"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra
"""

import sys
import json
import os
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain
from unittest.mock import MagicMock
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
sys.modules["langchain_community.llms.vertexai"] = MagicMock()
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import (
    load_knowledge_base,
    split_text,
    build_vectorstore,
)
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = """
Bạn là trợ lý AI hữu ích.
Chỉ sử dụng context được cung cấp để trả lời câu hỏi.
Trả lời ngắn gọn, trực tiếp trong khoảng 2-4 câu.

Context:
{context}

Nếu context không chứa đủ thông tin, hãy nói rõ rằng bạn không có đủ dữ liệu để trả lời.
""".strip()

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])


SYSTEM_V2 = """
Bạn là chuyên gia AI có nhiệm vụ trả lời câu hỏi dựa trên knowledge base.

Hãy:
1. Đọc kỹ context được cung cấp.
2. Xác định các thông tin liên quan trực tiếp đến câu hỏi.
3. Trả lời rõ ràng, có cấu trúc và chính xác trong khoảng 3-5 câu.
4. Không sử dụng kiến thức bên ngoài context.

Context:
{context}

Nếu context không đủ để trả lời, hãy nói rõ rằng thông tin hiện có chưa đủ.
""".strip()

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# Các tác vụ dưới đây độc lập nhau, nhưng vẫn cần giới hạn song song để tránh
# vượt rate limit của provider. Có thể hạ các giá trị này nếu provider trả 429.
def _positive_env_int(name: str, default: int) -> int:
    """Đọc một biến môi trường số nguyên dương, với giá trị mặc định an toàn."""
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


RAG_MAX_CONCURRENCY = _positive_env_int("RAG_MAX_CONCURRENCY", 8)
RAGAS_MAX_WORKERS = _positive_env_int("RAGAS_MAX_WORKERS", 8)
RAGAS_BATCH_SIZE = _positive_env_int("RAGAS_BATCH_SIZE", 16)


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG:
    contexts phải là list[str], không phải string đã ghép.

    Trả về:
        {
            "answer": str,
            "contexts": list[str]
        }
    """

    # Retrieve top-k documents
    docs = retriever.invoke(question)

    # Giữ riêng từng context cho RAGAS
    contexts = [
        doc.page_content
        for doc in docs
    ]

    # Ghép thành chuỗi để truyền vào prompt
    ctx_str = "\n\n".join(contexts)

    # RAG generation chain
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": ctx_str,
        "question": question,
    })

    return {
        "answer": answer,
        "contexts": contexts,
    }


def build_rag_chain(llm, prompt):
    """Tạo chain một lần để dùng lại cho toàn bộ QA pairs."""
    return prompt | llm | StrOutputParser()


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả QA pairs qua prompt version được chỉ định.

    Trả về list dict:
      question
      reference
      answer
      contexts
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = get_llm()
    prompt = PROMPTS[prompt_version]

    print(
        f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi "
        f"với prompt {prompt_version} "
        f"(tối đa {RAG_MAX_CONCURRENCY} tác vụ song song) ..."
    )

    # Retrieval và generation của mỗi QA pair độc lập. batch() giữ nguyên thứ
    # tự input, nên answers/contexts vẫn khớp với QA_PAIRS như cách chạy tuần tự.
    questions = [qa["question"] for qa in QA_PAIRS]
    batch_config = {"max_concurrency": RAG_MAX_CONCURRENCY}
    retrieved_docs = retriever.batch(questions, config=batch_config)
    contexts_per_question = [
        [doc.page_content for doc in docs]
        for docs in retrieved_docs
    ]

    chain = build_rag_chain(llm, prompt)
    answers = chain.batch(
        [
            {"context": "\n\n".join(contexts), "question": question}
            for question, contexts in zip(questions, contexts_per_question)
        ],
        config=batch_config,
    )

    results = []
    for i, (qa, answer, contexts) in enumerate(
        zip(QA_PAIRS, answers, contexts_per_question), 1
    ):
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": answer,
            "contexts": contexts,
        })

        print(
            f"  [{i:02d}/{len(QA_PAIRS)}] "
            f"{qa['question'][:60]}"
        )

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển kết quả RAG thành RAGAS EvaluationDataset.
    """

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(
        samples=samples
    )


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str, llm_eval, emb_eval) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.

    Trả về:
        {
            metric_name: mean_score
        }
    """

    print(
        f"\n📐 Đang đánh giá RAGAS cho prompt {version} ..."
    )

    dataset = build_ragas_dataset(
        rag_results
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS),
        batch_size=RAGAS_BATCH_SIZE,
    )

    # Tính mean cho từng metric
    scores = {}

    for key in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        raw = result[key]

        valid_values = [
            v
            for v in raw
            if v is not None
            and not np.isnan(v)
        ]

        scores[key] = (
            float(np.mean(valid_values))
            if valid_values
            else 0.0
        )

    print(
        f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:"
    )

    for k, v in scores.items():
        star = (
            " ⭐"
            if k == "faithfulness"
            and v >= 0.8
            else ""
        )

        print(
            f"  {k:30s}: "
            f"{v:.4f}{star}"
        )

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Chạy RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(
        vectorstore,
        "v1",
    )

    v2_results = collect_rag_outputs(
        vectorstore,
        "v2",
    )

    # Reuse evaluator clients between V1/V2; creating them twice does not add
    # value and can repeat connection/setup overhead.
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(
        v1_results,
        "v1",
        llm_eval,
        emb_eval,
    )

    v2_scores = run_ragas_eval(
        v2_results,
        "v2",
        llm_eval,
        emb_eval,
    )

    # ── Bảng so sánh ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(
        f"  {'Metric':30s}  "
        f"{'V1':>8}  "
        f"{'V2':>8}  "
        f"Winner"
    )
    print("=" * 65)

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]

        if s1 > s2:
            winner = "← V1"
        elif s2 > s1:
            winner = "← V2"
        else:
            winner = "Tie"

        print(
            f"  {metric:30s}  "
            f"{s1:>8.4f}  "
            f"{s2:>8.4f}  "
            f"{winner}"
        )

    # ── Kiểm tra target ───────────────────────────────────────────────────
    best_faith = max(
        v1_scores["faithfulness"],
        v2_scores["faithfulness"],
    )

    if best_faith >= 0.8:
        print(
            f"\n✅ Đạt mục tiêu: "
            f"faithfulness = {best_faith:.4f} ≥ 0.8"
        )
    else:
        print(
            f"\n⚠️  Chưa đạt mục tiêu "
            f"({best_faith:.4f} < 0.8)."
        )
        print(
            "   Gợi ý: giảm chunk_size, tăng k, "
            "hoặc điều chỉnh prompt."
        )

    # ── Lưu report ────────────────────────────────────────────────────────
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }

    report_path = (
        Path(__file__).parent.parent
        / "data"
        / "ragas_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"💾 Đã lưu báo cáo vào "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()
