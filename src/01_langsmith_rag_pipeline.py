"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.
    """

    # Khởi tạo embeddings từ factory
    embeddings = get_embeddings()

    # Đọc nội dung knowledge base
    text = load_knowledge_base()

    # Chia text thành chunks
    chunks = split_text(
        text,
        chunk_size=500,
        chunk_overlap=50,
    )

    print(f"📚 Đã chia thành {len(chunks)} chunks")

    # Tạo FAISS vectorstore
    vectorstore = build_vectorstore(chunks, embeddings)

    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\n"
            "Context:\n{context}",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:

        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """

    llm = get_llm()

    # Retriever lấy 3 documents gần nhất
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # Ghép page_content của các documents thành context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Xây dựng LCEL RAG chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Chạy RAG chain với một câu hỏi.

    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith
    như một trace riêng.
    """

    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    # Tạo FAISS vectorstore
    vectorstore = setup_vectorstore()

    # Build RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # Chạy toàn bộ sample questions
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)

        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {str(answer)[:100]}\n")

    print(
        f"\n✅ {len(SAMPLE_QUESTIONS)} traces đã gửi lên "
        f"LangSmith project '{config.LANGSMITH_PROJECT}'"
    )
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()