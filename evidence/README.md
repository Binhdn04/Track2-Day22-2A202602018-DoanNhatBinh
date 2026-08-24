# So sánh V1 và V2

Kết quả đánh giá RAGAS cho thấy **Prompt V1 hoạt động tốt hơn V2**, dù chênh lệch không lớn.

| Metric            |         V1 |     V2 |
| ----------------- | ---------: | -----: |
| Faithfulness      | **0.9566** | 0.9474 |
| Answer Relevancy  | **0.9177** | 0.9034 |
| Context Recall    |     1.0000 | 1.0000 |
| Context Precision |     0.9450 | 0.9450 |

Cả hai phiên bản đều đạt **Context Recall = 1.0** và **Context Precision ≈ 0.945**, cho thấy hệ thống retrieval cung cấp context đầy đủ và chính xác tương đương nhau.

Sự khác biệt chủ yếu nằm ở chất lượng sinh câu trả lời. **V1 có Faithfulness và Answer Relevancy cao hơn V2**, nghĩa là câu trả lời của V1 bám sát context và câu hỏi tốt hơn một chút.

**Kết luận:** Target `faithfulness ≥ 0.8` đã đạt ở cả hai phiên bản. Với kết quả hiện tại, **Prompt V1 nên được chọn làm phiên bản mặc định**, trong khi V2 cần tiếp tục tối ưu trước khi thay thế V1.
