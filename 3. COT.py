from utils import chat

# 3.1 Zero Shot CoT
# Bài toán tính chi phí token — rất thực tế trong AI Agent development!
problem = """
Một AI Agent cần xử lý 1,000 requests mỗi ngày.
Mỗi request tiêu tốn trung bình 800 tokens (input + output).
GPT-4o-mini có giá: $0.00015/1K input tokens, $0.0006/1K output tokens.
Giả sử tỷ lệ input:output = 70:30.
Tổng chi phí mỗi tháng (30 ngày) là bao nhiêu USD?
"""

# KHông dùng COT
print("WITHOUT COT:")
print(chat(problem))

print("\n" + "=" * 60)
print("WITH Zero-shot COT:")
print(chat(problem + "\n\nHãy suy nghĩ từng bước trước khi đưa ra câu trả lời."))

# 3.2 COT FEW SHOT:
cot_few_shot = """
Giải quyết các bài toán về AI Agent performance:
---
Q: Nếu agent gọi 3 tools song song, mỗi tool mất 2 giây,
   nhưng 1 tool timeout sau 5 giây, tổng thời gian là bao nhiêu?
A: Hãy suy nghĩ từng bước:
   1. 3 tools chạy song song → thời gian = max(thời gian từng tool)
   2. Tool 1: 2s, Tool 2: 2s, Tool 3: timeout = 5s
   3. max(2, 2, 5) = 5 seconds
   4. Thêm retry logic overhead: +2s
   → Kết quả: ~7s với retry, hoặc 5s nếu fail-fast

---
Q: Một agent có context window 128K tokens. Mỗi turn conversation
   trung bình 500 tokens. Agent cần lưu system prompt 2000 tokens
   và tool definitions 1000 tokens. Tối đa bao nhiêu turns?
A: Hãy suy nghĩ từng bước:
    1. Context window: 128,000 tokens
    2. Fixed overhead (system + tools): 2000 + 1000 = 3000 tokens
    3. Remaining for conversation: 128,000 - 3000 = 125,000 tokens
    4. Mỗi turn ~500 tokens → 125,000 / 500 = 250 turns
    5. Thêm safety margin: ~240 turns

--- 
Q: Agent dùng RAG pipeline: embedding 1000 docs, mỗi doc 512 tokens.
   Embedding model ada-002 giá $0.0001/1K tokens.
   Chi phí index toàn bộ corpus là bao nhiêu?
A: Hãy suy nghĩ từng bước:
"""

print("FEW-SHOT CoT:")
print(chat(cot_few_shot, temperature=0.1))

#3.3 SELF-CONSISTENCY — Majority voting để tăng accuracy
