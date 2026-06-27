from utils import chat, client
from typing import Callable

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

# ============================================================
# 3.3 SELF-CONSISTENCY — Majority voting để tăng accuracy
# Chạy cùng prompt nhiều lần với temperature > 0, lấy đáp án phổ biến nhất
# ============================================================

from collections import Counter


def self_consistency(
    prompt: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    extract_final_answer: Callable = None
) -> dict:
    """
    Chạy prompt n lần, trả về majority answer.
    Phù hợp cho bài toán có đáp án đúng/sai rõ ràng.
    Chi phí: n × đơn giá — dùng khi cần accuracy cao.
    """
    answers = []
    full_responses = []

    print(f"🔄 Chạy {n_samples} samples với temperature={temperature}...")

    for i in range(n_samples):
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=temperature,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        full_responses.append(full_text)

        # Extract final answer (dòng cuối cùng thường là answer)
        if extract_final_answer:
            answer = extract_final_answer(full_text)
        else:
            answer = full_text.strip().split("\n")[-1].strip()

        answers.append(answer)
        print(f"  Sample {i+1}: {answer[:80]}..." if len(answer) > 80 else f"  Sample {i+1}: {answer}")

    # Majority vote
    counter = Counter(answers)
    majority_answer, count = counter.most_common(1)[0]

    return {
        "majority_answer": majority_answer,
        "confidence": count / n_samples,
        "all_answers": dict(counter),
        "all_responses": full_responses
    }


# Test Self-Consistency
sc_prompt = """
Một AI startup có 3 engineers. Mỗi engineer build 2 AI agents.
Mỗi agent gọi trung bình 5 API calls/request.
Nếu system nhận 100 requests/ngày, tổng API calls trong 1 tuần là bao nhiêu?

Suy nghĩ từng bước và đưa ra kết quả cuối cùng là một con số.
"""

result = self_consistency(sc_prompt, n_samples=5)
print(f"\n📊 KẾT QUẢ SELF-CONSISTENCY:")
print(f"   Majority answer : {result['majority_answer']}")
print(f"   Confidence      : {result['confidence']:.0%}")
print(f"   Tất cả đáp án   : {result['all_answers']}")

# ============================================================
# 3.4 TREE-OF-THOUGHT (ToT)
# Khám phá nhiều reasoning branches song song, đánh giá và chọn tốt nhất
# ============================================================

tot_prompt = """
Problem: Thiết kế kiến trúc memory cho một AI Agent phải xử lý 10,000 sessions/ngày,
mỗi session có 50 turns conversation.

Đề xuất 3 approaches KHÁC NHAU:

Approach 1: [Tên kiến trúc]
→ Mô tả: ...
→ Pros: ...
→ Cons: ...
→ Score: .../10

Approach 2: [Tên kiến trúc]
→ Mô tả: ...
→ Pros: ...
→ Cons: ...
→ Score: .../10

Approach 3: [Tên kiến trúc]
→ Mô tả: ...
→ Pros: ...
→ Cons: ...
→ Score: .../10

Final Recommendation: [Approach được chọn và lý do]
"""

print("🌳 TREE-OF-THOUGHT:")
print(chat(tot_prompt, model='gpt-4o-mini', max_tokens=1500))