from utils import chat

# 2.1 zero shot prompting
# no example provided , knowledge only come from pre-training

zero_shot_prompts = """
Classify the following sentiment of the following review :
Review : "Sản phẩm tệ , không đáng tiền, giao hàng chậm "
Sentiment (Positive/Negative/Neutral):
"""

print("Zero shot prompting:")
print(chat(zero_shot_prompts))

# 2.2 few shot prompting
# provide example of the task and pattern to follow

system_prompt = "You are a sentiment analysis expert for an e-commerce platfor."
few_shot_prompt = """
# TASK: Phân tích cảm xúc của review sản phẩm
# FORMAT: Label (POSITIVE/NEGATIVE/NEUTRAL) + Score (0.0–1.0) + Reason (1 câu)
# EXAMPLES 1:
Review : "Giao hàng nhanh, đóng gói cẩn thận, sản phẩm đúng mô tả"
Label: POSITIVE, Score: 0.9, Reason: Đề cập tích cực về 3 yếu tố quan trọng: tốc độ, bao bì, và chất lượng

# EXAMPLES 2:
Review: "Chất lượng tạm được nhưng giá hơi cao so với thị trường"
Label: NEUTRAL, Score: 0.5, Reason: Có điểm tốt (chất lượng) nhưng cũng có điểm tiêu cực (giá cả)

# EXAMPLES 3:
Review: "Sản phẩm rất tệ, không đáng tiền, giao hàng chậm"
Reason: Nhiều vấn đề nghiêm trọng: chất lượng kém và dịch vụ tệ

# Now analyze:
Review: "Thiết kế đẹp nhưng pin không đúng như quảng cáo, hơi thất vọng"
Label:
Score:
Reason:
"""

print("Few shot prompting (3 examples):")
print(chat(few_shot_prompt, system_message=system_prompt))


# 2.3 Dynamic few shot prompting
# Kỹ thuật nâng cao: chọn examples phù hợp nhất với input hiện tại
# Trong production, dùng embedding similarity để chọn examples

EXAMPLE_BANK = [
    {
        "review": "Pin trâu, camera đẹp, màn hình sắc nét, rất đáng mua",
        "category": "phone",
        "label": "POSITIVE", "score": 0.92
    },
    {
        "review": "Máy tính xách tay tản nhiệt kém, hay nóng khi dùng lâu",
        "category": "laptop",
        "label": "NEGATIVE", "score": 0.20
    },
    {
        "review": "Tai nghe âm thanh tốt, kết nối ổn định, pin dùng được khoảng 8 tiếng",
        "category": "earphone",
        "label": "POSITIVE", "score": 0.80
    },
    {
        "review": "Điện thoại oke nhưng màn hình hay lag khi chơi game nặng",
        "category": "phone",
        "label": "NEUTRAL", "score": 0.50
    },
]

def select_examples(query: str, bank: list, n: int=2) -> list:
    """
    Simple keyword-based selection (production: dùng embedding cosine similarity).
    Chọn n examples liên quan nhất đến query.
    """
    # Trong thực tế: score = cosine_similarity(embed(query), embed(example['review']))
    # Đây là phiên bản simplified dùng keyword matching
    keywords = set(query.lower().split())
    scored = []
    for ex in bank:
        ex_words = set(ex["review"].lower().split())
        overlap = len(keywords & ex_words)
        scored.append((overlap, ex))
    
    # Sắp xếp theo độ trùng keyword giảm dần và lấy n examples
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:n]]

def build_dynamic_few_shot_prompt(query: str, n_examples: int = 2) -> str:
    examples = select_examples(query, EXAMPLE_BANK, n_examples)
    print("EXAMPLES SELECTED:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['review']} (Label: {ex['label']}, Score: {ex['score']})")
    prompt_parts = ["# Phân tích sentiment của review sản phẩm điện tử\n"]
    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"# Example {i}:")
        prompt_parts.append(f"Review: \"{ex['review']}\"")
        prompt_parts.append(f"Label: {ex['label']}")
        prompt_parts.append(f"Score: {ex['score']}\n")
    prompt_parts.append("# Now analyze:")
    prompt_parts.append(f"Review: \"{query}\"")
    prompt_parts.append("Label:\nScore:")
    return "\n".join(prompt_parts)

# Test dynamic selection
test_review = "Điện thoại đẹp nhưng pin yếu, chỉ dùng được nửa ngày"
dynamic_prompt = build_dynamic_few_shot_prompt(test_review)

print("📝 Dynamic Few-shot Prompt được tạo:")
print(dynamic_prompt)
print("\n" + "=" * 60)
print("🤖 Kết quả:")
print(chat(dynamic_prompt))
