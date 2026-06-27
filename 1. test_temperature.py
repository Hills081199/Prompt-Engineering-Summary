from utils import chat
prompt = "Viết 1 tới 5 câu mô tả về AI Agents ứng dụng trong lĩnh vực giáo dục"
print("="*60)
print("Temperature = 0.0 (deterministic)")
print("="*60)
print(chat(prompt, temperature=0.0))
print("\n" + "=" * 60)
print("Temperature = 1.0 (creative):")
print(chat(prompt, temperature=1.0))