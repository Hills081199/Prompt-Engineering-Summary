import os 
from openai import OpenAI, AsyncOpenAI
import tiktoken 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
DEFAULT_MODEL = "gpt-4o-mini"   # Rẻ nhất, phù hợp để học
def chat(
    user_message: str,
    system_message: str = "You are a helpful AI assistant.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    verbose: bool = True
) -> str:
    """
    Wrapper đơn giản cho OpenAI Chat API.
    Trả về text response và (tuỳ chọn) in thông tin usage.
    """
    response = client.chat.completions.create(
        model = model,
        temperature = temperature,
        max_tokens = max_tokens,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    result = response.choices[0].message.content
    if verbose:
        usage = response.usage
        print("Usage: ", usage)
        print(f"📊 Tokens — Input: {usage.prompt_tokens} | Output: {usage.completion_tokens} | Total: {usage.total_tokens}")
        print("-" * 60)
    return result

def count_tokens(text: str, model: str = "gpt-4o-mini"):
    """Đếm số token của một đoạn text."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# print("✅ Helper functions đã sẵn sàng!")

# # Test nhanh
# test_text = "Hello, I am learning AI Agents!"
# print(f"Token count của '{test_text}': {count_tokens(test_text)} tokens") #Token count của 'Hello, I am learning AI Agents!': 8 tokens