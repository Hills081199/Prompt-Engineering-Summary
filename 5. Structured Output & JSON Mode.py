# ============================================================
# 5.1 JSON MODE của OpenAI
# response_format={"type": "json_object"} đảm bảo output luôn là valid JSON
# ============================================================
from utils import client
import json

def get_json_output(prompt: str, schema_description: str) -> dict:
    """
    Luôn trả về valid JSON.
    Quan trọng: Phải mention "JSON" trong system hoặc user message khi dùng json_object mode.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1000,
        response_format={"type": "json_object"},  # ← Key parameter!
        messages=[
            {
                "role": "system",
                "content": f"""You are a data extraction agent.
                    ALWAYS respond with valid JSON following this schema:
                    {schema_description}
                    Rules:
                    - Output JSON only, no additional text
                    - Use null for missing data, never omit fields
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return json.loads(response.choices[0].message.content)

# Test: Extract structured info từ unstructured text
job_posting = """
Chúng tôi cần tuyển Senior AI Engineer với 5+ năm kinh nghiệm Python.
Yêu cầu: FastAPI, LangChain, Docker, kinh nghiệm deploy ML models.
Mức lương: 3000–5000 USD/tháng. Remote-first. Deadline nộp hồ sơ: 31/12/2024.
Email: career@aicompany.com
"""
schema = """
{
  "position": "string",
  "seniority": "junior|mid|senior|lead",
  "years_experience": "number or null",
  "required_skills": ["string"],
  "salary_range": {"min": "number or null", "max": "number or null", "currency": "string"},
  "work_mode": "remote|hybrid|onsite",
  "deadline": "YYYY-MM-DD or null",
  "contact": "string or null"
}
"""

result = get_json_output(f"Extract info from: {job_posting}", schema)
print("📊 Structured Output:")
print(json.dumps(result, indent=2, ensure_ascii=False))

# ============================================================
# 5.2 PYDANTIC VALIDATION cho structured outputs
# Đây là pattern production-grade nhất
# ============================================================
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum
class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class CodeIssue(BaseModel):
    type: str = Field(description="Issue type: SECURITY, PERFORMANCE, CORRECTNESS, STYLE")
    severity: Severity
    description: str
    line_number: Optional[int] = None
    fix_suggestion: str


class CodeReviewResult(BaseModel):
    summary: str
    approved: bool
    overall_score: int = Field(ge=0, le=10, description="0-10 score")
    issues: List[CodeIssue]
    positives: List[str]

    @field_validator("overall_score")
    @classmethod
    def validate_score(cls, v):
        if not 0 <= v <= 10:
            raise ValueError("Score must be 0-10")
        return v


def review_code_structured(code: str) -> CodeReviewResult:
    schema_json = CodeReviewResult.model_json_schema()

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert code reviewer.
Return JSON matching this schema exactly:
{json.dumps(schema_json, indent=2)}"""
            },
            {"role": "user", "content": f"Review this Python code:\n```python\n{code}\n```"}
        ]
    )

    raw = json.loads(response.choices[0].message.content)
    return CodeReviewResult(**raw)  # Pydantic validation!


# Test
sample_code = """
def divide(a, b):
    return a / b  # No error handling

result = divide(10, 0)  # Will crash!
"""

review = review_code_structured(sample_code)
print(f"✅ Pydantic validated output:")
print(f"   Approved    : {review.approved}")
print(f"   Score       : {review.overall_score}/10")
print(f"   Issues      : {len(review.issues)}")
print(f"   Summary     : {review.summary}")
if review.issues:
    print(f"\n   Issues found:")
    for issue in review.issues:
        print(f"   [{issue.severity}] {issue.type}: {issue.description}")
        print(f"   → Fix: {issue.fix_suggestion}")
     