from utils import chat, count_tokens, client, DEFAULT_MODEL, SMART_MODEL
from dataclasses import dataclass, field
import json
# ============================================================
# 6.1 JINJA2 TEMPLATES cho Production Prompts
# Trong production, prompts KHÔNG BAO GIỜ hardcoded
# ============================================================

from jinja2 import Template

AGENT_TEMPLATE = Template("""
# Role
{{ role }}

# Task
{{ task }}

# Context
{{ context }}

{% if constraints %}
# CONSTRAINTS
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{% if examples %}
# EXAMPLES
{% for ex in examples %}
Input: {{ ex.input }}
Output: {{ ex.output }}
---
{% endfor %}
{% endif %}

# Output Format
{{ output_format }}
"""
)

@dataclass
class PromptConfig:
    role: str
    task: str
    context: str
    output_format: str
    constraints: list = field(default_factory=list)
    examples: list = field(default_factory=list)

# Build prompt from template
config = PromptConfig(
    role="Senior Python Code Reviewer with security expertise",
    task="Review the provided Python code for bugs and security issues",
    context="Production FastAPI service, 10,000 requests/minute, Python 3.12",
    output_format='JSON: {"bugs": [...], "security_issues": [...], "score": 0-10}',
    constraints=[
        "Do not suggest a complete rewrite",
        "Max 5 issues",
        "Focus on security and correctness over style"
    ]
)

rendered_prompt = AGENT_TEMPLATE.render(**config.__dict__)
print("📝 Rendered System Prompt:")
print(rendered_prompt)
print(f"\n📊 Token count: {count_tokens(rendered_prompt)} tokens")


# ============================================================
# 6.2 PROMPT CHAINING PATTERN
# Chia task phức tạp thành nhiều steps, output step i → input step i+1
# ============================================================

def analyze_codebase_with_chaining(code: str) -> dict:
    """
    3-step prompt chain: Extract → Analyze → Recommend
    Tại sao chain thay vì 1 prompt to?
    - Mỗi step chuyên biệt → accuracy cao hơn
    - Dễ debug: biết step nào fail
    - Có thể cache intermediate results
    - Models xử lý tốt hơn khi task nhỏ hơn
    """
    print("🔗 PROMPT CHAIN - 3 STEPS")
    print("=" * 60)

    # STEP 1: Extract structure
    print("\n📍 STEP 1: Extracting code structure...")
    step1_response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a code parser. Extract the structure as JSON: {functions, classes, imports, dependencies}"
            },
            {"role": "user", "content": f"Parse this Python code:\n{code}"}
        ]
    )
    structure = step1_response.choices[0].message.content
    print(f"   Structure extracted: {len(structure)} chars")

    # STEP 2: Security analysis
    print("\n📍 STEP 2: Security analysis...")
    step2_response = client.chat.completions.create(
        model=SMART_MODEL,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a security expert. Analyze the code structure for vulnerabilities. Return JSON: {vulnerabilities: [{type, severity, description}]}"
            },
            {"role": "user", "content": f"Code structure:\n{structure}\n\nOriginal code:\n{code}"}
        ]
    )
    analysis = step2_response.choices[0].message.content
    print(f"   Analysis complete: {len(analysis)} chars")

    # STEP 3: Recommendations
    print("\n📍 STEP 3: Generating recommendations...")
    step3_response = client.chat.completions.create(
        model=SMART_MODEL,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": "You are a senior architect. Based on the security analysis, propose a concrete, prioritized action plan with code examples."
            },
            {
                "role": "user",
                "content": f"Security analysis:\n{analysis}\n\nOriginal code:\n{code}\n\nGenerate action plan with fixed code."
            }
        ]
    )
    plan = step3_response.choices[0].message.content

    print("\n📊 FINAL RESULT:")
    print("-" * 40)
    print(plan)

    return {
        "structure": json.loads(structure),
        "analysis": json.loads(analysis),
        "action_plan": plan
    }


# Test
insecure_code = """
import os
import sqlite3

def execute_command(cmd: str):
    result = os.system(cmd)  # Direct OS command execution!
    return result

def search_user(username: str, db_path: str):
    conn = sqlite3.connect(db_path)
    # SQL Injection vulnerability
    rows = conn.execute(f"SELECT * FROM users WHERE name = '{username}'").fetchall()
    return rows
"""

chain_result = analyze_codebase_with_chaining(insecure_code)