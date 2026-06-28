# 🚀 Prompt Engineering — Zero to Hero

> **Audience:** Python developers who want to become production-grade AI Engineers  
> **Format:** Theory + hands-on, runnable code for every concept  
> **Models:** `gpt-4o-mini` (default, cost-effective) · `gpt-4o` (complex tasks)

![Prompt Engineering Techniques](prompt%20engineering%20techniques.png)

---

## 📋 Table of Contents

- [Why This Repo Exists](#-why-this-repo-exists)
- [Roadmap Overview](#-roadmap-overview)
- [Repo Structure](#-repo-structure)
- [Quick Start](#-quick-start)
- [Part 1 — Foundations](#-part-1--foundations)
- [Part 2 — Advanced & Production](#-part-2--advanced--production)
- [Standalone Scripts](#-standalone-scripts)
- [Key Mental Models](#-key-mental-models)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🎯 Why This Repo Exists

Large Language Models are **general-purpose reasoning engines that you program with natural language**. The quality of everything you build on top of an LLM — chatbots, autonomous agents, RAG pipelines, copilots — is bounded by the quality of your prompts and the surrounding "prompt program."

**Prompt Engineering is a real engineering discipline**, not "talking nicely to a chatbot." It has:

- **Theory** — a mental model of what an LLM is doing when it predicts text, and *why* certain prompt structures reliably change its behavior
- **Patterns** — reusable techniques (few-shot, CoT, ReAct, self-consistency, etc.) backed by published research
- **Engineering practice** — versioning, testing, evaluation, cost/latency budgets, structured outputs, and production-grade tooling

Every theory section in this repo is immediately followed by **runnable code** on realistic, AI-agent-flavored examples: token cost calculations, code review agents, multi-tool reasoning — the kind of problems you'll face building real AI Agents.

---

## 🗺️ Roadmap Overview

```
Phase 1 (Weeks 1–2) ← YOU ARE HERE
├── LLM APIs (OpenAI · Anthropic · Gemini)
├── Prompt Engineering ← THIS REPO (Part 1 + Part 2)
│   ├── Zero/Few-shot · Chain-of-Thought · Role Prompting
│   └── Structured Output · Templates · ReAct · Self-Critique · Meta-Prompting
├── Tokens & Context Window Management
├── Embeddings & Semantic Search
└── Structured Output & Function/Tool Calling

Phase 2 → Tool Use · Agentic Loops · RAG · Vector DBs
Phase 3 → LangChain · LangGraph · CrewAI · Production Deployment
Phase 4 → Multi-Agent Systems · Planning · Human-in-the-loop
Phase 5 → Safety, Alignment, Fine-tuning · Frontier Research
```

---

## 📁 Repo Structure

```
Prompt-Engineering-Summary/
│
├── 📓 Notebooks (primary learning material)
│   ├── Prompt_Engineering_Zero_to_Hero_Part1.ipynb   ← Sections 0–7
│   ├── Prompt_Engineering_Zero_to_Hero_Part2.ipynb   ← Sections 7–12
│   └── AI_Agents_Prompt_Engineering.ipynb
│
├── 🐍 Standalone Scripts (run each file individually)
│   ├── utils.py                                       ← Shared helpers
│   ├── 1. test_temperature.py
│   ├── 2. Zero shot and few shot prompting.py
│   ├── 3. COT.py
│   ├── 4. Role Prompting and System Prompting.py
│   ├── 5. Structured Output & JSON Mode.py
│   ├── 6. Prompt template and Chaining.py
│   └── 7. ReAct Pattern.py
│
├── 📄 Docs
│   ├── Prompt_Engineering_AI_Agents.docx              ← Vietnamese
│   ├── Prompt_Engineering_AI_Agents_EN.docx           ← English
│   └── prompt engineering techniques.png
│
├── .env            ← API keys (never commit)
└── .gitignore
```

---

## ⚙️ Quick Start

**1. Clone & install**
```bash
git clone <repo-url>
cd Prompt-Engineering-Summary
pip install openai tiktoken jinja2 pydantic aiohttp python-dotenv
pip install jupyter        # for notebooks
```

**2. Set your API key**
```bash
# Recommended: environment variable (never hardcode in files you commit)
export OPENAI_API_KEY=sk-...
```
Or create `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
```

**3. Run a script**
```bash
python "1. test_temperature.py"
```

**4. Open a notebook**
```bash
jupyter notebook Prompt_Engineering_Zero_to_Hero_Part1.ipynb
```

---

## 📚 Part 1 — Foundations

`Prompt_Engineering_Zero_to_Hero_Part1.ipynb` · ~3–4 hours

| Section | Topic | Techniques |
|---------|-------|-----------|
| **0** | What Is Prompt Engineering? | LLM mental model, tokens, context window, 4 levers |
| **1** | Setup & API Basics | `temperature`, `top_p`, `seed`, `max_tokens`, tokenizer deep-dive |
| **2** | Zero / One / Few-Shot | In-context learning, dynamic example selection, embeddings |
| **3** | Chain-of-Thought | Zero-shot CoT, Few-shot CoT, Self-Consistency, Tree-of-Thought |
| **4** | Role & System Prompt | Persona engineering, system prompt anatomy, multi-agent personas |
| **5** | Structured Output | JSON mode, Pydantic validation, retry-on-failure pattern |
| **6** | Templates & Chaining | Jinja2 templating, multi-step prompt chains |
| **7** | ReAct Pattern | Thought→Action→Observation loop, tool parsing, agent from scratch |

---

### Section 0 — LLM Mental Model

An LLM is a function:
```
output = f(system_prompt, conversation_history, user_input, sampling_params)
```
**Prompt Engineering = deliberately engineering every argument of `f`** so the output distribution is shifted toward what you want, as reliably and cheaply as possible.

**The 4 Levers:**

| Lever | What it controls |
|-------|----------------|
| **Content** | What information/instructions are present |
| **Structure** | How information is organized/delimited |
| **Sampling** | How the model turns probabilities into tokens (`temperature`, `top_p`, `seed`) |
| **Control flow** | How many LLM calls, in what order, with what feedback (chaining, ReAct, self-consistency) |

---

### Section 1 — Sampling Parameters

| Parameter | Recommended | Use case |
|-----------|------------|---------|
| `temperature=0.0–0.2` | Production consistency | Classification, JSON output, code review |
| `temperature=0.7–1.0` | Creativity | Brainstorming, copywriting |
| `top_p` | Alternative to temperature | Use one or the other, not both at non-default values |
| `seed=42` | Reproducibility | Writing reproducible prompt tests |
| `max_tokens` | Always set explicitly | Never rely on defaults |

> ⚠️ `temperature=0` does **not** guarantee bit-for-bit identical output (GPU non-determinism), but gets you very close.

---

### Section 2 — Zero / One / Few-Shot Prompting

**Theory — In-Context Learning:**

| | Traditional ML training | In-context learning |
|---|---|---|
| Mechanism | Gradient descent updates weights | Model attends to examples in the context window |
| Persistence | Permanent | Only for this single prompt |
| Data needed | Hundreds to millions of examples | 0–10 examples |

**When zero-shot fails → move to few-shot when:**
- Inconsistent output **format**
- Domain-specific or ambiguous **labels**
- A **house style** you need the model to imitate exactly

**2.1 Zero-Shot**
```python
zero_shot_prompt = """
Classify the sentiment of the following review:

Review: "The product is bad, not worth the money, slow delivery"
Sentiment (POSITIVE/NEGATIVE/NEUTRAL):
"""
```

**2.2 One-Shot** — Pin down the output format with a single example:
```python
one_shot_prompt = """
Classify sentiment in exactly this format.

Example:
Review: "Fast shipping, great packaging, exactly as described"
Sentiment: POSITIVE | Confidence: 0.95

Now classify:
Review: "The product is bad, not worth the money, slow delivery"
Sentiment:
"""
```

**2.3 Few-Shot** — 3 examples covering POSITIVE / NEUTRAL / NEGATIVE cases, consistent format.

**2.4 Dynamic Few-Shot (Advanced)** — Select examples per-input via cosine similarity on embeddings:
```python
def select_examples_embedding(query, bank, n=2):
    query_emb = get_embedding(query)
    scored = [(cosine_similarity(query_emb, get_embedding(ex["review"])), ex) for ex in bank]
    return [ex for _, ex in sorted(scored, reverse=True)[:n]]
```
In production: pre-compute embeddings, store in a vector DB (FAISS, Pinecone, pgvector), do fast nearest-neighbor lookup at request time.

**Best Practices:**

| Principle | Detail |
|-----------|--------|
| Quality > Quantity | 3–5 excellent examples beat 10 mediocre ones |
| Diversity | Cover important edge cases and variations |
| Consistent format | All examples must share the exact same structure |
| Recency bias | The *last* example has the strongest influence |
| Task before examples | Describe the task before showing examples |
| Dynamic selection | Selecting relevant examples per-input can improve accuracy 15–20% |

---

### Section 3 — Chain-of-Thought (CoT)

**Why it works:** Forcing the model to emit intermediate reasoning tokens gives it more "compute" (more conditioning context) before committing to a final answer. Wei et al. (2022): boosted accuracy from ~18% to ~57% on arithmetic benchmarks.

**3.1 Zero-Shot CoT** — Append a trigger phrase:
```python
problem + "\n\nLet's think step by step before giving the final answer."
```

**3.2 Few-Shot CoT** — Show worked examples *with* their reasoning chains:
```
Q: If an agent calls 3 tools in parallel, each taking 2 seconds,
   but one times out at 5 seconds, what is the total time?
A: Let's think step by step:
   1. 3 tools run in parallel → total time = max(time of each tool)
   2. Tool 1: 2s, Tool 2: 2s, Tool 3: timeout = 5s
   3. max(2, 2, 5) = 5 seconds
   4. Add retry-logic overhead: +2s
   → Result: ~7s with retry, or 5s if fail-fast
```

**3.3 Self-Consistency** — Run `n` times at `temperature > 0`, take majority vote:
```python
def self_consistency(prompt, n_samples=5, temperature=0.7):
    answers = [run_once(prompt) for _ in range(n_samples)]
    majority, count = Counter(answers).most_common(1)[0]
    return {"majority_answer": majority, "confidence": count / n_samples}
```
Cost: `n × single-call price`. Use when accuracy truly matters.

**3.4 Tree-of-Thought** — Explore multiple reasoning branches, score and select the best. The notebook implements a single-prompt approximation where the model proposes 3 distinct approaches with pros/cons and a score.

| Technique | Use when | Trade-off |
|-----------|----------|-----------|
| Zero-shot CoT | Simple reasoning, one clear answer | Fast, cheap |
| Few-shot CoT | Domain-specific reasoning pattern needed | More tokens |
| Self-Consistency | High accuracy required | `n×` cost |
| Tree-of-Thought | Large solution space, need best option | Most complex and expensive |

---

### Section 4 — Role & System Prompt Design

**Why role prompting works:** It's a conditioning trick that shifts the output distribution toward text patterns associated with that role in training data. A vague role carries almost no useful signal; a precise role gives the model a narrower, more predictable target distribution.

**4.1 Vague vs Specific Role:**
```python
# ❌ Vague — zero conditioning signal
bad_system = "You are a helpful AI assistant."

# ✅ Specific — shifts distribution toward security-expert behavior
good_system = """
You are a Senior Python Engineer with 10+ years building production backend systems at scale.
You specialize in security vulnerabilities and database performance.

When reviewing code, you:
1. ALWAYS identify security vulnerabilities first (SQL injection, XSS, auth issues)
2. Then flag performance bottlenecks with specific metrics
3. Provide corrected code, not just descriptions
4. Explain WHY each issue matters in production
5. Rate severity: CRITICAL / HIGH / MEDIUM / LOW
"""
```

**4.2 Anatomy of a Professional System Prompt** — 6 required sections:

```
# IDENTITY      → Who the model is, and its relevant expertise
# OBJECTIVE     → What task, for what audience
# CAPABILITIES  → What kinds of things it should identify / be able to do
# CONSTRAINTS   → Explicit limits: what NOT to do, max items, etc.
# OUTPUT FORMAT → The exact structure expected (so downstream code can parse it)
# BEHAVIOR RULES → How to resolve ambiguity and priority conflicts
```

**4.3 Multi-Persona for Multi-Agent Systems:**

| Agent | Responsibility | Key Constraint |
|-------|---------------|----------------|
| **Orchestrator** | Decompose task, assign subtasks, synthesize results | Does NOT execute subtasks |
| **Research Agent** | Fetch and return raw information | No interpretation, no opinions |
| **Code Generation Agent** | Write code given a clear spec | Code only, no explanations unless asked |

Why separate personas? Each agent's context stays small and focused → fewer hallucinations. You can swap models per-agent. Easier to test, version, and debug independently. This mirrors AutoGen / CrewAI / LangGraph design.

---

### Section 5 — Structured Output & JSON Mode

**Three layers (weakest → strongest):**
1. Prompt-level formatting instructions — model *usually* complies but can drift
2. **JSON mode** (`response_format={"type": "json_object"}`) — API guarantees syntactically valid JSON, but **not** that it matches *your* schema
3. **Pydantic schema validation** — parse JSON and validate against strict schema. This is the **production-grade pattern**

```python
# 5.1 JSON Mode
response_format={"type": "json_object"}
# ⚠️ Must mention "JSON" somewhere in the system or user message

# 5.2 Pydantic Validation
class CodeIssue(BaseModel):
    type: str
    severity: Severity   # Enum: CRITICAL | HIGH | MEDIUM | LOW
    description: str
    fix_suggestion: str

class CodeReviewResult(BaseModel):
    summary: str
    approved: bool
    overall_score: int = Field(ge=0, le=10)
    issues: List[CodeIssue]

result = CodeReviewResult(**json.loads(response))  # Validates automatically!
```

**5.3 Retry-on-Validation-Failure** — Feed validation errors back to the model:
```python
def review_code_with_retry(code, max_retries=2):
    for attempt in range(max_retries + 1):
        response = call_llm(messages)
        try:
            return CodeReviewResult(**json.loads(response))
        except ValidationError as e:
            # Append the error back as a user message → model self-corrects
            messages.append({"role": "user",
                             "content": f"Your JSON failed: {e}\nReturn corrected JSON."})
```

---

### Section 6 — Prompt Templates & Chaining

**6.1 Jinja2 Templates** — Prompts in production are never hardcoded strings:

```python
AGENT_TEMPLATE = Template("""
# ROLE
{{ role }}

# TASK
{{ task }}

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

# OUTPUT FORMAT
{{ output_format }}
""")
```

Benefits: separation of concerns, conditional sections (shorter prompts = lower cost), loops for examples, reusability across agents just by swapping the config object.

**6.2 Prompt Chaining — 3-Step Pipeline:**

```
Code ──→ [Step 1: Parse Structure]  ──→ [Step 2: Security Analysis]  ──→ [Step 3: Action Plan]
          gpt-4o-mini                        gpt-4o                          gpt-4o
```

Why chain instead of one giant prompt?

| Benefit | Explanation |
|---------|------------|
| Higher accuracy | Each step is focused → fewer cognitive demands on the model |
| Debuggability | When something fails, you know exactly which step |
| Caching | Intermediate results (e.g., parsed structure) can be reused |
| Cost optimization | Cheap model for simple extraction, smart model for reasoning |

Trade-off: more latency and total token usage — a deliberate choice, not a default.

---

### Section 7 — ReAct Pattern (Part 1 intro)

The ReAct loop (`Thought → Action → Observation → Final Answer`) is built from scratch in Part 1 with 3 tools (`search`, `calculate`, `read_file`) and a basic regex parser.

Part 2 provides the full production-grade implementation. See [Section 7 below](#section-7--react-pattern-full-implementation).

---

### ✅ What You've Mastered After Part 1

| Technique | ✓ |
|-----------|---|
| LLM mental model, tokens, context window | ✅ |
| Sampling parameters (temperature, top_p, seed) | ✅ |
| Zero-shot / One-shot / Few-shot Prompting | ✅ |
| Dynamic Few-shot Selection (keyword + embeddings) | ✅ |
| Chain-of-Thought (Zero-shot & Few-shot CoT) | ✅ |
| Self-Consistency (Majority Voting) | ✅ |
| Tree-of-Thought | ✅ |
| Role Prompting & System Prompt Anatomy | ✅ |
| Multi-Persona prompting for multi-agent systems | ✅ |
| JSON Mode & Pydantic-validated Structured Output | ✅ |
| Validation-retry loop pattern | ✅ |
| Jinja2 Prompt Templates | ✅ |
| Prompt Chaining (multi-step pipelines) | ✅ |
| ReAct Pattern built from scratch | ✅ |

---

## 🤖 Part 2 — Advanced & Production

`Prompt_Engineering_Zero_to_Hero_Part2.ipynb` · ~4–6 hours · Prerequisite: Part 1

| Section | Topic | Difficulty |
|---------|-------|-----------|
| **7** | ReAct Pattern (full) | ⭐⭐⭐ |
| **8** | Meta-Prompting, Self-Critique, Constitutional AI | ⭐⭐⭐⭐ |
| **9** | Prompt Evaluation & A/B Testing | ⭐⭐⭐⭐ |
| **10** | Production Systems | ⭐⭐⭐⭐⭐ |
| **11** | Anti-Patterns | ⭐⭐ |
| **12** | Capstone Project | ⭐⭐⭐⭐⭐ |

---

### Section 7 — ReAct Pattern (Full Implementation)

**Why ReAct beats pure CoT:**

| Approach | Accuracy (HotpotQA) | Hallucination | Debuggable |
|----------|--------------------|--------------|-----------| 
| Direct answer (no tools) | ~29% | High | Black box |
| Chain-of-Thought only | ~33% | Medium | Hard |
| **ReAct** | **~35–50%** | **Low** | **Traceable** |
| ReAct + self-reflection | ~55%+ | Very Low | Full trace |

The key insight: **Observation** grounds subsequent **Thoughts** in real-world facts, breaking the hallucination cycle.

**7.2 Tool Registry** — 4 tools with mock implementations:
```python
TOOLS = {
    "search":       search_web,      # Mock → replace with Tavily / SerpAPI
    "calculate":    calculate,       # Safe eval() with blocked builtins
    "convert":      convert_units,   # Token ↔ USD, GB ↔ MB
    "analyze_code": analyze_code,    # Line count, functions, imports, patterns
}
```

**7.3 ReAct System Prompt** — The contract between you and the agent:
```
STRICT FORMAT:
Thought: [reasoning — what do you know? what do you need?]
Action: tool_name("parameters")

OR:
Thought: [final reasoning]
Final Answer: [complete answer]

RULES:
1. Always start with "Thought:" — never skip
2. One Action per response — no chaining actions in one turn
3. Never hallucinate tool results — wait for the Observation
```

**7.4 Robust Action Parser** — Handles multiple formats the LLM might generate:
```python
patterns = [
    r'Action:\s*(\w+)\(["\'`](.*?)["\'`]\)',   # tool("param")
    r'Action:\s*(\w+)\(([^)]+)\)',              # tool(expr) — no quotes
    r'Action:\s*(\w+):\s*["\'`]?(.*?)["\'`]?$' # tool: param
]
```

**7.5 Full Agent Loop** — with `AgentStep` dataclass for full tracing:
- Token usage & latency tracked per step
- Observation truncated at 500 chars to prevent context explosion
- Error recovery: tool errors fed back as Observation so model adapts
- Max steps guard to prevent runaway loops

**7.7 Retry & Model Fallback:**
```python
# Strategy: try SMART_MODEL → fall back to DEFAULT_MODEL → graceful error
models_to_try = [SMART_MODEL] + [DEFAULT_MODEL] * max_retries
```

**7.8 Exercises:**
- **7A**: Add `get_datetime(timezone)` using `zoneinfo` — IANA timezone names
- **7B**: Add `database_query(sql)` on in-memory SQLite seeded with AI model pricing
- **7C**: Detect tool-call loops (same tool + same input twice) and force a strategy change

---

### Section 8 — Advanced Techniques

#### 8.1 Three Pillars

| Technique | Best for | Cost multiplier | Quality gain |
|-----------|----------|----------------|-------------|
| Single-shot | Speed-sensitive tasks | 1× | baseline |
| Self-Critique (2-step) | Quality-sensitive tasks | ~2.5× | +15–25% |
| Self-Critique (3-step) | High-stakes content | ~4× | +25–40% |
| Constitutional (5+ principles) | Safety/compliance | ~5× | +30–50% |
| Meta-Prompting | Prompt optimization | ~2× | +5–15% |

#### 8.2 Meta-Prompting — LLMs Writing Prompts

```python
META_PROMPT_TEMPLATE = """You are a world-class Prompt Engineer with expertise in {model}.

Generate the OPTIMAL system prompt for this use case:
- Task:             {task}
- Output Format:    {output_format}
- User Persona:     {user_persona}
- Constraints:      {constraints}
- Quality Criteria: {quality_criteria}

Required components: specific role, step-by-step instructions, exact output format,
3 diverse inline examples, error handling for out-of-scope inputs, quality guardrails.

Output between <prompt> tags. Explain decisions in <reasoning>.
"""
```
Research (Zhou et al., 2022): Meta-Prompting outperforms human-written prompts by **5–15%** on BIG-Bench tasks.

#### 8.3 Self-Critique & Reflection Pipeline

```
Step 1 Generate  : First draft (gpt-4o-mini — cheap)
Step 2 Critique  : Identify issues against explicit criteria (gpt-4o — smart)
Step 3 Revise    : Fix each issue in a complete rewrite (gpt-4o — smart)
```

Using **different models per step** is intentional — cheap for generation, smart for critique.

The `CritiqueResult` dataclass captures: `task`, `initial_output`, `critique`, `improved_output`, `overall_score`.

#### 8.4 Constitutional AI — Principle-Based Refinement

Evaluate every response against a set of explicit principles ("the constitution"):
```python
AI_AGENT_CONSTITUTION = [
    "Accuracy: All technical claims must be verifiable; no speculation as fact",
    "Safety: Never suggest approaches causing data loss or security vulnerabilities",
    "Completeness: Cover the full scope of the question",
    "Practicality: Solutions must be implementable with standard Python libraries",
    "Clarity: Code examples must have comments; jargon explained on first use",
    "Neutrality: When comparing tools, present objective trade-offs, not opinions",
]
```
The model checks each principle and returns PASS/PARTIAL/FAIL per principle, then rewrites the response to fix failures. Stops early if all principles pass.

---

### Section 9 — Prompt Evaluation & A/B Testing

#### 9.1 LLM-as-Judge

Use a stronger LLM to evaluate outputs of another LLM against a rubric. Produces a score (0.0–1.0) and per-criterion pass/fail.

#### 9.2 Evaluation Framework

```python
@dataclass
class EvalCase:
    input:           str
    expected_output: str    # For exact-match scoring
    scoring_rubric:  str    # For LLM-as-judge scoring
    category:        str    # "easy" / "edge_case" / "adversarial"
    weight:          float = 1.0
```

A full `SENTIMENT_EVAL_SUITE` of test cases covers:
- **Easy cases** — unambiguous positive/negative
- **Edge cases** — sarcasm, surprised positive, numeric ratings
- **Adversarial** — contains the wrong label word in the text

#### 9.3 A/B/C Testing

`PromptEvaluator.evaluate_prompt(prompt, prompt_name)` runs all eval cases and returns:
```python
{
    "mean_score":       0.87,
    "pass_rate":        0.85,  # fraction with score >= 0.7
    "min_score":        0.60,
    "avg_latency_ms":   234.5,
    "p95_latency_ms":   410.0,
    "total_cost_usd":   0.00234,
    "cost_per_case":    0.000078,
    "by_category":      {"easy": 0.95, "edge_case": 0.82, "adversarial": 0.72}
}
```

Comparison table printed with winner per metric:
```
Prompt A (Minimal) vs Prompt B (Structured) vs Prompt C (Few-shot)
→ Compared on: mean_score, pass_rate, avg_latency_ms, cost_per_case
```

#### 9.4 Prompt Regression Suite — CI/CD for Prompts

```python
suite = PromptRegressionSuite(
    baseline_prompt=PROMPT_A,
    eval_cases=SENTIMENT_EVAL_SUITE,
    thresholds={
        "mean_score_min":   0.70,   # Quality floor
        "pass_rate_min":    0.65,   # Pass-rate floor
        "score_regression": 0.03,   # Block if score drops > 3%
    }
)
suite.establish_baseline()
decision = suite.test_candidate(PROMPT_C)
# → "APPROVED — Safe to deploy" or "BLOCKED — Do not deploy"
```

**Typical CI/CD workflow:**
1. Developer proposes prompt change
2. CI runs regression suite
3. If `mean_score` drops > 3% or `pass_rate` drops > 5% → **BLOCK** deploy
4. If metrics improve → **ALLOW** deploy with change log entry

---

### Section 10 — Production Systems

#### 10.1 Semantic Caching

```python
cache_key = hashlib.md5(f"{prompt}:{inputs}".encode()).hexdigest()
# Cache hit rate in production: typically 30–60% of traffic
# Saves API cost proportionally
```

TTL-based expiry configurable per use case.

#### 10.2 SmartPromptManager

Full production-grade prompt management:

```python
class PromptVersion(BaseModel):
    name:             str          # "sentiment"
    version:          str          # "v1.1"
    description:      str
    author:           str
    system_template:  str          # Jinja2 template
    user_template:    str
    model:            str
    temperature:      float
    variables:        list[str]
    changelog:        str
    deprecated:       bool = False
```

Features:
- **Version Control** — `register()`, `deprecate()`, rollback by calling an older version
- **Semantic Cache** — MD5 hash of rendered prompt + user template, TTL expiry
- **Stats Dashboard** — tracks `total_calls`, `cache_hits`, `total_cost_usd`, `p95_latency_ms`, `error_count`

```python
manager = SmartPromptManager(cache_ttl_seconds=300)
manager.register(PromptVersion(name="sentiment", version="v1.1", ...))
manager.deprecate("sentiment", "v1.0")
result = manager.run("sentiment", version="v1.1", domain="e-commerce", text=review)
manager.print_dashboard()
```

#### 10.3 Async Batch Processing

```python
async def batch_classify_async(
    reviews: list[str],
    system_prompt: str,
    max_concurrent: int = 5     # Semaphore limits concurrent API calls
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [classify_one(r, i) for i, r in enumerate(reviews)]
    return await asyncio.gather(*tasks)

# Speedup: ~4–8x vs sequential for I/O-bound workloads
# 8 reviews: sequential ~8s → async ~2s
```

---

### Section 11 — Anti-Patterns ❌

| # | Anti-Pattern | Observed impact | Fix |
|---|-------------|-----------------|-----|
| 1 | **Vague Role** — "You are a helpful assistant" | No domain knowledge activation | Specific role with expertise + stakes |
| 2 | **Negative-Only Instructions** — "Don't be verbose" | Violated ~25–35% (Shah et al.) | Replace with positive equivalent |
| 3 | **Monolith Prompt** — one prompt doing 5 things | Tasks at the end get ~40% lower quality | Chain into specialized prompts |
| 4 | **No Output Constraints** — "Analyze this ticket" | Wastes tokens, breaks downstream parsing | Specify exact schema + field types |
| 5 | **Inconsistent Few-Shot Examples** — mixed formats | ~2× higher format error rate | All examples: identical structure |
| 6 | **Wrong Temperature** — `temp=0` for creative | Repetitive, canonical-only outputs | 0.0–0.2 deterministic / 0.7–1.0 creative |
| 7 | **Ignoring Model Quirks** | Same prompt: 30–40% performance difference across models | Optimize per model (GPT-4o uses function calling; Claude prefers XML tags; Gemini: CoT in user turn) |
| 8 | **No Evaluation Before Deploy** | 30% of regressions invisible to manual spot-checking | Eval suite ≥ 30 cases + regression guard in CI |

---

### Section 12 — Capstone: Production AI Code Review Agent

Combines **every technique from Part 1 and Part 2**:

| Component | Technique |
|-----------|-----------|
| Agent loop | ReAct Pattern (Section 7) |
| Prompt generation | Meta-Prompting (Section 8.2) |
| Quality improvement | Self-Critique Pipeline (Section 8.3) |
| Quality measurement | LLM-as-Judge Evaluation (Section 9) |
| State management | SmartPromptManager (Section 10) |
| High throughput | Async Batch Processing (Section 10.3) |

**Production Deploy Checklist:**
```
□ Eval suite >= 30 cases (easy + edge + adversarial)
□ A/B test at least 2 prompt versions with stats
□ Regression guard in CI/CD (block if score drops > 3%)
□ Semantic caching enabled (expect 30-60% hit rate)
□ Cost and P95 latency tracked per prompt version
□ Retry logic + fallback model configured
□ Async processing for any batch workloads
□ Prompt version changelog maintained in code review
```

---

## 🐍 Standalone Scripts

Runnable Python files that mirror the notebook content — useful for quick experimentation without Jupyter:

| File | What it demonstrates |
|------|---------------------|
| `utils.py` | `chat()` wrapper, `count_tokens()`, OpenAI client setup |
| `1. test_temperature.py` | `temperature=0.0` vs `temperature=1.0` side-by-side |
| `2. Zero shot and few shot prompting.py` | Zero-shot, few-shot, dynamic few-shot with keyword selection |
| `3. COT.py` | Zero-shot CoT, few-shot CoT, self-consistency, Tree-of-Thought |
| `4. Role Prompting and System Prompting.py` | Vague vs specific role, full system prompt anatomy, multi-persona |
| `5. Structured Output & JSON Mode.py` | JSON mode, Pydantic validation with `CodeReviewResult` |
| `6. Prompt template and Chaining.py` | Jinja2 templates, 3-step chain (parse → analyze → recommend) |
| `7. ReAct Pattern.py` | Placeholder — full implementation in Part 2 notebook |

---

## 💡 Key Mental Models

1. **An LLM is a function of everything in its context window.** Every technique is a way of engineering that context more deliberately.

2. **Few-shot, CoT, and ReAct are all forms of "giving the model more useful conditioning before it commits to an answer."**

3. **Structured output + validation is what turns a chatbot into a programmable component.**

4. **An "agent" is just an LLM call wrapped in a loop with tools and a parser.** Frameworks (LangChain, LangGraph, CrewAI) automate this loop — but you should always be able to explain what they're doing under the hood, because you build it yourself in Section 7.

5. **The eval set is ground truth.** Protect it like production data; never train on it.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥ 1.0 | OpenAI Chat Completions & Embeddings API |
| `pydantic` | ≥ 2.0 | Structured output validation |
| `jinja2` | ≥ 3.0 | Prompt templating |
| `tiktoken` | latest | Token counting (OpenAI tokenizer) |
| `python-dotenv` | latest | API key management |
| `aiohttp` | latest | Async HTTP (for batch processing) |
| `asyncio` | built-in | Concurrent LLM calls |
| `sqlite3` | built-in | In-memory DB tool for exercises |

---

## 📖 References

| Paper | Technique |
|-------|-----------|
| [Wei et al., 2022 — Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | CoT |
| [Wang et al., 2022 — Self-Consistency](https://arxiv.org/abs/2203.11171) | Self-Consistency |
| [Yao et al., 2023 — Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Tree-of-Thought |
| [Yao et al., 2022 — ReAct](https://arxiv.org/abs/2210.03629) | ReAct Pattern |
| [Anthropic, 2022 — Constitutional AI](https://arxiv.org/abs/2212.08073) | Constitutional AI |
| [Zhou et al., 2022 — Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) | Meta-Prompting |
| [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | General |

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**⭐ If this repo helped you, a star is appreciated!**

</div>