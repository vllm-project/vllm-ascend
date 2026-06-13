#!/usr/bin/env python3
"""Test cases for the issue review prompt.

Run with:
    python .github/workflows/scripts/robot/tests/test_issue_review_prompt.py

Requires VLLM_BASE_URL and VLLM_API_KEY environment variables.
"""

import os
import sys
from pathlib import Path

import requests

# Add parent to path so we can import the step modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from call_llm import call_vllm
from prepare_system_prompt import load_system_prompt
from prepare_template import load_template

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Override for testing
os.environ["VLLM_BASE_URL"] = VLLM_BASE_URL
os.environ["VLLM_API_KEY"] = VLLM_API_KEY


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ── 1. Well-formed Bug Report ──────────────────────────────────────────
    {
        "name": "well_formed_bug",
        "issue_type": "[Bug]",
        "title": "[Bug]: DeepSeek V3 crashes with OOM on A2 when batch_size > 8",
        "body": """### Your current environment
<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.1.0
CANN version: 8.0.RC2
NPU: Ascend 910B2
vllm version: 0.6.3
vllm-ascend version: 0.1.0
```
</details>

### 🐛 Describe the bug
When running DeepSeek V3 with batch_size=16 on Atlas 800 A2, the process crashes with an out-of-memory error after processing ~100 tokens.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    max_num_seqs=16,
    gpu_memory_utilization=0.9,
)
prompts = ["Hello, how are you?"] * 16
outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
```

Error:
```
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 2.00 GiB
```
""",
    },
    # ── 2. Incomplete Bug Report (missing env) ─────────────────────────────
    {
        "name": "incomplete_bug_no_env",
        "issue_type": "[Bug]",
        "title": "[Bug]: Model inference is slow",
        "body": """### Your current environment
(not provided)

### 🐛 Describe the bug
The model runs very slowly. Please fix it.
""",
    },
    # ── 3. Empty Bug Report ────────────────────────────────────────────────
    {
        "name": "empty_bug",
        "issue_type": "[Bug]",
        "title": "[Bug]: It doesn't work",
        "body": """### Your current environment
```text
```
### 🐛 Describe the bug
It doesn't work.
""",
    },
    # ── 4. Installation Issue ──────────────────────────────────────────────
    {
        "name": "installation_issue",
        "issue_type": "[Installation]",
        "title": "[Installation]: pip install fails with CANN version mismatch",
        "body": """### Your current environment
```text
npu-smi 24.1.0
CANN: 8.0.RC1
Ubuntu 22.04
```

### How you are installing vllm and vllm-ascend
```sh
pip install vllm vllm-ascend
```

Error:
```
ERROR: Could not find a version that satisfies the requirement vllm-ascend
```
""",
    },
    # ── 5. Usage Question ──────────────────────────────────────────────────
    {
        "name": "usage_question",
        "issue_type": "[Usage]",
        "title": "[Usage]: How to run Qwen3-235B on multi-node A3 cluster?",
        "body": """### Your current environment
```text
CANN 8.0.RC2
Atlas 800 A3 x 4 nodes
```

### How would you like to use vllm on ascend
I want to run inference of Qwen3-235B-A22B across 4 A3 nodes with tensor parallelism. I tried setting --tensor-parallel-size 8 but got an error about NCCL initialization. What is the correct way to configure multi-node inference?
""",
    },
    # ── 6. Documentation Issue ─────────────────────────────────────────────
    {
        "name": "doc_issue",
        "issue_type": "[Doc]",
        "title": "[Doc]: Installation guide missing CANN 8.0.RC3 steps",
        "body": """### 📚 The doc issue
The installation guide at https://vllm-ascend.readthedocs.org only covers up to CANN 8.0.RC2. CANN 8.0.RC3 has been released and the installation steps have changed.

### Suggest a potential alternative/fix
Please add a section for CANN 8.0.RC3 with the updated pip install commands.
""",
    },
    # ── 7. Misc Issue ──────────────────────────────────────────────────────
    {
        "name": "misc_issue",
        "issue_type": "[Misc]",
        "title": "[Misc]: Request for ROCm support",
        "body": """### Anything you want to discuss about vllm on ascend.
Will vllm-ascend ever support AMD ROCm GPUs? We have a mixed cluster and would like to use the same serving stack.
""",
    },
    # ── 8. Bug with Chinese content ────────────────────────────────────────
    {
        "name": "bug_chinese",
        "issue_type": "[Bug]",
        "title": "[Bug]: GLM-4-9B 模型在 Ascend 910B 上输出乱码",
        "body": """### Your current environment
```text
CANN 8.0.RC2
Ascend 910B
```

### 🐛 Describe the bug
使用 GLM-4-9B-Chat 模型进行推理时，输出结果包含大量乱码字符。同样的代码在 NVIDIA GPU 上运行正常。

```python
from vllm import LLM, SamplingParams
llm = LLM(model="THUDM/glm-4-9b-chat")
outputs = llm.generate(["你好"], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
# 输出: 你好���æ˜¯ä¸€ä¸ªAIåŠ©æ‰‹ï¿½ï¿½
```
""",
    },
    # ── 9. Bug with very long body ─────────────────────────────────────────
    {
        "name": "bug_verbose",
        "issue_type": "[Bug]",
        "title": "[Bug]: KV cache corruption during long-context inference",
        "body": """### Your current environment
<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.1.0+git1234567
CANN version: 8.0.RC2.20240601
NPU driver: 24.1.0
NPU firmware: 1.82.0
vllm version: 0.6.3.post1
vllm-ascend version: 0.1.0.dev123
Python: 3.10.12
Ubuntu: 22.04.4 LTS
Kernel: 5.15.0-112-generic
```
</details>

### 🐛 Describe the bug

I am running long-context inference with a custom fine-tuned model based on DeepSeek-V2-Lite. After approximately 32K tokens of context, the model starts producing nonsensical output. I suspect KV cache corruption.

**Reproduction steps:**
1. Load model with max_model_len=65536
2. Feed a document of ~40K tokens
3. Ask a question about the document
4. Observe garbled output

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/fine-tuned-deepseek-v2-lite",
    max_model_len=65536,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

with open("long_document.txt") as f:
    doc = f.read()

prompt = f"Please summarize the following document:\\n\\n{doc}\\n\\nSummary:"
outputs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=500))
print(outputs[0].outputs[0].text)
```

**Expected behavior:** A coherent summary of the document.

**Actual behavior:** After ~32K tokens, output becomes random characters and repeated phrases.

**Additional context:**
- This does NOT happen on NVIDIA A100 with the same model and vLLM version.
- Reducing max_model_len to 32768 avoids the issue.
- I have tried both enforce_eager=True and False, same result.
- NPU memory usage is around 55GB out of 64GB during inference.
- I have attached the full npu-smi and dmesg output below.

```
[12345.678] npu 0: memory usage 55.2 GiB / 64.0 GiB
[12345.679] npu 1: memory usage 55.1 GiB / 64.0 GiB
```
""",
    },
    # ── 10. Minimal bug (barely any info) ──────────────────────────────────
    {
        "name": "minimal_bug",
        "issue_type": "[Bug]",
        "title": "[Bug]: crash",
        "body": "crashes when i run it",
    },
]

# ── 11. Edge case: wrong prefix format ────────────────────────────────────
# (This tests the extraction logic, not the LLM)
EDGE_CASES = [
    {
        "name": "no_prefix",
        "title": "This is just a regular issue without prefix",
        "body": "Some content",
    },
    {
        "name": "lowercase_prefix",
        "title": "[bug]: Something is broken",
        "body": "Details here",
    },
    {
        "name": "extra_spaces",
        "title": "[Bug] : Extra space after prefix",
        "body": "Details here",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test_case(case: dict, system_prompt: str) -> dict:
    """Run a single test case and return the result."""
    issue_type = case["issue_type"]
    template_text = load_template(issue_type)

    user_prompt = f"""## Issue Title
{case['title']}

## Issue Template (what the user was asked to fill in)
{template_text}

## Issue Content (what the user actually submitted)
{case['body']}

Please review this issue and provide your feedback in the specified format.
"""

    try:
        review = call_vllm(system_prompt, user_prompt)
        return {"name": case["name"], "status": "OK", "review": review}
    except Exception as e:
        return {"name": case["name"], "status": f"ERROR: {e}", "review": ""}


def main():
    system_prompt = load_system_prompt()

    print("=" * 80)
    print("ISSUE REVIEW PROMPT TEST SUITE")
    print(f"vLLM endpoint: {VLLM_BASE_URL}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("=" * 80)

    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"\n{'─' * 80}")
        print(f"[{i+1}/{len(TEST_CASES)}] {case['name']}")
        print(f"Title: {case['title']}")
        result = run_test_case(case, system_prompt)
        results.append(result)
        print(f"Status: {result['status']}")
        if result["review"]:
            print(f"\nReview output:\n{result['review'][:500]}...")
            if len(result["review"]) > 500:
                print(f"(truncated, total {len(result['review'])} chars)")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] != "OK")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailures:")
        for r in results:
            if r["status"] != "OK":
                print(f"  - {r['name']}: {r['status']}")

    # Edge case extraction tests (no LLM call needed)
    print(f"\n{'─' * 80}")
    print("EDGE CASE: Title prefix extraction")
    from extract_input import extract_issue_type

    for ec in EDGE_CASES:
        result = extract_issue_type(ec["title"])
        status = "✓" if result is None else f"✗ (got {result})"
        print(f"  {ec['name']}: {status}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
