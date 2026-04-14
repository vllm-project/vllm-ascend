# CI Failure Diagnosis Workflow

Diagnose and fix vLLM-Ascend CI failures caused by upstream vLLM main branch evolution. This implements a 4-phase pipeline: log mining, change analysis, report generation, and automated fix.

## Prerequisites (run these first, before Phase 1)

Before starting, verify the `gh` CLI is installed and authenticated:

```bash
gh auth status
```

If not authenticated, instruct the user to run `gh auth login`.

Also verify you are inside the vllm-ascend repository:

```bash
git rev-parse --show-toplevel  # Should end with vllm-ascend
```

Locate the vLLM upstream repo, if not found, prompt the user to specify the exact path to the vLLM git repository.

Before Phase 2, ensure the vllm repo has both the good and bad commits:

```bash
git cat-file -t <GOOD_COMMIT>  
git cat-file -t <BAD_COMMIT> 
```

---

## Token Budget Strategy

CI logs can be enormous (10K+ lines per job). To avoid exhausting your context on raw log text:

1. **Always use the repository summary script** for Phase 1. It processes logs in a subprocess and returns only the structured results — keeping your context clean for the higher-value Phase 2/3 analysis.
2. **Write a partial report early.** After Phase 1, immediately write a skeleton `vllm_error_analyze.md` with the Overview table, failed jobs, and error list. Then fill in the upstream commit details as you complete Phase 2. This ensures the user gets a useful report even if you run low on budget.
3. **Use the local vLLM repo** for all upstream code analysis (Phase 2). Run `git log`, `git diff`, `git show`, and read files directly from `$VLLM_LOCAL_DIR` — this is faster and more reliable than GitHub API calls, and avoids rate limits.
4. **If falling back to manual mode**, never pipe full logs into context. Always filter through `grep` with `head` limits first.

---

## Phase 1: Fault Context Acquisition — Use the Repository Script

This skill relies on `.github/workflows/scripts/ci_log_summary.py` to summarize failed jobs, failed tests, and distinct root-cause errors from a GitHub Actions run or local pytest log. **Always run the script first** to avoid wasting tokens on manual log parsing. The script prepares the failure inventory for Phase 1; Phase 2 upstream commit correlation is still manual.

### 1.1 Run the Summary Script

Run one of the following commands from the `vllm-ascend` repository root:

```bash
# With a specific run ID:
python3 <VLLM_ASCEND_DIR>/.github/workflows/scripts/ci_log_summary.py \
  --run-id <RUN_ID> \
  --format llm-json \
  --output /tmp/ci_analysis.json
```

The script will:

- Download logs for each completed non-skipped job when `--run-id` is provided
- Extract the **bad commit** from the vLLM version string in the logs (e.g., `vLLM 0.1.dev1+g6d4f9d3ad.empty` → `6d4f9d3ad`)
- Extract the **good commit** from `.github/workflows/pr_test_full.yaml` (the `vllm_version` matrix field)
- Parse failed test files and failed test cases from pytest summary output
- Extract root-cause exceptions (TypeError, AttributeError, ImportError, etc.)
- Skip wrapper errors (`Engine core initialization failed`, `Worker failed with error`)
- Filter downstream effects (`KeyError: 'choices'` caused by upstream engine crash)
- Detect environment flakes (`Stale file handle`, `ConnectionResetError`, `filelock` errors) — even when embedded inside assertion messages
- Deduplicate errors by normalized signature (stripping PIDs, timestamps, addresses, errno numbers)
- Output a structured JSON summary for LLM consumption

If the user does not provide a run ID or a log file, obtain the run ID first using `gh` commands, then invoke the script. Do not pretend the script can auto-discover the latest failed run by itself.

### 1.2 Read the Script Output

Load `/tmp/ci_analysis.json` and extract the key fields:

```json
{
  "run_id": 21646698906,
  "good_commit": "15d76f74e2fdb12a95ea00f0ca283acf6219a2b7",
  "bad_commit": "6d4f9d3ad5aa3750697edcf013ad080619ae25e9",
  "failed_test_files": ["tests/..."],
  "failed_test_cases": ["tests/...::test_xxx"],
  "code_bugs": [
    {
      "error_type": "TypeError",
      "error_message": "...",
      "category": "Code Bug",
      "context": [...],
      "error_failed_test_files": ["tests/..."],
      "error_failed_test_cases": ["tests/...::test_xxx"]
    }
  ],
  "env_flakes": []
}
```

**Phase 1 outputs:** `RUN_ID`,`GOOD_COMMIT`, `BAD_COMMIT`, `failed_test_cases`, `code_bugs`, and `env_flakes`

---

## Phase 2: Change Comparison & Adaptation Analysis

The goal is to **map each code bug to the specific upstream vLLM commit** that caused it. Only analyze `code_bugs`, not `env_flakes`.

All commands in this phase run against the **local vLLM repo** (`$VLLM_LOCAL_DIR`).

### 2.1 Get the Commit Diff

Compare changed files between good and bad commits under `vllm/vllm/` directory:

```bash
git diff  <GOOD_COMMIT>..<BAD_COMMIT> --name-only
```

List commits in the range:

```bash
git log --oneline <GOOD_COMMIT>..<BAD_COMMIT>
```

Focus on files in these critical paths:

- `vllm/platforms/` — Platform interface changes
- `vllm/model_executor/layers/attention/` — Attention backends
- `vllm/model_executor/layers/fused_moe/` — MoE layer
- `vllm/model_executor/layers/layernorm.py` — Normalization ops
- `vllm/model_executor/custom_op.py` — Custom op registration
- `vllm/v1/worker/` — Model runner and workers
- `vllm/distributed/` — Distributed communication
- `vllm/config*.py` — Configuration
- `vllm/compilation/` — Compilation passes

### 2.2 Root Cause Correlation

For each code bug from the script output, use the error type, message, and context to figure out how upstream changes caused it. Find the commit(s) that introduced the relevant change, then analyze the code diff to understand why it breaks vllm-ascend.

### 2.3 File Impact Mapping

Map vLLM changes to their vllm-ascend counterparts:

| vLLM Source Path | vllm-ascend Target Path |
|:---|:---|
| `vllm/platforms/` | `vllm_ascend/platform.py` |
| `vllm/model_executor/layers/attention/` | `vllm_ascend/attention/`, `vllm_ascend/ops/mm_encoder_attention.py` |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/moe.py` |
| `vllm/model_executor/layers/layernorm.py` | `vllm_ascend/ops/layernorm.py` |
| `vllm/model_executor/custom_op.py` | `vllm_ascend/ops/` (any file registering custom ops) |
| `vllm/v1/worker/gpu/model_runner.py` | `vllm_ascend/worker/model_runner_v1.py`, `vllm_ascend/worker/v2/model_runner.py` |
| `vllm/v1/worker/gpu/spec_decode/` | `vllm_ascend/spec_decode/` |
| `vllm/distributed/` | `vllm_ascend/distributed/` |
| `vllm/config*.py` | `vllm_ascend/ascend_config.py` |
| `vllm/compilation/` | `vllm_ascend/compilation/` or config overrides |

**Phase 2 outputs:** For each code bug, the causal upstream commit(s), the changed vLLM file(s), and the affected vllm-ascend file(s).

---

## Phase 3: Generate Diagnostic Report

Write `vllm_error_analyze.md` in the repository root **as early as possible**. Start writing it right after Phase 1 completes — fill in the Overview, Failed Jobs Summary, and error list immediately. Then update the Issue Analysis sections with upstream commit details as you complete Phase 2. This incremental approach ensures a useful report exists even if you can't finish all the tracing.

Use the script output JSON to populate it — do not re-download logs.

```markdown
# vLLM-Ascend CI Failure Analysis Report

## Overview

| Item                      | Value                      |
| :------------------------ | :------------------------- |
| **Run URL**               | <url>                      |
| **Run Date**              | <date>                     |
| **Good Commit (pinned)**  | `<good_commit>`            |
| **Bad Commit (tested)**   | `<bad_commit>`             |
| **Total Failed Jobs**     | X / Y                      |
| **Distinct Issues Found** | N code bugs + M env flakes |

## Failed Jobs Summary

| Job        | Conclusion | Failed Tests     |
|:---        |:---        |:---              |
| <job_name> | failure    | <test1>, <test2> |

## Issue Analysis

### Issue 1: <Short Description>

| Item                      | Detail                                   |
| :------------------------ | :--------------------------------------- |
| **Category**              | Code Bug / Environment Flake             |
| **Error Type**            | <exception class>                        |
| **Affected Tests**        | <list>                                   |
| **Root Cause Commit**     | `<sha>` — "<commit message>" (<PR link>) |
| **Changed File**          | `<vllm file path>`                       |
| **Impact in vllm-ascend** | `<ascend file path>`                     |

**Error Traceback:**
(use context from script output)

**Explanation:** <Why this change breaks vllm-ascend>

**Fix Suggestion:** <Specific code change needed>

### Issue 2: ...

## Summary Table

| #    | Error | Category | Upstream Commit | Affected Tests | Fix  |
| :--- | :---- | :------- | :-------------- | :------------- | :--- |
| 1    | ...   | ...      | ...             | ...            | ...  |

## Recommended Actions

1. <action item>
2. <action item>
```

---

## Phase 4: Automated Fix & Output Summary

### 4.1 Apply Fixes

Only fix `Code Bug` issues. Skip `Environment Flake` issues entirely.

Map each error to the corresponding fix pattern in the Common Error Patterns Reference in `reference/error-patterns.md`, which documents frequent upstream vLLM evolution issues with concrete fix examples.

### 4.2 Version Compatibility Pattern

Most fixes require `vllm_version_is()` guards to maintain compatibility with both the pinned release version and main branch. The compatible release version comes from the `vllm_version` matrix in `.github/workflows/pr_test_full.yaml`:

```python
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.16.0"):  # pinned version
    # Use old API
else:
    # Use new API
```

This pattern appears throughout the Common Error Patterns below.

### 4.3 Output Fix Summary

After all fixes are applied, output a structured summary in the conversation. This summary serves as the skill's primary output — it's what a Workflow consumes, and what gets used as PR body content in standalone mode.

```markdown
### CI Fix Summary (run ID: <RUN_ID>)

**Commit range:** `<GOOD_COMMIT_SHORT>`..`<BAD_COMMIT_SHORT>`

#### Issues Fixed
| Error | Upstream Cause Commit | Affected Files | Fix Description |
|:---|:---|:---|:---|
| `TypeError: forward_oot() got unexpected kwarg 'X'` | `abc1234` — "refactor attention API" | `vllm_ascend/attention/` | Added `vllm_version_is()` guard |

#### Issues Skipped (Environment Flakes)
- `OSError: Stale file handle` — no code fix needed

#### Files Changed
- `vllm_ascend/attention/...`
- `.github/workflows/...`
```

The "Upstream Cause Commit" column is critical — it links each fix back to the specific vLLM commit that caused the breakage, identified during Phase 2.
