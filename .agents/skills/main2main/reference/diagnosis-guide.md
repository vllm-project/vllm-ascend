# Diagnosis Guide

The goal of diagnosis isn't just "find the failing test" — it's to trace each failure back to the specific upstream change that caused it, so the fix addresses the root cause rather than the symptom.

**Write `vllm_error_analyze.md` immediately after Step 1** — start with just the skeleton (Overview table, error list), then fill in upstream commit details as Step 2 progresses. This ensures a useful record exists even if context runs low before finishing.

---

## Step 1: Extract structured errors

Run ci_log_summary.py to turn raw CI logs into structured data:

```bash
# From a local log file (local CI run):
python3 <ascend_path>/.github/workflows/scripts/ci_log_summary.py \
  --log-file /tmp/main2main/steps/<step-id>/ci/<round>.log \
  --format llm-json \
  --output /tmp/main2main/steps/<step-id>/ci/<round>-summary.json
```

The script does the heavy lifting: it extracts root-cause exceptions, filters wrapper errors (`Engine core initialization failed`), filters downstream effects (`KeyError: 'choices'` caused by engine crash), and deduplicates by normalized signature.

**Relevant output fields:**

```json
{
  "good_commit": "...",
  "bad_commit": "...",
  "code_bugs": [
    {
      "error_type": "TypeError",
      "error_message": "forward_oot() got an unexpected keyword argument 'kv_cache_dtype'",
      "context": ["...traceback lines..."],
      "error_failed_test_cases": ["tests/...::test_xxx"]
    }
  ],
  "env_flakes": [{ "error_type": "OSError", "error_message": "Stale file handle" }]
}
```

Only `code_bugs` need fixing. If only `env_flakes` remain, treat as pass.

**Immediately write the skeleton of `vllm_error_analyze.md`:**

```markdown
# CI Failure Analysis — step-<N>, round-<M>

## Overview
| Item | Value |
|:---|:---|
| Step | step-<N> |
| Round | <M> |
| Good commit | `<sha>` |
| Bad commit | `<sha>` |
| Code bugs | <count> |
| Env flakes | <count> |

## Issues
| # | Error type | Message | Root cause commit | Status |
|:---|:---|:---|:---|:---|
| 1 | TypeError | forward_oot() got... | TBD | open |

## Details
(fill in during Step 2)
```

---

## Step 2: Trace each bug to its upstream cause

Don't just look at what line failed — find the upstream commit that introduced the breaking change. That context makes the fix complete rather than a patch.

For each `code_bug`:

**1. Use the error type to narrow the mechanism:**
- `TypeError` → almost always a signature change (added/removed parameter)
- `AttributeError` → config field moved or renamed
- `ImportError` → module path changed
- `NotImplementedError` → new abstract method added to base class
- Unfamiliar downstream error (e.g., `KeyError: 'choices'`) → read the traceback upward to find the actual root cause

**2. Extract a search term from the error message** and search the step's upstream.patch:

```bash
grep -n 'kv_cache_dtype' /tmp/main2main/steps/<step-id>/upstream.patch
grep -n 'forward_oot' /tmp/main2main/steps/<step-id>/upstream.patch
```

This reveals the diff chunk that introduced the change — not just the symptom, but the full context of what changed and why.

**3. Identify the intent of the upstream change.** Was it a rename? A removal? A new parameter? This determines the fix:
- New parameter → add to vllm-ascend's override with a default, use `vllm_version_is()` guard
- Removal → delete the usage from vllm-ascend
- Rename → update to new name everywhere with `vllm_version_is()` guard
- New abstract method → implement in `AscendPlatform` or relevant class

**4. Update `vllm_error_analyze.md`** with the root cause commit and fix plan:

```markdown
### Issue 1: TypeError in forward_oot()

| Item | Detail |
|:---|:---|
| Error | `TypeError: forward_oot() got an unexpected keyword argument 'kv_cache_dtype'` |
| Affected tests | `tests/e2e/test_basic_correctness.py::test_chunked_prefill` |
| Root cause commit | `abc1234` — "refactor attention forward signature" |
| Changed file | `vllm/model_executor/layers/attention/backends/abstract.py` |
| vllm-ascend file | `vllm_ascend/attention/ascend_attn_backend.py` |
| Fix | Add `kv_cache_dtype` parameter with `vllm_version_is()` guard |
```

For matching known error types to fix patterns, consult `reference/error-patterns.md`.

---

## Step 3: Apply fixes and track progress

After fixing, re-run CI. Then compare error signatures with the previous round:

- **Fewer failing tests** → making progress, continue
- **Same error signatures two rounds in a row** → fix isn't working, trigger partial stop
- **New errors not in the previous round** → fix introduced a regression, revert and trigger partial stop

Update the Status column in `vllm_error_analyze.md` each round.

**Stop conditions** (same as in SKILL.md):
1. Only `env_flakes` remain → treat as pass
2. Two consecutive rounds with identical error signatures → partial stop
3. This round produced no code diff → partial stop
4. No actionable `code_bugs` in summary → partial stop
5. Hard cap of 5 rounds → partial stop

---

## Context management

CI logs can be enormous. Never read raw logs into context:
- Always use `ci_log_summary.py` first — it processes in a subprocess and returns only structured output
- To read a specific section of the raw log: `grep -A 10 '<pattern>' <logfile> | head -30`
- Write `vllm_error_analyze.md` incrementally — it serves as external memory for this task. Re-orient by reading the file rather than reconstructing from context