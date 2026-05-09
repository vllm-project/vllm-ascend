# Diagnosis Guide — How to Fix CI Failures

This is the reference material for the fix-ci phase. Come here when CI fails and you need the concrete commands or the progress judgment criteria. The thinking framework is in SKILL.md — this document is for lookup when you're in the middle of a fix round.

---

## Running ci_log_summary.py

The analysis script lives in the vllm-ascend repo. It processes raw CI logs into structured error data that fits in context without blowing your token budget.

```bash
python3 <ascend_path>/.github/workflows/scripts/ci_log_summary.py \
  --log-file /tmp/main2main/steps/<step-id>/ci/<round>.log \
  --format llm-json \
  --output /tmp/main2main/steps/<step-id>/ci/<round>-summary.json
```

### What the script does

- Parses failed test files and individual test case identifiers from pytest output
- Extracts root-cause exceptions (TypeError, AttributeError, ImportError, etc.)
- Skips wrapper errors (`Engine core initialization failed`, `Worker failed with error`)
- Filters downstream effects (`KeyError: 'choices'` caused by upstream engine crash)
- Detects environment flakes (Stale file handle, ConnectionResetError, filelock) even when embedded inside assertion messages
- Deduplicates errors by normalized signature (stripping PIDs, timestamps, addresses)

### Output format

```json
{
  "good_commit": "15d76f74...",
  "bad_commit": "6d4f9d3a...",
  "failed_test_files": ["tests/e2e/test_basic_correctness.py"],
  "failed_test_cases": ["tests/e2e/test_basic_correctness.py::test_chunked_prefill"],
  "code_bugs": [
    {
      "error_type": "TypeError",
      "error_message": "forward_oot() got an unexpected keyword argument 'kv_cache_dtype'",
      "category": "Code Bug",
      "context": ["...traceback lines..."],
      "error_failed_test_files": ["tests/..."],
      "error_failed_test_cases": ["tests/...::test_xxx"]
    }
  ],
  "env_flakes": [
    {
      "error_type": "OSError",
      "error_message": "Stale file handle",
      "category": "Environment Flake"
    }
  ]
}
```

Only `code_bugs` need fixing. If only `env_flakes` remain, the CI is passing in substance — proceed to commit.

---

## Root Cause Correlation

This is the step most people rush through and then waste time fixing symptoms. The goal is to connect each code_bug to a specific upstream change — not just "which line errors" but "what upstream commit caused this error."

### Method

1. **Start from the error_type.** The exception class tells you the mechanism:
   - `TypeError` → almost always a signature change
   - `AttributeError` → config field moved or renamed
   - `ImportError` → module path changed
   - Unfamiliar error → read the traceback upward to find the root cause

2. **Extract search terms from the error.** If the error says `forward_oot() got unexpected keyword argument 'kv_cache_dtype'`, search the upstream.patch for `kv_cache_dtype` or `forward_oot`.

3. **Find the upstream diff that introduced the change.**
   ```bash
   grep -n 'kv_cache_dtype' /tmp/main2main/steps/<step-id>/upstream.patch
   ```
   This gives you the full context of what changed — not just the error symptom.

4. **Understand the intent of the upstream change.** Was it a rename? A removal? A new parameter? This determines the fix strategy:
   - Rename → update vllm-ascend to use the new name
   - New parameter → add it to the OOT method signature with a default
   - Removal → delete the usage from vllm-ascend
   - Behavior change → adapt the override to match the new semantics

---

## Progress Judgment

After each fix round, compare error signatures with the previous round:

- **Fewer failing tests** → making progress, continue to next round.
- **Same error signatures for two consecutive rounds** → the fix is not working. Stop and trigger partial stop.
- **New errors that weren't in the previous round** → the fix introduced a regression. Revert this round's changes before trying something different.

---

## Stop Conditions

The fix-ci loop ends when any of these is true:

1. **Only env_flakes remain** → treat as pass, proceed to commit.
2. **Two consecutive rounds with identical error signatures** → no progress being made.
3. **This round produced no code diff** → nothing was attempted.
4. **ci_log_summary reports no actionable code_bugs** → nothing to fix.
5. **Hard cap of 5 rounds** → safety fuse against infinite loops.

When stopping due to conditions 2-5, execute a partial stop (save patch + summary, rollback to last_verified_head).

---

## Context Management

CI logs can be enormous (10K+ lines per job). Protect your context budget:

1. **Always use ci_log_summary.py first.** It processes logs in a subprocess and returns only structured results — keeping your context clean for analysis.
2. **Never pipe raw logs into your context.** If you need a specific section, use `grep -A 10 '<error_pattern>' <logfile> | head -30`.
3. **Write a diagnostic note early.** After parsing the summary JSON, jot down the key findings before diving into the patch. This helps if you need to resume after a context reset.


