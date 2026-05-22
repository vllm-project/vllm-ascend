# Diagnosis Guide

The goal of diagnosis isn't just "find the failing test" — it's to trace each failure back to the specific upstream change that caused it, so the fix addresses the root cause rather than the symptom.

**Write `vllm_error_analyze.md` immediately after Step 1** — start with just the skeleton (Overview table, error list), then fill in upstream commit details as Step 2 progresses. This ensures a useful record exists even if context runs low before finishing.

## Re-orient (every fix round, not just the first)

Re-read this file before each fix round. The error pattern table in Step 2 and
the `reference/error-pattern-examples.md` reference help match CI failures to
known fix patterns — this lookup produces better fixes than reasoning from the
error message alone, especially after multiple rounds when context is saturated.

Before starting, confirm:
- Current step, round number, compatible release tag (`main_vllm_tag`)
- Which issues from `vllm_error_analyze.md` are still open
- The upstream patch for this step: `/tmp/main2main/steps/<step-id>/upstream.patch`

---

## Step 1: Read structured CI output

`run_main2main_ci.py` already runs `ci_log_summary.py` after every CI round.
Start diagnosis from these files:

- `/tmp/main2main/steps/<step-id>/ci/round-<N>-result.json`
- `/tmp/main2main/steps/<step-id>/ci/round-<N>-summary.json`

Do not rerun `ci_log_summary.py` by hand during normal main2main execution. The
wrapper output is the source of truth for `ci_result`, `run_suite_exit_code`,
and the path to the structured summary.

The summary does the heavy lifting: it extracts root-cause exceptions, filters
wrapper errors (`Engine core initialization failed`), filters downstream effects
(`KeyError: 'choices'` caused by engine crash), and deduplicates by normalized
signature.

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

Only `code_bugs` need fixing. If only `env_flakes` remain, record CI as
`env_flake_pass` and proceed to commit.

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
| Compatible release | `<main_vllm_tag>` |
| Code bugs | <count> |
| Env flakes | <count> |

## Issues
| # | Error type | Message | Root cause commit | Version guard needed | Status |
|:---|:---|:---|:---|:---|:---|
| 1 | TypeError | forward_oot() got... | TBD | TBD | open |

## Details
(fill in during Step 2)
```

The **Version guard needed** column forces an explicit decision per issue —
fill in YES / NO / N/A during Step 2.

---

## Step 2: Root Cause Correlation and Apply Fixes

For each code bug from the script output, use the error type, message, and context to figure out how upstream changes caused it. Find the commit(s) that introduced the relevant change, then analyze the code diff to understand why it breaks vllm-ascend.

**1. Use the error type to narrow the mechanism:**
- `TypeError` → almost always a signature change (added/removed parameter)
- `AttributeError` → config field moved or renamed
- `ImportError` → module path changed
- `NotImplementedError` → new abstract method added to base class
- Unfamiliar downstream error (e.g., `KeyError: 'choices'`) → read the traceback upward to find the actual root cause

Then look up the matching pattern in `reference/error-pattern-examples.md` —
it has concrete fix examples for each error type. Don't skip this lookup in
later rounds; the reference covers edge cases (like signature consistency
across version-guarded branches) that are easy to miss when reasoning from
the error alone.

**2. Extract a search term from the error message** and search the step's upstream.patch:  `/tmp/main2main/steps/<step-id>/upstream.patch`

This reveals the diff chunk that introduced the change — not just the symptom, but the full context of what changed and why.

**3. Identify the intent of the upstream change.** Was it a rename? A removal? A new parameter? This determines the fix:
- New parameter → add to vllm-ascend's override with a default, use `vllm_version_is()` guard
- Removal → delete the usage from vllm-ascend, guarded by `vllm_version_is()` if release still has it
- Rename → update to new name with `vllm_version_is()` guard
- New abstract method → implement in `AscendPlatform` or relevant class

For each fix, decide whether a version guard is needed using the decision
tree in `reference/adapt-guide.md` Step 2.

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
| Version guard needed | **YES** — release uses old signature without `kv_cache_dtype` |

**Error Traceback:**
(use context from script output)

**Explanation:** <Why this change breaks vllm-ascend>

**Fix:** <Specific code change, with vllm_version_is() guard if YES above>
```

**5. Apply fixes**

For each issue, apply the fix from the plan above. Map each error to the
corresponding fix pattern in `reference/error-pattern-examples.md`.

---

## Step 3: Verify Before Re-running CI

Run `scripts/pre_ci_check.py` before re-running CI — same script as in the
adapt phase. Review any failures; a missing version guard on a YES issue from
`vllm_error_analyze.md` means the fix is incomplete.

```bash
python3 <skill_dir>/scripts/pre_ci_check.py \
  --ascend-path <ascend_path> \
  --release-tag <main_vllm_tag>
```

---

## Step 4: Re-run CI and Track Progress

Re-run CI using **Verify by CI** in SKILL.md. Then compare the new
`round-<N>-summary.json` error signatures with the previous round:

- **Fewer failing tests** → making progress, continue
- **Same error signatures two rounds in a row** → fix isn't working, trigger partial stop
- **New errors not in the previous round** → fix introduced a regression, revert and trigger partial stop

Update the Status column in `vllm_error_analyze.md` each round.

**Stop conditions** (authoritative list — check before each round):
1. Only `env_flakes` remain → record CI as `env_flake_pass`
2. Two consecutive rounds with identical error signatures → partial stop
3. This round produced no code diff → partial stop
4. No actionable `code_bugs` in summary → partial stop
5. Hard cap of 5 rounds → partial stop

---

## Context management

CI logs can be enormous. Never read raw logs into context:
- Always use `round-<N>-summary.json` first — it contains only structured output
- To read a specific section of the raw log: `grep -A 10 '<pattern>' <logfile> | head -30`
- Write `vllm_error_analyze.md` incrementally — it serves as external memory for this task. Re-orient by reading the file rather than reconstructing from context
- At the start of each fix round, re-read this file's Re-orient block and the open issues in `vllm_error_analyze.md` before writing any code