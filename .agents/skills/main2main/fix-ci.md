# Fix-CI Loop — Recovering from Test Failures

This document describes how to extract errors from CI logs, diagnose root
causes, and fix code. Read this when Phase 5 CI fails (entering Phase 6).

## Entry Condition

The e2e-main2main test suite returned a non-zero exit code. The raw log is at
`/tmp/main2main/steps/<step-id>/ci/<round>.log`.

## Step 1: Extract Errors

Use `ci_log_summary.py` to extract structured error information:

    python3 <ascend_path>/.github/workflows/scripts/ci_log_summary.py \
      --log-file /tmp/main2main/steps/<step-id>/ci/<round>.log \
      --format llm-json \
      --output /tmp/main2main/steps/<step-id>/ci/<round>-summary.json

## Step 2: Read the Summary

The summary JSON contains two categories of errors:

- **code_bugs** — code-level issues that need fixing. These are actionable.
- **env_flakes** — transient infrastructure failures (network timeouts, stale
  file handles, disk full, etc.). These cannot be fixed by code changes.

Focus on code_bugs. If only env_flakes remain, the CI is considered passing
in substance — proceed to Phase 7 (commit).

## Step 3: Diagnose and Fix

1. Read the `code_bugs` entries in the summary JSON.
2. Cross-reference with `/tmp/main2main/steps/<step-id>/upstream.patch` to
   determine whether the error originates from an upstream change. This is the
   reason adapt and fix share the same patch file — it provides root-cause
   traceability.
3. Consult `reference/error-patterns.md` for common failure patterns and
   proven fix strategies (signature changes, config moves, missing ops, etc.).
4. Apply the fix to vllm-ascend code.

## Step 4: Re-run CI

After fixing, re-run the test suite (back to Phase 5) with the next round
number. The log goes to `.../ci/<round+1>.log`.

## Progress Judgment

After each fix round, compare the error signatures with the previous round:

- **Fewer failing tests** → making progress, continue.
- **Same error signatures** for two consecutive rounds → fix is not effective,
  stop and trigger partial stop.
- **New errors appeared** that were not in the previous round → the fix
  introduced a regression. Revert this round's changes before continuing.

## Stop Conditions

The fix-ci loop stops when any of these conditions are met:

1. Only env_flakes remain → treat as pass.
2. Two consecutive rounds with identical error signatures → no progress.
3. This round produced no code diff → nothing to try.
4. ci_log_summary reports no actionable code_bugs → nothing to fix.
5. Hard cap of 5 rounds reached → safety fuse to prevent infinite loops.

## Round Ledger

Append a line to `/tmp/main2main/round-ledger.jsonl`:
```json
{"phase": "fix", "step": 1, "round": 2, "errors_before": 5, "errors_after": 3, "summary": "..."}
```
