---
name: main2main
description: >-
  Adapt vLLM-Ascend to upstream vLLM main branch evolution via a CI-verified
  step pipeline. Use this skill whenever the user mentions upgrading, bumping,
  or syncing vllm-ascend to a newer vLLM commit, wants to analyze upstream
  vLLM changes and their impact on vllm-ascend, mentions main2main or
  schedule_test_vllm_main, or asks to adapt code for vLLM API changes.
  Also use when the user provides a vllm path and vllm-ascend path and
  wants to bring them in sync, even if they don't explicitly say "main2main".
---

# main2main

vllm-ascend is a hardware adaptation plugin for vLLM. As vLLM's main branch
evolves, vllm-ascend must keep up. This skill splits upstream commit changes
into manageable steps, adapts vllm-ascend code for each step, and only commits
after the e2e-main2main test suite passes.

## Inputs

The user provides (or the skill auto-detects):
- **vllm_path**: path to the local vLLM repository
- **vllm_ascend_path**: path to the local vllm-ascend repository

## Guardrails

These rules protect the repository's clean state. Every commit produced by
this skill should be verified and intentional.

- **Only modify files inside the vllm-ascend repo.** The vLLM repo is an
  upstream reference — we adapt to it, never modify it. If a fix seems to
  require changing vLLM code, the adaptation approach is wrong.

- **Keep intermediate files in /tmp/main2main/.** Analysis reports, logs,
  patches, and ledgers belong outside the repo. Files left inside the repo
  pollute commit history and risk being accidentally committed.

- **Use `git add <files>` instead of `git add .`.** Explicit staging ensures
  only intentional changes enter a commit. Glob-add easily drags in debug
  artifacts, log files, and analysis documents.

- **Commit only after CI passes.** Unverified code in the main branch breaks
  other developers. If CI does not pass, save the work-in-progress as a patch
  file instead of committing.

- **Advance the vllm repo after each successful step.** After committing a
  verified step, checkout the vllm repo to that step's last upstream commit.
  Subsequent CI runs need to execute against the correct vLLM version —
  skipping this produces misleading test results.

## Workflow Overview

The pipeline has 8 phases. Phases 1–2 are preparation. Phases 3–7 repeat for
each step. Phase 8 produces the final output.

### Phase 1: Init & Detect

Run the detection script to determine what needs upgrading:

    python3 <skill_dir>/scripts/detect_commits.py \
      --vllm-path <vllm_path> \
      --ascend-path <ascend_path>

The script reads `main_vllm_commit` from `vllm-ascend/docs/source/conf.py`
as the base commit, and uses the vLLM repo's HEAD as the target. If they
match (no drift), the pipeline ends here.

### Phase 2: Plan Steps

Run the step planner to split the commit range into steps:

    python3 <skill_dir>/scripts/plan_steps.py \
      --vllm-path <vllm_path> \
      --base-commit <base> \
      --target-commit <target>

Read `/tmp/main2main/steps.json` for the plan. The step boundaries are
determined by the planner's risk analysis — do not rearrange them.

Record `last_verified_head = current HEAD` before starting execution.

*Phases 3–7 repeat for each step in the plan.*

### Phase 3: Prepare Step Context

Generate the upstream diff for this step. This patch file is the central
reference for both the adapt and fix-ci phases.

    git -C <vllm_path> diff <step_start>..<step_end> \
      > /tmp/main2main/steps/<step-id>/upstream.patch

    git -C <vllm_path> diff --name-only <step_start>..<step_end> \
      > /tmp/main2main/steps/<step-id>/changed-files.txt

### Phase 4: Adapt

Read `adapt.md` in this skill directory. It contains the concrete method for
analyzing the upstream patch and mapping changes to vllm-ascend files — the
file mapping table, adaptation rules, and version compatibility patterns.

The core task: translate upstream.patch changes into vllm-ascend modifications
that maintain interface compatibility.

### Phase 5: Verify via CI

Run the e2e-main2main test suite:

    cd <ascend_path>
    python3 .github/workflows/scripts/run_suite.py \
      --suite e2e-main2main --continue-on-error \
      > /tmp/main2main/steps/<step-id>/ci/round-1.log 2>&1

If all tests pass, proceed to Phase 7. If any fail, enter Phase 6.

### Phase 6: Fix-CI Loop

Read `fix-ci.md` in this skill directory. It describes how to extract errors
from CI logs using `ci_log_summary.py`, diagnose root causes by
cross-referencing the upstream patch, and apply targeted fixes.

Each fix round re-runs CI (Phase 5). The loop continues until tests pass or
a stop condition is met:

1. Only environment flakes remain (not code issues) → treat as pass.
2. Two consecutive rounds with identical error signatures → stop.
3. No code diff produced in the current round → stop.
4. No actionable code_bugs in the error summary → stop.
5. Hard cap of 5 rounds → safety stop.

### Phase 7: Commit & Advance

If CI passed, commit the verified changes:

    python3 <skill_dir>/scripts/check_and_commit.py \
      --ascend-path <ascend_path> \
      --step-id <step-id> \
      --message "<commit message>"

Then advance the vllm repo to match this step's endpoint:

    git -C <vllm_path> checkout <step_end_commit>

Update `last_verified_head` to the new HEAD.

**If the fix-ci loop is exhausted** (stop condition triggered), execute a
partial stop instead:

    git diff > /tmp/main2main/steps/<step-id>/failed.patch
    # Write failed-summary.json with error details
    git checkout -- .   # rollback to last_verified_head

Do not skip the failed step and continue to the next one — stop immediately.

### Phase 8: Final Summary

After all steps complete (or after a partial stop), output a structured
summary covering: steps completed, commit SHAs, base→target progress, and
any failure details with patch paths.

## Edge Cases

- **No drift (base == target)**: Phase 1 reports `has_drift=false`. Output
  "no upstream drift detected" and end.

- **0 steps planned** (all commits are docs/tests only): Skip Phases 3–7.
  Just update the commit reference in conf.py and commit.

- **First step fails and cannot be fixed**: Partial stop with 0 verified
  steps. Save the patch for manual review.

- **vllm repo path doesn't exist**: The detect script exits with an error.
  Ask the user to confirm the path.

- **CI timeout**: If `run_suite.py` produces no output for 60+ minutes,
  terminate the process and treat it as a CI failure.

## Version Compatibility Pattern

When code must work with both old and new vLLM versions, use version guards:

    from vllm_ascend.utils import vllm_version_is

    if vllm_version_is(">=0.19.0"):
        # new version logic
    else:
        # old version logic

The version number comes from conf.py's `main_vllm_tag` field. Use semantic
version tags rather than commit hashes for clarity and stability.

## Pre-Completion Checklist

Before declaring the task complete, verify these items — they are the most
commonly missed issues based on past experience:

- [ ] No temporary files left in the repo (vllm_changes.md, .log, .patch, etc.)
- [ ] All commits use `git commit -s` (sign-off required)
- [ ] All intermediate files are in /tmp/main2main/, not in the repo
- [ ] conf.py `main_vllm_commit` updated to the final target commit reached
- [ ] conf.py `main_vllm_tag` updated if the tag changed
- [ ] Any new `vllm_version_is()` calls use the correct version number
- [ ] Each commit message includes the upstream commit range it adapts
- [ ] If partial stop: failed.patch and failed-summary.json are saved
- [ ] Final summary has been output

## Output Contract

The final output should be structured as:

```yaml
status: completed | partial
base_commit: <sha>
target_commit: <sha>
reached_commit: <sha>
steps_completed: N
steps_total: M
commits:
  - sha: <sha>
    step: step-1
    message: "..."
  - ...
partial_stop:            # only if status == partial
  step_id: step-K
  patch_path: /tmp/main2main/steps/step-K/failed.patch
  summary_path: /tmp/main2main/steps/step-K/failed-summary.json
  reason: "..."
```
