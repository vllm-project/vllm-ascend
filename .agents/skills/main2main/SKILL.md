---
name: main2main
description: >-
  Adapt vLLM-Ascend to track upstream vLLM main branch changes. Use whenever the user mentions upgrading, syncing, or bumping vllm-ascend to a newer vLLM commit, or provides both a vllm path and vllm-ascend path for syncing. Also triggers on keywords: main2main, schedule_test_vllm_main, vLLM API changes.
---

# main2main

vllm-ascend is a hardware adaptation plugin that sits on top of vLLM. When upstream vLLM changes its interfaces — function signatures, config fields, module paths, base class methods — vllm-ascend breaks. This skill's job is to absorb those upstream changes incrementally: split the commit range into manageable steps, adapt vllm-ascend for each step, verify via CI, and commit only verified code.

The two hardest parts are **figuring out what to adapt** and **diagnosing why CI fails after adapting**. Everything else (detecting commits, planning steps, running CI, committing) is mechanical and handled by scripts. This document focuses on the judgment calls.

## Read order

1. Read this file (SKILL.md) first — it has the thinking frameworks and the full pipeline.
2. During adapt (Phase 4): consult `reference/adapt-guide.md` for the file mapping table and subsystem details.
3. During fix-ci (Phase 6): consult `reference/diagnosis-guide.md` for ci_log_summary usage, progress judgment, and stop conditions.
4. When diagnosing specific errors: consult `reference/error-patterns.md` for common failure patterns and fix examples.

## Inputs

- **vllm_path**: local vLLM repository (upstream reference, read-only)
- **vllm_ascend_path**: local vllm-ascend repository (this is what we modify)

## Guardrails

These protect the repo. The reasoning behind each one matters more than the rule:

- **Only modify vllm-ascend.** vLLM is the upstream reference. If a fix seems to require changing vLLM code, the adaptation approach is wrong — step back and rethink.

- **Intermediate files go in /tmp/main2main/.** Patches, logs, analysis reports inside the repo will get accidentally committed. This has happened before.

- **`git add <files>`, never `git add .`.** Debug artifacts, log files, and analysis documents sitting in the working tree will silently enter the commit.

- **Commit only after CI passes.** Unverified code in main breaks other developers. If CI won't pass, save a `.patch` file instead.

- **Advance vllm after each step.** Each step's CI must run against the correct upstream version. If vllm stays on an old commit, tests pass for the wrong reasons.

---

## How to Think About Adaptation

Adapting is not mechanically copying upstream changes. Upstream changes affect vllm-ascend through several common patterns — understanding these helps you respond more precisely:

- **Interface contract change**: upstream changed a function signature, deleted a parameter, or added an abstract method. CI reports `TypeError` or `Can't instantiate abstract class`. Find all call sites: `grep -rn '<function_name>' vllm_ascend/`

- **Data flow path change**: upstream moved a config field between classes, or renamed a variable. CI reports `AttributeError` or `KeyError`. Trace every access path — don't just fix the one that errored, there are usually more.

- **Module topology change**: upstream moved a module, renamed a class, or split/merged files. CI reports `ImportError`. Check `git diff --name-status` for `R` (rename) markers.

- **Behavioral semantic change**: upstream didn't change the interface but changed internal behavior (e.g., return type changed from None to float). CI may report a confusing downstream error. Read upstream.patch to understand whether your override still satisfies the contract.

These are not exclusive categories — a single upstream commit can combine multiple patterns. The point is to help you look for the right signals in the patch, not to require formal classification.

When you get a step's `upstream.patch`, start by scanning `changed-files.txt` to see which subsystems are involved. `reference/adapt-guide.md` has the file mapping table and subsystem-specific notes.

---

## How to Think About CI Failures

CI fails because your adaptation missed something. The goal of diagnosis is not "find which test broke" but "identify which upstream change wasn't fully adapted."

**The reasoning path:**

1. Run `ci_log_summary.py` to get structured error data. Only care about `code_bugs` — `env_flakes` are infrastructure noise that can't be fixed with code changes.

2. For each code_bug, the `error_type` tells you the likely pattern:
   - `TypeError` → signature change (most common)
   - `AttributeError` → config/field relocation
   - `ImportError` → module path change
   - Unfamiliar error → read the traceback upward to find the real root cause; wrapper errors like `Engine core initialization failed` are not the root cause

3. Search the `upstream.patch` for the function/class/field name from the error. This is the critical step — it connects the symptom to the upstream change, giving you the complete picture of what changed (not just what broke).

4. Fix based on the full change context, not just the error message. A one-line patch that silences the error but doesn't match the upstream intent will create worse problems in the next step.

**Context management:** CI logs can be 10K+ lines. Never read raw logs into context — always use `ci_log_summary.py` first. If you need a specific log section, filter with `grep -A 10 '<pattern>' <log> | head -30`.

Details on running the summary script, progress judgment, and stop conditions are in `reference/diagnosis-guide.md`.

---

## Version Compatibility Rules

When code must work with both the release version and upstream main:

```python
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.19.0"):
    # release version API
else:
    # upstream main API
```

Three rules that prevent subtle maintenance debt:

1. **Use `vllm_version_is()` — not `hasattr()`, not `try/except`, not a boolean flag.** The version string is the source of truth. `hasattr` hides the version boundary and makes future cleanup impossible to grep for.

2. **Call it at each branch point.** If two files diverge by version, each one imports and calls `vllm_version_is()` directly. Don't set a flag in one place and read it elsewhere — that turns a version boundary into a capability toggle that future maintainers won't know to delete.

3. **When in doubt, grep the existing codebase.** `grep -rn 'vllm_version_is' vllm_ascend/` shows how other version guards are structured. Follow the established pattern.

---

## Pipeline Execution

The mechanics of running the pipeline. Most of this is scripted — you just need to invoke the right commands and connect the outputs.

### Phase 1–2: Detect and Plan (once)

```bash
# Detect base/target commits
python3 <skill_dir>/scripts/detect_commits.py \
  --vllm-path <vllm_path> --ascend-path <ascend_path>

# Plan steps (reads detect.json, outputs steps.json)
python3 <skill_dir>/scripts/plan_steps.py \
  --vllm-path <vllm_path> \
  --base-commit <base> --target-commit <target>
```

If `has_drift` is false, stop — nothing to do. If the plan has 0 steps (all commits are docs/tests), just update conf.py and commit.

Record `last_verified_head` before starting.

### Phase 3–7: Per-step loop

For each step in `steps.json`:

```bash
# 1. Generate upstream patch
git -C <vllm_path> diff <step_start>..<step_end> \
  > /tmp/main2main/steps/<step-id>/upstream.patch
git -C <vllm_path> diff --name-only <step_start>..<step_end> \
  > /tmp/main2main/steps/<step-id>/changed-files.txt

# 2. Adapt (see "How to Think About Adaptation" above, and reference/adapt-guide.md)

# 3. Run CI
python3 <ascend_path>/.github/workflows/scripts/run_suite.py \
  --suite e2e-main2main --continue-on-error \
  2>&1 | tee /tmp/main2main/steps/<step-id>/ci/round-1.log

# 4. If CI fails, diagnose and fix (see "How to Think About CI Failures")
#    Re-run CI after each fix round. Stop conditions in reference/diagnosis-guide.md.

# 5. If CI passes, commit and advance
python3 <skill_dir>/scripts/check_and_commit.py \
  --ascend-path <ascend_path> --step-id <step-id> \
  --message "<commit message>"
git -C <vllm_path> checkout <step_end_commit>
```

**If the fix loop is exhausted** (stop conditions triggered in Phase 6):
- Save current changes: `git diff > /tmp/main2main/steps/<step-id>/failed.patch`
- Write failure details to `/tmp/main2main/steps/<step-id>/failed-summary.json`
- Rollback to `last_verified_head`: `git checkout -- .`
- Stop the pipeline. Don't skip the failed step and continue to the next one.

**If CI hangs (no output for 60+ minutes):** terminate the process and treat as CI failure, entering Phase 6.

### Phase 8: Final summary

Output a structured summary:

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
partial_stop:              # only if status == partial
  step_id: step-K
  patch_path: /tmp/main2main/steps/step-K/failed.patch
  reason: "..."
```

---

## Pre-Completion Checklist

These are the things most commonly missed, based on past experience:

- [ ] No temp files in the repo (vllm_changes.md, .log, .patch, .jsonl)
- [ ] All commits signed (`git commit -s`)
- [ ] All intermediate files in /tmp/main2main/, not in the repo
- [ ] conf.py `main_vllm_commit` updated to final target reached
- [ ] conf.py `main_vllm_tag` updated if the tag changed
- [ ] New `vllm_version_is()` calls use the correct version
- [ ] Each commit message includes the upstream commit range
- [ ] If partial stop: patch + failed-summary.json saved
- [ ] Final summary output to user
