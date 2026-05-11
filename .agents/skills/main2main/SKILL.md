---
name: main2main
description: >-
  Adapt vLLM-Ascend to track upstream vLLM main branch changes. Use whenever
  the user mentions upgrading, syncing, or bumping vllm-ascend to a newer vLLM
  commit, or provides both a vllm path and vllm-ascend path for syncing. Also
  triggers on keywords: main2main, vLLM API changes.
---

# main2main

vllm-ascend is a hardware adaptation plugin that sits on top of vLLM. When upstream vLLM changes — function signatures, config fields, module paths, base class methods, etc. — vllm-ascend breaks. This skill's job is to absorb those upstream changes incrementally: split the commit range into manageable steps, adapt vllm-ascend for each step, verify via CI, and commit only verified code.

The two hardest parts are **figuring out what to adapt** and **diagnosing why CI fails after adapting**. Everything else (detecting commits, planning steps, running CI, committing) is mechanical and handled by scripts. This document focuses on the judgment calls.

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

## Adaptation and CI Diagnosis

Each step has two phases: **adapt** (proactively modify vllm-ascend based on upstream.patch) and **fix** (react to CI failures your adaptation missed).

**Adapt:** Read upstream.patch, use `changed-files.txt` to identify which subsystems are affected, then find and update the corresponding code in vllm-ascend. The file mapping table and key areas are in `reference/adapt-guide.md`.

When adapting, the goal isn't to copy upstream changes mechanically — it's to ask "how does this upstream change affect the contract between vLLM and vllm-ascend?" For example:

> **Upstream** adds a new abstract method `get_cache_config()` to the Platform base class.
> **Wrong approach**: Ignore it because no test currently calls it.
> **Right approach**: Check whether `AscendPlatform` inherits from this class — if yes, vllm-ascend will fail to instantiate at runtime with `TypeError: Can't instantiate abstract class`. Add the method immediately, even if the body is a stub.

The pattern to internalize: upstream changes to **abstract methods, function signatures, and config field locations, etc.** always need vllm-ascend follow-up, because vllm-ascend overrides or reads these directly. Changes to **internal implementation** of methods vllm-ascend doesn't override can usually be skipped.

No-op adapt is allowed, but it does not skip CI: every step must run CI after the commit reference is updated.

**Fix:** When CI fails, run `ci_log_summary.py` to get structured error data. Focus only on `code_bugs` (ignore `env_flakes`). For each bug, search `upstream.patch` for the function/class/field name from the error — this connects the symptom to the upstream change, giving you the full picture. Fix based on that complete context, not just the error message. Detailed diagnostic workflow and progress judgment are in `reference/diagnosis-guide.md`.

The fix loop ends when any of these is true — check before each round:
1. Only `env_flakes` remain → treat as pass, proceed to commit
2. Two consecutive rounds with identical error signatures → partial stop
3. This round produced no code diff → partial stop
4. `ci_log_summary` reports no actionable `code_bugs` → partial stop
5. Hard cap of 5 rounds → partial stop

**Context management:** CI logs can be 10K+ lines. Never read raw logs into context — always use `ci_log_summary.py` first. If you need a specific log section, filter with `grep -A 10 '<pattern>' <log> | head -30`.

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
The version info source of truth is `vllm-ascend/docs/source/conf.py`. The compatible release version for `vllm_version_is()` guards comes from the `main_vllm_tag` .

Three rules that prevent subtle maintenance debt:

1. **Use `vllm_version_is()` — not `hasattr()`, not `try/except`, not a boolean flag.** The version string is the source of truth. `hasattr` hides the version boundary and makes future cleanup impossible to grep for.

2. **Call it at each branch point.** If two files diverge by version, each one imports and calls `vllm_version_is()` directly. Don't set a flag in one place and read it elsewhere — that turns a version boundary into a capability toggle that future maintainers won't know to delete.

3. **When in doubt, grep the existing codebase.** `grep -rn 'vllm_version_is' vllm_ascend/` shows how other version guards are structured. Follow the established pattern.

---

## Pipeline Execution

Most of this is scripted — scripts have `--help` for argument details. This section describes the flow and the non-obvious decisions at each phase.

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

If `has_drift` is false, stop — nothing to do.

Record `last_verified_head` before starting.

### Phase 3–7: Per-step loop

For each step in `steps.json`:

1. **Generate upstream patch** 
```bash
git -C <vllm_path> diff <step_start>..<step_end> \
  > /tmp/main2main/steps/<step-id>/upstream.patch
git -C <vllm_path> diff --name-only <step_start>..<step_end> \
  > /tmp/main2main/steps/<step-id>/changed-files.txt
```

2. **Update commit references.** 
Replace the previous vLLM commit hash with this step's target commit before CI — tests may depend on the correct version reference.
```bash
python3 <skill_dir>/scripts/update_commit_reference.py \
  --ascend-path <ascend_path> \
  --old-commit <step_start_commit> \
  --new-commit <step_end_commit>
```

3. **Adapt.** 
Read the patch, identify affected subsystems, update vllm-ascend code if needed. See "Adaptation and CI Diagnosis" above and `reference/adapt-guide.md`. If no code adaptation is needed, record that conclusion and continue to CI.

4. **Verify by CI (mandatory for every step)**
Run CI after the commit reference update and adapt phase, even when adapt made no extra code changes. A step is only complete after CI passes.

```bash
set -o pipefail
python3 <ascend_path>/.github/workflows/scripts/run_suite.py \
  --suite e2e-main2main --continue-on-error \
  2>&1 | tee /tmp/main2main/steps/<step-id>/ci/round-1.log
```

5. **If CI passes, commit and advance.**
Run `scripts/check_and_commit.py` to commit the changes (including the updated commit reference). Then checkout the step's end commit in the vLLM repo so the next step runs against the correct upstream.
```bash
python3 <skill_dir>/scripts/check_and_commit.py \
  --ascend-path <ascend_path> --step-id <step-id> \
  --message "<commit message>"
git -C <vllm_path> checkout <step_end_commit>
```

6. **If CI fails, diagnose and fix.** Follow `reference/diagnosis-guide.md` for the diagnostic workflow. Re-run CI after each fix round. Stop conditions listed above.

**If the fix loop is exhausted** (stop conditions triggered in "Adaptation and CI Diagnosis"):
- Save current changes: `git diff > /tmp/main2main/steps/<step-id>/failed.patch`
- Write failure details to `/tmp/main2main/steps/<step-id>/failed-summary.json`
- Rollback to `last_verified_head`: `git checkout -- .`
- Stop the pipeline. Don't skip the failed step and continue to the next one.

**If CI hangs (no output for 120+ minutes):** terminate and treat as CI failure.

7. **Write step summary.** After committing, write a brief summary for this step: what upstream changes were absorbed, what vllm-ascend files were modified, and any version guards added. Save to `/tmp/main2main/steps/<step-id>/summary.md`. This is important because later steps and the final report depend on it.

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
- [ ] conf.py `main_vllm_commit` updated at each step (not just at the end)
- [ ] conf.py `main_vllm_tag` updated if the tag changed
- [ ] Every step ran CI after the commit reference update, including no-op adapt steps
- [ ] New `vllm_version_is()` calls use the correct version
- [ ] Each commit message includes the upstream commit range
- [ ] Each step has a summary in `/tmp/main2main/steps/<step-id>/summary.md`
- [ ] If partial stop: patch + failure details saved
- [ ] Final summary output to user
