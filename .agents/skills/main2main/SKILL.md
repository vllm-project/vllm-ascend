---
name: main2main
description: >-
  Adapt vLLM-Ascend to track upstream vLLM main branch changes incrementally:
  detect commit drift, plan steps, adapt code, run CI, and commit verified
  changes. Use whenever the user mentions main2main, upgrading or syncing
  vllm-ascend to a newer vLLM commit, vLLM API changes breaking vllm-ascend,
  or provides both a vllm path and vllm-ascend path for syncing. Also triggers
  on: "vLLM broke our plugin", "bump the vLLM commit", "ascend CI failing
  after upstream update".
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

- **Do not stop because the run is long.** Estimated duration, session length, model cost, or the number of remaining steps is not a stop condition. After a step is committed and vllm is advanced, immediately continue to the next planned step.

---

## Adaptation and CI Diagnosis

Each step has two phases: **adapt** (proactively modify vllm-ascend based on upstream.patch) and **fix** (react to CI failures your adaptation missed).

The detailed workflows, file mapping tables, error pattern references, and stop conditions are in `reference/adapt-guide.md` and `reference/diagnosis-guide.md`. Read them at every step and fix round — they contain per-step lookup tables that surface different vllm-ascend files for each step's patch, not one-time background reading.

The core judgment call in both phases: upstream changes to **abstract methods, function signatures, config field locations, and import paths** always need vllm-ascend follow-up, because vllm-ascend overrides or reads these directly. Changes to **internal implementation** of methods vllm-ascend doesn't override can be skipped — unless vllm-ascend calls that method and depends on its return value, side effects, or error behavior.

No-op adapt is allowed, but it does not skip CI: every step must run CI after the commit reference is updated.

The only valid partial-stop reasons are the stop conditions listed in `reference/diagnosis-guide.md` Step 4. Do not create a partial final summary because the remaining steps would take many hours, because the current session is long, or because additional CI runs are required.

**Context management:** CI logs can be 10K+ lines. Never read raw logs into context — use the `round-N-summary.json` produced by `run_main2main_ci.py` first. If you need a specific log section, filter with `grep -A 10 '<pattern>' <log> | head -30`.

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

## Execution playbook

Most of this is scripted — scripts have `--help` for argument details.

### Phase 1: Detect drift and Plan steps (once)

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

### Phase 2: Step-wise adaptation and verification

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
Read `reference/adapt-guide.md` and follow its Steps 1-3. The guide contains
file mapping tables and key area lists that identify different vllm-ascend
files for each step's patch — this lookup must happen every step, not just
the first. If no code adaptation is needed, record that conclusion and
continue to CI.

4. **Verify by CI (mandatory for every step)**
Run CI after the commit reference update and adapt phase, even when adapt made no extra code changes. A step is only complete after CI passes.

```bash
python3 <skill_dir>/scripts/run_main2main_ci.py \
  --ascend-path <ascend_path> \
  --step-id <step-id> \
  --suite e2e-singlecard-light \
  --suite e2e-2card-light \
  --suite e2e-4card-light \
  --round 1
```

The wrapper writes:
- `/tmp/main2main/steps/<step-id>/ci/round-1.log`
- `/tmp/main2main/steps/<step-id>/ci/round-1-summary.json`
- `/tmp/main2main/steps/<step-id>/ci/round-1-result.json`

Run the CI wrapper in the foreground and wait for it to finish; do not read raw
CI logs or `/tmp/claude-*/tasks/*.output` for progress monitoring.

Use `round-1-result.json` as the CI source of truth. It preserves the raw
`run_suite.py` status in `run_suite_exit_code` and classifies the main2main
outcome in `ci_result`.

CI result labels:
- `passed`: `ci_result` is `passed`; `run_suite_exit_code` is 0.
- `env_flake_pass`: `ci_result` is `env_flake_pass`; `run_suite_exit_code`
  is non-zero, but the summary has only `env_flakes` and no `code_bugs`. This
  may proceed to commit, but do not call it `passed`.
- `failed`: `ci_result` is `failed`; CI failed and the summary is available for
  diagnosis.
- `summary_error`: `ci_result` is `summary_error`; do not commit. Fix the
  summary/log extraction issue or treat the step as a partial stop if no
  actionable diagnosis can be produced.

5. **If CI result allows commit, commit and advance.**
Proceed only when `ci_result` is `passed` or `env_flake_pass`.
Run `scripts/check_and_commit.py` to commit the changes (including the updated commit reference). Then checkout the step's end commit in the vLLM repo so the next step runs against the correct upstream.
```bash
python3 <skill_dir>/scripts/check_and_commit.py \
  --ascend-path <ascend_path> --step-id <step-id> \
  --message "<commit message>"
git -C <vllm_path> checkout <step_end_commit>
```

6. **If CI result does not allow commit, diagnose and fix.**
Read `reference/diagnosis-guide.md` and follow its Steps 1-4. The guide
contains error type → fix pattern mappings and references
`reference/error-pattern-examples.md` for concrete fix examples — use these
lookups each round rather than reasoning from the error message alone. Re-run
CI after each fix round. Stop conditions are listed in the diagnosis guide.

**If the fix loop is exhausted** (a stop condition from `reference/diagnosis-guide.md` is triggered):
- Save current changes: `git diff > /tmp/main2main/steps/<step-id>/failed.patch`
- Write failure details to `/tmp/main2main/steps/<step-id>/failed-summary.json`
- Rollback to `last_verified_head`: `git checkout -- .`
- Stop the pipeline. Don't skip the failed step and continue to the next one.

**If CI hangs (no output for 120+ minutes):** terminate and treat as CI failure.
Normal long-running CI with regular output is not a hang and must keep running.

7. **Write step summary.** After committing, write a brief summary for this step: what upstream changes were absorbed, what vllm-ascend files were modified, and any version guards added. Save to `/tmp/main2main/steps/<step-id>/summary.md`. This is important because later steps and the final report depend on it.

### Phase 3: Refresh adapt-guide tables (after all steps are committed)

The lookup tables in `reference/adapt-guide.md` drift as vllm-ascend evolves —
new modules appear, old paths get renamed or deleted. Refresh them once per run
*after every step is committed* (not per step), so the next main2main run reads
an up-to-date guide. Only run this when the run is on track to reach `completed`
or a clean partial stop; skip it if the pipeline rolled back.

1. Run the linter to surface drift:
```bash
python3 <skill_dir>/scripts/lint_adapt_guide.py \
  --ascend-path <ascend_path>
```
This writes `/tmp/main2main/adapt-guide-refresh/check_report.md` with three
sections: invalidated paths, uncovered vllm_ascend/ directories, and upstream
paths touched this run that the file-mapping table doesn't cover.

2. Update **only the three AUTO-MAINTAINED regions** in
`reference/adapt-guide.md` — `key-areas`, `file-locations`, `file-mapping`.
Use the report plus this run's `changed-files.txt` to decide what to add,
rename, or remove. Do not touch the surrounding prose, headings, or any other
section of the file.

3. Commit the refresh as a separate commit, only if the guide actually changed:
```bash
git -C <ascend_path> diff --quiet \
  .agents/skills/main2main/reference/adapt-guide.md \
  || git -C <ascend_path> commit -s \
       -m "docs(main2main): refresh adapt-guide tables" \
       -- .agents/skills/main2main/reference/adapt-guide.md
```
Use `git add <file>`, never `git add .`. Keep this commit separate from any
step's functional commit so reviewers can audit the table changes alone.

### Phase 4: Final summary

Output a reviewer-facing Markdown summary only when one of these is true:
1. Every planned step has been completed, verified, committed, and the target
   vLLM commit has been reached.
2. A listed partial-stop condition was triggered and the required failed patch
   plus failure summary were saved.

Do not write a final summary after a successful intermediate step just to report
remaining work. Build the final summary from the per-step summaries and CI
results. Use the exact structure in `reference/final-summary.md`.

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
- [ ] `lint_adapt_guide.py` ran and any updates to the three AUTO-MAINTAINED regions in `reference/adapt-guide.md` were committed as a separate commit
- [ ] Final summary output to user