---
name: main2main-error-analysis
description: |
  Automates root-cause analysis of vLLM-Ascend CI failures triggered by upstream vLLM main branch updates.
  Given a GitHub Actions workflow run, this skill extracts failed test cases,
  mines error logs for true root causes, traces failures to specific upstream vLLM commits, generates a
  structured diagnostic report (vllm_error_analyze.md), applies adaptation fixes, and creates a PR.

  Use this skill whenever:
  - The user shares a GitHub Actions URL or run ID related to vLLM-Ascend CI failures
  - The user mentions CI failures related to vLLM main branch updates or "main2main" test failures
  - The user asks to analyze, debug, or fix failures in vllm-ascend E2E tests caused by upstream changes
  - The user wants to understand what broke between a "good" pinned commit and a "bad" upstream commit
  - Even if the user just says "the nightly CI is red" or "schedule tests are failing" — use this skill
---

# main2main-error-analysis

Diagnose and fix vLLM-Ascend CI failures caused by upstream vLLM main branch evolution. This skill implements a 4-phase pipeline: context acquisition, log mining, change analysis, report generation, and automated fix with PR submission.

## Prerequisites

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

1. **Always use the bundled script** for Phases 1-2. It processes logs in a subprocess and returns only the structured results — keeping your context clean for the higher-value Phase 3 analysis.
2. **Write a partial report early.** After Phase 1, immediately write a skeleton `vllm_error_analyze.md` with the Overview table, failed jobs, and error list. Then fill in the upstream commit details as you complete Phase 2. This ensures the user gets a useful report even if you run low on budget.
3. **Use the local vLLM repo** for all upstream code analysis (Phase 2). Run `git log`, `git diff`, `git show`, and read files directly from `$VLLM_LOCAL_DIR` — this is faster and more reliable than GitHub API calls, and avoids rate limits.
4. **If falling back to manual mode**, never pipe full logs into context. Always filter through `grep` with `head` limits first.

---

## Phase 1: Fault Context Acquisition — Use the Script

This skill bundles `scripts/extract_and_analyze.py` which handles Phases 1 and 2 automatically. **Always run the script first** to avoid wasting tokens on manual log parsing.

### 1.1 Run the Extraction Script

Determine the skill's own directory path (where this SKILL.md is located), then run:

```bash
# With a specific run ID:
python3 <SKILL_DIR>/scripts/extract_and_analyze.py --run-id <RUN_ID> --llm-output -o /tmp/ci_analysis.json

# Or auto-find latest failed run:
python3 <SKILL_DIR>/scripts/extract_and_analyze.py --llm-output -o /tmp/ci_analysis.json
```

The script will:

- Find the failed workflow run and download logs for each failed job
- Extract the **bad commit** from the vLLM version string in the logs (e.g., `vLLM 0.1.dev1+g6d4f9d3ad.empty` → `6d4f9d3ad`)
- Extract the **good commit** from `.github/workflows/pr_test_full.yaml` (the `vllm_version` matrix field)
- Parse all failed test identifiers using three regex patterns (inline, summary block, pytest FAILED line)
- Extract root-cause exceptions (TypeError, AttributeError, ImportError, etc.)
- Skip wrapper errors (`Engine core initialization failed`, `Worker failed with error`)
- Filter downstream effects (`KeyError: 'choices'` caused by upstream engine crash)
- Detect environment flakes (`Stale file handle`, `ConnectionResetError`, `filelock` errors) — even when embedded inside assertion messages
- Deduplicate errors by normalized signature (stripping PIDs, timestamps, addresses, errno numbers)
- Output a structured JSON report

### 1.2 Read the Script Output

Load `/tmp/ci_analysis.json` and extract the key fields:

```python
{
  "run_id": 21646698906,
  "good_commit": "15d76f74e2fdb12a95ea00f0ca283acf6219a2b7",
  "bad_commit": "6d4f9d3ad5aa3750697edcf013ad080619ae25e9",
  "code_bugs": [
    {"error_type": "TypeError", "error_message": "...", "context": [...], "affected_tests": [...]}
  ],
}
```

**Phase 1 outputs:** `RUN_ID`, `GOOD_COMMIT`, `BAD_COMMIT`and list of `code_bugs`

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

## Phase 4: Automated Fix & PR Submission

### 4.1 Apply Fixes

Only fix `Code Bug` issues. Skip `Environment Flake` issues entirely.

Map each error to the corresponding fix pattern in **Common Error Patterns Reference** below. The patterns cover frequently seen upstream vLLM evolution issues with concrete fix examples.

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

### 4.3 Update vLLM Commit References

After applying code fixes, update all vllm commit references in vllm-ascend from the **good commit** to the **bad commit**. Use a repo-wide grep-and-replace:

```bash
# Find all files containing the good commit and replace with bad commit
grep -Frl "<GOOD_COMMIT>" . | xargs sed -i '' "s/<GOOD_COMMIT>/<BAD_COMMIT>/g"
```

Verify no old references remain:

```bash
grep -Frn "<GOOD_COMMIT>" .
# Should return nothing
```

### 4.4 Create Branch and PR

After applying all fixes and updating commit references:

```bash
# Remove the analysis report (it's for local diagnosis only, not for the repo)
rm -f vllm_error_analyze.md

# Create a descriptive branch
git checkout main
git checkout -b main2main-ci-$(date +%Y%m%d)

# Stage all modified files
pre-commit run --all-files 
git add -u

# Commit with structured message
git commit -m "fix: adapt to upstream vLLM changes (<GOOD_COMMIT_SHORT>..<BAD_COMMIT_SHORT>)

Root causes:
- <issue 1 one-line summary>
- <issue 2 one-line summary>

Upstream commit range: <GOOD_COMMIT>..<BAD_COMMIT>

Co-Authored-By: Claude Code <noreply@anthropic.com>"

# Push and create PR
git push -u origin main2main-ci-$(date +%Y%m%d)

gh pr create \
  --title "fix: adapt to upstream vLLM changes ($(date +%Y-%m-%d))" \
  --body "$(cat <<'EOF'
## Summary
Fixes CI failures in schedule_test_vllm_main caused by upstream vLLM changes.

**Commit range:** `<GOOD_COMMIT>`..`<BAD_COMMIT>`

### Issues Fixed
- <issue 1>
- <issue 2>

### Issues Skipped (Environment Flakes)
- <flake description> — no code fix needed

See `vllm_error_analyze.md` for full analysis.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Common Error Patterns Reference

These are the most frequently seen failure patterns when upstream vLLM evolves:

### Pattern: Method Signature Change

- **Error:** `TypeError: forward_oot() got an unexpected keyword argument 'X'` or `missing 1 required positional argument: 'X'`
- **Cause:** vLLM changed a method signature — parameter added, removed, renamed, or full API replacement (e.g., `disable_full` → `valid_modes`/`invalid_modes`).
- **Fix:** Compare signatures at good vs bad commit, then adapt:

```python
from vllm_ascend.utils import vllm_version_is

# Option 1: Add parameter conditionally to call site
kwargs = {"existing_param": value}
if not vllm_version_is("0.16.0"):  # version before the change
    kwargs["new_param"] = new_value
function(**kwargs)

# Option 2: Add default parameter to OOT method signature
def forward_oot(self, query, key, value, cu_seqlens=None, max_seqlen=None, new_param=None):
```

For full API replacements, adapt the call site to match the new API — do NOT blindly add the old parameter.
**Important:** When creating version-guarded branches, all branches must define the function with identical signatures (convert lambdas to `def` if needed). Mismatched signatures across branches cause mypy `[call-arg]` errors.

### Pattern: Config/Attribute Change

- **Error:** `AttributeError: 'CompilationConfig' object has no attribute 'X'`, `KeyError: 'field_name'`, or `Config object has no attribute 'Y'`
- **Cause:** Upstream moved an attribute/config field between classes, restructured a config class, or added a new required field (e.g., `bs_to_padded_graph_size` moved to `CudagraphDispatcher`, `uses_mrope` moved from target to draft model config, `enable_eplb` added to `FusedMoEParallelConfig`).
- **Fix:** Use `vllm_version_is()` to access from the correct location:

```python
if vllm_version_is('0.16.0'):
    value = self.vllm_config.old_location.attribute
else:
    value = self.new_class.new_location.attribute
```

For config access that changes frequently, consider helper methods like `_get_positions()` / `_set_positions()` to abstract the logic. For new required fields, add them to the config wrapper.

### Pattern: Method Return Type Change

- **Error:** `TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'` or similar comparison errors on None
- **Cause:** Upstream changed a method from returning `None` to returning a value (e.g., `float`), and the caller now uses it.
- **Fix:** Update the OOT override to return the expected value.

### Pattern: Module Reorganization

- **Error:** `ImportError: cannot import name 'X' from 'vllm.old.path'`, or  `error: Cannot find implementation or library stub for module named "vllm.X"  [import-not-found]`
- **Cause:** vLLM moved/renamed a module, or removed it entirely (e.g., `vllm._bc_linter`).
- **Fix:** For moved/renamed modules, use `vllm_version_is()` to branch imports. For removed modules, delete the import **and** all usages (decorators, function calls) — clean removal over `# type: ignore`.

### Pattern: Platform Interface Addition

- **Error:** `TypeError: Can't instantiate abstract class AscendPlatform with abstract method X`
- **Cause:** New abstract method added to vLLM's `Platform` base class
- **Fix:** Implement the method in `vllm_ascend/platform.py`

### Pattern: Environment Flakes (NO FIX NEEDED)

- `OSError: [Errno 116] Stale file handle` — multi-process NFS race
- `ConnectionResetError` — transient network
- `filelock` errors — model download contention
- These should be noted in the report but require no code changes

---
