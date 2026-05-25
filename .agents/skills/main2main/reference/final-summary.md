# Final Summary Guide

Use this guide at the end of a main2main run. The final summary is for a human reviewer, so it should explain what was completed, what was verified, and what still needs attention. It is not a debug dump.

Write the final summary only when the run reached a terminal state:
- `completed`: all planned steps were verified, committed, and the target vLLM commit was reached.
- `partial`: a valid partial-stop condition from `SKILL.md` was triggered.

Do not write this file after a successful intermediate step. Long remaining CI
time, session length, model cost, or "additional CI runs required" are not valid
partial-stop reasons.

Build the summary from:
- `/tmp/main2main/steps/<step-id>/summary.md`
- CI results for each step
- created vllm-ascend commits
- `round-N-summary.json` output for any partial stop

Keep exact SHAs where they matter for traceability. Keep raw logs and internal debug details out of the main report. Mention temp paths only when the run ends with a partial stop and the reviewer needs the saved patch or failure summary.

## Output Template

Use this Markdown structure:

```markdown
## Main2Main Summary

Status: completed | partial
Upstream range: <base_sha>..<target_sha>
Reached upstream commit: <reached_sha>
Steps: <completed>/<total>
CI suite: e2e-main2main

### Result
<One short paragraph describing whether the target commit was fully reached.
If partial, state the valid partial-stop condition.>

### Completed Steps
| Step | Upstream range | vllm-ascend commit | CI result | Summary |
| --- | --- | --- | --- | --- |
| step-1 | <start>..<end> | <sha> | passed | <main adaptation or "commit reference only"> |

### Changes Made
- Updated vLLM commit reference from <base_sha> to <reached_sha>.
- <Key vllm-ascend adaptation area or file group changed.>
- <Version compatibility guards added, if any.>

### CI Verification
- Passed: <steps or suites that passed>
- Treated as env flakes: <brief list, or "none">
- Last successful step: <step-id>

### Adapt Guide Refresh
Only include this section when `lint_adapt_guide.py` was run and produced
output. Helps the PR reviewer audit machine-driven changes to the lookup
tables in `reference/adapt-guide.md`.

- Lint report: `/tmp/main2main/adapt-guide-refresh/check_report.md`
- Adapt-guide commit: <sha or "no change — guide already up to date">
- What changed and why: <one bullet per AUTO-MAINTAINED region touched —
  e.g., "file-mapping: added row for `vllm/v1/foo/` since step-2's patch
  introduced it; removed row for `vllm/old_path/` since the directory is gone">

### Partial Stop
Only include this section when Status is `partial`.

- Stopped at: <step-id>, upstream range <start>..<end>
- Reason: <valid partial-stop condition from SKILL.md and concise explanation>
- Unresolved failures: <short error summary from round-N-summary.json>
- Saved patch: /tmp/main2main/steps/<step-id>/failed.patch
- Saved failure summary: /tmp/main2main/steps/<step-id>/failed-summary.json
- Repository state: rolled back to last verified vllm-ascend commit <sha>

### Follow-up
- <Concrete next action, only when needed>
```

## Writing Rules

- Prefer concise paragraphs and a small number of high-signal bullets.
- Do not include raw CI logs.
- Do not list every file unless the file list is small and important.
- Use `commit reference only` when a step only updated the vLLM commit hash and CI passed without extra code adaptation.
- Allowed Completed Steps `CI result` values are `passed` and
  `env_flake_pass`.
- Report CI as `passed` only when
  `/tmp/main2main/steps/<step-id>/ci/round-N-result.json` has
  `ci_result: "passed"` and `run_suite_exit_code: 0`.
- Report CI as `env_flake_pass` only when `ci_result` is `env_flake_pass`.
  This means `run_suite_exit_code` was non-zero, but the generated summary had
  only `env_flakes` and no actionable `code_bugs`. This is allowed to proceed
  to commit, but it is not the same as `passed`.
- For `partial`, make the unresolved failure actionable: name the failing test, exception type, and likely area if known.
- Include `Adapt Guide Refresh` only when the linter actually ran. If the guide was unchanged, still include the section so the reviewer sees the lint was performed and produced no diff.
- Do not add "Remaining Steps (Not Started)" for a normal in-progress run; continue executing the next step instead.
- Do not use "session time", "estimated remaining time", "too many hours", or "additional CI runs required" as the reason for `partial`.
- Omit `Follow-up` when there is no concrete next action.
