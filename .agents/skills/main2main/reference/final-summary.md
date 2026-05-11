# Final Summary Guide

Use this guide at the end of a main2main run. The final summary is for a human reviewer, so it should explain what was completed, what was verified, and what still needs attention. It is not a debug dump.

Build the summary from:
- `/tmp/main2main/steps/<step-id>/summary.md`
- CI results for each step
- created vllm-ascend commits
- `ci_log_summary.py` output for any partial stop

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
If partial, state where the run stopped and why.>

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

### Partial Stop
Only include this section when Status is `partial`.

- Stopped at: <step-id>, upstream range <start>..<end>
- Reason: <fix loop stop condition and concise explanation>
- Unresolved failures: <short error summary from ci_log_summary>
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
- For `partial`, make the unresolved failure actionable: name the failing test, exception type, and likely area if known.
- Omit `Follow-up` when there is no concrete next action.
