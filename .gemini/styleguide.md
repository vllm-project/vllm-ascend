# Pull Request Summary Style Guide

## Output Instructions

**IMPORTANT**: When doing PR review, you MUST output them in markdown code blocks so users can easily copy them:

1. **PR Title**: Output the generated title in a code block with triple backticks
2. **PR Summary**: Output the generated summary in a markdown code block with triple backticks

This allows users to directly copy the content without manual formatting.

## Pull Request Summary Format

The summary should follow the format:

   ```markdown
    ### What this PR does / why we need it?
    <!--
    - Please clarify what changes you are proposing. The purpose of this section is to outline the changes and how this PR fixes the issue.
    If possible, please consider writing useful notes for better and faster reviews in your PR.

    - Please clarify why the changes are needed. For instance, the use case and bug description.

    - Fixes #
    -->

    ### Does this PR introduce _any_ user-facing change?
    <!--
    Note that it means *any* user-facing change including all aspects such as API, interface or other behavior changes.
    Documentation-only updates are not considered user-facing changes.
    -->

    ### How was this patch tested?
    <!--
    CI passed with new added/existing test.
    If it was tested in a way different from regular unit tests, please clarify how you tested step by step, ideally copy and paste-able, so that other reviewers can test and check, and descendants can verify in the future.
    If tests were not added, please describe why they were not added and/or why it was difficult to add.
    -->
   ```

## Pull Request Title Format

The summary should also refresh the Pull Request Title to follow the format:

    ```txt
    [Branch][Module][Action] Pull Request Title
    ```

- Branch: The branch name where the PR is based. If the base branch is main, this prefix can be omitted.
- Module: The module or component being changed. It includes but is not limited to the following:
    - [Attention]
    - [Ops]
    - [Doc]
    - [Test]
    - [CI]
    - [Benchmark]
- Action: The action being performed. It includes but is not limited to the following:
    - [BugFix]
    - [Feature]
    - [Misc]

## Example Output Format

When providing a PR review, format your response like this:

**Suggested PR Title:**

```markdown
[Branch][Module][Action] Your generated title here
```

**Suggested PR Summary:**

```markdown
### What this PR does / why we need it?

Your analysis of what the PR does and why it's needed.

Fixes #issue_number

### Does this PR introduce _any_ user-facing change?

Your assessment of user-facing changes.

### How was this patch tested?

Your description of testing approach.
```

And please print your review suggestion in markdown format no matter the pull request description is empty or not.

## Logging Quality (PR incremental)

When reviewing changes that add or modify logger calls in `vllm_ascend/`, apply the following rules.
Hard violations (privacy, vague ERROR, missing diagnostic carrier, `init_logger(__name__)`) are enforced by pre-commit (`tools/check_log_quality.py`, `tools/check_logger.sh`).
Do **not** repeat those checks here. Focus on context-dependent MUST / MUST NOT issues only.
Stock / periodic inventory and rewrite candidates are handled by
`.agents/skills/log-quality-governance/` (`phase1-stock`); do not turn those
stock findings into PR-blocking comments unless they are also incremental MUST issues on this diff.

Comment only when confidence is HIGH. Avoid style-only suggestions.

| Issue type | Review focus |
|---|---|
| Error semantics | ERROR / WARNING must express failure semantics; do not require a fixed `reason=` field |
| Request tracing | Request-path logs should include `req_id` or legacy-compatible `request_id`; prefer `req_id` for new logs |
| Cross-component calls | Logs should be self-evident: clearly describe the call relationship and what happened (outcome, timing). Prefer `peer_component=`, `result=`, `duration_ms=`; add `peer_addr=` at process/network boundaries. Do not treat field names as a checklist |
| Retry loops | Include `attempt=`, `max_attempts=`, `final_result=`, `last_error=` where applicable; final failure must have a summary ERROR |
| Hot paths | Success paths should use DEBUG, sampling, or rate limiting to avoid log flooding |
| Long-running flows | Model load, KV link, resource allocation should have start / success / failure closure |
| Background tasks | Periodic or long-lived tasks should carry `task_id`, `job_id`, or a stable task name |
| Privacy | Do not log raw user input, keys, certificates, passwords, full HTTP body, large tensors, or large lists |
| Change boundary | Do not modify retry/backoff, return paths, thread loops, HTTP calls, or business control flow just to add logs |

If a PR only fixes pre-commit hard violations without context issues, do not add logging comments.
