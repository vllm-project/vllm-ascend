# Rewrite guide (stock remediation candidates)

Used by `log-quality-governance` after a `phase1-stock` scan.
Produces **candidates only** — human applies patches.

---

## Scope of edits

| Do | Do not |
|---|---|
| Change logger call level / message / fields | Change retry/backoff timing or count logic |
| Add `exc_info=True` or switch to `logger.exception` when appropriate | Change `return` / `raise` targets or control flow “to make room for logs” |
| Reuse existing rate-limit helpers already in the module | Introduce new threading, HTTP clients, or sleep policies |
| Keep English prose consistent with the file | Force Chinese templates or Hermes-style “可检查：…” blocks |
| Prefer `req_id=` on new request-path fields | Mass-rename working `request_id` / `trace_id` without need |

Never `git commit` or `git merge` as part of this skill.

---

## Priority filter

Default rewrite set = **P0 + P1** from the scan report.
Include **P2** only when the user asks.

Suggested batching:

1. P0 privacy + empty/vague ERROR
2. P0 silent critical failure paths
3. P1 diagnostic / `req_id` / cross-component / retry summary

---

## Style matching (vllm_ascend)

Before rewriting a site, sample nearby logger calls in the same file:

1. Method style: `logger.error(...)` vs `self.logger.error(...)`
2. Formatting: `%s` vs f-string vs `.format`
3. Natural language: almost always **English** in this tree
4. Whether structured `key=value` already appears

Rewrite must look native to that file.

---

## Fix patterns by finding type

### Privacy (P0)

- Remove or redact sensitive names (`messages`, raw token/secret/password, full body, huge containers)
- Keep safe dimensions: `input_len`, `num_tokens`, `prompt_hash` (allowlisted)

### Vague / no carrier (P0)

```python
# bad
logger.error("failed")

# better (illustrative — match local formatting)
logger.error(
    "NPU query failed: local_rank=%s error=%s",
    self.local_rank,
    e,
)
# or
logger.exception("NPU query failed: local_rank=%s", self.local_rank)
```

Do **not** invent peer IPs, ports, or root causes not available in scope.
If a value is not in scope, omit it rather than fabricating.

### Request path missing id (P1)

Add `req_id=` (or keep `request_id=` if that is the local convention) when the
id is already available in the function. If not available, note
`needs_context_plumbing` in the candidate and **do not** invent a random id
generator mid-function unless the user explicitly wants that design.

### Cross-component (P1)

When the log sits on an outbound/inbound RPC or connector boundary, prefer:

`peer_component=... result=... duration_ms=...` and `peer_addr=...` if address exists.

### Retry spam (P1)

- Per-attempt failures: WARNING/DEBUG inside the loop
- Final failure: one ERROR with `attempt=` / `max_attempts=` / `last_error=` / `final_result=`
- Do not delete per-attempt visibility entirely; downgrade level if needed

### Raise gap (P0/P1)

Add a logger call **adjacent** to the failure site only if it does not require
restructuring control flow. If the clean fix needs control-flow change, mark
the finding `needs_human_design` and skip auto-diff.

---

## Diff output rules

1. One finding → one hunk when possible
2. No drive-by refactors, import churn, or formatting-only noise
3. State clearly: “draft only — do not apply without review”
4. If a fix would touch non-logger statements, stop and escalate to the user
