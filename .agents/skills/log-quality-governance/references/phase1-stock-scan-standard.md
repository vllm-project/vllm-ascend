# Phase-1 stock scan standard (`phase1-stock`)

Authoritative scan standard for the default `phase1-stock` mode in
`.agents/skills/log-quality-governance`.
This is the **minimum quality line + stock scan coverage definition**, not the full ideal log standard.

Related Phase-1 layers:

- Incremental hard gate: `tools/check_log_quality.py` + `tools/log_quality_rules.toml`
- PR context review: `.gemini/styleguide.md` → Logging Quality

---

## Priority model (single scale)

| Priority | Meaning | Typical action |
|---|---|---|
| **P0** | Sensitive leak; totally vague ERROR; critical failure path with no log | Fix in first remediation batch |
| **P1** | Missing key diagnostic / cross-component fields; retry without final ERROR | Second batch |
| **P2** | Naming inconsistency; weak readability; incomplete start/success/failure closure | Defer unless asked |

Do not invent a parallel P0–P3 raise scale. Map raise gaps into this table.

---

## Minimum bar (must detect)

| Category | Stock requirement | Default priority if violated |
|---|---|---|
| Privacy | Do not log raw `messages`, passwords, secrets, private keys, certificates, bare tokens, full HTTP body, large tensors/lists | **P0** |
| Readable errors | ERROR / exception / critical must not be only `failed` / `error` / `timeout` (see `log_quality_rules.toml` `[hard_fail.vague]`) | **P0** |
| Diagnostic carrier | ERROR should carry args, key=value fields, exception text, or `exc_info=True` / `logger.exception` | **P0** if empty/short with no carrier; else **P1** if weak |
| Linkable | Request-path / cross-component / retry final failure should carry correlators (`req_id`/`request_id`, peer fields, etc.) | **P1** (not P0 in phase1-stock) |
| Maintainable | Remediation must not rewrite business control flow | Process rule (rewrite-guide) |

Token-name allowlist for privacy (do **not** flag): see
`tools/log_quality_rules.toml` → `allowlist_names` (`num_tokens`, `input_tokens`, …).

---

## Nine coverage buckets (scan report must cover)

| # | Bucket | What to extract / check | Default priority |
|---|---|---|---|
| 1 | Logger inventory | path, line, level, template, args | info |
| 2 | Level × module distribution | counts under `vllm_ascend/` | info |
| 3 | Sensitive hits | privacy identifiers in message / arg names | **P0** |
| 4 | Vague ERROR/WARNING candidates | exact vague literals / empty message | **P0** |
| 5 | Diagnostic carrier | ERROR/WARNING with no `%`/`{}`/kwargs/`exc_info` | **P0/P1** |
| 6 | Request-id coverage | high-risk request dirs missing `req_id`/`request_id` | **P1** |
| 7 | Cross-component fields | missing `peer_component` / `result` / `duration_ms` / `peer_addr` on call boundaries | **P1** |
| 8 | Loop / retry spam | ERROR inside tight retry loop without final summary | **P1** |
| 9 | Raise / failure gap | business failure path with no nearby warning/error/exception log | **P0** if silent critical path; else **P1** |

### High-risk request / link directories (hint list, not exclusive)

Prefer deeper `req_id` checks under paths that touch serving, connector, KV, worker RPC, platform init — e.g.:

- `vllm_ascend/worker/`
- `vllm_ascend/platform.py` and platform helpers
- connector / KV / disaggregate related modules under `vllm_ascend/`

If unsure whether a file is request-path, mark **P2** “possible missing req_id” instead of P1.

### Raise-gap precision rules

Skip:

- `tests/**`, generated protobuf, `__init__` trivial paths
- Framework raises (`KeyError`, `AttributeError`, …) that are programmer bugs
- Private helpers that are assert-like
- Raises already followed within ~20 lines by `logger.exception` / `logger.error` in the **same file**

Do not require a log before every `raise`.

---

## Field vocabulary (align with Gemini styleguide)

| Concern | Preferred fields |
|---|---|
| Request correlation | `req_id` (new), `request_id` (legacy OK) |
| Cross-component | `peer_component=`, `result=`, `duration_ms=`, `peer_addr=` when network/process boundary |
| Retry | `attempt=`, `max_attempts=`, `final_result=`, `last_error=`; final failure → summary ERROR |
| Background / periodic | `task_id` / `job_id` / stable task name |

`trace_id` is acceptable if already used in a file; do **not** mass-rename to `trace_id` in rewrites. Prefer `req_id` for new request-path text.

---

## Explicitly NOT hard-fail in `phase1-stock`

These may appear as **P2** suggestions only (or be ignored):

- Missing `event=`
- Missing fixed `reason=`
- Missing Chinese “可检查 / 根因” prose
- Missing `[vllm/…]` component prefix in the Python string
- Incomplete start/success/failure closure on non-critical flows
- Ops retention / shipping / aging

In **`full-standard`** mode (user opt-in only), agents may apply a broader ideal bar, but must still emit the same P0/P1/P2 labels and must still refuse business-control-flow edits.

---

## Mode switch

| Mode | When | Bar |
|---|---|---|
| `phase1-stock` (default) | Periodic governance, batch backlog | This file |
| `full-standard` | User explicitly asks for full ideal standard | Broader SHOULD rules; still no PR gate; still no auto-apply |
