# Runtime Guard

Runtime Guard is vLLM Ascend's online anomaly detection and incident response layer. It watches decode-time signals (token repetition, garbled output, non-finite logits, speculative acceptance drift, and more), writes structured reports under `runtime/report/`, and optionally captures per-request KV cache blocks via native device-to-host dump (`dump_kv`).

## When to use

- Intermittent quality bugs: repetition, gibberish, sudden NaN/Inf
- Need on-call artifacts (JSON report + optional `.pt` KV slices) without patching the model
- Suspected KV issues: capture with `dump_kv` for offline inspection

Default deployment: **detectors off, hot-reload off** — negligible overhead until you enable features in `runtime_config.json`.

## Quick start

**Detect + report only**

```bash
vllm serve Qwen/Qwen3-8B --additional-config '{
  "runtime_config_reload_interval": 5,
  "runtime_config": {
    "detector": {
      "token_repeat": { "enabled": true },
      "logits_finite": { "enabled": true }
    }
  }
}'
```

**Detect + report + KV dump on hit**

```bash
vllm serve Qwen/Qwen3-8B --additional-config '{
  "runtime_config_path": "/data/runtime/config/runtime_config.json",
  "runtime_config_reload_interval": 5
}'
```

Use the annotated template at `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc` and set:

- `detector.<name>.enabled`: `true`
- `detector.<name>.on_trigger`: `["report", "dump_kv"]`
- `dump.auto_max_times`: e.g. `3` (required for auto dump quota)

## Startup options

Configure through `--additional-config` (or `LLM(..., additional_config=...)`):

| Key | Type | Description |
|-----|------|-------------|
| `runtime_config_path` | str | Path to `runtime_config.json`. Default: `<cwd>/runtime/config/runtime_config.json` |
| `runtime_config_reload_interval` | float | Hot-reload period in seconds. `0` = static after startup (default) |
| `runtime_config` | dict | Startup overlay merged into JSON defaults |
| `runtime_report_dir` | str | Override report root (default `<cwd>/runtime/report`) |
| `runtime_dump_dir` | str | Seed KV dump root `dump.dump_dir` (default derived from report root). Hot-reload of `dump.dump_dir` in JSON wins after startup |

See [Additional Configuration](../configuration/additional_config.md#runtime_guard) and the full [runtime_config reference](../configuration/runtime_config.md).

## Architecture (summary)

```text
RuntimeGuardProcessor.bind(runner)
  → sync_for_step()        # config + wave + manual triggers
  → detector hooks         # before/after sample (and after spec)
  → ActionExecutor         # report | dump_kv | set_log_level (async queue)
```

Design details (Chinese): [runtime_guard_design.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/zh/design/runtime_guard_design.md)  
Operations runbook (Chinese): [runtime_guard_ops.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/zh/design/runtime_guard_ops.md)

## On-disk layout

```text
runtime/
  config/runtime_config.json
  report/
    <incident_type>/report_<ms>[_<req_id>]_pid*.json   # last-PP TP0 only
    kv_cache/<incident_type>/<req_id>/wave_*/dp*_tp*_pp*_cp*/*.pt
    kv_cache/<incident_type>/<req_id>/wave_*/request_info.json
```

Report JSON top-level includes `dump_attempted` (whether `dump_kv` was in `on_trigger`, not whether D2H finished), `dump_arm_wave`, and `dump_dir`. Same `(type, req_id)` is capped by `report.max_per_req` (default 1); reaching the cap stops detection for that request. Wave backoff (64, then doubles) spaces further writes when the cap is higher. Default `on_trigger` includes `report`.

## Detectors

| Type | Stage | Typical use |
|------|-------|-------------|
| `token_repeat` | after sample | Stutter / repetition |
| `output_substring` | after sample | Forbidden or garbage token patterns |
| `logits_finite` | before sample | NaN/Inf logits (``isfinite`` + ``.item()`` each step; hit resolves indices/kind then enqueues; after-sample drains). Unattributable rows: warning only (no guessed / null-req report). |
| `spec_acceptance` | after spec | Spec-decode acceptance drift (via `run_sample_phase` → `check_after_spec`; v2 stashes accept stats in `postprocess_sampled`) |

All detectors default to **disabled**. Enable individually under `detector.<name>.enabled`.

Online KV / position meta detectors are **not** in this release; use `dump_kv` for KV capture (offline compare tooling ships later).

> **Wiring:** v1/v2 `sample_tokens` call `RuntimeGuardProcessor.run_sample_phase` for post-pre-sample hooks (`mark_finished` / `check_after_spec` / waves / `check_after_sample` when host ids are ready). Pre-sample `check_before_sample` stays on the compute_logits wrap (before grammar; `logits_finite` only — `.item()` gate there; hit resolves then enqueues). After-sample for async scheduling **and** for v2 `AsyncOutput` (even under sync scheduling) runs in `AscendAsync*` `get_output()` after D2H + `num_sampled` trim — do not append padded `AsyncOutput.sampled_token_ids` in the sync hook (W2-3 / D-11). Then drain `logits_finite` incidents and enqueue CPU detectors (`token_repeat` / substring) on `ActionQueue` (finish does not wait). `dump_kv` is skipped if the request has already finished/reaped.

## Actions

| Action | Effect |
|--------|--------|
| `report` | Write JSON incident report (+ metric counter) |
| `dump_kv` | D2H paged KV for the request. **Last PP × all TP** (not other PP stages): TP0 records dump jobs this step and writes `request_info.json` (report-like metadata); every last-PP TP rank (including TP0) dumps on the next `sync_for_step` via TP `broadcast_object`. Same arm wave: each `req_id` at most once (`queue_kv_dump` dedupes by `(wave, req_id)`; duplicate enqueue logs INFO). Files: `{dump_root}/{type}/{req_id}/wave_<N>/request_info.json` and `{dump_root}/{type}/{req_id}/wave_<N>/dp*_tp*_pp*_cp*/*.pt` |
| `set_log_level` | Raise log verbosity synchronously on trigger |

Default `on_trigger` is `["report"]`. Per-detector overrides:

```json
"token_repeat": {
  "enabled": true,
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "request" }
}
```

**KV dump rank coverage:** detection and reports stay on last-PP TP0. `dump_kv` captures **last PP × every TP rank**. **PP>1 forces `sync_mode=file`**. On PP==1 broadcast, **wave-head** uses **one** `all_reduce([config_due, dump_due])` then separate `broadcast_object` per due lane (idle waves pay only that AR). Detector dumps are delivered the **next** wave (+1); config applies at wave-head for same-wave detection. Assemble shards from the `tp*` directories under the same `req_id`.

## Performance

- **No additional-config / defaults**: bind-only path; intended to be noise-free.
- **Hot-reload only** (`reload_interval > 0`, all detectors off, dump off): small periodic JSON sync; UT bounds ~1–2% CPU on reload path.
- **Detectors on**: cost depends on enabled checks (light: `token_repeat`; heavier: `logits_finite` with per-step `.item()`).
- **dump_kv on hit**: detector arms queue for the **next** wave-head dump lane, then D2H at that wave's `end_of_wave_sync` (after prepare). Manual dump D2H's locally at end-of-wave (no dump-job bcast). Idle steps only pay a cheap due-vector `all_reduce` at wave-head; per-lane `broadcast_object` runs only when that lane is due.

Live NPU A/B checklist: `vllm_ascend/runtime_guard/test/perf/README.md`.

## Related docs

- [runtime_config.md](../configuration/runtime_config.md) — JSON field reference  
- [runtime_guard_design.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/zh/design/runtime_guard_design.md) — full design  
- [runtime_guard_ops.md](https://github.com/vllm-project/vllm-ascend/blob/main/docs/zh/design/runtime_guard_ops.md) — ops / troubleshooting
