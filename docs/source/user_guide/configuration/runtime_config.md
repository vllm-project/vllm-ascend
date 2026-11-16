# runtime_config

JSON schema for Runtime Guard. Default path: `<cwd>/runtime/config/runtime_config.json`.

Defaults are defined in code (`runtime_config._defaults`); JSONC comments are supported if you hand-edit the live file.

**Startup overwrite:** on worker start the JSON writer materializes defaults ← `additional_config.runtime_config` overlay and **overwrites** any pre-existing file. Hand-edits made *before* start are lost unless they are in the overlay. Prefer: start with `runtime_config_hot_reload=true`, then edit the live file (or set the watermark via overlay).

Startup keys (`runtime_config_path`, `runtime_config_hot_reload`, overlay dict) are documented in [Additional Configuration](./additional_config.md#runtime_guard).

## Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `actions` | object | see below | Default incident actions |
| `dump` | object | see below | Auto dump quota and manual dump controls |
| `ascend_log` | object | see below | Ascend logger level overrides |
| `report` | object | see below | Report content and truncation |
| `detector` | object | see below | Detector sections + shared flags |

Hot-reload is **startup-only** (`additional_config.runtime_config_hot_reload`); it is not a JSON field. Sync transport is fixed: **last-PP × all TP** use the merged due bus; other processes poll this JSON file.

## actions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `defaults.on_trigger` | list[str] | `["report"]` | Actions when a detector section omits `on_trigger` |

Valid action names: `report`, `dump_kv`. ActionQueue capacity is a fixed internal constant (not exposed in JSON).

## dump

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_max_times` | int | `0` | Max auto `dump_kv` captures per process lifetime. `0` disables auto dump quota |
| `auto_cooldown_seconds` | float | `300` | Minimum seconds between auto dumps after a **successful** quota consume (refund clears this cooldown) |
| `manual_dump` | bool \| int | `false` | Manual dump watermark: `false`, positive int N (catch-up: dump once per armed wave while process `done < N`; **prefer bump 1→2→3**), or `true` (continuous; **not recommended**). Never rewritten by the process. If disk value ≤ `done`, skip. Setting N=5 with `done=0` fires about five dumps across subsequent waves. Reports / `.pt` carry `manual_dump_count` (1-based seq). Multi-DP: each replica tracks its own `done`. **Ops:** start the server first, then raise N in the live JSON (hot-reload required). |
| `dump_dir` | str \| null | derived | KV dump root (default `<report_dir>/kv_cache`). Layout: `<dump_root>/<incident_type>/<req_id>/dp*_tp*_pp*_cp*/*.pt`. Coverage is last PP × all TP (not other PP). |

Free-space headroom for `dump_kv` (`estimated payload + headroom`) is a fixed internal constant (5 GiB), not a JSON field.

Manual dump / manual trigger skip auto quota and cooldown. Requires `runtime_config_hot_reload=true`.

## report

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_sensitive_info` | bool | `false` | Persist prompt/output token ids **and** decode them to text |
| `max_prompt_token_ids` | int | `100000` | Truncate persisted prompt ids (`0` = unlimited) |
| `max_output_token_ids` | int | `100000` | Truncate persisted output ids |
| `max_per_req` | int | `1` | Max report files per `(incident_type, req_id)`; at cap, stop detecting that request. Wave backoff (64×2ⁿ) between writes when cap &gt; 1 |

GPU `block_ids` are always included in report detail. Decoding token ids to text follows `save_sensitive_info` (no separate toggle).

`[SamplingMeta]` is emitted at DEBUG on the after-sample path (TP0 + last PP). Enable with `ascend_log` / logger level for `vllm_ascend.observability.runtime_guard` — there is no JSON toggle.

## ascend_log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | str | `"INFO"` | Base Ascend log level |
| `debug` | list[str] | `[]` | Module path prefixes forced to DEBUG under `vllm_ascend` |
| `modules` | object | `{}` | Per-logger overrides, e.g. `{"vllm.worker": "WARNING"}` |

## detector (shared)

Stop-detect is controlled by ``report.max_per_req`` (write-full), not a shared detector flag.
Default ``actions.defaults.on_trigger`` includes ``report``.

Detector JSON sections are declared on each detector class (``schema``) and
registered in ``runtime_config.detector_catalog`` — add/remove a detector there
to refresh defaults, validation, and control-panel field lists.

Each nested detector section supports:

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Master switch (default `false`) |
| `on_trigger` | list[str] | Override actions for this incident type |
| `dump_kv` | object | Per-type dump options: `scope` (`request` \| `all_requests`). Finish-wave arms still dump; artifacts may set `request_finished_at_dump: true` (KV may already be freed/reused — treat as suspect). |

### spec_acceptance

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `10` | Rolling window size |
| `low_threshold` | float | `0.3` | Low acceptance rate threshold |
| `len_low_threshold` | float | `1.4` | Length ratio at low rate |
| `high_threshold` | float | `0.96` | High acceptance rate threshold |
| `len_high_threshold` | float | `2.8` | Length ratio at high rate |

Per-req INFO short-log throttle is a fixed internal interval (2s), not a JSON field.

### token_repeat

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `32` | Sliding content window |
| `repeat_sum_threshold` | int | `64` | Alert when sum of repeat scores exceeds this |
| `min_tokens` | int | `32` | Minimum content tokens before alerting |
| `consecutive_hits` | int | `1` | Required consecutive over-threshold steps |
| `ignore_token_ids` | list[int] | `[]` | Token ids excluded from window scoring |

### logits_finite

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Alert on NaN/Inf logits before sampling |
| `item_sync` | bool | `false` | `false`: async `.item()` — non-blocking D2H of the all-finite gate at pre-sample; wait in after-sample / `get_output`. **Logits may be mutated afterward** (e.g. grammar bitmask writes `-inf`); the gate still uses precomputed `row_finite`, but hit-time `finite_kind` may reflect post-mutation values. `true`: blocking `.item()` (+ hit resolve) at pre-sample so gate and kind match **pre-grammar** logits. |

Deferred alert-batch queue cap is a fixed internal constant, not a JSON field.

Each step runs a device ``isfinite`` reduction and one all-finite gate. Default
path launches a non-blocking host copy of that gate and waits in
``check_deferred``. On a hit only, bad rows / ``logits_indices`` / ``finite_kind``
are resolved (indices are snapshotted at pre-sample when needed). Set
``item_sync: true`` when you need kind/attribution against pristine pre-sample
logits. Retired: ``check_every_tokens`` (multi-step window) — ignored if present.

### manual_trigger

Not a detector — control-plane for `dump.manual_dump`. Always runs `dump_kv` over the live batch (`scope=all_requests`; configured `dump_kv.scope` is ignored). `on_trigger` may still list `report` / other actions; if omitted, `actions.defaults` apply and `dump_kv` is injected.

**Prefer `dump.manual_dump: 1` then bump to `2`, `3`, … for another capture** (after the server is already running with hot-reload). Continuous `true` / large jumps add little for debugging and can flood the action queue and disk (`N=5` with `done=0` catch-up-dumps across ~5 waves).

`N` is a **watermark**, not a remaining counter. The process keeps an in-memory `done` count (never written back to JSON). While `done < N`, each armed wave with scheduled tokens fires one dump and bumps `done` (even if `dump_kv` fails to queue — skip marker still written). If the value on disk is ≤ `done`, no dump. To dump again after finishing, raise `N` above `done`. Reports and dump payloads include `manual_dump_count` (1-based seq for that wave) and `manual_dump_target`. Multi-DP sharing one file: each replica tracks its own `done`.

```json
"manual_trigger": {
  "on_trigger": ["report", "dump_kv"]
}
```

## Example (detection + dump on repeat)

```json
{
  "dump": {
    "auto_max_times": 5,
    "auto_cooldown_seconds": 300
  },
  "detector": {
    "token_repeat": {
      "enabled": true,
      "window": 32,
      "repeat_sum_threshold": 64,
      "on_trigger": ["report", "dump_kv"],
      "dump_kv": { "scope": "request" }
    }
  },
  "report": {
    "save_sensitive_info": true,
    "max_output_token_ids": 500
  }
}
```

## Related docs

- [Runtime Guard feature guide](../feature_guide/runtime_guard.md)
- Design / ops (analysis branch): `docs/source/developer_guide/Design_Documents/runtime_guard_{design,ops}.md`
