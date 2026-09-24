# runtime_config

JSON schema for Runtime Guard. Default path: `<cwd>/runtime/config/runtime_config.json`.

Annotated example: `vllm_ascend/runtime_config/templates/runtime_config.example.jsonc`.

Startup keys (`runtime_config_path`, `runtime_config_reload_interval`, overlay dict) are documented in [Additional Configuration](./additional_config.md#runtime_guard).

## Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sync_mode` | str | `"broadcast"` | `"broadcast"` (PP==1: end-of-wave `all_reduce([config_due, dump_due])` + per-lane broadcast) or `"file"` (each rank polls). **PP>1 always forces `file`**. |
| `reload_interval_seconds` | number | `0` | Display only; effective interval is `runtime_config_reload_interval` at process start |
| `actions` | object | see below | Default incident actions |
| `dump` | object | see below | Auto dump quota and manual dump controls |
| `ascend_log` | object | see below | Ascend logger level overrides |
| `log` | object | see below | Ops logging switches (not stored in report JSON) |
| `report` | object | see below | Report content and truncation |
| `detector` | object | see below | Detector sections + shared flags |

## actions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `defaults.on_trigger` | list[str] | `["report"]` | Actions when a detector section omits `on_trigger` |
| `queue_max_size` | int | `64` | ActionQueue capacity at bind (raise under bursty dump/CPU detect) |

Valid action names: `report`, `dump_kv`, `set_log_level`.

## dump

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_max_times` | int | `0` | Max auto `dump_kv` captures per process lifetime. `0` disables auto dump quota |
| `auto_cooldown_seconds` | float | `300` | Minimum seconds between auto dumps after a **successful** quota consume (refund clears this cooldown) |
| `manual_dump` | bool \| int | `false` | Manual dump: `false`, positive int N (armed waves; **prefer `1` — one shot is enough**), or `true` (continuous every wave until hot-reload false; **not recommended** — little debug value, floods ActionQueue/disk). Each armed wave decrements in memory whether dump queued or failed (`dump_skipped.json`); JSON rewritten only when count reaches 0. Multi-DP sharing one file: each DP may dump up to N times while the file still shows N. |
| `dump_dir` | str \| null | derived | KV dump root (default `<report_dir>/kv_cache`). Layout: `<dump_root>/<incident_type>/<req_id>/dp*_tp*_pp*_cp*/*.pt`. Coverage is last PP × all TP (not other PP). |
| `free_headroom_bytes` | int | `5368709120` (5 GiB) | Skip `dump_kv` when `statvfs` free space is below **estimated payload + this headroom**. Not a fixed free-space floor. |

Manual dump / manual trigger skip auto quota and cooldown. Requires hot-reload interval &gt; 0.

## report

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_sensitive_info` | bool | `false` | Persist prompt/output token ids in reports |
| `decode_token_ids` | bool | `true` | Decode ids to text when sensitive info saved |
| `max_prompt_token_ids` | int | `1000` | Truncate persisted prompt ids (`0` = unlimited) |
| `max_output_token_ids` | int | `1000` | Truncate persisted output ids |
| `include_block_ids` | bool | `true` | Include GPU block ids in report detail |
| `max_per_req` | int | `1` | Max report files per `(incident_type, req_id)`; at cap, stop detecting that request. Wave backoff (64×2ⁿ) between writes when cap &gt; 1 |

## log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `print_output_on_finish` | bool | `false` | Log output token ids/text when any request finishes |

`[SamplingMeta]` is emitted at DEBUG on the after-sample path (TP0 + last PP). Enable with `ascend_log` / logger level for `vllm_ascend.runtime_guard` — there is no JSON toggle.

## ascend_log

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | str | `"INFO"` | Base Ascend log level |
| `debug` | list[str] | `[]` | Module path prefixes forced to DEBUG under `vllm_ascend` |
| `modules` | object | `{}` | Per-logger overrides, e.g. `{"vllm.worker": "WARNING"}` |

## detector (shared)

Stop-detect is controlled by ``report.max_per_req`` (write-full), not a shared detector flag.
Default ``actions.defaults.on_trigger`` includes ``report``.

Each nested detector section supports:

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Master switch (default `false`) |
| `on_trigger` | list[str] | Override actions for this incident type |
| `dump_kv` | object | Per-type dump options: `scope` (`request` \| `all_requests`) |
| `set_log_level` | object | For `set_log_level` action: `level`, `modules` |

### spec_acceptance

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window` | int | `10` | Rolling window size |
| `low_threshold` | float | `0.3` | Low acceptance rate threshold |
| `len_low_threshold` | float | `1.4` | Length ratio at low rate |
| `high_threshold` | float | `0.96` | High acceptance rate threshold |
| `len_high_threshold` | float | `2.8` | Length ratio at high rate |
| `short_log_interval_seconds` | float | `2.0` | Per-req throttle for INFO short logs |

### output_substring

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `patterns` | list | `[]` | Token-id subsequences or string patterns to match |
| `add_special_tokens` | bool | `false` | Include special tokens when encoding string patterns |
| `match_prefix` | bool | `false` | Match only at output prefix vs anywhere |

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
| `deferred_queue_max` | int | `256` | Cap in-flight deferred alert batches (async keeps ~1–2 steps) |

Each step runs a device ``isfinite`` reduction and one ``.item()`` gate on the
pre-sample hook. On a hit only, bad rows / ``logits_indices`` / ``finite_kind``
are resolved while tensors are live and host ``Incident``s are enqueued.
``check_deferred`` on after-sample / ``get_output`` drains that queue (for dump
timing). Retired: ``check_every_tokens`` (multi-step window) — ignored if present.

### manual_trigger

Not a detector — control-plane for `dump.manual_dump`. Always runs `dump_kv` over the live batch (`scope=all_requests`; configured `dump_kv.scope` is ignored). `on_trigger` may still list `report` / other actions; if omitted, `actions.defaults` apply and `dump_kv` is injected.

**Prefer `dump.manual_dump: 1` (one shot).** Continuous `true` / large N adds little for debugging and can flood the action queue and disk; hot-reload again when you need another capture.

Each armed wave with scheduled tokens decrements `manual_dump` **in memory** after handle (`true` = continuous, no decrement), **even if** `dump_kv` fails to queue. Skips write `{dump_root}/<type>/<req_id>/wave_<N>/dump_skipped.json` with a `reason` (same layout as successful `request_info.json`). The JSON file is updated to `false` **only when the count reaches 0**, not on every decrement. While the in-memory count is still >0, a hand-edit to the file still hot-reloads into memory. Multi-DP sharing one file: each DP may dump up to N times (file stays at N until some replica persists `false`).

```json
"manual_trigger": {
  "on_trigger": ["report", "dump_kv"]
}
```

## Example (detection + dump on repeat)

```json
{
  "sync_mode": "broadcast",
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
- [runtime_guard_design.md](../../developer_guide/Design_Documents/runtime_guard_design.md)
- [runtime_guard_ops.md](../../developer_guide/Design_Documents/runtime_guard_ops.md)
