# Runtime Guard Operations and Troubleshooting

> For deployment / on-call.  
> Design details: [runtime_guard_design.md](./runtime_guard_design.md);  
> Config fields: [runtime_config.md](../../user_guide/configuration/runtime_config.md).

## 1. Minimum Viable Configuration

Detection + report only (no KV dump):

```bash
vllm serve <model> --additional-config '{
  "runtime_config_path": "/data/runtime/config/runtime_config.json",
  "runtime_config_reload_interval": 5
}'
```

Example `/data/runtime/config/runtime_config.json`:

```json
{
  "detector": {
    "token_repeat": { "enabled": true },
    "logits_finite": { "enabled": true }
  },
  "dump": { "auto_max_times": 0, "manual_dump": false }
}
```

You can also overlay at startup via `additional_config.runtime_config` **without editing the JSON file**:

```bash
vllm serve <model> --additional-config '{
  "runtime_config_reload_interval": 5,
  "runtime_config": {
    "detector": {
      "token_repeat": { "enabled": true, "on_trigger": ["report", "dump_kv"] },
      "logits_finite": { "enabled": true }
    },
    "dump": { "auto_max_times": 3, "auto_cooldown_seconds": 300 }
  }
}'
```

Merge order: at startup `defaults ← additional_config.runtime_config`, with overwrite to disk; on hot reload `defaults ← JSON`.  
Hot reload re-reads the JSON file only; it does not re-apply the startup overlay.

## 2. Common Operations

### 2.1 Enable detection / dump_kv (independently)

| Goal | Configuration |
|------|------|
| Report only | `"on_trigger": ["report"]` or omit (default) |
| Report + KV | `"on_trigger": ["report", "dump_kv"]` and `dump.auto_max_times > 0` |
| Log level only | `"on_trigger": ["set_log_level"]` + nested `set_log_level` |

All detectors default off; set `enabled: true` per detector.

### 2.2 Manual `dump.manual_dump`

| Field | Meaning |
|------|------|
| `manual_dump: false` | Off |
| `manual_dump: 1` (recommended) | **One shot**: triggers on the next wave with scheduled tokens, then resets to zero |
| `manual_dump: N` (N>1) | One trigger per wave with scheduled tokens for N consecutive waves |
| `manual_dump: true` | Dump every wave until hot reload sets false (**not recommended**) |

**Recommendation: `1` (or occasionally `2`) is enough.** Continuous / `true` adds little value—duplicate dumps for the same batch, fills ActionQueue (heavy `.pt`), consumes disk, and helps debugging only marginally; hot reload again when you need another capture.

**Requires `runtime_config_reload_interval > 0`**. Manual events **skip auto quota/cooldown**.

Manual triggers use incident_type `manual_trigger`. **`dump_kv` is always injected** and **forced** to `scope=all_requests` (configured `dump_kv.scope` is ignored). `on_trigger` can still include `report` / `set_log_level`; when omitted, default includes `report`, then `dump_kv` is added automatically.

**Counting and on-disk behavior:**

- Every wave with scheduled tokens **always** decrements `manual_dump` in memory after `_handle_manual_trigger` (continuous `true` mode does not decrement), **regardless** of whether `dump_kv` enqueue succeeds.
- If dump is skipped (no batch / insufficient disk / finished / queue failure, etc.), write  
  `{dump_root}/<type>/<req_id>/wave_<N>/[rank_tag/]dump_skipped.json`  
  with `reason` + `stage` (`arm`|`drain`); **do not** retain count for retry.  
  At drain, **each last-PP TP** writes its own marker under `rank_tag/` (including `d2h_failed` / `no_tensors` / `torch_save_failed` / `empty_block_ids`). Markers mean **incomplete success** (side paths may already have `request_info` or partial `.pt`).
- **JSON is not rewritten on every decrement**; when the in-memory count reaches **0**, the JSON writer **writes once** with `manual_dump: false`.
- While in-memory count is still >0, if someone edits the config file (content change), hot reload still applies the new value to memory (can increase/decrease/stop mid-run).
- **Multiple DP sharing one `runtime_config.json`:** the file stays at the original `N` until some replica hits 0; each DP replica decrements in memory independently, so **each DP can dump at most N times**; cluster worst case ≈ `num_DP × N`. When one DP persists `false`, others hot reload to `false` and stop early.

### 2.3 Report fields and truncation

On disk (last-PP TP0 only): `{report_dir}/<type>/report_<ms_timestamp>[_<req_id>]_pid<pid>.json`

Common top-level fields: `incident_type`, `req_id`, `rank`, `dump_attempted` (on_trigger includes `dump_kv`; not the same as D2H success), `dump_arm_wave` (arm wave; may differ from actual `wave_*` when deferred), `dump_dir` (req root; `wave_*` underneath), `dump_count` / `dump_max_times`, `detail`.  
**Manual**: one report **per real req** in the batch (top-level `req_id` / `dump_dir` align to that req; `detail.trigger_req_id` is `__manual_trigger__`). No single synthetic aggregate report.

Same `(incident_type, req_id)`: `report.max_per_req` (default **1**); when full, **stop detection for that req**; multiple reports use **wave backoff** (64×2ⁿ). Default `on_trigger` includes `report`. Unrelated to `dump.auto_cooldown_seconds`.

| Field | Purpose |
|------|------|
| `report.save_sensitive_info` | Whether to write prompt/output token ids |
| `report.max_prompt_token_ids` | Truncation cap (0 = unlimited) |
| `report.max_output_token_ids` | Truncation cap |
| `report.max_per_req` | Max reports per (type, req); stop detection when full (default 1) |
| `report.include_block_ids` | Include GPU block_ids in detail |
| `report.decode_token_ids` | Decode text when sensitive mode is on |

If reports are too large when investigating hits, turn off `save_sensitive_info` or lower max_*.

### 2.4 KV dump layout and ranks

**Scope: last PP × all TP. Other PP stages do not dump.**

| Rank | When dump runs |
|------|-----------|
| last PP TP0 | On detect/manual hit, queue only; **same wave** D2H with other TP after prepare |
| last PP all TP | **Same wave** `end_of_wave_sync`: after `{req_id, ...}`, each reads local block table then D2H |
| Other PP | No dump |

```text
{dump_root}/<incident_type>/<req_id>/wave_<N>/
  request_info.json          # last-PP TP0, written at arm
  dp{D}_tp0_pp{last}_cp{C}/  # D2H same beat as other TP
  dp{D}_tp1_pp{last}_cp{C}/
  ...
    {req_id}_{layer}_req.pt
```

`dump_root` defaults to `<report_dir>/kv_cache`. Each `.pt` includes: `req_id`, `block_ids`, `layer`, `rank_tag`, `tp_rank` / `pp_rank` / `cp_rank`, `num_kv_heads`, `tensor`.

Combine **all `tp*` dirs under the same `req_id` + same `wave_*` on last PP** for that stage’s head split. Missing earlier `pp*` dirs is expected (those layers were not dumped).

- `scope=request` (default, **auto detection**): each rank uses local `block_ids_for_request`; skip if request finished or table empty.
- `scope=all_requests` (configurable for auto; **manual_trigger always this**): at arm, enumerate local batch via `iter_local_request_rows`, **per req** `block_ids_for_request`.
- **At most one dump pending per `req_id` per arm wave** (`queue_kv_dump` dedupes by `(wave, req_id)`; duplicate enqueue logs INFO and skips).
- **Report on ActionQueue**: skip duplicate when pending/running report commit exists for same `(wave, req_id)` (INFO).  
  **`.pt` is one heavy job per layer**; no `(wave, req_id)` dedupe (would keep only the first layer).
- After successful queue, last-PP TP0 writes request metadata to `{dump_root}/<type>/<req_id>/wave_<N>/request_info.json`.
- When **no complete `.pt`** (arm or drain), write  
  `{dump_root}/<type>/<req_id>/wave_<N>/[rank_tag/]dump_skipped.json`:  
  `reason` such as `finished_or_reaped` / `no_dump_targets` / `empty_block_ids` / `insufficient_free_space` / `quota_or_cooldown_blocked` / `queue_failed` / `d2h_failed` / `no_tensors` / `torch_save_failed`; `stage` is `arm` or `drain`. On drain failure **each TP writes under its rank dir**; file means incomplete success, not an empty directory.  
  Logs: `[runtime_guard dump_kv] skipped reason=...` / `skip empty local block_ids`, etc.
- Empty local `block_ids`: **auto detection skips dump**; **manual** still enqueues, resolves blocks after same-wave `prepare`, then at end of `run_sample_phase` via `end_of_wave_sync` (does not drag full pool KV).
- `.pt` tensor stays `[n_blocks, block_size, …]` (blocks for that request only).
- **Quota:** one `try_consume` per `prepare`; multiple req jobs under same `arm_id` **refund once** only if **all** fail to produce D2H at drain (restore count and clear cooldown). Async `torch.save` failure **does not** refund (quota counts D2H opportunity, not disk success). When ActionQueue is **full**, heavy (`.pt`) jobs are **dropped** with WARNING; light (report) runs inline.
- TP=1: no list broadcast; same-wave flush dumps one `tp0` dir only.
- Dump / config notification (PP==1 broadcast): **one** `all_reduce([config_due, dump_due])` at wave head; each due lane gets `broadcast_object` (normally one AR). Config applies this wave.
    - **Auto single-request dump**: detect enqueue this wave, **next wave head** bcast, D2H at that wave end (+1 wave).
    - **manual_dump**: each last-PP TP dumps **locally** at wave end (no dump-job bcast).
    - **PP>1 (forced file) / file**: config local poll; auto dump at wave head TP drain → D2H at end.
- D2H timing: arm queues only; **same wave** at end of `run_sample_phase` (`check_after_sample` then) via `end_of_wave_sync`; when no sample this step, sync path uses the same end entry.

### 2.5 Logging switches and UCM

| Config | Purpose |
|------|------|
| `log.print_output_on_finish` | Log output token ids / decoded text when request finishes (TP0) |
| `ascend_log.level` / `modules` | Module log levels; `[SamplingMeta]` at DEBUG after sample (enable DEBUG on `runtime_guard` to see) |
| `set_log_level` action | Temporarily raise logs on incident (nested `set_log_level` under detector section) |

**UCM hijack (common in Ascend containers)**

- Symptom: `ascend_log` / `set_log_level` level changes have no effect; DEBUG missing; UT `caplog` cannot capture `vllm_ascend.*` warnings.
- Cause: container UCM may replace `vllm.logger.init_logger`; loggers from that entry can have levels locked in C extensions.
- This repo: runtime_guard / Ascend modules use `init_logger_ascend` (stdlib + once methods), unaffected by UCM replacement.
- Note: upstream code still using `vllm.logger.init_logger` may remain affected by UCM.
- Preflight: `type(logging.getLogger("vllm_ascend.runtime_guard..."))` should be `logging.Logger`, and present in `Logger.manager.loggerDict` (UT checklist: `tests/ut/runtime_guard/TEST_MATRIX.md` L1–L5).

## 3. Troubleshooting Quick Reference

| Symptom | Check |
|------|------|
| No report | Detector `enabled`; last PP + TP0; rank skip logs |
| Report but no kv | `on_trigger` includes `dump_kv`; quota/cooldown; `auto_max_times` is 0 |
| Only `tp0`, no other `tp*` | TP>1; other last-PP TP reached same-wave `end_of_wave_sync`; logs `[runtime_guard dump_kv]` |
| Missing earlier PP layers | Expected: non-last PP not dumped today |
| Dump empty / missing layers | `block_ids`; `dump_skipped.json`; logs `skip` / `skipped reason=` / `action queue full` |
| Repeat arm, no second dump | Expected: same wave same req dedupe; log `skip enqueue: already pending` |
| Missing one report | `max_per_req` / wave backoff; or log `skip report enqueue (dedupe or stopping)` |
| Hot reload not applied | `runtime_config_reload_interval` >0; JSON path readable on all ranks |
| Repeated report spam | `report.max_per_req` (default 1, stop when full); confirm `on_trigger` includes `report` |
| Manual not firing | reload interval; idle dummy wave; `manual_dump` count exhausted |
| Performance regression | Disable all detectors, keep reload only, re-enable one by one (see feature guide) |

## 4. Log Keywords

| Keyword | Meaning |
|--------|------|
| `[runtime_guard sync]` | Per-wave config sync |
| `[runtime_guard manual_trigger]` | Manual dump control plane |
| `[runtime_guard dump_kv]` | KV queue / D2H / disk; includes `skip enqueue: already pending` (same-wave dedupe) |
| `[runtime_guard action]` | prepare/commit failures; `skip report enqueue` / `skip heavy enqueue` |
| `action queue skip enqueue: duplicate key` | Report commit dedupe for same `(wave, req_id)` |
| `action queue full` | heavy `.pt` dropped / light inline |

## 5. Disk and Quota

- Each auto `dump_kv` consumes quota (`auto_max_times` is per-process cumulative cap; need **refund** or **restart** after exhausted. `auto_cooldown_seconds` only spaces two **successful consume** events, does not raise cap).
- After successful `try_consume`, if enqueue fails (e.g. queue unavailable) or whole arm drains without D2H, **refund** (restore count and **clear cooldown**), avoiding empty deduct stuck in cooldown.
- Free space threshold: leader single-GPU payload estimate × `tp_size` + `free_headroom_bytes` (all TP write).
- Manual path does not consume auto quota.
- Long-running `dump_kv`: watch `{report_dir}/kv_cache/` disk; archive periodically or lower `auto_max_times`.

## 6. Related Documentation

- [runtime_guard_design.md](./runtime_guard_design.md)
- [runtime_guard.md](../../user_guide/feature_guide/runtime_guard.md)
- [runtime_config.md](../../user_guide/configuration/runtime_config.md)
