# Runtime Guard Design (vllm-ascend)

> Runtime anomaly detection and incident response control plane.  
> Code root: `vllm_ascend/observability/runtime_guard/`  
> Config module: `vllm_ascend/observability/runtime_config/`

## 1. Components and Flow

| Component | Module | Responsibility |
|------|------|------|
| Runtime Config | `runtime_config/config.py` (`RuntimeConfig`) | Single JSON file; optional hot reload (interval set at startup) |
| Detector | `detector/` | Anomaly detection; produces `Incident` |
| Action | `action/` | Async handling: `report`, `dump_kv`, `set_log_level` |
| Report | `report.py` (`ReportWriter`) | Short incident reports written under `runtime/report/` |
| KV dump | `kv_cache_reader.py` (`KvCacheReader`) | Native D2H per request blocks; async `.pt` writes |
| Processor | `processor.py` (`RuntimeGuardProcessor`) | Runner-side orchestration: bind / `sync_for_step` / sample hooks / `_handle_alert` → ActionExecutor; `[SamplingMeta]` DEBUG dump lives in `sampling_meta_debug.py` (log level only, no JSON switch) |
| Request state | `request_state.py` (`RequestGuardStore`) | Per-request shared state; delayed `clear` after `mark_finished` |
| I/O snapshot | `io_snapshot.py` (`RequestIoSnapshotManager`) | Report I/O view (normalize→Store + `snapshot`) |
| Quota | `quota.py` (`DumpQuota`) | Auto dump count cap and cooldown |
| Rank gate | `rank_gate.py` | Detection / report: last-PP TP0; dump: all TP on last PP |

Public entry: `from vllm_ascend.observability.runtime_guard import RuntimeGuardProcessor`

```text
additional_config
  ├─ runtime_config_path / runtime_config_reload_interval
  └─ AscendConfig.runtime_config (RuntimeConfig)
         │
Worker: RuntimeGuardProcessor.bind(runner)

  ① execute_model (this call)
  │  sync_for_step()                    # wave head: config hot reload + dump list due bus
  │                                      # static idle (reload=0, no detectors/print/manual):
  │                                      #   skip config bus; claim TP dump bus only if dump_enabled
  │  │  ├─ broadcast∧PP==1: 1×AR([config_due,dump_due]) + per-lane bcast as needed
  │  │  └─ file / PP>1: local poll JSON; dump list via last-PP TP bus
  │  try:
  │    prepare → forward
  │      · non-last PP: often only IntermediateTensors; this call ends with no sample
  │      · last PP and sampling needed: leave execute_model_state; engine invokes ② after return
  │  finally:
  │    if dummy_run or execute_model_state is None:
  │      end_of_wave_sync(allow_arm=False)   # wave tail when no follow-up sample (mutually exclusive with ②)
  │
  ② sample_tokens (separate call; only when ① left execute_model_state)
  │  run_sample_phase(sample_fn=…)
  │    ├─ sample_fn() (original sampling)
  │    │    └─ if need_pre_sample_hook [= logits_finite.enabled ∧ last-PP∧TP0]
  │    │         then wrap compute_logits → check_before_sample (before grammar)
  │    ├─ if not needs_sample_phase_hooks:
  │    │     sample_fn → (optional routed_experts) → end_of_wave_sync → return
  │    │     (still enters wave tail: flush wave-head deferred; no collective at wave tail)
  │    └─ else:
  │         sample_fn → mark_finished
  │         → if spec ∧ should_check_after_spec → check_after_spec
  │         → if sync∨(async∧TP0) → record_sample_waves
  │         → if NOT (use_async ∨ AsyncModelRunnerOutput)
  │              then same-wave check_after_sample
  │                   · leader (last-PP TP0): substring/repeat → report / queue dump_kv
  │                   · else: no detect
  │              else defer to AscendAsync*.get_output (after D2H+trim) then check_after_sample
  │         → end_of_wave_sync(allow_arm=True)   # wave tail when sample ran (mutually exclusive with ① finally)
  │
  ③ (optional, deferred) executor calls get_output on output rank → check_after_sample
  │
  end_of_wave_sync at same wave end (whether triggered by ① or ②, once per wave):
    ├─ deferred non-empty and last-PP TP → each rank local D2H (list synced at wave head)
    └─ allow_arm ∧ manual_dump → each last-PP TP local D2H
```

### Report / dump_kv pipeline

When `on_trigger` for one incident includes both `report` and `dump_kv`, the executor **enqueues report first, then queues dump_kv** (no D2H in this step).  
By default `dump_kv` dumps only the **paged blocks** the request occupies (`block_ids`), not the full KV pool.  
After successful queue, last-PP TP0 writes request metadata to `{dump_root}/<type>/<req_id>/wave_<N>/request_info.json` (field policy matches report: counts always; token ids gated by `report.save_sensitive_info`).  
`.pt` tensors stay `[n_sel_blocks, block_size, …]`.

Before write, free space on the target directory is checked using **leader single-GPU estimate × `tp_size` + `dump.free_headroom_bytes` (default 5GiB)** (all last-PP TP ranks write). Insufficient space skips queue (does not consume auto quota). After successful `try_consume`, if queue fails or the whole arm drains without D2H, **refund** (restore count and clear cooldown).  
After-sample CPU detection alerts slightly later on the ActionQueue. On any path, if the request is already `finished` or reaped, **always skip dump** (KV may be freed/reused). `logits_finite` already `.item()`’d at before-sample and (on hit) parsed and enqueued; `check_deferred` still runs at `get_output` / after-sample drain.

### dump_kv rank coverage (last PP × all TP)

Detection runs only on last-PP TP0. Other TP ranks have no incident and cannot decide dump alone. The last-PP **TP group for this stage** carries the list.

| Who | What |
|----|--------|
| last PP + TP0 | Detect, write report, `queue_kv_dump` records `{req_id, ...}` (no D2H this step) |
| last PP + all TP (incl. TP0) | **Same wave** `end_of_wave_sync`: after receiving `req_id`, each dumps **this rank’s** shard |
| Other PP stages | **No dump** (those layers’ KV is out of scope) |
| Other DP replicas | Not involved (no KV for this request) |

```text
step N  wave head: if broadcast∧PP==1 → merged AR+bcast; else file(+TP dump bus)
        prepare → forward (if non-last PP: often no sample)
        → if need_pre_sample_hook → check_before_sample (logits_finite)
        → run_sample_phase:
             if not needs_sample_phase_hooks → sample_fn → end_of_wave
               (no after hooks still enter wave tail: flush wave-head deferred; no bus at wave tail)
             else sample_fn → mark_finished
                  → if spec∧should_check_after_spec → check_after_spec
                  → if sync∨(async∧TP0) → record_sample_waves
                  → if sync∧non-AsyncOutput → check_after_sample (leader detect/arm)
                    else → check_after_sample after AscendAsync.get_output
                  → end_of_wave_sync
        → end_of_wave: local deferred/manual D2H (no collective; list sync only at wave head)
        (dump armed last wave often D2H at this wave head bcast/claim, typically +1 wave)
```

On the sync path, after-sample armed jobs can D2H in the same wave; async `get_output` and ActionQueue CPU alerts often land on the next wave flush. Requests already `mark_finished` this step still skip dump via the finished gate.  
When a complete `.pt` is not produced, relevant ranks write `{dump_root}/<type>/<req_id>/wave_<N>/[rank_tag/]dump_skipped.json` (`reason` + `stage=arm|drain`).

On disk:

```text
{dump_root}/{incident_type}/{req_id}/wave_{N}/dp{D}_tp{T}_pp{P}_cp{C}/{req_id}_{layer}_req.pt
```

Combine all `tp*` dirs under last PP for that stage’s full head split; **no** other `pp*` dirs is expected.

Cost: with PP==1 broadcast, dump and config share one `all_reduce` due bit; no `broadcast_object` when nothing is due. **PP>1 forces `sync_mode=file`** (each rank reads JSON for config); dump uses last-PP TP group `all_reduce(has_job)`. Does not change `SchedulerOutput` or use collectives on the PP group.

## 2. Runtime Config

### 2.1 Paths

| Item | Description |
|----|------|
| Default file | `<cwd>/runtime/config/runtime_config.json` |
| Explicit path | `additional_config.runtime_config_path` |
| Report root | Default `<cwd>/runtime/report`; override with `runtime_report_dir` |
| Example | `vllm_ascend/observability/runtime_config/templates/runtime_config.example.jsonc` |

### 2.2 Sync mode `sync_mode`

| Value | Behavior |
|----|------|
| `broadcast` (default) | **PP==1 only**: one `all_reduce([config_due, dump_due])` at wave head; each due lane gets its own `broadcast_object`. **PP>1 is forced to file**. |
| `file` | Each rank polls `runtime_config_path` (shared disk or per-node copy); dump uses last-PP TP `all_reduce(has_job)` |

**Note**: Config hot reload does **not** use a full-world collective across DP replicas. With multiple DP, each EngineCore maintains its own readable JSON.

### 2.3 Hot reload

- Enable: `additional_config.runtime_config_reload_interval > 0` (set at process start; `reload_interval_seconds` in JSON is display-only).
- `interval = 0`: config is static after startup; only startup overlay and one-shot `manual_trigger` remain.
- Hot reload failure (malformed JSON): keep old config; service continues.

Merge order: at startup `defaults ← additional_config.runtime_config` (overwrite on disk); on hot reload `defaults ← JSON`.

### 2.3.1 Startup persist

Startup synthesizes effective config once (defaults + overlay + ctor seeds). The writer **fully overwrites** existing JSON on `_bootstrap(persist=True)` / `ensure_persisted()`. No read-merge from disk, no skip rewrite for explicit paths / dump_dir backfill. Hot reload still reads JSON only.

### 2.4 Top-level JSON structure

| Section | Purpose |
|----|------|
| `sync_mode` | How config is synchronized |
| `actions.defaults.on_trigger` | Default action list when unspecified |
| `dump` | Auto dump quota, `manual_dump` |
| `detector.*` | Per-detector switches and thresholds; optional per-type `on_trigger` override |
| `report` | Report fields, sensitive info, block metadata |
| `log` | Ops log switches (not written into report JSON) |
| `ascend_log` | Ascend module log levels |

Field details: [runtime_config reference](../../user_guide/configuration/runtime_config.md).

### 2.5 Logging switches and UCM

In Ascend containers, UCM may hijack `vllm.logger.init_logger`, so `setLevel` / `ascend_log` may not affect loggers created through that entry. runtime_guard uses `init_logger_ascend` uniformly (stdlib `getLogger` + vLLM once methods), bypassing that hijack.

For troubleshooting steps and symptom mapping, see the ops doc [runtime_guard_ops.md §2.5](./runtime_guard_ops.md#25-logging-switches-and-ucm).

## 3. Detector

Detection runs on **last PP + TP0** (same rank as `is_action_leader_rank`).  
Async scheduling’s `unique_reply_rank` only sends the output rank (TP0) return value into `enqueue_output` / `get_output`; forcing `get_output()` on other TP ranks blocks the next step’s `execute_model` TP collectives.  
Reports are written only on last PP + TP0. `dump_kv` rank coverage is in the previous section (last PP × all TP).

| incident_type | Hook phase | Description |
|---------------|----------|------|
| `spec_acceptance` | after spec | Speculative decoding acceptance rate anomaly |
| `output_substring` | after sample | Output token subsequence match |
| `token_repeat` | after sample | Sliding-window repetition score |
| `logits_finite` | before sample | Logits NaN/Inf: per-step `isfinite` + `.item()`; only on hit, parse bad row / indices / kind inline and enqueue host `Incident`; drain again at `get_output` / after-sample (helps dump). |

Shared behavior: stop detection when ``report.max_per_req`` reports are full (default ``actions.defaults.on_trigger`` includes ``report``).  
`output_substring` / `token_repeat` run on the `ActionQueue`; `logits_finite` gates with `.item()` at before-sample, parses and enqueues on hit, drains after-sample.

> Online KV / position meta detectors are planned in a follow-up PR.

Each detector can override actions via nested `on_trigger`, for example:

```json
"token_repeat": {
  "enabled": true,
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "request" }
}
```

## 4. Action

| name | sync_only | Description |
|------|-----------|------|
| `report` | no | Write `runtime/report/<type>/report_*.json` |
| `dump_kv` | no | last PP × all TP: each writes `{dump_root}/<type>/<req_id>/wave_<N>/<rank_tag>/*.pt` (see “dump_kv rank coverage”) |
| `set_log_level` | yes | Adjust Ascend log levels immediately |

`dump_kv` config (per detector):

| Field | Default | Description |
|------|------|------|
| `scope` | `request` | `request`: incident request only; `all_requests`: every request in current batch (arm bootstraps via `iter_local_request_rows` + per-req `block_ids_for_request`). **`manual_trigger` always uses `all_requests`, ignoring config.** |

Quota: `dump.auto_max_times > 0` enables auto dump quota; `dump.auto_cooldown_seconds` controls cooldown.  
One `arm_id` is shared per `dump_kv` prepare: refund once only if the entire arm drains without D2H (restore count and clear cooldown); async `torch.save` failures do not refund quota.  
`manual_dump` / `manual_trigger` bypass auto quota (see ops doc).

## 5. Report

On disk: `{report_dir}/<incident_type>/report_<timestamp_ms>[_<req_id>]_pid<pid>.json` (last-PP TP0 only).

Common fields: `incident_type`, `req_id`, `rank`, `detail`, `dump_attempted`, `dump_arm_wave`, `dump_dir`, `dump_count` / `dump_max_times`.  
Same `(incident_type, req_id)`: `report.max_per_req` (**default 1**) caps report count; **after full, stop all detection for that req**. Multiple reports use **wave** backoff (first interval 64, then double). Default `on_trigger` includes `report`; if a detector overrides away `report`, detection does not stop on report cap.  
With `report.save_sensitive_info=true`, persist prompt/output token ids (truncation; `decode_token_ids` controls text decode).

## 6. Model Runner Integration

| Runner | bind | Main hooks |
|--------|------|----------|
| v1 | `model_runner_v1.py` ctor | `sync_for_step`, `run_sample_phase` (after_sample, etc.), pre-sample wrap, async `AscendAsync*` |
| v2 | `worker/v2/model_runner.py` ctor | same |

Idle DP: `worker.execute_dummy_batch` calls `sync_for_step(allow_arm=False)` to align config hot reload with busy ranks.

v1/v2 wrap `compute_logits` to insert `check_before_sample` (`runner_bridge.wrap_compute_logits_for_pre_sample`).

## 7. Process Roles and Multi-DP

- **`.bind(runner)`**: This process creates/binds the `RuntimeGuardProcessor` singleton to ModelRunner (once in v1/v2 ctor). Later `sync_for_step` / `run_sample_phase` use that singleton.
- **Worker**: Distributed worker process running NPU forward; config hot reload locksteps with `execute_model` / dummy batch (broadcast or file).
- **Non-worker**: e.g. API server, EngineCore front-end — **no** ModelRunner hot path. With hot reload enabled, `RuntimeConfig.start_non_worker_background_reload()` starts a background thread that **file-polls JSON + reapplies `ascend_log` only**, no worker collectives.
- **EngineCore**: vLLM V1 scheduler/engine process; with `data_parallel_size>1`, typically **one EngineCore per DP (+ its worker group)**. This is what the doc means by “multiple engines” — **present in production multi-DP**, not a separate runtime_guard product; each binds separately and reads its own (or shared-readable) JSON.
- **DP replica**: One EngineCore + that DP’s TP/PP workers, holding one data-parallel shard of the full model replica. **No** world collective hot reload across replicas.

## 8. Related Documentation

- Operations and troubleshooting: [runtime_guard_ops.md](./runtime_guard_ops.md)
- User feature guide: [runtime_guard.md](../../user_guide/feature_guide/runtime_guard.md)
- Config field table: [runtime_config.md](../../user_guide/configuration/runtime_config.md)
- Startup options: [additional_config.md](../../user_guide/configuration/additional_config.md)
