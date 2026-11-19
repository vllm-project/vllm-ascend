# Runtime Guard

Runtime Guard is vLLM Ascend's online anomaly detection and incident response layer. It watches decode-time signals (token repetition, garbled output, non-finite logits, speculative acceptance drift, and more), writes structured reports under `runtime/report/`, and optionally captures per-request KV cache blocks via native device-to-host dump (`dump_kv`).

## When to use

- Intermittent quality bugs: repetition, gibberish, sudden NaN/Inf
- Need on-call artifacts (JSON report + optional `.pt` KV slices) without patching the model
- Suspected KV issues: capture with `dump_kv` for offline inspection

Default deployment: **detectors off, hot-reload off** — negligible overhead until you enable features in `runtime_config.json`.

> **Review note (docs-only PR):** sections marked **Pending** document open feature PRs not yet on `main` (`#21492` pattern `token_repeat`, `#20138` `rare_chars`). They are included here so reviewers can see the full Guard docs surface in one place; landing those features still requires the corresponding code PRs.

## Quick start

**Detect + report only**

```bash
vllm serve Qwen/Qwen3-8B --additional-config '{
  "runtime_config_hot_reload": true,
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
  "runtime_config_hot_reload": true
}'
```

Edit the live `runtime_config.json` (JSONC comments/trailing commas OK) and set:

- `detector.<name>.enabled`: `true`
- `detector.<name>.on_trigger`: `["report", "dump_kv"]`
- `dump.auto_max_times`: e.g. `3` (required for auto dump quota)

**Manual one-shot dump (watermark):** start the server with `runtime_config_hot_reload=true` first (startup overwrites the JSON from defaults/overlay). Then edit the live file: set `dump.manual_dump` to `1`. After it fires, raise to `2`, `3`, … for another capture. Setting a large `N` while `done` is still `0` catch-up-dumps across that many later waves — prefer bumping by 1. See [runtime_config.md](../configuration/runtime_config.md#dump).

## Startup options

Configure through `--additional-config` (or `LLM(..., additional_config=...)`):

| Key | Type | Description |
|-----|------|-------------|
| `runtime_config_path` | str | Path to `runtime_config.json`. Default: `<cwd>/runtime/config/runtime_config.json` |
| `runtime_config_hot_reload` | bool | Enable hot-reload (startup-only; not a JSON field). `false` = static after startup (default) |
| `runtime_config` | dict | Startup overlay merged into JSON defaults |
| `runtime_report_dir` | str | Override report root (default `<cwd>/runtime/report`) |
| `runtime_dump_dir` | str | Seed KV dump root `dump.dump_dir` (default derived from report root). Hot-reload of `dump.dump_dir` in JSON wins after startup |

**Bootstrap overwrite:** At process start the effective config is built from **built-in defaults + `additional_config.runtime_config` overlay** (and dump-dir seed). Any pre-existing file at `runtime_config_path` is **ignored for that first load** and then **rewritten** when the worker leader persists (`(overwrote)` in the startup log). Put detectors / dump quotas you need at boot into the `runtime_config` overlay (or edit the JSON **after** start with hot-reload). Pre-seeding only the on-disk JSON without an overlay will look like “my file was wiped” — that is expected.

See [Additional Configuration](../configuration/additional_config.md#runtime_guard) and the full [runtime_config reference](../configuration/runtime_config.md).

## Architecture (summary)

```text
RuntimeGuardProcessor.bind(runner)
  → begin_wave()        # config + wave + manual triggers
  → detector hooks         # before/after sample (and after spec)
  → ActionExecutor         # report | dump_kv (async queue)
```

Design / ops runbooks live on the analysis branch (`docs/source/developer_guide/Design_Documents/runtime_guard_{design,ops}.md`), not in this product tree.

## On-disk layout

```text
runtime/
  config/runtime_config.json
  report/
    <incident_type>/report_<ms>[_<req_id>]_pid*.json   # last-PP TP0 only
    kv_cache/<incident_type>/<req_id>/wave_*/dp*_tp*_pp*_cp*/*.pt
    kv_cache/<incident_type>/<req_id>/wave_*/request_info.json
```

Report JSON top-level includes `dump_attempted` (whether `dump_kv` was in `on_trigger`, not whether D2H finished), `dump_arm_wave`, `dump_dir`, and optionally `request_finished_at_dump` (finish-wave arm — KV may be freed/reused). Same `(type, req_id)` is capped by `report.max_per_req` (default 1); reaching the cap stops detection for that request. Wave backoff (64, then doubles) spaces further writes when the cap is higher. Default `on_trigger` includes `report`.

## Detectors

| Type | Stage | Typical use |
|------|-------|-------------|
| `token_repeat` | after sample | Pattern-based stutter / repetition (in-tree native CPU extension — **Pending** `#21492`) |
| `logits_finite` | before sample | NaN/Inf logits (``isfinite`` + gate; default **async** ``.item()`` wait at after-sample — logits may be mutated by grammar before wait; set ``item_sync: true`` for sync pre-sample gate/kind). Hit resolves indices/kind then enqueues; after-sample drains. Unattributable rows: warning only (no guessed / null-req report). |
| `rare_chars` | after sample (Guard logits stash) | **Pending** [`#20138`](https://github.com/vllm-project/vllm-ascend/pull/20138): visible garbled / foreign-script / illegal UTF-8. Per-tokenizer table at warmup; Ascend Triton narrow collect on a **cloned** pre-grammar logits snapshot → `(top1, sampled)` only (no API ``logprobs``). Per-step `RareLogprobBatch` on `AscendAsyncOutput`; D2H + feed on `RareLogprobFeedQueue`. Shadow/report oriented. |
| `spec_acceptance` | after spec | Spec-decode acceptance drift (via `run_sample_phase` → `check_after_spec`; v2 stashes accept stats in `postprocess_sampled`) |

All detectors default to **disabled**. Enable individually under `detector.<name>.enabled`.

Online KV / position meta detectors are **not** in this release; use `dump_kv` for KV capture (offline compare tooling ships later).

> **Wiring (v2 only):** `@runtime_guard_sample_tokens` covers post-pre-sample hooks (`mark_finished` / `check_after_spec` / waves / `check_after_sample` when host ids are ready). Pre-sample `check_before_sample` is on the compute_logits wrap inside the sample decorator (before grammar; `logits_finite` only on `main` today — default async gate D2H there, wait in after-sample; `item_sync: true` for blocking `.item()`). **Pending `#20138`:** the same wrap may also stash logits for `rare_chars`; right after `sample_fn`, narrow collect fills a per-step `RareLogprobBatch`, and `get_output` submits it to `RareLogprobFeedQueue`. After-sample for async scheduling **and** for v2 `AsyncOutput` (even under sync scheduling) runs in `AscendAsyncOutput.get_output()` after D2H + `num_sampled` trim — do not append padded `AsyncOutput.sampled_token_ids` in the sync hook (W2-3 / D-11). Then drain `logits_finite` incidents and enqueue CPU detector (`token_repeat`) on `ActionQueue` (finish does not wait). `dump_kv` may still arm after `mark_finished` (finish can precede after-sample); reports / `request_info` / `.pt` set `request_finished_at_dump: true` in that case — KV may already be freed or reused, so treat tensors as suspect. Reaped ids are refused at arm.

## Actions

| Action | Effect |
|--------|--------|
| `report` | Write JSON incident report (+ metric counter) |
| `dump_kv` | D2H paged KV for the request. **Last PP × all TP** (not other PP stages): TP0 records dump jobs this step and writes `request_info.json` (report-like metadata); the next wave-head delivers the list on the merged due-bus dump lane (`config_due` may be false when hot-reload is off; `tp_size<=1` stashes locally with no TP collective), then each last-PP TP rank (including TP0) does local D2H in `end_of_wave`. Same arm wave: each `req_id` at most once (`queue_kv_dump` dedupes by `(wave, req_id)`; duplicate enqueue logs INFO). Files: `{dump_root}/{type}/{req_id}/wave_<N>/request_info.json` and `{dump_root}/{type}/{req_id}/wave_<N>/dp*_tp*_pp*_cp*/*.pt` |

Default `on_trigger` is `["report"]`. Per-detector overrides:

```json
"token_repeat": {
  "enabled": true,
  "think_end": -1,
  "on_trigger": ["report", "dump_kv"],
  "dump_kv": { "scope": "request" }
}
```

**KV dump rank coverage:** detection and reports stay on last-PP TP0. `dump_kv` captures **last PP × every TP rank**. **Only last-PP × all TP** join the merged due bus (config + dump); other processes poll the config JSON. Wave head **submits** one TP0-sourced `broadcast([wave_idx, config_due, dump_due])` (+ due-lane `broadcast_object`) to `DueBitsBusWorker`; **end-of-wave drains** (due-bcast ideally finished during forward). Detector dumps are delivered the **next** wave (+1); config apply lands at that wave's end-of-wave drain. Assemble shards from the `tp*` directories under the same `req_id`.

## Performance

- **No additional-config / defaults**: bind-only path; intended to be noise-free.
- **Hot-reload only** (`runtime_config_hot_reload=true`, all detectors off, dump off): on last-PP TP, the TP0 due-broadcast runs on the bus worker so the inference thread can overlap it with forward (C2); end-of-wave rate-limits a warning if the collective was not hidden. Other processes poll the config file.
- **Detectors on**: cost depends on enabled checks (light: `token_repeat`; heavier: `logits_finite` — default async gate overlaps sample; `item_sync: true` pays sync `.item()` each step).
- **dump_kv on hit**: detector arms queue for the **next** wave-head dump lane, then D2H at that wave's `end_of_wave` (after bus drain / prepare). Manual dump D2H's locally at end-of-wave (no dump-job bcast). Idle/broadcast steps submit a cheap due-vector broadcast at wave-head; per-lane `broadcast_object` runs only when that lane is due. Finish-wave hits set `request_finished_at_dump` on the artifacts.

Throughput A/B (with vs without Runtime Guard) should be measured on Ascend hardware; it is not gated by default CI.

## Feature compatibility

Runtime Guard is wired on **Model Runner v2** only (v1 is not wired). Tracking epic: [vllm-ascend#21976](https://github.com/vllm-project/vllm-ascend/issues/21976).

**白盒** = code-path analysis. **验证** = UT / live NPU. They are not the same.

| Area | Notes |
|------|--------|
| TP / PP | Detect last-PP TP0; `dump_kv` last PP × all TP. Exercised in functional runs. |
| DP | Per-replica report/dump; watch multi-DP `manual_dump` quota. |
| SP (MoE) | Sample-time rows are post all-gather; no Guard-specific SP handling needed. |
| PCP | Detection must use the post-`restore_for_sampling` batch (see #22026). Dump path tags need `pcp_manager.pcp_rank` (#22028). |
| DCP / CP interleave | Each rank dumps its **local dense shard**; no CP stitch in-tree. Treat `.pt` as per-rank shards. |
| Spec / MTP / adaptive | `spec_acceptance` needs real draft lengths, not padded row width (#22027). |
| Hybrid / multi KV group / draft KV | `dump_kv` must resolve `block_ids` per group (#22025). |
| C8 / TQ4 / kernel pages | Prefer forward-context tuple KV; expand manager→kernel ids when `blocks_per_kv_block > 1` (#22025). |
| Prefix cache / APC | Finish / +1-wave dump may see freed or reused pages (`request_finished_at_dump`). |
| KV Pool / Offload / PD disagg | Dump reads **device** `kv_caches` only; not a first-class offload/remote source. |
| Grammar / structured output | `logits_finite` default async gate may observe post-grammar logits; use `item_sync: true` for sync kind. |

Residual risks (PD cross-machine, DCP full-sequence stitch, offload-backed dump) stay documentation-only until product requires them.

## Token repetition (pattern, Pending)

> **Pending open PR:** [#21492](https://github.com/vllm-project/vllm-ascend/pull/21492).

See [token_repeat_pattern.md](token_repeat_pattern.md) for tokenizer warmup,
configuration, alert reasons, and validation. The native extension is built
with vllm-ascend (`native_pattern`); no separate install step. Sliding-window
`token_repeat` and multi-window `#19015` are not part of that PR.

## Related docs

- [runtime_config.md](../configuration/runtime_config.md) — JSON field reference  
- Compatibility epic: [vllm-ascend#21976](https://github.com/vllm-project/vllm-ascend/issues/21976)  
- Pending feature PRs (docs preview only): [#20138](https://github.com/vllm-project/vllm-ascend/pull/20138), [#21492](https://github.com/vllm-project/vllm-ascend/pull/21492)  
- Design / ops (analysis branch): `docs/source/developer_guide/Design_Documents/runtime_guard_{design,ops}.md`
