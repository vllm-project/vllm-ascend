# SPDX-License-Identifier: Apache-2.0

"""runtime_guard / runtime_config test matrix (native dump_kv).

Priorities: **P0** every PR / smoke; **P1** full suite; **P2** env-dependent.

## P0 — correctness & isolation (must not affect inference)

| ID | Area | What | Expect |
|----|------|------|--------|
| I1 | Isolation | Code present but **no** `--additional-config` / default reload=0, detectors off | Same tokens as baseline (temp=0); TPS within noise |
| I2 | Isolation | Feature wired, detectors off, dump off, reload=0 | Same as I1 (noop hot path) |
| I3 | Isolation | reload>0, all detectors off, dump off | TPS ≈ I1 (≤~1–2% typical); no report spam |
| I4 | Soft-fail | Malformed JSON on hot-reload | Keep old config; service alive |
| I5 | Soft-fail | Unknown / wrong-type fields | Soft warn; no crash |
| I6 | dump_kv | Empty `block_ids` | **No** full-cache D2H; skip / no dump |
| I7 | Output | Report / print_output must not mutate sampler outputs | HTTP body unchanged vs I1 |
| O1 | Output integrity | `print_output_on_finish` text vs HTTP (no MTP) | String-equal (trim) |

## P0 — function smoke (UT + short e2e)

| ID | What | Expect |
|----|------|--------|
| S1 | Config load defaults | `dump_enabled` false; detectors off |
| S2 | Hot-reload mtime/content | `sync_runtime_config` / `reload` picks detector enable |
| S3 | manual_dump / manual_trigger | Report + optional dump_kv files under `kv_cache/` |
| S4 | token_repeat detector (unit) | Hits on synthetic repeats; miss on unique ids |
| S5 | Wave + quota | wave stamp / consume / cooldown |
| S6 | ActionQueue | Async commit; full→inline fallback |
| S7 | *(deferred)* Offline KV compare tooling | Follow-up PR |

## P1 — detectors / report / dump_kv

| ID | What | Expect |
|----|------|--------|
| T1–T3 | output_substring / token_repeat / logits_finite | Hit/miss; max_per_req stop-detect |
| R1–R6 | block_ids / sensitive ids / truncate | Schema matches config flags |
| D1–D5 | dump_kv scope=request, quota, cooldown | Files only for armed req; quota respected |
| C1–C4 | v1 vs v2 runner hook parity | Same report fields / detector hits |

## P2 — stress / DP / MTP

| ID | What |
|----|------|
| X* | DP>1 sync, long prefill dump size |
| O2+ | MTP output integrity (known risk area) |

## P0 — isolation (what product UT proves)

| Experiment | In this repo UT? | How |
|------------|------------------|-----|
| **A2** Empty `block_ids` never full-cache D2H | **Yes** | `test_detectors_and_kv.py` |
| **A3** Soft-fail bad JSON | **Yes** | `tests/ut/runtime_config/test_runtime_config_core.py` |
| Hot-path gate / idle sync correctness | **Yes** | `test_hot_path_overhead.py` (no wall-clock) |
| Live NPU TPS / wall-clock microbench | **No** | Out of this product tree (manual Ascend runs) |

CI proves **functional isolation** and detector/report/dump contracts.  
Throughput / µs budgets need Ascend hardware and are not gated here.

## P0 — review regression suite (2026-09-01 white-box review; `test_review_regressions.py`)

| ID | Finding | What | Expect |
|----|---------|------|--------|
| V2 | P0-5 | `dumps_report_json` | np.int64 / torch scalar / NaN never lose the report |
| V3a–d | P0-1 | soft-fail contract | Detector/hook exceptions never reach engine loop / async copy thread / sampler |
| V4 | P0-2 | Shipped `runtime_config.example.jsonc` | Loads + validates as-is; reload(force) succeeds |
| V5 | P0-3 | Bootstrap invalid content | Falls back to defaults; service starts |
| V6 | C1 | `sync_mode` hot-reload | Frozen at first apply (DP collective safety) |
| V8a/b | B2 | Wave stamps lifecycle | discard on reap; no unbounded `_sample_waves` growth |
| V9a/b | B3 | ActionQueue full/stop | Heavy (dump) jobs dropped, never inline; stop works with full queue (drain + sentinel) |
| V9d | — | ActionQueue `dedupe_key` | Same key while queued/running → skip + INFO; different wave key still enqueues |
| V10 | C5 | Unknown detector sub-key | Reload rejected loudly (typo like `windw` no longer silently defaults) |
| V10b | W2-1/F-07 | Unknown top-level key | Typo like `{"windw":10}` rejected on validate/reload; old config kept; ERROR log |
| V12/12b | A1 | logits_finite attribution | Unattributable rows → warning only (no guess, no null-req report); decode rows still per-request |
| V13 | P0-2 | JSONC parsing | `//`, `/* */` comments + trailing commas accepted (string-aware) |
| V14/14b | B'4 | DumpQuota | Atomic `try_consume` (cap + cooldown); blocked consume doesn't burn; `refund` |
| V15/15b | B'2 | ReportWriter dedupe | `max_per_req` (default 1) + write-detect at cap; wave backoff (64,×2) between writes |
| V16/16b | — | detector `clear_finished` | Per-req state (buf/history) fully dropped, no cross-request leaks |
| V17 | B9 | spec_acceptance short batch | accepted shorter than req_ids: no IndexError |
| V17b/c | — | spec_acceptance v2 req_ids | `input_batch=None` + explicit `req_ids`; manager forwards to `check_all` |
| V18 | B'6 | dump payload | Carries `tp_rank` / `num_kv_heads` for cross-TP comparison |
| V21* | — | default-path merge | Missing JSON keys fall back to `_DEFAULTS` |
| V23* | — | dump_root / ReportWriter dump_dir | JSON / startup seed / provider |
| V25* | — | rank gate TP0-only | last PP + TP0 detect; leftover exec_scope / detector_placement ignored |

Related behavior contracts changed by the review fixes (mirrored in old UTs):
`_slice_blocks` returns `(tensor, used_ids)` (B'5c payload alignment); ActionQueue
`submit(job, heavy=)` replaces `sync_fallback=` (light → inline fallback, heavy → drop);
Store-side `sample_waves` FIFO removed — drain gating via `WaveTracker.pending` probe.

## P0 — logging (UCM bypass, 2026-09-12)

Ascend containers may unconditionally replace `vllm.logger.init_logger` (see
`docs/source/developer_guide/Design_Documents/runtime_guard_design.md` §2.5).
`init_logger_ascend` uses `logging.getLogger` + `_METHODS_TO_PATCH` so Ascend /
runtime_guard loggers stay on the stdlib tree.

| ID | What | How (UT) | Expect |
|----|------|----------|--------|
| L1 | Logger type | `import vllm_ascend.runtime_guard.detector.logits_finite as m; type(m.logger)` | `logging.Logger`, registered in `Logger.manager.loggerDict` (not `ucm.logger.Logger`) |
| L2 | Once helpers | `hasattr(logger, "info_once"/"debug_once"/"warning_once")` | All True; repeated calls dedupe |
| L3 | Level control | `logging.getLogger("vllm_ascend.x").setLevel(WARNING)` then `info` | INFO suppressed (UCM previously locked INFO) |
| L4 | caplog | `test_v12_logits_finite_unattributable_row_warns_not_misattributes` | caplog / leaf handler captures warning |
| L5 | Per-module levels | `apply_ascend_log_level(module_levels=...)` | Override applies (was a silent no-op under UCM) |

UT command: `pytest tests/ut/runtime_guard/ tests/ut/runtime_config/ -q`.
Live-NPU log-level smoke (`ascend_log` / `set_log_level` → DEBUG) is manual when needed.
