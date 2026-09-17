# runtime_guard anomaly injection test matrix

> Goal: prove each detector actually fires on its target anomaly, not just
> passes UT on synthetic happy-path inputs. Live NPU injection is the only
> way to close the "detection didn't trigger" risk — UT proves the detector
> logic, injection proves the wire-up.

## Why this exists

Ship injection for:

`output_substring` / `token_repeat` / `spec_acceptance` / `logits_finite`.

Still open even for shipped detectors:

- **max_per_req stop-detect cross-step skip NOT live-verified** — every
  prior detector alert was followed by server shutdown, so "subsequent steps
  skip this req_id after report write" was never observed.
- **Per-detector threshold boundaries NOT tested** — only `enabled=true/false`
  was toggled; window/threshold numerical behavior is UT-only.

This matrix covers those gaps via direct code injection (lower setup cost
than reverting real vllm-ascend bugfixes).

## Two paths — when to use which

| Path | Method | Fidelity | Setup cost | Use for |
|------|--------|----------|-----------|---------|
| **B (direct inject)** | env-gated debug hook writes NaN logits / loop tokens / etc. | Medium (synthetic but exercises the real detector code path) | 1-2 hours | All scenarios below |
| **A (revert real bugfix)** | `git revert <sha>` of a related bugfix + shrink `--block-size` if needed | High (real bug replay) | 0.5-1 day (rebuild if csrc) | Cross-check B when applicable |

Path A candidates (DSV2-Lite compatible) from
`task_spec/kv_cross_request_contamination_survey_20260903.md`:

- vllm main #18957 ComputedBlocksTracker outdated (refcount; pure Python, no rebuild)
- vllm-ascend #5030 KV Pool TP rank mismatch (C++; needs .so rebuild)
- vllm main #51482 LIFO free_blocks reuse order (pure Python)

Path A is gated on path B passing first — A is the cross-check, not the
primary evidence.

## Path B injection matrix (5 scenarios + 2 cross-cutting)

Double-gated: injection is dead in shipped builds. Arm by flipping
`_INJECT_MASTER_SWITCH = True` in `inject.py` (source edit, verification
builds only), then set `RG_INJECT`. Either gate absent → 0 overhead in prod.

```
RG_INJECT=scenario_name[:step_trigger][:param]
```

Injection entry points (as-built, one guarded call each in `processor.py`;
`step` counts engine waves and defaults to 5 — advanced on pre-sample when the
wrap runs, otherwise lazily on the first after-spec / after-sample inject of
that step so post-sample scenarios do not require `logits_finite`):

| Hook | Scenarios |
|------|-----------|
| `check_before_sample` | #1 `nan_logits`, #2 `inf_logits` (corrupt `logits[0, col]` pre-detect, one-shot) |
| `check_after_spec` | #5 `spec_all_reject` (zero `accepted_token_nums` pre-detect, one-shot) |
| `check_after_sample` | #3 `forbidden_substring` (row-0 token cycles pattern ids), #4 `token_loop` (anchor = the trigger wave's own row-0 last sampled token; pinned for `param` waves, default 40) |

### Detector coverage

> **Status (as-built):** `vllm_ascend/runtime_guard/inject.py` implements
> scenarios #1–#5 (`nan_logits` / `inf_logits` / `forbidden_substring` /
> `token_loop` / `spec_all_reject`) with synthetic UT
> (`vllm_ascend/runtime_guard/test/test_inject_scenarios.py`, no NPU). Live runs
> (§Live run procedure) are pending an NPU window; the live runner is
> workspace-side (`<rg_test>/run_inject.sh`, machine-specific server startup
> stays out of the repo).

| # | Scenario | Injection point | What gets corrupted | Detector | Expected report field |
|---|----------|------------------|----------------------|----------|----------------------|
| 1 | `nan_logits` | runner post-logits | `logits[0,5]=NaN` | logits_finite | `kind=nan` + row |
| 2 | `inf_logits` | runner post-logits | `logits[0,3]=Inf` | logits_finite | `kind=inf` + row |
| 3 | `forbidden_substring` | post-sampler | replace sampled_tokens with `李白` after first decode step | output_substring | `pattern=李白` |
| 4 | `token_loop` | post-sampler | repeat last sampled token 32 times | token_repeat | `repeat_sum` + `window` |
| 5 | `spec_all_reject` | spec_acceptance pre-call | `accepted_token_nums=[0]*bs` | spec_acceptance | `rate≈0` + `window=10` |

### Cross-cutting scenarios (verify max_per_req stop-detect)

| # | Scenario | Operation | Verification point |
|---|----------|-----------|--------------------|
| 11 | `max_per_req_stop` | run scenario #1 once (`max_per_req=1`), continue stepping | that req_id no longer enters detector after the report is written |
| 12 | `max_per_req_multi` | set `report.max_per_req=3`, run #1 across steps | up to 3 reports, then skip |

### Coverage summary

| Detector | Scenario | Status |
|----------|----------|--------|
| logits_finite | #1, #2 | pending |
| output_substring | #3 | pending |
| token_repeat | #4 | pending |
| spec_acceptance | #5 | pending ⚠️ DSV2-Lite not running MTP currently; either enable `--num-speculative-tokens` + Eagle speculator or fall back to UT-only coverage |

**4/4 shipped detectors covered by injection.**

## Injection mechanism design

1. **Zero prod path**: `RG_INJECT` env unset → `inject.py` `inject_for_step()` returns immediately. No overhead.
2. **Reentrant**: each scenario is an independent function with isolated state.
3. **Observable**: injection triggers print `[INJECT] scenario=X step=N` so it can be cross-referenced with detector hit log lines.
4. **One-shot vs armed**: `nan_logits` / `inf_logits` / `spec_all_reject` fire
   once; `forbidden_substring` / `token_loop` stay armed from `step` (they need
   multiple waves to cross the detector window), with `token_loop` bounded by
   its `param` waves. Re-arm = restart the process.

## Implementation (as-built)

```
vllm_ascend/runtime_guard/
└── inject.py                          # env parser + all 5 scenario hooks (no NPU deps)

vllm_ascend/runtime_guard/processor.py # 3 guarded call sites:
                                       #   check_before_sample / check_after_spec / check_after_sample
                                       #   (`if inject.ENABLED:` → zero overhead when unarmed / env unset)

vllm_ascend/runtime_guard/test/
├── test_inject_scenarios.py           # synthetic UT per scenario (no NPU)
└── ANOMALY_INJECTION.md               # this file

<rg_test workspace>                    # machine-specific launcher (NOT in repo)
└── run_inject.sh                      # per-scenario server restart (RG_INJECT is
                                       # read at process start) + report/log checks
```

## Live run procedure

For each scenario #1-#5, #11-#12:
1. Confirm guard server up on cards 6-7 (DeepSeek-V2-Lite, TP=2, port 8017)
2. Reset runtime_config.json: `report.max_per_req=1`, all 4 shipped detectors `enabled=true`
   (`spec_acceptance`, `output_substring`, `token_repeat`, `logits_finite`)
3. Clear stale `dump.manual_dump` in runtime_config.json if needed (no HTTP manual_trigger endpoint)
4. Flip `_INJECT_MASTER_SWITCH = True` in `vllm_ascend/runtime_guard/inject.py`, then
   `RG_INJECT=scenario_name:step_trigger[:param] python vllm_ascend/runtime_guard/test/perf/run_inject.py`
5. Wait for injection log `[INJECT] scenario=X step=N` to appear
6. Check `runtime_report_dir/<incident_type>/` for matching detector report
7. Cross-reference detector log line with `[INJECT]` line — both must appear in the same step window

## Pass criteria

- #1-#5: detector report file exists with expected `kind` / `pattern` / field
- #11: subsequent step's detector skip set includes the req_id (`stopped_req_ids`)
- #12: up to `max_per_req` reports for the same req_id, then skip (proves write-full stop)
