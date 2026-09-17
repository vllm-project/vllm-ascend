# runtime_guard system observability test plan

> Goal: 3 test axes — **perf+memory combined / log-level correctness /
> module-by-module log printing**. Each section is self-contained with shell
> commands so `run_system_observability.sh` can pick it up.
>
> UT expect ≈ **86 passed** on this branch.

## 0. Pre-flight (every section depends on this)

```bash
docker exec test-mrv2 bash -lc '
  cd /workspace/vllm-ascend &&
  git fetch origin &&
  test "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/runtime-guard-config)" \
    || { echo "HEAD != origin; run: git pull --ff-only"; exit 1; }
  python -m pytest vllm_ascend/runtime_guard/test/ tests/ut/runtime_config/ -q 2>&1 | tail -3
'
# Expect: 86 passed (runtime_guard 80 + runtime_config 6)
```

```bash
# NPU availability — need cards 6-7 free (HBM per card <5GB OR owned by this task)
npu-smi info | awk '/910B4|0000:/{print}'
```

## 1. Perf + memory combined test — T0 vs T1 vs T2 vs T3

### 1.1 What it proves

Per-terminal-state throughput + leak-back memory signature in one run.
Phase B (measured rounds) catches monotonic per-round growth (per-req
state containers); Phase C (5-min idle post-load) catches leak-back failure
(state not released after requests stop).

| State | Code | `--additional-config` | reload | detectors | server swap |
|---|---|---|---|---|---|
| **T0** | merge-base `37e382498` worktree | none | — | — | stop server → worktree vllm serve |
| **T1** | current HEAD | none (plain vllm serve) | 0 (default) | all off | stop T0 → restart from HEAD |
| **T2** | current HEAD | runtime_config_path + reload_interval=3 | 3 | all off | hot-toggle via set_detectors(False) |
| **T3** | current HEAD | same as T2 | 3 | 4 enabled | hot-toggle via set_detectors(True) |

### 1.2 Leak candidates Phase B must catch

| Container | File | Why suspect |
|---|---|---|
| `WaveTracker._sample_waves` | `wave_tracker.py` | Cleared only via `discard_many` in `_reap_finished_requests`; skipped if `wave_tracker is None` or exception. |
| `SpecAcceptanceDetector._history` | `detector/spec_acceptance.py` | `defaultdict(deque)` auto-creates on read; stray req_id never reaped. |
| `TokenRepeatDetector._states` / `._consumed_len` | `detector/token_repeat.py` | No cap; relies on reap. |
| `RequestGuardStore._by_req` | `request_state.py` | Unbounded (only `_reaped_ring` capped at 1024). |
| `ActionExecutor` queue | `action/executor.py` | `Queue(maxsize=64)` — should self-bound; verify drain works. |

### 1.3 Procedure — `perf_baseline.py` (T0 or T1) + `perf_ab_quick.py` (T2/T3)

The 3 scripts now sample memory per-round + run a 5-min leak-back phase.
Output files:

| Script | Output (perf) | Output (leakback) |
|---|---|---|
| `perf_baseline.py` (T0 worktree) | `logs/perf_t0.jsonl` | `logs/leakback_t0.jsonl` |
| `perf_baseline.py` (T1 HEAD) | `logs/perf_baseline.jsonl` | `logs/leakback_baseline.jsonl` |
| `perf_ab_quick.py` (T2 A + T3 B) | `logs/perf_ab_quick.jsonl` | `logs/leakback_ab.jsonl` |

```bash
# Section 1a: T0 baseline (worktree at merge-base)
docker exec test-mrv2 bash -lc '
  cd /workspace
  git -C vllm-ascend worktree add /tmp/rg-perf-base 37e382498
  # Stop current guard server on 6-7
  pgrep -f "[v]llm serve" | xargs -r kill -9
  npu-smi info | grep VLLMWorker | awk "{print \$3}" | xargs -r kill -9
  sleep 5

  cd /tmp/rg-perf-base
  ASCEND_RT_VISIBLE_DEVICES=6,7 VLLM_USE_V1=1 \
  nohup python -m vllm.entrypoints.openai.api_server \
    --model /data0/weights/DeepSeek-V2-Lite --served-model-name dsv2 \
    --pipeline-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
    --seed 1024 --trust-remote-code --gpu-memory-utilization 0.85 --enforce-eager \
    --port 8017 \
    > /tmp/rg-perf-t0.log 2>&1 &
  until curl -sf http://127.0.0.1:8017/health >/dev/null; do sleep 5; done

  cd /workspace/vllm-ascend/vllm_ascend/runtime_guard/test/perf
  RG_PERF_OUT_BASELINE=/workspace/djh-testruntime/rg_test/logs/perf_t0.jsonl \
  RG_PERF_OUT_LEAKBACK_BASELINE=/workspace/djh-testruntime/rg_test/logs/leakback_t0.jsonl \
  python perf_baseline.py
'

# Section 1b: T1 baseline (current HEAD, plain vllm serve)
docker exec test-mrv2 bash -lc '
  pgrep -f "[v]llm serve" | xargs -r kill -9
  npu-smi info | grep VLLMWorker | awk "{print \$3}" | xargs -r kill -9
  sleep 5
  git -C /workspace/vllm-ascend worktree remove /tmp/rg-perf-base --force

  cd /workspace/vllm-ascend
  ASCEND_RT_VISIBLE_DEVICES=6,7 VLLM_USE_V1=1 \
  nohup python -m vllm.entrypoints.openai.api_server \
    --model /data0/weights/DeepSeek-V2-Lite --served-model-name dsv2 \
    --pipeline-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
    --seed 1024 --trust-remote-code --gpu-memory-utilization 0.85 --enforce-eager \
    --port 8017 \
    > /tmp/rg-perf-t1.log 2>&1 &
  until curl -sf http://127.0.0.1:8017/health >/dev/null; do sleep 5; done

  cd vllm_ascend/runtime_guard/test/perf
  # perf_baseline.jsonl already has T1 pre-110ff191e data — back it up
  mv /workspace/djh-testruntime/rg_test/logs/perf_baseline.jsonl \
     /workspace/djh-testruntime/rg_test/logs/perf_baseline_pre110ff19.jsonl 2>/dev/null
  python perf_baseline.py
'

# Section 1c: T2+T3 (current T1 server + runtime_config + reload=3)
docker exec test-mrv2 bash -lc '
  # Stop T1, restart as T2 (server with reload=3)
  pgrep -f "[v]llm serve" | xargs -r kill -9
  npu-smi info | grep VLLMWorker | awk "{print \$3}" | xargs -r kill -9
  sleep 5
  cd /workspace/vllm-ascend
  ASCEND_RT_VISIBLE_DEVICES=6,7 VLLM_USE_V1=1 \
  nohup python -m vllm.entrypoints.openai.api_server \
    --model /data0/weights/DeepSeek-V2-Lite --served-model-name dsv2 \
    --pipeline-parallel-size 1 --tensor-parallel-size 2 --enable-expert-parallel \
    --seed 1024 --trust-remote-code --gpu-memory-utilization 0.85 --enforce-eager \
    --port 8017 \
    --additional-config "{\"runtime_config_path\":\"/workspace/djh-testruntime/rg_test/config/runtime_config.json\",\"runtime_config_reload_interval\":3,\"runtime_report_dir\":\"/workspace/djh-testruntime/rg_test/report\"}" \
    > /tmp/rg-perf-t23.log 2>&1 &
  until curl -sf http://127.0.0.1:8017/health >/dev/null; do sleep 5; done

  cd vllm_ascend/runtime_guard/test/perf
  python perf_ab_quick.py
'
```

### 1.4 Pass criteria

| Metric | Bar | How |
|---|---|---|
| **C1** (T0 vs T1) gap | ≤0.1% per tag (short/medium/long) | `perf_t0.jsonl` avg tps vs `perf_baseline.jsonl` avg tps |
| **C2** (T1 vs T2) gap | ≤0.1% per tag (post-`110ff191e` idle short-circuit should drop from 0.4% → ~0.1%) | T2 = perf_ab_quick A rounds avg tps vs T1 |
| **C3** (T2 vs T3) gap | ≤1% per tag | T3 = perf_ab_quick B rounds avg tps vs T2 A |
| **Phase B monotonic RSS growth** | per-round delta ≤20 MB | Compare `post_rss_kb - pre_rss_kb` across rounds within same state |
| **Phase C leak-back** | end-of-leakback `rss_delta_kb` ≤ 30 MB (30720) | `leakback_*.jsonl` last line `rss_delta_kb` field |
| **HBM stability** | end-of-leakback HBM ≤ Phase A end + 100 MB | Compare leakback end `hbm_mb` to first leakback sample |

Compute script:

```bash
docker exec test-mrv2 bash -lc '
  python3 << "PY"
import json
from pathlib import Path
LOGS = Path("/workspace/djh-testruntime/rg_test/logs")
def tps_avg(state_or_file, tag):
    if state_or_file == "t0":
        rs = [json.loads(l) for l in open(LOGS/"perf_t0.jsonl")]
    elif state_or_file == "t1":
        rs = [json.loads(l) for l in open(LOGS/"perf_baseline.jsonl")]
    elif state_or_file == "ab":
        rs = [json.loads(l) for l in open(LOGS/"perf_ab_quick.jsonl")]
    rs = [r for r in rs if r["tag"]==tag]
    return sum(r["out_tok_s"] for r in rs)/len(rs)
def leak_delta(name):
    p = LOGS/f"leakback_{name}.jsonl"
    if not p.exists(): return None
    last = [json.loads(l) for l in open(p) if "leakback_end" in l]
    return last[-1]["rss_delta_kb"] if last else None
print("=== Perf gaps ===")
for tag in ("short","medium","long"):
    try:
        t0, t1 = tps_avg("t0",tag), tps_avg("t1",tag)
        print(f"C1 {tag}: T0={t0:.2f} T1={t1:.2f} gap={(t0-t1)/t0*100:+.2f}%  {\"PASS\" if abs((t0-t1)/t0*100) <= 0.1 else \"FAIL\"}")
    except Exception as e: print(f"C1 {tag}: skip ({e})")
    ab = [json.loads(l) for l in open(LOGS/"perf_ab_quick.jsonl") if "tag" not in l or True]
    a = [r for r in ab if r["state"]=="A" and r["tag"]==tag]
    b = [r for r in ab if r["state"]=="B" and r["tag"]==tag]
    if a and b:
        ta = sum(r["out_tok_s"] for r in a)/len(a)
        tb = sum(r["out_tok_s"] for r in b)/len(b)
        print(f"C3 {tag}: T2={ta:.2f} T3={tb:.2f} gap={(ta-tb)/ta*100:+.2f}%  {\"PASS\" if abs((ta-tb)/ta*100) <= 1 else \"FAIL\"}")
print("=== Memory leak-back ===")
for n in ("t0","baseline","ab"):
    d = leak_delta(n)
    print(f"leakback_{n}: rss_delta_kb={d}  {\"PASS\" if d is None or abs(d) <= 30720 else \"FAIL\"}")
PY
'
```

### 1.5 Known pitfalls

- **First round faster** (cold cache): handled by `warmup(rounds=1)` in both scripts.
- **`pkill -f 'vllm serve'` matches own shell**: use `pgrep -f '[v]llm serve'` bracket trick + PID kill from `npu-smi info | grep VLLMWorker`.
- **TP workers survive parent kill**: explicitly kill worker PIDs from npu-smi.
- **Cross-session noise ±1%**: for tight comparisons (C1, C2), run T0 + T1 in same session back-to-back (procedure above does this).
- **`run_rg.sh baseline` ≠ T0**: that script still points PYTHONPATH at current branch — it's T1, not T0. T0 requires worktree checkout at `37e382498`.
- **HBM stays elevated after kill**: driver residual ~2.9 GB; this is normal, not a leak.

## 2. Debug log additions — request lifecycle + per-module snapshot

### 2.1 Per-module debug switch (use existing infra)

`apply_ascend_log_level(level, debug_modules, module_levels)` already supports
per-module DEBUG. The `runtime_config.json` `ascend_log` section hot-reloads
this. No new code needed — just exercise it (see §3).

### 2.2 New debug logs to add (code change — pending, tracked as task #17)

Each must be DEBUG level (default-off), gated by `logger.isEnabledFor(DEBUG)`
so prod pays nothing.

| # | Module:File | Log site | Log content | When |
|---|---|---|---|---|
| L1 | `request_state.py:RequestGuardStore.get_or_create` | after `_by_req[req_id] = state` | `[RG_STATE create] req_id=%s size=%d reapable=%d` | first sight |
| L2 | `request_state.py:RequestGuardStore.clear` | before pop | `[RG_STATE destroy] req_id=%s lifetime_ms=%d outputs=%d alerted=%s` | reap |
| L3 | `request_state.py:RequestGuardStore.clear_many` | after batch | `[RG_STATE reap_batch] count=%d remaining=%d reaped_ring=%d` | every reap sweep |
| L4 | `wave_tracker.py:WaveTracker.record_sample_waves` | after stamp | `[RG_WAVE record] req_id=%s wave=%d pending=%d` | hook 7 |
| L5 | `wave_tracker.py:WaveTracker.discard_many` | after batch | `[RG_WAVE discard] count=%d remaining=%d` | reap |
| L6 | `kv_block_meta.py:KvBlockMetaTracker.record_writes` | after loop | `[RG_KV write] req_id=%s blocks=%d wave=%d total_blocks_tracked=%d distinct_writers=%d` | hook (note_kv_block_writes) |
| L7 | `processor.py:_reap_finished_requests` | start + end | start: `[RG_REAP enter] live=%d reapable=%d`; end: `[RG_REAP leave] reaped=%d live=%d` | every step (DEBUG) |
| L8 | `processor.py:sync_for_step` | DEBUG-only snapshot every N=64 steps | `[RG_SYNC snapshot] step=%d store_size=%d wave_pending=%d kv_blocks=%d io_cache=%d action_q=%d` | periodic |
| L9 | `detector/spec_acceptance.py:SpecAcceptanceDetector._history` defaultdict access | when auto-create | `[RG_SPEC hist-create] req_id=%s history_size=%d` | read-on-missing |
| L11 | `detector/manager.py:DetectorManager.clear_finished` | after per-detector loop | `[RG_DET clear] req_id=%s cleared=%d` | reap |
| L12 | `action/executor.py:ActionExecutor.submit_heavy` | on enqueue + on drop | `[RG_ACT enqueue] heavy=%s qsize=%d` / `[RG_ACT drop] heavy=%s reason=full` | submit |

### 2.3 Per-request snapshot helper (new module — pending, tracked as task #18)

`vllm_ascend/runtime_guard/debug_snapshot.py:snapshot_summary()` aggregates
container sizes. Wire into `manual_trigger.py` as new trigger type
`dump_state` — live-triggerable via HTTP, no restart.

### 2.4 Comparisons this enables

| Comparison | How | What it surfaces |
|---|---|---|
| `store_size` vs sum of all per-req detector dict sizes | After each reap: `len(store) == len(spec_history) == len(token_repeat_states)` | Any detector dict out of sync → leak in that detector |
| `kv_blocks_tracked` vs `kv_distinct_writers` | Periodic snapshot | Block meta growth without new requests → block-id-level leak |
| `wave_pending` vs `store_size` | After every reap | Wave not discarded for reaped req → wave_tracker leak |
| `action_queue_qsize` | Periodic | Heavy action backlog growing → ActionExecutor drain stuck |
| `reaped_ring_size` | Periodic | Should converge to min(live, 1024) |

## 3. Multi-module log-level + log-printing test matrix

### 3.0 UCM 绕过前置检查（2026-09-12 新增）

Ascend 容器里 UCM 会**无条件**劫持 `vllm.logger.init_logger`（见
`docs/zh/design/runtime_guard_design.md` 2.5）。`init_logger_ascend` 已改为直接
`logging.getLogger` + `_METHODS_TO_PATCH` 绕过；否则本节的 **M5–M8（Layer B）会静默失效**
（级别锁死 INFO、不进 stdlib 树，`ascend_log` 调级对 runtime_guard 模块不生效）。
跑矩阵前先确认 logger 类型：

```bash
python -c "import vllm_ascend.runtime_guard.processor as m; \
print(type(m.logger).__module__ + '.' + type(m.logger).__name__)"
# Expect: logging.Logger（不是 ucm.logger.Logger）
```

### 3.1 Three repos / module layers

| Layer | Module prefix | Sample loggers | Source repo |
|---|---|---|---|
| **A. vllm core** | `vllm.*` | `vllm.engine_llm`, `vllm.transformers_utils.tokenizer`, `vllm.v1.worker.gpu_worker` | vllm main |
| **B. runtime_guard** | `vllm_ascend.runtime_guard.*`, `vllm_ascend.runtime_config.*` | `vllm_ascend.runtime_guard.processor.RuntimeGuardProcessor`, `vllm_ascend.runtime_config.config.RuntimeConfig` | vllm-ascend (this branch) |
| **C. ascend ops/worker** | `vllm_ascend.worker.*`, `vllm_ascend.ops.*` | `vllm_ascend.worker.model_runner_v1`, `vllm_ascend.ops.npu_fusion_attention` | vllm-ascend (csrc + Python wrapper) |

### 3.2 Test matrix (12 cases = 3 layers × 4 level toggles)

| # | Layer | Toggle | Expected | How to verify |
|---|---|---|---|---|
| M1 | A | `level=INFO` only | INFO+ lines appear, DEBUG lines don't | `grep -c DEBUG log` should be 0; `grep -c INFO log` > 0 |
| M2 | A | `level=DEBUG` | DEBUG lines for vllm.engine_llm appear | `grep "vllm.engine_llm" log \| grep -c DEBUG` > 0 |
| M3 | A | `modules=["vllm.transformers_utils.tokenizer"]` only (level=INFO) | Just tokenizer goes DEBUG; rest INFO | `grep "tokenizer.*DEBUG" log` > 0; `grep "engine_llm.*DEBUG" log` = 0 |
| M4 | A | bad module name `vllm.does_not_exist` | Config rejected, keeps old config | `curl runtime_config` returns old; rejected log line in server.log |
| M5 | B | `level=INFO`, no modules | `[runtime_guard sync]` DEBUG lines suppressed | `grep -c "runtime_guard sync" log` = 0 |
| M6 | B | `modules=["vllm_ascend.runtime_guard.processor"]` | `[runtime_guard sync] enter/leave` lines appear | `grep "runtime_guard sync" log \| grep -c enter` > 0 |
| M7 | B | `modules=["vllm_ascend.runtime_config.config"]` | `[runtime_config] hot-reload enabled` lines appear (only log_once) | `grep "runtime_config" log \| grep -c "hot-reload"` ≥ 1 |
| M8 | B | hot toggle: INFO → DEBUG → INFO | Levels change live without restart | Count DEBUG lines before/after each toggle in 60s windows |
| M9 | C | `modules=["vllm_ascend.worker.model_runner_v1"]` | runner-level DEBUG lines appear | `grep "model_runner_v1" log \| grep -c DEBUG` > 0 |
| M10 | C | `modules=["vllm_ascend.ops.npu_fusion_attention"]` | Op-level DEBUG lines (if logger exists) | `grep "fusion_attention" log` may be empty if op uses print(); document that gap |
| M11 | C | bad ascend_log level value `INFO_PLUS` (typo) | Rejected, falls back to INFO; announce line shows old level | `grep "applied level=INFO" log` exists, no INFO_PLUS |
| M12 | A+B+C | reset to all INFO (clear modules) | All DEBUG lines stop within 60s | `grep -c DEBUG log` (over 60s post-reset) = 0 |

### 3.3 Procedure template (per case)

```bash
docker exec test-mrv2 bash -lc '
  CFG=/workspace/djh-testruntime/rg_test/config/runtime_config.json
  python3 << PY
import json
cfg = json.load(open("$CFG"))
cfg["ascend_log"] = {"level": "INFO", "modules": ["vllm_ascend.runtime_guard.processor"]}
json.dump(cfg, open("$CFG", "w"), indent=2)
PY
  # Force a step so config picks up
  curl -s http://127.0.0.1:8017/v1/completions \
    -d "{\"model\":\"dsv2\",\"prompt\":\"warmup\",\"max_tokens\":2,\"temperature\":0}"
  sleep 5
  tail -2000 /workspace/djh-testruntime/rg_test/logs/server.log | grep -c "runtime_guard sync"
'
```

Repeat per matrix cell. The runner `run_log_matrix.sh` loops over M1-M12
with this template, parameterized by an env file.

### 3.4 Known pitfalls (from past sessions)

- **`logging.disable(NOTSET)` stuck**: `apply_ascend_log_level` clears it at entry (logger.py:99). Verify with M8 repeated toggle — if 2nd toggle fails to re-apply, the stuck-disable regression came back.
- **`vllm.handlers.level=INFO` gates `vllm.*` DEBUG**: prior fix added `setLevel(DEBUG)` on `vllm.logger`. M2 must verify `vllm.engine_llm` DEBUG actually prints, not just `vllm_ascend.*` DEBUG.
- **Without runtime_guard, branch uses `vllm` root logger**: handlers' level gate still applies. M5 + M8 together cover this.
- **Hot-reload takes 1 step to pick up**: always send warmup request after config edit, then wait ≥ `reload_interval_seconds` (default 3) + 1.
- **UCM 劫持（已修）**：`init_logger_ascend` 若委托给被 patch 的 `vllm.logger.init_logger`，会返回 UCM logger（级别锁死 INFO、不进 stdlib 树），导致 M5–M8 与 `ascend_log` 分模块调级对 Layer B 全部静默失效。已改为 `logging.getLogger` + `_METHODS_TO_PATCH`（挂 once 方法）绕过。回归时先跑 3.0 前置检查确认 logger 类型是 `logging.Logger`。

## 4. Auto-runner script

`vllm_ascend/runtime_guard/test/perf/run_system_observability.sh` (task #22) wraps
sections 1 + 3 in sequence:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run_system_observability.sh [section]
#   section ∈ {perf, log, all} (default: all)
SECTION="${1:-all}"
[[ "$SECTION" =~ ^(perf|log|all)$ ]] || { echo "bad section"; exit 1; }

case "$SECTION" in
  perf|all) bash vllm_ascend/runtime_guard/test/perf/run_perf_mem.sh ;;
esac
case "$SECTION" in
  log|all) bash vllm_ascend/runtime_guard/test/perf/run_log_matrix.sh ;;
esac
```

## 5. Status tracking

| Section | Status (2026-09-06) |
|---|---|
| 1 (perf + memory combined) | ⚠️ scripts extended (perf_lib + perf_baseline + perf_ab_quick), procedure ready, NOT run yet — needs cards 6-7 swap |
| 2 (debug log additions) | ❌ design only, L1-L12 not implemented (task #17) + debug_snapshot not created (task #18) |
| 3 (log-level matrix) | ❌ not run — needs server up (task #20) |
| 4 (auto-runner) | ❌ not created (task #22) |

## 6. Order of operations

1. **Commit perf_lib.py + perf_baseline.py + perf_ab_quick.py + this doc** (this session)
2. Wait for cards 6-7 free (or stop current guard server)
3. Run section 1 (perf C1+C2+C3 + memory leak-back) — ~1h live NPU (task #21 + #19 merged)
4. Run section 3 (log matrix M1-M12) — 1h live NPU (task #20)
5. Implement L1-L12 + debug_snapshot (task #17, #18) — ~2h code work, no NPU needed
6. Create `run_system_observability.sh` (task #22) — procedure fragments from §1.3 + §3.3
7. Update CURRENT_STATE.md perf + memory sections with results
