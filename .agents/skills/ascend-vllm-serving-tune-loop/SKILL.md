# ascend-vllm-serving-tune-loop

Autonomous, evidence-driven performance optimization loop for vLLM-Ascend serving on Huawei Ascend 910C NPU.

Runs a fixed baseline benchmark, then iterates through a layered tuning plan — from scheduler-level knobs down to ACLGraph compilation and kernel-level parameters — exhaustively trying all candidate levers across all phases. Produces a full optimization ledger and final comparison report.

## When to invoke

Use this skill when the user wants to:
- Squeeze lower latency / higher throughput from an existing vllm-ascend deployment on 910C
- Systematically explore Ascend-specific tuning knobs without guessing
- Get a documented, reproducible record of what was tried and what helped

Trigger phrases: "optimize", "tune", "调优", "性能优化", "latency optimization", "ascend tuning loop", "sota loop"

## Prerequisites

Same as `ascend-vllm-serving-auto-benchmark` plus:
- `msprof` available (optional, for human deep-dive profiling if Phase 7 identifies non-tunable bottlenecks)
- Enough NPU memory headroom to try `block_size` variants
- Write permission to working directory for ledger files

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_path` | Yes | Local path or model ID |
| `model_name` | Yes | Short name used in commands and ledger |
| `tensor_parallel_size` | No | NPU card count (default: 1) |
| `max_model_len` | No | Max context length (default: 8192) |
| `dtype` | No | `float16` / `bfloat16` (default: `float16`) |
| `port` | No | Server port (default: 5000) |
| `baseline_launch_command` | No | Full server launch command used as the Phase 1 baseline. If provided, the skill uses this command exactly as the starting point instead of the default template, preserving any pre-applied optimizations across the whole campaign unless a later iteration is explicitly testing that field. |
| `target_metric` | No | Primary metric to optimize: `ttft_avg` \| `tpot_avg` \| `latency_avg` (default: `ttft_avg`) |
| `low_conc_levels` | No | Concurrency levels considered "low" (default: `1 4 8`) |
| `requests_per_level` | No | Requests per concurrency level for each benchmark run (default: `20 80 160`) |
| `input_tokens` | No | Input token length (default: 1024) |
| `output_tokens` | No | Output token length (default: 512) |
| `improvement_threshold_pct` | No | Minimum improvement to count as a win (default: 1.0%) |
| `max_iterations` | No | Hard stop after N tuning iterations (default: 50) |
| `work_dir` | No | Directory for all artifacts (default: `./tuning_run/<timestamp>`) |

## Fixed benchmark contract

**These values are locked for the entire campaign and must never change between iterations:**

| What | Value |
|------|-------|
| Concurrency levels | `c = 1, 4, 8` (all three, every run) |
| Primary decision metric | `ttft_avg` at **c = 1** |
| Secondary guard metric | `tpot_avg` at **c = 1** — if it regresses > 5%, downgrade WIN to NEUTRAL |
| Full recorded metrics | `ttft_avg`, `tpot_avg`, `output_token_throughput` (TPS), `latency_avg` at each of c=1/4/8 |
| Input tokens | fixed at `input_tokens` parameter |
| Output tokens | fixed at `output_tokens` parameter |
| Requests per level | fixed at `requests_per_level` parameter |
| Baseline launch config | If `baseline_launch_command` is provided, use it exactly in Phase 1; otherwise use the default server template below |
| Warmup requests | 20 at c=1 before every measurement run |

**Why `ttft_avg` and not P99**: at low concurrency (20 requests per level), P99 represents a single outlier request and is too noisy to be a reliable signal. `ttft_avg` is statistically stable and directly reflects prefill path performance — the dominant factor at c=1.

**TPS** (`output_token_throughput`) is recorded at all concurrency levels for informational purposes. At c=1 it correlates closely with `tpot_avg`, but becomes more informative at c=4/8 where batching effects kick in.

The user may override the target metric via input parameters, but once Phase 1 starts the chosen metric is frozen.

## Core principle: RLCR (Refinement Loop with Continuous Revalidation)

Every tuning iteration must:
1. **Hypothesize** — pick one lever from the ranked candidate list with a stated expected impact
2. **Change one thing** — apply a single parameter change; never bundle multiple levers in one iteration
3. **Revalidate** — run evalscope at **all three concurrency levels (c=1/4/8)** with the fixed workload
4. **Decide on `ttft_avg` at c=1** — this single number is the gate for WIN/LOSS/NEUTRAL
5. **Record** — append to the ledger regardless of win/loss; include all c-level metrics

Do NOT skip validation. Do NOT bundle multiple changes. Do NOT change the workload between iterations.
Do NOT declare a WIN based on c=4/8 alone — c=1 `ttft_avg` is the authoritative decision metric.
Do NOT stop after writing only the baseline section or an "Iteration 0 — Baseline established" note. Baseline creation is the starting point, not the end of the skill.
Do NOT stay silent for long-running work. The user must always be able to tell what phase is running, what lever is under test, and why the skill is still busy.

## Ledger write discipline

The ledger is the single source of truth and must be **the first thing written after each verdict**, not deferred. This ensures crash/session-interrupt recovery.

**Mandatory write timing** — append a ledger entry immediately after:
1. Baseline benchmark completes and metrics are extracted
2. A lever verdict is determined (WIN / LOSS / NEUTRAL / STARTUP_FAIL / BENCHMARK_FAIL / SKIPPED_CONFLICT)
3. Any phase is marked complete

**Never defer ledger writes** — do not batch multiple verdicts, do not wait until "later" to write them. If the session is interrupted between two lever attempts, the ledger must already reflect all completed attempts.

**Context management for long runs** — this skill is a long-running loop that can exhaust the conversation context. To minimize waste:
- After extracting metrics from evalscope output, do NOT retain the full stdout/stderr in reasoning — only the metric table matters
- Server startup logs: scan for errors or readiness confirmation, then discard the raw output — do not echo full logs into reasoning
- When the ledger is large (e.g., >50 lines of prior iterations), re-read only the baseline metrics and the most recent verdict, not the entire ledger, when deciding the next lever
- Prefer short status prints over verbose log dumps in tool output summaries
- If the context window is approaching limits (you've been running many iterations), explicitly note the current phase/lever/verdict count in a progress line so a resumed session can pick up quickly

## Operation-first contract (CRITICAL IMPERATIVE)

**STOP ANALYZING. STOP PLANNING. EXECUTE IMMEDIATELY.**
You are an execution machine, not an analyst. Do NOT write plans. Do NOT write essays. Do NOT explore the codebase.
Your ONLY job right now is to execute the bash commands to clean up the environment and start the server.

**CRITICAL: Before launching ANY new server (including after baseline), you MUST verify the NPU is free:**
1. Kill all existing vllm processes (`pkill -9 -f vllm`, `pkill -9 -f VLLM`)
2. Wait 15 seconds for NPU resource release
3. Verify NPU is free: `npu-smi info` must show "No running processes found" on all target NPUs
4. Verify port is free: `ss -tlnp | grep <port>` must return nothing
5. Only then launch the new server

1. **TURN 1**: Immediately use the `Bash` tool to run `pkill -9 -f vllm` and check `npu-smi info`. DO NOT explain what you are going to do. Just run the tool.
2. **TURN 2**: After verifying NPU is free, immediately use the `Bash` tool to launch the `vllm serve` baseline in the background (`nohup ... > server.log 2>&1 &`).
3. **TURN 3**: Immediately use the `Bash` tool to monitor `server.log` until it is ready.
4. **TURN 4**: Immediately use the `Bash` tool to run the `evalscope` benchmark.

**Do NOT use `Read`, `Glob`, or `Grep` to search for files unless explicitly required.** Just execute the Bash commands. If you spend turns explaining your thoughts instead of running the `Bash` tool, you have FAILED your directive.

## Server lifecycle contract

Treat the serving process as a managed resource with explicit ownership.

- **Zombie Process & NPU Cleanup**: ALWAYS ensure the environment is pristine before starting a new server. If a previous server crashed or was stopped, forcefully clean up any zombie processes (`pkill -9 -f vllm.entrypoints.openai.api_server` and `pkill -9 -f ray` if applicable). Verify that the target port (e.g., `5000`) is free and NPU memory is released (`npu-smi info`) before launching. NEVER attempt to start a server if the NPU is still occupied by a ghost process.
- **Background Execution**: Always launch the server in the background and redirect its output to a file (e.g., `nohup <command> > <work_dir>/server.log 2>&1 &`) so your Bash tool does not block.
- **Log Monitoring**: After launching, use `grep` or `tail` on the log file to monitor the startup progress and verify readiness (look for "Uvicorn running on" or similar success messages).
- Before Phase 1 or any lever attempt, check whether the target server is already running on the intended port and whether it matches the intended config closely enough to reuse
- If the running server does not match the intended config, forcefully stop and clean it up before starting the next attempt
- Never leave multiple competing server processes running for the same campaign port
- Every iteration must end in exactly one of these states:
  - benchmarked and recorded
  - failed to start and recorded
  - skipped by precheck and recorded
- Do not keep an old server alive just to save time if it makes the config attribution ambiguous

### Post-iteration strict cleanup protocol (MANDATORY)

**Note: Baseline benchmark completion counts as an iteration verdict. The cleanup protocol MUST be executed after baseline before starting Phase 2.**

After EVERY iteration verdict (WIN/LOSS/NEUTRAL/STARTUP_FAIL/BENCHMARK_FAIL/SKIPPED_CONFLICT), the following cleanup sequence MUST be executed before starting the next lever. This is non-negotiable. Skipping cleanup leads to zombie process accumulation, NPU memory leaks, and cascading STARTUP_FAIL on subsequent iterations.

**Cleanup script — execute as a single Bash command after every iteration:**

```bash
# Step 1: Kill ALL vllm-related processes by multiple patterns
pkill -9 -f "vllm.entrypoints" 2>/dev/null
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "vllm_ascend" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "VLLM::Worker" 2>/dev/null
pkill -9 -f "multiproc_executor" 2>/dev/null
pkill -9 -f "patch_balance_schedule" 2>/dev/null
pkill -9 -f "ray::" 2>/dev/null

# Step 2: Kill by parent PID chain — find the vllm serve/bash wrapper and kill its process group
for pid in $(pgrep -f "vllm" 2>/dev/null); do
  kill -9 -"$pid" 2>/dev/null  # kill the process group
  kill -9 "$pid" 2>/dev/null
done

# Step 3: Wait for OS to release NPU resources and reap zombies
sleep 15

# Step 4: Verify — count remaining vllm-related processes
REMAINING=$(ps aux | grep -E "vllm|VLLM|EngineCore|Worker_TP|multiproc_executor" | grep -v grep | grep -v defunct | wc -l)
ZOMBIES=$(ps aux | grep -E "vllm|VLLM|EngineCore|Worker_TP" | grep -v grep | grep defunct | wc -l)
echo "Cleanup check: $REMAINING live processes, $ZOMBIES zombies"

# Step 5: If zombies remain, kill their parent (init/PID 1 or the shell that spawned them)
if [ "$ZOMBIES" -gt 0 ]; then
  echo "WARNING: $ZOMBIES zombie processes remain. Killing zombie parents..."
  ps aux | grep -E "vllm|VLLM|EngineCore|Worker_TP" | grep defunct | awk '{print $2}' | while read zpid; do
    PPID=$(ps -o ppid= -p "$zpid" 2>/dev/null | tr -d ' ')
    if [ -n "$PPID" ] && [ "$PPID" != "1" ] && [ "$PPID" != "0" ]; then
      kill -9 "$PPID" 2>/dev/null
      echo "Killed parent PID $PPID of zombie $zpid"
    fi
  done
  sleep 5
fi

# Step 6: Final verification — port must be free
if lsof -i :5001 2>/dev/null | grep -q LISTEN; then
  echo "ERROR: Port 5001 still occupied after cleanup!"
  lsof -i :5001 2>/dev/null
fi
```

**Key rules for cleanup:**
- Execute cleanup AFTER writing the ledger entry for the iteration, but BEFORE starting the next lever
- The 15-second sleep is mandatory — Ascend CANN runtime needs time to release NPU HBM after SIGKILL
- Zombies (state `Z`/`<defunct>`) do NOT hold NPU memory themselves, but their parent processes may still hold references to CANN device contexts. Killing the parent chain is necessary.
- If after cleanup there are still >0 live vllm processes, retry once with `kill -9` on each remaining PID, then wait another 10 seconds
- Do NOT proceed to the next lever until the cleanup verification passes (0 live processes, port free)
- This cleanup is a hard requirement — even if the server appeared to stop cleanly, residual EngineCore/Worker processes may still hold NPU device handles

## Timeout and retry contract

Long waits must be bounded.

- Define a startup wait budget for server readiness and treat timeout as `STARTUP_FAIL`
- Define a benchmark wait budget for warmup plus evalscope completion and treat timeout as `BENCHMARK_FAIL`
- Retry at most once for transient startup or benchmark issues; if retried, record that it was a retry and why
- After a failed attempt, clean up the candidate server before moving to the next lever

## Continuation contract

The skill must continue executing after Phase 1 in the same run unless one of the explicit exit conditions is hit.

- Baseline-only output is **incomplete** and must not be presented as the final result
- After Phase 1 finishes, immediately continue into Phase 2 and keep advancing through the ranked lever list
- Do not pause just because the ledger header, baseline table, or iteration-0 summary has been written
- Do not generate the final summary/report until at least one post-baseline lever attempt has been recorded, unless execution is blocked by an explicit failure outside the lever space (for example: missing dependency, no write permission, repeated server launch failure for the baseline itself, or user stop)
- If execution is blocked before any post-baseline lever can be attempted, write the blocking reason explicitly and mark the run as blocked rather than silently ending after the baseline

**Robustness guarantees** — the skill must:
- Catch and handle all exceptions during server startup, benchmark execution, and ledger writes
- If a lever attempt fails for any reason (server crash, module not found, timeout, etc.), immediately write a ledger entry with the appropriate failure verdict (`STARTUP_FAIL`, `BENCHMARK_FAIL`) and continue to the next lever
- Never exit the skill early due to individual lever failures — only the explicit exit conditions above can terminate the run
- If context window is approaching limits (detected via conversation length or explicit warning), write a checkpoint note to the ledger and exit gracefully so the wrapper can resume
- Treat missing dependencies, incompatible hardware features, or environment issues as `STARTUP_FAIL` or `SKIPPED_CONFLICT`, not as reasons to abort the entire run

## Progress visibility contract

The skill must behave like a visible long-running workflow, not a silent batch job.

- Before starting each major action, emit a short status update naming the current phase, lever, and action
- For any action expected to take more than 30 seconds, emit heartbeat progress updates at least every 30 seconds
- Each progress update must include as many of these as are known: current phase, current lever, current step, current command category (`server_start`, `warmup`, `benchmark`, `profiling`, `report_generation`), elapsed time, and next expected milestone
- When starting a lever attempt, explicitly print `Current lever:` and `Next lever if this finishes:` so the user can see forward progress
- When running a benchmark, explicitly print the fixed workload (`c=1/4/8`, request counts, input/output tokens) and the artifact directory being written
- When waiting on server readiness, say whether the skill is waiting for process start, health check success, warmup completion, or benchmark completion
- If a step appears slow, explain the likely reason in plain language (for example: model load, graph compilation, benchmark still running at higher concurrency, profiling trace export)
- Never disappear behind a single long terminal command without accompanying narrative updates
- If an iteration is retried, say why it is being retried and what changed
- Keep updates minimal: one or two short lines focused on action, not theory

---

## Workflow

### Phase 0 — Setup & Resume

Create the run directory and initialize the optimization ledger:

```
<work_dir>/
├── ledger.md               # optimization history, all attempts
├── baseline/               # baseline benchmark artifacts
├── iter_01/ iter_02/ ...   # per-iteration benchmark results
└── final_report.md         # synthesis after loop exits
```

**Resume logic** — before starting any work, check for an existing ledger in `work_dir`:

0. **Handle explicit fresh start** — if the user explicitly requests to start fresh (e.g. "重新开始", "fresh start", "run a complete flow from scratch"), do NOT resume. Archive or overwrite the existing `ledger.md`, forcefully kill any existing `vllm` zombie processes, and proceed directly to Phase 1.
1. **Read the ledger** and extract:
   - Baseline metrics (c=1/4/8 ttft_avg, tpot_avg, TPS) — these are immutable
   - All completed iteration verdicts (WIN / LOSS / NEUTRAL / STARTUP_FAIL / BENCHMARK_FAIL / SKIPPED_CONFLICT)
   - The most recent lever attempt: which phase, which lever number, and its verdict
   - Count of completed iterations per phase

2. **Handle abandoned `TESTING` state** — if the last ledger entry shows `Status: TESTING` or `Status: IN_PROGRESS` without a verdict:
   - This indicates the session was interrupted mid-lever
   - Do NOT count this as a completed attempt
   - The next action is to retry this same lever from scratch (stop any leftover server, restart, re-benchmark)
   - Append a note to the ledger: `## Iteration N (retry) — <phase.lever>` with reason: `Session interrupted during previous attempt; retrying`

3. **Determine the next lever** — based on the last completed verdict:
   - If baseline is complete but no Phase 2 levers attempted → start Phase 2.1
   - If Phase 2 has completed levers 2.1–2.3 → continue with 2.4
   - If Phase 2 is complete (all levers attempted) → start Phase 3.1
   - If a lever was `WIN` → carry its config forward as the new incumbent
   - If a lever was `LOSS` / `NEUTRAL` / failure → discard and use the prior incumbent
   - Continue through all phases (2–7) until all levers are exhausted or an exit condition is met

4. **Load the incumbent config** — reconstruct the exact server launch command by:
   - Starting from the Phase 1 baseline command
   - Applying all `WIN` levers in order (each WIN modifies one field)
   - This reconstructed command is the starting point for the next lever attempt

5. **Print resume status**:
```text
Phase 0/7 | resume | work_dir=<path> | last_completed=<phase.lever> verdict=<WIN|LOSS|...> | next=<phase.lever> | wins_so_far=N
```

**First run (no ledger)** — if no ledger exists, initialize a fresh one and proceed to Phase 1:
```text
Phase 0/7 | setup | action=initialize_work_dir | work_dir=... | next=baseline startup
```

### Phase 1 — Baseline Benchmark

Run the fixed-workload benchmark at `low_conc_levels` using the **baseline server configuration**:

- If `baseline_launch_command` is provided, use it exactly as the Phase 1 startup command
- Otherwise, start from the default template below
- The Phase 1 startup config becomes the initial incumbent; every later iteration must inherit it and change only one lever at a time
- Before launching, check whether a matching baseline server is already healthy on the target port; reuse it only if config attribution remains unambiguous, otherwise stop and relaunch

Default template when `baseline_launch_command` is not provided:

```bash
ASCEND_RT_VISIBLE_DEVICES=<devices> \
VLLM_ASCEND_ENABLE_ACLGRAPH=1 \
VLLM_ASCEND_ENABLE_NZ=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --served-model-name <model_name> \
  --tensor-parallel-size <tp> \
  --max-model-len <max_model_len> \
  --dtype <dtype> \
  --port <port> \
  --trust-remote-code
```

Benchmark command (warm-up 20 requests at c=1, then measure at all concurrency levels):
```bash
evalscope perf \
  --parallel <low_conc_levels> \
  --number <requests_per_level> \
  --warmup-num 20 \
  --model <model_name> \
  --url http://127.0.0.1:<port>/v1/chat/completions \
  --api openai --stream \
  --dataset random \
  --max-tokens <output_tokens> --min-tokens <output_tokens> \
  --min-prompt-length <input_tokens> --max-prompt-length <input_tokens> \
  --tokenizer-path <model_path>
```
Note: evalscope perf saves output to `outputs/<timestamp>/<model>/` in the current directory. Run the command from the appropriate work subdirectory (e.g., `cd <work_dir>/baseline && evalscope perf ...`) so artifacts land in the right place. Do NOT pass `--work-dir` — it is not a valid evalscope perf argument.

Incumbent inheritance rules after Phase 1:
- Preserve all env vars and CLI args from the Phase 1 baseline in every later iteration unless the current lever is explicitly testing one of those fields
- If a candidate lever conflicts with a user-provided baseline setting, change only that single field for the A/B run and keep the rest of the baseline untouched
- Record the exact effective startup command in the ledger so the winning config remains copy-pasteable

Extract and record in ledger:
- c=1: `ttft_avg`, `tpot_avg`, `output_token_throughput` (TPS), `latency_avg`
- c=4, c=8: same metrics
- These become the **immutable reference values** — never modified after Phase 1
- After recording the baseline, execute the post-iteration cleanup protocol to stop the baseline server, then proceed to the first Phase 2 lever attempt in the same session

Required status updates during Phase 1:
- Before startup: `Phase 1/7 | baseline | action=server_start | command_source=user_baseline|default_template`
- While waiting: `Phase 1/7 | baseline | action=wait_ready | state=process_started|healthcheck_pending|warmup_running`
- Before benchmark: `Phase 1/7 | baseline | action=benchmark | c=1/4/8 | requests=20/80/160 | artifacts=<work_dir>/baseline`
- After benchmark: `Phase 1/7 | baseline | action=record_baseline | ttft_c1=<value> | next=Phase 2 <first lever>`

### Phase 2 — Scheduler & Engine Shell Tuning

**Goal**: reduce queuing overhead and TTFT at low concurrency without touching kernel code.

Candidate levers (try in order):

#### 2.1 Balance scheduling
```bash
--additional-config '{"enable_balance_scheduling": true}'
```
Expected impact: reduces decode stalls at c=1 and c=4 when prefill and decode batches are mixed.

#### 2.2 CPU binding
```bash
--additional-config '{"enable_cpu_binding": true}'   # default true; try false if NUMA is fragmented
```

#### 2.3 PagedAttention block size
```bash
--block-size 128    # 910C optimal; try 64 if memory-constrained
```
Expected impact: larger blocks → fewer KV cache pointer dereferences → lower TTFT.
Note: `block_size=128` is the xlite_graph_config recommendation for 910C; always try this first.

#### 2.4 Chunked prefill configuration
```bash
--enable-chunked-prefill \
--max-num-batched-tokens 2048   # try: 1024, 2048, 4096
```
Expected impact on c=1: reduces head-of-line blocking from large prefills.

#### 2.5 max_num_seqs
```bash
--max-num-seqs 32    # try: 16, 32, 64, 128
```
At low concurrency, smaller values reduce scheduler overhead.

For each candidate:
- Precheck applicability first; if the lever is incompatible with the current model, TP mode, EP mode, or incumbent config, record `SKIPPED_CONFLICT` immediately and continue
- Start fresh server from the current incumbent command (Phase 1 baseline or prior WIN) with the single changed parameter
- If an incumbent server is still running but does not match the intended candidate config, stop it before launching the candidate
- If the server fails to start, never becomes healthy, or the benchmark cannot complete, still append a ledger entry for that attempt with a failure verdict and the captured reason
- Run 20 warmup requests at c=1, then run the **fixed benchmark**: `--parallel 1 4 8 --number <requests_per_level>`
- **Decision gate**: compare `ttft_avg` at **c=1** against the baseline value
  - WIN if improvement ≥ `improvement_threshold_pct` (default 1%) **and** `tpot_avg` at c=1 does not regress > 5%
  - NEUTRAL if `ttft_avg` improves but `tpot_avg` regresses > 5% (trade-off, not a clear win)
  - LOSS if c=1 `ttft_avg` regresses > 2%
  - NEUTRAL otherwise
- Failure verdicts:
  - `STARTUP_FAIL` if the process exits, crashes, or never reaches serving-ready state
  - `BENCHMARK_FAIL` if the server starts but warmup or `evalscope perf` fails before producing a complete c=1/4/8 result set
  - `SKIPPED_CONFLICT` if the lever is known in advance to be incompatible with the current model, hardware mode, or incumbent config and is intentionally skipped after a concrete conflict check
- Also record `ttft_avg` and `tpot_avg` at c=4/8 — these appear in the ledger but do NOT affect WIN/LOSS
- Log result in ledger (`WIN` / `LOSS` / `NEUTRAL` / `STARTUP_FAIL` / `BENCHMARK_FAIL` / `SKIPPED_CONFLICT`)
- If WIN: carry this change into the next phase as the new incumbent

Required status updates for every lever:
- Before change: `Phase <n>/7 | lever=<phase.lever> | action=prepare_attempt | incumbent=<short name> | next_lever=<next candidate>`
- Before stop/reuse decision: `Phase <n>/7 | lever=<phase.lever> | action=server_check | port=<port> | decision=reuse|stop_and_restart`
- Before startup: `Phase <n>/7 | lever=<phase.lever> | action=server_start | changed_field=<single field>`
- Before benchmark: `Phase <n>/7 | lever=<phase.lever> | action=benchmark | c=1/4/8 | artifacts=<work_dir>/iter_N`
- After verdict: `Phase <n>/7 | lever=<phase.lever> | action=record_verdict | verdict=WIN|LOSS|NEUTRAL|STARTUP_FAIL|BENCHMARK_FAIL|SKIPPED_CONFLICT | carry_forward=YES|NO | next=<next candidate>`

### Phase 3 — Ascend Fusion & Graph Compilation Tuning

**Goal**: enable or tune Ascend-specific kernel fusion and graph compilation backends.

Try in order:

#### 3.0 CUDAGraph mode (graph compilation decision)

```bash
--compilation-config '{"cudagraph_mode": "FULL"}'              # vs baseline FULL_DECODE_ONLY
```
This is the most upstream graph-compilation lever: it decides whether decode (and prefill) run inside a captured ACL graph. `cudagraph_mode` is the field on `--compilation-config`, NOT an `--additional-config` key.

Applicable `CUDAGraphMode` values on vllm-ascend:
- `FULL_DECODE_ONLY` (= `FULL` decode, `NONE` prefill) — decode captured, prefill not. Common baseline choice.
- `FULL` (= `FULL` decode, `FULL` prefill) — prefill also captured. A/B against `FULL_DECODE_ONLY`: same decode graph, but prefill runs as a graph too (longer startup, possibly lower prefill latency / TTFT).
- `FULL_AND_PIECEWISE` (= `FULL` decode, `PIECEWISE` prefill) — try if `FULL` prefill capture is memory-constrained.
- `PIECEWISE` — piecewise graph only (decode NOT fully captured → decode TPOT usually regresses; useful only as a diagnostic baseline).
- `NONE` — no graph (equivalent to `--enforce-eager`); decode TPOT usually regresses. Use only to measure the graph's contribution.

**When to run this lever**: ALWAYS, as the first Phase 3 attempt. `cudagraph_mode` is the single field with the most direct effect on decode TPOT — skipping it leaves the most upstream decode switch untested.

**Primary A/B**: `FULL` vs the baseline's graph mode. Only change the `cudagraph_mode` field; preserve the rest of `--compilation-config` and the incumbent command. If the baseline did not set `cudagraph_mode` (vllm-ascend default is `FULL`), test `FULL_DECODE_ONLY` as the candidate and compare against the default-`FULL` incumbent.

Expected impact: if the baseline is `FULL_DECODE_ONLY`, switching decode-path capture rarely helps (decode is already captured); the meaningful delta is on TTFT/prefill. If the baseline is `FULL` (default), `FULL_DECODE_ONLY` can reduce startup + prefill overhead with identical decode TPOT — a likely WIN on TTFT-neutral workloads. `PIECEWISE`/`NONE` are expected regressions on TPOT; record them as LOSS/NEUTRAL for attribution, do not carry forward.

#### 3.1 NZ weight format
```bash
VLLM_ASCEND_ENABLE_NZ=1    # default 1; also try 2 (enable for all ops)
VLLM_ASCEND_ENABLE_NZ=2
```
Expected impact: NZ format reduces memory bandwidth on 910C for matmul-heavy layers.

#### 3.2 npugraph_ex with static kernel
```bash
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true}}'
```
Expected impact: static kernel pre-compiles operator binaries for fixed batch shapes; reduces JIT overhead at c=1. Note: adds startup time.

#### 3.3 Norm-quant fusion
```bash
--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": true, "fuse_qknorm_rope": true}}'
```
Default is already true; try disabling if baseline seems slow on normalization:
```bash
--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'
```

#### 3.4 AllReduce-RMSNorm fusion (TP > 1 only)
```bash
--additional-config '{"ascend_compilation_config": {"fuse_allreduce_rms": true}}'
```
Expected impact: fuses allreduce + rmsnorm → reduces synchronization latency in TP runs.

#### 3.5 MatMul-AllReduce fusion (TP > 1 only)
```bash
VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1
```
Expected impact: overlaps matmul and allreduce; helps TPOT at low concurrency.

#### 3.6 GMM-SwiGLU-Quant fusion (MoE models only)
```bash
--additional-config '{"ascend_fusion_config": {"fusion_ops_gmmswigluquant": true}}'
```

#### 3.7 XliteGraph mode
```bash
--additional-config '{"xlite_graph_config": {"enabled": true, "full_mode": false}}'
```
Requires `block_size=128`. Expected impact: Xlite graph backend can significantly accelerate decode on 910C. Try `full_mode: true` if `false` wins.

### Phase 4 — FlashComm & Communication Overlap Tuning (TP > 1 only)

Skip this phase if `tensor_parallel_size == 1`.

#### 4.1 FlashComm1 (prefill-side comm-compute overlap)
```bash
--additional-config '{"enable_flashcomm1": true}'
```
Expected impact: overlaps TP allreduce with compute during prefill → lower TTFT.

#### 4.2 FlashComm2 (decode-side comm overlap)
```bash
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=<tp>
```
Try values: `1`, `tp/2`, `tp`. Expected impact: further reduces allreduce stalls during decode.

#### 4.3 Prefill comm-compute overlap
```bash
--additional-config '{"prefill_comm_compute_overlap": true}'
```

#### 4.4 HCCL buffer size
```bash
HCCL_BUFFSIZE=200    # try: 100, 200, 400 (MB)
```
Larger buffer can reduce HCCL fragmentation overhead.

### Phase 5 — Weight Prefetch & Memory Layout

#### 5.1 Weight prefetch
```bash
--additional-config '{"weight_prefetch_config": {"enabled": true}}'
```
Expected impact: prefetches next layer weights during current layer compute; masks HBM latency.
Try adjusting `prefetch_ratio` for attention vs MoE separately if overall win is marginal:
```bash
--additional-config '{"weight_prefetch_config": {"enabled": true, "prefetch_ratio": {"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 1.0}}}}'
```

#### 5.2 MLAPO (DeepSeek W8A8 only)
```bash
VLLM_ASCEND_ENABLE_MLAPO=1    # default 1; disable if memory is tight
```

#### 5.3 Fused MC2 (MoE W8A8 + EP only)
```bash
VLLM_ASCEND_ENABLE_FUSED_MC2=1   # try: 0, 1, 2
```

### Phase 6 — Multistream & DSA Overlap (Advanced)

These levers involve deeper NPU pipeline overlapping. Apply only after Phases 2–5 are exhausted.

#### 6.1 Multistream shared-expert overlap (MoE only)
```bash
--additional-config '{"multistream_overlap_shared_expert": true}'
```

#### 6.2 Multistream gate overlap (MoE only)
```bash
--additional-config '{"multistream_overlap_gate": true}'
```

#### 6.3 DSA preprocess overlap
```bash
--additional-config '{"multistream_dsa_preprocess": true}'
```

#### 6.4 DSv4 DSA overlap (DeepSeek V4 only)
```bash
--additional-config '{"multistream_dsv4_dsa_overlap": true}'
```

### Phase 7 — Profiling-Driven Triton Kernel Tuning

**Trigger condition**: Apply after all Phases 2–6 levers have been attempted.

This phase uses vLLM's built-in profiler to identify the slowest operators, then tunes their block size parameters if they are in the tunable category.

#### 7.1 Collect profiling trace

Server must be started with profiling enabled:
```bash
--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir=<work_dir>/profiling
```

Then at c=1:
```bash
# Start profiling
curl -X POST http://127.0.0.1:5000/start_profile

# Send 20 requests at c=1 to capture a representative trace
evalscope perf --parallel 1 --number 20 ...

# Stop profiling
curl -X POST http://127.0.0.1:5000/stop_profile
```

The trace is saved as JSON in `<work_dir>/profiling/`. Claude reads the JSON and identifies:
- **Top 5 longest ops** (by total GPU time)
- **NPU idle gaps** between consecutive ops (indicating CPU-side bottleneck or launch overhead)
- **Allreduce / HCCL ops** (if TP > 1, check if communication is on the critical path)

#### 7.2 Classify ops into tunable vs. non-tunable

After reading the trace, classify each slow op:

**Tunable (Triton kernels in vllm-ascend source):**

| Op pattern | Source file | Tunable parameter |
|------------|-------------|-------------------|
| `rms_norm` | `ops/triton/rms_norm.py:52` | `ROW_BLOCK_SIZE` (default 16) |
| `layernorm_gated` | `ops/triton/layernorm_gated.py:155` | `BLOCK_M` (default 64) |
| `matmul` / `gemm` | `ops/triton/batch_invariant/matmul.py:137` | `BLOCK_M/N/K` (default 128/128/64) |
| `flash_attention` / `fa3` | `attention/fa3_v1.py:90` | `num_splits` (default 1), `attention_chunk` (default 0) |

**Non-tunable (CANN C++ compiled ops):**

| Op pattern | Reason |
|------------|--------|
| `grouped_matmul_swiglu_quant` | Tiling params (`baseM/baseN/baseK`) are C++ compile-time constants in `csrc/gmm/` headers |
| `paged_attention` / `npu_fused_infer_attention_score` | Implemented via ACLNN, no user-controllable parameters |
| Any op prefixed with `aclnn` | CANN runtime op, tiling decided at op compilation time |

#### 7.3 Tune tunable ops

For each tunable op found in 7.2:

1. **Edit the source file** — change the block size constant to a new value (try 2× or 0.5× of default)
2. **Restart server** with the same configuration (no other changes)
3. **Run benchmark** at c=1/4/8
4. **Record in ledger** with verdict WIN/LOSS/NEUTRAL based on `ttft_avg` at c=1
5. If WIN: keep the edit; if LOSS/NEUTRAL: revert the edit and try the next parameter

Example for `rms_norm`:
```python
# ops/triton/rms_norm.py line 52
ROW_BLOCK_SIZE = 32  # was 16, try doubling
```

For `fa3_v1.py` `num_splits`:
```python
# attention/fa3_v1.py line 90
num_splits=2,  # was 1, try splitting attention across NPU cores
```

#### 7.4 Record non-tunable findings

For non-tunable ops that appear in the top-5 slow list, append to `final_report.md` under a "Manual Intervention" section:

```markdown
## Manual Intervention Required

The following operators were identified as bottlenecks but cannot be tuned via source modification:

- `aclnnGroupedMatmul` (18% of total GPU time): Tiling parameters `baseM/baseN` are C++ compile-time constants. To tune, use CANN's AOE (Ascend Operator Expert) tool offline with the specific input shapes observed in this run.
- `aclnnFlashAttention` (12% of total GPU time): No user-controllable parameters. Consider contacting Huawei for updated ACLNN binaries or checking if a newer CANN version has optimized this op.
```

This allows the user to decide whether to invest time in offline AOE tuning or escalate to Huawei support.

---

## Ledger format

Append one entry per iteration to `ledger.md`.

- For successful benchmarked attempts (`WIN` / `LOSS` / `NEUTRAL`), **all three concurrency rows (c=1/4/8) are mandatory**
- For failed or skipped attempts (`STARTUP_FAIL` / `BENCHMARK_FAIL` / `SKIPPED_CONFLICT`), use the failure template below instead of the metric table

```markdown
## Iteration N — <phase>.<lever_name>

**Hypothesis**: <why this lever should help, citing phase doc>
**Change**: `<exact env var or --additional-config JSON>`
**Incumbent config before**: `<previous winning command>`

### Results vs Baseline (fixed benchmark: c=1/4/8, primary=ttft_avg@c=1, guard=tpot_avg@c=1)

| Concurrency | ttft_avg (ms) | ttft_avg delta | tpot_avg (ms) | tpot_avg delta | TPS (tok/s) | TPS delta |
|-------------|---------------|----------------|---------------|----------------|-------------|-----------|
| **1** ⭐ | 241.0 | **-15.4% ← DECISION** | 27.1 | -4.2% | 36.9 | +4.4% |
| 4 | 420.0 | -8.3% | 29.5 | +1.0% | 135.6 | -1.0% |
| 8 | 610.0 | -5.2% | 31.0 | +2.1% | 258.1 | -2.1% |

**Verdict**: WIN  ← ttft_avg c=1 improved -15.4% (≥1%), tpot_avg c=1 improved -4.2% (no regression)
**Carry forward**: YES
**Notes**: <any anomalies, warnings from server log>
```

Failure / skipped attempt template:

```markdown
## Iteration N — <phase>.<lever_name>

**Hypothesis**: <why this lever should help, or why it was checked>
**Change**: `<exact env var or --additional-config JSON>`
**Incumbent config before**: `<previous winning command>`
**Verdict**: STARTUP_FAIL
**Failure stage**: server_startup
**Reason**: <short reason, e.g. incompatible with xlite_graph_config + block_size=64>
**Evidence**:
- Exit code: <code or N/A>
- Health check: <never ready / failed after warmup / benchmark aborted>
- Log excerpt: `<relevant stderr or server log lines>`
**Carry forward**: NO
**Notes**: <follow-up, retry hint, or why this lever was skipped>
```

Rules:
- A lever is a **WIN** if `ttft_avg` at **c=1** improves ≥ `improvement_threshold_pct` **and** `tpot_avg` at c=1 does not regress > 5%
- A lever is **NEUTRAL** if `ttft_avg` at c=1 improves but `tpot_avg` regresses > 5% (trade-off)
- A lever is a **LOSS** if `ttft_avg` at c=1 regresses > 2%
- A lever is **STARTUP_FAIL** if no valid serving endpoint is established for the attempt
- A lever is **BENCHMARK_FAIL** if serving starts but the benchmark does not finish with a complete c=1/4/8 result set
- A lever is **SKIPPED_CONFLICT** if the incompatibility is known before launch and is backed by a concrete config or model constraint
- c=4/8 rows are recorded for trend analysis only — they never override the c=1 verdict
- Wins are carried forward cumulatively; losses, neutrals, failures, and skipped conflicts are discarded
- Never modify a prior ledger entry — only append

---

## Exit conditions

**Run all phases (1-7) completely.** Do not stop early based on consecutive no-WIN phases or plateau patterns.

Stop only when:

1. **Budget**: `max_iterations` reached (safety valve, default 50)
2. **Manual stop**: user explicitly says "stop" or "enough"
3. **All phases complete**: Phase 7 has been attempted, regardless of WIN/LOSS/NEUTRAL outcomes
4. **Blocked before tuning can continue**: baseline itself cannot be made runnable, required tooling is missing, or the environment prevents any further lever attempt from being executed safely

Non-exit examples:
- Baseline benchmark completed successfully
- Ledger header or baseline report has been written
- No WIN found yet
- A single lever failed to start
- An iteration was marked `STARTUP_FAIL`, `BENCHMARK_FAIL`, or `SKIPPED_CONFLICT`

The skill should exhaust all tuning levers across all phases before generating the final report.

---

## Running the loop to completion (auto-resume wrapper)

The full campaign spans 7 phases and dozens of lever attempts — far more than a single `claude` session can survive before the context window fills. The skill is designed around this: each session runs until it must yield (context approaching limits — see the checkpoint-and-exit rule), then a **wrapper** re-launches a fresh session that reads the ledger and continues. This is how a fragmented set of sessions becomes one complete optimization run.

Use `scripts/auto_resume_wrapper.sh` to drive this. It wraps the `claude` CLI call and relaunches it until `final_report.md` appears in `work_dir` (the skill's only completion signal — see "Final report"). The ledger in `work_dir` is the hand-off state between sessions: a freshly launched session follows the "Resume logic" section, reads the ledger, and picks up at the next unfinished phase.lever.

**Usage:**

```bash
bash scripts/auto_resume_wrapper.sh \
  --work-dir <work_dir> \
  -- claude -p "<skill prompt with model/work_dir/etc.>" --max-turns 0
```

- `--work-dir` — the campaign's `work_dir`; the wrapper watches `<work_dir>/final_report.md` for completion and `<work_dir>/ledger.md` for resume state.
- Everything after `--` is the full `claude` command, passed through verbatim to each relaunch. Use `--max-turns 0` (or a bounded value) so each session yields on its own when the context fills, letting the wrapper decide whether to relaunch.
- The wrapper retries up to 20 times, sleeping 5s between attempts, and exits 0 as soon as `final_report.md` exists. It checks for completion both **before** launching (so an already-finished run is not re-launched) and **after** each session exits.

**What the wrapper is NOT:**
- It does not read or parse the ledger — the skill does that on resume. The wrapper only checks for the presence of `final_report.md`.
- It does not decide which lever to try next — that is the skill's job per the RLCR principle.
- It does not recover from a corrupted or contradictory ledger; if the skill cannot resume cleanly, it should surface that to the user rather than silently looping.

**Recommended run shape:** launch the wrapper once in the background, let it drive the whole campaign to `final_report.md`, then read the report. Do not manually relaunch `claude` mid-campaign — that races with the wrapper and can spawn two sessions writing to the same ledger.

---

## Final report

After the loop exits, write `final_report.md` using `scripts/generate_tuning_report.py`:

```markdown
# vLLM-Ascend Tuning Report — <model_name>

## Summary
- Total iterations: N
- Winning levers: M
- Failed or skipped levers: K
- Overall improvement on target metric (c=1): X%
- Completion status: `completed` | `blocked` | `stopped_by_user` | `budget_exhausted`

## Baseline vs Best Configuration

| Concurrency | Metric | Baseline | Best | Improvement |
|-------------|--------|----------|------|-------------|
| 1 | TTFT avg | ... | ... | ... |
| 1 | TPOT avg | ... | ... | ... |
| 4 | TTFT avg | ... | ... | ... |
| 8 | TTFT avg | ... | ... | ... |

## Winning Configuration (copy-pasteable)

```bash
ASCEND_RT_VISIBLE_DEVICES=...
VLLM_ASCEND_ENABLE_NZ=...
...
python3 -m vllm.entrypoints.openai.api_server \
  --model ... \
  --additional-config '...' \
  ...
```

## Optimization History

<condensed ledger — one line per iteration>

## Levers That Did Not Help

<list with brief reason>

## Failed / Incompatible Levers

| Lever | Verdict | Stage | Reason | Evidence |
|-------|---------|-------|--------|----------|
| `xlite_graph.full_mode=true` | `STARTUP_FAIL` | `server_startup` | incompatible with current block size | `server exited with code 1: ...` |
| `fuse_allreduce_rms=true` | `SKIPPED_CONFLICT` | `precheck` | TP=1 so lever is not applicable | `tensor_parallel_size == 1` |

Rules for this section:
- Include every `STARTUP_FAIL`, `BENCHMARK_FAIL`, and `SKIPPED_CONFLICT` entry from the ledger
- Preserve the first actionable failure reason instead of collapsing all failures into "did not help"
- Include a short evidence string, preferably an exit code, failed health check, or log excerpt

## Manual Intervention Required

<Non-tunable ops identified by Phase 7 profiling — op name, GPU time percentage, reason it cannot be tuned, and suggested action (AOE / Huawei support)>

## Recommended Next Steps

<if gains are still possible: profiling suggestions, kernel-level work>
```

Finalization rules:
- Never treat the baseline section alone as the final deliverable
- Only generate the final report after an actual exit condition is met
- If the run is blocked before tuning can proceed, the final report must say `Completion status: blocked` and name the blocking reason
- If only baseline succeeded and no tuning iteration was attempted, that is a blocked or interrupted run, not a completed optimization campaign
- Before final report generation, emit a final progress line stating whether the run is exiting because of completion, block, user stop, or budget exhaustion

---

## Constraints

- **Execution over narration**: once prerequisites are satisfied, prefer the next concrete server/benchmark action over more discussion
- **One lever per iteration**: never bundle two changes; it destroys attribution
- **Fixed workload**: `input_tokens`, `output_tokens`, and `low_conc_levels` never change after Phase 1
- **Baseline inheritance**: if `baseline_launch_command` is provided, Phase 1 uses it exactly and every later iteration must preserve it except for the single field under test
- **Server ownership**: at most one campaign-owned server may be active on the target port at any time
- **Mandatory post-iteration cleanup**: execute the full cleanup script from "Post-iteration strict cleanup protocol" after EVERY iteration before starting the next lever. This kills all vllm/EngineCore/Worker processes, waits for NPU resource release, and verifies port availability. Skipping cleanup leads to zombie accumulation and NPU memory leaks.
- **Restart server cleanly** between every iteration — no warm state carry-over
- **Bounded waiting**: startup and benchmark waits must use explicit timeout budgets rather than unbounded waiting
- **Failure visibility**: startup failures, benchmark failures, and prechecked incompatibilities must still be written to the ledger and surfaced in the final report
- **Progress visibility**: long-running steps require heartbeat updates at least every 30 seconds with phase, lever, action, and next milestone
- **Warm up before every measurement**: 20 warmup requests at c=1 before measurement
- **Record everything**: even a 0.1% change gets a ledger entry
- **TP-only levers**: FlashComm, MatMul-AllReduce, AllReduce-RMSNorm only apply when `tensor_parallel_size > 1`
- **MoE-only levers**: GMM-SwiGLU fusion, multistream shared-expert, DSA only apply to MoE architectures

---

## Example invocation

```
/ascend-vllm-serving-tune-loop
model_path=/data/models/Qwen3-32B
model_name=Qwen3-32B
tensor_parallel_size=4
target_metric=ttft_avg
input_tokens=1024
output_tokens=512
```

```
/ascend-vllm-serving-tune-loop
model_path=/data/models/DeepSeek-V3
model_name=DeepSeek-V3
tensor_parallel_size=8
baseline_launch_command='ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ASCEND_ENABLE_ACLGRAPH=1 VLLM_ASCEND_ENABLE_NZ=2 python3 -m vllm.entrypoints.openai.api_server --model /data/models/DeepSeek-V3 --served-model-name DeepSeek-V3 --tensor-parallel-size 8 --max-model-len 32768 --dtype bfloat16 --block-size 128 --max-num-seqs 32 --additional-config "{\"enable_balance_scheduling\": true}" --port 5000 --trust-remote-code'
target_metric=ttft_avg
low_conc_levels="1 4 8"
requests_per_level="20 80 160"
max_iterations=50
```