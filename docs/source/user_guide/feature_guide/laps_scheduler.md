# LAPS-Inspired Prefill Scheduling

`vllm-ascend` now provides an engine-side, NPU-oriented adaptation of the core
LAPS scheduling idea from the paper *Length-Aware Prefill Scheduling for LLM
Serving*.

## What Was Ported

The original LAPS artifact on top of SGLang contains three layers:

1. Dual-queue scheduling for short and long prefills.
2. A short-request waiting window to build better short-prefill batches.
3. Dynamic GPU allocation in the router/load balancer.

For `vllm-ascend`, the currently ported pieces are the **engine-local**
mechanisms:

- `dual-queue`: split the waiting queue into short and long prompt classes.
- `waiting window`: optionally hold an isolated short request for a small time
  window so that more short requests can join the same batch.

The CUDA-specific `attention-in-graph` optimization from the LAPS SGLang branch
was intentionally **not** ported, because it is tightly coupled to CUDA graph
capture and FlashAttention internals. Likewise, router-level dynamic allocation
has **not** been merged into the Ascend proxy layer yet.

## Why It Fits `vllm-ascend`

The vLLM scheduler already has a centralized waiting queue in the engine core,
which makes the LAPS queueing policy portable without changing model execution
code. This is a good match for NPU deployment because the main benefit here is
request isolation and batching behavior, not CUDA-only kernel machinery.

## Configuration

Enable the feature with environment variables before launching `vllm serve`:

```bash
export VLLM_ASCEND_LAPS_SCHEDULING=1
export VLLM_ASCEND_LAPS_THRESHOLD=256
export VLLM_ASCEND_LAPS_WAIT_WINDOW_MS=5
export VLLM_ASCEND_LAPS_WAIT_MAX_BATCH=4
```

### Variables

- `VLLM_ASCEND_LAPS_SCHEDULING`
  - `1` enables the LAPS-style waiting queue patch.
- `VLLM_ASCEND_LAPS_THRESHOLD`
  - Prompts with `num_prompt_tokens <= threshold` are treated as short.
- `VLLM_ASCEND_LAPS_WAIT_WINDOW_MS`
  - `0` means short requests dispatch immediately.
  - Positive values keep an isolated short batch waiting briefly.
- `VLLM_ASCEND_LAPS_WAIT_MAX_BATCH`
  - Dispatch short requests early once this many short requests are queued,
    even if the wait window has not expired.

## Scheduling Semantics

- If short requests are ready, the scheduler prefers them over long requests.
- If a short waiting window is active and no short batch is ready yet, long
  requests are still allowed to run.
- If only short requests are queued and the waiting window has not expired, the
  engine stays idle briefly instead of dispatching a tiny short batch
  immediately.

## Current Scope and Limitations

- The current implementation is enabled only for the **FCFS** scheduler policy.
- The adaptation is **engine-local**; it does not yet rebalance prefill and
  decode instances across nodes or proxies.
- The implementation targets the vLLM waiting-queue layer and is intended for
  PD / EPD style serving where prompt length skew is a dominant bottleneck.
