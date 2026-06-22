# LAPS-Inspired Prefill Scheduling

`vllm-ascend` now provides an engine-side, NPU-oriented adaptation of the core LAPS scheduling idea from the paper *Length-Aware Prefill Scheduling for LLM Serving*.

## What Was Ported

The original LAPS artifact on top of SGLang contains three layers:

1. Dual-queue scheduling for short and long prefills.
2. A short-request waiting window to build better short-prefill batches.
3. Dynamic GPU allocation in the router/load balancer.

For `vllm-ascend`, the currently ported pieces are the **engine-local** mechanisms:

- `triple-queue`: split the waiting queue into immediate, short, and long prompt classes.
- `anti-starvation (aging)`: bound how long a long prefill can be held behind short prefills, so a sustained short-request stream cannot starve long requests indefinitely.

The original LAPS short-request *waiting window* was **not** ported: vLLM v1 already performs continuous batching (it re-batches all running requests and admits new ones up to the token/running budget every step, with chunked prefill), so manually accumulating a short batch is redundant and only adds latency.

The CUDA-specific `attention-in-graph` optimization from the LAPS SGLang branch was intentionally **not** ported, because it is tightly coupled to CUDA graph capture and FlashAttention internals (and is incompatible with the two-stage sparse attention used by models such as DeepSeek-V4). Likewise, router-level dynamic allocation has **not** been merged into the Ascend proxy layer yet.

## Why It Fits `vllm-ascend`

The vLLM scheduler already has a centralized waiting queue in the engine core, which makes the LAPS queueing policy portable without changing model execution code. This is a good match for NPU deployment because the main benefit here is request isolation and batching behavior, not CUDA-only kernel machinery.

## Configuration

LAPS is configured through `--additional-config` (the canonical source), under a `laps_config` block. It requires `recompute_scheduler_enable: true` in the same `additional_config`:

```bash
vllm serve <model> \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "laps_config": {
      "enabled": true,
      "threshold": 256,
      "long_max_wait_ms": 2000,
      "long_token_reservation": 0.2,
      "stats_log_interval_s": 0
    }
  }'
```

### `laps_config` fields

- `enabled` (bool, default `false`)
    - `true` enables the Ascend LAPS scheduler.
- `threshold` (int, default `256`)
    - Prompts with `num_prompt_tokens <= threshold` are treated as short.
- `long_max_wait_ms` (float, default `0`)
    - Anti-starvation aging bound for long prefills, in milliseconds.
    - `0` disables aging (strict short-priority).
    - Once the oldest long has waited `>= long_max_wait_ms` it becomes *eligible* to be promoted ahead of waiting shorts, but the actual promotion goes through the token-reservation bucket only (see `long_token_reservation`), which rate-limits aged-long admissions. There is **no unconditional deadline bypass**: the bucket is the single admission channel, so a backlog of aged longs can never flip the queue into long-first.
    - Requires `long_token_reservation > 0` when set (enforced by config validation); otherwise the bucket never refills and aging would be inert.
- `long_token_reservation` (float, default `0.0`, range `[0.0, 1.0]`)
    - Average fraction of token throughput reserved for **admitting** aged-long prefills ahead of waiting shorts. This is the *only* channel through which an aged long jumps a waiting short.
    - Implemented as a token bucket: it refills by `reservation * per-step token budget` every scheduling step (burst capped at a few steps' worth), and admitting an aged-long ahead of shorts spends that credit. An aged-long is promoted only while the bucket has credit; once spent, shorts are preferred until the bucket refills (typically within a handful of steps). This caps the aged-long admission rate at the reservation fraction and prevents a backlog of aged longs from flipping the queue into long-first and starving shorts.
    - Must be `> 0` whenever `long_max_wait_ms > 0` (config validation rejects the `long_max_wait_ms > 0, reservation = 0` combination). Larger values drain aged-long requests faster at the cost of short-request latency; the degradation to shorts is bounded by roughly the reservation fraction.
    - The bucket bounds the *admission rate*, not total compute: once admitted, a long prefill proceeds chunk-by-chunk through the normal running loop.
    - The bucket refills against `max_num_scheduled_tokens` (the full per-step budget), not the budget left after running requests. Under sustained backlog the bucket therefore tends to stay topped up, so aged longs are admitted at close to the full reservation rate. This errs toward stronger anti-starvation and is an intentional trade-off (see "Behavior under saturation" below).
- `long_burst_steps` (int, default `4`, range `>= 1`)
    - Burst window, in scheduling steps, the aged-long bucket may accumulate: capacity = `long_token_reservation * per-step token budget * long_burst_steps`. Advanced knob; the default suits most workloads.
- `stats_log_interval_s` (float, default `0`)
    - Periodic LAPS stats logging interval, in seconds. `0` disables it. Intended for benchmark observability without enabling global DEBUG logging.

## How It Is Selected

`vllm-ascend` selects the scheduler at config time:

- `laps_config.enabled = false`
    - Keep the normal scheduler path.
- `laps_config.enabled = true` and `recompute_scheduler_enable = true`
    - Keep `RecomputeScheduler`, and install the LAPS waiting queue inside it.
- `laps_config.enabled = true` and `recompute_scheduler_enable = false`
    - LAPS is not activated. The platform logs a warning and keeps the default scheduler path.
- `SLO_limits_for_dynamic_batch != -1`
    - `SchedulerDynamicBatch` takes precedence and LAPS is ignored.

In other words, the effective priority is:

`dynamic batch > recompute (+ optional LAPS) > default`

## Minimal Examples

Enable recompute scheduler together with LAPS:

```bash
vllm serve <model> \
  --additional-config '{
    "recompute_scheduler_enable": true,
    "laps_config": {
      "enabled": true,
      "threshold": 256,
      "long_max_wait_ms": 2000,
      "long_token_reservation": 0.2
    }
  }'
```

Disable LAPS explicitly (the default):

```bash
vllm serve <model> \
  --additional-config '{"laps_config": {"enabled": false}}'
```

## Scheduling Semantics

The `LapsRequestQueue` manages three queues:

- **immediate queue**: Requests that must be dispatched immediately, such as preempted requests or requests with already-computed tokens (recovery flows).
- **short queue**: Short prefills where `num_prompt_tokens <= threshold`.
- **long queue**: Long prefills where `num_prompt_tokens > threshold`.

Dispatch priority is: immediate > aged-long > short > long.

- Immediate requests are dispatched as soon as they arrive.
- Short requests are dispatched whenever the short queue is non-empty (vLLM's continuous batching groups them together each step).
- Long requests are normally only dispatched when no immediate or short requests are schedulable.
- **Anti-starvation (token-bucket aging):** aging kicks in once the oldest long request has waited longer than `laps_config.long_max_wait_ms`. The aged long is then promoted ahead of shorts **only** through the token-reservation bucket: while the bucket has credit it is promoted; once the credit is spent, shorts are preferred until the bucket refills (a few steps). There is deliberately no unconditional deadline bypass, so the aged-long admission rate is capped at the reservation fraction and the scheduler can never flip into long-first.
- **Token reservation (the single admission channel):** the bucket refills `laps_config.long_token_reservation` of the per-step token budget each step (burst capped at a few steps' worth). Admitting an aged-long ahead of shorts spends the long's **full remaining prefill** from the bucket — not just the current chunk. This matters because a long prefill monopolizes the per-step token budget for many steps; charging only the first chunk would let the per-step refill outpace the charge, so the bucket would never drain and the reservation would fail to throttle (a backlog of longs could then starve shorts entirely). Charging the full cost instead drives the bucket well below zero for a big long; the per-step refill repays that debt over the steps the long actually runs, and meanwhile shorts are preferred (a negative bucket blocks further promotions). When no short request is waiting, long is dispatched regardless of the bucket (stall avoidance) and is **not** charged or counted as a starvation promotion, since it did not jump the queue. Larger reservations drain aged-long requests faster at the cost of short-request latency; the cost to shorts is bounded by roughly the reservation fraction.

### Behavior under saturation (overload)

The aging signal is **absolute wall-clock wait time**. That assumes a long which has waited past `long_max_wait_ms` is being singled out by a short stream — true only when the queue is *not* saturated. Under sustained overload, where the whole queue's residence time (e.g. tens of seconds) dwarfs any practical `long_max_wait_ms`, *every* long is perpetually "aged", so the signal can no longer distinguish a starved long from a generally overloaded system.

This is why the bucket is the single admission channel and the deadline bypass was removed: under overload an unconditional bypass would promote aged longs back-to-back and reintroduce the head-of-line blocking LAPS exists to remove, inflating both mean and (especially) tail TTFT. With the bucket as the sole channel, the contract is:

- Aging **never** flips scheduling into long-first; shorts always retain `1 - reservation` of throughput.
- The cost of aging to short-request latency is **bounded and tunable** at ~`reservation`. It is not free — any anti-starvation that promotes longs takes some throughput from shorts — but a small reservation (e.g. `0.05`–`0.1`) keeps the short-side impact minor while still giving longs a steady reserved drain.
- The trade-off is that there is no hard wall-clock upper bound on an individual long's wait; under overload such a bound is unachievable anyway (the system simply cannot keep up). Long wait is instead bounded by the reserved drain rate. Tune `reservation` up if long tail latency matters more than short TTFT.

## Observability

LAPS behavior is exposed through a periodic, opt-in log line (`laps_config.stats_log_interval_s > 0`): an aggregate `LAPS stats: ...` line with queue sizes, dispatch/skip counters, `long_starvation_promotions`, `long_tokens_charged`, and the current bucket/capacity. A steadily rising `long_starvation_promotions` with the bucket hovering near empty indicates the reservation is fully utilized by aged-long demand; raise `long_token_reservation` if long requests still wait too long. This is useful for benchmark runs without enabling global DEBUG logging, and is disabled by default (`0`).

## Current Scope and Limitations

- The current implementation is enabled only for the **FCFS** scheduler policy.
- The adaptation is **engine-local**; it does not yet rebalance prefill and decode instances across nodes or proxies.
- The implementation targets the vLLM waiting-queue layer and is intended for PD / EPD style serving where prompt length skew is a dominant bottleneck.
- LAPS currently requires `recompute_scheduler_enable=true`; setting only `laps_config.enabled=true` is not sufficient.
- When dynamic batch is selected through `SLO_limits_for_dynamic_batch`, LAPS is not applied.

## Design Rationale

A few deliberate scoping decisions, recorded here so the trade-offs are explicit for reviewers and future maintainers.

- **Why coupled to `recompute_scheduler_enable=true`.** LAPS only swaps the *waiting-queue policy*; it does not change how a request is executed. The Ascend `RecomputeScheduler` is the scheduler that PD / EPD-style reference configurations already enable, so coupling LAPS to it means the feature lands exactly where the length-skew bottleneck actually hurts, with a small, focused intrusion into `RecomputeScheduler` (a per-step `begin_step` hook plus routing waiting-queue pops through the LAPS accounting) instead of a second scheduler subclass to maintain. When `recompute_scheduler_enable=false`, LAPS stays inert and the default scheduler path is untouched.
- **Why FCFS-only for now.** The triple-queue split and the aging promotion order assume a single, total arrival order so that "oldest long" and "shorts ahead of longs" are well defined. Under the PRIORITY policy the head-comparison semantics differ (see the base scheduler's `_select_waiting_queue_for_scheduling`), so LAPS deliberately declines to engage and logs once at config time, keeping the default queue. Extending to PRIORITY is possible later but is intentionally out of scope for the first landing.
- **vllm-ascend first, vLLM core later.** The mechanism is a generic waiting-queue policy with no NPU-specific kernel dependency, so it could eventually live in vLLM core. We land it in vllm-ascend first to validate the policy on real Ascend PD/EPD workloads before proposing a core-level change.
