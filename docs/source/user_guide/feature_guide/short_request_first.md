# ShortRequestFirst Prefill Scheduling

ShortRequestFirst reduces head-of-line blocking during prefill by letting short prompts run before long prompts. It is designed for mixed prompt-length traffic where a few long requests can otherwise delay many short ones.

## When to use it

Enable ShortRequestFirst when:

- request lengths are highly skewed
- short-request TTFT matters more than strict FCFS ordering
- the prefill node already runs with the recompute scheduler enabled

Keep it disabled if the workload is mostly uniform, or if FCFS ordering is more important than short-request latency.

## Configuration

Add `short_request_first_config` to the prefill (P) node's `additional_config` in a PD-disaggregated deployment. Because ShortRequestFirst is wired through the recompute scheduler, keep `recompute_scheduler_enable=true` in the same P-node config:

```json
{
  "recompute_scheduler_enable": true,
  "short_request_first_config": {
    "enabled": true,
    "threshold": 256,
    "long_max_wait_ms": 2000
  }
}
```

### Fields

- `enabled` (bool, default `false`)
  Turns ShortRequestFirst on or off.
- `threshold` (int, default `256`)
  Requests with `num_prompt_tokens <= threshold` are treated as short requests.
- `long_max_wait_ms` (float, default `0`)
  Maximum time a long request may wait behind short requests before it can be promoted ahead of them.
  `0` disables long-request promotion and keeps strict short-request priority.

## Threshold tuning

If `threshold` is too low, most requests fall into the `long` lane and the split largely stops helping. If it is too high, most requests fall into the `short` lane and the split also stops helping.

In practice, `threshold` should sit near the boundary between the short-request main cluster and the long-request tail. The goal is not to pick a fixed midpoint of all request lengths, and not to move the threshold close to the longest requests.

If you do not yet have a refined traffic model, start with `P70-P85` of prompt length as an engineering baseline and then refine it using the real request-length histogram. For a clear bimodal workload, the threshold usually belongs near the upper edge of the short-request cluster rather than near the longest prompts.

In short: the right threshold is usually the valley between short-request and long-request populations. When detailed distribution modeling is unavailable, `P70-P85` is a practical starting range.

## long_max_wait_ms tuning

`long_max_wait_ms` is a fairness gate, not a throughput-optimization knob. It controls when a long request stops waiting under normal short-first behavior and becomes eligible for anti-starvation promotion.

Tune `threshold` first. If degradation warnings appear frequently, first suspect that `threshold` is too small for the current traffic mix, then consider increasing `long_max_wait_ms`.

A practical tuning flow is:

- start with `long_max_wait_ms = 0` to establish a strict short-first baseline
- measure the waiting-time distribution of long requests under that baseline
- define `W_normal` as the normal upper tail of long-request waiting, typically `P90` or `P95`
- define `W_slo` as the maximum additional queueing delay that your service can tolerate for long requests
- set `long_max_wait_ms` somewhere inside `[W_normal, W_slo]`

Within that interval, bias the value toward `W_slo` if short-request TTFT matters more, or toward `W_normal` if long-request fairness matters more. If `W_normal` is already close to `W_slo`, the main issue is more likely the threshold choice or overall system capacity than the wait parameter itself.

Do not size `long_max_wait_ms` from average short-request TTFT. It is usually better anchored to the tail of long-request waiting under strict short-first behavior.

In short: `long_max_wait_ms` should be derived from the long-request waiting tail under strict short-first scheduling and bounded by the long-request queueing delay your service can accept. If aged-long promotions happen repeatedly, increase `threshold` first and only then increase `long_max_wait_ms`.

## Scheduling behavior

The waiting queue is split into three lanes:

- `immediate`
  Recovery-style requests such as preempted requests or requests with already computed tokens.
- `short`
  Requests whose prompt length is at or below `threshold`.
- `long`
  Requests whose prompt length is above `threshold`.

Dispatch priority is:

`immediate > aged-long > short > long`

That means:

- immediate requests always win
- short requests run before ordinary long requests
- if the oldest long request waits past `long_max_wait_ms`, it can jump ahead of waiting short requests

## Degradation warning

If long requests are promoted ahead of waiting short requests for three consecutive dispatches, vLLM Ascend emits a warning. This indicates the feature is drifting toward long-request priority, usually because the threshold is too small for the current traffic mix.

When this warning appears:

- increase `short_request_first_config.threshold`
- or disable `short_request_first_config.enabled`

ShortRequestFirst also emits an aggregate stats log every 5 seconds so queue behavior is visible without adding extra configuration.

## Relationship with recompute scheduler

ShortRequestFirst only changes the waiting-queue policy and is wired into the recompute scheduler. With `recompute_scheduler_enable=false`, the normal scheduler path is used and ShortRequestFirst is not activated.

## Minimal examples

P-node `additional_config` example:

```json
{
  "recompute_scheduler_enable": true,
  "short_request_first_config": {
    "enabled": true,
    "threshold": 256,
    "long_max_wait_ms": 2000
  }
}
```

Disable it explicitly:

```bash
vllm serve <model> \
  --additional-config '{"short_request_first_config": {"enabled": false}}'
```
