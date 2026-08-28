# DyntraLB: Dynamic Intra-Decoder Data-Parallel Load Balancing

DyntraLB is a load-aware scheduler for data-parallel Decode nodes in a
prefill/decode (P/D) disaggregated deployment. It reduces the time that lightly
loaded DP ranks spend waiting for the busiest rank when request lengths are
uneven.

DyntraLB does not move requests between DP ranks. Instead, every DP rank shares a
snapshot of its running and admissible waiting requests. The planner estimates
each request's load from its KV-cache block count and tells each rank which
local requests to admit, keep running, or pause. A request paused by DyntraLB
keeps its allocated KV cache and can be resumed by the same rank later.

## When to use it

Consider enabling DyntraLB when all of the following are true:

- the deployment uses P/D disaggregation
- the Decode instance uses internal data parallelism on a single node
- request lengths are skewed enough to create visible DP bubbles
- Decode throughput is more important than preserving strict local request
  execution order

Keep it disabled for a uniformly sized workload, a single-DP-rank Decode
instance, or a multi-node Decode DP instance.

## Requirements

DyntraLB has the following startup requirements:

- it must be enabled only on a P/D-disaggregated Decode node with
  `kv_role="kv_consumer"`
- `data_parallel_size` must be greater than `1`
- all DP ranks of the Decode instance must be on one node

Configure DyntraLB only on Decode nodes. Do not add `dyntra_lb_config` with
`enabled=true` to Prefill nodes.

DyntraLB cannot be combined with:

- `scheduler_config.enable_balance_scheduling`
- `scheduler_config.profiling_chunk_config.enabled`

The service rejects these unsupported combinations during startup.

## Configuration

Add `dyntra_lb_config` under `scheduler_config` in the Decode node's
`--additional-config`.

DyntraLB and recompute scheduling remain independent sibling configurations.
When both `dyntra_lb_config.enabled` and `recompute_scheduler_enable` are `true`,
the Decode node uses the combined DyntraLB recompute scheduler.

Add the following arguments to an existing, working single-node Decode command.
This fragment enables dynamic mode by default, which is recommended for production scenario:

```bash
  --additional-config '{
    "scheduler_config": {
      "dyntra_lb_config": {
        "enabled": true
      }
    }
  }'
```

Keep the existing `--kv-transfer-config` for the Decode node and verify that it
sets `kv_role` to `kv_consumer`.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enables DyntraLB and selects `DyntraLBScheduler` on the Decode node. |
| `mode` | str | `"dynamic"` | Selects `"static"` or `"dynamic"` activation. Use `"static"` only for troubleshooting; use `"dynamic"` for production scenarios. |
| `start_step` | int | `250` | First completed engine-step snapshot from which DyntraLB can generate a plan for the next step. Must be greater than or equal to `0`. |
| `end_step` | int | `-1` | Exclusive final snapshot step. `-1` means no final step. Otherwise, it must be greater than `start_step`. |
| `bubble_threshold` | float | `5.0` | Minimum imbalance that triggers planner modifications. Values greater than or equal to `1` are interpreted as an absolute KV-block difference between the maximum and average rank load. Values below `1` are interpreted as the normalized ratio `(maximum - average) / maximum`. Therefore, `1` means one KV-cache block, not 100%. |
| `long_req_block_threshold` | int | `700` | In dynamic mode, a newly added request with more than this number of KV-cache blocks activates balancing. |
| `dynamic_max_step` | int | `256` | In dynamic mode, disables balancing after this many active steps without another newly added long request. |
| `enable_diagnostics` | bool | `false` | Enable verbose logs for feature validation and debugging only. It is disabled by default and should remain disabled in production.  |

`long_req_block_threshold` and an absolute `bubble_threshold` are expressed in
KV-cache blocks, not tokens. Their token equivalents therefore depend on the
configured block size.

## Static and dynamic modes

### Static mode

Static mode evaluates the DP load on every coordinated engine step in the
configured interval:

```json
{
  "scheduler_config": {
    "dyntra_lb_config": {
      "enabled": true,
      "mode": "static"
    }
  }
}
```

Static mode is intended only for troubleshooting and short validation runs. It
keeps load balancing active throughout the configured interval, which makes
DyntraLB behavior easier to observe. Do not use static mode for production.
Set `start_step` to `0` during a short diagnostic run so that the run actually
exercises DyntraLB.

### Dynamic mode

Dynamic mode remains inactive until a newly added request exceeds
`long_req_block_threshold`. It then evaluates load until
`dynamic_max_step` consecutive active steps pass without another newly added
long request:

```json
{
  "scheduler_config": {
    "dyntra_lb_config": {
      "enabled": true,
      "mode": "dynamic"
    }
  }
}
```

Dynamic mode is recommended for production because it activates load
balancing only when a long request indicates that DP imbalance is likely. The
`start_step` and `end_step` interval still applies in dynamic mode.

## How scheduling works

DyntraLB pipelines each load-balancing decision across two engine steps:

1. the current step runs scheduling and model execution
2. all DP ranks synchronize their wave/idle state
3. each active rank prepares the local waiting requests whose remote KV cache
   is ready
4. the ranks all-gather the KV-block counts of running and admissible waiting
   requests across the Decode DP group
5. the planner generates a per-rank plan that can pause local running requests,
   admit selected local waiting requests, or freeze new admission
6. the next engine step applies that plan before normal scheduling and model
   execution

The first engine step of a new wave has no plan from a preceding active step and
therefore follows the normal scheduler path. In dynamic mode, detecting a new
long request after that step immediately generates a plan for the following
step. `start_step` and `end_step` select the completed-step snapshots from which
plans may be generated; each generated plan is consumed by the next step.

The planner uses request block counts as a lightweight load estimate. The
estimate is intended to reduce DP bubbles; it is not a latency or throughput
guarantee for every model and traffic distribution.

## Tuning and validation

Start with a controlled A/B comparison using the same model, prompts, sampling
parameters, concurrency, and P/D topology.

1. Set `start_step=0`, `mode="static"`, and
   `enable_diagnostics=true` for a short troubleshooting or functional run.
2. Confirm that every Decode rank enters the same workload-collection steps
   and that requests complete without persistent KV-waiting states.
3. Compare throughput and latency with DyntraLB disabled.
4. Turn diagnostics off for performance measurement.
5. Switch to `mode="dynamic"` for workload testing. Tune `bubble_threshold`,
   `long_req_block_threshold`, and `dynamic_max_step` from the observed request
   block-count distribution.

A very small `bubble_threshold` can cause frequent request pausing and
admission changes. A very large value makes DyntraLB rarely modify the normal
scheduler plan.

## Diagnostics

Set `enable_diagnostics=true` only while validating or troubleshooting:

```json
{
  "scheduler_config": {
    "dyntra_lb_config": {
      "enabled": true,
      "enable_diagnostics": true
    }
  }
}
```

Rank 0 prints `[dyntra_lb]` workload snapshots and the generated `Out`, `In`, and
`Freeze` modifications. Scheduler summaries and engine-step diagnostics are
also emitted. These logs can be verbose and should normally remain disabled in
throughput benchmarks.

## Limitations

- Only single-node, internally data-parallel Decode instances are supported.
- DyntraLB does not route or migrate a request to another DP rank.
- The planner estimates work from KV-cache blocks and uses a fixed throughput
  heuristic. Model-specific costs, speculative-draft work, and operator-level
  variation may require workload-specific tuning.
- Combinations with speculative decoding or async scheduling should be
  validated with the exact model, vLLM version, and vLLM Ascend version used in
  production.

For the complete `additional_config` schema, see
[Additional Configuration](../configuration/additional_config.md).
