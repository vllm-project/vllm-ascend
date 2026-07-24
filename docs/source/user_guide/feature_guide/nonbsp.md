# NonBSP Decode Scheduling

NonBSP is a load-aware scheduler for data-parallel Decode nodes in a
prefill/decode (P/D) disaggregated deployment. It reduces the time that lightly
loaded DP ranks spend waiting for the busiest rank when request lengths are
uneven.

NonBSP does not move requests between DP ranks. Instead, every DP rank shares a
snapshot of its running and admissible waiting requests. The planner estimates
each request's load from its KV-cache block count and tells each rank which
local requests to admit, keep running, or pause. A request paused by NonBSP
keeps its allocated KV cache and can be resumed by the same rank later.

## When to use it

Consider enabling NonBSP when all of the following are true:

- the deployment uses P/D disaggregation
- the Decode instance uses internal data parallelism on a single node
- request lengths are skewed enough to create visible DP bubbles
- Decode throughput is more important than preserving strict local request
  execution order

Keep it disabled for a uniformly sized workload, a single-DP-rank Decode
instance, or a multi-node Decode DP instance.

## Requirements

NonBSP has the following startup requirements:

- it must be enabled only on a P/D-disaggregated Decode node with
  `kv_role="kv_consumer"`
- `data_parallel_size` must be greater than `1`
- all DP ranks of the Decode instance must be on one node
- the vLLM V1 engine and the normal scheduler must be used

Configure NonBSP only on Decode nodes. Do not add `nonbsp_config` with
`enabled=true` to Prefill nodes.

NonBSP cannot be combined with:

- `scheduler_config.enable_balance_scheduling`
- `scheduler_config.recompute_scheduler_enable`
- `scheduler_config.profiling_chunk_config.enabled`

The service rejects these unsupported combinations during startup.

## Configuration

Add `nonbsp_config` under `scheduler_config` in the Decode node's
`--additional-config`. The legacy top-level `NONBSP_*` keys do not enable this
feature.

Add the following arguments to an existing, working single-node Decode command.
This fragment enables dynamic mode, which is recommended for normal workloads:

```bash
--data-parallel-size 2 \
  --additional-config '{
    "scheduler_config": {
      "nonbsp_config": {
        "enabled": true,
        "mode": "dynamic"
      }
    }
  }'
```

Keep the existing `--kv-transfer-config` for the Decode node and verify that it
sets `kv_role` to `kv_consumer`.

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enables NonBSP and selects `NonBSPScheduler` on the Decode node. |
| `enable_diagnostics` | bool | `false` | Prints NonBSP configuration, per-rank workload snapshots, scheduling modifications, and scheduler summaries. |
| `mode` | str | `"static"` | Selects `"static"` or `"dynamic"` activation. Use `"static"` only for troubleshooting; use `"dynamic"` for normal workloads. |
| `start_step` | int | `250` | First engine step at which balancing can run. Must be greater than or equal to `0`. |
| `end_step` | int | `-1` | Exclusive final balancing step. `-1` means no final step. Otherwise, it must be greater than `start_step`. |
| `bubble_threshold` | float | `5.0` | Minimum imbalance that triggers planner modifications. Values greater than `1` are interpreted as an absolute KV-block difference between the maximum and average rank load. Values at or below `1` are interpreted as the normalized ratio `(maximum - average) / maximum`. |
| `long_req_block_threshold` | int | `700` | In dynamic mode, a newly added request with more than this number of KV-cache blocks activates balancing. |
| `dynamic_max_step` | int | `256` | In dynamic mode, disables balancing after this many active steps without another newly added long request. |

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
    "nonbsp_config": {
      "enabled": true,
      "mode": "static"
    }
  }
}
```

Static mode is intended only for troubleshooting and short validation runs. It
keeps load balancing active throughout the configured interval, which makes
NonBSP behavior easier to observe. Do not use static mode for normal workloads.
Set `start_step` to `0` during a short diagnostic run so that the run actually
exercises NonBSP.

### Dynamic mode

Dynamic mode remains inactive until a newly added request exceeds
`long_req_block_threshold`. It then evaluates load until
`dynamic_max_step` consecutive active steps pass without another newly added
long request:

```json
{
  "scheduler_config": {
    "nonbsp_config": {
      "enabled": true,
      "mode": "dynamic"
    }
  }
}
```

Dynamic mode is recommended for normal workloads because it activates load
balancing only when a long request indicates that DP imbalance is likely. The
`start_step` and `end_step` interval still applies in dynamic mode.

## How scheduling works

For each active balancing step, NonBSP:

1. prepares the local waiting requests whose remote KV cache is ready
2. all-gathers the KV-block counts of running and admissible waiting requests
   across the Decode DP group
3. compares each rank's estimated running load with the DP-wide load
   distribution
4. generates a per-rank plan that can pause local running requests, admit
   selected local waiting requests, or freeze new admission for that step
5. runs the normal scheduling and model-execution path using that plan

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
3. Compare throughput and latency with NonBSP disabled.
4. Turn diagnostics off for performance measurement.
5. Switch to `mode="dynamic"` for workload testing. Tune `bubble_threshold`,
   `long_req_block_threshold`, and `dynamic_max_step` from the observed request
   block-count distribution.

A very small `bubble_threshold` can cause frequent request pausing and
admission changes. A very large value makes NonBSP rarely modify the normal
scheduler plan.

## Diagnostics

Set `enable_diagnostics=true` only while validating or troubleshooting:

```json
{
  "scheduler_config": {
    "nonbsp_config": {
      "enabled": true,
      "enable_diagnostics": true
    }
  }
}
```

Rank 0 prints `[nonbsp]` workload snapshots and the generated `Out`, `In`, and
`Freeze` modifications. Scheduler summaries and engine-step diagnostics are
also emitted. These logs can be verbose and should normally remain disabled in
throughput benchmarks.

## Limitations

- Only single-node, internally data-parallel Decode instances are supported.
- NonBSP does not route or migrate a request to another DP rank.
- The planner estimates work from KV-cache blocks and uses a fixed throughput
  heuristic. Model-specific costs, speculative-draft work, and operator-level
  variation may require workload-specific tuning.
- Combinations with speculative decoding or async scheduling should be
  validated with the exact model, vLLM version, and vLLM Ascend version used in
  production.

For the complete `additional_config` schema, see
[Additional Configuration](../configuration/additional_config.md).
