# Expert Parallelism Load Balancer (EPLB)

## Feature Introduction

Expert balancing for MoE (Mixture of Experts) models in LLM (Large Language Model) serving is essential for optimal performance.
Dynamically rearranging experts during inference can negatively impact TTFT (Time To First Token) and TPOT (Time Per Output Token), as it requires global synchronization pauses that stall all in-flight requests.
Our solution aims to minimize the impact of these pauses. EPLB feature provides the following benefits:

- Reduced Latency: Dynamically balances expert loads to minimize TTFT and TPOT by distributing workloads evenly across experts.
- Adaptive Scaling: Automatically adjusts to workload fluctuations while maintaining stable performance.

### Usage Scenarios

EPLB is not recommended in the following scenarios because the load-balancing benefit may not offset its runtime overhead:

- P node workloads with input sequences shorter than `1024` tokens.
- D node workloads where the number of experts per die is `> 8` (`> 16` on 950DT Products), or where the per-die load is below `128` tokens.

!!!WARNING

    Meeting the above conditions may lead to performance degradation.
    When there are around 8 experts per die, the EPLB benefit may be comparable to its overhead. Benchmark the actual workload and enable EPLB only after confirming a performance gain.

vLLM Ascend uses the upstream vLLM EPLB controller, configuration, load window,
asynchronous worker, and rearrangement lifecycle. Ascend adds the STAIR policy
(the default policy on Ascend), the HIXL and Gloo movement adapters, and the
`load_collection_phase` extension.

The legacy vLLM Ascend dynamic, recording, and static EPLB modes belonged to
Model Runner V1 and have been removed.

### Constraints and Limitations

#### Models

EPLB applies only to MoE models that support expert parallelism and whose MoE
quantization method exposes a complete expert-weight movement layout. Support
also depends on the selected model runner and hardware generation.

The model-level validation uses Qwen3-30B-A3B W8A8 with asynchronous EPLB. Validate accuracy and performance with the target model, topology, and
traffic before production deployment.

!!!IMPORTANT

    A2 does not support redundant experts. On A2, `num_redundant_experts` must be set to `0` in both runners. Examples in this page that use a non-zero `num_redundant_experts` apply only to hardware that supports redundant experts, such as A3 or 950PR&950DT Products.

#### Weight Formats

The following table summarizes the code paths supported by MRv2 EPLB. W8A8 is covered by an in-repository model-level regression test that runs on Ascend NPUs (`tests/e2e/pull_request/four_card/test_qwen3_mrv2_eplb.py`). Other enabled formats have not undergone the same level of validation and must be validated with the target model and hardware configuration before production deployment.

| Weight format | MRv2 EPLB | Notes |
| --- | --- | --- |
| BF16 / FP16 | Enabled | Uses the unquantized expert weights and biases. |
| W8A8 / W8A8 Dynamic | Enabled | Uses persistent per-expert weight and scale tensors. |
| W4A8 | Enabled | Uses persistent per-expert weight, scale, and scale-bias tensors. |
| W4A4 MXFP | Enabled | 950PR&950DT Products; keeps native ND expert tensors. |
| W8A8 MXFP | Enabled | 950PR&950DT Products; keeps native ND expert tensors. |
| W4A8 MXFP | Rejected | The expert-weight layout does not support EPLB movement; startup fails with `EPLB is not supported ...`. |
| W4A16 | Rejected | The expert-weight layout does not support EPLB movement; startup fails with `EPLB is not supported ...`. |
| W4A16 MXFP | Rejected | The expert-weight layout does not support EPLB movement; startup fails with `EPLB is not supported ...`. |

"Rejected" means startup fails before serving: the quantization method does not
declare EPLB support (`supports_eplb = False`). This rejection is independent
of hardware generation.

## Feature Usage

### Asynchronous EPLB

On Ascend, the V2 runner handles every request because Model Runner V1 has been
removed, so no environment variable is needed to select it.
Enable expert parallelism and upstream EPLB. Ascend selects the STAIR policy
by default and automatically uses the HIXL communicator when it is available
on every EPLB rank, falling back to Gloo otherwise. Movement is asynchronous
only.

The STAIR defaults do not require tuning; see
[Tuning Suggestions](#tuning-suggestions) for the advanced STAIR parameters.
A minimal production configuration only needs the redundant-expert budget:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config.num_redundant_experts 16
```

For short benchmark or accuracy-validation runs where you want quicker load
feedback instead of production stability, shrink the load window and the
rearrangement interval together (for example to `200`). These values
increase rearrangement frequency; do not use them as production defaults:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config.window_size 200 \
  --eplb-config.step_interval 200 \
  --eplb-config.num_redundant_experts 16
```

MRv2 uses the upstream `EPLBConfig` fields:

| Parameter | Required | Value range | Default | Description |
| --- | --- | --- | --- | --- |
| `window_size` | No | positive integer | `1000` | Number of recent steps used for expert-load recording. |
| `step_interval` | No | positive integer | `3000` | Interval between expert rearrangements. If it exceeds `window_size`, only the last `window_size` steps of metrics are used. |
| `num_redundant_experts` | No | non-negative integer | `0` | Number of redundant physical experts. Must be `0` on A2. `(num_experts + num_redundant_experts)` must be divisible by the expert-parallel size. |
| `use_async` | No | `true` / `false` | `true` | Ascend MRv2 always runs asynchronously. `false` is normalized to `true` with a warning. |
| `policy` | No | `stair`, `default` | `stair` on Ascend | Select `stair` for the Ascend policy or `default` for upstream-policy comparison experiments. |
| `log_balancedness` | No | `true` / `false` | `false` | Log expert balancedness metrics. Incurs extra communication; recommended to leave disabled and only enable it temporarily when debugging. Do not set it in production configuration. |
| `log_balancedness_interval` | No | positive integer | `1` | Interval between balancedness log entries. Only relevant when `log_balancedness` is explicitly enabled for debugging. |
| `communicator` | No | `hixl`, `torch_gloo`, unset | Auto (`None`) on Ascend | Leave unset to use HIXL when available and Gloo otherwise; set `hixl` or `torch_gloo` to select a backend explicitly. |

Every field is optional: the engine starts with the defaults. Why the examples
above deviate from them, and when to do the same:

- `num_redundant_experts`: the default `0` means no redundancy, so EPLB can
  only shuffle whole experts between ranks. For production, budget a small
  hot-expert reserve that keeps `(num_experts + num_redundant_experts)`
  divisible by the expert-parallel size — for example `16` for
  Qwen3-30B-A3B with EP 16, which gives every rank one spare expert slot.
- `window_size` / `step_interval`: keep the `1000` / `3000` defaults in
  production. The `200` / `200` values in the validation example exist only
  to shorten the feedback loop of short benchmark or accuracy runs; they
  increase rearrangement frequency and trade stability for faster response.

These fields may also be passed together as JSON through `--eplb-config`.
They must not be placed in `--additional-config` for MRv2: the only MRv2 EPLB
fields under `additional_config.eplb_config` are `load_collection_phase` and
`stair_config`; any other field is rejected at startup.

Ascend extends the upstream `policy` field without adding a second selector.
For example, use `--eplb-config.policy default` to run the upstream policy;
omit it or set it to `stair` to run STAIR.

#### MRv2 Communicator Selection

Automatic selection requires the same HIXL binding class on every EPLB rank:
each rank resolves either the official `hixl` Python package distributed with
CANN HIXL or vllm_ascend's ctypes binding, which drives the local toolkit
libraries directly (`libcann_hixl.so`, for CANN distributions that ship the
library without the package). The ranks reach a group-wide consensus at
startup; if any rank has no usable HIXL binding, or the binding classes
differ between ranks, the whole group falls back to `torch_gloo` CPU staging
and the decision is logged.

STAIR leaves both per-rank and cross-node migration limits unrestricted by
default (`-1`) so that HIXL can use the available bandwidth. When the
communicator falls back to Gloo automatically, Ascend clamps only the limits
you have not set explicitly to `1` to avoid excessive CPU-staged transfers,
and logs the clamp; explicitly configured values are kept (for example
`cross_node_transfer_limit: 0` still forbids cross-node migrations) with a
warning. Explicitly setting `torch_gloo` or `hixl` also keeps your configured
limits untouched.

HIXL requires the CANN HIXL runtime and working device connectivity on every
EPLB rank. HIXL initialization or runtime failures fail fast instead of
falling back mid-flight, because ranks may disagree on transfer progress.

#### MRv2 Load Collection Phase

`load_collection_phase` is an MRv2 EPLB field under
`additional_config.eplb_config`. It controls which batch phases contribute to
the load window when the STAIR policy is active; it has no effect with
`--eplb-config.policy default`, and it does not disable routing or MoE
computation for non-matching batches.

| Value | Behavior | Typical use |
| --- | --- | --- |
| `all` | Collect load from every batch. This is the default. | General and mixed workloads. |
| `prefill` | Collect only from batches containing at least one prefill request. | Optimize prefill balance and TTFT. |
| `decode` | Collect only from batches containing decode requests and no prefill request. | Optimize decode balance and TPOT. |

Additional rules:

- A value other than `all` requires `--enable-eplb`.
- Classification is performed once per batch. A batch containing any prefill
  request is classified entirely as prefill; otherwise it is decode. On
  disaggregated decode nodes, requests that recompute a prompt tail segment
  are not counted as prefill, so such batches are classified as decode.
- A batch that does not match `load_collection_phase` contributes zero load
  but still advances the shared EPLB window. This keeps load-window slots,
  scheduling, and communication aligned across data-parallel ranks.
- If no rank has recorded fresh load since the last rearrangement (for
  example, `prefill` is selected but traffic is pure decode), the next
  rearrangement is skipped instead of running on an empty window.

For example, to collect only prefill load:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-expert-parallel \
  --enable-eplb \
  --additional-config '{"eplb_config":{"load_collection_phase":"prefill"}}'
```

#### MRv2 Constraints

MRv2 supports asynchronous EPLB only and normalizes `use_async=false` to
asynchronous movement. It rejects:

- elastic expert parallelism,
- legacy `dynamic_eplb`, recording/static-map fields, and the `DYNAMIC_EPLB`
  and `EXPERT_MAP_RECORD` environment variables, all of which belonged to the
  removed Model Runner V1 EPLB implementation,
- communicators other than Gloo and HIXL,
- weight offload and prefetch: enabling EPLB with `cpu_offload_gb > 0` or
  `offload_group_size > 0` raises a configuration error.

Validate the target model, topology, graph mode, and traffic independently
before production use.

## Performance Ceiling Analysis (`enable_force_eplb`)

`enable_force_eplb` is an analysis-only switch that forces a deterministic
round-robin expert routing instead of the real MoE router output. It gives every
expert-parallel rank a synthetic, perfectly balanced token distribution, so you can
measure the performance upper bound (ceiling) of the expert-parallel / EPLB path
under ideal load balance.

!!!WARNING

    `enable_force_eplb` replaces the real routing result, so model results and
    **accuracy are invalid**. It is **not for production** and must not be used for
    correctness or accuracy validation.

- Enable it through `additional_config`: `--additional-config '{"enable_force_eplb": true}'`.
- It requires expert parallelism and a MoE model; `num_logical_experts` must be divisible by the expert-parallel size.
- It cannot be combined with `eplb_config.dynamic_eplb`.
- See [`enable_force_eplb`](../configuration/additional_config.md) for the full parameter reference.

## Tuning Suggestions

The default STAIR policy already covers general workloads; change the
advanced parameters below only after measuring a specific need. They live
under `additional_config.eplb_config.stair_config`. Numeric fields must not
be booleans, and each field is range checked at startup.

| Parameter | Default | Range | Description |
| --- | --- | --- | --- |
| `load_window_bins` | `64` | 2–256 | Maximum chronological bins used to compress the upstream EPLB load window. |
| `load_risk_quantile` | `0.75` | (0.5, 1.0) exclusive | One-sided standard-normal quantile converted to a z-score for mean-plus-deviation expert and rank risk. |
| `relative_balance_threshold` | `0.95` | (0, 1] | Rebalance when current balance divided by the last committed balance is at or below this value. |
| `absolute_balance_threshold` | `0.90` | (0, 1] | Rebalance when current balance is at or below this value. |
| `rank_transfer_limit` | `-1` | -1 or positive | Maximum outgoing and incoming expert transfers for each rank in one layer plan. `-1` removes the limit. |
| `cross_node_transfer_limit` | `-1` | -1 or non-negative | Maximum outgoing and incoming cross-node expert transfers for each node in one layer plan. `-1` removes the limit; `0` disables cross-node transfers. |
| `replica_search_num_stages` | `4` | 1–8 | Number of risk-ordered expert groups handled by the FlashTree-style replica search. |
| `replica_search_radius` | `8` | 0–32 | Maximum distance from the greedy extra-replica budget explored at each search stage. |
| `replica_search_beam_size` | `64` | 1–128 | Maximum unique replica-count candidates kept after each search stage. |
| `placement_search_backtrack_limit` | `32` | 0–64 | Maximum feasible-branch reversals while constrained LPT places one candidate. `0` disables them. |

Example:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-expert-parallel \
  --enable-eplb \
  --additional-config '{"eplb_config":{"stair_config":{"rank_transfer_limit":2,"cross_node_transfer_limit":0}}}'
```

Balance is the reciprocal of the mean max-to-average rank-load ratio; STAIR
rebalances a layer only when the hysteresis gates above fire, so lowering
`step_interval` alone does not force continuous movement.

## Critical Considerations

1. Parameter Tuning:
    - Tune `window_size` and `step_interval` against the target workload.
    - `num_redundant_experts` must make `(num_experts +
      num_redundant_experts)` divisible by the expert-parallel size.

2. Hardware Requirements:
    - Ensure that all NPUs have identical memory capacity and compute capabilities.
    - Network bandwidth must support expert redistribution traffic (≥ 10 Gbps recommended).
    - shm needs to be mounted for container

3. Monitoring & Validation:
    - Temporarily enable `--eplb-config.log_balancedness true` to log
      `EPLB step: ... balancedness=...` per window sample, and check
      `Ascend async EPLB cycle committed` lines for the per-cycle movement
      summary (`mean ... -> ...`, changed layers, transfer counts).
    - Use vLLM monitor to detect imbalances during runtime.
    - Always verify expert map JSON structure before loading (validate with jq or similar tools).
