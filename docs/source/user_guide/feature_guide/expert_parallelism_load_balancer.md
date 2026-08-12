# Expert Parallelism Load Balancer (EPLB)

## Overview

Expert balancing for MoE (Mixture of Experts) models in LLM (Large Language) serving is essential for optimal performance. Dynamically changing experts during inference can negatively impact TTFT (Time To First Token) and TPOT (Time Per Output Token) due to stop-the-world operations. Our solution aims to minimize the negative impacts caused by the operation.

vLLM Ascend provides two EPLB integration paths:

- **Model Runner V2 (MRv2)** uses the upstream vLLM EPLB controller,
  configuration, policy, load window, and rearrangement lifecycle. Ascend adds
  an HCCL weight-transfer backend and the `load_collection_phase` extension.
- **Model Runner V1 (MRv1)** retains the legacy vLLM Ascend dynamic, recording,
  and static EPLB modes.

The two paths use different switches and configuration schemas. Do not mix
MRv1 environment variables or legacy fields with MRv2 EPLB configuration.

## EPLB Effects

- Reduced Latency: Dynamically balances expert loads to minimize TTFT and TPOT by distributing workloads evenly across experts.
- Adaptive Scaling: Automatically adjusts to workload fluctuations while maintaining stable performance.

## Support Scenarios

### Models

EPLB applies only to MoE models that support expert parallelism and whose MoE
quantization method exposes a complete expert-weight movement layout. Support
also depends on the selected model runner and hardware generation.

Legacy MRv1 performance has primarily been verified on DeepSeek-V3.1/R1. The
initial MRv2 model-level validation uses Qwen3-30B-A3B W8A8 with synchronous
EPLB. Validate accuracy and performance with the target model, topology, and
traffic before production deployment.

> [!IMPORTANT]
> Ascend 950 Products does not support using EPLB with quant type "W4A8MXFP4", "W4A16", "W4A16MXFP4".

### Model Runner V2 Weight Formats

The following table describes the MRv2 EPLB code paths. W8A8 has an in-tree
model-level NPU regression. Other enabled formats require model-level
validation on the target hardware before production use.

| Weight format | MRv2 EPLB | Notes |
| --- | --- | --- |
| BF16 / FP16 | Enabled | Uses the unquantized expert weights and biases. |
| W8A8 / W8A8 Dynamic | Enabled | Uses persistent per-expert weight and scale tensors. |
| W4A8 | Enabled | Uses persistent per-expert weight, scale, and scale-bias tensors. |
| W4A4 MXFP | Enabled | Ascend 950 products; keeps native ND expert tensors. |
| W8A8 MXFP | Enabled | Ascend 950 products; keeps native ND expert tensors. |
| W4A16 | Rejected | The expert-weight layout has not completed independent EPLB validation. |
| W4A16 MXFP | Rejected | The expert-weight layout has not completed independent EPLB validation. |
| W4A8 MXFP | Rejected | The expert-weight layout has not completed independent EPLB validation. |

### Model Runner V1 Quantization and Hardware

| QuantType                       | Supported Hardware          |
| ------------------------------- | --------------------------- |
| W8A8 / W8A8-Dynamic             | A2, A3 |
| W4A8 (with fused MC2 enabled)   | A2, A3 |
| MXFP4                           | Ascend 950 Products         |
| MXFP8                           | Ascend 950 Products         |

### Usage Recommendations

EPLB is not recommended in the following scenarios because the load-balancing benefit may not offset its runtime overhead:

- P node workloads with input sequences shorter than `1024` tokens.
- D node workloads where the number of experts per die is `<= 8` (`<= 16` on 950DT), or where the per-die load is below `128` tokens.

> [!WARNING]
> Meeting the above conditions may lead to performance degradation.
> When there are around 8 experts per die, the EPLB benefit may be comparable to its overhead. Benchmark the actual workload and enable EPLB only after confirming a performance gain.

## How to Use EPLB

### Model Runner V2: Upstream Synchronous EPLB

Select MRv2 explicitly when the model or environment does not select it by
default. Enable expert parallelism and upstream EPLB, and set `use_async=false`.
The upstream default is asynchronous, which is not yet supported by MRv2 on
Ascend.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
unset DYNAMIC_EPLB
unset EXPERT_MAP_RECORD

vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config.window_size 50 \
  --eplb-config.step_interval 50 \
  --eplb-config.num_redundant_experts 16 \
  --eplb-config.use_async false \
  --eplb-config.log_balancedness true \
  --eplb-config.log_balancedness_interval 1 \
  --additional-config '{"eplb_config":{"load_collection_phase":"all"}}'
```

MRv2 uses the upstream `EPLBConfig` fields:

| Parameter | Default | Description |
| --- | --- | --- |
| `window_size` | `1000` | Number of recent steps used for expert-load recording. |
| `step_interval` | `3000` | Interval between expert rearrangements. |
| `num_redundant_experts` | `0` | Number of redundant physical experts. |
| `use_async` | `true` | Must be set to `false` on Ascend MRv2. |
| `policy` | `default` | Upstream EPLB placement policy. |
| `log_balancedness` | `false` | Log expert balancedness metrics. |
| `log_balancedness_interval` | `1` | Interval between balancedness log entries. |
| `communicator` | `None` | Do not set this on Ascend; HCCL is selected automatically. |

These fields may also be passed together as JSON through `--eplb-config`.
They must not be placed in `--additional-config` for MRv2.

#### MRv2 Load Collection Phase

`load_collection_phase` is the only MRv2 EPLB field under
`additional_config.eplb_config`. It controls which batch phases contribute to
the upstream load window; it does not disable routing or MoE computation for
non-matching batches.

| Value | Behavior | Typical use |
| --- | --- | --- |
| `all` | Collect load from every batch. This is the default. | General and mixed workloads. |
| `prefill` | Collect only from batches containing at least one prefill request. | Optimize prefill balance and TTFT. |
| `decode` | Collect only from batches containing decode requests and no prefill request. | Optimize decode balance and TPOT. |

Classification is performed once per batch. A batch containing any prefill
request is classified entirely as prefill; otherwise it is decode. A batch
that does not match `load_collection_phase` does not contribute load and does
not advance the EPLB load window. It still participates in the global EPLB
scheduling and communication sequence so that data-parallel ranks remain
synchronized.

For example, to collect only prefill load:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config.use_async false \
  --additional-config '{"eplb_config":{"load_collection_phase":"prefill"}}'
```

> [!IMPORTANT]
> MRv2 currently supports synchronous EPLB only. It rejects legacy
> `dynamic_eplb`, recording/static-map fields, `DYNAMIC_EPLB`, and
> `EXPERT_MAP_RECORD`. It also rejects an explicitly selected communicator.
> The initial validated execution scope is eager mode with the standard
> non-fused MoE communication path. Validate graph, multi-node, speculative
> decoding, and other communication combinations independently before use.

### Model Runner V1: Legacy EPLB

Legacy MRv1 EPLB has three usage modes:

| Mode | Config in `eplb_config` | Env Variable |
| ---- | ----------------------- | ------------ |
| **Dynamic EPLB** | `dynamic_eplb: true` | `DYNAMIC_EPLB=true` |
| **Recording** (generate expert map) | `expert_map_record_path` | `DYNAMIC_EPLB=true` or `EXPERT_MAP_RECORD=true` |
| **Static EPLB** (load pre-recorded map) | `expert_map_path` | none required |

> [!IMPORTANT]
> For Dynamic EPLB and Recording modes, the env variable acts as a safety guard: setting `dynamic_eplb: true` in config alone is not enough — the assertion requires `DYNAMIC_EPLB=true` or `EXPERT_MAP_RECORD=true`. Static EPLB (loading a pre-recorded map via `expert_map_path`) does **not** require an env variable.

#### Dynamic EPLB

We need to add environment variable `export DYNAMIC_EPLB="true"` to enable vLLM-Ascend EPLB. Enable dynamic balancing with auto-tuned parameters. Adjust expert_heat_collection_interval and algorithm_execution_interval based on workload patterns. In the current version, we recommend using the following: policy of SwiftBalanceEplb(2).

| Parameter | Description | Default |
| --- | --- | --- |
| dynamic_eplb | Enable dynamic EPLB. | False |
| expert_heat_collection_interval | Interval for collecting expert heat. | 600 |
| algorithm_execution_interval | Interval for executing the balancing algorithm. | 50 |
| eplb_policy_type | EPLB policy type. | 2 |
| num_redundant_experts | Number of redundant experts. | 0 |
| eplb_heat_collection_stage | Request stage used to collect expert heat. Available values: `all`, `prefill`, and `decode`. | `all` |

```shell
graph TB
   A[start] --> B(collect_heat)
   B --> C(execute_algorithm)
   C --> D(update_layer one by one)
   D --> B
   D --> F[termination upon service termination]
```

```shell
# D node or colocation
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 600,
    "algorithm_execution_interval": 50,
    "eplb_policy_type": 2,
    "num_redundant_experts": 16
    }}'

# P node
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 50,
    "algorithm_execution_interval": 5,
    "eplb_policy_type": 2,
    "num_redundant_experts": 16
    }}'
```

##### EPLB Policy Types

The `eplb_policy_type` parameter selects the balancing algorithm used during dynamic expert redistribution:

| Value | Policy | Description |
|-------|--------|-------------|
| `0` | Random | Randomly swaps experts between ranks. Suitable for basic testing only. |
| `1` | DefaultEplb | Open-source EPLB algorithm. Adds redundant experts to the hottest, packs via balanced assignment with local constraint exchange. |
| `2` | SwiftBalanceEplb | Optimized for low-bandwidth environments. Supports intra-node and inter-node expert redundancy, joint optimization of expert placement. **(Recommended)** |
| `3` | FlashLB | Statistical method using sliding-window mean/variance/covariance of expert loads. Uses FlashTree layered search for optimal replica allocation and `minimize_redeploy` for incremental adjustment. Best for high-frequency load fluctuations. |

##### Selective Expert Heat Collection

The `eplb_heat_collection_stage` option is intended for prefill-decode aggregation scenarios. Prefill requests usually process many tokens in one iteration, while decode requests usually process fewer tokens. As a result, the expert workload distribution can differ between the two stages. Collecting heat from both stages may hide the imbalance of the stage whose latency you want to optimize.

> [!IMPORTANT]
> This section describes the MRv1-only `eplb_heat_collection_stage` field.
> MRv2 uses `load_collection_phase` as described above; the two fields have
> different batch-classification semantics and are not interchangeable.

Use `eplb_heat_collection_stage` to select the stage whose expert heat contributes to EPLB:

| Value | Behavior | Typical use |
| ----- | -------- | ----------- |
| `all` | Collect expert heat from both prefill and decode iterations. | General workloads; this is the default. |
| `prefill` | Collect expert heat only from iterations classified as prefill. | Optimize prefill workload balance and TTFT. |
| `decode` | Collect expert heat only from iterations classified as decode. | Optimize decode workload balance and TPOT. |

Choose the stage according to the actual workload. The following values can be used as initial tuning guidance:

- For workloads whose typical input sequence length is greater than `1024` tokens, start with `prefill`.
- For workloads whose typical input sequence length is less than `1024` tokens but concurrency is greater than `1024`, try `decode` or `all`.
- For other or mixed workloads, benchmark `all`, `prefill`, and `decode` against the target TTFT or TPOT before choosing a setting.

These thresholds are empirical starting points rather than strict requirements. Production traffic distribution, concurrency, model configuration, and hardware topology can all affect the optimal stage.

For example, to collect only prefill heat:

```shell
export DYNAMIC_EPLB="true"

vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 600,
    "algorithm_execution_interval": 50,
    "eplb_policy_type": 2,
    "num_redundant_experts": 16,
    "eplb_heat_collection_stage": "prefill"
  }}'
```

To collect only decode heat, set:

```json
{
  "eplb_config": {
    "dynamic_eplb": true,
    "eplb_heat_collection_stage": "decode"
  }
}
```

> [!NOTE]
> Stage selection applies to dynamic EPLB heat collection. Internally, vLLM-Ascend classifies each forward iteration by comparing its padded scheduled token count with the maximum expected token count of a decode iteration. An iteration above the threshold is treated as prefill; an iteration at or below the threshold is treated as decode. Classification is therefore performed per forward iteration rather than per individual request.

When an iteration does not match the selected stage, its expert load is not accumulated and it does not advance the heat-collection interval. Once heat collection is complete, balancing calculation and layer-by-layer expert weight updates continue normally.

#### Static EPLB

> [!WARNING]
> Static EPLB is scheduled for removal in v0.25.1.

##### Initial Setup (Record Expert Map)

We need to add environment variable `export EXPERT_MAP_RECORD="true"` to record expert map. Generate the initial expert distribution map using expert_map_record_path. This creates a baseline configuration for future deployments.

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "expert_map_record_path": "/path/to/eplb.json",
    "num_redundant_experts": 16,
    "expert_heat_collection_interval": 400,
    "algorithm_execution_interval": 30
  }}'
```

##### Subsequent Deployments (Use Recorded Map)

Load the pre-recorded expert map for consistent performance. This avoids recalculating distributions at runtime.

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{
    "eplb_config": {"expert_map_path": "/path/to/eplb.json"}
  }'
```

## Critical Considerations

1. Parameter Tuning:
   - For MRv2, tune `window_size` and `step_interval` against the target
     workload. For MRv1, tune `expert_heat_collection_interval` and
     `algorithm_execution_interval`.
   - `num_redundant_experts` must make `(num_experts +
     num_redundant_experts)` divisible by the expert-parallel size.

2. Hardware Requirements:
   - Ensure that all NPUs have identical memory capacity and compute capabilities.
   - Network bandwidth must support expert redistribution traffic (≥ 10 Gbps recommended).
   - shm needs to be mounted for container

3. Monitoring & Validation:
   - Track metrics: Search for [Expert Hotness] in the log. We will calculate the peak-to-average ratio of the load for each layer at different ranks, and then find their mean and maximum values. Current means actual peak-to-average ratio, update means estimated peak-to-average ratio after algorithm adjustment.
   - Use vLLM monitor to detect imbalances during runtime.
   - Always verify expert map JSON structure before loading (validate with jq or similar tools).
