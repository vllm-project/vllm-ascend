# Expert Parallelism Load Balancer (EPLB)

## Overview

Expert balancing for MoE (Mixture of Experts) models in LLM (Large Language) serving is essential for optimal performance. Dynamically changing experts during inference can negatively impact TTFT (Time To First Token) and TPOT (Time Per Output Token) due to stop-the-world operations. Our solution aims to minimize the negative impacts caused by the operation.

## EPLB Effects

- Reduced Latency: Dynamically balances expert loads to minimize TTFT and TPOT by distributing workloads evenly across experts.
- Adaptive Scaling: Automatically adjusts to workload fluctuations while maintaining stable performance.

## Support Scenarios

### Models

All MOE models supported by vLLM-Ascend.
But we have only verified the performance on deepseek-v3.1/r1 models.

### MOE QuantType

| QuantType                       | Supported Hardware          |
| ------------------------------- | --------------------------- |
| W8A8 / W8A8-Dynamic             | A2, A3 |
| W4A8 (with fused MC2 enabled)   | A2, A3 |
| MXFP4                           | Ascend 950 Products         |
| MXFP8                           | Ascend 950 Products         |

## How to Use EPLB

EPLB has three usage modes:

| Mode | Config in `eplb_config` | Env Variable |
| ---- | ----------------------- | ------------ |
| **Dynamic EPLB** | `dynamic_eplb: true` | `DYNAMIC_EPLB=true` |
| **Recording** (generate expert map) | `expert_map_record_path` | `DYNAMIC_EPLB=true` or `EXPERT_MAP_RECORD=true` |
| **Static EPLB** (load pre-recorded map) | `expert_map_path` | none required |

> [!IMPORTANT]
> For Dynamic EPLB and Recording modes, the env variable acts as a safety guard: setting `dynamic_eplb: true` in config alone is not enough — the assertion requires `DYNAMIC_EPLB=true` or `EXPERT_MAP_RECORD=true`. Static EPLB (loading a pre-recorded map via `expert_map_path`) does **not** require an env variable.

### Dynamic EPLB

We need to add environment variable `export DYNAMIC_EPLB="true"` to enable vLLM-Ascend EPLB. Enable dynamic balancing with auto-tuned parameters. Adjust expert_heat_collection_interval and algorithm_execution_interval based on workload patterns. In the current version, we recommend using the following: policy of swift balancer(2).

| Parameter | Description | Default |
| --- | --- | --- |
| dynamic_eplb | Enable dynamic EPLB. | False |
| expert_heat_collection_interval | Interval for collecting expert heat. | 600 |
| algorithm_execution_interval | Interval for executing the balancing algorithm. | 50 |
| eplb_policy_type | EPLB policy type. | 2 |
| num_redundant_experts | Number of redundant experts. | 0 |

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

#### EPLB Policy Types

The `eplb_policy_type` parameter selects the balancing algorithm used during dynamic expert redistribution:

| Value | Policy | Description |
|-------|--------|-------------|
| `0` | Random | Randomly swaps experts between ranks. Suitable for basic testing only. |
| `1` | DefaultEplb | Open-source EPLB algorithm. Adds redundant experts to the hottest, packs via balanced assignment with local constraint exchange. |
| `2` | SwiftBalanceEplb | Optimized for low-bandwidth environments. Supports intra-node and inter-node expert redundancy, joint optimization of expert placement. **(Recommended)** |
| `3` | FlashLB | Statistical method using sliding-window mean/variance/covariance of expert loads. Uses FlashTree layered search for optimal replica allocation and `minimize_redeploy` for incremental adjustment. Best for high-frequency load fluctuations. |

### Static EPLB

#### Initial Setup (Record Expert Map)

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

#### Subsequent Deployments (Use Recorded Map)

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
   - expert_heat_collection_interval: Higher values (e.g., 600+) for stable workloads; lower values (e.g., 50-100) for fluctuating traffic.
   - algorithm_execution_interval: Should be ≥ 50 to avoid premature balancing during startup.
   - num_redundant_experts: Must match (num_experts + num_redundant_experts) is divisible by expert-parallel size.

2. Hardware Requirements:
   - Ensure that all NPUs have identical memory capacity and compute capabilities.
   - Network bandwidth must support expert redistribution traffic (≥ 10 Gbps recommended).
   - shm needs to be mounted for container

3. Monitoring & Validation:
   - Track metrics: Search for [Expert Hotness] in log, we will calculate the peak-to-average ratio of the load for each layer at different ranks, and then find their mean and maximum values. Current means actual peak-to-average ratio, update means estimated peak-to-average ratio after algorithm adjustment.
   - Use vLLM monitor to detect imbalances during runtime.
   - Always verify expert map JSON structure before loading (validate with jq or similar tools).
