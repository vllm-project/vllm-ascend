# Expert Load Balance (EPLB)

## Overview

Expert balancing for MoE models in LLM serving is essential for optimal performance. Dynamically changing experts during inference can negatively impact TTFT (Time To First Token) and TPOT (Time Per Output Token) due to stop-the-world operations. SwiftBalancer enables asynchronous expert load balancing with zero-overhead expert movement, ensuring seamless service continuity.

## EPLB Effects

- Reduced Latency: Dynamically balances expert loads to minimize TTFT and TPOT by distributing workloads evenly across experts.
- Enhanced Throughput: Optimizes GPU utilization, increasing token generation speed under high-concurrency scenarios.
- Zero-Overhead Movement: Expert redistribution occurs asynchronously without interrupting ongoing inference requests.
- Adaptive Scaling: Automatically adjusts to workload fluctuations while maintaining stable performance.
- Fault Tolerance: Redundant expert placement ensures system resilience during hardware failures.

## Support Scenarios

### Models

DeepSeekV3/V3.1/R1, Qwen3-MoE

### MOE QuantType

W8A8-Dynamic

## How to Use EPLB

### Dynamic EPLB

We need to add environment variable `export DYNAMIC_EPLB="true"` to enable vLLM EPLB. Enable dynamic balancing with auto-tuned parameters. Adjust expert_heat_collection_interval and algorithm_execution_interval based on workload patterns.

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{ "eplb_config": {
    "dynamic_eplb": true,
    "expert_heat_collection_interval": 400,
    "algorithm_execution_interval": 30
    }}'
```

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
    "expert_map_path": "/path/to/eplb.json"
  }'
```

## Critical Considerations

1. Parameter Tuning:
   - expert_heat_collection_interval: Higher values (e.g., 400+) for stable workloads; lower values (e.g., 100-200) for fluctuating traffic.
   - algorithm_execution_interval: Should be ≥ 30 to avoid premature balancing during startup.
   - num_redundant_experts: Must match tensor-parallel size (e.g., 16 for 16 GPUs) to ensure sufficient redundancy.

2. Hardware Requirements:
   - Ensure that all GPUs have identical memory capacity and compute capabilities.
   - Network bandwidth must support expert redistribution traffic (≥ 10 Gbps recommended).

3. Model Compatibility:
   - Only MoE models with explicit expert parallelism support (e.g., Qwen3 MoE models) are compatible.
   - Verify model architecture supports dynamic expert routing through --enable-expert-parallel.

4. Monitoring & Validation:
   - Track metrics: expert_load_balance_ratio, ttft_p99, tpot_avg, and gpu_utilization.
   - Use vLLM monitor to detect imbalances during runtime.
   - Always verify expert map JSON structure before loading (validate with jq or similar tools).

5. Startup Behavior:
   - Initial requests may experience higher latency during the first balancing cycle (typically 1-2 minutes).
   - Avoid sudden traffic spikes during the warm-up phase.

6. Common Pitfalls:
   - Incorrect tensor-parallel-size vs. actual GPU count → causes resource underutilization.
   - Using expert_map_path without generating the map first → runtime errors.
   - Setting num_redundant_experts > available GPUs → system failure.
