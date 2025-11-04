# Expert Load Balance (EPLB)

## Overview

Expert balancing for MoE models in LLM serving is essential for optimal performance. Dynamically changing experts during inference can negatively impact TTFT (Time To First Token) and TPOT (Time Per Output Token) due to stop-the-world operations. SwiftBalancer enables asynchronous expert load balancing with zero-overhead expert movement, ensuring seamless service continuity.

## EPLB Effects

- Reduced Latency: Dynamically balances expert loads to minimize TTFT and TPOT by distributing workloads evenly across experts.
- Enhanced Throughput: Optimizes GPU utilization, increasing token generation speed under high-concurrency scenarios.
- Zero-Overhead Movement: Expert redistribution occurs asynchronously without interrupting ongoing inference requests.
- Adaptive Scaling: Automatically adjusts to workload fluctuations while maintaining stable performance.
- Fault Tolerance: Redundant expert placement ensures system resilience during hardware failures.

## How to Use EPLB

### Dynamic EPLB

We need to add the environment variable `export PYTHONOPTIMIZE=1` to get context of the vllm process. Enable dynamic balancing with auto-tuned parameters. Adjust num_iterations_eplb_update and num_wait_worker_iterations based on workload patterns.

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{
    "dynamic_eplb": true,
    "num_iterations_eplb_update": 400,
    "num_wait_worker_iterations": 30
  }'
```

### Static EPLB
#### Initial Setup (Record Expert Map)

Generate the initial expert distribution map using expert_map_record_path. This creates a baseline configuration for future deployments.

```shell
vllm serve Qwen/Qwen3-235B-A22 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --additional-config '{
    "expert_map_record_path": "/path/to/eplb.json",
    "init_redundancy_expert": 16,
    "num_iterations_eplb_update": 400,
    "num_wait_worker_iterations": 30
  }'
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
   - num_iterations_eplb_update: Higher values (e.g., 400+) for stable workloads; lower values (e.g., 100-200) for fluctuating traffic.
   - num_wait_worker_iterations: Should be ≥ 30 to avoid premature balancing during startup.
   - init_redundancy_expert: Must match tensor-parallel size (e.g., 16 for 16 GPUs) to ensure sufficient redundancy.

2. Hardware Requirements:
   - Ensure that all GPUs have identical memory capacity and compute capabilities.
   - Network bandwidth must support expert redistribution traffic (≥ 10 Gbps recommended).

3. Model Compatibility:
   - Only MoE models with explicit expert parallelism support (e.g., Qwen3-235B-A22) are compatible.
   - Verify model architecture supports dynamic expert routing through --enable-expert-parallel.

4. Gating Configuration:
   - When gate_eplb=true, validate that the gating mechanism can handle expert movement without routing errors.
   - Test with synthetic workloads before production deployment.

5. Monitoring & Validation:
   - Track metrics: expert_load_balance_ratio, ttft_p99, tpot_avg, and gpu_utilization.
   - Use vllm monitor to detect imbalances during runtime.
   - Always verify expert map JSON structure before loading (validate with jq or similar tools).

6. Startup Behavior:
   - Initial requests may experience higher latency during the first balancing cycle (typically 1-2 minutes).
   - Avoid sudden traffic spikes during the warm-up phase.

7. Common Pitfalls:
   - Incorrect tensor-parallel-size vs. actual GPU count → causes resource underutilization.
   - Using expert_map_path without generating the map first → runtime errors.
   - Setting init_redundancy_expert > available GPUs → system failure.
