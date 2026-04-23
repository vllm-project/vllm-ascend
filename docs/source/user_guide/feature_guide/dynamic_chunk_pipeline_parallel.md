# Chunked Pipeline Parallel Guide

## Overview

Dynamic Chunk for Chunked Pipeline Parallelism (CPP) is a profiling-based dynamic chunking strategy that optimizes prefill performance for long sequences in Pipeline Parallelism (PP) scenarios. This approach addresses the computational efficiency problem caused by growing KVCache during long sequence processing.

The core idea is borrowed from [SGLang's dynamic chunking mechanism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/), but has been fully implemented and adapted for the vLLM-Ascend framework.

## Why Dynamic Chunk?

### The Problem with Fixed Chunking

In PP + Chunked Prefill scenarios, long sequences are split into fixed-size chunks that pass through the pipeline sequentially. However, due to the O(n²) computational complexity of Self-Attention, **chunks of the same size take increasingly longer to process as the prefix sequence grows**:

```
Chunk 1 (history=0):     ██████         → Time T1
Chunk 2 (history=4K):    ████████       → Time T2 > T1
Chunk 3 (history=8K):    ██████████     → Time T3 > T2
Chunk 4 (history=12K):   ████████████   → Time T4 > T3
```

This time variance propagates across pipeline stages, causing increased idle waiting (Pipeline Bubble) in subsequent stages and significantly reducing GPU utilization.

### The Solution

Chunked Pipeline Parallel uses a **profile-first, then predict** strategy:

1. During engine startup, perform forward passes with different chunk sizes to measure actual latency
2. Fit the latency data to a quadratic function model
3. At runtime, dynamically predict the optimal chunk size based on current prefix length to keep each chunk's execution time consistent

## How It Works

### Quadratic Latency Model

Transformer prefill latency can be modeled as a quadratic function of sequence length:

$$f(l) = a \cdot l^2 + b \cdot l + c$$

Where:
- $a$: Quadratic coefficient for attention (reflecting O(n²) attention overhead)
- $b$: Linear coefficient (reflecting FFN, projection, residual connections)
- $c$: Constant term (reflecting fixed overhead like kernel launch)

### Startup Phase: Profiling

During engine initialization:

1. **Sampling**: Uniformly sample 64 different chunk sizes from `base_chunk_size` down to near 0
2. **Execution**: Perform real model forward passes for each chunk size and precisely measure latency (milliseconds)
3. **Fitting**: Fit the quadratic model using least squares: $f(l) = a \cdot l^2 + b \cdot l + c$
4. **Target Setting**: Calculate target latency $T = f(\text{base\_chunk\_size}) - f(0)$

In PP mode, all workers execute forward passes to stay synchronized, but only the first PP rank's timing results are used for scheduling decisions.

### Runtime Phase: Dynamic Prediction

Given current prefix length $L$ and target latency $T$, solve for next chunk size $x$:

$$f(L + x) - f(L) = T$$

This expands to the quadratic equation:

$$a \cdot x^2 + (2aL + b) \cdot x - T = 0$$

Solved using the quadratic formula:

$$x = \frac{-(2aL + b) + \sqrt{(2aL + b)^2 + 4aT}}{2a}$$

The prediction then goes through post-processing:
1. **Smoothing**: `smoothed = base_chunk + smooth_factor × (x - base_chunk)`
2. **Alignment**: Round down to a multiple of `page_size` (minimum 64)
3. **Constraints**: Not exceeding `max_model_len - history_len` and `max_num_scheduled_tokens`

### Online Calibration

Since the profile phase uses sequences up to `max_num_batched_tokens` (which is smaller than actual long sequence lengths), the system collects real execution data during runtime for online model calibration:

**Fitting model**: $f(C, H) = a \cdot C(C+H) + b \cdot (C+H) + c$

Where $C$ is chunk size and $H$ is prefix history length.

After each batch execution, feature vectors `[Σ(C+H)·C, Σ(C+H), N]` and actual execution time are recorded. Once enough data points accumulate (5-30), model parameters are updated using least squares.

**Note**: For best results, warm up with 3-5 real data samples after service startup.

## How to Use Chunked Pipeline Parallel

### Enable Chunked Pipeline Parallel

Chunked Pipeline Parallel requires Pipeline Parallelism (PP > 1) and enable Chunked Prefill. **Notably, the TTFT of CPP is very sensitive to `max-num-batched-tokens` (considered the initial chunksize for dynamic solving).** Because if it is too large, it will introduce significant computational voids, and if it is too small, it will lead to a decrease in operator efficiency. To leave enough room for dynamic adjustments, we recommend that the longer the sequence being processed, the larger the `max-num-batched-tokens` should be set.
For fixed-length sequences, we obtained some empirical values for sequence lengths through experiments with DeepSeek v3.1.

| seq_len | `max-num-batched-tokens` |
|-----------|------|
| 64k | 20480 |
| 128k | 32768 |

You can enable dynamic chunking through the `additional_config` parameter:

**Online serving:**

```bash
vllm serve <model_path> \
    --pipeline-parallel-size 2 \
    --additional-config '{"profiling_chunk_config": {"enabled": true}}'
```

**Offline inference:**

```python
from vllm import LLM

llm = LLM(
    model="<model_path>",
    pipeline_parallel_size=2,
    additional_config={"profiling_chunk_config": {"enabled": True}},
)
```

### Addtional Configuration Parameters

The `profiling_chunk_config` accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable/disable dynamic chunked pipeline parallel |
| `smooth_factor` | float | 1.0 | Smoothing factor (0 < x ≤ 1.0). Higher values trust the dynamic strategy more |
| `min_chunk` | int | 4096 | Minimum chunk size for dynamic calculation |

**Parameter tuning guidelines:**

- `enabled`: The main switch for this feature
- `smooth_factor`: Controls how much to trust the dynamic prediction
  - `1.0`: Strictly follow model prediction
  - `0.6~0.85`: Balance between dynamic adjustment and scheduling overhead
  - `0.0`: No dynamic adjustment (degrades to fixed chunking)
- `min_chunk`: Generally doesn't need adjustment. Should be smaller than `max-num-batched-tokens`

### Warm-up with Real Data

For optimal performance, warm up the service with real data before production use. You can use aisbench to generate fixed-length random datasets:

1. Modify `ais_bench/datasets/synthetic/synthetic_config.py`:

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 5,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 131072, "MaxValue": 131072}  # Your max sequence length
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 1}
        }
    },
}
```

2. Run warm-up:

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug
```

**Important considerations:**
- The length of data for Warm-up should be your max sequence length
- Use `batch_size=1` in aisbench configuration
- If prefix cache is enabled, ensure warm-up data is different from each other to avoid cache hits affecting calibration

## Performance Tuning

Reference [SGLang's tuning recommendations](https://docs.sglang.com.cn/advanced_features/pipeline_parallelism.html):

1. **Find optimal fixed chunk size**: Experiment with 2K~16K range to find the best TTFT
2. **Set initial chunk size**: Multiply optimal fixed value by 2~3x for dynamic chunking
3. **Adjust smooth factor**:
   - `1.0`: Strict dynamic adjustment
   - `0.6~0.85`: Balance dynamic adjustment with scheduling frequency
   - `0.0`: Fixed chunking (disabled)

**Example configurations for DeepSeek V3.1:**
- 64K input: `max-num-batched-tokens=20480`
- 128K input: `max-num-batched-tokens=32768`

## Constraints

- **Pipeline Parallelism Required**: Must set `--pipeline-parallel-size > 1`
- **Incompatible with Balance Scheduling**: Cannot enable `VLLM_ASCEND_BALANCE_SCHEDULING` environment variable simultaneously
- **Startup Overhead**: Profiling phase executes ~64 forward passes, adding tens of seconds to startup time
- **Memory**: No additional runtime memory overhead; profiling reuses existing dummy_run mechanism

## Key Components

| Component | Location | Responsibility |
|-----------|----------|---------------|
| **ChunkSizePredictor** | `vllm_ascend/core/profiling_chunk_predictor.py` | Quadratic model fitting and prediction |
| **ProfilingChunkManager** | `vllm_ascend/core/profiling_chunk_predictor.py` | Manage profiling workflow and predictor |
| **Scheduler** | `vllm_ascend/core/scheduler_profiling_chunk.py` | Integrate dynamic chunk scheduling |
| **EngineCore** | `vllm_ascend/patch/platform/patch_profiling_chunk.py` | Startup profiling, record execution time |
| **NPUWorker** | `vllm_ascend/worker/worker.py` | Execute real forward pass profiling |
| **NPUModelRunner** | `vllm_ascend/worker/model_runner_v1.py` | `profile_cpp=True` mode |

## Comparison with SGLang

| Feature | SGLang Dynamic Chunking | Profile CPP (This Solution) |
|---------|------------------------|----------------------------|
| Profiling method | Preset quadratic function | Real forward pass profiling at startup |
| Model fitting | $f(l) = a \cdot l^2 + b \cdot l + c$ | Same + online calibration $f(C,H)$ |
| Online updates | None | History-based fitting |