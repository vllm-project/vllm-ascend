# Dynamic Chunked Pipeline Parallel Guide

## Overview

Dynamic Chunked Pipeline Parallel (CPP) is a profiling-based dynamic chunking strategy that optimizes prefill performance for long sequences in Pipeline Parallelism (PP) scenarios. This approach addresses the computational efficiency problem caused by growing KVCache during long sequence processing.

The core idea is borrowed from [SGLang's dynamic chunking mechanism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/), and has been adapted for the vLLM-Ascend framework with additional enhancements such as online calibration for improved prediction accuracy.

### When to Use

Because PP has relatively low communication overhead, Dynamic Chunked Pipeline Parallel is particularly effective in the following scenarios:

- **Variable-length sequence serving**: PP does not introduce degradation on short sequences, and on long sequences it gains benefits through dynamic chunks.
- **Ultra-long sequence inference across multiple machines**: For sequences that exceed single-machine memory capacity (e.g., 1M tokens), PP is required to distribute layers across machines. In this setting, pipeline bubbles become the dominant bottleneck, and dynamic chunking significantly reduces idle time across stages.

## Why Dynamic Chunked Pipeline Parallel?

### The Problem with Fixed Chunking

In PP + Chunked Prefill scenarios, long sequences are split into fixed-size chunks that pass through the pipeline sequentially. However, due to the O(n²) computational complexity of Self-Attention, **chunks of the same size take increasingly longer to process as the prefix sequence grows**:

```
Chunk 1 (history=0):     ██████         → Time T1
Chunk 2 (history=4K):    ████████       → Time T2 > T1
Chunk 3 (history=8K):    ██████████     → Time T3 > T2
Chunk 4 (history=12K):   ████████████   → Time T4 > T3
```

This time variance propagates across pipeline stages, causing increased idle waiting (Pipeline Bubble) in subsequent stages and significantly reducing GPU utilization.

The following diagram illustrates how fixed chunking creates pipeline bubbles, and how dynamic chunking eliminates them by equalizing per-chunk latency:

```
Fixed Chunking (equal chunk size, unequal time):

          Stage 0  |■■■■|■■■■■■|■■■■■■■■|■■■■■■■■■■|
          Stage 1  |    |■■■■  |■■■■■■  |■■■■■■■■  |■■■■■■■■■■|
                        ↑ bubble  ↑ bubble   ↑ bubble

Dynamic Chunking (unequal chunk size, equal time):

          Stage 0  |■■■■■■|■■■■■■|■■■■■■|■■■■■■|
          Stage 1  |      |■■■■■■|■■■■■■|■■■■■■|■■■■■■|
                          ↑ no bubble — stages stay in sync
```

### The Solution

Dynamic Chunked Pipeline Parallel uses a **profile-first, then predict** strategy:

1. During engine startup, perform forward passes with different chunk sizes to measure actual latency
2. Fit the latency data to a quadratic function model
3. At runtime, dynamically predict the optimal chunk size based on current prefix length to keep each chunk's execution time consistent

## How It Works

### Quadratic Latency Model

Transformer prefill latency grows quadratically with sequence length due to the O(n²) Self-Attention mechanism. The system models this relationship as $f(l) = a \cdot l^2 + b \cdot l + c$, where the three terms capture attention overhead, linear operations (FFN, projection), and fixed overhead (kernel launch) respectively.

### Startup Phase: Profiling

During engine initialization, the system profiles actual model performance to build the latency model:

1. **Sampling**: Uniformly sample 64 different chunk sizes from `base_chunk_size` down to near 0
2. **Execution**: Perform real model forward passes for each chunk size and precisely measure latency (milliseconds)
3. **Fitting**: Fit the quadratic model using least squares
4. **Target Setting**: Calculate the target per-chunk latency based on `base_chunk_size`

In PP mode, all workers execute forward passes to stay synchronized, but only the first PP rank's timing results are used for scheduling decisions.

### Runtime Phase: Dynamic Prediction

At runtime, given the current prefix length, the system solves for the chunk size that would produce a latency equal to the target. The result goes through post-processing:

1. **Smoothing**: Blend the predicted chunk size with `base_chunk_size` using `smooth_factor`
2. **Alignment**: Round down to a multiple of `page_size` (minimum 64)
3. **Constraints**: Not exceeding `max_model_len - history_len` and `max_num_scheduled_tokens`

<details>
<summary>Mathematical Details</summary>

The latency model is:

$$f(l) = a \cdot l^2 + b \cdot l + c$$

Given current prefix length $L$ and target latency $T = f(\text{base\_chunk\_size}) - f(0)$, solve for the next chunk size $x$:

$$f(L + x) - f(L) = T$$

This expands to:

$$a \cdot x^2 + (2aL + b) \cdot x - T = 0$$

Solved using the quadratic formula:

$$x = \frac{-(2aL + b) + \sqrt{(2aL + b)^2 + 4aT}}{2a}$$

</details>

### Online Calibration

Since the profiling phase only covers sequences up to `max_num_batched_tokens` (typically shorter than real workloads), the system continuously collects execution data at runtime to refine the latency model. After each batch, feature vectors and actual execution time are recorded. Once enough data points accumulate (5-30), model parameters are updated using least squares.

<details>
<summary>Mathematical Details</summary>

The online fitting model extends to two variables:

$$f(C, H) = a \cdot C(C+H) + b \cdot (C+H) + c$$

Where $C$ is chunk size and $H$ is prefix history length. After each batch, feature vectors `[Σ(C+H)·C, Σ(C+H), N]` and actual execution time are recorded for least squares fitting.

</details>

**Note**: For best results, warm up with 3-5 real data samples after service startup.

## How to Use Dynamic Chunked Pipeline Parallel

### Enable Dynamic Chunked Pipeline Parallel

Dynamic Chunked Pipeline Parallel requires Pipeline Parallelism (PP > 1) and enable Chunked Prefill. **Notably, the TTFT of CPP is very sensitive to `max-num-batched-tokens` (considered the initial chunksize for dynamic solving).** Because if it is too large, it will introduce significant computational voids, and if it is too small, it will lead to a decrease in operator efficiency. To leave enough room for dynamic adjustments, we recommend that the longer the sequence being processed, the larger the `max-num-batched-tokens` should be set.
For fixed-length sequences, we obtained some empirical values for sequence lengths through experiments with DeepSeek v3.1.

| seq_len | `max-num-batched-tokens` |
|-----------|------|
| 64k | 20480 |
| 128k | 32768 |

You can enable Dynamic Chunked Pipeline Parallel through the `additional_config` parameter:

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

### Additional Configuration Parameters

The `profiling_chunk_config` accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable/disable Dynamic Chunked Pipeline Parallel |
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
2. **Set initial chunk size**: Multiply optimal fixed value by 2~3x for Dynamic Chunked Pipeline Parallel
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
| **Scheduler** | `vllm_ascend/core/scheduler_profiling_chunk.py` | Integrate Dynamic Chunked Pipeline Parallel scheduling |
| **EngineCore** | `vllm_ascend/patch/platform/patch_profiling_chunk.py` | Startup profiling, record execution time |
| **NPUWorker** | `vllm_ascend/worker/worker.py` | Execute real forward pass profiling |
| **NPUModelRunner** | `vllm_ascend/worker/model_runner_v1.py` | `profile_cpp=True` mode |

## Comparison with SGLang

| Feature | SGLang Dynamic Chunking | Dynamic Chunked Pipeline Parallel (This Solution) |
|---------|------------------------|----------------------------|
| Profiling method | Preset quadratic function | Real forward pass profiling at startup |
| Model fitting | $f(l) = a \cdot l^2 + b \cdot l + c$ | Same + online calibration $f(C,H)$ |
| Online updates | None | History-based fitting |