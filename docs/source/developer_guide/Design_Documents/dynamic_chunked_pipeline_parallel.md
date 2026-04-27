# Dynamic Chunked Pipeline Parallel (CPP)

TL;DR CPP uses profiling-based dynamic chunking to equalize per-chunk latency and eliminate pipeline bubbles in PP scenarios.

## Background

### Problem Statement

In Pipeline Parallelism (PP) + Chunked Prefill scenarios, long sequences are split into fixed-size chunks that pass through the pipeline sequentially. Due to the O(n²) computational complexity of Self-Attention, **chunks of the same size take increasingly longer to process as the prefix sequence grows**:

```
Chunk 1 (history=0):     ██████         → Time T1
Chunk 2 (history=4K):    ████████       → Time T2 > T1
Chunk 3 (history=8K):    ██████████     → Time T3 > T2
Chunk 4 (history=12K):   ████████████   → Time T4 > T3
```

This time variance propagates across pipeline stages, causing increased idle waiting (Pipeline Bubble) and significantly reducing GPU utilization.

### Solution Overview

Dynamic Chunked Pipeline Parallel uses a **profile-first, then predict** strategy:

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

The core idea is borrowed from [SGLang's dynamic chunking mechanism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/), with additional enhancements such as online calibration.

## Design

### Quadratic Latency Model

Transformer prefill latency grows quadratically with sequence length due to the O(n²) Self-Attention mechanism:

$$f(l) = a \cdot l^2 + b \cdot l + c$$

Where:
- $a \cdot l^2$: Attention overhead (quadratic)
- $b \cdot l$: Linear operations (FFN, projection)
- $c$: Fixed overhead (kernel launch)

### Startup Phase: Profiling

During engine initialization, the system profiles actual model performance:

1. **Sampling**: Uniformly sample 64 different chunk sizes from `base_chunk_size` down to near 0
2. **Execution**: Perform real model forward passes for each chunk size and precisely measure latency (milliseconds)
3. **Fitting**: Fit the quadratic model using least squares
4. **Target Setting**: Calculate target per-chunk latency based on `base_chunk_size`

In PP mode, all workers execute forward passes to stay synchronized, but only the first PP rank's timing results are used for scheduling decisions.

### Runtime Phase: Dynamic Prediction

Given current prefix length $L$ and target latency $T = f(\text{base\_chunk\_size}) - f(0)$, the system solves for the next chunk size $x$:

$$f(L + x) - f(L) = T$$

Expanding to:

$$a \cdot x^2 + (2aL + b) \cdot x - T = 0$$

Solved using the quadratic formula:

$$x = \frac{-(2aL + b) + \sqrt{(2aL + b)^2 + 4aT}}{2a}$$

The result goes through post-processing:

1. **Smoothing**: Blend predicted chunk size with `base_chunk_size` using `smooth_factor`
2. **Alignment**: Round down to multiple of `page_size` (minimum 64)
3. **Constraints**: Not exceeding `max_model_len - history_len` and `max_num_scheduled_tokens`

### Online Calibration

Since profiling only covers sequences up to `max_num_batched_tokens` (typically shorter than real workloads), the system continuously refines the model at runtime.

**Extended Model (two variables):**

$$f(C, H) = a \cdot C(C+H) + b \cdot (C+H) + c$$

Where $C$ is chunk size and $H$ is prefix history length.

After each batch, feature vectors `[Σ(C+H)·C, Σ(C+H), N]` and actual execution time are recorded. Once enough data points accumulate (5-30), model parameters are updated using least squares.

## Architecture

### Key Components

| Component | Location | Responsibility |
|-----------|----------|---------------|
| **ChunkSizePredictor** | `vllm_ascend/core/profiling_chunk_predictor.py` | Quadratic model fitting and prediction |
| **ProfilingChunkManager** | `vllm_ascend/core/profiling_chunk_predictor.py` | Manage profiling workflow and predictor |
| **Scheduler** | `vllm_ascend/core/scheduler_profiling_chunk.py` | Integrate CPP scheduling |
| **EngineCore** | `vllm_ascend/patch/platform/patch_profiling_chunk.py` | Startup profiling, record execution time |
| **NPUWorker** | `vllm_ascend/worker/worker.py` | Execute real forward pass profiling |
| **NPUModelRunner** | `vllm_ascend/worker/model_runner_v1.py` | `profile_cpp=True` mode |

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Startup Phase                            │
├─────────────────────────────────────────────────────────────┤
│  1. EngineCore.init() triggers profiling                    │
│  2. ProfilingChunkManager samples 64 chunk sizes            │
│  3. NPUWorker executes forward passes                       │
│  4. ChunkSizePredictor fits quadratic model                 │
│  5. Target latency = f(base_chunk_size) - f(0)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Runtime Phase                            │
├─────────────────────────────────────────────────────────────┤
│  For each prefill chunk:                                    │
│    1. Scheduler queries ChunkSizePredictor                  │
│    2. Given history length L, solve for optimal chunk size  │
│    3. Apply smoothing and alignment                         │
│    4. Execute chunk                                         │
│    5. Record actual timing for online calibration           │
│    6. Update model if enough samples collected              │
└─────────────────────────────────────────────────────────────┘
```

## Comparison with SGLang

| Feature | SGLang Dynamic Chunking | Dynamic Chunked Pipeline Parallel |
|---------|------------------------|-----------------------------------|
| Profiling method | Preset quadratic function | Real forward pass profiling at startup |
| Model fitting | $f(l) = a \cdot l^2 + b \cdot l + c$ | Same + online calibration $f(C,H)$ |
| Online updates | None | History-based fitting |
| Accuracy | May deviate on different hardware | Adapts to actual hardware performance |
| Startup cost | None | ~64 forward passes (tens of seconds) |

## Constraints

- **Pipeline Parallelism Required**: Must set `--pipeline-parallel-size > 1`
- **Chunked Prefill Required**: Must enable `--enable-chunked-prefill`
- **Incompatible with Balance Scheduling**: Cannot enable `VLLM_ASCEND_BALANCE_SCHEDULING`
- **Startup Overhead**: Profiling phase adds tens of seconds to initialization
- **Memory**: No additional runtime memory overhead; profiling reuses existing dummy_run mechanism

## References

- [SGLang Dynamic Chunking Blog](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)
- [User Guide](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md)
- [Tutorial](../../tutorials/features/dynamic_chunked_pipeline_parallel.md)