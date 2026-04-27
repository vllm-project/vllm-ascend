# Dynamic Chunked Pipeline Parallel

:::{note}
For design details and mathematical models, see [Design Document](../../developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md). For deployment tutorial, see [Dynamic Chunked Pipeline Parallel Tutorial](../../tutorials/features/dynamic_chunked_pipeline_parallel.md).
:::

## Overview

Dynamic Chunked Pipeline Parallel (CPP) is a profiling-based dynamic chunking strategy that optimizes prefill performance for long sequences in Pipeline Parallelism (PP) scenarios.

### When to Use

- **Variable-length sequence serving**: PP does not introduce degradation on short sequences, and gains benefits through dynamic chunks on long sequences.
- **Ultra-long sequence inference**: For sequences exceeding single-machine memory capacity (e.g., 1M tokens), dynamic chunking significantly reduces pipeline idle time.

## Supported Scenarios

Currently CPP mainly focuses on optimization during the prefill phase. It is better to be used in PD disaggregation scenarios. Supported features are as follows:

|         | Eager | Graph | Prefix <br> Cache | Chunked <br> Prefill |
| ------- | ----- | ----- | ------ | ------ |
| **CPP** | ✅    | ✅     | ✅      | ✅       |


## How to Enable

### Online Serving

```bash
vllm serve <model_path> \
    --pipeline-parallel-size 2 \
    --enable-chunked-prefill \
    --additional-config '{"profiling_chunk_config": {"enabled": true}}'
```

### Offline Inference

```python
from vllm import LLM

llm = LLM(
    model="<model_path>",
    pipeline_parallel_size=2,
    additional_config={"profiling_chunk_config": {"enabled": True}},
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable/disable Dynamic Chunked Pipeline Parallel |
| `smooth_factor` | float | 1.0 | Smoothing factor (0 < x ≤ 1.0). Higher values trust dynamic prediction more |
| `min_chunk` | int | 4096 | Minimum chunk size for dynamic calculation |
| `need_timing` | bool | True | Enable/disable Online Calibration |

### Parameter Tuning

- `smooth_factor`: Controls trust level in dynamic prediction
  - `1.0`: Strictly follow model prediction
  - `0.6~0.85`: Balance dynamic adjustment and scheduling overhead
  - `0.0`: No dynamic adjustment (degrades to fixed chunking)
- `min_chunk`: Generally doesn't need adjustment. Should be smaller than `max-num-batched-tokens`

## Recommended Settings

### max-num-batched-tokens

**Notably, the TTFT of CPP is very sensitive to `max-num-batched-tokens` (considered the initial chunksize for dynamic solving).** Because if it is too large, it will introduce si
gnificant computational voids, and if it is too small, it will lead to a decrease in operator efficiency. To leave enough room for dynamic adjustments, we recommend that the longer the sequence being processed, the larger the `max-num-batched-tokens` should be set. Recommended values:

| Sequence Length | `max-num-batched-tokens` |
|-----------------|--------------------------|
| 64k             | 20480                    |
| 128k            | 32768                    |

### Online Calibration

For optimal performance, online calibrate with real data before production:

You can use aisbench to generate fixed-length random datasets. Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

1. Modify `<YOUR_AISBENCH_PATH>/benchmark/ais_bench/datasets/synthetic/synthetic_config.py`:

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 5,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 131072, "MaxValue": 131072}  # Your max sequence length, max-model-len
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 1}
        }
    },
}
```

2. Run for online calibration:
```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --debug
```

Configure online calibration data length to match your `max-model-len`. Use `batch_size=1` and ensure data differs to avoid cache hits if prefix caching is enabled.

## Performance

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

To evaluate the effectiveness of Dynamic Chunked Pipeline Parallel in long sequence LLM inference scenarios, we use **DeepSeek-V3.1-W8A8** and **Qwen3-235B**, deploy P instance in Ascend Atlas A3 inference products*64G (A3), the configuration and performance data are as follows.

**Fixed-length requests, concurrency=1**:
- DeepSeek-V3.1-W8A8:

    | Configuration | CPP <br> (Dynamic Chunk, <br> chunksize=32k) | PP <br>(Static Chunk, <br> chunksize=32k) |
    | ----------------------------- | ------------------------- | ------------------------- | 
    | Input length  128k    | TTFT: 22.5s | TTFT: 27.0s |

- Qwen3-235B:

    | Configuration | CPP <br> (Dynamic Chunk, <br> chunksize=32k) | PP <br>(Static Chunk, <br> chunksize=32k) |
    | ----------------------------- | ------------------------- | ------------------------- | 
    | Input length  256k    | TTFT: 53.5s | TTFT: 61.4s |

**Variable-length requests**:
- DeepSeek-V3.1-W8A8:

    | Configuration | 4k~64k Input, mean=32k, std=32k <br> prefix hit rate=99% |
    | ----------------------------- | ------------------------- | 
    |  CPP2TP8   | Input throughput: 22424 tps/card |
    |  DP2TP8   | Input throughput: 16150 tps/card |
    |  PCP2TP8   | Input throughput: 18197 tps/card |
    |  TP16   | Input throughput: 18875 tps/card |


## Constraints

- **Pipeline Parallelism Required**: `--pipeline-parallel-size > 1`
- **Chunked Prefill Required**: `--enable-chunked-prefill`
- **Incompatible with Balance Scheduling**: Cannot enable `VLLM_ASCEND_BALANCE_SCHEDULING`
- **Startup Overhead**: Profiling adds ~64 forward passes (tens of seconds)
