# Dynamic Chunked Pipeline Parallel (DeepSeek-V3.1)

## Getting Started

vLLM-Ascend supports Dynamic Chunked Pipeline Parallel (CPP) for optimizing prefill performance in Pipeline Parallelism scenarios. This guide demonstrates deployment with DeepSeek-V3.1 on 1 Atlas 800T A3 server (64G × 16).

For configuration details, see the [Feature Guide](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md). For design details, see the [Design Document](../../developer_guide/Design_Documents/dynamic_chunked_pipeline_parallel.md).

## Environment Preparation

### Model Weight

- `DeepSeek-V3.1-w8a8` (Quantized version): 1 Atlas 800T A3 (64G × 16) node

Download to shared directory such as `/mnt/weight/`

### Run with Docker

```{code-block} bash
   :substitutions:
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
export NAME=vllm-ascend

docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci8 \
--device /dev/davinci9 \
--device /dev/davinci10 \
--device /dev/davinci11 \
--device /dev/davinci12 \
--device /dev/davinci13 \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/weight:/mnt/weight \
-it $IMAGE bash
```

## Deployment

### Startup Script

```shell
#!/bin/sh
unset https_proxy
unset http_proxy

export OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=2048
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

vllm serve /mnt/weight/DeepSeek-V3.1-w8a8 \
  --host 0.0.0.0 \
  --port 8003 \
  --served-model-name model \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --quantization ascend \
  --additional-config '{
    "profiling_chunk_config":{"enabled":true, "smooth_factor":1.0, "min_chunk":4096}
  }'
```

### Key Parameters

- `--pipeline-parallel-size 2`: Enables Pipeline Parallelism (required)
- `--enable-chunked-prefill`: Enables Chunked Prefill (required)
- `--max-num-batched-tokens 32768`: Initial chunk size (recommended for 128K sequences)
- `profiling_chunk_config.enabled`: Enables Dynamic Chunked Pipeline Parallel
- `profiling_chunk_config.smooth_factor`:  Smoothing factor (0 < x ≤ 1.0). Higher values trust dynamic prediction more
- `profiling_chunk_config.min_chunk`: Minimum chunk size for dynamic calculation. Should be smaller than `max-num-batched-tokens`

For configuration details, see the [Feature Guide](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md).

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

## Accuracy Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

| dataset | accuracy |
|---------|----------|
| gsm8k   | 95.83    |

## Performance Benchmark

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
