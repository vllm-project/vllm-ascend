# Qwen3-Coder-30B-A3B

## 1 Introduction

Qwen3-Coder-30B-A3B is a Mixture-of-Experts (MoE) model in the Qwen3 Coder series, sharing the same architecture as Qwen3-30B-A3B with 30.5B total parameters and 3.3B activated per token. Built upon the Qwen3 base architecture, it delivers significant optimizations in agentic coding, extended context support of up to 1M tokens, and versatile function calling capabilities.

This document will demonstrate the main validation steps for Qwen3-Coder-30B-A3B in the vLLM-Ascend environment, including supported features, environment preparation, single-node deployment, as well as accuracy and performance evaluation.

The Qwen3-Coder-30B-A3B model is first supported in **v0.10.0rc1**. This document is validated and written based on **vLLM-Ascend v0.20.2**. All **v0.20.2 and later versions** can run stably. To use the latest features, it is recommended to use v0.20.2 or a later version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory across multiple nodes (e.g., `/root/.cache/`).

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-Coder-30B-A3B-Instruct (BF16) | Atlas 800I A2 or A3 (64G, 1~4 cards) | [Download](https://modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B-Instruct-W8A8 | Atlas 800I A2 or A3 (64G, 1~4 cards) | [Download](https://modelscope.cn/models/Eco-Tech/Qwen3-Coder-30B-A3B-Instruct-w8a8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

:::{note}
If the W8A8 quantized weights are not available for direct download, you can obtain them by quantizing the BF16 model using **msmodelslim**. Refer to the [Quantization Guide](../../user_guide/feature_guide/quantization.md) for details. All model paths in this document should be adjusted to your actual local paths.
:::

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

**Docker Pull:**

```bash
docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7], Atlas A3: /dev/davinci[0-15]).
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|

docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

The default workdir is `/workspace`. vLLM and vLLM-Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`), so changes take effect immediately without requiring a new installation.

**Installation Verification:**

After starting the container, run the following command to verify the installation:

```bash
docker ps | grep vllm-ascend-env
```

Expected result: The container is listed with status `Up`. You can also verify the vllm-ascend version inside the container:

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, matching the pulled image version.

### 4.2 Source Code Installation

If you prefer not to use the Docker image, you can build from source:

1. Clone the repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   ```

2. Install in development mode:

   ```bash
   pip install -e .
   ```

**Installation Verification:**

```bash
pip show vllm-ascend
```

Expected result: The version information is displayed, confirming a successful installation.

:::{note}
If deploying a multi-node environment, set up the environment on each node.
:::

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios. For the Qwen3-Coder-30B-A3B MoE model, Expert Parallelism (EP) is required to distribute experts across NPUs.

```bash

export VLLM_USE_MODELSCOPE=True
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 32768 \
    --quantization ascend \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --port 1025
```

**Key Parameter Description:**

> Replace `your_model_path` with the actual model path (e.g., ModelScope ID or local path).
> Qwen3-Coder-30B-A3B natively supports 256K token context length. Adjust `--max-model-len` based on your use case and available memory.
> The startup command above shows a basic configuration. Additional parameters used in performance-optimized scenarios (see Section 9.1) are also listed below for reference.

| Parameter | Description |
|-----------|-------------|
| `--async-scheduling` | Enables asynchronous scheduling to improve concurrent request processing. |
| `--compilation-config` | FullGraph optimization that captures and replays the entire decode graph, reducing scheduling latency. |
| `--distributed_executor_backend "mp"` | Uses the multi-process distributed backend for parallel execution. |
| `--enable-expert-parallel` | Enables Expert Parallelism, which is required for MoE models to distribute experts across NPUs. |
| `--gpu-memory-utilization 0.95` | Proportion of NPU memory allocated for the KV cache. Higher values increase cache capacity but risk OOM. |
| `--max-model-len 32768` | Maximum context length. Adjust based on your use case and available NPU memory. Larger values increase KV cache usage. |
| `--max-num-batched-tokens` | Maximum tokens processed in a single batch. Balances prefill chunking with memory usage. |
| `--max-num-seqs` | Maximum number of concurrent requests. Adjust based on workload and available KV cache memory. |
| `--no-enable-prefix-caching` | Disables prefix caching. Recommended for general scenarios to reduce memory overhead. |
| `--port 1025` | Port number for the API server. Adjust to avoid conflicts with other services. |
| `--quantization ascend` | Enables W8A8 quantization inference. Remove this parameter when using the BF16 model. |
| `--served-model-name qwen3-coder` | The model name exposed by the service, used as the `model` field in API calls. |
| `--speculative-config` | Speculative decoding configuration. Uses eagle3 draft model to reduce decode latency. |
| `--tensor-parallel-size 4` | Tensor parallelism degree. For Atlas A2 64G, at least 2 is required; 4 is recommended for optimal performance. |
| `--trust-remote-code` | Allows custom model code to be executed from remote repositories. Required for Modelscope models. |

**Service Verification:**

After the service is started, verify it is running:

```bash
curl http://localhost:1025/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder",
        "messages": [
            {"role": "user", "content": "Hello."}
        ],
        "max_tokens": 10
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```shell
curl http://localhost:1025/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-coder",
        "messages": [
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_completion_tokens": 4096
    }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field with generated text.

## 7 Accuracy Evaluation

### Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

<!-- TODO: Add accuracy evaluation results when available -->

### Using Language Model Evaluation Harness

Using the `openai_humaneval` dataset as an example, run the accuracy evaluation in online mode:

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval`:

   ```shell
   lm_eval \
     --model local-completions \
     --model_args model=/root/.cache/Qwen/Qwen3-Coder-30B-A3B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
     --tasks openai_humaneval \
     --output_path ./
   ```

## 8 Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

Take the `serve` subcommand as an example:

```shell
vllm bench serve \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3-coder \
    --port 1025 \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

<!-- TODO: Add performance evaluation results when available -->

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|------------------------|
| High Throughput | Single-Node (TP1) | 1 (A2) | W8A8 | Single-card deployment maximizes concurrent request processing |
| Low Latency | Single-Node (TP4) | 4 (A2) | W8A8 | Multi-card TP reduces per-token latency with expert parallelism |
| Long Context | Single-Node (TP4) | 4 (A2) | W8A8 | Reduces concurrent sequences to accommodate longer max-model-len |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | #NPUs | TP | DP | BS | Concurrency | Max Context Length | FUSED_MC2 | EP Switch | FC+CP Switch | Async Scheduling |
|----------|---------------|-------|----|----|----|-------------|--------------------|-----------|-----------|--------------|------------------|
| High Throughput | Single-Node | 1 | 1 | 1 | 32 | 100 | 37364 | Off | Off | Off | On |
| Low Latency | Single-Node | 4 | 4 | 1 | 32 | 100 | 37364 | Off | On | On | On |
| Long Context | Single-Node | 4 | 4 | 1 | 32 | 14 | 135000 | Off | On | On | On |

> For detailed parameter descriptions, please refer to the deployment examples in Section 5.

**Low Latency Configuration:**

```shell

export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --port 1025 \
    --speculative-config '{"method": "eagle3","model": "/mnt/share/weight/Qwen3-Coder-30B-A3B-EAGLE3", "num_speculative_tokens": 3}'
```

**High Throughput Configuration:**

```shell

export ASCEND_RT_VISIBLE_DEVICES=15
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 1 \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --port 1025 \
    --speculative-config '{"method": "eagle3","model": "/mnt/share/weight/Qwen3-Coder-30B-A3B-EAGLE3", "num_speculative_tokens": 3}'
```

**Long Context Configuration:**

```shell

export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve your_model_path \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 14 \
    --max-model-len 135000 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed_executor_backend "mp" \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --port 1025 \
    --speculative-config '{"method": "eagle3","model": "/mnt/share/weight/Qwen3-Coder-30B-A3B-EAGLE3", "num_speculative_tokens": 3}'
```

### 9.2 Tuning Guidelines

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This chapter only covers model-specific issues.

### Q: What hardware is required for Qwen3-Coder-30B-A3B?

The model can run on Atlas 800I A2 or A3 with 64G NPUs, typically using 1 to 4 cards.

### Q: How do I enable long context (beyond 32K)?

Qwen3-Coder-30B-A3B natively supports 256K token context length, extendable up to 1M with YaRN rope scaling. For contexts beyond the default 32K, use YaRN rope scaling via `--hf-overrides` (for vLLM >= v0.12.0):

```bash
--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'
```

### Q: Why is `--enable-expert-parallel` required?

Qwen3-Coder-30B-A3B is a Mixture-of-Experts model where different experts reside on different NPUs. Expert Parallelism (EP) distributes these experts across devices, which is required for the model to fit in memory and run efficiently. vLLM does not support mixing ETP and EP; MoE layers use either pure EP or pure TP.

### Q: When should I use `VLLM_ASCEND_ENABLE_NZ=2`?

Set `VLLM_ASCEND_ENABLE_NZ=2` when you want maximum throughput and your model uses fp16/bf16 weights. This forces FRACTAL_NZ weight format conversion for all weight types, improving compute efficiency. The default value `1` only converts quantized weights. Set to `0` for RL scenarios where NZ format may cause precision issues.

### Q: What makes Qwen3-Coder different from Qwen3-30B-A3B?

Qwen3-Coder-30B-A3B shares the same MoE architecture (30.5B/3.3B) as the base Qwen3-30B-A3B but is specifically fine-tuned for coding tasks, with optimizations for agentic coding, function calling, and extended context support up to 1M tokens.
