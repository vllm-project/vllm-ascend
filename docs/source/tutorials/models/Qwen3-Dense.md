# Qwen3-Dense (Qwen3-0.6B/1.7B/4B/8B/14B/32B, W8A8, W4A8, W4A4)

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. The Dense variants covered in this document include Qwen3-0.6B, 1.7B, 4B, 8B, 14B, and 32B, along with their quantized versions (W8A8, W4A8, W4A4) optimized for Ascend NPU deployment.

This document will demonstrate the main validation steps for Qwen3 Dense models in the vLLM-Ascend environment, including supported features, environment preparation, model quantization, single-node and multi-node deployment, as well as accuracy and performance evaluation. By tailoring service-level configurations to fit different use cases, you can ensure optimal performance across various scenarios.

The Qwen3 Dense models are first supported in v0.8.4rc2. W8A8 quantization was first supported in v0.8.4rc2, W4A8 quantization is supported since v0.9.1rc2, and W4A4 is supported since v0.11.0rc1. This document is validated and written based on **vLLM-Ascend v0.13.0**. All **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use v0.13.0 or a later version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory across multiple nodes (e.g., `/root/.cache/`).

**BF16 Versions:**

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-0.6B | 1 Atlas 800I A3 (64G × 2) or 1 Atlas 800I A2 (64G × 1) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-0.6B) |
| Qwen3-1.7B | 1 Atlas 800I A3 (64G × 2) or 1 Atlas 800I A2 (64G × 1) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-1.7B) |
| Qwen3-4B | 1 Atlas 800I A3 (64G × 2) or 1 Atlas 800I A2 (64G × 1) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-4B) |
| Qwen3-8B | 1 Atlas 800I A3 (64G × 2) or 1 Atlas 800I A2 (64G × 1) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-8B) |
| Qwen3-14B | 1 Atlas 800I A3 (64G × 2) or 2 Atlas 800I A2 (64G × 1) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-14B) |
| Qwen3-32B | 2 Atlas 800I A3 (64G × 4) or 4 Atlas 800I A2 (64G × 4) | [Download](https://modelers.cn/models/Modelers_Park/Qwen3-32B) |

**Quantized Versions (Pre-converted):**

| Model | Quantization | Hardware Requirement | Download |
|-------|-------------|---------------------|----------|
| Qwen3-8B-W4A8 | W4A8 | 1 Atlas 800I A3 (64G × 2) or 1 Atlas 800I A2 (64G × 1) | [Download](https://www.modelscope.cn/models/vllm-ascend/Qwen3-8B-W4A8) |
| Qwen3-32B-W4A4 | W4A4 | 1 Atlas 800I A3 (64G × 2) or 1 Atlas 800I A2 (64G × 1) | [Download](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4) |
| Qwen3-32B-W8A8 | W8A8 | 2 Atlas 800I A3 (64G × 4) or 4 Atlas 800I A2 (64G × 4) | [Download](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W8A8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Verify Multi-node Communication

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 Dense models.

**Docker Pull:**

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

**Docker Run:**

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7], Atlas A3:/dev/davinci[0-15]).
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --privileged=true \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

The default workdir is `/workspace`. vLLM and vLLM-Ascend code are placed in `/vllm-workspace` and installed in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) (`pip install -e`), so changes take effect immediately without requiring a new installation.

To verify the successful installation of the environment, please refer to [installation](../../installation.md).

If deploying a multi-node environment, set up the environment on each node.

### 4.2 Source Code Installation

In addition, if you don't want to use the Docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

If you want to deploy a multi-node environment, you need to set up environment on each node.

### 4.3 Model Quantization (Optional)

If you wish to quantize the model yourself, refer to the [MindStudio ModelSlim documentation](https://gitcode.com/Ascend/msit) for W4A8 and W4A4 quantization procedures. Pre-converted quantized weights are also available for download (see [3.1 Model Weight](#31-model-weight)).

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios.

#### BF16 Models (e.g., Qwen3-8B)

```bash
export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=Modelers_Park/Qwen3-8B
vllm serve ${MODEL_PATH} --served-model-name "qwen3-8b" --max-model-len 4096
```

#### Quantized Models

For W4A8 (e.g., Qwen3-8B-W4A8):

```bash
export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=vllm-ascend/Qwen3-8B-W4A8
vllm serve ${MODEL_PATH} --served-model-name "qwen3-8b-w4a8" --max-model-len 4096 --quantization ascend
```

For W4A4 (e.g., Qwen3-32B-W4A4):

```bash
export MODEL_PATH=/home/models/Qwen3-32B-w4a4
vllm serve ${MODEL_PATH} --served-model-name "qwen3-32b-w4a4" --max-model-len 4096 --quantization ascend
```

:::{note}
To enable quantization for Ascend, the quantization method must be `"ascend"`. If the model is not a quantized model, remove the `--quantization ascend` parameter.
:::

### 5.2 Multi-Node Online Deployment

Multi-node deployment leverages Tensor Parallelism (TP) to distribute the model across multiple NPUs, suitable for large models that exceed single-card memory or require higher throughput.

The following example demonstrates best practices for Qwen3-32B-W8A8 on an Atlas 800I A3 (64G × 16) with DP=1, TP=4, targeting optimal throughput at batch_size=72 with fixed-length input of 3.5K and output of 1.5K.

:::{note}
If the machine is an **Atlas 800I A2 (64G × 8)**, the deployment approach stays identical — adjust `--device` mappings and TP size accordingly.

If you are already inside the container (see [Section 4.1](#41-docker-image-installation)), skip the Docker run step and proceed directly to **Start the server**.
:::

**Start the server:**

```bash
# Set the NPU device number
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1

# [Optional] jemalloc for better performance
# if libjemalloc.so is installed:
# Ubuntu:
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# openEuler:
# export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE="AIV"

# Enable FlashComm_v1 optimization when tensor parallel is enabled
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve vllm-ascend/Qwen3-32B-W8A8 \
    --served-model-name qwen3 \
    --trust-remote-code \
    --async-scheduling \
    --quantization ascend \
    --distributed-executor-backend mp \
    --tensor-parallel-size 4 \
    --max-model-len 5500 \
    --max-num-batched-tokens 40960 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"pa_shape_list":[48,64,72,80], "weight_prefetch_config":{"enabled":true}}' \
    --port 8113 \
    --block-size 128 \
    --gpu-memory-utilization 0.9
```

The key parameters are explained as follows:

- `--tensor-parallel-size 4`: Distributes the model across 4 NPUs using Tensor Parallelism. Adjust based on available NPU count and model size.
- `--quantization ascend`: Enables Ascend-specific quantization for W8A8/W4A8/W4A4 models. **Remove this parameter for BF16 models.**
- `--async-scheduling`: Enables asynchronous scheduling to overlap CPU and NPU operations, reducing NPU idle time and improving throughput.
- `--distributed-executor-backend mp`: Uses the multi-process (mp) distributed backend for tensor parallelism.
- `--max-model-len 5500`: Sets the maximum sequence length the model can handle. Increase for longer contexts (subject to memory constraints).
- `--max-num-batched-tokens 40960`: Caps the total number of tokens across all requests in a single batch. Tune this to balance throughput and memory usage.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables FullGraph optimization to capture and replay the entire decoding graph, significantly reducing scheduling latency. `FULL_DECODE_ONLY` mode captures only the decode phase.
- `--additional-config '{"pa_shape_list":[48,64,72,80], "weight_prefetch_config":{"enabled":true}}'`:
  - `pa_shape_list`: Specifies batch sizes at which to switch from the default FIA operator to the PA operator for attention computation. This is a temporary tuning knob; in the future, FIA will be optimized for these scenarios.
  - `weight_prefetch_config`: Enables prefetching MLP weights into L2 cache during vector computation time, improving throughput in MLP-intensive scenarios.
- `--block-size 128`: Controls the size of KV cache blocks. Larger values reduce scheduling granularity and can improve performance.
- `--gpu-memory-utilization 0.9`: Limits the fraction of NPU memory used by the model and KV cache, leaving headroom for other operations. Increase to 0.95 if memory is constrained.

:::{note}
- `vllm-ascend/Qwen3-32B-W8A8` is the default model path, replace this with your actual path.
- If the model is not a quantized model, remove the `--quantization ascend` parameter.
- For additional parameter details, refer to the [vLLM Serving Arguments documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
:::

**Verify the deployment:**

After starting the server, verify it is running correctly:

```bash
# Check that the server process is listening
curl http://localhost:8113/v1/models

# Expected output: a JSON response listing available models, including "qwen3"
```

If the server responds with a model list, the deployment is successful.

### 5.3 Offline Inference

#### Single-NPU

For quantized models on a single NPU:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/home/models/Qwen3-8B-w4a8",
          max_model_len=4096,
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

#### Multi-NPU

For models requiring tensor parallelism:

```python
import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)
llm = LLM(model="vllm-ascend/Qwen3-32B-W8A8",
          tensor_parallel_size=4,
          trust_remote_code=True,
          distributed_executor_backend="mp",
          max_model_len=5500,
          max_num_batched_tokens=5500,
          quantization="ascend",
          compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```

:::{note}
- `vllm-ascend/Qwen3-32B-W8A8` is the default model path, replace this with your actual path.
- If the model is not a quantized model, remove the `quantization="ascend"` parameter.
:::

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Completions API:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-8b-w4a8",
        "prompt": "what is large language model?",
        "max_completion_tokens": 128,
        "top_p": 0.95,
        "top_k": 40,
        "temperature": 0.0
    }'
```

**Chat Completions API:**

```bash
curl http://localhost:8113/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
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

### 7.1 Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

## 8 Performance

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take `serve` as an example:

```shell
vllm bench serve \
    --model vllm-ascend/Qwen3-32B-W8A8 \
    --served-model-name qwen3 \
    --port 8113 \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

After several minutes, you will get the performance evaluation result.

## 9 Performance Tuning

> **Important**: The configurations provided in this section are validated in specific test environments and are **not** guaranteed to be globally optimal. Actual performance depends on factors such as input/output length distribution, request rate, prefix cache hit rate, hardware configuration, and precision requirements. It is strongly recommended to use the following as a starting point and refer to [Section 9.2](#92-tuning-guidelines) for tuning based on your own workload.

### 9.1 Recommended Configurations

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
|----------|----------------|-------------|----------------|---------------------|
| High Throughput<br>(3.5K → 1.5K) | Single-node | 4 (A3) | Qwen3-32B-W8A8 (PDMix) | For short-sequence high throughput, try adjusting TP size (4 or 8), enabling FlashComm and weight prefetch, tuning `pa_shape_list` via `--additional-config`, and expanding `cudagraph_capture_sizes` for larger batch concurrency |
| High Throughput<br>(3.5K → 1.5K) | Single-node | 8 (A3) | Qwen3-32B-W8A8 (PDMix) | For short-sequence high throughput with more NPUs, try adjusting `cudagraph_capture_sizes` and `pa_shape_list` for the higher concurrency range enabled by TP=8 |
| Long Context<br>(up to 135K) | Single-node | 4 (A3) | Qwen3-32B-W8A8 (PDMix) | For long-context scenarios, try adjusting yarn RoPE parameters (`--hf-overrides`), `--max-num-batched-tokens` for chunked prefill behavior, and enabling FlashComm for TP communication efficiency |
| Low Latency<br>(real-time/interactive) | Single-node | 8 (A3) | Qwen3-32B-W8A8 (PDMix) | For low-latency interactive scenarios, try using `cudagraph_mode: FULL_DECODE_ONLY`, tuning `cudagraph_capture_sizes` for small batch sizes, enabling `--async-scheduling`, and adjusting `--block-size` |

> **Note**: `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | #NPUs | TP | DP | BS | Concurrency | Max Context Length | MTP Speculation Num | FUSED_MC2 | EP Switch | FC+CP Switch | Async Scheduling |
|----------|---------------|-------|----|----|----|-------------|--------------------|---------------------|-----------|-----------|--------------|------------------|
| High Throughput | Server-D Node (TP=4) | 4 | 4 | 1 | - | - | 5500 | 3 (eagle3) | - | - | - | Enabled |
| High Throughput | Server-D Node (TP=8) | 8 | 8 | 1 | - | - | 5500 | 3 (eagle3) | - | - | - | Enabled |
| Long Context | Server-D Node | 4 | 4 | 1 | - | - | 135000 | 3 (eagle3) | - | - | - | Enabled |
| Low Latency | Server-D Node | 8 | 8 | 1 | - | - | 5500 | 3 (eagle3) | - | - | - | Enabled |

> **Note**: BS (Batch Size) and Concurrency values depend on the specific workload and request pattern. Refer to the reference configurations below for detailed `cudagraph_capture_sizes` and `pa_shape_list` settings.

For complete startup commands and parameter descriptions, please refer to the deployment examples in [Section 5](#5-online-service-deployment) and the reference configurations in [Section 9.3](#93-reference-configurations).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

For general performance tuning methods applicable to all models (e.g., compilation-config tuning, chunked prefill, block-size, and scheduling optimizations), please refer to:

- [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md)
- [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions

#### 9.2.2 Model-Specific Optimizations

In this section, we introduce the key optimization points that can significantly improve the performance of Qwen3 Dense models. These techniques aim to improve throughput and efficiency in various scenarios.

##### Optimizations Enabled by Default

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
|------------------------|--------------------|---------------------|
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges address-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| Zero-like Elimination | Removes unnecessary zero-tensor operations in Attention forward pass | Reduces memory footprint, improves matrix operation efficiency |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

##### Optimizations That Require Explicit Enabling

| Optimization Technique | Applicable Scenarios | Enablement Method | Technical Principle | Precautions |
|------------------------|----------------------|-------------------|--------------------|-------------|
| FlashComm_v1 | High-concurrency, Tensor Parallelism (TP) scenarios | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | Threshold protection: only takes effect when the actual number of tokens exceeds the threshold to avoid performance degradation in low-concurrency scenarios |
| Matmul-ReduceScatter Fusion | Large-scale distributed environments | Automatically enabled after enabling FlashComm_v1 | Fuses matrix multiplication and Reduce-Scatter operations to achieve pipelined parallel processing | Same as FlashComm_v1, has threshold protection |
| Weight Prefetch | MLP-intensive scenarios (Dense models) | `--additional-config '{"weight_prefetch_config":{"enabled":true}}'` | Utilizes vector computation time to prefetch MLP weights into L2 cache in advance | Requires coordination with prefetch buffer size adjustment |
| Asynchronous Scheduling | Large-scale models, high-concurrency scenarios | `--async-scheduling` | Non-blocking task scheduling to improve concurrent processing capability | Should be used in coordination with FullGraph optimization |

##### Optimization Highlights

Building on the example scenarios outlined earlier, the following points were most critical for achieving optimal performance:

**Prefetch Buffer Size**

Setting the right prefetch buffer size is essential for optimizing weight loading. The size of this buffer is directly related to the time that can be hidden by vector computations. To achieve a near-perfect overlap between the prefetch and computation streams, flexibly adjust the buffer size by profiling and observing the degree of overlap at different buffer sizes.

In the real-world scenario above (Qwen3-32B-W8A8, TP=4), setting the prefetch buffer size for the MLP `gate_up_proj` and `down_proj` to 18 MB allows the vector computations of RMSNorm and SiLU to effectively hide the prefetch stream, thereby accelerating the Matmul computations of the two linear layers.

**max-num-batched-tokens**

The `max-num-batched-tokens` parameter determines the maximum number of tokens that can be processed in a single batch. Setting this value too small can negatively impact end-to-end performance, as fewer tokens are processed per batch. Conversely, setting it too large increases the risk of OOM errors due to excessive memory consumption.

When chunked prefill is enabled, also account for the accumulation of decode tokens. If the value is set too small, a single request may be chunked multiple times, and during the early stages of inference, a batch may contain only a small number of decode tokens, resulting in end-to-end throughput falling short of expectations.

**cudagraph_capture_sizes**

The `cudagraph_capture_sizes` parameter controls the granularity of graph captures during inference. If this list is not manually specified, it will be filled with a series of evenly distributed values, which typically ensures good performance. However, manually specifying the values yields better results, because if the batch size falls between two sizes, the framework will automatically pad the token count to the larger size, often causing actual performance to deviate from expectations.

When adjusting benchmark request concurrency, always ensure that the concurrency is actually included in the `cudagraph_capture_sizes` list. This way, during the decode phase, padding operations are essentially avoided, ensuring reliable experimental data.

:::{note}
If FlashComm_v1 is enabled, the values in this list must be integer multiples of the TP size. Any values that do not meet this condition will be automatically filtered out. It is recommended to incrementally add concurrency based on the TP size after enabling FlashComm_v1.
:::

### 9.3 Reference Configurations

This section provides complete, annotated launch commands for representative model variants. These configurations have been validated in production environments and can serve as a starting point for your deployment.

#### 9.3.1 Qwen3-8B (BF16) — General Purpose / Low Latency

This configuration targets low-latency serving with TP=2, FlashComm, and speculative decoding on 2 NPUs:

```bash
# NPU device selection
export ASCEND_RT_VISIBLE_DEVICES=0,1

# Memory allocator configuration
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Operator dispatch pipeline optimization
export TASK_QUEUE_ENABLE=1

# AIVector core for ROCE communication scheduling
export HCCL_OP_EXPANSION_MODE="AIV"

# Enable FlashComm_v1 optimization
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# [Optional] System-level performance tuning (requires root)
# echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# sysctl -w vm.swappiness=0
# sysctl -w kernel.numa_balancing=0
# sysctl kernel.sched_migration_cost_ns=50000

# [Optional] jemalloc for better memory allocation performance
# Ubuntu:
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# openEuler:
# export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

# [Optional] Torch profiler for debugging
# export VLLM_TORCH_PROFILER_DIR="./profile/online"
# export VLLM_TORCH_PROFILER_WITH_STACK=0

vllm serve /mnt/share/Qwen3-8B \
    --served-model-name qwen3 \
    --trust-remote-code \
    --distributed-executor-backend mp \
    --tensor-parallel-size 2 \
    --max-model-len 8000 \
    --max-num-batched-tokens 40960 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 8, 16, 24, 32, 48, 64, 72, 76, 96, 128, 144, 160, 192]}' \
    --speculative_config '{"method": "eagle3", "model": "/mnt/share/weight/qwen3_8b_eagle3", "enforce_eager": true, "num_speculative_tokens": 3}' \
    --port 8153 \
    --block-size 128 \
    --gpu-memory-utilization 0.85
```

:::{note}
- Replace `/mnt/share/Qwen3-8B` with your local model path.
- `cudagraph_capture_sizes` is tuned for this specific scenario. Adjust the values based on your target concurrency (see [9.2.2](#922-model-specific-optimizations)).
- Speculative decoding with eagle3 significantly reduces time-to-first-token latency. The eagle3 model path must point to a compatible draft model. Remove `--speculative_config` if not needed.
- Disabling prefix caching (`--no-enable-prefix-caching`) avoids the overhead of cache management in high-throughput scenarios.
:::

#### 9.3.2 Qwen3-32B-W8A8-PDMix — Max Throughput with Speculative Decoding

This configuration targets maximum throughput with W8A8 PD-Mix quantization (dynamic W8A8 for prefill, static W8A8 for decode), FlashComm, weight prefetch, and eagle3 speculative decoding on 4 NPUs:

```bash
# NPU device selection
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Memory allocator configuration
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Operator dispatch pipeline optimization
export TASK_QUEUE_ENABLE=1

# AIVector core for ROCE communication scheduling
export HCCL_OP_EXPANSION_MODE="AIV"

# Enable FlashComm_v1 optimization
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/qwen3-32b-pdmix \
    --served-model-name qwen3 \
    --trust-remote-code \
    --distributed-executor-backend mp \
    --tensor-parallel-size 4 \
    --max-model-len 5500 \
    --max-num-batched-tokens 40960 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 64, 72, 76, 80, 96, 100, 120, 140, 144, 160, 192, 216, 240, 252, 288, 320, 336, 360, 384, 400, 408, 416, 420, 432, 480, 540, 576, 600]}' \
    --additional-config '{"pa_shape_list": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256], "weight_prefetch_config": {"enabled": true}}' \
    --speculative_config '{"method": "eagle3", "model": "/mnt/share/weights/qwen-eagle3/qwen3_32B_rot/", "enforce_eager": true, "num_speculative_tokens": 3}' \
    --port 2000 \
    --block-size 128 \
    --gpu-memory-utilization 0.9
```

:::{note}
- Replace `/mnt/share/qwen3-32b-pdmix` with your local W8A8 PD-Mix quantized model path.
- The `W8A8_MIX` quantization method uses dynamic W8A8 for the prefill phase and static W8A8 for the decode phase, providing an optimal balance between accuracy and throughput.
- `cudagraph_capture_sizes` and `pa_shape_list` are tuned for this specific deployment. The `pa_shape_list` covers a broad range of batch sizes (32–256) to ensure the PA operator is used where FIA performance is suboptimal. See [9.2.2](#922-model-specific-optimizations) for tuning guidance.
- FlashComm_v1 is threshold-protected and only activates when the token count exceeds the threshold. The `cudagraph_capture_sizes` values must be integer multiples of TP=4 when FlashComm_v1 is enabled.
- `weight_prefetch_config` enables MLP weight prefetching, which overlaps weight loading with vector computation to hide memory latency.
- Disabling prefix caching (`--no-enable-prefix-caching`) avoids the overhead of cache management in high-throughput scenarios.
:::

#### 9.3.3 Qwen3-32B-W4A4 — Memory-Efficient Deployment

For memory-constrained environments, W4A4 quantization enables running the 32B model on a single NPU:

```bash
vllm serve /home/models/Qwen3-32B-w4a4 \
    --served-model-name "qwen3-32b-w4a4" \
    --max-model-len 4096 \
    --quantization ascend
```

:::{note}
- For model quantization and conversion steps, see [Qwen3-32B-W4A4](Qwen3-32B-W4A4.md).
- Pre-converted W4A4 weights are available at [Qwen3-32B-W4A4 on ModelScope](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4).
- Adjust `--max-model-len` according to your use case and available memory.
:::

#### 9.3.4 Qwen3-32B-W8A8 (PDMix) — Low Latency

For real-time or interactive scenarios with tight latency requirements:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

vllm serve /mnt/share/qwen3-32b-pdmix \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,408,416,420,432,480,540,576,600]}' \
  --speculative_config '{"method": "eagle3", "model":"/mnt/share/weights/qwen-eagle3/qwen3_32B_rot/", "enforce_eager": true, "num_speculative_tokens": 3}' \
  --port 2000 \
  --block-size 128 \
  --gpu-memory-utilization 0.9
```

#### 9.3.5 Qwen3-32B-W8A8 (PDMix) — Max Throughput

For throughput-oriented scenarios with high concurrency:

**Option A: TP=4 (4×NPUs)**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/qwen3-32b-pdmix \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,408,416,420,432,480,540,576,600,640,660,680,700,720]}' \
  --additional-config '{"weight_prefetch_config":{"enabled":true}, "pa_shape_list":[32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256]}'\
  --speculative_config '{"method": "eagle3", "model":"/mnt/share/weights/qwen-eagle3/qwen3_32B_rot/", "enforce_eager": true, "num_speculative_tokens": 3}' \
  --host 141.61.133.127 \
  --port 2000 \
  --block-size 128 \
  --gpu-memory-utilization 0.9
```

**Option B: TP=8 (8×NPUs)**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/qwen3-32b-pdmix \
  --served-model-name qwen3 \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 8 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,64,72,76,80,96,100,120,140,144,160,192,216,240,252,288,320,336,360,384,400,416,420,432,480,540,580,600,620,800,840,860,880]}' \
  --additional-config '{"weight_prefetch_config":{"enabled":true}, "pa_shape_list":[32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256]}'\
  --speculative_config '{"method": "eagle3", "model":"/mnt/share/weights/qwen-eagle3/qwen3_32B_rot/", "enforce_eager": true, "num_speculative_tokens": 3}' \
  --port 2000 \
  --block-size 128 \
  --gpu-memory-utilization 0.9
```

#### 9.3.6 Qwen3-32B-W8A8 (PDMix) — Long Context

For scenarios requiring extended context length (up to 135K tokens):

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/qwen3-32b-pdmix \
  --host 0.0.0.0 \
  --port 8004 \
  --served-model-name qwen \
  --trust-remote-code \
  --seed 1024 \
  --max-model-len 135000 \
  --max-num-batched-tokens 40960 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend "mp" \
  --async-scheduling \
  --no-enable-prefix-caching \
  --speculative_config '{"method": "eagle3", "model":"/mnt/share/weights/qwen-eagle3/qwen3_32B_rot/", "num_speculative_tokens": 3}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":131072}}' \
  --gpu-memory-utilization 0.9 \
  --quantization ascend
```

:::{note}
- **Low Latency**: No FlashComm or weight prefetch is needed — the focus is on minimizing per-request latency with FullGraph (`cudagraph_mode: FULL_DECODE_ONLY`) and tuned `cudagraph_capture_sizes`.
- **Max Throughput**: Enable FlashComm (`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`) and weight prefetch (`"weight_prefetch_config":{"enabled":true}` in `--additional-config`). Use `pa_shape_list` in `--additional-config` to optimize page attention for the target concurrency range.
- **Long Context**: FlashComm is enabled for TP communication efficiency. The yarn RoPE extension (`--hf-overrides`) is required for context lengths beyond the original 131K training limit.
- Adjust paths (`/mnt/share/...`, `--host`, `--port`) to match your environment.
:::

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section only covers issues specific to Qwen3 Dense models.

### Q: How do I choose between single-node and multi-node deployment?

Single-node deployment is recommended when the model fits within the memory of a single node's NPUs. For models like Qwen3-32B (BF16), which requires 4 × 64G cards, multi-NPU within a single node (TP) is sufficient. Multi-node deployment is only needed when the total NPU count exceeds a single node's capacity.

### Q: What quantization method should I use?

- **BF16**: Best accuracy, highest memory footprint. Use for accuracy-critical applications or when memory is sufficient.
- **W8A8**: Good balance of accuracy and memory reduction. Use for large models (e.g., 32B) on memory-constrained hardware.
- **W4A8/W4A4**: Maximum memory reduction. Suitable for deploying larger models on smaller hardware configurations, with some accuracy trade-off.

### Q: When should I enable FlashComm_v1?

Enable FlashComm_v1 (`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`) when using Tensor Parallelism (TP ≥ 2) with high concurrency. It is threshold-protected and will not activate in low-concurrency scenarios where it could degrade performance.

### Q: What is the difference between FIA and PA operators for attention?

FIA (Flash Attention) is the default attention operator in vLLM-Ascend. In some batch-size settings (particularly medium concurrency), FIA may exhibit suboptimal performance. The PA (Page Attention) operator can be manually enabled via `pa_shape_list` in `--additional-config`. When the runtime batch size matches a value in `pa_shape_list`, the framework switches to PA. This is a temporary tuning knob — future FIA optimizations will make this parameter obsolete.
