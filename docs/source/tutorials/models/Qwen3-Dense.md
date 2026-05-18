# Qwen3-Dense (Qwen3-0.6B/1.7B/4B/8B/14B/32B, W8A8, W4A8, W4A4)

## 1 Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support. The Dense variants covered in this document include Qwen3-0.6B, 1.7B, 4B, 8B, 14B, and 32B, along with their quantized versions (W8A8, W4A8, W4A4) optimized for Ascend NPU deployment.

This document will demonstrate the main validation steps for Qwen3 Dense models in the vLLM-Ascend environment, including supported features, environment preparation, model quantization, single-node and multi-node deployment, as well as accuracy and performance evaluation. By tailoring service-level configurations to fit different use cases, you can ensure optimal performance across various scenarios.

The Qwen3 Dense models are first supported in v0.8.4rc2. W4A8 quantization is supported since v0.9.1rc2, and W4A4 is supported since v0.11.0rc1. This document is validated and written based on **vLLM-Ascend v0.13.0**. All **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use v0.13.0 or a later version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Environment Preparation

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

### 3.2 Verify Multi-node Communication (Optional)

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

If you prefer not to use the Docker image, you can build from source. Refer to [installation](../../installation.md) for details.

If deploying a multi-node environment, set up the environment on each node.

### 4.2 Model Quantization (Optional)

If you wish to quantize the model yourself, refer to the [MindStudio Model Slim documentation](https://gitcode.com/Ascend/msit) for W4A8 and W4A4 quantization procedures. Pre-converted quantized weights are also available for download (see [3.1 Model Weight](#31-model-weight)).

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

### 5.2 Multi-NPU Online Deployment

Multi-NPU deployment leverages Tensor Parallelism (TP) to distribute the model across multiple NPUs, suitable for large models that exceed single-card memory or require higher throughput.

The following example demonstrates best practices for Qwen3-32B-W8A8 on an Atlas 800I A3 (64G × 16) with DP=1, TP=4, targeting optimal throughput at batch_size=72 with fixed-length input of 3.5K and output of 1.5K.

:::{note}
If the machine is an **Atlas 800I A2 (64G × 8)**, the deployment approach stays identical — adjust `--device` mappings and TP size accordingly.
:::

```{code-block} bash
   :substitutions:

# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8113:8113 \
    -it $IMAGE bash
```

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

:::{note}
- `vllm-ascend/Qwen3-32B-W8A8` is the default model path, replace this with your actual path.
- If the model is not a quantized model, remove the `--quantization ascend` parameter.
- **[Optional]** `--additional-config '{"pa_shape_list":[48,64,72,80]}'`: `pa_shape_list` specifies the batch sizes where you want to switch to the PA operator. This is a temporary tuning knob. Currently, the attention operator dispatch defaults to the FIA operator. In some batch-size (concurrency) settings, FIA may have suboptimal performance. By setting `pa_shape_list`, when the runtime batch size matches one of the listed values, vLLM-Ascend will replace FIA with the PA operator to prevent performance degradation. In the future, FIA will be optimized for these scenarios and this parameter will be removed.
- If ultimate performance is desired, the `cudagraph_capture_sizes` parameter can be enabled (see [10.2 Optimization Highlights](#102-optimization-highlights)). Example for batch_size=72: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,8,24,48,60,64,72,76]}'`.
:::

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

### Using AISBench

For details, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

For reference, the following are the accuracy results of `Qwen3-32B-W8A8` on `vllm-ascend:0.11.0rc2`:

| dataset | version | metric   | mode | task name                           | vllm-api-general-chat |
|---------|---------|----------|------|--------------------------------------|-----------------------|
| gsm8k   | -       | accuracy | gen  | gsm8k_gen_0_shot_noncot_chat_prompt  | 96.44                 |
| math500 | -       | accuracy | gen  | math500_gen_0_shot_cot_chat_prompt   | 97.60                 |
| aime    | -       | accuracy | gen  | aime2024_gen_0_shot_chat_prompt      | 76.67                 |

### Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example, run the accuracy evaluation in online mode:

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval`:

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/vllm-ascend/Qwen3-32B-W8A8,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

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

## 9 Best Practices

The following configurations are recommended for different deployment scenarios to achieve optimal performance with Qwen3 Dense models.

### Long Sequence

For scenarios with long input/output sequences (e.g., document processing, long-form generation):

- Set `--max-model-len` to accommodate the target sequence length (e.g., 32768 or higher for supported models).
- Use chunked prefill to manage long prefill sequences without exceeding memory limits.
- Adjust `--max-num-batched-tokens` to balance the number of tokens processed per batch, ensuring sufficient tokens to fill the batch while avoiding OOM.
- Enable `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` to reduce scheduling overhead during long decodes.

### Low Latency

For interactive or real-time applications requiring minimal response time:

- Enable `--async-scheduling` and FullGraph optimization: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`.
- Manually specify `cudagraph_capture_sizes` to cover the target concurrency range and avoid padding overhead during decode (see [10.2](#102-optimization-highlights)).
- Set `TASK_QUEUE_ENABLE=1` to optimize operator dispatch pipeline.
- Use `--block-size 128` to reduce scheduling granularity.
- Enable jemalloc if available on your system (see [5.2](#52-multi-npu-online-deployment)).

### High Throughput

For batch processing or serving scenarios maximizing tokens-per-second:

- Enable FlashComm_v1: `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` (see [10.1.2](#1012-advanced-optimizations-require-explicit-enablement)).
- Enable weight prefetch: `--additional-config '{"weight_prefetch_config":{"enabled":true}}'`.
- Set `export HCCL_OP_EXPANSION_MODE="AIV"` to enable AIVector core for ROCE communication scheduling.
- Use a larger `--max-num-batched-tokens` value (e.g., 40960) to maximize batch utilization.
- Enable chunked prefill to interleave prefill and decode tokens within a batch.
- Set `--gpu-memory-utilization 0.9` to maximize available KV cache memory.

## 10 Performance Tuning

### 10.1 Key Optimization Points

In this section, we introduce the key optimization points that can significantly improve the performance of Qwen3 Dense models. These techniques aim to improve throughput and efficiency in various scenarios.

#### 10.1.1 Basic Optimizations

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
|------------------------|--------------------|---------------------|
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges address-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| Zero-like Elimination | Removes unnecessary zero-tensor operations in Attention forward pass | Reduces memory footprint, improves matrix operation efficiency |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### 10.1.2 Advanced Optimizations (Require Explicit Enablement)

| Optimization Technique | Technical Principle | Enablement Method | Applicable Scenarios | Precautions |
|------------------------|--------------------|-------------------|----------------------|-------------|
| FlashComm_v1 | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | High-concurrency, Tensor Parallelism (TP) scenarios | Threshold protection: only takes effect when the actual number of tokens exceeds the threshold to avoid performance degradation in low-concurrency scenarios |
| Matmul-ReduceScatter Fusion | Fuses matrix multiplication and Reduce-Scatter operations to achieve pipelined parallel processing | Automatically enabled after enabling FlashComm_v1 | Large-scale distributed environments | Same as FlashComm_v1, has threshold protection |
| Weight Prefetch | Utilizes vector computation time to prefetch MLP weights into L2 cache in advance | `--additional-config '{"weight_prefetch_config":{"enabled":true}}'` | MLP-intensive scenarios (Dense models) | Requires coordination with prefetch buffer size adjustment |
| Asynchronous Scheduling | Non-blocking task scheduling to improve concurrent processing capability | `--async-scheduling` | Large-scale models, high-concurrency scenarios | Should be used in coordination with FullGraph optimization |

### 10.2 Optimization Highlights

Building on the example scenarios outlined earlier, the following points were most critical for achieving optimal performance:

#### Prefetch Buffer Size

Setting the right prefetch buffer size is essential for optimizing weight loading. The size of this buffer is directly related to the time that can be hidden by vector computations. To achieve a near-perfect overlap between the prefetch and computation streams, flexibly adjust the buffer size by profiling and observing the degree of overlap at different buffer sizes.

In the real-world scenario above (Qwen3-32B-W8A8, TP=4), setting the prefetch buffer size for the MLP `gate_up_proj` and `down_proj` to 18 MB allows the vector computations of RMSNorm and SiLU to effectively hide the prefetch stream, thereby accelerating the Matmul computations of the two linear layers.

#### max-num-batched-tokens

The `max-num-batched-tokens` parameter determines the maximum number of tokens that can be processed in a single batch. Setting this value too small can negatively impact end-to-end performance, as fewer tokens are processed per batch. Conversely, setting it too large increases the risk of OOM errors due to excessive memory consumption.

When chunked prefill is enabled, also account for the accumulation of decode tokens. If the value is set too small, a single request may be chunked multiple times, and during the early stages of inference, a batch may contain only a small number of decode tokens, resulting in end-to-end throughput falling short of expectations.

#### cudagraph_capture_sizes

The `cudagraph_capture_sizes` parameter controls the granularity of graph captures during inference. If this list is not manually specified, it will be filled with a series of evenly distributed values, which typically ensures good performance. However, manually specifying the values yields better results, because if the batch size falls between two sizes, the framework will automatically pad the token count to the larger size, often causing actual performance to deviate from expectations.

When adjusting benchmark request concurrency, always ensure that the concurrency is actually included in the `cudagraph_capture_sizes` list. This way, during the decode phase, padding operations are essentially avoided, ensuring reliable experimental data.

:::{note}
If FlashComm_v1 is enabled, the values in this list must be integer multiples of the TP size. Any values that do not meet this condition will be automatically filtered out. It is recommended to incrementally add concurrency based on the TP size after enabling FlashComm_v1.
:::

## 11 FAQ

### Q: How do I choose between single-node and multi-node deployment?

Single-node deployment is recommended when the model fits within the memory of a single node's NPUs. For models like Qwen3-32B (BF16), which requires 4 × 64G cards, multi-NPU within a single node (TP) is sufficient. Multi-node deployment is only needed when the total NPU count exceeds a single node's capacity.

### Q: What quantization method should I use?

- **BF16**: Best accuracy, highest memory footprint. Use for accuracy-critical applications or when memory is sufficient.
- **W8A8**: Good balance of accuracy and memory reduction. Use for large models (e.g., 32B) on memory-constrained hardware.
- **W4A8/W4A4**: Maximum memory reduction. Suitable for deploying larger models on smaller hardware configurations, with some accuracy trade-off.

### Q: When should I enable FlashComm_v1?

Enable FlashComm_v1 (`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`) when using Tensor Parallelism (TP ≥ 2) with high concurrency. It is threshold-protected and will not activate in low-concurrency scenarios where it could degrade performance.
