# Qwen3-30B-A3B

## 1 Introduction

Qwen3-30B-A3B is a Mixture-of-Experts (MoE) model in the Qwen3 series, featuring 30B total parameters with 3B activated per token. The sparse MoE architecture enables efficient training and inference, delivering strong performance across reasoning, instruction-following, and agent capabilities while maintaining lower computational cost compared to dense models of similar capability.

This document will demonstrate the main validation steps for Qwen3-30B-A3B in the vLLM-Ascend environment, including supported features, environment preparation, single-node and multi-node deployment, as well as accuracy and performance evaluation.

The Qwen3-30B-A3B model is first supported in v0.8.4rc2. This document is validated and written based on **vLLM-Ascend v0.13.0**. All **v0.13.0 and later versions** can run stably.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Environment Preparation

### 3.1 Model Weight

The following model variants are available. It is recommended to download the model weight to a shared directory across multiple nodes (e.g., `/root/.cache/`).

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-30B-A3B (BF16) | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B) |
| Qwen3-30B-A3B-W8A8 | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelscope.cn/models/vllm-ascend/Qwen3-30B-A3B-W8A8) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 MoE models.

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

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node, suitable for development, testing, and small-to-medium scale inference scenarios. For the Qwen3-30B-A3B MoE model, Expert Parallelism (EP) is required to distribute experts across NPUs.

#### BF16 Model

```bash
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name qwen3 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 4096
```

#### W8A8 Quantized Model

```bash
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve vllm-ascend/Qwen3-30B-A3B-W8A8 \
    --served-model-name qwen3 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --quantization ascend
```

:::{note}
- `vllm-ascend/Qwen3-30B-A3B-W8A8` is the default model path, replace this with your actual path.
- For an Atlas A2 with 64 GB NPU memory, `--tensor-parallel-size` should be at least 2; for 32 GB memory, at least 4.
- `--enable-expert-parallel` enables Expert Parallelism, which is required for MoE models. vLLM does not support mixing ETP and EP; MoE layers use either pure EP or pure TP.
- If the model is not a quantized model, remove the `--quantization ascend` parameter.
:::

### 5.2 Offline Inference

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

llm = LLM(model="Qwen/Qwen3-30B-A3B",
          tensor_parallel_size=4,
          distributed_executor_backend="mp",
          max_model_len=4096,
          enable_expert_parallel=True)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```

:::{note}
- If using a quantized model, add `quantization="ascend"` to the LLM constructor.
- Replace `Qwen/Qwen3-30B-A3B` with your local model path.
:::

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
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

### Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example, run the accuracy evaluation in online mode:

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval`:

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/vllm-ascend/Qwen3-30B-A3B-W8A8,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
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
    --model vllm-ascend/Qwen3-30B-A3B-W8A8 \
    --served-model-name qwen3 \
    --port 8000 \
    --dataset-name random \
    --random-input 200 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

After several minutes, you will get the performance evaluation result.

## 9 Best Practices

The following configurations are recommended for different deployment scenarios to achieve optimal performance with Qwen3-30B-A3B.

### 9.1 Long Sequence

For scenarios with long input/output sequences:

- Set `--max-model-len` to accommodate the target sequence length (e.g., 32768 or higher).
- Use a moderate `--max-num-batched-tokens` value (e.g., 16384) to balance prefill chunking and memory usage.
- Enable `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` to reduce scheduling overhead during long decodes.

### 9.2 Low Latency

For interactive or real-time applications requiring minimal response time:

- Enable `--async-scheduling` and FullGraph optimization: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`.
- Set `export TASK_QUEUE_ENABLE=1` to optimize operator dispatch pipeline.
- Use speculative decoding (eagle3) to reduce time-to-first-token latency.
- Set `export HCCL_OP_EXPANSION_MODE="AIV"` to enable AIVector core for ROCE communication scheduling.

### 9.3 High Throughput

For batch processing or serving scenarios maximizing tokens-per-second:

- Enable FlashComm_v1: `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.
- Set `export VLLM_ASCEND_ENABLE_NZ=2` to enable FRACTAL_NZ format for all weight types, improving compute efficiency.
- Set `export HCCL_BUFFSIZE=1024` to optimize HCCL communication buffer size.
- Use a larger `--max-num-seqs` value (e.g., 100) to maximize concurrent requests.
- Set `--gpu-memory-utilization 0.95` to maximize available KV cache memory.

### 9.4 Reference Configuration

This configuration targets maximum throughput with W8A8 quantization, FlashComm, and eagle3 speculative decoding on 4 NPUs:

```bash
# Enable vLLM V1 engine
export VLLM_USE_V1=1

# NPU device selection
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# AIVector core for ROCE communication scheduling
export HCCL_OP_EXPANSION_MODE="AIV"

# HCCL communication buffer size optimization
export HCCL_BUFFSIZE=1024

# Thread affinity configuration
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

# Memory allocator configuration
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Enable FRACTAL_NZ format for all weight types (improves compute efficiency)
export VLLM_ASCEND_ENABLE_NZ=2

# Enable FlashComm_v1 optimization
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/weight/Qwen3-30B-A3B-W8A8 \
    --served-model-name qwen3 \
    --trust-remote-code \
    --max-num-seqs 100 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --port 1999 \
    --distributed-executor-backend mp \
    --async-scheduling \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --speculative_config '{"method": "eagle3", "model": "/mnt/share/weight/Qwen3-30B-A3B-EAGLE3", "num_speculative_tokens": 3}'
```

:::{note}
- Replace `/mnt/share/weight/Qwen3-30B-A3B-W8A8` with your local W8A8 quantized model path.
- `VLLM_ASCEND_ENABLE_NZ=2` forces FRACTAL_NZ weight format for fp16/bf16 and quantized weights, which can improve compute efficiency. The default value is `1` (quantized weights only). Set to `0` to disable NZ format entirely.
- Speculative decoding with eagle3 reduces time-to-first-token latency. The eagle3 model path must point to a compatible draft model. Remove `--speculative_config` if not needed.
- `--max-num-seqs 100` sets the maximum number of concurrent requests. Adjust based on your workload and available KV cache memory.
:::

## 10 Performance Tuning

### 10.1 Key Optimization Points

In this section, we introduce the key optimization points that can significantly improve the performance of Qwen3-30B-A3B.

#### 10.1.1 Basic Optimizations

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
|------------------------|--------------------|---------------------|
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges address-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### 10.1.2 Advanced Optimizations (Require Explicit Enablement)

| Optimization Technique | Technical Principle | Enablement Method | Applicable Scenarios | Precautions |
|------------------------|--------------------|-------------------|----------------------|-------------|
| FlashComm_v1 | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | High-concurrency, Tensor Parallelism (TP) scenarios with MoE models | Currently only supported for MoE in scenarios where TP > 1 |
| NZ Weight Format | Converts weight tensors to FRACTAL_NZ format for improved compute efficiency | `export VLLM_ASCEND_ENABLE_NZ=2` | All scenarios with fp16/bf16 or quantized weights | Default is `1` (quantized only). Set to `2` for maximum performance; set to `0` for RL scenarios |
| Asynchronous Scheduling | Non-blocking task scheduling to improve concurrent processing capability | `--async-scheduling` | Large-scale models, high-concurrency scenarios | Should be used in coordination with FullGraph optimization |
| Speculative Decoding | Uses a lightweight draft model to predict future tokens, reducing decode latency | `--speculative_config '{"method": "eagle3", ...}'` | Low-latency scenarios | Requires a compatible eagle3 draft model |

### 10.2 Optimization Highlights

#### max-num-batched-tokens

The `max-num-batched-tokens` parameter determines the maximum number of tokens that can be processed in a single batch. For MoE models like Qwen3-30B-A3B, a moderate value (e.g., 16384) helps balance prefill chunking with memory usage. Setting this value too high increases activation memory pressure; setting it too low causes excessive chunking and reduces throughput.

#### max-num-seqs

`--max-num-seqs` indicates the maximum number of requests that each DP group can process concurrently. Requests exceeding this limit will wait in the queue. When benchmarking performance, ensure `max-num-seqs` × `data-parallel-size` >= the actual target concurrency to avoid artificial queuing delays.

#### gpu-memory-utilization

`--gpu-memory-utilization` controls the proportion of HBM used for KV cache. During warm-up, vLLM profiles peak memory usage with an input of `max-num-batched-tokens` tokens, then calculates available KV cache as: `gpu-memory-utilization` × HBM size − peak memory. Higher values increase KV cache capacity but risk OOM. For MoE models, a value of 0.95 is typically safe on well-balanced EP configurations.

## 11 FAQ

### Q: What hardware is required for Qwen3-30B-A3B?

The BF16 model requires 1 Atlas 800I A3 (64G × 16) node or 1 Atlas 800I A2 (64G × 8) node. The W8A8 quantized version has similar hardware requirements but uses less memory per card.

### Q: Why is `--enable-expert-parallel` required?

Qwen3-30B-A3B is a Mixture-of-Experts model where different experts reside on different NPUs. Expert Parallelism (EP) distributes these experts across devices, which is required for the model to fit in memory and run efficiently. vLLM does not support mixing ETP and EP; MoE layers use either pure EP or pure TP.

### Q: When should I use `VLLM_ASCEND_ENABLE_NZ=2`?

Set `VLLM_ASCEND_ENABLE_NZ=2` when you want maximum throughput and your model uses fp16/bf16 weights. This forces FRACTAL_NZ weight format conversion for all weight types, improving compute efficiency. The default value `1` only converts quantized weights. Set to `0` for RL scenarios where NZ format may cause precision issues.
