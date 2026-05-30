# Qwen3-Coder-30B-A3B

## 1 Introduction

Qwen3-Coder-30B-A3B is a Mixture-of-Experts (MoE) model in the Qwen3 Coder series, featuring 30B total parameters with 3B activated per token. Built upon the Qwen3 base architecture, it delivers significant optimizations in agentic coding, extended context support of up to 1M tokens, and versatile function calling capabilities.

This document will demonstrate the main validation steps for Qwen3-Coder-30B-A3B in the vLLM-Ascend environment, including supported features, environment preparation, single-node deployment, as well as accuracy and performance evaluation.

The Qwen3-Coder-30B-A3B model is first supported in **v0.10.0rc1**. This document is validated and written based on **vLLM-Ascend v0.13.0**. All **v0.13.0 and later versions** can run stably.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Environment Preparation

### 3.1 Model Weight

The following model variant is available. It is recommended to download the model weight to a shared directory across multiple nodes (e.g., `/root/.cache/`).

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| Qwen3-Coder-30B-A3B-Instruct (BF16) | 1 Atlas 800I A3 (64G × 16) or 1 Atlas 800I A2 (64G × 8) | [Download](https://modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct) |

These are the recommended numbers of cards, which can be adjusted according to the actual situation.

### 3.2 Verify Multi-node Communication (Optional)

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image for Qwen3 Coder models.

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

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. For the Qwen3-Coder-30B-A3B MoE model, Expert Parallelism (EP) is required to distribute experts across NPUs.

```bash
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 32768
```

:::{note}
- For an Atlas A2 with 64 GB NPU memory, `--tensor-parallel-size` should be at least 2; for 32 GB memory, at least 4.
- `--enable-expert-parallel` enables Expert Parallelism, which is required for MoE models. vLLM does not support mixing ETP and EP; MoE layers use either pure EP or pure TP.
- Qwen3-Coder supports up to 1M token context length. Adjust `--max-model-len` based on your use case and available memory.
- Replace `Qwen/Qwen3-Coder-30B-A3B-Instruct` with your local model path.
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
    "def fibonacci(n):",
    "Write a function to reverse a linked list.",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
          tensor_parallel_size=4,
          distributed_executor_backend="mp",
          max_model_len=32768,
          enable_expert_parallel=True)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
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

For reference, the following are the accuracy results of `Qwen3-Coder-30B-A3B-Instruct` on `vllm-ascend:v0.11.0rc0`:

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| openai_humaneval | f4a973 | humaneval_pass@1 | gen | 94.51 |

### Using Language Model Evaluation Harness

Using the `humaneval` dataset as an example, run the accuracy evaluation in online mode:

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

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take `serve` as an example:

```shell
vllm bench serve \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3-coder \
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

The following configurations are recommended for different deployment scenarios to achieve optimal performance with Qwen3-Coder-30B-A3B.

### 9.1 Long Sequence

For coding scenarios with long context (e.g., repository-level code generation):

- Set `--max-model-len` to accommodate the target context length. Qwen3-Coder supports up to 1M tokens; adjust based on available memory.
- Use a moderate `--max-num-batched-tokens` value (e.g., 16384) to balance prefill chunking with memory usage.
- Enable `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` to reduce scheduling overhead.
- Consider using `--hf-overrides` with yarn rope-scaling for context lengths beyond the default 32K.

### 9.2 Low Latency

For interactive coding assistants requiring minimal response time:

- Enable `--async-scheduling` and FullGraph optimization.
- Set `export TASK_QUEUE_ENABLE=1` to optimize operator dispatch pipeline.
- Set `export HCCL_OP_EXPANSION_MODE="AIV"` to enable AIVector core for ROCE communication scheduling.

### 9.3 High Throughput

For batch code processing or serving scenarios maximizing tokens-per-second:

- Enable FlashComm_v1: `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.
- Set `export VLLM_ASCEND_ENABLE_NZ=2` to enable FRACTAL_NZ format for all weight types.
- Set `export HCCL_BUFFSIZE=1024` to optimize HCCL communication buffer size.
- Use a larger `--max-num-seqs` to maximize concurrent requests.
- Set `--gpu-memory-utilization 0.9` or higher to maximize KV cache.

### 9.4 Reference Configuration

This configuration targets high-throughput coding assistance on 4 NPUs:

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

# Enable FRACTAL_NZ format for all weight types
export VLLM_ASCEND_ENABLE_NZ=2

# Enable FlashComm_v1 optimization
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3-coder \
    --trust-remote-code \
    --max-num-seqs 64 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.9
```

:::{note}
- Replace `Qwen/Qwen3-Coder-30B-A3B-Instruct` with your local model path.
- `VLLM_ASCEND_ENABLE_NZ=2` forces FRACTAL_NZ weight format for fp16/bf16 and quantized weights, improving compute efficiency. The default value is `1` (quantized weights only).
- Qwen3-Coder supports up to 1M context. For longer contexts, add yarn rope-scaling via `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`.
- Adjust `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` based on your use case and available NPU memory.
:::

## 10 Performance Tuning

### 10.1 Key Optimization Points

In this section, we introduce the key optimization points that can significantly improve the performance of Qwen3-Coder-30B-A3B.

#### 10.1.1 Basic Optimizations

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
|------------------------|--------------------|---------------------|
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### 10.1.2 Advanced Optimizations (Require Explicit Enablement)

| Optimization Technique | Technical Principle | Enablement Method | Applicable Scenarios | Precautions |
|------------------------|--------------------|-------------------|----------------------|-------------|
| FlashComm_v1 | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | High-concurrency, TP > 1 scenarios with MoE models | Currently only supported for MoE in scenarios where TP > 1 |
| NZ Weight Format | Converts weight tensors to FRACTAL_NZ format for improved compute efficiency | `export VLLM_ASCEND_ENABLE_NZ=2` | All scenarios with fp16/bf16 or quantized weights | Default is `1` (quantized only). Set to `2` for maximum performance; set to `0` for RL scenarios |
| Asynchronous Scheduling | Non-blocking task scheduling to improve concurrent processing capability | `--async-scheduling` | Large-scale models, high-concurrency scenarios | Should be used in coordination with FullGraph optimization |

### 10.2 Optimization Highlights

#### Long Context Tuning

Qwen3-Coder supports up to 1M token context. For contexts beyond the default 32K, use yarn rope-scaling:
- For vLLM >= v0.12.0: `--hf-overrides '{"rope_parameters": {"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`
- For vLLM < v0.12.0: `--rope_scaling '{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}'`

This is not required for model variants that already support long contexts natively.

#### max-num-batched-tokens

For MoE models like Qwen3-Coder-30B-A3B, a moderate value (e.g., 16384) helps balance prefill chunking with memory usage. Setting this value too high increases activation memory pressure; setting it too low causes excessive chunking and reduces throughput. For coding scenarios with typically shorter outputs, values in the range 8192–16384 are usually appropriate.

#### Expert Parallelism Configuration

MoE models require `--enable-expert-parallel` to distribute experts across NPUs. The tensor-parallel-size should be set based on available NPU memory per card. For Atlas A2 with 64 GB memory, TP ≥ 2; for 32 GB memory, TP ≥ 4.

## 11 FAQ

### Q: What hardware is required for Qwen3-Coder-30B-A3B?

The model requires 1 Atlas 800I A3 (64G × 16) node or 1 Atlas 800I A2 (64G × 8) node for BF16 inference.

### Q: How do I enable long context (beyond 32K)?

Use yarn rope-scaling via `--hf-overrides` (vLLM >= v0.12.0) or `--rope_scaling` (vLLM < v0.12.0). See [10.2 Optimization Highlights](#102-optimization-highlights) for exact parameters. Qwen3-Coder supports up to 1M tokens context length.

### Q: What makes Qwen3-Coder different from Qwen3-30B-A3B?

Qwen3-Coder-30B-A3B shares the same MoE architecture (30B/3B) as Qwen3-30B-A3B but is specifically fine-tuned for coding tasks, with optimizations for agentic coding, function calling, and extended context support up to 1M tokens.
