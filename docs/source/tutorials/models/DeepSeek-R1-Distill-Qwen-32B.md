# DeepSeek-R1-Distill-Qwen-32B

## 1 Introduction

DeepSeek-R1-Distill-Qwen-32B is a 32B dense reasoning model distilled from DeepSeek-R1, based on the Qwen2.5-32B architecture. It inherits strong chain-of-thought reasoning abilities from DeepSeek-R1 and generates answers in a `<think>...</think>` reasoning format before providing the final response.

This document demonstrates the main validation steps for DeepSeek-R1-Distill-Qwen-32B in the vLLM-Ascend environment, including supported features, environment preparation, online service deployment, and accuracy evaluation on Atlas A2 inference products.

This document is validated and written based on **vLLM-Ascend v0.23.0**. All **v0.23.0 and later versions** can run stably. To use the latest features, it is recommended to use the latest release candidate or official version.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix. Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

It is recommended to download the model weight to a shared directory accessible to all nodes.

| Model | Hardware Requirement | Download |
|-------|---------------------|----------|
| DeepSeek-R1-Distill-Qwen-32B (BF16, ~65GB) | 4 Atlas A2 inference products (64GB) or more | [Download](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |

> **Note**: The BF16 weights are about 65GB in total. A single Atlas A2 card (64GB HBM) cannot hold the weights, so at least 2 cards are required. Tensor parallelism of 4 is the recommended and validated configuration.

### 3.2 Verify Multi-node Communication

If you need to deploy a multi-node environment, verify the multi-node communication according to [Verify Multi-node Communication Environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one Docker image.

```bash
docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|
```

Refer to [Qwen3 Dense](Qwen3-Dense.md#41-docker-image-installation) for the detailed `docker run` command for Atlas A2 inference products.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend following the [Installation Guide](../../installation.md).

To verify the installation:

```bash
pip show vllm-ascend
```

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

**Start the server:**

> The following command is an example configuration. Adjust the parameters based on your actual scenario.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
vllm serve your_model_path \
  --served-model-name deepseek-r1-distill-qwen-32b \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser deepseek_r1 \
  --port 8000
```

> **Note**:
> - `--reasoning-parser deepseek_r1` separates the `<think>...</think>` reasoning content from the final answer in the API response (`reasoning_content` vs `content`), which is required for clean downstream consumption and correct benchmark scoring.
> - For debug or first-time validation, `--enforce-eager` can be added to skip graph compilation.
> - [vLLM Serving Arguments documentation](https://docs.vllm.com.cn/en/latest/cli/serve/?h=block+size#arguments) — Additional parameter details for vLLM serve commands.
> - [Environment Variables](../../user_guide/configuration/env_vars.md) — Ascend-specific environment variables (`HCCL_*`, etc.).

**Service Verification:** If the service starts successfully, the following startup log will be displayed:

```text
(APIServer pid=) INFO:     Started server process []
(APIServer pid=) INFO:     Waiting for application startup.
(APIServer pid=) INFO:     Application startup complete.
```

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distill-qwen-32b",
    "messages": [
      {"role": "user", "content": "Give me a short introduction to large language models."}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_completion_tokens": 4096
  }'
```

Expected result: HTTP 200 with a JSON response containing the `choices` field. The `reasoning_content` field carries the chain-of-thought, and `content` carries the final answer.

## 7 Accuracy Evaluation

The GSM8K accuracy of DeepSeek-R1-Distill-Qwen-32B can be evaluated with `lm-eval` against the running service.

```bash
pip install "lm-eval"
lm_eval --model local-chat-completions \
  --model_args base_url=http://127.0.0.1:8000/v1,tokenizer=your_model_path,num_concurrent=16 \
  --tasks gsm8k_cot \
  --apply_chat_template \
  --gen_kwargs temperature=0,max_gen_toks=4096 \
  --output_path ./gsm8k-result
```

**Validated accuracy on Atlas A2 inference products (4 NPUs, TP=4, BF16, vLLM-Ascend v0.23.0, `--enforce-eager`):**

| Task | Metric | Result | Notes |
|------|--------|--------|-------|
| GSM8K-CoT (0-shot, chat template) | exact_match,flexible-extract | **0.8218 ± 0.0105** | Effective metric for reasoning models |
| GSM8K-CoT (0-shot, chat template) | exact_match,strict-match | 0.0008 ± 0.0008 | Not applicable: the model does not emit the `#### answer` format that strict-match expects |

> **Note**: A small fraction (~3%) of requests return null content when the chain-of-thought exceeds the 4096-token generation budget; these are scored as incorrect. Raising `max_gen_toks` (and `--max-model-len` accordingly) can recover part of this loss. The `strict-match` metric is listed for schema consistency but is not meaningful for R1-style reasoning models.

## 8 Performance Evaluation

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for latency, serving and throughput benchmarking, same as other dense models. See also the `vllm bench serve` example in [Qwen3 Dense](Qwen3-Dense.md#8-performance-evaluation).

## 9 Performance Tuning

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Recommended starting points on Atlas A2 inference products:

- TP=4 for balanced throughput/latency; TP=8 for lower latency.
- `--enforce-eager` off (graph mode) for production after functional validation.
- Refer to [Qwen3 Dense tuning section](Qwen3-Dense.md#9-performance-tuning) for environment variables such as `TASK_QUEUE_ENABLE` and `HCCL_OP_EXPANSION_MODE`.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [vLLM-Ascend FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This section only covers issues specific to DeepSeek-R1-Distill-Qwen-32B.
