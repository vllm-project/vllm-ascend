# MiniCPM-2B-dpo-bf16

## 1 Introduction

MiniCPM-2B-dpo-bf16 is a text-only large language model developed by ModelBest Inc. and TsinghuaNLP, with 2.4B parameters excluding embeddings and a context length of 4096 tokens.

This document describes the supported features, environment preparation, single-node deployment, functional verification, and accuracy evaluation of MiniCPM-2B-dpo-bf16 on Atlas A2.

MiniCPM-2B-dpo-bf16 is supported by upstream vLLM through the `MiniCPMForCausalLM` implementation, so no extra adaptation is required in vLLM Ascend. This document is written based on the latest vLLM Ascend main branch and is verified on Atlas A2.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

MiniCPM-2B-dpo-bf16 supports ACLGraph execution on Atlas A2, with TP=1 and TP=2.

## 3 Prerequisites

### 3.1 Model Weight

Download the model weight to a directory that is accessible from the deployment environment, such as `/root/.cache/`. In the following examples, replace `<YOUR_MODEL_PATH>` with your actual model path.

| Weight Version | Hardware Requirements | Download Links |
| --- | --- | --- |
| `OpenBMB/MiniCPM-2B-dpo-bf16` | one Atlas A2 64 GB NPU | [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM-2B-dpo-bf16) \| [Hugging Face](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16) |

The model weights are licensed under the [General Model License (GML)](https://github.com/OpenBMB/General-Model-License). Commercial use requires authorization from the model provider.

## 4 Installation

### 4.1 Docker Image Installation

{% include "getting_started/installation/ascend_docker_runtime.inc.md" %}

Pull the vLLM Ascend image for Atlas A2 and start a container. Replace the model-weight mount with the path used in your environment. To use two NPUs, set `ASCEND_VISIBLE_DEVICES=0-1`.

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export MODEL_PATH=<YOUR_MODEL_PATH>

docker pull ${IMAGE}

docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    -e ASCEND_VISIBLE_DEVICES=0 \
    -v "$MODEL_PATH:$MODEL_PATH" \
    -p 8000:8000 \
    -it $IMAGE bash
```

After entering the container, verify the environment:

```bash
npu-smi info
pip show vllm vllm-ascend
```

Expected result: `npu-smi info` lists the Ascend device, and `pip show` displays version information for both packages.

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM Ascend following the [Installation Guide](../../getting_started/installation.md#installation-software-environment).

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on one NPU, making it suitable for functional verification and single-node serving.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH=<YOUR_MODEL_PATH>

vllm serve ${MODEL_PATH} \
  --served-model-name MiniCPM-2B-dpo-bf16 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

- `--trust-remote-code`: loads the custom modeling code shipped with the model.
- `--tensor-parallel-size 1`: uses one NPU. The model has 36 attention heads, so the TP size must be a divisor of 36.
- `--max-model-len 4096`: matches the context length of the model.

When the service starts successfully, the log contains `Application startup complete`. If startup fails, see the [Public FAQs](../../faqs.md).

### 5.2 Multi-Card Deployment

To deploy with TP=2, use the following command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MODEL_PATH=<YOUR_MODEL_PATH>

vllm serve ${MODEL_PATH} \
  --served-model-name MiniCPM-2B-dpo-bf16 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000
```

## 6 Functional Verification

After the service is started, the model can be invoked through OpenAI-compatible APIs.

**Completions API:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM-2B-dpo-bf16",
        "prompt": "The capital of China is",
        "max_tokens": 32,
        "temperature": 0
    }'
```

Expected result: HTTP 200 and a JSON response containing the generated text in the `choices` field.

**Chat Completions API:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM-2B-dpo-bf16",
        "messages": [
            {
                "role": "user",
                "content": "用一句话介绍你自己。"
            }
        ],
        "max_tokens": 128,
        "temperature": 0
    }'
```

Expected result: HTTP 200 and a JSON response containing the generated text in the `choices` field.

**Offline Inference:**

```python
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(model="<YOUR_MODEL_PATH>", trust_remote_code=True, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(["The capital of France is"], sampling_params)
    print(outputs[0].outputs[0].text)
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

### 7.2 Using Language Model Evaluation Harness

Using the `gsm8k` and `ceval-valid` datasets as examples, run the accuracy evaluation for MiniCPM-2B-dpo-bf16 in offline mode. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for installation and usage details.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
lm_eval \
  --model vllm \
  --model_args pretrained=<YOUR_MODEL_PATH>,trust_remote_code=True,tensor_parallel_size=1,max_model_len=4096,gpu_memory_utilization=0.8 \
  --tasks gsm8k,ceval-valid \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./
```

The results on Atlas A2 (TP=1, lm-eval 0.4.12) are for reference only.

| Dataset | Metric | Score |
| --- | --- | --- |
| GSM8K | exact_match (strict-match) | 47.86 |
| GSM8K | exact_match (flexible-extract) | 48.32 |
| C-Eval | acc | 49.55 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following configurations are for reference only. Tune the parameters according to your actual workload.

| Scenario | Deployment Mode | Total NPUs | Key Considerations |
| --- | --- | --- | --- |
| Functional verification | Single-node online serving | 1 | Use TP=1 to verify model loading and basic generation. |
| Single-node serving | Single-node online serving | 1 or 2 | TP=1 provides a KV cache of about 145k tokens on one 64 GB NPU. Use TP=2 for multi-card serving. |

### 9.2 Tuning Guidelines

Refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.
