# MiniCPM-2B-dpo-bf16

## Introduction

MiniCPM-2B is a compact 2-billion-parameter language model from the OpenBMB team. The `dpo-bf16` variant is fine-tuned with Direct Preference Optimization (DPO) for improved alignment. It features a standard dense decoder-only transformer architecture with RMSNorm and SiLU activation, making it a lightweight yet capable model for general text generation tasks despite its small size.

This document describes the main verification steps of the model, including supported features, environment preparation, single-node deployment, functional verification, and accuracy evaluation on the GSM8K benchmark.

## Supported Features

| Feature          | MiniCPM-2B-dpo-bf16 |
|------------------|---------------------|
| Dense Model      | ✅                  |
| BF16 Inference   | ✅                  |
| ACLGraph         | ✅                  |
| Trust Remote Code| ✅ (required)       |

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) for the full feature matrix.

## Environment Preparation

### Model Weight

- `MiniCPM-2B-dpo-bf16` (BF16 version): requires 1 Ascend 910B (1 x 64G NPU). [Download model weight from HuggingFace](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16)

It is recommended to place the model weight in a shared cache directory, such as `/root/.cache/` or your local model path.

> **Note**: MiniCPM-2B-dpo-bf16 uses custom modeling code shipped with the model repository (`modeling_minicpm.py`). You must set `--trust-remote-code` when loading.

### Installation

MiniCPM-2B-dpo-bf16 can be deployed with `vllm-ascend` in a compatible runtime environment.

You can use the official docker image for deployment:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/models:/data/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

If you do not want to use the docker image, you can also build from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

Start the online serving service with the following command:

```bash
# Using HuggingFace model ID (auto-download)
vllm serve "openbmb/MiniCPM-2B-dpo-bf16" \
  --served-model-name minicpm-2b-dpo \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000

# Using local model weights
vllm serve /path/to/MiniCPM-2B-dpo-bf16 \
  --served-model-name minicpm-2b-dpo \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000
```

**Key parameters**:

- `--trust-remote-code`: **required** — MiniCPM-2B-dpo-bf16 includes custom model code that must be loaded from the model directory.
- `--tensor-parallel-size 1`: The model fits on a single NPU (2B parameters in BF16 ≈ 4GB).
- `--max-model-len 4096`: The model supports up to 4096 tokens.
- `--gpu-memory-utilization 0.85`: Balances memory for weights, KV cache, and activations.

## Functional Verification

Once your server is started, verify basic functionality:

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-2b-dpo",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-2b-dpo",
    "prompt": "The capital of France is",
    "max_tokens": 64,
    "temperature": 0
  }'
```

Expected response should contain relevant and coherent text.

### Python API

```python
from vllm import LLM, SamplingParams

# Using HuggingFace model ID (auto-download)
llm = LLM(
    model="openbmb/MiniCPM-2B-dpo-bf16",
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

# Using local model weights
# llm = LLM(
#     model="/path/to/MiniCPM-2B-dpo-bf16",
#     trust_remote_code=True,
#     max_model_len=4096,
#     gpu_memory_utilization=0.85,
# )

prompts = [
    "Hello, my name is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=64)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print("-" * 40)
```

## Accuracy Evaluation

Use `lm-evaluation-harness` to evaluate on GSM8K:

```bash
# Single card evaluation
python tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/MiniCPM-2B-dpo-bf16.yaml \
  --tp-size 1
```

Expected accuracy (GSM8K 5-shot): approximately 40-45% (strict-match).

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `ImportError: cannot import name 'MiniCPMConfig'` | Missing `--trust-remote-code` | Add `--trust-remote-code` to serve command |
| `OutOfMemoryError` | Insufficient NPU memory | Reduce `max-model-len` or increase `gpu-memory-utilization` |
| Slow response | ACLGraph not enabled | Remove `--enforce-eager` to use graph mode |
| `KeyError: 'minicpm'` | vLLM version too old | Upgrade to latest vllm-ascend image |
