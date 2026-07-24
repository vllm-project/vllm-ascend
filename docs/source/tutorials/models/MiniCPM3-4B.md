# MiniCPM3-4B

## Introduction

MiniCPM3-4B is the third-generation small language model from the OpenBMB team, featuring Multi-head Latent Attention (MLA) architecture similar to DeepSeek-V2. Despite its compact 4B parameter size, it delivers competitive performance through efficient attention mechanisms and the LongRoPE position encoding strategy supporting up to 32K context length.

This document describes the main verification steps of the model, including supported features, environment preparation, single-node deployment, functional verification, and accuracy evaluation on the GSM8K benchmark.

## Supported Features

| Feature          | MiniCPM3-4B |
|------------------|-------------|
| Dense Model      | ✅          |
| MLA(Multi-head Latent Attention) | ✅ |
| LongRoPE         | ✅          |
| BF16 Inference   | ✅          |
| ACLGraph         | ✅          |
| Trust Remote Code| ✅ (required)|

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) for the full feature matrix.

## Environment Preparation

### Model Weight

- `MiniCPM3-4B`(BF16 version): requires 1 Ascend 910B (1 x 64G NPU). [Download model weight from HuggingFace](https://huggingface.co/openbmb/MiniCPM3-4B)

It is recommended to place the model weight in a shared cache directory, such as `/root/.cache/` or your local model path.

> **Note**: MiniCPM3-4B uses custom modeling code shipped with the model repository (`modeling_minicpm.py`). You must set `--trust-remote-code` when loading.

### Installation

MiniCPM3-4B can be deployed with `vllm-ascend` in a compatible runtime environment.

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
vllm serve "openbmb/MiniCPM3-4B" \
  --served-model-name minicpm3-4b \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000

# Using local model weights
vllm serve /path/to/MiniCPM3-4B \
  --served-model-name minicpm3-4b \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000
```

**Key parameters**:

- `--trust-remote-code`: **required** — MiniCPM3-4B includes custom model code that must be loaded from the model directory.
- `--tensor-parallel-size 1`: The model fits on a single NPU (4B parameters in BF16 ≈ 8GB).
- `--max-model-len 4096`: Conservative setting; the model supports up to 32768 tokens via LongRoPE.
- `--gpu-memory-utilization 0.85`: Balances memory for weights, KV cache, and activations.

## Functional Verification

Once your server is started, verify basic functionality:

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm3-4b",
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
    "model": "minicpm3-4b",
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
    model="openbmb/MiniCPM3-4B",
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

# Using local model weights
# llm = LLM(
#     model="/path/to/MiniCPM3-4B",
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
  --config tests/e2e/models/configs/MiniCPM3-4B.yaml \
  --tp-size 1
```

Expected accuracy (GSM8K 5-shot): approximately 45-50% (strict-match).

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `ImportError: cannot import name 'MiniCPM3Config'` | Missing `--trust-remote-code` | Add `--trust-remote-code` to serve command |
| `OutOfMemoryError` | Insufficient GPU memory | Reduce `max-model-len` or increase `gpu-memory-utilization` |
| Slow response | ACLGraph not enabled | Remove `--enforce-eager` to use graph mode |
| `KeyError: 'minicpm3'` | vLLM version too old | Upgrade to latest vllm-ascend image |
| `RuntimeError: Failed to compile ... precompiled.h.gch` | Triton-Ascend PCH compilation bug | Update triton-ascend driver, remove `-shared` from GCH compilation flags in `triton/backends/ascend/utils.py` |
| `is pie differs in PCH file vs. current file` | PCH compiled without `-fPIC`, but launcher uses `-fPIC` | Use `-fPIC` (not `-shared`) for GCH compilation in `triton/backends/ascend/utils.py` |
