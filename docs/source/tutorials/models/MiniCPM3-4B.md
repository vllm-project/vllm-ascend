# MiniCPM3-4B

## 1 Introduction

MiniCPM3-4B is the third-generation dense language model in OpenBMB's MiniCPM series. It combines a 62-layer, 2560-hidden-size decoder with Multi-head Latent Attention (MLA), a SwiGLU MLP, and LongRoPE position encoding. The checkpoint is approximately 4B parameters and natively supports a 32K context window. It is intended for English/Chinese chat, summarization, and other text-only generation workloads.

`MiniCPM3ForCausalLM` is implemented by upstream vLLM, including its attention, weight mapping, sampling integration, and LongRoPE handling. vLLM Ascend therefore reuses the upstream model implementation without a model-specific patch. This tutorial covers the Ascend-specific configuration, deployment, functional verification, accuracy evaluation, and known limitations.

This document is validated against vLLM Ascend's current main development stack. The model is marked experimental in the support matrix until the new single-card and tensor-parallel E2E cases pass in CI on Atlas A2 hardware.

## 2 Supported Features

| Feature | Status | Notes |
| --- | --- | --- |
| BF16 inference | ✅ | `--dtype bfloat16`; the checkpoint is BF16 |
| FP16 inference | 🟡 | Not part of the validated configuration |
| MLA attention | ✅ | Reuses upstream `MiniCPM3Attention` |
| LongRoPE | ✅ | 32K theoretical context; 4K is the E2E-tested length |
| Tensor parallelism | 🟡 | TP1 is validated; a TP2 E2E case is included but pending CI |
| Expert parallelism | ❌ | Dense model, no MoE experts |
| Pipeline / data parallelism | 🟡 | Handled by the generic paths; not separately qualified for this model |
| Chunked prefill, prefix caching | 🟡 | Enabled by default; not separately qualified |
| ACLGraph | 🟡 | Supported by the generic path; `--enforce-eager` is the fallback |
| Quantization (W8A8/W4A8) | ❌ | No Ascend quantized checkpoint is provided or validated |
| LoRA, speculative decoding, MTP | ❌ | Not provided by this checkpoint and not validated |
| Atlas 300I DUO (310P) | ❌ | Not supported by this configuration |

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md) page for the authoritative support matrix and to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

### Supported dtype, devices, and limits

- **dtype:** BF16 only for the validated path. Use `--dtype bfloat16`; do not use FP16 or an Ascend quantized checkpoint.
- **devices:** Atlas A2 inference products. Atlas A3 uses the same software path but is not part of the current validated matrix. Atlas 300I DUO (310P) is unsupported.
- **parallelism:** single NPU (`--tensor-parallel-size 1`) is the default. TP2 can be enabled when `num_attention_heads=40` is divisible by the TP size; it is covered by the E2E case and awaits CI evidence.
- **context length:** theoretical maximum is 32768 tokens from `max_position_embeddings`. Start with `--max-model-len 4096`, which is the value used by the E2E configuration, and raise it only after checking NPU memory.
- **memory:** BF16 weights are approximately 7.6 GiB, so a single 32 GB card is sufficient. The E2E configuration uses `--gpu-memory-utilization 0.8` with `--enforce-eager`, which keeps several GB of headroom on 32 GB cards and avoids aclgraph capture spikes in memory-constrained containers. On 64 GB cards both values can be relaxed.

## 3 Prerequisites

### 3.1 Model Weight

| Weight version | Hardware requirement | Download |
| --- | --- | --- |
| `MiniCPM3-4B` (BF16) | 1 Atlas A2 NPU (64 GB recommended) | [Hugging Face](https://huggingface.co/openbmb/MiniCPM3-4B) \| [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM3-4B) \| [Modelers](https://modelers.cn/models/Modelers_Park/MiniCPM3-4B) |

> **Path description:** Download the checkpoint to a directory of your choice and record the absolute path, for example `/models/MiniCPM3-4B`. Replace `<YOUR_MODEL_PATH>` in the commands below with that path.

The repository contains custom `configuration_minicpm.py` and `modeling_minicpm.py` files. `--trust-remote-code` is mandatory.

### 3.2 Environment Check

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.npu.device_count(), torch.npu.get_device_name(0))"
```

The command should report at least one available Ascend NPU.

## 4 Installation

Use the vLLM Ascend image that matches the vLLM version installed in the container. The following example uses the documented release placeholder; replace it with the actual image tag for your environment.

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

docker run --rm \
  --name vllm-ascend-minicpm3 \
  --shm-size=1g \
  --net=host \
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
  -v /models:/models \
  -it $IMAGE bash
```

Verify the runtime:

```bash
pip show vllm vllm-ascend
```

Both packages must be present. For source installation, follow the [Installation Guide](../../getting_started/installation.md).

## 5 Online Service Deployment

### 5.1 Single-NPU Deployment

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve <YOUR_MODEL_PATH> \
  --served-model-name minicpm3-4b \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

Successful startup contains:

```text
INFO:     Application startup complete.
```

Recommended parameter meanings:

| Parameter | Recommended value | Reason |
| --- | --- | --- |
| `--trust-remote-code` | required | Loads the checkpoint's custom configuration/modeling code |
| `--dtype` | `bfloat16` | Matches the checkpoint and validated kernels |
| `--max-model-len` | `4096` | Conservative tested length; theoretical maximum is 32768 |
| `--tensor-parallel-size` | `1` | Single-card baseline; use `2` only with two free NPUs |
| `--gpu-memory-utilization` | `0.8` | Leaves ~6 GB headroom on a 32 GB card for activations |
| `--enforce-eager` | recommended on 32 GB cards | Skips aclgraph capture, whose compile/capture phase spikes both NPU and host memory |

### 5.2 Tensor-Parallel Deployment

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

vllm serve <YOUR_MODEL_PATH> \
  --served-model-name minicpm3-4b \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

For single-card and TP2 verification, set `HCCL_OP_EXPANSION_MODE=AIV` and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` if the host requires the project's standard multi-card environment settings.

## 6 Functional Verification

Chat Completions request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm3-4b",
    "messages": [
      {"role": "user", "content": "用一句话介绍杭州。"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

Expected result: HTTP 200, a non-empty `choices[0].message.content`, and `finish_reason` equal to `stop` or `length`. A non-greedy request such as `"temperature": 0.7, "top_p": 0.7, "seed": 42` should also return coherent text and be reproducible with the same seed.

Offline API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="<YOUR_MODEL_PATH>",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
)
outputs = llm.generate(
    ["Hello, my name is", "人工智能的未来是"],
    SamplingParams(temperature=0.0, max_tokens=32),
)
for output in outputs:
    print(output.outputs[0].text)
```

## 7 Accuracy Evaluation

The E2E configuration evaluates GSM8K with 5-shot prompting and chat templates. Selecting the configuration also enables the opt-in generation matrix:

```bash
python -m pytest -sv tests/e2e/models \
  --config tests/e2e/models/configs/MiniCPM3-4B.yaml \
  --tp-size 1
```

`MiniCPM3-4B.yaml` gates on strict-match 0.77 and flexible-extract 0.78, the expected GSM8K level for this model (official model-card result 81.1), and compares the measured value against them with the existing `RTOL=0.05` in `test_lm_eval_correctness.py`. The reference run on Atlas A2 (910B4, TP1, BF16, `--enforce-eager`) measured strict-match 0.7566 and flexible-extract 0.7597.

| Covered requirement | Test entry |
| --- | --- |
| GSM8K accuracy, 5-shot, chat template | `test_lm_eval_correctness.py` via `configs/accuracy.txt` |
| Offline vs OpenAI-compatible greedy equality | `test_minicpm3_generation.py::test_minicpm3_offline_online` |
| Seeded non-greedy reproducibility | same test, two identical requests per prompt |
| English / Chinese, short / 2K-token prompts | same test, built from `generation.long_prompt_tokens` |
| Repeated-run NPU memory growth | same test, bounded by `generation.max_memory_growth_mb` |
| TP1 / TP2 | run the same commands with `--tp-size 1` and `--tp-size 2` |

The nightly accuracy job only runs `test_lm_eval_correctness.py` for every entry in `tests/e2e/models/configs/accuracy.txt`, which now includes `MiniCPM3-4B.yaml`. The generation matrix is opt-in and must be requested explicitly with the command above; the `TP2` case skips itself when fewer than two NPUs are visible.

For an online lm-eval run after starting the server:

```bash
lm_eval \
  --model local-completions \
  --model_args model=minicpm3-4b,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --output_path ./accuracy/minicpm3-4b
```

## 8 Performance Evaluation

Use a workload representative of the target service. The following example measures 200 random requests with 1K input and 256 output tokens:

```bash
vllm bench serve \
  --model minicpm3-4b \
  --dataset-name random \
  --random-input 1024 \
  --random-output 256 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./benchmarks/minicpm3-4b
```

Record TTFT, TPOT, output throughput, request throughput, and NPU memory. Performance depends on the Atlas model, driver/CANN version, `max-model-len`, batch size, and whether ACLGraph is enabled; do not compare runs collected with different settings.

## 9 Known Limitations and Troubleshooting

- **BF16 only.** W8A8/W4A8, FP16, and quantized community checkpoints are outside the validated path.
- **`--trust-remote-code` is mandatory.** Omitting it fails while resolving the custom `MiniCPM3Config`.
- **32K is theoretical.** The E2E configuration uses 4096 tokens. Long-context accuracy and peak memory at 32K have not been qualified.
- **TP2 is experimental.** The model requires `num_attention_heads` to be divisible by TP size; the new TP2 test must pass before the matrix changes from experimental.
- **No token-level `transformers` parity case.** The checkpoint's remote code assumes the `transformers` 4.x cache and generation contract; on 5.x it fails on `seen_tokens`, `get_max_cache_shape` and `get_usable_length`, and returns NaN logits when driven without a cache. Accuracy is instead gated by GSM8K over the full test split.
- **LongRoPE warning.** Some vLLM versions print `rope_parameters short_factor ... length` warnings while normalizing the checkpoint's `rope_scaling`. If generation remains correct, this is a benign compatibility warning; if startup fails, report the vLLM version and complete traceback.
- **transformers 5.x and the checkpoint's remote code.** `modeling_minicpm.py` still reads `past_key_values.seen_tokens`, which `transformers` 5 removes, so building the Transformers reference for a parity check fails with `AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`. Inference through vLLM is unaffected; use the `transformers` version pinned by `requirements-ascend.txt` when running the Transformers parity test.
- **Triton PCH failures.** Older `triton-ascend` builds may fail while compiling `precompiled.h.gch` with a `-shared` error. Upgrade to a fixed `triton-ascend`; as a diagnostic fallback, use `--enforce-eager` and clear stale caches (`~/.triton/cache`, `/tmp/triton`).
- **Out of memory or container instability.** Lower `--max-model-len` first, then reduce `--gpu-memory-utilization` (0.8 is the validated default; go lower only if other processes share the NPU). Keep `--enforce-eager` on 32 GB cards: aclgraph capture spikes both NPU and host memory and can destabilize memory-constrained containers. For graph-capture failures, `--enforce-eager` also distinguishes a graph issue from a model/operator issue.

For environment and general parameter issues, see the [Public FAQs](../../faqs.md).
