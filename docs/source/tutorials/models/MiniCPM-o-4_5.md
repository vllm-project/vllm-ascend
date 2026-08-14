# MiniCPM-o 4.5

## 1 Introduction

[MiniCPM-o 4.5](https://github.com/OpenBMB/MiniCPM-o) is a 9B omni-modal model from OpenBMB. It unifies vision, audio and text understanding in a single end-to-end architecture, and reaches vision-language performance comparable to much larger models on benchmarks such as MMBench and MathVista. The model is released under the Apache-2.0 license.

This document shows the verification steps of `OpenBMB/MiniCPM-o-4_5` on Ascend NPU, covering text and vision understanding through the OpenAI-compatible API.

:::{note}
Scope: the OpenAI-compatible server covers text + image/video understanding. Real-time full-duplex speech interaction (streaming audio in/out) is served by the official MiniCPM-o streaming stack and is out of scope of this document.
:::

:::{warning}
`MiniCPM-o-4_5` requires a vLLM build that contains the upstream resampler device fix (see [vllm#42322](https://github.com/vllm-project/vllm/issues/42322): `pos_embed` of the MiniCPM-V resampler may stay on CPU during profiling and crashes with `Expected all tensors to be on the same device`). The fix is present in current vLLM `main`. If you are on an older build (e.g. vLLM 0.21.0), apply the one-line workaround shown in section 6 before starting the server.
:::

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Environment Preparation

Verified environment:

- 1 × Ascend 910C NPU (64 GB HBM), driver 25.5.0, CANN 9.0.0
- vllm-ascend 0.21.0rc1 + vLLM 0.21.0

### 3.1 Model Weight

`OpenBMB/MiniCPM-o-4_5` (~19 GB, bf16 safetensors, 4 shards):

- [Download from ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5) (recommended in CN regions)

  ```bash
  pip install modelscope
  modelscope download --model OpenBMB/MiniCPM-o-4_5 --local_dir /user_data/models/MiniCPM-o-4_5
  ```

- [Download from HuggingFace](https://huggingface.co/openbmb/MiniCPM-o-4_5)

  Alternatively set `VLLM_USE_MODELSCOPE=True` and let vLLM pull the model automatically.

## 4 Installation

You can use the official docker image, or build from source. Refer to [installation](../../installation.md) for both options.

Make sure the Ascend environment variables are set before starting the server:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh   # required by torch_npu ATB ops (libatb.so)
```

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

`OpenBMB/MiniCPM-o-4_5` (bf16, ~19 GB) fits on a single 64 GB NPU:

```bash
vllm serve /user_data/models/MiniCPM-o-4_5 \
    --trust-remote-code \
    --max-model-len 4096 \
    --port 8000
```

Wait until `Application startup complete` appears in the log (model loading plus multimodal profiling takes about 2-3 minutes).

## 6 Known Issues

### 6.1 Resampler `pos_embed` device mismatch (older vLLM builds)

Symptom: engine crashes during startup profiling with

```text
RuntimeError: Expected all tensors to be on the same device. Expected NPU tensor,
please check whether the input tensor device is correct.
```

Root cause: in `vllm/model_executor/models/minicpmv.py`, `_adjust_pos_cache` only re-creates the 2D position cache when the requested size exceeds the cached size; the cache created on CPU at `__init__` is therefore never moved to the NPU, and `x + pos_embed_2d` fails. Same issue on CUDA: [vllm#42322](https://github.com/vllm-project/vllm/issues/42322).

Workaround for affected builds: patch `_adjust_pos_cache` to move the cache when the device does not match:

```python
# vllm/model_executor/models/minicpmv.py, in Resampler2_5._adjust_pos_cache
if max_h > self.max_size[0] or max_w > self.max_size[1]:
    self.max_size = (
        max(max_h, self.max_size[0]),
        max(max_w, self.max_size[1]),
    )
    self._set_2d_pos_cache(self.max_size, device)
elif self.pos_embed.device != device:        # <-- add these two lines
    self.pos_embed = self.pos_embed.to(device)
```

## 7 Functional Verification

Once your server is started, you can query the model with input prompts.

Text:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "/user_data/models/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": "Hello, introduce yourself in one sentence."}],
    "max_tokens": 100
    }'
```

Vision (engineering drawing understanding — verified on 910C with a base64-encoded DXF rendering of a flange drawing):

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "/user_data/models/MiniCPM-o-4_5",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_IMAGE>"}},
        {"type": "text", "text": "How many bolt holes are on this drawing? What is the outer diameter?"}
    ]}],
    "max_tokens": 150
    }'
```

Expected Result (the drawing carries `6-φ12 均布` and `φ200` annotations):

```text
6 bolt holes, outer diameter φ200.
```

## 8 Accuracy / Performance Evaluation

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details, or [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/).

## 9 FAQ

- Common Issues Tip: If you encounter issues, Refer to [FAQs](../../faqs.md).
