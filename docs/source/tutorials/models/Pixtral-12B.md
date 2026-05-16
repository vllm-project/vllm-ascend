# Pixtral-12B-2409

## Introduction

Pixtral-12B-2409 is a 12-billion parameter multimodal model developed by Mistral AI, released in September 2024. It combines a 12B parameter language model (Mistral Nemo) with a newly trained 400M parameter vision encoder, enabling native image understanding alongside text. The model supports variable image sizes and aspect ratios, and can handle multiple images in a single context window.

This document describes how to deploy Pixtral-12B-2409 on Ascend NPU using vLLM-Ascend, including the known compatibility patches required and the verification steps.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Hardware Requirements

- 1× Atlas 800I A2 (64G HBM) — tested on Ascend 910B2C
- CANN: 8.3.RC2 or later

### Model Weight

Download the model weight from Hugging Face or ModelScope:

- `mistralai/Pixtral-12B-2409`: [HuggingFace](https://huggingface.co/mistralai/Pixtral-12B-2409) | [ModelScope](https://modelscope.cn/models/mistralai/Pixtral-12B-2409)

The model requires approximately 24 GB of HBM memory.

### Installation

Run the vllm-ascend docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
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
-v /data:/data \
-p 8000:8000 \
-it $IMAGE bash
```

After entering the container, set up the environment:

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_USE_V1=1
export HF_HOME=/root/.cache
```

### Dependency Fixes

Pixtral-12B-2409 requires specific package versions due to compatibility issues between the default environment and the model's dependencies:

```shell
# Fix pkg_resources availability (setuptools 82.x may be incomplete in some containers)
pip install "setuptools==65.6.3" --force-reinstall --no-cache-dir

# Pin mistral_common to 1.8.2 (1.9+ removed ImageChunk used by vllm's pixtral.py)
pip install "mistral_common[image,audio]==1.8.2"

# Pin numpy to avoid numba incompatibility introduced by mistral_common upgrade
pip install "numpy==1.26.4"
```

### NPU Compatibility Patches for pixtral.py

The vLLM `pixtral.py` vision encoder contains two NPU-incompatible patterns that must be patched:

**Patch 1 — complex64 tensor indexing** (`pixtral.py` line ~810)

Ascend NPU's `aclnnIndex` does not support `DT_COMPLEX64`. Replace the direct complex tensor index with a `view_as_real` round-trip:

```python
# Before
freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

# After
freqs_cis_real = torch.view_as_real(self.freqs_cis)
freqs_cis = torch.view_as_complex(
    freqs_cis_real[positions[:, 0], positions[:, 1]].contiguous()
)
```

**Patch 2 — xformers attention and BlockDiagonalMask** (`pixtral.py` lines ~674 and ~814)

xformers is a CUDA-only library and is not available on Ascend NPU. Replace with PyTorch native `F.scaled_dot_product_attention`:

```python
# In Attention.forward() — replace xformers call:
# Before
out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)

# After — GQA: repeat KV heads to match query heads count before calling SDPA
# Pixtral-12B uses n_heads=32 and n_kv_heads=8; SDPA requires matching head counts
q_t = q.transpose(1, 2)
k_t = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2).transpose(1, 2)
v_t = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2).transpose(1, 2)
out = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2)
```

```python
# In VisionEncoder.forward() — replace BlockDiagonalMask:
# Before
if USE_XFORMERS_OPS:
    mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(...)
else:
    raise ImportError("Xformers is required ...")

# After
mask = None  # NPU: full attention; no cross-image masking
```

> **Limitation**: Setting `mask = None` is only safe for **single-image inference**
> (`max_num_seqs=1`). In multi-image batch scenarios, full attention allows cross-image
> token interactions, which can degrade output quality. A proper tensor-based
> block-diagonal mask for NPU is a known gap tracked in
> [vllm-ascend#7322](https://github.com/vllm-project/vllm-ascend/issues/7322).

Apply the patches with the following one-liner:

```shell
PIXTRAL_PY=$(python -c "import vllm.model_executor.models.pixtral as m; print(m.__file__)")

python3 - <<'EOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()

# Patch 1: complex64 indexing
old1 = '        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]'
new1 = ('        freqs_cis_real = torch.view_as_real(self.freqs_cis)\n'
        '        freqs_cis = torch.view_as_complex('
        'freqs_cis_real[positions[:, 0], positions[:, 1]].contiguous())')
content = content.replace(old1, new1)

# Patch 2: xformers attention
old2 = ('        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)\n'
        '        out = out.reshape(batch, patches, self.n_heads * self.head_dim)')
new2 = ('        q_t = q.transpose(1, 2)\n'
        '        k_t = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2).transpose(1, 2)\n'
        '        v_t = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2).transpose(1, 2)\n'
        '        out = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2)\n'
        '        out = out.reshape(batch, patches, self.n_heads * self.head_dim)')
content = content.replace(old2, new2)

# Patch 3: BlockDiagonalMask
old3 = ('        if USE_XFORMERS_OPS:\n'
        '            mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(\n'
        '                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], )\n'
        '        else:\n'
        '            raise ImportError("Xformers is required for Pixtral inference "\n'
        '                              "with the Mistral format")')
new3 = '        mask = None  # NPU: full attention; no cross-image masking'
content = content.replace(old3, new3)

with open(path, 'w') as f:
    f.write(content)
print('All patches applied successfully.')
EOF
$PIXTRAL_PY
```

## Deployment

### Offline Inference (Python API)

```python
import os
os.environ["VLLM_USE_V1"] = "1"

import vllm_ascend  # noqa: F401 — registers Ascend platform plugin
from vllm import LLM, SamplingParams

MODEL_PATH = "/root/.cache/mistralai/Pixtral-12B-2409"

llm = LLM(
    model=MODEL_PATH,
    tokenizer_mode="mistral",
    max_model_len=4096,
    max_num_seqs=1,
    dtype="bfloat16",
    trust_remote_code=True,
    allowed_local_media_path="/data",   # required for local file:// URLs
    enforce_eager=True,                  # required: ACL Graph incompatible with vision encoder
)

sampling_params = SamplingParams(max_tokens=256, temperature=0.0)

messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "file:///data/test_image.jpg"}},
        {"type": "text", "text": "Describe this image in detail."},
    ],
}]

outputs = llm.chat(messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

### Service Deployment (OpenAI-compatible API)

```shell
vllm serve mistralai/Pixtral-12B-2409 \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer-mode mistral \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager \
  --allowed-local-media-path /data
```

Query the service:

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Pixtral-12B-2409",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"}},
          {"type": "text", "text": "What is shown in this image?"}
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

## Functional Verification

After applying the patches and starting the service (or running offline), verify that the model correctly processes image inputs and generates coherent descriptions.

A simple smoke-test script:

```python
from PIL import Image
import os

# Create a test image
img = Image.new("RGB", (224, 224), color=(255, 128, 0))
img.save("/data/test_smoke.jpg")

# Confirm output is generated without error
outputs = llm.chat([{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "file:///data/test_smoke.jpg"}},
        {"type": "text", "text": "What color is this image?"},
    ],
}], SamplingParams(max_tokens=32, temperature=0.0))

assert len(outputs[0].outputs[0].text) > 0, "No output generated"
print("Verification passed:", outputs[0].outputs[0].text)
```

## Known Issues and Upstream Gaps

| Issue | Root Cause | Status |
|-------|-----------|--------|
| `aclnnIndex` fails on complex64 tensor | Ascend NPU does not support `DT_COMPLEX64` indexing | Workaround applied locally; upstream fix pending |
| xformers `memory_efficient_attention` unavailable | xformers is CUDA-only | Replaced with `F.scaled_dot_product_attention` |
| `BlockDiagonalMask` unavailable | Depends on xformers | Replaced with `mask=None`; **only safe for single-image inference** (`max_num_seqs=1`) |
| `mistral_common>=1.9` removes `ImageChunk` | API breaking change | Pin to `mistral_common==1.8.2` |

The patches to `pixtral.py` should be contributed upstream to vLLM-Ascend to enable Pixtral to run without manual patching. See [vllm-ascend#7322](https://github.com/vllm-project/vllm-ascend/issues/7322) for tracking.
