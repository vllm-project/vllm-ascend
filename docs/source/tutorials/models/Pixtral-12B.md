# Pixtral-12B-2409

## Introduction

Pixtral-12B-2409 is a 12-billion parameter multimodal model developed by Mistral AI, released in September 2024. It combines a 12B parameter language model (Mistral Nemo) with a newly trained 400M parameter vision encoder, enabling native image understanding alongside text. The model supports variable image sizes and aspect ratios, and can handle multiple images in a single context window.

This document describes how to deploy Pixtral-12B-2409 on Ascend NPU using vLLM-Ascend, including environment preparation, deployment, and verification steps.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

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

Pixtral-12B-2409 requires specific package versions. These are installed automatically by the CI pipeline. When deploying manually, run the following commands inside the container:

```shell
# Fix pkg_resources availability (setuptools 82.x may be incomplete in some containers)
pip install "setuptools==65.6.3" --force-reinstall --no-cache-dir

# Pin mistral_common to 1.8.2 (1.9+ removed ImageChunk used by vllm's pixtral.py)
pip install "mistral_common[image,audio]==1.8.2"

# Pin numpy to avoid numba incompatibility introduced by mistral_common upgrade
pip install "numpy==1.26.4"
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

After starting the service (or running offline), verify that the model correctly processes image inputs and generates coherent descriptions.

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

## Accuracy Evaluation

The repository already contains an end-to-end accuracy baseline for this model in `tests/e2e/models/configs/Pixtral-12B-2409.yaml`.

| dataset | platform | metric | value |
|----- | ----- | ----- | ----- |
| mmmu_val | A2 | acc,none | 0.52 |
