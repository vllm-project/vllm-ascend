# DeepSeek-VL2-Tiny

## Introduction

`deepseek-ai/deepseek-vl2-tiny` is a compact vision-language model from DeepSeek.

## Current Status

**The model is currently unsupported by vLLM** due to a model configuration compatibility issue.

## Environment Preparation

### Model Weight

- `deepseek-ai/deepseek-vl2-tiny`: [Download model weight](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)

### Installation

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \\
    --name vllm-ascend \\
    --shm-size=1g \\
    --net=host \\
    --device /dev/davinci0 \\
    --device /dev/davinci_manager \\
    --device /dev/devmm_svm \\
    --device /dev/hisi_hdc \\
    -v /usr/local/dcmi:/usr/local/dcmi \\
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \\
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \\
    -v /etc/ascend_install.info:/etc/ascend_install.info \\
    -v /root/.cache:/root/.cache \\
    -it $IMAGE bash
```

## Validation Attempt

### Model Loading Test

```python
from vllm import LLM
model = LLM(model="deepseek-ai/deepseek-vl2-tiny")
```

### Error Encountered

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, No model architectures are specified
```

## Root Cause Analysis

DeepSeek-VL2 uses a nested config.json structure where architectures is inside language_config.

### Adaptation Suggestions

Modify the model config parsing logic to check language_config.architectures when no top-level architectures is found.
