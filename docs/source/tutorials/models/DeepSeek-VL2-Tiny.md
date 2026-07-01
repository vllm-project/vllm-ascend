# DeepSeek-VL2-Tiny

## Introduction

`deepseek-ai/deepseek-vl2-tiny` is a compact vision-language model from DeepSeek designed for efficient multimodal understanding.

## Current Status

The model is currently unsupported by vLLM due to a model configuration compatibility issue.

## Environment Preparation

### Model Weight

Download model weight from [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny).

### Installation

Use the official docker image.

## Validation

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

DeepSeek-VL2 config.json uses a nested structure:
- No architectures at top level
- Info is inside `language_config.architectures`

vLLM only looks for architectures at the top level.

## Adaptation Suggestions

Modify model config parsing to check `language_config.architectures` when no top-level architectures is found.
