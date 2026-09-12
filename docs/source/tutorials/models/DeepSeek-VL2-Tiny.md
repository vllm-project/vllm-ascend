# DeepSeek-VL2-Tiny

## Introduction

DeepSeek-VL2-Tiny is a compact vision-language model.

## Current Status

Currently unsupported by vLLM.

## Environment

Download from HuggingFace. Use official docker image.

## Validation

```python
from vllm import LLM
model = LLM(model="deepseek-ai/deepseek-vl2-tiny")
```

Error: ValidationError - No model architectures specified

## Root Cause

config.json has architectures inside language_config.

## Adaptation

Check language_config.architectures.
