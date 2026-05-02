# Facebook Chameleon-7B Deployment Guide

## Overview

[Chameleon](https://huggingface.co/facebook/chameleon-7b) is a mixed-modal model from Meta.
This guide shows how to run `facebook/chameleon-7b` with vLLM Ascend on Huawei NPUs.

## Prerequisites

- Hardware: Huawei Ascend NPU (for example, Atlas A2/910B)
- Environment: Completed vLLM Ascend runtime installation

## Quick Start

```python
from vllm import LLM, SamplingParams

model_path = "facebook/chameleon-7b"

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    enforce_eager=True,
)

prompts = [
    "Hello, who are you?",
    "The capital of France is",
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=150)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```
