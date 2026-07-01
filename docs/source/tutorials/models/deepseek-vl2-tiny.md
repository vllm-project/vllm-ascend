# DeepSeek-VL2-Tiny Model Validation Report

## Basic Information

Model Name: deepseek-ai/deepseek-vl2-tiny

Model Type: Multimodal Vision-Language Model

Release Date: December 2024

Validation Environment: Ascend 910B + vLLM 0.11.0 + PyTorch 2.7.1

## Validation Conclusion

The model cannot be run directly and requires adaptation for support.

## Problem Description

### Error Phenomenon

The following error occurs when loading the model:

pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig

Value error, No model architectures are specified

### Root Cause Analysis

1. Special model configuration format

DeepSeek-VL2's config.json uses a nested structure:

- No architectures field at the top level

- Architecture information is located in language_config.architectures

2. vLLM does not adapt to nested format

The current vLLM version (0.11.0) only looks for the architectures field at the top level when reading the model configuration, with no logic for recursive lookup in nested fields.

### Validation Method

cat config.json | grep -A 5 "language_config"

Output:

"language_config": {
"architectures": ["DeepseekV2ForCausalLM"]
}

## Adaptation Gap Analysis

Ascend NPU Environment: Normal

vLLM Basic Dependencies: Normal

Model Architecture Recognition: Missing

## Adaptation Suggestions

### Code Modification Direction

Modify the create_model_config method in vllm/engine/arg_utils.py:

def get_architectures(config):
    # First look at top level
    if "architectures" in config:
        return config["architectures"]
    # Then look in language_config
    elif "language_config" in config and "architectures" in config["language_config"]:
        return config["language_config"]["architectures"]
    else:
        raise ValueError("No model architectures are specified")

### Environment Constraints

- Ascend NPU requires torch-npu, which supports PyTorch up to 2.9.0

- vLLM 0.18.0 requires PyTorch 2.10.0

- The two cannot be satisfied simultaneously; adaptation must be based on vLLM 0.11.0

## Test Steps

### 1. Environment Setup

Create an Ascend 910B container instance

Image: PyTorch 2.7.1 + CANN 8.3.RC2

### 2. Dependency Check

npu-smi info

python -c "import torch_npu; print('NPU OK')"

### 3. Model Loading Test

from vllm import LLM

model = LLM(model="deepseek-ai/deepseek-vl2-tiny")

## Related Resources

Model HuggingFace Page: https://huggingface.co/deepseek-ai/deepseek-vl2-tiny

vLLM Documentation: https://docs.vllm.ai/

vLLM Ascend Documentation: https://docs.vllm.ai/projects/ascend/en/latest/

Task Issue #7319: https://github.com/vllm-project/vllm-ascend/issues/7319

## Report Date

March 21, 2026
