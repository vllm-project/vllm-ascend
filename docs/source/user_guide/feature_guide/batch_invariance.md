# Batch Invariance

```{note}
Batch invariance is currently in beta. Some features are still under active development.
Track progress and planned improvements at <https://github.com/vllm-project/vllm-ascend/issues/5487>
```

This document shows how to enable batch invariance in vLLM-Ascend. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance currently requires Ascend 910B NPUs, because only the 910B supports batch invariance with HCCL communication for now.
We will support other NPUs in the future.

## Software Requirements

Batch invariance requires a customed operator library for 910B.
We will release the customed operator library in future versions.

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-8B
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="Qwen/Qwen3-8B",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Tested Models

Batch invariance has been tested and verified on the following models:

- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm-ascend/issues/new/choose).

## Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism

```{note}
Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.
```

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional NPUs series
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm-ascend/issues/5487).
