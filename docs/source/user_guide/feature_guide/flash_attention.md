# Flash Attention 3

```{note}
Flash Attention 3 on Ascend is currently in beta. The `flash_attn_npu` package required for FA3 has not yet been open-sourced and is expected to be released by the end of May.
```

This document shows how to enable Flash Attention 3 (FA3) in vLLM-Ascend. FA3 provides a training-inference consistent attention implementation for Ascend NPUs.

## Motivation

In RL training frameworks such as veRL, the attention computation during training uses Flash Attention. When vLLM-Ascend serves as the inference backend, the default Fused Infer Attention (FIA) implementation differs from the training-side Flash Attention, which can lead to training-inference inconsistency. To address this, vLLM-Ascend introduces the FA3 attention backend to maintain consistency with the training side.

FA3 is crucial for the following scenarios:

- **Training-inference consistency**: Ensures that the attention computation during inference matches the training side, which is essential for RL workflows (e.g., veRL) where inference results are used to compute training signals.
- **Framework debugging**: Consistent attention implementations make it easier to debug issues by eliminating discrepancies between training and inference.
- **Reinforcement Learning (RL)**: RL training often requires deterministic and consistent rollouts for reproducibility and stable training.

## Differences from GPU Implementation

```{note}
This section is to be supplemented. FA3 on Ascend NPU has certain differences from the GPU-side Flash Attention implementation. Specific differences will be documented here in future updates.
```

## Hardware Requirements

FA3 currently requires Ascend Atlas A2 and A3 inference products NPUs.
We will support other NPUs in the future.

## Software Requirements

FA3 requires the `flash_attn_npu` package, which provides the `flash_attn_v3` module with the `flash_attn_with_kvcache` operator.

```{warning}
The `flash_attn_npu` package has not yet been open-sourced. It is expected to be released by the end of the month. Until then, external users cannot directly use Flash Attention 3.
```

Once `flash_attn_npu` is available, install it before using FA3:

```bash
pip install flash_attn_npu
```

## Enabling Flash Attention 3

To enable FA3, you need to:

1. Set the environment variable `export VLLM_BATCH_INVARIANT=1` to enable batch invariant mode
2. Specify the attention backend as `FLASH_ATTN` via the LLM parameter `attention_backend="FLASH_ATTN"`

### Online Inference (Server Mode)

To start a vLLM server with FA3 enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-8B --attention-backend FLASH_ATTN
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

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

For offline batch inference with FA3:

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
    attention_backend="FLASH_ATTN",
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Limitations

- **Package not yet open-sourced**: The `flash_attn_npu` package required for FA3 has not yet been released. External users cannot use FA3 until the package is available.
- **Sliding window not supported**: FA3 does not support sliding window attention. Models that require sliding window need to use the default FIA backend.
- **ACL graph capture not supported**: FA3 does not support ACL graph capture with `FULL_DECODE_ONLY` mode. Please use `enforce_eager=True` when enabling FA3.
- **MLA and SFA not supported**: FA3 does not support Multi-head Latent Attention (MLA) or Sparse Flash Attention (SFA).

```{note}
Enabling FA3 may cause performance degradation compared to the default FIA backend. This trade-off is intentional to guarantee training-inference consistency.
```

## Tested Models

FA3 has been tested and verified on the following models:

- **Qwen3 (Dense)**: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`

Other models have not been tested yet and will be supported in the future if not supported after been tested.

## Future Improvements

The FA3 feature is under active development. Planned improvements include:

- Open-source the `flash_attn_npu` package
- Document differences from GPU-side Flash Attention implementation
- Support for additional NPUs series
- Expanded model coverage
- Performance optimizations
- Additional testing and validation
