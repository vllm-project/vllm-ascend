# Flash Attention 3

```{note}
Flash Attention 3 on Ascend is currently in beta. The `flash_attn_npu` package required for FA3 has not yet been open-sourced and is expected to be released after May 20th.
```

This document shows how to enable Flash Attention 3 (FA3) in vLLM-Ascend. FA3 provides a training-inference consistent attention implementation for Ascend NPUs.

## Motivation

In RL training frameworks such as veRL, the attention computation during training uses Flash Attention. When vLLM-Ascend serves as the inference backend, the default Fused Infer Attention (FIA) implementation differs from the training-side Flash Attention, which can lead to training-inference inconsistency. To address this, vLLM-Ascend introduces the FA3 attention backend to maintain consistency with the training side.

FA3 is crucial for the following scenarios:

- **Training-inference consistency**: Ensures that the attention computation during inference matches the training side, which is essential for RL workflows (e.g., veRL) where inference results are used to compute training signals.
- **Framework debugging**: Consistent attention implementations make it easier to debug issues by eliminating discrepancies between training and inference.
- **Reinforcement Learning (RL)**: RL training often requires deterministic and consistent rollouts for reproducibility and stable training.

## Feature Comparison

The following table compares the features of `flash_attn_with_kvcache` between GPU FA3 and Ascend NPU FA3:

| Feature | GPU FA3 | NPU FA3 |
|---------|---------|---------|
| FP16 (float16) | ✅ | ✅ |
| BF16 (bfloat16) | ✅ | ✅ |
| Causal Attention | ✅ | ✅ |
| Sliding Window Attention | ✅ | - |
| MQA/GQA | ✅ | ✅ |
| Paged KV Cache | ✅ | ✅ |
| Rotary Position Embedding (RoPE) | ✅ | - |
| ALiBi | - | - |
| Softcapping | ✅ | - |
| FP8 Quantization | ✅ | - |
| Variable-length Sequences | ✅ | ✅ |

### Differences from GPU Implementation

The `flash_attn_with_kvcache` interface on NPU is semantically consistent with the GPU FA3 version in terms of API parameters. The key differences are:

1. **Unsupported features on NPU FA3**: Sliding window attention, RoPE, ALiBi, Softcapping, and FP8 quantization are not yet supported. These will be added in future releases.
2. **Graph capture**: The tiling of `flash_attn_with_kvcache` is processed on the host side and is currently being optimized. It does not support ACL graph capture (i.e., cannot be captured into a computational graph for acceleration). Please use `enforce_eager=True` when enabling FA3.

## Hardware Requirements

FA3 currently requires Ascend Atlas A2 and A3 inference products NPUs.
We will support other NPUs in the future.

## Software Requirements

FA3 requires the `flash_attn_npu` package, which provides the `flash_attn_v3` module with the `flash_attn_with_kvcache` operator.

```{warning}
The `flash_attn_npu` package has not yet been open-sourced. It is expected to be released after May 20th. Until then, external users cannot directly use Flash Attention 3.
```

### Installation

Install the `flash_attn_npu` wheel package as follows:

```bash
pip3 install flash_attn_npu-x.x.x-cp3xx-cp3xx-linux_aarch64.whl
```

```{note}
Replace `x.x.x` with the actual package version, and `cp3xx` with the actual Python version tag matching your environment (e.g., `cp310` for Python 3.10, `cp311` for Python 3.11).
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
- **ACL graph capture not supported**: The tiling of `flash_attn_with_kvcache` is processed on the host side and currently does not support ACL graph capture. Please use `enforce_eager=True` when enabling FA3.
- **RoPE not supported**: FA3 does not support rotary position embedding within the attention kernel. vLLM-Ascend patches this by using the PyTorch native RoPE fallback instead.
- **ALiBi not supported**: FA3 does not support ALiBi (Attention with Linear Biases).
- **Softcapping not supported**: FA3 does not support attention logit softcapping.
- **FP8 quantization not supported**: FA3 does not support FP8 quantized attention.
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
- Support ACL graph capture (host-side tiling optimization)
- Support sliding window attention
- Support RoPE within the attention kernel
- Support ALiBi
- Support Softcapping
- Support FP8 quantization
- Support for additional NPUs series
- Expanded model coverage
- Performance optimizations
- Additional testing and validation
