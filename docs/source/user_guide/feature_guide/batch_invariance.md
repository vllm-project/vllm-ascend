# Batch Invariance

!!! note

    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at [tracking issue #5487](https://github.com/vllm-project/vllm-ascend/issues/5487)

!!! note

    To install the batch invariance custom operator library, set `VLLM_BATCH_INVARIANT=1` before building vllm-ascend.
    For installation instructions, see [installing in an existing CANN environment](../../getting_started/installation.md#installation-existing-cann-install).

This document shows how to enable batch invariance in vLLM-Ascend. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance currently requires Ascend Atlas A2 and A3 inference products NPUs.
We will support Ascend 950 Products and other NPUs in the future.

## Software Requirements

Batch invariance requires custom operators for Atlas A2 and A3 inference products. Set `VLLM_BATCH_INVARIANT=1` before building vllm-ascend from source to build and install the required operator packages.

The A2 and A3 Dockerfiles set `VLLM_BATCH_INVARIANT=1` while installing vllm-ascend from source, so the resulting images include both the batch-invariant operator run package and the `batch_invariant_ops` wheel. This is a build-time setting only; set `VLLM_BATCH_INVARIANT=1` again when starting the server or running offline inference to enable batch invariance at runtime.

The `batch_invariant_ops` build and installation process consists of two stages as in the [build_batch_invariant_ops.sh](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/build_batch_invariant_ops.sh), which must run in order:

1. Install the operator run package. It provides the device-side batch-invariant operators implemented with AscendC.
2. Build and install the `batch_invariant_ops` wheel. It provides the PyTorch extension interfaces that invoke the AscendC operators.

### Option 1: Install vllm-ascend from source

The environment variable is consumed by the source build. It works with both a regular source installation and an editable source installation when custom kernel compilation is enabled:

```bash
# Regular source installation
COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=1 \
    pip install . --no-build-isolation

# Editable source installation
COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=1 \
    pip install -e . --no-build-isolation
```

Setting `VLLM_BATCH_INVARIANT=1` while installing a prebuilt vllm-ascend wheel does not rebuild these operators. Use a source installation as above, or install the operator packages separately as described below.

### Option 2: vllm-ascend is already installed

Obtain a vllm-ascend source tree that matches the installed package version, then build and install the operator packages from that source tree.

!!! note

    A prebuilt vllm-ascend wheel does not include the `csrc` directory or `csrc/build_batch_invariant_ops.sh`. Therefore, the following commands cannot be run from the wheel's `site-packages` installation directory. `<vllm-ascend-source-dir>` must point to a matching vllm-ascend source checkout.

**A2:**

```bash
cd <vllm-ascend-source-dir>
bash csrc/build_batch_invariant_ops.sh ascend910b
```

**A3:**

```bash
cd <vllm-ascend-source-dir>
bash csrc/build_batch_invariant_ops.sh ascend910_93
```

### Quick Check

After installation, verify the ops are available:

```bash
python -c "
import batch_invariant_ops
import torch
op = torch.ops.batch_invariant_ops.npu_matmul_batch_invariant
print(op)
assert 'npu_matmul_batch_invariant' in str(op)
"
```

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
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
    compilation_config={"cudagraph_mode": "PIECEWISE"},
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
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-235B-A22B`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm-ascend/issues/new/choose).

## Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism

!!! note

    The batch invariance attention operators currently do not support
    `FULL`,`FULL_DECODE_ONLY` cudagraph mode.

!!! note

    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional NPUs series
- Support `FULL`,`FULL_DECODE_ONLY` cudagraph mode with batch invariance attention operators
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm-ascend/issues/5487).
