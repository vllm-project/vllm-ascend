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

Batch invariance supports Atlas A2, A3, and Ascend 950 products.

## Software Requirements

Batch invariance requires custom operators for Atlas A2, A3, and Ascend 950 products. Set `VLLM_BATCH_INVARIANT=1` before building vllm-ascend from source to build and install the required operator packages.

The `batch_invariant_ops` build and installation process consists of two stages as in the [build_batch_invariant_ops.sh](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/build_batch_invariant_ops.sh), which must run in order:

1. Install the operator run package. It provides the device-side batch-invariant operators implemented with AscendC.
2. Build and install the `batch_invariant_ops` wheel. It provides the PyTorch extension interfaces that invoke the AscendC operators.

!!! note

    A prebuilt vllm-ascend wheel does not include the `csrc` directory or `csrc/build_batch_invariant_ops.sh`, and setting `VLLM_BATCH_INVARIANT=1` while installing that wheel does not rebuild the operators. Manual operator installation requires a matching vllm-ascend source checkout. `<vllm-ascend-source-dir>` in the following commands refers to that checkout, not the wheel's `site-packages` directory.

### Install from source

#### Option 1: Install vllm-ascend and the operator packages together

The environment variable is consumed by the source build. It works with both a regular source installation and an editable source installation when custom kernel compilation is enabled:

```bash
cd <vllm-ascend-source-dir>

# Regular source installation
COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=1 \
    pip install . --no-build-isolation

# Editable source installation
COMPILE_CUSTOM_KERNELS=1 VLLM_BATCH_INVARIANT=1 \
    pip install -e . --no-build-isolation
```

#### Option 2: Install the operator packages if vllm-ascend is already installed

Obtain a vllm-ascend source tree that matches the installed package version, then build and install the operator packages from that source tree.

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

**Ascend 950:**

```bash
cd <vllm-ascend-source-dir>
bash csrc/build_batch_invariant_ops.sh ascend950
```

### Use Docker images

The A2, A3, and Ascend 950 Docker images for Ubuntu and openEuler build vllm-ascend from source with `VLLM_BATCH_INVARIANT=1`, so the image build installs both the AscendC operator run package and the `batch_invariant_ops` wheel. This build-time environment variable is not retained as a runtime setting. Set `VLLM_BATCH_INVARIANT=1` when starting the server or running offline inference to enable batch invariance.

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
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 128
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
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    block_size=128,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Scheduling Limitations

Chunked prefill, prefix caching, and request preemption (eviction and recomputation) are not supported with batch invariance.

These scheduling features are not disabled automatically. You must explicitly disable chunked prefill and prefix caching in your configuration, and pair the chunked prefill disabling with a KV cache block size of 128 — pass `--block-size 128` together with `--no-enable-chunked-prefill` when starting the server, or `block_size=128` together with `enable_chunked_prefill=False` for offline inference — as shown in the examples above.

Request preemption is triggered when the KV cache runs out: the preempted request is evicted and recomputed later. The recomputed prefill includes the tokens generated before the preemption, so attention processes them through the prefill (P) path instead of the original decode (D) path — the P and D computations cannot be aligned, which breaks batch invariance. To reduce the chance of preemption, increase the available KV cache or lower the per-request and concurrent pressure:

- Decrease `--max-num-seqs` so fewer requests share the KV cache.
- Set `--max-model-len` to the smallest value your workload needs, and cap the per-request output length (`max_tokens`).
- Increase `--gpu-memory-utilization` to leave more memory for the KV cache.

Use the startup logs (`GPU KV cache size` and `Maximum concurrency for ... tokens per request`) to size your workload against the KV cache capacity, and watch the engine stats: `GPU KV cache usage` approaching 100% signals imminent preemption. See the [preemption FAQ](../../faqs.md#22-why-does-tpot-increase-drastically-as-concurrency-grows) for details.

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

    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional NPUs series
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm-ascend/issues/5487).
