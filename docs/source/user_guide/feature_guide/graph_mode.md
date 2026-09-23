# Graph Mode Guide

## Overview

This guide explains how graph mode is used in vLLM Ascend.

vLLM already provides the generic graph-mode architecture, mode definitions, and compile integration. For those upstream concepts, see:

- [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)

This document focuses on the Ascend-specific view: how graph mode works on Ascend, which components are involved, how to configure them, and what constraints users should keep in mind.

## Current Status on Ascend

- Graph mode is currently available on both Model Runner V1/V2.
- **ACLGraph** (capture/replay via `torch.npu.NPUGraph`) is the runtime graph execution mechanism used by the default graph path on Ascend.
- **Npugraph_ex** is a compile-time FX graph optimization layer, enabled by default in FULL/FULL_DECODE_ONLY modes. It optimizes the graph before ACLGraph captures it.
- **XliteGraph** is an optional graph path for selected model families and environments.
- In context parallel scenarios, `cudagraph_mode="FULL"` is not sufficiently supported yet.

## Graph Paths on Ascend

vLLM Ascend provides two graph paths:

| Graph Path | Default | Description | Since |
|---|---|---|---|
| ACLGraph (+ Npugraph_ex) | Yes | Compile-time FX optimization (Npugraph_ex) + runtime capture/replay (ACLGraph) | v0.9.0rc1 (Npugraph_ex since v0.15.0rc1) |
| XliteGraph | No | Preconfigured graph path for selected model families. Requires separate installation | v0.11.0 |

## How Graph Mode Works on Ascend

The default graph path on Ascend involves two stages: **compile-time optimization** and **runtime capture/replay**. ACLGraph handles the runtime capture/replay. The compile-time stage differs by `cudagraph_mode`:

- **FULL_AND_PIECEWISE**: Default mode, same as the upstream vLLM strategy. The compile-time path follows PIECEWISE compilation, while the runtime may still use full-graph behavior for uniform decode batches.
- **FULL / FULL_DECODE_ONLY**: Npugraph_ex optimizes the FX graph via npugraph_ex (`force_eager=True`, compile-time only, no capture). The optimized callable is then captured and replayed by ACLGraph at runtime.
- **PIECEWISE**: Npugraph_ex is disabled. Only basic FX fusion passes are applied at compile-time. ACLGraph captures and replays the resulting callable at runtime.
- **NONE**: No compilation or graph capture. The model runs in eager mode.

| `cudagraph_mode` | Compile-time | Runtime | Npugraph_ex |
|---|---|---|---|
| FULL_AND_PIECEWISE | Piecewise compilation path | Mixed: PIECEWISE for mixed batches, FULL-capable for uniform decode batches | Disabled |
| FULL / FULL_DECODE_ONLY | Npugraph_ex FX optimization | ACLGraph capture/replay | Enabled |
| PIECEWISE | Fusion pass only | ACLGraph capture/replay | Disabled |
| NONE | None | Eager execution | Disabled |

Ascend sets `use_inductor=False` and selects its own compilation backend. That backend applies npugraph_ex optimizations or FX fusion passes according to the configuration.

Additionally, **XliteGraph** is available as an optional alternative graph path for selected model families (see [Using XliteGraph](#using-xlitegraph)).

## Using ACLGraph

ACLGraph is the runtime graph capture/replay mechanism on Ascend. It is enabled automatically when graph mode is active (i.e., `cudagraph_mode` is not `NONE`), and does not require explicit configuration.

### Basic usage

Offline example:

```python
from vllm import LLM

llm = LLM(model="path/to/Qwen3-0.6B")
outputs = llm.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen3-0.6B
```

### Explicit `cudagraph_mode` configuration

The generic `cudagraph_mode` options come from upstream vLLM. On Ascend, the final effective mode may still be adjusted according to platform and backend support, so the official vLLM CUDA Graphs document remains the canonical reference for mode semantics.

CLI example:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

Python example:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    compilation_config={"cudagraph_mode": "PIECEWISE"},
)
```

For the detailed meaning of `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, and `FULL_AND_PIECEWISE`, as well as the generic fallback policy, see the upstream [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/) design doc.

### Default capture sizes for speculative PD decode

When neither `max_cudagraph_capture_size` nor `cudagraph_capture_sizes` is set,
Ascend derives the initial maximum from `max_num_seqs * decode_query_len`.
For speculative decoding, `decode_query_len` is `1 + num_speculative_tokens`.
The usual default cap is 512 scheduled tokens. A disaggregated decode instance
with `kv_role="kv_consumer"`, speculative decoding, `FULL_DECODE_ONLY`, and no
`enforce_eager` instead uses a cap of 1024. Explicit size settings take precedence.

For example, `max_num_seqs=256` and three speculative tokens give a default
maximum of 1024, covering the configured uniform decode capacity of each DP
rank. With the previous 512-token cap, 128 requests fit but 129 require 516
tokens. A balanced DP2 deployment can cross that boundary at a total concurrency
of 257, causing synchronized ranks to leave full-graph execution.

The cap is a resource limit, not a guarantee of full-graph coverage for every
configuration. For example, 256 requests with seven speculative tokens can
require 2048 tokens. Ascend warns when the configured uniform decode capacity
exceeds the 1024-token default cap. Check graph memory and KV-cache capacity
before explicitly increasing it. Larger capture sizes can increase capture
time and graph memory, reduce available KV-cache space, and affect communication
buffer sizing and method selection.

The upstream token budget, size alignment, and Ascend graph filtering still
apply, so inspect the final capture sizes in the startup logs. At runtime, the
dispatcher selects from the captured sizes; setting a maximum of 1024 does not
pad every batch to 1024. Requests beyond the scheduler's running capacity wait
rather than increasing the graph size automatically.

### Attention backend compatibility

Not all attention backends support all graph modes. vLLM checks attention backend compatibility during compatibility checks and, when possible, automatically adjusts `cudagraph_mode` to a more compatible mode instead of failing immediately. In practice, this means a requested full-graph mode may be narrowed to a mixed or piecewise mode, and if the backend cannot support graph execution at all, graph mode may be disabled.

On Ascend, the current attention backend support levels are:

| Attention backend | Declared support | Practical meaning |
|---|---|---|
| `attention_v1` | `ALWAYS` | Supports graph execution for mixed prefill/decode batches |
| `context_parallel/attention_cp` | `ALWAYS` | Supports graph execution for mixed prefill/decode batches |
| `mla_v1` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `context_parallel/mla_cp` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `sfa_v1` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `context_parallel/sfa_cp` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |

This is why the effective graph mode on Ascend may differ from the mode requested in configuration.

### Troubleshooting capture resource exhaustion

If ACLGraph capture fails because the configured graph sizes exceed the runtime resources available on the current stack, vLLM Ascend now raises a dedicated error with mitigation guidance. In practice, the most useful actions are:

- upgrade to a newer HDK/CANN stack if one is available;
- reduce `cudagraph_capture_sizes` or `max_cudagraph_capture_size`;
- prefer `FULL` or `FULL_DECODE_ONLY` when the workload is mostly uniform decode;
- temporarily disable graph mode to confirm the issue is capture-related.

This is most likely to appear in `PIECEWISE` or `FULL_AND_PIECEWISE` configurations because those paths tend to capture more graphs than uniform full-graph decode.

## Using Npugraph_ex

As introduced in the [RFC](https://github.com/vllm-project/vllm-ascend/issues/4715), Npugraph_ex is a compile-time FX graph optimization layer that works together with ACLGraph. It optimizes the model's FX graph before ACLGraph captures it at runtime. Its performance benefits mainly come from fusing multiple operators into single kernels (e.g., add + rms_norm → npu_add_rms_norm) to reduce kernel launch overhead.

!!! note "Atlas 300I DUO"

    Atlas 300I DUO and Atlas 200I Pro do not support `enable_npugraph_ex`. Set --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'.

### Default behavior

Npugraph_ex is **enabled by default** on supported hardware when `cudagraph_mode` is `FULL` or `FULL_DECODE_ONLY`. It is automatically disabled in `FULL_AND_PIECEWISE`, `PIECEWISE`, and `NONE` modes.

Select a full graph mode to use this default:

```python
from vllm import LLM

# Npugraph_ex is enabled by default in FULL/FULL_DECODE_ONLY mode
llm = LLM(model="path/to/Qwen2-7B-Instruct", compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})
outputs = llm.generate("Hello, how are you?")
```

### Explicit configuration

To explicitly control Npugraph_ex:

Offline example:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true}}'
```

To disable Npugraph_ex explicitly:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

### Static kernel compilation

Static kernel compilation is an **optional** feature that pre-compiles operator binaries with fixed shapes at compile time, reducing runtime overhead for networks with static or near-static shapes. It is **disabled by default** and must be explicitly enabled.

!!! note

    Enabling static kernel triggers a compilation pass during the graph capture phase at service startup. This may add **several minutes to tens of minutes** to the startup time depending on the number of operators to compile and model complexity. Once completed, subsequent request processing is not affected.

Offline example:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

#### Verifying static kernel is active

The recommended way to verify static kernel is in effect is through **Ascend Profiling**:

1. Collect a profiling trace of your running model using [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md) (`torch_npu.profiler`).
2. Open the generated `op_statistic.csv` file.
3. Look for operators whose `op_type` or `name` column contains the keyword **`static_kernel`**. If such entries exist, static kernel compilation has taken effect for those operators.

During the compilation phase, you will see a Python warning (visible by default):

```text
Starting static kernel compilation, the build directory is <path>
```

This confirms that compilation has been triggered. The absence of this message means static kernel was not enabled or the cached result was reused directly.

### Super Kernel optimization

[Super Kernel](https://www.hiascend.com/document/detail/zh/Pytorch/latest/devguide/TorchAir/docs/zh/npugraph_ex/advanced/superkernel.md) is an **optional** operator-binary fusion optimization. Unlike source-level fusion, it works on compiled kernel binaries: eligible subgraphs are identified, their child kernels are combined into a larger kernel, and synchronization is inserted according to graph dependencies. Compared with launching operators individually, this can reduce task scheduling waits, launch overhead, and operator-head overhead.

In vLLM Ascend, Super Kernel optimization is applied during ACLGraph capture. It depends on both static kernel and Npugraph_ex:

```text
Super Kernel -> static kernel -> Npugraph_ex
```

!!! note

    Static kernel is disabled by default, so Super Kernel is also disabled under the default configuration. When `enable_static_kernel` is explicitly set to `true` and `enable_super_kernel` is omitted, Super Kernel follows static kernel and is enabled. Set `enable_super_kernel` explicitly to override this inherited behavior. Super Kernel cannot be enabled when static kernel is disabled.

    Not every operator is eligible for Super Kernel fusion. An unsupported operator may split the fusion range, so the actual coverage and performance benefit depend on the model and operator sequence. Changing the Super Kernel setting also changes the static-kernel compilation configuration and may cause the static-kernel cache to be rebuilt.

Offline example:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": True,
            # `enable_super_kernel` is automatically enabled with `enable_static_kernel=True`.
            # It can also be set explicitly if preferred.
            # "enable_super_kernel": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

To use static kernel without Super Kernel:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true, "enable_super_kernel":false}}'
```

#### Verifying Super Kernel is active

When Super Kernel optimization is applied during ACLGraph capture, vLLM Ascend emits:

```text
Super kernel optimization is enabled for ACL graph capture.
```

This message confirms that the optimization step was invoked. To verify the resulting fusion coverage and performance benefit, collect an **Ascend Profiling** trace and inspect `kernel_details.csv`. Compare the kernel/task entries and latency with Super Kernel disabled; the profiler also reports the launch core count for generated Super Kernels in the `Block Dim` field.

For more details about Npugraph_ex, see the [npugraph_ex guide](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md).

## Using XliteGraph {: #using-xlitegraph }

XliteGraph is an optional graph path for the Llama and Qwen dense series, the Qwen MoE series, Qwen3-VL, and DeepSeek-V3 / V3.2-class MoE models (including GLM-4 / GLM-5 / GLM-5.1 and MiniMax-M2). It requires the `xlite` package to be installed and is configured through the `xlite_graph_config` additional-config key (see the `xlite_graph_config` table under [Configuration options](../configuration/additional_config.md#configuration-options)).

Install Xlite first:

```bash
pip install xlite
```

### Decode-only vs. full mode

XliteGraph has two modes, selected by `xlite_graph_config.full_mode`. In both modes, every batch is routed by its **token count** (the maximum number of tokens across all data-parallel ranks): batches that fit within the token budget xlite preallocates are executed by the xlite runtime — regardless of whether they contain prefill or decode — while larger batches fall back to the native runnable, captured/replayed by ACLGraph. The modes differ in the token budget xlite preallocates:

| Mode | `full_mode` | Preallocated token budget | Batches handled by the xlite runtime |
|---|---|---|---|
| **Decode-only** (default) | `False` | `max_num_seqs × (1 + num_speculative_tokens)` | Batches within the decode-sized budget — typically decode steps (including MTP), and any small prefill/mixed batch that happens to fit |
| **Full** | `True` | `max-num-batched-tokens` | Any batch up to `max-num-batched-tokens`, i.e., prefill, decode, and mixed batches alike |

In **decode-only mode** (the default), the budget is sized for a decode step: the maximum number of running requests times the per-request decode tokens (1, plus the speculative tokens when MTP-style speculative decoding is enabled). Batches that exceed this budget — such as prefill and large mixed batches — fall back to the native runnable. This keeps the xlite hidden-state workspace small, and is the recommended option — `Xlite` is still evolving, and decode-only mode is more performant and robust for most workloads.

In **full mode**, the budget is `max-num-batched-tokens`, so the xlite runtime drives prefill and decode alike; only batches that exceed `max-num-batched-tokens` (e.g., certain profile runs) fall back. Because xlite itself manages the full forward, ACLGraph capture should be disabled, especially under memory constraints, for higher concurrency and better performance.

!!! note

    Since v0.28.0, batches are routed by **token count** instead of the batch attention state: any batch within the preallocated budget runs on the xlite runtime, whether prefill, decode, or mixed. Before v0.28.0, decode-only mode only handled batches whose attention state was decode — any batch with a prefill was routed back to ACLGraph — so small prefill/mixed batches could never take the xlite path.

### Supported models

The xlite adapter registers architectures by their `config.json` `architectures` field. The currently supported set:

| Architecture | Attention | Adapter |
|---|---|---|
| `LlamaForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM` | MHA | `StandardXliteModel` |
| `Qwen3VLForConditionalGeneration` | MHA (multimodal) | `StandardXliteModel` |
| `Qwen3MoeForCausalLM`, `Qwen3VLMoeForConditionalGeneration` | MHA + MoE | `QwenMoeXliteModel` |
| `Glm4MoeForCausalLM` | MHA + MoE | `Glm4MoeXliteModel` |
| `DeepseekV3ForCausalLM` | MLA + MoE | `DeepseekV3XliteModel` |
| `DeepseekV32ForCausalLM`, `GlmMoeDsaForCausalLM` | DSA + MoE | `DeepseekV32XliteModel` |
| `MiniMaxM2ForCausalLM` | MHA + MoE | `MiniMaxM2XliteModel` |

### Example

Serving `Qwen3-30B-A3B` (`Qwen3MoeForCausalLM`, MHA + MoE) on 4 NPU cards with tensor parallelism of 4, and expert parallelism enabled. Full mode is shown; drop `"full_mode": true` for decode-only.

Offline example:

```python
import os

from vllm import LLM

os.environ["HCCL_BUFFSIZE"] = "1024"
os.environ["TASK_QUEUE_ENABLE"] = "1"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

llm = LLM(
    model="path/to/Qwen3-30B-A3B",
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    enforce_eager=True,  # recommended in xlite full mode without speculative decoding
    additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}},
)
outputs = llm.generate("Hello, how are you?")
```

Online example:

```bash
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve path/to/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --enforce-eager \
  --additional-config '{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

For more details about Xlite, see the [Xlite README](https://atomgit.com/openeuler/GVirt/blob/master/xlite/README.md).

## Common Limitations and Caveats

- XliteGraph should be treated as an alternative graph path, not as a drop-in replacement for ACLGraph in all scenarios.
- Model and backend coverage is still evolving, so a configuration that works for one model family may not yet be recommended for another.
- Encoder-decoder models currently do not keep `FULL_AND_PIECEWISE`; on Ascend they fall back to `PIECEWISE` or `NONE` depending on compilation support.

## Fallback to Eager Mode

If you encounter issues with graph mode, you can temporarily fall back to eager mode by setting `enforce_eager=True`.

If ACL graph capture fails with the confirmed stream-resource signature in the error text, such as `207008` together with `Stream resources are insufficient` or `Insufficient_Stream_Resources`, vLLM Ascend will re-raise that capture failure with targeted mitigation guidance. In practice, the main levers are: upgrading to a newer HDK/CANN stack, reducing `cudagraph_capture_sizes`, lowering `max_cudagraph_capture_size`, or preferring `FULL` / `FULL_DECODE_ONLY` when the workload is mostly uniform decode.

**Offline example:**

```python
from vllm import LLM

llm = LLM(model="path/to/your/model", enforce_eager=True)
outputs = llm.generate("Hello, how are you?")
```

**Online example:**

```bash
vllm serve path/to/your/model --enforce-eager
```

## References

- [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)
- [Xlite README](https://atomgit.com/openeuler/GVirt/blob/master/xlite/README.md)
- [Npugraph_ex guide](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md)
- [Npugraph_ex RFC](https://github.com/vllm-project/vllm-ascend/issues/4715)
- [ACL Graph Developer Guide](../../developer_guide/Design_Documents/ACL_Graph.md)
