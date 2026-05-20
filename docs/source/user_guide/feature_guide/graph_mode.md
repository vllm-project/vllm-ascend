# Graph Mode Guide

## Overview

This guide explains how graph mode is used in vLLM Ascend.

vLLM already provides the generic graph-mode architecture, mode definitions, and compile integration. For those upstream concepts, see:

- [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)

This document focuses on the Ascend-specific view: how graph mode works on Ascend, which components are involved, how to configure them, and what constraints users should keep in mind.

## Current Status on Ascend

- Graph mode is currently available only on the **V1 Engine**.
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

- **FULL / FULL_DECODE_ONLY**: Npugraph_ex optimizes the FX graph via torchair (`run_eagerly=True`, compile-time only, no capture). The optimized callable is then captured and replayed by ACLGraph at runtime.
- **PIECEWISE**: Npugraph_ex is disabled. Only basic FX fusion passes are applied at compile-time. ACLGraph captures and replays the resulting callable at runtime.
- **NONE**: No compilation or graph capture. The model runs in eager mode.

| `cudagraph_mode` | Compile-time | Runtime | Npugraph_ex |
|---|---|---|---|
| FULL / FULL_DECODE_ONLY | Npugraph_ex FX optimization | ACLGraph capture/replay | Enabled (default) |
| PIECEWISE | Fusion pass only | ACLGraph capture/replay | Disabled |
| NONE | None | Eager execution | Disabled |

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

## Using Npugraph_ex

As introduced in the [RFC](https://github.com/vllm-project/vllm-ascend/issues/4715), Npugraph_ex is a compile-time FX graph optimization layer that works together with ACLGraph. It optimizes the model's FX graph before ACLGraph captures it at runtime. Its performance benefits mainly come from fusing multiple operators into single kernels (e.g., add + rms_norm → npu_add_rms_norm) to reduce kernel launch overhead.

### Default behavior

Npugraph_ex is **enabled by default** when `cudagraph_mode` is `FULL` or `FULL_DECODE_ONLY`. It is automatically disabled in `PIECEWISE` or `NONE` modes.

This means for most users, Npugraph_ex is active without any explicit configuration:

```python
from vllm import LLM

# Npugraph_ex is enabled by default in FULL/FULL_DECODE_ONLY mode
llm = LLM(model="path/to/Qwen2-7B-Instruct")
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

```{note}
Enabling static kernel triggers a compilation pass during the graph capture phase at service startup. This may add **several minutes to tens of minutes** to the startup time depending on the number of operators to compile and model complexity. Once completed, subsequent request processing is not affected.
```

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

For more details about Npugraph_ex, see the [torchair guide](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00021.html).

## Using XliteGraph

XliteGraph is an optional path for Llama, Qwen dense series models, Qwen MoE series models, and Qwen3-VL. It requires Xlite to be installed and configured through `xlite_graph_config`.

Install Xlite first:

```bash
pip install xlite
```

Offline example:

```python
from vllm import LLM

# Xlite supports decode-only mode by default.
# Full mode can be enabled with "full_mode": True.
llm = LLM(
    model="path/to/Qwen3-32B",
    tensor_parallel_size=8,
    additional_config={
        "xlite_graph_config": {
            "enabled": True,
            "full_mode": True,
        }
    },
)
outputs = llm.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve path/to/Qwen3-32B \
  --tensor-parallel-size 8 \
  --additional-config '{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

For more details about Xlite, see the [Xlite README](https://atomgit.com/openeuler/GVirt/blob/master/xlite/README.md).

## Common Limitations and Caveats

- XliteGraph should be treated as an alternative graph path, not as a drop-in replacement for ACLGraph in all scenarios.
- Model and backend coverage is still evolving, so a configuration that works for one model family may not yet be recommended for another.

## Fallback to Eager Mode

If you encounter issues with graph mode, you can temporarily fall back to eager mode by setting `enforce_eager=True`.

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
- [Npugraph_ex torchair guide](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/torchairuseguide/torchair_00021.html)
- [Npugraph_ex RFC](https://github.com/vllm-project/vllm-ascend/issues/4715)
- [ACL Graph Developer Guide](../../developer_guide/Design_Documents/ACL_Graph.md)
