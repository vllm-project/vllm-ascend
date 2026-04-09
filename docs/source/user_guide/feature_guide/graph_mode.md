# Graph Mode Guide

## Overview

This guide explains how graph mode is used in vLLM Ascend.

vLLM already provides the generic graph-mode architecture, mode definitions, and compile integration. For those upstream concepts, see:

- [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)

This document focuses on the Ascend-specific view: which graph backends are available, how to enable them, and what constraints users should keep in mind on Ascend.

## Current Status on Ascend

- Graph mode is currently available only on the **V1 Engine**.
- **ACLGraph** is the default graph path in vLLM Ascend.
- **XliteGraph** is an optional graph path for selected model families and environments.
- In context parallel scenarios, `cudagraph_mode="FULL"` is not sufficiently supported yet.

## Graph Backends on Ascend

vLLM Ascend currently exposes two graph backends:

| Backend | Default | Typical usage | Notes | Since |
|---|---|---|---|---|
| ACL Graph | Yes | General graph mode on Ascend | Default path in vLLM Ascend | v0.9.0rc1 |
| XliteGraph | No | Selected models with Xlite installed | Requires additional installation and config | v0.11.0 |

## Using ACLGraph

ACLGraph is enabled by default when the model runs on the V1 Engine and graph mode is available.

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

- ACLGraph and XliteGraph have different support coverage. XliteGraph should be treated as an alternative backend, not as a drop-in replacement for all ACLGraph scenarios.
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
- [ACL Graph Developer Guide](../../developer_guide/Design_Documents/ACL_Graph.md)
