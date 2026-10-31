# Non-Contiguous KV Cache

## 1. Overview

vLLM normally exposes a logical KV-cache block as a regular tensor view. On
Ascend, hybrid models can allocate Attention and Mamba cache groups from a
common page-aligned cache pool. A physical page may contain alignment padding
and different cache views may use different regions or backing tensors, so
treating every cache as a dense K/V tensor can waste memory or produce an
incorrect stride.

This feature keeps the logical block table and block IDs unchanged while
constructing strided tensor views over the existing physical allocation. It is
implemented in Model Runner V1 and applies to hybrid Attention + Mamba models
and supported pure GQA Attention models.

## 2. Goals and Non-Goals

The implementation aims to:

- preserve vLLM's logical block and block-table semantics;
- avoid copying padded pages into a dense K/V allocation;
- expose correct shape, stride, and storage-offset metadata to Ascend kernels;
- support separate Attention and Mamba cache groups in one allocation plan;
- make the layout observable through temporary debug logs.

It does not change token scheduling, block-table numbering, or the logical
Attention/Mamba cache specifications.

## 3. Physical and Logical Layout

A logical block is the unit addressed by the scheduler. A physical page is the
byte range reserved by the allocator. Their sizes can differ when a page is
aligned for a hybrid cache group.

For an Attention cache, the raw allocation is viewed as a block-major tensor.
When the physical page contains padding, the view uses a block stride based on
the full page size rather than the unpadded logical tensor size:

```text
logical view:  [block, kv_head, token, head_dim]
physical page: [logical elements][alignment padding]
block stride:  physical page size / element size
```

The view is non-contiguous by design. K and V remain separate logical views,
even when they share the same backing allocation.

### 3.1 Contiguous Compatibility Layout

The following layout is the legacy compatibility path used when the
implementation needs all state tensors to remain contiguous:

```text
tensor 1: [KV padding][conv state]...
tensor 2: [K blocks][SSM state]...
tensor 3: [V blocks][Mamba padding]...
```

This diagram must not be interpreted as the non-contiguous layout. It describes
how the physical buffer is split when padding is materialized explicitly.

### 3.2 Non-Contiguous Layout

The non-contiguous path does not require K, V, convolution state, and SSM state
to be concatenated into one logical tensor. Instead, each cache group is
exposed through its own view:

```text
Attention raw allocation  -> K view and V view
Mamba raw allocation      -> conv-state view and SSM-state view
```

The views can have different shapes, strides, and storage offsets. They use the
same logical block numbering and, where the cache pool requires it, the same
physical page stride. The implementation must not assume that an Attention K
region, an Mamba SSM region, and an Attention V region are always adjacent or
interleaved within one raw tensor.

Whether two layers share a backing tensor is determined by the cache allocator
and layer mapping. A shared backing tensor is interpreted through the existing
shared-layout path; otherwise each layer receives its own strided view.

### 3.3 Layout Comparison

| Aspect | Contiguous compatibility path | Non-contiguous path |
| --- | --- | --- |
| K/V representation | Dense or explicitly sliced tensors | Independent strided K/V views |
| Mamba state representation | Explicitly split padded regions | Views advanced by physical page stride |
| Padding | Materialized as tensor regions | Skipped by stride or storage offset |
| Data movement | May require a dense staging view | Designed to avoid a full-page copy |
| Main invariant | Keep every state tensor contiguous | Preserve the physical allocation and correct strides |

The two paths use the same logical block IDs, but they do not have the same
physical interpretation. A change to the contiguous compatibility layout must
not be used as evidence about the non-contiguous layout.

### 3.4 Example: Block Stride

Assume a logical Attention block contains 48 `float16` elements, while the
hybrid page is aligned to 128 bytes:

```text
logical payload = 48 elements * 2 bytes = 96 bytes
physical page   = 128 bytes
block stride    = 128 / 2 = 64 elements
```

The next logical block therefore starts 64 elements after the current block,
not 48 elements after it. The 16-element gap is not part of the logical view.

For a combined K/V allocation, a simplified two-lane view can be represented
as:

```text
raw allocation: [K lane][V lane]
logical shape:  (2, blocks, block_tokens, kv_heads, head_dim)
```

The first lane addresses K and the second lane addresses V. The lane stride and
the block stride are derived from the physical allocation; they are not inferred
from the logical tensor size alone.

## 4. Cache Construction Flow

`NPUModelRunner._reshape_kv_cache_tensors` performs the following steps:

1. Read the logical `KVCacheSpec` for each cache group.
2. Select the kernel block size for the group.
3. Identify whether the raw allocation is a separate K/V tuple, a combined
   tensor, or a Mamba state allocation.
4. Build typed views with `view`, `as_strided`, or the existing Ascend layout
   helper without copying the backing storage.
5. Preserve the logical layer-to-cache mapping used by the worker.

The relevant implementation points are:

```text
_allocate_kv_cache_tensors()
    Allocate the raw backing tensor for each cache layer or group.

_reshape_kv_cache_tensors()
    Select the Attention/Mamba path and construct typed cache views.

_adjust_kv_layout()
    Build Mamba views whose block dimension advances by the physical page size.

gather_ssm_states()
    Gather indexed strided state rows into a kernel-friendly tensor.

scatter_ssm_states_()
    Write updated kernel state rows back to the original allocation.
```

The scheduler continues to operate on logical block IDs. These helpers only
change how a worker maps a logical ID to a physical address.

Hybrid Attention uses the common hybrid allocation contract when
`use_hybrid_blocks` is enabled. Pure GQA Attention can use the block-major
non-contiguous view without requiring a Mamba layer in the same model.

## 5. Mamba State Access

Mamba convolution and SSM states can have a physical stride larger than their
logical row size. The Mamba state helpers therefore gather the requested rows
into a kernel-friendly tensor and scatter updates back to the original strided
allocation.

The gather/scatter path preserves:

- the request-to-state index mapping;
- the `has_initial_state` mask;
- the storage padding between physical pages;
- in-place updates of the original cache allocation.

The state index tests cover both zero padding and padded physical rows.

The access flow is therefore:

```text
block table / state indices
          |
          v
strided physical state
          |
          +--> gather --> FLA or Mamba kernel
          |                  |
          +<-- scatter <-----+
```

## 6. FLA Operator Integration

The standard GDN preprocessing and recurrent paths use the FLA AscendC
interfaces:

```python
from fla_npu.ops.ascendc import (
    causal_conv1d_fn,
    causal_conv1d_update,
    recurrent_gated_delta_rule,
)
```

The FLA tests use independent PyTorch golden implementations and are placed
outside the vLLM-Ascend `e2e` and `ops` conftest trees. This prevents unrelated
Triton/custom-op initialization from changing the FLA runtime state in the
same process.

## 7. Observability

While the layout is still under validation, Model Runner emits debug messages
with the following prefix:

```text
[non-contiguous-kv-cache]
```

Attention logs include the mode (`hybrid` or `gqa`), shape, stride, and K/V
contiguity. Mamba logs include page size, state shapes, strides, storage
offsets, and contiguity. These logs are temporary observation points and can be
removed after the layout implementation is considered stable.

## 8. Compatibility and Limitations

- The feature is implemented for Model Runner V1 cache construction.
- The logical block table and scheduler-visible block IDs remain unchanged.
- The exact physical layout depends on the model's Attention and Mamba cache
  specifications and the selected kernel block sizes.
- Sparse, compressed, offloaded, or connector-specific cache paths must retain
  their own layout contracts; they should not be assumed to accept a strided
  view without dedicated validation.
- Graph capture, PD disaggregation, and KV connectors require end-to-end
  validation for each supported configuration.

When a path uses a connector, sparse cache, compressed cache, or offload
backend, its registration and transfer contract may impose a separate physical
layout. The non-contiguous view must not be enabled for that path solely because
the model is a hybrid or GQA model; the connector-specific path needs its own
validation.

## 9. Testing

The PR covers the following test categories:

- Model Runner unit tests for hybrid and pure-GQA cache views;
- Mamba state gather/scatter tests, including physical padding;
- FLA causal convolution and recurrent GDN numerical tests;
- GDN layerwise KV connector tests with FLA operator mocks;
- debug-log assertions confirming hybrid and GQA non-contiguous layouts.

The intended validation matrix is:

| Area | Cases |
| --- | --- |
| Cache construction | Hybrid Attention + Mamba; pure GQA; padded pages; combined K/V allocation |
| State access | Zero padding; non-zero padding; indexed gather; in-place scatter |
| FLA operators | `causal_conv1d_fn`; `causal_conv1d_update`; `recurrent_gated_delta_rule` |
| Execution modes | Eager execution; graph capture after the layout is enabled |
| Scheduling | Prefill; decode; MTP/speculative metadata where the model path enables it |
| Distributed paths | PD disaggregation and KV connectors only after their transfer layout is validated |

The unit tests validate shapes, strides, storage offsets, and dispatch metadata.
FLA single-operator numerical tests are maintained in the FLA repository.
The Ascend nightly state-index tests compare the local Triton helpers against
PyTorch references, including unselected rows and physical page padding.
End-to-end model tests are still required for each
graph, connector, and distributed configuration claimed as supported.

Representative NPU commands are:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py
pytest -q tests/ut/worker/test_model_runner_v1.py
```
