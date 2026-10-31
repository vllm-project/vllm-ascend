# scatter_k_scales

## Description

- **Location**: `vllm_ascend/ops/triton/mxfp_k_scale.py`; Python entry point `scatter_k_scales`, private kernel `_scatter_k_scale_kernel`.
- **Function**: Scatter per-token K-scale bytes into the packet-packed PA_NZ cache used by C8-MXFP attention. Only the addressed token/head/group elements are written. Other cache pages, K/V payloads, V scales, and page padding remain untouched. The operator copies bytes; it does not calculate scales, quantize keys, or validate E8M0 values.
- **Formula**: For token `t`, head `h`, scale group `g`, and pair element `p`:

  ```text
  b = block_ids[t]
  s = segment_ids[t]
  f = fragment_ids[t]
  cache[b, h, s, g, f, p] = key_scale[t, h, g, p]

  destination_offset = b * cache.stride(0)
                     + h * cache.stride(1)
                     + s * cache.stride(2)
                     + g * cache.stride(3)
                     + f * cache.stride(4)
                     + p * cache.stride(5)
  ```

  All destination offsets are relative to `cache.data_ptr()`, which already includes the view's storage offset. For a slot `q` and kernel block size `B`, the caller derives `b = q // B`, `s = (q % B) // 16`, and `f = q % 16`.

- **Algorithm flow**:
    1. Return without launching a kernel when the block-index tensor is empty.
    2. Reinterpret source scales as uint8 and make the source contiguous. Read head/group counts and all six destination strides from cache metadata. Validate the source element count before launching the kernel.
    3. Launch one Triton program per 256 flattened source bytes, masking elements beyond `T * H * G * 2`.
    4. Decompose each flattened index into token, head, group, and pair coordinates. Load block/segment/fragment indices using their actual strides.
    5. Promote address coordinates to int64, calculate the destination offset, and store the unchanged source byte directly. No cache-wide temporary view or writeback is created.
- **Supported modes**: Ascend NPU eager execution and ACL graph capture/replay with Triton-Ascend. Hardware validation was performed on 950DT (Ascend A5). Atlas A2, Atlas A3, and 950PR validation is N/A; this document does not establish additional hardware support.

## Parameters

All three arguments are required. The Python function returns `None` and updates `cache` in place. `T` is the number of token rows, `H` the local KV-head count, `D` the Q/K head dimension, `G = D // 64`, and `B` the kernel block size.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `key_scale` | Input | Source scales of shape `[T, H, G, 2]`; viewed as raw bytes without numerical conversion | One-byte storage; uint8 and E8M0 views are tested | ND; noncontiguous sources are made contiguous |
| `cache` | Input/Output | K-scale view of shape `[num_blocks, H, B // 16, G, 16, 2]`, backed by packet-packed KV storage | uint8 | PA_NZ, with explicit six-dimensional strides and an optional storage offset |
| `slot_index` | Input | Tuple `(block_ids, segment_ids, fragment_ids)`; each tensor has shape `[T]` | int64 in the caller and tests | 1D; each index tensor's stride is used |
| Return value | Output | N/A; the destination view is modified in place | `None` | N/A |
| `num_elements` | Internal attribute | `T * H * G * 2`; excluded from JIT specialization with `do_not_specialize` | Integer, runtime | Scalar |
| `num_heads`, `num_groups` | Internal attributes | `H` and `G` | Integer, constexpr | Scalars |
| Cache and index strides | Internal attributes | Element strides supplied by the input tensor metadata | Integer, constexpr | Scalars |
| `BLOCK_SIZE`, `num_warps` | Internal attributes | 256 bytes per program and 4 warps | Integer, compile-time launch settings | Scalars |

## Constraints

- Source, destination, and all index tensors must reside on the same NPU. CPU inference is not supported by this update path. CPU unit tests mock the Triton entry point; numerical comparisons execute the real kernel on NPU against test-only CPU references.
- Source shape and element size must satisfy `key_scale.view(torch.uint8).numel() == T * H * G * 2`. For a nonempty index batch, a mismatched source element count raises `ValueError` before kernel launch, including under `python -O`. The current C8 caller uses `B=512`, `D=256`, and local KV-head counts of 1 or 2. The test allocation may use fewer pages than a model deployment while preserving these model layout dimensions.
- The cache must be a valid, non-overlapping PA_NZ byte view. Its last dimension is 2 and its token-fragment dimension is 16. Actual page strides may include K/V payloads, V scales, and padding; the kernel uses the supplied strides rather than assuming contiguous pages.
- Indices must already be valid: `0 <= block_ids < num_blocks`, `0 <= segment_ids < B // 16`, and `0 <= fragment_ids < 16`. The kernel does not clamp or bounds-check their values. The attention caller clamps padded slot `-1` to the reserved null slot before constructing these indices.
- Real tokens must not target the same destination element. Padded rows may target the reserved null slot, whose final value is ignored. Deterministic ordering for overlapping writes is N/A.
- Source and index storage must not overlap the destination writes. Concurrent writers must address disjoint destination bytes. This kernel preserves writes to unselected pages; it does not synchronize competing writes to the same bytes or replace transfer-completion dependencies required by readers.
- Byte patterns are preserved exactly, including any non-finite E8M0 encoding already present in the input. Scale validation, saturation, and NaN correction are N/A.
- Capture and replay keep buffer addresses, shapes, strides, and launch arguments fixed. Values in the captured source/index buffers may change before replay. `num_elements` is a runtime scalar excluded from JIT specialization, so varying token counts between eager calls or separate captures reuses the compiled kernel when other specialization attributes stay fixed. Layout attributes remain compile-time parameters; different layouts can produce different compiled variants.
- Address arithmetic uses int64. The caller must still supply valid allocated storage for every destination.
- Forward inference only; autograd backward support is N/A.

## Origin and Differences

- **Origin**: Developed from `scatter_mxfp_k_scale_cache` in `vllm_ascend/attention/attention_c8_mxfp.py`. The original implementation used advanced-index assignment into a strided cache view; this is not a port of an upstream vLLM Triton operator.
- **Differences**:
    - On NPU, the original IndexPut path can materialize the strided K-scale view and copy it back, overwriting a concurrent KV transfer to an unselected page. The targeted kernel stores only the addressed bytes.
    - Preserve the byte assignment and existing PA_NZ layout while honoring page padding, storage offsets, and all destination strides.
    - Keep the C8 cache representation, attention operator interfaces, and launch configuration unchanged. Every nonempty update uses Triton; the production path contains no CPU fallback.
    - Add no host-side tensor-value reads or device-wide synchronization to the production update path.

## Test Cases

The NPU suite is located under the nightly single-card Triton operator directory:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mxfp_k_scale_scatter.py
```

For a focused run that includes the operator fixtures without loading model-level E2E fixtures:

```bash
pytest --confcutdir=tests/e2e/nightly/single_node/ops -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mxfp_k_scale_scatter.py
```

The suite defines 64 NPU cases covering:

- Model layout dimensions `B=512`, `D=256`, and `H=1/2`, with 1, 64, and 65 token rows. Page padding of 0/256 bytes, storage offsets of 0/4096 bytes, and contiguous/noncontiguous sources yield 48 combinations. Both uint8 and E8M0 source views are exercised, including partial 256-byte programs.
- Eager and ACL graph replay in each combination, changing source values and slot indices in the captured buffers. The complete raw allocation, including unrelated cache components and padding, is checked against a CPU reference.
- An empty source/index batch preserving the complete cache allocation.
- Six migrated numerical cases: preservation of unselected elements in a dense view; valid/padded slots; all-padding batches; real tokens alongside padding; prefill equivalence to the former read-modify-write reference; and padded-decode equivalence outside the null slot. The null-slot cases use `D=64` and `H=2`; reference writes remain test-only on CPU.
- Six source-size mismatch cases for `H=1/2` and zero, too few, or too many source rows. Each requires `ValueError`, no kernel launch, and unchanged cache bytes. These cases can also run under `python -O`.
- Two compilation-reuse cases with `H=1/2`, checking byte-exact results at token counts 1, 2, 31, 32, 33, 63, 64, 65, and 129. Prefix slices keep buffer addresses and strides fixed, and a JIT compilation hook verifies that changing only the token count does not trigger further compilation after the first call.
- An independently scheduled ACL SDMA transfer of a 4096-byte marker into an unselected page during graph replay. Two submission orders, five delays, and four repetitions exercise 40 interleavings. The scatter addresses block 1 and the transfer addresses block 50.

Every byte-copy comparison requires `rtol=0, atol=0`; the concurrent-transfer marker must be byte-identical. The race test allocates approximately 1.03 GiB of cache storage plus transfer buffers. It requires the CANN Python `acl` module; if that module is unavailable, this specific case is skipped, which must not be reported as concurrent-transfer validation.

CPU dispatch tests remain in the unit-test suite with a mocked Triton entry point:

```bash
pytest -q tests/ut/attention/test_mxfp_k_scale_scatter.py
```

Those tests verify a single delegation without inspecting the cache device and an empty-batch return without importing the kernel or accessing cache storage. Numerical cache writes, padding behavior, and concurrent-transfer preservation are tested on NPU.

## Example

This example uses the C8 model's `B=512`, `D=256`, and two local KV heads. The allocation is reduced to four pages for demonstration. Four distinct slots receive scale byte 120.

```python
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_c8_mxfp import (
    mxfp_k_scale_slot_index,
    mxfp_packet_size_bytes,
    mxfp_paged_cache_views,
)
from vllm_ascend.ops.triton.mxfp_k_scale import scatter_k_scales

blocks, heads, dim, block_size = 4, 2, 256, 512
pitch = mxfp_packet_size_bytes(heads, block_size, dim, dim)
raw = torch.zeros(blocks * pitch, dtype=torch.int8, device="npu")
cache = mxfp_paged_cache_views(raw, blocks, heads, dim, dim, block_size)[2]
slots = torch.tensor([512, 527, 528, 1023], dtype=torch.int64, device="npu")
indices = mxfp_k_scale_slot_index(slots, block_size)
values = torch.full((4, heads, dim // 64, 2), 120, dtype=torch.uint8, device="npu")

scatter_k_scales(values, cache, indices)
torch.npu.synchronize()
block_ids, segment_ids, fragment_ids = (index.cpu() for index in indices)
actual = cache.cpu()[block_ids, :, segment_ids, :, fragment_ids, :]
torch.testing.assert_close(actual, values.cpu(), rtol=0, atol=0)
```
