# scatter_kv_payload

## Description

- **Location**: `vllm_ascend/ops/triton/qfa_kv_scatter.py`; Python entry point `scatter_kv_payload`, private kernel `_scatter_kv_payload_kernel`.
- **Function**: Scatter one quantized int8 K or V payload into the packet-packed PA_NZ cache used by the A3 QFA INT8 (`int8_qfa`) KV path. Only the addressed token/head bytes are written; other cache pages, the K descale section, page padding, and unrelated layers' storage remain untouched. The operator copies bytes; it does not quantize, descale, or validate payload values.
- **Formula**: For token `t`, head `h`, dim fragment `f`, and lane element `l`:

  ```text
  b = block_ids[t]
  p = pos_ids[t]
  cache[b, h, f, p, l] = payload[t, h, f * 32 + l]

  destination_offset = b * cache.stride(0)
                     + h * cache.stride(1)
                     + f * cache.stride(2)
                     + p * cache.stride(3)
                     + l * cache.stride(4)
  ```

  All destination offsets are relative to `cache.data_ptr()`, which already includes the view's storage offset. For a slot `q` and kernel block size `B`, the caller derives `b = q // B` and `p = q % B`.

- **Algorithm flow**:
    1. Return without launching a kernel when the block-index tensor is empty.
    2. Reinterpret the payload as raw int8 bytes and make it contiguous. Read head/fragment counts and the first four destination strides from cache metadata. Validate the source element count before launching the kernel.
    3. Launch one Triton program per 256 flattened source bytes, masking elements beyond `T * H * D`.
    4. Decompose each flattened index into token, head, fragment, and lane coordinates. Load block/pos indices using their actual strides.
    5. Promote address coordinates to int64, calculate the destination offset, and store the unchanged source byte directly. No cache-wide temporary view or writeback is created.
- **Supported modes**: Ascend NPU eager execution and ACL graph capture/replay with Triton-Ascend. Hardware validation status is tracked in the A3 QFA workstream (probe_scatter_pa_v8 byte-exact parity plus service-level regression); this document does not establish additional hardware support beyond that evidence.

## Parameters

All three arguments are required. The Python function returns `None` and updates `cache` in place. `T` is the number of token rows, `H` the local KV-head count, `D` the Q/K head dimension, and `B` the kernel block size.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `payload` | Input | Quantized payload of shape `[T, H, D]` | int8 | ND; noncontiguous sources are made contiguous |
| `cache` | Input/Output | K or V view of shape `[num_blocks, H, D // 32, B, 32]`, backed by packet-packed KV storage | int8 | PA_NZ, with explicit five-dimensional strides and an optional storage offset |
| `slot_index` | Input | Tuple `(block_ids, pos_ids)`; each tensor has shape `[T]` | int64 in the caller | 1D; each index tensor's stride is used |
| Return value | Output | N/A; the destination view is modified in place | `None` | N/A |
| `num_elements` | Internal attribute | `T * H * D`; excluded from JIT specialization with `do_not_specialize` | Integer, runtime | Scalar |
| `num_heads`, `num_frags` | Internal attributes | `H` and `D // 32` | Integer, constexpr | Scalars |
| Cache and index strides | Internal attributes | Element strides supplied by the input tensor metadata | Integer, constexpr | Scalars |
| `BLOCK_SIZE`, `num_warps` | Internal attributes | 256 bytes per program and 4 warps | Integer, compile-time launch settings | Scalars |

## Constraints

- Source, destination, and all index tensors must reside on the same NPU. CPU inference is not supported by this update path.
- Source shape and element size must satisfy `payload.numel() == T * H * D`. For a nonempty index batch, a mismatched source element count raises `ValueError` before kernel launch.
- The cache must be a valid, non-overlapping PA_NZ byte view whose last dimension is 32. Actual page strides include the interleaved K-descale section and page padding; the kernel uses the supplied strides rather than assuming contiguous pages. The fused `npu_scatter_pa_kv_cache` operator is not usable on this layout: it addresses blocks by the shape's inner product and ignores the dim0 stride (probe_scatter_pa_v8, 2026-11-20).
- Indices must already be valid: `0 <= block_ids < num_blocks` and `0 <= pos_ids < B`. The kernel does not clamp or bounds-check their values. The attention caller clamps padded slot `-1` to the reserved null slot before constructing these indices.
- Real tokens must not target the same destination element. Padded rows may target the reserved null slot, whose final value is ignored.
- Source and index storage must not overlap the destination writes. Concurrent writers must address disjoint destination bytes.
- Byte patterns are preserved exactly; payload validation and saturation are N/A.
- Capture and replay keep buffer addresses, shapes, strides, and launch arguments fixed. Values in the captured source/index buffers may change before replay. `num_elements` is a runtime scalar excluded from JIT specialization, so varying token counts between eager calls or separate captures reuses the compiled kernel when other specialization attributes stay fixed.
- Address arithmetic uses int64.
- Forward inference only; autograd backward support is N/A.

## Origin and Differences

- **Origin**: Developed for the A3 QFA INT8 KV write path in `vllm_ascend/attention/attention_a3_qfa.py`. Modeled on `scatter_k_scales` in `vllm_ascend/ops/triton/mxfp_k_scale.py`; this is not a port of an upstream vLLM Triton operator.
- **Differences**:
    - The original write path used advanced-index assignment into the strided 5-D packet views. On NPU, IndexPut materializes the whole per-layer pool per call (~8.5 ms/layer at 17k blocks) and can copy the strided view back, overwriting unrelated pages. The targeted kernel stores only the addressed bytes.
    - Preserve the byte assignment and the existing per-page packet layout `[K | K_descale | V]` while honoring page strides, storage offsets, and all destination strides.
    - Keep the `int8_qfa` cache representation, QFA attention operator interfaces, and launch configuration unchanged.
    - Add no host-side tensor-value reads or device-wide synchronization to the production update path.

## Test Cases

Byte-exact parity against the strided index_put_ reference, including sentinel write-range checks over the whole group-packed allocation, runs from the A3 workstream probe:

```bash
VLLM_ASCEND_REPO=<vllm-ascend checkout> \
ASCEND_RT_VISIBLE_DEVICES=<cards> python3 probe_scatter_pa_v8.py
```

The probe covers: builder geometry cross-check; same-page, page-crossing, prefill (4-page), shuffled-slot, and H=4 cases; overlapping slots (last write wins); padded slots clamped to the null slot; and a service-scale (~100k block) timing comparison against the index_put_ reference. Service-level regression (L1-L6 alternating burst plus the standard burst suite) and decode throughput A/B (`VLLM_ASCEND_QFA_SCATTER_WRITE=1` vs `0`) complete the validation.

## Example

This example uses the service geometry `D=256`, `H=2`, kernel block size 128, with a reduced four-page allocation.

```python
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_a3_qfa import qfa_int8_paged_cache_views
from vllm_ascend.ops.triton.qfa_kv_scatter import scatter_kv_payload

blocks, heads, dim, block_size = 4, 2, 256, 128
raw = torch.zeros(blocks * (2 * heads * block_size * dim + 4 * heads * block_size),
                  dtype=torch.int8, device="npu")
k_nz, _, _ = qfa_int8_paged_cache_views(raw, blocks, heads, dim, block_size)
slots = torch.tensor([128, 129, 257, 383], dtype=torch.int64, device="npu")
pages, pos = slots // block_size, slots % block_size
payload = torch.randint(-127, 128, (4, heads, dim), dtype=torch.int8, device="npu")

scatter_kv_payload(payload, k_nz, (pages, pos))
torch.npu.synchronize()
actual = k_nz.cpu()[pages.cpu(), :, :, pos.cpu(), :]
torch.testing.assert_close(actual.reshape(4, heads, dim), payload.cpu(), rtol=0, atol=0)
```
