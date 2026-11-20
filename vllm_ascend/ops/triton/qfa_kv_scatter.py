# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Write int8_qfa K/V payloads without materializing strided packet views.

Strided index_put_ on the 5-D PA_NZ packet views first materializes the
whole per-layer pool (~8.5 ms/layer at 17k blocks) and NPU IndexPut can
copy the strided view back, overwriting bytes that do not belong to the
current step. The kernel below writes only the addressed bytes instead,
consuming the packet views' strides exactly like
``vllm_ascend.ops.triton.mxfp_k_scale.scatter_k_scales`` does for the
C8-MXFP scale cache.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_elements"])
def _scatter_kv_payload_kernel(
    source,
    cache,
    block_ids,
    pos_ids,
    num_elements,
    num_heads: tl.constexpr,
    num_frags: tl.constexpr,
    block_stride: tl.constexpr,
    head_stride: tl.constexpr,
    frag_stride: tl.constexpr,
    pos_stride: tl.constexpr,
    block_index_stride: tl.constexpr,
    pos_index_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = index < num_elements
    elems_per_token = num_heads * num_frags * 32
    token = index // elems_per_token
    rem = index % elems_per_token
    head = rem // (num_frags * 32)
    rem2 = rem % (num_frags * 32)
    frag = rem2 // 32
    lane = rem2 % 32
    block = tl.load(block_ids + token * block_index_stride, mask=valid, other=0).to(tl.int64)
    pos = tl.load(pos_ids + token * pos_index_stride, mask=valid, other=0).to(tl.int64)
    offset = (
        block * block_stride
        + head.to(tl.int64) * head_stride
        + frag.to(tl.int64) * frag_stride
        + pos * pos_stride
        + lane.to(tl.int64)
    )
    value = tl.load(source + index, mask=valid, other=0)
    tl.store(cache + offset, value, mask=valid)


def scatter_kv_payload(
    payload: torch.Tensor,
    cache: torch.Tensor,
    slot_index: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scatter one quantized payload [T, N, D] into a 5-D PA_NZ packet view.

    ``cache`` is ``(blocks, N, D // 32, block_size, 32)`` with dim0 stride
    equal to the packet pitch (the per-page ``[K | K_descale | V]`` sections
    interleave, so dim0 stride never equals the shape's inner product);
    ``slot_index`` is the step-shared ``(block_ids, pos_ids)`` pair derived
    from the slot mapping. Padded rows arrive clamped to slot 0 and land in
    the null block, which the block pool never hands to a request.

    Capture safe: no host-device sync, no data-dependent shapes.
    """
    block_ids, pos_ids = slot_index
    num_tokens = block_ids.numel()
    if num_tokens == 0:
        return
    num_heads, num_frags = cache.shape[1], cache.shape[2]
    source = payload.contiguous().view(torch.int8).view(-1)
    num_elements = source.numel()
    expected_elements = num_tokens * num_heads * num_frags * 32
    if num_elements != expected_elements:
        raise ValueError(
            f"KV payload contains {num_elements} elements; expected {expected_elements}."
        )
    block_size = 256
    _scatter_kv_payload_kernel[(triton.cdiv(num_elements, block_size),)](
        source,
        cache,
        block_ids,
        pos_ids,
        num_elements,
        num_heads,
        num_frags,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        block_ids.stride(0),
        pos_ids.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
