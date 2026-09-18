# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_mla_kv_kernel(
    latent,
    rope,
    latent_cache,
    rope_cache,
    slots,
    scale_reciprocal,
    source_slots,
    INPUT_LATENT_STRIDE: tl.constexpr,
    INPUT_ROPE_STRIDE: tl.constexpr,
    INPUT_LATENT_ROW_STRIDE: tl.constexpr,
    INPUT_ROPE_ROW_STRIDE: tl.constexpr,
    SOURCE_PAGE_SIZE: tl.constexpr,
    PAGED_SOURCE: tl.constexpr,
    LATENT_PAGE_STRIDE: tl.constexpr,
    LATENT_ROW_STRIDE: tl.constexpr,
    ROPE_PAGE_STRIDE: tl.constexpr,
    ROPE_ROW_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    valid = slot >= 0
    safe_slot = tl.maximum(slot, 0)
    page, row = safe_slot // PAGE_SIZE, safe_slot % PAGE_SIZE
    if PAGED_SOURCE:
        source_slot = tl.load(source_slots + token)
        valid = valid & (source_slot >= 0)
        safe_source = tl.maximum(source_slot, 0)
        source_page, source_row = safe_source // SOURCE_PAGE_SIZE, safe_source % SOURCE_PAGE_SIZE
        latent_offset = source_page * INPUT_LATENT_STRIDE + source_row * INPUT_LATENT_ROW_STRIDE
        rope_offset = source_page * INPUT_ROPE_STRIDE + source_row * INPUT_ROPE_ROW_STRIDE
    else:
        latent_offset = token * INPUT_LATENT_STRIDE
        rope_offset = token * INPUT_ROPE_STRIDE
    scale = tl.load(scale_reciprocal)
    c = tl.arange(0, 512)
    kv = tl.load(latent + latent_offset + c, valid, other=0).to(tl.float32)
    kv = tl.minimum(tl.maximum(kv * scale, -448.0), 448.0)
    tl.store(latent_cache + page * LATENT_PAGE_STRIDE + row * LATENT_ROW_STRIDE + c, kv, valid)
    r = tl.arange(0, 64)
    pe = tl.load(rope + rope_offset + r, valid, other=0)
    tl.store(rope_cache + page * ROPE_PAGE_STRIDE + row * ROPE_ROW_STRIDE + r, pe, valid)


def quantize_mla_kv(
    latent: torch.Tensor,
    rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    slots: torch.Tensor,
    scale_reciprocal: torch.Tensor,
    *,
    source_slots: torch.Tensor | None = None,
) -> None:
    """Write normalized MLA rows into static FP8/BF16 caches without staging KV.

    The one-head caches have independent page strides; BLHNC padding and
    neighboring layers are never read or copied. Negative slots are padding or
    tokens owned by another DCP rank. Only the 512-dimensional latent is scaled;
    the 64-dimensional component stays in the model dtype.

    With ``source_slots``, latent/rope are paged BF16 views with page and row
    axes first. Read those slots directly, allowing MLAPO's BF16 current-cache
    output to populate persistent C8 history without gathering a token buffer.
    Negative source slots are skipped as well as negative destination slots.
    """
    if slots.numel() == 0:
        return
    _quantize_mla_kv_kernel[(slots.numel(),)](
        latent,
        rope,
        latent_cache,
        rope_cache,
        slots,
        scale_reciprocal,
        source_slots,
        INPUT_LATENT_STRIDE=latent.stride(0),
        INPUT_ROPE_STRIDE=rope.stride(0),
        INPUT_LATENT_ROW_STRIDE=latent.stride(1) if source_slots is not None else 0,
        INPUT_ROPE_ROW_STRIDE=rope.stride(1) if source_slots is not None else 0,
        SOURCE_PAGE_SIZE=latent.shape[1] if source_slots is not None else 1,
        PAGED_SOURCE=source_slots is not None,
        LATENT_PAGE_STRIDE=latent_cache.stride(0),
        LATENT_ROW_STRIDE=latent_cache.stride(1),
        ROPE_PAGE_STRIDE=rope_cache.stride(0),
        ROPE_ROW_STRIDE=rope_cache.stride(1),
        PAGE_SIZE=latent_cache.shape[1],
        multibuffer=False,
    )
