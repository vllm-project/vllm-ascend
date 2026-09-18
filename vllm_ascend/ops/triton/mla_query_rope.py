# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.quantize_mla_kv import _quantize_mla_kv_kernel, quantize_mla_kv
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _scale_mla_query_rope_kernel(
    rope,
    query_scale,
    kv_scale,
    output,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    ROPE_T: tl.constexpr,
    ROPE_H: tl.constexpr,
    ROPE_D: tl.constexpr,
    SCALE_T: tl.constexpr,
    SCALE_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ROWS * 64 + tl.arange(0, BLOCK_ROWS * 64)
    rows, cols = offsets // 64, offsets % 64
    tokens, heads = rows // HEADS, rows % HEADS
    valid = rows < ROWS
    scale_ptrs = query_scale + tokens * SCALE_T + heads * SCALE_H
    sq = tl.load(scale_ptrs, valid, other=1.0)
    # Dynamic quantization returns scale=0 for an all-zero latent row. Its
    # FP8 values remain zero when the descale becomes one; the independent
    # BF16 component still contributes normally to QK.
    sq = tl.where(sq == 0.0, 1.0, sq)
    tl.store(scale_ptrs, sq, valid & (cols == 0))
    sk = tl.load(kv_scale)
    values = tl.load(
        rope + tokens * ROPE_T + heads * ROPE_H + cols * ROPE_D,
        valid,
        other=0.0,
    ).to(tl.float32)
    scaled = (values / sq) / sk
    tl.store(output + offsets, scaled, valid)


def scale_mla_query_rope(
    rope: torch.Tensor,
    query_scale: torch.Tensor,
    kv_scale: torch.Tensor,
) -> torch.Tensor:
    """Prepare C8 MLA's BF16 component and normalize zero Q descales in place.

    ``rope`` is [T,H,64] and may be a strided projection view. FP32 query
    descales are [T,H] or [T,H,1]; the static KV descale is a scalar tensor.
    No latent Q values are changed. Each row belongs to exactly one program,
    so the descale update and corresponding output have no cross-core race.
    """
    tokens, heads, dim = rope.shape
    assert dim == 64 and rope.dtype == torch.bfloat16
    assert query_scale.shape[:2] == (tokens, heads)
    assert query_scale.numel() == tokens * heads and query_scale.dtype == torch.float32
    assert kv_scale.numel() == 1 and kv_scale.dtype == torch.float32
    output = torch.empty((tokens, heads, 64), dtype=rope.dtype, device=rope.device)
    rows = tokens * heads
    if rows == 0:
        return output
    # At most 2,048 FP32 elements per program; leave ample UB for the
    # compiler's broadcast/division temporaries, including the strided path.
    block_rows = 8 if rows <= 512 else 32
    _scale_mla_query_rope_kernel[(triton.cdiv(rows, block_rows),)](
        rope,
        query_scale,
        kv_scale,
        output,
        ROWS=rows,
        HEADS=heads,
        ROPE_T=rope.stride(0),
        ROPE_H=rope.stride(1),
        ROPE_D=rope.stride(2),
        SCALE_T=query_scale.stride(0),
        SCALE_H=query_scale.stride(1),
        BLOCK_ROWS=block_rows,
        multibuffer=False,
    )
    return output


@triton.jit
def _quantize_mla_query_kernel(
    query,
    rope,
    kv_scale,
    quantized,
    scaled_rope,
    query_scale,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY_T: tl.constexpr,
    QUERY_H: tl.constexpr,
    QUERY_D: tl.constexpr,
    ROPE_T: tl.constexpr,
    ROPE_H: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCKS_PER_CORE: tl.constexpr,
):
    columns = tl.arange(0, 512)
    rope_columns = tl.arange(0, 64)
    sk = tl.load(kv_scale)
    for block in range(BLOCKS_PER_CORE):
        rows = (tl.program_id(0) * BLOCKS_PER_CORE + block) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        tokens, heads = rows // HEADS, rows % HEADS
        valid = rows < ROWS
        values = tl.load(
            query + tokens[:, None] * QUERY_T + heads[:, None] * QUERY_H + columns[None, :] * QUERY_D,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        # Match CANN dynamic quantization's rounded FP32 reciprocal multiply.
        sq = tl.max(tl.abs(values), 1) * (1.0 / 448.0)
        sq = tl.where(sq == 0.0, 1.0, sq)
        normalized = tl.div_rn(values, sq[:, None])
        normalized = tl.minimum(tl.maximum(normalized, -448.0), 448.0)
        tl.store(quantized + rows[:, None] * 512 + columns[None, :], normalized, valid[:, None])
        tl.store(query_scale + rows, sq, valid)
        pe = tl.load(
            rope + tokens[:, None] * ROPE_T + heads[:, None] * ROPE_H + rope_columns[None, :] * ROPE_D,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        scaled = (pe / sq[:, None]) / sk
        tl.store(scaled_rope + rows[:, None] * 64 + rope_columns[None, :], scaled, valid[:, None])


def quantize_mla_query(
    query: torch.Tensor,
    rope: torch.Tensor,
    kv_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse per-head FP8 query quantization and the independent BF16 term.

    Inputs [T,H,512] and [T,H,64] can have independent token/head strides,
    with contiguous final dimensions. Zero latent heads retain their BF16
    contribution by using descale one.
    """
    tokens, heads, dim = query.shape
    assert dim == 512 and query.dtype == torch.bfloat16
    assert rope.shape == (tokens, heads, 64) and rope.dtype == torch.bfloat16
    assert query.stride(-1) == rope.stride(-1) == 1
    assert kv_scale.numel() == 1 and kv_scale.dtype == torch.float32
    quantized = torch.empty((tokens, heads, 512), dtype=torch.float8_e4m3fn, device=query.device)
    scaled_rope = torch.empty((tokens, heads, 64), dtype=rope.dtype, device=rope.device)
    query_scale = torch.empty((tokens, heads), dtype=torch.float32, device=query.device)
    rows = tokens * heads
    if rows:
        init_device_properties_triton()
        # Sixteen FP32 scale elements per block avoid cross-core partial
        # cache-line writes. A 16x512 tile fits the A5 vector UB with all
        # reduction and division temporaries.
        block_rows = 16
        blocks = triton.cdiv(rows, block_rows)
        grid = min(blocks, get_vectorcore_num())
        _quantize_mla_query_kernel[(grid,)](
            query,
            rope,
            kv_scale,
            quantized,
            scaled_rope,
            query_scale,
            rows,
            heads,
            *query.stride(),
            *rope.stride(),
            block_rows,
            triton.cdiv(blocks, grid),
            multibuffer=False,
        )
    return quantized, scaled_rope, query_scale


@triton.jit
def _quantize_mla_query_and_kv_kernel(
    query,
    rope,
    kv_scale,
    quantized,
    scaled_rope,
    query_scale,
    local_query,
    latent,
    kv_rope,
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
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY_T: tl.constexpr,
    QUERY_H: tl.constexpr,
    QUERY_D: tl.constexpr,
    ROPE_T: tl.constexpr,
    ROPE_H: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCKS_PER_CORE: tl.constexpr,
    WRITE_LOCAL: tl.constexpr,
    LOCAL_HEAD_START: tl.constexpr,
):
    _quantize_mla_query_kernel(
        query,
        rope,
        kv_scale,
        quantized,
        scaled_rope,
        query_scale,
        ROWS,
        HEADS,
        QUERY_T,
        QUERY_H,
        QUERY_D,
        ROPE_T,
        ROPE_H,
        ROPE_D,
        BLOCK_ROWS,
        BLOCKS_PER_CORE,
    )
    _quantize_mla_kv_kernel(
        latent,
        kv_rope,
        latent_cache,
        rope_cache,
        slots,
        scale_reciprocal,
        source_slots,
        INPUT_LATENT_STRIDE,
        INPUT_ROPE_STRIDE,
        INPUT_LATENT_ROW_STRIDE,
        INPUT_ROPE_ROW_STRIDE,
        SOURCE_PAGE_SIZE,
        PAGED_SOURCE,
        LATENT_PAGE_STRIDE,
        LATENT_ROW_STRIDE,
        ROPE_PAGE_STRIDE,
        ROPE_ROW_STRIDE,
        PAGE_SIZE,
    )
    if WRITE_LOCAL:
        # Fixed affine rows retain SIMD lowering; a head mask inside the
        # quantization tile turns these copies into costly SIMT scatter.
        token = tl.program_id(0)
        columns = tl.arange(0, 512)
        rope_columns = tl.arange(0, 64)
        for head in tl.static_range(12):
            value = tl.load(query + token * QUERY_T + (LOCAL_HEAD_START + head) * QUERY_H + columns * QUERY_D)
            pe = tl.load(rope + token * ROPE_T + (LOCAL_HEAD_START + head) * ROPE_H + rope_columns * ROPE_D)
            offset = (token * 12 + head) * 576
            tl.store(local_query + offset + columns, value)
            tl.store(local_query + offset + 512 + rope_columns, pe)


def quantize_mla_query_and_kv(
    query: torch.Tensor,
    rope: torch.Tensor,
    kv_scale: torch.Tensor,
    latent: torch.Tensor,
    kv_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    slots: torch.Tensor,
    scale_reciprocal: torch.Tensor,
    *,
    source_slots: torch.Tensor | None = None,
    local_query: torch.Tensor | None = None,
    local_head_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One program per token; each owns 96 Q heads plus one optional KV row.

    The negative destination/source slot rules and independent page/row strides
    match quantize_mla_kv. Optional local_query receives the original BF16
    current query; the three quantized outputs and their rounding are unchanged.
    """
    init_device_properties_triton()
    if local_query is not None:
        assert query.shape == (64, 96, 512)
        assert local_query.shape == (64, 12, 576) and local_query.is_contiguous()
        assert local_query.dtype == query.dtype == torch.bfloat16
        assert local_query.device == query.device
        assert 0 <= local_head_start <= 84
    # Six 16-head tiles map one program to one token only for this shape.
    if query.shape != (64, 96, 512) or get_vectorcore_num() != 64:
        quantize_mla_kv(
            latent,
            kv_rope,
            latent_cache,
            rope_cache,
            slots,
            scale_reciprocal,
            source_slots=source_slots,
        )
        if local_query is not None:
            local_end = local_head_start + 12
            torch.cat((query[:, local_head_start:local_end], rope[:, local_head_start:local_end]), -1, out=local_query)
        return quantize_mla_query(query, rope, kv_scale)
    assert query.dtype == torch.bfloat16
    assert rope.shape == (64, 96, 64) and rope.dtype == torch.bfloat16
    assert query.stride(-1) == rope.stride(-1) == 1
    assert slots.numel() == 64
    assert source_slots is None or source_slots.numel() == 64
    assert kv_scale.numel() == scale_reciprocal.numel() == 1
    assert kv_scale.dtype == scale_reciprocal.dtype == torch.float32
    quantized = torch.empty((64, 96, 512), dtype=torch.float8_e4m3fn, device=query.device)
    scaled_rope = torch.empty((64, 96, 64), dtype=rope.dtype, device=rope.device)
    query_scale = torch.empty((64, 96), dtype=torch.float32, device=query.device)
    _quantize_mla_query_and_kv_kernel[(64,)](
        query,
        rope,
        kv_scale,
        quantized,
        scaled_rope,
        query_scale,
        local_query,
        latent,
        kv_rope,
        latent_cache,
        rope_cache,
        slots,
        scale_reciprocal,
        source_slots,
        INPUT_LATENT_STRIDE=latent.stride(0),
        INPUT_ROPE_STRIDE=kv_rope.stride(0),
        INPUT_LATENT_ROW_STRIDE=latent.stride(1) if source_slots is not None else 0,
        INPUT_ROPE_ROW_STRIDE=kv_rope.stride(1) if source_slots is not None else 0,
        SOURCE_PAGE_SIZE=latent.shape[1] if source_slots is not None else 1,
        PAGED_SOURCE=source_slots is not None,
        LATENT_PAGE_STRIDE=latent_cache.stride(0),
        LATENT_ROW_STRIDE=latent_cache.stride(1),
        ROPE_PAGE_STRIDE=rope_cache.stride(0),
        ROPE_ROW_STRIDE=rope_cache.stride(1),
        PAGE_SIZE=latent_cache.shape[1],
        ROWS=64 * 96,
        HEADS=96,
        QUERY_T=query.stride(0),
        QUERY_H=query.stride(1),
        QUERY_D=query.stride(2),
        ROPE_T=rope.stride(0),
        ROPE_H=rope.stride(1),
        ROPE_D=rope.stride(2),
        BLOCK_ROWS=16,
        BLOCKS_PER_CORE=6,
        WRITE_LOCAL=local_query is not None,
        LOCAL_HEAD_START=local_head_start,
        multibuffer=False,
    )
    return quantized, scaled_rope, query_scale
