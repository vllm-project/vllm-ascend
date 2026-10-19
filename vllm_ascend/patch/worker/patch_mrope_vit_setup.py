# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound the ViT mRoPE setup kernel's live vectors for Ascend UB capacity."""

from vllm.triton_utils import tl, triton

try:
    import vllm.model_executor.layers.rotary_embedding.mrope_vit_setup as mrope_vit_setup
except ModuleNotFoundError as exc:
    if exc.name != "vllm.model_executor.layers.rotary_embedding.mrope_vit_setup":
        raise
    mrope_vit_setup = None


@triton.jit(
    do_not_specialize=["num_tokens", "num_grids"],
    do_not_specialize_on_alignment=["grids_ptr"],
)
def _vit_mrope_setup_kernel(
    inv_freq_t_ptr,
    inv_freq_h_ptr,
    inv_freq_w_ptr,
    cos_out_ptr,
    sin_out_ptr,
    cu_seqlens_ptr,
    grids_ptr,
    num_tokens,
    num_grids,
    half_t: tl.constexpr,
    half_h: tl.constexpr,
    half_w: tl.constexpr,
    spatial_merge: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Keep upstream's launch grid and signature: imported vit_mrope_setup
    # aliases also resolve this kernel through the original module globals.
    # A flat tile caps all intermediates, including the compiler's fp32 trig
    # expansion, instead of materializing BLOCK_N x BLOCK_D live tensors.
    TILE_ELEMENTS: tl.constexpr = 256
    half_rot: tl.constexpr = half_t + half_h + half_w
    plane = num_tokens.to(tl.int64) * half_rot
    for tile in range(tl.cdiv(BLOCK_N * BLOCK_D, TILE_ELEMENTS)):
        element = tile * TILE_ELEMENTS + tl.arange(0, TILE_ELEMENTS)
        tok = tl.program_id(0) * BLOCK_N + element // BLOCK_D
        d = element % BLOCK_D
        tok_mask = (element < BLOCK_N * BLOCK_D) & (tok < num_tokens)

        seg = tl.zeros((TILE_ELEMENTS,), tl.int32)
        for g in range(num_grids):
            end = tl.load(cu_seqlens_ptr + g + 1)
            seg += (tok >= end).to(tl.int32)
        seg_start = tl.load(cu_seqlens_ptr + seg, mask=tok_mask, other=0)
        grid_h = tl.load(grids_ptr + seg * 3 + 1, mask=tok_mask, other=1)
        grid_w = tl.load(grids_ptr + seg * 3 + 2, mask=tok_mask, other=1)
        local = tok - seg_start
        rem = local % (grid_h * grid_w)
        tpos = local // (grid_h * grid_w)
        block_h = rem // (spatial_merge * grid_w)
        within_row = rem % (spatial_merge * grid_w)
        block_w = within_row // (spatial_merge * spatial_merge)
        within_block = within_row % (spatial_merge * spatial_merge)
        hpos = block_h * spatial_merge + within_block // spatial_merge
        wpos = block_w * spatial_merge + within_block % spatial_merge
        row = tok.to(tl.int64) * half_rot + d

        in_t = d < half_t
        inv_t = tl.load(inv_freq_t_ptr + d, mask=in_t, other=0.0)
        freq = tpos.to(tl.float32) * inv_t
        mask = tok_mask & in_t
        tl.store(cos_out_ptr + row, tl.cos(freq), mask=mask)
        tl.store(sin_out_ptr + row, tl.sin(freq), mask=mask)

        in_h = (d >= half_t) & (d < half_t + half_h)
        inv_h = tl.load(inv_freq_h_ptr + d - half_t, mask=in_h, other=0.0)
        freq = hpos.to(tl.float32) * inv_h
        mask = tok_mask & in_h
        tl.store(cos_out_ptr + plane + row, tl.cos(freq), mask=mask)
        tl.store(sin_out_ptr + plane + row, tl.sin(freq), mask=mask)

        in_w = (d >= half_t + half_h) & (d < half_rot)
        inv_w = tl.load(inv_freq_w_ptr + d - half_t - half_h, mask=in_w, other=0.0)
        freq = wpos.to(tl.float32) * inv_w
        mask = tok_mask & in_w
        tl.store(cos_out_ptr + 2 * plane + row, tl.cos(freq), mask=mask)
        tl.store(sin_out_ptr + 2 * plane + row, tl.sin(freq), mask=mask)


if mrope_vit_setup is not None:
    mrope_vit_setup._vit_mrope_setup_kernel = _vit_mrope_setup_kernel
