# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernel for DSA context-parallel local token metadata.

See docs/dsa_local_metadata.md for the full description, semantics,
constraints and Ascend-specific lowering notes.
"""

import torch
from vllm.triton_utils import tl, triton

# Wrapper fallback capacity; production derives block from the actual
# buffer capacity (see docs/dsa_local_metadata.md).
DSA_LOCAL_METADATA_BLOCK = 512


@triton.jit(do_not_specialize=["local_start", "local_end", "num_reqs"])
def build_local_metadata_kernel(
    query_start_loc_ptr,  # [num_reqs + 1] int32, 1D contiguous
    seq_lens_ptr,  # [num_reqs] int32, 1D contiguous
    local_query_start_loc_ptr,  # [BLOCK + 1] int32 (output, fully overwritten)
    local_seq_lens_ptr,  # [BLOCK] int32 (output, fully overwritten)
    start_pos_out_ptr,  # [BLOCK] int32 (output, fully overwritten)
    local_start,  # runtime scalar: this rank's local token slice start
    local_end,  # runtime scalar: this rank's local token slice end
    num_reqs,  # runtime scalar: current batch size, 0..BLOCK
    COMPUTE_START_POS: tl.constexpr,
    BLOCK: tl.constexpr,  # fixed capacity; never derived from num_reqs
    SUB_N: tl.constexpr,  # cumsum fold rows (BLOCK = SUB_N * COLS)
    COLS: tl.constexpr,  # cumsum fold cols
):
    """Fused local-token-metadata kernel; one BLOCK-lane vector pass.

    Semantics: see docs/dsa_local_metadata.md. Lanes with i >= num_reqs
    store 0, so the full output buffers are deterministic regardless of
    prior contents.
    """
    offs = tl.arange(0, BLOCK)
    # int32 vector < runtime int scalar lowers to a scalar compare loop on
    # Ascend; casting only the VECTOR side to fp32 rides the vector compare
    # units (num_reqs < 2^24 promotes exactly). The scalar side must stay a
    # bare do_not_specialize int — calling .to() on that runtime scalar
    # produced an illegal instruction (NPU error 507057) on 910B4.
    valid = offs.to(tl.float32) < num_reqs

    # Vector loads: int32 -> fp32 for the clamp/compare chain (values
    # < 2^24, exact roundtrip). The int32 register copies are kept for
    # the integer-domain sp path below.
    q_start_i = tl.load(query_start_loc_ptr + offs, mask=valid, other=0)
    q_end_i = tl.load(query_start_loc_ptr + offs + 1, mask=valid, other=0)
    seq_len_i = tl.load(seq_lens_ptr + offs, mask=valid, other=0)
    q_start = q_start_i.to(tl.float32)
    q_end = q_end_i.to(tl.float32)
    seq_len = seq_len_i.to(tl.float32)
    ls_f = local_start.to(tl.float32)
    le_f = local_end.to(tl.float32)

    lqs = tl.minimum(tl.maximum(q_start, ls_f), le_f)
    lqe = tl.minimum(tl.maximum(q_end, ls_f), le_f)
    lql = lqe - lqs

    # Output 1: [0] + inclusive cumsum of local query lens, folded into a
    # column-major (SUB_N, COLS) view so cumsum rides the vector units
    # (see docs); each column is compensated with the prefix of the
    # preceding columns' totals.
    x_col = tl.trans(tl.reshape(lql, (COLS, SUB_N)))  # (SUB_N, COLS) column-major
    cum_col = tl.cumsum(x_col, axis=0)  # vector path (not last dim)
    col_sums = tl.sum(x_col, axis=0)  # (COLS,) totals per column
    col_prefix = tl.cumsum(col_sums, axis=0) - col_sums  # [0, s0, s0+s1, ...]
    y = cum_col + col_prefix[None, :]  # global inclusive scan
    cum = tl.reshape(tl.trans(y), (BLOCK,))  # back to row-major order
    # Position [0] is the scalar 0 prefix; store it through a 1-lane
    # arange (triton 3.2.0 requires a block pointer for a block value).
    tl.store(local_query_start_loc_ptr + tl.arange(0, 1), tl.zeros((1,), dtype=tl.int32))
    # Masked lanes load q_start=q_end=0, so lql=0 there, but cumsum at
    # lane i >= num_reqs still carries the running total from valid
    # lanes — the where(valid, ...) tail-zeroing here is REQUIRED.
    tl.store(
        local_query_start_loc_ptr + 1 + offs,
        tl.where(valid, cum, 0.0).to(tl.int32),
    )

    # Output 2: masked local seq lens.
    offset = q_end - lqe
    lsl = tl.where(
        (lql > 0.0) & (seq_len > 0.0),
        tl.maximum(seq_len - offset, 0.0),
        0.0,
    )
    tl.store(local_seq_lens_ptr + offs, lsl.to(tl.int32))

    # Output 3: start_pos (constexpr branch, not a runtime scalar-if).
    # Computed in the int32 domain from the raw load copies: sp is a pure
    # integer expression, so the fp32 round-trip would only add two
    # 512-wide vcast(trunc) round-trips for zero benefit. Masked lanes
    # load q_start=q_end=seq_len=0, so sp = 0 - (0 - 0) = 0 naturally.
    if COMPUTE_START_POS:
        sp = seq_len_i - (q_end_i - q_start_i)
        tl.store(start_pos_out_ptr + offs, sp)


def build_local_metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    local_query_start_loc: torch.Tensor,
    local_seq_lens: torch.Tensor,
    local_start: int,
    local_end: int,
    num_reqs: int,
    start_pos_out=None,
    block: int = DSA_LOCAL_METADATA_BLOCK,
):
    """Launch the fused local-token-metadata kernel.

    Output buffers must be sized for the full capacity (block+1 / block)
    and are fully overwritten — callers do not need to pre-zero. With
    num_reqs == 0 the launch is skipped and the buffers are zero-filled.
    See docs/dsa_local_metadata.md for the full contract.
    """
    # SUB_N x COLS fold requires the capacity to be a multiple of SUB_N.
    assert block % 8 == 0, f"block size {block} must be a multiple of 8"
    if num_reqs == 0:
        local_query_start_loc.zero_()
        local_seq_lens.zero_()
        if start_pos_out is not None:
            start_pos_out.zero_()
        return

    # Zero-size inputs carry a NULL data_ptr on NPU (torch.empty(0)); the
    # masked-load lowering still dereferences the pointer, so substitute a
    # live 1-element dummy. Every lane is masked out at num_reqs == 0, but
    # this branch only guards num_reqs > 0 with degenerate input tensors.
    if seq_lens.numel() == 0:
        seq_lens = seq_lens.new_zeros(1)
    if query_start_loc.numel() == 0:
        query_start_loc = query_start_loc.new_zeros(1)

    build_local_metadata_kernel[(1,)](
        query_start_loc,
        seq_lens,
        local_query_start_loc,
        local_seq_lens,
        start_pos_out if start_pos_out is not None else local_query_start_loc.new_zeros(1),
        local_start,
        local_end,
        num_reqs,
        COMPUTE_START_POS=start_pos_out is not None,
        BLOCK=block,
        SUB_N=8,  # block = 8 * (block / 8); column-major fold for vector cumsum
        COLS=block // 8,
    )
