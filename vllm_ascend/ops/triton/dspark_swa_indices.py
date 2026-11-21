# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-Ascend fused implementation of build_dspark_swa_indices.

See docs/dspark_swa_indices.md for the full description, semantics,
constraints and Ascend-specific lowering notes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(
    do_not_specialize=[
        "num_blocks",
        "num_reqs",
        "num_rows_padded",
        "num_rows",
        "num_query_per_req",
        "num_slots",
    ]
)
def _dspark_swa_indices_kernel(
    bt_ptr,  # block_table       [R_alloc, num_blocks]  i32/i64 (narrowed in-kernel)
    stride_bt_r,  # block_table.stride(0): row pitch of a possibly sliced/padded table
    qsl_ptr,  # query_start_loc  [R_alloc + 1]  i32/i64
    seq_lens_ptr,  # seq_lens         [R_alloc]  i32/i64
    slots_ptr,  # out_slots        [num_rows_padded, 1, INDEX_W]  i32
    lens_ptr,  # out_lens         [num_rows_padded]  i64
    num_blocks,  # block_table.shape[1]
    num_reqs,  # active request count == block_table.shape[0]
    num_rows_padded,  # padded row count == R_alloc * num_query_per_req
    num_rows,  # active row count == sum(query_lens)
    num_query_per_req,  # row-expansion factor per request slot
    num_slots,  # request-slot capacity (row bands are [r*nqp, (r+1)*nqp))
    WINDOW_SIZE: tl.constexpr,  # sliding window size (model constant)
    BLOCK_SIZE: tl.constexpr,  # DSA block size (power of two)
    INDEX_W: tl.constexpr,  # aligned index width
    BLOCK_W: tl.constexpr,  # column-block width
    ROW_POW2: tl.constexpr,  # next_pow2(num_blocks): UB preload tile shape
    Q_POW2: tl.constexpr,  # next_pow2(spec + 1): output row-tile height
):
    # NUM_CB must be ceil: a floor division sends tail items' r past
    # num_slots, reading qsl/seq_lens out of bounds (MTE "DDR address
    # out of range" faults on non-divisible widths).
    NUM_CB: tl.constexpr = (INDEX_W + BLOCK_W - 1) // BLOCK_W
    ALIGNED_W: tl.constexpr = (INDEX_W % BLOCK_W) == 0
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_items = num_slots * NUM_CB

    # Grid-stride: each program claims w = pid, pid + num_progs, ...
    for w in range(pid, total_items, num_progs):
        r = w // NUM_CB
        cb = w % NUM_CB

        # Distributed pad-row cleanup: reset the slice of
        # [num_rows, num_rows_padded) in this item's row band
        # [r*nqp, (r+1)*nqp). Must precede every guard below.
        row_lo = tl.maximum(r * num_query_per_req, num_rows)
        row_hi = tl.minimum((r + 1) * num_query_per_req, num_rows_padded)
        if row_lo < row_hi:
            pad_offs = tl.arange(0, BLOCK_W)
            pad_mask = (cb * BLOCK_W + pad_offs) < INDEX_W
            for row in range(row_lo, row_hi):
                base = row * INDEX_W + cb * BLOCK_W
                tl.store(slots_ptr + base + pad_offs, -1, mask=pad_mask)
                if cb == 0:
                    tl.store(lens_ptr + row, 0)

        # Positive guards (Triton has no `continue`): skipping an item
        # must not drop the remaining items this program owns.
        if r < num_reqs:
            q0 = tl.load(qsl_ptr + r).to(tl.int32)
            q1 = tl.load(qsl_ptr + r + 1).to(tl.int32)
            q_len = q1 - q0
            seq_len = tl.load(seq_lens_ptr + r).to(tl.int32)
            prefix_len = seq_len - q_len
            start_pos = tl.maximum(prefix_len - WINDOW_SIZE, 0)
            visible_len = seq_len - start_pos

            if q_len > 0:
                offs_w = cb * BLOCK_W + tl.arange(0, BLOCK_W)
                pos = start_pos + offs_w
                blk_num = pos // BLOCK_SIZE
                # Clamp so gather never reads OOB; clamped lanes are
                # discarded by the visible mask below.
                safe_num = tl.minimum(tl.maximum(blk_num, 0), num_blocks - 1)

                # triton-ascend's tl.gather rejects int32 payloads; ride
                # the id through fp32 bitcast (bits preserved). Narrow the
                # loaded row to int32 first: no-op on i32 tables, a
                # value-preserving truncation on i64 (same pattern as
                # qsl/seq_lens above; physical block IDs fit int32).
                r_offs = tl.arange(0, ROW_POW2)
                bt_row = tl.load(
                    bt_ptr + r * stride_bt_r + r_offs,
                    mask=r_offs < num_blocks,
                    other=0,
                ).to(tl.int32)
                block_id = tl.gather(bt_row.to(tl.float32, bitcast=True), safe_num, 0).to(tl.int32, bitcast=True)

                blk_off = pos - blk_num * BLOCK_SIZE
                slot = block_id * BLOCK_SIZE + blk_off
                slot = tl.where(offs_w < visible_len, slot, -1)

                is_first_cb = cb == 0
                lens_val = visible_len.to(tl.int64)

                # One 2D store for the [q_len, INDEX_W] tile: rows
                # [q0, q0+q_len) broadcast the same slot vector (the
                # uniform-query contract keeps q_len <= Q_POW2).
                t_offs = tl.arange(0, Q_POW2)
                slot_tile = tl.broadcast_to(slot[None, :], (Q_POW2, BLOCK_W))
                tile_ptrs = slots_ptr + (q0 + t_offs)[:, None] * INDEX_W + offs_w[None, :]
                if ALIGNED_W:
                    tl.store(tile_ptrs, slot_tile, mask=(t_offs < q_len)[:, None])
                else:
                    tl.store(
                        tile_ptrs,
                        slot_tile,
                        mask=(t_offs < q_len)[:, None] & (offs_w < INDEX_W)[None, :],
                    )
                tl.store(lens_ptr + q0 + t_offs, lens_val, mask=(t_offs < q_len) & is_first_cb)


# Column-block width for the 1D grid split (optimum of the 910B4 benchmark
# ladder: 1024 -> 66.7x vs eager, 512 -> 65.6x, 256 -> 51.0x).
_DEFAULT_BLOCK_W = 1024

# UB-preload tile cap: next_pow2(block_table_width) beyond this no longer
# fits the per-program UB budget, and the wrapper falls back to eager.
_MAX_ROW_POW2 = 8192


def dspark_swa_indices_supported(
    block_size: int,
    block_table_width: int,
) -> bool:
    """Host-only fast-path eligibility check (no device synchronization).

    Conditions:
    * triton-ascend is importable (HAS_TRITON);
    * ``block_size`` is a power of two (kernel lowers ``//`` and ``%`` to
      shift/mask, which requires a constexpr pow2 divisor);
    * the UB-preload tile fits: ``next_pow2(block_table_width)`` lanes must
      stay within the kernel's register/UB budget (ROW_POW2 cap).
    """

    if not HAS_TRITON:
        return False
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return False
    return triton.next_power_of_2(block_table_width) <= _MAX_ROW_POW2


class DsparkSwaIndicesKernel(AscendTritonJitKernel["DsparkSwaIndicesKernel.CompileKey"]):
    kernel = _dspark_swa_indices_kernel

    @dataclass(frozen=True)
    class CompileKey:
        window_size: int
        block_size: int
        index_w: int
        block_w: int
        row_pow2: int
        q_pow2: int
        # Representative of stride_bt_r's Triton specialization class (the
        # contiguous block-table row pitch == block_table_width).
        bt_stride: int

    def dispatch(
        self,
        *,
        window_size: int,
        block_size: int,
        index_w: int,
        block_w: int,
        row_pow2: int,
        q_pow2: int,
        bt_stride: int,
    ) -> DsparkSwaIndicesKernel.CompileKey:
        return self.CompileKey(
            window_size=window_size,
            block_size=block_size,
            index_w=index_w,
            block_w=block_w,
            row_pow2=row_pow2,
            q_pow2=q_pow2,
            bt_stride=bt_stride,
        )

    def get_warmup_keys(
        self,
        *,
        window_size: int,
        block_size: int,
        index_width: int,
        block_table_width: int,
        num_speculative_tokens: int,
    ) -> list[DsparkSwaIndicesKernel.CompileKey]:
        # The whole constexpr set is pinned by the startup configuration, so
        # each registration expands to exactly one key. All capacity scalars
        # are do_not_specialize; only stride_bt_r carries a value class.
        W = int(index_width)
        return self._trace_dispatch(self.dispatch)(
            window_size=int(window_size),
            block_size=int(block_size),
            index_w=W,
            block_w=min(_DEFAULT_BLOCK_W, W),
            row_pow2=triton.next_power_of_2(int(block_table_width)),
            q_pow2=triton.next_power_of_2(int(num_speculative_tokens) + 1),
            bt_stride=triton_scalar_specialization_rep(int(block_table_width)),
        )

    def warmup_inputs(self, compile_key: DsparkSwaIndicesKernel.CompileKey) -> dict[str, Any]:
        W = compile_key.index_w
        # Pointer specialization of the runtime buffers (all torch-allocated,
        # 16-byte aligned) is reproduced with aligned TritonWarmupTensors; the
        # kernels are compiled, never launched, during warmup.
        return dict(
            block_table=TritonWarmupTensor(torch.int32, shape=(1, 1), strides=(compile_key.bt_stride, 1)),
            query_start_loc=TritonWarmupTensor(torch.int32, shape=(2,)),
            seq_lens=TritonWarmupTensor(torch.int32, shape=(1,)),
            out_slots=TritonWarmupTensor(torch.int32, shape=(1, 1, W), strides=(W, W, 1)),
            out_lens=TritonWarmupTensor(torch.int64, shape=(1,)),
            num_blocks=1,
            num_reqs=1,
            num_rows_padded=1,
            num_rows=1,
            num_query_per_req=1,
            num_slots=1,
            WINDOW_SIZE=compile_key.window_size,
            BLOCK_SIZE=compile_key.block_size,
            INDEX_W=W,
            BLOCK_W=compile_key.block_w,
            ROW_POW2=compile_key.row_pow2,
            Q_POW2=compile_key.q_pow2,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        out_slots: torch.Tensor,
        out_lens: torch.Tensor,
        *,
        num_blocks: int,
        num_reqs: int,
        num_rows_padded: int,
        num_rows: int,
        num_query_per_req: int,
        num_slots: int,
        WINDOW_SIZE: int,
        BLOCK_SIZE: int,
        INDEX_W: int,
        BLOCK_W: int,
        ROW_POW2: int,
        Q_POW2: int,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
            # The kernel's short pointer names cannot be derived from the
            # launcher's tensor argument names; bind them explicitly.
            bt_ptr=block_table,
            qsl_ptr=query_start_loc,
            seq_lens_ptr=seq_lens,
            slots_ptr=out_slots,
            lens_ptr=out_lens,
            stride_bt_r=block_table.stride(0),
        )


_DSPARK_SWA_INDICES_KERNEL = DsparkSwaIndicesKernel()


def register_dspark_swa_indices_warmup(
    *,
    window_size: int,
    block_size: int,
    index_width: int,
    block_table_width: int,
    num_speculative_tokens: int,
) -> bool:
    """Register the DSpark SWA indices kernel for one draft attention group.

    The block-table width (and therefore ROW_POW2 and the stride
    specialization) is only final once the draft proposer and the input batch
    exist, so this runs from ``kernel_warmup`` inside its own activation
    window rather than from the metadata builder's ``__init__``. No-op
    outside an active window.
    """
    if not dspark_swa_indices_supported(block_size, block_table_width):
        return False
    _DSPARK_SWA_INDICES_KERNEL.register_warmup(
        window_size=window_size,
        block_size=block_size,
        index_width=index_width,
        block_table_width=block_table_width,
        num_speculative_tokens=num_speculative_tokens,
    )
    return True


def build_dspark_swa_slot_indices_triton(
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int | None = None,
    index_width: int | None = None,
    indices_output: torch.Tensor | None = None,
    lens_output: torch.Tensor | None = None,
    max_num_reqs: int | None = None,
    num_query_per_req: int | None = None,
    block_w: int = _DEFAULT_BLOCK_W,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton-Ascend build of DSpark non-causal SWA indices.

    Fast path for ``vllm_ascend.attention.dsa_v1.build_dspark_swa_indices``
    (same leading signature and index_width semantics). One intentional API
    difference: the lens output is always int64, while the eager chain's
    ``per_token_lens`` follows the dtype of ``seq_lens``. See
    docs/dspark_swa_indices.md for the output contract, the graph-capture
    buffer rules and the parameter semantics.
    """

    num_reqs = query_start_loc.shape[0] - 1
    if index_width is None:
        # Same alignment rule as dsa_v1._aligned_dspark_index_width.
        min_width = int(window_size) + int(num_speculative_tokens)
        index_width = ((min_width + 127) // 128) * 128
    W = int(index_width)

    # num_decode_tokens avoids the only D2H sync on this path
    # (query_start_loc[num_reqs].item()).
    if num_decode_tokens is not None:
        num_rows = int(num_decode_tokens)
    else:
        num_rows = int(query_start_loc[num_reqs].item())

    # The grid must stay fixed across replays: derive it from the
    # capture-time bound max_num_reqs, never from num_reqs.
    R_alloc = int(max_num_reqs) if max_num_reqs is not None else num_reqs
    if R_alloc < num_reqs:
        R_alloc = num_reqs
    if num_query_per_req is None:
        num_query_per_req = int(num_speculative_tokens) + 1
    num_query_per_req = int(num_query_per_req)
    if num_query_per_req < 1:
        raise ValueError(f"dspark_swa_indices num_query_per_req must be >= 1, got {num_query_per_req}")
    num_rows_padded = R_alloc * num_query_per_req

    if indices_output is not None:
        if indices_output.dtype != torch.int32:
            raise ValueError(f"dspark_swa_indices indices_output must be int32, got {indices_output.dtype}")
        out_slots = indices_output
        if out_slots.shape[0] < num_rows:
            raise ValueError(
                "dspark_swa_indices indices_output has fewer rows than active tokens: "
                f"output={out_slots.shape[0]}, active={num_rows}"
            )
        if out_slots.shape[0] < num_rows_padded:
            # Clamp the pad cleanup to the buffer extent; warn when the
            # caller signaled graph intent but passed an active-sized slice,
            # which silences the cleanup.
            if max_num_reqs is not None and R_alloc > num_reqs:
                warnings.warn(
                    "dspark_swa_indices indices_output has fewer rows than the "
                    "capacity grid "
                    f"(output={out_slots.shape[0]}, capacity={R_alloc * num_query_per_req}): "
                    "the pad-row cleanup is disabled. Pass the FULL buffer "
                    "(indices_output=buffer, not buffer[:num_rows]) so captured "
                    "graphs never replay stale rows.",
                    UserWarning,
                    stacklevel=2,
                )
            num_rows_padded = out_slots.shape[0]
    else:
        # Fresh active-sized allocation: clamp num_rows_padded so the pad cleanup
        # stays in bounds.
        out_slots = torch.empty((num_rows, 1, W), dtype=torch.int32, device=block_table.device)
        num_rows_padded = min(num_rows_padded, num_rows)

    if lens_output is not None:
        out_lens = lens_output
        if out_lens.shape[0] < num_rows:
            raise ValueError(
                f"dspark_swa_indices lens_output has fewer rows than active tokens: "
                f"output={out_lens.shape[0]}, active={num_rows}"
            )
        num_rows_padded = min(num_rows_padded, out_lens.shape[0])
    elif indices_output is not None:
        # No persistent lens buffer: allocate scratch sized to the cleanup
        # extent.
        out_lens = torch.empty((max(num_rows, num_rows_padded),), dtype=torch.int64, device=block_table.device)
    else:
        out_lens = torch.empty((num_rows,), dtype=torch.int64, device=block_table.device)

    num_blocks = block_table.shape[1]
    row_pow2 = triton.next_power_of_2(num_blocks)
    eff_block_w = min(block_w, W)  # BLOCK_W > INDEX_W would zero NUM_CB
    num_cb = triton.cdiv(W, eff_block_w)

    # num_slots (R_alloc) stays unclamped so buffer-extent clamping of
    # num_rows_padded does not shrink the kernel's claim space. The kernel
    # claims work items grid-stride, so the core count is a performance
    # knob only; any grid >= 1 covers the work.
    total_items = R_alloc * num_cb
    init_device_properties_triton()
    grid = min(get_vectorcore_num(), total_items)

    _DSPARK_SWA_INDICES_KERNEL(
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        out_slots=out_slots,
        out_lens=out_lens,
        num_blocks=num_blocks,
        num_reqs=num_reqs,
        num_rows_padded=num_rows_padded,
        num_rows=num_rows,
        num_query_per_req=num_query_per_req,
        num_slots=R_alloc,
        WINDOW_SIZE=int(window_size),
        BLOCK_SIZE=int(block_size),
        INDEX_W=W,
        BLOCK_W=eff_block_w,
        ROW_POW2=row_pow2,
        Q_POW2=triton.next_power_of_2(int(num_speculative_tokens) + 1),
        grid_size=grid,
    )

    # Return views sized to the active extent, mirroring the eager function.
    return out_slots[:num_rows], out_lens[:num_rows]
