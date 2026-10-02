# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-Ascend fused implementation of build_dspark_swa_indices.

Replaces the eager 12-op chain (arange / compare / div / gather / where /
repeat_interleave / copy_ ...) used by the DSA metadata builder for DSpark
non-causal parallel drafting (see vllm_ascend.attention.dsa_v1).

Graph-capture contract:
* Launched on a fixed grid (``min(AIV cores, max_num_reqs * NUM_CB)``); the
  kernel claims the capacity work items grid-stride, so the launch shape is
  stable across ACL-graph replays while the per-step batch size varies.
* Programs past the last active request reset the padded rows
  ``[num_rows, num_rows_padded)`` to (-1, 0) so a captured graph never replays
  stale rows; the eager path leaves those rows untouched.
* ``indices_output`` / ``lens_output`` accept pre-allocated persistent
  buffers so tensor addresses stay stable across replays.

Precision: tl.gather on Ascend only accepts fp16/fp32/bf16/fp8/int8 sources
and int32 //, clamp and compares lower to scalar ops; an fp32 roundtrip
handles both and stays exact below 2^24.
"""

from __future__ import annotations

import warnings

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

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
        bt_ptr,  # block_table       [R_alloc, num_blocks]  i32/i64
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
                    blk_f = blk_num.to(tl.float32)
                    safe_num = tl.minimum(tl.maximum(blk_f, 0.0), (num_blocks - 1).to(tl.float32)).to(tl.int32)

                    # fp32 roundtrip: tl.gather rejects int32 sources, and
                    # fp32 math rides the vector unit; exact < 2^24.
                    r_offs = tl.arange(0, ROW_POW2)
                    bt_row_f32 = tl.load(
                        bt_ptr + r * stride_bt_r + r_offs,
                        mask=r_offs < num_blocks,
                        other=0,
                    ).to(tl.float32)
                    block_id = tl.gather(bt_row_f32, safe_num, 0).to(tl.int32)

                    blk_off = pos - blk_num * BLOCK_SIZE
                    slot = block_id * BLOCK_SIZE + blk_off
                    slot = tl.where(offs_w.to(tl.float32) < visible_len.to(tl.float32), slot, -1)

                    is_first_cb = cb.to(tl.float32) == 0.0
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

# AIV core count bounding the grid-stride launch (a performance knob: any
# grid >= 1 is correct). Fetched from the device properties, falling back
# to the A2-class count the gate admits.
_NUM_AIV_CORES = 40


def _num_aiv_cores() -> int:
    try:
        props = torch.npu.get_device_properties(0)
        vector_core_num = getattr(props, "vector_core_num", None)
        if vector_core_num is not None and vector_core_num > 0:
            return int(vector_core_num)
    except Exception:
        pass
    return _NUM_AIV_CORES


def dspark_swa_indices_supported(
    block_size: int,
    block_table_width: int,
) -> bool:
    """Host-only fast-path eligibility check (no device synchronization).

    Conditions:
    * triton-ascend is importable (HAS_TRITON);
    * the SoC belongs to a validated family (A2 / 910B series; A3 and later
      need on-device benchmark sign-off before being added here);
    * ``block_size`` is a power of two (kernel lowers ``//`` and ``%`` to
      shift/mask, which requires a constexpr pow2 divisor);
    * the UB-preload tile fits: ``next_pow2(block_table_width)`` lanes must
      stay within the kernel's register/UB budget (ROW_POW2 cap).
    """

    if not HAS_TRITON:
        return False
    try:
        from vllm_ascend.device.device_config import get_ascend_device_type

        if get_ascend_device_type().name != "A2":
            return False
    except Exception:
        return False
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return False
    return triton.next_power_of_2(block_table_width) <= _MAX_ROW_POW2


def build_dspark_swa_indices_triton(
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

    Drop-in fast path for ``vllm_ascend.attention.dsa_v1.build_dspark_swa_indices``
    (same leading signature and index_width semantics). Differences that are
    intentional and graph-mode specific:

    * ``indices_output`` / ``lens_output`` are written in place; when omitted,
      fresh tensors sized to the active rows are allocated.
    * ``max_num_reqs`` sizes the capacity grid: the claim space is
      ``max_num_reqs * NUM_CB`` work items, and the launch grid is
      ``min(num_aiv_cores, max_num_reqs * NUM_CB)`` — fixed across steps so
      the launch shape is stable for ACL-graph capture while the per-step
      batch size varies. Rows in ``[num_rows, num_rows_padded)`` of the
      output buffers are explicitly reset to (-1, 0) — a strict superset of
      the eager behavior, which leaves those rows stale.
    * Pass the FULL ``indices_output`` buffer (``buffer``, not
      ``buffer[:num_rows]``) to keep the pad cleanup armed: the cleanup
      extent is clamped to the buffer, so an active-sized slice silences it
      (flagged by a UserWarning under graph intent).
    * ``num_query_per_req`` overrides the capacity-grid row-expansion
      factor. It defaults to ``num_speculative_tokens + 1`` (DFlash-style
      contract); the DSpark anchor-sampling mode uses
      ``num_speculative_tokens`` instead — pass the exact factor when it
      is known.
    * ``num_decode_tokens`` (== ``num_reqs * num_query_per_req`` under the
      uniform-query contract) supplies the num_rows scalar, avoiding any
      D2H sync. The contract bounds each request's query count by
      ``num_speculative_tokens + 1``; the kernel's output tile relies on it.
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
    # num_rows_padded does not shrink the kernel's claim space.
    total_items = R_alloc * num_cb
    grid = min(_num_aiv_cores(), total_items)

    _dspark_swa_indices_kernel[(grid,)](
        block_table,
        block_table.stride(0),
        query_start_loc,
        seq_lens,
        out_slots,
        out_lens,
        num_blocks,
        num_reqs,
        num_rows_padded,
        num_rows,
        num_query_per_req,
        R_alloc,
        WINDOW_SIZE=int(window_size),
        BLOCK_SIZE=int(block_size),
        INDEX_W=W,
        BLOCK_W=eff_block_w,
        ROW_POW2=row_pow2,
        Q_POW2=triton.next_power_of_2(int(num_speculative_tokens) + 1),
    )

    # Return views sized to the active extent, mirroring the eager function.
    return out_slots[:num_rows], out_lens[:num_rows]


def warmup_dspark_swa_indices_triton(
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int,
    index_width: int,
    indices_output: torch.Tensor,
    lens_output: torch.Tensor | None,
    max_num_reqs: int,
) -> None:
    """JIT-warm the kernel for the exact shapes seen at capture time.

    ROW_POW2 derives from block_table.shape[1] and num_blocks can change
    between capture and steady state (physical-page remapping across kernel
    block sizes). A post-capture JIT would block the replay path, so
    ``enable_dspark_device_metadata`` should call this once per (num_blocks, W)
    combination, including the pad-cleanup branch (num_rows_padded > num_rows).
    """

    build_dspark_swa_indices_triton(
        block_table,
        num_speculative_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        num_decode_tokens=num_decode_tokens,
        index_width=index_width,
        indices_output=indices_output,
        lens_output=lens_output,
        max_num_reqs=max_num_reqs,
    )
