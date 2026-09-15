# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-Ascend fused implementation of build_dspark_swa_indices.

Replaces the eager 12-op chain (arange / compare / div / gather / where /
repeat_interleave / copy_ ...) used by the DSA metadata builder for DSpark
non-causal parallel drafting (see vllm_ascend.attention.dsa_v1).

Design notes (cross-reviewed against the ACL-graph integration plan):

* Capacity grid: the kernel is launched with ``grid=(MAX_REQS * NUM_CB,)``
  where MAX_REQS is stable across graph replays. The active request count is
  read on device from ``query_start_loc``; programs with ``r >= active_r``
  return early. This keeps the launch shape fixed for capture while the
  per-step batch size varies.
* Pad-row cleanup: programs past the last active request fill the padded
  output rows with the docstring contract values (-1 slots / 0 lens) so the
  interval ``[T_active, T_padded)`` never carries stale data into a captured
  graph. This is a strict superset of the eager behavior (the eager path
  leaves those rows untouched); diffs there are expected and intentional.
* Out-variant: ``indices_output`` / ``lens_output`` accept pre-allocated
  persistent buffers so tensor addresses stay stable across ACL-graph
  replays (the DSA operator freezes ``ori_sparse_indices``'s pointer at
  capture time).

dtype/precision contract (validated in triton_ascend_output benchmarks):
- tl.gather on Ascend only accepts fp16/fp32/bf16/fp8/int8 sources, so the
  block-table row is preloaded as fp32 and gathered, then cast back to
  int32. Block ids < 2^24 keep this lossless in fp32.
- int32 //, clamp and compares in the column chain lower to scalar ops on
  Ascend; the same math in fp32 (< 2^24, exact) rides the vector unit.
"""

from __future__ import annotations

import warnings

import torch

try:
    from vllm.triton_utils import HAS_TRITON, tl, triton
except ImportError:  # standalone test environments without vllm installed
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True

if HAS_TRITON:

    @triton.jit(
        do_not_specialize=[
            "B",
            "R",
            "T_padded",
            "T_active",
        ]
    )
    def _dspark_swa_indices_kernel(
        bt_ptr,  # block_table       [R_alloc, B]  i32/i64
        stride_bt_r,  # block_table.stride(0): row pitch of a possibly sliced/padded table
        qsl_ptr,  # query_start_loc  [R_alloc + 1]  i32/i64
        seq_lens_ptr,  # seq_lens         [R_alloc]  i32/i64
        slots_ptr,  # out_slots        [T_padded, 1, INDEX_W]  i32
        lens_ptr,  # out_lens         [T_padded]  i64
        B,  # block_table.shape[1]
        R,  # active request count == block_table.shape[0]
        T_padded,  # padded row count == R_alloc * num_query_per_req
        T_active,  # active row count == sum(query_lens)
        WINDOW_SIZE: tl.constexpr,  # sliding window size (model constant)
        BLOCK_SIZE: tl.constexpr,  # DSA block size (power of two)
        INDEX_W: tl.constexpr,  # aligned index width
        BLOCK_W: tl.constexpr,  # column-block width
        ROW_POW2: tl.constexpr,  # next_pow2(B): UB preload tile shape
    ):
        # 1D capacity grid: pid = r * NUM_CB + cb. NUM_CB must match the
        # wrapper's cdiv(W, BLOCK_W): a floor division here would send the
        # tail programs' r past R_alloc and read qsl/seq_lens out of bounds
        # (observed as MTE "DDR address out of range" vector-core faults on
        # non-divisible widths, e.g. INDEX_W=8320 vs BLOCK_W=1024).
        NUM_CB: tl.constexpr = (INDEX_W + BLOCK_W - 1) // BLOCK_W
        # Aligned widths keep the mask-free (KVR) store path; a tail column
        # block on non-divisible widths must mask lanes >= INDEX_W.
        ALIGNED_W: tl.constexpr = (INDEX_W % BLOCK_W) == 0
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        r = pid // NUM_CB
        cb = pid % NUM_CB

        # ---- pad-row cleanup branch (CUDA-graph safety) ----
        # Mirrors _prepare_dflash_inputs_kernel_ascend: the last allocated
        # program per column block fills [T_active, T_padded) with the
        # docstring contract values so a captured graph never replays stale
        # rows. The loop body is empty (and effectively free) in the common
        # no-padding case; ``range`` with runtime bounds is the dynamic-loop
        # form proven by the dflash production kernel.
        # NOTE: must run before any early-return so pad rows are always
        # cleaned, including when the owning program's request is inactive
        # or out of the allocation extent.
        if pid >= num_progs - NUM_CB:
            pad_offs = tl.arange(0, BLOCK_W)
            pad_mask = (cb * BLOCK_W + pad_offs) < INDEX_W
            for row in range(T_active, T_padded):
                base = row * INDEX_W + cb * BLOCK_W
                tl.store(slots_ptr + base + pad_offs, -1, mask=pad_mask)
                if cb == 0:
                    tl.store(lens_ptr + row, 0)

        # Out-of-allocation requests (r >= R): the production caller pads
        # qsl/seq_lens/block_table to the capacity extent so these lanes read
        # q_len == 0 and early-out below; but an exact-sized tensor set would
        # make the loads read past the end of the arrays (MTE OOB). Guard
        # explicitly so both allocation styles are safe.
        if r >= R:
            return

        # ---- per-request scalar derivation ----
        # qsl comes from the upstream draft metadata builder, which clamps
        # the cumulative series at the real num_reqs: padded requests see
        # q_len == 0 and the series stops at T_active.
        q0 = tl.load(qsl_ptr + r).to(tl.int32)
        q1 = tl.load(qsl_ptr + r + 1).to(tl.int32)
        q_len = q1 - q0
        seq_len = tl.load(seq_lens_ptr + r).to(tl.int32)
        prefix_len = seq_len - q_len
        start_pos = tl.maximum(prefix_len - WINDOW_SIZE, 0)
        visible_len = seq_len - start_pos

        # Early-out for inactive requests (q_len == 0 on padded rows): their
        # output rows are covered by the cleanup branch above.
        if q_len <= 0:
            return

        # ---- column-block vector compute ----
        offs_w = cb * BLOCK_W + tl.arange(0, BLOCK_W)
        pos = start_pos + offs_w
        # BLOCK_SIZE is a constexpr power of two: div lowers to shift/mask in
        # int32 and stays vectorized (int32-vec // constexpr is fine; only
        # fp32 // was rejected by the Ascend frontend).
        blk_num = pos // BLOCK_SIZE
        # Clamp to valid block-table columns so gather never reads OOB; the
        # clamped lanes are discarded by the visible mask below anyway.
        # fp32 min/max ride the vector unit; values < 2^24 stay exact.
        blk_f = blk_num.to(tl.float32)
        safe_num = tl.minimum(tl.maximum(blk_f, 0.0), (B - 1).to(tl.float32)).to(tl.int32)

        # ---- UB-preload row + tl.gather ----
        # triton-ascend tl.gather only accepts fp16/fp32/bf16/fp8/int8 sources
        # (int32/int64 raise CompilationError). block-table values fit in 24
        # bits, so an fp32 roundtrip is lossless for the gather payload.
        r_offs = tl.arange(0, ROW_POW2)
        bt_row_f32 = tl.load(bt_ptr + r * stride_bt_r + r_offs, mask=r_offs < B, other=0).to(tl.float32)
        block_id = tl.gather(bt_row_f32, safe_num, 0).to(tl.int32)

        # ---- slot arithmetic + visible mask ----
        blk_off = pos - blk_num * BLOCK_SIZE  # == pos % BLOCK_SIZE
        slot = block_id * BLOCK_SIZE + blk_off
        # int32 compare -> fp32 vector compare (rides the vector unit).
        slot = tl.where(offs_w.to(tl.float32) < visible_len.to(tl.float32), slot, -1)

        # ---- fused repeat_interleave (row-broadcast store) ----
        # Aligned widths drop the store mask entirely (KVR); a tail column
        # block on non-divisible widths masks lanes >= INDEX_W.
        is_first_cb = cb.to(tl.float32) == 0.0
        lens_val = visible_len.to(tl.int64)
        for t in range(0, q_len):
            if ALIGNED_W:
                tl.store(slots_ptr + (q0 + t) * INDEX_W + offs_w, slot)
            else:
                tl.store(slots_ptr + (q0 + t) * INDEX_W + offs_w, slot, mask=offs_w < INDEX_W)
            tl.store(lens_ptr + q0 + t, lens_val, mask=is_first_cb)


# Column-block width for the 1D grid split. Benchmark ladder on 910B4
# (triton_ascend_output block-size scan, R=8 uniform decode): 1024 is the
# optimum (1024 -> 66.7x vs eager, 512 -> 65.6x, 256 -> 51.0x); BLOCK_W
# beyond INDEX_W would make NUM_CB == 0 and break the grid math, so it is
# clamped to INDEX_W by the wrapper.
_DEFAULT_BLOCK_W = 1024

# ROW_POW2 = next_pow2(block_table_width) shapes the fp32 UB-preload tile of
# the block-table row. DeepSeek-V4 decode keeps B well below 2^13 (~8k
# blocks = 1M tokens); beyond the cap the tile no longer fits the per-program
# UB budget, so the wrapper falls back to the eager path.
_MAX_ROW_POW2 = 8192


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
    * ``max_num_reqs`` sizes the capacity grid: ``grid=(max_num_reqs * NUM_CB,)``
      is fixed across steps so the launch shape is stable for ACL-graph
      capture. Rows in ``[T_active, T_padded)`` of the output buffers are
      explicitly reset to (-1, 0) — a strict superset of the eager behavior,
      which leaves those rows stale.
    * **Callers that want the pad cleanup to fire must pass the FULL buffer**
      (``indices_output=buffer``, not ``buffer[:T_active]``): the cleanup
      extent is clamped to ``indices_output.shape[0]``, so a slice cut to
      the active rows silences it (a UserWarning flags this when
      ``max_num_reqs`` signals graph intent). The returned view is still
      sized to the active rows.
    * ``num_query_per_req`` overrides the row-expansion factor used for the
      capacity grid. It defaults to ``num_speculative_tokens + 1`` (the
      DFlash-style drafting contract); the DSpark anchor-sampling mode uses
      ``num_speculative_tokens`` instead
      (``dspark_proposer.num_query_per_req``). With the default in that
      mode, ``T_padded`` overshoots the true padded row count — the extra
      rows are simply skipped — and buffer sizing is only safe by
      coincidence of the caller's allocation quota, so pass the exact
      factor when it is known.
    * ``num_decode_tokens`` must equal ``num_reqs * num_query_per_req``
      when given (the uniform-query contract of DSpark parallel drafting);
      it is used for the T_active scalar, avoiding any D2H sync.
    """

    R = query_start_loc.shape[0] - 1
    if index_width is None:
        # Same alignment rule as dsa_v1._aligned_dspark_index_width.
        min_width = int(window_size) + int(num_speculative_tokens)
        index_width = ((min_width + 127) // 128) * 128
    W = int(index_width)

    # Active row count. num_decode_tokens is the caller's authoritative
    # count under the uniform-query contract (== num_reqs * num_query_per_req
    # for the active requests); without it we fall back to reading the clamped
    # cumulative end — the only D2H sync on this path. The production caller
    # always passes num_decode_tokens.
    if num_decode_tokens is not None:
        T_active = int(num_decode_tokens)
    else:
        T_active = int(query_start_loc[R].item())

    # Capacity sizing. The grid must stay fixed across replays, so it is
    # derived from max_num_reqs (capture-time bound), never from R.
    R_alloc = int(max_num_reqs) if max_num_reqs is not None else R
    if R_alloc < R:
        R_alloc = R
    if num_query_per_req is None:
        num_query_per_req = int(num_speculative_tokens) + 1
    num_query_per_req = int(num_query_per_req)
    if num_query_per_req < 1:
        raise ValueError(f"dspark_swa_indices num_query_per_req must be >= 1, got {num_query_per_req}")
    T_padded = R_alloc * num_query_per_req

    if indices_output is not None:
        if indices_output.dtype != torch.int32:
            raise ValueError(f"dspark_swa_indices indices_output must be int32, got {indices_output.dtype}")
        out_slots = indices_output
        if out_slots.shape[0] < T_active:
            raise ValueError(
                "dspark_swa_indices indices_output has fewer rows than active tokens: "
                f"output={out_slots.shape[0]}, active={T_active}"
            )
        if out_slots.shape[0] < T_padded:
            # Buffer sized below the capacity grid: clamp the pad cleanup to
            # the buffer extent rather than erroring out. A slice cut to the
            # active rows (the classic eager-style call site) silences the
            # cleanup entirely; flag that when the caller signaled graph
            # intent via max_num_reqs.
            if max_num_reqs is not None and R_alloc > R:
                warnings.warn(
                    "dspark_swa_indices indices_output has fewer rows than the "
                    "capacity grid "
                    f"(output={out_slots.shape[0]}, capacity={R_alloc * num_query_per_req}): "
                    "the pad-row cleanup is disabled. Pass the FULL buffer "
                    "(indices_output=buffer, not buffer[:T_active]) so captured "
                    "graphs never replay stale rows.",
                    UserWarning,
                    stacklevel=2,
                )
            T_padded = out_slots.shape[0]
    else:
        # No persistent buffer: pad cleanup beyond the fresh active-sized
        # allocation would write out of bounds, so clamp T_padded down. The
        # graph-capture caller always passes indices_output and keeps
        # T_padded at the full capacity extent.
        out_slots = torch.empty((T_active, 1, W), dtype=torch.int32, device=block_table.device)
        T_padded = min(T_padded, T_active)

    if lens_output is not None:
        out_lens = lens_output
        if out_lens.shape[0] < T_active:
            raise ValueError(
                f"dspark_swa_indices lens_output has fewer rows than active tokens: "
                f"output={out_lens.shape[0]}, active={T_active}"
            )
        # Clamp the pad cleanup to the lens buffer extent as well.
        T_padded = min(T_padded, out_lens.shape[0])
    elif indices_output is not None:
        # Persistent indices buffer without a lens buffer: the caller treats
        # lens as scratch (the production call site discards it). Allocate a
        # transient tensor sized to the (already clamped) cleanup extent so
        # the kernel can still write it.
        out_lens = torch.empty((max(T_active, T_padded),), dtype=torch.int64, device=block_table.device)
    else:
        out_lens = torch.empty((T_active,), dtype=torch.int64, device=block_table.device)

    B = block_table.shape[1]
    row_pow2 = triton.next_power_of_2(B)
    # BLOCK_W > INDEX_W would zero NUM_CB and break the grid math; clamp.
    eff_block_w = min(block_w, W)
    num_cb = triton.cdiv(W, eff_block_w)

    _dspark_swa_indices_kernel[(R_alloc * num_cb,)](
        block_table,
        block_table.stride(0),
        query_start_loc,
        seq_lens,
        out_slots,
        out_lens,
        B,
        R,
        T_padded,
        T_active,
        WINDOW_SIZE=int(window_size),
        BLOCK_SIZE=int(block_size),
        INDEX_W=W,
        BLOCK_W=eff_block_w,
        ROW_POW2=row_pow2,
    )

    # Return views sized to the active extent, mirroring the eager function.
    return out_slots[:T_active], out_lens[:T_active]


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

    ROW_POW2 derives from block_table.shape[1] and B can change between
    capture and steady state (physical-page remapping across kernel block
    sizes). A post-capture JIT would block the replay path, so
    ``enable_dspark_device_metadata`` should call this once per (B, W)
    combination, including the pad-cleanup branch (T_padded > T_active).
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
