# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified Triton-Ascend implementation of DSpark SWA index construction.

Supports both physical KV slot IDs (DSA) and request-local logical sequence
positions (DSA V4.1 / SparseFlashMLA) through a single fused kernel and shared
entry point. See docs/dspark_swa_indices.md for the full description,
semantics, constraints, and Ascend-specific lowering notes.
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

# Column-block width for the 1D grid split (optimum of the 910B4 benchmark
# ladder: 1024 -> 66.7x vs eager, 512 -> 65.6x, 256 -> 51.0x).
_DEFAULT_BLOCK_W = 1024

# Bound the 2D output store tile height so (Q_POW2, BLOCK_W) fits comfortably
# in UB even when a caller supplies large query counts per request.
_MAX_Q_TILE = 16

_INDEX_WIDTH_ALIGNMENT = 128
_MIN_BT_WIN_POW2 = 2
_ACCELERATOR_DEVICE_TYPES = ("npu", "cuda")
_SUPPORTED_METADATA_DTYPES = (torch.int32, torch.int64)

OUTPUT_MODE_PHYSICAL = "physical"
OUTPUT_MODE_LOGICAL = "logical"
_VALID_OUTPUT_MODES = (OUTPUT_MODE_PHYSICAL, OUTPUT_MODE_LOGICAL)


def normalize_dspark_swa_output_mode(
    output_mode: str | None = None,
    *,
    use_logical_indices: bool = False,
) -> str:
    """Normalize and validate DSpark SWA output mode selection."""
    if output_mode is None:
        return OUTPUT_MODE_LOGICAL if use_logical_indices else OUTPUT_MODE_PHYSICAL
    if output_mode not in _VALID_OUTPUT_MODES:
        raise ValueError(f"Unsupported DSpark SWA output_mode={output_mode!r}; expected one of {_VALID_OUTPUT_MODES}")
    if use_logical_indices and output_mode != OUTPUT_MODE_LOGICAL:
        raise ValueError(
            f"Conflicting DSpark SWA mode arguments: output_mode={output_mode!r} with use_logical_indices=True"
        )
    return output_mode


def _compute_eff_block_w(index_width: int, block_w: int = _DEFAULT_BLOCK_W) -> int:
    """Return a power-of-two column-tile width bounded by next_pow2(index_width)."""
    return min(
        triton.next_power_of_2(max(1, int(block_w))),
        triton.next_power_of_2(max(1, int(index_width))),
    )


def _compute_bt_win_pow2(block_w: int, block_size: int, use_logical_indices: bool) -> int:
    """Return power-of-two UB window size covering one column tile's block span.

    For any start_pos >= 0 and column block of width block_w, the integer block
    indices pos // block_size span at most floor((block_w + block_size - 2) /
    block_size) + 1 consecutive block-table columns, independent of total width.
    """
    if use_logical_indices:
        return 1
    max_span_blocks = (int(block_w) + int(block_size) - 2) // int(block_size) + 1
    return max(_MIN_BT_WIN_POW2, triton.next_power_of_2(max_span_blocks))


def _compute_q_pow2(num_speculative_tokens: int) -> int:
    """Return bounded power-of-two row-tile height for the 2D output store."""
    return min(_MAX_Q_TILE, triton.next_power_of_2(max(1, int(num_speculative_tokens) + 1)))


@triton.jit(
    do_not_specialize=[
        "num_blocks",
        "num_reqs",
        "num_rows_padded",
        "num_query_per_req",
        "num_slots",
    ]
)
def _dspark_swa_indices_kernel(
    bt_ptr,  # block_table [R_alloc, num_blocks] i32/i64 (unused in logical mode)
    stride_bt_r,  # block_table.stride(0): row pitch of a possibly sliced/padded table
    qsl_ptr,  # query_start_loc [R_alloc + 1] i32/i64
    seq_lens_ptr,  # seq_lens [R_alloc] i32/i64
    slots_ptr,  # out_slots [num_rows_padded, 1, INDEX_W] i32
    lens_ptr,  # out_lens [num_rows_padded] or [num_rows_padded, 1] i32/i64
    num_blocks,  # block_table.shape[1]
    num_reqs,  # active request count
    num_rows_padded,  # padded cleanup upper bound
    num_query_per_req,  # row-expansion factor per request slot
    num_slots,  # request-slot grid capacity
    WINDOW_SIZE: tl.constexpr,  # sliding window size (model constant)
    BLOCK_SIZE: tl.constexpr,  # DSA KV block size (any positive integer)
    INDEX_W: tl.constexpr,  # aligned index width
    BLOCK_W: tl.constexpr,  # column-block width (power of two)
    BT_WIN_POW2: tl.constexpr,  # bounded block-table window width (power of two)
    Q_POW2: tl.constexpr,  # output row-tile height (power of two)
    USE_LOGICAL_INDICES: tl.constexpr,  # True -> sequence positions; False -> physical slots
):
    # NUM_CB must be ceil so every column up to INDEX_W is visited.
    NUM_CB: tl.constexpr = (INDEX_W + BLOCK_W - 1) // BLOCK_W
    ALIGNED_W: tl.constexpr = (INDEX_W % BLOCK_W) == 0
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_items = num_slots * NUM_CB
    active_rows = tl.load(qsl_ptr + num_reqs).to(tl.int32)
    col_offs = tl.arange(0, BLOCK_W)
    t_base = tl.arange(0, Q_POW2)

    for w in range(pid, total_items, num_progs):
        r = w // NUM_CB
        cb = w % NUM_CB
        offs_w = cb * BLOCK_W + col_offs
        col_mask = offs_w < INDEX_W

        # Distributed pad-row cleanup for [active_rows, num_rows_padded).
        row_lo = tl.maximum(r * num_query_per_req, active_rows)
        row_hi = tl.minimum((r + 1) * num_query_per_req, num_rows_padded)
        if row_lo < row_hi:
            for row in range(row_lo, row_hi):
                base = tl.cast(row, tl.int64) * INDEX_W + cb * BLOCK_W
                tl.store(slots_ptr + base + col_offs, -1, mask=col_mask)
                if cb == 0:
                    tl.store(lens_ptr + row, 0)

        if r < num_reqs:
            q0 = tl.load(qsl_ptr + r).to(tl.int32)
            q1 = tl.load(qsl_ptr + r + 1).to(tl.int32)
            q_len = q1 - q0
            if q_len > 0:
                seq_len = tl.load(seq_lens_ptr + r).to(tl.int32)
                prefix_len = seq_len - q_len
                start_pos = tl.maximum(prefix_len - WINDOW_SIZE, 0)
                visible_len = seq_len - start_pos
                pos = start_pos + offs_w
                if USE_LOGICAL_INDICES:
                    slot = pos
                else:
                    blk_num = pos // BLOCK_SIZE
                    safe_num = tl.minimum(tl.maximum(blk_num, 0), num_blocks - 1)
                    blk_base = tl.minimum(tl.maximum((start_pos + cb * BLOCK_W) // BLOCK_SIZE, 0), num_blocks - 1)
                    bt_cols = blk_base + tl.arange(0, BT_WIN_POW2)
                    # Narrow block IDs to int32 and bitcast through fp32 for tl.gather.
                    bt_win = tl.load(
                        bt_ptr + r.to(tl.int64) * stride_bt_r + bt_cols,
                        mask=bt_cols < num_blocks,
                        other=0,
                    ).to(tl.int32)
                    block_id = tl.gather(bt_win.to(tl.float32, bitcast=True), safe_num - blk_base, 0).to(
                        tl.int32, bitcast=True
                    )
                    blk_off = pos - blk_num * BLOCK_SIZE
                    slot = block_id * BLOCK_SIZE + blk_off
                slot = tl.where(offs_w < visible_len, slot, -1)
                lens_val = visible_len.to(lens_ptr.dtype.element_ty)
                slot_tile = tl.broadcast_to(slot[None, :], (Q_POW2, BLOCK_W))
                for q_off in range(0, q_len, Q_POW2):
                    t_offs = q_off + t_base
                    row_mask = t_offs < q_len
                    tile_ptrs = slots_ptr + (q0 + t_offs).to(tl.int64)[:, None] * INDEX_W + offs_w[None, :]
                    if ALIGNED_W:
                        tl.store(tile_ptrs, slot_tile, mask=row_mask[:, None])
                    else:
                        tl.store(tile_ptrs, slot_tile, mask=row_mask[:, None] & col_mask[None, :])
                    if cb == 0:
                        tl.store(lens_ptr + q0 + t_offs, lens_val, mask=row_mask)


def dspark_swa_indices_supported(
    block_size: int = 1,
    block_table_width: int = 1,
    *,
    output_mode: str = OUTPUT_MODE_PHYSICAL,
) -> bool:
    """Host-only eligibility check for the unified DSpark SWA Triton kernel."""
    if not HAS_TRITON or output_mode not in _VALID_OUTPUT_MODES:
        return False
    if output_mode == OUTPUT_MODE_LOGICAL:
        return True
    return int(block_size) > 0 and int(block_table_width) > 0


class DsparkSwaIndicesKernel(AscendTritonJitKernel["DsparkSwaIndicesKernel.CompileKey"]):
    kernel = _dspark_swa_indices_kernel

    @dataclass(frozen=True)
    class CompileKey:
        window_size: int
        block_size: int
        index_w: int
        block_w: int
        bt_win_pow2: int
        q_pow2: int
        bt_stride: int
        use_logical_indices: bool
        lens_dtype: torch.dtype

    def dispatch(
        self,
        *,
        window_size: int,
        block_size: int,
        index_w: int,
        block_w: int,
        bt_win_pow2: int,
        q_pow2: int,
        bt_stride: int,
        use_logical_indices: bool,
        lens_dtype: torch.dtype,
    ) -> DsparkSwaIndicesKernel.CompileKey:
        return self.CompileKey(
            window_size=window_size,
            block_size=block_size,
            index_w=index_w,
            block_w=block_w,
            bt_win_pow2=bt_win_pow2,
            q_pow2=q_pow2,
            bt_stride=bt_stride,
            use_logical_indices=use_logical_indices,
            lens_dtype=lens_dtype,
        )

    def get_warmup_keys(
        self,
        *,
        window_size: int,
        block_size: int,
        index_width: int,
        block_table_width: int,
        num_speculative_tokens: int,
        output_mode: str = OUTPUT_MODE_PHYSICAL,
        lens_dtype: torch.dtype | None = None,
    ) -> list[DsparkSwaIndicesKernel.CompileKey]:
        mode = normalize_dspark_swa_output_mode(output_mode)
        use_logical = mode == OUTPUT_MODE_LOGICAL
        eff_width = int(index_width)
        eff_block_w = _compute_eff_block_w(eff_width)
        eff_block_size = 1 if use_logical else int(block_size)
        eff_lens_dtype = (torch.int32 if use_logical else torch.int64) if lens_dtype is None else lens_dtype
        eff_bt_stride = 1 if use_logical else triton_scalar_specialization_rep(int(block_table_width))
        return self._trace_dispatch(self.dispatch)(
            window_size=int(window_size),
            block_size=eff_block_size,
            index_w=eff_width,
            block_w=eff_block_w,
            bt_win_pow2=_compute_bt_win_pow2(eff_block_w, eff_block_size, use_logical),
            q_pow2=_compute_q_pow2(num_speculative_tokens),
            bt_stride=eff_bt_stride,
            use_logical_indices=use_logical,
            lens_dtype=eff_lens_dtype,
        )

    def warmup_inputs(self, compile_key: DsparkSwaIndicesKernel.CompileKey) -> dict[str, Any]:
        width = compile_key.index_w
        lens_shape = (1, 1) if compile_key.use_logical_indices else (1,)
        return dict(
            block_table=TritonWarmupTensor(torch.int32, shape=(1, 1), strides=(compile_key.bt_stride, 1)),
            query_start_loc=TritonWarmupTensor(torch.int32, shape=(2,)),
            seq_lens=TritonWarmupTensor(torch.int32, shape=(1,)),
            out_slots=TritonWarmupTensor(torch.int32, shape=(1, 1, width), strides=(width, width, 1)),
            out_lens=TritonWarmupTensor(compile_key.lens_dtype, shape=lens_shape),
            num_blocks=1,
            num_reqs=1,
            num_rows_padded=1,
            num_query_per_req=1,
            num_slots=1,
            WINDOW_SIZE=compile_key.window_size,
            BLOCK_SIZE=compile_key.block_size,
            INDEX_W=width,
            BLOCK_W=compile_key.block_w,
            BT_WIN_POW2=compile_key.bt_win_pow2,
            Q_POW2=compile_key.q_pow2,
            USE_LOGICAL_INDICES=compile_key.use_logical_indices,
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
        num_query_per_req: int,
        num_slots: int,
        WINDOW_SIZE: int,
        BLOCK_SIZE: int,
        INDEX_W: int,
        BLOCK_W: int,
        BT_WIN_POW2: int,
        Q_POW2: int,
        USE_LOGICAL_INDICES: bool,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(
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
    output_mode: str = OUTPUT_MODE_PHYSICAL,
    lens_dtype: torch.dtype | None = None,
) -> bool:
    """Register the unified DSpark SWA indices kernel for one draft attention group."""
    mode = normalize_dspark_swa_output_mode(output_mode)
    if not dspark_swa_indices_supported(block_size, block_table_width, output_mode=mode):
        return False
    _DSPARK_SWA_INDICES_KERNEL.register_warmup(
        window_size=window_size,
        block_size=block_size,
        index_width=index_width,
        block_table_width=block_table_width,
        num_speculative_tokens=num_speculative_tokens,
        output_mode=mode,
        lens_dtype=lens_dtype,
    )
    return True


def _validate_device_residency(
    *,
    use_logical: bool,
    block_table: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    indices_output: torch.Tensor | None,
    lens_output: torch.Tensor | None,
) -> torch.device:
    """Validate that all device-dereferenced tensors live on one accelerator."""
    device = query_start_loc.device
    if device.type not in _ACCELERATOR_DEVICE_TYPES:
        raise ValueError(
            f"dspark_swa_indices requires an accelerator device, got query_start_loc on {device}; "
            "the triton kernel would dereference host memory"
        )
    tensors: list[tuple[str, torch.Tensor | None]] = [
        ("query_start_loc", query_start_loc),
        ("seq_lens", seq_lens),
        ("indices_output", indices_output),
        ("lens_output", lens_output),
    ]
    if not use_logical:
        tensors.append(("block_table", block_table))
    for name, tensor in tensors:
        if tensor is not None and tensor.device != device:
            raise ValueError(
                f"dspark_swa_indices tensors must live on the same device, got "
                f"{name} on {tensor.device} while query_start_loc is on {device}"
            )
    return device


def _validate_inputs(
    *,
    use_logical: bool,
    block_table: torch.Tensor | None,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    block_w: int,
    max_num_reqs: int | None,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
) -> int:
    """Validate shapes, dtypes, and scalar arguments; return active request count."""
    if int(num_speculative_tokens) < 0:
        raise ValueError(f"dspark_swa_indices num_speculative_tokens must be >= 0, got {num_speculative_tokens}")
    if int(window_size) < 0:
        raise ValueError(f"dspark_swa_indices window_size must be >= 0, got {window_size}")
    if int(block_w) <= 0:
        raise ValueError(f"dspark_swa_indices block_w must be positive, got {block_w}")
    if max_num_reqs is not None and int(max_num_reqs) < 0:
        raise ValueError(f"dspark_swa_indices max_num_reqs must be >= 0, got {max_num_reqs}")
    if query_start_loc.dtype not in _SUPPORTED_METADATA_DTYPES:
        raise ValueError(
            f"dspark_swa_indices query_start_loc must have int32 or int64 dtype, got {query_start_loc.dtype}"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 1:
        raise ValueError(
            f"dspark_swa_indices query_start_loc must be 1D with >= 1 element, got {query_start_loc.shape}"
        )
    num_reqs = int(query_start_loc.shape[0]) - 1
    if seq_lens.dtype not in _SUPPORTED_METADATA_DTYPES:
        raise ValueError(f"dspark_swa_indices seq_lens must have int32 or int64 dtype, got {seq_lens.dtype}")
    if seq_lens.ndim != 1 or seq_lens.shape[0] < num_reqs:
        raise ValueError(f"dspark_swa_indices seq_lens must cover num_reqs={num_reqs}, got shape {seq_lens.shape}")
    if not use_logical:
        if int(block_size) <= 0:
            raise ValueError(f"dspark_swa_indices block_size must be positive, got {block_size}")
        if block_table is None:
            raise ValueError("dspark_swa_indices requires block_table in physical output mode")
        if block_table.dtype not in _SUPPORTED_METADATA_DTYPES:
            raise ValueError(f"dspark_swa_indices block_table must have int32 or int64 dtype, got {block_table.dtype}")
        if block_table.ndim != 2 or block_table.shape[0] < num_reqs or block_table.shape[1] < 1:
            raise ValueError(
                f"dspark_swa_indices block_table must have shape [>={num_reqs}, >=1], got {tuple(block_table.shape)}"
            )
        if block_table.stride(1) != 1:
            raise ValueError(f"dspark_swa_indices block_table must have unit column stride, got {block_table.stride()}")
    return num_reqs


def _prepare_outputs(
    *,
    device: torch.device,
    use_logical: bool,
    num_reqs: int,
    num_rows: int,
    index_width: int,
    num_query_per_req: int,
    max_num_reqs: int | None,
    indices_output: torch.Tensor | None,
    lens_output: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Validate/allocate persistent or fresh outputs and compute cleanup bounds."""
    r_alloc = max(int(max_num_reqs), num_reqs) if max_num_reqs is not None else num_reqs
    if max_num_reqs is not None:
        num_rows_padded = max(num_rows, r_alloc * num_query_per_req)
    else:
        num_rows_padded = num_rows

    if indices_output is not None:
        if (
            indices_output.dtype != torch.int32
            or indices_output.ndim != 3
            or indices_output.shape[1] != 1
            or indices_output.shape[2] != index_width
            or not indices_output.is_contiguous()
        ):
            raise ValueError(
                "dspark_swa_indices indices_output must be contiguous int32 with shape "
                f"[>={num_rows}, 1, {index_width}], got {indices_output.dtype}{tuple(indices_output.shape)}"
            )
        if indices_output.shape[0] < num_rows:
            raise ValueError(
                "dspark_swa_indices indices_output has fewer rows than active tokens: "
                f"output={indices_output.shape[0]}, active={num_rows}"
            )
        if indices_output.shape[0] < num_rows_padded:
            if max_num_reqs is not None and r_alloc > num_reqs:
                warnings.warn(
                    "dspark_swa_indices indices_output has fewer rows than the capacity grid "
                    f"(output={indices_output.shape[0]}, capacity={r_alloc * num_query_per_req}): "
                    "the pad-row cleanup is disabled. Pass the FULL buffer "
                    "(indices_output=buffer, not buffer[:num_rows]) so captured "
                    "graphs never replay stale rows.",
                    UserWarning,
                    stacklevel=3,
                )
            num_rows_padded = indices_output.shape[0]
        out_slots = indices_output
    else:
        out_slots = torch.empty((num_rows, 1, index_width), dtype=torch.int32, device=device)
        num_rows_padded = min(num_rows_padded, num_rows)

    if lens_output is not None:
        valid_shape = (lens_output.ndim == 1) or (lens_output.ndim == 2 and lens_output.shape[1] == 1)
        if not valid_shape or lens_output.dtype not in (torch.int32, torch.int64) or not lens_output.is_contiguous():
            raise ValueError(
                "dspark_swa_indices lens_output must be contiguous int32/int64 with shape [>=N] or [>=N, 1], "
                f"got {lens_output.dtype}{tuple(lens_output.shape)}"
            )
        if lens_output.shape[0] < num_rows:
            raise ValueError(
                f"dspark_swa_indices lens_output has fewer rows than active tokens: "
                f"output={lens_output.shape[0]}, active={num_rows}"
            )
        out_lens = lens_output
        num_rows_padded = min(num_rows_padded, out_lens.shape[0])
    else:
        alloc_rows = max(num_rows, num_rows_padded) if indices_output is not None else num_rows
        lens_shape = (alloc_rows, 1) if use_logical else (alloc_rows,)
        lens_dtype = torch.int32 if use_logical else torch.int64
        out_lens = torch.empty(lens_shape, dtype=lens_dtype, device=device)

    num_slots = max(r_alloc, triton.cdiv(num_rows_padded, num_query_per_req)) if num_rows_padded > 0 else r_alloc
    return out_slots, out_lens, num_rows_padded, num_slots


def _resolve_launch_dims(
    *,
    window_size: int,
    num_speculative_tokens: int,
    index_width: int | None,
    num_decode_tokens: int | None,
    num_query_per_req: int | None,
    query_start_loc: torch.Tensor,
    num_reqs: int,
) -> tuple[int, int, int]:
    """Resolve aligned index_width, output row count, and rows per request."""
    if index_width is None:
        min_width = int(window_size) + int(num_speculative_tokens)
        index_width = ((min_width + _INDEX_WIDTH_ALIGNMENT - 1) // _INDEX_WIDTH_ALIGNMENT) * _INDEX_WIDTH_ALIGNMENT
    eff_width = int(index_width)
    if eff_width <= 0:
        raise ValueError(f"dspark_swa_indices index_width must be positive, got {eff_width}")
    num_rows = int(num_decode_tokens) if num_decode_tokens is not None else int(query_start_loc[num_reqs].item())
    if num_rows < 0:
        raise ValueError(f"dspark_swa_indices num_decode_tokens must be >= 0, got {num_rows}")
    nqp = int(num_speculative_tokens) + 1 if num_query_per_req is None else int(num_query_per_req)
    if nqp < 1:
        raise ValueError(f"dspark_swa_indices num_query_per_req must be >= 1, got {nqp}")
    return eff_width, num_rows, nqp


def build_dspark_swa_indices_triton(
    block_table: torch.Tensor | None,
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
    *,
    output_mode: str = OUTPUT_MODE_PHYSICAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified Triton-Ascend entry point for DSpark non-causal SWA indices.

    Supports ``output_mode="physical"`` (physical KV slot IDs via bounded
    block-table window loading for any positive block size and arbitrary
    block-table width) and ``output_mode="logical"`` (request-local sequence
    positions for SparseFlashMLA).
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for build_dspark_swa_indices_triton.")
    mode = normalize_dspark_swa_output_mode(output_mode)
    use_logical = mode == OUTPUT_MODE_LOGICAL
    device = _validate_device_residency(
        use_logical=use_logical,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        indices_output=indices_output,
        lens_output=lens_output,
    )
    num_reqs = _validate_inputs(
        use_logical=use_logical,
        block_table=block_table,
        num_speculative_tokens=num_speculative_tokens,
        window_size=window_size,
        block_size=block_size,
        block_w=block_w,
        max_num_reqs=max_num_reqs,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
    )
    query_start_loc, seq_lens = query_start_loc.contiguous(), seq_lens.contiguous()
    eff_width, num_rows, nqp = _resolve_launch_dims(
        window_size=window_size,
        num_speculative_tokens=num_speculative_tokens,
        index_width=index_width,
        num_decode_tokens=num_decode_tokens,
        num_query_per_req=num_query_per_req,
        query_start_loc=query_start_loc,
        num_reqs=num_reqs,
    )
    out_slots, out_lens, num_rows_padded, num_slots = _prepare_outputs(
        device=device,
        use_logical=use_logical,
        num_reqs=num_reqs,
        num_rows=num_rows,
        index_width=eff_width,
        num_query_per_req=nqp,
        max_num_reqs=max_num_reqs,
        indices_output=indices_output,
        lens_output=lens_output,
    )
    if num_rows == 0 and num_rows_padded == 0:
        return out_slots[:0], out_lens[:0]

    eff_block_w = _compute_eff_block_w(eff_width, block_w)
    eff_block_size = 1 if use_logical else int(block_size)
    bt_arg = query_start_loc if use_logical else block_table
    assert bt_arg is not None
    num_blocks = 1 if use_logical else int(bt_arg.shape[1])
    init_device_properties_triton()
    grid = min(get_vectorcore_num(), num_slots * triton.cdiv(eff_width, eff_block_w))
    _DSPARK_SWA_INDICES_KERNEL(
        block_table=bt_arg,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        out_slots=out_slots,
        out_lens=out_lens,
        num_blocks=num_blocks,
        num_reqs=num_reqs,
        num_rows_padded=num_rows_padded,
        num_query_per_req=nqp,
        num_slots=num_slots,
        WINDOW_SIZE=int(window_size),
        BLOCK_SIZE=eff_block_size,
        INDEX_W=eff_width,
        BLOCK_W=eff_block_w,
        BT_WIN_POW2=_compute_bt_win_pow2(eff_block_w, eff_block_size, use_logical),
        Q_POW2=_compute_q_pow2(num_speculative_tokens),
        USE_LOGICAL_INDICES=use_logical,
        grid_size=grid,
    )
    return out_slots[:num_rows], out_lens[:num_rows]


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
    """Compatibility alias forwarding to ``build_dspark_swa_indices_triton`` in physical mode."""
    return build_dspark_swa_indices_triton(
        block_table=block_table,
        num_speculative_tokens=num_speculative_tokens,
        window_size=window_size,
        block_size=block_size,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_decode_tokens=num_decode_tokens,
        index_width=index_width,
        indices_output=indices_output,
        lens_output=lens_output,
        max_num_reqs=max_num_reqs,
        num_query_per_req=num_query_per_req,
        block_w=block_w,
        output_mode=OUTPUT_MODE_PHYSICAL,
    )
