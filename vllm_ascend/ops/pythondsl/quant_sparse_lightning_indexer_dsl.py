"""Fused QSLI MXFP4 with metadata-selected LD/no-LD scheduling.

Both AIVs gather 256 tokens each for a 512-token Cube tile.
Packed K and E8M0 scales pass through a private GM ring because this DSL
cannot lower the E8M0 BDN transform directly from UB to L1. Two BF16 FIXP
copies feed the AIVs through ping-pong buffers. Each AIV reduces all heads
for its token half; their scores are joined before AIV0 runs local TopK.
LD partials are merged in the same kernel after the cross-core barrier.
No-LD records write directly to public outputs.
"""

from __future__ import annotations

import threading

import cannbotdsl
import torch
from cannbotdsl import MemLoc, Tensor, channel_rewind, dtypes
from cannbotdsl import select as dyn_select
from cannbotdsl._mlir import ir
from cannbotdsl._mlir.dialects import arith, ascvec
from cannbotdsl.ascir import extract_buffer
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.core.scalar_conversion import coerce_scalar_value
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_intra_arrive,
    cube_sync_intra_wait,
    global_sync_all,
    vec_sync_intra_arrive,
    vec_sync_intra_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    local_slice,
    make_layout,
    make_pointer,
    make_tensor,
    tile_view,
)

from vllm_ascend.ops.pythondsl.utils import INDEXER_MAX_WORKERS, get_indexer_worker_count


def _to_index(value):
    return coerce_scalar_value(value, ir.IndexType.get())


TILE_M = 32
TILE_N = 512
LOGICAL_D = 128
PACKED_D = LOGICAL_D // 2
N1 = 32
N2 = 1
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
CANDIDATE_BLOCKS_PER_TILE = TILE_N // CANDIDATE_BLOCK_SIZE
CANDIDATE_BLOCKS_PER_AIV = CANDIDATE_BLOCKS_PER_TILE // 2
STAGING_DEPTH = 4
# Cross-core sync ids 0..4 are reserved by the runtime/Channel lowering.
# Keep the explicit GM staging handshakes in the validated user range.
VECTOR0_READY_ID = 6
AIV1_SYNC_ID_OFFSET = 16
TOPK_TRUNK_LEN = 16384
MAX_CUBE_WORKERS = INDEXER_MAX_WORKERS


def _offset_view(tensor, offsets):
    """Rebase an ND input with the current DSL's zero-copy interval slicing."""
    if all(isinstance(offset, int) and offset == 0 for offset in offsets):
        return tensor
    return tensor[tuple(slice(offset, None) for offset in offsets)]


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _validate_pa_dim0_stride(tensor: torch.Tensor, name: str) -> int:
    """Return PA axis-0 stride after enforcing ASC's inner-contiguous scope."""
    strides = tuple(int(value) for value in tensor.stride())
    expected = [1]
    for dimension in reversed(tuple(int(value) for value in tensor.shape[1:])):
        expected.append(expected[-1] * dimension)
    expected_inner = tuple(reversed(expected[:-1]))
    if strides[1:] != expected_inner:
        raise ValueError(
            f"{name} only supports non-contiguous storage on axis 0; "
            f"expected inner strides {expected_inner}, got {strides[1:]}"
        )
    if strides[0] <= 0:
        raise ValueError(f"{name} axis-0 stride must be positive")
    return strides[0]


class QsliMergeTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(self, max_trunk_len: int, max_topk: int):
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + self.max_trunk_len, 128) * 128
        self.current_trunk = Channel(
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
        )
        self.merge_key = Buffer(MemLoc.UB, (1, self.merge_len_pad), dtypes.uint16)
        self.tmp_idx = Buffer(MemLoc.UB, (self.merge_len_pad,), dtypes.uint16)
        self.histogram = Buffer(MemLoc.UB, (256,), dtypes.uint16)
        self.high_target_carrier = Buffer(MemLoc.UB, (64,), dtypes.int32)
        self.history_key = Buffer(MemLoc.UB, (2, self.topk_pad), dtypes.uint16)
        self.next_history_key = Channel(
            MemLoc.UB,
            (1, self.topk_pad),
            dtypes.uint16,
            depth=1,
        )
        self.history_idx = Buffer(MemLoc.UB, (2, self.topk_pad), dtypes.int32)
        self.next_idx = Buffer(MemLoc.UB, (self.topk_pad,), dtypes.int32)


class QsliRawTopKSelector:
    """QSLI-local uint16 radix selector used by local TopK and LD merge."""

    def __init__(
        self,
        topk: int,
        workspace: QsliMergeTopKWorkspace,
    ):
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = TOPK_TRUNK_LEN
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(self.topk_pad + TOPK_TRUNK_LEN, 128) * 128
        self.workspace = workspace
        if (
            TOPK_TRUNK_LEN > workspace.max_trunk_len  # noqa: SIM300
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = Channel(MemLoc.UB, (1, self.topk), dtypes.int32, depth=1)
        self.output_value_bits_stage = Channel(MemLoc.UB, (1, self.topk), dtypes.uint16, depth=1)

    @jit
    def _find_indices(self, sort_key, valid_len, tmp_idx, histogram, high_target_carrier):
        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                h0 = rr.vhistogram_accumulate(h0, high, mask=mask8, bin=0)
                h1 = rr.vhistogram_accumulate(h1, high, mask=mask8, bin=1)
            rr.vstore(histogram, 0, h0, full16)
            rr.vstore(histogram, 128, h1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"),
                    dtypes.int32,
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            high_target8 = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b8"),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_count = zero16
            for base in range_constexpr(0, 256, 128):
                hv = rr.vload(histogram, base)
                idx = rr.varange(base, dtypes.uint16)
                at_prev = rr.veq(idx, prev, mask=full16)
                picked = rr.vselect(hv, zero16, cond_mask=at_prev)
                prev_count = rr.vadd(
                    prev_count,
                    rr.vreduce_max(picked, mask=full16),
                    mask=full16,
                )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            rr.vstore(tmp_idx, 0, next_k, lane0)

            l0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range((valid_len + 255) // 256, unroll=1):
                valid_count = valid_len - chunk * 256
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(sort_key, chunk * 256, width="b8")
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0 = rr.vhistogram_accumulate(l0, low, mask=eq_high, bin=0)
                l1 = rr.vhistogram_accumulate(l1, low, mask=eq_high, bin=1)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(idx, mask=rr.vgt(keys, kth, mask=mask16))
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(idx, mask=rr.veq(keys, kth, mask=mask16))
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")


class LdTopKSelector(QsliRawTopKSelector):
    def __init__(self, topk, workspace, splits):
        super().__init__(topk, workspace)
        self.ld_length = int(splits) * self.topk
        self.ld_indices = Buffer(MemLoc.UB, (1, self.ld_length), dtypes.int32)
        self.ld_bits = Buffer(MemLoc.UB, (1, self.ld_length), dtypes.uint16)

    @jit
    def merge_ld(
        self,
        partial_indices,
        partial_bits,
        gm_sparse_indices,
        gm_selected_value_bits,
        valid_length,
        output_offset=0,
        preloaded=False,
    ):
        if not preloaded:
            mem_copy(local_slice(self.ld_indices, (1, valid_length)), partial_indices[0:1, 0:valid_length])
            mem_copy(local_slice(self.ld_bits, (1, valid_length)), partial_bits[0:1, 0:valid_length])
        current_trunk = self.workspace.merge_key
        history_key = self.workspace.history_key  # noqa: F841
        history_idx = self.workspace.history_idx
        with vf(mode="raw"):
            mask16 = rr.update_mask(64, elem_bits=16)[0]
            mask32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(valid_length // 64, unroll=1):
                bits = rr.vload(self.ld_bits, chunk * 64)
                indices = rr.vload(self.ld_indices, chunk * 64)
                positive_key = rr.vbitwise_xor(bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16)
                negative_key = rr.vbitwise_xor(bits, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16)
                negative = rr.veqs(
                    rr.vbitwise_and(bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16),
                    0x8000,
                    mask=mask16,
                )
                keys = rr.vselect(negative_key, positive_key, cond_mask=negative)
                valid32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid16 = rr.vpack(rr.vreinterpret(valid32, dtypes.uint32), dtypes.uint16, part="lower")
                keys = rr.vselect(
                    keys, rr.vdups(0, dtypes.uint16, mask=mask16), cond_mask=rr.veqs(valid16, 1, mask=mask16)
                )
                rr.vstore(current_trunk, chunk * 64, keys, mask16)
            rr.vmem_bar("vst_vld")
        self._find_indices(
            current_trunk,
            valid_length,
            self.workspace.tmp_idx,
            self.workspace.histogram,
            self.workspace.high_target_carrier,
        )
        output_idx_stage = self.output_idx_stage
        out_value_bits = self.output_value_bits_stage
        output_valid_count = dtypes.int32(self.topk)
        final_idx = history_idx  # noqa: F841
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                positions = rr.vunpack(rr.vload(self.workspace.tmp_idx, chunk * 64), dtypes.uint32, part="lower")
                indices = rr.vgather(self.ld_indices, rr.vreinterpret(positions, dtypes.uint32), mask=mask32)
                positions16 = rr.vload(self.workspace.tmp_idx, chunk * 64)
                keys = rr.vgather(current_trunk, positions16, mask=mask16)
                value_bits = rr.vgather(self.ld_bits, positions16, mask=mask16)
                valid_rank32 = rr.vlts(  # noqa: F841
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(  # noqa: F841
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                key_valid16 = rr.vselect(
                    rr.vdups(1, dtypes.uint16, mask=mask16),
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.vges(keys, 1, mask=mask16),
                )
                key_valid32 = rr.vunpack(key_valid16, dtypes.uint32, part="lower")
                indices = rr.vselect(
                    indices, rr.vdups(-1, dtypes.int32, mask=mask32), cond_mask=rr.veqs(key_valid32, 1, mask=mask32)
                )
                valid_flags32 = rr.vselect(
                    rr.vdups(1, dtypes.int32, mask=mask32),
                    rr.vdups(0, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                valid_flags16 = rr.vpack(rr.vreinterpret(valid_flags32, dtypes.uint32), dtypes.uint16, part="lower")
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.vges(indices, 0, mask=mask32),
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=rr.veqs(valid_flags16, 1, mask=mask16),
                )
                indices = rr.vselect(
                    rr.vadds(indices, output_offset, mask=mask32),
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=rr.veqs(valid_flags32, 1, mask=mask32),
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, output_idx_stage)
        mem_copy(gm_selected_value_bits, out_value_bits)


class LdMergeStage:
    def __init__(self, topk, splits):
        self.topk = topk
        self.length = topk * splits
        self.workspace = QsliMergeTopKWorkspace(TOPK_TRUNK_LEN, topk)
        self.selector = LdTopKSelector(topk, self.workspace, splits)

    @jit
    def __call__(
        self,
        partial_indices: Tensor,
        partial_bits: Tensor,
        indices: Tensor,
        bits: Tensor,
        metadata: Tensor,
        rows,
        workers,
        cu_seqlens_q: Tensor,
        query_tile_rows,
        has_cu,
        output_offset_address=0,
    ):
        fd = dtypes.int64(36 + get_block_idx() * 2 + get_subblock_id())
        if metadata[fd, 0] != 0:
            batch_idx = dtypes.int64(metadata[fd, 1])
            begin = dtypes.int64(0)
            if has_cu:
                begin = dtypes.int64(cu_seqlens_q[batch_idx])
            row_base = begin + dtypes.int64(metadata[fd, 2]) * query_tile_rows
            workspace_begin = dtypes.int64(metadata[fd, 3])
            part_count = dtypes.int64(metadata[fd, 4])
            m_begin = dtypes.int64(metadata[fd, 5])
            m_end = m_begin + dtypes.int64(metadata[fd, 6])
            for m in dsl_range(m_begin, m_end, unroll=1):
                row = row_base + m
                output_offset = dtypes.int32(0)
                if output_offset_address != 0:
                    offsets = make_tensor(
                        make_pointer(dtypes.int32, dtypes.int64(output_offset_address), MemLoc.GM),
                        make_layout((rows, 1), stride=(1, 1)),
                    )
                    output_offset = offsets[row, 0]
                for first in dsl_range(0, part_count, self.length // self.topk - 1, unroll=1):
                    count = min(part_count - first, self.length // self.topk - 1)
                    carry = dtypes.int64(dyn_select(first > 0, 1, 0))
                    vec_sync_notify(PIPE.V, PIPE.MTE2, 5)
                    vec_sync_wait(PIPE.V, PIPE.MTE2, 5)
                    if first > 0:
                        mem_copy(
                            local_slice(self.selector.ld_indices, (1, self.topk)),
                            tile_view(indices, (1, self.topk), (row, 0)),
                        )
                        mem_copy(
                            local_slice(self.selector.ld_bits, (1, self.topk)),
                            tile_view(bits, (1, self.topk), (row, 0)),
                        )
                    for part in dsl_range(0, count, unroll=1):
                        source_row = (workspace_begin + first + part) * query_tile_rows + m
                        mem_copy(
                            tile_view(self.selector.ld_indices, (1, self.topk), (0, part + carry)),
                            tile_view(partial_indices, (1, self.topk), (source_row, 0)),
                        )
                        mem_copy(
                            tile_view(self.selector.ld_bits, (1, self.topk), (0, part + carry)),
                            tile_view(partial_bits, (1, self.topk), (source_row, 0)),
                        )
                    vec_sync_notify(PIPE.MTE2, PIPE.V, 5)
                    vec_sync_wait(PIPE.MTE2, PIPE.V, 5)
                    offset = dtypes.int32(dyn_select(first + count == part_count, output_offset, 0))
                    self.selector.merge_ld(
                        partial_indices,
                        partial_bits,
                        tile_view(indices, (1, self.topk), (row, 0)),
                        tile_view(bits, (1, self.topk), (row, 0)),
                        (count + carry) * self.topk,
                        offset,
                        True,
                    )
                    vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 5)
                    vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 5)


class QsliLocalTopKWorkspace:
    """Only radix scratch is needed for QSLI's single candidate trunk."""

    def __init__(self, max_trunk_len, max_topk):
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.merge_len_pad = ceil_div(ceil_div(self.max_topk, 256) * 256 + self.max_trunk_len, 128) * 128
        self.tmp_idx = Buffer(MemLoc.UB, (self.merge_len_pad,), dtypes.uint16)
        self.histogram = Buffer(MemLoc.UB, (256,), dtypes.uint16)
        self.high_target_carrier = Buffer(MemLoc.UB, (64,), dtypes.int32)


class QsliLocalTopKSelector(QsliRawTopKSelector):
    """QSLI scores fit one 16K trunk; select directly from the producer UB."""

    @jit
    def select_local(self, keys, total_tokens, output_valid_count, gm_indices, gm_bits, need_values):
        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            zero = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(total_tokens // 128, (self.topk + 127) // 128, unroll=1):
                rr.vstore(keys, chunk * 128, zero, full16)
            rr.vmem_bar("vst_vld")
        self._find_indices(
            keys,
            max(total_tokens, self.topk),
            self.workspace.tmp_idx,
            self.workspace.histogram,
            self.workspace.high_target_carrier,
        )
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                mask = rr.update_mask(self.topk - chunk * 64, elem_bits=32)[0]
                positions = rr.vunpack(rr.vload(self.workspace.tmp_idx, chunk * 64), dtypes.uint32, part="lower")
                valid = rr.vlts(rr.varange(chunk * 64, dtypes.uint32), output_valid_count, mask=mask)
                rr.vstore(
                    self.output_idx_stage,
                    chunk * 64,
                    rr.vselect(
                        rr.vreinterpret(positions, dtypes.int32), rr.vdups(-1, dtypes.int32, mask=mask), cond_mask=valid
                    ),
                    mask,
                )
            rr.vmem_bar("vst_vld")
        if need_values != 0:
            with vf(mode="raw"):
                for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                    mask = rr.update_mask(self.topk - chunk * 128, elem_bits=16)[0]
                    pos = rr.vload(self.workspace.tmp_idx, chunk * 128)
                    selected = rr.vgather(keys, pos, mask=mask)
                    sign = rr.vdups(0x8000, dtypes.uint16, mask=mask)
                    positive = rr.veq(rr.vbitwise_and(selected, sign, mask=mask), sign, mask=mask)
                    bits = rr.vselect(
                        rr.vbitwise_xor(selected, sign, mask=mask),
                        rr.vbitwise_xor(selected, rr.vdups(0xFFFF, dtypes.uint16, mask=mask), mask=mask),
                        cond_mask=positive,
                    )
                    valid = rr.vlts(rr.varange(chunk * 128, dtypes.uint16), output_valid_count, mask=mask)
                    rr.vstore(
                        self.output_value_bits_stage,
                        chunk * 128,
                        rr.vselect(bits, rr.vdups(0, dtypes.uint16, mask=mask), cond_mask=valid),
                        mask,
                    )
                rr.vmem_bar("vst_vld")
            mem_copy(gm_bits, self.output_value_bits_stage)


class QsliCube:
    """AIC-owned native MXFP4 QK resources."""

    def __init__(self):
        self.q_l1 = Channel(
            MemLoc.L1,
            (TILE_M, LOGICAL_D),
            dtypes.fp4x2_e2m1,
            depth=2,
            data_format="nz",
        )
        self.k_l1 = Channel(
            MemLoc.L1,
            (TILE_N, LOGICAL_D),
            dtypes.fp4x2_e2m1,
            depth=2,
            data_format="nz",
        )
        self.q_scale_l1 = Channel(
            MemLoc.L1,
            (TILE_M, 4),
            dtypes.float8_e8m0,
            depth=2,
            data_format="zn",
        )
        self.k_scale_l1 = Channel(
            MemLoc.L1,
            (4, TILE_N),
            dtypes.float8_e8m0,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (TILE_M, LOGICAL_D), dtypes.fp4x2_e2m1, depth=2)
        self.l0b = Channel(
            MemLoc.L0B,
            (TILE_N, LOGICAL_D),
            dtypes.fp4x2_e2m1,
            depth=2,
            data_format="nz",
        )
        self.l0c = Channel(MemLoc.L0C, (TILE_M, TILE_N), dtypes.float32, depth=2)
        self.nd2nz_fp4 = make_copy_engine(
            format_transform="nd2nz",
            dtype=dtypes.fp4x2_e2m1,
            pad_value=0.0,
        )
        self.scale_a = make_copy_engine(
            format_transform="mx_scale_and",
            dtype=dtypes.float8_e8m0,
            pad_value=0.0,
        )
        self.scale_b = make_copy_engine(
            format_transform="mx_scale_bdn",
            dtype=dtypes.float8_e8m0,
            pad_value=0.0,
        )
        self.fixpipe_qk = make_copy_engine(dtype=dtypes.bfloat16, dual_dst_ctl=0, sub_block_id=0, unit_flag_mode=3)

        self.fixpipe_qk1 = make_copy_engine(dtype=dtypes.bfloat16, dual_dst_ctl=0, sub_block_id=1, unit_flag_mode=3)

    def load_query(self, query, query_scale):
        mem_copy(self.q_l1, query, engine=self.nd2nz_fp4)
        mem_copy(self.q_scale_l1, query_scale, engine=self.scale_a)
        mem_copy(self.l0a, self.q_l1, mx_scale=self.q_scale_l1)

    def compute_qk(self, key_tile, key_scale_tile, qk_handoff, slot):
        mem_copy(self.k_l1, key_tile, engine=self.nd2nz_fp4)
        mem_copy(self.k_scale_l1, key_scale_tile, engine=self.scale_b)
        mem_copy(self.l0b, self.k_l1, mx_scale=self.k_scale_l1)
        dsl_matmul(self.l0c, self.l0a, self.l0b, init=True, unit_flag=3)
        cube_sync_intra_wait(PIPE.FIXPIPE, 4 + slot)
        cube_sync_intra_wait(PIPE.FIXPIPE, 20 + slot)
        destination = tile_view(qk_handoff, (TILE_M, TILE_N // 2), (slot, 0))
        mem_copy(
            destination,
            tile_view(self.l0c, (TILE_M, TILE_N // 2), (0, 0)),
            engine=self.fixpipe_qk,
        )
        mem_copy(
            destination,
            tile_view(self.l0c, (TILE_M, TILE_N // 2), (0, 1)),
            engine=self.fixpipe_qk1,
        )
        cube_sync_intra_arrive(PIPE.FIXPIPE, 14 + slot)
        cube_sync_intra_arrive(PIPE.FIXPIPE, 30 + slot)


def _split_page_pow2(value, reciprocal, divisor, shift, mask):
    page = rr.vshr(value, shift, mask=mask)
    offset = rr.vbitwise_and(value, rr.vdups(divisor - 1, dtypes.uint32, mask=mask), mask=mask)
    return page, offset


def _split_page_general(value, reciprocal, divisor, shift, mask):
    low, page = rr.vmull(value, reciprocal, dtypes.uint32, mask=mask)
    product = rr.vmuls(page, divisor, mask=mask)
    page = rr.vselect(
        rr.vsub(page, rr.vdups(1, dtypes.uint32, mask=mask), mask=mask),
        page,
        cond_mask=rr.vgt(product, value, mask=mask),
    )
    return page, rr.vsub(value, rr.vmuls(page, divisor, mask=mask), mask=mask)


class QsliVector0:
    """Each AIV gathers packed K/scales through two UB buffers into the GM ring."""

    def __init__(self, subblock_idx, pa_block_size):
        self.subblock_idx = subblock_idx
        self.pa_block_size = int(pa_block_size)
        self.page_divisor = int(pa_block_size) // CANDIDATE_BLOCK_SIZE
        self.page_shift = self.page_divisor.bit_length() - 1
        self.page_split = _split_page_pow2 if self.page_divisor & (self.page_divisor - 1) == 0 else _split_page_general
        self.page_reciprocal = ((1 << 32) + self.page_divisor - 1) // self.page_divisor
        self.candidate_ub = Buffer(MemLoc.UB, (1, CANDIDATE_CAPACITY), dtypes.int32)
        self.pair_swaps = Buffer(MemLoc.UB, (1, CANDIDATE_CAPACITY // 2), dtypes.uint32)
        self.physical_blocks = Buffer(MemLoc.UB, (1, CANDIDATE_CAPACITY), dtypes.uint32)
        self.pa_table_capacity = 2048  # UB cache capacity; larger tables use the scalar path.
        self.page_table_ub = Buffer(MemLoc.UB, (1, self.pa_table_capacity), dtypes.int32)
        self.packed_buffer = Buffer(MemLoc.UB, (2, CANDIDATE_BLOCKS_PER_AIV, 544), dtypes.uint8)

    @jit
    def preload_table(self, table, batch, has_block_table):
        if has_block_table and table.shape[1] <= self.pa_table_capacity:
            dst, dst_offset = extract_buffer(self.page_table_ub, access="write")
            src, src_offset = extract_buffer(table, access="read")
            ascvec.copy_gm2ub(
                dst,
                dst_offset,
                src,
                arith.addi(src_offset, _to_index(batch * table.stride[0])),
                dtypes.int32(1),
                dtypes.int32(table.shape[1] * 4),
                dtypes.int64(table.shape[1] * 4),
                dtypes.int32(table.shape[1] * 4),
                dtypes.int32(0),
                dtypes.int32(0),
            )
            vec_sync_notify(PIPE.MTE2, PIPE.V, 0)
            vec_sync_wait(PIPE.MTE2, PIPE.V, 0)

    @jit
    def prepare_addresses(
        self,
        candidates,
        row,
        valid_blocks,
        key_stride,
        candidate_begin,
        candidate_end,
        key_extent_bytes,
        table_length,
        has_block_table,
    ):
        if has_block_table and table_length <= self.pa_table_capacity:
            if key_extent_bytes <= 2147483648:
                self._prepare_addresses32(candidates, row, valid_blocks, key_stride, candidate_begin, candidate_end)
            else:
                self._prepare_addresses64(candidates, row, valid_blocks, key_stride, candidate_begin, candidate_end)
        else:
            # The scalar fallback preserves candidate order.
            with vf(mode="raw"):
                mask = rr.full_mask()
                zero = rr.vdups(0, dtypes.uint32, mask=mask)
                for chunk in dsl_range(CANDIDATE_CAPACITY // 128, unroll=1):
                    rr.vstore(self.pair_swaps, chunk * 64, zero, mask)

    @jit
    def _prepare_addresses32(self, candidates, row, valid_blocks, key_stride, candidate_begin, candidate_end):
        mem_copy(
            self.candidate_ub,
            tile_view(candidates, (1, 1, CANDIDATE_CAPACITY), (row, 0, 0)).view(1, CANDIDATE_CAPACITY),
        )
        vec_sync_notify(PIPE.MTE2, PIPE.V, 0)
        vec_sync_wait(PIPE.MTE2, PIPE.V, 0)
        with vf(mode="raw"):
            mask = rr.full_mask()
            all_lanes = mask
            lane_position = rr.varange(0, dtypes.uint32)
            lane = lane_position
            in_half = rr.vbitwise_and(lane_position, rr.vdups(31, dtypes.uint32, mask=mask), mask=mask)
            next_tile = rr.vmuls(rr.vshr(lane_position, 5, mask=mask), 64, mask=mask)
            position_offsets = rr.vadd(in_half, next_tile, mask=mask)
            reciprocal = rr.vdups(self.page_reciprocal, dtypes.uint32, mask=mask)
            stride_low = rr.vdups(dtypes.uint32(key_stride % 4294967296), dtypes.uint32, mask=mask)
            even_lane = rr.vbitwise_and(lane, rr.vdups(0xFFFFFFFE, dtypes.uint32, mask=mask), mask=mask)
            odd_lane = rr.vadd(even_lane, rr.vdups(1, dtypes.uint32, mask=mask), mask=mask)
            one = rr.vdups(1, dtypes.uint32, mask=mask)
            zero = rr.vdups(0, dtypes.uint32, mask=mask)
            for chunk in dsl_range(candidate_begin // 64, candidate_end // 64, 2, unroll=1):
                logical_position = rr.vadds(
                    position_offsets, dtypes.uint32(chunk * 64 + self.subblock_idx * 32), mask=mask
                )
                safe_position = rr.vmins(logical_position, dtypes.uint32(valid_blocks - 1), mask=mask)
                candidate = rr.vmaxs(rr.vgather(self.candidate_ub, safe_position, mask=mask), 0, mask=mask)
                unsigned_candidate = rr.vreinterpret(candidate, dtypes.uint32)
                page, offset = self.page_split(unsigned_candidate, reciprocal, self.page_divisor, self.page_shift, mask)
                physical = rr.vgather(self.page_table_ub, rr.vreinterpret(page, dtypes.uint32), mask=mask)
                physical = rr.vreinterpret(physical, dtypes.uint32)
                low = rr.vmul(physical, stride_low, mask=mask)
                offset_bytes = rr.vreinterpret(rr.vmuls(offset, 544, mask=mask), dtypes.uint32)
                low = rr.vadd(low, offset_bytes, mask=mask)
                high = zero  # noqa: F841
                even_low = rr.vgather_reg(low, even_lane)
                odd_low = rr.vgather_reg(low, odd_lane)
                low_gt = rr.vselect(one, zero, cond_mask=rr.vgt(even_low, odd_low, mask=all_lanes))
                swap = low_gt
                permutation = rr.vbitwise_xor(lane, swap, mask=all_lanes)
                low = rr.vgather_reg(low, permutation)
                rr.vstore(
                    self.pair_swaps,
                    chunk * 32,
                    rr.vmuls(swap, 4, mask=all_lanes),
                    rr.update_mask((candidate_end // 64 - chunk) * 32, elem_bits=32)[0],
                )
                rr.vstore(
                    self.physical_blocks,
                    chunk * 32,
                    low,
                    rr.update_mask((candidate_end // 64 - chunk) * 32, elem_bits=32)[0],
                )
        vec_sync_notify(PIPE.V, PIPE.S, 6)
        vec_sync_wait(PIPE.V, PIPE.S, 6)

    @jit
    def _prepare_addresses64(self, candidates, row, valid_blocks, key_stride, candidate_begin, candidate_end):
        mem_copy(
            self.candidate_ub,
            tile_view(candidates, (1, 1, CANDIDATE_CAPACITY), (row, 0, 0)).view(1, CANDIDATE_CAPACITY),
        )
        vec_sync_notify(PIPE.MTE2, PIPE.V, 0)
        vec_sync_wait(PIPE.MTE2, PIPE.V, 0)
        with vf(mode="raw"):
            mask = rr.full_mask()
            all_lanes = mask
            lane_position = rr.varange(0, dtypes.uint32)
            lane = lane_position
            in_half = rr.vbitwise_and(lane_position, rr.vdups(31, dtypes.uint32, mask=mask), mask=mask)
            next_tile = rr.vmuls(rr.vshr(lane_position, 5, mask=mask), 64, mask=mask)
            position_offsets = rr.vadd(in_half, next_tile, mask=mask)
            reciprocal = rr.vdups(self.page_reciprocal, dtypes.uint32, mask=mask)
            stride_low = rr.vdups(dtypes.uint32(key_stride % 4294967296), dtypes.uint32, mask=mask)
            stride_high = rr.vdups(dtypes.uint32(key_stride // 4294967296), dtypes.uint32, mask=mask)
            even_lane = rr.vbitwise_and(lane, rr.vdups(0xFFFFFFFE, dtypes.uint32, mask=mask), mask=mask)
            odd_lane = rr.vadd(even_lane, rr.vdups(1, dtypes.uint32, mask=mask), mask=mask)
            one = rr.vdups(1, dtypes.uint32, mask=mask)
            zero = rr.vdups(0, dtypes.uint32, mask=mask)
            for chunk in dsl_range(candidate_begin // 64, candidate_end // 64, 2, unroll=1):
                logical_position = rr.vadds(
                    position_offsets, dtypes.uint32(chunk * 64 + self.subblock_idx * 32), mask=mask
                )
                safe_position = rr.vmins(logical_position, dtypes.uint32(valid_blocks - 1), mask=mask)
                candidate = rr.vmaxs(rr.vgather(self.candidate_ub, safe_position, mask=mask), 0, mask=mask)
                unsigned_candidate = rr.vreinterpret(candidate, dtypes.uint32)
                page, offset = self.page_split(unsigned_candidate, reciprocal, self.page_divisor, self.page_shift, mask)
                physical = rr.vgather(self.page_table_ub, rr.vreinterpret(page, dtypes.uint32), mask=mask)
                physical = rr.vreinterpret(physical, dtypes.uint32)
                low, high = rr.vmull(physical, stride_low, dtypes.uint32, mask=mask)
                offset_bytes = rr.vreinterpret(rr.vmuls(offset, 544, mask=mask), dtypes.uint32)
                carry, low = rr.vaddco(low, offset_bytes, mask=mask)
                carry_out, high = rr.vaddc(high, rr.vmul(physical, stride_high, mask=mask), carry, mask=mask)
                even_low = rr.vgather_reg(low, even_lane)
                odd_low = rr.vgather_reg(low, odd_lane)
                even_high = rr.vgather_reg(high, even_lane)
                odd_high = rr.vgather_reg(high, odd_lane)
                low_gt = rr.vselect(one, zero, cond_mask=rr.vgt(even_low, odd_low, mask=all_lanes))
                high_gt = rr.vselect(one, zero, cond_mask=rr.vgt(even_high, odd_high, mask=all_lanes))
                swap = rr.vselect(low_gt, high_gt, cond_mask=rr.veq(even_high, odd_high, mask=all_lanes))
                permutation = rr.vbitwise_xor(lane, swap, mask=all_lanes)
                low = rr.vgather_reg(low, permutation)
                high = rr.vgather_reg(high, permutation)
                rr.vstore(
                    self.pair_swaps,
                    chunk * 32,
                    rr.vmuls(swap, 4, mask=all_lanes),
                    rr.update_mask((candidate_end // 64 - chunk) * 32, elem_bits=32)[0],
                )
                interleaved0, interleaved1 = rr.vinterleave(low, high)
                store_mask = rr.full_mask()
                rr.vstore(self.physical_blocks, chunk * 64, interleaved0, store_mask)
                rr.vstore(
                    self.physical_blocks,
                    chunk * 64 + 64,
                    interleaved1,
                    rr.update_mask((candidate_end // 64 - chunk - 1) * 64, elem_bits=32)[0],
                )
        vec_sync_notify(PIPE.V, PIPE.S, 6)
        vec_sync_wait(PIPE.V, PIPE.S, 6)

    @jit
    def _gather_half(
        self,
        half,
        key,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
        packed_ub,
        slot,
    ):
        if not has_block_table or block_table.shape[1] > self.pa_table_capacity:
            self._gather_half_scalar(
                half,
                key,
                candidate_block_indices,
                candidate_block_length,
                block_table,
                task_query_row,
                batch_idx,
                candidate_tile,
                staging_key,
                staging_scale,
                task_idx,
                has_block_table,
                packed_ub,
                slot,
            )
        elif key.shape[0] * key.stride[0] <= 2147483648:
            self._gather_half32(
                half,
                key,
                candidate_block_indices,
                candidate_block_length,
                block_table,
                task_query_row,
                batch_idx,
                candidate_tile,
                staging_key,
                staging_scale,
                task_idx,
                has_block_table,
                packed_ub,
                slot,
            )
        else:
            self._gather_half64(
                half,
                key,
                candidate_block_indices,
                candidate_block_length,
                block_table,
                task_query_row,
                batch_idx,
                candidate_tile,
                staging_key,
                staging_scale,
                task_idx,
                has_block_table,
                packed_ub,
                slot,
            )

    @jit
    def _gather_half_scalar(
        self,
        half,
        key,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
        packed_ub,
        slot,
    ):
        staging_row = (
            dtypes.int64(task_idx) * STAGING_DEPTH + dtypes.int64(candidate_tile) % STAGING_DEPTH
        ) * TILE_N + dtypes.int64(half) * (TILE_N // 2)
        first_candidate = (
            dtypes.int64(candidate_tile) * CANDIDATE_BLOCKS_PER_TILE + dtypes.int64(half) * CANDIDATE_BLOCKS_PER_AIV
        )
        valid_blocks = dtypes.int64(candidate_block_length)
        if slot == 0:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 3)
        packed_dst, packed_offset = extract_buffer(packed_ub, access="write")
        key_src, key_offset = extract_buffer(key, access="read")
        if self.batch_consistency != 0:
            for block_idx in range(CANDIDATE_BLOCKS_PER_AIV):
                pos = first_candidate + block_idx
                safe_pos = dtypes.int64(dyn_select(pos < valid_blocks, pos, valid_blocks - 1))
                block = dtypes.int64(candidate_block_indices[task_query_row, 0, safe_pos])
                page = block // (self.pa_block_size // CANDIDATE_BLOCK_SIZE)
                within_page = block % (self.pa_block_size // CANDIDATE_BLOCK_SIZE)
                physical = page
                if has_block_table:
                    physical = dtypes.int64(block_table[batch_idx, page])
                address = physical * key.stride[0] + within_page * 544
                mem_copy(
                    tile_view(packed_ub, (1, 544), (block_idx, 0)),
                    tile_view(key, (1, 1, 544), (physical, within_page, 0)).view(1, 544),
                )
        else:
            for pair in range(CANDIDATE_BLOCKS_PER_AIV // 2):
                addresses = []
                for item in range_constexpr(2):
                    block_idx = pair * 2 + item
                    pos = first_candidate + block_idx
                    safe_pos = dtypes.int64(dyn_select(pos < valid_blocks, pos, valid_blocks - 1))
                    block = dtypes.int64(candidate_block_indices[task_query_row, 0, safe_pos])
                    page = block // (self.pa_block_size // CANDIDATE_BLOCK_SIZE)
                    within_page = block % (self.pa_block_size // CANDIDATE_BLOCK_SIZE)
                    physical = page
                    if has_block_table:
                        physical = dtypes.int64(block_table[batch_idx, page])
                    address = physical * key.stride[0] + within_page * 544
                    addresses.append(address)
                first = addresses[0]
                gap = addresses[1] - first
                # Preserve candidate order: descending/duplicate addresses use single rows.
                pair_ok = dyn_select(gap >= 544, dyn_select(gap - 544 <= 549755813887, 1, 0), 0)
                if pair_ok != 0:
                    ascvec.copy_gm2ub(
                        packed_dst,
                        arith.addi(packed_offset, _to_index(pair * 2 * 544)),
                        key_src,
                        arith.addi(key_offset, _to_index(first)),
                        dtypes.int32(2),
                        dtypes.int32(544),
                        dtypes.int64(gap),
                        dtypes.int32(544),
                        dtypes.int32(0),
                        dtypes.int32(0),
                    )
                else:
                    for item in range_constexpr(2):
                        ascvec.copy_gm2ub(
                            packed_dst,
                            arith.addi(packed_offset, _to_index((pair * 2 + item) * 544)),
                            key_src,
                            arith.addi(key_offset, _to_index(addresses[item])),
                            dtypes.int32(1),
                            dtypes.int32(544),
                            dtypes.int64(544),
                            dtypes.int32(544),
                            dtypes.int32(0),
                            dtypes.int32(0),
                        )
        vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 2)
        vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 2)
        packed_src, packed_src_offset = extract_buffer(packed_ub, access="read")
        k_dst, k_offset = extract_buffer(staging_key, access="write")
        scale_dst, scale_offset = extract_buffer(staging_scale, access="write")
        # Gather all K rows and all scale rows into separate contiguous ND staging.
        ascvec.copy_ub2gm(
            k_dst,
            arith.addi(k_offset, _to_index(staging_row * PACKED_D)),
            packed_src,
            packed_src_offset,
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(512),
            dtypes.int64(512),
            dtypes.int32(544),
        )
        ascvec.copy_ub2gm(
            scale_dst,
            arith.addi(scale_offset, _to_index(staging_row * 4)),
            packed_src,
            arith.addi(packed_src_offset, _to_index(512)),
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(32),
            dtypes.int64(32),
            dtypes.int32(544),
        )
        if slot == 0:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 3)

    @jit
    def _gather_half32(
        self,
        half,
        key,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
        packed_ub,
        slot,
    ):
        staging_row = (
            dtypes.int64(task_idx) * STAGING_DEPTH + dtypes.int64(candidate_tile) % STAGING_DEPTH
        ) * TILE_N + dtypes.int64(half) * (TILE_N // 2)
        first_candidate = dtypes.int64(candidate_tile) * CANDIDATE_BLOCKS_PER_AIV
        valid_blocks = dtypes.int64(candidate_block_length)  # noqa: F841
        if slot == 0:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 3)
        packed_dst, packed_offset = extract_buffer(packed_ub, access="write")
        key_src, key_offset = extract_buffer(key, access="read")
        if self.batch_consistency != 0:
            for block_idx in range(CANDIDATE_BLOCKS_PER_AIV):
                pos = first_candidate + block_idx
                safe_pos = pos
                address = dtypes.int64(self.physical_blocks[0, safe_pos])
                physical = address // key.stride[0]
                within_page = (address % key.stride[0]) // 544
                mem_copy(
                    tile_view(packed_ub, (1, 544), (block_idx, 0)),
                    tile_view(key, (1, 1, 544), (physical, within_page, 0)).view(1, 544),
                )
        else:
            for pair in range(CANDIDATE_BLOCKS_PER_AIV // 2):
                packed_pair = self.physical_blocks.reinterpret(dtypes.uint64, (1, CANDIDATE_CAPACITY // 2))[
                    0, first_candidate // 2 + pair
                ]
                first = dtypes.int64(dtypes.uint32(packed_pair))
                second = dtypes.int64(packed_pair // 4294967296)
                gap = second - first
                if gap >= 544:
                    ascvec.copy_gm2ub(
                        packed_dst,
                        arith.addi(packed_offset, _to_index(pair * 2 * 544)),
                        key_src,
                        arith.addi(key_offset, _to_index(first)),
                        dtypes.int32(2),
                        dtypes.int32(544),
                        dtypes.int64(gap),
                        dtypes.int32(544),
                        dtypes.int32(0),
                        dtypes.int32(0),
                    )
                else:
                    ascvec.copy_gm2ub(
                        packed_dst,
                        arith.addi(packed_offset, _to_index(pair * 2 * 544)),
                        key_src,
                        arith.addi(key_offset, _to_index(first)),
                        dtypes.int32(1),
                        dtypes.int32(544),
                        dtypes.int64(544),
                        dtypes.int32(544),
                        dtypes.int32(0),
                        dtypes.int32(0),
                    )
                    ascvec.copy_gm2ub(
                        packed_dst,
                        arith.addi(packed_offset, _to_index((pair * 2 + 1) * 544)),
                        key_src,
                        arith.addi(key_offset, _to_index(second)),
                        dtypes.int32(1),
                        dtypes.int32(544),
                        dtypes.int64(544),
                        dtypes.int32(544),
                        dtypes.int32(0),
                        dtypes.int32(0),
                    )
        vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 2)
        vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 2)
        packed_src, packed_src_offset = extract_buffer(packed_ub, access="read")
        k_dst, k_offset = extract_buffer(staging_key, access="write")
        scale_dst, scale_offset = extract_buffer(staging_scale, access="write")
        # Gather all K rows and all scale rows into separate contiguous ND staging.
        ascvec.copy_ub2gm(
            k_dst,
            arith.addi(k_offset, _to_index(staging_row * PACKED_D)),
            packed_src,
            packed_src_offset,
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(512),
            dtypes.int64(512),
            dtypes.int32(544),
        )
        ascvec.copy_ub2gm(
            scale_dst,
            arith.addi(scale_offset, _to_index(staging_row * 4)),
            packed_src,
            arith.addi(packed_src_offset, _to_index(512)),
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(32),
            dtypes.int64(32),
            dtypes.int32(544),
        )
        if slot == 0:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 3)

    @jit
    def _gather_half64(
        self,
        half,
        key,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
        packed_ub,
        slot,
    ):
        staging_row = (
            dtypes.int64(task_idx) * STAGING_DEPTH + dtypes.int64(candidate_tile) % STAGING_DEPTH
        ) * TILE_N + dtypes.int64(half) * (TILE_N // 2)
        first_candidate = dtypes.int64(candidate_tile) * CANDIDATE_BLOCKS_PER_AIV
        valid_blocks = dtypes.int64(candidate_block_length)  # noqa: F841
        if slot == 0:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 3)
        packed_dst, packed_offset = extract_buffer(packed_ub, access="write")
        key_src, key_offset = extract_buffer(key, access="read")
        if self.batch_consistency != 0:
            for block_idx in range(CANDIDATE_BLOCKS_PER_AIV):
                pos = first_candidate + block_idx
                safe_pos = pos
                address = self.physical_blocks.reinterpret(dtypes.int64, (1, CANDIDATE_CAPACITY // 2))[0, safe_pos]
                physical = address // key.stride[0]
                within_page = (address % key.stride[0]) // 544
                mem_copy(
                    tile_view(packed_ub, (1, 544), (block_idx, 0)),
                    tile_view(key, (1, 1, 544), (physical, within_page, 0)).view(1, 544),
                )
        else:
            for pair in range(CANDIDATE_BLOCKS_PER_AIV // 2):
                addresses = []
                for item in range_constexpr(2):
                    block_idx = pair * 2 + item
                    pos = first_candidate + block_idx
                    safe_pos = pos
                    address = self.physical_blocks.reinterpret(dtypes.int64, (1, CANDIDATE_CAPACITY // 2))[0, safe_pos]
                    addresses.append(address)
                first = addresses[0]
                gap = addresses[1] - first
                # Preserve candidate order: descending/duplicate addresses use single rows.
                pair_ok = dyn_select(gap >= 544, dyn_select(gap - 544 <= 549755813887, 1, 0), 0)
                if pair_ok != 0:
                    ascvec.copy_gm2ub(
                        packed_dst,
                        arith.addi(packed_offset, _to_index(pair * 2 * 544)),
                        key_src,
                        arith.addi(key_offset, _to_index(first)),
                        dtypes.int32(2),
                        dtypes.int32(544),
                        dtypes.int64(gap),
                        dtypes.int32(544),
                        dtypes.int32(0),
                        dtypes.int32(0),
                    )
                else:
                    for item in range_constexpr(2):
                        ascvec.copy_gm2ub(
                            packed_dst,
                            arith.addi(packed_offset, _to_index((pair * 2 + item) * 544)),
                            key_src,
                            arith.addi(key_offset, _to_index(addresses[item])),
                            dtypes.int32(1),
                            dtypes.int32(544),
                            dtypes.int64(544),
                            dtypes.int32(544),
                            dtypes.int32(0),
                            dtypes.int32(0),
                        )
        vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 2)
        vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 2)
        packed_src, packed_src_offset = extract_buffer(packed_ub, access="read")
        k_dst, k_offset = extract_buffer(staging_key, access="write")
        scale_dst, scale_offset = extract_buffer(staging_scale, access="write")
        # Gather all K rows and all scale rows into separate contiguous ND staging.
        ascvec.copy_ub2gm(
            k_dst,
            arith.addi(k_offset, _to_index(staging_row * PACKED_D)),
            packed_src,
            packed_src_offset,
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(512),
            dtypes.int64(512),
            dtypes.int32(544),
        )
        ascvec.copy_ub2gm(
            scale_dst,
            arith.addi(scale_offset, _to_index(staging_row * 4)),
            packed_src,
            arith.addi(packed_src_offset, _to_index(512)),
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(32),
            dtypes.int64(32),
            dtypes.int32(544),
        )
        if slot == 0:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 3)

    @jit
    def gather(
        self,
        key,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
    ):
        slot = candidate_tile % 2
        packed = tile_view(self.packed_buffer, (1, CANDIDATE_BLOCKS_PER_AIV, 544), (slot, 0, 0)).view(
            CANDIDATE_BLOCKS_PER_AIV, 544
        )
        self._gather_half(
            self.subblock_idx,
            key,
            candidate_block_indices,
            candidate_block_length,
            block_table,
            task_query_row,
            batch_idx,
            candidate_tile,
            staging_key,
            staging_scale,
            task_idx,
            has_block_table,
            packed,
            slot,
        )
        vec_sync_intra_arrive(PIPE.MTE3, VECTOR0_READY_ID + candidate_tile % STAGING_DEPTH)

    @jit
    def begin_buffers(self):
        for slot in range_constexpr(2):
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2 + slot)

    @jit
    def drain_buffers(self):
        for slot in range_constexpr(2):
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2 + slot)

    @jit
    def wait_ready(self, tile):
        slot = tile % STAGING_DEPTH
        cube_sync_intra_wait(PIPE.MTE2, VECTOR0_READY_ID + slot)
        cube_sync_intra_wait(PIPE.MTE2, VECTOR0_READY_ID + slot + 16)

    @jit
    def await_slot(self, tile):
        vec_sync_intra_wait(PIPE.MTE2, 10 + tile % STAGING_DEPTH)

    @jit
    def signal_consumed(self, tile):
        slot = tile % STAGING_DEPTH
        cube_sync_intra_arrive(PIPE.MTE2, 10 + slot)
        cube_sync_intra_arrive(PIPE.MTE2, 26 + slot)


class QsliVector1:
    """Compute the one-query QSLI score row and its logical token payload."""

    def __init__(self, tokens):
        self.tokens = int(tokens)
        self.weight_ub = Channel(MemLoc.UB, (1, N1), dtypes.float32, depth=1)
        self.weight_bf16 = Buffer(MemLoc.UB, (N1,), dtypes.bfloat16)
        self.key_ub = Buffer(MemLoc.UB, (1, TOPK_TRUNK_LEN), dtypes.uint16)
        self.valid_counts = Buffer(MemLoc.UB, (CANDIDATE_CAPACITY,), dtypes.int32)
        self.valid_ub = Buffer(MemLoc.UB, (TILE_N,), dtypes.uint16)

    @jit
    def load_weights(self, weights, task_query_row):
        mem_copy(self.weight_ub, tile_view(weights, (1, N1), (task_query_row, 0)))
        with vf(mode="raw"):
            mask = rr.update_mask(N1, elem_bits=32)[0]
            value = rr.vcast(rr.vload(self.weight_ub, 0), dtypes.bfloat16, mask=mask, reg_layout=rr.RegLayout.ZERO)
            packed = rr.vpack(rr.vreinterpret_lanes(value, dtypes.uint32), dtypes.uint16, part="lower")
            rr.vstore(
                self.weight_bf16, 0, rr.vreinterpret(packed, dtypes.bfloat16), rr.update_mask(N1, elem_bits=16)[0]
            )
            rr.vmem_bar("vst_vld")

    @jit
    def compute(self, qk_handoff, candidate_tile, subblock_idx, workspace_tile, pair_swaps):
        group_count = dtypes.int32(N1)
        half_count = dtypes.int32(4)
        with vf(mode="raw"):
            full32 = rr.full_mask()
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            half16 = rr.update_mask(64, elem_bits=16)[0]
            one32 = rr.vdups(1, dtypes.uint32, mask=full32)
            zero32 = rr.vdups(0, dtypes.uint32, mask=full32)
            for half in dsl_range(half_count, unroll=1):
                lane = rr.varange(half * 64, dtypes.uint32)
                offset = rr.vbitwise_and(lane, rr.vdups(7, dtypes.uint32, mask=full32), mask=full32)
                counts = rr.vload_broadcast(
                    self.valid_counts,
                    candidate_tile * CANDIDATE_BLOCKS_PER_TILE
                    + subblock_idx * CANDIDATE_BLOCKS_PER_AIV
                    + dtypes.int64(half) * 8,
                    mode="datablock",
                )
                valid = rr.vselect(one32, zero32, cond_mask=rr.vlt(offset, counts, mask=full32))
                rr.vstore(self.valid_ub, half * 64, rr.vpack(valid, dtypes.uint16, part="lower"), half16)
            rr.vmem_bar("vst_vld")
            acc0 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            acc1 = rr.vdups(0.0, dtypes.bfloat16, mask=mask16)
            for group_block in dsl_range(group_count // 4, unroll=1):
                w0 = rr.vload_broadcast(self.weight_bf16, group_block * 4 + 0)
                q0_0 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 0) * (TILE_N // 2) + 0), 0.0, mask=mask16)
                q0_1 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 0) * (TILE_N // 2) + 128), 0.0, mask=mask16)
                w1 = rr.vload_broadcast(self.weight_bf16, group_block * 4 + 1)
                q1_0 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 1) * (TILE_N // 2) + 0), 0.0, mask=mask16)
                q1_1 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 1) * (TILE_N // 2) + 128), 0.0, mask=mask16)
                w2 = rr.vload_broadcast(self.weight_bf16, group_block * 4 + 2)
                q2_0 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 2) * (TILE_N // 2) + 0), 0.0, mask=mask16)
                q2_1 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 2) * (TILE_N // 2) + 128), 0.0, mask=mask16)
                w3 = rr.vload_broadcast(self.weight_bf16, group_block * 4 + 3)
                q3_0 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 3) * (TILE_N // 2) + 0), 0.0, mask=mask16)
                q3_1 = rr.vmaxs(rr.vload(qk_handoff, (group_block * 4 + 3) * (TILE_N // 2) + 128), 0.0, mask=mask16)
                acc0 = rr.vmadd(q0_0, w0, acc0, mask=mask16)
                acc1 = rr.vmadd(q0_1, w0, acc1, mask=mask16)
                acc0 = rr.vmadd(q1_0, w1, acc0, mask=mask16)
                acc1 = rr.vmadd(q1_1, w1, acc1, mask=mask16)
                acc0 = rr.vmadd(q2_0, w2, acc0, mask=mask16)
                acc1 = rr.vmadd(q2_1, w2, acc1, mask=mask16)
                acc0 = rr.vmadd(q3_0, w3, acc0, mask=mask16)
                acc1 = rr.vmadd(q3_1, w3, acc1, mask=mask16)
            word_lane = rr.varange(0, dtypes.uint32)
            local_block = rr.vshr(word_lane, 2, mask=full32)
            block0 = rr.vadds(local_block, dtypes.uint32(candidate_tile * CANDIDATE_BLOCKS_PER_AIV), mask=full32)
            block1 = rr.vadds(block0, 16, mask=full32)
            perm0 = rr.vbitwise_xor(word_lane, rr.vgather(pair_swaps, block0, mask=full32), mask=full32)
            perm1 = rr.vbitwise_xor(word_lane, rr.vgather(pair_swaps, block1, mask=full32), mask=full32)
            acc0 = rr.vreinterpret_lanes(
                rr.vgather_reg(rr.vreinterpret_lanes(acc0, dtypes.uint32), perm0), dtypes.bfloat16
            )
            acc1 = rr.vreinterpret_lanes(
                rr.vgather_reg(rr.vreinterpret_lanes(acc1, dtypes.uint32), perm1), dtypes.bfloat16
            )
            value0 = rr.vadd(acc0, rr.vdups(0.0, dtypes.bfloat16, mask=mask16), mask=mask16)
            bits0 = rr.vreinterpret(value0, dtypes.uint16)
            sign0 = rr.vbitwise_and(bits0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16)
            negative0 = rr.veqs(sign0, 0x8000, mask=mask16)
            positive_key0 = rr.vbitwise_xor(bits0, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16)
            negative_key0 = rr.vbitwise_xor(bits0, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16)
            keys0 = rr.vselect(negative_key0, positive_key0, cond_mask=negative0)
            keys0 = rr.vselect(
                keys0,
                rr.vdups(0, dtypes.uint16, mask=mask16),
                cond_mask=rr.veqs(rr.vload(self.valid_ub, 0), 1, mask=mask16),
            )
            rr.vstore(self.key_ub, workspace_tile * (TILE_N // 2) + 0, keys0, mask16)
            value1 = rr.vadd(acc1, rr.vdups(0.0, dtypes.bfloat16, mask=mask16), mask=mask16)
            bits1 = rr.vreinterpret(value1, dtypes.uint16)
            sign1 = rr.vbitwise_and(bits1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16)
            negative1 = rr.veqs(sign1, 0x8000, mask=mask16)
            positive_key1 = rr.vbitwise_xor(bits1, rr.vdups(0x8000, dtypes.uint16, mask=mask16), mask=mask16)
            negative_key1 = rr.vbitwise_xor(bits1, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16), mask=mask16)
            keys1 = rr.vselect(negative_key1, positive_key1, cond_mask=negative1)
            keys1 = rr.vselect(
                keys1,
                rr.vdups(0, dtypes.uint16, mask=mask16),
                cond_mask=rr.veqs(rr.vload(self.valid_ub, 128), 1, mask=mask16),
            )
            rr.vstore(self.key_ub, workspace_tile * (TILE_N // 2) + 128, keys1, mask16)
            rr.vmem_bar("vst_vld")

    @jit
    def join_scores(self, key_workspace, worker_idx, tiles, subblock_idx):
        vec_sync_notify(PIPE.V, PIPE.MTE3, 0)
        vec_sync_wait(PIPE.V, PIPE.MTE3, 0)
        src, src_offset = extract_buffer(self.key_ub, access="read")
        dst, dst_offset = extract_buffer(key_workspace, access="write")
        ascvec.copy_ub2gm(
            dst,
            arith.addi(dst_offset, _to_index(worker_idx * self.tokens + subblock_idx * (TILE_N // 2))),
            src,
            src_offset,
            dtypes.int32(tiles),
            dtypes.int32(TILE_N),
            dtypes.int64(TILE_N * 2),
            dtypes.int32(TILE_N),
        )
        vec_sync_intra_arrive(PIPE.MTE3, 6)
        cube_sync_intra_wait(PIPE.MTE2, 6)
        cube_sync_intra_wait(PIPE.MTE2, 22)
        cube_sync_intra_arrive(PIPE.MTE2, 7)
        cube_sync_intra_arrive(PIPE.MTE2, 23)
        vec_sync_intra_wait(PIPE.MTE2, 7)
        if subblock_idx == dtypes.int64(0):
            mem_copy(
                local_slice(self.key_ub, (1, self.tokens)), tile_view(key_workspace, (1, self.tokens), (worker_idx, 0))
            )
            vec_sync_notify(PIPE.MTE2, PIPE.V, 0)
            vec_sync_wait(PIPE.MTE2, PIPE.V, 0)


class QsliPositionMapper:
    """Map compact TopK positions back to sparse logical token indices."""

    def __init__(self, topk, candidate_capacity):
        self.topk = int(topk)
        self.candidate_capacity = int(candidate_capacity)
        self.count_ub = Buffer(MemLoc.UB, (8,), dtypes.int32)
        # Half-vector loads may fetch a full register. Reserve private padding
        # so the final AIV1 load does not read an adjacent UB allocation.
        self.candidate_ub = Channel(
            MemLoc.UB,
            (self.candidate_capacity,),
            dtypes.int32,
            depth=1,
            capacity=(ceil_div(self.candidate_capacity, 64) * 64 + 64,),
        )
        self.output_ub = Channel(
            MemLoc.UB,
            (1, self.topk),
            dtypes.int32,
            depth=1,
        )

    @jit
    def count_visible(
        self,
        candidate_block_indices,
        query_row,
        candidate_length,
        valid_s2,
        candidate_begin,
        candidate_end,
        valid_counts,
    ):
        mem_copy(
            self.candidate_ub,
            tile_view(_offset_view(candidate_block_indices, (query_row, 0, 0)), (self.candidate_capacity,), (0,)),
        )
        with vf(mode="raw"):
            mask = rr.update_mask(64, elem_bits=32)[0]
            total = rr.vdups(0, dtypes.int32, mask=mask)
            for chunk in dsl_range(candidate_begin // 64, (candidate_end + 63) // 64, unroll=1):
                ids = rr.vload(self.candidate_ub, chunk * 64)
                count = rr.vsub(
                    rr.vdups(valid_s2, dtypes.int32, mask=mask),
                    rr.vmuls(ids, CANDIDATE_BLOCK_SIZE, mask=mask),
                    mask=mask,
                )
                count = rr.vmins(rr.vmaxs(count, 0, mask=mask), CANDIDATE_BLOCK_SIZE, mask=mask)
                count = rr.vselect(
                    count,
                    rr.vdups(0, dtypes.int32, mask=mask),
                    cond_mask=rr.mask_and(
                        rr.vlts(rr.varange(chunk * 64, dtypes.int32), candidate_length, mask=mask),
                        rr.mask_and(
                            rr.vges(rr.varange(chunk * 64, dtypes.int32), candidate_begin, mask=mask),
                            rr.vlts(rr.varange(chunk * 64, dtypes.int32), candidate_end, mask=mask),
                            exec_mask=mask,
                        ),
                        exec_mask=mask,
                    ),
                )
                rr.vstore(valid_counts, chunk * 64, count, mask)
                total = rr.vadd(total, count, mask=mask)
            rr.vstore(self.count_ub, 0, rr.vreduce_sum(total, mask=mask), rr.update_mask(1, elem_bits=32)[0])
            rr.vmem_bar("vst_vld")
        # Publish now; consume only when local TopK needs the scalar count.
        vec_sync_notify(PIPE.V, PIPE.S, 0)

    @jit
    def read_count(self):
        vec_sync_wait(PIPE.V, PIPE.S, 0)
        return dtypes.int32(self.count_ub[0])

    @jit
    def load_own_counts(
        self,
        candidate_block_indices,
        query_row,
        candidate_length,
        valid_s2,
        candidate_begin,
        candidate_end,
        valid_counts,
    ):
        mem_copy(
            self.candidate_ub,
            tile_view(_offset_view(candidate_block_indices, (query_row, 0, 0)), (self.candidate_capacity,), (0,)),
        )
        with vf(mode="raw"):
            mask = rr.update_mask(32, elem_bits=32)[0]
            for chunk in dsl_range(candidate_begin // 64, (candidate_end + 63) // 64, unroll=1):
                ids = rr.vload(self.candidate_ub, chunk * 64 + 32)
                count = rr.vsub(
                    rr.vdups(valid_s2, dtypes.int32, mask=mask),
                    rr.vmuls(ids, CANDIDATE_BLOCK_SIZE, mask=mask),
                    mask=mask,
                )
                count = rr.vmins(rr.vmaxs(count, 0, mask=mask), CANDIDATE_BLOCK_SIZE, mask=mask)
                count = rr.vselect(
                    count,
                    rr.vdups(0, dtypes.int32, mask=mask),
                    cond_mask=rr.mask_and(
                        rr.vlts(rr.varange(chunk * 64 + 32, dtypes.int32), candidate_length, mask=mask),
                        rr.mask_and(
                            rr.vges(rr.varange(chunk * 64 + 32, dtypes.int32), candidate_begin, mask=mask),
                            rr.vlts(rr.varange(chunk * 64 + 32, dtypes.int32), candidate_end, mask=mask),
                            exec_mask=mask,
                        ),
                        exec_mask=mask,
                    ),
                )
                rr.vstore(valid_counts, chunk * 64 + 32, count, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def apply(
        self,
        selected_positions,
        candidate_block_indices,
        output_indices,
        query_row,
        valid_count,
        output_idx_offset,
        position_base,
    ):
        mem_copy(
            self.candidate_ub,
            tile_view(
                _offset_view(candidate_block_indices, (query_row, 0, 0)),
                (self.candidate_capacity,),
                (0,),
            ),
        )
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = self.topk - chunk * 64
                mask = rr.update_mask(count, elem_bits=32)[0]
                position = rr.vreinterpret(rr.vload(selected_positions, chunk * 64), dtypes.uint32)
                valid = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    valid_count,
                    mask=mask,
                )
                # Raw TopK emits -1 positions after valid_count.  Sanitize
                # those lanes before vgather; masking only the final store is
                # too late because uint32(-1) would address beyond candidate_ub.
                position = rr.vselect(
                    rr.vadds(position, position_base, mask=mask),
                    rr.vdups(0, dtypes.uint32, mask=mask),
                    cond_mask=valid,
                )
                block_position = rr.vshr(position, 3, mask=mask)
                block_id = rr.vgather(self.candidate_ub, block_position, mask=mask)
                logical_index = rr.vadd(
                    rr.vmuls(block_id, CANDIDATE_BLOCK_SIZE, mask=mask),
                    rr.vreinterpret(
                        rr.vbitwise_and(
                            position,
                            rr.vdups(
                                CANDIDATE_BLOCK_SIZE - 1,
                                dtypes.uint32,
                                mask=mask,
                            ),
                            mask=mask,
                        ),
                        dtypes.int32,
                    ),
                    mask=mask,
                )
                logical_index = rr.vadds(logical_index, output_idx_offset, mask=mask)
                rr.vstore(
                    self.output_ub,
                    chunk * 64,
                    rr.vselect(
                        logical_index,
                        rr.vdups(-1, dtypes.int32, mask=mask),
                        cond_mask=valid,
                    ),
                    mask,
                )
            rr.vmem_bar("vst_vld")

        mem_copy(output_indices, self.output_ub)


class QsliOutputWriter:
    """Initialize public rows inside the fused kernel."""

    def __init__(self, public_topk):
        self.public_topk = int(public_topk)
        self.index_ub = Channel(
            MemLoc.UB,
            (1, self.public_topk),
            dtypes.int32,
            depth=1,
        )
        self.value_ub = Channel(
            MemLoc.UB,
            (1, self.public_topk),
            dtypes.uint16,
            depth=1,
        )

    @jit
    def initialize(self, sparse_indices, sparse_value_bits, query_row):
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.public_topk, 64), unroll=1):
                count = self.public_topk - chunk * 64
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.mask_and(
                    rr.update_mask(count, elem_bits=16)[0],
                    rr.update_mask(64, elem_bits=16)[0],
                    exec_mask=rr.update_mask(64, elem_bits=16)[0],
                )
                rr.vstore(
                    self.index_ub,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    self.value_ub,
                    chunk * 64,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")

        mem_copy(
            tile_view(sparse_indices, (1, self.public_topk), (query_row, 0)),
            self.index_ub,
        )
        mem_copy(
            tile_view(sparse_value_bits, (1, self.public_topk), (query_row, 0)),
            self.value_ub,
        )


class QsliFusedKernel:
    def __init__(
        self,
        tokens,
        topk,
        candidate_capacity,
        mask_mode,
        cmp_ratio,
        has_cu,
        has_seqused_q,
        has_seqused_k,
        has_block_table,
        has_output_offset,
        pa_block_size,
        splits,
    ):
        self.splits = int(splits)
        self.tokens = int(tokens)
        self.token_tiles = self.tokens // TILE_N
        self.topk_count = int(topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_cu = bool(has_cu)
        self.has_seqused_q = bool(has_seqused_q)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_residual = self.mask_mode == 3 and self.cmp_ratio != 1
        self.has_block_table = bool(has_block_table)
        self.has_output_offset = bool(has_output_offset)
        # Explicit per-slot credits protect both BF16 destinations. The current
        # Channel split-N path does not support FP32-to-BF16 narrowing.
        self.qk_handoff = Buffer(MemLoc.UB, (TILE_M * 2, TILE_N // 2), dtypes.bfloat16)
        self.cube = QsliCube()
        self.vector0 = QsliVector0(get_subblock_id(), pa_block_size)
        self.vector1 = QsliVector1(tokens)
        self.output_writer = QsliOutputWriter(self.topk_count)
        self.topk_workspace = QsliLocalTopKWorkspace(TOPK_TRUNK_LEN, self.topk_count)
        self.topk = QsliLocalTopKSelector(self.topk_count, self.topk_workspace)
        self.position_mapper = QsliPositionMapper(self.topk_count, candidate_capacity)

    @jit
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        candidate_block_indices: Tensor,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        block_table: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_scale: Tensor,
        staging_scale_e8m0: Tensor,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
        metadata: Tensor,
        return_values,
        partial_idx_address,
        partial_bits_address,
        final_idx_address,
        final_bits_address,
        batch_consistency,
    ):
        self.vector0.batch_consistency = batch_consistency
        worker_idx = dtypes.int64(get_block_idx())
        subblock_idx = dtypes.int64(get_subblock_id())
        workspace_cursor = dtypes.int64(metadata[worker_idx, 7])
        if metadata[worker_idx, 0] != 0:
            first_b = dtypes.int64(metadata[worker_idx, 1])
            last_b = dtypes.int64(metadata[worker_idx, 4])
            for batch_idx in dsl_range(first_b, min(last_b + 1, batch_count), unroll=1):
                begin = dtypes.int64(0)
                end = dtypes.int64(query_rows)
                if self.has_cu:
                    begin = dtypes.int64(cu_seqlens_q[batch_idx])
                    end = dtypes.int64(cu_seqlens_q[batch_idx + 1])
                used_q = dtypes.int32(end - begin)
                if self.has_seqused_q:
                    used_q = seqused_q[batch_idx]
                actual = dtypes.int32(logical_k_capacity)
                if self.has_seqused_k:
                    actual = seqused_k[batch_idx]
                residual = dtypes.int32(0)
                if self.has_residual:
                    residual = cmp_residual_k[batch_idx]
                first_m = dtypes.int64(0)
                last_m = end - begin
                if batch_idx == first_b:
                    first_m = dtypes.int64(metadata[worker_idx, 2])
                if batch_idx == last_b:
                    last_m = dtypes.int64(metadata[worker_idx, 5]) + dtypes.int64(
                        dyn_select(metadata[worker_idx, 6] > 0, 1, 0)
                    )
                self.vector0.preload_table(block_table, batch_idx, self.has_block_table)
                prefetched_tiles = dtypes.int64(0)
                for m_idx in dsl_range(first_m, last_m, unroll=1):
                    query_row = begin + m_idx
                    candidate_length = dtypes.int32(candidate_block_length[query_row, 0])
                    tile_begin = dtypes.int64(0)
                    tile_end = (dtypes.int64(candidate_length) + TILE_N // CANDIDATE_BLOCK_SIZE - 1) // (
                        TILE_N // CANDIDATE_BLOCK_SIZE
                    )
                    if batch_idx == first_b and m_idx == first_m:
                        tile_begin = dtypes.int64(metadata[worker_idx, 3])
                    if batch_idx == last_b and m_idx == dtypes.int64(metadata[worker_idx, 5]):
                        tile_end = min(tile_end, dtypes.int64(metadata[worker_idx, 6]))
                    valid_s2 = actual
                    if self.mask_mode == 3:
                        valid_s2 = min(
                            actual,
                            max(
                                dtypes.int32(0),
                                (actual * self.cmp_ratio + residual - used_q + dtypes.int32(m_idx) + 1)
                                // self.cmp_ratio,
                            ),
                        )
                    if m_idx >= dtypes.int64(used_q):
                        valid_s2 = dtypes.int32(0)
                    full_tiles = (dtypes.int64(candidate_length) + TILE_N // CANDIDATE_BLOCK_SIZE - 1) // (
                        TILE_N // CANDIDATE_BLOCK_SIZE
                    )
                    row_ld = valid_s2 > 0 and (tile_begin > 0 or tile_end < full_tiles)
                    output_row = dtypes.int64(dyn_select(row_ld, workspace_cursor, query_row))
                    idx_address = dtypes.int64(dyn_select(row_ld, partial_idx_address, final_idx_address))
                    bits_address = dtypes.int64(dyn_select(row_ld, partial_bits_address, final_bits_address))
                    sparse_indices = make_tensor(
                        make_pointer(dtypes.int32, idx_address, MemLoc.GM),
                        make_layout((max(query_rows, 64), self.topk_count), stride=(self.topk_count, 1)),
                    )
                    sparse_value_bits = make_tensor(
                        make_pointer(dtypes.uint16, bits_address, MemLoc.GM),
                        make_layout((max(query_rows, 64), self.topk_count), stride=(self.topk_count, 1)),
                    )
                    if subblock_idx == dtypes.int64(0):
                        self.output_writer.initialize(sparse_indices, sparse_value_bits, output_row)
                    if row_ld:
                        workspace_cursor += 1
                    # Metadata boundaries directly index complete compute tiles.
                    candidate_begin = tile_begin * (TILE_N // CANDIDATE_BLOCK_SIZE)
                    candidate_end = tile_end * (TILE_N // CANDIDATE_BLOCK_SIZE)
                    if candidate_length > 0 and tile_end > tile_begin and valid_s2 > 0:
                        valid_count = dtypes.int32(0)
                        if valid_s2 > dtypes.int32(0):
                            if subblock_idx == dtypes.int64(0):
                                self.position_mapper.count_visible(
                                    candidate_block_indices,
                                    query_row,
                                    candidate_length,
                                    valid_s2,
                                    dtypes.int32(candidate_begin),
                                    dtypes.int32(candidate_end),
                                    self.vector1.valid_counts,
                                )
                            else:
                                self.position_mapper.load_own_counts(
                                    candidate_block_indices,
                                    query_row,
                                    candidate_length,
                                    valid_s2,
                                    dtypes.int32(candidate_begin),
                                    dtypes.int32(candidate_end),
                                    self.vector1.valid_counts,
                                )

                        # Keep the Cube/AIV control flow keyed only by sequence
                        # metadata, which is uniform across the fused core group.
                        # Candidate-derived valid_count remains an AIV TopK guard.
                        if valid_s2 > dtypes.int32(0):
                            self.cube.load_query(
                                tile_view(query, (N1, LOGICAL_D), (query_row, 0)),
                                tile_view(query_scale, (N1, 2, 2), (query_row, 0, 0)),
                            )
                            self.vector1.load_weights(weights, query_row)
                            if prefetched_tiles == 0:
                                self.vector0.prepare_addresses(
                                    candidate_block_indices,
                                    query_row,
                                    candidate_length,
                                    key.stride[0],
                                    candidate_begin,
                                    candidate_end,
                                    key.shape[0] * key.stride[0],
                                    block_table.shape[1],
                                    self.has_block_table,
                                )
                                self.vector0.begin_buffers()
                            for qk_slot in range_constexpr(2):
                                vec_sync_intra_arrive(PIPE.V, 4 + qk_slot)
                            for tick in dsl_range(tile_begin, tile_end + 1, unroll=1):
                                if tick < tile_end and tick >= tile_begin + prefetched_tiles:
                                    if tick >= tile_begin + STAGING_DEPTH:
                                        self.vector0.await_slot(tick)
                                    self.vector0.gather(
                                        key,
                                        candidate_block_indices,
                                        candidate_length,
                                        block_table,
                                        query_row,
                                        batch_idx,
                                        tick,
                                        staging_key,
                                        staging_scale,
                                        worker_idx,
                                        self.has_block_table,
                                    )
                                if tick > tile_begin:
                                    candidate_tile = tick - 1
                                    self.vector0.wait_ready(candidate_tile)
                                    staging_row = (
                                        worker_idx * STAGING_DEPTH + dtypes.int64(candidate_tile) % STAGING_DEPTH
                                    ) * TILE_N
                                    self.cube.compute_qk(
                                        tile_view(
                                            _offset_view(staging_key_fp4, (staging_row, 0)),
                                            (TILE_N, LOGICAL_D),
                                            (0, 0),
                                        ),
                                        tile_view(
                                            _offset_view(
                                                staging_scale_e8m0,
                                                (staging_row, 0, 0),
                                            ),
                                            (TILE_N, 2, 2),
                                            (0, 0, 0),
                                        ),
                                        self.qk_handoff,
                                        candidate_tile % 2,
                                    )
                                    self.vector0.signal_consumed(candidate_tile)
                                    vec_sync_intra_wait(PIPE.V, 14 + candidate_tile % 2)
                                    self.vector1.compute(
                                        tile_view(self.qk_handoff, (TILE_M, TILE_N // 2), (candidate_tile % 2, 0)),
                                        candidate_tile,
                                        subblock_idx,
                                        candidate_tile - tile_begin,
                                        self.vector0.pair_swaps,
                                    )
                                    vec_sync_intra_arrive(PIPE.V, 4 + candidate_tile % 2)

                            for final_tick in dsl_range(max(tile_begin, tile_end - STAGING_DEPTH), tile_end, unroll=1):
                                self.vector0.await_slot(final_tick)
                            self.vector0.drain_buffers()
                            for qk_slot in range_constexpr(2):
                                cube_sync_intra_wait(PIPE.FIXPIPE, 4 + qk_slot)
                                cube_sync_intra_wait(PIPE.FIXPIPE, 20 + qk_slot)
                            self.vector1.join_scores(key_workspace, worker_idx, tile_end - tile_begin, subblock_idx)
                            prefetched_tiles = dtypes.int64(0)
                            next_m = m_idx + 1
                            if next_m < last_m and next_m < dtypes.int64(used_q):
                                next_row = query_row + 1
                                next_length = dtypes.int32(candidate_block_length[next_row, 0])
                                next_end = (
                                    dtypes.int64(next_length) + CANDIDATE_BLOCKS_PER_TILE - 1
                                ) // CANDIDATE_BLOCKS_PER_TILE
                                if batch_idx == last_b and next_m == dtypes.int64(metadata[worker_idx, 5]):
                                    next_end = min(next_end, dtypes.int64(metadata[worker_idx, 6]))
                                next_visible = actual
                                if self.mask_mode == 3:
                                    next_visible = min(
                                        actual,
                                        max(
                                            dtypes.int32(0),
                                            (actual * self.cmp_ratio + residual - used_q + dtypes.int32(next_m) + 1)
                                            // self.cmp_ratio,
                                        ),
                                    )
                                if next_length > 0 and next_end > 0 and next_visible > 0:
                                    self.vector0.prepare_addresses(
                                        candidate_block_indices,
                                        next_row,
                                        next_length,
                                        key.stride[0],
                                        dtypes.int64(0),
                                        next_end * CANDIDATE_BLOCKS_PER_TILE,
                                        key.shape[0] * key.stride[0],
                                        block_table.shape[1],
                                        self.has_block_table,
                                    )
                                    self.vector0.begin_buffers()
                                    prefetched_tiles = min(dtypes.int64(STAGING_DEPTH), next_end)
                                    for future_tile in dsl_range(prefetched_tiles, unroll=1):
                                        self.vector0.gather(
                                            key,
                                            candidate_block_indices,
                                            next_length,
                                            block_table,
                                            next_row,
                                            batch_idx,
                                            future_tile,
                                            staging_key,
                                            staging_scale,
                                            worker_idx,
                                            self.has_block_table,
                                        )
                            if subblock_idx == dtypes.int64(0):
                                valid_count = self.position_mapper.read_count()
                                if valid_count > dtypes.int32(0):
                                    self.topk.select_local(
                                        self.vector1.key_ub,
                                        (tile_end - tile_begin) * TILE_N,
                                        valid_count,
                                        tile_view(
                                            selected_position_workspace,
                                            (1, self.topk_count),
                                            (worker_idx, 0),
                                        ),
                                        tile_view(
                                            sparse_value_bits,
                                            (1, self.topk_count),
                                            (output_row, 0),
                                        ),
                                        dtypes.int64(dyn_select(metadata[36, 0] != 0, 1, return_values)),
                                    )
                                    output_offset = dtypes.int32(0)
                                    if self.has_output_offset and not row_ld:
                                        output_offset = dtypes.int32(output_idx_offset[query_row, 0])
                                    self.position_mapper.apply(
                                        self.topk.output_idx_stage,
                                        candidate_block_indices,
                                        tile_view(
                                            sparse_indices,
                                            (1, self.topk_count),
                                            (output_row, 0),
                                        ),
                                        query_row,
                                        valid_count,
                                        output_offset,
                                        dtypes.uint32(tile_begin * TILE_N),
                                    )


_COMPILED_KERNEL = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _build_compiled_fused_runner(
    tokens,
    topk,
    candidate_capacity,
    mask_mode,
    cmp_ratio,
    has_cu,
    has_seqused_q,
    has_seqused_k,
    has_block_table,
    has_output_offset,
    pa_block_size,
    splits,
    auto_metadata,
):
    """Compile the dynamic TensorSpec contract using the standard DSL entry."""
    has_residual = int(mask_mode) == 3 and int(cmp_ratio) != 1

    @kernel
    def fused_body(
        query_address,
        key,
        weights: Tensor,
        query_scale: Tensor,
        candidate_block_indices,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        block_table: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_key,
        staging_key_fp4,
        staging_scale,
        staging_scale_e8m0,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices_address,
        sparse_value_address,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
        metadata: Tensor,
        final_sparse_indices_address,
        final_sparse_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
        batch_consistency,
    ):
        ld_enabled = metadata[36, 0] != 0
        final_sparse_indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(final_sparse_indices_address), MemLoc.GM),
            make_layout((query_rows, topk), stride=(topk, 1)),
        )
        final_sparse_bits = make_tensor(
            make_pointer(dtypes.uint16, dtypes.int64(final_sparse_bits_address), MemLoc.GM),
            make_layout((query_rows, topk), stride=(topk, 1)),
        )
        query = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(query_address), MemLoc.GM),
            make_layout((query_rows * N1, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        index_pointer = make_pointer(dtypes.int32, dtypes.int64(sparse_indices_address), MemLoc.GM)
        value_pointer = make_pointer(dtypes.uint16, dtypes.int64(sparse_value_address), MemLoc.GM)
        sparse_indices = make_tensor(index_pointer, make_layout((64, topk), stride=(topk, 1)))
        sparse_value_bits = make_tensor(value_pointer, make_layout((64, topk), stride=(topk, 1)))
        merge_indices = make_tensor(index_pointer, make_layout((64, topk), stride=(topk, 1)))
        merge_bits = make_tensor(value_pointer, make_layout((64, topk), stride=(topk, 1)))
        QsliFusedKernel(
            tokens,
            topk,
            candidate_capacity,
            mask_mode,
            cmp_ratio,
            has_cu,
            has_seqused_q,
            has_seqused_k,
            has_block_table,
            has_output_offset,
            pa_block_size,
            splits,
        )(
            query,
            key,
            weights,
            query_scale,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_q,
            seqused_q,
            block_table,
            seqused_k,
            cmp_residual_k,
            output_idx_offset,
            staging_key,
            staging_key_fp4,
            staging_scale,
            staging_scale_e8m0,
            key_workspace,
            selected_position_workspace,
            sparse_indices,
            sparse_value_bits,
            worker_count,
            query_rows,
            batch_count,
            logical_k_capacity,
            metadata,
            return_values,
            sparse_indices_address,
            sparse_value_address,
            final_sparse_indices_address,
            final_sparse_bits_address,
            batch_consistency,
        )
        if ld_enabled:
            global_sync_all()
            channel_rewind(reset_sync_id=True)
            LdMergeStage(topk, splits)(
                merge_indices,
                merge_bits,
                final_sparse_indices,
                final_sparse_bits,
                metadata,
                query_rows,
                worker_count * 2,
                cu_seqlens_q,
                1,
                has_cu,
                output_offset_address,
            )

    def run_fused(
        query_address,
        key,
        weights: Tensor,
        query_scale: Tensor,
        candidate_block_indices,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        block_table: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_key,
        staging_key_fp4,
        staging_scale,
        staging_scale_e8m0,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices_address,
        sparse_value_address,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
        metadata: Tensor,
        final_sparse_indices_address,
        final_sparse_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
        batch_consistency,
    ):
        fused_body[worker_count](
            query_address,
            key,
            weights,
            query_scale,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_q,
            seqused_q,
            block_table,
            seqused_k,
            cmp_residual_k,
            output_idx_offset,
            staging_key,
            staging_key_fp4,
            staging_scale,
            staging_scale_e8m0,
            key_workspace,
            selected_position_workspace,
            sparse_indices_address,
            sparse_value_address,
            worker_count,
            query_rows,
            batch_count,
            logical_k_capacity,
            metadata,
            final_sparse_indices_address,
            final_sparse_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
            batch_consistency,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    physical_blocks_dim = cannbotdsl.Dim("PA")
    batch_dim = cannbotdsl.Dim("B")
    worker_dim = cannbotdsl.Dim("WORKERS", min=1, max=MAX_CUBE_WORKERS)
    # Packed FP4 entry tensors need static host-carrier shapes in DSL 0.5.
    # Reserve bounded staging capacity; launch count and TopK workspaces stay dynamic.
    staging_workers = MAX_CUBE_WORKERS
    block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    key_stride_dim = cannbotdsl.Dim("K_S0")
    fake = cannbotdsl.TensorSpec
    dummy_spec = fake((query_rows_dim, N2), dtypes.int32)
    cu_spec = fake((batch_dim + 1,), dtypes.int32) if has_cu else dummy_spec
    seqused_q_spec = fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_spec
    block_table_spec = fake((batch_dim, block_table_width_dim), dtypes.int32) if has_block_table else dummy_spec
    seqused_k_spec = fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_spec
    residual_spec = fake((batch_dim,), dtypes.int32) if has_residual else dummy_spec
    offset_spec = fake((query_rows_dim, N2), dtypes.int32)
    compiled = jit(run_fused).compile(
        dtypes.int64,
        fake(
            (physical_blocks_dim, pa_block_size // CANDIDATE_BLOCK_SIZE, 544),
            dtypes.uint8,
            stride=(key_stride_dim, 544, 1),
        ),
        fake((query_rows_dim, N1), dtypes.float32),
        fake((query_rows_dim * N1, 2, 2), dtypes.float8_e8m0),
        fake((query_rows_dim, N2, candidate_capacity), dtypes.int32),
        fake((query_rows_dim, N2), dtypes.int32),
        cu_spec,
        seqused_q_spec,
        block_table_spec,
        seqused_k_spec,
        residual_spec,
        offset_spec,
        fake(
            (staging_workers * STAGING_DEPTH * TILE_N, PACKED_D),
            dtypes.uint8,
        ),
        fake(
            (staging_workers * STAGING_DEPTH * TILE_N, LOGICAL_D),
            dtypes.fp4x2_e2m1,
        ),
        fake((staging_workers * STAGING_DEPTH * TILE_N, 4), dtypes.uint8),
        fake(
            (staging_workers * STAGING_DEPTH * TILE_N, 2, 2),
            dtypes.float8_e8m0,
        ),
        fake((worker_dim, tokens), dtypes.uint16),
        fake((worker_dim, topk), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((128, 8), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int32,
    )
    return compiled


def clear_caches():
    """Close cached dynamic executables."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNEL.values():
            compiled.close()
        _COMPILED_KERNEL.clear()


def _get_compiled_fused_runner(*config):
    """Cache the dynamic QSLI executable; shape axes remain dynamic."""
    cache_key = tuple(config)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_compiled_fused_runner(*config)
            _COMPILED_KERNEL[cache_key] = compiled
        return compiled


def _batch_consistency_enabled():
    """Use the framework's batch-consistency level without changing global settings."""
    try:
        import torch_npu

        return int(torch_npu.npu._get_deterministic_level()) == 3
    except (AttributeError, ImportError):
        return False


def quant_sparse_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    cmp_residual_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    output_idx_offset: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    *,
    descale_k: torch.Tensor | None = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Run fused QSLI without a Host-built scheduling metadata table.

    ``candidate_block_length[t, 0]`` is the exact number of valid leading
    entries in ``candidate_block_indices[t, 0]`` and must be in ``[0, 2048]``.
    Every entry in that prefix is a logical eight-token block id valid for the
    row's batch; the suffix is ignored.  These value constraints are consumed
    as the device-side contract so Host does not copy candidates or lengths
    back to CPU.  ``metadata`` must supply reusable device-produced S2 shard records.
    """
    if layout_k == "TND":
        return _run_tnd(
            q,
            k,
            w,
            descale_q,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            output_idx_offset,
            metadata,
            descale_k=descale_k,
            topk=topk,
            candidate_block_size=candidate_block_size,
            quant_mode=quant_mode,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            layout_q=layout_q,
            layout_k=layout_k,
            return_value=return_value,
        )
    if metadata is None:
        raise ValueError("metadata is required; call quant_sparse_lightning_indexer_metadata before this operator")
    if int(quant_mode) != 1:
        raise ValueError("quant_sparse_lightning_indexer supports quant_mode=1 only")
    if layout_q != "TND" or layout_k != "PA_BBND":
        raise ValueError("current QSLI layouts must be TND and PA_BBND")
    if descale_k is not None:
        raise ValueError("descale_k is only supported with TND K; PA_BBND stores scales in packed k")
    if q.device.type != "npu" or k.device.type != "npu":
        raise ValueError("quant_sparse_lightning_indexer requires NPU q/k tensors")
    if q.dtype != torch.uint8 or k.dtype != torch.uint8:
        raise TypeError("q/k storage must be uint8 packed MXFP4")
    if q.ndim != 3 or tuple(q.shape[1:]) != (N1, PACKED_D):
        raise ValueError("q must have shape (T1,32,64)")
    if k.ndim != 3 or int(k.shape[2]) != 544:
        raise ValueError("k must have shape (block_num,block_size/8,544), each block K[512] followed by scale[32]")
    pa_block_size = int(k.shape[1]) * CANDIDATE_BLOCK_SIZE
    if pa_block_size <= 0 or pa_block_size > 1024 or pa_block_size % 16 != 0:
        raise ValueError("PA block_size must be a multiple of 16 in (0, 1024]")
    query_rows = int(q.shape[0])
    physical_k_blocks = int(k.shape[0])
    if query_rows <= 0 or physical_k_blocks <= 0:
        raise ValueError("T1 and the physical K block count must be positive")
    _validate_pa_dim0_stride(k, "k")

    if tuple(w.shape) != (query_rows, N1) or w.dtype != torch.float32:
        raise ValueError("w must be FP32 with shape (T1,32)")
    if w.device.type != "npu":
        raise ValueError("w must be an NPU tensor")
    expected_q_scale = (query_rows, N1, LOGICAL_D // 64, 2)
    if descale_q is None or tuple(descale_q.shape) != expected_q_scale:
        raise ValueError("descale_q is required with shape (T1,32,D/64,2)")
    scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
    if descale_q.dtype not in scale_dtypes or descale_q.device.type != "npu":
        raise TypeError("descale_q must be an NPU E8M0/uint8 tensor")

    if int(candidate_block_size) != CANDIDATE_BLOCK_SIZE:
        raise ValueError("candidate_block_size must be 8")
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("topk must be in [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("cmp_ratio must be in [1,128]")
    if int(max_seqlen_q) < -1:
        raise ValueError("max_seqlen_q must be -1 or nonnegative")

    if (
        candidate_block_indices.ndim != 3
        or tuple(candidate_block_indices.shape[:2]) != (query_rows, N2)
        or int(candidate_block_indices.shape[2]) != CANDIDATE_CAPACITY
    ):
        raise ValueError("candidate_block_indices must have shape (T1,1,2048)")
    if candidate_block_indices.dtype != torch.int32 or candidate_block_indices.device.type != "npu":
        raise TypeError("candidate_block_indices must be an NPU int32 tensor")
    if (
        candidate_block_length.dtype != torch.int32
        or candidate_block_length.device.type != "npu"
        or int(candidate_block_length.numel()) != query_rows * N2
    ):
        raise ValueError("candidate_block_length must be NPU int32 with T1*N2 entries")
    candidate_length_2d = candidate_block_length.reshape(query_rows, N2)

    if cu_seqlens_q is None:
        batch = 1
    else:
        if (
            cu_seqlens_q.dtype != torch.int32
            or cu_seqlens_q.device.type != "npu"
            or cu_seqlens_q.ndim != 1
            or int(cu_seqlens_q.numel()) < 2
        ):
            raise ValueError("cu_seqlens_q must be NPU int32 with shape (B+1,)")
        batch = int(cu_seqlens_q.numel()) - 1
    if seqused_q is not None and (
        tuple(seqused_q.shape) != (batch,) or seqused_q.dtype != torch.int32 or seqused_q.device.type != "npu"
    ):
        raise ValueError("seqused_q must be NPU int32 with shape (B,)")

    if block_table is None:
        if batch != 1:
            raise ValueError("block_table is required when batch is greater than one")
        logical_k_capacity = physical_k_blocks * pa_block_size
    else:
        if (
            block_table.ndim != 2
            or int(block_table.shape[0]) != batch
            or block_table.dtype != torch.int32
            or block_table.device.type != "npu"
        ):
            raise ValueError("block_table must be NPU int32 with shape (B,max_blocks)")
        logical_k_capacity = int(block_table.shape[1]) * pa_block_size
    if seqused_k is not None and (
        tuple(seqused_k.shape) != (batch,) or seqused_k.dtype != torch.int32 or seqused_k.device.type != "npu"
    ):
        raise ValueError("seqused_k must be NPU int32 with shape (B,)")

    needs_residual = int(mask_mode) == 3 and int(cmp_ratio) != 1
    if needs_residual:
        if cmp_residual_k is None or (
            tuple(cmp_residual_k.shape) != (batch,)
            or cmp_residual_k.dtype != torch.int32
            or cmp_residual_k.device.type != "npu"
        ):
            raise ValueError("cmp_residual_k must be NPU int32 with shape (B,) for causal compression")
    elif cmp_residual_k is not None:
        raise ValueError("cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1")
    if output_idx_offset is not None and (
        tuple(output_idx_offset.shape) != (query_rows, N2)
        or output_idx_offset.dtype != torch.int32
        or output_idx_offset.device.type != "npu"
    ):
        raise ValueError("output_idx_offset must be NPU int32 with shape (T1,1)")

    # Static capacity supplies the rolled loop bound; dynamic lengths and
    # candidate prefixes are consumed in the kernel and never copied to CPU.
    # Compile the candidate-capacity workspace once.  ``logical_k_capacity``
    # remains a runtime scalar used by masking and position mapping, so S2 and
    # PA/block-table shapes no longer create shape-specialized binaries.
    full_tokens = CANDIDATE_CAPACITY * CANDIDATE_BLOCK_SIZE
    # Streaming-merge workspace capacity; actual shard boundaries come from metadata.
    split_count = max(1, min(8, 16384 // int(topk)))
    auto_metadata = False
    worker_count = get_indexer_worker_count(q.device)
    if metadata.dtype != torch.int32 or metadata.ndim != 1 or metadata.numel() != 1024 or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous int32 [1024] in ASC LI/LD boundary format")
    if metadata.device != q.device:
        raise ValueError("metadata must be on the same device as q")
    metadata = metadata.view(128, 8)
    tokens = full_tokens
    worker_count = get_indexer_worker_count(q.device)

    sparse_indices = torch.empty(
        (64, N2, int(topk)),
        dtype=torch.int32,
        device=q.device,
    )
    sparse_values = torch.empty(
        (64, N2, int(topk)),
        dtype=torch.bfloat16,
        device=w.device,
    )
    staging_key = torch.empty(
        (MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N, PACKED_D),
        dtype=q.dtype,
        device=q.device,
    )
    staging_scale = torch.empty(
        (MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N, 4),
        dtype=q.dtype,
        device=q.device,
    )
    key_workspace = torch.empty((worker_count, tokens), dtype=torch.uint16, device=q.device)
    selected_position_workspace = torch.empty(
        (worker_count, int(topk)),
        dtype=torch.int32,
        device=q.device,
    )

    # Optional inputs use an existing int32 tensor as an ABI placeholder.
    # Their constexpr flags remove every dummy access from generated device code.
    int32_dummy = candidate_length_2d
    cu_npu = cu_seqlens_q if cu_seqlens_q is not None else int32_dummy
    seqused_q_npu = seqused_q if seqused_q is not None else int32_dummy
    block_table_npu = block_table if block_table is not None else int32_dummy
    seqused_k_npu = seqused_k if seqused_k is not None else int32_dummy
    residual_npu = cmp_residual_k if cmp_residual_k is not None else int32_dummy
    offset_npu = output_idx_offset if output_idx_offset is not None else int32_dummy

    query_scale_bytes = descale_q.view(torch.uint8)
    final_sparse_indices = torch.empty((query_rows, int(topk)), dtype=torch.int32, device=q.device)
    final_sparse_values = torch.empty((query_rows, int(topk)), dtype=torch.bfloat16, device=q.device)
    _get_compiled_fused_runner(
        tokens,
        int(topk),
        CANDIDATE_CAPACITY,
        int(mask_mode),
        int(cmp_ratio),
        cu_seqlens_q is not None,
        seqused_q is not None,
        seqused_k is not None,
        block_table is not None,
        output_idx_offset is not None,
        pa_block_size,
        split_count,
        auto_metadata,
    )(
        q.data_ptr(),
        k,
        w,
        query_scale_bytes.reshape(query_rows * N1, 2, 2).view(torch.float8_e8m0fnu),
        candidate_block_indices,
        candidate_length_2d,
        cu_npu,
        seqused_q_npu,
        block_table_npu,
        seqused_k_npu,
        residual_npu,
        offset_npu,
        staging_key,
        staging_key.view(torch.int8),
        staging_scale,
        staging_scale.reshape(-1, 2, 2).view(torch.float8_e8m0fnu),
        key_workspace,
        selected_position_workspace,
        sparse_indices.data_ptr(),
        sparse_values.data_ptr(),
        worker_count,
        query_rows,
        batch,
        logical_k_capacity,
        metadata,
        final_sparse_indices.data_ptr(),
        final_sparse_values.view(torch.uint16).data_ptr(),
        int(return_value),
        min(64, query_rows),
        0 if output_idx_offset is None else output_idx_offset.data_ptr(),
        int(_batch_consistency_enabled()),
    )
    sparse_indices, sparse_values = final_sparse_indices, final_sparse_values
    sparse_indices = sparse_indices.view(query_rows, N2, int(topk))
    sparse_values = sparse_values.view(query_rows, N2, int(topk))
    return (
        sparse_indices,
        sparse_values if return_value else sparse_values.reshape(-1)[:0],
    )


__all__ = ["quant_sparse_lightning_indexer"]


class QsliTndGather(QsliVector0):
    """Use plain DMA for full contiguous blocks, NDDMA for tails/strides."""

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.packed_buffer = Buffer(MemLoc.UB, (2, CANDIDATE_BLOCKS_PER_AIV, 544), dtypes.uint8)
        self.pair_swaps = Buffer(MemLoc.UB, (1, CANDIDATE_CAPACITY // 2), dtypes.uint32)

    @jit
    def prepare_identity(self):
        with vf(mode="raw"):
            mask = rr.full_mask()
            zero = rr.vdups(0, dtypes.uint32, mask=mask)
            for chunk in dsl_range(CANDIDATE_CAPACITY // 128, unroll=1):
                rr.vstore(self.pair_swaps, chunk * 64, zero, mask)

    @jit
    def _copy_block(self, packed_ub, block_idx, key, key_scale, source_row, count):
        dst, dst_offset = extract_buffer(packed_ub, access="write")
        key_src, key_offset = extract_buffer(key, access="read")
        scale_src, scale_offset = extract_buffer(key_scale, access="read")
        # NDDMA loop order is innermost first. Padding is on token loop 1.
        if count == 8 and key.stride[0] == 64:
            ascvec.copy_gm2ub(
                dst,
                arith.addi(dst_offset, _to_index(block_idx * 544)),
                key_src,
                arith.addi(key_offset, _to_index(source_row * key.stride[0])),
                dtypes.int32(1),
                dtypes.int32(512),
                dtypes.int64(512),
                dtypes.int32(512),
                dtypes.int32(0),
                dtypes.int32(0),
            )
        else:
            ascvec.copy_gm2ub_nddma(
                dst,
                arith.addi(dst_offset, _to_index(block_idx * 544)),
                key_src,
                arith.addi(key_offset, _to_index(source_row * key.stride[0])),
                dtypes.int32(64),
                dtypes.int32(count),
                dtypes.int32(1),
                dtypes.int32(1),
                dtypes.int32(1),
                dtypes.int64(1),
                dtypes.int64(key.stride[0]),
                dtypes.int64(0),
                dtypes.int64(0),
                dtypes.int64(0),
                dtypes.int64(1),
                dtypes.int64(64),
                dtypes.int64(0),
                dtypes.int64(0),
                dtypes.int64(0),
                [0, 0, 0, 0, 0],
                [0, 8 - count, 0, 0, 0],
                nearest=True,
            )
        if count == 8 and key_scale.stride[0] == 4:
            ascvec.copy_gm2ub(
                dst,
                arith.addi(dst_offset, _to_index(block_idx * 544 + 512)),
                scale_src,
                arith.addi(scale_offset, _to_index(source_row * key_scale.stride[0])),
                dtypes.int32(1),
                dtypes.int32(32),
                dtypes.int64(32),
                dtypes.int32(32),
                dtypes.int32(0),
                dtypes.int32(0),
            )
        else:
            ascvec.copy_gm2ub_nddma(
                dst,
                arith.addi(dst_offset, _to_index(block_idx * 544 + 512)),
                scale_src,
                arith.addi(scale_offset, _to_index(source_row * key_scale.stride[0])),
                dtypes.int32(4),
                dtypes.int32(count),
                dtypes.int32(1),
                dtypes.int32(1),
                dtypes.int32(1),
                dtypes.int64(1),
                dtypes.int64(key_scale.stride[0]),
                dtypes.int64(0),
                dtypes.int64(0),
                dtypes.int64(0),
                dtypes.int64(1),
                dtypes.int64(4),
                dtypes.int64(0),
                dtypes.int64(0),
                dtypes.int64(0),
                [0, 0, 0, 0, 0],
                [0, 8 - count, 0, 0, 0],
                nearest=True,
            )

    @jit
    def _gather_half_tnd(
        self,
        key,
        key_scale,
        candidates,
        candidate_length,
        cu_k,
        query_row,
        batch_idx,
        tile,
        staging_key,
        staging_scale,
        worker,
        packed_ub,
        slot,
    ):
        staging_row = (
            dtypes.int64(worker) * STAGING_DEPTH + dtypes.int64(tile) % STAGING_DEPTH
        ) * TILE_N + dtypes.int64(self.subblock_idx) * (TILE_N // 2)
        first_candidate = (
            dtypes.int64(tile) * CANDIDATE_BLOCKS_PER_TILE + dtypes.int64(self.subblock_idx) * CANDIDATE_BLOCKS_PER_AIV
        )
        base = dtypes.int64(cu_k[batch_idx])
        length = dtypes.int64(cu_k[batch_idx + 1]) - base
        if slot == 0:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_wait(PIPE.MTE3, PIPE.MTE2, 3)
        for block_idx in range(CANDIDATE_BLOCKS_PER_AIV):
            pos = first_candidate + block_idx
            safe_pos = dtypes.int64(
                dyn_select(pos < dtypes.int64(candidate_length), pos, dtypes.int64(candidate_length) - 1)
            )
            block = dtypes.int64(candidates[query_row, 0, safe_pos])
            start = block * CANDIDATE_BLOCK_SIZE
            count = min(dtypes.int64(8), length - start)
            if count == 8:
                self._copy_block(packed_ub, block_idx, key, key_scale, base + start, 8)
            else:
                # Padding attributes are static; only the matching branch runs.
                for tail in range_constexpr(1, 8):
                    if count == tail:
                        self._copy_block(packed_ub, block_idx, key, key_scale, base + start, tail)
        vec_sync_notify(PIPE.MTE2, PIPE.MTE3, 2)
        vec_sync_wait(PIPE.MTE2, PIPE.MTE3, 2)
        src, src_offset = extract_buffer(packed_ub, access="read")
        k_dst, k_offset = extract_buffer(staging_key, access="write")
        s_dst, s_offset = extract_buffer(staging_scale, access="write")
        ascvec.copy_ub2gm(
            k_dst,
            arith.addi(k_offset, _to_index(staging_row * PACKED_D)),
            src,
            src_offset,
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(512),
            dtypes.int64(512),
            dtypes.int32(544),
        )
        ascvec.copy_ub2gm(
            s_dst,
            arith.addi(s_offset, _to_index(staging_row * 4)),
            src,
            arith.addi(src_offset, _to_index(512)),
            dtypes.int32(CANDIDATE_BLOCKS_PER_AIV),
            dtypes.int32(32),
            dtypes.int64(32),
            dtypes.int32(544),
        )
        if slot == 0:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 2)
        else:
            vec_sync_notify(PIPE.MTE3, PIPE.MTE2, 3)

    @jit
    def gather(
        self,
        key,
        key_scale,
        candidate_block_indices,
        candidate_block_length,
        cu_seqlens_k,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_cu_k,
    ):
        slot = candidate_tile % 2
        packed = tile_view(self.packed_buffer, (1, CANDIDATE_BLOCKS_PER_AIV, 544), (slot, 0, 0)).view(
            CANDIDATE_BLOCKS_PER_AIV, 544
        )
        self._gather_half_tnd(
            key,
            key_scale,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_k,
            task_query_row,
            batch_idx,
            candidate_tile,
            staging_key,
            staging_scale,
            task_idx,
            packed,
            slot,
        )
        vec_sync_intra_arrive(PIPE.MTE3, VECTOR0_READY_ID + candidate_tile % STAGING_DEPTH)


class QsliTndFusedKernel:
    def __init__(
        self,
        tokens,
        topk,
        candidate_capacity,
        mask_mode,
        cmp_ratio,
        has_cu,
        has_seqused_q,
        has_seqused_k,
        has_cu_k,
        has_output_offset,
        pa_block_size,
        splits,
    ):
        self.splits = int(splits)
        self.tokens = int(tokens)
        self.token_tiles = self.tokens // TILE_N
        self.topk_count = int(topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_cu = bool(has_cu)
        self.has_seqused_q = bool(has_seqused_q)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_residual = self.mask_mode == 3 and self.cmp_ratio != 1
        self.has_cu_k = bool(has_cu_k)
        self.has_output_offset = bool(has_output_offset)
        self.qk_handoff = Buffer(MemLoc.UB, (TILE_M * 2, TILE_N // 2), dtypes.bfloat16)
        self.cube = QsliCube()
        self.vector0 = QsliTndGather(get_subblock_id())
        self.vector1 = QsliVector1(tokens)
        self.output_writer = QsliOutputWriter(self.topk_count)
        self.topk_workspace = QsliLocalTopKWorkspace(TOPK_TRUNK_LEN, self.topk_count)
        self.topk = QsliLocalTopKSelector(self.topk_count, self.topk_workspace)
        self.position_mapper = QsliPositionMapper(self.topk_count, candidate_capacity)

    @jit
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        key_scale: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        candidate_block_indices: Tensor,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        cu_seqlens_k: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_scale: Tensor,
        staging_scale_e8m0: Tensor,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
        metadata: Tensor,
        return_values,
        partial_idx_address,
        partial_bits_address,
        final_idx_address,
        final_bits_address,
    ):
        worker_idx = dtypes.int64(get_block_idx())
        subblock_idx = dtypes.int64(get_subblock_id())
        workspace_cursor = dtypes.int64(metadata[worker_idx, 7])
        if metadata[worker_idx, 0] != 0:
            first_b = dtypes.int64(metadata[worker_idx, 1])
            last_b = dtypes.int64(metadata[worker_idx, 4])
            for batch_idx in dsl_range(first_b, min(last_b + 1, batch_count), unroll=1):
                begin = dtypes.int64(0)
                end = dtypes.int64(query_rows)
                if self.has_cu:
                    begin = dtypes.int64(cu_seqlens_q[batch_idx])
                    end = dtypes.int64(cu_seqlens_q[batch_idx + 1])
                used_q = dtypes.int32(end - begin)
                if self.has_seqused_q:
                    used_q = seqused_q[batch_idx]
                actual = dtypes.int32(cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx])
                if self.has_seqused_k:
                    actual = seqused_k[batch_idx]
                residual = dtypes.int32(0)
                if self.has_residual:
                    residual = cmp_residual_k[batch_idx]
                first_m = dtypes.int64(0)
                last_m = end - begin
                if batch_idx == first_b:
                    first_m = dtypes.int64(metadata[worker_idx, 2])
                if batch_idx == last_b:
                    last_m = dtypes.int64(metadata[worker_idx, 5]) + dtypes.int64(
                        dyn_select(metadata[worker_idx, 6] > 0, 1, 0)
                    )
                self.vector0.prepare_identity()
                prefetched_tiles = dtypes.int64(0)
                for m_idx in dsl_range(first_m, last_m, unroll=1):
                    query_row = begin + m_idx
                    candidate_length = dtypes.int32(candidate_block_length[query_row, 0])
                    tile_begin = dtypes.int64(0)
                    tile_end = (dtypes.int64(candidate_length) + TILE_N // CANDIDATE_BLOCK_SIZE - 1) // (
                        TILE_N // CANDIDATE_BLOCK_SIZE
                    )
                    if batch_idx == first_b and m_idx == first_m:
                        tile_begin = dtypes.int64(metadata[worker_idx, 3])
                    if batch_idx == last_b and m_idx == dtypes.int64(metadata[worker_idx, 5]):
                        tile_end = min(tile_end, dtypes.int64(metadata[worker_idx, 6]))
                    valid_s2 = actual
                    if self.mask_mode == 3:
                        valid_s2 = min(
                            actual,
                            max(
                                dtypes.int32(0),
                                (actual * self.cmp_ratio + residual - used_q + dtypes.int32(m_idx) + 1)
                                // self.cmp_ratio,
                            ),
                        )
                    if m_idx >= dtypes.int64(used_q):
                        valid_s2 = dtypes.int32(0)
                    full_tiles = (dtypes.int64(candidate_length) + TILE_N // CANDIDATE_BLOCK_SIZE - 1) // (
                        TILE_N // CANDIDATE_BLOCK_SIZE
                    )
                    row_ld = valid_s2 > 0 and (tile_begin > 0 or tile_end < full_tiles)
                    output_row = dtypes.int64(dyn_select(row_ld, workspace_cursor, query_row))
                    idx_address = dtypes.int64(dyn_select(row_ld, partial_idx_address, final_idx_address))
                    bits_address = dtypes.int64(dyn_select(row_ld, partial_bits_address, final_bits_address))
                    sparse_indices = make_tensor(
                        make_pointer(dtypes.int32, idx_address, MemLoc.GM),
                        make_layout((max(query_rows, 64), self.topk_count), stride=(self.topk_count, 1)),
                    )
                    sparse_value_bits = make_tensor(
                        make_pointer(dtypes.uint16, bits_address, MemLoc.GM),
                        make_layout((max(query_rows, 64), self.topk_count), stride=(self.topk_count, 1)),
                    )
                    if subblock_idx == dtypes.int64(0):
                        self.output_writer.initialize(sparse_indices, sparse_value_bits, output_row)
                    if row_ld:
                        workspace_cursor += 1
                    candidate_begin = tile_begin * (TILE_N // CANDIDATE_BLOCK_SIZE)
                    candidate_end = tile_end * (TILE_N // CANDIDATE_BLOCK_SIZE)
                    if candidate_length > 0 and tile_end > tile_begin and valid_s2 > 0:
                        valid_count = dtypes.int32(0)
                        if valid_s2 > dtypes.int32(0):
                            if subblock_idx == dtypes.int64(0):
                                self.position_mapper.count_visible(
                                    candidate_block_indices,
                                    query_row,
                                    candidate_length,
                                    valid_s2,
                                    dtypes.int32(candidate_begin),
                                    dtypes.int32(candidate_end),
                                    self.vector1.valid_counts,
                                )
                            else:
                                self.position_mapper.load_own_counts(
                                    candidate_block_indices,
                                    query_row,
                                    candidate_length,
                                    valid_s2,
                                    dtypes.int32(candidate_begin),
                                    dtypes.int32(candidate_end),
                                    self.vector1.valid_counts,
                                )

                        if valid_s2 > dtypes.int32(0):
                            self.cube.load_query(
                                tile_view(query, (N1, LOGICAL_D), (query_row, 0)),
                                tile_view(query_scale, (N1, 2, 2), (query_row, 0, 0)),
                            )
                            self.vector1.load_weights(weights, query_row)
                            if prefetched_tiles == 0:
                                self.vector0.begin_buffers()
                            for qk_slot in range_constexpr(2):
                                vec_sync_intra_arrive(PIPE.V, 4 + qk_slot)
                            for tick in dsl_range(tile_begin, tile_end + 1, unroll=1):
                                if tick < tile_end and tick >= tile_begin + prefetched_tiles:
                                    if tick >= tile_begin + STAGING_DEPTH:
                                        self.vector0.await_slot(tick)
                                    self.vector0.gather(
                                        key,
                                        key_scale,
                                        candidate_block_indices,
                                        candidate_length,
                                        cu_seqlens_k,
                                        query_row,
                                        batch_idx,
                                        tick,
                                        staging_key,
                                        staging_scale,
                                        worker_idx,
                                        self.has_cu_k,
                                    )
                                if tick > tile_begin:
                                    candidate_tile = tick - 1
                                    self.vector0.wait_ready(candidate_tile)
                                    staging_row = (
                                        worker_idx * STAGING_DEPTH + dtypes.int64(candidate_tile) % STAGING_DEPTH
                                    ) * TILE_N
                                    self.cube.compute_qk(
                                        tile_view(
                                            _offset_view(staging_key_fp4, (staging_row, 0)),
                                            (TILE_N, LOGICAL_D),
                                            (0, 0),
                                        ),
                                        tile_view(
                                            _offset_view(
                                                staging_scale_e8m0,
                                                (staging_row, 0, 0),
                                            ),
                                            (TILE_N, 2, 2),
                                            (0, 0, 0),
                                        ),
                                        self.qk_handoff,
                                        candidate_tile % 2,
                                    )
                                    self.vector0.signal_consumed(candidate_tile)
                                    vec_sync_intra_wait(PIPE.V, 14 + candidate_tile % 2)
                                    self.vector1.compute(
                                        tile_view(self.qk_handoff, (TILE_M, TILE_N // 2), (candidate_tile % 2, 0)),
                                        candidate_tile,
                                        subblock_idx,
                                        candidate_tile - tile_begin,
                                        self.vector0.pair_swaps,
                                    )
                                    vec_sync_intra_arrive(PIPE.V, 4 + candidate_tile % 2)

                            for final_tick in dsl_range(max(tile_begin, tile_end - STAGING_DEPTH), tile_end, unroll=1):
                                self.vector0.await_slot(final_tick)
                            self.vector0.drain_buffers()
                            for qk_slot in range_constexpr(2):
                                cube_sync_intra_wait(PIPE.FIXPIPE, 4 + qk_slot)
                                cube_sync_intra_wait(PIPE.FIXPIPE, 20 + qk_slot)
                            self.vector1.join_scores(key_workspace, worker_idx, tile_end - tile_begin, subblock_idx)
                            prefetched_tiles = dtypes.int64(0)
                            next_m = m_idx + 1
                            if next_m < last_m and next_m < dtypes.int64(used_q):
                                next_row = query_row + 1
                                next_length = dtypes.int32(candidate_block_length[next_row, 0])
                                next_end = (
                                    dtypes.int64(next_length) + CANDIDATE_BLOCKS_PER_TILE - 1
                                ) // CANDIDATE_BLOCKS_PER_TILE
                                if batch_idx == last_b and next_m == dtypes.int64(metadata[worker_idx, 5]):
                                    next_end = min(next_end, dtypes.int64(metadata[worker_idx, 6]))
                                next_visible = actual
                                if self.mask_mode == 3:
                                    next_visible = min(
                                        actual,
                                        max(
                                            dtypes.int32(0),
                                            (actual * self.cmp_ratio + residual - used_q + dtypes.int32(next_m) + 1)
                                            // self.cmp_ratio,
                                        ),
                                    )
                                if next_length > 0 and next_end > 0 and next_visible > 0:
                                    self.vector0.begin_buffers()
                                    prefetched_tiles = min(dtypes.int64(STAGING_DEPTH), next_end)
                                    for future_tile in dsl_range(prefetched_tiles, unroll=1):
                                        self.vector0.gather(
                                            key,
                                            key_scale,
                                            candidate_block_indices,
                                            next_length,
                                            cu_seqlens_k,
                                            next_row,
                                            batch_idx,
                                            future_tile,
                                            staging_key,
                                            staging_scale,
                                            worker_idx,
                                            self.has_cu_k,
                                        )
                            if subblock_idx == dtypes.int64(0):
                                valid_count = self.position_mapper.read_count()
                                if valid_count > dtypes.int32(0):
                                    self.topk.select_local(
                                        self.vector1.key_ub,
                                        (tile_end - tile_begin) * TILE_N,
                                        valid_count,
                                        tile_view(
                                            selected_position_workspace,
                                            (1, self.topk_count),
                                            (worker_idx, 0),
                                        ),
                                        tile_view(
                                            sparse_value_bits,
                                            (1, self.topk_count),
                                            (output_row, 0),
                                        ),
                                        dtypes.int64(dyn_select(metadata[36, 0] != 0, 1, return_values)),
                                    )
                                    output_offset = dtypes.int32(0)
                                    if self.has_output_offset and not row_ld:
                                        output_offset = dtypes.int32(output_idx_offset[query_row, 0])
                                    self.position_mapper.apply(
                                        self.topk.output_idx_stage,
                                        candidate_block_indices,
                                        tile_view(
                                            sparse_indices,
                                            (1, self.topk_count),
                                            (output_row, 0),
                                        ),
                                        query_row,
                                        valid_count,
                                        output_offset,
                                        dtypes.uint32(tile_begin * TILE_N),
                                    )


_TND_COMPILED_KERNEL = {}
_TND_COMPILED_KERNEL_LOCK = threading.Lock()


def _build_tnd_compiled_fused_runner(
    tokens,
    topk,
    candidate_capacity,
    mask_mode,
    cmp_ratio,
    has_cu,
    has_seqused_q,
    has_seqused_k,
    has_cu_k,
    has_output_offset,
    pa_block_size,
    splits,
    auto_metadata,
):
    """Compile the dynamic TensorSpec contract using the standard DSL entry."""
    has_residual = int(mask_mode) == 3 and int(cmp_ratio) != 1

    @kernel
    def fused_body(
        query_address,
        key,
        key_scale,
        weights: Tensor,
        query_scale: Tensor,
        candidate_block_indices,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        cu_seqlens_k: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_key,
        staging_key_fp4,
        staging_scale,
        staging_scale_e8m0,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices_address,
        sparse_value_address,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
        metadata: Tensor,
        final_sparse_indices_address,
        final_sparse_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        ld_enabled = metadata[36, 0] != 0
        final_sparse_indices = make_tensor(
            make_pointer(dtypes.int32, dtypes.int64(final_sparse_indices_address), MemLoc.GM),
            make_layout((query_rows, topk), stride=(topk, 1)),
        )
        final_sparse_bits = make_tensor(
            make_pointer(dtypes.uint16, dtypes.int64(final_sparse_bits_address), MemLoc.GM),
            make_layout((query_rows, topk), stride=(topk, 1)),
        )
        query = make_tensor(
            make_pointer(dtypes.fp4x2_e2m1, dtypes.int64(query_address), MemLoc.GM),
            make_layout((query_rows * N1, LOGICAL_D), stride=(LOGICAL_D, 1)),
        )
        index_pointer = make_pointer(dtypes.int32, dtypes.int64(sparse_indices_address), MemLoc.GM)
        value_pointer = make_pointer(dtypes.uint16, dtypes.int64(sparse_value_address), MemLoc.GM)
        sparse_indices = make_tensor(index_pointer, make_layout((64, topk), stride=(topk, 1)))
        sparse_value_bits = make_tensor(value_pointer, make_layout((64, topk), stride=(topk, 1)))
        merge_indices = make_tensor(index_pointer, make_layout((64, topk), stride=(topk, 1)))
        merge_bits = make_tensor(value_pointer, make_layout((64, topk), stride=(topk, 1)))
        QsliTndFusedKernel(
            tokens,
            topk,
            candidate_capacity,
            mask_mode,
            cmp_ratio,
            has_cu,
            has_seqused_q,
            has_seqused_k,
            has_cu_k,
            has_output_offset,
            pa_block_size,
            splits,
        )(
            query,
            key,
            key_scale,
            weights,
            query_scale,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_q,
            seqused_q,
            cu_seqlens_k,
            seqused_k,
            cmp_residual_k,
            output_idx_offset,
            staging_key,
            staging_key_fp4,
            staging_scale,
            staging_scale_e8m0,
            key_workspace,
            selected_position_workspace,
            sparse_indices,
            sparse_value_bits,
            worker_count,
            query_rows,
            batch_count,
            logical_k_capacity,
            metadata,
            return_values,
            sparse_indices_address,
            sparse_value_address,
            final_sparse_indices_address,
            final_sparse_bits_address,
        )
        if ld_enabled:
            global_sync_all()
            channel_rewind(reset_sync_id=True)
            LdMergeStage(topk, splits)(
                merge_indices,
                merge_bits,
                final_sparse_indices,
                final_sparse_bits,
                metadata,
                query_rows,
                worker_count * 2,
                cu_seqlens_q,
                1,
                has_cu,
                output_offset_address,
            )

    def run_fused(
        query_address,
        key,
        key_scale,
        weights: Tensor,
        query_scale: Tensor,
        candidate_block_indices,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        cu_seqlens_k: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_key,
        staging_key_fp4,
        staging_scale,
        staging_scale_e8m0,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices_address,
        sparse_value_address,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
        metadata: Tensor,
        final_sparse_indices_address,
        final_sparse_bits_address,
        return_values,
        merge_workers,
        output_offset_address,
    ):
        fused_body[worker_count](
            query_address,
            key,
            key_scale,
            weights,
            query_scale,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_q,
            seqused_q,
            cu_seqlens_k,
            seqused_k,
            cmp_residual_k,
            output_idx_offset,
            staging_key,
            staging_key_fp4,
            staging_scale,
            staging_scale_e8m0,
            key_workspace,
            selected_position_workspace,
            sparse_indices_address,
            sparse_value_address,
            worker_count,
            query_rows,
            batch_count,
            logical_k_capacity,
            metadata,
            final_sparse_indices_address,
            final_sparse_bits_address,
            return_values,
            merge_workers,
            output_offset_address,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    key_rows_dim = cannbotdsl.Dim("T2")
    batch_dim = cannbotdsl.Dim("B")
    boundary_dim = cannbotdsl.Dim("B_PLUS_1")
    worker_dim = cannbotdsl.Dim("WORKERS", min=1, max=MAX_CUBE_WORKERS)
    # Packed FP4 entry tensors need static host-carrier shapes in DSL 0.5.
    # Reserve bounded staging capacity; launch count and TopK workspaces stay dynamic.
    staging_workers = MAX_CUBE_WORKERS
    key_stride_dim = cannbotdsl.Dim("K_S0")
    scale_stride_dim = cannbotdsl.Dim("KS_S0")
    fake = cannbotdsl.TensorSpec
    dummy_spec = fake((query_rows_dim, N2), dtypes.int32)
    cu_spec = fake((boundary_dim,), dtypes.int32) if has_cu else dummy_spec
    seqused_q_spec = fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_spec
    cu_seqlens_k_spec = fake((boundary_dim,), dtypes.int32)
    seqused_k_spec = fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_spec
    residual_spec = fake((batch_dim,), dtypes.int32) if has_residual else dummy_spec
    offset_spec = fake((query_rows_dim, N2), dtypes.int32)
    compiled = jit(run_fused).compile(
        dtypes.int64,
        fake((key_rows_dim, 1, PACKED_D), dtypes.uint8, stride=(key_stride_dim, PACKED_D, 1)),
        fake((key_rows_dim, 1, 2, 2), dtypes.uint8, stride=(scale_stride_dim, 4, 2, 1)),
        fake((query_rows_dim, N1), dtypes.float32),
        fake((query_rows_dim * N1, 2, 2), dtypes.float8_e8m0),
        fake((query_rows_dim, N2, candidate_capacity), dtypes.int32),
        fake((query_rows_dim, N2), dtypes.int32),
        cu_spec,
        seqused_q_spec,
        cu_seqlens_k_spec,
        seqused_k_spec,
        residual_spec,
        offset_spec,
        fake(
            (staging_workers * STAGING_DEPTH * TILE_N, PACKED_D),
            dtypes.uint8,
        ),
        fake(
            (staging_workers * STAGING_DEPTH * TILE_N, LOGICAL_D),
            dtypes.fp4x2_e2m1,
        ),
        fake((staging_workers * STAGING_DEPTH * TILE_N, 4), dtypes.uint8),
        fake(
            (staging_workers * STAGING_DEPTH * TILE_N, 2, 2),
            dtypes.float8_e8m0,
        ),
        fake((worker_dim, tokens), dtypes.uint16),
        fake((worker_dim, topk), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        fake((128, 8), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return compiled


def clear_tnd_caches():
    """Close cached dynamic executables."""
    with _TND_COMPILED_KERNEL_LOCK:
        for compiled in _TND_COMPILED_KERNEL.values():
            compiled.close()
        _TND_COMPILED_KERNEL.clear()


def _get_tnd_compiled_fused_runner(*config):
    """Cache the dynamic QSLI executable; shape axes remain dynamic."""
    cache_key = tuple(config)
    with _TND_COMPILED_KERNEL_LOCK:
        compiled = _TND_COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_tnd_compiled_fused_runner(*config)
            _TND_COMPILED_KERNEL[cache_key] = compiled
        return compiled


def _validate_tnd_sequence(tensor, name, device, count=None):
    if tensor is None or tensor.dtype != torch.int32 or tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int32 tensor")
    if tensor.device != device or device.type != "npu" or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the input NPU device")
    if count is not None and tensor.numel() != count:
        raise ValueError(f"{name} must have {count} entries")


def _validate_tnd_geometry(lengths, cu_q, cu_k, used_q, used_k, residual, *, topk, mask_mode, cmp_ratio):
    if (
        lengths.ndim != 2
        or lengths.shape[1] != 1
        or lengths.shape[0] <= 0
        or lengths.dtype != torch.int32
        or lengths.device.type != "npu"
        or not lengths.is_contiguous()
    ):
        raise ValueError("candidate_block_length must be contiguous NPU int32 (T1,1), T1>0")
    device = lengths.device
    _validate_tnd_sequence(cu_q, "cu_seqlens_q", device)
    if cu_q.numel() < 2:
        raise ValueError("cu_seqlens_q must contain at least two boundaries")
    batch = cu_q.numel() - 1
    _validate_tnd_sequence(cu_k, "cu_seqlens_k", device, batch + 1)
    for name, tensor in (("seqused_q", used_q), ("seqused_k", used_k)):
        if tensor is not None:
            _validate_tnd_sequence(tensor, name, device, batch)
    if not 1 <= int(topk) <= 8192 or int(mask_mode) not in (0, 3) or not 1 <= int(cmp_ratio) <= 128:
        raise ValueError("invalid topk, mask_mode or cmp_ratio")
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        _validate_tnd_sequence(residual, "cmp_residual_k", device, batch)
    elif residual is not None:
        raise ValueError("cmp_residual_k requires mask_mode=3 and cmp_ratio!=1")
    return batch


def _run_tnd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    cmp_residual_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    output_idx_offset: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    *,
    descale_k: torch.Tensor | None = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Run fused QSLI without a Host-built scheduling metadata table.

    ``candidate_block_length[t, 0]`` is the exact number of valid leading
    entries in ``candidate_block_indices[t, 0]`` and must be in ``[0, 2048]``.
    Every entry in that prefix is a logical eight-token block id valid for the
    row's batch; the suffix is ignored.  These value constraints are consumed
    as the device-side contract so Host does not copy candidates or lengths
    back to CPU.  ``metadata`` can supply reusable device-produced S2 shard records.
    """
    if int(quant_mode) != 1:
        raise ValueError("quant_sparse_lightning_indexer supports quant_mode=1 only")
    if layout_q != "TND" or layout_k != "TND":
        raise ValueError("TND path requires TND Q and K")
    if block_table is not None:
        raise ValueError("block_table must be None for TND K")
    if q.device.type != "npu" or k.device.type != "npu":
        raise ValueError("quant_sparse_lightning_indexer requires NPU q/k tensors")
    if q.dtype != torch.uint8 or k.dtype != torch.uint8:
        raise TypeError("q/k storage must be uint8 packed MXFP4")
    if q.ndim != 3 or tuple(q.shape[1:]) != (N1, PACKED_D):
        raise ValueError("q must have shape (T1,32,64)")
    query_rows = int(q.shape[0])
    if query_rows <= 0 or k.ndim != 3 or tuple(k.shape[1:]) != (1, PACKED_D) or k.shape[0] <= 0:
        raise ValueError("TND k must have shape (T2,1,64), T1/T2>0")
    _validate_pa_dim0_stride(k, "k")
    if descale_k is None or tuple(descale_k.shape) != (k.shape[0], 1, 2, 2):
        raise ValueError("descale_k is required with shape (T2,1,2,2)")
    if descale_k.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("descale_k must contain E8M0 or uint8 bits")
    _validate_pa_dim0_stride(descale_k, "descale_k")
    logical_k_capacity = int(k.shape[0])
    pa_block_size = 0  # unused by TND; preserved runner configuration position
    for name, tensor in (
        ("q", q),
        ("w", w),
        ("descale_q", descale_q),
        ("candidate_block_indices", candidate_block_indices),
    ):
        if tensor is None or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, tensor in (
        ("k", k),
        ("w", w),
        ("descale_q", descale_q),
        ("descale_k", descale_k),
        ("candidate_block_indices", candidate_block_indices),
        ("candidate_block_length", candidate_block_length),
    ):
        if tensor.device != q.device:
            raise ValueError(f"{name} must share q's device")

    if tuple(w.shape) != (query_rows, N1) or w.dtype != torch.float32:
        raise ValueError("w must be FP32 with shape (T1,32)")
    if w.device.type != "npu":
        raise ValueError("w must be an NPU tensor")
    expected_q_scale = (query_rows, N1, LOGICAL_D // 64, 2)
    if descale_q is None or tuple(descale_q.shape) != expected_q_scale:
        raise ValueError("descale_q is required with shape (T1,32,D/64,2)")
    scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
    if descale_q.dtype not in scale_dtypes or descale_q.device.type != "npu":
        raise TypeError("descale_q must be an NPU E8M0/uint8 tensor")

    if int(candidate_block_size) != CANDIDATE_BLOCK_SIZE:
        raise ValueError("candidate_block_size must be 8")
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("topk must be in [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("cmp_ratio must be in [1,128]")
    if int(max_seqlen_q) < -1:
        raise ValueError("max_seqlen_q must be -1 or nonnegative")

    if (
        candidate_block_indices.ndim != 3
        or tuple(candidate_block_indices.shape[:2]) != (query_rows, N2)
        or int(candidate_block_indices.shape[2]) != CANDIDATE_CAPACITY
    ):
        raise ValueError("candidate_block_indices must have shape (T1,1,2048)")
    if candidate_block_indices.dtype != torch.int32 or candidate_block_indices.device.type != "npu":
        raise TypeError("candidate_block_indices must be an NPU int32 tensor")
    if (
        candidate_block_length.dtype != torch.int32
        or candidate_block_length.device.type != "npu"
        or int(candidate_block_length.numel()) != query_rows * N2
    ):
        raise ValueError("candidate_block_length must be NPU int32 with T1*N2 entries")
    candidate_length_2d = candidate_block_length.reshape(query_rows, N2)

    batch = _validate_tnd_geometry(
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )
    if output_idx_offset is not None and (
        tuple(output_idx_offset.shape) != (query_rows, N2)
        or output_idx_offset.dtype != torch.int32
        or output_idx_offset.device != q.device
        or not output_idx_offset.is_contiguous()
    ):
        raise ValueError("output_idx_offset must be contiguous int32 (T1,1) on q's device")

    # Static capacity supplies the rolled loop bound; dynamic lengths and
    # candidate prefixes are consumed in the kernel and never copied to CPU.
    # Compile the candidate-capacity workspace once.  ``logical_k_capacity``
    # is overridden by the per-batch effective used_k in TND.
    full_tokens = CANDIDATE_CAPACITY * CANDIDATE_BLOCK_SIZE
    split_count = max(1, min(8, 16384 // int(topk)))
    auto_metadata = False
    worker_count = get_indexer_worker_count(q.device)
    if metadata is None:
        raise ValueError("metadata is required; call quant_sparse_lightning_indexer_metadata before this operator")
    if metadata.device != q.device:
        raise ValueError("metadata must be on q's device")
    if metadata.dtype != torch.int32 or metadata.ndim != 1 or metadata.numel() != 1024 or not metadata.is_contiguous():
        raise ValueError("metadata must be contiguous int32 [1024] in ASC LI/LD boundary format")
    metadata = metadata.view(128, 8)
    tokens = full_tokens
    worker_count = get_indexer_worker_count(q.device)

    sparse_indices = torch.empty(
        (64, N2, int(topk)),
        dtype=torch.int32,
        device=q.device,
    )
    sparse_values = torch.empty(
        (64, N2, int(topk)),
        dtype=torch.bfloat16,
        device=w.device,
    )
    staging_key = torch.empty(
        (MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N, PACKED_D),
        dtype=q.dtype,
        device=q.device,
    )
    staging_scale = torch.empty(
        (MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N, 4),
        dtype=q.dtype,
        device=q.device,
    )
    key_workspace = torch.empty((worker_count, tokens), dtype=torch.uint16, device=q.device)
    selected_position_workspace = torch.empty(
        (worker_count, int(topk)),
        dtype=torch.int32,
        device=q.device,
    )

    # Optional inputs use an existing int32 tensor as an ABI placeholder.
    # Their constexpr flags remove every dummy access from generated device code.
    int32_dummy = candidate_length_2d
    cu_npu = cu_seqlens_q if cu_seqlens_q is not None else int32_dummy
    seqused_q_npu = seqused_q if seqused_q is not None else int32_dummy
    cu_k_npu = cu_seqlens_k
    seqused_k_npu = seqused_k if seqused_k is not None else int32_dummy
    residual_npu = cmp_residual_k if cmp_residual_k is not None else int32_dummy
    offset_npu = output_idx_offset if output_idx_offset is not None else int32_dummy

    query_scale_bytes = descale_q.view(torch.uint8)
    final_sparse_indices = torch.full((query_rows, int(topk)), -1, dtype=torch.int32, device=q.device)
    final_sparse_values = torch.zeros((query_rows, int(topk)), dtype=torch.bfloat16, device=q.device)
    _get_tnd_compiled_fused_runner(
        tokens,
        int(topk),
        CANDIDATE_CAPACITY,
        int(mask_mode),
        int(cmp_ratio),
        cu_seqlens_q is not None,
        seqused_q is not None,
        seqused_k is not None,
        True,
        output_idx_offset is not None,
        pa_block_size,
        split_count,
        auto_metadata,
    )(
        q.data_ptr(),
        k,
        descale_k.view(torch.uint8),
        w,
        query_scale_bytes.reshape(query_rows * N1, 2, 2).view(torch.float8_e8m0fnu),
        candidate_block_indices,
        candidate_length_2d,
        cu_npu,
        seqused_q_npu,
        cu_k_npu,
        seqused_k_npu,
        residual_npu,
        offset_npu,
        staging_key,
        staging_key.view(torch.int8),
        staging_scale,
        staging_scale.reshape(-1, 2, 2).view(torch.float8_e8m0fnu),
        key_workspace,
        selected_position_workspace,
        sparse_indices.data_ptr(),
        sparse_values.data_ptr(),
        worker_count,
        query_rows,
        batch,
        logical_k_capacity,
        metadata,
        final_sparse_indices.data_ptr(),
        final_sparse_values.view(torch.uint16).data_ptr(),
        int(return_value),
        min(64, query_rows),
        0 if output_idx_offset is None else output_idx_offset.data_ptr(),
    )
    sparse_indices, sparse_values = final_sparse_indices, final_sparse_values
    sparse_indices = sparse_indices.view(query_rows, N2, int(topk))
    sparse_values = sparse_values.view(query_rows, N2, int(topk))
    return (
        sparse_indices,
        sparse_values if return_value else sparse_values.reshape(-1)[:0],
    )
