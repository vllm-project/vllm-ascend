# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fused Chunk Kimi Delta Attention public operator."""

import dataclasses
import threading
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple

_RAW_K_FORMAT = "bf16"
import cannbotdsl
import torch
from cannbotdsl import dtypes
from cannbotdsl.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.channel_arena import channel_rewind
from cannbotdsl.constexpr import const_expr
from cannbotdsl.control_flow import range as dsl_range
from cannbotdsl.delay_line import DelayLineGroup
from cannbotdsl.integer import Int64
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import cast, cumsum, matmul
from cannbotdsl.raw_reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup_lane0,
    vdup_scalar,
    vexp,
    vload,
    vload_brc,
    vload_unpack,
    vmax,
    vmem_bar,
    vmin,
    vmins,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_min,
    vreduce_sum,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
    vsqrt,
)
from cannbotdsl.raw_reg import vselect as vselect_raw
from cannbotdsl.sync import (
    cube_fill_l1_zero,
    cube_raw_l1_to_l0a,
    cube_raw_l1_to_l0b,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    _layout_op_wrapper,
    idx2crd,
    partition_view,
    local_slice,
    tile_view,
    make_copy_engine,
    make_layout,
    make_partition_tiler,
    mem_copy,
)
from cannbotdsl.typing.types import ChannelKind, MemLoc, PIPE, Tensor
from cannbotdsl.vf import vf

CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
HALF_CHUNK_SIZE = CHUNK_SIZE // 2
D_HALF = SUPPORTED_HEAD_DIM // 2
MAX_GROUP_CHUNKS = 8
LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32
_DYNAMIC_KERNEL_CACHE: dict[
    tuple[int, int, int, str, int, int, bool, int], object
] = {}


def get_group_config(heads: int, seq_len: int) -> int:
    """选 chunk group：最大 2 的幂 g 使 L2 足迹 heads*g ≤ 861，再夹到 [填满32核, chunks]。

    heads 越大 g 越小（查表，断点 = 861//g）；BN 合轴后 heads = batch*heads。
    """
    chunks = seq_len // CHUNK_SIZE
    if   heads <= 1:    g = 512
    elif heads <= 3:    g = 256
    elif heads <= 6:    g = 128
    elif heads <= 13:   g = 64
    elif heads <= 26:   g = 32
    elif heads <= 53:   g = 16
    elif heads <= 107:  g = 8
    elif heads <= 215:  g = 4
    elif heads <= 430:  g = 2
    else:               g = 1
    g = min(g, chunks)                       # 不超过总 chunks
    while heads * g < 32 and g < chunks:     # 下界：填满 32 核
        g *= 2
    return max(1, g)


def get_group_config_bn(bn: int) -> int:
    """动态 S 版 group：按 bn(=heads) 选，**不依赖 seq_len**（末尾 group 由 group_count 运行时夹到实际
    chunks，效果同 static 的 chunks 夹取）。传一个足够大的 seq 让 get_group_config 的 chunks 夹取不生效，
    得到纯 heads-based g（L2 足迹 heads*g≤861 由其 heads 表保证）。这样 group 只随 bn 变、S 仍可复用。"""
    return get_group_config(bn, 512 * CHUNK_SIZE)


def get_dv_base_config(bn: int, seq_len: int) -> int:
    """Select the Dv columns owned by one core."""
    if seq_len // CHUNK_SIZE < 1:
        return SUPPORTED_HEAD_DIM
    if bn <= 4:
        return 16
    if bn <= 8:
        return 32
    if bn <= 16:
        return 64
    return 128


def _o_head_chunk_tile(gm_O, seq_idx, chunk_idx, head_dim, dv_base, dv_idx):
    """Return one logical BNSD output chunk while preserving physical strides."""
    nv = gm_O.shape[1]
    batch_idx = seq_idx // nv
    head_idx = seq_idx - batch_idx * nv
    full_chunk = tile_view(
        gm_O[batch_idx, head_idx, None, None],
        (CHUNK_SIZE, head_dim),
        (chunk_idx, Int64(0)),
    )
    return tile_view(full_chunk, (CHUNK_SIZE, dv_base), (Int64(0), dv_idx))


class _Stage1Constants(NamedTuple):
    lower_mask_tiles: torch.Tensor
    identity16: torch.Tensor


_STAGE1_CONSTANTS_CACHE: dict[tuple[str, int | None, torch.dtype, torch.dtype], _Stage1Constants] = {}


@dataclasses.dataclass
class StageOneWorkspace:
    Q_decayed: torch.Tensor
    Mqk: torch.Tensor
    K_restored: torch.Tensor
    gamma_C: torch.Tensor
    U_pre: torch.Tensor
    W: torch.Tensor
    identity16: torch.Tensor
    lower_mask_tiles: torch.Tensor


def _empty_like_device(
    shape: tuple[int, ...],
    *,
    ref: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=ref.device)


def allocate_qk_exchange_workspace(
    *, active_block_num: int, ref: torch.Tensor,
) -> torch.Tensor:
    """Allocate one pair of fp32 Q/K half-exchange tiles per physical block."""
    assert active_block_num > 0, "active_block_num must be positive"
    return _empty_like_device(
        (active_block_num, 2, CHUNK_SIZE, D_HALF), ref=ref, dtype=HIGH_DTYPE,
    )


def qk_exchange_slices(
    workspace: torch.Tensor, *, block_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this block's Q-to-K and K-to-Q exchange directions."""
    assert workspace.dtype == HIGH_DTYPE
    assert workspace.dim() == 4 and tuple(workspace.shape[1:]) == (2, CHUNK_SIZE, D_HALF)
    assert 0 <= block_idx < workspace.shape[0], "block_idx is outside the exchange workspace"
    return workspace[block_idx, 0], workspace[block_idx, 1]


def _resolved_device_key(ref: torch.Tensor) -> tuple[str, int | None]:
    device = ref.device
    device_type = getattr(device, "type", "cpu")
    device_index = getattr(device, "index", None)
    if device_type in {"npu", "privateuseone"} and device_index is None:
        npu = getattr(torch, "npu", None)
        current_device = getattr(npu, "current_device", None) if npu is not None else None
        if current_device is not None:
            device_index = int(current_device())
    return device_type, device_index


def _stage1_constants_for(
    ref: torch.Tensor,
    *,
    mask_dtype: torch.dtype = HIGH_DTYPE,
    identity_dtype: torch.dtype = torch.float16,
) -> _Stage1Constants:
    device_type, device_index = _resolved_device_key(ref)
    key = (device_type, device_index, mask_dtype, identity_dtype)
    cached = _STAGE1_CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached

    mask_cols = CHUNK_SIZE // 4
    strict_zero = torch.triu(torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool, device=ref.device), diagonal=0)
    casual_zero = torch.triu(torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool, device=ref.device), diagonal=1)
    strict_bytemask = strict_zero.contiguous().view(torch.float32)
    casual_bytemask = casual_zero.contiguous().view(torch.float32)
    lower_mask_tiles = torch.empty((CHUNK_SIZE, mask_cols * 2), dtype=torch.float32, device=ref.device)
    lower_mask_tiles[:, :mask_cols] = strict_bytemask
    lower_mask_tiles[:, mask_cols:] = casual_bytemask
    constants = _Stage1Constants(
        lower_mask_tiles=lower_mask_tiles.contiguous(),
        identity16=torch.eye(16, dtype=identity_dtype, device=ref.device).contiguous(),
    )
    _STAGE1_CONSTANTS_CACHE[key] = constants
    return constants


def allocate_stage1_workspace(
    batch: int,
    heads: int,
    eff_group: int,
    dim: int,
    *,
    ref: torch.Tensor,
) -> StageOneWorkspace:
    # Scratch is reused for each chunk group rather than spanning the sequence.
    bn_chunks = batch * heads * eff_group
    constants = _stage1_constants_for(ref)
    return StageOneWorkspace(
        Q_decayed=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        Mqk=_empty_like_device((bn_chunks * CHUNK_SIZE, CHUNK_SIZE), ref=ref, dtype=LOW_DTYPE),
        K_restored=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        gamma_C=_empty_like_device((bn_chunks * dim, 1), ref=ref, dtype=HIGH_DTYPE),
        U_pre=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        W=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        identity16=constants.identity16,
        lower_mask_tiles=constants.lower_mask_tiles,
    )


@dataclasses.dataclass
class _ExecutionBuffers:
    workspace: StageOneWorkspace
    qk_exchange: torch.Tensor
    launch_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class _ExecutionCache:
    """Bounded cache of stream-private internal FlashKDA buffers."""

    def __init__(self, max_entries: int = 16):
        self._max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    @property
    def size(self):
        return len(self._entries)

    @staticmethod
    def _stream_identity(ref):
        device = ref.device
        if device.type not in {"npu", "privateuseone"}:
            return device.type, device.index, None
        device_index = ref.get_device() if hasattr(ref, "get_device") else device.index
        stream = torch.npu.current_stream(device_index)
        stream_id = getattr(stream, "npu_stream", stream)
        return device.type, device_index, int(stream_id)

    def get_or_create(
        self,
        *,
        ref,
        batch: int,
        value_heads: int,
        eff_group: int,
        dim: int,
        block_num: int,
        key_heads: Optional[int] = None,
        seq_len: Optional[int] = None,
        layout_qkv: str = "BNSD",
    ):
        key_heads = value_heads if key_heads is None else key_heads
        seq_len = eff_group * CHUNK_SIZE if seq_len is None else seq_len
        key = (
            *self._stream_identity(ref), batch, key_heads, value_heads,
            seq_len, dim, eff_group, block_num, layout_qkv,
        )
        with self._lock:
            cached = self._entries.pop(key, None)
            if cached is None:
                cached = _ExecutionBuffers(
                    workspace=allocate_stage1_workspace(batch, value_heads, eff_group, dim, ref=ref),
                    qk_exchange=allocate_qk_exchange_workspace(active_block_num=block_num, ref=ref),
                )
            self._entries[key] = cached
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return cached

    def clear(self):
        with self._lock:
            self._entries.clear()


_EXECUTION_CACHE = _ExecutionCache()


def clear_execution_cache():
    _EXECUTION_CACHE.clear()


ELEM_BYTES = 4
VL = 64  # 一个 256B 向量寄存器的 f32/b32 lane 数（raw VF 步长）；此处恰等于 CHUNK_SIZE，一行 = 一寄存器

NEUMANN_BLOCK_SIZE = 16
NEUMANN_DIAG_BLOCK_NUM = 4
NEUMANN_FRACTAL_ELEMS = 256                                                     # 一个 16x16 块的元素数。
NEUMANN_DIAG_SRC_STRIDE = NEUMANN_BLOCK_SIZE * CHUNK_SIZE + NEUMANN_BLOCK_SIZE  # ND 源矩阵相邻对角块起点间隔。
NEUMANN_DIAG_DST_STRIDE = (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS  # packed/NZ 目标相邻对角块起点间隔。

class StageOneMatmul:
    """StageOne Cube 侧矩阵乘计算。"""

    def __init__(self, *, head_dim: int):
        self.head_dim = head_dim
        # Neumann packed64 求逆工作区。
        self.tile_power_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.packed_inv64_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.tile_inv_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.scratch_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.neumann_power_handoff_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1, kind=ChannelKind.CrossCore)
        self.packed_identity_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.zero_l1 = Buffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.resident_inv_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1)
        # L0 — double-buffer + Neumann16
        self.l0a_db = Channel(MemLoc.L0A, (CHUNK_SIZE, head_dim), dtypes.bfloat16, depth=2)
        self.l0b_db = Channel(MemLoc.L0B, (CHUNK_SIZE, head_dim), dtypes.bfloat16, depth=2)
        self.l0c_db = Channel(MemLoc.L0C, (CHUNK_SIZE, head_dim), dtypes.float32, depth=2)
        self.per_d_left_k_l0a = Channel(MemLoc.L0A, (NEUMANN_BLOCK_SIZE, head_dim), dtypes.bfloat16, depth=1)
        self.per_d_left_q_l0a = Channel(MemLoc.L0A, (NEUMANN_BLOCK_SIZE, head_dim), dtypes.bfloat16, depth=1)
        self.per_d_l0b_db = Channel(MemLoc.L0B, (NEUMANN_BLOCK_SIZE, head_dim), dtypes.bfloat16, depth=2)
        self.raw_l0a_db = Channel(MemLoc.L0A, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0b_db = Channel(MemLoc.L0B, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0c_main = Channel(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1)
        self.raw_l0c_aux = Channel(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1)
        # 引擎
        self.gm2l1_identity16_diag = make_copy_engine(
            format_transform="nd2nz",
            dtype=dtypes.float16,
            pad_value=0.0,
            nd_num=NEUMANN_DIAG_BLOCK_NUM,
            src_nd_stride=0,
            dst_nd_stride=NEUMANN_DIAG_DST_STRIDE,
            dst_c0_stride=CHUNK_SIZE,
        )
        self.fixpipe_engine = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)
        self.fixpipe_l0c2packed_l1 = make_copy_engine(dtype=dtypes.float32, dst_c0_stride=CHUNK_SIZE, unit_flag_mode=3)

    def load_k_decayed(self, k_decayed_l1):
        mem_copy(self.l0a_db, k_decayed_l1)

    def load_q_decayed(self, q_decayed_l1):
        mem_copy(self.l0a_db, q_decayed_l1)

    def load_k_inv(self, k_inv_l1):
        mem_copy(self.l0b_db, k_inv_l1)

    def mm_kkt(self):
        """KKT = K_decayed @ K_inv^T；K_inv 的 L0B slot 留给 Mqk 复用。"""
        matmul(local_slice(self.l0c_db, (CHUNK_SIZE, CHUNK_SIZE), offset=0), self.l0a_db, self.l0b_db, init=True)
        return self.l0b_db

    def mm_mqk(self, k_inv_l0b):
        """Mqk = Q_decayed @ K_inv^T，消费 mm_kkt 保留的 K_inv slot。"""
        matmul(local_slice(self.l0c_db, (CHUNK_SIZE, CHUNK_SIZE), offset=0), self.l0a_db, k_inv_l0b, init=True)

    def load_per_d_left_k(self, l1_channel):
        mem_copy(self.per_d_left_k_l0a, l1_channel)

    def load_per_d_left_q(self, l1_channel):
        mem_copy(self.per_d_left_q_l0a, l1_channel)

    def load_per_d_right_k(self, l1_channel):
        mem_copy(self.per_d_l0b_db, l1_channel)

    def mm_per_d_kkt(self, row_block: int, col_block: int):
        kkt_tile = tile_view(
            self.raw_l0c_main,
            (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            (row_block, col_block),
        )
        matmul(kkt_tile, self.per_d_left_k_l0a, self.per_d_l0b_db, init=True)

    def mm_per_d_mqk(self, row_block: int, col_block: int):
        mqk_tile = tile_view(
            self.raw_l0c_aux,
            (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            (row_block, col_block),
        )
        matmul(mqk_tile, self.per_d_left_q_l0a, self.per_d_l0b_db, init=True)

    def finish_per_d_matrices(self, ub_kkt, ub_mqk):
        mem_copy(
            local_slice(ub_kkt, (HALF_CHUNK_SIZE, CHUNK_SIZE), offset=0),
            local_slice(self.raw_l0c_main, (CHUNK_SIZE, CHUNK_SIZE), offset=0),
            engine=self.fixpipe_engine,
        )
        mem_copy(
            local_slice(ub_mqk, (HALF_CHUNK_SIZE, CHUNK_SIZE), offset=0),
            local_slice(self.raw_l0c_aux, (CHUNK_SIZE, CHUNK_SIZE), offset=0),
            engine=self.fixpipe_engine,
        )

    def store_l0c_to_gm(self, gm_tile, *, shape=(CHUNK_SIZE, CHUNK_SIZE)):
        mem_copy(gm_tile, local_slice(self.l0c_db, shape, offset=0))

    def store_l0c_to_ub(self, ub_buffer, *, shape=(CHUNK_SIZE, CHUNK_SIZE), ub_shape=None):
        ub_shape = ub_shape or shape
        mem_copy(local_slice(ub_buffer, ub_shape, offset=0), local_slice(self.l0c_db, shape, offset=0), engine=self.fixpipe_engine)

    def packed64_block(self, l1_matrix, row_block: int, col_block: int):
        # l1_matrix 是 64x64 NZ compact 矩阵。逻辑 16x16 子块按 packed64 block offset 定位。
        offset_elems = (col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE)
        tile = local_slice(l1_matrix, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), stride=(CHUNK_SIZE, 1), offset=offset_elems * 2)
        return tile, l1_matrix, row_block, col_block

    def _packed64_block_offset_elems(self, row_block: int, col_block: int):
        return ( col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
                + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE)

    def l0_matrix_block_offset_elems(self, row_block: int, col_block: int):
        return (row_block * NEUMANN_DIAG_BLOCK_NUM + col_block) * NEUMANN_FRACTAL_ELEMS

    def c64_diag_nz_offset_elems(self, diag_block: int):
        return diag_block * (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS

    def c64_odd_even_pair_dst_gap_fractals(self):
        return (self.c64_diag_nz_offset_elems(2) // NEUMANN_FRACTAL_ELEMS) - 1

    def _raw_l0a_full(self, l0a_slot):
        return local_slice(l0a_slot, (CHUNK_SIZE, CHUNK_SIZE), offset=0)

    def _raw_l0b_full(self, l0b_slot):
        return local_slice(l0b_slot, (CHUNK_SIZE, CHUNK_SIZE), offset=0)

    def _raw_l0a_block(self, l0a_slot, row_block: int, col_block: int):
        return local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
                            offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2,)

    def _raw_l0b_block(self, l0b_slot, row_block: int, col_block: int):
        return local_slice(l0b_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
                            offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2,)

    def _load_l1_block_to_l0a_block(
        self, l1_block, l0a_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 装入 full64 L0A 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0a_block(l0a_slot, row_block, col_block)
        cube_raw_l1_to_l0a(l0_tile, l1_parent, l1_offset_elems=col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE,
                            m_start=0, k_start=0, m_step=1, k_step=1, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=1, transpose=False)

    def _load_l1_block_to_l0b_block(
        self, l1_block, l0b_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 转置装入 full64 L0B 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0b_block(l0b_slot, row_block, col_block)
        cube_raw_l1_to_l0b(l0_tile, l1_parent, l1_offset_elems=col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE,
                            m_start=0, k_start=0, m_step=1, k_step=1, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=1, transpose=True)

    def _load_packed32_block_to_full64_l0b_raw(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = True,
    ):
        src_stride = (src_packed_size + NEUMANN_BLOCK_SIZE - 1) // NEUMANN_BLOCK_SIZE
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems((dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE)
            l0_tile = local_slice(l0b_slot, (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            cube_raw_l1_to_l0b(l0_tile, l1_matrix, l1_offset_elems=l1_offset, m_start=0, k_start=0,
                                m_step=1, k_step=2, src_stride=src_stride, dst_stride=1, transpose=transpose)

    def _load_packed32_block_to_full64_l0a_raw(
        self,
        l1_matrix,
        l0a_slot,
        *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = False,
    ):
        """把 packed32 L1 block 装入 full64 L0A。"""
        src_stride = (src_packed_size + NEUMANN_BLOCK_SIZE - 1) // NEUMANN_BLOCK_SIZE
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems((dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE)
            l0_offset -= 6 * NEUMANN_FRACTAL_ELEMS
            l0_tile = local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            cube_raw_l1_to_l0a(l0_tile, l1_matrix, l1_offset_elems=l1_offset, m_start=0, k_start=0,
                                m_step=1, k_step=2, src_stride=src_stride, dst_stride=1, transpose=transpose)

    def _load_packed16_pair_to_full64_l0a_with_gap(
        self,
        l1_matrix,
        l0a_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 装入 full64 L0A，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        for src_row_block, src_col_block, dst_block in (
            (src0_row_block, src0_col_block, dst0_block),
            (src1_row_block, src1_col_block, dst1_block),
        ):
            src_offset = self._packed64_block_offset_elems(src_row_block, src_col_block)
            # 3510 L0A 的 16x16 单块地址仍需补偿一个 full64 M span。
            dst_offset = (dst_block - (NEUMANN_DIAG_BLOCK_NUM - 1)) * NEUMANN_FRACTAL_ELEMS
            l0_tile = local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), offset=dst_offset * 2)
            cube_raw_l1_to_l0a(l0_tile, l1_matrix, l1_offset_elems=src_offset, m_start=0, k_start=0,
                                m_step=1, k_step=1, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=1, transpose=False,)

    def _load_packed16_pair_to_full64_l0b_trans_with_gap(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 转置装入 full64 L0B，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        self._load_l1_block_to_l0b_block( self.packed64_block(l1_matrix, src0_row_block, src0_col_block),
                                          l0b_slot, dst0_block // NEUMANN_DIAG_BLOCK_NUM, dst0_block % NEUMANN_DIAG_BLOCK_NUM,)
        self._load_l1_block_to_l0b_block( self.packed64_block(l1_matrix, src1_row_block, src1_col_block),
                                          l0b_slot, dst1_block // NEUMANN_DIAG_BLOCK_NUM, dst1_block % NEUMANN_DIAG_BLOCK_NUM,)

    def zero_half_l1(self, l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    def zero_packed64_l0_operands(self, l0a_slot, l0b_slot):
        mem_copy(self._raw_l0a_full(l0a_slot), self.zero_l1)
        mem_copy(self._raw_l0b_full(l0b_slot), self.zero_l1)

    def load_packed64_diag_only_operands_to_l0(
        self, left_l1, right_l1, l0a_slot, l0b_slot, *, clear_l0: bool = True,
    ):
        if clear_l0:
            self.zero_packed64_l0_operands(l0a_slot, l0b_slot)
        for diag_block in range(NEUMANN_DIAG_BLOCK_NUM):
            self._load_l1_block_to_l0a_block(self.packed64_block(left_l1, diag_block, diag_block), l0a_slot, diag_block, diag_block)
            self._load_l1_block_to_l0b_block( self.packed64_block(right_l1, diag_block, diag_block), l0b_slot, diag_block, diag_block,)

    def issue_packed64_diag_only_operands_to_l0_db(
        self, left_l1, right_l1, *, clear_l0: bool = True,
    ):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.load_packed64_diag_only_operands_to_l0(left_l1, right_l1, l0a_write, l0b_write, clear_l0=clear_l0)

    def begin_mmad_raw_l0ab_db_packed64(
        self, c_tile, *, init_c: bool, unit_flag: int,
    ):
        """发起 full64 raw MMAD，显式指定 init/accumulate。"""
        l0a_read = self.raw_l0a_db
        l0b_read = self.raw_l0b_db
        matmul(c_tile, self._raw_l0a_full(l0a_read), self._raw_l0b_full(l0b_read), init=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only(self, left_l1, right_l1, c_tile, *, init_c: bool, unit_flag: int, clear_l0: bool = True):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l1, right_l1, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only_db_pair(self, left_l10, right_l10, left_l11, right_l11, c_tile, *, init_c0: bool, unit_flag0: int,
                                            init_c1: bool, unit_flag1: int, clear_l0: bool = True):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l10, right_l10, clear_l0=clear_l0)
        self.issue_packed64_diag_only_operands_to_l0_db(left_l11, right_l11, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c0, unit_flag=unit_flag0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c1, unit_flag=unit_flag1)

    def fixpipe_packed64_full_to_half_l1( self, dst_channel, c_tile,):
        dst_write = dst_channel
        for row_block in (0, 1, 2, 3):
            for col_block in (0, 1, 2, 3):
                c_offset = ( (col_block * NEUMANN_DIAG_BLOCK_NUM + row_block) * NEUMANN_FRACTAL_ELEMS* ELEM_BYTES)
                c_block = local_slice(c_tile, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), offset=c_offset)
                dst_block, _, _, _ = self.packed64_block(dst_write, row_block, col_block)
                mem_copy(dst_block, c_block, engine=self.fixpipe_l0c2packed_l1)

    def fixpipe_packed64_full_to_bf16_l1(
        self,
        dst_channel,
        c_tile,
    ):
        dst_write = dst_channel
        for row_block in (0, 1, 2, 3):
            for col_block in (0, 1, 2, 3):
                c_offset = (col_block * NEUMANN_DIAG_BLOCK_NUM + row_block) * NEUMANN_FRACTAL_ELEMS * ELEM_BYTES
                c_block = local_slice(c_tile, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), offset=c_offset)
                dst_block, _, _, _ = self.packed64_block(dst_write, row_block, col_block)
                mem_copy(dst_block, c_block, engine=self.fixpipe_l0c2packed_l1)

    def half_l1_power_packed64_diag_only_on_c(
        self, left_l1, right_l1, dst_channel, *, clear_l0: bool,
    ):
        c_power = self.raw_l0c_aux
        self.raw_mmad_packed64_diag_only(left_l1, right_l1, c_power, init_c=True, unit_flag=3, clear_l0=clear_l0)
        c_power_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(dst_channel, c_power_read)

    def prepare_packed64_identity_l1(self, gm_identity16):
        """构造 packed 64x64 identity。"""
        identity_write = self.packed_identity_l1
        self.zero_half_l1(identity_write)
        mem_copy(identity_write, gm_identity16, engine=self.gm2l1_identity16_diag)
        return self.packed_identity_l1

    def issue_negative_t_identity_operands(self, negative_t, identity):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        mem_copy(self._raw_l0a_full(l0a_write), negative_t)
        mem_copy(self._raw_l0b_full(l0b_write), identity)

    def issue_identity_identity_operands(self, identity):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        mem_copy(self._raw_l0a_full(l0a_write), identity)
        mem_copy(self._raw_l0b_full(l0b_write), identity)

    def prepare_inv_init_from_negative_t_cube(self, negative_t, identity):
        """Cube 侧在 L0C 构造 scratch_l1 = (-T) @ I + I @ I。"""
        self.issue_negative_t_identity_operands(negative_t, identity)
        self.issue_identity_identity_operands(identity)
        c_inv_init = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=True, unit_flag=2)
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.scratch_l1, c_inv_read)
        return self.scratch_l1

    def load_packed64_diagonal_inputs_from_l1_handoff(
        self, power_handoff_l1, gm_identity16,
    ):
        """消费 AIV `-T` handoff，并在 Cube 侧构造 `I + (-T)`。"""
        self.zero_half_l1(self.zero_l1)
        identity = self.prepare_packed64_identity_l1(gm_identity16)
        negative_t = power_handoff_l1
        scratch = self.prepare_inv_init_from_negative_t_cube(negative_t, identity)
        return negative_t, identity, scratch

    def neumann_diag_power2_update(self, negative_t, identity, scratch):
        self.half_l1_power_packed64_diag_only_on_c(negative_t, negative_t, self.tile_inv_l1, clear_l0=True)
        power2 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only_db_pair( scratch, identity, scratch, power2, c_inv, init_c0=True,
                                                  unit_flag0=2, init_c1=False, unit_flag1=3, clear_l0=False,)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power2, self.packed_inv64_l1

    def neumann_diag_power4_update(self, power2, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power2, power2, self.tile_power_l1, clear_l0=False)
        power4 = self.tile_power_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power4, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power4, self.packed_inv64_l1

    def neumann_diag_power8_update(self, power4, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power4, power4, self.tile_inv_l1, clear_l0=False)
        power8 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power8, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap( scratch, l0a_write, src0_row_block=1, src0_col_block=0, src1_row_block=3, src1_col_block=2,
                                                          dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
                                                          dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap( packed_inv, l0b_write, src0_row_block=0, src0_col_block=0,
                                                               src1_row_block=2, src1_col_block=2,
                                                               dst_offset_elems=self.c64_diag_nz_offset_elems(0),
                                                               dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),)

    def issue_odd_inv_odd_tmp_full64(
        self, packed_inv, tmp64,
    ):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 1, 1), l0a_write, 1, 1)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 3, 3), l0a_write, 3, 3)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap( tmp64, l0b_write, src0_row_block=1, src0_col_block=0,
                                                               src1_row_block=3, src1_col_block=2,
                                                               dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
                                                               dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),)

    def compose_odd_even_lower16_to32_accum_full64(
        self, scratch, packed_inv,
    ):
        """组合两个 32x32 对角子块内部的 lower 逆矩。"""
        self.issue_odd_lower_even_inv_full64(scratch, packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.issue_odd_inv_odd_tmp_full64(packed_inv, tmp64)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd32_lower_even32_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(self._raw_l0a_full(l0a_write), scratch)
        self._load_packed32_block_to_full64_l0b_raw( packed_inv, l0b_write, src_row=0, src_col=0, dst_row=0, dst_col=0,
                                                     transpose=True,)

    def prepare_odd32_inv_odd32_tmp_full64(self, packed_inv):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(self._raw_l0a_full(l0a_write), packed_inv)
        return l0a_write, l0b_write

    def complete_odd32_inv_odd32_tmp_full64(
        self, tmp64, l0a_write, l0b_write,
    ):
        self._load_packed32_block_to_full64_l0b_raw( tmp64, l0b_write, src_row=HALF_CHUNK_SIZE, src_col=0,
                                                     dst_row=HALF_CHUNK_SIZE, dst_col=0, transpose=True,)

    def compose_odd_even_lower32_to64_accum_full64(
        self, scratch, packed_inv,
    ):
        """组合 64x64 左下 32x32 lower 逆矩并生成 BF16 resident inverse。"""
        self.issue_odd32_lower_even32_inv_full64(scratch, packed_inv)
        second_l0a, second_l0b = (
            self.prepare_odd32_inv_odd32_tmp_full64(packed_inv)
        )
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.complete_odd32_inv_odd32_tmp_full64(tmp64, second_l0a, second_l0b)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_bf16_l1(self.resident_inv_l1, c_inv_read)

    def _load_resident_inv_to_l0a(self):
        resident_inv = self.resident_inv_l1
        l0a_write = self.l0a_db
        cube_raw_l1_to_l0a(local_slice(l0a_write, (CHUNK_SIZE, CHUNK_SIZE), offset=0), resident_inv, l1_offset_elems=0, m_start=0, k_start=0,
                           m_step=NEUMANN_DIAG_BLOCK_NUM, k_step=NEUMANN_DIAG_BLOCK_NUM, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=NEUMANN_DIAG_BLOCK_NUM, transpose=False,)
        return self.l0a_db

    def _load_beta_to_l0b(self, beta_channel):
        beta_read = beta_channel
        l0b_write = self.l0b_db
        cube_raw_l1_to_l0b(l0b_write, beta_read, l1_offset_elems=0, m_start=0, k_start=0,
                           m_step=NEUMANN_DIAG_BLOCK_NUM, k_step=SUPPORTED_HEAD_DIM // NEUMANN_BLOCK_SIZE, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=SUPPORTED_HEAD_DIM // NEUMANN_BLOCK_SIZE, transpose=True,)
        return self.l0b_db

    def _mm_resident_inv_beta(self, l0a_resident, l0b_beta):
        l0c_write = self.l0c_db
        matmul(l0c_write, local_slice(l0a_resident, (CHUNK_SIZE, CHUNK_SIZE), offset=0), l0b_beta, init=True)

    def compute_u_pre_and_w_from_resident_inv_beta(
        self,
        gm_U_pre_scratch,
        gm_W_scratch,
        linear_idx,
        v_beta_l1,
        k_decayed_beta_l1,
    ):
        # β·K 先产出，故先算 W；W 的 MMAD/store 与后续 β·V handoff 重叠。
        l0a_resident = self._load_resident_inv_to_l0a()

        l0b_beta = self._load_beta_to_l0b(k_decayed_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        gm_W_chunk = tile_view(gm_W_scratch, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear_idx, Int64(0)))
        self.store_l0c_to_gm(gm_W_chunk, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM))

        l0b_beta = self._load_beta_to_l0b(v_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        u_pre_part = tile_view(gm_U_pre_scratch, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear_idx, Int64(0)))
        self.store_l0c_to_gm(u_pre_part, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM))


class StageOneVector:
    """Vector 侧标量、mask、搬运和状态更新封装。"""

    def __init__(self, *, head_dim: int, scale_value, subblock_idx):
        self.head_dim = head_dim
        self.scale_value = scale_value
        self.subblock_idx = subblock_idx
        self.ub_q = Channel(MemLoc.UB, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.ub_k = Channel(MemLoc.UB, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.ub_qk_exchange = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_g_raw = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1)
        self.ub_gate_activated = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32, depth=1)
        self.ub_dt_bias = Channel(MemLoc.UB, (D_HALF,), dtypes.float32, depth=1)
        self.ub_alpha = Channel(MemLoc.UB, (1,), dtypes.float32, depth=1)
        self.ub_beta_raw = Channel(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.bfloat16, depth=1)
        self.ub_beta = Buffer(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.float32)
        self.ub_q_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_k_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_v_raw = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1)
        self.ub_gamma = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_gamma_c = Channel(MemLoc.UB, (1, D_HALF), dtypes.float32, depth=2)
        self.ub_cs_snapshot = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32) for _ in range(2)]
        self.ub_mqk = Channel(
            MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=2, kind=ChannelKind.CrossCore
        )
        self.ub_bf16_nz = Channel(
            MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=2,
            data_format="nz", n1_pad=16,
        )
        self.ub_cast_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_kkt_fp16 = Buffer(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.ub_mqk_bf16 = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_g_cumsum = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32, depth=1)
        self.ub_raw_k = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16) for _ in range(2)]
        self.ub_kkt = Channel(
            MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1, kind=ChannelKind.CrossCore
        )
        self.ub_kkt_nz = Channel(
            MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1,
            data_format="nz", n1_pad=16,
        )
        self.ub_per_d_nd = Channel(MemLoc.UB, (NEUMANN_BLOCK_SIZE, D_HALF), dtypes.bfloat16, depth=2)
        self.ub_per_d_nz = Channel(
            MemLoc.UB, (NEUMANN_BLOCK_SIZE, D_HALF), dtypes.bfloat16, depth=2,
            data_format="nz", n1_pad=16,
        )
        # One FIFO carries target-scoped left K/Q and column-scoped right K.
        # Its three original L1 slots let AIV prepare left Q while AIC consumes
        # the first right K; Cube keeps left K/Q resident in distinct L0A roots.
        self.l1_per_d_operand = Channel(MemLoc.L1, (NEUMANN_BLOCK_SIZE, self.head_dim), dtypes.bfloat16, depth=3, kind=ChannelKind.CrossCore)
        self.l1_V_beta = Channel(MemLoc.L1, (CHUNK_SIZE, self.head_dim), dtypes.bfloat16, depth=1, kind=ChannelKind.CrossCore)
        self.l1_K_decayed_beta_for_W = Channel( MemLoc.L1, (CHUNK_SIZE, self.head_dim), dtypes.bfloat16,
                                                depth=1, kind=ChannelKind.CrossCore,)
        self.full_d_column_partition = make_partition_tiler(
            (CHUNK_SIZE, self.head_dim),
            (CHUNK_SIZE, self.head_dim),
            axis=1,
        )
        self.per_d_full_d_partition = make_partition_tiler(
            (NEUMANN_BLOCK_SIZE, D_HALF * 2),
            (NEUMANN_BLOCK_SIZE, D_HALF * 2),
            axis=1,
        )
        self.full_square_partition = make_partition_tiler((CHUNK_SIZE, CHUNK_SIZE), (CHUNK_SIZE, CHUNK_SIZE))
        self.ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.per_d_ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.fp16_ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16, pad_value=0.0)

    def scratch_half_tile(self, gm_scratch, block_idx, cols: int):
        return tile_view(gm_scratch, (HALF_CHUNK_SIZE, cols), (block_idx * Int64(2) + self.subblock_idx, Int64(0)))

    def scratch_dhalf_tile(self, gm_scratch, block_idx):
        return tile_view(gm_scratch, (CHUNK_SIZE, D_HALF), (block_idx, self.subblock_idx))

    @jit
    def publish_normalized_qk_half(self, q_to_k, outgoing_half):
        """Publish this AIV's normalized Q/K half before unrelated V work."""
        vec_sync_notify(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        mem_copy(q_to_k, outgoing_half)
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, QK_PUBLISH_FLAG_ID, mode=1)

    @jit
    def consume_normalized_qk_half(self, k_to_q, incoming_half):
        """Consume the peer half after independent gate activation has run."""
        vec_sync_block_wait(PIPE.MTE2, QK_PUBLISH_FLAG_ID, mode=1)
        mem_copy(incoming_half, k_to_q)
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE2, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_block_wait(PIPE.MTE3, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_notify(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)

    @jit
    def normalize_full_d_bf16(self, tile):
        """Normalize each bf16 [128] row in place using fp32 arithmetic."""
        with vf(mode="raw"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdup_scalar(1.0, dtypes.float32, mask=lane_mask)
            for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                offset = row * Int64(SUPPORTED_HEAD_DIM)
                first = vcast(vload_unpack(tile, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=lane_mask)
                second = vcast(vload_unpack(tile, offset + Int64(D_HALF), mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=lane_mask)
                sum_squares = vreduce_sum(vadd(vmul(first, first, mask=lane_mask), vmul(second, second, mask=lane_mask), mask=lane_mask), mask=lane_mask)
                inverse_norm = vdiv(one, vsqrt(vadds(vdup_lane0(sum_squares, mask=lane_mask), 1.0e-6, mask=lane_mask), mask=lane_mask), mask=lane_mask)
                vstore_pack(tile, offset, vcast(vmul(first, inverse_norm, mask=lane_mask), dtypes.bfloat16, mask=lane_mask), lane_mask, mode=PackMode.B32_TO_B16)
                vstore_pack(tile, offset + Int64(D_HALF), vcast(vmul(second, inverse_norm, mask=lane_mask), dtypes.bfloat16, mask=lane_mask), lane_mask, mode=PackMode.B32_TO_B16)
            vmem_bar("vst_vld")

    @jit
    def prepare_normalized_qk_local_half(
        self, gm_K, gm_Q, gm_qk_exchange, batch_idx, qk_head_idx, chunk_idx, block_idx,
    ):
        """Normalize full Q/K rows and leave the peer half ready for publication."""
        q_to_k = gm_qk_exchange[block_idx, Int64(0), None, None]
        k_to_q = gm_qk_exchange[block_idx, Int64(1), None, None]
        q_chunk = tile_view(gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, Int64(0)))
        k_chunk = tile_view(gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, Int64(0)))

        if self.subblock_idx == Int64(0):
            mem_copy(self.ub_q, q_chunk)
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
            self.normalize_full_d_bf16(self.ub_q)
            with vf(outputs=[self.ub_q_bf16]):
                mem_copy(self.ub_q_bf16, local_slice(self.ub_q, (CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1), offset=0))
            with vf(outputs=[self.ub_qk_exchange]):
                cast(self.ub_qk_exchange, local_slice(self.ub_q, (CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1), offset=D_HALF * 2))
            self.publish_normalized_qk_half(q_to_k, self.ub_qk_exchange)
        else:
            mem_copy(self.ub_k, k_chunk)
            vec_sync_all()
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
            self.normalize_full_d_bf16(self.ub_k)
            with vf(outputs=[self.ub_k_bf16]):
                mem_copy(self.ub_k_bf16, local_slice(self.ub_k, (CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1), offset=D_HALF * 2))
            with vf(outputs=[self.ub_qk_exchange]):
                cast(self.ub_qk_exchange, local_slice(self.ub_k, (CHUNK_SIZE, D_HALF), stride=(SUPPORTED_HEAD_DIM, 1), offset=0))
            self.publish_normalized_qk_half(k_to_q, self.ub_qk_exchange)

        return q_to_k, k_to_q

    @jit
    def finish_normalized_qk_exchange(self, q_to_k, k_to_q):
        """Receive the peer half and return local normalized K/Q halves."""
        if self.subblock_idx == Int64(0):
            self.consume_normalized_qk_half(k_to_q, self.ub_qk_exchange)
            with vf(outputs=[self.ub_k_bf16]):
                cast(self.ub_k_bf16, self.ub_qk_exchange)
        else:
            self.consume_normalized_qk_half(q_to_k, self.ub_qk_exchange)
            with vf(outputs=[self.ub_q_bf16]):
                cast(self.ub_q_bf16, self.ub_qk_exchange)
        return self.ub_k_bf16, self.ub_q_bf16

    def _copy_bf16_gm_half_tile_to_ub(self, gm_tile, ub_channel, *, cols: int = SUPPORTED_HEAD_DIM):
        ub_write = ub_channel
        bf16_valid = local_slice(ub_write, (HALF_CHUNK_SIZE, cols), offset=0)
        mem_copy(bf16_valid, gm_tile)

    def _cast_bf16_ub_tile_to_f32(self, ub_cast_bf16, ub_f32_tile, *, rows: int = HALF_CHUNK_SIZE, cols: int = SUPPORTED_HEAD_DIM):
        bf16_valid = local_slice(ub_cast_bf16, (rows, cols), offset=0)
        f32_valid = local_slice(ub_f32_tile, (rows, cols), offset=0)
        with vf(outputs=[ub_f32_tile]):
            cast(f32_valid, bf16_valid)

    def _store_f32_ub_tile_to_bf16_gm(self, gm_tile, ub_f32_tile, ub_cast_channel, *, rows: int = HALF_CHUNK_SIZE, cols: int = SUPPORTED_HEAD_DIM):
        cast_write = ub_cast_channel
        bf16_valid = local_slice(cast_write, (rows, cols), offset=0)
        with vf(outputs=[cast_write]):
            cast(bf16_valid, ub_f32_tile)
        cast_read = ub_cast_channel
        mem_copy(gm_tile, local_slice(cast_read, (rows, cols), offset=0))

    @jit
    def activate_raw_gate(self, gm_g, gm_A_log, gm_dt_bias, batch_idx, head_idx, chunk_idx, lower_bound):
        """Apply the source gate formula to one D half, then take its cumsum."""
        g_chunk = tile_view(gm_g[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        bias_half = tile_view(gm_dt_bias[head_idx, None], (D_HALF,), (self.subblock_idx,))
        alpha = tile_view(gm_A_log, (1,), (head_idx,))
        mem_copy(self.ub_g_raw, g_chunk)
        mem_copy(self.ub_dt_bias, bias_half)
        mem_copy(self.ub_alpha, alpha)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        with vf(mode="raw"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            bound = vdup_scalar(lower_bound, dtypes.float32, mask=lane_mask)
            one = vdup_scalar(1.0, dtypes.float32, mask=lane_mask)
            alpha_exp = vexp(vload_brc(self.ub_alpha, 0), mask=lane_mask)
            neg_exp_a = vmuls(alpha_exp, -1.0, mask=lane_mask)
            for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                offset = row * Int64(D_HALF)
                raw_g = vcast(vload_unpack(self.ub_g_raw, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=lane_mask)
                gate_logit = vmul(neg_exp_a, vadd(raw_g, vload(self.ub_dt_bias, 0), mask=lane_mask), mask=lane_mask)
                gate = vdiv(bound, vadd(one, vexp(gate_logit, mask=lane_mask), mask=lane_mask), mask=lane_mask)
                vstore(self.ub_gate_activated, offset, gate, lane_mask)
            vmem_bar("vst_vld")
        vec_sync_all()
        cumsum(self.ub_g_cumsum, self.ub_gate_activated, axis=0)
        return self.ub_g_cumsum

    @jit
    def activate_raw_beta(self, gm_beta, batch_idx, head_idx, chunk_idx):
        """Apply sigmoid to one raw BF16 beta-logit chunk."""
        beta_chunk = tile_view(gm_beta[batch_idx, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, Int64(0)))
        mem_copy(self.ub_beta_raw, beta_chunk)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        with vf(outputs=[self.ub_beta]):
            cast(self.ub_beta, local_slice(self.ub_beta_raw, (CHUNK_SIZE, 1), offset=0))
        with vf(mode="raw"):
            lane_mask, _ = update_mask(VL, elem_bits=32)
            one = vdup_scalar(1.0, dtypes.float32, mask=lane_mask)
            raw_beta = vload(self.ub_beta, 0)
            beta = vdiv(one, vadd(one, vexp(vmuls(raw_beta, -1.0, mask=lane_mask), mask=lane_mask), mask=lane_mask), mask=lane_mask)
            vstore(self.ub_beta, 0, beta, lane_mask)
        return self.ub_beta

    @jit
    def apply_beta_and_finish_t_for_neumann(self, beta_slot, power_handoff_l1, subblock_base: int, buffer_parity: int):
        row_base = subblock_base * HALF_CHUNK_SIZE   # 本半块全局行起点(0 或 32)
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdup_scalar(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(Int64(0), Int64(HALF_CHUNK_SIZE), Int64(1), unroll=4):
                row_off = row * Int64(CHUNK_SIZE)
                t_true = vload(self.ub_kkt, row_off)
                beta_brc = vload_brc(beta_slot, Int64(row_base) + row)        # β[row] 广播到整行(逐行标量)
                t_beta = vmul(t_true, beta_brc, mask=row_mask)               # T[row,:] *= β[row]
                # 严格下三角：全局行 i 保留前 i 列(对角也清)；update_mask(i) 的前 i 个 lane = 保留谓词。
                tril_mask, _ = update_mask(Int64(row_base) + row, elem_bits=32)
                t_tril = vselect_raw(t_beta, zero_reg, cond_mask=tril_mask)   # lane<i 取 t_beta，其余取 0
                neg_t = vmuls(t_tril, -1.0, mask=row_mask)                    # -T
                neg_t_fp16 = vcast(neg_t, dtypes.float16, mask=row_mask)            # f32 -> fp16
                vstore_pack(self.ub_kkt_fp16, row_off, neg_t_fp16, row_mask, mode="b32_to_b16")
        # nd2nz 拆到 raw 之外（skill 第4步；同 V pipe 程序序保 RaW）：ub_kkt_fp16(nd) -> ub_kkt_nz(nz)
        with vf(outputs=[self.ub_kkt_nz]):
            mem_copy(self.ub_kkt_nz, self.ub_kkt_fp16, engine=self.fp16_ub2l1)
        self._store_neumann_tmp_to_handoff_l1(power_handoff_l1, subblock_base)

    @jit
    def prepare_beta_weighted_c3_operands(self, gm_V, beta_slot, batch_idx, head_idx, chunk_idx, buffer_parity: int):
        """Construct the two C3 operands from one full S by D-half vector tile."""
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        v_half = tile_view(gm_V[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        self._copy_bf16_gm_half_tile_to_ub(v_half, self.ub_v_raw, cols=D_HALF)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)

        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                offset = row * Int64(D_HALF)
                neg_beta = vmuls(vload_brc(beta_slot, row), -1.0, mask=row_mask)
                cs_value = vload(self.ub_gamma, offset)
                if const_expr(_RAW_K_FORMAT == "fp32"):
                    k_value = vload(raw_k_snapshot, offset)
                else:
                    k_value = vcast(vload_unpack(raw_k_snapshot, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=row_mask)
                k_decayed_beta = vmul(vmul(k_value, vexp(cs_value, mask=row_mask), mask=row_mask), neg_beta, mask=row_mask)
                vstore_pack(cast_write, offset, vcast(k_decayed_beta, dtypes.bfloat16, mask=row_mask), row_mask, mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        with vf(outputs=[self.ub_bf16_nz]):
            mem_copy(self.ub_bf16_nz, cast_read, engine=self.ub2l1)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_K_decayed_beta_for_W, self.ub_bf16_nz)

        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                offset = row * Int64(D_HALF)
                beta = vload_brc(beta_slot, row)
                v_value = vcast(vload_unpack(self.ub_v_raw, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=row_mask)
                vstore_pack(cast_write, offset, vcast(vmul(v_value, beta, mask=row_mask), dtypes.bfloat16, mask=row_mask), row_mask, mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        with vf(outputs=[self.ub_bf16_nz]):
            mem_copy(self.ub_bf16_nz, cast_read, engine=self.ub2l1)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_V_beta, self.ub_bf16_nz)

    @jit
    def prepare_normalized_chunk_inputs(self, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity: int):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        log_gamma_c_write = self.ub_gamma_c
        self.save_gamma_c_row(gate_cumsum_dhalf, log_gamma_c_write)
        gate_cumsum_tile = local_slice(gate_cumsum_dhalf, (CHUNK_SIZE, D_HALF), offset=0)
        with vf(outputs=[self.ub_gamma]):
            mem_copy(self.ub_gamma, gate_cumsum_tile)
        mem_copy(gate_cumsum_snapshot, gate_cumsum_tile)

        if const_expr(_RAW_K_FORMAT == "fp32"):
            with vf(mode="raw"):
                row_mask, _ = update_mask(VL, elem_bits=32)
                for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                    offset = row * Int64(D_HALF)
                    k_value = vcast(vload_unpack(normalized_k_dhalf, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=row_mask)
                    vstore(raw_k_snapshot, offset, k_value, row_mask)
        else:
            mem_copy(raw_k_snapshot, normalized_k_dhalf)

        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                offset = row * Int64(D_HALF)
                exp_value = vexp(vload(self.ub_gamma, offset), mask=row_mask)
                q_value = vcast(vload_unpack(normalized_q_dhalf, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=row_mask)
                q_decayed = vmuls(vmul(q_value, exp_value, mask=row_mask), self.scale_value, mask=row_mask)
                vstore_pack(cast_write, offset, vcast(q_decayed, dtypes.bfloat16, mask=row_mask), row_mask, mode="b32_to_b16")
        return self.ub_cast_bf16, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf

    def _store_dhalf_nz_ub_to_full_l1(self, l1_dst, ub_nz_src):
        l1_part = partition_view(l1_dst, self.full_d_column_partition, self.subblock_idx)
        mem_copy(l1_part, ub_nz_src)

    def _store_neumann_tmp_to_handoff_l1(self, l1_dst, subblock_base: int):
        l1_part = partition_view(l1_dst, self.full_square_partition, Int64(subblock_base))
        mem_copy(l1_part, self.ub_kkt_nz)

    @jit
    def _store_masked_mqk_to_bf16_gm(self, gm_tile, buffer_parity: int):
        # raw：Mqk(32×64) 取含对角下三角(casual: update_mask(全局行 i+1) 谓词+vselect 零寄存器) → cast bf16 → GM(nd)。2 行 unroll。
        mqk_slot = self.ub_mqk
        bf16_write = self.ub_mqk_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdup_scalar(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(Int64(0), Int64(HALF_CHUNK_SIZE), Int64(1), unroll=4):
                off = row * Int64(CHUNK_SIZE)
                keep, _ = update_mask(self.subblock_idx * Int64(HALF_CHUNK_SIZE) + row + Int64(1), elem_bits=32)
                mqk = vload(mqk_slot, off)
                mqk_tril = vselect_raw(mqk, zero_reg, cond_mask=keep)   # 保留 lane<=i（含对角）
                mqk_bf16 = vcast(mqk_tril, dtypes.bfloat16, mask=row_mask)
                vstore_pack(bf16_write, off, mqk_bf16, row_mask, mode="b32_to_b16")
        bf16_read = self.ub_mqk_bf16
        mem_copy(gm_tile, local_slice(bf16_read, (HALF_CHUNK_SIZE, CHUNK_SIZE), offset=0))

    def finish_mqk(self, gm_Mqk_scratch, block_idx, subblock_base: int, buffer_parity: int):
        Mqk_part = self.scratch_half_tile(gm_Mqk_scratch, block_idx, CHUNK_SIZE)
        self._store_masked_mqk_to_bf16_gm(Mqk_part, buffer_parity)
        return self.ub_gamma_c

    def store_q_decayed_scratch(self, gm_Q_decayed_scratch, block_idx, q_decayed_slot):
        # Q_decayed 的 bf16 已在 prepare_chunk 的 Q_decay vf 里 cast 好、留在 ub_cast_bf16(buf9)；
        # 本方法紧跟 prepare_chunk、中间无人碰 buf9 → 直接把它搬到 GM，省一次 f32→bf16 cast。
        Q_decayed_slot_part = self.scratch_dhalf_tile(gm_Q_decayed_scratch, block_idx)
        mem_copy(Q_decayed_slot_part, q_decayed_slot)

    def save_gamma_c_row(self, g_slot, gamma_c_write):
        src_pre = (CHUNK_SIZE - 1) * D_HALF   # cumsum 最后一行元素起点
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            vstore(gamma_c_write, 0, vload(g_slot, src_pre), row_mask)

    @jit
    def store_restored_state_inputs(
        self, gm_K_restored_scratch, gm_gamma_C_scratch, block_idx, gamma_c_slot, buffer_parity: int,
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        K_restored_dhalf = self.scratch_dhalf_tile(gm_K_restored_scratch, block_idx)
        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            cs_last = vload(gamma_c_slot, 0)
            for row in dsl_range(Int64(0), Int64(CHUNK_SIZE), Int64(1), unroll=4):
                offset = row * Int64(D_HALF)
                if const_expr(_RAW_K_FORMAT == "fp32"):
                    k_value = vload(raw_k_snapshot, offset)
                else:
                    k_value = vcast(vload_unpack(raw_k_snapshot, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=row_mask)
                restored = vmul(k_value, vexp(vsub(cs_last, vload(gate_cumsum_snapshot, offset), mask=row_mask), mask=row_mask), mask=row_mask)
                vstore_pack(cast_write, offset, vcast(restored, dtypes.bfloat16, mask=row_mask), row_mask, mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        mem_copy(K_restored_dhalf, cast_read)

        # gamma_C can underflow to zero, which is the correct state-decay value; it is never multiplied by K_inv.
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            vstore(gamma_c_slot, 0, vexp(vload(gamma_c_slot, 0), mask=row_mask), row_mask)
        gamma_C_gm = tile_view(gm_gamma_C_scratch, (D_HALF, 1), (block_idx * Int64(2) + self.subblock_idx, Int64(0)))
        mem_copy(gamma_C_gm, gamma_c_slot)

    @jit
    def _write_per_d_full_d_handoff(self, full_channel, nd_slot):
        mem_copy(self.ub_per_d_nz, nd_slot, engine=self.per_d_ub2l1)
        full_part = partition_view(full_channel, self.per_d_full_d_partition, self.subblock_idx)
        mem_copy(full_part, self.ub_per_d_nz)

    @jit
    def build_per_d_left_k(self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, left_k_l1_operand):
        k_read = local_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), offset=row_block * NEUMANN_BLOCK_SIZE * D_HALF * 2)
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_per_d_nd
        with vf(mode="raw"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(Int64(0), Int64(NEUMANN_BLOCK_SIZE), Int64(1), unroll=2):
                offset = row * Int64(D_HALF)
                g_offset = (Int64(row_block * NEUMANN_BLOCK_SIZE) + row) * D_HALF
                k = vcast(vload_unpack(k_read, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=mask)
                value = vmul(k, vexp(vsub(vload(gate_cumsum_dhalf, g_offset), ref, mask=mask), mask=mask), mask=mask)
                vstore_pack(nd_write, offset, vcast(value, dtypes.bfloat16, mask=mask), mask, mode="b32_to_b16")
        nd_read = self.ub_per_d_nd
        self._write_per_d_full_d_handoff(left_k_l1_operand, nd_read)

    @jit
    def build_per_d_left_q(self, normalized_q_dhalf, gate_cumsum_dhalf, row_block: int, left_q_l1_operand):
        q_read = local_slice(normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), offset=row_block * NEUMANN_BLOCK_SIZE * D_HALF * 2)
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF
        nd_write = self.ub_per_d_nd
        with vf(mode="raw"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(Int64(0), Int64(NEUMANN_BLOCK_SIZE), Int64(1), unroll=2):
                offset = row * Int64(D_HALF)
                g_offset = (Int64(row_block * NEUMANN_BLOCK_SIZE) + row) * D_HALF
                q = vcast(vload_unpack(q_read, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=mask)
                value = vmuls(vmul(q, vexp(vsub(vload(gate_cumsum_dhalf, g_offset), ref, mask=mask), mask=mask), mask=mask), self.scale_value, mask=mask)
                vstore_pack(nd_write, offset, vcast(value, dtypes.bfloat16, mask=mask), mask, mode="b32_to_b16")
        nd_read = self.ub_per_d_nd
        self._write_per_d_full_d_handoff(left_q_l1_operand, nd_read)

    @jit
    def build_per_d_right(self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, col_block: int, right_k_l1_operand):
        k_read = local_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), offset=col_block * NEUMANN_BLOCK_SIZE * D_HALF * 2)
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_per_d_nd
        with vf(mode="raw"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(Int64(0), Int64(NEUMANN_BLOCK_SIZE), Int64(1), unroll=2):
                offset = row * Int64(D_HALF)
                g_offset = (Int64(col_block * NEUMANN_BLOCK_SIZE) + row) * D_HALF
                k = vcast(vload_unpack(k_read, offset, mode=UnpackMode.B16_TO_B32), dtypes.float32, mask=mask)
                value = vmul(k, vexp(vsub(ref, vload(gate_cumsum_dhalf, g_offset), mask=mask), mask=mask), mask=mask)
                vstore_pack(nd_write, offset, vcast(value, dtypes.bfloat16, mask=mask), mask, mode="b32_to_b16")
        nd_read = self.ub_per_d_nd
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_read)

@jit
def _run_stage1_body(
    head_dim: int,
    gm_K,
    gm_V,
    gm_Q,
    gm_beta_brc,
    gm_g,
    gm_qk_exchange,
    gm_A_log,
    gm_dt_bias,
    gm_q_decayed,
    gm_Mqk,
    gm_k_restored,
    gm_gamma,
    gm_U_pre,
    gm_w,
    gm_lower_mask_tiles,
    gm_identity16,
    batch: int,          # 兼容位(不再使用；BN 合轴后 head 由 head_base/head_count 表达)
    head_count: int,
    chunks_total: int,
    head_base: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    scale_value: float,
    lower_bound: float,
):
    """Chunk Kimi Delta Attention Stage1。

    Stage1 对每个 chunk 计算 K_inv、K_decayed、Q_decayed 和 beta 加权的 KKT 下三角矩阵，
    使用 Neumann 展开求解块内三角系统，生成 Mqk、U_pre、W、K_restored 和 gamma_C。

    计算 [head_base, head_base+head_count) 这些(平铺后)head 一组 chunk 的 StageOne workspace。
    输入保持原始 (B, N, S, D)；用 idx2crd 把平铺 head 反解回 (batch, head) 再 2D 索引（免 reshape）。"""
    subblock_idx = get_subblock_id()
    block_idx = get_block_idx()
    block_num = get_block_num()
    batch_count = Int64(gm_K.shape[0])
    value_head_count = Int64(gm_V.shape[1])
    key_head_count = Int64(gm_K.shape[1])
    heads_per_key_head = value_head_count // key_head_count

    total_tasks = head_count * group_count
    active_core_num = block_num
    if active_core_num > total_tasks:
        active_core_num = total_tasks
    tasks_per_core = (total_tasks + active_core_num - Int64(1)) // active_core_num
    task_start = block_idx * tasks_per_core
    task_end = task_start + tasks_per_core
    if task_end > total_tasks:
        task_end = total_tasks

    if block_idx < active_core_num:
        # 严格/casual 下三角改用 raw update_mask 谓词（apply_beta / _store_masked_mqk），
        # 常驻 byte-mask UB(原 buf21/22) 与其 GM load 已删；gm_lower_mask_tiles 入参保留但不再使用。
        # matmul + vector 提到循环外(对齐 stage2)。identity 仍每 chunk 载(不变)。
        matmul = StageOneMatmul(head_dim=head_dim)
        vector = StageOneVector(head_dim=head_dim, scale_value=scale_value, subblock_idx=subblock_idx)
        task_delay_line = DelayLineGroup(2, "idx")
        pipeline_tick = 0
        issued_task_count = 0
        for local_task_index in range(task_start, task_end):
            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_index = task_delay_line.idx.tap(1)
                local_head_index = delayed_task_index // group_count
                group_chunk_index = delayed_task_index - local_head_index * group_count
                chunk_index = group_start + group_chunk_index
                global_head_index = head_base + local_head_index
                batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
                buffer_parity = delayed_task_index % Int64(2)
                vector.apply_beta_and_finish_t_for_neumann(vector.ub_beta, matmul.neumann_power_handoff_l1, subblock_idx, buffer_parity)
                vector.prepare_beta_weighted_c3_operands(gm_V, vector.ub_beta, batch_index, value_head_index, chunk_index, buffer_parity)
                negative_lower_matrix, identity_matrix, neumann_scratch = matmul.load_packed64_diagonal_inputs_from_l1_handoff(matmul.neumann_power_handoff_l1, gm_identity16)
                second_power, packed_inverse = matmul.neumann_diag_power2_update(negative_lower_matrix, identity_matrix, neumann_scratch)
                fourth_power, packed_inverse = matmul.neumann_diag_power4_update(second_power, packed_inverse)
                packed_inverse = matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
                packed_inverse = matmul.compose_odd_even_lower16_to32_accum_full64(neumann_scratch, packed_inverse)
                matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)

            task_delay_line.push(idx=local_task_index)
            local_head_index = local_task_index // group_count
            group_chunk_index = local_task_index - local_head_index * group_count
            chunk_index = group_start + group_chunk_index
            global_head_index = head_base + local_head_index
            batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
            query_key_head_index = value_head_index // heads_per_key_head
            workspace_task_index = global_head_index * eff_group + group_chunk_index
            buffer_parity = local_task_index % Int64(2)

            q_to_k_exchange, k_to_q_exchange = vector.prepare_normalized_qk_local_half(gm_K, gm_Q, gm_qk_exchange, batch_index, query_key_head_index, chunk_index, block_idx)
            gate_cumsum_dhalf = vector.activate_raw_gate(gm_g, gm_A_log, gm_dt_bias, batch_index, value_head_index, chunk_index, lower_bound)
            normalized_k_dhalf, normalized_q_dhalf = vector.finish_normalized_qk_exchange(q_to_k_exchange, k_to_q_exchange)
            decayed_q_dhalf, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf = vector.prepare_normalized_chunk_inputs(normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity)
            vector.store_q_decayed_scratch(gm_q_decayed, workspace_task_index, decayed_q_dhalf)

            vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 0, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 0, 0, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(0, 0)
            vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 0, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            matmul.mm_per_d_mqk(0, 0)

            vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 1, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 0, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(1, 0)
            vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 1, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            matmul.mm_per_d_mqk(1, 0)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 1, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(1, 1)
            matmul.mm_per_d_mqk(1, 1)

            vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 2, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 0, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(2, 0)
            vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 2, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            matmul.mm_per_d_mqk(2, 0)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 1, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(2, 1)
            matmul.mm_per_d_mqk(2, 1)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 2, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(2, 2)
            matmul.mm_per_d_mqk(2, 2)

            vector.build_per_d_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, vector.l1_per_d_operand)
            matmul.load_per_d_left_k(vector.l1_per_d_operand)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 0, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 0)
            vector.build_per_d_left_q(normalized_q_dhalf, gate_cumsum_dhalf, 3, vector.l1_per_d_operand)
            matmul.load_per_d_left_q(vector.l1_per_d_operand)
            matmul.mm_per_d_mqk(3, 0)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 1, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 1)
            matmul.mm_per_d_mqk(3, 1)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 2, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 2)
            matmul.mm_per_d_mqk(3, 2)
            vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 3, vector.l1_per_d_operand)
            matmul.load_per_d_right_k(vector.l1_per_d_operand)
            matmul.mm_per_d_kkt(3, 3)
            matmul.mm_per_d_mqk(3, 3)

            vector.activate_raw_beta(gm_beta_brc, batch_index, value_head_index, chunk_index)
            matmul.finish_per_d_matrices(vector.ub_kkt, vector.ub_mqk)

            issued_task_count = issued_task_count + 1
            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_index = task_delay_line.idx.tap(1)
                local_head_index = delayed_task_index // group_count
                group_chunk_index = delayed_task_index - local_head_index * group_count
                workspace_task_index = (head_base + local_head_index) * eff_group + group_chunk_index
                buffer_parity = delayed_task_index % Int64(2)
                log_gamma_c_dhalf = vector.finish_mqk(gm_Mqk, workspace_task_index, subblock_idx, buffer_parity)
                vector.store_restored_state_inputs(gm_k_restored, gm_gamma, workspace_task_index, log_gamma_c_dhalf, buffer_parity)
                matmul.compute_u_pre_and_w_from_resident_inv_beta(gm_U_pre, gm_w, workspace_task_index, vector.l1_V_beta, vector.l1_K_decayed_beta_for_W)
            task_delay_line.advance()
            pipeline_tick = pipeline_tick + 1

        if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
            delayed_task_index = task_delay_line.idx.tap(1)
            local_head_index = delayed_task_index // group_count
            group_chunk_index = delayed_task_index - local_head_index * group_count
            chunk_index = group_start + group_chunk_index
            global_head_index = head_base + local_head_index
            batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
            workspace_task_index = global_head_index * eff_group + group_chunk_index
            buffer_parity = delayed_task_index % Int64(2)
            vector.apply_beta_and_finish_t_for_neumann(vector.ub_beta, matmul.neumann_power_handoff_l1, subblock_idx, buffer_parity)
            vector.prepare_beta_weighted_c3_operands(gm_V, vector.ub_beta, batch_index, value_head_index, chunk_index, buffer_parity)
            negative_lower_matrix, identity_matrix, neumann_scratch = matmul.load_packed64_diagonal_inputs_from_l1_handoff(matmul.neumann_power_handoff_l1, gm_identity16)
            second_power, packed_inverse = matmul.neumann_diag_power2_update(negative_lower_matrix, identity_matrix, neumann_scratch)
            fourth_power, packed_inverse = matmul.neumann_diag_power4_update(second_power, packed_inverse)
            packed_inverse = matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
            packed_inverse = matmul.compose_odd_even_lower16_to32_accum_full64(neumann_scratch, packed_inverse)
            matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)
            log_gamma_c_dhalf = vector.finish_mqk(gm_Mqk, workspace_task_index, subblock_idx, buffer_parity)
            vector.store_restored_state_inputs(gm_k_restored, gm_gamma, workspace_task_index, log_gamma_c_dhalf, buffer_parity)
            matmul.compute_u_pre_and_w_from_resident_inv_beta(gm_U_pre, gm_w, workspace_task_index, vector.l1_V_beta, vector.l1_K_decayed_beta_for_W)

class StageTwoMatmul:

    def __init__(self, *, head_dim: int, dv_base: int):
        self.head_dim = head_dim
        self.dv_base = dv_base
        self.l0a_chunk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, head_dim), dtype=dtypes.bfloat16, depth=2)
        self.l0a_kr_t = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1, data_format="zn")
        self.l0a_mqk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=1)
        self.l0b_u = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1)
        self.l0b_state = Channel(MemLoc.L0B, shape=(dv_base, head_dim), dtype=dtypes.bfloat16, depth=1)
        self.l0b_kr = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, head_dim), dtype=dtypes.bfloat16, depth=1)
        self.l0c_u = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1)
        self.l0c_o = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1)
        self.l0c_state = Channel(MemLoc.L0C, shape=(dv_base, head_dim), dtype=dtypes.float32, depth=1)
        self.gm2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.fixpipe_state = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=2)
        self.fixpipe_u = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

    def load_k_cumdecay(self, k_cumdecay_l1, gm_tile):
        mem_copy(k_cumdecay_l1, gm_tile, engine=self.gm2l1)

    def load_q_decayed(self, q_decayed_l1, gm_tile):
        mem_copy(q_decayed_l1, gm_tile, engine=self.gm2l1)

    def load_k_restored(self, k_restored_l1, gm_tile):
        mem_copy(k_restored_l1, gm_tile, engine=self.gm2l1)

    def load_mqk(self, mqk_l1, gm_tile):
        mem_copy(mqk_l1, gm_tile, engine=self.gm2l1)

    def load_state(self, state_l1):
        mem_copy(self.l0b_state, state_l1, transpose=False)

    def compute_w_state(self, w_l1):
        mem_copy(self.l0a_chunk, w_l1)
        matmul(self.l0c_u, self.l0a_chunk, self.l0b_state, init=True)

    def compute_q_state(self, q_decayed_l1):
        mem_copy(self.l0a_chunk, q_decayed_l1)
        matmul(self.l0c_o, self.l0a_chunk, self.l0b_state, init=True)

    def update_state(self, k_restored_l1, u_slot):
        u_src = local_slice(u_slot, (CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0a_kr_t, u_src, transpose=True)
        mem_copy(self.l0b_kr, k_restored_l1, transpose=True)
        matmul(self.l0c_state, self.l0a_kr_t, self.l0b_kr, init=True)

    def accumulate_mqk_u(self, mqk_l1, u_slot):
        mem_copy(self.l0a_mqk, mqk_l1)
        u_tile = local_slice(u_slot, (CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0b_u, u_tile, transpose=True)
        matmul(self.l0c_o, self.l0a_mqk, self.l0b_u, init=False)

    def store_u_delta(self, ub_ch):
        mem_copy(ub_ch, local_slice(self.l0c_u, (CHUNK_SIZE, self.dv_base), offset=0), engine=self.fixpipe_u)

    def store_output(self, gm_tile):
        """读取包含 O_state + Mqk@U 的同一 L0C Channel 事务。"""
        mem_copy(gm_tile, local_slice(self.l0c_o, (CHUNK_SIZE, self.dv_base), offset=0))

    def store_state_delta(self, ub_ch):
        shape = (self.dv_base, self.head_dim)
        split_n = make_partition_tiler(shape, shape, axis=1)
        mem_copy(ub_ch, local_slice(self.l0c_state, shape, offset=0), engine=self.fixpipe_state, partition=split_n)


class StageTwoVector:
    def __init__(self, *, u_row_block: int, head_dim: int, dv_base: int):
        self.head_dim = head_dim
        self.dv_base = dv_base
        self.u_row_block = u_row_block
        self.half_dk = head_dim // 2
        self.ub_state_fp32 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32)
        self.tmp_state_bf16 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16)
        self.tmp_u_bf16 = Buffer(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16)
        self.state_nd_ch = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1)
        self.state_nz_ch = Channel(
            MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1,
            data_format="nz", n1_pad=16,
        )
        self.u_nz_ch = Channel(
            MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16, depth=1,
            data_format="nz", n1_pad=16,
        )
        self.state_in_ch = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1)
        self.state_out_ch = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1)
        self.u_in_ch = Channel(MemLoc.UB, (u_row_block, dv_base), dtypes.bfloat16, depth=1)
        self.ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)

    def load_gamma_col(self, gamma_ub, gamma_col):
        slot = gamma_ub
        ub_gamma = local_slice(slot, (self.half_dk, 1), offset=0)
        mem_copy(ub_gamma, gamma_col)

    def _produce_state_l1(self, state_l1, subblock_idx):
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        nd_slot = self.state_nd_ch
        with vf(outputs=[self.state_nd_ch]):
            cast(nd_slot, ub_state)
        nd_read = self.state_nd_ch
        nz_slot = self.state_nz_ch
        with vf():
            mem_copy(nz_slot, nd_read, engine=self.ub2l1)
        nz_read = self.state_nz_ch
        slot = state_l1
        state_l1_rows = tile_view(slot, (self.dv_base, self.half_dk), (Int64(0), subblock_idx))
        mem_copy(state_l1_rows, nz_read)

    def load_initial_state_and_write_l1(self, current_state, state_l1, subblock_idx):
        in_slot = self.state_in_ch
        mem_copy(in_slot, current_state)
        in_read = self.state_in_ch
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        with vf():
            mem_copy(ub_state, in_read)
        self._produce_state_l1(state_l1, subblock_idx)

    def write_state_l1_snapshot(self, state_l1, subblock_idx):
        """Publish the accumulated state for the next Cube step."""
        self._produce_state_l1(state_l1, subblock_idx)

    @jit
    def prepare_scaled_state(self, gamma_ub):
        """ub_state_fp32 *= gamma（broadcast 列向量）—— raw VF 直读 gamma channel slot。"""
        gamma_read = gamma_ub
        ub_gamma = local_slice(gamma_read, (1, self.half_dk), offset=0)
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)

        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            gamma_vec = vload(ub_gamma, 0)      # gamma[0:half_dk]=gamma[0:VL]
            for base in dsl_range(Int64(0), Int64(self.dv_base * self.half_dk), Int64(VL), unroll=4):
                state_v = vload(ub_state, base)
                state_scaled = vmul(state_v, gamma_vec, mask=mask)
                vstore(ub_state, base, state_scaled, mask)

    @jit
    def add_delta(self, tmp_delta):
        """ub_state_fp32 += tmp_delta（CrossCore Channel）—— raw vf 直读 channel slot。"""
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        delta = tmp_delta
        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            for base in dsl_range(Int64(0), Int64(self.dv_base * self.half_dk), Int64(VL), unroll=4):
                state_v = vload(ub_state, base)
                delta_v = vload(delta, base)
                state_next = vadd(state_v, delta_v, mask=mask)
                vstore(ub_state, base, state_next, mask)

    @jit
    def add_u_delta_to_l1(self, gm_u_tile, u_l1, tmp_u_delta, subblock_idx):
        gm_u_rows = tile_view(gm_u_tile, (self.u_row_block, self.dv_base), (subblock_idx, Int64(0)))
        u_l1_rows = tile_view(u_l1, (self.u_row_block, self.dv_base), (subblock_idx, Int64(0)))
        tmp_u_bf16 = local_slice(self.tmp_u_bf16, (self.u_row_block, self.dv_base), offset=0)

        u_in_slot = self.u_in_ch
        mem_copy(u_in_slot, gm_u_rows)
        u_in_read = self.u_in_ch
        with vf(outputs=[self.tmp_u_bf16]):
            mem_copy(tmp_u_bf16, u_in_read)

        delta = tmp_u_delta
        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            for base in range(0, self.u_row_block * self.dv_base, 2 * VL):
                u_raw_pre = vload_unpack(tmp_u_bf16, base, mode=UnpackMode.B16_TO_B32)
                u_raw_post = vload_unpack(tmp_u_bf16, base + VL, mode=UnpackMode.B16_TO_B32)
                u_fp32_pre = vcast(u_raw_pre, dtypes.float32, mask=mask)
                u_fp32_post = vcast(u_raw_post, dtypes.float32, mask=mask)
                delta_pre = vload(delta, base)
                delta_post = vload(delta, base + VL)
                u_next_pre = vadd(u_fp32_pre, delta_pre, mask=mask)
                u_next_post = vadd(u_fp32_post, delta_post, mask=mask)
                u_bf16_pre = vcast(u_next_pre, dtypes.bfloat16, mask=mask)
                u_bf16_post = vcast(u_next_post, dtypes.bfloat16, mask=mask)
                vstore_pack(tmp_u_bf16, base, u_bf16_pre, mask, mode=PackMode.B32_TO_B16)
                vstore_pack(tmp_u_bf16, base + VL, u_bf16_post, mask, mode=PackMode.B32_TO_B16)

        u_nz_slot = self.u_nz_ch
        with vf():
            mem_copy(u_nz_slot, tmp_u_bf16, engine=self.ub2l1)
        u_nz_read = self.u_nz_ch
        mem_copy(u_l1_rows, u_nz_read)

    def store_state_to_gm(self, gm_state_rows):
        """末态 ub_state_fp32 → GM（FP32）。累加在 V pipe、写 GM 在 MTE3，跨 pipe，经 Channel 交接。"""
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        out_slot = self.state_out_ch
        with vf():
            mem_copy(out_slot, ub_state)
        out_read = self.state_out_ch
        mem_copy(gm_state_rows, out_read)


@jit
def _run_stage2_body(
    head_dim: int,
    gm_k_restored,
    gm_gamma,
    gm_q_decayed,
    gm_w,
    gm_U,
    gm_O,
    gm_state_in,
    gm_state_out,
    gm_Mqk,
    *,
    dv_base: int,
    head_base: int,
    head_count: int,
    chunks_total: int,
    group_start: int,
    group_count: int,
    eff_group: int,
):

    block_idx = get_block_idx()
    block_num = get_block_num()
    subblock_idx = get_subblock_id()
    num_dv_base = head_dim // dv_base
    half_dk = head_dim // 2
    u_row_block = CHUNK_SIZE // 2

    q_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, head_dim), dtype=dtypes.bfloat16, depth=2)
    w_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, head_dim), dtype=dtypes.bfloat16, depth=2)
    k_restored_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, head_dim), dtype=dtypes.bfloat16, depth=2, data_format="nz")
    mqk_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=2)
    state_l1 = Channel(MemLoc.L1, shape=(dv_base, head_dim), dtype=dtypes.bfloat16, depth=2, data_format="nz", kind=ChannelKind.CrossCore)
    u_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, head_dim), dtype=dtypes.bfloat16, depth=2, kind=ChannelKind.CrossCore)
    s2_matmul = StageTwoMatmul(head_dim=head_dim, dv_base=dv_base)
    gamma_ub = Channel(MemLoc.UB, shape=(half_dk, 1), dtype=dtypes.float32, depth=2)
    tmp_delta = Channel(MemLoc.UB, shape=(dv_base, half_dk), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore)
    tmp_u_delta = Channel(MemLoc.UB, shape=(u_row_block, dv_base), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore)

    s2_vector = StageTwoVector(u_row_block=u_row_block, head_dim=head_dim, dv_base=dv_base)

    total_items = head_count * Int64(num_dv_base)
    logical_core_num = block_num
    if logical_core_num > total_items:
        logical_core_num = total_items
    items_per_core = (total_items + logical_core_num - Int64(1)) // logical_core_num
    item_start = block_idx * items_per_core
    item_end = item_start + items_per_core
    if item_end > total_items:
        item_end = total_items

    if block_idx < logical_core_num:
        for item in range(item_start, item_end):
            head_id, dv_idx = idx2crd(item, [head_count, Int64(num_dv_base)])
            seq_idx = head_base + head_id
            write_state_ref = tile_view(gm_state_out, (head_dim, head_dim), (seq_idx, Int64(0)))

            first_linear = seq_idx * eff_group
            first_w_chunk = tile_view(gm_w, (CHUNK_SIZE, head_dim), (first_linear, Int64(0)))
            first_q_chunk = tile_view(gm_q_decayed, (CHUNK_SIZE, head_dim), (first_linear, Int64(0)))
            first_gamma_col_full = tile_view(gm_gamma, (head_dim, 1), (first_linear, Int64(0)))
            first_gamma_col = tile_view(first_gamma_col_full, (half_dk, 1), (subblock_idx, Int64(0)))
            s2_matmul.load_k_cumdecay(w_l1, first_w_chunk)
            s2_matmul.load_q_decayed(q_l1, first_q_chunk)
            if group_start == Int64(0):
                in_state_ref = tile_view(gm_state_in, (head_dim, head_dim), (seq_idx, Int64(0)))
                first_state_rows = tile_view(in_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            else:
                first_state_rows = tile_view(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            s2_vector.load_gamma_col(gamma_ub, first_gamma_col)

            for chunk_id in range(Int64(0), group_count):
                k = group_start + chunk_id
                linear = seq_idx * eff_group + chunk_id
                u_tile_full = tile_view(gm_U, (CHUNK_SIZE, head_dim), (linear, Int64(0)))
                u_tile = tile_view(u_tile_full, (CHUNK_SIZE, dv_base), (Int64(0), dv_idx))
                o_chunk = _o_head_chunk_tile(gm_O, seq_idx, k, head_dim, dv_base, dv_idx)
                mqk_tile = tile_view(gm_Mqk, (CHUNK_SIZE, CHUNK_SIZE), (linear, Int64(0)))

                s2_matmul.load_state(state_l1)
                s2_matmul.compute_w_state(w_l1)
                s2_vector.prepare_scaled_state(gamma_ub)
                if chunk_id + Int64(1) < group_count:
                    next_linear = linear + Int64(1)
                    next_q_chunk = tile_view(gm_q_decayed, (CHUNK_SIZE, head_dim), (next_linear, Int64(0)))
                    s2_matmul.load_q_decayed(q_l1, next_q_chunk)
                s2_matmul.store_u_delta(tmp_u_delta)

                s2_matmul.compute_q_state(q_l1)
                k_chunk = tile_view(gm_k_restored, (CHUNK_SIZE, head_dim), (linear, Int64(0)))
                s2_matmul.load_k_restored(k_restored_l1, k_chunk)
                s2_matmul.load_mqk(mqk_l1, mqk_tile)

                s2_vector.add_u_delta_to_l1(u_tile, u_l1, tmp_u_delta, subblock_idx)

                s2_matmul.update_state(k_restored_l1, u_l1)
                if chunk_id + Int64(1) < group_count:
                    next_linear = linear + Int64(1)
                    next_w_chunk = tile_view(gm_w, (CHUNK_SIZE, head_dim), (next_linear, Int64(0)))
                    next_gamma_col_full = tile_view(gm_gamma, (head_dim, 1), (next_linear, Int64(0)))
                    next_gamma_col = tile_view(next_gamma_col_full, (half_dk, 1), (subblock_idx, Int64(0)))
                    s2_vector.load_gamma_col(gamma_ub, next_gamma_col)
                    s2_matmul.load_k_cumdecay(w_l1, next_w_chunk)
                s2_matmul.store_state_delta(tmp_delta)

                s2_matmul.accumulate_mqk_u(mqk_l1, u_l1)
                s2_matmul.store_output(o_chunk)

                s2_vector.add_delta(tmp_delta)
                if chunk_id + Int64(1) < group_count:
                    s2_vector.write_state_l1_snapshot(state_l1, subblock_idx)
                else:
                    state_rows = tile_view(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                    s2_vector.store_state_to_gm(state_rows)


def _bsnd_to_bnsd(t: Tensor) -> Tensor:
    """View physical BSND storage as logical BNSD by swapping layout metadata."""
    def _swap_1_2(layout):
        shape, stride = layout.shape, layout.stride
        return make_layout(
            (shape[0], shape[2], shape[1], shape[3]),
            stride=(stride[0], stride[2], stride[1], stride[3]),
        )

    return _layout_op_wrapper(_swap_1_2, t)
# mode=0 below is a whole-grid barrier.  The launch must fit in one resident
# cube-core wave, otherwise resident blocks wait for blocks that cannot be
# scheduled.  Keep 32 as the algorithmic cap and select the actual launch size
# from the target device on the host.
ALGORITHM_MAX_BLOCK_NUM = 32
DEFAULT_BLOCK_NUM = ALGORITHM_MAX_BLOCK_NUM

STAGE1_BOUNDARY_FLAG_ID = 0
STAGE23_BOUNDARY_FLAG_ID = 1
_STAGE_BOUNDARY_AIC_FLAG_OFFSET = 4
_STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET = 8
QK_PUBLISH_FLAG_ID = 10
QK_CONSUME_FLAG_ID = 11
MTE2_TO_V_EVENT_ID = 0
MTE2_TO_V_V_STAGING_EVENT_ID = 1
MTE2_TO_V_SUBBLOCK1_EVENT_ID = 4
QK_V_TO_MTE3_EVENT_ID = 2
QK_MTE2_TO_V_EVENT_ID = 3


def _rewind_buffers():
    channel_rewind(reset_sync_id=False)


def _stage_boundary_reset(flag_id: int):
    cube_sync_all()
    vec_sync_all()

    aiv_ready_flag_id = flag_id
    aic_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_FLAG_OFFSET
    aic_aiv_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET

    vec_sync_block_arrive(PIPE.MTE3, aiv_ready_flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, aiv_ready_flag_id, mode=2)

    cube_sync_block_arrive(PIPE.FIXPIPE, aic_ready_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, aic_ready_flag_id, mode=0)

    cube_sync_block_arrive(PIPE.MTE3, aic_aiv_ready_flag_id, mode=2)
    vec_sync_block_wait(PIPE.S, aic_aiv_ready_flag_id, mode=2)
    _rewind_buffers()


@kernel
class flash_kda_kernel:
    """Fuse the frontend-compiled stage-one and stage-two bodies and write O once."""

    def __init__(
        self,
        head_dim: int = SUPPORTED_HEAD_DIM,
        group: int = MAX_GROUP_CHUNKS,
        dv_base: int = SUPPORTED_HEAD_DIM,
    ):
        self.head_dim = head_dim
        self.group = int(group)
        self.dv_base = int(dv_base)

    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta_brc: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_k_restored: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_w: Tensor,
        gm_Mqk: Tensor,
        gm_U: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_lower_mask_tiles: Tensor,
        gm_identity16: Tensor,
        bn_total: int,
        seq_len: int,
        block: int,
        num_outer: int,
        scale_value: float,
        lower_bound: float,
    ):
        chunks_total = seq_len // Int64(CHUNK_SIZE)
        GROUP = self.group
        eff_group = chunks_total
        if eff_group > Int64(GROUP):
            eff_group = Int64(GROUP)
        for outer in range(Int64(0), num_outer):
            head_base = outer * block
            head_count = block
            if head_base + block > bn_total:          # 尾轮：不足 block
                head_count = bn_total - head_base
            for group_start in range(Int64(0), chunks_total, Int64(GROUP)):
                group_count = chunks_total - group_start
                if group_count > Int64(GROUP):
                    group_count = Int64(GROUP)

                _run_stage1_body(
                    self.head_dim,
                    gm_K,
                    gm_V,
                    gm_Q,
                    gm_beta_brc,
                    gm_g,
                    gm_qk_exchange,
                    gm_A_log,
                    gm_dt_bias,
                    gm_q_decayed,
                    gm_Mqk,
                    gm_k_restored,
                    gm_gamma,
                    gm_U,
                    gm_w,
                    gm_lower_mask_tiles,
                    gm_identity16,
                    bn_total,
                    head_count,
                    chunks_total,
                    head_base,
                    group_start,
                    group_count,
                    eff_group,
                    scale_value,
                    lower_bound,
                )
                _stage_boundary_reset(STAGE1_BOUNDARY_FLAG_ID)

                _run_stage2_body(
                    self.head_dim,
                    gm_k_restored,
                    gm_gamma,
                    gm_q_decayed,
                    gm_w,
                    gm_U,
                    gm_O,
                    gm_state_in,
                    gm_state_out,
                    gm_Mqk,
                    dv_base=self.dv_base,
                    head_base=head_base,
                    head_count=head_count,
                    chunks_total=chunks_total,
                    group_start=group_start,
                    group_count=group_count,
                    eff_group=eff_group,
                )

                has_next_group = (
                    group_start + group_count < chunks_total or outer + Int64(1) < num_outer
                )
                if has_next_group:
                    _stage_boundary_reset(STAGE23_BOUNDARY_FLAG_ID)


class FlashKDA:
    """host 下发：head_dim / group 编译期常量经 self 携带，@jit run 建 kernel 并 op[block](...) 下发。"""

    def __init__(
        self,
        head_dim: int = SUPPORTED_HEAD_DIM,
        group: int = MAX_GROUP_CHUNKS,
        layout_qkv: str = "BNSD",
        dv_base: int = SUPPORTED_HEAD_DIM,
        block_num: Optional[int] = None,
    ):
        self.head_dim = head_dim
        self.group = int(group)
        self.layout_qkv = layout_qkv
        self.dv_base = int(dv_base)
        self.block_num = _device_block_num() if block_num is None else int(block_num)
        if not 1 <= self.block_num <= ALGORITHM_MAX_BLOCK_NUM:
            raise ValueError(
                "block_num must be in "
                f"[1, {ALGORITHM_MAX_BLOCK_NUM}], got {self.block_num}"
            )

    @jit
    def run(self, K: Tensor, V: Tensor, Q: Tensor, beta: Tensor, g: Tensor,
            qk_exchange: Tensor, A_log: Tensor, dt_bias: Tensor,
            kr: Tensor, gamma: Tensor, qd: Tensor, w: Tensor, mqk: Tensor,
            u: Tensor, o: Tensor, st_in: Tensor, st_out: Tensor,
            masks: Tensor, ident: Tensor,
            bnt: int, s: int, blk: int, no: int, scale: float, lower_bound: float):
        if const_expr(self.layout_qkv == "BSND"):
            K = _bsnd_to_bnsd(K)
            V = _bsnd_to_bnsd(V)
            Q = _bsnd_to_bnsd(Q)
            beta = _bsnd_to_bnsd(beta)
            g = _bsnd_to_bnsd(g)
            o = _bsnd_to_bnsd(o)
        op = flash_kda_kernel(self.head_dim, self.group, self.dv_base)
        op[self.block_num](K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kr, gamma, qd, w, mqk, u, o, st_in, st_out, masks, ident, bnt, s, blk, no, scale, lower_bound)


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """Return a grid-barrier-safe launch size for the selected NPU."""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return ALGORITHM_MAX_BLOCK_NUM

    npu = getattr(torch, "npu", None)
    if npu is None:
        return ALGORITHM_MAX_BLOCK_NUM

    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            # Preserve CPU-only/offline tracing.  A targeted offline compile
            # can still pass block_num explicitly.
            return ALGORITHM_MAX_BLOCK_NUM

    properties = npu.get_device_properties(device_index)
    try:
        cube_core_num = int(properties.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"NPU {device_index} does not expose a valid cube_core_num"
        ) from exc
    if cube_core_num <= 0:
        raise RuntimeError(
            f"NPU {device_index} reports invalid cube_core_num={cube_core_num}"
        )
    return min(cube_core_num, ALGORITHM_MAX_BLOCK_NUM)


def _dynamic_sequence_tensor(dtype, batch, seq_len, heads, width, layout_qkv):
    shape = (batch, heads, seq_len, width)
    if layout_qkv == "BNSD":
        return cannbotdsl.TensorSpec(shape, dtype)
    batch_stride = seq_len * heads * width
    head_stride = batch_stride if heads == 1 else width
    return cannbotdsl.TensorSpec(
        shape,
        dtype,
        stride=(batch_stride, head_stride, heads * width, 1),
    )


def _dynamic_workspace_tokens(
    batch, seq_len, heads, group, scratch_uses_sequence
):
    if scratch_uses_sequence:
        return batch * heads * seq_len
    return batch * heads * group * CHUNK_SIZE


def flash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashKDA forward (Flash Kimi Delta Attention).
        Args:
            q (torch.Tensor): Query, bf16, shape ``[batch_size, seqlens, head_num, head_dim_qk]``.
            k (torch.Tensor): Key, bf16, shape ``[batch_size, seqlens, head_num, head_dim_qk]``.
            v (torch.Tensor): Value, bf16, shape ``[batch_size, seqlens, head_num_v, head_dim_v]``.
            g (torch.Tensor): Raw gate input, bf16, shape ``[batch_size, seqlens, head_num_v, head_dim_qk]``.
            beta (torch.Tensor): Raw beta logits, bf16, shape ``[batch_size, seqlens, head_num_v]``.
            scale (float): Scaling factor.
            initial_state (torch.Tensor, optional): Initial recurrent state, fp32, shape ``[batch_size, head_num_v, head_dim_v, head_dim_qk]``,
            layout_qkv (str):  "BSND" "BNSD" "TND".
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int32, shape ``[batch_size+1]``.

        Return:
            out (torch.Tensor): Output, bf16, shape ``[batch_size, seqlens, head_num_v, head_dim_v]``.
            final_state (torch.Tensor): Final recurrent state, same dtype/shape rules as ``initial_state``.
        Notes:
            * Currently requires ``Dk = Dv = 128``.
            * All input tensors must be NPU, contiguous, and have the dtypes listed above.
            * head_num_v <= 96
            * head_num_v % head_num_qk == 0
            * lower_bound is in [-5, 0]
    """
    assert layout_qkv in ("BNSD", "BSND"), f"layout_qkv must be BNSD or BSND, got {layout_qkv!r}"
    assert cu_seqlens is None, "cu_seqlens is not supported"
    assert q.dim() == 4, "q/k/v must be rank-4"
    batch, dim = q.shape[0], q.shape[3]
    seq_len = q.shape[1] if layout_qkv == "BSND" else q.shape[2]
    n_v = v.shape[2] if layout_qkv == "BSND" else v.shape[1]
    n_kv = q.shape[2] if layout_qkv == "BSND" else q.shape[1]
    assert n_v % n_kv == 0, f"GQA requires Nv % Nk == 0, got Nv={n_v}, Nk={n_kv}"
    if layout_qkv == "BSND":
        qk_shape = (batch, seq_len, n_kv, dim)
        v_shape = (batch, seq_len, n_v, dim)
        beta_shape = (batch, seq_len, n_v)      # 对外 beta 只收 3 维 BSN
    else:
        qk_shape = (batch, n_kv, seq_len, dim)
        v_shape = (batch, n_v, seq_len, dim)
        beta_shape = (batch, n_v, seq_len)      # 对外 beta 只收 3 维 BNS
    assert tuple(q.shape) == qk_shape and tuple(k.shape) == qk_shape, "q/k shape does not match layout"
    assert tuple(v.shape) == v_shape, "v shape does not match layout"
    assert tuple(g.shape) == v_shape, "g shape does not match layout"
    # beta 对外接口是 3 维 BNS/BSN，校验也只按 3 维；末轴的 1 由内部补齐喂 kernel。
    assert beta.dim() == 3 and tuple(beta.shape) == beta_shape, "beta must be 3-D BNS/BSN matching layout"
    beta = beta.unsqueeze(-1)                    # 内部补 1 → BNS1/BSN1（kernel 期望 4 维）
    assert tuple(initial_state.shape) == (batch, n_v, dim, dim), "initial_state must be (B, Nv, D, D)"
    assert dim == SUPPORTED_HEAD_DIM, f"Flash KDA only supports D={SUPPORTED_HEAD_DIM}"
    assert seq_len % CHUNK_SIZE == 0, f"S must be a multiple of {CHUNK_SIZE}"

    assert q.dtype == LOW_DTYPE and k.dtype == LOW_DTYPE and v.dtype == LOW_DTYPE, "q/k/v must be bf16"
    assert tuple(A_log.shape) == (n_v,), f"A_log must have shape ({n_v},)"
    assert tuple(dt_bias.shape) == (n_v, dim), f"dt_bias must have shape ({n_v}, {dim})"
    assert -5.0 <= float(lower_bound) <= 0.0, "lower_bound must be in [-5, 0]"
    assert g.dtype == LOW_DTYPE, "g must be bf16"
    assert beta.dtype == LOW_DTYPE, "beta must be bf16"
    assert initial_state.dtype == torch.float32, "initial_state must be fp32"
    assert A_log.dtype == HIGH_DTYPE, "A_log must be fp32"
    assert dt_bias.dtype == HIGH_DTYPE, "dt_bias must be fp32"
    tensors = (q, k, v, g, beta, initial_state, A_log, dt_bias)
    assert all(t.device == q.device for t in tensors), "all inputs must be on the same device"
    assert all(t.is_contiguous() for t in tensors), "all inputs must be contiguous"

    scale_value = float(scale)
    bn_total = batch * n_v
    group = get_group_config_bn(bn_total)
    eff_group = min(seq_len // CHUNK_SIZE, group)
    if layout_qkv == "BSND":
        out_physical = torch.empty(batch, seq_len, n_v, dim, dtype=LOW_DTYPE, device=q.device)
        out = out_physical.transpose(1, 2)
        if n_v == 1:
            fixed_stride = list(out.stride())
            fixed_stride[1] = seq_len * n_v * dim
            out = torch.as_strided(out, out.shape, tuple(fixed_stride))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)
    else:
        out = torch.empty(batch, n_v, seq_len, dim, dtype=LOW_DTYPE, device=q.device)
        out_physical = out
    final_state = torch.empty(
        initial_state.shape,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    # group / dv_base 均按实际 bn 分桶（dv_base：少头切 Dv；group：少头用大 group 减 barrier），
    # 一并作为下面动态 kernel 缓存的 key；B/S 仍动态复用（末尾 group 由 group_count 运行时夹取）。
    dv_base = get_dv_base_config(bn_total, seq_len)

    # min(S / CHUNK_SIZE, group) 不属于 Dim 表达式。按 min 的两个分支分别
    # specialization，既保持 B/S 动态复用，也让 workspace 与输入关系精确可校验。
    scratch_uses_sequence = seq_len // CHUNK_SIZE <= group
    # block_num 决定 launch grid，必须入 key，防止跨设备误复用。
    # FlashKDA 恒 layout_qkv="BNSD"：BSND 由 host 转置 + _dynamic_sequence_tensor 的 stride 表达，
    # 不走 run 内的 const_expr swap；fake 张量按实际 layout_qkv 编码 stride。
    block_num = _device_block_num(q)
    _kernel_key = (
        n_kv,
        n_v,
        dim,
        layout_qkv,
        dv_base,
        group,
        scratch_uses_sequence,
        block_num,
    )
    fn = _DYNAMIC_KERNEL_CACHE.get(_kernel_key)
    if fn is None:
        sym_batch = cannbotdsl.Dim("B")
        sym_seq = cannbotdsl.Dim("S", multiple_of=CHUNK_SIZE)
        scratch_tokens = _dynamic_workspace_tokens(
            sym_batch,
            sym_seq,
            n_v,
            group,
            scratch_uses_sequence,
        )
        gamma_rows = (scratch_tokens // CHUNK_SIZE) * dim
        fake = cannbotdsl.TensorSpec
        op = FlashKDA(
            dim,
            group=group,
            layout_qkv="BNSD",
            dv_base=dv_base,
            block_num=block_num,
        )
        fn = op.run.compile(
            _dynamic_sequence_tensor(dtypes.bfloat16, sym_batch, sym_seq, n_kv, dim, layout_qkv),
            _dynamic_sequence_tensor(dtypes.bfloat16, sym_batch, sym_seq, n_v, dim, layout_qkv),
            _dynamic_sequence_tensor(dtypes.bfloat16, sym_batch, sym_seq, n_kv, dim, layout_qkv),
            _dynamic_sequence_tensor(dtypes.bfloat16, sym_batch, sym_seq, n_v, 1, layout_qkv),
            _dynamic_sequence_tensor(dtypes.bfloat16, sym_batch, sym_seq, n_v, dim, layout_qkv),
            fake((block_num, 2, CHUNK_SIZE, D_HALF), dtypes.float32),
            fake((n_v,), dtypes.float32),
            fake((n_v, dim), dtypes.float32),
            fake((scratch_tokens, dim), dtypes.bfloat16),
            fake((gamma_rows, 1), dtypes.float32),
            fake((scratch_tokens, dim), dtypes.bfloat16),
            fake((scratch_tokens, dim), dtypes.bfloat16),
            fake((scratch_tokens, CHUNK_SIZE), dtypes.bfloat16),
            fake((scratch_tokens, dim), dtypes.bfloat16),
            _dynamic_sequence_tensor(dtypes.bfloat16, sym_batch, sym_seq, n_v, dim, layout_qkv),
            fake((sym_batch, n_v, dim, dim), dtypes.float32),
            fake((sym_batch, n_v, dim, dim), dtypes.float32),
            fake((CHUNK_SIZE, CHUNK_SIZE // 2), dtypes.float32),
            fake((16, 16), dtypes.float16),
            dtypes.int64,
            dtypes.int64,
            dtypes.int64,
            dtypes.int64,
            dtypes.float32,
            dtypes.float32,
        )
        _DYNAMIC_KERNEL_CACHE[_kernel_key] = fn
    execution = _EXECUTION_CACHE.get_or_create(
        ref=q,
        batch=batch,
        key_heads=n_kv,
        value_heads=n_v,
        seq_len=seq_len,
        eff_group=eff_group,
        dim=dim,
        block_num=block_num,
        layout_qkv=layout_qkv,
    )
    ws = execution.workspace
    with execution.launch_lock:
        fn(
            k, v, q, beta, g, execution.qk_exchange, A_log, dt_bias,
            ws.K_restored, ws.gamma_C, ws.Q_decayed, ws.W, ws.Mqk,
            ws.U_pre, out, initial_state, final_state, ws.lower_mask_tiles,
            ws.identity16, bn_total, seq_len, bn_total, 1, scale_value,
            float(lower_bound),
        )
    return out_physical, final_state
