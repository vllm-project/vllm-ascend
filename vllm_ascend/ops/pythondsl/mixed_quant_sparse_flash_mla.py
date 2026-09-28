import math
import os
import sys
import threading
import cannbotdsl
import torch
from enum import IntEnum

from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.types.delay_line import DelayLineGroup
from cannbotdsl.ops.arch import get_subblock_id, get_block_idx, get_subblock_dim
from cannbotdsl import select as dyn_select
from cannbotdsl.types import Int32, Int64
from cannbotdsl._mlir.dialects import arith, ascvec, cannir
from cannbotdsl.ir_transport import unwrap_operand
from cannbotdsl.ascir import extract_buffer
from cannbotdsl.core.scalar_conversion import coerce_scalar_value
from cannbotdsl import dtypes
from cannbotdsl.types import Float8E4M3FN
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.types import ChannelKind, MemLoc, PIPE, Tensor
from cannbotdsl.tensor import (
    ceil_div,
    partition_view,
    local_slice,
    tile_view,
    make_partition_tiler,
    make_tensor,
    make_layout,
    reinterpret,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl._mlir import ir
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.sync import (
    cube_raw_l1_to_l0a,
    cube_sync_block_arrive,
    cube_sync_notify,
    cube_sync_wait,
    cube_sync_intra_arrive,
    cube_sync_intra_wait,
    global_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_intra_arrive,
    vec_sync_intra_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.arena import channel_rewind
from cannbotdsl.ops import reg as rr

# ---- AICPU metadata 算子（同目录 mixed_quant_sparse_flash_mla_metadata.py）----
# 布局常量 / 核数查询 / host 入口自该模块导入；mixed_quant_sparse_flash_mla_metadata
# 在此 re-export，保持 ops.mixed_quant_sparse_flash_mla.* 的既有外部调用路径不变。
if __package__:
    from .mixed_quant_sparse_flash_mla_metadata import (
        AIC_CORE_MAX_NUM, MQSMLA_METADATA_TOTAL_SIZE, FA_METADATA_SIZE,
        FA_BN2_START_INDEX, FA_M_START_INDEX, FA_S2_START_INDEX,
        FA_BN2_END_INDEX, FA_M_END_INDEX, FA_S2_END_INDEX,
        FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, FD_METADATA_BASE, FD_METADATA_SIZE,
        FD_CORE_ENABLE_INDEX, FD_BN2_IDX_INDEX, FD_M_IDX_INDEX, FD_WORKSPACE_IDX_INDEX,
        FD_WORKSPACE_NUM_INDEX, FD_M_START_INDEX, FD_USED_VEC_NUM_WORD,
        _get_cube_core_num, mixed_quant_sparse_flash_mla_metadata,
    )
else:
    from mixed_quant_sparse_flash_mla_metadata import (
        AIC_CORE_MAX_NUM, MQSMLA_METADATA_TOTAL_SIZE, FA_METADATA_SIZE,
        FA_BN2_START_INDEX, FA_M_START_INDEX, FA_S2_START_INDEX,
        FA_BN2_END_INDEX, FA_M_END_INDEX, FA_S2_END_INDEX,
        FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, FD_METADATA_BASE, FD_METADATA_SIZE,
        FD_CORE_ENABLE_INDEX, FD_BN2_IDX_INDEX, FD_M_IDX_INDEX, FD_WORKSPACE_IDX_INDEX,
        FD_WORKSPACE_NUM_INDEX, FD_M_START_INDEX, FD_USED_VEC_NUM_WORD,
        _get_cube_core_num, mixed_quant_sparse_flash_mla_metadata,
    )

# ---- FP4 硬件解码 compile-hook（2026-09-16，ops-nn 式改造；默认启用）----
# cannbotdsl 0.5.0 的 MLIR→AscendC 翻译层没有 fp4x2_e2m1→bf16 的映射：vcast
# 前端可通过（full_mask 后改 elem_bits=4 绕过校验），但发射器产出不存在的
# asc_cast_unknown（发射 .so 的符号表中无 asc_e2m1x22bfloat16，C1/本轮二进制
# 检查双重实证）。本 hook 在 .asc 文本层把它替换为真机验证过的等价硬件调用
# asc_e2m1x22bfloat16(dst, src, mask, DISPERSE_FIRST_QUARTER)：单条 Q0 输出
# 即 [lo(b0),hi(b0),...] 全序（真机 bit 级 128/128，tuning/fp4new/）。替换只
# 在文本含 asc_cast_unknown 时发生（该符号仅本文件的新解码路径会发出）。
# （历史开关 MQSMLA_FP4_VCVT / MQSMLA_FP4_ASC_DUMP 已于 2026-09-18 移除，
#   新解码路径与 hook 无条件启用。）


# ---- 地址向量化（2026-09-17，tune/20260917-addr-vec；参考 ops-transformer ----
# arch35 的 IS_VEC_S2PHYADDR 前置预计算 + cannbot-arena main samples/mqsmla 的
# cannbotdsl 移植）。
#
# 主循环前 AIV 两扫（ori/cmp）做一趟前置 pass——sp 行 staging + VF 分块
# （vshr/vgather/vmuls，列按 w 钳位）+ u32 行号写 GM 表 [T1, K1+K2]，block 内
# AIV 对握手，消费侧逐 token 读表回放 pair-DMA。
#
# **唯一开关**：`ADDR_VEC_ENABLED`，默认 On。置 False（或环境变量
# MQSMLA_ADDR_VEC=0）整体回退基线标量路径（逐 token _pa_blk_off 现算），
# 用于 A/B 对比。此外形状不满足资格（bs 非 2 的幂、池 stride != bs*行距、
# 不可字面量特化、UB 装不下）时，_get_compiled_kernel 在编译期静默回退同一条
# 标量路径——这是保护性的，不是开关。
#
# bit 级等价依据：addr = blk*page_stride + off*ROW_BYTES == (blk*bs+off)*ROW_BYTES
# （page_stride == bs*ROW_BYTES 时），整数恒等，钳位语义逐位同值。所以标量路径
# 是向量路径在任意形状/任意 K 下的合法逐位参考。
ADDR_VEC_ENABLED = os.environ.get("MQSMLA_ADDR_VEC", "1") != "0"

# _get_compiled_kernel 在编译前按形状资格改写它（不满足回退标量路径），
# 编译锁内安全；锁外恒为 None。
_ADDR_VEC_OVERRIDE = None


def _addr_vec_mode():
    if _ADDR_VEC_OVERRIDE is not None:
        return _ADDR_VEC_OVERRIDE
    return "pre" if ADDR_VEC_ENABLED else "off"


def _addr_vec_eligible(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
    """向量臂形状资格：bs 2 的幂、池 stride==bs*行距（行号×行距恒等的前提）。

    返回 (ok, bs_ori, btp_ori, k1, bs_cmp, btp_cmp, k2)；不满足时 ok=False，
    本次编译回退标量基线路径（符号维规格）。
    """
    def pool_ok(kv, row_bytes):
        bs = kv.shape[1]
        return (isinstance(bs, int) and bs >= 16 and bs & (bs - 1) == 0
                and kv.stride(0) == bs * row_bytes), bs

    if not isinstance(ori_bt.shape[1], int) or not isinstance(ori_idx.shape[2], int):
        return False, None, None, None, None, None, None
    ok_w, bs_w = pool_ok(ori_kv, KV_ROW_BYTES_ORI)
    ok = ok_w
    bs_c, btp_c, k2 = None, None, None
    if cmp_kv is not None:
        if not isinstance(cmp_bt.shape[1], int) or not isinstance(cmp_idx.shape[2], int):
            return False, None, None, None, None, None, None
        ok_c, bs_c = pool_ok(cmp_kv, KV_ROW_BYTES_CMP)
        ok = ok and ok_c
        btp_c, k2 = cmp_bt.shape[1], cmp_idx.shape[2]
    btp = ori_bt.shape[1] if cmp_kv is None else max(ori_bt.shape[1], btp_c)
    sp_w = ori_idx.shape[2] if cmp_kv is None else max(ori_idx.shape[2], k2)
    # _ch_addr_out 按补齐宽度分配（见 _addr_tab_w），比 sp_w 最多多 _TILE_ROWS-1，
    # UB 预算必须按补齐后的值算，否则 K 接近上限时会超 UB 容量。
    out_w = (_addr_tab_w(ori_idx.shape[2]) if cmp_kv is None else
             max(_addr_tab_w(ori_idx.shape[2]), _addr_tab_w(k2)))
    if (btp + sp_w + out_w) * 4 > _ADDR_VEC_UB_HEADROOM:
        return False, None, None, None, None, None, None
    return ok, bs_w, ori_bt.shape[1], ori_idx.shape[2], bs_c, btp_c, k2


import re as _re
from cannbotdsl.core.toolchain import ascendc as _asc_toolchain

_FP4_Q0_ARG = ("std::integral_constant<asc_position_quarter_mode, "
               "asc_position_quarter_mode::DISPERSE_FIRST_QUARTER>{}")
_FP4_CAST_PAT = _re.compile(r"asc_cast_unknown\(([^;]+?)\);")
_fp4_orig_compile = _asc_toolchain.AscendCCompiler.compile_ascendc_to_so

def _fp4_vcast_compile(self, c_code, **kw):
    if "asc_cast_unknown" in c_code:
        c_code = ('#include <type_traits>\n#include "c_api/asc_simd.h"\n'
                  + c_code)
        c_code = _FP4_CAST_PAT.sub(
            rf"asc_e2m1x22bfloat16(\1, {_FP4_Q0_ARG});", c_code)
    return _fp4_orig_compile(self, c_code, **kw)

_asc_toolchain.AscendCCompiler.compile_ascendc_to_so = _fp4_vcast_compile
# _NZ_CHUNK 是 face 间距（BF16 元素数）；padding 只存在于 UB，copy-out 时跳过。
_NZ_ROW = 16
_NZ_PAD_ROWS = 17
_NZ_CHUNK = _NZ_PAD_ROWS * _NZ_ROW
# 输出落地手动同步的 HardEvent id（与 sync_intra id / sync_block flag id 空间相互独立）：
# finalize 的 cast(V) → out 拷贝(MTE3) 只需这一对「MTE3 等 V」；空行路径（drain 段连续
# 写零、无 AIC 生产链门控）在覆写前额外补一对「V 等 MTE3」排空上一笔出拷贝。
_OUT_V_TO_MTE3_EVENT_ID = 0
_OUT_MTE3_TO_V_EVENT_ID = 1

# 地址向量臂在基线 UB 预算之上额外占用 UB：_ch_addr_bt(btp) + _ch_addr_sp(sp_w)
# + _ch_addr_out(sp_w)，各 int32（(1,N) Channel，按字节紧凑排布，无额外 pad）。
# 256 KiB 片上 UB 扣掉标量基线固定预算后，剩下的头空间就是向量臂可用的上限。
# 形状（block_table 宽 btp、稀疏列宽 sp_w）过大时须回退标量路径，否则
# cannir-plan-onchip-memory 报 "UB allocation exceeds the 262144-byte capacity"。
_UB_CAPACITY_BYTES = 262144
_UB_BASE_BUDGET_BYTES = 250624
_ADDR_VEC_UB_HEADROOM = _UB_CAPACITY_BYTES - _UB_BASE_BUDGET_BYTES  # 11520 B

def _to_index(value):
    return coerce_scalar_value(value, ir.IndexType.get())

# ---- 编译期形状契约（冻结值，见接口文档「约束说明」）----
D = 512                  # 逻辑 head dim：nope 448 + rope 64（计算序 nope 在前）
N1 = 64                  # Q 头数唯一取值（SPLIT_G=128 与 runtime_n1 已删，2026-09-07）
N2 = 1                   # KV 头数（G=N1/N2=64）
TILE_N = 128             # n-tile：一个 s2 任务的 KV token 数
WS_DEPTH = 3             # GM ws 三深环：复用前借 softmax(T−3) 确认 loadQK(T−3) 已完成
D_ROPE = 64              # rope 段元素数（位于 nope 之后）
D_NOPE = 448             # nope 段元素数
GROUP_SIZE_ORI = 32
GROUP_SIZE_CMP = 16
NUM_GROUPS_ORI = D // GROUP_SIZE_ORI
NUM_GROUPS_CMP = D // GROUP_SIZE_CMP

# ---- per-token-group 物理布局 ----
# 物理字节序 = nope[448] ‖ rope[64]；ori 每 32 元素、cmp 每 16 元素共享 1 个 BF16 scale。
# scale 内联在数据段尾、无 pad 字段，均以 uint8 字节视图传入：
#   ori_kv (FP8 E4M3)    544 B/token = 512B fp8 数据 + 32B scale
#   cmp_kv (FP4 E2M1 x2) 320 B/token = 256B fp4x2 数据（偶元素低 nibble）+ 64B scale
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
KV_SCALE_BYTES_ORI = NUM_GROUPS_ORI * 2
KV_SCALE_BYTES_CMP = NUM_GROUPS_CMP * 2
ROW_BF16_ORI = KV_ROW_BYTES_ORI // 2      # ori 池 bf16 视图行宽 272
ROW_BF16_CMP = KV_ROW_BYTES_CMP // 2      # cmp 池 bf16 视图行宽 160
SCALE_BF16_OFF_ORI = 256                  # scale 段在 ori 行内的 bf16 偏移（512B/2）
SCALE_BF16_OFF_CMP = 128                  # scale 段在 cmp 行内的 bf16 偏移（256B/2）

class QuantMode(IntEnum):
    CONTIGUOUS = 1

class Scenario(IntEnum):
    ORI_SPARSE = 0
    ORI_CMP_SPARSE = 1

# ---- FD workspace staging（区域布局 [O slots][max slots][sum slots]，对齐
# AscendC S2SplitFdStagingLayout；常规 FD 每核最多 2 份分部：首行续切 + 尾行起切）----
# metadata 采用 int32[1024] 固定布局（FA 9×36 ‖ FD 8×72 ‖ fdUsedVecNum），
# 分核计划与 FD 归约计划由 mixed_quant_sparse_flash_mla_metadata 在 host 侧生成。
FD_SLOTS_PER_CORE = 2
FD_O_ELEMS = N1 * D                       # 每 slot O 区：64×512 fp32
FD_MS_ELEMS = 2 * N1                      # 每 slot max[64] + sum[64] fp32
FD_SLOT_ELEMS = FD_O_ELEMS + FD_MS_ELEMS  # 32896 fp32 = 131584 B
V0_BYTES_PER_CORE = WS_DEPTH * TILE_N * D * 2          # 393216
FD_BYTES_PER_CORE = FD_SLOTS_PER_CORE * FD_SLOT_ELEMS * 4  # 263168
WS_BYTES_PER_CORE = V0_BYTES_PER_CORE + FD_BYTES_PER_CORE  # 656384

# ⚠️ UB 妥协（2026-09-19，后续要改掉）：
# 为保输出路径不退化（decode 常规 case 不差于基线），FD 常驻 UB 被压到
# 极限（总 261888/262144 B，仅 256 B 余量），只能把下面两个量压到最小：
#   _FD_CHUNK_ROWS=1：FD 搬运分块行数。ascendc 参考是 16，fd Python 中间版是 8；压到 1 后
#     fd_ub=(1,512)=2KiB，但归约/stage 变成 32 片小搬运，FD 路径最慢（实测 long-s2 贡献
#     约 24us 额外开销）。要抬回 ≥4 需先释放 addr-vec 的 8KiB Channel（把主循环 buffer 也搬
#     进 __call__ 用 channel_rewind，或 dsl.UB.view 手工分区）。
#   _FD_MP_SLOTS=8：FD 归约 m_p/t_p 缓存槽数。fd 原分支开 24（覆盖 cmp_topk≤2944）；压到 8
#     只覆盖 k≤8（当前 ori_topk=128/cmp_topk=512 时 k≤5，够用，但更大 topk 会越界）。
# 2026-09-21 起输出改为 res_o 原地 cast（out_ub Channel 已删，见 _OUT_*_EVENT_ID 注释），
# 常驻 UB 释放 16 KiB，FD 抬块的重评估余量已具备（单变量原则，另行实验）。
# 对应地 metadata 侧 SUPPORT_FD 已置 False（FD 暂不开），见 mixed_quant_sparse_flash_mla_metadata.py。
_FD_CHUNK_ROWS = 1                        # FD 搬运分块行数：(1,512) fp32 = 2 KiB UB
_FD_NEG_INF = -1.0e38                     # 非首分部 softmax 种子（exp 下溢为 0）
_FD_MP_SLOTS = 8                          # FD 归约 m_p 缓存槽数（非 batch-consistency 每行至多 5 份分部）

# FD 编译期开关（与 metadata 的 SUPPORT_FD 一致）。False 时 FD 专属 buffer（fd_ub 2 KiB +
# fd_mp_tb 1 KiB）与 stage_partial/fd_reduce 整段不参与编译，把这 3 KiB UB 归还给主循环
# （例如 pv_ub depth 1→2 的双缓冲）。注意：metadata 侧 SUPPORT_FD 也必须保持 False，否则
# 运行时 fd_any!=0 会进入未编译的 FD 分支。
_FD_COMPILE = False

# AIV→L1 直写 K̃（省掉 GM ws 往返的 mte2 1.13us/tile）。True 时 vec0 把反量化后的 K̃ 用
# copy_ub2l1_strided 直接写进 L1 kv_wide 的 tick%3 槽，load_qk 不再做 GM→L1；跨 AIV↔AIC
# 同步改成 ready(MTE3→MTE1) + free(MTE1→MTE3) 两对 intra sync。False 时走 GM ws 往返基线。
# 注意：copy_ub2l1_strided 的 burst_len/src_gap/dst_gap 单位是 32B datablock（不是字节/元素）。
_AIV2L1 = True

def _ws_rows(blocks):
    # GM workspace 的 v0 区行数：核数 × ws_depth(3) × tile_n(128) 行，每行 512 bf16。
    return blocks * WS_DEPTH * TILE_N

# ori 侧派生偏移（kernel 侧缺省口径；cmp 侧在 `_vec0_body` 内按池分派）：
_ROW_BF16 = ROW_BF16_ORI          # ch_batch 槽行距 272 bf16（两池统一按 ori 的 544B 槽行距）
_SCALE_BF16_OFF = SCALE_BF16_OFF_ORI  # scale 段在 ori 槽行内的 bf16 偏移

_VL = 128                  # vec0 反量化的单笔向量宽度（bf16 lanes）
_DEFAULT_BLOCK_SIZE = 64
_TILE_ROWS = 16            # vec0 子块行数：一个 128-token tile 切 8 个 16 行子块（分形 face 高）
_ADDR_HALF = 64            # 地址向量化：VF 每批算的行号宽度（列宽 64 lane）
# B4-(1)：batch 已跑到最后一个 span 时的边界哨兵。任何合法 m_idx <= T1 都小于它，
# 所以 while 条件恒假 ⇒ 与 _batch_of 的 hi=b_rt 钳位等价，且不会越界读 cu_q。
_ADDR_M_INF = 1 << 40


def _addr_tab_w(k):
    """行号表里每侧实际要写的列数：真实 K 按 _TILE_ROWS 上取整。

    消费侧 vec 臂**不做** last_off 钳位（`_vec0_body` 的向量分支直接按列取，
    钳位是 pass 侧的职责），而它每个 16 行子块恒读满
    sub_row..sub_row+_TILE_ROWS-1 列，于是最多越过真实 K 读到
    align_up(K, _TILE_ROWS)-1。只写 K 列时这些越界列会落进相邻 side 的
    区段（K1=133 时 ori 读到第 143 列 = cmp 区），静默取到别的池的行号。
    pass 侧把这些列一并按 w-1 钳位写出，与标量分支 min(ii, last_off)
    逐位同值。K 本身是 _TILE_ROWS 整数倍时返回 K，与修复前逐位相同
    （128/512 走这一支）。
    """
    return -(-k // _TILE_ROWS) * _TILE_ROWS


def _raw_ch2gm(gm, ch, elem_off, n_elems):
    """Channel 槽的 UB→GM 单笔搬运（读括号消费，与 write 括号 1:1）。"""
    ub_base, ub_off = extract_buffer(ch, access="read")
    gm_base, gm_off = extract_buffer(gm, access="write")
    ascvec.copy_ub2gm(gm_base, arith.addi(gm_off, _to_index(elem_off)),
                      ub_base, ub_off,
                      Int32(1), Int32(n_elems * 4),
                      dtypes.int64(0), Int32(0))


def _raw_gm2ch(ch, gm, elem_off, burst_bytes):
    """Channel 槽的裸 GM→UB 单笔搬运（write 括号生产；同步由 Channel 锁承担）。

    elem_off 按缓冲 dtype 的【元素】计——extract_buffer 返回类型化指针，
    指针加法是元素步长（板上实证：i32 表上传字节偏移会 ×4 走到第 4m 行）。
    burst_bytes 才是字节。
    """
    ub_base, ub_off = extract_buffer(ch, access="write")
    gm_base, gm_off = extract_buffer(gm, access="read")
    ascvec.copy_gm2ub(ub_base, ub_off,
                      gm_base, arith.addi(gm_off, _to_index(elem_off)),
                      Int32(1), Int32(burst_bytes),
                      dtypes.int64(0), Int32(0), Int32(0), Int32(0))


def _i64(gm, *idx):
    raw = gm[idx] if len(idx) > 1 else gm[idx[0]]
    # GM 索引为 int32；在物理地址乘法前显式扩展到 int64。
    return dtypes.int64(raw)

class Kvcache:
    """PA_BBND 两级寻址（blockTable 值是全局物理块号，batch 只在下标上）：
    blk = s2_idx // blockSize; off = s2_idx % blockSize;
    phys_row = blockTable[bo][blk] * blockSize + off
    两侧均按运行时 blockSize 寻址，不要求页大小是 2 的幂。
    """

    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
        self.block_size = int(block_size)

    def get_key_offset(self, gm_block_table, s2_idx, bo_idx=0, block_size=None):
        bs = self.block_size if block_size is None else block_size
        phys_blk, off = self._pa_blk_off(gm_block_table, s2_idx, bo_idx, bs)
        return phys_blk * bs + off

    @jit
    def _pa_blk_off(self, gm_block_table, s2_idx, bo_idx, bs):
        # Common page size avoids runtime division; all other sizes remain valid.
        blk = dtypes.int64(0)
        off = dtypes.int64(0)
        if bs == 128:
            blk = s2_idx // 128
            off = s2_idx % 128
        else:
            blk = s2_idx // bs
            off = s2_idx % bs
        return _i64(gm_block_table, bo_idx, blk), off

TILE = 128
D_TILES = D // TILE          # D=512 切 4 个 128 列 chunk（bmm1 的 K 轴 / bmm2 的 N 轴）

# ---- 每 AIC 的 L1 预算：K̃ 384 KiB + Q 96 KiB + p_l1 32 KiB = 512 KiB ----
# GM ws 用 tick%3，L1 K̃ 按 tick%3 轮转；ws 在 loadQK 后即可复用，
# L1 K̃ 的消费范围覆盖 QK 和 PV。Q 按半 D 装载，在一个 M 内常驻，以事件保护代际轮转。
_KV_RING = 3                 # K̃ 环深度；一槽保存一份供 QK/PV 共读的 KV tile

_Q_RING = 3                  # Q Buffer 的 3 个 [64,256] BF16 半槽，共 96 KiB
_Q_SLOT_ELEMS = N1 * (D // 2)
_Q_CHUNK_ELEMS = N1 * TILE
_Q_EVENT_BASE = 3            # KV 使用 0..2；Q 三个物理半槽使用 3..5

_P_L1_DEPTH = 2              # p_l1 跨核 Channel 深度（AIV store_p → AIC bmm2）

# ---- ChannelArena id 纪律（烧过三次的教训：手挑 id 曾「不报错、只间歇挂死」）----
# sync_id 全空间 [0,15]：0..4 跨核 Channel 自取；本侧向 arena 申请并烧号到地板 5 让开；
# 保留段 {11..14}（AscendC 运行时）与 {10}（板上负面证据）整段避让。
# block-sync flag 必须在所有 Channel 构造之后申请（先申请会顶走 Channel 的 sync_id）。
_PLANNER_SYNC_ID_FLOOR = 8

def _burn_to_floor(alloc, floor, *alloc_args):
    while True:
        got = alloc(*alloc_args)
        if min(got) >= floor:
            return got

_AIV1_ID_OFFSET = 16

def _alloc_ws_intra_ids():
    from cannbotdsl.arena import _current_channel_arena
    arena = _current_channel_arena()
    return _burn_to_floor(arena.alloc_sync_ids, _PLANNER_SYNC_ID_FLOOR, 1)

_KV_NUM_PER_LOOP = 128
_TILE_SIZE = 64
_COMBINE_DIM = _ROW_BF16 * 2

class Matmul:
    """AIC 侧：把 Q/K̃ 装入 L1，完成 QK、PV 两次矩阵乘并经 fixpipe 送 AIV。

    q_wide 是包含 3 个 [64,256] BF16 半槽的 Buffer（96 KiB）。
    Q 每个 M 只加载一次，两个 K chunk 共读半槽；事件管理 MTE2/MTE1 同步。
    kv_wide 保存 3 份 [128,512] BF16 K̃（384 KiB），同一份数据依次用于 QK 和 PV，
    KV 事件 覆盖 QK/PV，直到 PV 的 L1→L0B 装载结束才释放。
    K̃ 采用逻辑 (token, 3*D) 的 NZ Buffer，按 D 列切出每槽；GM workspace 的 (D/16, token, 16)
    分形字节通过 NZ 视图原样搬入 L1。QK/PV 共用两个 [64,512] FP32 L0C 槽；
    同一个 Channel 管理写入、FIXPIPE 消费和跨迭代轮转，QK 只使用槽的前 128 列。
    """

    def __init__(self, tile_cube_m, tile_vec_m, tile_n):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D

        self.nd2nz = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.identity = make_copy_engine(format_transform="identity", dtype=dtypes.bfloat16)
        self.fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

        # Q 按 M 常驻 L1：一代占连续两个半 D 槽，槽基 (2*m)%3。
        # 三槽 credit 保护下一 M 的 MTE2 覆写与上一 M 末 tile 的 MTE1 读取。
        self.q_wide = Buffer(MemLoc.L1, (tile_cube_m, _Q_RING * (D // 2)),
                             dtypes.bfloat16)
        self.kv_wide = Buffer(MemLoc.L1, (tile_n, _KV_RING * D), dtypes.bfloat16,
                              data_format="nz")
        self.l0a = Channel(MemLoc.L0A, shape=(tile_cube_m, TILE), dtype=dtypes.bfloat16, depth=2)
        self.l0a_p = Channel(MemLoc.L0A, shape=(tile_cube_m, tile_n), dtype=dtypes.bfloat16, depth=2)
        self.l0b = Channel(MemLoc.L0B, shape=(TILE, TILE), dtype=dtypes.bfloat16, depth=2)
        # Match AscendC: one shared ring, two 128 KiB slots, M/FIX ownership.
        self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, D), dtype=dtypes.float32,
                           depth=2, kind=ChannelKind.SameCore)

    @jit
    def kv_event(self, tick: int, src_pipe, dst_pipe, notify):
        # Event IDs are compile-time constants; branch on the runtime ring slot.
        # These pipe pairs are KV-only; Channels use logical buffer locks.
        for slot in range_constexpr(_KV_RING):
            if tick % _KV_RING == slot:
                if const_expr(notify):
                    cube_sync_notify(src_pipe, dst_pipe, slot)
                else:
                    cube_sync_wait(src_pipe, dst_pipe, slot)

    @jit
    def kv_events_init(self):
        # Every slot starts free, including on cores with no work.
        for slot in range_constexpr(_KV_RING):
            cube_sync_notify(PIPE.MTE1, PIPE.MTE2, slot)

    @jit
    def kv_events_drain(self):
        # PV has returned every used slot; consume unused initial events too.
        for slot in range_constexpr(_KV_RING):
            cube_sync_wait(PIPE.MTE1, PIPE.MTE2, slot)

    def kv_slot(self, tick):
        return tile_view(self.kv_wide, (self.tile_n, D), (0, tick % _KV_RING))

    @jit
    def _q_slot(self, m_seq, half):
        return (m_seq * 2 + half) % _Q_RING

    @jit
    def q_event(self, m_seq, half, src_pipe, dst_pipe, notify):
        # AscendC INNERCORE_L1Q assigns one hard event to each physical half
        # slot. Branching keeps event IDs compile-time constants in the DSL.
        q_slot = self._q_slot(m_seq, half)
        for slot in range_constexpr(_Q_RING):
            if q_slot == slot:
                if const_expr(notify):
                    cube_sync_notify(src_pipe, dst_pipe,
                                     _Q_EVENT_BASE + slot)
                else:
                    cube_sync_wait(src_pipe, dst_pipe,
                                   _Q_EVENT_BASE + slot)

    @jit
    def q_events_init(self):
        for slot in range_constexpr(_Q_RING):
            cube_sync_notify(PIPE.MTE1, PIPE.MTE2, _Q_EVENT_BASE + slot)

    @jit
    def q_events_drain(self):
        for slot in range_constexpr(_Q_RING):
            cube_sync_wait(PIPE.MTE1, PIPE.MTE2, _Q_EVENT_BASE + slot)

    @jit
    def load_q(self, q_tile_gm, m_seq):
        # Each half follows AscendC's wait-copy-notify sequence independently,
        # allowing bmm1 to start from half 0 while half 1 readiness is pending.
        for h in range_constexpr(2):
            self.q_event(m_seq, h, PIPE.MTE1, PIPE.MTE2, False)
            dst = tile_view(self.q_wide, (self.tile_cube_m, D // 2),
                            (0, self._q_slot(m_seq, h)))
            src = tile_view(q_tile_gm, (self.tile_cube_m, D // 2), (0, h))
            mem_copy(dst, src, engine=self.nd2nz)
            self.q_event(m_seq, h, PIPE.MTE2, PIPE.MTE1, True)

    @jit
    def _bmm1_chunk_fanout(self, qk_slot, kv_slot, j, m_seq, init):
        q_off = (self._q_slot(m_seq, j // 2) * _Q_SLOT_ELEMS
                 + (j % 2) * _Q_CHUNK_ELEMS)
        cube_raw_l1_to_l0a(self.l0a, self.q_wide, l1_offset_elems=q_off,
                           m_start=0, k_start=0, m_step=self.tile_cube_m // 16,
                           k_step=TILE // 16, src_stride=self.tile_cube_m // 16,
                           dst_stride=self.tile_cube_m // 16)
        mem_copy(self.l0b, tile_view(kv_slot, (self.tile_n, TILE), (0, j)))
        matmul(qk_slot, self.l0a, self.l0b, init=init)

    @jit
    def load_qk(self, k_tile_gm, row_tile, kv_slot, q_tile_gm, m_seq, tick,
                ws_ready_id):
        # Match AscendC IterateLoadQK issue order exactly: bootstrap Q first,
        # acquire the KV write slot, wait for both Vec0 workspace producers,
        # copy KV, then publish KV ready. Subsequent Q loads are still issued
        # by the preceding M's bmm1-tail prefetch.
        if tick == dtypes.int64(0):
            self.load_q(q_tile_gm, m_seq)
        if const_expr(_AIV2L1):
            # AIV→L1：K̃ 已由 vec0 直写进 L1 槽，load_qk 只剩 bootstrap Q，不再做 GM→L1；
            # ready/free 握手在 vec0/_stage_qk_softmax/compute_pv_fanout 三处完成。
            return
        self.kv_event(tick, PIPE.MTE1, PIPE.MTE2, False)
        cube_sync_intra_wait(PIPE.MTE2, ws_ready_id)
        cube_sync_intra_wait(PIPE.MTE2, ws_ready_id + _AIV1_ID_OFFSET)
        # lag1：GM ws → L1 K̃ 环；一次原样搬运生产一个完整 KV tile。
        # GM 已去除 UB padding；[32,128,16] 分形与 [128,512] 的 NZ 物理布局一致。
        # workspace 中各 tile 是独立连续的 NZ 块；索引前导维只选择块，不重排数据。
        ws_tiles = k_tile_gm.shape[0] // self.tile_n
        ws_nz = make_tensor(cannir.extract_pointer(unwrap_operand(k_tile_gm)),
                            make_layout((ws_tiles, self.tile_n, D),
                                        stride=(self.tile_n * D, D, 1)),
                            data_format="nz")
        rt = 0 if row_tile is None else row_tile
        src = tile_view(ws_nz[rt, None, None], (self.tile_n, D), (0, 0))
        # GM 已是 NZ 字节序，identity 仅作 ND2ND 原样搬运，不做 ND2NZ。
        mem_copy(kv_slot, src, engine=self.identity)
        self.kv_event(tick, PIPE.MTE2, PIPE.MTE1, True)

    @jit
    def bmm1_fanout_q(self, kv_slot, m_seq, is_first, is_last):
        # lag2：S[64,128] = Q[64,512] @ K̃[128,512]^T，FP32 L0C 累加。
        # 每个 M 的本核首个 tile 取得 Q 读 credit，本核末 tile 归还；中间 tile 直接复用 L1。
        # FD 切分行的分部起点/终点即 credit 边界（is_first/is_last 由 metadata 范围派生）。
        # QK 等待 K̃ 装载就绪，PV 读完后通知槽可复用。
        qk_dst = tile_view(self.l0c, (self.tile_cube_m, TILE), (0, 0))
        for j in range_constexpr(D_TILES):
            if const_expr(j % 2 == 0):
                if is_first == dtypes.int64(1):
                    self.q_event(m_seq, j // 2, PIPE.MTE2, PIPE.MTE1, False)
            self._bmm1_chunk_fanout(qk_dst, kv_slot, j, m_seq, init=(j == 0))
        if is_last == dtypes.int64(1):
            for h in range_constexpr(2):
                self.q_event(m_seq, h, PIPE.MTE1, PIPE.MTE2, True)

    def store_s(self, qk_ub_ch, partition):
        # S 从 L0C 经 fixpipe（dual_dst、split-M partition）跨核送到 AIV 的 qk_ub。
        # split-M 将 [64,128] 的 64 个头对半分给两个 AIV。
        # AIV0 处理 head[0:32]，AIV1 处理 head[32:64]，store_p/finalize 保持同一划分。
        mem_copy(qk_ub_ch, tile_view(self.l0c, (self.tile_cube_m, TILE), (0, 0)),
                 engine=self.fixpipe, partition=partition)

    def compute_pv_fanout(self, p_l1_ch, pv_ub_ch, partition, kv_slot, actual_n, tick):
        # lag3：PV[64,512] = P[64,128] @ K̃[128,512]；此时 P 是 exp(score−running_max)，
        # 尚未除以分母。D 切为 4 个独立的 128 列输出 chunk，各自 init=True。
        # K̃ 复用 QK 的 L1 slot，通过 L0B 的 transpose=True 改变矩阵乘的读取方向。
        # P 只装入 L0A 一次并供 4 个 chunk 共读；各列块先写入完整 L0C。
        # 一次 FIXPIPE 写出完整 PV，Channel 自动管理跨核生产和消费。
        # 最后一次 MTE1 读取后发布 KV free，再将整块 PV 通过 FIXPIPE 写出。
        p_slot = p_l1_ch
        a_slot = self.l0a_p
        mem_copy(a_slot, p_slot)

        # Only reduce initialized 16-row subblocks; masked P=0 cannot suppress NaN V.
        pv_k = ceil_div(actual_n, _TILE_ROWS) * _TILE_ROWS
        a_view = local_slice(self.l0a_p, (self.tile_cube_m, pv_k),
                             stride=(self.tile_n, 1))
        for n in range(D_TILES):
            mem_copy(self.l0b, tile_view(kv_slot, (self.tile_n, TILE), (0, n)),
                     transpose=True)
            b_view = local_slice(self.l0b, (pv_k, TILE), stride=(TILE, 1))
            col = tile_view(self.l0c, (self.tile_cube_m, TILE), (0, n))
            matmul(col, a_view, b_view, init=True)
        # Return KV after the final MTE1 read, before FIXPIPE can wait on AIV.
        # The hard event orders all preceding L1 reads before a slot overwrite.
        if const_expr(not _AIV2L1):
            self.kv_event(tick, PIPE.MTE1, PIPE.MTE2, True)
        mem_copy(pv_ub_ch, self.l0c, engine=self.fixpipe, partition=partition)

class Vector:
    """AIV 侧：每核负责同一 query 的 32 个头，完成 online softmax 与输出累加。

    对一个头，设旧状态为 (m,l,o)，当前 tile 的缩放后 score 为 s：
      m' = max(m, tile_max)，alpha = exp(m−m')，p_j = exp(s_j−m')；
      l' = alpha*l + Σ_j p_j，o' = alpha*o + BF16(p) @ K̃。
    仅有效前缀参与指数与求和；残块的 tile_max 还可能包含 masked score 的零占位。
    sink 播种 (m=sink,l=1,o=0)，末 tile 才输出 BF16(o/l) 与 log(l)+m。
    sink 是未乘 softmax_scale 的 logit；传 0 仍贡献 exp(0)=1 的分母项。

    max/sum 属于 query（m_seq%2），alpha 属于 KV tile（tick%2）。两套双槽分别防止
    下一 query 的 softmax 覆盖上一 query 收尾所需的分母，以及相邻 tile 覆盖 alpha。
    res_o 只用一块 UB：lag2 按任务顺序消费，上一 query finalize 完才开始下一 query。
    """

    def __init__(self, tile_vec_m, tile_n, tile_d, subblock_idx, sinks_span):
        self.sinks_span = sinks_span
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # max/sum 按 query、exp 按任务轮转；索引均为 %2，各分配两槽。
        self.sm_max_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        self.sm_sum_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        self.sm_exp_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        # FD 归约第 1 遍把各分部 m_p 全部缓存（供第 2/4 遍复用，去 m_p 重复 load）。
        # 分部数 k = 一行被跨核切分的份数；非 batch-consistency 下每行至多
        # ceil(ori_topk/s2_base) + ceil(cmp_topk/s2_base) = 5 份（ori=128,cmp=512，
        # s2_base=128），开 8 槽留余量。每槽 (32,1) fp32。
        # 此 __init__ 槽是死代码：真正的槽在 _setup_fd_reduce_buffers（barrier 后）重建；
        # _FD_COMPILE=False 时不分配，归还 1 KiB UB。
        self.fd_mp_tb = []
        if _FD_COMPILE:
            self.fd_mp_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
                             for _ in range(_FD_MP_SLOTS)]

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.lse_ub = Channel(MemLoc.UB, shape=(tile_vec_m, 1), dtype=dtypes.float32, depth=1)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)
        # 输出落地 = res_o 前 32 KiB 的 BF16 同储别名：cast 原地写入，不再另设 out_ub
        # Channel（其 depth=1 槽复用会在两次写出间引入 V 等 MTE3，打断 AIV 流水）。
        # FP32(32,512)=64 KiB ≥ BF16(32,512)=32 KiB，别名不新增 UB 分配。
        self.out_view = reinterpret(self.res_o, dtypes.bfloat16,
                                    (tile_vec_m, tile_d))

        p_n1_pad = 32 // 2
        self.p_ub = Channel(
            MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.bfloat16, depth=2,
            data_format="nz", n1_pad=p_n1_pad,
        )
        self.sinks_ub = Channel(MemLoc.UB, shape=(1, self.sinks_span), dtype=dtypes.float32, depth=1)

    def _nz_params(self):
        # 从 p_ub 的真实物理 stride 推导落点，包含 Channel 的 NZ padding；
        # 此布局与 vec0 的 NZ17 由不同对象管理，不能直接套用 _NZ_CHUNK。
        s = self.p_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return m0, s_m1, s_m0, s_n1 // n0

    def _softmax_fold_row(self, qk_ch, base, nz_off, max_brc_buf, row,
                          ve_mask, vo_mask, b16, b16_full, block_stride):
        # softmax 第二遍（行内）：exp(s−max) 后把 f32 偶/奇 lane 各 cast bf16（RegLayout
        # ZERO/ONE 分别落偶/奇 b16 lanes），vbitwise_or 合并成 128 连续 bf16 = P，
        # 按 NZ 分形 vstore_strided 存 p_ub。返回 (ve, vo) 供行求和。
        # 指数掩码按偶/奇列数 ceil(actual_n/2)、floor(actual_n/2) 分开，兼容奇数尾。
        # 最后用 b16_full 写满 128 列，将尾部 P=0 一并写入，避免复用槽里的旧 P 残留。
        mx = rr.vload_broadcast(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, dtypes.bfloat16, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, dtypes.bfloat16, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16)
        rr.vstore_strided(self.p_ub, nz_off, merged, b16_full, block_stride=block_stride, repeat_stride=0)
        return ve, vo

    def _pass_a_row(self, qk_ch, scale, sm_max_dst, row, row_stride, VL_T,
                    half0_mask, half1_mask, full_mask, half_only=False):
        # 第一遍：有效 score 乘 scale 并原位回写，同时求用于稳定指数的行 max。
        # 每行分成前/后各 64 列；vmuls 的非活跃 lane 置零，而 max 用 full_mask 归约，
        # 因此残块的零也可能抬高 max（不等价于用 −inf 填尾）。后续指数和 P 仍按
        # actual_n 屏蔽尾列。half_only 只在 actual_n 为编译期整数且 ≤64 时裁掉后半。
        base = row * row_stride
        if const_expr(half_only):
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rmax = rr.vreduce_max(v0, mask=full_mask)
        else:
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            v1 = rr.vmuls(rr.vload(qk_ch, base + VL_T), scale, mask=half1_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rr.vstore(qk_ch, base + VL_T, v1, half1_mask)
            rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
        rr.vstore_first(sm_max_dst, row, rmax)

    @staticmethod
    def _half_only(actual_n, VL_T):
        return isinstance(actual_n, int) and actual_n <= VL_T

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        # 用本 tile 的新 max 合并归一化分母：alpha = exp(old_max−new_max)，
        # new_sum = alpha * old_sum + Σ_j exp(score_j−new_max)。
        # alpha 存入本任务的 sm_exp 槽，供 lag3 将旧的 O 分子缩放到同一个指数基准。
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)

    @jit
    def softmax_rest(self, qk_ch, scale, m_axis_triple: int, tile_triple: int, actual_n):
        # 合并当前 tile 与已保存的 running 状态；有 sinks 时首 tile 也走此函数。
        # ori→cmp 时 m_seq 不变、n 不归零，继续使用同一 max/sum 槽，不做分池 softmax。
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]
        VL_T = 2048 // 32
        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        half_only = self._half_only(actual_n, VL_T)
        with vf(mode="raw"):
            rows = qk_ch.shape[0]
            src_row_stride = qk_ch.stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                if const_expr(half_only):
                    half1_mask = None
                else:
                    half1_mask, _ = rr.update_mask(max(0, actual_n - VL_T), elem_bits=32)
                self._pass_a_row(qk_ch, scale, self.tmp_new_max, row, src_row_stride, VL_T,
                                 half0_mask, half1_mask, full, half_only=half_only)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                b16_full, _ = rr.update_mask(N, elem_bits=16)
                ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
                vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
                b16, _ = rr.update_mask(actual_n, elem_bits=16)
                base = row * src_row_stride
                nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
                ve, vo = self._softmax_fold_row(
                    qk_ch, base, nz_off, self.tmp_new_max, row,
                    ve_mask, vo_mask, b16, b16_full, block_stride
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
                rr.vstore_first(self.tmp_sum, row, rsum)
            rr.vmem_bar("vst_vld")
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def seed_from_sinks(self, sinks_view, m_axis_triple: int):
        # sinks 播种（仅第一个 n-tile）：max ← sink[head]、sum ← 1.0（exp(sink−sink)=1），
        # 此后每 tile 走 softmax_rest 的 update 路径 —— sink 只吸收分母质量、无 value 贡献。
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        span = self.sinks_span
        with vf(mode="raw"):
            full, _ = rr.update_mask(2048 // 32, elem_bits=32)
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            for row in tuple(range(self.tile_vec_m)):
                rr.vstore_first(sm_max, row, rr.vload_broadcast(sinks_view, row % span))
                rr.vstore_first(sm_sum, row, one)
            rr.vmem_bar("vst_vld")

    @jit
    def seed_empty(self, m_axis_triple: int):
        # FD 非首分部播种（对齐 AscendC 跨核切分行续算语义）：max ← FP32 下界、
        # sum ← 0；sinks 只属于行的首个分部。alpha=exp(下界−new_max) 下溢为 0，
        # 首 tile 后状态等价于从零开始的在线 softmax。
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            neg = rr.vdups(_FD_NEG_INF, dtypes.float32, mask=rowmask)
            zero = rr.vdups(0.0, dtypes.float32, mask=rowmask)
            rr.vstore(sm_max, 0, neg, rowmask)
            rr.vstore(sm_sum, 0, zero, rowmask)
            rr.vmem_bar("vst_vld")

    def store_p(self, p_l1_ch, partition):
        # P 从 p_ub 跨核送 p_l1：每 AIV 填自己的 32 头半块，L1 两槽合计 32 KiB。
        piece = partition_view(p_l1_ch, partition, self.subblock_idx)
        mem_copy(piece, self.p_ub)

    @jit
    def init_o(self, pv_ch):
        # 首 tile 的旧分子为 0（sink 无 Value），直接复制 PV，无需读未初始化的 res_o。
        mem_copy(local_slice(self.res_o, (pv_ch.shape[0], self.tile_d)), pv_ch)

    @jit
    def update_o(self, pv_ch, exp_idx: int):
        # res_o 保存尚未除分母的 FP32 分子：new_o = alpha * old_o + PV。
        # alpha 由同一 tick 的 softmax 产生；PV 中的 P 已转 BF16，矩阵乘累加为 FP32。
        # full_mask 复用于 D=512 的八个 64-lane chunk，D 的整除约束在构造时检查。
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def update_o_last(self, pv_ch, exp_idx: int, sum_idx: int):
        # 末 tile：累加后在 FP32 域除 sum，随后 finalize 才转 BF16，避免先舍入再归一化。
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            one = rr.vdups(1.0, dtypes.float32)
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                inv_sum = rr.vdiv(one, sum_b, mask=full)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=full)
                    o = rr.vmul(o, inv_sum, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def init_o_last(self, pv_ch, sum_idx: int):
        # 整个 query 只有一个 KV tile：没有跨 tile 累加，直接用 PV/最终分母初始化 O。
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        VL_T = 2048 // 32
        with vf(mode="raw"):
            full = rr.full_mask()
            one = rr.vdups(1.0, dtypes.float32)
            for row in range(self.tile_vec_m):
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                inv_sum = rr.vdiv(one, sum_b, mask=full)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    cur = rr.vload(pv_ch, off)
                    rr.vstore(self.res_o, off, rr.vmul(cur, inv_sum, mask=full), full)

    @jit
    def _finalize_lse_vf(self, lse_gm, lse_base, sum_idx: int):
        # lse = log(sum) + max；独立 Channel 管理 V 写/MTE3 读及槽复用。
        sm_max_buf = self.sm_max_tb[sum_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        with vf(mode="raw"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            total = rr.vload(sm_sum_buf, 0)
            maximum = rr.vload(sm_max_buf, 0)
            rr.vstore(self.lse_ub, 0, rr.vadd(rr.vlog(total, mask=mask), maximum, mask=mask), mask)
        mem_copy(tile_view(lse_gm, (self.tile_vec_m,), (lse_base,)),
                 local_slice(self.lse_ub, (self.tile_vec_m,)))

    @jit
    def _cast_output(self):
        # 原地 cast：res_o(32,512) FP32 → out_view 前 32 KiB BF16。B32_TO_B16 以相同
        # 元素 offset 落位：第 o 块写 BF16 字节 [2o, 2o+128) ⊂ 读 FP32 字节 [4o, 4o+256)，
        # 且下一块读起点 4(o+64) = 4o+256 > 2o+128，块间无交叉；V 核内按序执行，覆写安全。
        with vf(mode="raw"):
            full = rr.full_mask()
            for offset in range(0, self.tile_vec_m * self.tile_d, 64):
                value = rr.vload(self.res_o, offset)
                result = rr.vcast(value, dtypes.bfloat16, mask=full)
                rr.vstore_pack(self.out_view, offset, result, full, pack_mode=rr.PackMode.B32_TO_B16)

    def finalize_o(self, o_tile_gm, sum_idx: int,
                   lse_gm=None, lse_base=None):
        # init_o_last/update_o_last 已完成除分母；此处计算可选 LSE，并转 BF16 写出。
        # 整块 32 行原地 cast 后只插一对「MTE3 等 V」，再单笔 (32,512) 拷出；两笔 MTE3
        # （lse、out）按序执行，V 侧无任何等待。出拷贝的源 out_view 是 res_o 前
        # 32 KiB 的同储别名：下一 query 首笔 res_o 写（init_o/init_o_last）前必须等
        # 本拷贝完成。原先依赖 pv_ub 生产链（AIC bmm1+bmm2+fixpipe）慢于 32 KiB
        # 拷贝的"天然门控"，但末两任务为相邻单 tile query 时下一 bmm2 已在 cube
        # 队首、fixpipe 在上一 update 消费后立即到货，VF 写会追上拷贝尾部
        # （out_view 本地行 24-31）⇒ GM 混入下一 query 的 FP32 原始位型
        # （偶 lane=低 16 位 → NaN，b=1 s1=42 实测 head 56-63 散点 NaN）。故在
        # 拷贝后入队 _OUT_MTE3_TO_V 信号，由 wait_prev_out_copy 消费（空行路径
        # 无生产链门控，早有同构显式排空，见 finalize_empty）。
        split_m = make_partition_tiler(o_tile_gm.shape, (self.tile_m, self.tile_d))
        half = partition_view(o_tile_gm, split_m, self.subblock_idx)

        if const_expr(lse_gm is not None):
            self._finalize_lse_vf(lse_gm, lse_base, sum_idx)

        self._cast_output()
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(half, self.out_view)
        # 信号紧随出拷贝入队（早于后续 vec0 的 L1 写），等待只覆盖拷贝本身；
        # 末 query 的信号无人消费，悬置无害（sync id 每次发射独立）。
        vec_sync_notify(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)

    @jit
    def wait_prev_out_copy(self):
        # V 等 MTE3：上一 query 出拷贝完成后才允许本 query 首笔 res_o 覆写。
        # 仅在 tick>0（此前存在任务 ⇒ 上一 query 非空且已 finalize）时调用，
        # 否则无信号可等会挂死；稳态下拷贝早已完成，信号已就绪，等待零阻塞。
        vec_sync_wait(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)

    @jit
    def finalize_empty(self, o_tile_gm, sinks_gm, lse_gm=None, lse_base=None):
        # No KV contributes to the numerator; the denominator contains only
        # the sink, so O=0 and LSE=sink exactly. 空行在 drain 段连续写出，相邻两笔
        # 之间没有 AIC 生产链做天然门控：覆写 out_view 前先等上一笔出拷贝（含主循环
        # 末尾的 finalize 拷贝）在 MTE3 上排空，再写零 + 单对「MTE3 等 V」+ 单笔拷出。
        vec_sync_notify(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)
        split_m = make_partition_tiler(o_tile_gm.shape, (self.tile_m, self.tile_d))
        half = partition_view(o_tile_gm, split_m, self.subblock_idx)
        with vf(mode="raw"):
            full, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, dtypes.bfloat16, mask=full)
            for offset in range(0, self.tile_vec_m * self.tile_d, 128):
                rr.vstore(self.out_view, offset, zero, full)
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(half, self.out_view)
        if const_expr(lse_gm is not None):
            # Keep lse_ub's producer on V, as in _finalize_lse_vf; using
            # MTE2 here would change the shared Channel's event protocol.
            mem_copy(self.sinks_ub, tile_view(sinks_gm, (1, self.sinks_span),
                                            (0, self.subblock_idx)))
            with vf(mode="raw"):
                mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                rr.vstore(self.lse_ub, 0, rr.vload(self.sinks_ub, 0), mask)
            mem_copy(tile_view(lse_gm, (self.tile_vec_m,), (lse_base,)),
                     local_slice(self.lse_ub, (self.tile_vec_m,)))

    # ---- FD 辅助（每个 VF 块独立成 @jit 方法：runtime for 循环体内不得绑定
    # VF 寄存器/mask 局部量，否则会被提升为 scf.for 迭代状态导致类型错乱）----

    @jit
    def _fd_stage_chunk(self, fd_ub, c: int):
        # fd_ub ← res_o 行 [c*_FD_CHUNK_ROWS, c*_FD_CHUNK_ROWS+_FD_CHUNK_ROWS)（FP32 原值）
        with vf(mode="raw"):
            full = rr.full_mask()
            for r in range_constexpr(_FD_CHUNK_ROWS):
                for col in range(0, self.tile_d, 2048 // 32):
                    v = rr.vload(self.res_o,
                                 (c * _FD_CHUNK_ROWS + r) * self.tile_d + col)
                    rr.vstore(fd_ub, r * self.tile_d + col, v, full)

    @jit
    def _fd_ms_to_ub(self, which: int, sum_idx: int):
        # lse_ub ← sm_max（which=0）/ sm_sum（which=1）槽
        if const_expr(which == 0):
            buf = self.sm_max_tb[sum_idx]
        else:
            buf = self.sm_sum_tb[sum_idx]
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(self.lse_ub, 0, rr.vload(buf, 0), rowmask)

    @jit
    def _fd_init_acc(self):
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(self.sm_max_tb[0], 0,
                      rr.vdups(_FD_NEG_INF, dtypes.float32, mask=rowmask), rowmask)
            rr.vstore(self.sm_sum_tb[0], 0,
                      rr.vdups(0.0, dtypes.float32, mask=rowmask), rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_max_save(self, p: int):
        # lse_ub 已载 m_p：acc_max = max(acc_max, m_p) 且 m_p 存 fd_mp_tb[p]（供第 2 遍复用；第 2 遍将其改写为 t_p，供第 4 遍复用）
        mp_buf = self.fd_mp_tb[p]
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            mp = rr.vload(self.lse_ub, 0)
            nm = rr.vmax(rr.vload(self.sm_max_tb[0], 0), mp, mask=rowmask)
            rr.vstore(self.sm_max_tb[0], 0, nm, rowmask)
            rr.vstore(mp_buf, 0, mp, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_sum(self, p: int):
        # lse_ub 已载 s_p：t_p = s_p·exp(m_p−M)；acc_sum += t_p；t_p 存回 fd_mp_tb[p]
        # （复用第 1 遍的 m_p 槽，第 4 遍以 t_p/G 求权重，= AscendC ComputeScaleValue_8_VF）
        mp_buf = self.fd_mp_tb[p]
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            e = rr.vexp_sub(rr.vload(mp_buf, 0),
                            rr.vload(self.sm_max_tb[0], 0), mask=rowmask)
            t = rr.vmul(rr.vload(self.lse_ub, 0), e, mask=rowmask)
            acc = rr.vadd(rr.vload(self.sm_sum_tb[0], 0), t, mask=rowmask)
            rr.vstore(self.sm_sum_tb[0], 0, acc, rowmask)
            rr.vstore(mp_buf, 0, t, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_lse_to_ub(self):
        # lse_ub ← M + log(G)
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            lse = rr.vadd(rr.vlog(rr.vload(self.sm_sum_tb[0], 0), mask=rowmask),
                          rr.vload(self.sm_max_tb[0], 0), mask=rowmask)
            rr.vstore(self.lse_ub, 0, lse, rowmask)

    @jit
    def _fd_weight(self, p: int):
        # w_p = t_p / G → tmp_new_max；t_p 复用第 2 遍存回 fd_mp_tb[p] 的值
        mp_buf = self.fd_mp_tb[p]
        with vf(mode="raw"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            w = rr.vdiv(rr.vload(mp_buf, 0),
                        rr.vload(self.sm_sum_tb[0], 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, w, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_zero_o(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            for off in range(0, self.tile_vec_m * self.tile_d, 2048 // 32):
                rr.vstore(self.res_o, off, zero, full)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_chunk(self, fd_ub, c: int):
        # res_o 行 [c*_FD_CHUNK_ROWS, ...) += w_p ⊙ fd_ub（fd_ub 已载归一化 O_p 分块）；
        # 权重 w_p = t_p/G 取 tmp_new_max 的第 c*_FD_CHUNK_ROWS+r 个 head（与 chunk 行一一对应）。
        # vmadd(x, y, z) = y*x + z（与 update_o 同约定）；要 res_o + w_p·O，
        # 故传 vmadd(cur=O, wb=w_p, pre=res_o)。
        with vf(mode="raw"):
            full = rr.full_mask()
            for r in range_constexpr(_FD_CHUNK_ROWS):
                wb = rr.vload_broadcast(self.tmp_new_max, c * _FD_CHUNK_ROWS + r)
                for col in range(0, self.tile_d, 2048 // 32):
                    off = (c * _FD_CHUNK_ROWS + r) * self.tile_d + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(fd_ub, r * self.tile_d + col)
                    rr.vstore(self.res_o, off,
                              rr.vmadd(cur, wb, pre, mask=full), full)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_full(self, fd_ub):
        # res_o += w_p ⊙ fd_ub：fd_ub 是 post-rewind 的整块 (32,512) fp32 落地，
        # w_p = t_p/G 在 tmp_new_max（与 update_o 同 vmadd(cur, wb, pre) 约定）。
        with vf(mode="raw"):
            full = rr.full_mask()
            for r in range(self.tile_vec_m):
                wb = rr.vload_broadcast(self.tmp_new_max, r)
                for col in range(0, self.tile_d, 2048 // 32):
                    off = r * self.tile_d + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(fd_ub, off)
                    rr.vstore(self.res_o, off,
                              rr.vmadd(cur, wb, pre, mask=full), full)
            rr.vmem_bar("vst_vld")

    def _setup_fd_reduce_buffers(self):
        # FD 归约专属 buffer 组：在 global_sync_all + channel_rewind 之后分配，复用
        # 主循环（已结束、barrier 隔离）的 UB 地址。fd_ub 抬到整块 (32,512) fp32
        # （64KiB）⇒ 每个跨核切分分部的 O 一整笔 GM→UB，替代 _FD_CHUNK_ROWS=1 的
        # 32 笔 2KiB 小搬运（此前归约 76.8us 回退的根因）。归约累加器/落地属性原地
        # 重指到新 buffer，fd_reduce/_fd_* 辅助方法无需改签名即可跑在新 buffer 上。
        self.fd_ub = Channel(MemLoc.UB, shape=(self.tile_vec_m, self.tile_d),
                             dtype=dtypes.float32, depth=1)
        self.res_o = Buffer(MemLoc.UB, (self.tile_vec_m, self.tile_d), dtypes.float32)
        self.out_view = reinterpret(self.res_o, dtypes.bfloat16,
                                    (self.tile_vec_m, self.tile_d))
        self.lse_ub = Channel(MemLoc.UB, shape=(self.tile_vec_m, 1),
                              dtype=dtypes.float32, depth=1)
        self.sm_max_tb[0] = Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
        self.sm_sum_tb[0] = Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
        self.tmp_new_max = Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
        self.fd_mp_tb = [Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
                         for _ in range(_FD_MP_SLOTS)]

    @jit
    def stage_partial(self, fd_o, fd_ms, fd_ub, slot, sum_idx: int):
        # FD 暂存（= AscendC StageVec1Lse/StageVec2PartialO 的常规 FD 路径）：
        # 跨核切分行在本核的分部结束时，把已归一化 O（FP32，res_o 已除以分母
        # sum，由调用方 init_o_last/update_o_last 完成）与本 AIV 32 个 head 的
        # max/sum 写入 workspace slot。
        # O 经 fd_ub 以 (_FD_CHUNK_ROWS,512) 分块走 V→MTE3；max/sum 复用 lse_ub（32 fp32）。
        # slot 布局：O 区 (slots*64, 512) fp32；max/sum 区 (slots*128,) fp32，
        # 每 slot 128 元素 = max[0:64] ‖ sum[64:128]。
        sub = self.subblock_idx
        for c in range_constexpr(self.tile_vec_m // _FD_CHUNK_ROWS):
            self._fd_stage_chunk(fd_ub, c)
            o_coord = slot * (N1 // _FD_CHUNK_ROWS) + sub * (self.tile_vec_m // _FD_CHUNK_ROWS) + c
            mem_copy(tile_view(fd_o, (_FD_CHUNK_ROWS, self.tile_d), (o_coord, 0)),
                     fd_ub)
        self._fd_ms_to_ub(0, sum_idx)
        mem_copy(tile_view(fd_ms, (self.tile_vec_m,),
                           (slot * (FD_MS_ELEMS // self.tile_vec_m) + sub,)),
                 local_slice(self.lse_ub, (self.tile_vec_m,)))
        self._fd_ms_to_ub(1, sum_idx)
        mem_copy(tile_view(fd_ms, (self.tile_vec_m,),
                           (slot * (FD_MS_ELEMS // self.tile_vec_m) + (N1 // self.tile_vec_m) + sub,)),
                 local_slice(self.lse_ub, (self.tile_vec_m,)))

    @jit
    def fd_reduce(self, fd_o, fd_ms, fd_ub, out_gm, lse_gm,
                  row, slot0, k, h0):
        # FD 归约（= AscendC ProcessFlashDecode/ReduceWithLse）：本 AIV 负责行 row
        # 的 32 个 head（h0 ∈ {0,32}），合并 k 份分部：
        #   M = max_p(m_p)；e_p = exp(m_p−M)；t_p = s_p·e_p；G = Σ_p t_p；
        #   out = Σ_p t_p/G · O_p（O_p 已归一化）；lse = M + log(G)。
        # 分部按 slot 序号从左到右累加，顺序由 metadata 确定 ⇒ 结果确定。
        # 全部 UB 复用主循环既有对象（barrier 后主循环已结束）：
        # acc_max→sm_max_tb[0]、acc_sum→sm_sum_tb[0]、m_p/t_p→fd_mp_tb[p]、
        # w_p→tmp_new_max、累加器→res_o、落地→fd_ub/lse_ub/out_view。
        sub = h0 // self.tile_vec_m
        self._fd_init_acc()
        # 1) M = max_p(m_p)；同时把 m_p 存 fd_mp_tb[p]，供第 2 遍复用（省重复小 load）
        for p in range(k):
            mem_copy(self.lse_ub,
                     tile_view(fd_ms, (self.tile_vec_m,),
                               ((slot0 + p) * (FD_MS_ELEMS // self.tile_vec_m) + sub,)))
            self._fd_acc_max_save(p)
        # 2) G = Σ_p s_p·exp(m_p−M)（t_p 存回 fd_mp_tb[p]，供第 4 遍复用）
        for p in range(k):
            mem_copy(self.lse_ub,
                     tile_view(fd_ms, (self.tile_vec_m,),
                               ((slot0 + p) * (FD_MS_ELEMS // self.tile_vec_m) + (N1 // self.tile_vec_m) + sub,)))
            self._fd_acc_sum(p)
        # 3) 可选 LSE = M + log(G)
        if const_expr(lse_gm is not None):
            self._fd_lse_to_ub()
            mem_copy(tile_view(lse_gm, (self.tile_vec_m,), (row * 2 + sub,)),
                     local_slice(self.lse_ub, (self.tile_vec_m,)))
        # 4) res_o = Σ_p t_p/G ⊙ O_p（O_p 已归一化；t_p 复用第 2 遍已存值）
        self._fd_zero_o()
        for p in range(k):
            self._fd_weight(p)
            # post-rewind fd_ub 是整块 (32,512) fp32 落地：每个分部一整笔 64KiB GM→UB，
            # 替代 _FD_CHUNK_ROWS=1 的 32 笔 2KiB 小搬运（归约延迟的主要来源）。
            mem_copy(fd_ub,
                     tile_view(fd_o, (self.tile_vec_m, self.tile_d),
                               ((slot0 + p) * (N1 // self.tile_vec_m) + sub, 0)))
            self._fd_acc_full(fd_ub)
        # 5) 输出：res_o 已归一化，整块原地转 BF16 写出（与 finalize_o 同路径）。
        # out 行号 = row*64 + h0 + [0,32)，(32,512) tile 坐标 = row*(N1/tile_vec_m) + sub；
        # barrier（global_sync_all 的 vec_sync_all）已排空主循环的全部 MTE3，无反向等待。
        self._cast_output()
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(tile_view(out_gm, (self.tile_vec_m, self.tile_d),
                           (row * (N1 // self.tile_vec_m) + sub, 0)),
                 self.out_view)

@kernel
class MqsmlaKernel:
    """Device attention pipeline; tensor views and launch live in _run_mqsmla."""

    def __init__(self, tile_cube_m, tile_vec_m, tile_n,
                 n_heads=64, has_cmp=False, ws_depth=WS_DEPTH):
        self.n_heads = n_heads
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D
        self.has_cmp = has_cmp
        self.qk_ub = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32,
                             depth=2, kind=ChannelKind.CrossCore)
        # FIX(AIC) → V(AIV)：必须使用 CrossCore；kind 不代替整块 PV 的事务边界。
        self.pv_ub = Channel(MemLoc.UB, shape=(tile_vec_m, self.tile_d), dtype=dtypes.float32,
                             depth=1, kind=ChannelKind.CrossCore)
        self.p_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_n), dtype=dtypes.bfloat16,
                            depth=_P_L1_DEPTH, kind=ChannelKind.CrossCore)

        # 三条跨核数据链：QK(AIC→AIV，depth=2)、P(AIV→AIC，depth=2)、PV(AIC→AIV，depth=1)。
        # qk_ub/pv_ub 的 shape 是每 AIV 半块，p_l1 的 shape 是 AIC 接收的完整 64 头块。

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.sinks_span = tile_vec_m

        self.matmul = Matmul(tile_cube_m, tile_vec_m, tile_n)
        self.vector = Vector(tile_vec_m, tile_n, self.tile_d, self.subblock_idx,
                             sinks_span=self.sinks_span)

        self.ws_depth = int(ws_depth)
        (self._ws_ready_id,) = _alloc_ws_intra_ids()
        # AIV→L1 的 free 握手 id（AIC MTE1 读完 L1 槽 → 通知 AIV MTE3 可覆写下一份）。
        # 与 ws_ready 同源分配、独立 id；AIV1 侧经 +_AIV1_ID_OFFSET 区分。
        self._kv_free_id = _alloc_ws_intra_ids()[0] if _AIV2L1 else None
        self.kvcache = Kvcache()
        # The existing scalar PA path accepts runtime page sizes and does not
        # allocate an on-chip buffer proportional to the block-table capacity.
        ch_rows = _TILE_ROWS
        # MTE2 → V 均在同一 AIV，SameCore 自动管理逐行填充和反量化的事务。
        self.ch_batch_bytes = Channel(MemLoc.UB,
                                      shape=(ch_rows, _COMBINE_DIM),
                                      dtype=dtypes.uint8, depth=2, kind=ChannelKind.SameCore)
        # DMA 和 VF 共用一个字节 Channel；scale 在寄存器中按 BF16 位解释，
        # 避免同一槽的两个 Channel dtype 别名生成重复消费事务。
        self.n_sub = tile_n // _TILE_ROWS
        # FD staging/归约共用的 (_FD_CHUNK_ROWS,512) FP32 落地 Channel：
        # 暂存方向 V→MTE3，归约方向 MTE2→V，两阶段被 global_sync_all 隔开。
        # _FD_COMPILE=False 时不分配（stage_partial 未编译，无人引用），归还 2 KiB UB。
        self.fd_ub = None
        if _FD_COMPILE:
            self.fd_ub = Channel(MemLoc.UB, shape=(_FD_CHUNK_ROWS, D),
                                 dtype=dtypes.float32, depth=1)

    # ---- 坐标与有效宽度：m_seq 是核内 query 序号，m_idx 是全局 query token 序号 ----
    # n 是同一 query 上 ori→cmp 串接的 tile 序号；tick 则跨 query 连续编号。
    # 每侧 w = topk_length（None 时为 K），tiles = ceil(w/128)，actual_n = min(128,w−128*n)。
    # gather 的有效前缀与 softmax 的有效列数必须同源；切到 cmp 后须先减去 ori tile 数。
    # *_s2_start 是稀疏索引表列偏移，不是逻辑 token 值。

    def _m_idx(self, m_seq: int):
        # metadata 连续区间映射：m_start 在 __call__ 里读一次存 SSA，这里只做单次加法。
        return self._m_start + m_seq

    @jit
    def _batch_of(self, m_idx: int):
        # Dynamic upper-bound search over the B query spans.
        lo, hi = 0, self._b_rt
        while lo < hi:
            mid = (lo + hi) // 2
            take = _i64(self._cu_q, 0, mid + 1) <= m_idx
            lo = dyn_select(take, mid + 1, lo)
            hi = dyn_select(take, hi, mid)
        return lo

    @jit
    def _bound_after(self, batch: int, b_last: int):
        # cu_q[batch+1]（= 下一个 batch 的起始 query 序号）。batch 已在最后一个
        # span 上时返回 +INF 哨兵，既不越界读 cu_q，也让增量推进自然停在 b_rt，
        # 与 _batch_of 的 hi=b_rt 钳位逐位一致。单次 GM 标量读。
        nxt = dyn_select(batch < b_last, batch + 1, b_last)
        v = _i64(self._cu_q, 0, nxt)
        return dyn_select(batch < b_last, v, dtypes.int64(_ADDR_M_INF))

    def _cmp_tiles_rt(self, m_idx: int):
        w = self._csa_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    @jit
    def _ori_valid_w(self, m_idx: int):
        # ori 侧有效宽度：min(K1, ori_topk_length[m_idx])。
        # 无 topk_length 时也强制 Int64 提升为 SSA：FD 主循环的 t0/t1/split
        # dyn_select 脚手架要求 tiles 是运行时值（否则 select 谓词退化为 Python bool）。
        if const_expr(self._ori_topk_length is not None):
            return min(self._ori_idx.shape[1], _i64(self._ori_topk_length, m_idx))
        return Int64(self._ori_idx.shape[1])

    @jit
    def _cmp_topk_len(self, m_idx: int):
        if const_expr(self._cmp_topk_length is not None):
            return min(self._cmp_idx.shape[1], _i64(self._cmp_topk_length, m_idx))
        return Int64(self._cmp_idx.shape[1])

    def _ori_tiles_rt(self, m_idx: int):
        w = self._ori_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    def _n_end_rt(self, m_idx: int):
        # 本 query 的 n-tile 趟数（运行时）：ori_tiles + cmp_tiles。
        ori_tiles = self._ori_tiles_rt(m_idx)
        if const_expr(not self.has_cmp):
            return ori_tiles
        return ori_tiles + self._cmp_tiles_rt(m_idx)

    def _actual_n(self, m_idx: int, n):
        # 根据 n 所在的池返回当前 tile 的有效 KV 列数，供 softmax 的指数/P 掩码使用。
        # 只有被选中的池内坐标有效；dyn_select 未选中的候选宽度不代表实际任务。
        ori_w = self._ori_width(m_idx, n)
        if const_expr(not self.has_cmp):
            return ori_w
        ori_tiles = self._ori_tiles_rt(m_idx)
        c = n - ori_tiles
        cmp_w = self._cmp_valid(m_idx, c)
        return dyn_select(n >= ori_tiles, cmp_w, ori_w)

    @jit
    def _ori_width(self, m_idx: int, n):
        w = self._ori_valid_w(m_idx)
        return min(self.tile_n, w - n * self.tile_n)

    def _csa_valid_w(self, m_idx: int):
        return self._cmp_topk_len(m_idx)

    @jit
    def _cmp_valid(self, m_idx: int, c):
        return min(self.tile_n, self._csa_valid_w(m_idx) - c * self.tile_n)

    def _split_m(self):
        return make_partition_tiler(
            (self.tile_cube_m, self.tile_n), (self.tile_cube_m, self.tile_n),
        )

    @staticmethod
    def _scale_idx(grp_lane, base: int, m16):
        # scale gather 的索引向量：idx[lane m] = 组基号 base + m//group_size（u16）。
        # ★ u16 运算必须用 b16 粒度掩码：用 b32 的 full_mask 会把移位结果整寄存器清零
        # ⇒ gather 恒取 srow[0] ⇒ 全列小误差（0.018 量级，板证教训）。
        return rr.vadd(grp_lane, rr.vdups(base, dtypes.uint16), mask=m16)

    @jit
    def _antiquant_vf_fp8_g32(self, kv_fp8, out):
        # ori：nope448|rope64|scale16，数据顺序与计算顺序一致。
        # 四段各128维：FP8→FP32→BF16，deinterleave压紧偶lane，乘group32 BF16 scale。
        # 每段scale基号为4*j，字节偏移为128*j；写入同列的BF16 NZ Channel。
        SUB = _TILE_ROWS
        with vf(mode="raw"):
            m32, _ = rr.update_mask(_TILE_SIZE, elem_bits=32)
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            m8, _ = rr.update_mask(256, elem_bits=8)
            grp_lane = rr.vshr(rr.varange(0, dtypes.uint16), 5, mask=m16)
            row_bf16, row_fp8 = _ROW_BF16, _COMBINE_DIM

            # j遍历四段连续128维，i遍历子块内token。
            for j in tuple(range(D // _KV_NUM_PER_LOOP)):
                src_j = j * _KV_NUM_PER_LOOP
                dst_j = j * _KV_NUM_PER_LOOP
                idx_j = self._scale_idx(grp_lane, 4 * j, m16)
                for i in range(SUB):
                    s = i * row_fp8 + src_j
                    u0 = rr.vload_unpack(kv_fp8, s, unpack_mode="b8_to_b32")
                    u1 = rr.vload_unpack(kv_fp8, s + _TILE_SIZE,
                                         unpack_mode="b8_to_b32")
                    e0 = rr.vreinterpret_lanes(u0, Float8E4M3FN)
                    e1 = rr.vreinterpret_lanes(u1, Float8E4M3FN)
                    f0 = rr.vcast(e0, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                    f1 = rr.vcast(e1, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                    b0 = rr.vcast(f0, dtypes.bfloat16, mask=m32)
                    b1 = rr.vcast(f1, dtypes.bfloat16, mask=m32)
                    raw, _ = rr.vdeinterleave(b0, b1)
                    srow = rr.vreinterpret_lanes(
                        rr.vload(kv_fp8, 2 * (_SCALE_BF16_OFF + i * row_bf16)), dtypes.bfloat16)
                    svec = rr.vgather_reg(srow, idx_j)
                    packed = rr.vmul(raw, svec, mask=m16)
                    self._v0_store(out, i, dst_j, packed, m16)

    @jit
    def _antiquant_vf_fp4_g16(self, kv_fp8, out):
        # cmp：nope224B|rope32B|scale64B，四段各128维连续反量化。
        # 2026-09-16 ops-nn 式（参照 anti_mx_quant + 本轮真机/cannsim 探针结论）：
        #   解码 = UNPACK4 装载(字节j→lane 4j) + reinterpret fp4x2 + 单条
        #          vcast(ZERO)。vcast 的 fp4 分支被 cannbotdsl 0.5.0 翻译层降为
        #          asc_cast_unknown（发射器二进制内无 asc_e2m1x22bfloat16，
        #          C1 实证），由本文件模块级 compile-hook 将其文本替换为
        #          asc_e2m1x22bfloat16(..., DISPERSE_FIRST_QUARTER)。单条 Q0
        #          vcvt 输出即为 [lo(b0),hi(b0),...] 全序，无需 intlv（真机
        #          探针 r15 对照 20/128 证伪了 intlv 版）。（旧 and/shr/
        #          intlv+LUT 解码路径已随开关移除，见 TUNING-REPORT.md。）
        #   scale = vload_broadcast(ELEM2DATABLOCK)：8 个 g16 scale 各填满一个
        #          32B block=16 lanes，恰为分组形态；替代 srow vload+vgather_reg
        #          (EXQ lat16)。真机实证 E2B 位级传输正确且容忍 16B 对齐（段
        #          p=1,3 的 scale 在字节 272/304）。
        # cannsim 稳态 II：540.1(基线) → 181.6（EX 540.7→169.0）；bit 级精度
        # 真机 ND/NZ 全合成均 128/128（tuning/fp4new/）。输出落位 NZ17 不变。
        # （C2 全宽解码版 2026-09-16 前已回退，见 TUNING-REPORT.md。）
        SUB = _TILE_ROWS
        with vf(mode="raw"):
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            row_fp8 = _COMBINE_DIM
            scale_off = SCALE_BF16_OFF_CMP

            # scale 侧沿用 vload+vgather_reg：E2B(vload_broadcast
            # ELEM2DATABLOCK) 需要 16 位 dtype 视图,而 Channel.reinterpret
            # 别子无论建在 vf 区内/外都在 cannir-resolve-channel-operands
            # 触发 "operand does not dominate this use"(DSL 0.5.0 缺陷,
            # 两处落点均实证);手搓 RawRegLoadBrcOp 又被 op 校验拒绝
            # (e2b_b16 requires 16-bit source buffer element)。留档待上游
            # 修复;真机/cannsim 层面 E2B 位级与对齐已单独验证(见
            # tuning/fp4new/)。idx 与旧路径同构(lane//16 + 8p)。
            grp_lane = rr.vshr(rr.varange(0, dtypes.uint16), 4, mask=m16)
            for p in tuple(range(D // _KV_NUM_PER_LOOP)):
                src_p = p * _TILE_SIZE
                dst_p = p * _KV_NUM_PER_LOOP
                idx_p = self._scale_idx(grp_lane, 8 * p, m16)
                for i in range(SUB):
                    u = rr.vload_unpack(kv_fp8, src_p + i * row_fp8,
                                        unpack_mode=rr.UnpackMode.UNPACK4)
                    f4 = rr.vreinterpret_lanes(u, dtypes.fp4x2_e2m1)
                    fm = rr.full_mask()
                    fm.elem_bits = 4
                    vals = rr.vcast(f4, dtypes.bfloat16, mask=fm,
                                    reg_layout=rr.RegLayout.ZERO)
                    srow = rr.vreinterpret_lanes(
                        rr.vload(kv_fp8, 2 * (scale_off + i * _ROW_BF16)),
                        dtypes.bfloat16)
                    packed = rr.vmul(vals, rr.vgather_reg(srow, idx_p),
                                     mask=m16)
                    self._v0_store(out, i, dst_p, packed, m16)
    def _v0_store(self, out, i, col, val, mask):
        # i 为子块内 token 行，col 为逻辑 D 列，所有偏移均以 BF16 元素计。
        # NZ17 的 UB 地址 = (col//16)*(17*16) + i*16 + col%16；vstore_strided
        # 每写 16 列跨到下一 face，间距 17 个 32B block。第 17 行是 padding。
        rr.vstore_strided(out, (col // 16) * _NZ_CHUNK + i * _NZ_ROW,
                          val, mask, block_stride=_NZ_PAD_ROWS,
                          repeat_stride=0)

    @jit
    def _vec0_body(self, kv_pool, blk_table, blk_size, max_blk, s2_start: int, slot: int,
                   n_real, idx_tab=None, idx_row=None, batch=None,
                   is_cmp=False):
        # lag0：128-token tile 分成 8 个 16 行子块；AIV0 负责前 4 个，AIV1 负责后 4 个。
        # 当前每子块依次执行以下操作（不是性能记录中的 prime/replenish 发射方式）：
        #   1. 先算16行地址；全部pair可合并时每对按物理地址排序，用8笔双行DMA；
        #      否则整子块逐行搬运。尾行钳到最后有效索引，避免 -1 参与 PA 寻址。
        #   2. 字节槽自动同步后反量化：ori FP8 / cmp FP4，scale 都在同一行内。
        #   3. V→MTE3 就绪事件后，将 BF16 数据写入本核 GM ws 的 tick%3 槽。
        # 两池 UB 行距统一为 544B；cmp 每行只搬入前 320B，反量化仅使用有效数据/scale。
        # 输出 _v0_out 双槽由 Channel 轮转，MTE3→V 依赖阻止覆盖尚未搬完的源槽。
        # 只处理 sub_row<n_real 的子块；完全无效的子块不写，PV 的归约范围排除这些行。
        # 对已处理残块的尾行重读，可避免未初始化值进入 PV 的 0*Value 路径。
        SUB = _TILE_ROWS
        if const_expr(is_cmp):
            _ROW_BYTES = KV_ROW_BYTES_CMP
            page_stride = self._cmp_page_stride
        else:
            _ROW_BYTES = KV_ROW_BYTES_ORI
            page_stride = self._ori_page_stride
        n_rt = n_real
        # 显式覆盖从首字节到末页末字节的物理 storage（含页间空洞），不是 token 折叠。
        # mem_copy 保留 Channel 可识别的生产者。
        storage_bytes = ((kv_pool.shape[0] - 1) * page_stride + blk_size * _ROW_BYTES)
        kv_bytes = kv_pool.view(storage_bytes)

        part = self.n_sub // 2
        t0 = self.subblock_idx * part
        _mode = _addr_vec_mode()
        _consumer_vec = _mode == "pre"
        tab = None
        col_base = 0
        if const_expr(_mode == "pre"):
            # 行号已在主循环前的 pass 里写进 GM 表；col_base 选池区列基址。
            tab = self._addr_tab
            col_base = self._avec_k1_tab if const_expr(is_cmp) else 0
        for lt in range(part):
            sub_b = t0 + lt
            sub_row = sub_b * SUB
            if sub_row < n_rt:
                buf = self.ch_batch_bytes
                bo_sp = 0 if batch is None else batch
                # 寻址前钳位到有效前缀；残块尾行重读最后一个有效 token。
                last_off = n_rt - 1 - sub_row
                addresses = []
                pair_first = []
                pair_gap = []
                all_pairs_ok = Int32(1)
                if const_expr(_consumer_vec):
                    # 向量臂：行号×行距 = 基线的 blk*page_stride + off*行距
                    # （host 已断言 page_stride == bs*行距，整数恒等）；
                    # 钳位已在 pass 侧逐位完成，这里直接取列。
                    for pair in range_constexpr(SUB // 2):
                        ii = pair * 2
                        c0 = col_base + s2_start + sub_row + ii
                        c1 = c0 + 1
                        r0 = _i64(tab, idx_row, c0)
                        r1 = _i64(tab, idx_row, c1)
                        addr0 = r0 * _ROW_BYTES
                        addr1 = r1 * _ROW_BYTES
                        first = min(addr0, addr1)
                        gap = max(addr0, addr1) - first
                        addresses.append(addr0)
                        addresses.append(addr1)
                        pair_first.append(first)
                        pair_gap.append(gap)
                        all_pairs_ok = dyn_select(gap >= _ROW_BYTES,
                            dyn_select(gap - _ROW_BYTES <= 549755813887,
                                       all_pairs_ok, Int32(0)), Int32(0))
                else:
                    for pair in range_constexpr(SUB // 2):
                        ii = pair * 2
                        pos0 = s2_start + sub_row + min(ii, last_off)
                        pos1 = s2_start + sub_row + min(ii + 1, last_off)
                        tok0 = _i64(idx_tab, idx_row, pos0)
                        tok1 = _i64(idx_tab, idx_row, pos1)
                        blk0, off0 = self.kvcache._pa_blk_off(
                            blk_table, tok0, bo_sp, blk_size)
                        blk1, off1 = self.kvcache._pa_blk_off(
                            blk_table, tok1, bo_sp, blk_size)
                        addr0 = blk0 * page_stride + off0 * _ROW_BYTES
                        addr1 = blk1 * page_stride + off1 * _ROW_BYTES
                        first = min(addr0, addr1)
                        gap = max(addr0, addr1) - first
                        addresses.append(addr0)
                        addresses.append(addr1)
                        pair_first.append(first)
                        pair_gap.append(gap)
                        all_pairs_ok = dyn_select(gap >= _ROW_BYTES,
                            dyn_select(gap - _ROW_BYTES <= 549755813887,
                                       all_pairs_ok, Int32(0)), Int32(0))
                # ★ 批量一致性（deterministic level==3）：禁用成对双行 DMA，退化成逐行搬运。
                #   成对合并会按物理地址交换行序；batch 一致性要求严格按 sparse_indices
                #   顺序一条一条搬，故强制清零 all_pairs_ok 走逐行 mem_copy 分支。
                all_pairs_ok = dyn_select(self._batch_consistency != 0, Int32(0), all_pairs_ok)
                # 每个分支填满16行再提交同一Channel槽。排序同时移动KV和内联scale，
                # QK与PV读取同一排列；尾部重复地址触发逐行回退，保留有效前缀。
                if all_pairs_ok != 0:
                    ub_base, ub_offset = extract_buffer(buf, access="write")
                    gm_base, gm_offset = extract_buffer(kv_bytes, access="read")
                    for pair in range_constexpr(SUB // 2):
                        ascvec.copy_gm2ub(
                            ub_base, arith.addi(ub_offset, _to_index(pair * 2 * _COMBINE_DIM)),
                            gm_base, arith.addi(gm_offset, _to_index(pair_first[pair])),
                            Int32(2), Int32(_ROW_BYTES), dtypes.int64(pair_gap[pair]),
                            Int32(_COMBINE_DIM), Int32(0), Int32(0))
                else:
                    for ii in range_constexpr(SUB):
                        addr = addresses[ii]
                        row = kv_bytes[addr:addr + _ROW_BYTES,].view(1, _ROW_BYTES)
                        mem_copy(tile_view(buf, (1, _ROW_BYTES), (ii, 0)), row)
                # 只消费一次字节槽，VF 内的 scale 位解释不引入第二个 Channel。
                rslot_bytes = self.ch_batch_bytes

                v0_out = self._v0_out
                if const_expr(is_cmp):
                    self._antiquant_vf_fp4_g16(rslot_bytes, v0_out)
                else:
                    self._antiquant_vf_fp8_g32(rslot_bytes, v0_out)
                ub_buf, ub_off = extract_buffer(v0_out, access="read")
                if const_expr(_AIV2L1):
                    # AIV→L1 直写：K̃ 反量化后直接写进 L1 kv_wide 的 tick%3 槽，省掉
                    # GM ws 往返（copy_ub2gm MTE3 + load_qk MTE2 ≈ 0.9+1.13 us/tile）。
                    # L1 槽基址（元素）=(tick%3)*tile_n*D + sub_b*SUB*16，与 GM tile_base 同构。
                    # copy_ub2l1_strided 单位是 32B datablock，且 src_gap/dst_gap 是【间隙】
                    # （stride−burst_len），不是全 stride：每 face 16行×16列×2B=512B=16，
                    # NZ17 间隙 544−512=32B=1，L1 相邻 face(列块) 间隙 4096−512=3584B=112。
                    l1_slot = slot % self.ws_depth
                    l1_buf, l1_off = extract_buffer(self.matmul.kv_wide, access="write")
                    ascvec.copy_ub2l1_strided(
                        l1_buf, arith.addi(l1_off, _to_index(
                            l1_slot * (self.tile_n * D) + sub_b * (SUB * 16))),
                        ub_buf, ub_off,
                        Int32(D // 16), Int32(SUB * 16 * 2 // 32),
                        Int32((_NZ_CHUNK - SUB * 16) * 2 // 32),
                        Int32((self.tile_n - SUB) * 16 * 2 // 32))
                else:
                    gm_buf, gm_off = extract_buffer(self._ws, access="write")
                    # 去掉 UB 的第 17 行，将 32 个 face 写入 GM 的 [32,128,16] 槽：
                    # GM BF16 offset = slot*128*512 + (col//16)*128*16 + token*16 + col%16。
                    # 每笔 burst 搬 16×16×2=512B；相邻 face 的 GM/UB 起点间距为 4096/544B。
                    tile_base = (slot * (self.tile_n * D)
                                 + sub_b * (SUB * 16))
                    # 裸 DMA 的 stride 参数要求 i64 SSA，不能直接传 Python 常量。
                    ascvec.copy_ub2gm(
                        gm_buf, arith.addi(gm_off, _to_index(tile_base)),
                        ub_buf, ub_off,
                        Int32(D // 16), Int32(SUB * 16 * 2),
                        dtypes.int64(self.tile_n * 16 * 2), Int32(_NZ_CHUNK * 2))

    @jit
    def _addr_precompute(self, m_count):
        """R' 前置 pass（Channel 1:1 纪律版）。

        两扫（ori→cmp）把每 query 每池的 u32 物理行号写进 GM 表 [T1, k1+k2]。
        valid-counter 连续等分到本 block 两 AIV（行唯一写者）。每 query：
        stage sp（一次生产）→ 单 vf 块算完整行（全部 gather 消费 + vstore
        out 一次生产）→ 整行一笔写表（一次消费）。列按 min(col, w-1) 钳位
        （与消费侧逐子块钳位逐位等价）。计数器与键的合并全部 dyn_select。
        末尾网格屏障三连（NVF 诊断已隔离验证屏障本身无罪）。
        """
        tab_cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
        # ---- B4-(1) 增量式 batch 推进 ----
        # m_idx = _m_start + m_seq 是连号单调的，两个 side 又跑同一串 m_idx，
        # 所以整个前置段只需要 **一次** 二分查找（_batch_of(_m_start)），之后
        # 每行把 batch 往前推到 cu_q[batch+1] > m_idx 为止。m_idx 每轮只 +1 ⇒
        # 整个循环跨界次数 <= 覆盖的 span 数 ⇒ 摊销 O(1) 次 GM 标量读，取代原来
        # 每 side 每行一整条 5 层相互依赖的二分链（主 case 30 次串行 GM 读）。
        b_last = dtypes.int64(self._b_rt)
        batch0 = self._batch_of(self._m_start)
        bound0 = self._bound_after(batch0, b_last)
        for side in range_constexpr(2 if self.has_cmp else 1):
            if const_expr(side == 0):
                idx_tab, bt_gm = self._ori_idx, self._pa_bt
                bs, kwidth, col0 = self._avec_bs_ori, self._avec_k1, 0
                ktab = self._avec_k1_tab

                def w_fn(mi):
                    return self._ori_valid_w(mi)
            else:
                idx_tab, bt_gm = self._cmp_idx, self._pa_bt_cmp
                bs, kwidth, col0 = (self._avec_bs_cmp, self._avec_k2,
                                    self._avec_k1_tab)
                ktab = self._avec_k2_tab

                def w_fn(mi):
                    return self._cmp_topk_len(mi)
            btp = bt_gm.shape[1]
            shift = bs.bit_length() - 1
            # 覆盖宽度是 ktab（按 _TILE_ROWS 上取整）而不是真实 kwidth：消费侧
            # 每个 16 行子块恒读满一整子块的列，见 _addr_tab_w。
            # ktab 不是 _ADDR_HALF 的整数倍时必须向上取整并给尾块单独出掩码：
            # 整除会让 VF 少算最后那几列，而收尾的 ch2gm 照样搬 ktab 个，
            # 尾列搬出去的是 _ch_addr_out（depth=1 复用）里上一行的残留行号
            # ——是"看起来合法"的地址，静默读到别的 query 的 KV，不会报错。
            # ktab % _ADDR_HALF == 0 时 tail_lanes == _ADDR_HALF、尾掩码即全掩码，
            # 与修复前逐位等价（K=128/512 是这一支，零行为变化）。
            chunks = -(-ktab // _ADDR_HALF)
            tail_lanes = ktab - (chunks - 1) * _ADDR_HALF
            total_valid = Int64(0)
            for m_seq in range(0, m_count):
                w0 = w_fn(self._m_idx(m_seq))
                total_valid = total_valid + dyn_select(w0 > Int64(0), Int64(1), Int64(0))
            per = total_valid // 2
            tail = total_valid % 2
            core = dtypes.int64(self.subblock_idx)
            # B3：两侧的取整方向相反 ⇒ 等价于把 (query, side) 按 query-major 序
            # [(q0,ori),(q0,cmp),(q1,ori),...] 连续等分给两个 AIV。
            # total_valid 为偶数时与旧划分逐位相同（对照组零风险）；为奇数时
            # 多出来的 ori 归 aiv0、多出来的 cmp 归 aiv1，把 block 内 2:1 的
            # 偏斜压成 (2 ori+1 cmp) : (1 ori+2 cmp)。零额外标量运算。
            if const_expr(side == 0):
                start = per * core + min(core, tail)
                count = per + dyn_select(core < tail, Int64(1), Int64(0))
            else:
                start = per * core
                count = per + dyn_select(core < Int64(1), Int64(0), tail)
            counter = Int64(0)
            bt_staged = Int64(-1)
            # batch 必须在 ForOp 之前绑定：增量臂里它只在 while 体内被改写，
            # while 可能零轮 ⇒ 前端 FE201 要求它是显式 loop-carried state。
            # 每个 side 从同一个起点重新推进（SSA 拷贝，无 GM 读）。
            batch = batch0
            bound = bound0
            for m_seq in range(0, m_count):
                m_idx = self._m_idx(m_seq)
                while m_idx >= bound:
                    batch = batch + 1
                    bound = self._bound_after(batch, b_last)
                w = w_fn(m_idx)
                active = dyn_select(w > Int64(0), Int64(1), Int64(0)) * \
                    dyn_select(counter >= start, Int64(1), Int64(0)) * \
                    dyn_select(counter < start + count, Int64(1), Int64(0))
                if active == Int64(1):
                    _raw_gm2ch(self._ch_addr_sp, idx_tab,
                               m_idx * kwidth, kwidth * 4)
                    key = batch * 2 + side
                    if key != bt_staged:
                        _raw_gm2ch(self._ch_addr_bt, bt_gm,
                                   batch * btp, btp * 4)
                    with vf(mode="raw"):
                        m32, _ = rr.update_mask(_ADDR_HALF, elem_bits=32)
                        # 尾掩码在循环外备好（本仓既有写法：多掩码并存）。
                        m_tail = (m32 if tail_lanes == _ADDR_HALF else
                                  rr.update_mask(tail_lanes, elem_bits=32)[0])
                        lane = rr.varange(0, dtypes.int32)
                        for chunk in range(chunks):
                            mk = m32 if chunk < chunks - 1 else m_tail
                            col = rr.vadds(lane, chunk * _ADDR_HALF, mask=mk)
                            col = rr.vmins(col, w - Int64(1), mask=mk)
                            idx = rr.vgather(self._ch_addr_sp,
                                             rr.vreinterpret(col, dtypes.uint32),
                                             mask=mk)
                            blk = rr.vshr(idx, shift, mask=mk)
                            off = rr.vsub(idx, rr.vmuls(blk, bs, mask=mk), mask=mk)
                            p = rr.vgather(self._ch_addr_bt,
                                           rr.vreinterpret(blk, dtypes.uint32),
                                           mask=mk)
                            row = rr.vadd(rr.vmuls(p, bs, mask=mk), off, mask=mk)
                            rr.vstore(self._ch_addr_out, chunk * _ADDR_HALF,
                                      row, mk)
                    dst = m_idx * tab_cols + col0
                    # 写 ktab 列（含按 w-1 钳位的补齐列）；ktab 是 _TILE_ROWS 的
                    # 整数倍 ⇒ 每笔起止都落在 32B 边界，UB→GM 无 pad 能力也安全。
                    _raw_ch2gm(self._addr_tab, self._ch_addr_out, dst, ktab)
                counter = counter + dyn_select(w > Int64(0), Int64(1), Int64(0))
                # active==1 已蕴含 w>0，直接合并键
                bt_staged = dyn_select(active == Int64(1), batch * 2 + side, bt_staged)
        # 收尾：block 内 AIV 对握手（V6b）。行号表的每行唯一写者是本核的 query，
        # 读者只有本 block 的两个 AIV（vec0 按 subblock 分工），AIC 不读表
        # ⇒ 跨 block 零依赖。两 AIV 各以 MTE3 管 intra arrive 同一 sid（管语义 =
        # 本核写表的 DMA 已完成；AIC 侧按 +_AIV1_ID_OFFSET 自动区分 AIV1）→
        # AIC 等齐两侧后 block arrive flag → AIV 的 S 管 block wait。零轮询，
        # 且快核的 vec0 可与慢核仍在跑的 pre 重叠；wait 挂 S 管，主循环的标量读表
        # 自然排在其后。跨 AIV 的数据可见性全靠它，不可删。
        #
        # 这里曾经还有一条 `vec_sync_all()`（本 AIV 全管排空）。它之所以看起来
        # "不可删"，是因为当时 AIC 中继的放行挂在 MTE3、等待挂在 MTE1，跨管无序，
        # 屏障其实被整个越过（见下面 __call__ 里的注释）——那条排空是当时唯一在
        # 起作用的定序手段，删掉它自然会炸（b=32,s1=6 复现 507015 2/2）。
        # 屏障修好之后重测：删除排空在 b=32s6 / b=32s1 / b=4s6 K=133/517 /
        # b=16s6 K=133/517 四组共 28 次全部 bit-identical，故移除。
        vec_sync_intra_arrive(PIPE.MTE3, self._addr_bar_sid)
        vec_sync_block_wait(PIPE.S, self._addr_bar_flag, mode=2)

    @jit
    def _stage_vec0(self, tick: int, m_seq: int, n):
        # 按 n 选择 ori/cmp 的池、索引表和有效宽度，写同一条 ws 任务环。
        # cmp 的局部 tile 号 c=n−ori_tiles，从索引表 c*128 列开始取数；其地址表行偏移为 T1。
        # 两 AIV 即使本 tile 无子块可处理，也都在末尾发布 ws-ready，供 AIC 成对等待。
        m_idx = self._m_idx(m_seq)
        # pre 向量臂的寻址走预计算行号表，bo_sp 仅标量路径消费；逐 tile 的
        # _batch_of 二叉搜索（每 tile ~log2(B) 次 GM 标量读）在 pre 下纯属浪费。
        batch = None
        if const_expr(_addr_vec_mode() != "pre"):
            batch = self._batch_of(m_idx)
        max_blk = self._pa_bt.shape[1]
        max_blk_cmp = self._pa_bt_cmp.shape[1] if self.has_cmp else None
        slot = self._ws_core_slot0 + tick % self.ws_depth

        if const_expr(_AIV2L1):
            # free 握手：等 bmm2(tick-3) 读完本 L1 槽后才可覆写（3 深环同槽复用）。
            vec_sync_intra_wait(PIPE.MTE3, self._kv_free_id)

        if const_expr(not self.has_cmp):
            self._vec0_body(self._pa_phys, self._pa_bt, self._pa_bs_rt, max_blk,
                            n * self.tile_n, slot, self._ori_width(m_idx, n),
                            idx_tab=self._ori_idx, idx_row=m_idx, batch=batch,
                            is_cmp=False)
        else:
            ori_tiles = self._ori_tiles_rt(m_idx)
            ori_bound = ori_tiles
            if n >= ori_bound:
                c = n - ori_bound
                self._vec0_body(self._pa_phys_cmp, self._pa_bt_cmp, self._pa_cmp_bs_rt,
                                max_blk_cmp, c * self.tile_n, slot,
                                self._cmp_valid(m_idx, c),
                                idx_tab=self._cmp_idx, idx_row=m_idx, batch=batch,
                                is_cmp=True)
            else:
                self._vec0_body(self._pa_phys, self._pa_bt, self._pa_bs_rt, max_blk,
                                n * self.tile_n, slot, self._ori_width(m_idx, n),
                                idx_tab=self._ori_idx, idx_row=m_idx, batch=batch,
                                is_cmp=False)

        self._ws_ready_arrive()

    @jit
    def _ws_ready_arrive(self):
        # ws-ready 握手（AIV MTE3 写完槽 → arrive）；走 mode=4 指令路径不过 FFTS。
        vec_sync_intra_arrive(PIPE.MTE3, self._ws_ready_id)

    @jit
    def _stage_bmm1_fanout(self, kv_slot, m_seq: int, is_first: int, is_last: int,
                           row_count):
        # lag2 cube：按 AscendC 的 Q 半槽事件读取常驻 Q，QK 经 fixpipe 送 qk_ub。
        self.matmul.bmm1_fanout_q(kv_slot, m_seq, is_first, is_last)
        # bmm1_fanout_q returns the Q credit on the part's last KV tile. Issue
        # the next M copy now, before lag1 reaches it, retaining cross-M
        # overlap. Empty rows never run bmm1 and thus never consume/return a
        # Q credit; the prefetch must skip them or the ring deadlocks (the
        # metadata splitter can leave empty rows inside a core's range).
        if is_last == dtypes.int64(1):
            # 双状态收敛扫描（与 _batch_of 同构：两 i64 状态都出现在条件里，
            # 避免非对称 scf.while 状态导致 AscendC 翻译失败）：
            # 命中非空行时 stop←j 使条件立即为假，j 停在目标行。
            j = m_seq + dtypes.int64(1)
            stop = row_count
            while j < stop:
                found = self._n_end_rt(self._m_idx(j)) != 0
                stop = dyn_select(found, j, stop)
                j = dyn_select(found, j, j + dtypes.int64(1))
            if j < row_count:
                q_queries = self._q.view(self._q.shape[0] // self.tile_cube_m,
                                         self.tile_cube_m, self.tile_d)
                next_q = q_queries[self._m_idx(j), None, None]
                self.matmul.load_q(next_q, j)
        self.matmul.store_s(self.qk_ub, self._split_m())

    @jit
    def _stage_softmax(self, tick: int, m_seq: int, n, is_first: int):
        # lag2 AIV：Channel 自动管理 qk_ub 的消费事务 →
        # online softmax（分部首 tile 播种：整行首分部用 sinks，FD 续算分部用
        # (FP32_MIN, 0)）→ store_p 送 p_l1。
        # 槽轮转：max/sum 按 m_seq%2、exp 按 tick%2（承重，见 Vector 类注释）。
        m_idx = self._m_idx(m_seq)
        actual_n = self._actual_n(m_idx, n)
        split_m = self._split_m()
        actual_vec_m = self.tile_vec_m

        m_axis_triple = m_seq % 2
        tile_triple = tick % 2

        qk_slot = self.qk_ub
        qk_view = local_slice(qk_slot, (actual_vec_m, self.tile_n), stride=(self.tile_n, 1))
        # 续算分部 = 本核首行(m_seq==0)且 s2_start>0；其首 tile 不得再播种 sinks。
        cont_part = dyn_select(m_seq == 0,
                               dyn_select(self._s2_start > 0, Int32(1), Int32(0)),
                               Int32(0))
        if is_first == dtypes.int64(1):
            if cont_part == Int32(1):
                self.vector.seed_empty(m_axis_triple)
            else:
                sinks_coord = self.subblock_idx
                sk = self.vector.sinks_ub
                mem_copy(sk, tile_view(self._sinks, (1, self.sinks_span),
                                       (0, sinks_coord)))
                skr = self.vector.sinks_ub
                self.vector.seed_from_sinks(skr, m_axis_triple)
        self.vector.softmax_rest(qk_view, self._scale, m_axis_triple, tile_triple, actual_n)

        self.vector.store_p(self.p_l1, split_m)

    @jit
    def _stage_pv_fanout(self, kv_slot, m_seq: int, n, tick):
        # lag3 cube：O_tile = P@K̃（P 读 p_l1，K̃ 转置读同一代 L1 环）→ fixpipe 送 pv_ub。
        split_m = make_partition_tiler(
            (self.tile_cube_m, self.tile_d), (self.tile_cube_m, self.tile_d),
        )
        self.matmul.compute_pv_fanout(self.p_l1, self.pv_ub, split_m,
                                      kv_slot, self._actual_n(self._m_idx(m_seq), n), tick)
        if const_expr(_AIV2L1):
            # free 握手：bmm2 已读完 L1 槽，通知两 AIV 可覆写下一份（MTE3 写同槽）。
            cube_sync_intra_arrive(PIPE.MTE1, self._kv_free_id)
            cube_sync_intra_arrive(PIPE.MTE1, self._kv_free_id + _AIV1_ID_OFFSET)

    @jit
    def _stage_loadqk(self, tick: int, m_seq: int):
        row_tile = self._ws_core_slot0 + tick % self.ws_depth
        q_queries = self._q.view(self._q.shape[0] // self.tile_cube_m,
                                 self.tile_cube_m, self.tile_d)
        q_tile_gm = q_queries[self._m_idx(m_seq), None, None]
        self.matmul.load_qk(self._ws, row_tile, self.matmul.kv_slot(tick),
                            q_tile_gm, m_seq, tick, self._ws_ready_id)

    @jit
    def _stage_qk_softmax(self, tick: int, m_seq: int, n, is_last: int,
                          is_first: int, row_count):
        if const_expr(_AIV2L1):
            # AIV→L1：bmm1 读 L1 槽前等两 AIV 的 MTE3 写完（ready 握手）。
            cube_sync_intra_wait(PIPE.MTE1, self._ws_ready_id)
            cube_sync_intra_wait(PIPE.MTE1, self._ws_ready_id + _AIV1_ID_OFFSET)
        else:
            self.matmul.kv_event(tick, PIPE.MTE2, PIPE.MTE1, False)
        self._stage_bmm1_fanout(self.matmul.kv_slot(tick), m_seq, is_first,
                                is_last, row_count)
        self._stage_softmax(tick, m_seq, n, is_first)

    @jit
    def _stage_pv_update(self, tick: int, m_seq: int, n, is_last: int,
                         is_first: int, is_split: int):
        self._stage_pv_fanout(self.matmul.kv_slot(tick), m_seq, n, tick)
        self._stage_update(tick, m_seq, n, is_last, is_first, is_split)

    @jit
    def _stage_update(self, tick: int, m_seq: int, n, is_last: int,
                      is_first: int, is_split: int):
        # lag3 按任务顺序消费 PV；exp_idx=tick%2，sum_idx=m_seq%2，与 softmax 写槽一致。
        # (首,末)=(1,1)：PV/sum；(1,0)：复制 PV；(0,0)：alpha*O+PV；
        # (0,1)：(alpha*O+PV)/sum。只有整行末 tile 才 finalize 写 O/LSE。
        # FD 切分行的分部末 tile 不除分母、不写输出：未归一化 O 与 max/sum
        # 暂存到 workspace slot（首行分部用 first_fd+0，尾行分部用 first_fd+1，
        # = AscendC firstFdDataWorkspaceIdx + s2SplitIdx 语义），归约在 barrier 后。
        # 两侧宽度同时为 0 时无流水任务；drain 后单独写出 O=0、LSE=sink。
        exp_idx = tick % 2
        sum_idx = m_seq % 2

        m_idx = self._m_idx(m_seq)
        o_row_tile = m_idx
        lse_tile_base = (
            o_row_tile * (self.tile_cube_m // self.tile_vec_m)
            + self.subblock_idx) if const_expr(self._lse is not None) else None
        # lse_tile_base 是 32 元素 tile 的编号；对应扁平 LSE 元素偏移 m_idx*64+aiv_id*32。
        if is_last == 1:
            if is_first == 1:
                if tick > dtypes.int64(0):
                    self.vector.wait_prev_out_copy()
                self.vector.init_o_last(self.pv_ub, sum_idx)
            else:
                self.vector.update_o_last(self.pv_ub, exp_idx, sum_idx)
            if const_expr(_FD_COMPILE):
                if is_split == 1:
                    # slot = first_fd + s2SplitIdx（= AscendC 语义）：本核至多暂存两份
                    # 分部——首行续算分部（m_seq==0，s2SplitIdx=0）与尾行起切分部
                    # （m_seq==row_count-1）。尾行分部的 s2SplitIdx 仅当首行也被暂存
                    # （即本核以续算分部开头，s2_start>0）时才为 1，否则为 0。
                    # 此处已按 AscendC 语义除以分母 sum（init_o_last/update_o_last），
                    # 暂存的是归一化 O。
                    part_idx = dyn_select(
                        m_seq == 0, dtypes.int64(0),
                        dyn_select(self._s2_start > 0, dtypes.int64(1), dtypes.int64(0)))
                    slot = self._first_fd + part_idx
                    self.vector.stage_partial(self._fd_o, self._fd_ms, self.fd_ub,
                                              slot, sum_idx)
                else:
                    o_tile_gm = tile_view(self._out, (self.tile_cube_m, self.tile_d),
                                          (o_row_tile, 0))
                    self.vector.finalize_o(o_tile_gm, sum_idx,
                                           lse_gm=self._lse, lse_base=lse_tile_base)
            else:
                o_tile_gm = tile_view(self._out, (self.tile_cube_m, self.tile_d),
                                      (o_row_tile, 0))
                self.vector.finalize_o(o_tile_gm, sum_idx,
                                       lse_gm=self._lse, lse_base=lse_tile_base)
        else:
            if is_first == 1:
                if tick > dtypes.int64(0):
                    self.vector.wait_prev_out_copy()
                self.vector.init_o(self.pv_ub)
            else:
                self.vector.update_o(self.pv_ub, exp_idx)

    def __call__(self, out_gm: Tensor, q_gm: Tensor,
                 sinks_gm: Tensor,
                 pa_phys_gm: Tensor = None, pa_bt_gm: Tensor = None, ws_gm: Tensor = None,
                 pa_phys_cmp_gm: Tensor = None, pa_bt_cmp_gm: Tensor = None,
                 cmp_idx_gm: Tensor = None,
                 ori_idx_gm: Tensor = None,
                 cu_seqlens_q: Tensor = None,
                 softmax_lse: Tensor = None,
                 ori_topk_length: Tensor = None, cmp_topk_length: Tensor = None,
                 metadata_gm: Tensor = None, softmax_scale: dtypes.float32 = 1.0,
                 batch_consistency: dtypes.int32 = 0,
                 addr_ws_gm: Tensor = None):
        # 固定 1024 metadata 不再携带核数；发射网格与 FD slot 数由 workspace
        # 容量反推（construct_workspace 与 metadata 生产者查询同一核数来源）。
        self.blocks = ws_gm.shape[0] // WS_BYTES_PER_CORE
        self._b_rt = cu_seqlens_q.shape[1] - 1
        self._scale = softmax_scale
        self._batch_consistency = batch_consistency
        self._ori_page_stride = pa_phys_gm.stride[0]
        self._pa_bs_rt = pa_phys_gm.shape[1]
        if const_expr(self.has_cmp):
            self._cmp_page_stride = pa_phys_cmp_gm.stride[0]
            self._pa_cmp_bs_rt = pa_phys_cmp_gm.shape[1]
        v0_rows = self.blocks * self.ws_depth * self.tile_n
        # workspace 布局：[v0 区 blocks*3*128 行 bf16][FD O 区][FD max/sum 区]。
        # v0 区 = 每核三深 NZ 环；FD 区为 fp32：O (slots*64, 512)，max/sum (slots*128,)，
        # slots = 2/核（常规 FD 每核最多暂存首/尾两个分部），区域序对齐 AscendC
        # S2SplitFdStagingLayout 的 [O][max][sum] 排布。
        # v0 视图不做区间切片：load_qk 需要对底层指针做 NZ 重解释。
        ws_u8 = ws_gm
        ws_gm = ws_u8.view(dtypes.bfloat16).view(ws_u8.shape[0] // (D * 2), D)
        slots = self.blocks * FD_SLOTS_PER_CORE
        ws_f32 = ws_u8.view(dtypes.float32)
        fd_o0 = v0_rows * D // 2
        fd_o = ws_f32[fd_o0:fd_o0 + slots * FD_O_ELEMS,].view(slots * N1, D)
        fd_ms0 = fd_o0 + slots * FD_O_ELEMS
        fd_ms = ws_f32[fd_ms0:fd_ms0 + slots * FD_MS_ELEMS,].view(slots * FD_MS_ELEMS)
        self._fd_o, self._fd_ms = fd_o, fd_ms
        self._q = q_gm
        self._sinks = sinks_gm
        self._out = out_gm
        self._lse = softmax_lse
        self._ori_topk_length = ori_topk_length
        self._cmp_topk_length = cmp_topk_length

        self._cu_q = cu_seqlens_q

        self._pa_phys, self._pa_bt, self._ws = pa_phys_gm, pa_bt_gm, ws_gm
        self._ori_idx = ori_idx_gm
        if const_expr(self.has_cmp):
            self._pa_phys_cmp, self._pa_bt_cmp = pa_phys_cmp_gm, pa_bt_cmp_gm
            self._cmp_idx = cmp_idx_gm
        self._ws_core_slot0 = self.block_idx * self.ws_depth
        _v0_shape = (D // 16, _NZ_PAD_ROWS, 16)
        self._v0_out = Channel(MemLoc.UB, shape=_v0_shape, dtype=dtypes.bfloat16, depth=2)

        # ---- 地址向量化臂的核内资源（仅向量臂编译时存在；off 的 UB/id 预算不变）----
        # Channel 锁括号是本仓唯一经 bit 级验证的 MTE2→V→MTE3 定序原语
        # （pair-DMA→dequant 路径）。其纪律 = 每次"生产"（一次 write 括号）
        # 配一次"消费"（一个 vf 块的全部读 / 一笔读括号拷贝）；跨 chunk 的
        # 多次消费会让槽轮转计数失步（板上实证：表 98.6% 陈旧值）。因此
        # pre 的 VF 每 query 单块算完整行、整行单笔写表；tile 同理。
        _avec = _addr_vec_mode()
        if const_expr(_avec == "pre"):
            from cannbotdsl.arena import _current_channel_arena
            self._avec_bs_ori = pa_phys_gm.shape[1]
            self._avec_k1 = ori_idx_gm.shape[1]
            self._avec_k1_tab = _addr_tab_w(self._avec_k1)
            self._avec_k2_tab = 0
            if const_expr(self.has_cmp):
                self._avec_bs_cmp = pa_phys_cmp_gm.shape[1]
                self._avec_k2 = cmp_idx_gm.shape[1]
                self._avec_k2_tab = _addr_tab_w(self._avec_k2)
            btp = max(pa_bt_gm.shape[1],
                      pa_bt_cmp_gm.shape[1] if self.has_cmp else 0)
            self._ch_addr_bt = Channel(MemLoc.UB, shape=(1, btp),
                                       dtype=dtypes.int32, depth=1,
                                       kind=ChannelKind.SameCore)
            sp_w = max(self._avec_k1, self._avec_k2 if self.has_cmp else 0)
            self._ch_addr_sp = Channel(MemLoc.UB, shape=(1, sp_w),
                                       dtype=dtypes.int32, depth=1,
                                       kind=ChannelKind.SameCore)
            # out 侧按补齐宽度分配：VF 要算到 ktab 列，比 sp_w 最多多 15 个。
            out_w = max(self._avec_k1_tab, self._avec_k2_tab)
            self._ch_addr_out = Channel(MemLoc.UB, shape=(1, out_w),
                                        dtype=dtypes.int32, depth=1,
                                        kind=ChannelKind.SameCore)
            tab_cols = self._avec_k1_tab + (self._avec_k2_tab
                                            if self.has_cmp else 0)
            self._addr_tab = addr_ws_gm.view(dtypes.int32).view(
                addr_ws_gm.shape[0] // 4 // tab_cols, tab_cols)
            self._addr_bar_flag = _burn_to_floor(
                _current_channel_arena().alloc_flag_ids, 8, 1)[0]
            # block 内 AIV 对握手（V6b）用的 intra sync id：两个 AIV arrive 同一
            # id，AIC 侧按 +_AIV1_ID_OFFSET 区分 AIV1，故只需分配一个。
            self._addr_bar_sid = _burn_to_floor(
                _current_channel_arena().alloc_sync_ids, 8, 1)[0]

        # ---- FA metadata（faMetadata[block_idx][*]，字段对齐 AscendC）----
        # 区间语义：本核处理行 [m_start, m_end]（s2_end==0 时末行为 m_end-1）；
        # 首行从 tile s2_start 起、末行到 tile s2_end 止（0=整行）。
        # start[i] == end[i-1]，即 metadata 编码 (行, tile) 工作空间的连续半开划分。
        # 禁用核整行字段为 0 ⇒ row_count=0，自然跳过主循环，只参与 barrier。
        fa = self.block_idx * FA_METADATA_SIZE
        bn2_start = _i64(metadata_gm, fa + FA_BN2_START_INDEX)
        m_start_g = _i64(metadata_gm, fa + FA_M_START_INDEX)
        bn2_end = _i64(metadata_gm, fa + FA_BN2_END_INDEX)
        m_end_g = _i64(metadata_gm, fa + FA_M_END_INDEX)
        self._m_start = _i64(self._cu_q, 0, bn2_start) + m_start_g
        self._s2_start = _i64(metadata_gm, fa + FA_S2_START_INDEX)
        m_end = _i64(self._cu_q, 0, bn2_end) + m_end_g
        s2_end = _i64(metadata_gm, fa + FA_S2_END_INDEX)
        self._first_fd = _i64(metadata_gm, fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX)
        row_count = m_end - self._m_start + (1 if s2_end > 0 else 0)
        # FD 全局使能计数（= 2*归约任务数，全核一致）：提前读一次，供主循环用
        # 运行时 if 在 FD-off 时跳过 t0/t1/split 的 dyn_select 脚手架；FD-off 时
        # metadata 已保证 s2_start=0 且 s2_end=0（每核整行处理），恒有
        # t0=0 / t1=tiles / split=0。
        fd_any = _i64(metadata_gm, FD_USED_VEC_NUM_WORD)
        # DelayLineGroup 只延迟任务坐标，不保存张量；深度 4 对应最大 lag=3。
        # g 是本核流水时钟，s.t 是被取出的任务编号；每级必须用 s.t 选择该任务的环槽。
        # m/n/last/first/split 随任务一起传递，跨 query 边界也不能拿当前循环坐标
        # 替代延迟坐标；first/split 由 metadata 行区间派生（FD 分部边界）。
        lag_load = 1
        lag_b = 2
        lag_c = lag_b + 1
        DEPTH = lag_c + 1
        DRAIN = lag_c
        dl = DelayLineGroup(DEPTH, 'm', 'n', 't', 'last', 'first', 'split')
        g = 0
        if const_expr(_AIV2L1):
            # free 握手初始化：3 个 L1 槽起始都空闲，预发 3 份 free（AIC MTE1 → 两 AIV）。
            # 与 kv_events_init 的 3 槽 credit 同构；否则首 3 个 vec0 会等一份永不来的 free。
            for slot in range_constexpr(_KV_RING):
                cube_sync_intra_arrive(PIPE.MTE1, self._kv_free_id)
                cube_sync_intra_arrive(PIPE.MTE1, self._kv_free_id + _AIV1_ID_OFFSET)
        else:
            self.matmul.kv_events_init()
        self.matmul.q_events_init()

        if const_expr(_addr_vec_mode() == "pre"):
            # R' 前置 pass：行号表写完 + 屏障之后，主循环才能消费读表。
            self._addr_precompute(row_count)
            # V6b 的 AIC 中继：等齐本 block 两个 AIV 的写表完成，再放行它们。
            # wait 挂 MTE1——队首空闲，且 MTE1 的首个真实消费者 bmm1 本就在 MTE2
            # 下游，不增加延迟；挂 MTE2 会挡住 q 预取，移到 trace 尾则与 loadQK
            # 成环死锁（两者皆为上游板上实证）。
            cube_sync_intra_wait(PIPE.MTE1, self._addr_bar_sid)
            cube_sync_intra_wait(PIPE.MTE1,
                                 self._addr_bar_sid + _AIV1_ID_OFFSET)
            # 放行必须与等待**同管**。原先 arrive 挂 MTE3，而两次 wait 挂 MTE1，
            # 两条管并行 ⇒ AIC 开头 MTE3 队列为空时 arrive 会立刻发出，不等那两次
            # wait，整个屏障被越过。此时正确性退化成"AIV1 读表时 AIV0 写完没有"，
            # 而 m_count==1 是唯一一种某个 AIV 前置段完全空转的配置（它直接冲进
            # 主循环读另一个 AIV 还在写的表）⇒ 静默错数，b=32,s1=1 实测 7/12。
            # 改挂 MTE1 后同形状 0/12；等价修法（arrive 前加 cube_sync_all）同样
            # 归零，共同点只有"放行排在等待之后"。零新增指令。
            cube_sync_block_arrive(PIPE.MTE1, self._addr_bar_flag, mode=2)

        # ---- warmup / steady：每核 g 从 0 递增，跨 query 及 ori→cmp 边界不重置 ----
        # g>=lag 才存在该级任务；每级按 Channel 依赖分别发往 AIC/AIV，Python 调用顺序
        # 不等于所有硬件串行执行。每个 query 仅 n 从 t0 起计数；first 触发 softmax/O
        # 初始化（续算分部用空种子），last+split 触发 FD 暂存而非 finalize。
        # row_count=0 的核不发任务；分页查表不需要全核地址预计算屏障。
        for m_seq in range(0, row_count):
            m_idx = self._m_idx(m_seq)
            tiles = self._n_end_rt(m_idx)
            t0 = dtypes.int64(0)
            t1 = dtypes.int64(tiles)
            split = dtypes.int64(0)
            if const_expr(_FD_COMPILE):
                if fd_any != 0:
                    t0 = dyn_select(m_seq == 0, self._s2_start, dtypes.int64(0))
                    t1_tail = dyn_select(s2_end > 0, s2_end, tiles)
                    t1 = dyn_select(m_seq == row_count - 1, t1_tail, tiles)
                    split = dyn_select(t0 > 0, dtypes.int64(1),
                                       dyn_select(t1 < tiles, dtypes.int64(1), dtypes.int64(0)))
            for n in range(t0, t1):
                is_last = 1 if n == t1 - 1 else 0
                is_first = 1 if n == t0 else 0
                dl.push(m=m_seq, n=n, t=g, last=is_last, first=is_first, split=split)

                self._stage_vec0(g, m_seq, n)
                if g >= lag_load:
                    s0 = dl.tap(lag_load)
                    self._stage_loadqk(s0.t, s0.m)
                if g >= lag_b:
                    s1 = dl.tap(lag_b)
                    self._stage_qk_softmax(s1.t, s1.m, s1.n, s1.last, s1.first,
                                           row_count)

                if g >= lag_c:
                    s2 = dl.tap(lag_c)
                    self._stage_pv_update(s2.t, s2.m, s2.n, s2.last, s2.first,
                                          s2.split)

                dl.advance()
                g = g + 1
        # ---- drain：不再 push 新任务，固定推进 3 拍，让末尾坐标走完所有后级 ----
        # r<lag 排除该级已排空的拍；g>=lag 排除任务数过少时尚未填充的 tap。
        # 即使只发 1 个任务，也会依次补跑 loadQK、QK/softmax、PV/update；空核各守卫均不通过。
        for r in range_constexpr(DRAIN):
            if const_expr(r < lag_load):
                if g >= lag_load:
                    s0 = dl.tap(lag_load)
                    self._stage_loadqk(s0.t, s0.m)
            if const_expr(r < lag_b):
                if g >= lag_b:
                    s1 = dl.tap(lag_b)
                    self._stage_qk_softmax(s1.t, s1.m, s1.n, s1.last, s1.first,
                                           row_count)
            if const_expr(r < lag_c):
                if g >= lag_c:
                    s2 = dl.tap(lag_c)
                    self._stage_pv_update(s2.t, s2.m, s2.n, s2.last, s2.first,
                                          s2.split)
            dl.advance()
            g = g + 1

        if const_expr(not _AIV2L1):
            self.matmul.kv_events_drain()
        self.matmul.q_events_drain()

        # Empty queries never enter the delay line. Write their outputs after
        # draining so the shared output Channel cannot disturb pending PVs.
        for m_seq in range(0, row_count):
            m_idx = self._m_idx(m_seq)
            if self._n_end_rt(m_idx) == 0:
                o_tile = tile_view(self._out, (self.tile_cube_m, self.tile_d),
                                   (m_idx, 0))
                lse_base = (m_idx * 2 + self.subblock_idx
                            if const_expr(self._lse is not None) else None)
                self.vector.finalize_empty(o_tile, self._sinks, self._lse, lse_base)

        # ---- FD 归约阶段（= AscendC Process：ProcessMainLoop → SyncAll →
        # ProcessFlashDecode）：全网格 barrier 后，fdMetadata 使能的 AIV 合并
        # 跨核切分行的分部结果并写出最终 O/LSE。禁用 AIV 直接结束。
        # fd_any 是全核一致的 metadata 字（= 2*归约任务数，主循环前已读）：为 0 时
        # 无任何切分，整段 barrier+归约跳过 ⇒ FD-off 用例与无 FD 基线逐指令等价、
        # 零额外开销。barrier 必须全核同进同出；fd_any 各核读同一 GM 字，分支天然一致。
        if const_expr(_FD_COMPILE):
            if fd_any != 0:
                global_sync_all()
                # 主循环已结束、global_sync_all 已隔离：rewind 后分配归约专属 buffer，
                # 复用主循环 UB 地址，把归约落地 fd_ub 抬到整块 (32,512) fp32（64KiB）。
                channel_rewind(reset_sync_id=False)
                self.vector._setup_fd_reduce_buffers()
                aiv_idx = self.block_idx * get_subblock_dim() + self.subblock_idx
                fdb = FD_METADATA_BASE + aiv_idx * FD_METADATA_SIZE
                fd_enable = _i64(metadata_gm, fdb + FD_CORE_ENABLE_INDEX)
                if fd_enable != 0:
                    fd_bn2 = _i64(metadata_gm, fdb + FD_BN2_IDX_INDEX)
                    fd_m = _i64(metadata_gm, fdb + FD_M_IDX_INDEX)
                    fd_row = _i64(self._cu_q, 0, fd_bn2) + fd_m
                    fd_ws = _i64(metadata_gm, fdb + FD_WORKSPACE_IDX_INDEX)
                    fd_k = _i64(metadata_gm, fdb + FD_WORKSPACE_NUM_INDEX)
                    fd_h0 = _i64(metadata_gm, fdb + FD_M_START_INDEX)
                    self.vector.fd_reduce(self._fd_o, self._fd_ms, self.vector.fd_ub,
                                          self._out, self._lse,
                                          fd_row, fd_ws, fd_k, fd_h0)

mqsmla = sys.modules[__name__]

def _batch_consistency_enabled():
    """批量一致性开关：torch_npu 确定性等级 == 3 时开启。

    用 `torch_npu.npu._get_deterministic_level()` 读取，level == 3 时 VEC0 的 sparse KV
    搬入从「成对双行 DMA」退化成「逐行搬运」（见 `_vec0_body` 的 `all_pairs_ok` 覆盖）。
    """
    try:
        import torch_npu
        return int(torch_npu.npu._get_deterministic_level()) == 3
    except (AttributeError, ImportError):
        return False

def _validate_inputs(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
):
    """Validate tensor metadata only; never read tensor values on the host.

    Callers supply valid prefix sums, sequence lengths and active indices.
    seqused_q, when supplied, must equal the cu_seqlens_q spans (no padding).
    """
    if str(layout_q).upper() != "TND" or str(layout_kv).upper() != "PA_BBND":
        raise ValueError("mixed_quant_sparse_flash_mla supports only "
                         "layout_q='TND' and layout_kv='PA_BBND'")
    if int(quant_mode) != 1:
        raise ValueError("mixed_quant_sparse_flash_mla supports only "
                         "quant_mode=1")

    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.dtype != torch.bfloat16 or q.dim() != 3 or int(q.shape[-1]) != D:
        raise ValueError("q must be a BF16 TND tensor with shape [T1, N1, 512]")
    rows, n1 = int(q.shape[0]), int(q.shape[1])
    if n1 != N1:
        raise ValueError(f"N1 must be {N1} (the only supported query-head "
                         f"count), got {n1}")
    if rows <= 0:
        raise ValueError("q must contain at least one token")

    if cu_seqlens_q is None:
        raise ValueError(
            "cu_seqlens_q is required for TND varlen and is never synthesized: "
            "pass an int32 [B+1] prefix-sum tensor ending at q.shape[0]")
    if (not isinstance(cu_seqlens_q, torch.Tensor)
            or cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.dim() != 1):
        raise ValueError("cu_seqlens_q must be an int32 tensor of shape (B+1,)")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must describe at least one batch")
    b = cu_seqlens_q.numel() - 1

    if seqused_q is not None:
        if not isinstance(seqused_q, torch.Tensor):
            raise TypeError("seqused_q must be a torch.Tensor when provided")
        if seqused_q.dtype != torch.int32 or seqused_q.dim() != 1:
            raise ValueError("seqused_q must be a 1-D int32 tensor of shape (B,)")
        if seqused_q.numel() != b:
            raise ValueError("seqused_q must have one length per cu_seqlens_q batch")

    for name, seqused in (("seqused_ori_kv", seqused_ori_kv),
                          ("seqused_cmp_kv", seqused_cmp_kv)):
        if seqused is None:
            continue
        if not isinstance(seqused, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor when provided")
        if seqused.dtype != torch.int32 or seqused.dim() != 1:
            raise ValueError(f"{name} must be a 1-D int32 tensor of shape (B,)")
        if int(seqused.numel()) != b:
            raise ValueError(f"{name} must have one entry per batch element (B={b})")

    has_cmp = (cmp_kv is not None or cmp_sparse_indices is not None
               or cmp_block_table is not None)
    if has_cmp:
        if (cmp_kv is None or cmp_sparse_indices is None
                or cmp_block_table is None):
            raise ValueError("ORI_CMP_SPARSE requires cmp_kv, cmp_sparse_indices, "
                             "cmp_block_table together")
    elif cmp_topk_length is not None:
        raise ValueError("cmp inputs are not valid for ORI_SPARSE")

    _validate_pa_side(ori_kv, ori_block_table, name="ori_kv",
                      row_bytes=KV_ROW_BYTES_ORI)
    _validate_indices(ori_sparse_indices, name="ori_sparse_indices", rows=rows)
    _validate_lengths(ori_topk_length, name="ori_topk_length", rows=rows)

    if has_cmp:
        _validate_pa_side(cmp_kv, cmp_block_table, name="cmp_kv",
                          row_bytes=KV_ROW_BYTES_CMP)
        _validate_indices(cmp_sparse_indices, name="cmp_sparse_indices", rows=rows)
        _validate_lengths(cmp_topk_length, name="cmp_topk_length", rows=rows)

    if sinks is None:
        raise ValueError("sinks is required: pass an FP32 [N1] tensor shared by all batches")
    if (not isinstance(sinks, torch.Tensor) or sinks.dtype != torch.float32
            or tuple(sinks.shape) != (n1,)):
        raise ValueError("sinks must be an FP32 tensor with shape [N1], shared by all batches")

    if metadata is None:
        raise ValueError(
            "metadata is required and is never synthesized: generate it with "
            "mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1, has_cmp_kv=...) and pass "
            "the result in (two-stage call)")
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("metadata must be a torch.Tensor")
    if (metadata.dtype != torch.int32 or metadata.dim() != 1
            or metadata.numel() != MQSMLA_METADATA_TOTAL_SIZE):
        raise ValueError(f"metadata must be a 1-D int32 tensor with exactly "
                         f"{MQSMLA_METADATA_TOTAL_SIZE} elements (fixed shape)")


def _validate_pa_side(kv, block_table, *, name, row_bytes):
    if not isinstance(kv, torch.Tensor) or not isinstance(block_table, torch.Tensor):
        raise TypeError(f"{name} and {name}_block_table must be torch.Tensor")
    if (kv.dtype != torch.uint8 or kv.dim() != 4
            or int(kv.shape[2]) != N2 or int(kv.shape[-1]) != row_bytes):
        side = "FP8 (ori)" if name != "cmp_kv" else "FP4 (cmp)"
        raise ValueError(
            f"{name} must be a uint8 [blocknum, blocksize, 1, {row_bytes}] "
            f"tensor ({side} byte view); got shape {tuple(kv.shape)}, "
            f"dtype {kv.dtype}")
    if int(kv.shape[0]) <= 0 or int(kv.shape[1]) <= 0:
        raise ValueError(f"{name} must be non-empty")
    if (block_table.dtype != torch.int32 or block_table.dim() != 2
            or int(block_table.shape[0]) <= 0 or int(block_table.shape[1]) <= 0):
        raise ValueError(f"{name}_block_table must be a non-empty int32 "
                         f"[batch, num_blocks] tensor")
    bs = int(kv.shape[1])
    strides = tuple(kv.stride())
    page_stride = int(kv.stride(0))  # uint8 元素数即字节数；独立于页内 token 数。
    if strides[1:] != (N2 * row_bytes, row_bytes, 1):
        raise ValueError(f"{name} only supports axis-0 non-contiguity; "
                         f"page dimensions must be contiguous, got strides {strides}")
    if page_stride < bs * N2 * row_bytes:
        raise ValueError(f"{name} page stride must be at least {bs * N2 * row_bytes} "
                         "bytes (positive, non-overlapping pages)")
    if bs > 1024:
        raise ValueError("PA block size must be in 1..1024")

def _validate_indices(idx, *, name, rows):
    if not isinstance(idx, torch.Tensor) or idx.dtype != torch.int32:
        raise ValueError(f"{name} must be an int32 torch.Tensor")
    if (idx.dim() != 3 or int(idx.shape[0]) != rows
            or int(idx.shape[1]) != N2):
        raise ValueError(f"{name} must be int32 with shape [T1, 1, K] "
                         f"(one row per query token, {rows} rows)")
    k = int(idx.shape[2])
    if k <= 0:
        raise ValueError(f"{name} must have at least one index column")

def _validate_lengths(x, *, name, rows):
    if x is None:
        return None
    if not isinstance(x, torch.Tensor) or x.dtype != torch.int32:
        raise ValueError(f"{name} must be an int32 torch.Tensor")
    if x.dim() != 2 or int(x.shape[0]) != rows or int(x.shape[1]) != N2:
        raise ValueError(f"{name} must be int32 with shape [T1, 1]")

def construct_workspace(metadata, addr_bytes=0):
    # v0 环 + FD staging 均按核数线性缩放；固定 1024 metadata 不再携带核数，
    # 与 metadata 生产者同源查询（同设备 ⇒ 同值）。kernel 从容量反推网格。
    blocks = _get_cube_core_num(metadata.device)
    if not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError(f"core count must be in 1..{AIC_CORE_MAX_NUM}, got {blocks}")
    ws_bytes = blocks * WS_BYTES_PER_CORE
    # Allocation only: avoid a zeros device op during ACLGraph capture.
    # vec0 initializes every subblock consumed by PV on every replay; FD slots
    # are fully written by their staging core before the barrier publishes them.
    # addr_bytes：pre 臂的行号表区（T1×(K1+K2)×4B），同一次 device 分配，
    # 不增加 host 内存与 H2D（kernel 自写自读，消费侧只读被写过的列）。
    return torch.empty(ws_bytes + addr_bytes, dtype=torch.uint8,
                       device=metadata.device)

@jit
def _run_mqsmla(q_gm: Tensor, ori_kv_gm: Tensor, ori_idx_gm: Tensor,
                ori_bt_gm: Tensor, ori_len_gm: Tensor, out_gm: Tensor,
                sinks_gm: Tensor, cu_seqlens_q: Tensor,
                metadata_gm: Tensor, workspace_gm: Tensor,
                cmp_kv_gm: Tensor = None, cmp_idx_gm: Tensor = None,
                cmp_bt_gm: Tensor = None, cmp_len_gm: Tensor = None,
                lse_gm: Tensor = None, softmax_scale: dtypes.float32 = 1.0,
                batch_consistency: dtypes.int32 = 0,
                addr_ws_gm: Tensor = None):
    """Adapt dynamic tensor views and launch the device kernel from a JIT entry."""
    # trace 期只变 DSL Tensor 视图，不做 torch 变形或数值转换：
    # Q/out [T1,64,512]→[T1*64,512]，故设备 m-tile 行号就是全局 query token 号；
    # KV 保留四维 uint8 原视图；两侧页大小和 page stride 从运行时 DSL Tensor 获取；
    # indices [T1,1,K]→[T1,K]；lengths [T1,1]→[T1]；cu [B+1]→[1,B+1]；
    # sinks [64]→[1,64]（所有 batch/query 共用）；LSE [1,T1,64]→[T1*64]。
    # KV 按字节寻址，避免把页间 padding 当成 token；vec0 搬到 UB 后按 FP8/FP4 解码。
    m_rows = q_gm.shape[0] * q_gm.shape[1]
    q_flat = q_gm.view(m_rows, D)
    out_flat = out_gm.view(m_rows, D)
    ori_pool = ori_kv_gm
    ori_idx_flat = ori_idx_gm.view(
        ori_idx_gm.shape[0] * ori_idx_gm.shape[1], ori_idx_gm.shape[2])
    ori_len_flat = None
    if const_expr(ori_len_gm is not None):
        ori_len_flat = ori_len_gm.view(
            ori_len_gm.shape[0] * ori_len_gm.shape[1])
    cu_q = cu_seqlens_q.view(1, cu_seqlens_q.shape[0])
    sinks_k = sinks_gm.view(1, q_gm.shape[1])
    lse_flat = None
    if const_expr(lse_gm is not None):
        lse_flat = lse_gm.view(lse_gm.shape[1] * lse_gm.shape[2])
    cmp_pool = None
    cmp_idx_flat = None
    cmp_len_flat = None
    if const_expr(cmp_kv_gm is not None):
        cmp_pool = cmp_kv_gm
        cmp_idx_flat = cmp_idx_gm.view(
            cmp_idx_gm.shape[0] * cmp_idx_gm.shape[1], cmp_idx_gm.shape[2])
        if const_expr(cmp_len_gm is not None):
            cmp_len_flat = cmp_len_gm.view(
                cmp_len_gm.shape[0] * cmp_len_gm.shape[1])
    op = MqsmlaKernel(
        tile_cube_m=N1, tile_vec_m=N1 // 2, tile_n=TILE_N,
        n_heads=q_gm.shape[1], has_cmp=(cmp_kv_gm is not None),
        ws_depth=WS_DEPTH)
    # 固定 1024 metadata 不携带核数；发射网格由 workspace 容量反推（每核
    # WS_BYTES_PER_CORE = v0 三深环 + 2 个 FD staging slot），与 metadata
    # 生产者的核数查询同源，保持动态网格 ⇒ 单一二进制适配任意核数。
    blocks = workspace_gm.shape[0] // WS_BYTES_PER_CORE
    if const_expr(_addr_vec_mode() == "pre"):
        op[blocks](out_flat, q_flat, sinks_k,
                   ori_pool, ori_bt_gm, workspace_gm,
                   cmp_pool, cmp_bt_gm, cmp_idx_flat,
                   ori_idx_flat, cu_q, lse_flat, ori_len_flat,
                   cmp_len_flat, metadata_gm,
                   softmax_scale, batch_consistency, addr_ws_gm)
    else:
        op[blocks](out_flat, q_flat, sinks_k,
                   ori_pool, ori_bt_gm, workspace_gm,
                   cmp_pool, cmp_bt_gm, cmp_idx_flat,
                   ori_idx_flat, cu_q, lse_flat, ori_len_flat,
                   cmp_len_flat, metadata_gm,
                   softmax_scale, batch_consistency)

_COMPILED_KERNELS = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def clear_caches():
    """Close cached executables, as in flash_kda.clear_caches()."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNELS.values():
            compiled.close()
        _COMPILED_KERNELS.clear()


def _get_compiled_kernel(
    q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
    out, sinks, cu_seqlens_q, metadata, workspace,
    cmp_kv=None, cmp_sparse_indices=None, cmp_block_table=None,
    cmp_topk_length=None, lse=None, addr_ws=None,
):
    """Cache the dynamic executable by optional-input structure.

    The host entry validates the fixed tensor contract before calling here.
    Query/KV sizes, K1/K2, page strides and scale remain runtime values.
    地址向量化（默认启用）额外要求 bs/btp/K 为字面量并回写 _ADDR_VEC_OVERRIDE，
    使 kernel trace 期的 _addr_vec_mode() 与本规格一致；不满足资格时该次编译
    回退标量基线路径（符号规格，与 fp4 版逐字节相同）。
    """
    has_cmp = cmp_kv is not None
    env_mode = _addr_vec_mode()
    ok, bs_w, btp_w, k1, bs_c, btp_c, k2 = _addr_vec_eligible(
        ori_kv, ori_block_table, ori_sparse_indices,
        cmp_kv, cmp_block_table, cmp_sparse_indices)
    _vec_modes = ("pre",)
    eff_mode = env_mode if (env_mode in _vec_modes and ok) else (
        env_mode if env_mode not in _vec_modes else "off")
    lit = (bs_w, btp_w, k1, bs_c, btp_c, k2) if eff_mode in _vec_modes else None
    # Only optional Tensor presence selects a binary; fixed dtypes and head
    # dimensions follow the host contract. Look up before constructing specs.
    key = (has_cmp, ori_topk_length is not None, cmp_topk_length is not None,
           lse is not None, eff_mode, lit)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        global _ADDR_VEC_OVERRIDE
        _ADDR_VEC_OVERRIDE = eff_mode
        try:
            return _compile_mqsmla_locked(
                q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
                out, sinks, cu_seqlens_q, metadata, workspace,
                cmp_kv, cmp_sparse_indices, cmp_block_table,
                cmp_topk_length, lse, addr_ws, has_cmp, eff_mode, lit, key)
        finally:
            _ADDR_VEC_OVERRIDE = None


def _compile_mqsmla_locked(
        q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
        out, sinks, cu_seqlens_q, metadata, workspace,
        cmp_kv, cmp_sparse_indices, cmp_block_table,
        cmp_topk_length, lse, addr_ws, has_cmp, eff_mode, lit, key):
    if True:

        rows = cannbotdsl.Dim("T1", min=1)
        batch = cannbotdsl.Dim("B", min=1)
        # 固定 1024：静态 shape，图捕获友好；分核计划在其中动态编码。
        metadata_size = MQSMLA_METADATA_TOTAL_SIZE
        spec = cannbotdsl.TensorSpec
        # 向量臂用真实字面量特化（vshr 移位数/表列基址/驻留槽宽都要编译期值）；
        # off/消融保持符号维，规格与 fp4 版完全一致。
        if lit is not None:
            bs_w, btp_w, k1, bs_c, btp_c, k2 = lit
            stride_w = bs_w * KV_ROW_BYTES_ORI
            stride_c = bs_c * KV_ROW_BYTES_CMP if has_cmp else None
        else:
            bs_w = cannbotdsl.Dim("ORI_BS", min=1, max=1024)
            btp_w = cannbotdsl.Dim("ORI_BT", min=1)
            k1 = cannbotdsl.Dim("K1", min=1)
            bs_c = cannbotdsl.Dim("CMP_BS", min=1, max=1024)
            btp_c = cannbotdsl.Dim("CMP_BT", min=1)
            k2 = cannbotdsl.Dim("K2", min=1)
            stride_w = cannbotdsl.Dim("ORI_STRIDE", min=1)
            stride_c = cannbotdsl.Dim("CMP_STRIDE", min=1)
        q_spec = spec((rows, N1, D), dtypes.bfloat16)
        ori_kv_spec = spec(
            (cannbotdsl.Dim("ORI_PAGES", min=1),
             bs_w, N2, KV_ROW_BYTES_ORI),
            dtypes.uint8,
            stride=(stride_w,
                    KV_ROW_BYTES_ORI, KV_ROW_BYTES_ORI, 1))
        ori_indices_spec = spec((rows, N2, k1), dtypes.int32)
        ori_bt_spec = spec((batch, btp_w), dtypes.int32)
        ori_length_spec = None
        if ori_topk_length is not None:
            ori_length_spec = spec((rows, N2), dtypes.int32)
        out_spec = spec((rows, N1, D), dtypes.bfloat16)
        sinks_spec = spec((N1,), dtypes.float32)
        cu_spec = spec((batch + 1,), dtypes.int32)
        metadata_spec = spec((metadata_size,), dtypes.int32)
        # workspace = blocks × WS_BYTES_PER_CORE（v0 环 + FD staging），
        # 整除约束让 kernel 从容量反推动态发射网格。
        workspace_spec = spec(
            (cannbotdsl.Dim("WORKSPACE_BYTES", min=WS_BYTES_PER_CORE,
                            multiple_of=WS_BYTES_PER_CORE),),
            dtypes.uint8)
        cmp_kv_spec = None
        cmp_indices_spec = None
        cmp_bt_spec = None
        cmp_length_spec = None
        if has_cmp:
            cmp_kv_spec = spec(
                (cannbotdsl.Dim("CMP_PAGES", min=1),
                 bs_c, N2, KV_ROW_BYTES_CMP),
                dtypes.uint8,
                stride=(stride_c,
                        KV_ROW_BYTES_CMP, KV_ROW_BYTES_CMP, 1))
            cmp_indices_spec = spec((rows, N2, k2), dtypes.int32)
            cmp_bt_spec = spec((batch, btp_c), dtypes.int32)
        if cmp_topk_length is not None:
            cmp_length_spec = spec((rows, N2), dtypes.int32)
        lse_spec = None
        if lse is not None:
            lse_spec = spec((N2, rows, N1), dtypes.float32)
        # addr_ws_gm 尾置：None 规格缺席时不再扰动 scale 的参数对位；
        # pre 给真规格（ADDR_WS_BYTES 符号维，大小由 host 分配保证）。
        addr_ws_spec = (spec((cannbotdsl.Dim("ADDR_WS_BYTES", min=1),), dtypes.uint8)
                        if eff_mode == "pre"
                        else None)
        compiled = _run_mqsmla.compile(
            q_spec, ori_kv_spec, ori_indices_spec, ori_bt_spec, ori_length_spec,
            out_spec, sinks_spec, cu_spec, metadata_spec, workspace_spec,
            cmp_kv_spec, cmp_indices_spec, cmp_bt_spec, cmp_length_spec,
            lse_spec, dtypes.float32, dtypes.int32, addr_ws_spec,
        )
        # Publish only after compilation succeeds; failures can be retried.
        _COMPILED_KERNELS[key] = compiled
    return compiled


def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
):
    _validate_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    for name, tensor, shape, dtype in (
        ("out", out, tuple(q.shape), torch.bfloat16),
        ("lse", lse, (N2, q.shape[0], q.shape[1]), torch.float32),
    ):
        if name == "lse" and not return_softmax_lse:
            shape = (0,)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a caller-allocated tensor")
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {dtype} {shape}")
    # 地址向量臂的行号表区：仅 pre 需要（tile 的行号住 UB）；资格与编译侧一致。
    _ok, _bs_w, _btp_w, _k1, _bs_c, _btp_c, _k2 = _addr_vec_eligible(
        ori_kv, ori_block_table, ori_sparse_indices,
        cmp_kv, cmp_block_table, cmp_sparse_indices)
    _env = _addr_vec_mode()
    _vmodes = ("pre",)
    _eff = _env if (_env in _vmodes and _ok) else (
        _env if _env not in _vmodes else "off")
    addr_ws = None
    addr_bytes = 0
    blocks_ = _get_cube_core_num(metadata.device)
    ws_bytes_ = blocks_ * WS_BYTES_PER_CORE
    if _eff == "pre":
        # u32 行号表：[T1, K1+K2]，与 v0+FD workspace 同一次 device 分配；
        # workspace_gm 只传 v0+FD 前缀（满足 WS_BYTES_PER_CORE 整除），addr_ws 传尾区。
        # 列数按 _addr_tab_w 补齐（消费侧按 _TILE_ROWS 整子块读列），与 kernel
        # 侧的 tab_cols 必须同源，否则行距对不上、整张表错位。
        addr_bytes = q.shape[0] * (_addr_tab_w(_k1)
                                   + (_addr_tab_w(_k2) if _k2 is not None
                                      else 0)) * 4
        workspace_full = construct_workspace(metadata, addr_bytes)
        workspace = workspace_full[:ws_bytes_]
        addr_ws = workspace_full[ws_bytes_:]
    else:
        workspace = construct_workspace(metadata)
    tensors = (
        q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
        out, sinks, cu_seqlens_q, metadata, workspace,
        cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    compiled = _get_compiled_kernel(
        q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
        out, sinks, cu_seqlens_q, metadata, workspace,
        cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length,
        lse if return_softmax_lse else None, addr_ws,
    )
    # AOT fixes absent Tensor parameters at compile time and removes them from
    # the runtime signature. Remaining Tensor arguments keep their order.
    scale = (float(softmax_scale) if softmax_scale is not None
             else 1.0 / math.sqrt(q.shape[-1]))
    batch_consistency = 1 if _batch_consistency_enabled() else 0
    compiled(*(tensor for tensor in tensors if tensor is not None), scale,
             batch_consistency,
             *( [addr_ws] if addr_ws is not None else [] ))
# Native export profiles describe optional-input structures and example tensors.
# T1, K1/K2 and physical page sizes/counts stay dynamic in the public TensorSpecs.
from cannbotdsl.package.native import register


@register("mqsmla")
def export_mqsmla():
    """Export without allocating NPU inputs or launching an attention kernel.

    CANNBOTDSL_DS41_PROFILES may name a JSON file with a ``mqsmla`` list.
    Each entry selects optional-input structure and example tensor shapes.
    K1/K2, like other sizes, become symbolic TensorSpecs and do not select a binary.
    """
    import json

    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    # Export every active-side length-presence combination in both public modes.
    profiles = [
        {"has_ori_length": ori, "has_cmp_length": cmp}
        for ori in (True, False) for cmp in (True, False)
    ] + [{"k2": 0, "has_ori_length": ori} for ori in (True, False)]
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            profiles = json.load(stream).get("mqsmla", profiles)
    defaults = dict(b=1, k1=128, k2=512, bs=128, bs2=128,
                    s2_cap=8192, s2c_cap=4096, blocks=28)
    clear_caches()
    try:
        for profile in profiles:
            config = dict(defaults, **profile)
            if config.pop("sinks_dim", 1) != 1:
                raise ValueError("MQSMLA sinks must have shape [N1]; sinks_dim must be 1")
            has_lse = config.pop("return_softmax_lse", False)
            has_ori_length = config.pop("has_ori_length", True)
            has_cmp_length = config.pop("has_cmp_length", True)
            config.setdefault("ori_page_stride", config["bs"] * 544)
            config.setdefault("cmp_page_stride", config["bs2"] * 320 if config["k2"] else 0)
            def tensor(shape, dtype, stride=None):
                if stride is not None:
                    return torch.empty_strided(shape, stride, dtype=dtype, device="meta")
                return torch.empty(shape, dtype=dtype, device="meta")
            _get_compiled_kernel(
                tensor((1, N1, D), torch.bfloat16),
                tensor((1, config["bs"], 1, 544), torch.uint8,
                       (config["ori_page_stride"], 544, 544, 1)),
                tensor((1, 1, config["k1"]), torch.int32),
                tensor((config["b"], config["s2_cap"] // config["bs"]), torch.int32),
                tensor((1, 1), torch.int32) if has_ori_length else None,
                tensor((1, N1, D), torch.bfloat16),
                tensor((N1,), torch.float32),
                tensor((config["b"] + 1,), torch.int32),
                tensor((MQSMLA_METADATA_TOTAL_SIZE,), torch.int32),
                tensor((WS_BYTES_PER_CORE * config["blocks"],), torch.uint8),
                tensor((1, config["bs2"], 1, 320), torch.uint8,
                       (config["cmp_page_stride"], 320, 320, 1)) if config["k2"] else None,
                tensor((1, 1, config["k2"]), torch.int32) if config["k2"] else None,
                tensor((config["b"], config["s2c_cap"] // config["bs2"]), torch.int32) if config["k2"] else None,
                tensor((1, 1), torch.int32) if config["k2"] and has_cmp_length else None,
                tensor((1, 1, N1), torch.float32) if has_lse else None,
            )
    finally:
        clear_caches()
