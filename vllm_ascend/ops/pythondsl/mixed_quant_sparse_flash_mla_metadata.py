# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on how to use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

"""mqsmla metadata —— AscendC AICPU 分核算子 mixed_quant_sparse_flash_mla_metadata 的
Python 逐行移植（host 侧计算，flash_attn 方法）。

镜像对象（**算法基线**）：
`ops-transformer/attention/mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/
mixed_quant_sparse_flash_mla_metadata_aicpu.{h,cpp}`
（与 `sparse_flash_mla_metadata_aicpu.{h,cpp}` 共用同一分核 + FD 算法，移植骨架
来自 `ref/smla_metadata.py`。）

方法（= `samples/flash_attn/flash_attn.py` 的 `_compute_load_balance`）：canndsl 不保留
AICPU kernel 形态，metadata 在 **host 侧 Python** 算好，产出与 AscendC 完全同布局的
int32[1024] 张量（FA 9 字段 × AIC_CORE_MAX_NUM + FD 8 字段 × AIV_CORE_MAX_NUM），
由主算子 `mixed_quant_sparse_flash_mla` 当普通 GM 张量消费。

**移植纪律**：本文件的算法函数与 AICPU C++ **一一对应、原封不动** —— 函数名、控制流、
算术（含 uint32 回绕、int64 截断）逐条镜像，每个方法 docstring 标注 C++ 行号。

对接细节（消费侧在 `mixed_quant_sparse_flash_mla.py`）：
- AICPU 游标 `(curBN2Idx, curS1GIdx)` 在 (batch, 行) 字典序上连续 ⇒ 映到全局扁平行号后
  每核是一段连续区间 `[m0, m1)`：`m0 = row(bn2_start) + gs1_start`、
  `m1 = row(bn2_end) + gs1_end`（幻影行 s1GIdx ≥ baseNum 也落在该式内 —— 它就是下一
  batch 的前几行，覆盖仍是精确划分）。消费侧按 `cu_seqlens_q[bn2] + m` 重建全局行号。
- N2=1 ⇒ `group_size_ = num_heads_q/num_heads_kv = 64 = m_base_size_`，每个 s1G 块恰为
  一个 query 行，`FA_M_START/END`（= gS1 游标）即 batch 内行号，`FA_BN2_START/END` 即
  batch 号；`fd_m_size` 恒 64 head。
- **FD 开启逻辑**：AscendC 里 `supportFd_` 成员默认 true（aicpu.h:368），`ParamsInit` 的
  `if (isBatchConsistency_) supportFd_ = true;`（aicpu.cpp:426-428）是恒真 no-op，即参考侧
  FD 始终开启。本移植改用顶部 `SUPPORT_FD` 开关控制（当前 False，见顶部「FD 开关」注释）。
"""

import math
import tempfile
import threading

from cannbotdsl.aicpu import GmIn, GmOut, I32, aicpu_kernel, current_raw_stream

# ---- 常量（metadata.h:24-50 / aicpu.h:29-58，逐条照抄）----
AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
MQSMLA_METADATA_TOTAL_SIZE = 1024

FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE   # 324
# faMetadata(324) + fdMetadata(576) = 900 字之后的首个保留字：存放 fdUsedVecNum
# （= FD 归约 AIV 数）。消费侧据此统一决定是否执行 grid barrier + FD 归约阶段。
FD_USED_VEC_NUM_WORD = AIC_CORE_MAX_NUM * FA_METADATA_SIZE + AIV_CORE_MAX_NUM * FD_METADATA_SIZE  # 900

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7
FA_S2_MAX_NUM = 8

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6

FA_TOLERANCE_RATIO = 2
COST_WEIGHT_M = 6
COST_WEIGHT_S2 = 10
BATCH_CONSISTENCY_MAX_REDUCTION_PARTS = 32
ORI_KV = False
CMP_KV = True
NO_MASK = 0
HAS_MASK = 1

INT64_MAX = 2 ** 63 - 1
UINT32_MAX = 2 ** 32 - 1


# ---- FD（flash decode）开关（2026-09-19 重新关闭，实验结论见 MQSMLA_FD_OPTIMIZATION_PLAN.md）----
# 本会话实测（kernel 侧已做 post-barrier channel_rewind + 整块 64KiB 归约落地，把 FD-on
# 从 76.8us 压到 59.0us，但仍 > 基线 43.9us）：decode-6-long-s2 的 tile/行只有 5，负载不均衡
# 仅 33%（15 vs 11.25 tile），FD 切分 26 行带来的 O 暂存+归约流量（~6.6MB）串行尾段 ~15us，
# 远超负载均衡省下的 ~2.5us mac。故对 5-tile/行 的 decode 形状 FD 不划算，保持关闭。
# 对「每行 tile 数多、FA 不均衡大」的长行形状，FD 仍可能净赚（此时暂存流量相对节省可忽略），
# 重开只需把本开关置回 True；归约落地优化与 support_fd 门限（_apply_support_fd_gate）已保留。
# decode-1/decode-6 完全均衡，FD 门限/自然不分切 ⇒ fd_used_vec_num=0，与无 FD 基线逐指令等价。
SUPPORT_FD = False


class _BlockType:
    """aicpu.h:38-44。"""
    ORI_NORMAL_BLOCK = 0
    ORI_TAIL_BLOCK = 1
    CMP_NORMAL_BLOCK = 2
    CMP_TAIL_BLOCK = 3
    BLOCK_MAX_TYPE = 4


class _SparseMode:
    """aicpu.h:46-53。"""
    DEFAULT_MASK = 0
    ALL_MASK = 1
    LEFT_UP_CAUSAL = 2
    RIGHT_DOWN_CAUSAL = 3
    BAND = 4


def _u32(value):
    """C++ uint32_t 转换（回绕）。"""
    return int(value) & UINT32_MAX


def _clip(value, lo, hi):
    """aicpu.h:67-76 Clip<T>。"""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _is_within_tolerance(limit, tolerance, value):
    """aicpu.h:78-82 IsWithinTolerance<T>：`limit + tolerance >= value`。"""
    return limit + tolerance >= value


def _i64_div(a, b):
    """C++ int64/int64 除法：向零截断（Python `//` 向下取整，负数时不同）。"""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


class _FlashDecodeResult:
    """aicpu.h:85-108 FlashDecodeResult。向量长度按 (aicNum, aivNum) 定长。"""

    def __init__(self, aic_num, aiv_num):
        self.fd_used_vec_num = 0
        self.fd_bn2_idx = [0] * aic_num
        self.fd_m_idx = [0] * aic_num
        self.fd_workspace_idx = [0] * aic_num
        self.fd_s2_split_num = [0] * aic_num
        self.fd_m_size = [0] * aic_num
        self.fd_idx = [0] * aiv_num
        self.fd_m_start = [0] * aiv_num
        self.fd_m_num = [0] * aiv_num


class _SplitResult:
    """aicpu.h:111-129 SplitResult（构造即带 fdRes(aicNum, aivNum)，aicpu.cpp:31）。"""

    def __init__(self, aic_num, aiv_num):
        self.used_core_num = 0
        self.bn2_end = [0] * aic_num
        self.gs1_end = [0] * aic_num
        self.s2_end = [0] * aic_num
        self.first_fd_data_workspace_idx = [0] * aic_num
        self.max_cost = 0
        self.num_of_fd_head = 0
        self.max_s2_split_num = 0
        self.max_s2_loop_num = 0
        self.fd_res = _FlashDecodeResult(aic_num, aiv_num)


class _SplitInfo:
    """aicpu.h:132-149 SplitInfo。"""

    def __init__(self, batch_size):
        self.s1_g_base_num = [0] * batch_size
        self.ori_s2_base_num = [0] * batch_size
        self.cmp_s2_base_num = [0] * batch_size
        self.s1_g_tail_size = [0] * batch_size
        self.ori_s2_tail_size = [0] * batch_size
        self.cmp_s2_tail_size = [0] * batch_size
        self.is_kv_seq_all_zero = True


class _CostInfo:
    """aicpu.h:152-167 CostInfo。"""

    def __init__(self, batch_size):
        self.bn2_cost_of_each_batch = [0] * batch_size
        self.bn2_block_of_each_batch = [0] * batch_size
        self.bn2_s2_loop_of_each_batch = [0] * batch_size
        self.bn2_last_block_cost_of_each_batch = [0] * batch_size
        self.total_block_num = 0
        self.total_cost = 0
        self.max_s1g_cost = 0


class _SplitContext:
    """aicpu.h:170-178 SplitContext。"""

    def __init__(self, batch_size):
        self.split_info = _SplitInfo(batch_size)
        self.cost_info = _CostInfo(batch_size)


class _BatchCache:
    """aicpu.h:181-191 BatchCache。"""

    def __init__(self):
        self.b_idx = 0
        self.s1_size = 0
        self.ori_s2_size = 0
        self.cmp_revert_s2_size = 0
        self.ori_pre_token_left_up = 0
        self.ori_next_token_left_up = 0
        self.cmp_pre_token_left_up = 0
        self.cmp_next_token_left_up = 0
        self.type_cost = [[0] * _BlockType.BLOCK_MAX_TYPE for _ in range(_BlockType.BLOCK_MAX_TYPE)]


class _S1GCache:
    """aicpu.h:194-220 S1GCache。oriS2TailSize/cmpS2TailSize 是 int64（C++ 原类型如此）。"""

    def __init__(self):
        self.b_idx = 0
        self.s1_g_idx = 0
        self.s2_start = 0
        self.s2_end = 0
        self.ori_s2_start = 0
        self.ori_s2_end = 0
        self.cmp_s2_start = 0
        self.cmp_s2_end = 0
        self.s1_g_cost = 0
        self.s1_g_last_block_cost = 0
        self.s1_g_block = 0
        self.s2_loop = 0
        self.ori_s1_g_block = 0
        self.ori_s1_g_cost = 0
        self.ori_s1_g_last_block_cost = 0
        self.ori_s1_g_normal_block_cost = 0
        self.cmp_s1_g_block = 0
        self.cmp_s1_g_cost = 0
        self.cmp_s1_g_last_block_cost = 0
        self.cmp_s1_g_normal_block_cost = 0
        self.ori_s2_tail_size = 0
        self.cmp_s2_tail_size = 0
        self.act_ori_s2_size = 0
        self.act_cmp_s2_size = 0
        self.reduction_block_size = 0


class _CoreCache:
    """aicpu.h:223-228 CoreCache。"""

    def __init__(self):
        self.cost_limit = 0
        self.cost = 0
        self.block = 0
        self.s2_loop = 0


class _AssignContext:
    """aicpu.h:231-248 AssignContext。C++ `= {}` 零初始化。"""

    def __init__(self):
        self.cur_b_idx = 0
        self.cur_bn2_idx = 0
        self.cur_s1g_idx = 0
        self.cur_s2_idx = 0
        self.cur_core_idx = 0
        self.unassigned_cost = 0
        self.cur_kv_split_part = 1
        self.pre_fd_data_num = 0
        self.bn2_cost = 0
        self.bn2_block = 0
        self.bn2_s2_loop = 0
        self.is_finished = False
        self.batch_cache = _BatchCache()
        self.s1g_cache = _S1GCache()
        self.core_cache = _CoreCache()


class SparseFlashMlaMetadataCpuKernel:
    """`SparseFlashMlaMetadataCpuKernel`（aicpu.h:250-392）的逐行移植。

    输入张量一律是 **host int list 或 None**（= C++ 的 `Tensor*`/`GetData()` 视图）；
    NPU/CPU torch 张量由模块级 `mixed_quant_sparse_flash_mla_metadata()` 负责落地转换。
    """

    def __init__(self, *,
                 # required attrs（Prepare，aicpu.cpp:51-53）
                 num_heads_q, num_heads_kv=1, head_dim=512,
                 # optional attrs（aicpu.cpp:57-76 GetAttrValueOpt）
                 soc_version="Ascend950", aic_core_num=AIC_CORE_MAX_NUM,
                 aiv_core_num=AIV_CORE_MAX_NUM, batch_size=0, max_seqlen_q=0,
                 max_seqlen_ori_kv=0, max_seqlen_cmp_kv=0, ori_topk=0, cmp_topk=0,
                 cmp_ratio=1, ori_mask_mode=_SparseMode.BAND,
                 cmp_mask_mode=_SparseMode.RIGHT_DOWN_CAUSAL, ori_win_left=127,
                 ori_win_right=0, layout_q="BSND", layout_kv="PA_BBND",
                 has_ori_kv=True, has_cmp_kv=True, is_batch_consistency=False):
        # aicpu.h:336-374 成员默认值，逐条照抄。
        self.num_heads_q_ = int(num_heads_q)
        self.num_heads_kv_ = int(num_heads_kv)
        self.head_dim_ = int(head_dim)
        self.batch_size_ = int(batch_size)
        self.max_seqlen_q_ = int(max_seqlen_q)
        self.max_seqlen_ori_kv_ = int(max_seqlen_ori_kv)
        self.max_seqlen_cmp_kv_ = int(max_seqlen_cmp_kv)
        self.ori_top_k_ = int(ori_topk)
        self.cmp_top_k_ = int(cmp_topk)
        self.cmp_ratio_ = int(cmp_ratio)
        self.ori_mask_mode_ = int(ori_mask_mode)
        self.cmp_mask_mode_ = int(cmp_mask_mode)
        self.ori_win_left_ = int(ori_win_left)
        self.ori_win_right_ = int(ori_win_right)
        self.layout_q_ = str(layout_q)
        self.layout_kv_ = str(layout_kv)
        self.has_ori_kv_ = bool(has_ori_kv)
        self.has_cmp_kv_ = bool(has_cmp_kv)
        self.aic_core_num_ = int(aic_core_num)
        self.aiv_core_num_ = int(aiv_core_num)
        self.is_batch_consistency_ = bool(is_batch_consistency)
        self.soc_version_ = str(soc_version)
        # aicpu.h:360-375（派生态，ParamsInit 填）
        self.ori_pre_token_ = 0
        self.ori_next_token_ = 0
        self.cmp_pre_token_ = 0
        self.cmp_next_token_ = 0
        self.group_size_ = 0
        self.m_base_size_ = 0
        self.s2_base_size_ = 128
        self.is_s1g_ = True
        # aicpu.h:368 成员默认 true；此处改为受 SUPPORT_FD 开关控制（默认 False，见顶部注释）。
        self.support_fd_ = SUPPORT_FD
        self.ori_attention_mode_ = HAS_MASK
        self.cmp_attention_mode_ = HAS_MASK
        self._type_cost_ = [[0] * _BlockType.BLOCK_MAX_TYPE
                            for _ in range(_BlockType.BLOCK_MAX_TYPE)]
        self.is_split_g_ = False
        self.is_sparse_ori_kv_ = False
        self.is_sparse_cmp_kv_ = False
        self.remained_block_num_ = 0

    # ------------------------------------------------------------------
    # Compute / Prepare / 校验（aicpu.cpp:25-417）
    # ------------------------------------------------------------------

    def compute(self, *, cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None, seqused_q=None, seqused_ori_kv=None,
                seqused_cmp_kv=None, cmp_residual_kv=None,
                ori_topk_length=None, cmp_topk_length=None):
        """aicpu.cpp:25-34 Compute：Prepare → BalanceSchedule → GenMetadata。
        返回长度 1024 的 int list（= metadata 输出张量的平坦视图）。"""
        self.cu_seqlens_q_ = _as_i32_list(cu_seqlens_q)
        self.cu_seqlens_ori_kv_ = _as_i32_list(cu_seqlens_ori_kv)
        self.cu_seqlens_cmp_kv_ = _as_i32_list(cu_seqlens_cmp_kv)
        self.seqused_q_ = _as_i32_list(seqused_q)
        self.seqused_ori_kv_ = _as_i32_list(seqused_ori_kv)
        self.seqused_cmp_kv_ = _as_i32_list(seqused_cmp_kv)
        self.cmp_residual_kv_ = _as_i32_list(cmp_residual_kv)
        self.ori_topk_length_ = _as_i32_list(ori_topk_length)
        self.cmp_topk_length_ = _as_i32_list(cmp_topk_length)
        self._params_check()
        self._params_init()
        split_res = _SplitResult(self.aic_core_num_, self.aiv_core_num_)
        if not self._balance_schedule(split_res):
            raise ValueError("BalanceSchedule failed")
        if not self._gen_metadata(split_res):
            raise ValueError("GenMetadata failed")
        return self.metadata_

    def _shape_len(self, tensor, ndim_expect=None):
        """C++ `GetTensorShape()->GetDimSize(0/1/2)` 的等价物：list 视图配显式形状。"""
        if tensor is None:
            return None
        return tensor  # (data, shape) 元组在 _as_i32_list 处理；这里只透传

    def _topk_length_size(self, tensor):
        """C++ `GetDimSize(0)[*GetDimSize(1)[*GetDimSize(2)]]`（不足的维按 1 补）。"""
        shape = tensor[1]
        dims = tuple(shape) + (1,) * (3 - len(shape))
        return dims[0] * dims[1] * dims[2]

    def get_query_batch_size(self):
        """aicpu.cpp:348-367。⚠️ C++ 读的是 `GetDimSize(0)`（第一维长度，不是元素总数）——
        cu_seqlens_q / seqused_q 在参考侧是 **1-D [B+1]/[B]**，两者恰好相等；为忠实起见
        端口也按 dim0 取。"""
        if self.seqused_q_ is not None:
            return _u32(self.seqused_q_[1][0])
        if self.layout_q_ == "TND":
            if self.cu_seqlens_q_ is not None:
                return _u32(self.cu_seqlens_q_[1][0] - 1)
        return _u32(self.batch_size_)

    def get_sum_of_query_seq(self):
        """aicpu.cpp:320-346。"""
        batch_size = self.get_query_batch_size()
        if self.seqused_q_ is not None:
            return sum(self.seqused_q_[0][:batch_size])
        if self.layout_q_ == "TND":
            if self.cu_seqlens_q_ is not None:
                return self.cu_seqlens_q_[0][batch_size]
        return self.batch_size_ * self.max_seqlen_q_

    def _params_check(self):
        """aicpu.cpp:80-318 ParamsCheck（校验失败 ⇒ ValueError，= KERNEL_STATUS_PARAM_INVALID）。"""
        batch_size = self.get_query_batch_size()
        if self.layout_q_ == "TND":
            if self.cu_seqlens_q_ is not None:
                p = self.cu_seqlens_q_[0]
                if p[0] != 0:
                    raise ValueError(f"The first element of cu_seqlens_q should be 0, but got {p[0]}")
                for i in range(1, batch_size + 1):
                    if p[i - 1] > p[i]:
                        raise ValueError(
                            f"The elements in cu_seqlens_q must be in ascending order, "
                            f"but got cu_seqlens_q[{i - 1}] = {p[i - 1]}, cu_seqlens_q[{i}] = {p[i]}")
        if self.seqused_q_ is not None:
            sq = self.seqused_q_[0]
            cu_q = self.cu_seqlens_q_[0] if (self.layout_q_ == "TND"
                                             and self.cu_seqlens_q_ is not None) else None
            for i in range(batch_size):
                if sq[i] < 0:
                    raise ValueError(f"The elements in seqused_q should be >= 0, but got seqused_q[{i}] = {sq[i]}")
                if self.layout_q_ == "BSND" and sq[i] > self.max_seqlen_q_:
                    raise ValueError(
                        f"The elements in seqused_q should not be greater than max_seqlen_q "
                        f"{self.max_seqlen_q_}, but got seqused_q[{i}] = {sq[i]}")
                if cu_q is not None:
                    seq_len = cu_q[i + 1] - cu_q[i]
                    if sq[i] > seq_len:
                        raise ValueError(
                            f"The elements in seqused_q should not be greater than the sequence length "
                            f"from cu_seqlens_q {seq_len}, but got seqused_q[{i}] = {sq[i]}")
        if self.has_ori_kv_:
            if self.layout_kv_ == "TND":
                if self.cu_seqlens_ori_kv_ is not None:
                    p = self.cu_seqlens_ori_kv_[0]
                    if p[0] != 0:
                        raise ValueError(f"The first element of cu_seqlens_ori_kv should be 0, but got {p[0]}")
                    for i in range(1, batch_size + 1):
                        if p[i - 1] > p[i]:
                            raise ValueError(
                                f"The elements in cu_seqlens_ori_kv must be in ascending order, "
                                f"but got cu_seqlens_ori_kv[{i - 1}] = {p[i - 1]}, "
                                f"cu_seqlens_ori_kv[{i}] = {p[i]}")
            if self.seqused_ori_kv_ is not None:
                sv = self.seqused_ori_kv_[0]
                cu_ori = self.cu_seqlens_ori_kv_[0] if (self.layout_kv_ == "TND"
                                                        and self.cu_seqlens_ori_kv_ is not None) else None
                for i in range(batch_size):
                    if sv[i] < 0:
                        raise ValueError(
                            f"The elements in seqused_ori_kv should be >= 0, but got seqused_ori_kv[{i}] = {sv[i]}")
                    if self.layout_kv_ == "BSND" and sv[i] > self.max_seqlen_ori_kv_:
                        raise ValueError(
                            f"The elements in seqused_ori_kv should not be greater than "
                            f"max_seqlen_ori_kv {self.max_seqlen_ori_kv_}, but got seqused_ori_kv[{i}] = {sv[i]}")
                    if cu_ori is not None:
                        seq_len = cu_ori[i + 1] - cu_ori[i]
                        if sv[i] > seq_len:
                            raise ValueError(
                                f"The elements in seqused_ori_kv should not be greater than the sequence "
                                f"length from cu_seqlens_ori_kv {seq_len}, but got seqused_ori_kv[{i}] = {sv[i]}")
            if (self.ori_top_k_ != 0
                    and self.ori_mask_mode_ == _SparseMode.DEFAULT_MASK
                    and self.ori_topk_length_ is not None):
                sum_of_query_seq = self.get_sum_of_query_seq()
                data, _shape = self.ori_topk_length_
                ori_topk_length_size = self._topk_length_size(self.ori_topk_length_)
                if ori_topk_length_size < sum_of_query_seq:
                    raise ValueError(
                        f"The size of ori_topk_length {ori_topk_length_size} should not be smaller than "
                        f"the sum of query sequence {sum_of_query_seq}!")
                for i in range(ori_topk_length_size):
                    if data[i] < 0:
                        raise ValueError(
                            f"The elements in ori_topk_length should be >= 0, "
                            f"but got ori_topk_length[{i}] = {data[i]}")
        if self.has_cmp_kv_:
            if self.layout_kv_ == "TND":
                if self.cu_seqlens_cmp_kv_ is not None:
                    p = self.cu_seqlens_cmp_kv_[0]
                    if p[0] != 0:
                        raise ValueError(f"The first element of cu_seqlens_cmp_kv should be 0, but got {p[0]}")
                    for i in range(1, batch_size + 1):
                        if p[i - 1] > p[i]:
                            raise ValueError(
                                f"The elements in cu_seqlens_cmp_kv must be in ascending order, "
                                f"but got cu_seqlens_cmp_kv[{i - 1}] = {p[i - 1]}, "
                                f"cu_seqlens_cmp_kv[{i}] = {p[i]}")
            if self.seqused_cmp_kv_ is not None:
                sv = self.seqused_cmp_kv_[0]
                cu_cmp = self.cu_seqlens_cmp_kv_[0] if (self.layout_kv_ == "TND"
                                                        and self.cu_seqlens_cmp_kv_ is not None) else None
                for i in range(batch_size):
                    if sv[i] < 0:
                        raise ValueError(
                            f"The elements in seqused_cmp_kv should be >= 0, but got seqused_cmp_kv[{i}] = {sv[i]}")
                    if self.layout_kv_ == "BSND" and sv[i] > self.max_seqlen_cmp_kv_:
                        raise ValueError(
                            f"The elements in seqused_cmp_kv should not be greater than "
                            f"max_seqlen_cmp_kv {self.max_seqlen_cmp_kv_}, but got seqused_cmp_kv[{i}] = {sv[i]}")
                    if cu_cmp is not None:
                        seq_len = cu_cmp[i + 1] - cu_cmp[i]
                        if sv[i] > seq_len:
                            raise ValueError(
                                f"The elements in seqused_cmp_kv should not be greater than the sequence "
                                f"length from cu_seqlens_cmp_kv {seq_len}, but got seqused_cmp_kv[{i}] = {sv[i]}")
            if self.cmp_residual_kv_ is not None:
                cr = self.cmp_residual_kv_[0]
                for i in range(batch_size):
                    if cr[i] < 0 or cr[i] >= self.cmp_ratio_:
                        raise ValueError(
                            f"The elements in cmp_residual_kv should be in [0, cmpRatio_({self.cmp_ratio_})), "
                            f"but got cmp_residual_kv[{i}] = {cr[i]}")
            if (self.cmp_top_k_ != 0
                    and self.cmp_mask_mode_ == _SparseMode.DEFAULT_MASK
                    and self.cmp_topk_length_ is not None):
                sum_of_query_seq = self.get_sum_of_query_seq()
                data, _shape = self.cmp_topk_length_
                cmp_topk_length_size = self._topk_length_size(self.cmp_topk_length_)
                if cmp_topk_length_size < sum_of_query_seq:
                    raise ValueError(
                        f"The size of cmp_topk_length {cmp_topk_length_size} should not be smaller than "
                        f"the sum of query sequence {sum_of_query_seq}!")
                for i in range(cmp_topk_length_size):
                    if data[i] < 0:
                        raise ValueError(
                            f"The elements in cmp_topk_length should be >= 0, "
                            f"but got cmp_topk_length[{i}] = {data[i]}")

    def _calc_ori_mask_mode(self):
        """aicpu.cpp:369-384。"""
        if self.ori_mask_mode_ == _SparseMode.DEFAULT_MASK:
            self.ori_pre_token_ = INT64_MAX
            self.ori_next_token_ = INT64_MAX
            self.ori_attention_mode_ = NO_MASK
        elif self.ori_mask_mode_ == _SparseMode.RIGHT_DOWN_CAUSAL:
            self.ori_pre_token_ = INT64_MAX
            self.ori_next_token_ = 0
            self.ori_attention_mode_ = HAS_MASK
        else:  # SparseMode = 4
            self.ori_pre_token_ = self.ori_win_left_ if self.ori_win_left_ > -1 else INT64_MAX
            self.ori_next_token_ = self.ori_win_right_ if self.ori_win_right_ > -1 else INT64_MAX
            self.ori_attention_mode_ = HAS_MASK

    def _calc_cmp_mask_mode(self):
        """aicpu.cpp:386-401。⚠️ 照抄参考的笔误级行为：cmp 侧读的也是 **ori**WinLeft/Right。"""
        if self.cmp_mask_mode_ == _SparseMode.DEFAULT_MASK:
            self.cmp_pre_token_ = INT64_MAX
            self.cmp_next_token_ = INT64_MAX
            self.cmp_attention_mode_ = NO_MASK
        elif self.cmp_mask_mode_ == _SparseMode.RIGHT_DOWN_CAUSAL:
            self.cmp_pre_token_ = INT64_MAX
            self.cmp_next_token_ = 0
            self.cmp_attention_mode_ = HAS_MASK
        else:  # SparseMode = 4
            self.cmp_pre_token_ = self.ori_win_left_ if self.ori_win_left_ > -1 else INT64_MAX
            self.cmp_next_token_ = self.ori_win_right_ if self.ori_win_right_ > -1 else INT64_MAX
            self.cmp_attention_mode_ = HAS_MASK

    def _params_init(self):
        """aicpu.cpp:413-442。"""
        self.batch_size_ = self.get_query_batch_size()
        self._calc_ori_mask_mode()
        self._calc_cmp_mask_mode()
        self.is_s1g_ = self.layout_q_ in ("BSND", "BSH", "TND")
        self.group_size_ = _u32(self.num_heads_q_ // self.num_heads_kv_) if self.num_heads_kv_ else 0
        if self.has_ori_kv_ and self.ori_top_k_ != 0:
            self.is_sparse_ori_kv_ = True
        if self.has_cmp_kv_ and self.cmp_top_k_ != 0:
            self.is_sparse_cmp_kv_ = True
        if self.is_batch_consistency_:
            self.support_fd_ = True
        if "Ascend950" in self.soc_version_:
            if self.group_size_ > 64:
                self.is_split_g_ = True
                self.aic_core_num_ //= 2  # 2：核心数减半以平衡负载
            self.m_base_size_ = self.group_size_
            self.s2_base_size_ = 128
        else:
            self.m_base_size_ = self.group_size_
            self.s2_base_size_ = 128

    # ------------------------------------------------------------------
    # 逐 batch 长度 / 寻址（aicpu.cpp:444-568）
    # ------------------------------------------------------------------

    def get_s1_idx(self, s1_size, s1_g_idx):
        """aicpu.cpp:444-454。isS1G_ 恒真 ⇒ `s1Idx = s1GToken / groupSize_`。"""
        s1_g_token = _u32(s1_g_idx * self.m_base_size_)
        if self.is_s1g_:
            return _u32(s1_g_token // self.group_size_)
        return _u32(s1_g_token % s1_size)

    def get_bs_stride(self, b_idx, s1_idx):
        """aicpu.cpp:456-468。"""
        if self.layout_q_ == "TND":
            if self.cu_seqlens_q_ is not None:
                return _u32(self.cu_seqlens_q_[0][b_idx] + s1_idx)
        return _u32(b_idx * self.max_seqlen_q_ + s1_idx)

    def get_ori_topk_length(self, bs_stride):
        """aicpu.cpp:470-480。"""
        if (self.ori_top_k_ != 0
                and self.ori_mask_mode_ == _SparseMode.DEFAULT_MASK
                and self.ori_topk_length_ is not None):
            return _u32(self.ori_topk_length_[0][bs_stride])
        return _u32(self.ori_top_k_)

    def get_cmp_topk_length(self, bs_stride):
        """aicpu.cpp:482-492。"""
        if (self.cmp_top_k_ != 0
                and self.cmp_mask_mode_ == _SparseMode.DEFAULT_MASK
                and self.cmp_topk_length_ is not None):
            return _u32(self.cmp_topk_length_[0][bs_stride])
        return _u32(self.cmp_top_k_)

    def get_s1_seq_size(self, b_idx):
        """aicpu.cpp:494-511。"""
        if self.seqused_q_ is not None:
            return _u32(self.seqused_q_[0][b_idx])
        if self.layout_q_ == "TND":
            if self.cu_seqlens_q_ is not None:
                p = self.cu_seqlens_q_[0]
                return _u32(p[b_idx + 1] - p[b_idx])
        return _u32(self.max_seqlen_q_)

    def get_ori_s2_seq_size(self, b_idx):
        """aicpu.cpp:513-534。"""
        if self.seqused_ori_kv_ is not None:
            return _u32(self.seqused_ori_kv_[0][b_idx])
        if self.layout_kv_ == "TND":
            if self.cu_seqlens_ori_kv_ is not None:
                p = self.cu_seqlens_ori_kv_[0]
                return _u32(p[b_idx + 1] - p[b_idx])
        if (self.layout_kv_ == "PA_BBND" or self.max_seqlen_ori_kv_ == 0) and self.is_sparse_ori_kv_:
            return UINT32_MAX
        return _u32(self.max_seqlen_ori_kv_)

    def get_cmp_s2_seq_size(self, b_idx):
        """aicpu.cpp:536-557。"""
        if self.seqused_cmp_kv_ is not None:
            return _u32(self.seqused_cmp_kv_[0][b_idx])
        if self.layout_kv_ == "TND":
            if self.cu_seqlens_cmp_kv_ is not None:
                p = self.cu_seqlens_cmp_kv_[0]
                return _u32(p[b_idx + 1] - p[b_idx])
        if (self.layout_kv_ == "PA_BBND" or self.max_seqlen_cmp_kv_ == 0) and self.is_sparse_cmp_kv_:
            return UINT32_MAX
        return _u32(self.max_seqlen_cmp_kv_)

    def get_revert_s2_size(self, b_idx):
        """aicpu.cpp:559-568。"""
        cmp_s2_size = self.get_cmp_s2_seq_size(b_idx)
        if self.cmp_residual_kv_ is not None:
            return self.cmp_residual_kv_[0][b_idx] + cmp_s2_size * self.cmp_ratio_
        return cmp_s2_size * self.cmp_ratio_

    # ------------------------------------------------------------------
    # 切分统计 / cost 模型（aicpu.cpp:570-698）
    # ------------------------------------------------------------------

    def _calc_split_info(self, split_context):
        """aicpu.cpp:570-591 CalcSplitInfo。"""
        split_info = split_context.split_info
        for b_idx in range(self.batch_size_):
            s1_size = self.get_s1_seq_size(b_idx)
            split_info.s1_g_base_num[b_idx] = _u32(
                (s1_size * self.group_size_ + (self.m_base_size_ - 1)) // self.m_base_size_)
            split_info.s1_g_tail_size[b_idx] = _u32(
                (s1_size * self.group_size_) % self.m_base_size_)
            if self.has_ori_kv_:
                cur_ori_s2_size = self.get_ori_s2_seq_size(b_idx)
                split_info.ori_s2_base_num[b_idx] = _u32(
                    (cur_ori_s2_size + self.s2_base_size_ - 1) // self.s2_base_size_)
            if self.has_cmp_kv_:
                cur_cmp_s2_size = self.get_cmp_s2_seq_size(b_idx)
                split_info.cmp_s2_base_num[b_idx] = _u32(
                    (cur_cmp_s2_size + self.s2_base_size_ - 1) // self.s2_base_size_)
            if split_info.s1_g_base_num[b_idx] != 0 and (
                    split_info.ori_s2_base_num[b_idx] != 0
                    or split_info.cmp_s2_base_num[b_idx] != 0):
                split_info.is_kv_seq_all_zero = False

    def _calc_ori_pre_token_left_up(self, s1_size, s2_size):
        """aicpu.cpp:593-601。"""
        if self.ori_mask_mode_ == _SparseMode.BAND:
            return INT64_MAX if self.ori_pre_token_ == INT64_MAX \
                else s1_size - s2_size + self.ori_pre_token_
        return self.ori_pre_token_

    def _calc_ori_next_token_left_up(self, s1_size, s2_size):
        """aicpu.cpp:603-620。"""
        mode = self.ori_mask_mode_
        if mode in (_SparseMode.DEFAULT_MASK, _SparseMode.ALL_MASK, _SparseMode.LEFT_UP_CAUSAL):
            return self.ori_next_token_
        if mode == _SparseMode.RIGHT_DOWN_CAUSAL:
            return s2_size - s1_size
        if mode == _SparseMode.BAND:
            return INT64_MAX if self.ori_next_token_ == INT64_MAX \
                else s2_size - s1_size + self.ori_next_token_
        return self.ori_next_token_

    def _calc_cmp_pre_token_left_up(self, s1_size, s2_size):
        """aicpu.cpp:622-630。"""
        if self.cmp_mask_mode_ == _SparseMode.BAND:
            return INT64_MAX if self.cmp_pre_token_ == INT64_MAX \
                else s1_size - s2_size + self.cmp_pre_token_
        return self.cmp_pre_token_

    def _calc_cmp_next_token_left_up(self, s1_size, s2_size):
        """aicpu.cpp:632-649。"""
        mode = self.cmp_mask_mode_
        if mode in (_SparseMode.DEFAULT_MASK, _SparseMode.ALL_MASK, _SparseMode.LEFT_UP_CAUSAL):
            return self.cmp_next_token_
        if mode == _SparseMode.RIGHT_DOWN_CAUSAL:
            return s2_size - s1_size
        if mode == _SparseMode.BAND:
            return INT64_MAX if self.cmp_next_token_ == INT64_MAX \
                else s2_size - s1_size + self.cmp_next_token_
        return self.cmp_next_token_

    def _ori_calc_cost(self, basic_m, basic_s2):
        """aicpu.cpp:651-658。"""
        align_m = (basic_m + 15) // 16
        align_s2 = (basic_s2 + 63) // 64
        return COST_WEIGHT_M * align_m + COST_WEIGHT_S2 * align_s2

    def _cmp_calc_cost(self, basic_m, basic_s2):
        """aicpu.cpp:660-667。"""
        align_m = (basic_m + 15) // 16
        align_s2 = (basic_s2 + 63) // 64
        return COST_WEIGHT_M * align_m + COST_WEIGHT_S2 * align_s2

    def _calc_cost_table(self, s1_g_tail_size, reduction_block_size, ori_s2_tail_size,
                         cmp_s2_tail_size):
        """aicpu.cpp:669-698 CalcCostTable（写 self._type_cost_）。"""
        tc = self._type_cost_
        if self.has_ori_kv_:
            tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_NORMAL_BLOCK] = (
                self._ori_calc_cost(self.m_base_size_, reduction_block_size)
                if self.is_batch_consistency_
                else self._ori_calc_cost(self.m_base_size_, self.s2_base_size_))
            tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_NORMAL_BLOCK] = (
                0 if s1_g_tail_size == 0 else
                (self._ori_calc_cost(s1_g_tail_size, reduction_block_size)
                 if self.is_batch_consistency_
                 else self._ori_calc_cost(s1_g_tail_size, self.s2_base_size_)))
            tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_TAIL_BLOCK] = (
                0 if ori_s2_tail_size == 0 else self._ori_calc_cost(self.m_base_size_, ori_s2_tail_size))
            tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_TAIL_BLOCK] = (
                0 if (s1_g_tail_size == 0 or ori_s2_tail_size == 0)
                else self._ori_calc_cost(s1_g_tail_size, ori_s2_tail_size))
        if self.has_cmp_kv_:
            tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_NORMAL_BLOCK] = (
                self._cmp_calc_cost(self.m_base_size_, reduction_block_size)
                if self.is_batch_consistency_
                else self._cmp_calc_cost(self.m_base_size_, self.s2_base_size_))
            tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_NORMAL_BLOCK] = (
                0 if s1_g_tail_size == 0 else
                (self._cmp_calc_cost(s1_g_tail_size, reduction_block_size)
                 if self.is_batch_consistency_
                 else self._cmp_calc_cost(s1_g_tail_size, self.s2_base_size_)))
            tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_TAIL_BLOCK] = (
                0 if cmp_s2_tail_size == 0 else self._cmp_calc_cost(self.m_base_size_, cmp_s2_tail_size))
            tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_TAIL_BLOCK] = (
                0 if (s1_g_tail_size == 0 or cmp_s2_tail_size == 0)
                else self._cmp_calc_cost(s1_g_tail_size, cmp_s2_tail_size))

    def _calc_s2_token_range(self, s1_g_idx, batch_cache, is_cmp_kv):
        """aicpu.cpp:700-758 CalcS2TokenRange：返回 (first, last) int64 对。"""
        if not is_cmp_kv:
            if batch_cache.s1_size == 0 or batch_cache.ori_s2_size == 0:
                return (0, 0)
        else:
            if batch_cache.s1_size == 0 or batch_cache.cmp_revert_s2_size == 0:
                return (0, 0)
        s2_size = batch_cache.cmp_revert_s2_size if is_cmp_kv else batch_cache.ori_s2_size
        has_mask = self.cmp_attention_mode_ if is_cmp_kv else self.ori_attention_mode_
        if not has_mask:
            return (0, s2_size - 1)
        s1_g_first_token = s1_g_idx * self.m_base_size_
        s1_g_last_token = min(s1_g_first_token + self.m_base_size_,
                              batch_cache.s1_size * self.group_size_) - 1
        if self.is_s1g_:
            s1_first_token = s1_g_first_token // self.group_size_
            s1_last_token = s1_g_last_token // self.group_size_
        else:
            if s1_g_first_token // batch_cache.s1_size == s1_g_last_token // batch_cache.s1_size:
                s1_first_token = s1_g_first_token % batch_cache.s1_size
                s1_last_token = s1_g_last_token % batch_cache.s1_size
            else:
                s1_first_token = 0
                s1_last_token = batch_cache.s1_size
        if not is_cmp_kv:
            s2_first_token = s1_first_token - batch_cache.ori_pre_token_left_up
            s2_last_token = (INT64_MAX if batch_cache.ori_next_token_left_up == INT64_MAX
                             else s1_last_token + batch_cache.ori_next_token_left_up)
        else:
            s2_first_token = s1_first_token - batch_cache.cmp_pre_token_left_up
            s2_last_token = (INT64_MAX if batch_cache.cmp_next_token_left_up == INT64_MAX
                             else s1_last_token + batch_cache.cmp_next_token_left_up)
        return (s2_first_token, s2_last_token)

    def _calc_batch_cache(self, b_idx, split_context, batch_cache):
        """aicpu.cpp:760-777。"""
        batch_cache.b_idx = b_idx
        batch_cache.s1_size = self.get_s1_seq_size(b_idx)
        if self.has_ori_kv_:
            batch_cache.ori_s2_size = self.get_ori_s2_seq_size(b_idx)
            batch_cache.ori_pre_token_left_up = self._calc_ori_pre_token_left_up(
                batch_cache.s1_size, batch_cache.ori_s2_size)
            batch_cache.ori_next_token_left_up = self._calc_ori_next_token_left_up(
                batch_cache.s1_size, batch_cache.ori_s2_size)
        if self.has_cmp_kv_:
            batch_cache.cmp_revert_s2_size = self.get_revert_s2_size(b_idx)
            batch_cache.cmp_pre_token_left_up = self._calc_cmp_pre_token_left_up(
                batch_cache.s1_size, batch_cache.cmp_revert_s2_size)
            batch_cache.cmp_next_token_left_up = self._calc_cmp_next_token_left_up(
                batch_cache.s1_size, batch_cache.cmp_revert_s2_size)

    def _calc_ori_s1g_cache(self, s1g_cache, split_info):
        """aicpu.cpp:779-809。"""
        if s1g_cache.ori_s2_start >= s1g_cache.ori_s2_end:
            s1g_cache.ori_s1_g_block = 0
            s1g_cache.ori_s1_g_cost = 0
            s1g_cache.ori_s1_g_last_block_cost = 0
            s1g_cache.ori_s1_g_normal_block_cost = 0
        else:
            s1g_cache.ori_s1_g_block = s1g_cache.ori_s2_end - s1g_cache.ori_s2_start
            cur_ori_tail_s2_num = 1 if s1g_cache.ori_s2_tail_size != 0 else 0
            cur_ori_normal_s2_num = s1g_cache.ori_s1_g_block - cur_ori_tail_s2_num
            tc = self._type_cost_
            if (s1g_cache.s1_g_idx == split_info.s1_g_base_num[s1g_cache.b_idx] - 1
                    and split_info.s1_g_tail_size[s1g_cache.b_idx] != 0):
                s1g_cache.ori_s1_g_cost = (
                    tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_NORMAL_BLOCK] * cur_ori_normal_s2_num
                    + tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_TAIL_BLOCK] * cur_ori_tail_s2_num)
                s1g_cache.ori_s1_g_last_block_cost = (
                    tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_TAIL_BLOCK] if cur_ori_tail_s2_num > 0
                    else tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_NORMAL_BLOCK])
                s1g_cache.ori_s1_g_normal_block_cost = (
                    tc[_BlockType.ORI_TAIL_BLOCK][_BlockType.ORI_NORMAL_BLOCK])
            else:
                s1g_cache.ori_s1_g_cost = (
                    tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_NORMAL_BLOCK] * cur_ori_normal_s2_num
                    + tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_TAIL_BLOCK] * cur_ori_tail_s2_num)
                s1g_cache.ori_s1_g_last_block_cost = (
                    tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_TAIL_BLOCK] if cur_ori_tail_s2_num > 0
                    else tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_NORMAL_BLOCK])
                s1g_cache.ori_s1_g_normal_block_cost = (
                    tc[_BlockType.ORI_NORMAL_BLOCK][_BlockType.ORI_NORMAL_BLOCK])

    def _calc_cmp_s1g_cache(self, s1g_cache, split_info):
        """aicpu.cpp:811-841。"""
        if s1g_cache.cmp_s2_start >= s1g_cache.cmp_s2_end:
            s1g_cache.cmp_s1_g_block = 0
            s1g_cache.cmp_s1_g_cost = 0
            s1g_cache.cmp_s1_g_last_block_cost = 0
            s1g_cache.cmp_s1_g_normal_block_cost = 0
        else:
            s1g_cache.cmp_s1_g_block = s1g_cache.cmp_s2_end - s1g_cache.cmp_s2_start
            cur_cmp_tail_s2_num = 1 if s1g_cache.cmp_s2_tail_size != 0 else 0
            cur_cmp_normal_s2_num = s1g_cache.cmp_s1_g_block - cur_cmp_tail_s2_num
            tc = self._type_cost_
            if (s1g_cache.s1_g_idx == split_info.s1_g_base_num[s1g_cache.b_idx] - 1
                    and split_info.s1_g_tail_size[s1g_cache.b_idx] != 0):
                s1g_cache.cmp_s1_g_cost = (
                    tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_NORMAL_BLOCK] * cur_cmp_normal_s2_num
                    + tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_TAIL_BLOCK] * cur_cmp_tail_s2_num)
                s1g_cache.cmp_s1_g_last_block_cost = (
                    tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_TAIL_BLOCK] if cur_cmp_tail_s2_num > 0
                    else tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_NORMAL_BLOCK])
                s1g_cache.cmp_s1_g_normal_block_cost = (
                    tc[_BlockType.CMP_TAIL_BLOCK][_BlockType.CMP_NORMAL_BLOCK])
            else:
                s1g_cache.cmp_s1_g_cost = (
                    tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_NORMAL_BLOCK] * cur_cmp_normal_s2_num
                    + tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_TAIL_BLOCK] * cur_cmp_tail_s2_num)
                s1g_cache.cmp_s1_g_last_block_cost = (
                    tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_TAIL_BLOCK] if cur_cmp_tail_s2_num > 0
                    else tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_NORMAL_BLOCK])
                s1g_cache.cmp_s1_g_normal_block_cost = (
                    tc[_BlockType.CMP_NORMAL_BLOCK][_BlockType.CMP_NORMAL_BLOCK])

    def _calc_ori_block_range(self, ori_s2_token_range, batch_cache, s1g_cache):
        """aicpu.cpp:843-868。"""
        ori_s2_first_token = ori_s2_token_range[0]
        ori_s2_last_token = ori_s2_token_range[1]
        s1g_cache.ori_s2_start = 0
        if (ori_s2_first_token >= batch_cache.ori_s2_size or ori_s2_last_token < 0
                or ori_s2_last_token < ori_s2_first_token):
            s1g_cache.ori_s2_end = 0
            s1g_cache.ori_s2_tail_size = 0
        else:
            ori_s2_first_token = _clip(ori_s2_first_token, 0, batch_cache.ori_s2_size - 1)
            ori_s2_last_token = _clip(ori_s2_last_token, 0, batch_cache.ori_s2_size - 1)
            s1_idx = self.get_s1_idx(batch_cache.s1_size, s1g_cache.s1_g_idx)
            bs_stride = self.get_bs_stride(s1g_cache.b_idx, s1_idx)
            ori_topk_size = self.get_ori_topk_length(bs_stride)
            s1g_cache.act_ori_s2_size = (
                min(ori_s2_last_token - ori_s2_first_token + 1, ori_topk_size)
                if self.is_sparse_ori_kv_
                else ori_s2_last_token - ori_s2_first_token + 1)
            s1g_cache.ori_s2_end = (0 if s1g_cache.act_ori_s2_size == 0
                                    else (s1g_cache.act_ori_s2_size - 1) // self.s2_base_size_ + 1)
            s1g_cache.ori_s2_tail_size = s1g_cache.act_ori_s2_size % self.s2_base_size_

    def _calc_cmp_block_range(self, cmp_revert_s2_token_range, batch_cache, s1g_cache):
        """aicpu.cpp:870-908。"""
        cmp_revert_s2_first_token = cmp_revert_s2_token_range[0]
        cmp_revert_s2_last_token = cmp_revert_s2_token_range[1]
        s1g_cache.cmp_s2_start = s1g_cache.ori_s2_end
        if (cmp_revert_s2_first_token >= batch_cache.cmp_revert_s2_size
                or cmp_revert_s2_last_token < 0
                or cmp_revert_s2_last_token < cmp_revert_s2_first_token):
            s1g_cache.cmp_s2_end = s1g_cache.cmp_s2_start
            s1g_cache.cmp_s2_tail_size = 0
        else:
            cmp_revert_s2_first_token = _clip(cmp_revert_s2_first_token, 0,
                                              batch_cache.cmp_revert_s2_size - 1)
            cmp_revert_s2_last_token = _clip(cmp_revert_s2_last_token, 0,
                                             batch_cache.cmp_revert_s2_size - 1)
            if (cmp_revert_s2_last_token + 1) // self.cmp_ratio_ == 0:
                s1g_cache.cmp_s2_end = s1g_cache.cmp_s2_start
                s1g_cache.cmp_s2_tail_size = 0
                return
            cmp_s2_first_token = (0 if (cmp_revert_s2_first_token + 1) // self.cmp_ratio_ == 0
                                  else (cmp_revert_s2_first_token + 1) // self.cmp_ratio_ - 1)
            cmp_s2_last_token = (cmp_revert_s2_last_token + 1) // self.cmp_ratio_ - 1
            s1_idx = self.get_s1_idx(batch_cache.s1_size, s1g_cache.s1_g_idx)
            bs_stride = self.get_bs_stride(s1g_cache.b_idx, s1_idx)
            cmp_topk_size = self.get_cmp_topk_length(bs_stride)
            s1g_cache.act_cmp_s2_size = (
                min(cmp_s2_last_token - cmp_s2_first_token + 1, cmp_topk_size)
                if self.is_sparse_cmp_kv_
                else cmp_s2_last_token - cmp_s2_first_token + 1)
            s1g_cache.cmp_s2_end = (s1g_cache.cmp_s2_start if s1g_cache.act_cmp_s2_size == 0
                                    else s1g_cache.cmp_s2_start
                                    + (s1g_cache.act_cmp_s2_size - 1) // self.s2_base_size_ + 1)
            s1g_cache.cmp_s2_tail_size = s1g_cache.act_cmp_s2_size % self.s2_base_size_

    def _gather_ori_and_cmp_cache(self, s1g_cache):
        """aicpu.cpp:910-922。"""
        s1g_cache.s2_start = 0
        if s1g_cache.cmp_s1_g_block > 0:
            s1g_cache.s1_g_last_block_cost = s1g_cache.cmp_s1_g_last_block_cost
            s1g_cache.s2_end = s1g_cache.cmp_s2_end
        else:
            s1g_cache.s1_g_last_block_cost = s1g_cache.ori_s1_g_last_block_cost
            s1g_cache.s2_end = s1g_cache.ori_s2_end
        s1g_cache.s1_g_block = s1g_cache.ori_s1_g_block + s1g_cache.cmp_s1_g_block
        s1g_cache.s1_g_cost = s1g_cache.ori_s1_g_cost + s1g_cache.cmp_s1_g_cost

    def _calc_s1g_cache(self, s1_g_idx, split_context, batch_cache, s1g_cache):
        """aicpu.cpp:924-998。⚠️ 调用方可能传入 s1GIdx == baseNum（幻影行游标）——
        参考此时照算不拒，本移植同样**不加任何 clamp**（游标落点由它决定）。"""
        split_info = split_context.split_info
        if split_info.s1_g_base_num[batch_cache.b_idx] == 0:
            s1g_cache.s1_g_cost = 0
            s1g_cache.s1_g_last_block_cost = 0
            s1g_cache.ori_s1_g_normal_block_cost = 0
            s1g_cache.ori_s1_g_last_block_cost = 0
            s1g_cache.cmp_s1_g_normal_block_cost = 0
            s1g_cache.cmp_s1_g_last_block_cost = 0
            s1g_cache.s1_g_block = 0
            s1g_cache.s2_loop = 0
            s1g_cache.s2_start = 0
            s1g_cache.cmp_s2_start = 0
            s1g_cache.s2_end = 0
            return
        s1g_cache.b_idx = batch_cache.b_idx
        s1g_cache.s1_g_idx = s1_g_idx
        s1g_cache.act_ori_s2_size = 0
        s1g_cache.act_cmp_s2_size = 0
        if self.has_ori_kv_:
            ori_s2_token_range = self._calc_s2_token_range(s1_g_idx, batch_cache, ORI_KV)
            self._calc_ori_block_range(ori_s2_token_range, batch_cache, s1g_cache)
        else:
            s1g_cache.ori_s2_start = 0
            s1g_cache.ori_s2_end = s1g_cache.ori_s2_start
            s1g_cache.ori_s2_tail_size = 0
        if self.has_cmp_kv_:
            cmp_revert_s2_token_range = self._calc_s2_token_range(s1_g_idx, batch_cache, CMP_KV)
            self._calc_cmp_block_range(cmp_revert_s2_token_range, batch_cache, s1g_cache)
        else:
            s1g_cache.cmp_s2_start = s1g_cache.ori_s2_end
            s1g_cache.cmp_s2_end = s1g_cache.cmp_s2_start
            s1g_cache.cmp_s2_tail_size = 0
        if self.is_batch_consistency_:
            act_total_s2_size = s1g_cache.act_ori_s2_size + s1g_cache.act_cmp_s2_size
            s1g_cache.reduction_block_size = (
                (act_total_s2_size // BATCH_CONSISTENCY_MAX_REDUCTION_PARTS
                 + self.s2_base_size_ - 1) // self.s2_base_size_ * self.s2_base_size_)
            s1g_cache.reduction_block_size = (self.s2_base_size_
                                              if s1g_cache.reduction_block_size == 0
                                              else s1g_cache.reduction_block_size)
            s1g_cache.ori_s2_end = (0 if s1g_cache.act_ori_s2_size == 0
                                    else (s1g_cache.act_ori_s2_size - 1)
                                    // s1g_cache.reduction_block_size + 1)
            s1g_cache.ori_s2_tail_size = s1g_cache.act_ori_s2_size % s1g_cache.reduction_block_size
            s1g_cache.cmp_s2_end = (s1g_cache.cmp_s2_start if s1g_cache.act_cmp_s2_size == 0
                                    else s1g_cache.cmp_s2_start
                                    + (s1g_cache.act_cmp_s2_size - 1)
                                    // s1g_cache.reduction_block_size + 1)
            s1g_cache.cmp_s2_tail_size = s1g_cache.act_cmp_s2_size % s1g_cache.reduction_block_size
        self._calc_cost_table(split_info.s1_g_tail_size[s1g_cache.b_idx],
                              s1g_cache.reduction_block_size, s1g_cache.ori_s2_tail_size,
                              s1g_cache.cmp_s2_tail_size)
        self._calc_ori_s1g_cache(s1g_cache, split_info)
        self._calc_cmp_s1g_cache(s1g_cache, split_info)
        self._gather_ori_and_cmp_cache(s1g_cache)
        s1g_cache.s2_loop = (
            (s1g_cache.act_ori_s2_size + self.s2_base_size_ - 1) // self.s2_base_size_
            + (s1g_cache.act_cmp_s2_size + self.s2_base_size_ - 1) // self.s2_base_size_)

    def _calc_batch_cost(self, b_idx, split_context, cost_info):
        """aicpu.cpp:1000-1045。"""
        cost_info.bn2_cost_of_each_batch[b_idx] = 0
        cost_info.bn2_block_of_each_batch[b_idx] = 0
        cost_info.bn2_s2_loop_of_each_batch[b_idx] = 0
        cost_info.bn2_last_block_cost_of_each_batch[b_idx] = 0
        if self.get_s1_seq_size(b_idx) == 0:
            return
        if not self.has_ori_kv_ and not self.has_cmp_kv_:
            return
        if not self.has_ori_kv_:
            if self.get_cmp_s2_seq_size(b_idx) == 0:
                return
        elif not self.has_cmp_kv_:
            if self.get_ori_s2_seq_size(b_idx) == 0:
                return
        else:
            if self.get_ori_s2_seq_size(b_idx) == 0 and self.get_cmp_s2_seq_size(b_idx) == 0:
                return
        b_cache = _BatchCache()
        s1g_cache = _S1GCache()
        self._calc_batch_cache(b_idx, split_context, b_cache)
        for s1_g_idx in range(split_context.split_info.s1_g_base_num[b_idx]):
            self._calc_s1g_cache(s1_g_idx, split_context, b_cache, s1g_cache)
            cost_info.bn2_cost_of_each_batch[b_idx] += s1g_cache.s1_g_cost
            cost_info.bn2_block_of_each_batch[b_idx] += s1g_cache.s1_g_block
            cost_info.bn2_s2_loop_of_each_batch[b_idx] += s1g_cache.s2_loop
            if s1g_cache.s1_g_cost > cost_info.max_s1g_cost:
                cost_info.max_s1g_cost = s1g_cache.s1_g_cost
            if s1g_cache.s1_g_block > 0:
                cost_info.bn2_last_block_cost_of_each_batch[b_idx] = s1g_cache.s1_g_last_block_cost

    def _calc_cost_info(self, split_context):
        """aicpu.cpp:1047-1064。"""
        split_info = split_context.split_info
        cost_info = split_context.cost_info
        if split_info.is_kv_seq_all_zero:
            cost_info.total_cost = 0
            cost_info.total_block_num = 0
            return
        for b_idx in range(self.batch_size_):
            self._calc_batch_cost(b_idx, split_context, cost_info)
            cost_info.total_cost += cost_info.bn2_cost_of_each_batch[b_idx] * self.num_heads_kv_
            cost_info.total_block_num += cost_info.bn2_block_of_each_batch[b_idx] * self.num_heads_kv_

    # ------------------------------------------------------------------
    # 分配（aicpu.cpp:1066-1432）
    # ------------------------------------------------------------------

    def _update_cursor(self, split_context, assign_context):
        """aicpu.cpp:1066-1114。"""
        split_info = split_context.split_info
        cost_info = split_context.cost_info
        update_s1g = False
        update_batch = False
        if assign_context.cur_s2_idx >= assign_context.s1g_cache.s2_end:
            assign_context.cur_s2_idx = 0
            assign_context.cur_s1g_idx += 1
            update_s1g = True
        if assign_context.cur_s1g_idx >= split_info.s1_g_base_num[assign_context.cur_b_idx]:
            assign_context.cur_s1g_idx = 0
            assign_context.cur_bn2_idx += 1
        if assign_context.cur_bn2_idx == self.batch_size_ * self.num_heads_kv_:
            assign_context.cur_s1g_idx = 0
            assign_context.cur_s2_idx = 0
            assign_context.is_finished = True
            return
        if assign_context.cur_bn2_idx // self.num_heads_kv_ != assign_context.cur_b_idx:
            assign_context.cur_b_idx = assign_context.cur_bn2_idx // self.num_heads_kv_
            assign_context.cur_s1g_idx = 0
            update_batch = True
            update_s1g = True
        if update_batch:
            self._calc_batch_cache(assign_context.cur_b_idx, split_context,
                                   assign_context.batch_cache)
            assign_context.bn2_cost = cost_info.bn2_cost_of_each_batch[assign_context.cur_b_idx]
            assign_context.bn2_block = cost_info.bn2_block_of_each_batch[assign_context.cur_b_idx]
            assign_context.bn2_s2_loop = cost_info.bn2_s2_loop_of_each_batch[assign_context.cur_b_idx]
        if update_s1g:
            self._calc_s1g_cache(assign_context.cur_s1g_idx, split_context,
                                 assign_context.batch_cache, assign_context.s1g_cache)
            assign_context.cur_s2_idx = (assign_context.s1g_cache.ori_s2_start
                                         if self.support_fd_ else 0)

    def _assign_by_batch(self, split_context, assign_context):
        """aicpu.cpp:1116-1153。"""
        if assign_context.is_finished:
            return
        cost_info = split_context.cost_info
        while (assign_context.bn2_cost == 0
               or _is_within_tolerance(
                   assign_context.core_cache.cost_limit,
                   cost_info.bn2_last_block_cost_of_each_batch[assign_context.cur_b_idx]
                   // FA_TOLERANCE_RATIO,
                   assign_context.core_cache.cost + assign_context.bn2_cost)):
            assign_context.core_cache.cost += assign_context.bn2_cost
            assign_context.core_cache.block += assign_context.bn2_block
            assign_context.core_cache.s2_loop += assign_context.bn2_s2_loop
            assign_context.cur_bn2_idx += 1
            if assign_context.cur_bn2_idx == self.batch_size_ * self.num_heads_kv_:
                assign_context.cur_s1g_idx = 0
                assign_context.cur_s2_idx = 0
                assign_context.is_finished = True
                return
            if assign_context.cur_bn2_idx // self.num_heads_kv_ != assign_context.cur_b_idx:
                assign_context.cur_b_idx = assign_context.cur_bn2_idx // self.num_heads_kv_
                self._calc_batch_cache(assign_context.cur_b_idx, split_context,
                                       assign_context.batch_cache)
            assign_context.bn2_cost = cost_info.bn2_cost_of_each_batch[assign_context.cur_b_idx]
            assign_context.bn2_block = cost_info.bn2_block_of_each_batch[assign_context.cur_b_idx]
            assign_context.bn2_s2_loop = cost_info.bn2_s2_loop_of_each_batch[assign_context.cur_b_idx]
            assign_context.cur_s1g_idx = 0
            self._calc_s1g_cache(assign_context.cur_s1g_idx, split_context,
                                 assign_context.batch_cache, assign_context.s1g_cache)
            assign_context.cur_s2_idx = assign_context.s1g_cache.s2_start

    def _assign_by_row(self, split_context, assign_context):
        """aicpu.cpp:1155-1185。⚠️ do-while 可把游标推到 baseNum（幻影行）——照抄。"""
        if assign_context.is_finished:
            return
        while _is_within_tolerance(
                assign_context.core_cache.cost_limit,
                assign_context.s1g_cache.s1_g_last_block_cost // FA_TOLERANCE_RATIO,
                assign_context.core_cache.cost + assign_context.s1g_cache.s1_g_cost):
            assign_context.core_cache.cost += assign_context.s1g_cache.s1_g_cost
            assign_context.core_cache.block += assign_context.s1g_cache.s1_g_block
            assign_context.core_cache.s2_loop += assign_context.s1g_cache.s2_loop
            assign_context.bn2_cost = (assign_context.bn2_cost - assign_context.s1g_cache.s1_g_cost
                                       if assign_context.bn2_cost > assign_context.s1g_cache.s1_g_cost
                                       else 0)
            assign_context.bn2_block = (assign_context.bn2_block - assign_context.s1g_cache.s1_g_block
                                        if assign_context.bn2_block > assign_context.s1g_cache.s1_g_block
                                        else 0)
            assign_context.bn2_s2_loop = (assign_context.bn2_s2_loop - assign_context.s1g_cache.s2_loop
                                          if assign_context.bn2_s2_loop > assign_context.s1g_cache.s2_loop
                                          else 0)
            while True:
                assign_context.cur_s1g_idx += 1
                self._calc_s1g_cache(assign_context.cur_s1g_idx, split_context,
                                     assign_context.batch_cache, assign_context.s1g_cache)
                if assign_context.s1g_cache.s1_g_block != 0:
                    break
            assign_context.cur_s2_idx = assign_context.s1g_cache.s2_start

    def _calc_cur_block_cost(self, assign_context):
        """aicpu.cpp:1187-1202。"""
        cur_cost = 0
        if assign_context.cur_s2_idx < assign_context.s1g_cache.cmp_s2_start:
            cur_cost = assign_context.s1g_cache.ori_s1_g_normal_block_cost
            if assign_context.cur_s2_idx == assign_context.s1g_cache.cmp_s2_start - 1:
                cur_cost = assign_context.s1g_cache.ori_s1_g_last_block_cost
        else:
            cur_cost = assign_context.s1g_cache.cmp_s1_g_normal_block_cost
            if assign_context.cur_s2_idx == assign_context.s1g_cache.s2_end - 1:
                cur_cost = assign_context.s1g_cache.cmp_s1_g_last_block_cost
        return cur_cost

    def _calc_cur_block_s2_loop(self, assign_context):
        """aicpu.cpp:1204-1219。"""
        if not self.is_batch_consistency_:
            return 1
        s1g_cache = assign_context.s1g_cache
        block_size = s1g_cache.reduction_block_size
        if assign_context.cur_s2_idx < s1g_cache.cmp_s2_start:
            if assign_context.cur_s2_idx + 1 == s1g_cache.cmp_s2_start and s1g_cache.ori_s2_tail_size != 0:
                block_size = s1g_cache.ori_s2_tail_size
        elif assign_context.cur_s2_idx + 1 == s1g_cache.s2_end and s1g_cache.cmp_s2_tail_size != 0:
            block_size = s1g_cache.cmp_s2_tail_size
        return (block_size + self.s2_base_size_ - 1) // self.s2_base_size_

    def _assign_by_block(self, split_context, assign_context):
        """aicpu.cpp:1221-1250。supportFd_ 恒真（aicpu.h:368），走 FD 按块分配路径。"""
        if assign_context.is_finished or not self.support_fd_:
            return
        cur_cost = self._calc_cur_block_cost(assign_context)
        cur_s2_loop = self._calc_cur_block_s2_loop(assign_context)
        while _is_within_tolerance(assign_context.core_cache.cost_limit,
                                   cur_cost // FA_TOLERANCE_RATIO,
                                   assign_context.core_cache.cost + cur_cost):
            assign_context.core_cache.cost += cur_cost
            assign_context.core_cache.block += 1
            assign_context.core_cache.s2_loop += cur_s2_loop
            assign_context.cur_s2_idx += 1
            assign_context.bn2_cost = assign_context.bn2_cost - cur_cost
            assign_context.s1g_cache.s1_g_cost = assign_context.s1g_cache.s1_g_cost - cur_cost
            assign_context.bn2_block = _u32(assign_context.bn2_block - 1)
            assign_context.s1g_cache.s1_g_block = _u32(assign_context.s1g_cache.s1_g_block - 1)
            assign_context.bn2_s2_loop = (assign_context.bn2_s2_loop - cur_s2_loop
                                          if assign_context.bn2_s2_loop > cur_s2_loop else 0)
            assign_context.s1g_cache.s2_loop = (assign_context.s1g_cache.s2_loop - cur_s2_loop
                                                if assign_context.s1g_cache.s2_loop > cur_s2_loop else 0)
            cur_cost = self._calc_cur_block_cost(assign_context)
            cur_s2_loop = self._calc_cur_block_s2_loop(assign_context)

    def _force_assign(self, split_context, assign_context):
        """aicpu.cpp:1252-1276。"""
        if assign_context.is_finished:
            return
        cur_cost = self._calc_cur_block_cost(assign_context)
        cur_s2_loop = self._calc_cur_block_s2_loop(assign_context)
        assign_context.core_cache.cost += cur_cost
        assign_context.core_cache.block += 1
        assign_context.core_cache.s2_loop += cur_s2_loop
        assign_context.cur_s2_idx += 1
        assign_context.bn2_cost = assign_context.bn2_cost - cur_cost
        assign_context.bn2_block = _u32(assign_context.bn2_block - 1)
        assign_context.bn2_s2_loop = (assign_context.bn2_s2_loop - cur_s2_loop
                                      if assign_context.bn2_s2_loop > cur_s2_loop else 0)
        assign_context.s1g_cache.s1_g_cost = assign_context.s1g_cache.s1_g_cost - cur_cost
        assign_context.s1g_cache.s1_g_block = _u32(assign_context.s1g_cache.s1_g_block - 1)
        assign_context.s1g_cache.s2_loop = (assign_context.s1g_cache.s2_loop - cur_s2_loop
                                            if assign_context.s1g_cache.s2_loop > cur_s2_loop else 0)
        self._update_cursor(split_context, assign_context)

    def _is_need_record_fd_info(self, assign_context, split_res):
        """aicpu.cpp:1278-1296。"""
        if assign_context.cur_core_idx == 0:
            return False
        if assign_context.cur_kv_split_part <= 1:
            return False
        if (assign_context.cur_bn2_idx == split_res.bn2_end[assign_context.cur_core_idx - 1]
                and assign_context.cur_s1g_idx == split_res.gs1_end[assign_context.cur_core_idx - 1]):
            return False
        return True

    def _is_first_reduction_block(self, assign_context, split_res):
        """aicpu.cpp:1298-1315。"""
        if assign_context.cur_core_idx == 0:
            return True
        if split_res.s2_end[assign_context.cur_core_idx - 1] == 0:
            return True
        if (assign_context.cur_bn2_idx != split_res.bn2_end[assign_context.cur_core_idx - 1]
                or assign_context.cur_s1g_idx != split_res.gs1_end[assign_context.cur_core_idx - 1]):
            return True
        return False

    def _record_fd_info(self, split_context, assign_context, result):
        """aicpu.cpp:1317-1340。"""
        split_info = split_context.split_info
        split_b_idx = result.bn2_end[assign_context.cur_core_idx - 1] // self.num_heads_kv_
        split_s1g_idx = result.gs1_end[assign_context.cur_core_idx - 1]
        s1_size = self.get_s1_seq_size(split_b_idx)
        cur_fd_s1g_size = (
            s1_size * self.group_size_ - split_s1g_idx * self.m_base_size_
            if split_s1g_idx == split_info.s1_g_base_num[split_b_idx] - 1
            else self.m_base_size_)
        result.max_s2_split_num = max(result.max_s2_split_num, assign_context.cur_kv_split_part)
        result.fd_res.fd_bn2_idx[result.num_of_fd_head] = result.bn2_end[assign_context.cur_core_idx - 1]
        result.fd_res.fd_m_idx[result.num_of_fd_head] = result.gs1_end[assign_context.cur_core_idx - 1]
        result.fd_res.fd_workspace_idx[result.num_of_fd_head] = assign_context.pre_fd_data_num
        result.fd_res.fd_s2_split_num[result.num_of_fd_head] = assign_context.cur_kv_split_part
        result.fd_res.fd_m_size[result.num_of_fd_head] = cur_fd_s1g_size
        result.num_of_fd_head += 1

    def _assign_blocks_to_core(self, split_context, assign_context, result):
        """aicpu.cpp:1342-1397。"""
        cost_info = split_context.cost_info
        result.first_fd_data_workspace_idx[assign_context.cur_core_idx] = (
            assign_context.pre_fd_data_num + assign_context.cur_kv_split_part - 1)
        avg_cost = _i64_div(assign_context.unassigned_cost,
                            self.aic_core_num_ - assign_context.cur_core_idx)
        assign_context.core_cache = _CoreCache()
        if not self.support_fd_:
            assign_context.core_cache.cost_limit = max(avg_cost, cost_info.max_s1g_cost)
        else:
            assign_context.core_cache.cost_limit = avg_cost
        self._assign_by_batch(split_context, assign_context)
        self._assign_by_row(split_context, assign_context)
        self._assign_by_block(split_context, assign_context)
        if assign_context.core_cache.block == 0 and self.support_fd_:
            self._force_assign(split_context, assign_context)
        result.bn2_end[assign_context.cur_core_idx] = assign_context.cur_bn2_idx
        result.gs1_end[assign_context.cur_core_idx] = assign_context.cur_s1g_idx
        result.s2_end[assign_context.cur_core_idx] = assign_context.cur_s2_idx
        result.max_cost = max(result.max_cost, assign_context.core_cache.cost)
        assign_context.unassigned_cost -= assign_context.core_cache.cost
        result.max_s2_loop_num = max(assign_context.core_cache.s2_loop, result.max_s2_loop_num)
        if self._is_need_record_fd_info(assign_context, result):
            if self.is_batch_consistency_:
                assign_context.cur_kv_split_part += self.remained_block_num_ - 1
            self._record_fd_info(split_context, assign_context, result)
            assign_context.pre_fd_data_num += assign_context.cur_kv_split_part
            assign_context.cur_kv_split_part = 1
        if (assign_context.cur_s2_idx > assign_context.s1g_cache.s2_start
                and assign_context.cur_s2_idx <= assign_context.s1g_cache.s2_end):
            if self.is_batch_consistency_:
                if self._is_first_reduction_block(assign_context, result):
                    assign_context.cur_kv_split_part += 1
                else:
                    assign_context.cur_kv_split_part += (
                        result.s2_end[assign_context.cur_core_idx]
                        - result.s2_end[assign_context.cur_core_idx - 1])
                self.remained_block_num_ = assign_context.s1g_cache.s1_g_block
            else:
                assign_context.cur_kv_split_part += 1

    def _calc_split_plan(self, cost_limit, split_context, result):
        """aicpu.cpp:1399-1432。"""
        cost_info = split_context.cost_info
        if self.aic_core_num_ == 0:
            return
        result.max_cost = 0
        result.used_core_num = 0
        assign_context = _AssignContext()
        assign_context.cur_b_idx = 0
        assign_context.cur_s1g_idx = 0
        assign_context.unassigned_cost = cost_info.total_cost
        assign_context.bn2_cost = cost_info.bn2_cost_of_each_batch[assign_context.cur_b_idx]
        assign_context.bn2_block = cost_info.bn2_block_of_each_batch[assign_context.cur_b_idx]
        assign_context.bn2_s2_loop = cost_info.bn2_s2_loop_of_each_batch[assign_context.cur_b_idx]
        self._calc_batch_cache(assign_context.cur_b_idx, split_context,
                               assign_context.batch_cache)
        self._calc_s1g_cache(assign_context.cur_s1g_idx, split_context,
                             assign_context.batch_cache, assign_context.s1g_cache)
        assign_context.cur_s2_idx = assign_context.s1g_cache.s2_start
        for i in range(self.aic_core_num_):
            if result.max_cost > cost_limit:
                return
            if assign_context.is_finished or assign_context.unassigned_cost <= 0:
                break
            assign_context.cur_core_idx = i
            self._assign_blocks_to_core(split_context, assign_context, result)
        result.used_core_num = assign_context.cur_core_idx + 1

    def _split_fd(self, split_res):
        """aicpu.cpp:1434-1475。"""
        total_fd_load = 0
        for i in range(split_res.num_of_fd_head):
            total_fd_load += split_res.fd_res.fd_s2_split_num[i] * split_res.fd_res.fd_m_size[i]
        empty_vector_num = self.aiv_core_num_ - split_res.num_of_fd_head
        average_load = (total_fd_load + self.aiv_core_num_ - 1) // self.aiv_core_num_
        cur_core_index = 0
        for i in range(split_res.num_of_fd_head):
            if empty_vector_num == 0:
                split_res.fd_res.fd_idx[cur_core_index] = i
                split_res.fd_res.fd_m_start[cur_core_index] = 0
                split_res.fd_res.fd_m_num[cur_core_index] = split_res.fd_res.fd_m_size[i]
                cur_core_index += 1
                continue
            cur_fd_vector_num = (split_res.fd_res.fd_s2_split_num[i]
                                 * split_res.fd_res.fd_m_size[i]) // average_load
            cur_fd_vector_num = max(1, cur_fd_vector_num)
            cur_ave_m_size = (split_res.fd_res.fd_m_size[i] + cur_fd_vector_num - 1) // cur_fd_vector_num
            cur_fd_vector_num = (split_res.fd_res.fd_m_size[i] + cur_ave_m_size - 1) // cur_ave_m_size
            cur_fd_vector_num = min(cur_fd_vector_num, empty_vector_num + 1)
            for vid in range(cur_fd_vector_num):
                split_res.fd_res.fd_idx[cur_core_index] = i
                split_res.fd_res.fd_m_start[cur_core_index] = vid * cur_ave_m_size
                split_res.fd_res.fd_m_num[cur_core_index] = (
                    cur_ave_m_size if vid < cur_fd_vector_num - 1
                    else split_res.fd_res.fd_m_size[i] - vid * cur_ave_m_size)
                cur_core_index += 1
            empty_vector_num -= cur_fd_vector_num - 1
        split_res.fd_res.fd_used_vec_num = cur_core_index

    def _balance_schedule(self, split_res):
        """aicpu.cpp:1477-1504。"""
        split_context = _SplitContext(self.batch_size_)
        self._calc_split_info(split_context)
        if split_context.split_info.is_kv_seq_all_zero:
            split_res.used_core_num = 1
            split_res.bn2_end[0] = self.batch_size_ * self.num_heads_kv_
            split_res.gs1_end[0] = 0
            split_res.s2_end[0] = 0
            return True
        self._calc_cost_info(split_context)
        self._apply_support_fd_gate(split_context)
        split_res.max_cost = INT64_MAX
        split_res.used_core_num = 1
        self._calc_split_plan(split_res.max_cost, split_context, split_res)
        if split_res.num_of_fd_head > 0:
            self._split_fd(split_res)
        split_res.used_core_num = max(split_res.used_core_num, 1)
        return True

    def _apply_support_fd_gate(self, split_context):
        """support_fd 门限（perf+precision critical，= ref/fd_version.py 语义）。

        只在 tile 级切分确实压低关键路径时开 FD：`(total_tiles > blocks) AND
        (ceil(rows/blocks)*max_row_tiles > ceil(total_tiles/blocks))`。用 cost 等价
        形式（max_s1g_cost / total_cost 均为 cost 单位，每 tile cost 44 恒等）：
          uniform_crit = ceil(rows/blocks) * max_s1g_cost   # 整行粒度关键路径上界
          fd_floor     = ceil(total_cost / blocks)          # 完美 tile 均衡下界
        关闭场景：T1 很小（total_tiles <= blocks，如 T1=1 的 5 tile）——此时 FD 把单行
        切成 5 份无意义归约，重结合误差把 0.94% 元素推出严格 BF16 比较器（99.057%<99.5%
        → FAIL）；且 K2=1024 时 k=9 会越 _FD_MP_SLOTS=8。batch_consistency 恒需 FD，不门控。
        """
        if self.is_batch_consistency_ or not self.support_fd_:
            return
        cost_info = split_context.cost_info
        blocks = self.aic_core_num_
        total_tiles = cost_info.total_block_num
        if total_tiles <= blocks:
            self.support_fd_ = False
            return
        rows = self.get_sum_of_query_seq()
        uniform_crit = ((rows + blocks - 1) // blocks) * cost_info.max_s1g_cost
        fd_floor = (cost_info.total_cost + blocks - 1) // blocks
        self.support_fd_ = uniform_crit > fd_floor

    def _gen_metadata(self, split_res):
        """aicpu.cpp:1506-1582。产出平铺 int32[1024]（FA 段 9×36 在前，FD 段 8×72 在后）。"""
        self.metadata_ = [0] * MQSMLA_METADATA_TOTAL_SIZE
        m = self.metadata_

        def fa(core, field, value):
            m[FA_METADATA_SIZE * core + field] = _u32(value)

        def fd(core, field, value):
            m[FA_METADATA_SIZE * AIC_CORE_MAX_NUM + FD_METADATA_SIZE * core + field] = _u32(value)

        if self.is_split_g_:
            for i in range(self.aic_core_num_):
                fa(2 * i, FA_S2_MAX_NUM, split_res.max_s2_loop_num)
                fa(2 * i + 1, FA_S2_MAX_NUM, split_res.max_s2_loop_num)
                if i >= split_res.used_core_num:
                    fa(2 * i, FA_CORE_ENABLE_INDEX, 0)
                    fa(2 * i + 1, FA_CORE_ENABLE_INDEX, 0)
                    continue
                fa(2 * i, FA_CORE_ENABLE_INDEX, 1)
                fa(2 * i + 1, FA_CORE_ENABLE_INDEX, 1)
                fa(2 * i, FA_BN2_START_INDEX, 0 if i == 0 else split_res.bn2_end[i - 1])
                fa(2 * i, FA_M_START_INDEX, 0 if i == 0 else split_res.gs1_end[i - 1])
                fa(2 * i, FA_S2_START_INDEX, 0 if i == 0 else split_res.s2_end[i - 1])
                fa(2 * i + 1, FA_BN2_START_INDEX, 0 if i == 0 else split_res.bn2_end[i - 1])
                fa(2 * i + 1, FA_M_START_INDEX, 0 if i == 0 else split_res.gs1_end[i - 1])
                fa(2 * i + 1, FA_S2_START_INDEX, 0 if i == 0 else split_res.s2_end[i - 1])
                fa(2 * i, FA_BN2_END_INDEX, split_res.bn2_end[i])
                fa(2 * i, FA_M_END_INDEX, split_res.gs1_end[i])
                fa(2 * i, FA_S2_END_INDEX, split_res.s2_end[i])
                fa(2 * i + 1, FA_BN2_END_INDEX, split_res.bn2_end[i])
                fa(2 * i + 1, FA_M_END_INDEX, split_res.gs1_end[i])
                fa(2 * i + 1, FA_S2_END_INDEX, split_res.s2_end[i])
                fa(2 * i, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                   split_res.first_fd_data_workspace_idx[i])
                fa(2 * i + 1, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                   split_res.first_fd_data_workspace_idx[i])
        else:
            for i in range(self.aic_core_num_):
                if i >= split_res.used_core_num:
                    fa(i, FA_CORE_ENABLE_INDEX, 0)
                    continue
                fa(i, FA_CORE_ENABLE_INDEX, 1)
                fa(i, FA_BN2_START_INDEX, 0 if i == 0 else split_res.bn2_end[i - 1])
                fa(i, FA_M_START_INDEX, 0 if i == 0 else split_res.gs1_end[i - 1])
                fa(i, FA_S2_START_INDEX, 0 if i == 0 else split_res.s2_end[i - 1])
                fa(i, FA_BN2_END_INDEX, split_res.bn2_end[i])
                fa(i, FA_M_END_INDEX, split_res.gs1_end[i])
                fa(i, FA_S2_END_INDEX, split_res.s2_end[i])
                fa(i, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                   split_res.first_fd_data_workspace_idx[i])
        for i in range(self.aiv_core_num_):
            if i >= split_res.fd_res.fd_used_vec_num:
                fd(i, FD_CORE_ENABLE_INDEX, 0)
                continue
            fd(i, FD_CORE_ENABLE_INDEX, 1)
            cur_fd_idx = split_res.fd_res.fd_idx[i]
            fd(i, FD_BN2_IDX_INDEX, split_res.fd_res.fd_bn2_idx[cur_fd_idx])
            fd(i, FD_M_IDX_INDEX, split_res.fd_res.fd_m_idx[cur_fd_idx])
            fd(i, FD_WORKSPACE_IDX_INDEX, split_res.fd_res.fd_workspace_idx[cur_fd_idx])
            fd(i, FD_WORKSPACE_NUM_INDEX, split_res.fd_res.fd_s2_split_num[cur_fd_idx])
            fd(i, FD_M_START_INDEX, split_res.fd_res.fd_m_start[i])
            fd(i, FD_M_NUM_INDEX, split_res.fd_res.fd_m_num[i])
        # mqsmla 消费侧统一 FD 使能计数（= FD 归约 AIV 数），barrier 门控用。
        m[FD_USED_VEC_NUM_WORD] = _u32(split_res.fd_res.fd_used_vec_num)
        return True

    # ------------------------------------------------------------------
    # 消费侧 helper（我方主算子用；AICPU 侧无此代码，纯派生）
    # ------------------------------------------------------------------

    def core_row_ranges(self, global_row_of_batch):
        """把 FA 游标 (bn2, gs1) 映成每核全局扁平行区间 [m0, m1)。

        `global_row_of_batch(b)`：BSND 等长 = `b * S1`；TND = `cu_q[b]`。
        依据：AICPU 游标在 (batch, 行) 字典序上连续，且幻影行（gs1 ≥ baseNum）的
        `row(b) + gs1` 恰好落在下一 batch 的前几行 —— 覆盖恒为 [0, 总行数) 的精确划分。
        返回 `[(m0, m1, core), ...]`，split_g 时 core 是 pair 序号（2i/2i+1 同区间）。"""
        n_range = self.aic_core_num_  # split_g 时已被减半 = pair 数
        out = []
        for i in range(n_range):
            if self.is_split_g_:
                core = 2 * i
            else:
                core = i
            if self.metadata_[FA_METADATA_SIZE * core + FA_CORE_ENABLE_INDEX] == 0:
                out.append((0, 0, core))
                continue
            bn2_s = self.metadata_[FA_METADATA_SIZE * core + FA_BN2_START_INDEX]
            gs1_s = self.metadata_[FA_METADATA_SIZE * core + FA_M_START_INDEX]
            bn2_e = self.metadata_[FA_METADATA_SIZE * core + FA_BN2_END_INDEX]
            gs1_e = self.metadata_[FA_METADATA_SIZE * core + FA_M_END_INDEX]
            m0 = global_row_of_batch(bn2_s) + gs1_s
            m1 = global_row_of_batch(bn2_e) + gs1_e
            out.append((m0, m1, core))
        return out

    def max_rows_per_core(self, global_row_of_batch):
        """SPLIT_G 全 grid 屏障配额用的 `m_fixed`：max over 核（split_g = pair）的行数。"""
        return max(m1 - m0 for m0, m1, _ in self.core_row_ranges(global_row_of_batch))


def _as_i32_list(tensor):
    """输入张量 → `(data:list[int], shape:tuple)`；None 透传。

    接受 host list/tuple、torch.Tensor（CPU 或 NPU，int32）。shape 缺省按 2-D [1, N]
    补齐（校验路径只用前 3 维长度）。"""
    if tensor is None:
        return None
    if isinstance(tensor, tuple) and len(tensor) == 2:
        return tensor
    import torch
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().to(torch.int32).flatten().tolist()
        return (data, tuple(tensor.shape))
    data = [int(x) for x in tensor]
    return (data, (1, len(data)))


def _resolve_device_core_nums(aic_core_num=None, aiv_core_num=None, device=None):
    """Return metadata scheduling core counts from the active NPU.

    Explicit values keep the CPU planner independently testable.  When omitted,
    use the same device properties consumed by ``sparse_flash_mla`` for its
    launch ``block_dim``.  The 36/72 constants describe metadata capacity only.
    """
    if aic_core_num is not None and aiv_core_num is not None:
        return int(aic_core_num), int(aiv_core_num)
    import torch
    if device is None:
        device = torch.npu.current_device()
    props = torch.npu.get_device_properties(device)
    if aic_core_num is None:
        aic_core_num = int(props.cube_core_num)
    if aiv_core_num is None:
        aiv_core_num = int(props.vector_core_num)
    if not 0 < int(aic_core_num) <= AIC_CORE_MAX_NUM:
        raise ValueError(f"aic_core_num must be in [1, {AIC_CORE_MAX_NUM}], got {aic_core_num}")
    if not 0 < int(aiv_core_num) <= AIV_CORE_MAX_NUM:
        raise ValueError(f"aiv_core_num must be in [1, {AIV_CORE_MAX_NUM}], got {aiv_core_num}")
    return int(aic_core_num), int(aiv_core_num)


def _get_cube_core_num(device=None):
    """Return the cube-core count of the active NPU (same source as the consumer)."""
    import torch
    return int(torch.npu.get_device_properties(device).cube_core_num)


# ---- AICPU metadata 算子（device 侧，msprof 可见的 AI_CPU 任务）----
# 上面的 SparseFlashMlaMetadataCpuKernel 是 AscendC AICPU 规划器的 host 逐行移植
# （FD 开关 SUPPORT_FD 控制；当前 False）。本 AICPU kernel 是该规划器在
# SUPPORT_FD=False（整行均匀划分、无 FD 切分）时的 device 等价实现：产出与 host
# 移植同布局的 int32[1024]（FA 段填均匀划分，FD 段全零、fd_used_vec_num=0）。
# 消费侧 mixed_quant_sparse_flash_mla 据此走非 FD 均匀路径（与 SUPPORT_FD=False
# 的 host 移植语义一致）。FD 重新打开仍需走 host 移植（或未来把 FD 规划器也下放
# AICPU），本 kernel 不实现 FD 切分。
class _MqsmlaMetadataArgs:
    metadata: GmOut(I32)
    cu_q: GmIn(I32)
    rows: I32
    blocks: I32
    batch: I32


@aicpu_kernel
def _mqsmla_metadata_kernel(a: _MqsmlaMetadataArgs):
    # 清零责任收归本 AICPU kernel：host 侧用 torch.empty 分配，不再调 torch.zeros。
    # 消费侧内核会无条件读取本 kernel 未填写的字段——每个使能核的 s2_start/s2_end/
    # first_fd、fd_used_vec_num(word 900)，以及 fd_any!=0 时的 FD 段(324..899)——
    # 这些位置必须为 0。只清有效区 [0, FD_USED_VEC_NUM_WORD+1)（FA 324 + FD 576 +
    # fdUsedVecNum 1 = 901 字）；尾部 901..1023 是布局填充，消费侧与 pytest 的
    # metadata-backend 比对都不触碰，留 undefined（不写）。
    for i in range(0, FD_USED_VEC_NUM_WORD + 1):
        a.metadata[i] = 0
    if a.rows <= 0 or a.blocks <= 0:
        return 1
    rows_per_core = a.rows // a.blocks
    extra = a.rows % a.blocks
    start = 0
    batch1 = a.batch + 1
    for core in range(0, a.blocks):
        count = rows_per_core
        if core < extra:
            count += 1
        end = start + count
        if count > 0:
            bn2_s = 0
            for b in range(0, batch1):
                if a.cu_q[b] <= start:
                    bn2_s = b
            m_s = start - a.cu_q[bn2_s]
            bn2_e = 0
            for b in range(0, batch1):
                if a.cu_q[b] <= end:
                    bn2_e = b
            m_e = end - a.cu_q[bn2_e]
            fa = FA_METADATA_SIZE * core
            a.metadata[fa + FA_CORE_ENABLE_INDEX] = 1
            a.metadata[fa + FA_BN2_START_INDEX] = bn2_s
            a.metadata[fa + FA_M_START_INDEX] = m_s
            a.metadata[fa + FA_BN2_END_INDEX] = bn2_e
            a.metadata[fa + FA_M_END_INDEX] = m_e
        start = end
    return 0


_METADATA_COMPILED = None
_METADATA_DIRECTORY = None
_METADATA_LOCK = threading.Lock()


def _get_compiled_metadata():
    """Load or compile the mqsmla metadata AICPU kernel once per process."""
    global _METADATA_COMPILED, _METADATA_DIRECTORY
    from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

    with _METADATA_LOCK:
        if _METADATA_COMPILED is None:
            _METADATA_DIRECTORY = tempfile.TemporaryDirectory(prefix="mqsmla_metadata_aicpu_")
            _METADATA_COMPILED = compile_aicpu_kernel(
                _mqsmla_metadata_kernel, workdir=_METADATA_DIRECTORY.name,
                launch_mode="interface")
        return _METADATA_COMPILED


def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
):
    """Generate MQSMLA ``int32[1024]`` metadata on an AICPU core.

    Uniform whole-row partition (FD off): the TND query rows are split evenly
    across the cube cores; each core's contiguous ``[m_start, m_end)`` row range
    is written into the FA metadata section as ``(batch, row-within-batch)`` via
    ``cu_seqlens_q``.  The FD section is left all-zero (``fd_used_vec_num=0``),
    so the attention kernel runs its uniform non-FD path — matching the host
    planner ``SparseFlashMlaMetadataCpuKernel`` with ``SUPPORT_FD=False``.
    """
    import torch

    for name, value in (("ori_topk_length", ori_topk_length),
                        ("cmp_topk_length", cmp_topk_length)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int32
                or value.dim() != 2 or value.shape[1] != num_heads_kv
                or value.shape[0] <= 0):
            raise ValueError(f"{name} must be an int32 tensor of shape [T1, N2]")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous (flat AICPU indexing)")
    if ori_topk_length.shape != cmp_topk_length.shape:
        raise ValueError("topk_length tensors must have the same shape")
    if (num_heads_q, num_heads_kv, head_dim, quant_mode) != (64, 1, 512, 1):
        raise ValueError("metadata supports num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1")
    if layout_q != "TND" or layout_kv != "PA_BBND" or not has_ori_kv:
        raise ValueError("metadata supports TND / PA_BBND with has_ori_kv=True")
    device = ori_topk_length.device
    if cmp_topk_length.device != device:
        raise ValueError("topk_length tensors must live on the same device")

    rows = ori_topk_length.shape[0]
    blocks = _get_cube_core_num(device)
    if not 0 < blocks <= AIC_CORE_MAX_NUM or rows > 2147483647:
        raise ValueError("rows and cube core count must fit positive int32 metadata")

    # No input tensor values are read on the host (see the baseline contract):
    # cu_seqlens_q only needs to be a well-formed [B+1] int32 span on the device.
    if cu_seqlens_q is None:
        cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    else:
        cu_q = cu_seqlens_q
        if (cu_q.dtype != torch.int32 or cu_q.dim() != 1 or cu_q.numel() < 2
                or cu_q.device != device):
            raise ValueError("cu_seqlens_q must be a 1-D int32 tensor on the same device")
        if not cu_q.is_contiguous():
            raise ValueError("cu_seqlens_q must be contiguous")

    batch = cu_q.numel() - 1
    compiled = _get_compiled_metadata()
    with torch.npu.device(device):
        metadata = torch.empty(MQSMLA_METADATA_TOTAL_SIZE, dtype=torch.int32, device=device)
        compiled.launch(current_raw_stream(device.index),
                        metadata=metadata.data_ptr(),
                        cu_q=cu_q.data_ptr(),
                        rows=rows, blocks=blocks, batch=batch)
    return metadata
