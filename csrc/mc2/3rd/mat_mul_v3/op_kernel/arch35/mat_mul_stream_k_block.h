/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_stream_k_block.h
 * \brief
 */
#ifndef MMV3_MATMUL_STREAM_K_BLOCK_H
#define MMV3_MATMUL_STREAM_K_BLOCK_H

#include "mat_mul_asw_block.h"

namespace Mc2MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

constexpr uint64_t BLOCK_BASE_M = 256;
constexpr uint64_t BLOCK_BASE_N = 256;
constexpr uint64_t MM_FIX_PIPE_UNIT_FLAG = 3;

struct StreamKAivArgs {
    uint64_t offsetWorkspaceSrc = 0;
    uint64_t mBurstCnt = 0;
    uint64_t mBurstBase = 0;
    uint64_t mBurstTail = 0;

    uint64_t aivMte2Num = 0;

    uint64_t copyGm2UbKCnt = 0;
    uint64_t copyGm2UbMBurst = 0;
    uint64_t copyGm2UbMBurstOri = 0;
    uint64_t copyGm2UbBurstLen = 0;
    int64_t copyGm2UbSrcGap = 0;

    uint64_t offsetCGm = 0;
    uint64_t copyUb2GmMBurst = 0;
    uint64_t copyUb2GmBurstLen = 0;
    int64_t copyUb2GmSrcGap = 0;
    int64_t copyUb2GmDstGap = 0;
};

struct StreamKAicArgs {
    uint64_t kCntIndex = 0;
    uint64_t lastLoopTotalCnt = 0;
    uint64_t blockBaseK = 0;
    uint64_t kBaseTail = 0;
    uint64_t singleCoreK = 0;
    uint64_t offsetCWorkspace = 0;
    uint64_t alignSingleCoreN = 0;
};


class MatmulStreamKBlock : public Mc2MatmulAswBlock {
public:
    __aicore__ inline MatmulStreamKBlock()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);
    __aicore__ inline uint64_t UpdateLoopIndex(uint64_t roundIdx);
    template <const bool ALIGN_FLAG>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t roundIdx);
    __aicore__ inline void CalcAivBaseParams();
    __aicore__ inline void UpdateAivParams(uint64_t index, uint64_t roundIdx);

public:
    StreamKAivArgs aivParams_;
    StreamKAicArgs aicParams_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulStreamKBlock::Init(const void *tilingData)
{
    Mc2MatmulAswBlock::Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    // sk过程k方向切分块数
    if (params_.round >= 2) {
        aicParams_.blockBaseK =
            MMV3CeilAlign(matmulTilingData_->tCubeTiling.Ka / matmulTilingData_->kTailCnt, BLOCK_SIZE);
    } else {
        aicParams_.blockBaseK = matmulTilingData_->tCubeTiling.singleCoreK;
    }
    // 刷新总的使用切分块数
    aicParams_.lastLoopTotalCnt =
        params_.mCnt * params_.nCnt % matmulTilingData_->tCubeTiling.usedCoreNum * matmulTilingData_->kTailCnt;
    params_.totalCnt = (params_.round - 1) * matmulTilingData_->tCubeTiling.usedCoreNum + aicParams_.lastLoopTotalCnt;

    aicParams_.kBaseTail =
        matmulTilingData_->tCubeTiling.Ka - (matmulTilingData_->kTailCnt - 1) * aicParams_.blockBaseK;
    // SK场景配置为2，把单次串行的vec计算优化成pingpang2次计算，优化VEC和MTE3一半的耗时;
    // DPSK场景减小tilesize来降低带宽，避免抢占AIC带宽
    aivParams_.aivMte2Num = params_.round >= NUM_TWO ? BLOCK_SIZE : NUM_TWO;
}

__aicore__ inline void MatmulStreamKBlock::UpdateBasicIndex(uint64_t roundIdx)
{
    uint64_t newBlockIdx;
    if ASCEND_IS_AIC {
        newBlockIdx = (roundIdx == params_.round - 1) ? (GetBlockIdx() / matmulTilingData_->kTailCnt) : GetBlockIdx();
        aicParams_.kCntIndex = (roundIdx == params_.round - 1) ? GetBlockIdx() % matmulTilingData_->kTailCnt : 0;
    }
    if ASCEND_IS_AIV {
        newBlockIdx = GetBlockIdx() / (GetTaskRation() * matmulTilingData_->kTailCnt);
        aicParams_.kCntIndex = GetBlockIdx() % (GetTaskRation() * matmulTilingData_->kTailCnt);
    }
    Mc2MatmulAswBlock::UpdateBasicIndex(roundIdx, newBlockIdx);
}

__aicore__ inline uint64_t MatmulStreamKBlock::UpdateLoopIndex(uint64_t roundIdx)
{
    if (params_.round < NUM_TWO) {
        return roundIdx;
    } else {
        // DP+SK把最后尾块的部分提前到倒数第二轮
        if (roundIdx == params_.round - NUM_TWO) { // 交换倒数第一轮和倒数第二轮顺序
            return params_.round - 1;
        } else if (roundIdx == params_.round - 1) {
            return params_.round - NUM_TWO;
        } else {
            return roundIdx;
        }
    }
}

template <const bool ALIGN_FLAG>
__aicore__ inline void MatmulStreamKBlock::UpdateBlockParams(uint64_t roundIdx)
{
    if (params_.round < NUM_TWO) {
        params_.singleCoreM = params_.mCntIndex != (params_.mCnt - 1) ? params_.blockBaseM : params_.mBaseTail;
        params_.singleCoreN = params_.nCntIndex != (params_.nCnt - 1) ? params_.blockBaseN : params_.nBaseTail;
    } else {
        params_.singleCoreM = BLOCK_BASE_M;
        params_.singleCoreN = BLOCK_BASE_N;
    }
    if constexpr (ALIGN_FLAG) {
        aicParams_.alignSingleCoreN = MMV3CeilAlign(params_.singleCoreN, BLOCK_BYTE_SIZE);
    } else {
        aicParams_.alignSingleCoreN = params_.singleCoreN;
    }

    if ASCEND_IS_AIC {
        // DP使用原始k
        if (roundIdx != params_.round - 1) {
            aicParams_.singleCoreK = matmulTilingData_->tCubeTiling.Ka;
        } else {
            // sk使用切分k
            if (GetBlockIdx() >= aicParams_.lastLoopTotalCnt) {
                aicParams_.singleCoreK = 0;
            } else if (GetBlockIdx() % matmulTilingData_->kTailCnt == matmulTilingData_->kTailCnt - 1) {
                aicParams_.singleCoreK = aicParams_.kBaseTail;
            } else {
                aicParams_.singleCoreK = aicParams_.blockBaseK;
            }
        }
    }
}

template <class A_TYPE, class B_TYPE>
__aicore__ inline void MatmulStreamKBlock::CalcGMOffset(uint64_t roundIdx)
{
    if ASCEND_IS_AIC {
        if constexpr (A_TYPE::isTrans) {
            offset_.offsetA = params_.mCntIndex * params_.blockBaseM +
                              aicParams_.kCntIndex * aicParams_.blockBaseK * matmulTilingData_->tCubeTiling.M;
        } else {
            offset_.offsetA = params_.mCntIndex * params_.blockBaseM * matmulTilingData_->tCubeTiling.Ka +
                              aicParams_.kCntIndex * aicParams_.blockBaseK;
        }
        if constexpr (B_TYPE::format == CubeFormat::ND) {
            if constexpr (B_TYPE::isTrans) {
                offset_.offsetB = params_.nCntIndex * params_.blockBaseN * matmulTilingData_->tCubeTiling.Kb +
                                  aicParams_.kCntIndex * aicParams_.blockBaseK;
            } else {
                offset_.offsetB = params_.nCntIndex * params_.blockBaseN +
                                  aicParams_.kCntIndex * aicParams_.blockBaseK * matmulTilingData_->tCubeTiling.N;
            }
        } else {
            if constexpr (B_TYPE::isTrans) {
                offset_.offsetB = params_.nCntIndex * params_.blockBaseN * params_.kbAlignSize +
                                  aicParams_.kCntIndex * aicParams_.blockBaseK *
                                      MMV3CeilAlign(matmulTilingData_->tCubeTiling.N, params_.nAlignSize);
            } else {
                offset_.offsetB = params_.nCntIndex * params_.blockBaseN *
                                      MMV3CeilAlign(matmulTilingData_->tCubeTiling.Kb, params_.kbAlignSize) +
                                  aicParams_.kCntIndex * aicParams_.blockBaseK * params_.nAlignSize;
            }
        }
        offset_.offsetC = (params_.nCntIndex * params_.blockBaseN) +
                          (params_.mCntIndex * params_.blockBaseM) * matmulTilingData_->tCubeTiling.N;
        aicParams_.offsetCWorkspace =
            ((params_.index - roundIdx * matmulTilingData_->tCubeTiling.usedCoreNum) * matmulTilingData_->kTailCnt +
             aicParams_.kCntIndex) *
            BLOCK_BASE_M * BLOCK_BASE_N;
        if (matmulTilingData_->tCubeTiling.isBias) {
            offset_.offsetBias = params_.nCntIndex * params_.blockBaseN;
        }
    }
}

__aicore__ inline void MatmulStreamKBlock::CalcAivBaseParams()
{
    if ASCEND_IS_AIV {
        aivParams_.copyGm2UbKCnt = matmulTilingData_->kTailCnt;
        // 主轮一次搬得m行数，至少需要搬运32个FP32的数
        aivParams_.mBurstBase =
            MMV3CeilAlign(MMV3DivCeil(params_.singleCoreM, matmulTilingData_->kTailCnt * GetTaskRation()),
                          MMV3DivCeil(BLOCK_BYTE_SIZE, aicParams_.alignSingleCoreN));
        // 按照对齐后的行数，确认需要用到的aiv数量
        aivParams_.mBurstCnt = MMV3DivCeil(params_.singleCoreM, aivParams_.mBurstBase);
        // 确认最后一块的行数
        aivParams_.mBurstTail = params_.singleCoreM - (aivParams_.mBurstCnt - 1) * aivParams_.mBurstBase;
        // 实际每个aiv一次搬运的行数
        if (aicParams_.kCntIndex >= aivParams_.mBurstCnt) {
            aivParams_.copyGm2UbMBurstOri = 0;
        } else {
            aivParams_.copyGm2UbMBurstOri =
                (aicParams_.kCntIndex == aivParams_.mBurstCnt - 1) ? aivParams_.mBurstTail : aivParams_.mBurstBase;
        }
    }
}

__aicore__ inline void MatmulStreamKBlock::UpdateAivParams(uint64_t index, uint64_t roundIdx)
{
    aivParams_.copyGm2UbMBurst = MMV3DivCeil(aivParams_.copyGm2UbMBurstOri, aivParams_.aivMte2Num);

    // 计算workspace每个aiv的起始地址
    aivParams_.offsetWorkspaceSrc =
        (params_.index - roundIdx * matmulTilingData_->tCubeTiling.usedCoreNum) * matmulTilingData_->kTailCnt *
            BLOCK_BASE_M * BLOCK_BASE_N +
        (aicParams_.kCntIndex * aivParams_.mBurstBase + aivParams_.copyGm2UbMBurst * index) *
            aicParams_.alignSingleCoreN;
    // 计算aiv搬入到最终C矩阵的地址
    aivParams_.offsetCGm = params_.nCntIndex * params_.blockBaseN +
                           params_.mCntIndex * params_.blockBaseM * matmulTilingData_->tCubeTiling.N +
                           (aicParams_.kCntIndex * aivParams_.mBurstBase + aivParams_.copyGm2UbMBurst * index) *
                               matmulTilingData_->tCubeTiling.N;
    uint64_t singleCnt = MMV3DivCeil(aivParams_.copyGm2UbMBurstOri, aivParams_.copyGm2UbMBurst);
    if (params_.round == 1 &&
        // 纯sk场景一次搬运超过16K才使用pingpang优化，否则不使用pingpang优化
        aivParams_.copyGm2UbMBurstOri * aicParams_.alignSingleCoreN * matmulTilingData_->kTailCnt * DATA_SIZE_FP32 <
            BLOCK_SIZE * BANK_SIZE * DATA_SIZE_FP32) {
        singleCnt = 1;
    }
    if (index == singleCnt - 1) {
        aivParams_.copyGm2UbMBurst = aivParams_.copyGm2UbMBurstOri - (singleCnt - 1) * aivParams_.copyGm2UbMBurst;
    } else if (index >= singleCnt) {
        aivParams_.copyGm2UbMBurst = 0;
    }
    // vec一次搬运的数据量 对齐到32B
    aivParams_.copyGm2UbBurstLen = MMV3CeilAlign(aivParams_.copyGm2UbMBurst * aicParams_.alignSingleCoreN, BLOCK_SIZE);
    // vec 一次搬运2个burst之间的srcgap
    aivParams_.copyGm2UbSrcGap = BLOCK_BASE_M * BLOCK_BASE_N - aivParams_.copyGm2UbBurstLen;

    // vec ub -> gm 搬运的参数
    aivParams_.copyUb2GmMBurst = aivParams_.copyGm2UbMBurst;
    aivParams_.copyUb2GmBurstLen = params_.singleCoreN;
    aivParams_.copyUb2GmDstGap = matmulTilingData_->tCubeTiling.N - params_.singleCoreN;
    aivParams_.copyUb2GmSrcGap = aicParams_.alignSingleCoreN - params_.singleCoreN;
}
} // namespace Mc2MatmulV3Advanced

#endif // MMV3_MATMUL_STREAM_K_BLOCK_H
