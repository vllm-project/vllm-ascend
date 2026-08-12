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
 * \file mat_mul_asw_kernel.h
 * \brief
 */
#ifndef MMV3_MATMUL_ASW_KERNEL_H
#define MMV3_MATMUL_ASW_KERNEL_H

#include "mat_mul_asw_block.h"
#include "mat_mul_dasw_block.h"

namespace Mc2MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulAswKernel {
public:
    __aicore__ inline Mc2MatmulAswKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);
    __aicore__ inline void UpdateBias(uint64_t kIndex);
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void Process(uint8_t enAtomic = 0);
    __aicore__ inline void End()
    {
        mm_.End();
    }
    __aicore__ inline const BLOCK_TYPE GetBlock()
    {
        return block_;
    }

protected:
    BLOCK_TYPE block_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    InitInputs(aGM, bGM, cGM, biasGM);
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->tCubeTiling, pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(GM_ADDR aGM, GM_ADDR bGM,
                                                                                      GM_ADDR cGM, GM_ADDR biasGM)
{
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->tCubeTiling.M) *
                                 block_.matmulTilingData_->tCubeTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->tCubeTiling.Kb) *
                                 block_.matmulTilingData_->tCubeTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->tCubeTiling.M) *
                                 block_.matmulTilingData_->tCubeTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block_.matmulTilingData_->tCubeTiling.N);
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateBias(uint64_t kIndex)
{
    if (block_.matmulTilingData_->tCubeTiling.isBias) {
        if (kIndex == block_.params_.splitKRound - 1) {
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
        } else {
            mm_.ClearBias();
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }

    SetAtomicNone();
    mm_.SetHF32(block_.matmulTilingData_->isHf32, 1);
    SetMMLayoutTransform(true); // fixp使用n搬出，达到cube和fixp并行的效果
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        uint64_t newBlockIdx = block_.GetNewBlockIdx(j);
        block_.UpdateBasicIndex(j, newBlockIdx); // update asw index
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.template UpdateBlockParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(j);
            if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
                for (uint64_t kIndex = 0; kIndex < block_.params_.splitKRound; kIndex++) {
                    uint64_t singleShapeK = kIndex == block_.params_.splitKRound - 1 ? block_.params_.singleShapeKTail :
                                                                                       block_.params_.singleCoreSplitK;
                    mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, singleShapeK);
                    block_.template CalcSplitKGMOffset<A_TYPE, B_TYPE>(kIndex);
                    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
                    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
                    UpdateBias(kIndex);
                    mm_.Iterate();
                    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic || kIndex != 0);
                }
            }
        }
    }
    SetMMLayoutTransform(false);
    mm_.SetHF32(false, 0);
}

} // namespace Mc2MatmulV3Advanced

#endif // MMV3_MATMUL_ASW_KERNEL_H
