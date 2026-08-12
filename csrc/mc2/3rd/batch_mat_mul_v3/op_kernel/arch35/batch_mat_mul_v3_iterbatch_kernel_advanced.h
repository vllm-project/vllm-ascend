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
 * \file batch_mat_mul_v3_iterbatch_kernel_advanced.h
 * \brief
 */
#ifndef BATCH_MAT_MUL_V3_ITERBATCH_KERNEL_ADVANCED_H
#define BATCH_MAT_MUL_V3_ITERBATCH_KERNEL_ADVANCED_H

#include "batch_mat_mul_v3_iterbatch_block_advanced.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

namespace Mc2BatchMatMulV3Advanced {
using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2BatchMatMulMultiBatchBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_MULTI_BATCH_OUT_BATCH_BIAS>
class Mc2BatchMatMulMultiBatchKernel {
public:
    __aicore__ inline Mc2BatchMatMulMultiBatchKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);
    __aicore__ inline void Process();
    __aicore__ inline void End()
    {
        mm_.End();
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

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2BatchMatMulMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    block_.Init(tilingData);

    if (GetCurrentBlockIdx() >= block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum) {
        return;
    }
    InitInputs(aGM, bGM, cGM, biasGM);
    SetAtomicNone();
    pipe_ = pipe;
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling, pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(GM_ADDR aGM,
                                                                                                  GM_ADDR bGM,
                                                                                                  GM_ADDR cGM,
                                                                                                  GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
                                 block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Ka *
                                 block_.batchMatmulTilingData_->aBatchDimAll);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.Kb) *
                                 block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N *
                                 block_.batchMatmulTilingData_->bBatchDimAll);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.M) *
                                 block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N *
                                 block_.batchMatmulTilingData_->cBatchDimAll);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.N *
                                    block_.batchMatmulTilingData_->biasBatchDimAll);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2BatchMatMulMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    if (GetCurrentBlockIdx() >= block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.usedCoreNum) {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    for (uint64_t loopIndex = 0; loopIndex < block_.params_.LoopTimes; loopIndex++) {
        block_.GetMultiBatchInfo(loopIndex);
        if (block_.params_.batchANum > 0) {
            block_.CalcGMOffset();

            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.batchMatmulTilingData_->matMulTilingData.tCubeTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.SetNBatchOutNum(block_.batchMatmulTilingData_->batchOutNum);
            mm_.SetBatchNum(block_.params_.batchANum, block_.params_.batchBNum);
            mm_.IterateBatch(cGlobal_[block_.offset_.offsetC], 0, 0, 0, block_.params_.singleASize,
                             block_.params_.singleBSize);
        }
    }
    mm_.SetHF32(false, 0);
}
} // namespace Mc2BatchMatMulV3Advanced
#endif // BATCH_MAT_MUL_V3_ITERBATCH_KERNEL_ADVANCED_H
