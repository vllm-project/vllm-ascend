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
 * \file mc2_mat_mul_asw_kernel.h
 * \brief
 */

#ifndef NEW_MC2_MAT_MUL_ASW_KERNEL_H
#define NEW_MC2_MAT_MUL_ASW_KERNEL_H

#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#include "mc2_mat_mul_asw_block.h"

namespace MC2MatmulV3 {

using namespace AscendC;
using namespace matmul;
using namespace Mc2MatmulV3Advanced;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class MC2MatmulAswKernelDerive : public Mc2MatmulAswKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> {
public:
    __aicore__ inline MC2MatmulAswKernelDerive()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe, Mc2Tiling::RCSTiling cfg,
                                bool isTail, bool isGather);
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, bool isGather);
    __aicore__ inline void UpdateSlice(uint32_t idx, bool isTail);
    __aicore__ inline void Process(bool isLast = true, uint8_t enAtomic = 0);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MC2MatmulAswKernelDerive<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe, Mc2Tiling::RCSTiling cfg, bool isTail, bool isGather)
{
    this->pipe_ = pipe;
    this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    this->block_.InitForMC2(tilingData, cfg, isTail, isGather);
    InitInputs(aGM, bGM, cGM, biasGM, isGather);
    this->mm_.SetSubBlockIdx(0);
    this->mm_.Init(&this->block_.matmulTilingData_->tCubeTiling, this->pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void MC2MatmulAswKernelDerive<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, bool isGather)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    uint32_t adjustVal = isGather ? this->block_.cfg_.rankDim : 1;
    this->aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                                   static_cast<uint64_t>(this->block_.rankMK_ * adjustVal));
    this->bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), static_cast<uint64_t>(this->block_.rankKN_));
    this->cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                                   static_cast<uint64_t>(this->block_.rankMN_ * adjustVal));
    this->biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), this->block_.cfg_.rankN);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MC2MatmulAswKernelDerive<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateSlice(uint32_t idx, bool isTail)
{
    this->block_.UpdateOffset(idx, isTail);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
MC2MatmulAswKernelDerive<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(bool isLast, uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }

    SetAtomicNone();
    this->mm_.SetHF32(this->block_.matmulTilingData_->isHf32, 1);
    set_ctrl(sbitset1(get_ctrl(), 51)); // fixp使用n搬出，达到cube和fixp并行的效果
    this->block_.Update();
    for (uint64_t j = 0; j < this->block_.params_.round; j++) {
        uint64_t newBlockIdx = ((j == this->block_.params_.round - 1) && isLast) ?
                                   (block_idx / this->block_.params_.totalSplitCnt) :
                                   block_idx;
        if ((j == 0) && (newBlockIdx < this->block_.preCoreNum_)) {
            continue;
        }
        this->block_.UpdateBasicIndex(j, newBlockIdx); // update asw index
        if (this->block_.params_.index < this->block_.params_.totalCnt - this->block_.preCoreNum_) {
            this->block_.UpdateBlockParams(j, isLast);
            if ((this->block_.params_.singleCoreM > 0) && (this->block_.params_.singleCoreN > 0)) {
                this->block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();

                this->mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                                         this->block_.matmulTilingData_->tCubeTiling.singleCoreK);
                this->mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], A_TYPE::isTrans);
                this->mm_.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], B_TYPE::isTrans);
                if (this->block_.matmulTilingData_->tCubeTiling.isBias) {
                    this->mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
                }
                this->mm_.Iterate();
                this->mm_.GetTensorC(this->cGlobal_[this->block_.offset_.offsetC], enAtomic);
            }
        }
    }

    this->block_.preCoreNum_ =
        this->block_.params_.totalCnt % (this->block_.matmulTilingData_->tCubeTiling.usedCoreNum);
    set_ctrl(sbitset0(get_ctrl(), 51));
    this->mm_.SetHF32(false, 0);
}

} // namespace MC2MatmulV3

#endif // MC2_MAT_MUL_ASW_KERNEL_H
