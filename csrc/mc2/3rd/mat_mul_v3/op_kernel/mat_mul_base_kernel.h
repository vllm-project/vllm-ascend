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
 * \file mat_mul_base_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_BASE_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_BASE_KERNEL_H__


#include "mat_mul_base_block.h"

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class Mc2MatmulBaseKernel {
public:
    __aicore__ inline Mc2MatmulBaseKernel()
    {
    }

    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void SetOrgShape();

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);
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
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM);

    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    LocalTensor<uint8_t> buf = ubBuf_.template Get<uint8_t>();
    mm_.SetLocalWorkspace(buf);
#endif
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) *
                                 block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Kb) *
                                 block_.matmulTilingData_->matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.M) *
                                 block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), block_.matmulTilingData_->matmulTiling.N);
    SetL2CacheEnable(block_.matmulTilingData_->l2cacheUseInfo, aGlobal_, bGlobal_, cGlobal_, biasGlobal_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SetOrgShape()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN, block_.params_.alignedKaSize,
                        block_.params_.alignedKbSize, block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.matmulTilingData_->matmulTiling.N,
                        block_.params_.alignedKaSize, block_.matmulTilingData_->matmulTiling.Kb,
                        block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.matmulTilingData_->matmulTiling.M, block_.params_.alignedOriN,
                        block_.matmulTilingData_->matmulTiling.singleCoreK, block_.params_.alignedKbSize,
                        block_.matmulTilingData_->matmulTiling.N);
    }
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM);
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(uint64_t index,
                                                                                           uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    bool reverse = true;
    for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            block_.InitBlockIndex(index);
            for (uint64_t j = 0; j < block_.params_.realRound; j++) {
                if (block_.params_.rowOrder == 0) {
                    block_.UpdateBasicIndex(j); // 使能错位分核更新Index
                }
                if (block_.params_.index < block_.params_.totalTileCnt) {
                    block_.UpdateBlockParams(mTileIndex, nTileIndex);

                    block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);

                    mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                                       block_.matmulTilingData_->matmulTiling.singleCoreK);
                    mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], block_.params_.isTransposeA);
                    mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], block_.params_.isTransposeB);
                    if (block_.matmulTilingData_->matmulTiling.isBias) {
                        mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
                    }
                    mm_.Iterate();
                    mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID7);
#endif
                }
                block_.UpdateBlockIndex();
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}
#endif // MMV3_MATMUL_KERNEL_H
