/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_fixpipe_opti.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_FIXPIPE_OPTI_H__
#define __OP_KERNEL_MATMUL_V3_FIXPIPE_OPTI_H__

#include "mat_mul_asw_block.h"
#include "mm_extension_interface/mm_custom_mm_policy.h"

namespace Mc2MatmulV3Advanced {
using namespace AscendC;
using namespace matmul;

#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulFixpipeOptiKernel {
public:
    __aicore__ inline Mc2MatmulFixpipeOptiKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void AicProcess(bool aicNeedWaitAiv);
    __aicore__ inline void AivProcess(uint64_t roundIdx);
    __aicore__ inline void Process();

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using C_TYPE_FIXPIPE_OPTI = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND_ALIGN, C_T>;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_FIXPIPE_OPTI, BIAS_TYPE, MM_CFG> mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulFixpipeOptiKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
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
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->tCubeTiling, pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulFixpipeOptiKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::AivProcess(uint64_t roundIdx)
{
    if ASCEND_IS_AIV {
        uint64_t subIdx = GetSubBlockIdx();
        if (subIdx == 1) {
            return;
        }
        LocalTensor<C_T> ubTensor = ubBuf_.template Get<C_T>();
        uint16_t vecM = static_cast<uint16_t>(block_.params_.singleCoreM);
        uint32_t blockLen = static_cast<uint32_t>(block_.params_.singleCoreN * sizeof(C_T));
        uint32_t dstStride =
            static_cast<uint32_t>((block_.matmulTilingData_->tCubeTiling.N - block_.params_.singleCoreN)) * sizeof(C_T);
        DataCopyExtParams copyParams{vecM, blockLen, 0, dstStride, 0};
        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
        DataCopyPad<C_T>(cGlobal_[block_.offset_.offsetC], ubTensor, copyParams);
        if (roundIdx != block_.params_.round - 1) {
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulFixpipeOptiKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::AicProcess(bool aicNeedWaitAiv)
{
    if ASCEND_IS_AIC {
        // fixpipe_l0c2ub
        mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                           block_.matmulTilingData_->tCubeTiling.singleCoreK);
        mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
        mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
        if (block_.matmulTilingData_->tCubeTiling.isBias) {
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
        }
        mm_.Iterate();
        if (aicNeedWaitAiv) {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        }
        LocalTensor<C_T> ubTensor = ubBuf_.template Get<C_T>();
        // localTensor, enAtomic, enSequentialWrite
        mm_.GetTensorC(ubTensor, 0UL, true);
        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulFixpipeOptiKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    if ASCEND_IS_AIV {
        // aiv初始等待AIC时可以load所有的.o
        ICachePreLoad(2);
    }
    bool aicNeedWaitAiv = false;
    SetAtomicNone();
    mm_.SetHF32(block_.matmulTilingData_->isHf32, 1);
    uint64_t curBlockIdx = GetCurrentBlockIdx();
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        uint64_t newBlockIdx =
            (j == block_.params_.round - 1) ? (curBlockIdx / block_.params_.totalSplitCnt) : curBlockIdx;
        block_.UpdateBasicIndex(j, newBlockIdx); // 更新Index
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.template UpdateBlockParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(j);
            if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
                aicNeedWaitAiv = j > 0;
                AicProcess(aicNeedWaitAiv);
                AivProcess(j);
            }
        }
    }
    mm_.SetHF32(false, 0);
    return;
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulFixpipeOptiDualDstKernel {
public:
    __aicore__ inline Mc2MatmulFixpipeOptiDualDstKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void AicProcess(bool aicNeedWaitAiv);
    __aicore__ inline void AivProcess(uint64_t roundIdx);
    __aicore__ inline void Process();

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using C_TYPE_FIXPIPE_OPTI = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND_ALIGN, C_T>;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE_FIXPIPE_OPTI, BIAS_TYPE, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
               MatmulCommon::Mc2MMCustomMatmulPolicy>
        mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulFixpipeOptiDualDstKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
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
    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.matmulTilingData_->tCubeTiling, pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulFixpipeOptiDualDstKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::AivProcess(uint64_t roundIdx)
{
    if ASCEND_IS_AIV {
        LocalTensor<C_T> ubTensor = ubBuf_.template Get<C_T>();
        uint16_t vecM = MMV3DivCeil(static_cast<uint16_t>(block_.params_.singleCoreM), NUM_TWO);
        uint16_t vecMReal = ((block_.params_.singleCoreM & 1UL) > 0) ? vecM - GetSubBlockIdx() : vecM;
        uint32_t blockLen = static_cast<uint32_t>(block_.params_.singleCoreN * sizeof(C_T));
        uint32_t dstStride =
            static_cast<uint32_t>((block_.matmulTilingData_->tCubeTiling.N - block_.params_.singleCoreN)) * sizeof(C_T);
        DataCopyExtParams copyParams{vecMReal, blockLen, 0, dstStride, 0};
        uint64_t offsetC = block_.offset_.offsetC + GetSubBlockIdx() * vecM * block_.matmulTilingData_->tCubeTiling.N;
        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
        if (vecMReal > 0) {
            DataCopyPad<C_T>(cGlobal_[offsetC], ubTensor, copyParams);
        }
        if (roundIdx != block_.params_.round - 1) {
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulFixpipeOptiDualDstKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::AicProcess(
    bool aicNeedWaitAiv)
{
    if ASCEND_IS_AIC {
        // fixpipe_l0c2ub
        mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                           block_.matmulTilingData_->tCubeTiling.singleCoreK);
        mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
        mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
        if (block_.matmulTilingData_->tCubeTiling.isBias) {
            mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
        }
        mm_.Iterate();
        if (aicNeedWaitAiv) {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + VEC0_FLAG_ID_OFFSET);
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + VEC1_FLAG_ID_OFFSET);
        }
        LocalTensor<C_T> ubTensor = ubBuf_.template Get<C_T>();
        // localTensor, enAtomic, enSequentialWrite
        mm_.GetTensorC(ubTensor, 0UL, true);
        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC0_FLAG_ID_OFFSET);
        CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + VEC1_FLAG_ID_OFFSET);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulFixpipeOptiDualDstKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    if ASCEND_IS_AIV {
        // aiv初始等待AIC时可以load所有的.o
        ICachePreLoad(2);
    }
    bool aicNeedWaitAiv = false;
    SetAtomicNone();
    mm_.SetHF32(block_.matmulTilingData_->isHf32, 1);
    uint64_t curBlockIdx = GetCurrentBlockIdx();
    for (uint64_t j = 0; j < block_.params_.round; j++) {
        uint64_t newBlockIdx =
            (j == block_.params_.round - 1) ? (curBlockIdx / block_.params_.totalSplitCnt) : curBlockIdx;
        block_.UpdateBasicIndex(j, newBlockIdx); // 更新Index
        if (block_.params_.index < block_.params_.totalCnt) {
            block_.template UpdateBlockParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(j);
            if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
                block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
                aicNeedWaitAiv = j > 0;
                AicProcess(aicNeedWaitAiv);
                AivProcess(j);
            }
        }
    }
    mm_.SetHF32(false, 0);
    return;
}

} // namespace Mc2MatmulV3Advanced

#endif
