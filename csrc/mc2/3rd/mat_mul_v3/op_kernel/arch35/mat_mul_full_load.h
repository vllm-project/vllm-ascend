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
 * \file mat_mul_full_load.h
 * \brief
 */
#ifndef MMV3_MATMUL_FULL_LOAD_H
#define MMV3_MATMUL_FULL_LOAD_H

#include "mat_mul_asw_block.h"
#include "mat_mul_v3_full_load_kernel_helper.h"

namespace Mc2MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulAswKernelAL1FullLoad {
public:
    __aicore__ inline Mc2MatmulAswKernelAL1FullLoad()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Process(uint8_t enAtomic = 0);

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using A_TYPE_NEW = MatmulL1GmType<AscendC::TPosition::TSCM, A_TYPE::format, A_T, A_TYPE::isTrans>;
    MatmulImpl<A_TYPE_NEW, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TQue<QuePosition::A1, 1> InQueueAL1_;
    LocalTensor<A_T> al1Local_;
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulAswKernelAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE_NEW, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
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
__aicore__ inline void
Mc2MatmulAswKernelAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    bool isMMultiCore = block_.matmulTilingData_->tCubeTiling.singleCoreM < block_.matmulTilingData_->tCubeTiling.M;
    if (unlikely(isMMultiCore)) {
        block_.UpdateBasicIndex(0, GetBlockIdx()); // 更新Index
    }
    AswAL1FullLoadKernelCopyInA1<A_TYPE_NEW, A_T, BLOCK_TYPE>(block_, block_.matmulTilingData_, isMMultiCore, aGlobal_,
                                                              InQueueAL1_, al1Local_);
    AswAL1FullLoadKernelMainLoop<A_TYPE_NEW, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>(
        mm_, block_, block_.matmulTilingData_, bGlobal_, cGlobal_, biasGlobal_, InQueueAL1_, al1Local_, enAtomic);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulAswKernelBL1FullLoad {
public:
    __aicore__ inline Mc2MatmulAswKernelBL1FullLoad()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Process(uint8_t enAtomic = 0);

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using B_TYPE_NEW = MatmulL1GmType<AscendC::TPosition::TSCM, B_TYPE::format, B_T, B_TYPE::isTrans>;
    MatmulImpl<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TQue<QuePosition::B1, 1> InQueueBL1_;
    LocalTensor<B_T> bl1Local_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulAswKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE>(tilingData);
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
__aicore__ inline void
Mc2MatmulAswKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    bool isNMultiCore = block_.matmulTilingData_->tCubeTiling.singleCoreN < block_.matmulTilingData_->tCubeTiling.N;
    if (unlikely(isNMultiCore)) {
        block_.UpdateBasicIndex(0, GetBlockIdx()); // 更新Index
    }
    AswBL1FullLoadKernelCopyInB1<B_TYPE_NEW, B_T, BLOCK_TYPE>(block_, block_.matmulTilingData_, isNMultiCore, bGlobal_,
                                                              InQueueBL1_, bl1Local_);
    AswBL1FullLoadKernelMainLoop<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>(
        mm_, block_, block_.matmulTilingData_, aGlobal_, cGlobal_, biasGlobal_, InQueueBL1_, bl1Local_, enAtomic);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulAswBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulAswKernelABL1FullLoad {
public:
    __aicore__ inline Mc2MatmulAswKernelABL1FullLoad()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Process(uint8_t enAtomic = 0);

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using C_T = typename C_TYPE::T;
    using A_TYPE_NEW = MatmulL1GmType<AscendC::TPosition::TSCM, A_TYPE::format, A_T, A_TYPE::isTrans>;
    using B_TYPE_NEW = MatmulL1GmType<AscendC::TPosition::TSCM, B_TYPE::format, B_T, B_TYPE::isTrans>;
    using BIAS_TYPE_NEW = MatmulType<AscendC::TPosition::TSCM, CubeFormat::ND, BiasT>;
    MatmulImpl<A_TYPE_NEW, B_TYPE_NEW, C_TYPE, BIAS_TYPE_NEW, MM_CFG> mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TQue<QuePosition::A1, 1> InQueueAL1_;
    TQue<QuePosition::B1, 1> InQueueBL1_;
    TQue<QuePosition::B1, 1> InQueueBiasL1_;
    LocalTensor<A_T> al1Local_;
    LocalTensor<B_T> bl1Local_;
    LocalTensor<BiasT> biasL1Local_;

private:
    __aicore__ inline void CopyInBias(const Mc2MatMulV3TilingData &matmulTilingData, bool isNMultiCore);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulAswKernelABL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    block_.template Init<A_TYPE_NEW, B_TYPE_NEW, C_TYPE, BIAS_TYPE_NEW>(tilingData);
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
__aicore__ inline void
Mc2MatmulAswKernelABL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    bool isMMultiCore = block_.matmulTilingData_->tCubeTiling.singleCoreM < block_.matmulTilingData_->tCubeTiling.M;
    bool isNMultiCore = block_.matmulTilingData_->tCubeTiling.singleCoreN < block_.matmulTilingData_->tCubeTiling.N;
    block_.UpdateBasicIndex(0, GetBlockIdx()); // 使能错位分核更新Index
    AswAL1FullLoadKernelCopyInA1<A_TYPE_NEW, A_T, BLOCK_TYPE>(block_, block_.matmulTilingData_, isMMultiCore, aGlobal_,
                                                              InQueueAL1_, al1Local_);
    AswBL1FullLoadKernelCopyInB1<B_TYPE_NEW, B_T, BLOCK_TYPE>(block_, block_.matmulTilingData_, isNMultiCore, bGlobal_,
                                                              InQueueBL1_, bl1Local_);
    if (block_.matmulTilingData_->tCubeTiling.isBias) {
        CopyInBias(*(block_.matmulTilingData_), isNMultiCore);
    }
    mm_.SetSubBlockIdx(0);
    mm_.SetHF32(block_.matmulTilingData_->isHf32, 1);
    mm_.Init(&block_.matmulTilingData_->tCubeTiling, pipe_);
    SetMMLayoutTransform(true); // fixp使用n搬出，达到cube和fixp并行的效果
    SetAtomicNone();
    if (block_.params_.index < block_.params_.totalCnt) {
        block_.template UpdateBlockParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(0);
        if (block_.params_.singleCoreM > 0 && block_.params_.singleCoreN > 0) {
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            mm_.SetTensorA(al1Local_, A_TYPE::isTrans);
            mm_.SetTensorB(bl1Local_, B_TYPE::isTrans);
            if (block_.matmulTilingData_->tCubeTiling.isBias) {
                mm_.SetBias(biasL1Local_);
            }
            mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                               block_.matmulTilingData_->tCubeTiling.singleCoreK);

            // MDL模板，L1输入场景默认bl1N=N，分核全载需要通过设置SetOrgShape指定al1N=singleCoreN
            mm_.SetOrgShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                            block_.matmulTilingData_->tCubeTiling.Ka);
            mm_.Iterate();
            // MDL模板，N为内轴，L0C->GM需重置shape
            mm_.SetOrgShape(block_.matmulTilingData_->tCubeTiling.M, block_.matmulTilingData_->tCubeTiling.N,
                            block_.matmulTilingData_->tCubeTiling.Ka);
            mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
        }
    }
    InQueueAL1_.FreeTensor(al1Local_);
    InQueueBL1_.FreeTensor(bl1Local_);
    if (block_.matmulTilingData_->tCubeTiling.isBias) {
        InQueueBiasL1_.FreeTensor(biasL1Local_);
    }
    mm_.SetHF32(false, 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulAswKernelABL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CopyInBias(
    const Mc2MatMulV3TilingData &matmulTilingData, bool isNMultiCore)
{
    uint64_t nAligned = MMV3CeilAlign(matmulTilingData.tCubeTiling.singleCoreN, block_.params_.nAlignSize);
    pipe_->InitBuffer(InQueueBiasL1_, 1, nAligned * sizeof(BiasT));
    biasL1Local_ = InQueueBiasL1_.AllocTensor<BiasT>();

    uint64_t nCntIdx = block_.params_.nCntIndex;
    uint64_t offsetBias = nCntIdx * matmulTilingData.tCubeTiling.singleCoreN;
    uint64_t instrBias = isNMultiCore && nCntIdx == block_.params_.nCnt - 1 ?
                             block_.params_.nBaseTail :
                             static_cast<uint64_t>(matmulTilingData.tCubeTiling.singleCoreN);
    uint64_t instrBiasAlign = MMV3CeilAlign(instrBias, static_cast<uint64_t>(BLOCK_SIZE));
    uint64_t instrBiasAlignSize = instrBiasAlign * sizeof(BiasT);
    DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(instrBiasAlignSize),
                                        static_cast<uint32_t>(instrBiasAlignSize),
                                        static_cast<uint32_t>(instrBiasAlignSize), 0};
    DataCopyPad(biasL1Local_, biasGlobal_[offsetBias], dataCopyExtParams, {false, 0, 0, 0});
    InQueueBiasL1_.EnQue(biasL1Local_);
    biasL1Local_ = InQueueBiasL1_.DeQue<BiasT>();
}

} // namespace Mc2MatmulV3Advanced

#endif // MMV3_MATMUL_FULL_LOAD_H
