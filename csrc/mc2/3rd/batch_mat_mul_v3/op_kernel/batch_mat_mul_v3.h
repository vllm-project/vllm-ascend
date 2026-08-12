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
 * \file batch_mat_mul_v3.h
 * \brief
 */
#ifndef BATCH_MATMUL_V3_H
#define BATCH_MATMUL_V3_H

#include "batch_mat_mul_v3_block.h"
#include "batch_mat_mul_v3_com_base_block.h"
#include "../../mat_mul_v3/op_kernel/mat_mul_unaligned_base_kernel.h"
#include "../../mat_mul_v3/op_kernel/mat_mul_l1_full_load.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

const uint64_t CV_SYNC_FLAG = 4UL;
const uint64_t AIC_SYNC_FLAG = 6UL;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2BatchMatMulMultiBatchBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_ORDER_M>
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
    template <class A_TP, class B_TP, class C_TP, class BIAS_TP, class BLOCK_TP, const MatmulConfig &MM_CFG_VAL>
    friend class Mc2BatchMatMulUnalignedMultiBatchKernel;

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

    if (GetBlockIdx() >= block_.batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum) {
        return;
    }
    InitInputs(aGM, bGM, cGM, biasGM);
    SetAtomicNone();
    pipe_ = pipe;
    mm_.SetSubBlockIdx(0);
    if constexpr (ToMatmulConfig(MM_CFG).bmmOutMode != BatchOutMode::SINGLE_BATCH) {
        uint64_t batchOutNum =
            TOTAL_L0C_SIZE / (block_.batchMatmulTilingData_->matmulTiling.matmulTiling.baseM *
                              block_.batchMatmulTilingData_->matmulTiling.matmulTiling.baseN *
                              block_.batchMatmulTilingData_->matmulTiling.matmulTiling.dbL0C * sizeof(float));
        mm_.SetNBatchOutNum(batchOutNum);
    }

    mm_.Init(&block_.batchMatmulTilingData_->matmulTiling.matmulTiling, pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2BatchMatMulUnalignedKernel {
public:
    __aicore__ inline Mc2BatchMatMulUnalignedKernel()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2BatchMatmulTilingData *matmulTilingData, TPipe *pipe);

    __aicore__ inline void Process();

protected:
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR offsetWGM_;
    GM_ADDR workspaceGM_;
    Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD> mm_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    TPipe *pipe_;
    const Mc2BatchMatmulTilingData *tilingPtr_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2BatchMatMulUnalignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    const AscendC::tiling::TCubeTiling &tiling = tilingPtr_->matmulTiling.matmulTiling;
    const uint32_t &batchA1 = tilingPtr_->Mc2multiBatchInfo.aBatchDim0;
    const uint32_t &batchA2 = tilingPtr_->Mc2multiBatchInfo.aBatchDim1;
    const uint32_t &batchA3 = tilingPtr_->Mc2multiBatchInfo.aBatchDim2;
    const uint32_t &batchA4 = tilingPtr_->Mc2multiBatchInfo.aBatchDim3;
    const uint32_t &batchB1 = tilingPtr_->Mc2multiBatchInfo.bBatchDim0;
    const uint32_t &batchB2 = tilingPtr_->Mc2multiBatchInfo.bBatchDim1;
    const uint32_t &batchB3 = tilingPtr_->Mc2multiBatchInfo.bBatchDim2;
    const uint32_t &batchB4 = tilingPtr_->Mc2multiBatchInfo.bBatchDim3;
    const uint32_t &batchC1 = tilingPtr_->Mc2multiBatchInfo.cBatchDim0;
    const uint32_t &batchC2 = tilingPtr_->Mc2multiBatchInfo.cBatchDim1;
    const uint32_t &batchC3 = tilingPtr_->Mc2multiBatchInfo.cBatchDim2;
    const uint32_t &batchC4 = tilingPtr_->Mc2multiBatchInfo.cBatchDim3;

    uint64_t offsetA = static_cast<uint64_t>(tiling.M) * tiling.Ka * sizeof(A_T);
    uint64_t offsetB = static_cast<uint64_t>(tiling.Kb) * tiling.N * sizeof(B_T);
    uint64_t offsetC = static_cast<uint64_t>(tiling.M) * tiling.N * sizeof(C_T);
    uint64_t batchBiasOffset = 0;
    if (tilingPtr_->Mc2multiBatchInfo.biasWithBatch) {
        batchBiasOffset = static_cast<uint64_t>(tiling.N) * sizeof(BiasT);
    }
    bool isBatchA1One = (batchA1 == 1);
    bool isBatchB1One = (batchB1 == 1);
    bool isBatchA2One = (batchA2 == 1);
    bool isBatchB2One = (batchB2 == 1);
    bool isBatchA3One = (batchA3 == 1);
    bool isBatchB3One = (batchB3 == 1);
    bool isBatchA4One = (batchA4 == 1);
    bool isBatchB4One = (batchB4 == 1);
    for (uint64_t i1 = 0; i1 < batchC1; i1++) {
        uint64_t iA1 = isBatchA1One ? 0 : (i1 * batchA2 * batchA3 * batchA4);
        uint64_t iB1 = isBatchB1One ? 0 : (i1 * batchB2 * batchB3 * batchB4);
        uint64_t iC1 = i1 * batchC2 * batchC3 * batchC4;
        for (uint64_t i2 = 0; i2 < batchC2; i2++) {
            uint64_t iA2 = isBatchA2One ? 0 : (i2 * batchA3 * batchA4);
            uint64_t iB2 = isBatchB2One ? 0 : (i2 * batchB3 * batchB4);
            uint64_t iC2 = i2 * batchC3 * batchC4;
            for (uint64_t i3 = 0; i3 < batchC3; i3++) {
                uint64_t iA3 = isBatchA3One ? 0 : (i3 * batchA4);
                uint64_t iB3 = isBatchB3One ? 0 : (i3 * batchB4);
                uint64_t iC3 = i3 * batchC4;
                for (uint64_t i4 = 0; i4 < batchC4; i4++) {
                    uint64_t iA4 = isBatchA4One ? 0 : i4;
                    uint64_t iB4 = isBatchB4One ? 0 : i4;
                    uint64_t iA = iA1 + iA2 + iA3 + iA4;
                    uint64_t iB = iB1 + iB2 + iB3 + iB4;
                    uint64_t iC = iC1 + iC2 + iC3 + i4;
                    mm_.UpdateGlobalTensor(aGM_ + offsetA * iA, bGM_ + offsetB * iB, cGM_ + offsetC * iC,
                                           biasGM_ + batchBiasOffset * iC, offsetWGM_, workspaceGM_);
                    mm_.Process();
                    if ASCEND_IS_AIC {
                        NotifyEvent<PIPE_FIX>(4);
                    }
                    if ASCEND_IS_AIV {
                        WaitEvent(4);
                        SyncAll();
                        PipeBarrier<PIPE_ALL>();
                    }
                }
            }
        }
    }
#endif
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2BatchMatMulUnalignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2BatchMatmulTilingData *matmulTilingDataPtr, TPipe *pipePtr)
{
    pipe_ = pipePtr;
    tilingPtr_ = matmulTilingDataPtr;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    offsetWGM_ = offsetWGM;
    workspaceGM_ = workspaceGM;
    mm_.Init(aGM_, bGM_, cGM_, biasGM_, offsetWGM_, workspaceGM_, &(tilingPtr_->matmulTiling), pipe_);
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
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka *
                                 block_.batchMatmulTilingData_->Mc2multiBatchInfo.aBatchDimAll);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Kb) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N *
                                 block_.batchMatmulTilingData_->Mc2multiBatchInfo.bBatchDimAll);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N *
                                 block_.batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll);

    if (block_.batchMatmulTilingData_->Mc2multiBatchInfo.biasWithBatch) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                    static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N) *
                                        block_.batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2BatchMatMulMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    if (GetBlockIdx() >= block_.batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum) {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    for (uint64_t loopIndex = 0; loopIndex < block_.params_.loopTimes; loopIndex++) {
        block_.GetMultiBatchInfo(loopIndex);
        if (block_.params_.batchNum > 0) {
            block_.CalcGMOffset();

            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.batchMatmulTilingData_->Mc2multiBatchInfo.biasWithBatch) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }


            mm_.SetBatchNum(block_.params_.batchNum, block_.params_.batchNum);
            mm_.IterateBatch(cGlobal_[block_.offset_.offsetC], 0, 0, 0, block_.params_.singleASize,
                             block_.params_.singleBSize);
        }
    }
    mm_.SetHF32(false, 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2BatchMatMulCommonBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class Mc2BatchMatMulCommonKernel {
public:
    __aicore__ inline Mc2BatchMatMulCommonKernel()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);

    __aicore__ inline void Process();

    __aicore__ inline void InnerProcess(uint64_t mTileIndex, uint64_t nTileIndex);

    __aicore__ inline void End()
    {
        mm_.End();
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

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void SetOrgShape();
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void Mc2BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    // cube core cases, ignore vector core
    if ASCEND_IS_AIV {
        return;
    }
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);

    if (GetBlockIdx() >= block_.batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum) {
        return;
    }
    SetAtomicNone();
    pipe_ = pipe;
    InitInputs(aGM, bGM, cGM, biasGM);

    mm_.SetSubBlockIdx(0);
    mm_.Init(&block_.batchMatmulTilingData_->matmulTiling.matmulTiling, pipe_);
    mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    LocalTensor<uint8_t> buf = ubBuf_.template Get<uint8_t>();
    mm_.SetLocalWorkspace(buf);
#endif
    SetOrgShape();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitInputs(GM_ADDR aGM,
                                                                                                     GM_ADDR bGM,
                                                                                                     GM_ADDR cGM,
                                                                                                     GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             block_.params_.aBatchDimAll *
                                 static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             block_.params_.bBatchDimAll *
                                 static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Kb) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             block_.params_.cBatchDimAll *
                                 static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);


    if (block_.batchMatmulTilingData_->matmulTiling.matmulTiling.isBias) {
        auto biasSize = block_.params_.biasWithBatch ?
                            block_.params_.cBatchDimAll *
                                static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N) :
                            block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N;
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), biasSize);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::SetOrgShape()
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    // ND2NZ后需要设置对齐后的shape信息
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.params_.alignedOriN, block_.params_.alignedKaSize,
                        block_.params_.alignedKbSize, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.params_.alignedOriM, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N,
                        block_.params_.alignedKaSize, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Kb,
                        block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M, block_.params_.alignedOriN,
                        block_.batchMatmulTilingData_->matmulTiling.matmulTiling.singleCoreK,
                        block_.params_.alignedKbSize, block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    }
#endif
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    InitInputs(aGM, bGM, cGM, biasGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InnerProcess(
    uint64_t mTileIndex, uint64_t nTileIndex)
{
    for (uint64_t j = 0; j < block_.params_.realRound; ++j) {
        if (block_.params_.rowOrder == 0) {
            block_.UpdateBasicIndex(j, mTileIndex, nTileIndex); // 使能错位分核更新Index
        }
        if (block_.params_.index < block_.params_.totalTileCnt) {
            block_.UpdateBlockParams(mTileIndex, nTileIndex);
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
            mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                               block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka);
            mm_.SetTensorA(aGlobal_[block_.offset_.offsetA], A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.batchMatmulTilingData_->matmulTiling.matmulTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.IterateAll(cGlobal_[block_.offset_.offsetC]);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
            auto eventMTE3MTE2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
#endif
        }
        // 行/列优先走这
        block_.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2BatchMatMulCommonKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process()
{
    // cube core cases, ignore vector core
    if ASCEND_IS_AIV {
        return;
    }
    if (GetBlockIdx() >= block_.batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum) {
        return;
    }

    Mc2MatmulV3::ctx.isFirst = true;
    Mc2MatmulV3::ctx.inputDtypeSize = sizeof(typename A_TYPE::T);
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
            block_.InitBlockIndex();
            InnerProcess(mTileIndex, nTileIndex);
        }
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm_.SetHF32(false, 0);
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2BatchMatMulUnalignedMultiBatchBaseBlock, const MatmulConfig &MM_CFG = MM_CFG_ORDER_M>
class Mc2BatchMatMulUnalignedMultiBatchKernel {
    struct Mc2UnAlignedKernelParams {
        bool isTransposeA;
        bool isTransposeB;
        ND2NZ_SELECT nd2nzFlag; // 2表示B矩阵做nd2nz，1表示A矩阵做nd2nz
        int32_t batchA;
        GM_ADDR aGMND;
        GM_ADDR bGMND;
        GM_ADDR workspaceGMNZ;
        GM_ADDR workspaceGMabNZ;
        bool alignedA;
        bool alignedB;
        uint64_t batchL2Cnt;
        uint64_t nd2nzBatchNum;
        uint64_t aGMOffset;
        uint64_t bGMOffset;
        uint64_t workspaceGMOffset;
        uint64_t workspaceGMbOffset;
    };

public:
    __aicore__ inline Mc2BatchMatMulUnalignedMultiBatchKernel(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2BatchMatmulTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void CalculateabGM(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                         GM_ADDR workspaceGM, uint64_t c0Size);
    __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                              GM_ADDR workspaceGM);
    __aicore__ inline void UpdateabGM(uint64_t loopIndex);
    __aicore__ inline void CalculateAlign(uint64_t c0Size);
    __aicore__ inline void ProcessNDtoNZ();
    __aicore__ inline void Process();
    __aicore__ inline void End();

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans, LayoutMode::NORMAL>;
    using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans, LayoutMode::NORMAL>;
    Mc2BatchMatMulMultiBatchKernel<aType, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mma_;
    Mc2BatchMatMulMultiBatchKernel<A_TYPE, bType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmb_;
    Mc2BatchMatMulMultiBatchKernel<aType, bType, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mmab_;

    Mc2UnAlignedKernelParams innerParams_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    TPipe *pipe_;
    TBuf<> ubBuf_;
    const Mc2BatchMatmulTilingData *tilingPtr_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2BatchMatmulTilingData *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(ubBuf_, TOTAL_UB_SIZE);
    tilingPtr_ = tilingData;
    innerParams_.isTransposeA = tilingPtr_->matmulTiling.matmulRunInfo.transA;
    innerParams_.isTransposeB = tilingPtr_->matmulTiling.matmulRunInfo.transB;
    bool nd2nzA = tilingPtr_->matmulTiling.matmulRunInfo.nd2nzA;
    bool nd2nzB = tilingPtr_->matmulTiling.matmulRunInfo.nd2nzB;

    innerParams_.alignedA = true;
    innerParams_.alignedB = true;

    if (nd2nzA) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_A;
        innerParams_.alignedA = false;
    }
    if (nd2nzB) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::ONLY_B;
        innerParams_.alignedB = false;
    }
    if (nd2nzA && nd2nzB) {
        innerParams_.nd2nzFlag = ND2NZ_SELECT::BOTH_AB;
    }

    uint64_t c0Size;
    GetSizeC0<A_T>(c0Size);

    CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM, c0Size);

    innerParams_.batchL2Cnt =
        DivCeil(tilingPtr_->Mc2multiBatchInfo.cBatchDimAll, tilingPtr_->Mc2multiBatchInfo.batchTileBlock);
    innerParams_.nd2nzBatchNum = tilingPtr_->Mc2multiBatchInfo.batchTileBlock;

    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        mmb_.block_.SetC0(c0Size);
        mmb_.block_.SetNd2nzFlag(static_cast<uint64_t>(innerParams_.nd2nzFlag));
        mmb_.Init(aGM, innerParams_.workspaceGMNZ, cGM, biasGM, offsetWGM, workspaceGM, tilingPtr_, pipe_);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        mma_.block_.SetC0(c0Size);
        mma_.block_.SetNd2nzFlag(static_cast<uint64_t>(innerParams_.nd2nzFlag));
        mma_.Init(innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM, tilingPtr_, pipe_);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        mmab_.block_.SetC0(c0Size);
        mmab_.block_.SetNd2nzFlag(static_cast<uint64_t>(innerParams_.nd2nzFlag));
        mmab_.Init(innerParams_.workspaceGMNZ, innerParams_.workspaceGMabNZ, cGM, biasGM, offsetWGM, workspaceGM,
                   tilingPtr_, pipe_);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CalculateabGM(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM, uint64_t c0Size)
{
    innerParams_.aGMND = aGM;
    innerParams_.bGMND = bGM;

    CalculateAlign(c0Size);

    innerParams_.workspaceGMNZ = workspaceGM;
    innerParams_.workspaceGMabNZ =
        tilingPtr_->Mc2multiBatchInfo.cBatchDimAll == tilingPtr_->Mc2multiBatchInfo.batchTileBlock ?
            workspaceGM + innerParams_.workspaceGMOffset :    // 左右矩阵都要做nd2nz但不超L2
            workspaceGM + innerParams_.workspaceGMOffset * 2; // 偏移两个左矩阵batchNum
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        if (innerParams_.alignedA) {
            innerParams_.workspaceGMNZ = aGM;
        }
    }
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        if (innerParams_.alignedB) {
            innerParams_.workspaceGMNZ = bGM;
        }
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        if (innerParams_.alignedB) {
            innerParams_.workspaceGMabNZ = bGM;
        }
        if (innerParams_.alignedA) {
            innerParams_.workspaceGMNZ = aGM;
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CalculateAlign(
    uint64_t c0Size)
{
    uint64_t innerA =
        innerParams_.isTransposeA ? tilingPtr_->matmulTiling.matmulTiling.M : tilingPtr_->matmulTiling.matmulTiling.Ka;
    uint64_t outterA =
        innerParams_.isTransposeA ? tilingPtr_->matmulTiling.matmulTiling.Ka : tilingPtr_->matmulTiling.matmulTiling.M;
    uint64_t innerB =
        innerParams_.isTransposeB ? tilingPtr_->matmulTiling.matmulTiling.Kb : tilingPtr_->matmulTiling.matmulTiling.N;
    uint64_t outterB =
        innerParams_.isTransposeB ? tilingPtr_->matmulTiling.matmulTiling.N : tilingPtr_->matmulTiling.matmulTiling.Kb;

    // 表示左右矩阵按照nz格式读取
    if (innerA == c0Size && outterA % ALIGNED_H == 0) {
        innerParams_.alignedA = true;
    }
    if (innerB == c0Size && outterB % ALIGNED_H == 0) {
        innerParams_.alignedB = true;
    }

    uint64_t alignedMSize =
        innerParams_.isTransposeA ?
            MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.M, c0Size) * c0Size :
            MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.M, ALIGNED_H) * ALIGNED_H; // M轴转换成分型
    uint64_t alignedKaSize =
        innerParams_.isTransposeA ?
            MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.Ka, ALIGNED_H) * ALIGNED_H :
            MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.Ka, c0Size) * c0Size; // K轴转换成分型
    uint64_t alignedKbSize = innerParams_.isTransposeB ?
                                 MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.Kb, c0Size) * c0Size :
                                 MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    uint64_t alignedNSize = innerParams_.isTransposeB ?
                                MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.N, ALIGNED_H) * ALIGNED_H :
                                MMV3DivCeil(tilingPtr_->matmulTiling.matmulTiling.N, c0Size) * c0Size;
    innerParams_.bGMOffset = static_cast<uint64_t>(tilingPtr_->Mc2multiBatchInfo.batchTileBlock) * sizeof(B_T) *
                             tilingPtr_->matmulTiling.matmulTiling.Kb * tilingPtr_->matmulTiling.matmulTiling.N;
    innerParams_.aGMOffset = static_cast<uint64_t>(tilingPtr_->Mc2multiBatchInfo.batchTileBlock) * sizeof(A_T) *
                             tilingPtr_->matmulTiling.matmulTiling.M * tilingPtr_->matmulTiling.matmulTiling.Ka;
    innerParams_.workspaceGMOffset = static_cast<uint64_t>(tilingPtr_->Mc2multiBatchInfo.batchTileBlock) * sizeof(A_T) *
                                     alignedMSize * alignedKaSize;
    innerParams_.workspaceGMbOffset = static_cast<uint64_t>(tilingPtr_->Mc2multiBatchInfo.batchTileBlock) *
                                      sizeof(B_T) * alignedKbSize * alignedNSize;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateGlobalTensor(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM)
{
    CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        mmb_.UpdateGlobalTensor(aGM, innerParams_.workspaceGMNZ, cGM, biasGM, offsetWGM, workspaceGM);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        mma_.UpdateGlobalTensor(innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        mmab_.UpdateGlobalTensor(innerParams_.workspaceGMNZ, innerParams_.workspaceGMabNZ, cGM, biasGM, offsetWGM,
                                 workspaceGM);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::UpdateabGM(
    uint64_t loopIndex)
{
    if (loopIndex == 0) {
        return;
    }

    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B && !innerParams_.alignedB) {
        innerParams_.bGMND += innerParams_.bGMOffset;
        if (loopIndex % 2 == 1) {
            innerParams_.workspaceGMNZ += innerParams_.workspaceGMbOffset;
        } else {
            innerParams_.workspaceGMNZ -= innerParams_.workspaceGMbOffset;
        }
    }
    if ((innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A || innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) &&
        !innerParams_.alignedA) {
        innerParams_.aGMND += innerParams_.aGMOffset;
        if (loopIndex % 2 == 1) {
            innerParams_.workspaceGMNZ += innerParams_.workspaceGMOffset;
        } else {
            innerParams_.workspaceGMNZ -= innerParams_.workspaceGMOffset;
        }
    }
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB && !innerParams_.alignedB) {
        innerParams_.bGMND += innerParams_.bGMOffset;
        if (loopIndex % 2 == 1) {
            innerParams_.workspaceGMabNZ += innerParams_.workspaceGMbOffset;
        } else {
            innerParams_.workspaceGMabNZ -= innerParams_.workspaceGMbOffset;
        }
    }
    // 外提batch轴尾块
    if (loopIndex + 1 >= innerParams_.batchL2Cnt &&
        tilingPtr_->Mc2multiBatchInfo.cBatchDimAll % tilingPtr_->Mc2multiBatchInfo.batchTileBlock != 0) {
        innerParams_.nd2nzBatchNum =
            tilingPtr_->Mc2multiBatchInfo.cBatchDimAll % tilingPtr_->Mc2multiBatchInfo.batchTileBlock;
        if ASCEND_IS_AIV {
            return;
        }
        if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
            mmb_.block_.CalculateLoopTimes(innerParams_.nd2nzBatchNum);
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
            mma_.block_.CalculateLoopTimes(innerParams_.nd2nzBatchNum);
        } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
            mmab_.block_.CalculateLoopTimes(innerParams_.nd2nzBatchNum);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessNDtoNZ()
{
    // ND2NZ
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        MatrixBtoNZV2<B_T>(innerParams_.workspaceGMNZ, innerParams_.bGMND, tilingPtr_->matmulTiling.matmulTiling,
                           innerParams_.isTransposeB, ubBuf_, 0, 0, innerParams_.nd2nzBatchNum);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        MatrixAtoNZV2<A_T>(innerParams_.workspaceGMNZ, innerParams_.aGMND, tilingPtr_->matmulTiling.matmulTiling,
                           innerParams_.isTransposeA, ubBuf_, 0, 0, innerParams_.nd2nzBatchNum);
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        if (!innerParams_.alignedA) {
            MatrixAtoNZV2<A_T>(innerParams_.workspaceGMNZ, innerParams_.aGMND, tilingPtr_->matmulTiling.matmulTiling,
                               innerParams_.isTransposeA, ubBuf_, 0, 0, innerParams_.nd2nzBatchNum);
        }
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        if (!innerParams_.alignedB) {
            MatrixBtoNZV2<B_T>(innerParams_.workspaceGMabNZ, innerParams_.bGMND, tilingPtr_->matmulTiling.matmulTiling,
                               innerParams_.isTransposeB, ubBuf_, 0, 0, innerParams_.nd2nzBatchNum);
        }
    }
    SyncAll();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process()
{
    int32_t pingpongId = 0;
    for (uint64_t loopIndex = 0; loopIndex < innerParams_.batchL2Cnt; loopIndex++) {
        pingpongId = loopIndex & 0x1;
        UpdateabGM(loopIndex);
        // aiv前两轮循环可以连着做
        if ASCEND_IS_AIV {
            if (!innerParams_.alignedA || !innerParams_.alignedB) {
                if (loopIndex > 1) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ >= 220
                    CrossCoreWaitFlag(CV_SYNC_FLAG + pingpongId);
#endif
                }
                ProcessNDtoNZ();
                CrossCoreSetFlag<0x2, PIPE_MTE3>(CV_SYNC_FLAG + pingpongId);
            }
        }
        if ASCEND_IS_AIC {
            if (!innerParams_.alignedA || !innerParams_.alignedB) {
                CrossCoreWaitFlag(CV_SYNC_FLAG + pingpongId);
            }
            if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
                mmb_.block_.CalculateBatchOffset(loopIndex, innerParams_.alignedA, innerParams_.alignedB);
                mmb_.Process();
            } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
                mma_.block_.CalculateBatchOffset(loopIndex, innerParams_.alignedA, innerParams_.alignedB);
                mma_.Process();
            } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
                mmab_.block_.CalculateBatchOffset(loopIndex, innerParams_.alignedA, innerParams_.alignedB);
                mmab_.Process();
            }
            // aic最后两轮循环可以连着做
            if (!innerParams_.alignedA || !innerParams_.alignedB) {
                if (loopIndex < innerParams_.batchL2Cnt - 2) {
                    CrossCoreSetFlag<0x0, PIPE_MTE2>(AIC_SYNC_FLAG + pingpongId);
                    CrossCoreWaitFlag(AIC_SYNC_FLAG + pingpongId);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ >= 220
                    CrossCoreSetFlag<0x2, PIPE_FIX>(CV_SYNC_FLAG + pingpongId);
#endif
                }
            }
        }
    }
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2BatchMatMulUnalignedMultiBatchKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::End()
{
    if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_B) {
        mmb_.End();
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::ONLY_A) {
        mma_.End();
    } else if (innerParams_.nd2nzFlag == ND2NZ_SELECT::BOTH_AB) {
        mmab_.End();
    }
}

__aicore__ inline void BmmCopyAL1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM,
                                  int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2BatchMatmulTilingData *tilingDataPtr = reinterpret_cast<Mc2BatchMatmulTilingData *>(tilingPtr);
    if (tilingDataPtr == nullptr || !Mc2MatmulV3::ctx.isFirst) {
        return;
    }
    Mc2MatmulV3::DataCopyL1FullLoad(true, aMatrix, gm, static_cast<uint64_t>(useM), static_cast<uint64_t>(useK), 0UL,
                                    tilingDataPtr->matmulTiling);
    Mc2MatmulV3::ctx.isFirst = false;
}

__aicore__ inline void BmmCopyBL1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK,
                                  int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2BatchMatmulTilingData *tilingDataPtr = reinterpret_cast<Mc2BatchMatmulTilingData *>(tilingPtr);
    if (tilingDataPtr == nullptr || !Mc2MatmulV3::ctx.isFirst) {
        return;
    }

    Mc2MatmulV3::DataCopyL1FullLoad(false, bMatrix, gm, 0UL, static_cast<uint64_t>(useK), static_cast<uint64_t>(useN),
                                    tilingDataPtr->matmulTiling);
    Mc2MatmulV3::ctx.isFirst = false;
}

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          class BLOCK_TYPE = Mc2BatchMatMulMultiBatchFullLoadBlock>
class Mc2BatchMatMulMultiBatchFullLoadKernel {
public:
    __aicore__ inline Mc2BatchMatMulMultiBatchFullLoadKernel()
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
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using A_TYPE_NEW = MatmulL1GmType<AscendC::TPosition::TSCM, A_TYPE::format, A_T, A_TYPE::isTrans, LayoutMode::NONE>;
    static constexpr MatmulConfig CFG_MDL = GetMDLConfig(false, false, 0, false, false, false, true);
    MatmulImpl<A_TYPE_NEW, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;
    TQue<QuePosition::A1, 1> InQueueAL1_;
    uint64_t c0Size;

private:
    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM);
    __aicore__ inline void MultiBatchCopyInL1(LocalTensor<A_T> al1Local, GlobalTensor<A_T> aGlobal, uint64_t nDim,
                                              uint64_t dDim);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE>
__aicore__ inline void Mc2BatchMatMulMultiBatchFullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    GetSizeC0<A_T>(c0Size);
    block_.SetC0(c0Size);

    block_.Init(tilingData);

    if (GetBlockIdx() >= block_.batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum) {
        return;
    }

    InitInputs(aGM, bGM, cGM, biasGM);
    SetAtomicNone();
    pipe_ = pipe;
    mm_.SetSubBlockIdx(0);

    uint64_t nAligned = MMV3CeilAlign(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M, ALIGNED_H);
    uint64_t dAligned = MMV3CeilAlign(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka, c0Size);
    pipe_->InitBuffer(InQueueAL1_, 1,
                      block_.params_.mainLoopPreCoreBatchNum * block_.params_.singleASizeL1 * sizeof(A_T));
    mm_.Init(&block_.batchMatmulTilingData_->matmulTiling.matmulTiling, pipe_);
    SetMMLayoutTransform(true);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE>
__aicore__ inline void
Mc2BatchMatMulMultiBatchFullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE>::InitInputs(GM_ADDR aGM,
                                                                                                  GM_ADDR bGM,
                                                                                                  GM_ADDR cGM,
                                                                                                  GM_ADDR biasGM)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka *
                                 block_.batchMatmulTilingData_->Mc2multiBatchInfo.aBatchDimAll);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Kb) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N *
                                 block_.batchMatmulTilingData_->Mc2multiBatchInfo.bBatchDimAll);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                                 block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N *
                                 block_.batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll);

    if (block_.batchMatmulTilingData_->Mc2multiBatchInfo.biasWithBatch) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                    static_cast<uint64_t>(block_.batchMatmulTilingData_->matmulTiling.matmulTiling.N) *
                                        block_.batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE>
__aicore__ inline void
Mc2BatchMatMulMultiBatchFullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE>::MultiBatchCopyInL1(
    LocalTensor<A_T> al1Local, GlobalTensor<A_T> aGlobal, uint64_t nDim, uint64_t dDim)
{
    uint64_t nDimAligned = MMV3CeilAlign(nDim, ALIGNED_H);
    uint64_t dDimAligned = MMV3CeilAlign(dDim, c0Size);

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = block_.params_.batchNum;
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = block_.params_.singleASize;
    nd2nzParams.srcDValue = dDim;
    nd2nzParams.dstNzC0Stride = nDimAligned;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = nDimAligned * dDimAligned;

    DataCopy(al1Local, aGlobal, nd2nzParams);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE>
__aicore__ inline void Mc2BatchMatMulMultiBatchFullLoadKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    SetAtomicNone();
    if (GetBlockIdx() >= block_.batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum) {
        return;
    }
    mm_.SetHF32(false, 0);
    if (block_.params_.isHf32) {
        mm_.SetHF32(true, 1);
    }
    for (uint64_t loopIndex = 0; loopIndex < block_.params_.loopTimes; loopIndex++) {
        block_.GetMultiBatchInfo(loopIndex);
        if (block_.params_.batchNum > 0) {
            block_.CalcGMOffset();

            int32_t shapeM = block_.batchMatmulTilingData_->matmulTiling.matmulTiling.M;
            int32_t shapeK = block_.batchMatmulTilingData_->matmulTiling.matmulTiling.Ka;
            uint64_t nDim = static_cast<uint64_t>(shapeM);
            uint64_t dDim = static_cast<uint64_t>(shapeK);
            if constexpr (A_TYPE::isTrans) {
                nDim = static_cast<uint64_t>(shapeK);
                dDim = static_cast<uint64_t>(shapeM);
            }
            uint64_t nDimAligned = MMV3CeilAlign(nDim, ALIGNED_H);
            uint64_t dDimAligned = MMV3CeilAlign(dDim, c0Size);

            LocalTensor<A_T> aLocal_ = InQueueAL1_.AllocTensor<A_T>();
            MultiBatchCopyInL1(aLocal_, aGlobal_[block_.offset_.offsetA], nDim, dDim);
            InQueueAL1_.EnQue(aLocal_);
            aLocal_ = InQueueAL1_.DeQue<A_T>();

            uint64_t innerLoopIndex = block_.params_.batchNum / block_.params_.mainLoopPreCoreBatchNumB;
            for (uint64_t innerIndex = 0; innerIndex < innerLoopIndex; ++innerIndex) {
                mm_.SetTensorA(
                    aLocal_[block_.params_.mainLoopPreCoreBatchNumB * innerIndex * nDimAligned * dDimAligned],
                    A_TYPE::isTrans);
                mm_.SetTensorB(bGlobal_[block_.offset_.offsetB + block_.params_.singleBSize * innerIndex *
                                                                     block_.params_.mainLoopPreCoreBatchNumB],
                               B_TYPE::isTrans);
                if (block_.batchMatmulTilingData_->Mc2multiBatchInfo.biasWithBatch) {
                    mm_.SetBias(biasGlobal_[block_.offset_.offsetBias + block_.params_.singleBiasSize * innerIndex *
                                                                            block_.params_.mainLoopPreCoreBatchNumB]);
                }
                mm_.IterateAll(cGlobal_[block_.offset_.offsetC + block_.params_.singleCSize * innerIndex *
                                                                     block_.params_.mainLoopPreCoreBatchNumB]);
            }
            InQueueAL1_.FreeTensor(aLocal_);
        }
    }
    mm_.SetHF32(false, 0);
}

#endif // BATCH_MATMUL_V3_H
