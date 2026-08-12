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
 * \file mat_mul_cvp_base_kernel.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_CVP_BASE_KERNEL__H__
#define __OP_KERNEL_MATMUL_V3_CVP_BASE_KERNEL__H__


#include "mat_mul_base_block.h"
#include "mat_mul_l1_full_load.h"
#include "mat_mul_nd2nz.h"

using namespace AscendC;
using namespace matmul;


namespace Mc2MatmulV3 {

__BLOCK_LOCAL__ __inline__ uint64_t procNum;

const uint32_t PP_GM = 2;
const uint16_t C_NOTIFY_V = 2;
const uint16_t V_NOTIFY_C = 4;

template <typename T>
__aicore__ inline void CopyAImpl(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM,
                                 int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2MatmulV3TilingData *tilingDataPtr = reinterpret_cast<Mc2MatmulV3TilingData *>(tilingPtr);
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    LocalTensor<T> dst = aMatrix.ReinterpretCast<T>();
    GlobalTensor<T> src;

    uint32_t baseM = tilingDataPtr->matmulTiling.baseM;
    uint32_t baseK = tilingDataPtr->matmulTiling.baseK;
    uint32_t singleCoreM = static_cast<uint32_t>(dataPtr >> 32);
    uint32_t pingpong_gm = static_cast<uint32_t>((dataPtr << 32) >> 32);
    uint32_t singleCoreK = tilingDataPtr->matmulTiling.singleCoreK;
    uint32_t n, d, n_aligned, d_aligned, useN, useD, useRow, useCol;
    if (tilingDataPtr->matmulRunInfo.transA) {
        n = singleCoreK;
        d = singleCoreM;
        useN = AlignUp(useK, ALIGNED_H);
        useD = AlignUp(useM, c0Size);
        useRow = col * baseK;
        useCol = row * baseM;
    } else {
        n = singleCoreM;
        d = singleCoreK;
        useN = AlignUp(useM, ALIGNED_H);
        useD = AlignUp(useK, c0Size);
        useRow = row * baseM;
        useCol = col * baseK;
    }
    n_aligned = AlignUp(n, ALIGNED_H);
    d_aligned = AlignUp(d, c0Size);
    src.SetGlobalBuffer((__gm__ T *)gm, n_aligned * d_aligned);
    uint32_t srcStride = n_aligned - useN;
    uint64_t srcOffset = (int64_t)useRow * (int64_t)c0Size + (int64_t)useCol * (int64_t)n_aligned;
    if (col == 0 && row == 0) {
        procNum = useM * useK;
        CrossCoreWaitFlag(V_NOTIFY_C + pingpong_gm);
    } else {
        procNum += useM * useK;
    }
    DataCopy(dst, src[srcOffset],
             {static_cast<uint16_t>(useD / c0Size), static_cast<uint16_t>(useN), static_cast<uint16_t>(srcStride), 0});
    if (procNum >= singleCoreM * singleCoreK) {
        CrossCoreSetFlag<0x2, PIPE_MTE2>(C_NOTIFY_V + pingpong_gm);
    }
    return;
}

template <typename T>
__aicore__ inline void CopyBImpl(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK,
                                 int useNn, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2MatmulV3TilingData *tilingDataPtr = reinterpret_cast<Mc2MatmulV3TilingData *>(tilingPtr);
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    LocalTensor<T> dst = bMatrix.ReinterpretCast<T>();
    GlobalTensor<T> src;

    uint32_t baseN = tilingDataPtr->matmulTiling.baseN;
    uint32_t baseK = tilingDataPtr->matmulTiling.baseK;
    uint32_t singleCoreN = static_cast<uint32_t>(dataPtr >> 32);
    uint32_t pingpong_gm = static_cast<uint32_t>((dataPtr << 32) >> 32);
    uint32_t singleCoreK = tilingDataPtr->matmulTiling.singleCoreK;
    uint32_t n, d, n_aligned, d_aligned, useN, useD, useRow, useCol;
    if (tilingDataPtr->matmulRunInfo.transB) {
        n = singleCoreN;
        d = singleCoreK;
        useN = AlignUp(useNn, ALIGNED_H);
        useD = AlignUp(useK, c0Size);
        useRow = col * baseN;
        useCol = row * baseK;
    } else {
        n = singleCoreK;
        d = singleCoreN;
        useN = AlignUp(useK, ALIGNED_H);
        useD = AlignUp(useNn, c0Size);
        useRow = row * baseK;
        useCol = col * baseN;
    }
    n_aligned = AlignUp(n, ALIGNED_H);
    d_aligned = AlignUp(d, c0Size);
    src.SetGlobalBuffer((__gm__ T *)gm, n_aligned * d_aligned);
    uint32_t srcStride = n_aligned - useN;
    uint64_t srcOffset = (int64_t)useRow * (int64_t)c0Size + (int64_t)useCol * (int64_t)n_aligned;
    if (col == 0 && row == 0) {
        procNum = useNn * useK;
        CrossCoreWaitFlag(V_NOTIFY_C + pingpong_gm);
    } else {
        procNum += useNn * useK;
    }
    DataCopy(dst, src[srcOffset],
             {static_cast<uint16_t>(useD / c0Size), static_cast<uint16_t>(useN), static_cast<uint16_t>(srcStride), 0});
    if (procNum >= singleCoreN * singleCoreK) {
        CrossCoreSetFlag<0x2, PIPE_MTE2>(C_NOTIFY_V + pingpong_gm);
    }
    return;
}

__aicore__ inline void CopyA(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM,
                             int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2MatmulV3TilingData *tilingDataPtr = reinterpret_cast<Mc2MatmulV3TilingData *>(tilingPtr);
    if (tilingDataPtr->matmulRunInfo.nd2nzA) {
        if (ctx.inputDtypeSize == 2) {
            CopyAImpl<half>(aMatrix, gm, row, col, useM, useK, tilingPtr, dataPtr);
        } else {
            CopyAImpl<float>(aMatrix, gm, row, col, useM, useK, tilingPtr, dataPtr);
        }
        return;
    }

    if (tilingDataPtr == nullptr || !ctx.isFirst) {
        return;
    }
    Mc2MatmulV3::DataCopyL1FullLoad(true, aMatrix, gm, useM, useK, 0UL, *tilingDataPtr);
    ctx.isFirst = false;
}

__aicore__ inline void CopyB(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK,
                             int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2MatmulV3TilingData *tilingDataPtr = reinterpret_cast<Mc2MatmulV3TilingData *>(tilingPtr);
    if (tilingDataPtr->matmulRunInfo.nd2nzB) {
        if (ctx.inputDtypeSize == 2) {
            CopyBImpl<half>(bMatrix, gm, row, col, useK, useN, tilingPtr, dataPtr);
        } else {
            CopyBImpl<float>(bMatrix, gm, row, col, useK, useN, tilingPtr, dataPtr);
        }
        return;
    }
    if (tilingDataPtr == nullptr || !ctx.isFirst) {
        return;
    }
    Mc2MatmulV3::DataCopyL1FullLoad(false, bMatrix, gm, 0UL, useK, useN, *tilingDataPtr);
    ctx.isFirst = false;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, CopyA, CopyB>>
class Mc2MatmulCvpBaseKernel
    : public Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB> {
public:
    __aicore__ inline Mc2MatmulCvpBaseKernel()
    {
    }

    __aicore__ inline void InitInputs(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM);

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);
    __aicore__ inline void AicProcess(uint32_t pingpong_gm, uint8_t enAtomic);
    __aicore__ inline void AivProcess(uint32_t pingpong_gm);

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    GlobalTensor<A_T> wsGlobal_;
    uint64_t c0Size;
    uint32_t aNz;
    uint64_t wsBufferSize;
    uint64_t oneBufferSize;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void Mc2MatmulCvpBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    this->pipe_ = pipe;
    aNz = this->block_.matmulTilingData_->matmulRunInfo.nd2nzA;
    GetSizeC0<A_T>(c0Size);

    this->mm_.SetSubBlockIdx(0);
    this->mm_.Init(&this->block_.matmulTilingData_->matmulTiling, this->pipe_);
    this->mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
    this->pipe_->InitBuffer(this->ubBuf_, TOTAL_UB_SIZE);
    this->SetOrgShape();

    InitInputs(aGM, bGM, cGM, biasGM, workspaceGM);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void Mc2MatmulCvpBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::InitInputs(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR workspaceGM)
{
    this->aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                                   static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.M) *
                                       this->block_.matmulTilingData_->matmulTiling.Ka);
    this->bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                                   static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.Kb) *
                                       this->block_.matmulTilingData_->matmulTiling.N);
    this->cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                                   static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.M) *
                                       this->block_.matmulTilingData_->matmulTiling.N);

    uint64_t singleCoreM = this->block_.matmulTilingData_->matmulTiling.singleCoreM;
    uint64_t singleCoreN = this->block_.matmulTilingData_->matmulTiling.singleCoreN;
    uint64_t singleCoreK = this->block_.matmulTilingData_->matmulTiling.singleCoreK;
    uint32_t usedCoreNum = this->block_.matmulTilingData_->matmulTiling.usedCoreNum;

    oneBufferSize = AlignUp((aNz ? singleCoreM : singleCoreN), ALIGNED_H) * AlignUp(singleCoreK, c0Size);
    wsBufferSize = oneBufferSize * PP_GM * usedCoreNum; // 开启double buffer
    wsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(workspaceGM) +
                                  GetCurrentBlockIdx() * oneBufferSize * PP_GM,
                              static_cast<uint64_t>(oneBufferSize * PP_GM));

    this->biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                      this->block_.matmulTilingData_->matmulTiling.N);
    SetL2CacheEnable(this->block_.matmulTilingData_->l2cacheUseInfo, this->aGlobal_, this->bGlobal_, this->cGlobal_,
                     this->biasGlobal_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulCvpBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(uint64_t index,
                                                                                              uint8_t enAtomic)
{
    this->mm_.SetHF32(false, 0);
    if (this->block_.params_.isHf32) {
        this->mm_.SetHF32(true, 1);
    }
    ctx.isFirst = true;
    ctx.inputDtypeSize = sizeof(A_T);
    bool reverse = true;
    uint32_t pingpong_gm = 0;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<0x2, PIPE_MTE2>(C_NOTIFY_V);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(C_NOTIFY_V + 1);
    }
    for (uint64_t mTileIndex = 0; mTileIndex < this->block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < this->block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (this->block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            this->block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            this->block_.InitBlockIndex(index);
            for (uint64_t j = 0; j < this->block_.params_.realRound; j++) {
                if (this->block_.params_.rowOrder == 0) {
                    this->block_.UpdateBasicIndex(j); // 使能错位分核更新Index
                }
                if (this->block_.params_.index < this->block_.params_.totalTileCnt) {
                    this->block_.UpdateBlockParams(mTileIndex, nTileIndex);
                    this->block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
                    AivProcess(pingpong_gm);
                    AicProcess(pingpong_gm, enAtomic);
                    pingpong_gm = (pingpong_gm + 1) % PP_GM;
                }
                this->block_.UpdateBlockIndex();
            }
        }
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(C_NOTIFY_V);
        CrossCoreWaitFlag(C_NOTIFY_V + 1);
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    this->mm_.SetHF32(false, 0);
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulCvpBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AicProcess(uint32_t pingpong_gm,
                                                                                                 uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    this->mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                             this->block_.matmulTilingData_->matmulTiling.singleCoreK);
    uint64_t singleCoreTile;
    if (aNz) {
        this->mm_.SetTensorA(wsGlobal_[pingpong_gm * oneBufferSize], this->block_.params_.isTransposeA);
        this->mm_.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], this->block_.params_.isTransposeB);
        singleCoreTile = this->block_.params_.singleCoreM;
    } else {
        this->mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], this->block_.params_.isTransposeA);
        this->mm_.SetTensorB(wsGlobal_[pingpong_gm * oneBufferSize], this->block_.params_.isTransposeB);
        singleCoreTile = this->block_.params_.singleCoreN;
    }
    this->mm_.SetSelfDefineData(static_cast<uint64_t>((singleCoreTile << 32) + pingpong_gm));

    if (this->block_.matmulTilingData_->matmulTiling.isBias) {
        this->mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
    }
    this->mm_.IterateAll(this->cGlobal_[this->block_.offset_.offsetC], enAtomic);
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulCvpBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AivProcess(uint32_t pingpong_gm)
{
    if ASCEND_IS_AIC {
        return;
    }
    GlobalTensor<A_T> curGlobal;
    uint64_t curOffset;
    uint32_t nCalcLen, dCalcLen;

    if (aNz) {
        curGlobal = this->aGlobal_;
        curOffset = this->block_.offset_.offsetA;
        nCalcLen = this->block_.params_.isTransposeA ? this->block_.matmulTilingData_->matmulTiling.singleCoreK :
                                                       this->block_.params_.singleCoreM;
        dCalcLen = this->block_.params_.isTransposeA ? this->block_.params_.singleCoreM :
                                                       this->block_.matmulTilingData_->matmulTiling.singleCoreK;
    } else {
        curGlobal = this->bGlobal_;
        curOffset = this->block_.offset_.offsetB;
        nCalcLen = this->block_.params_.isTransposeB ? this->block_.params_.singleCoreN :
                                                       this->block_.matmulTilingData_->matmulTiling.singleCoreK;
        dCalcLen = this->block_.params_.isTransposeB ? this->block_.matmulTilingData_->matmulTiling.singleCoreK :
                                                       this->block_.params_.singleCoreN;
    }
    CrossCoreWaitFlag(C_NOTIFY_V + pingpong_gm);
#if defined(__DAV_C220_VEC__)
    auto &&tempDst = wsGlobal_[pingpong_gm * oneBufferSize];
    auto &&tempSrc = curGlobal[curOffset];
    Nd2nzVnchwMM<A_T, Nd2NzMode::SINGLE_CORE>(tempDst, tempSrc, nCalcLen, dCalcLen, 1, this->ubBuf_, GetTaskRation());
#endif
    CrossCoreSetFlag<0x2, PIPE_MTE3>(V_NOTIFY_C + pingpong_gm);
    return;
}
} // namespace Mc2MatmulV3
#endif // MMV3_MATMUL_KERNEL_H
