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
 * \file mat_mul_l1_full_load.h
 * \brief
 */

#ifndef __OP_KERNEL_MATMUL_V3_BL1_FULL_LOAD_H__
#define __OP_KERNEL_MATMUL_V3_BL1_FULL_LOAD_H__

#include "mat_mul_base_kernel.h"

using namespace AscendC;
using namespace matmul;

namespace Mc2MatmulV3 {

struct CallBackContext {
    bool isFirst;
    uint64_t inputDtypeSize;
};

__BLOCK_LOCAL__ __inline__ CallBackContext ctx;

template <class T>
__aicore__ inline void DataCopy2L1(const LocalTensor<int8_t> &matrix, const __gm__ void *gm, uint64_t l1Size,
                                   uint64_t calCount)
{
    LocalTensor<T> dst = matrix.ReinterpretCast<T>();
    GlobalTensor<T> src;
    src.SetGlobalBuffer((__gm__ T *)gm, l1Size);
    DataCopy(dst, src, calCount);
}

template <class T>
__aicore__ inline void DataCopy2L1(const LocalTensor<int8_t> &matrix, const __gm__ void *gm,
                                   const Nd2NzParams &nd2nzParams, uint64_t l1Size)
{
    LocalTensor<T> dst = matrix.ReinterpretCast<T>();
    GlobalTensor<T> src;
    src.SetGlobalBuffer((__gm__ T *)gm, l1Size);
    DataCopy(dst, src, nd2nzParams);
}

__aicore__ inline void DataCopyL1FullLoad(bool isAl1FullLoad, const LocalTensor<int8_t> &matrix, const __gm__ void *gm,
                                          uint64_t useM, uint64_t useK, uint64_t useN,
                                          const Mc2MatmulV3TilingData &tilingData)
{
    const bool isFp32 = ctx.inputDtypeSize == sizeof(float);
    const uint64_t c0 = BLOCK_BYTE_SIZE / ctx.inputDtypeSize;
    const uint64_t l1Size = isAl1FullLoad ? (tilingData.matmulTiling.Ka * tilingData.matmulTiling.M) :
                                            (tilingData.matmulTiling.Kb * tilingData.matmulTiling.N);
    uint64_t nDim;
    uint64_t dDim;
    if (isAl1FullLoad) {
        nDim = tilingData.matmulRunInfo.transA ? useK : useM;
        dDim = tilingData.matmulRunInfo.transA ? useM : useK;
    } else {
        nDim = tilingData.matmulRunInfo.transB ? useN : useK;
        dDim = tilingData.matmulRunInfo.transB ? useK : useN;
    }
    if ((dDim == c0) && (nDim % BLOCK_SIZE == 0)) {
        // 走连续拷贝
        return isFp32 ? DataCopy2L1<float>(matrix, gm, l1Size, nDim * dDim) :
                        DataCopy2L1<half>(matrix, gm, l1Size, nDim * dDim);
    }
    // 走Nd2Nz
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = dDim;
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nDim, static_cast<uint64_t>(BLOCK_SIZE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    return isFp32 ? DataCopy2L1<float>(matrix, gm, nd2nzParams, l1Size) :
                    DataCopy2L1<half>(matrix, gm, nd2nzParams, l1Size);
}

__aicore__ inline void CopyAL1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM,
                               int useK, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2MatmulV3TilingData *tilingDataPtr = reinterpret_cast<Mc2MatmulV3TilingData *>(tilingPtr);
    if (tilingDataPtr == nullptr || !ctx.isFirst) {
        return;
    }

    DataCopyL1FullLoad(true, aMatrix, gm, useM, useK, 0UL, *tilingDataPtr);
    ctx.isFirst = false;
}

__aicore__ inline void CopyBL1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK,
                               int useN, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    Mc2MatmulV3TilingData *tilingDataPtr = reinterpret_cast<Mc2MatmulV3TilingData *>(tilingPtr);
    if (tilingDataPtr == nullptr || !ctx.isFirst) {
        return;
    }

    DataCopyL1FullLoad(false, bMatrix, gm, 0UL, useK, useN, *tilingDataPtr);
    ctx.isFirst = false;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_MDL>
class Mc2MatmulBaseKernelAL1FullLoad {
public:
    __aicore__ inline Mc2MatmulBaseKernelAL1FullLoad()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

protected:
    BLOCK_TYPE block_;
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using A_TYPE_NEW = MatmulType<AscendC::TPosition::TSCM, A_TYPE::format, A_T, A_TYPE::isTrans>;
    MatmulImpl<A_TYPE_NEW, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    TQue<QuePosition::A1, 1> InQueueAL1_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;

private:
    __aicore__ inline void CopyInA1(const Mc2MatmulV3TilingData &matmulTilingData, LocalTensor<A_T> al1Local);
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulBaseKernelAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
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
    mm_.SetSubBlockIdx(0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseKernelAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint64_t index,
                                                                                               uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint64_t innerAlignedBlock = BLOCK_BYTE_SIZE / sizeof(A_T);
    uint64_t mAligned =
        A_TYPE::isTrans ?
            MMV3CeilAlign(static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.singleCoreM),
                          innerAlignedBlock) :
            MMV3CeilAlign(static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.singleCoreM), BLOCK_SIZE);
    uint64_t kaAligned =
        A_TYPE::isTrans ?
            MMV3CeilAlign(static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Ka), BLOCK_SIZE) :
            MMV3CeilAlign(static_cast<uint64_t>(block_.matmulTilingData_->matmulTiling.Ka), innerAlignedBlock);
    pipe_->InitBuffer(InQueueAL1_, 1, mAligned * kaAligned * sizeof(A_T));
    LocalTensor<A_T> al1Local = InQueueAL1_.AllocTensor<A_T>();
    CopyInA1(*(block_.matmulTilingData_), al1Local);
    al1Local = InQueueAL1_.DeQue<A_T>();

    mm_.SetHF32(block_.params_.isHf32, 1);
    mm_.Init(&block_.matmulTilingData_->matmulTiling, pipe_);
    SetAtomicNone();
    block_.UpdateBlockCnt(0, 0);
    block_.InitBlockIndex(index);
    for (uint64_t j = 0; j < block_.params_.realRound; j++) {
        if (block_.params_.index < block_.params_.totalTileCnt) {
            block_.UpdateBlockParams(0, 0);
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(0, 0);
            mm_.SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN,
                               block_.matmulTilingData_->matmulTiling.singleCoreK);
            mm_.SetTensorA(al1Local, A_TYPE::isTrans);
            mm_.SetTensorB(bGlobal_[block_.offset_.offsetB], B_TYPE::isTrans);
            if (block_.matmulTilingData_->matmulTiling.isBias) {
                mm_.SetBias(biasGlobal_[block_.offset_.offsetBias]);
            }
            mm_.Iterate();
            mm_.GetTensorC(cGlobal_[block_.offset_.offsetC], enAtomic);
        }
        block_.UpdateBlockIndex();
    }
    mm_.SetHF32(false, 0);
    InQueueAL1_.FreeTensor(al1Local);
    PipeBarrier<PIPE_ALL>();
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulBaseKernelAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CopyInA1(
    const Mc2MatmulV3TilingData &matmulTilingData, LocalTensor<A_T> al1Local)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    int32_t shapeM = matmulTilingData.matmulTiling.M;
    int32_t shapeK = matmulTilingData.matmulTiling.Ka;
    uint64_t nDim = A_TYPE::isTrans ? static_cast<uint64_t>(shapeK) : static_cast<uint64_t>(shapeM);
    uint64_t dDim = A_TYPE::isTrans ? static_cast<uint64_t>(shapeM) : static_cast<uint64_t>(shapeK);
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = static_cast<uint64_t>(A_TYPE::isTrans ? shapeM : shapeK);
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nDim, static_cast<uint64_t>(BLOCK_SIZE));
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(al1Local, aGlobal_, nd2nzParams);
    InQueueAL1_.EnQue(al1Local);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD>
class Mc2MatmulBaseKernelBL1FullLoad
    : public Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> {
public:
    __aicore__ inline Mc2MatmulBaseKernelBL1FullLoad()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using B_TYPE_NEW = MatmulType<AscendC::TPosition::TSCM, B_TYPE::format, B_T, B_TYPE::isTrans>;
    MatmulImpl<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE, MM_CFG> mm_;
    TQue<QuePosition::B1, 1> InQueueBL1_;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    TPipe *pipe_;

private:
    __aicore__ inline void CopyInB1(const Mc2MatmulV3TilingData &matmulTilingData, LocalTensor<B_T> bl1Local);
    __aicore__ inline void SetOrgShapeFull();
    __aicore__ inline void ProcessRound(uint64_t mTileIndex, uint64_t nTileIndex, uint8_t enAtomic);
};


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    this->block_.template Init<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE>(tilingData);
    pipe_ = pipe;
    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM),
                             static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.M) *
                                 this->block_.matmulTilingData_->matmulTiling.Ka);
    bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM),
                             static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.Kb) *
                                 this->block_.matmulTilingData_->matmulTiling.N);
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM),
                             static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.M) *
                                 this->block_.matmulTilingData_->matmulTiling.N);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM),
                                this->block_.matmulTilingData_->matmulTiling.N);
    SetL2CacheEnable(this->block_.matmulTilingData_->l2cacheUseInfo, aGlobal_, bGlobal_, cGlobal_, biasGlobal_);
    mm_.SetSubBlockIdx(0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::SetOrgShapeFull()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE_NEW::format == CubeFormat::NZ) {
        mm_.SetOrgShape(this->block_.params_.alignedOriM, this->block_.matmulTilingData_->matmulTiling.singleCoreN,
                        this->block_.params_.alignedKaSize, this->block_.params_.alignedKbSize,
                        this->block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm_.SetOrgShape(this->block_.params_.alignedOriM, this->block_.matmulTilingData_->matmulTiling.singleCoreN,
                        this->block_.params_.alignedKaSize, this->block_.matmulTilingData_->matmulTiling.Kb,
                        this->block_.matmulTilingData_->matmulTiling.N);
    } else if constexpr (B_TYPE_NEW::format == CubeFormat::NZ) {
        mm_.SetOrgShape(this->block_.matmulTilingData_->matmulTiling.M,
                        this->block_.matmulTilingData_->matmulTiling.singleCoreN,
                        this->block_.matmulTilingData_->matmulTiling.singleCoreK, this->block_.params_.alignedKbSize,
                        this->block_.matmulTilingData_->matmulTiling.N);
    } else {
        mm_.SetOrgShape(this->block_.matmulTilingData_->matmulTiling.M, this->block_.params_.singleCoreN,
                        this->block_.matmulTilingData_->matmulTiling.Ka,
                        this->block_.matmulTilingData_->matmulTiling.Kb,
                        this->block_.matmulTilingData_->matmulTiling.N);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::ProcessRound(uint64_t mTileIndex,
                                                                                                    uint64_t nTileIndex,
                                                                                                    uint8_t enAtomic)
{
    for (uint64_t j = 0; j < this->block_.params_.realRound; j++) {
        if (this->block_.params_.index < this->block_.params_.totalTileCnt) {
            this->block_.UpdateBlockParams(mTileIndex, nTileIndex);
            this->block_.template CalcGMOffset<A_TYPE, B_TYPE_NEW, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
            LocalTensor<B_T> bl1Local = InQueueBL1_.AllocTensor<B_T>();
            CopyInB1(*(this->block_.matmulTilingData_), bl1Local);
            SetOrgShapeFull();
            this->mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                                     this->block_.matmulTilingData_->matmulTiling.singleCoreK);
            this->mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], this->block_.params_.isTransposeA);
            this->mm_.SetTensorB(bl1Local, this->block_.params_.isTransposeB);
            if (this->block_.matmulTilingData_->matmulTiling.isBias) {
                this->mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
            }
            this->mm_.IterateAll(this->cGlobal_[this->block_.offset_.offsetC], enAtomic);
            InQueueBL1_.FreeTensor(bl1Local);
        }
        this->block_.UpdateBlockIndex();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void
Mc2MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::Process(uint64_t index,
                                                                                               uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }

    uint64_t innerAlignedBlock = BLOCK_BYTE_SIZE / sizeof(B_T);
    uint64_t nAligned =
        B_TYPE_NEW::isTrans ?
            MMV3CeilAlign(static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.singleCoreN), BLOCK_SIZE) :
            MMV3CeilAlign(static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.singleCoreN),
                          innerAlignedBlock);
    uint64_t kbAligned =
        B_TYPE_NEW::isTrans ?
            MMV3CeilAlign(static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.Kb), innerAlignedBlock) :
            MMV3CeilAlign(static_cast<uint64_t>(this->block_.matmulTilingData_->matmulTiling.Kb), BLOCK_SIZE);
    pipe_->InitBuffer(InQueueBL1_, 1, nAligned * kbAligned * sizeof(B_T));
    this->mm_.SetHF32(false, 0);
    if (this->block_.params_.isHf32) {
        this->mm_.SetHF32(true, 1);
    }
    this->mm_.Init(&this->block_.matmulTilingData_->matmulTiling, this->pipe_);
    for (uint64_t mTileIndex = 0; mTileIndex < this->block_.params_.mTileCntL2; mTileIndex++) {
        for (uint64_t nTileIndex = 0; nTileIndex < this->block_.params_.nTileCntL2; nTileIndex++) {
            this->block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            this->block_.InitBlockIndex(index);
            ProcessRound(mTileIndex, nTileIndex, enAtomic);
        }
    }
    this->mm_.SetHF32(false, 0);
    PipeBarrier<PIPE_ALL>();
    return;
}

template <class B_TYPE>
__aicore__ inline void CalCopyBL1Nd2NzParams(const Mc2MatmulV3TilingData &matmulTilingData, Nd2NzParams &nd2nzParams,
                                             uint64_t instrN)
{
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = B_TYPE::isTrans ? instrN : static_cast<uint64_t>(matmulTilingData.matmulTiling.Kb);
    nd2nzParams.dValue = B_TYPE::isTrans ? static_cast<uint64_t>(matmulTilingData.matmulTiling.Kb) : instrN;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue =
        static_cast<uint64_t>(B_TYPE::isTrans ? matmulTilingData.matmulTiling.Kb : matmulTilingData.matmulTiling.N);
    nd2nzParams.dstNzC0Stride = MMV3CeilAlign(nd2nzParams.nValue, BLOCK_SIZE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
}

template <class B_TYPE, class B_T, class BLOCK_TYPE>
__aicore__ inline void CalCopyBL1Nz2NzParams(const BLOCK_TYPE &block, const Mc2MatmulV3TilingData &matmulTilingData,
                                             bool isNMultiCore, DataCopyParams &dataCopyParams, uint64_t instrN)
{
    if (B_TYPE::isTrans && isNMultiCore) {
        dataCopyParams.blockCount = MMV3DivCeil(matmulTilingData.matmulTiling.Kb, block.params_.kbAlignSize);
        dataCopyParams.blockLen =
            block.params_.kbAlignSize * MMV3CeilAlign(instrN, block.params_.nAlignSize) * sizeof(B_T) / BLOCK_BYTE_SIZE;
        dataCopyParams.srcStride = block.params_.kbAlignSize *
                                   (MMV3CeilAlign(matmulTilingData.matmulTiling.N, block.params_.nAlignSize) -
                                    MMV3CeilAlign(instrN, block.params_.nAlignSize)) *
                                   sizeof(B_T) / BLOCK_BYTE_SIZE;
        dataCopyParams.dstStride = 0;
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = MMV3CeilAlign(matmulTilingData.matmulTiling.Kb, block.params_.kbAlignSize) *
                                  MMV3CeilAlign(instrN, block.params_.nAlignSize) * sizeof(B_T) / BLOCK_BYTE_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG>
__aicore__ inline void Mc2MatmulBaseKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG>::CopyInB1(
    const Mc2MatmulV3TilingData &matmulTilingData, LocalTensor<B_T> bl1Local)
{
    bool isNMultiCore = matmulTilingData.matmulTiling.singleCoreN < matmulTilingData.matmulTiling.N;
    uint64_t instrN = this->block_.params_.singleCoreN;
    if constexpr (B_TYPE_NEW::format == CubeFormat::ND) {
        Nd2NzParams nd2nzParams;
        CalCopyBL1Nd2NzParams<B_TYPE_NEW>(matmulTilingData, nd2nzParams, instrN);
        DataCopy(bl1Local, bGlobal_[this->block_.offset_.offsetB], nd2nzParams);
    } else {
        DataCopyParams dataCopyParams;
        CalCopyBL1Nz2NzParams<B_TYPE_NEW, B_T, BLOCK_TYPE>(this->block_, matmulTilingData, isNMultiCore, dataCopyParams,
                                                           instrN);
        DataCopy(bl1Local, bGlobal_[this->block_.offset_.offsetB], dataCopyParams);
    }
    InQueueBL1_.EnQue(bl1Local);
    bl1Local = InQueueBL1_.DeQue<B_T>();
}

// Current Kernel support only nd2nzA. No need to do nd2nz for B.
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
class Mc2MatmulBaseUnAlignedKernelBL1FullLoad
    : public Mc2MatmulBaseUnAlignedKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> {
public:
    __aicore__ inline Mc2MatmulBaseUnAlignedKernelBL1FullLoad()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

protected:
    using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
    Mc2MatmulBaseKernelBL1FullLoad<aType, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG> mma_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulBaseUnAlignedKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    this->pipe_ = pipe;
    this->pipe_->InitBuffer(this->ubBuf_, TOTAL_UB_SIZE);
    this->matmulTilingData_ = matmulTilingData;
    this->innerParams_.isTransposeA = this->matmulTilingData_->matmulRunInfo.transA;
    this->innerParams_.isTransposeB = this->matmulTilingData_->matmulRunInfo.transB;
    bool nd2nzA = this->matmulTilingData_->matmulRunInfo.nd2nzA;
    this->innerParams_.nd2nzFlag = nd2nzA ? static_cast<int32_t>(ND2NZ_SELECT::ONLY_A) : 0;
    this->innerParams_.baseAN = matmulTilingData->baseAN;
    this->innerParams_.baseAD = matmulTilingData->baseAD;
    this->CalculateabGM(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM);
    mma_.Init(this->innerParams_.workspaceGMNZ, bGM, cGM, biasGM, offsetWGM, workspaceGM, this->matmulTilingData_,
              this->pipe_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB>
__aicore__ inline void
Mc2MatmulBaseUnAlignedKernelBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(
    uint64_t index, uint8_t enAtomic)
{
    this->ProcessNDtoNZ();
    mma_.Process(index, enAtomic);
}

} // namespace Mc2MatmulV3
#endif // MMV3_MATMUL_BL1_FULL_LOAD_H
