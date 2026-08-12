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
 * \file mat_mul_optimized_fixpipe_algorithm.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_OPTIMIZED_FIXPIPE_ALGORITHM_H__
#define __OP_KERNEL_MATMUL_V3_OPTIMIZED_FIXPIPE_ALGORITHM_H__

#include "mat_mul_base_block.h"
#include "mat_mul_base_kernel.h"
#include "mat_mul_l1_full_load.h"

using namespace AscendC;
using namespace matmul;
using namespace Mc2MatmulV3;

#if defined(__CCE_KT_TEST__)
using namespace std;
#endif
// 512 byte
const uint32_t MM_ALIGN_SIZE = 512;
const uint8_t AIV_DB_SYNC_FLAG = 0x2;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
          FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE_ENABLE_ALIGNOUT>
class Mc2MatmulBaseUnalignedNKernel
    : public Mc2MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB> {
public:
    using C_T = typename C_TYPE::T;
    __aicore__ inline Mc2MatmulBaseUnalignedNKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
    __aicore__ inline void AicProcess(GlobalTensor<C_T> &cTensor, uint8_t enAtomic, bool aicNeedWaitAiv,
                                      uint8_t pingPongId);
    __aicore__ inline void AivProcess(GlobalTensor<C_T> &cTensor, uint8_t pingPongId);
    __aicore__ inline void Process(uint64_t index = 0UL, uint8_t enAtomic = 0UL);
    __aicore__ inline void AivNz2NdProcess(GlobalTensor<C_T> cNzGlobal, uint8_t pingPongId);
    __aicore__ inline void CopyInWithNz(LocalTensor<C_T> tensorNZ, GlobalTensor<C_T> cNzGlobal, uint64_t ubProcessM,
                                        uint8_t pingPongId);
    __aicore__ inline void Nz2NdAndGatherMask(LocalTensor<C_T> tensorND, LocalTensor<C_T> tensorNZ, uint64_t ubProcessM,
                                              uint8_t pingPongId);
    __aicore__ inline void UpdateOffsetC(bool isNd);

protected:
    GlobalTensor<C_T> tempCGlobal_;
    GatherMaskParams params_;
    uint64_t baseSize_ = 0UL;
    uint64_t alignedN_ = 0UL;
    uint64_t c0Size_ = BLOCK_SIZE;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const void *tilingData, TPipe *pipe)
{
    GetSizeC0<C_T>(c0Size_);
    uint64_t cDtypeSize = sizeof(C_T);
    this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    this->InitInputs(aGM, bGM, cGM, biasGM);

    this->pipe_ = pipe;
    this->pipe_->InitBuffer(this->ubBuf_, TOTAL_UB_SIZE);

    int64_t originShapeM = 0;
    if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
        alignedN_ = AlignUp(this->block_.matmulTilingData_->matmulTiling.N, BLOCK_SIZE);
        // for MatMul highlevel API copyout NZ format consecutively
        originShapeM = this->block_.matmulTilingData_->matmulTiling.baseM;
    } else {
        alignedN_ = AlignUp(this->block_.matmulTilingData_->matmulTiling.N, MM_ALIGN_SIZE / cDtypeSize);
        originShapeM = this->block_.matmulTilingData_->matmulTiling.M;
    }
    baseSize_ = alignedN_ * this->block_.matmulTilingData_->matmulTiling.baseM;
    params_.src0BlockStride = 1;
    params_.src0RepeatStride = alignedN_ / c0Size_;
    params_.src1RepeatStride = 0;

    tempCGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(workspaceGM),
                                 baseSize_ * NUM_TWO * this->block_.matmulTilingData_->matmulTiling.usedCoreNum);
    this->mm_.SetSubBlockIdx(0);
    this->mm_.Init(&this->block_.matmulTilingData_->matmulTiling, pipe);
    this->mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
    this->mm_.SetOrgShape(originShapeM, this->block_.params_.alignedOriN,
                          this->block_.matmulTilingData_->matmulTiling.singleCoreK, this->block_.params_.alignedKbSize,
                          this->alignedN_);
    if (this->block_.params_.isHf32) {
        this->mm_.SetHF32(true, 1);
    } else {
        this->mm_.SetHF32(false, 0);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::AivProcess(
    GlobalTensor<C_T> &cTensor, uint8_t pingPongId)
{
    if ASCEND_IS_AIV {
        uint32_t vBlockIndex = GetBlockIdx();
        if (vBlockIndex >= (this->block_.matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        CrossCoreWaitFlag(0x4 + pingPongId);

        uint64_t cDtypeSize = sizeof(C_T);
        LocalTensor<C_T> ubTensor = this->ubBuf_.template Get<C_T>();
        // aic : aiv is 1 : 2, singlecore cal half of baseM.
        uint64_t vecM = min(MMV3CeilAlign(this->block_.params_.singleCoreM / NUM_TWO, c0Size_),
                            static_cast<uint64_t>(this->block_.params_.singleCoreM));
        uint64_t subIdx = GetSubBlockIdx();
        uint64_t srcOffset = 0UL;
        uint64_t dstOffset = 0UL;
        if (subIdx == 1) {
            srcOffset = alignedN_ * vecM;
            dstOffset = vecM * this->block_.matmulTilingData_->matmulTiling.N;
            vecM = this->block_.params_.singleCoreM - vecM;
        }
        if (vecM == 0UL) {
            return;
        }
        uint64_t ubOffset = (pingPongId * TOTAL_UB_SIZE >> 1) / cDtypeSize;
        DataCopy<C_T>(ubTensor[ubOffset], cTensor[srcOffset], vecM * alignedN_);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(0x6 + pingPongId);

        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
        params_.repeatTimes = vecM;
        uint64_t rsvdCnt = 0UL;
        // src1Pattern is 7; mask is this->block_.matmulTilingData_->matmulTiling.N;
        GatherMask(ubTensor[ubOffset], ubTensor[ubOffset], 7, true, this->block_.matmulTilingData_->matmulTiling.N,
                   params_, rsvdCnt);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
        DataCopy<C_T>(this->cGlobal_[this->block_.offset_.offsetC + dstOffset], ubTensor[ubOffset],
                      AlignUp(vecM * this->block_.matmulTilingData_->matmulTiling.N, c0Size_));
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingPongId));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingPongId));
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::AicProcess(
    GlobalTensor<typename C_TYPE::T> &cTensor, uint8_t enAtomic, bool aicNeedWaitAiv, uint8_t pingPongId)
{
    if ASCEND_IS_AIC {
        this->mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                                 this->block_.matmulTilingData_->matmulTiling.singleCoreK);
        this->mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], A_TYPE::isTrans);
        this->mm_.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], B_TYPE::isTrans);
        if (this->block_.matmulTilingData_->matmulTiling.isBias) {
            this->mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
        }
        this->mm_.Iterate();
        if (aicNeedWaitAiv) {
            CrossCoreWaitFlag(0x6 + pingPongId);
        }
        this->mm_.GetTensorC(cTensor, enAtomic);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x4 + pingPongId);
#endif
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::CopyInWithNz(
    LocalTensor<C_T> tensorNZ, GlobalTensor<C_T> cNzGlobal, uint64_t ubProcessM, uint8_t pingPongId)
{
    // AIV cNzGlobal start address has updated
    size_t NfractualNum = alignedN_ / BLOCK_SIZE;
    DataCopyParams copyParams;
    copyParams.blockCount = NfractualNum;
    copyParams.blockLen = ubProcessM * NUM_TWO;
    copyParams.srcStride = (this->block_.matmulTilingData_->matmulTiling.baseM - ubProcessM) * NUM_TWO;
    copyParams.dstStride = 0;
    DataCopy(tensorNZ, cNzGlobal, copyParams);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::UpdateOffsetC(
    bool isNd)
{
    uint64_t mCntIndex = this->block_.params_.index / this->block_.params_.nCntUse;
    uint64_t nCntIndex = this->block_.params_.index % this->block_.params_.nCntUse;
    if (isNd) {
        this->block_.offset_.offsetC =
            (nCntIndex * this->block_.matmulTilingData_->matmulTiling.singleCoreN +
             mCntIndex * this->block_.matmulTilingData_->matmulTiling.singleCoreM *
                 this->block_.matmulTilingData_->matmulTiling.N +
             (this->block_.params_.mTileAddrOffset * this->block_.matmulTilingData_->matmulTiling.N +
              this->block_.params_.nTileAddrOffset));
    } else {
        this->block_.offset_.offsetC =
            (nCntIndex * this->block_.matmulTilingData_->matmulTiling.singleCoreN *
                 this->block_.matmulTilingData_->matmulTiling.M +
             mCntIndex * this->block_.matmulTilingData_->matmulTiling.singleCoreM * BLOCK_SIZE +
             (this->block_.params_.mTileAddrOffset * BLOCK_SIZE +
              this->block_.params_.nTileAddrOffset * this->block_.matmulTilingData_->matmulTiling.M));
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB,
                              FIXPIPE_OPT>::Nz2NdAndGatherMask(LocalTensor<C_T> tensorND, LocalTensor<C_T> tensorNZ,
                                                               uint64_t ubProcessM, uint8_t pingPongId)
{
    size_t NfractalNum = alignedN_ / BLOCK_SIZE;
    uint8_t repeatTimes = MMV3DivCeil(ubProcessM, 8); // ub calc 8 row every repeat
    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    UnaryRepeatParams mulsRepeatParams;
    mulsRepeatParams.srcBlkStride = 2;
    mulsRepeatParams.dstBlkStride = alignedN_ / c0Size_;
    mulsRepeatParams.dstRepStride = alignedN_;
    mulsRepeatParams.srcRepStride = 16;
    // float calc 2 cols per loop
    if constexpr (std::is_same_v<typename A_TYPE::T, float>) {
        for (size_t inLoop = 0; inLoop < NfractalNum; inLoop++) {
            Muls(tensorND[inLoop * BLOCK_SIZE], tensorNZ[inLoop * ubProcessM * BLOCK_SIZE], static_cast<C_T>(1.0), mask,
                 repeatTimes, mulsRepeatParams);
            Muls(tensorND[inLoop * BLOCK_SIZE + c0Size_], tensorNZ[inLoop * ubProcessM * BLOCK_SIZE + c0Size_],
                 static_cast<C_T>(1.0), mask, repeatTimes, mulsRepeatParams);
        }
    }
    PipeBarrier<PIPE_V>();
    // GatherMask
    uint64_t rsvdCnt = 0;
    params_.repeatTimes = ubProcessM;
    GatherMask(tensorND, tensorND, 7, true, this->block_.matmulTilingData_->matmulTiling.N, params_, rsvdCnt);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB,
                                                     FIXPIPE_OPT>::AivNz2NdProcess(GlobalTensor<C_T> cNzGlobal,
                                                                                   uint8_t pingPongId)
{
    if ASCEND_IS_AIV {
        uint32_t vBlockIndex = GetBlockIdx();
        if (vBlockIndex >= (this->block_.matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        CrossCoreWaitFlag(0x4 + pingPongId);
        LocalTensor<C_T> ubTensor = this->ubBuf_.template Get<C_T>();
        uint64_t cDtypeSize = sizeof(C_T);
        uint64_t ubProcessMNum = min(MMV3CeilAlign(this->block_.params_.singleCoreM / NUM_TWO, c0Size_),
                                     static_cast<uint64_t>(this->block_.params_.singleCoreM));
        int64_t subIdx = GetSubBlockIdx();
        uint64_t srcGmOffset = 0UL;
        uint64_t dstGmOffset = 0UL;

        if (subIdx == 1) {
            srcGmOffset = ubProcessMNum * ALIGNED_H; // for aiv 1 offset
            dstGmOffset = ubProcessMNum * this->block_.matmulTilingData_->matmulTiling.N;
            ubProcessMNum = this->block_.params_.singleCoreM - ubProcessMNum;
        }
        if (ubProcessMNum == 0UL) {
            return;
        }
        uint64_t ndOffset = (TOTAL_UB_SIZE >> 2) / cDtypeSize;
        uint64_t pingpongOffset = (TOTAL_UB_SIZE >> 1) / cDtypeSize;

        size_t NfractalNum = alignedN_ / ALIGNED_H;

        LocalTensor<C_T> tensorNZ = ubTensor[pingPongId * pingpongOffset];
        LocalTensor<C_T> tensorND = ubTensor[pingPongId * pingpongOffset + ndOffset];
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
        CopyInWithNz(tensorNZ, cNzGlobal[srcGmOffset], ubProcessMNum, pingPongId);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(0x6 + pingPongId); // 通知AIC MTE2可进行搬运
        Nz2NdAndGatherMask(tensorND, tensorNZ, ubProcessMNum, pingPongId);
        this->UpdateOffsetC(true);

        DataCopy(this->cGlobal_[this->block_.offset_.offsetC + dstGmOffset], tensorND,
                 AlignUp(ubProcessMNum * this->block_.matmulTilingData_->matmulTiling.N, c0Size_));
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Process(
    uint64_t index, uint8_t enAtomic)
{
    bool reverse = true;
    int8_t pingPongId = 0;
    bool aicNeedWaitAiv = false;
    ctx.isFirst = true;
    ctx.inputDtypeSize = sizeof(typename A_TYPE::T);
    GlobalTensor<C_T> tempCGlobal = tempCGlobal_;
    for (uint64_t mTileIndex = 0; mTileIndex < this->block_.params_.mTileCntL2; mTileIndex++) {
        reverse = !reverse;
        for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < this->block_.params_.nTileCntL2; nTileIndexTemp++) {
            uint64_t nTileIndex = reverse ? (this->block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
            this->block_.UpdateBlockCnt(mTileIndex, nTileIndex);
            this->block_.InitBlockIndex(index);
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG));
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + 1));
            for (uint64_t j = 0; j < this->block_.params_.realRound; j++) {
                tempCGlobal = tempCGlobal_[baseSize_ * (GetCurrentBlockIdx() * 2 + pingPongId)];
                if (this->block_.params_.rowOrder == 0) {
                    this->block_.UpdateBasicIndex(j); // 使能错位分核更新Index
                }
                if (this->block_.params_.index < this->block_.params_.totalTileCnt) {
                    this->block_.UpdateBlockParams(mTileIndex, nTileIndex);
                    this->block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
                    AicProcess(tempCGlobal, enAtomic, aicNeedWaitAiv, pingPongId);
                    if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
                        AivNz2NdProcess(tempCGlobal, pingPongId);
                    } else {
                        AivProcess(tempCGlobal, pingPongId);
                    }
                    aicNeedWaitAiv = aicNeedWaitAiv || bool(pingPongId);
                    pingPongId = (pingPongId + 1) & 1;
                }
                this->block_.UpdateBlockIndex();
            }
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + 1));
        }
    }
    if (this->block_.params_.isHf32) {
        this->mm_.SetHF32(false, 0);
    }
    PipeBarrier<PIPE_ALL>();
    return;
}

// Current Kernel support only nd2nzA. No need to do nd2nz for B.
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = Mc2MatmulBaseBlock,
          const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
          FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE_ENABLE_ALIGNOUT>
class Mc2MatmulBaseAToNZWithBL1FixpipeKernel
    : public Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT> {
    struct BaseUnAlignedNKernelParams {
        GM_ADDR aGMNZ;
        GM_ADDR workspaceGMNZ;
        uint64_t baseAN;
        uint64_t baseAD;
    };

public:
    __aicore__ inline Mc2MatmulBaseAToNZWithBL1FixpipeKernel()
    {
    }

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                GM_ADDR workspaceGM, const Mc2MatmulV3TilingData *tilingData, TPipe *pipe);

    __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);

protected:
    using C_T = typename C_TYPE::T;
    BaseUnAlignedNKernelParams fixpipeInnerParams_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void
Mc2MatmulBaseAToNZWithBL1FixpipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
    const Mc2MatmulV3TilingData *matmulTilingData, TPipe *pipe)
{
    GetSizeC0<C_T>(this->c0Size_);
    uint64_t cDtypeSize = sizeof(C_T);
    uint64_t alignedN = AlignUp(matmulTilingData->matmulTiling.N, MM_ALIGN_SIZE / cDtypeSize);
    uint64_t baseSize = alignedN * matmulTilingData->matmulTiling.baseM;

    Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
        workspaceGM + baseSize * NUM_TWO * matmulTilingData->matmulTiling.usedCoreNum * cDtypeSize, bGM, cGM, biasGM,
        offsetWGM, workspaceGM, matmulTilingData, pipe);
    this->fixpipeInnerParams_.baseAN = this->block_.matmulTilingData_->baseAN;
    this->fixpipeInnerParams_.baseAD = this->block_.matmulTilingData_->baseAD;
    this->fixpipeInnerParams_.aGMNZ = aGM;
    this->fixpipeInnerParams_.workspaceGMNZ =
        workspaceGM + this->baseSize_ * NUM_TWO * this->block_.matmulTilingData_->matmulTiling.usedCoreNum * cDtypeSize;
    this->mm_.SetOrgShape(this->block_.params_.alignedOriM, this->block_.matmulTilingData_->matmulTiling.N,
                          this->block_.params_.alignedKaSize, this->block_.matmulTilingData_->matmulTiling.Kb,
                          this->alignedN_);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
          class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void Mc2MatmulBaseAToNZWithBL1FixpipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
                                                              MM_CB, FIXPIPE_OPT>::Process(uint64_t index,
                                                                                           uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        MatrixAtoNZV2<typename A_TYPE::T>(this->fixpipeInnerParams_.workspaceGMNZ, this->fixpipeInnerParams_.aGMNZ,
                                          this->block_.matmulTilingData_->matmulTiling, A_TYPE::isTrans, this->ubBuf_,
                                          this->fixpipeInnerParams_.baseAN, this->fixpipeInnerParams_.baseAD);
        SyncAll();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CV_FLAG);
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(CV_FLAG);
    }
    Mc2MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Process(
        index, enAtomic);
}

#endif
