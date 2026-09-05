/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_nonquant_nz.h
 * \brief
 */

#ifndef ASCENDC_FFN_NONQUANT_NZ_H
#define ASCENDC_FFN_NONQUANT_NZ_H

#include "ffn_kernel.h"

namespace FFN {

template <typename ComputeType>
class FFNProcess {
private:
    ComputeType &computeOp; // internal computation operator
    const FFNTilingData *__restrict tilingData;

    uint32_t k1;
    uint32_t n1;
    uint32_t k2;
    uint32_t n2;
    uint32_t ubBaseM;
    uint32_t ubBaseN;
    uint32_t blockIdx;
    uint32_t coreIdx;
    uint32_t tokens;
    uint32_t coreNum;

public:
    /** @brief constructor */
    __aicore__ inline FFNProcess(ComputeType &computeOp_)
        : computeOp(computeOp_)
    {
    }

    __aicore__ inline void Init(const FFNTilingData *__restrict tiling);

    __aicore__ inline void Process();

    __aicore__ inline void MM1Process(MNConfig &mnConfig);

    __aicore__ inline void MM2Process(MNConfig &mnConfig);

    __aicore__ inline void MNConfigProcess(MNConfig &mnConfig, const TCubeTiling &mmTilingData, const uint32_t k,
                                           const uint32_t n);

    __aicore__ inline bool ZeroN1Process();
};

template <typename ComputeType>
__aicore__ inline void FFNProcess<ComputeType>::Init(const FFNTilingData *__restrict tiling)
{
    blockIdx = GetBlockIdx();
    coreIdx = blockIdx / GetTaskRation();
    tilingData = tiling;

    k1 = tilingData->ffnBaseParams.k1;
    n1 = tilingData->ffnBaseParams.n1;
    k2 = n1;
    n2 = tilingData->ffnBaseParams.n2;
    tokens = tilingData->ffnBaseParams.maxTokens;
    coreNum = tilingData->ffnBaseParams.coreNum;
    ubBaseM = tilingData->ffnSingleCoreParams.baseM1;
    ubBaseN = tilingData->ffnSingleCoreParams.baseN1;
}

template <typename ComputeType>
__aicore__ inline void FFNProcess<ComputeType>::Process()
{
    if (unlikely(ZeroN1Process())) {
        return;
    }

    MNConfig mnConfig1;
    MNConfigProcess(mnConfig1, tilingData->mm1TilingData, k1, n1);
    mnConfig1.m = tokens;
    mnConfig1.blockDimM = Ceil(mnConfig1.m, mnConfig1.baseM);
    MM1Process(mnConfig1);

    SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    computeOp.AllCoreSync();

    MNConfig mnConfig2;
    MNConfigProcess(mnConfig2, tilingData->mm2TilingData, k2, n2);
    mnConfig2.m = tokens;
    mnConfig2.blockDimM = Ceil(mnConfig2.m, mnConfig2.baseM);
    MM2Process(mnConfig2);
}

template <typename ComputeType>
__aicore__ inline void FFNProcess<ComputeType>::MNConfigProcess(MNConfig &mnConfig, const TCubeTiling &mmTilingData,
                                                                const uint32_t k, const uint32_t n)
{
    mnConfig.k = k;
    mnConfig.n = n;
    mnConfig.baseM = mmTilingData.baseM;
    mnConfig.baseN = mmTilingData.baseN;
    mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.baseN);
}

template <typename ComputeType>
__aicore__ inline void FFNProcess<ComputeType>::MM1Process(MNConfig &mnConfig)
{
    uint32_t totalBlock = mnConfig.blockDimM * mnConfig.blockDimN;
    uint32_t curBlock = coreIdx;
    while (curBlock < totalBlock) {
        mnConfig.mIdx = curBlock / mnConfig.blockDimN;
        mnConfig.nIdx = curBlock % mnConfig.blockDimN;
        mnConfig.singleM = mnConfig.baseM;
        mnConfig.singleN = mnConfig.baseN;
        computeOp.MM1Compute(mnConfig);
        curBlock += coreNum;
    }
}

template <typename ComputeType>
__aicore__ inline void FFNProcess<ComputeType>::MM2Process(MNConfig &mnConfig)
{
    uint32_t totalBlock = mnConfig.blockDimM * mnConfig.blockDimN;
    uint32_t curBlock = coreIdx;
    while (curBlock < totalBlock) {
        mnConfig.mIdx = curBlock / mnConfig.blockDimN;
        mnConfig.nIdx = curBlock % mnConfig.blockDimN;
        mnConfig.singleM = mnConfig.baseM;
        mnConfig.singleN = mnConfig.baseN;
        computeOp.MM2Compute(mnConfig);
        curBlock += coreNum;
    }
}

template <typename ComputeType>
__aicore__ inline bool FFNProcess<ComputeType>::ZeroN1Process()
{
    if (likely(n1 > 0)) {
        return false;
    }

    if (tokens > 0) {
        MNConfig mnConfig;
        mnConfig.m = tokens;
        mnConfig.n = n2;
        mnConfig.baseN = tilingData->ffnSingleCoreParams.baseN2;
        mnConfig.singleN = Ceil(mnConfig.n, coreNum);
        mnConfig.singleN = AlignUp<CUBE_BASE_ALIGN_FACTOR>(mnConfig.singleN);
        computeOp.ZeroN1Compute(mnConfig, coreIdx);
    }

    return true;
}

template <typename T, class mm1Type, class mm2Type = mm1Type, typename c1T = T, typename c2T = c1T, typename BiasT = T>
class FFNCompute {
private:
    // define matmul
    typename mm1Type::MT &mm1;
    typename mm2Type::MT &mm2;

    TPipe *pipe;
    GlobalTensor<T> xGm;
    GlobalTensor<T> weight1Gm;
    GlobalTensor<BiasT> bias1Gm;
    GlobalTensor<T> weight2Gm;
    GlobalTensor<BiasT> bias2Gm;
    GlobalTensor<c2T> yGm;
    GlobalTensor<T> mm2WorkspaceGm;
    GlobalTensor<int32_t> syncGm;
    LocalTensor<uint8_t> ubTemp;

    // define the queue/buffer
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TBuf<TPosition::VECCALC> tmpBuff;

    bool hasBias1 = false;
    bool hasBias2 = false;
    uint32_t activeType;
    int32_t activeBuffBias;

public:
    /** @brief constructor */
    __aicore__ inline FFNCompute(typename mm1Type::MT &mm1_, typename mm2Type::MT &mm2_)
        : mm1(mm1_),
          mm2(mm2_)
    {
    }

    /** Init function before process function
     * @param x: input 2D matrix.
     * @param weight1: input weight matrix of MM1.
     * @param weight2: input weight matrix of MM2.
     * @param expertTokens: expert information of x matrix.
     * @param bias1: whether MM1 has bias.
     * @param bias2: whether MM2 has bias.
     * @param mm1Workspace: global workspace address pointer.
     * @param tiling: tiling pointer.
     * @param tPipe: memory manager.
     */
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                __gm__ uint8_t *bias1, __gm__ uint8_t *bias2, __gm__ uint8_t *y,
                                __gm__ uint8_t *workspace, const FFNTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void CalcOffset(MNConfig &mnConfig, uint32_t &posN, uint64_t &xOffset, uint64_t &wOffset,
                                      uint64_t &outOffset);

    __aicore__ inline void MM1Compute(MNConfig &mnConfig);

    __aicore__ inline void MM2Compute(MNConfig &mnConfig);

    __aicore__ inline void Elewise(const MNConfig &mnConfig, const uint64_t activeOffset);

    __aicore__ inline void ElewiseCompute(const uint32_t computeSize);

    __aicore__ inline void ZeroN1Compute(MNConfig &mnConfig, const uint32_t coreIdx);

    __aicore__ inline uint32_t CursingleMNCompute(const uint32_t singleMN, const uint32_t mOrN, const uint32_t offset);

    __aicore__ inline void ZeroN1WithoutBiasCompute(const MNConfig &mnConfig, const uint32_t coreIdx);

    __aicore__ inline void ZeroN1WithBiasCompute(MNConfig &mnConfig, const uint32_t coreIdx);

    __aicore__ inline void AllCoreSync();
};

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::Init(
    __gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
    __gm__ uint8_t *y, __gm__ uint8_t *workspace, const FFNTilingData *__restrict tiling, TPipe *tPipe)
{
    activeType = tiling->ffnBaseParams.activeType;
    pipe = tPipe;
    // init global buffer
    xGm.SetGlobalBuffer((__gm__ T *)x);
    weight1Gm.SetGlobalBuffer((__gm__ T *)weight1);
    weight2Gm.SetGlobalBuffer((__gm__ T *)weight2);
    if (bias1 != nullptr) {
        bias1Gm.SetGlobalBuffer((__gm__ BiasT *)bias1);
    }
    if (bias2 != nullptr) {
        bias2Gm.SetGlobalBuffer((__gm__ BiasT *)bias2);
    }
    hasBias1 = bias1 != nullptr;
    hasBias2 = bias2 != nullptr;
    yGm.SetGlobalBuffer((__gm__ T *)y);
    syncGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
    mm2WorkspaceGm.SetGlobalBuffer(
        (__gm__ T *)(workspace + tiling->ffnBaseParams.syncWorkspaceSize + tiling->ffnBaseParams.workspace1Size));

    uint32_t ubCalSize = tiling->ffnSingleCoreParams.ubCalSize;
    pipe->InitBuffer(vecInQueue, 1, ubCalSize * sizeof(c1T) + UB_BLOCK_UNIT_SIZE);
    pipe->InitBuffer(vecOutQueue, 1, ubCalSize * sizeof(T));
    int32_t mm1BufferSize = tiling->mm1TilingData.baseN * sizeof(BiasT);
    int32_t mm2BufferSize = tiling->mm2TilingData.baseN * sizeof(BiasT) +
                            tiling->mm2TilingData.baseN * tiling->mm2TilingData.baseM * sizeof(c2T);
    pipe->InitBuffer(tmpBuff, tiling->ffnSingleCoreParams.ubRestBytes - UB_BLOCK_UNIT_SIZE);
    ubTemp = tmpBuff.Get<uint8_t>();
    LocalTensor<uint8_t> mm1Ub = ubTemp[tiling->ffnBaseParams.syncWorkspaceSize];
    mm1Ub.SetSize(mm1BufferSize);
    mm1.SetLocalWorkspace(mm1Ub);
    LocalTensor<uint8_t> mm2Ub = ubTemp[tiling->ffnBaseParams.syncWorkspaceSize];
    mm2Ub.SetSize(mm2BufferSize);
    mm2.SetLocalWorkspace(mm2Ub);

    // zeroing the soft synchronization space
    uint32_t eachCoreNum = UB_BLOCK_UNIT_SIZE / sizeof(int32_t);
    LocalTensor<int32_t> initLocal = ubTemp.template ReinterpretCast<int32_t>();
    Duplicate(initLocal, 0, eachCoreNum);
    DataCopy(syncGm[eachCoreNum * GetBlockIdx()], initLocal, eachCoreNum);
    activeBuffBias = tiling->ffnBaseParams.syncWorkspaceSize + mm1BufferSize;
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::AllCoreSync()
{
    LocalTensor<int32_t> syncLocal = ubTemp.template ReinterpretCast<int32_t>();
    SyncAll(syncGm, syncLocal);
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void
FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::CalcOffset(MNConfig &mnConfig, uint32_t &posN, uint64_t &xOffset,
                                                             uint64_t &wOffset, uint64_t &outOffset)
{
    uint32_t posM = mnConfig.mIdx * mnConfig.baseM;
    posN = mnConfig.nIdx * mnConfig.baseN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        mnConfig.singleN = mnConfig.n - posN;
    }
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        mnConfig.singleM = mnConfig.m - posM;
    }
    xOffset = posM * CUBE_BASE_ALIGN_FACTOR;
    wOffset = posN * mnConfig.k;
    outOffset = (uint64_t)posN * mnConfig.m + xOffset;
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::MM1Compute(MNConfig &mnConfig)
{
    uint32_t posN;
    uint64_t xOffset;
    uint64_t wOffset;
    uint64_t outOffset;
    CalcOffset(mnConfig, posN, xOffset, wOffset, outOffset);

    mm1.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm1.SetSingleShape(mnConfig.singleM, mnConfig.singleN, mnConfig.k);
    mm1.SetTensorA(xGm[xOffset]);
    mm1.SetTensorB(weight1Gm[wOffset]);
    if (hasBias1) {
        mm1.SetBias(bias1Gm[posN]);
    } else {
        mm1.ClearBias();
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t *)0);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(global);
#endif

    mm1.template Iterate<false>();

    LocalTensor<c1T> inLocal = vecInQueue.AllocTensor<c1T>();
    auto inLocalTmp = inLocal[CUBE_BASE_ALIGN_FACTOR];

    mm1.template GetTensorC<false>(inLocalTmp);
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    vecInQueue.EnQue(inLocal);
    Elewise(mnConfig, outOffset);
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::Elewise(const MNConfig &mnConfig,
                                                                                 const uint64_t activeOffset)
{
    uint32_t computeSize = mnConfig.singleM * mnConfig.singleN;
    ElewiseCompute(computeSize);

    LocalTensor<T> activeResUb = vecOutQueue.DeQue<T>();
    uint32_t dstStride = mnConfig.m - mnConfig.singleM;
    uint16_t blockCount = mnConfig.singleN / CUBE_BASE_ALIGN_FACTOR;
    if (dstStride >= UINT16_MAX) {
        uint32_t mAlign = mnConfig.m * CUBE_BASE_ALIGN_FACTOR;
        uint32_t singleMAlign = mnConfig.singleM * CUBE_BASE_ALIGN_FACTOR;
        for (uint32_t nLoop = 0; nLoop < blockCount; nLoop++) {
            DataCopy(mm2WorkspaceGm[activeOffset + nLoop * mAlign], activeResUb[nLoop * singleMAlign],
                     {1, static_cast<uint16_t>(mnConfig.singleM), 0, 0});
        }
    } else {
        DataCopy(mm2WorkspaceGm[activeOffset], activeResUb,
                 {blockCount, static_cast<uint16_t>(mnConfig.singleM), 0, static_cast<uint16_t>(dstStride)});
    }
    vecOutQueue.FreeTensor(activeResUb);
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::ElewiseCompute(const uint32_t computeSize)
{
    LocalTensor<uint8_t> tmpLocal = ubTemp[activeBuffBias];
    LocalTensor<T> activeResUb = vecOutQueue.AllocTensor<T>();
    ActiveType active = ActiveType(activeType);
    LocalTensor<T> mm1ResUbTmp = vecInQueue.DeQue<T>();
    auto mm1ResUb = mm1ResUbTmp[16];
    ApplyActivation(active, activeResUb, mm1ResUb, tmpLocal, computeSize);
    vecInQueue.FreeTensor(mm1ResUbTmp);
    vecOutQueue.EnQue<T>(activeResUb);
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::MM2Compute(MNConfig &mnConfig)
{
    uint32_t posN;
    uint64_t xOffset;
    uint64_t wOffset;
    uint64_t outOffset;
    CalcOffset(mnConfig, posN, xOffset, wOffset, outOffset);

    mm2.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm2.SetSingleShape(mnConfig.singleM, mnConfig.singleN, mnConfig.k);
    mm2.SetTensorA(mm2WorkspaceGm[xOffset]);
    mm2.SetTensorB(weight2Gm[wOffset]);
    if (hasBias2) {
        mm2.SetBias(bias2Gm[posN]);
    } else {
        mm2.ClearBias();
    }
    mm2.template IterateAll<false>(yGm[outOffset]);
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::ZeroN1Compute(MNConfig &mnConfig,
                                                                                       const uint32_t coreIdx)
{
    if (hasBias2) {
        ZeroN1WithBiasCompute(mnConfig, coreIdx);
    } else {
        ZeroN1WithoutBiasCompute(mnConfig, coreIdx);
    }
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline uint32_t FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::CursingleMNCompute(const uint32_t singleMN,
                                                                                                const uint32_t mOrN,
                                                                                                const uint32_t offset)
{
    uint32_t cursingleMN = singleMN;
    if (offset >= mOrN) {
        cursingleMN = 0;
    } else if (mOrN - offset < singleMN) {
        cursingleMN = mOrN - offset;
    }

    return cursingleMN;
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void
FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::ZeroN1WithoutBiasCompute(const MNConfig &mnConfig,
                                                                           const uint32_t coreIdx)
{
    uint32_t offset = mnConfig.singleN * coreIdx;
    uint32_t cursingleN = CursingleMNCompute(mnConfig.singleN, mnConfig.n, offset);
    if (cursingleN == 0) {
        return;
    }

    InitOutput<c2T>(yGm[offset * mnConfig.m], cursingleN * mnConfig.m, 0);
}

template <typename T, class mm1Type, class mm2Type, typename c1T, typename c2T, typename BiasT>
__aicore__ inline void FFNCompute<T, mm1Type, mm2Type, c1T, c2T, BiasT>::ZeroN1WithBiasCompute(MNConfig &mnConfig,
                                                                                               const uint32_t coreIdx)
{
    if (mnConfig.singleN < mnConfig.baseN) {
        mnConfig.singleN = mnConfig.baseN;
    }
    uint32_t offset = mnConfig.singleN * coreIdx;
    uint32_t cursingleN = CursingleMNCompute(mnConfig.singleN, mnConfig.n, offset);
    if (cursingleN == 0) {
        return;
    }

    uint32_t n2Loops = (cursingleN + mnConfig.baseN - 1) / mnConfig.baseN;
    uint32_t curBaseN2 = mnConfig.baseN;
    DataCopyPadParams padParams;
    for (int n2InnerIdx = 0; n2InnerIdx < n2Loops; n2InnerIdx++) {
        if (n2InnerIdx == n2Loops - 1) {
            curBaseN2 = cursingleN - n2InnerIdx * mnConfig.baseN;
        }
        uint64_t biasOffset = offset + n2InnerIdx * mnConfig.baseN;
        LocalTensor<T> inLocalBias2 = vecInQueue.AllocTensor<T>();
        DataCopyParams intriParams1{1, static_cast<uint16_t>(curBaseN2 * sizeof(T)), 0, 0};
        DataCopy(inLocalBias2, bias2Gm[biasOffset], intriParams1);
        vecInQueue.EnQue<T>(inLocalBias2);
        inLocalBias2 = vecInQueue.DeQue<T>();
        LocalTensor<T> outLocalBias2 = vecOutQueue.AllocTensor<T>();
        Adds(outLocalBias2, inLocalBias2, (T)0, curBaseN2);
        vecInQueue.FreeTensor(inLocalBias2);
        vecOutQueue.EnQue<T>(outLocalBias2);
        outLocalBias2 = vecOutQueue.DeQue<T>();

        uint16_t blockCount = curBaseN2 / CUBE_BASE_ALIGN_FACTOR;
        uint32_t dstStride = mnConfig.m - 1;
        for (uint32_t loopCnt = 0; loopCnt < mnConfig.m; loopCnt++) {
            if (dstStride > UINT16_MAX) {
                for (uint32_t nLoop = 0; nLoop < blockCount; nLoop++) {
                    DataCopy(yGm[biasOffset * mnConfig.m + (nLoop * mnConfig.m + loopCnt) * CUBE_BASE_ALIGN_FACTOR],
                             outLocalBias2[nLoop * CUBE_BASE_ALIGN_FACTOR], {1, 1, 0, 0});
                }
            } else {
                DataCopy(yGm[biasOffset * mnConfig.m + loopCnt * CUBE_BASE_ALIGN_FACTOR], outLocalBias2,
                         {blockCount, 1, 0, static_cast<uint16_t>(dstStride)});
            }
        }
        vecOutQueue.FreeTensor(outLocalBias2);
    }
}
} // namespace FFN

#endif // ASCENDC_FFN_NONQUANT_NZ_H
