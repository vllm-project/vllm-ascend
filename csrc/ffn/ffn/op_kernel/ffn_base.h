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
 * \file ffn_base.h
 * \brief
 */

#ifndef ASCENDC_FFN_BASE_H
#define ASCENDC_FFN_BASE_H

#include "ffn_kernel.h"

namespace FFN {

template <typename T, class mm1Type, class mm2Type = mm1Type, typename c1T = T, typename c2T = c1T, typename BiasT = T>
class FFNBase {
protected:
    using Btype = typename mm1Type::BT;

    const FFNTilingData *__restrict tilingData;
    // define matmul
    typename mm1Type::MT &mm1;
    typename mm2Type::MT &mm2;

    TPipe *pipe;
    GlobalTensor<T> xGm;
    GlobalTensor<int64_t> expertTokensGm;
    GlobalTensor<T> weight1Gm;
    GlobalTensor<BiasT> bias1Gm;
    GlobalTensor<T> weight2Gm;
    GlobalTensor<BiasT> bias2Gm;
    GlobalTensor<c2T> yGm;
    GlobalTensor<c1T> mm1WorkspaceGm;
    GlobalTensor<T> mm2WorkspaceGm;
    LocalTensor<uint8_t> ubTemp;
    LocalTensor<int64_t> ubTokens;

    // define the queue/buffer
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;

    bool hasBias1 = false;
    bool hasBias2 = false;
    uint32_t k1;
    uint32_t n1;
    uint32_t k2;
    uint32_t n2;
    uint32_t ubBaseM;
    uint32_t ubBaseN;
    uint32_t blockIdx;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t tokens;
    uint32_t tokensOffset;
    uint32_t expertNum;
    uint32_t activeType;
    uint32_t maxCoreNum;
    uint32_t aivCoreNum;

public:
    /** @brief constructor */
    __aicore__ inline FFNBase(typename mm1Type::MT &mm1_, typename mm2Type::MT &mm2_)
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
                                __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                                __gm__ uint8_t *y, __gm__ uint8_t *workspace, const FFNTilingData *__restrict tiling,
                                TPipe *tPipe)
    {
        blockIdx = GetBlockIdx();
        subBlockIdx = GetSubBlockIdx();
        coreIdx = blockIdx / GetTaskRation();
        pipe = tPipe;
        tilingData = tiling;
        InitTilingData();

        // init global buffer
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weight1Gm.SetGlobalBuffer((__gm__ T *)weight1);
        if (bias1 != nullptr) {
            hasBias1 = true;
            bias1Gm.SetGlobalBuffer((__gm__ BiasT *)bias1);
        }
        weight2Gm.SetGlobalBuffer((__gm__ T *)weight2);
        if (bias2 != nullptr) {
            hasBias2 = true;
            bias2Gm.SetGlobalBuffer((__gm__ BiasT *)bias2);
        }
        yGm.SetGlobalBuffer((__gm__ T *)y);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ c1T *)workspace);
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ T *)(workspace + tilingData->ffnBaseParams.workspace1Size));

        uint32_t ubCalSize = tilingData->ffnSingleCoreParams.ubCalSize;
        pipe->InitBuffer(vecInQueue, 1, ubCalSize * sizeof(c1T));
        pipe->InitBuffer(vecOutQueue, 1, ubCalSize * sizeof(T));
        if (likely(tilingData->ffnSingleCoreParams.ubRestBytes > 0)) {
            TBuf<TPosition::VECCALC> tmpBuff;
            pipe->InitBuffer(tmpBuff, tilingData->ffnSingleCoreParams.ubRestBytes);
            ubTemp = tmpBuff.Get<uint8_t>();
        }
        ubTokens = GetUbTokens(expertTokens, expertTokensGm, tilingData, pipe);
    }

protected:
    __aicore__ inline void InitTilingData()
    {
        k1 = tilingData->ffnBaseParams.k1;
        n1 = tilingData->ffnBaseParams.n1;
        k2 = n1;
        n2 = tilingData->ffnBaseParams.n2;
        expertNum = tilingData->ffnBaseParams.expertNum;
        activeType = tilingData->ffnBaseParams.activeType;
        tokens = 0;
        tokensOffset = 0;
        maxCoreNum = tilingData->ffnBaseParams.coreNum;
        aivCoreNum = tilingData->ffnBaseParams.coreNum * GetTaskRation();

        ubBaseM = tilingData->ffnSingleCoreParams.baseM1;
        ubBaseN = tilingData->ffnSingleCoreParams.baseN1;
    }

    __aicore__ inline bool CheckSyncBeforeMM1(MNConfig &mnConfig, uint32_t maxMM1ExpertParallelism,
                                              uint32_t maxMM2ExpertParallelism)
    {
        UpdateKernelTilingInfo info{n1, n2, expertNum, maxMM1ExpertParallelism};

        uint32_t maxMM1UsedCubeCore;
        UpdateKernelTilingBeforeMM1(mnConfig, maxMM1UsedCubeCore, tokens, info, tilingData);

        uint32_t minMM2UsedCubeCore = mnConfig.blockDimM * mnConfig.blockDimN;
        return minMM2UsedCubeCore < maxMM1UsedCubeCore;
    }

    __aicore__ inline void SyncBeforeMM1(bool waitIterateAll, bool &mm2WaitStatus, bool &firstMM1)
    {
        if (mm2WaitStatus) {
            mm2.WaitIterateAll();
            mm2.End();
            mm2WaitStatus = false;
        }
        if (unlikely(firstMM1)) {
            firstMM1 = false;
        } else {
            if (waitIterateAll) {
                SyncAll<true>();
            }
        }
    }

    __aicore__ inline void MM1Compute(MNConfig &mnConfig, uint32_t baseBlockIdx, uint32_t expertIdx,
                                      uint32_t tokensOffset, uint32_t outRowOffset)
    {
        uint32_t mIdx = baseBlockIdx / mnConfig.blockDimN;
        uint32_t nIdx = baseBlockIdx % mnConfig.blockDimN;
        uint32_t tailN = nIdx * mnConfig.singleN;
        uint32_t curSingleN = (nIdx == mnConfig.blockDimN - 1) ? (mnConfig.n - tailN) : mnConfig.singleN;
        uint32_t curSingleM =
            (mIdx == mnConfig.blockDimM - 1) ? (mnConfig.m - mIdx * mnConfig.singleM) : mnConfig.singleM;
        uint64_t outOffset = uint64_t(outRowOffset + mIdx * mnConfig.singleM) * n1 + tailN;
        uint64_t xCoreOffset = uint64_t(tokensOffset + mIdx * mnConfig.singleM) * k1;
        uint64_t w1CoreOffset = expertIdx * (uint64_t)k1 * n1;
        if constexpr (Btype::format == CubeFormat::NZ) {
            w1CoreOffset += tailN * k1;
        } else {
            w1CoreOffset += tailN;
        }

        mm1.SetOrgShape(mnConfig.m, n1, k1);
        mm1.SetSingleShape(curSingleM, curSingleN, k1);
        mm1.SetTensorA(xGm[xCoreOffset]);
        mm1.SetTensorB(weight1Gm[w1CoreOffset]);
        if (hasBias1) {
            mm1.SetBias(bias1Gm[expertIdx * n1 + tailN]);
        } else {
            mm1.ClearBias();
        }
        mm1.template IterateAll<true>(mm1WorkspaceGm[outOffset], false);
        mm1.End();
        if constexpr (IsSameType<c1T, float>::value) {
            Elewise1HighPrecision(curSingleM, curSingleN, outOffset, outOffset);
        } else {
            if (n1 <= curSingleN && tilingData->ffnSingleCoreParams.ubCalSize > n1) {
                Elewise2(curSingleM, curSingleN, outOffset, outOffset);
            } else {
                Elewise1(curSingleM, curSingleN, outOffset, outOffset);
            }
        }
    }

    __aicore__ inline void Elewise2(uint32_t curSingleM, uint32_t curSingleN, uint64_t mm1OutOffset,
                                    uint64_t activeOffset)
    {
        DataCopyPadParams padParams;
        DataCopyParams intriParams1;
        DataCopyParams intriParams2;
        uint32_t computeSize;
        uint32_t ubBaseM_ = tilingData->ffnSingleCoreParams.ubCalSize / n1;
        uint32_t curBaseM = ubBaseM_;

        for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += ubBaseM_) {
            if (offsetM + ubBaseM_ >= curSingleM) {
                curBaseM = curSingleM - offsetM;
            }
            LocalTensor<c1T> inLocal = vecInQueue.AllocTensor<c1T>();
            computeSize = curBaseM * n1;
            intriParams1.blockLen = computeSize * sizeof(c1T);
            intriParams1.blockCount = 1;
            intriParams1.srcStride = 0;
            intriParams1.dstStride = 0;
            DataCopyPad(inLocal, mm1WorkspaceGm[mm1OutOffset + uint64_t(offsetM) * n1], intriParams1, padParams);
            vecInQueue.EnQue(inLocal);
            Elewise1Compute(computeSize);
            LocalTensor<T> activeResUb = vecOutQueue.DeQue<T>();
            intriParams2.blockLen = computeSize * sizeof(T);
            intriParams2.blockCount = 1;
            intriParams2.srcStride = 0;
            intriParams2.dstStride = 0;
            DataCopyPad(mm2WorkspaceGm[activeOffset + uint64_t(offsetM) * n1], activeResUb, intriParams2);
            vecOutQueue.FreeTensor(activeResUb);
        }
    }

    __aicore__ inline void Elewise1(uint32_t curSingleM, uint32_t curSingleN, uint64_t mm1OutOffset,
                                    uint64_t activeOffset)
    {
        DataCopyPadExtParams<c1T> padParams;
        DataCopyExtParams intriParams1;
        DataCopyExtParams intriParams2;
        uint32_t computeBaseN1;
        uint32_t computeSize;
        uint32_t curBaseM = ubBaseM;

        for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += ubBaseM) {
            if (offsetM + ubBaseM >= curSingleM) {
                curBaseM = curSingleM - offsetM;
            }
            uint32_t curBaseN1 = ubBaseN;
            for (uint32_t offsetN = 0; offsetN < curSingleN; offsetN += ubBaseN) {
                if (offsetN + ubBaseN >= curSingleN) {
                    curBaseN1 = curSingleN - offsetN;
                }
                computeBaseN1 = AlignUp<GetNumInUbBlock<c1T>()>(curBaseN1);
                computeSize = curBaseM * computeBaseN1;
                LocalTensor<c1T> inLocal = vecInQueue.AllocTensor<c1T>();
                intriParams1.blockLen = curBaseN1 * sizeof(c1T);
                intriParams1.blockCount = curBaseM;
                intriParams1.srcStride = (n1 - curBaseN1) * sizeof(c1T);
                intriParams1.dstStride = 0;
                uint64_t offset = (uint64_t)(offsetM)*n1 + offsetN;
                DataCopyPad(inLocal, mm1WorkspaceGm[mm1OutOffset + offset], intriParams1, padParams);
                vecInQueue.EnQue(inLocal);
                Elewise1Compute(computeSize);
                LocalTensor<T> activeResUb = vecOutQueue.DeQue<T>();
                intriParams2.blockLen = curBaseN1 * sizeof(T);
                intriParams2.blockCount = curBaseM;
                intriParams2.srcStride = 0;
                intriParams2.dstStride = (n1 - curBaseN1) * sizeof(T);
                DataCopyPad(mm2WorkspaceGm[activeOffset + offset], activeResUb, intriParams2);
                vecOutQueue.FreeTensor(activeResUb);
            }
        }
    }

    __aicore__ inline void Elewise1HighPrecision(uint32_t curSingleM, uint32_t curSingleN, uint64_t mm1OutOffset,
                                                 uint64_t activeOffset)
    {
        uint32_t curBaseM = ubBaseM / 2; // 2: half size
        DataCopyPadExtParams<c1T> padParams;
        DataCopyExtParams intriParams1;
        DataCopyExtParams intriParams2;
        uint32_t computeBaseN1;
        uint32_t computeSize;
        for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += (ubBaseM / 2)) { // 2: half size
            if (offsetM + (ubBaseM / 2) >= curSingleM) {                             // 2: half size
                curBaseM = curSingleM - offsetM;
            }
            uint32_t curBaseN1 = ubBaseN;
            for (uint32_t offsetN = 0; offsetN < curSingleN; offsetN += ubBaseN) {
                if (offsetN + ubBaseN >= curSingleN) {
                    curBaseN1 = curSingleN - offsetN;
                }
                computeBaseN1 = AlignUp<16>(curBaseN1); // 16 elements alignment, due to Cast operator.
                computeSize = curBaseM * computeBaseN1;
                LocalTensor<c1T> inLocal = vecInQueue.template AllocTensor<c1T>();
                intriParams1.blockLen = curBaseN1 * sizeof(c1T);
                intriParams1.blockCount = curBaseM;
                intriParams1.srcStride = (n1 - curBaseN1) * sizeof(c1T);
                intriParams1.dstStride = (computeBaseN1 - curBaseN1) * sizeof(c1T) / 32; // 32: unit of stride
                uint64_t offset = (uint64_t)(offsetM)*n1 + offsetN;
                DataCopyPad(inLocal, mm1WorkspaceGm[mm1OutOffset + offset], intriParams1, padParams);
                vecInQueue.EnQue(inLocal);
                Elewise1ComputeHighPrecision(computeSize);
                LocalTensor<T> activeResUb = vecOutQueue.template DeQue<T>();
                intriParams2.blockLen = curBaseN1 * sizeof(T);
                intriParams2.blockCount = curBaseM;
                intriParams2.srcStride = 0;
                intriParams2.dstStride = (n1 - curBaseN1) * sizeof(T);
                DataCopyPad(mm2WorkspaceGm[activeOffset + offset], activeResUb, intriParams2);
                vecOutQueue.FreeTensor(activeResUb);
            }
        }
    }

    __aicore__ inline void Elewise1Compute(uint32_t computeSize)
    {
        LocalTensor<T> mm1ResUb = vecInQueue.DeQue<T>();
        LocalTensor<T> activeResUb = vecOutQueue.AllocTensor<T>();

        ActiveType active = ActiveType(activeType);
        if (active == ActiveType::FASTGELU) {
            FasterGelu(activeResUb, mm1ResUb, ubTemp, computeSize);
        } else if (active == ActiveType::RELU) {
            Relu(activeResUb, mm1ResUb, computeSize);
        } else if (active == ActiveType::SILU) {
            Silu(activeResUb, mm1ResUb, computeSize);
        } else if (active == ActiveType::GELU) {
            Gelu(activeResUb, mm1ResUb, ubTemp, computeSize);
        }
        vecInQueue.FreeTensor(mm1ResUb);
        vecOutQueue.EnQue<T>(activeResUb);
    }

    __aicore__ inline void Elewise1ComputeHighPrecision(uint32_t computeSize)
    {
        LocalTensor<c1T> mm1ResUb = vecInQueue.template DeQue<c1T>();
        LocalTensor<T> activeResUb = vecOutQueue.template AllocTensor<T>();
        LocalTensor<c1T> activeResUbFp32 = ubTemp.template ReinterpretCast<c1T>()[computeSize];
        LocalTensor<uint8_t> tmpLocal = ubTemp[computeSize * sizeof(c1T)].template ReinterpretCast<uint8_t>();

        ActiveType active = ActiveType(activeType);
        if (active == ActiveType::FASTGELU) {
            FasterGelu(activeResUbFp32, mm1ResUb, tmpLocal, computeSize);
        } else if (active == ActiveType::RELU) {
            Relu(activeResUbFp32, mm1ResUb, computeSize);
        } else if (active == ActiveType::SILU) {
            Silu(activeResUbFp32, mm1ResUb, computeSize);
        } else if (active == ActiveType::GELU) {
            Gelu(activeResUbFp32, mm1ResUb, tmpLocal, computeSize);
        }
        PipeBarrier<PIPE_V>();
        vecInQueue.FreeTensor(mm1ResUb);
        Cast(activeResUb, activeResUbFp32, RoundMode::CAST_ROUND, computeSize);
        vecOutQueue.template EnQue<T>(activeResUb);
    }

    __aicore__ inline void MM1Process(ExpertParallInfo &mm1ExpertParallInfo, bool waitIterateAll, MNConfig &mnConfig,
                                      bool &mm2WaitStatus, bool &firstMM1)
    {
        uint32_t coreNumEachExpert = maxCoreNum / mm1ExpertParallInfo.expertParallelism;
        mnConfig.SetConstriant(tokens, n1, tilingData->mm1TilingData.baseM, tilingData->mm1TilingData.baseN,
                               coreNumEachExpert);
        KernelTiling(mnConfig);
        coreNumEachExpert = mnConfig.blockDimM * mnConfig.blockDimN;
        size_t expertOrderInBuf =
            Min<uint32_t>(mm1ExpertParallInfo.start + coreIdx / coreNumEachExpert, mm1ExpertParallInfo.maxSize - 1);
        // make sure which expert each core/cube needs to compute
        uint32_t expertIMM = mm1ExpertParallInfo.expertIdxBuf[expertOrderInBuf];
        tokens = ubTokens.GetValue(expertIMM);
        SyncBeforeMM1(waitIterateAll, mm2WaitStatus, firstMM1);

        if (coreIdx < mm1ExpertParallInfo.expertParallelism * coreNumEachExpert && subBlockIdx == 0) {
            // assert mm1ExpertParallInfo.size == mm1ExpertParallInfo.start + mm1ParallelExpertsNum
            // For this expertOrderInBuf/expert, determine the offset of output
            // local m-offset of the expert
            mnConfig.m = tokens;
            uint32_t baseBlockIdx = coreIdx % coreNumEachExpert;
            uint32_t outRowOffset = mm1ExpertParallInfo.LocalOffset[expertOrderInBuf] -
                                    mm1ExpertParallInfo.LocalOffset[mm1ExpertParallInfo.start];
            MM1Compute(mnConfig, baseBlockIdx, expertIMM, mm1ExpertParallInfo.GlobalOffset[expertOrderInBuf],
                       outRowOffset);
        }
    }

    template <bool sync = false>
    __aicore__ inline void MM2Compute(MNConfig &mnConfig, uint32_t baseBlockIdx, uint32_t expertIdx,
                                      uint32_t tokensRowOffset, uint32_t outRowOffset, bool waitIterateAll)
    {
        uint32_t mIdx = baseBlockIdx / mnConfig.blockDimN;
        uint32_t nIdx = baseBlockIdx % mnConfig.blockDimN;
        uint32_t curSingleN = mnConfig.singleN;
        uint32_t tailN = nIdx * mnConfig.singleN;
        if (nIdx == mnConfig.blockDimN - 1) {
            curSingleN = mnConfig.n - tailN;
            // If there is no sync op after mm2, the last core needs the same computation load as others
            if (!waitIterateAll && mnConfig.n > mnConfig.singleN &&
                Ceil(mnConfig.singleN, 64) > Ceil(curSingleN, 64)) { // 64: an experimental threshold
                tailN = mnConfig.n - mnConfig.singleN;
                curSingleN = mnConfig.singleN;
            }
        }
        uint32_t curSingleM = mnConfig.singleM;
        if (mIdx == mnConfig.blockDimM - 1) {
            curSingleM = mnConfig.m - mIdx * curSingleM;
        }
        uint64_t outOffset = uint64_t(outRowOffset + mIdx * mnConfig.singleM) * n2 + tailN;
        uint64_t xCoreOffset = uint64_t(tokensRowOffset + mIdx * mnConfig.singleM) * k2;
        uint64_t w2CoreOffset = expertIdx * (uint64_t)k2 * n2;
        if constexpr (Btype::format == CubeFormat::NZ) {
            w2CoreOffset += tailN * k2;
        } else {
            w2CoreOffset += tailN;
        }
        mm2.SetOrgShape(mnConfig.m, n2, k2);
        mm2.SetSingleShape(curSingleM, curSingleN, k2);
        mm2.SetTensorA(mm2WorkspaceGm[xCoreOffset]);
        mm2.SetTensorB(weight2Gm[w2CoreOffset]);
        if (hasBias2) {
            mm2.SetBias(bias2Gm[expertIdx * n2 + tailN]);
        } else {
            mm2.ClearBias();
        }
        mm2.template IterateAll<sync>(yGm[outOffset], 0, false, !sync && waitIterateAll);
        if (sync || (!sync && !waitIterateAll)) {
            mm2.End();
        }
    }

    __aicore__ inline void MM2Process(ExpertParallInfo &mm2ExpertParallInfo, bool waitIterateAll, MNConfig &mnConfig,
                                      bool &mm2WaitStatus)
    {
        for (uint32_t i = mm2ExpertParallInfo.start; i < mm2ExpertParallInfo.size;
             i += mm2ExpertParallInfo.expertParallelism) {
            if (i + mm2ExpertParallInfo.expertParallelism > mm2ExpertParallInfo.size) {
                mm2ExpertParallInfo.expertParallelism = mm2ExpertParallInfo.size - i;
            }
            uint32_t coreNumEachExpert = maxCoreNum / mm2ExpertParallInfo.expertParallelism;

            if (coreIdx >= coreNumEachExpert * mm2ExpertParallInfo.expertParallelism || subBlockIdx != 0) {
                continue;
            }
            uint32_t expertOrderInBuf = i + coreIdx / coreNumEachExpert;
            uint32_t expertIMM = mm2ExpertParallInfo.expertIdxBuf[expertOrderInBuf];
            tokens = ubTokens.GetValue(expertIMM);
            mnConfig.SetConstriant(tokens, n2, tilingData->mm2TilingData.baseM, tilingData->mm2TilingData.baseN,
                                   coreNumEachExpert);
            KernelTiling(mnConfig);
            uint32_t baseBlockIdx = coreIdx % coreNumEachExpert;
            coreNumEachExpert = mnConfig.blockDimM * mnConfig.blockDimN;
            if (baseBlockIdx < coreNumEachExpert) {
                uint32_t tokensRowOffset = mm2ExpertParallInfo.LocalOffset[expertOrderInBuf] -
                                           mm2ExpertParallInfo.LocalOffset[mm2ExpertParallInfo.start];
                uint32_t outRowOffset = mm2ExpertParallInfo.GlobalOffset[expertOrderInBuf];
                if (mm2WaitStatus) {
                    mm2.WaitIterateAll();
                    mm2.End();
                    mm2WaitStatus = false;
                }
                MM2Compute<false>(mnConfig, baseBlockIdx, expertIMM, tokensRowOffset, outRowOffset, waitIterateAll);
                mm2WaitStatus = waitIterateAll;
            }
        }
    }

    __aicore__ inline void ComputeExpertParallNum(const uint32_t &expertI, const uint32_t &baseM,
                                                  ExpertParallInfo &expertParallInfo)
    {
        if (expertI == expertNum) {
            expertParallInfo.expertParallelism = Min(expertParallInfo.size, expertParallInfo.maxExpertParallelism);
            expertParallInfo.start = 0;
            tokens = ubTokens.GetValue(expertParallInfo.expertIdxBuf[0]);
            return;
        }
        ComputeExpertParallelNum(expertI, baseM, tokens, tokensOffset, expertParallInfo);
    }

    __aicore__ inline void ComputeZeroN1WithoutBias(uint32_t coreIdx, uint32_t expertIdx)
    {
        uint32_t singleM1 = Ceil(tokens, tilingData->ffnBaseParams.coreNum);
        singleM1 = AlignUp<CUBE_BASE_ALIGN_FACTOR>(singleM1);
        uint32_t cursingleM = singleM1;
        if (singleM1 * coreIdx >= tokens) {
            cursingleM = 0;
        } else if (tokens - singleM1 * coreIdx < singleM1) {
            cursingleM = tokens - singleM1 * coreIdx;
        }
        if (cursingleM > 0) {
            InitOutput<c2T>(yGm[(tokensOffset + singleM1 * coreIdx) * n2], cursingleM * n2, 0);
        }
    }

    __aicore__ inline void ComputeStepZeroN1WithBias(uint32_t expertIdx, uint32_t baseN2, uint32_t offset,
                                                     uint32_t curBaseN2, uint32_t n2InnerIdx)
    {
        DataCopyPadParams padParams;
        if constexpr (IsSameType<c1T, float>::value) {
            LocalTensor<BiasT> inLocalBias2 = vecInQueue.template AllocTensor<BiasT>();
            DataCopyParams intriParams1{1, static_cast<uint16_t>(curBaseN2 * sizeof(BiasT)), 0, 0};
            DataCopyPad(inLocalBias2, bias2Gm[expertIdx * n2 + offset + n2InnerIdx * baseN2], intriParams1, padParams);
            vecInQueue.template EnQue<BiasT>(inLocalBias2);
            inLocalBias2 = vecInQueue.template DeQue<BiasT>();
            LocalTensor<c2T> outLocalBias2 = vecOutQueue.template AllocTensor<c2T>();
            Cast(outLocalBias2, inLocalBias2, RoundMode::CAST_ROUND, curBaseN2);
            vecInQueue.template FreeTensor(inLocalBias2);
            vecOutQueue.template EnQue<c2T>(outLocalBias2);
            outLocalBias2 = vecOutQueue.template DeQue<c2T>();
            DataCopyParams intriParams2{1, static_cast<uint16_t>(curBaseN2 * sizeof(c2T)), 0, 0};
            for (uint32_t loopCnt = 0; loopCnt < tokens; loopCnt++) {
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(yGm[(tokensOffset + loopCnt) * n2 + offset + n2InnerIdx * baseN2], outLocalBias2,
                            intriParams2);
            }
            vecOutQueue.template FreeTensor(outLocalBias2);
        } else {
            LocalTensor<T> inLocalBias2 = vecInQueue.AllocTensor<T>();
            DataCopyParams intriParams1{1, static_cast<uint16_t>(curBaseN2 * sizeof(T)), 0, 0};
            DataCopyPad(inLocalBias2, bias2Gm[expertIdx * n2 + offset + n2InnerIdx * baseN2], intriParams1, padParams);
            vecInQueue.EnQue<T>(inLocalBias2);
            inLocalBias2 = vecInQueue.DeQue<T>();
            LocalTensor<T> outLocalBias2 = vecOutQueue.AllocTensor<T>();
            Adds(outLocalBias2, inLocalBias2, (T)0, curBaseN2);
            vecInQueue.FreeTensor(inLocalBias2);
            vecOutQueue.EnQue<T>(outLocalBias2);
            outLocalBias2 = vecOutQueue.DeQue<T>();
            for (uint32_t loopCnt = 0; loopCnt < tokens; loopCnt++) {
                DataCopyPad(yGm[(tokensOffset + loopCnt) * n2 + offset + n2InnerIdx * baseN2], outLocalBias2,
                            intriParams1);
            }
            vecOutQueue.FreeTensor(outLocalBias2);
        }
    }

    __aicore__ inline void ComputeZeroN1WithBias(uint32_t coreIdx, uint32_t expertIdx)
    {
        uint32_t singleN2;
        if constexpr (IsSameType<c1T, float>::value) {
            singleN2 = Ceil(n2, aivCoreNum);
        } else {
            singleN2 = Ceil(n2, tilingData->ffnBaseParams.coreNum);
        }
        uint32_t baseN2 = tilingData->ffnSingleCoreParams.baseN2;
        if (singleN2 < baseN2) {
            singleN2 = baseN2;
        }
        singleN2 = AlignUp<CUBE_BASE_ALIGN_FACTOR>(singleN2);
        uint32_t offset = singleN2 * coreIdx;
        uint32_t cursingleN2 = singleN2;
        if (singleN2 * coreIdx >= n2) {
            cursingleN2 = 0;
        } else if (n2 - singleN2 * coreIdx < singleN2) {
            cursingleN2 = n2 - singleN2 * coreIdx;
        }
        if (cursingleN2 == 0) {
            return;
        }
        uint32_t n2Loops = (cursingleN2 + baseN2 - 1) / baseN2;
        uint32_t curBaseN2 = baseN2;
        for (uint32_t n2InnerIdx = 0; n2InnerIdx < n2Loops; n2InnerIdx++) {
            if (n2InnerIdx == n2Loops - 1) {
                curBaseN2 = cursingleN2 - n2InnerIdx * baseN2;
            }
            ComputeStepZeroN1WithBias(expertIdx, baseN2, offset, curBaseN2, n2InnerIdx);
        }
    }

    __aicore__ inline bool ProcessZeroN1()
    {
        if (likely(n1 > 0)) {
            return false;
        }
        for (uint32_t expertI = 0; expertI < expertNum; ++expertI) {
            tokens = ubTokens.GetValue(expertI);
            if (tokens == 0) {
                continue;
            }
            if (hasBias2) {
                ComputeZeroN1WithBias(blockIdx, expertI);
            } else {
                ComputeZeroN1WithoutBias(blockIdx, expertI);
            }
            tokensOffset += tokens;
        }
        return true;
    }

    __aicore__ inline void ProcessNormal()
    {
        uint32_t tokensThisLoop = 0; // backup tokens value, bacasue `tokens` will be modified soon after.
        ExpertParallInfo mm1ExpertParallInfo(maxCoreNum, Ceil(n1, tilingData->mm1TilingData.baseN));
        ExpertParallInfo mm2ExpertParallInfo(maxCoreNum, Ceil(n2, tilingData->mm2TilingData.baseN));
        if (mm2ExpertParallInfo.maxSize < mm1ExpertParallInfo.maxSize) {
            // MM1 first computes experts, then MM2. If an expert is not processed by mm1, it cannot be processed by
            // mm2. Expertsin MM1 buffer are all unprocessed, so the buffer of MM2 should hold these experts too. This
            // requires MM2's maxSize >= MM1's maxSize, no matter what relative value of both of maxExpertParallelism.
            mm2ExpertParallInfo.maxSize = mm1ExpertParallInfo.maxSize;
        }
        if (mm2ExpertParallInfo.maxExpertParallelism > mm1ExpertParallInfo.maxExpertParallelism) {
            // Now MM2's expert parallelism is not supported to be larger than MM1's.
            // If it happens, one should consider adjusting workspace1Size and workspace2Size.
            mm2ExpertParallInfo.maxExpertParallelism = mm1ExpertParallInfo.maxExpertParallelism;
            mm2ExpertParallInfo.maxSize = mm1ExpertParallInfo.maxSize;
        }
        MNConfig mnConfig;
        bool firstMM1 = true;
        bool mm2WaitStatus = false;
        bool waitIterateAll = CheckSyncBeforeMM1(mnConfig, mm1ExpertParallInfo.maxExpertParallelism,
                                                 mm2ExpertParallInfo.maxExpertParallelism);

        for (uint32_t expertI(0); expertI < expertNum || mm1ExpertParallInfo.size > 0 || mm2ExpertParallInfo.size > 0;
             ++expertI) {
            tokensOffset += tokensThisLoop; // cannot ignore Step5
            if (likely(expertI < expertNum)) {
                tokensThisLoop = ubTokens.GetValue(expertI);
                if (tokensThisLoop == 0) {
                    continue;
                }
                tokens = tokensThisLoop;
            }
            // Step0: determine expert parallalism and core number for each expert.
            ComputeExpertParallNum(expertI, tilingData->mm1TilingData.baseM, mm1ExpertParallInfo);
            ComputeExpertParallNum(expertI, tilingData->mm1TilingData.baseM, mm2ExpertParallInfo);

            // Step1: mm1
            if (mm1ExpertParallInfo.expertParallelism > 0) {
                MM1Process(mm1ExpertParallInfo, waitIterateAll, mnConfig, mm2WaitStatus, firstMM1);
                // Step2: sync
                SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
                SyncAll<true>();
            }

            // Step3: mm2
            if (mm2ExpertParallInfo.expertParallelism > 0) {
                MM2Process(mm2ExpertParallInfo, waitIterateAll, mnConfig, mm2WaitStatus);
            }

            // Step4: post-process ...
            if (mm1ExpertParallInfo.expertParallelism > 0) {
                mm1ExpertParallInfo.Clear(mm1ExpertParallInfo.start);
            }
            if (mm2ExpertParallInfo.expertParallelism > 0) {
                mm2ExpertParallInfo.Clear(mm2ExpertParallInfo.start);
            }
        }
    }
};
} // namespace FFN

#endif // ASCENDC_FFN_BASE_H
