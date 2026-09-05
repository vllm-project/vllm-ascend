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
 * \file ffn_antiquant.h
 * \brief
 */

#ifndef ASCENDC_FFN_ANTI_QUANT_H
#define ASCENDC_FFN_ANTI_QUANT_H

#include "ffn_kernel.h"

namespace FFN {
/*@brief store variables for castWeight configuration
 */
struct CastWeightConfig {
    uint32_t n = 0;
    uint32_t curSingleK = 0;
    uint32_t curSingleN = 0;
    uint32_t kInOffset = 0;
    uint64_t wInOffset = 0;
    uint64_t wOutOffset = 0;
    uint64_t scaleOffset = 0;
    uint32_t groupSize = 0;
    uint32_t curBaseN = 0;
    uint32_t curBaseK = 0;
    uint32_t alignBaseN = 0;
};

/** @brief FFN Antiquant operator Class
 */
template <typename T, typename wT, typename mm1Type, typename mm2Type = mm1Type, typename c1T = T, typename c2T = T,
          typename biasT = T, bool isPerGroup = false>
class FFNAntiQuant {
public:
    __aicore__ inline FFNAntiQuant(mm1Type &mm1_, mm2Type &mm2_)
        : mm1(mm1_),
          mm2(mm2_)
    {
    }
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                                __gm__ uint8_t *antiQuantScale1, __gm__ uint8_t *antiQuantScale2,
                                __gm__ uint8_t *antiQuantOffset1, __gm__ uint8_t *antiQuantOffset2, __gm__ uint8_t *y,
                                __gm__ uint8_t *workSpace, const FFNTilingData *__restrict tiling, TPipe *tPipe)
    {
        tilingData = tiling;
        pipe = tPipe;
        InitTilingData();

        // init global buffer
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weight1Gm.SetGlobalBuffer((__gm__ int8_t *)weight1);
        if (bias1 != nullptr) {
            bias1Gm.SetGlobalBuffer((__gm__ biasT *)bias1);
            hasBias1 = true;
        }
        weight2Gm.SetGlobalBuffer((__gm__ int8_t *)weight2);
        if (bias2 != nullptr) {
            bias2Gm.SetGlobalBuffer((__gm__ biasT *)bias2);
            hasBias2 = true;
        }

        scale1WorkspaceGm.SetGlobalBuffer((__gm__ T *)antiQuantScale1);
        scale2WorkspaceGm.SetGlobalBuffer((__gm__ T *)antiQuantScale2);
        offset1WorkspaceGm.SetGlobalBuffer((__gm__ T *)antiQuantOffset1);
        offset2WorkspaceGm.SetGlobalBuffer((__gm__ T *)antiQuantOffset2);

        yGm.SetGlobalBuffer((__gm__ c2T *)y);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ c1T *)workSpace);
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ T *)workSpace);
        // init w1 and w2 workspace
        uint64_t offAddr = workspace1Size + workspace2Size;
        w1WorkspaceGm.SetGlobalBuffer((__gm__ T *)(workSpace + offAddr));
        offAddr += (k1 * n1 * dataTypeSize * 2);
        w2WorkspaceGm.SetGlobalBuffer((__gm__ T *)(workSpace + offAddr));

        InitLocalBuff(expertTokens, tPipe);
    }

    __aicore__ inline void Process()
    {
        tokensOffset = 0;
        tokens = 0;
        while (currentExpert < expertNum) {
            tokens = static_cast<uint32_t>(ubTokens.GetValue(currentExpert));
            if (tokens > 0) {
                ComputeExpertSplitMN();
            }
            currentExpert += 1;
            tokensOffset += tokens;
        }
    }

    // define matmul
    mm1Type &mm1;
    mm2Type &mm2;

protected:
    const FFNTilingData *__restrict tilingData;
    TPipe *pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleInQueue;
    TQue<QuePosition::VECIN, 1> offsetInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<int64_t> ubTokens;
    LocalTensor<T> tmpUb;

    GlobalTensor<T> xGm;
    GlobalTensor<int64_t> expertTokensGm;
    GlobalTensor<int8_t> weight1Gm;
    GlobalTensor<biasT> bias1Gm;
    GlobalTensor<int8_t> weight2Gm;
    GlobalTensor<biasT> bias2Gm;
    GlobalTensor<c2T> yGm;

    GlobalTensor<c1T> mm1WorkspaceGm;
    GlobalTensor<T> mm2WorkspaceGm;
    GlobalTensor<T> w1WorkspaceGm;
    GlobalTensor<T> w2WorkspaceGm;

    GlobalTensor<T> scale1WorkspaceGm;
    GlobalTensor<T> scale2WorkspaceGm;
    GlobalTensor<T> offset1WorkspaceGm;
    GlobalTensor<T> offset2WorkspaceGm;

    LocalTensor<T> scaleInUb;
    LocalTensor<T> offsetInUb;

    // tiling data
    uint32_t totalTokens;
    uint32_t maxTokens;
    uint32_t k1;
    uint32_t n1;
    uint32_t k2;
    uint32_t n2;
    uint32_t expertNum;
    uint32_t coreNum;
    uint32_t activeType;
    uint32_t baseM1;
    uint32_t baseN1;
    uint32_t baseN2;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t dataTypeSize;
    uint32_t mm1DataTypeSize;
    uint64_t workspace1Size;
    uint64_t workspace2Size;
    uint32_t antiQuantUbsize;
    uint32_t mm2ResUbSize;
    uint32_t scale1GroupNum;
    uint32_t scale2GroupNum;
    uint32_t scale1GroupSize;
    uint32_t scale2GroupSize;

    bool hasBias1 = false;
    bool hasBias2 = false;
    uint32_t curBlockIdx;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t tokens;
    uint32_t tokensOffset; // tokensOffset = tilingData->ffnBaseParams.tokensArr[0...i-1];
    uint32_t singleM1;
    uint32_t singleM2;
    uint32_t singleM1Tail;
    uint32_t singleM2Tail;
    uint32_t singleN1;
    uint32_t singleN1Tail;
    uint32_t singleN2;
    uint32_t singleN2Tail;
    uint32_t castWeightSingleK1;
    uint32_t castWeightSingleK1Tail;
    uint32_t castWeightSingleK2;
    uint32_t castWeightSingleK2Tail;
    uint32_t castWeightSingleN1;
    uint32_t castWeightSingleN1Tail;
    uint32_t castWeightSingleN2;
    uint32_t castWeightSingleN2Tail;
    uint32_t m1Loops;
    uint32_t m2Loops;
    uint32_t n1Loops;
    uint32_t n2Loops;
    uint32_t castWeightK1Loops;
    uint32_t castWeightK2Loops;
    uint32_t castWeightN1Loops;
    uint32_t castWeightN2Loops;
    uint32_t mInnerLoops;
    uint32_t n1InnerLoops;
    uint32_t maxMLoops;
    uint32_t maxNLoops;
    uint32_t maxUsedCore;
    uint64_t xCoreOffset;
    uint64_t w1CoreOffset;
    uint64_t bias1CoreOffset;
    uint64_t mm2CoreOffset;
    uint64_t w2CoreOffset;
    uint64_t bias2CoreOffset;
    uint64_t outOffset;
    uint64_t activeOffset;
    bool mm2WaitStatue = false;
    uint32_t currentExpert = 0;
    uint32_t bestCopySize = FP16_INT8_BEST_DATACOPY_BASE_SIZE;
    uint32_t reciprocalOfOneByteMultiple = 1;

    __aicore__ inline void InitLocalBuff(__gm__ uint8_t *expertTokens, TPipe *tPipe)
    {
        pipe = tPipe;

        // scale should bigger than singleN, 32 alignment is required
        pipe->InitBuffer(scaleInQueue, 2, bestCopySize * sizeof(T));
        pipe->InitBuffer(offsetInQueue, 2, bestCopySize * sizeof(T));
        pipe->InitBuffer(vecInQueue, 2, ubCalSize * sizeof(c1T));
        pipe->InitBuffer(vecOutQueue, 2, ubCalSize * sizeof(T));
        pipe->InitBuffer(tmpBuff, ubRestBytes);

        tmpUb = tmpBuff.Get<T>();
        ubTokens = GetUbTokens(expertTokens, expertTokensGm, tilingData, pipe);
    }

    __aicore__ inline void InitTilingData()
    {
        curBlockIdx = GetBlockIdx();
        subBlockIdx = GetSubBlockIdx();
        coreIdx = curBlockIdx / GetTaskRation();

        totalTokens = tilingData->ffnBaseParams.totalTokens;
        maxTokens = tilingData->ffnBaseParams.maxTokens;
        k1 = tilingData->ffnBaseParams.k1;
        n1 = tilingData->ffnBaseParams.n1;
        k2 = n1;
        n2 = tilingData->ffnBaseParams.n2;
        expertNum = tilingData->ffnBaseParams.expertNum;
        coreNum = tilingData->ffnBaseParams.coreNum;
        activeType = tilingData->ffnBaseParams.activeType;
        dataTypeSize = sizeof(T);
        mm1DataTypeSize = sizeof(c1T);

        baseM1 = tilingData->ffnSingleCoreParams.baseM1;
        baseN1 = tilingData->ffnSingleCoreParams.baseN1;
        baseN2 = tilingData->ffnSingleCoreParams.baseN2;
        ubCalSize = tilingData->ffnSingleCoreParams.ubCalSize;
        ubRestBytes = tilingData->ffnSingleCoreParams.ubRestBytes;
        workspace1Size = tilingData->ffnBaseParams.workspace1Size;
        workspace2Size = tilingData->ffnBaseParams.workspace2Size;
        scale1GroupNum = tilingData->ffnBaseParams.scale1GroupNum;
        scale2GroupNum = tilingData->ffnBaseParams.scale2GroupNum;
        scale1GroupSize = k1 / scale1GroupNum;
        scale2GroupSize = k2 / scale2GroupNum;

        if constexpr (IsSameType<c1T, float>::value) {
            bestCopySize = BF16_INT8_BEST_DATACOPY_BASE_SIZE;
        }
        if constexpr (IsSameType<wT, int4b_t>::value) {
            reciprocalOfOneByteMultiple = 2; // 2: the reciprocal of half Byte.
        }
    }

    __aicore__ inline void FindCoreSplit(uint32_t m, uint32_t n, uint32_t tilingCoreNum, uint32_t &nLoops,
                                         uint32_t &mLoops)
    {
        uint32_t baseN = nLoops;
        uint32_t baseM = mLoops;
        uint32_t maxNLoops = Ceil(n, baseN);
        uint32_t maxMLoops = Ceil(m, baseM);
        nLoops = maxNLoops;
        mLoops = maxMLoops;
        uint32_t minSingleCore = m * n; // calc loop on the single core
        uint32_t curNLoops = Min(maxNLoops, tilingCoreNum);
        while (curNLoops > 0) {
            uint32_t curSingleN = Ceil(n, curNLoops);
            curSingleN = Ceil(curSingleN, baseN) * baseN;
            curNLoops = Ceil(n, curSingleN);
            if (curNLoops == 0) {
                break;
            }
            uint32_t curMLoops = Min(tilingCoreNum / curNLoops, maxMLoops);
            uint32_t curSingleM = Ceil(m, curMLoops);
            curSingleM = Max(AlignUp<CUBE_BASE_ALIGN_FACTOR>(curSingleM), baseM);
            curSingleM = Min(curSingleM, m);
            curMLoops = Ceil(m, curSingleM);
            uint32_t curSingleCore = curSingleN * curSingleM;
            // select the smaller calc loop on the single core, preferred split N
            if (curSingleCore < minSingleCore ||
                (curSingleCore == minSingleCore && curNLoops * curMLoops < nLoops * mLoops)) {
                nLoops = curNLoops;
                mLoops = curMLoops;
                minSingleCore = curSingleCore;
            }
            // skip curNLoops in range (maxNLoops/2) + 1 to (maxNLoops - 1)
            curNLoops = Min(curNLoops - 1, Ceil(n, curSingleN + baseN));
        }
    }

    __aicore__ inline void KernelTiling(uint32_t baseM, uint32_t baseN, uint32_t Nlength, bool isMatMul1)
    {
        uint32_t nLoops = baseN; // core num used on n axis
        uint32_t mLoops = baseM; // core num used on m axis
        FindCoreSplit(tokens, Nlength, coreNum, nLoops, mLoops);

        uint32_t singleN = Ceil(Nlength, nLoops);
        singleN = AlignUp<CUBE_BASE_ALIGN_FACTOR>(singleN);
        nLoops = Ceil(Nlength, singleN);
        uint32_t singleM = Ceil(tokens, mLoops); // mLoops >= 1
        singleM = AlignUp<CUBE_BASE_ALIGN_FACTOR>(singleM);
        singleM = singleM > baseM ? singleM : baseM;
        singleM = singleM > tokens ? tokens : singleM;
        mLoops = Ceil(tokens, singleM);
        if (isMatMul1) {
            n1Loops = nLoops;
            m1Loops = mLoops;
            singleM1 = singleM;                               // compute C matrix block length along m direction for each cube
            singleN1 = singleN;                               // compute C matrix block length along n direction for each cube
            singleM1Tail = tokens - (m1Loops - 1) * singleM1; // recompute last block length along m direction
            singleN1Tail = n1 - (n1Loops - 1) * singleN1;     // recompute last block length along n direction
            castWeightSingleN1 = n1;
            castWeightN1Loops = 1;
            castWeightSingleN1Tail = n1;
            castWeightK1Loops = coreNum;
            castWeightK1Loops = k1 / castWeightK1Loops > 0 ? castWeightK1Loops : k1;
            castWeightSingleK1 = Ceil(k1, castWeightK1Loops);
            castWeightK1Loops = Ceil(k1, castWeightSingleK1);
            castWeightSingleK1Tail = k1 - (castWeightK1Loops - 1) * castWeightSingleK1;
        } else {
            n2Loops = nLoops;
            m2Loops = mLoops;
            singleM2 = singleM;
            singleN2 = singleN;
            singleM2Tail = tokens - (m2Loops - 1) * singleM2;
            singleN2Tail = n2 - (n2Loops - 1) * singleN2;
            castWeightSingleN2 = n2;
            castWeightN2Loops = 1;
            castWeightSingleN2Tail = n2;
            castWeightK2Loops = coreNum;
            castWeightK2Loops = k2 / castWeightK2Loops > 0 ? castWeightK2Loops : k2;
            castWeightSingleK2 = Ceil(k2, castWeightK2Loops);
            castWeightK2Loops = Ceil(k2, castWeightSingleK2);
            castWeightSingleK2Tail = k2 - (castWeightK2Loops - 1) * castWeightSingleK2;
        }
    }

    __aicore__ inline void ComputeExpertSplitMN()
    {
        KernelTiling(tilingData->mm1TilingData.baseM, tilingData->mm1TilingData.baseN, n1, true);
        SplitMM1();
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
        SyncAll<true>();

        SplitMM2();
    }

    __aicore__ inline void MM1Compute(uint32_t expertIdx, uint64_t OffsetTail)
    {
        mm1.SetTensorA(xGm[xCoreOffset]);
        mm1.SetTensorB(w1WorkspaceGm[w1CoreOffset]);
        if (hasBias1) {
            bias1CoreOffset = expertIdx * n1 + OffsetTail;
            mm1.SetBias(bias1Gm[bias1CoreOffset]);
        }
        mm1.template IterateAll<false>(mm1WorkspaceGm[outOffset], 0, false, true);
    }

    __aicore__ inline void InitActivationFunction(LocalTensor<c1T> activeResUb, uint32_t computeSize,
                                                  uint32_t activeUbOffset)
    {
        LocalTensor<c1T> mm1ResUb = vecInQueue.DeQue<c1T>();
        LocalTensor<uint8_t> tmpLocal = tmpUb[activeUbOffset].template ReinterpretCast<uint8_t>();

        ActiveType active = ActiveType(activeType);
        ApplyActivation(active, activeResUb, mm1ResUb, tmpLocal, computeSize);
        vecInQueue.FreeTensor(mm1ResUb);
    }

    __aicore__ inline void Elewise1Compute(uint32_t computeSize)
    {
        LocalTensor<T> activeResUb = vecOutQueue.AllocTensor<T>();
        uint32_t activeUbOffset = 0;

        if constexpr (IsSameType<c1T, float>::value) {
            LocalTensor<float> activeResUbFp32 = tmpUb.template ReinterpretCast<float>();
            activeUbOffset = computeSize * sizeof(float);
            InitActivationFunction(activeResUbFp32, computeSize, activeUbOffset);
            Cast(activeResUb, activeResUbFp32, RoundMode::CAST_ROUND, computeSize);
        } else {
            InitActivationFunction(activeResUb, computeSize, activeUbOffset);
        }

        vecOutQueue.EnQue<T>(activeResUb);
    }

    __aicore__ inline void Elewise1(uint32_t curSingleM, uint32_t curSingleN1, uint64_t mm1OutOffset,
                                    uint64_t activeOffset)
    {
        // in bf16, vector uses fp32 data.compared with fp16, baseM is reduced by half.
        uint32_t realBaseM1 = ubCalSize / baseN1;
        uint32_t curBaseM = realBaseM1;
        DataCopyPadExtParams<c1T> padParams;
        uint32_t computeBaseN1;
        uint32_t computeSize;
        for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += realBaseM1) {
            if (offsetM + realBaseM1 >= curSingleM) {
                curBaseM = curSingleM - offsetM;
            }
            uint32_t curBaseN1 = baseN1;
            for (uint32_t offsetN = 0; offsetN < curSingleN1; offsetN += baseN1) {
                if (offsetN + baseN1 >= curSingleN1) {
                    curBaseN1 = curSingleN1 - offsetN;
                }
                // mm1 is float16 and 32-byte aligned. mm1 is float32 and 64-byte aligned.
                computeBaseN1 = AlignUp<GetNumInUbBlock<T>()>(curBaseN1);
                computeSize = curBaseM * computeBaseN1;
                // copy mm1 output from workspace
                LocalTensor<c1T> inLocal = vecInQueue.AllocTensor<c1T>();

                DataCopyExtParams intriParams1;
                intriParams1.blockLen = curBaseN1 * mm1DataTypeSize;
                intriParams1.blockCount = curBaseM;
                intriParams1.srcStride = (n1 - curBaseN1) * mm1DataTypeSize;
                intriParams1.dstStride = (computeBaseN1 - curBaseN1) * mm1DataTypeSize / UB_BLOCK_UNIT_SIZE;
                DataCopyPad(inLocal, mm1WorkspaceGm[mm1OutOffset + uint64_t(offsetM) * n1 + offsetN], intriParams1,
                            padParams);
                vecInQueue.EnQue(inLocal);

                Elewise1Compute(computeSize);

                // ResultCopy2GM
                LocalTensor<T> activeResUb = vecOutQueue.DeQue<T>();

                DataCopyExtParams intriParams2;
                intriParams2.blockLen = curBaseN1 * dataTypeSize;
                intriParams2.blockCount = curBaseM;
                intriParams2.srcStride = 0;
                intriParams2.dstStride = (n1 - curBaseN1) * dataTypeSize;
                DataCopyPad(mm2WorkspaceGm[activeOffset + uint64_t(offsetM) * n1 + offsetN], activeResUb, intriParams2);
                vecOutQueue.FreeTensor(activeResUb);
            }
        }
    }

    __aicore__ inline void MM2Compute(uint32_t expertIdx, uint64_t OffsetTail)
    {
        mm2.SetTensorA(mm2WorkspaceGm[mm2CoreOffset]);
        mm2.SetTensorB(w2WorkspaceGm[w2CoreOffset]);

        if (hasBias2) {
            bias2CoreOffset = expertIdx * n2 + OffsetTail;
            mm2.SetBias(bias2Gm[bias2CoreOffset]);
        }
        mm2.template IterateAll<false>(yGm[outOffset], 0, false, true);
        mm2WaitStatue = true;
    }

    __aicore__ inline void SplitMM1()
    {
        uint32_t tokensOffsetInner = tokensOffset;
        uint32_t curSingleM = singleM1;
        // calc mm1 tiling
        uint32_t m1Idx = coreIdx / n1Loops; // 0 <= m1Idx < 2, m1Idx=0 is FIRST expert, m1Idx=1 is SECOND expert.
        uint32_t n1Idx = coreIdx % n1Loops;
        uint32_t curSingleN1 = singleN1;
        uint32_t expertIdx = currentExpert;
        uint32_t tilingCoreNum = n1Loops * m1Loops;
        bool isValidCore = (coreIdx < tilingCoreNum && subBlockIdx == 0);
        uint64_t OffsetTail = n1Idx * singleN1;
        if (isValidCore) {
            if (n1Idx == n1Loops - 1) {
                curSingleN1 = singleN1Tail;
            }
            if (m1Idx == m1Loops - 1) {
                curSingleM = singleM1Tail;
            }
            tokensOffsetInner = tokensOffset + m1Idx * singleM1;
            outOffset = uint64_t(m1Idx * singleM1) * n1 + OffsetTail;

            mm1.SetOrgShape(tokens, n1, k1);
            mm1.SetSingleShape(curSingleM, curSingleN1, k1);
            xCoreOffset = uint64_t(tokensOffsetInner) * k1;
            w1CoreOffset = OffsetTail;
        }
        if (coreIdx < castWeightN1Loops * castWeightK1Loops) {
            CalcOffsetAndCastWeight(true);
        }
        SyncBeforeMM1();
        SyncAll<true>();
        if (isValidCore) {
            MM1Compute(expertIdx, OffsetTail);
        }
        KernelTiling(tilingData->mm2TilingData.baseM, tilingData->mm2TilingData.baseN, n2, false);
        if (coreIdx < castWeightN2Loops * castWeightK2Loops) {
            CalcOffsetAndCastWeight(false);
        }
        if (isValidCore) {
            mm1.WaitIterateAll();
            mm1.End();
            activeOffset = workspace1Size / dataTypeSize + outOffset;
            Elewise1(curSingleM, curSingleN1, outOffset, activeOffset);
        }
    }

    __aicore__ inline void SplitMM2()
    {
        uint32_t curSingleN2 = singleN2;
        uint32_t expertIdx = currentExpert;
        tokens = static_cast<uint32_t>(ubTokens.GetValue(expertIdx));
        uint32_t tilingCoreNum = n2Loops * m2Loops;
        uint32_t m2Idx = coreIdx / n2Loops;
        uint32_t n2Idx = coreIdx % n2Loops;
        uint32_t curSingleM = singleM2;
        uint64_t OffsetTail = n2Idx * singleN2;

        if (coreIdx < tilingCoreNum) {
            curSingleN2 = singleN2;
            if (m2Idx == m2Loops - 1) {
                curSingleM = singleM2Tail;
            }
            if (n2Idx == n2Loops - 1) {
                curSingleN2 = singleN2Tail;
            }

            w2CoreOffset = OffsetTail;
        }
        if (coreIdx < tilingCoreNum && subBlockIdx == 0) {
            // mm2 compute
            mm2.SetOrgShape(tokens, n2, k2);
            mm2.SetSingleShape(curSingleM, curSingleN2, k2);
            mm2CoreOffset = workspace1Size / dataTypeSize + uint64_t(m2Idx * singleM2) * k2;
            outOffset = uint64_t(tokensOffset + m2Idx * singleM2) * n2 + OffsetTail;
            MM2Compute(expertIdx, OffsetTail);
        }
    }

    __aicore__ inline void CastWeightCompute(uint32_t curCalcK, uint32_t curCalcAlignN, LocalTensor<T> scaleInUb,
                                             LocalTensor<T> offsetInUb)
    {
        LocalTensor<wT> w1InUb = vecInQueue.DeQue<wT>();
        w1InUb.SetSize(curCalcK * curCalcAlignN);
        LocalTensor<T> w1ResUb = vecOutQueue.AllocTensor<T>();

        LocalTensor<uint8_t> tmpLocal = tmpUb.template ReinterpretCast<uint8_t>();
        AntiQuantShapeInfo shapeInfo = {1, curCalcAlignN, 1, curCalcAlignN};
        // fp16 tempbuff is 0, bf16 tempbuff = offset.GetSize() * 2 * sizeof(float) + 64 * K * sizeof(float)
        AscendAntiQuant<wT, T, false>(w1ResUb, w1InUb, offsetInUb, scaleInUb, tmpLocal, curCalcK, shapeInfo);

        vecInQueue.FreeTensor(w1InUb);
        vecOutQueue.EnQue<T>(w1ResUb);
    }

    __aicore__ inline void DataCopyAndComputeW(uint32_t offsetN, uint32_t offsetK, GlobalTensor<int8_t> weightGm,
                                               GlobalTensor<T> wWorkspaceGm, CastWeightConfig &castWeightCfg)
    {
        // copy mm1 output from workspace
        LocalTensor<int8_t> inLocal = vecInQueue.AllocTensor<int8_t>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen =
            castWeightCfg.curBaseN / reciprocalOfOneByteMultiple; // int4 weight are copied based on int8 type
        intriParams1.blockCount = castWeightCfg.curBaseK;
        intriParams1.srcStride = (castWeightCfg.n - castWeightCfg.curBaseN) / reciprocalOfOneByteMultiple;
        intriParams1.dstStride = 0;
        DataCopyPadExtParams<int8_t> padParams;
        DataCopyPad(
            inLocal,
            weightGm[(castWeightCfg.wInOffset + offsetK * castWeightCfg.n + offsetN) / reciprocalOfOneByteMultiple],
            intriParams1, padParams);
        vecInQueue.EnQue(inLocal);

        CastWeightCompute(castWeightCfg.curBaseK, castWeightCfg.alignBaseN, scaleInUb, offsetInUb);

        // ResultCopy2GM
        LocalTensor<T> w1ResUb = vecOutQueue.DeQue<T>();

        DataCopyExtParams intriParams2;
        intriParams2.blockLen = castWeightCfg.curBaseN * dataTypeSize;
        intriParams2.blockCount = castWeightCfg.curBaseK;
        intriParams2.srcStride = (castWeightCfg.alignBaseN - castWeightCfg.curBaseN) / GetNumInUbBlock<T>();
        intriParams2.dstStride = (castWeightCfg.n - castWeightCfg.curBaseN) * dataTypeSize;
        DataCopyPad(wWorkspaceGm[castWeightCfg.wOutOffset + offsetK * castWeightCfg.n + offsetN], w1ResUb,
                    intriParams2);
        vecOutQueue.FreeTensor(w1ResUb);
    }

    __aicore__ inline void SelectCastWeight(GlobalTensor<int8_t> weightGm, GlobalTensor<T> wWorkspaceGm,
                                            GlobalTensor<T> scaleWorkspaceGm, GlobalTensor<T> offsetWorkspaceGm,
                                            CastWeightConfig &castWeightCfg)
    {
        if constexpr (isPerGroup == true) {
            CastWeightPerGroup(weightGm, wWorkspaceGm, scaleWorkspaceGm, offsetWorkspaceGm, castWeightCfg);
        } else {
            CastWeightNormal(weightGm, wWorkspaceGm, scaleWorkspaceGm, offsetWorkspaceGm, castWeightCfg);
        }
    }

    __aicore__ inline void CastWeightPerGroup(GlobalTensor<int8_t> weightGm, GlobalTensor<T> wWorkspaceGm,
                                              GlobalTensor<T> scaleWorkspaceGm, GlobalTensor<T> offsetWorkspaceGm,
                                              CastWeightConfig &castWeightCfg)
    {
        uint32_t newBaseN = bestCopySize;
        // ensure when cast weight, newBaseN align to 32, compute size will not larger than ubCalSize
        uint32_t newBaseK = Min(ubCalSize / newBaseN, castWeightCfg.groupSize);
        castWeightCfg.curBaseN = newBaseN;
        uint32_t usedGroupSize = castWeightCfg.groupSize * (castWeightCfg.kInOffset / castWeightCfg.groupSize + 1);
        uint32_t subCoreCount = 0;

        for (uint32_t offsetN = 0; offsetN < castWeightCfg.curSingleN; offsetN += newBaseN) {
            if (offsetN + newBaseN >= castWeightCfg.curSingleN) {
                castWeightCfg.curBaseN = castWeightCfg.curSingleN - offsetN;
            }
            castWeightCfg.alignBaseN =
                AlignUp(castWeightCfg.curBaseN, UB_BLOCK_UNIT_SIZE * reciprocalOfOneByteMultiple);
            uint32_t curUsedGroupSize = usedGroupSize;
            for (uint32_t offsetK = 0; offsetK < castWeightCfg.curSingleK; offsetK += castWeightCfg.curBaseK) {
                if (unlikely(offsetK + newBaseK + castWeightCfg.kInOffset >= curUsedGroupSize)) {
                    castWeightCfg.curBaseK = curUsedGroupSize - offsetK - castWeightCfg.kInOffset;
                    curUsedGroupSize += castWeightCfg.groupSize;
                } else if (unlikely(offsetK + newBaseK >= castWeightCfg.curSingleK)) {
                    castWeightCfg.curBaseK = castWeightCfg.curSingleK - offsetK;
                } else {
                    castWeightCfg.curBaseK = newBaseK;
                }

                subCoreCount += 1;
                if (subBlockIdx == subCoreCount % 2) { // 2: enable both subcores to cast weight
                    continue;
                }

                DataCopyScaleAndOffset(offsetN, castWeightCfg.kInOffset + offsetK, scaleWorkspaceGm, offsetWorkspaceGm,
                                       castWeightCfg);
                DataCopyAndComputeW(offsetN, offsetK, weightGm, wWorkspaceGm, castWeightCfg);
                scaleInQueue.FreeTensor(scaleInUb);
                offsetInQueue.FreeTensor(offsetInUb);
            }
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }

    __aicore__ inline void CastWeightNormal(GlobalTensor<int8_t> weightGm, GlobalTensor<T> wWorkspaceGm,
                                            GlobalTensor<T> scaleWorkspaceGm, GlobalTensor<T> offsetWorkspaceGm,
                                            CastWeightConfig &castWeightCfg)
    {
        uint32_t newBaseN = bestCopySize;
        // ensure when cast weight, newBaseN align to 32, compute size will not larger than ubCalSize
        uint32_t newBaseK = ubCalSize / newBaseN;
        castWeightCfg.curBaseN = newBaseN;
        uint32_t subCoreCount = 0;

        for (uint32_t offsetN = 0; offsetN < castWeightCfg.curSingleN; offsetN += newBaseN) {
            if (offsetN + newBaseN >= castWeightCfg.curSingleN) {
                castWeightCfg.curBaseN = castWeightCfg.curSingleN - offsetN;
            }
            castWeightCfg.alignBaseN =
                AlignUp(castWeightCfg.curBaseN, UB_BLOCK_UNIT_SIZE * reciprocalOfOneByteMultiple);
            DataCopyScaleAndOffset(offsetN, 0, scaleWorkspaceGm, offsetWorkspaceGm, castWeightCfg);
            castWeightCfg.curBaseK = newBaseK;
            for (uint32_t offsetK = 0; offsetK < castWeightCfg.curSingleK; offsetK += castWeightCfg.curBaseK) {
                if (unlikely(offsetK + newBaseK >= castWeightCfg.curSingleK)) {
                    castWeightCfg.curBaseK = castWeightCfg.curSingleK - offsetK;
                }
                subCoreCount++;
                if (subBlockIdx == subCoreCount % 2) { // 2: enable both subcores to cast weight
                    continue;
                }

                DataCopyAndComputeW(offsetN, offsetK, weightGm, wWorkspaceGm, castWeightCfg);
            }
            scaleInQueue.FreeTensor(scaleInUb);
            offsetInQueue.FreeTensor(offsetInUb);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }

    __aicore__ inline void CalcOffsetAndCastWeight(bool isMatMul1)
    {
        uint32_t kIdx;
        uint32_t nIdx;
        uint64_t wCoreOffsetKMulN;
        CastWeightConfig castWeightCfg;
        castWeightCfg.curSingleN = isMatMul1 ? castWeightSingleN1 : castWeightSingleN2;
        castWeightCfg.curSingleK = isMatMul1 ? castWeightSingleK1 : castWeightSingleK2;
        if (isMatMul1) {
            kIdx = coreIdx / castWeightN1Loops;
            nIdx = coreIdx % castWeightN1Loops;
            if (nIdx == castWeightN1Loops - 1) {
                castWeightCfg.curSingleN = castWeightSingleN1Tail;
            }
            if (kIdx == castWeightK1Loops - 1) {
                castWeightCfg.curSingleK = castWeightSingleK1Tail;
            }
            uint32_t alignCastWeightSingleN1 = AlignUp(castWeightSingleN1, reciprocalOfOneByteMultiple);
            castWeightCfg.n = n1;
            castWeightCfg.kInOffset = kIdx * castWeightSingleK1;
            wCoreOffsetKMulN = k1 * n1;
            castWeightCfg.wInOffset = currentExpert * wCoreOffsetKMulN + nIdx * alignCastWeightSingleN1 +
                                      static_cast<uint64_t>(castWeightCfg.kInOffset) * n1;
            castWeightCfg.wOutOffset = static_cast<uint64_t>(castWeightCfg.kInOffset) * n1 + nIdx * castWeightSingleN1;
            castWeightCfg.scaleOffset =
                static_cast<uint64_t>(currentExpert) * n1 * scale1GroupNum + nIdx * castWeightSingleN1;
            castWeightCfg.groupSize = scale1GroupSize;
            SelectCastWeight(weight1Gm, w1WorkspaceGm, scale1WorkspaceGm, offset1WorkspaceGm, castWeightCfg);
        } else {
            kIdx = coreIdx / castWeightN2Loops;
            nIdx = coreIdx % castWeightN2Loops;
            if (nIdx == castWeightN2Loops - 1) {
                castWeightCfg.curSingleN = castWeightSingleN2Tail;
            }
            if (kIdx == castWeightK2Loops - 1) {
                castWeightCfg.curSingleK = castWeightSingleK2Tail;
            }
            uint32_t alignCastWeightSingleN2 = AlignUp(castWeightSingleN2, reciprocalOfOneByteMultiple);
            castWeightCfg.n = n2;
            castWeightCfg.kInOffset = kIdx * castWeightSingleK2;
            wCoreOffsetKMulN = k2 * n2;
            castWeightCfg.wInOffset = currentExpert * wCoreOffsetKMulN + nIdx * alignCastWeightSingleN2 +
                                      static_cast<uint64_t>(castWeightCfg.kInOffset) * n2;
            castWeightCfg.wOutOffset = static_cast<uint64_t>(castWeightCfg.kInOffset) * n2 + nIdx * castWeightSingleN2;
            castWeightCfg.scaleOffset =
                static_cast<uint64_t>(currentExpert) * n2 * scale2GroupNum + nIdx * castWeightSingleN2;
            castWeightCfg.groupSize = scale2GroupSize;
            SelectCastWeight(weight2Gm, w2WorkspaceGm, scale2WorkspaceGm, offset2WorkspaceGm, castWeightCfg);
        }
    }

    __aicore__ inline void DataCopyScaleAndOffset(uint32_t offsetN, uint32_t offsetK, GlobalTensor<T> scaleWorkspaceGm,
                                                  GlobalTensor<T> offsetWorkspaceGm, CastWeightConfig &castWeightCfg)
    {
        uint64_t realScaleOffset = castWeightCfg.scaleOffset + offsetN;
        if constexpr (isPerGroup == true) {
            realScaleOffset += (offsetK / castWeightCfg.groupSize * castWeightCfg.n);
        }

        // copy scale and offset from GM
        DataCopyPadParams padParams;
        DataCopyParams scaleParams;
        scaleParams.blockLen = castWeightCfg.curBaseN * dataTypeSize;
        scaleParams.blockCount = 1;
        scaleParams.srcStride = 0;
        scaleParams.dstStride = 0;
        LocalTensor<T> scaleLocal = scaleInQueue.AllocTensor<T>();
        DataCopyPad(scaleLocal, scaleWorkspaceGm[realScaleOffset], scaleParams, padParams);
        scaleInQueue.EnQue(scaleLocal);

        LocalTensor<T> offsetLocal = offsetInQueue.AllocTensor<T>();
        DataCopyPad(offsetLocal, offsetWorkspaceGm[realScaleOffset], scaleParams, padParams);
        offsetInQueue.EnQue(offsetLocal);

        scaleInUb = scaleInQueue.DeQue<T>();
        scaleInUb.SetSize(castWeightCfg.alignBaseN);
        offsetInUb = offsetInQueue.DeQue<T>();
        offsetInUb.SetSize(castWeightCfg.alignBaseN);
    }

    __aicore__ inline void SyncBeforeMM1()
    {
        if (mm2WaitStatue) {
            mm2.WaitIterateAll();
            mm2.End();
            mm2WaitStatue = false;
        }
    }
};
} // namespace FFN

#endif // ASCENDC_FFN_ANTI_QUANT_H
