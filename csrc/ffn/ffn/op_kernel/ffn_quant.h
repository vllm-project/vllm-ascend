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
 * \file ffn_quant.h
 * \brief
 */

#ifndef ASCENDC_FFN_QUANT_H
#define ASCENDC_FFN_QUANT_H

#include "ffn_kernel.h"

namespace FFN {
template <typename T, typename mm1Type, typename mm2Type, typename c1T, typename c2T, typename biasT, typename actT,
          typename dequantT, bool isSmooth = false>
class FFNQuant {
public:
    __aicore__ inline FFNQuant(mm1Type &mm1_, mm2Type &mm2_)
        : mm1(mm1_),
          mm2(mm2_){};
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                                __gm__ uint8_t *scale, __gm__ uint8_t *offset, __gm__ uint8_t *deqScale1,
                                __gm__ uint8_t *deqScale2, __gm__ uint8_t *y, __gm__ uint8_t *workSpace,
                                const FFNTilingData *__restrict tiling, TPipe *tPipe)
    {
        curBlockIdx = GetBlockIdx();
        subBlockIdx = GetSubBlockIdx();
        coreIdx = curBlockIdx / GetTaskRation();
        tilingData = tiling;
        pipe = tPipe;
        InitTilingData();

        // init global buffer
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weight1Gm.SetGlobalBuffer((__gm__ T *)weight1);
        if (bias1 != nullptr) {
            hasBias1 = true;
            bias1Gm.SetGlobalBuffer((__gm__ biasT *)bias1);
        }
        weight2Gm.SetGlobalBuffer((__gm__ T *)weight2);
        if (bias2 != nullptr) {
            hasBias2 = true;
            bias2Gm.SetGlobalBuffer((__gm__ biasT *)bias2);
        }
        yGm.SetGlobalBuffer((__gm__ c2T *)y);
        mm1WorkspaceGm.SetGlobalBuffer((__gm__ c1T *)workSpace);
        mm2WorkspaceGm.SetGlobalBuffer((__gm__ T *)workSpace);
        mm2OutWorkspaceGm.SetGlobalBuffer((__gm__ c1T *)workSpace);

        if constexpr (isSmooth) {
            ScaleGm.SetGlobalBuffer((__gm__ float *)scale);
        } else {
            quantScale = (__gm__ float *)scale;
        }
        quantOffset = (__gm__ float *)offset;

        InitDequantScale(workSpace, deqScale1, deqScale2);

        InitQueue();
        ubTokens = GetUbTokens(expertTokens, expertTokensGm, tilingData, pipe);
    }
    /** main logical function
     */
    __aicore__ inline void Process()
    {
        if (unlikely(this->ProcessZeroN1())) {
            return;
        }

        tokensOffset = 0;
        uint32_t tokensBak = 0; // backup tokens value, bacasue `tokens` will be modified soon after.
        ExpertParallInfo mm1ExpertParallInfo(
            cubeCoreNum, Ceil(n1, tilingData->mm1TilingData.baseN * tilingData->mm1TilingData.stepN));
        ExpertParallInfo mm2ExpertParallInfo(
            cubeCoreNum, Ceil(n2, tilingData->mm2TilingData.baseN * tilingData->mm2TilingData.stepN));
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
        bool whetherFirstMM1 = true;
        WhetherSyncBeforeMM1(mnConfig, mm1ExpertParallInfo.maxExpertParallelism,
                             mm2ExpertParallInfo.maxExpertParallelism);

        for (uint32_t expertI(0); expertI < expertNum || mm1ExpertParallInfo.size > 0 || mm2ExpertParallInfo.size > 0;
             ++expertI) {
            tokensOffset += tokensBak; // cannot ignore Step5
            if (likely(expertI < expertNum)) {
                tokensBak = ubTokens.GetValue(expertI);
                if (tokensBak == 0) {
                    continue;
                }
                tokens = tokensBak;
            }
            // Step0: determine expert parallalism and core number for each expert.
            ComputeExpertParallNum(expertI, tilingData->mm1TilingData.baseM, mm1ExpertParallInfo);
            ComputeExpertParallNum(expertI, tilingData->mm1TilingData.baseM, mm2ExpertParallInfo);

            // Step1: mm1
            if (mm1ExpertParallInfo.expertParallelism > 0) {
                MM1Process(mm1ExpertParallInfo, mnConfig, whetherFirstMM1);
                // Step2: sync
                SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
                SyncAll<true>();
            }

            // Step3: mm2
            if (mm2ExpertParallInfo.expertParallelism > 0) {
                MM2Process(mm2ExpertParallInfo, mnConfig, whetherWaitMM2);
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

    // define matmul
    mm1Type &mm1;
    mm2Type &mm2;

protected:
    const FFNTilingData *__restrict tilingData;
    TPipe *pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> scaleQueue;

    TBuf<TPosition::VECCALC> dequantOutBuf;
    LocalTensor<dequantT> dequantTmpOut;

    LocalTensor<int64_t> ubTokens;
    LocalTensor<actT> dequantOut;
    LocalTensor<actT> actOut;
    LocalTensor<uint8_t> actTmp;
    LocalTensor<uint8_t> quantTmp;
    LocalTensor<uint8_t> tmpBuff;
    LocalTensor<uint32_t> gatherIndex;
    GlobalTensor<T> xGm;
    GlobalTensor<int64_t> expertTokensGm;
    GlobalTensor<T> weight1Gm;
    GlobalTensor<biasT> bias1Gm;
    GlobalTensor<T> weight2Gm;
    GlobalTensor<biasT> bias2Gm;
    GlobalTensor<c2T> yGm;
    GlobalTensor<c1T> mm1WorkspaceGm;
    GlobalTensor<T> mm2WorkspaceGm;
    GlobalTensor<c1T> mm2OutWorkspaceGm;

    GlobalTensor<dequantT> deqScale1Gm;
    GlobalTensor<dequantT> deqScale2Gm;

    GlobalTensor<uint64_t> deqScale1UInt64Gm;
    GlobalTensor<uint64_t> deqScale2UInt64Gm;

    GlobalTensor<uint32_t> deqScale1FloatGm;
    GlobalTensor<uint32_t> deqScale2FloatGm;
    GlobalTensor<uint32_t> deqScale1GmVector;
    GlobalTensor<uint32_t> deqScale2GmVector;

    __gm__ float *quantScale;
    __gm__ float *quantOffset;
    GlobalTensor<float> ScaleGm;

    // tiling data
    uint32_t totalTokens;
    uint32_t maxTokens;
    uint32_t k1;
    uint32_t n1;
    uint32_t n2;
    uint32_t k2;
    uint32_t expertNum;
    uint32_t coreNum; // number of aiv
    uint32_t cubeCoreNum;
    uint32_t activeType;
    uint32_t baseM1;
    uint32_t baseN1;
    uint32_t baseM2;
    uint32_t baseN2;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t dataTypeSize;
    uint32_t outTypeSize;
    uint32_t mmOutTypeSize;
    uint64_t workspace1Size;
    uint64_t workspace2Size;
    uint32_t mm2ResUbSize;

    bool hasBias1 = false;
    bool hasBias2 = false;
    uint32_t tokens;
    uint32_t tokensOffset; // tokensOffset = tilingData->ffnBaseParams.tokensArr[0...i-1];
    uint32_t curBlockIdx;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t singleM1;
    uint32_t singleM2;
    uint32_t singleM1Tail;
    uint32_t singleM2Tail;
    uint32_t singleN1;
    uint32_t singleN2;
    uint32_t singleN1Tail;
    uint32_t singleN2Tail;
    uint32_t m1Loops;
    uint32_t m2Loops;
    uint32_t n1Loops;
    uint32_t n2Loops;
    uint32_t mInnerLoops;
    uint32_t n1InnerLoops;
    uint32_t maxMLoops;
    uint32_t maxNLoops;
    uint32_t maxUsedCore;
    uint64_t xCoreOffset;
    uint64_t w1CoreOffset;
    uint64_t w2CoreOffset;
    uint64_t bias1CoreOffset;
    uint64_t bias2CoreOffset;
    uint64_t mm2CoreOffset;
    uint64_t outOffset;
    uint64_t activeOffset;
    int32_t twoExpertIdsWithSmallShape[2] = {-1, -1}; // store expert Ids for parallelism
    uint32_t tokensCum[2] = {0, 0};                   // store tokensOffset for two parallel experts
    uint32_t expertNumI = 0;
    bool whetherWaitMM2 = false;
    bool whetherSyncBeforeMM1 = true;

    uint32_t deqscale1Offset;
    uint32_t scaleOffset;
    uint32_t deqscale2Offset;
    uint64_t mm2OutOffset;
    uint32_t dequantCountNum;
    uint32_t dequantParamsNum;

    /** init function for dequantscale.
     */
    __aicore__ inline void InitDequantScale(__gm__ uint8_t *workSpace, __gm__ uint8_t *deqScale1,
                                            __gm__ uint8_t *deqScale2)
    {
        if constexpr (IsSameType<dequantT, float>::value) {
            deqScale1GmVector.SetGlobalBuffer((__gm__ uint32_t *)deqScale1);
            deqScale2GmVector.SetGlobalBuffer((__gm__ uint32_t *)deqScale2);
            deqScale1UInt64Gm.SetGlobalBuffer((__gm__ uint64_t *)workSpace);
            deqScale2UInt64Gm.SetGlobalBuffer((__gm__ uint64_t *)workSpace);
            deqScale1FloatGm.SetGlobalBuffer((__gm__ uint32_t *)workSpace);
            deqScale2FloatGm.SetGlobalBuffer((__gm__ uint32_t *)workSpace);
        }
        deqScale1Gm.SetGlobalBuffer((__gm__ dequantT *)deqScale1);
        deqScale2Gm.SetGlobalBuffer((__gm__ dequantT *)deqScale2);
    }

    /** init function for Queue.
     */
    __aicore__ inline void InitQueue()
    {
        if (n1 != 0) {
            pipe->InitBuffer(vecInQueue, 1, ubCalSize * sizeof(c1T));
            if constexpr (IsSameType<c2T, bfloat16_t>::value) {
                // for mm2 output(int32), init size not less than 8*baseN for float32 deq_scale
                pipe->InitBuffer(vecOutQueue, 1, ubCalSize * sizeof(c2T));
            } else {
                // for quant output(int8), init size not less than 8*baseN for float32 deq_scale
                pipe->InitBuffer(vecOutQueue, 1, ubCalSize * sizeof(T));
            }
            if constexpr (IsSameType<dequantT, uint64_t>::value == false || isSmooth) {
                uint32_t scaleSize = AlignUp<UB_BLOCK_UNIT_SIZE>(baseN1 * sizeof(float));
                pipe->InitBuffer(scaleQueue, 1, scaleSize);
                ubRestBytes -= scaleSize;
            }
            TBuf<TPosition::VECCALC> tmpTBuf;
            pipe->InitBuffer(tmpTBuf, ubRestBytes);
            tmpBuff = tmpTBuf.Get<uint8_t>();
            actOut = tmpBuff.ReinterpretCast<actT>();
            actTmp = tmpBuff[ubCalSize * sizeof(actT)];
            quantTmp = actTmp;
            if constexpr (IsSameType<dequantT, bfloat16_t>::value) {
                dequantOut = actOut[ubCalSize];
                actTmp = tmpBuff[ubCalSize * sizeof(actT) * 2]; // 2: double
            }
        } else {
            pipe->InitBuffer(vecInQueue, 1, AlignUp<UB_BLOCK_UNIT_SIZE>(baseN2 * sizeof(biasT)));
            pipe->InitBuffer(vecOutQueue, 1, AlignUp<UB_BLOCK_UNIT_SIZE>(baseN2 * sizeof(c2T)));
            pipe->InitBuffer(scaleQueue, 1, AlignUp<UB_BLOCK_UNIT_SIZE>(baseN2 * sizeof(dequantT)));
            if constexpr (IsSameType<c2T, half>::value && IsSameType<dequantT, float>::value) {
                pipe->InitBuffer(dequantOutBuf, AlignUp<UB_BLOCK_UNIT_SIZE>(baseN2 * sizeof(dequantT)));
                dequantTmpOut = dequantOutBuf.Get<dequantT>();
            }
        }
    }

    /** init function for TilingData of mm1 and mm2.
     */
    __aicore__ inline void InitTilingData()
    {
        totalTokens = tilingData->ffnBaseParams.totalTokens;
        maxTokens = tilingData->ffnBaseParams.maxTokens;
        k1 = tilingData->ffnBaseParams.k1;
        n1 = tilingData->ffnBaseParams.n1;
        k2 = n1;
        n2 = tilingData->ffnBaseParams.n2;
        expertNum = tilingData->ffnBaseParams.expertNum;
        cubeCoreNum = tilingData->ffnBaseParams.coreNum;
        coreNum = cubeCoreNum * GetTaskRation();
        activeType = tilingData->ffnBaseParams.activeType;
        dataTypeSize = sizeof(T);
        outTypeSize = sizeof(c2T);
        mmOutTypeSize = sizeof(c1T);

        baseM1 = tilingData->ffnSingleCoreParams.baseM1;
        baseN1 = tilingData->ffnSingleCoreParams.baseN1;
        baseN2 = tilingData->ffnSingleCoreParams.baseN2;
        ubCalSize = tilingData->ffnSingleCoreParams.ubCalSize;
        ubRestBytes = tilingData->ffnSingleCoreParams.ubRestBytes;
        workspace1Size = tilingData->ffnBaseParams.workspace1Size;
        workspace2Size = tilingData->ffnBaseParams.workspace2Size;
    }

    /** function for Activation function. Currently, it contains FASTGELU, RELU and SILU.
     */
    __aicore__ inline void ActivationFunction(LocalTensor<actT> activeResUb, LocalTensor<actT> actInput,
                                              uint32_t computeSize)
    {
        ActiveType active = ActiveType(activeType);
        ApplyActivation(active, activeResUb, actInput, actTmp, computeSize);
    }

    /** Determine mm1Experts value for mm1
     * @param expertI: current expert I.
     * @param baseM: baseM for mm;
     * @param expertParalInfo: expert parallel info
     */
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

    /** Decide whether need to do syncall before mm1.
     * @param mnConfig: M and N config info for matmul.
     * @param maxMM1ExpertParallelism: max mm1 expert parallel num.
     * @param maxMM2ExpertParallelism: max mm2 expert parallel num.
     */
    __aicore__ inline void WhetherSyncBeforeMM1(MNConfig &mnConfig, uint32_t maxMM1ExpertParallelism,
                                                uint32_t maxMM2ExpertParallelism)
    {
        UpdateKernelTilingInfo info{n1, n2, expertNum, maxMM1ExpertParallelism};

        uint32_t maxMM1UsedCubeCore;
        UpdateKernelTilingBeforeMM1(mnConfig, maxMM1UsedCubeCore, tokens, info, tilingData);

        uint32_t minMM2UsedCubeCore = mnConfig.blockDimM * mnConfig.blockDimN;
        if (minMM2UsedCubeCore >= maxMM1UsedCubeCore && n1 > INT8_SYNC_N1_SIZE) {
            whetherSyncBeforeMM1 = false;
        }
    }

    // /** main computation function for experts with n1 > 0.
    // * @param expertIdxFlag1: current expert id that `expertNmI` pointing to.
    // * @param mm1Experts: number of experts for mm1 at this cycle.
    // * @param whetherFirstMM1: judgement of whether this mm1 is the first mm1 in this execution of FFN,
    // */
    // __aicore__ inline void ComputeExpertSplitMN(uint32_t expertIdxFlag1, uint32_t mm1Experts, bool& whetherFirstMM1);

    /** Do SyncAll before MM1
     * @param whetherFirstMM1: judgement of whether this mm1 is the first mm1 in this execution of FFN,
     */
    __aicore__ inline void SyncBeforeMM1(bool whetherFirstMM1)
    {
        if constexpr (IsSameType<c1T, half>::value) {
            if (whetherWaitMM2) {
                mm2.WaitIterateAll();
                mm2.End();
                whetherWaitMM2 = false;
            }
        }

        if (whetherSyncBeforeMM1 && !whetherFirstMM1) {
            SyncAll<true>();
        }
    }

    /** mm1 computation function
     * @param mnConfig: M and N config info for matmul.
     * @param baseBlockIdx: current block idx.
     * @param expertIdx: current expert idx for calculation.
     * @param tokensOffset: token offset for current expert.
     * @param outRowOffset: out offset for mm1.
     */
    __aicore__ inline void MM1Compute(MNConfig &mnConfig, uint32_t baseBlockIdx, uint32_t expertIdx,
                                      uint32_t tokensOffset, uint32_t outRowOffset)
    {
        uint32_t mIdx = baseBlockIdx / mnConfig.blockDimN;
        uint32_t nIdx = baseBlockIdx % mnConfig.blockDimN;
        uint32_t tailN = nIdx * mnConfig.singleN;
        uint32_t curSingleM =
            (mIdx == mnConfig.blockDimM - 1) ? (mnConfig.m - mIdx * mnConfig.singleM) : mnConfig.singleM;
        uint32_t curSingleN = (nIdx == mnConfig.blockDimN - 1) ? (mnConfig.n - tailN) : mnConfig.singleN;
        uint64_t outOffset = uint64_t(outRowOffset + mIdx * mnConfig.singleM) * n1 + tailN;
        uint64_t xCoreOffset = uint64_t(tokensOffset + mIdx * mnConfig.singleM) * k1;
        uint64_t w1CoreOffset = expertIdx * (uint64_t)k1 * n1 + tailN;

        mm1.SetOrgShape(mnConfig.m, n1, k1);
        mm1.SetSingleShape(curSingleM, curSingleN, k1);
        mm1.SetTensorA(xGm[xCoreOffset]);
        mm1.SetTensorB(weight1Gm[w1CoreOffset]);
        if (hasBias1) {
            mm1.SetBias(bias1Gm[expertIdx * n1 + tailN]);
        } else {
            mm1.ClearBias();
        }
        if constexpr (isSmooth) {
            scaleOffset = expertIdx * n1 + tailN;
        }
        if constexpr (IsSameType<c1T, half>::value) {
            if constexpr (IsSameType<dequantT, float>::value) {
                CastDeqScale(expertIdx * n1 + tailN, n1, curSingleM, curSingleN, true);
                mm1.SetQuantVector(
                    deqScale1UInt64Gm[(workspace1Size + workspace2Size) / sizeof(uint64_t) + expertIdx * n1 + tailN]);
            } else {
                mm1.SetQuantVector(deqScale1Gm[expertIdx * n1 + tailN]);
            }
        } else {
            deqscale1Offset = expertIdx * n1 + tailN;
        }
        mm1.template IterateAll<true>(mm1WorkspaceGm[outOffset], false);
        mm1.End();
        activeOffset = workspace1Size / dataTypeSize + outOffset;
        Elewise1(curSingleM, curSingleN, outOffset, activeOffset, expertIdx);
    }

    /** mm1 process function
     * @param mm1ExpertParallInfo: mm1 expert parallel info.
     * @param mnConfig: M and N config info for matmul..
     * @param whetherFirstMM1: whether the first mm1 for calculation.
     */
    __aicore__ inline void MM1Process(ExpertParallInfo &mm1ExpertParallInfo, MNConfig &mnConfig, bool &whetherFirstMM1)
    {
        uint32_t coreNumEachExpert = cubeCoreNum / mm1ExpertParallInfo.expertParallelism;
        mnConfig.SetConstriant(tokens, n1, tilingData->mm1TilingData.baseM,
                               tilingData->mm1TilingData.baseN * tilingData->mm1TilingData.stepN, coreNumEachExpert);
        KernelTiling(mnConfig);
        coreNumEachExpert = mnConfig.blockDimM * mnConfig.blockDimN;
        size_t expertOrderInBuf =
            Min<uint32_t>(mm1ExpertParallInfo.start + coreIdx / coreNumEachExpert, mm1ExpertParallInfo.maxSize - 1);
        // make sure which expert each core/cube needs to compute
        uint32_t expertIMM = mm1ExpertParallInfo.expertIdxBuf[expertOrderInBuf];
        tokens = ubTokens.GetValue(expertIMM);
        SyncBeforeMM1(whetherFirstMM1);

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
        whetherFirstMM1 = false;
    }

    /** quant computation function
     */
    __aicore__ inline void QuantCompute(uint32_t computeSize, uint32_t expertIdx, uint32_t computeBaseN1)
    {
        // quant compute
        uint32_t tmpsize = AlignUp<UB_BLOCK_UNIT_SIZE>(computeSize);
        if constexpr (isSmooth && IsSameType<c1T, int32_t>::value) {
            // bf16 per-channel mode
            LocalTensor<float> scaleSrcUb = scaleQueue.DeQue<float>();
            LocalTensor<T> quantOutUb = vecOutQueue.AllocTensor<T>();
            LocalTensor<half> srcUbFp16 = quantTmp.template ReinterpretCast<half>();
            LocalTensor<half> scalecUbFp16 = srcUbFp16[ubCalSize];
            LocalTensor<uint8_t> quantTmp1 = scalecUbFp16[computeBaseN1].template ReinterpretCast<uint8_t>();
            Cast(srcUbFp16, actOut, RoundMode::CAST_NONE, tmpsize);
            PipeBarrier<PIPE_V>();
            Cast(scalecUbFp16, scaleSrcUb, RoundMode::CAST_NONE, computeBaseN1);
            PipeBarrier<PIPE_V>();
            AscendQuant(quantOutUb, srcUbFp16, quantTmp1, scalecUbFp16, static_cast<half>(quantOffset[expertIdx]),
                        computeBaseN1, tmpsize);
            scaleQueue.FreeTensor(scaleSrcUb);
            vecOutQueue.EnQue<T>(quantOutUb);
        } else if constexpr (isSmooth && IsSameType<c1T, half>::value) {
            // fp16 per-channel mode
            LocalTensor<float> scaleSrcUb = scaleQueue.DeQue<float>();
            LocalTensor<half> scaleSrcUbFp16 = tmpBuff.ReinterpretCast<half>();
            scaleSrcUbFp16 = scaleSrcUbFp16[ubCalSize];
            quantTmp = tmpBuff[ubCalSize * sizeof(half) + computeBaseN1 * sizeof(half)];
            Cast(scaleSrcUbFp16, scaleSrcUb, RoundMode::CAST_NONE, computeBaseN1);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> quantOutUb = vecOutQueue.AllocTensor<T>();
            AscendQuant(quantOutUb, actOut, quantTmp, scaleSrcUbFp16, static_cast<half>(quantOffset[expertIdx]),
                        computeBaseN1, tmpsize);
            scaleQueue.FreeTensor(scaleSrcUb);
            vecOutQueue.EnQue<T>(quantOutUb);
        } else {
            // bf16 and fp16 per-tensor mode
            LocalTensor<T> quantOutUb = vecOutQueue.AllocTensor<T>();
            AscendQuant(quantOutUb, actOut, quantTmp, quantScale[expertIdx], quantOffset[expertIdx], tmpsize);
            vecOutQueue.EnQue<T>(quantOutUb);
        }
    }

    /** copy dequant from gm to ub function
     */
    __aicore__ inline void DequantDataCopy(uint32_t curBaseN1)
    {
        LocalTensor<dequantT> dequantLocal = scaleQueue.AllocTensor<dequantT>();
        DataCopyExtParams intriParams3;
        intriParams3.blockLen = curBaseN1 * outTypeSize;
        intriParams3.blockCount = 1;
        intriParams3.srcStride = (n1 - curBaseN1) * outTypeSize;
        intriParams3.dstStride = 0;
        DataCopyPadExtParams<dequantT> padParams;
        DataCopyPad(dequantLocal, deqScale1Gm[deqscale1Offset], intriParams3, padParams);
        scaleQueue.EnQue(dequantLocal);
    }

    /** copy scale from gm to ub function
     */
    __aicore__ inline void ScaleDataCopy(uint32_t curBaseN1, uint32_t offsetN)
    {
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        DataCopyExtParams intriParams;
        intriParams.blockLen = curBaseN1 * sizeof(float);
        intriParams.blockCount = 1;
        intriParams.srcStride = (n1 - curBaseN1) * sizeof(float);
        intriParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        DataCopyPad(scaleLocal, ScaleGm[scaleOffset + offsetN], intriParams, padParams);
        scaleQueue.EnQue(scaleLocal);
    }

    __aicore__ inline void CopyFromWorkspaceMM1(uint32_t curBaseM, uint32_t curBaseN1, uint32_t computeBaseN1,
                                                uint64_t offsetWorkspace, const DataCopyPadExtParams<c1T> &padParams)
    {
        LocalTensor<c1T> inLocal = vecInQueue.AllocTensor<c1T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = curBaseM;
        copyParams.blockLen = curBaseN1 * mmOutTypeSize;
        copyParams.srcStride = (n1 - curBaseN1) * mmOutTypeSize;
        copyParams.dstStride = (computeBaseN1 - curBaseN1) * mmOutTypeSize / UB_BLOCK_UNIT_SIZE;
        DataCopyPad(inLocal, mm1WorkspaceGm[offsetWorkspace], copyParams, padParams);
        vecInQueue.EnQue(inLocal);
    }

    __aicore__ inline void CopyToWorksapceMM2(uint32_t curBaseM, uint32_t curBaseN1, uint64_t offsetWorkspace)
    {
        LocalTensor<T> activeResUb = vecOutQueue.DeQue<T>();

        DataCopyExtParams copy2GmParams;
        copy2GmParams.blockCount = curBaseM;
        copy2GmParams.blockLen = curBaseN1 * dataTypeSize;
        copy2GmParams.srcStride = 0;
        copy2GmParams.dstStride = (n1 - curBaseN1) * dataTypeSize;
        DataCopyPad(mm2WorkspaceGm[offsetWorkspace], activeResUb, copy2GmParams);
        vecOutQueue.FreeTensor(activeResUb);
    }

    /** Entry point to Elewise activation function, containing vector tiling stage.
     * @param curSingleM: input matrix M size.
     * @param curSingleN1: input matrix N size.
     * @param mm1OutOffset: matrix address offset.
     * @param activeOffset: activation address offset.
     * @param expertIdx: current expert id at this cycle.
     */
    __aicore__ inline void Elewise1(uint32_t curSingleM, uint32_t curSingleN1, uint64_t mm1OutOffset,
                                    uint64_t activeOffset, uint32_t expertIdx)
    {
        uint32_t curBaseM = baseM1;
        DataCopyPadExtParams<c1T> padParams;
        uint32_t computeBaseN1;
        uint32_t computeSize;
        for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += baseM1) {
            if (offsetM + baseM1 >= curSingleM) {
                curBaseM = curSingleM - offsetM;
            }
            uint32_t curBaseN1 = baseN1;
            for (uint32_t offsetN = 0; offsetN < curSingleN1; offsetN += baseN1) {
                if (offsetN + baseN1 >= curSingleN1) {
                    curBaseN1 = curSingleN1 - offsetN;
                }
                computeBaseN1 = AlignUp<UB_BLOCK_UNIT_SIZE>(curBaseN1);
                computeSize = curBaseM * computeBaseN1;

                CopyFromWorkspaceMM1(curBaseM, curBaseN1, computeBaseN1,
                                     mm1OutOffset + uint64_t(offsetM) * n1 + offsetN, padParams);

                LocalTensor<actT> actInput;
                if constexpr (IsSameType<c1T, int32_t>::value) {
                    DequantDataCopy(curBaseN1);

                    dequantCountNum = curBaseM * computeBaseN1;
                    dequantParamsNum = computeBaseN1;
                    CastCompute(true); // cast int32 to float
                    actInput = dequantOut;
                } else {
                    actInput = vecInQueue.DeQue<actT>();
                }
                if constexpr (isSmooth) {
                    ScaleDataCopy(curBaseN1, offsetN);
                }
                ActivationFunction(actOut, actInput, computeSize);
                if constexpr (IsSameType<c1T, int32_t>::value == false) {
                    vecInQueue.FreeTensor(actInput);
                }
                QuantCompute(computeSize, expertIdx, computeBaseN1);

                CopyToWorksapceMM2(curBaseM, curBaseN1, activeOffset + uint64_t(offsetM) * n1 + offsetN);
            }
        }
    }

    /** Entry point to mm2 cast int32 to bf16 function, containing vector tiling stage.
     * @param curSingleM: input matrix M size.
     * @param curSingleN2: input matrix N size.
     * @param mm2OutOffset: matrix address offset.
     * @param activeOffset: out address offset.
     * @param expertIdx: current expert id at this cycle.
     */
    __aicore__ inline void Elewise2(uint32_t curSingleM, uint32_t curSingleN2, uint64_t mm2OutOffset,
                                    uint64_t activeOffset, uint32_t expertIdx)
    {
        uint32_t curBaseM = baseM1;
        DataCopyPadExtParams<c1T> padParams;
        DataCopyPadExtParams<dequantT> deqPadParams;
        uint32_t computeBaseN2;
        for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += baseM1) {
            if (offsetM + baseM1 >= curSingleM) {
                curBaseM = curSingleM - offsetM;
            }
            uint32_t curBaseN2 = baseN1;
            for (uint32_t offsetN = 0; offsetN < curSingleN2; offsetN += baseN1) {
                if (offsetN + baseN1 >= curSingleN2) {
                    curBaseN2 = curSingleN2 - offsetN;
                }
                computeBaseN2 = AlignUp<GetNumInUbBlock<c2T>()>(curBaseN2); // pad
                // copy mm1 output from workspace
                LocalTensor<c1T> inLocal = vecInQueue.AllocTensor<c1T>();

                DataCopyExtParams intriParams1;
                intriParams1.blockLen = curBaseN2 * mmOutTypeSize;
                intriParams1.blockCount = curBaseM;
                intriParams1.srcStride = (n2 - curBaseN2) * mmOutTypeSize;
                intriParams1.dstStride = (computeBaseN2 - curBaseN2) * mmOutTypeSize / UB_BLOCK_UNIT_SIZE;
                DataCopyPad(inLocal, mm2OutWorkspaceGm[mm2OutOffset + uint64_t(offsetM) * n2 + offsetN], intriParams1,
                            padParams);
                vecInQueue.EnQue(inLocal);

                LocalTensor<dequantT> dequantLocal = scaleQueue.AllocTensor<dequantT>();
                DataCopyExtParams intriParams3;
                intriParams3.blockLen = curBaseN2 * outTypeSize;
                intriParams3.blockCount = 1;
                intriParams3.srcStride = (n2 - curBaseN2) * outTypeSize;
                intriParams3.dstStride = 0;
                DataCopyPad(dequantLocal, deqScale2Gm[deqscale2Offset], intriParams3, deqPadParams);
                scaleQueue.EnQue(dequantLocal);

                dequantCountNum = curBaseM * computeBaseN2;
                dequantParamsNum = computeBaseN2;
                CastCompute(false); // cast int32 to bf16

                // output from buffer to gm
                LocalTensor<c2T> quantSrcUb = vecOutQueue.DeQue<c2T>();

                DataCopyExtParams intriParams2;
                intriParams2.blockLen = curBaseN2 * outTypeSize;
                intriParams2.blockCount = curBaseM;
                intriParams2.srcStride = 0;
                intriParams2.dstStride = (n2 - curBaseN2) * outTypeSize;
                DataCopyPad(yGm[activeOffset + uint64_t(offsetM) * n2 + offsetN], quantSrcUb, intriParams2);
                vecOutQueue.FreeTensor(quantSrcUb);
            }
        }
    }

    /**  mm2 computation function
     */
    __aicore__ inline void MM2Compute(MNConfig &mnConfig, uint32_t baseBlockIdx, uint32_t expertIdx,
                                      uint32_t tokensRowOffset, uint32_t outRowOffset)
    {
        uint32_t mIdx = baseBlockIdx / mnConfig.blockDimN;
        uint32_t nIdx = baseBlockIdx % mnConfig.blockDimN;
        uint32_t tailN = nIdx * mnConfig.singleN;
        uint32_t curSingleN = (nIdx == mnConfig.blockDimN - 1) ? (mnConfig.n - tailN) : mnConfig.singleN;
        uint32_t curSingleM =
            (mIdx == mnConfig.blockDimM - 1) ? (mnConfig.m - mIdx * mnConfig.singleM) : mnConfig.singleM;
        uint64_t outOffset = uint64_t(outRowOffset + mIdx * mnConfig.singleM) * (uint64_t)n2 + tailN;
        uint64_t xCoreOffset = workspace1Size / dataTypeSize + uint64_t(tokensRowOffset + mIdx * mnConfig.singleM) * k2;
        uint64_t w2CoreOffset = expertIdx * (uint64_t)k2 * n2 + tailN;

        mm2.SetOrgShape(mnConfig.m, n2, k2);
        mm2.SetSingleShape(curSingleM, curSingleN, k2);
        mm2.SetTensorA(mm2WorkspaceGm[xCoreOffset]);
        mm2.SetTensorB(weight2Gm[w2CoreOffset]);
        if (hasBias2) {
            mm2.SetBias(bias2Gm[expertIdx * n2 + tailN]);
        } else {
            mm2.ClearBias();
        }
        if constexpr (IsSameType<c1T, half>::value) {
            if constexpr (IsSameType<dequantT, float>::value) {
                CastDeqScale(expertIdx * n2 + tailN, n2, curSingleM, curSingleN, false);
                mm2.SetQuantVector(
                    deqScale2UInt64Gm[(workspace1Size + workspace2Size + expertNum * n1 * sizeof(uint64_t)) /
                                          sizeof(uint64_t) +
                                      expertIdx * n2 + tailN]);
            } else {
                mm2.SetQuantVector(deqScale2Gm[expertIdx * n2 + tailN]);
            }
            mm2.template IterateAll<false>(yGm[outOffset], 0, false, whetherSyncBeforeMM1);
        } else {
            deqscale2Offset = expertIdx * n2 + tailN;
            mm2OutOffset = (workspace1Size + workspace2Size) / mmOutTypeSize + outOffset;
            mm2.template IterateAll<true>(mm2OutWorkspaceGm[mm2OutOffset], false);
            Elewise2(curSingleM, curSingleN, mm2OutOffset, outOffset, expertIdx);
        }
    }

    /**  mm2 process function
     */
    __aicore__ inline void MM2Process(ExpertParallInfo &mm2ExpertParallInfo, MNConfig &mnConfig, bool &whetherWaitMM2)
    {
        for (uint32_t i = mm2ExpertParallInfo.start; i < mm2ExpertParallInfo.size;
             i += mm2ExpertParallInfo.expertParallelism) {
            if (i + mm2ExpertParallInfo.expertParallelism > mm2ExpertParallInfo.size) {
                mm2ExpertParallInfo.expertParallelism = mm2ExpertParallInfo.size - i;
            }
            uint32_t coreNumEachExpert = cubeCoreNum / mm2ExpertParallInfo.expertParallelism;

            if (coreIdx >= coreNumEachExpert * mm2ExpertParallInfo.expertParallelism || subBlockIdx != 0) {
                continue;
            }
            uint32_t expertOrderInBuf = i + coreIdx / coreNumEachExpert;
            uint32_t expertIMM = mm2ExpertParallInfo.expertIdxBuf[expertOrderInBuf];
            tokens = ubTokens.GetValue(expertIMM);
            mnConfig.SetConstriant(tokens, n2, tilingData->mm2TilingData.baseM,
                                   tilingData->mm2TilingData.baseN * tilingData->mm2TilingData.stepN,
                                   coreNumEachExpert);
            KernelTiling(mnConfig);
            uint32_t baseBlockIdx = coreIdx % coreNumEachExpert;
            coreNumEachExpert = mnConfig.blockDimM * mnConfig.blockDimN;
            if (baseBlockIdx < coreNumEachExpert) {
                uint32_t tokensRowOffset = mm2ExpertParallInfo.LocalOffset[expertOrderInBuf] -
                                           mm2ExpertParallInfo.LocalOffset[mm2ExpertParallInfo.start];
                uint32_t outRowOffset = mm2ExpertParallInfo.GlobalOffset[expertOrderInBuf];
                if constexpr (IsSameType<c1T, half>::value) {
                    ControlMM2();
                }
                MM2Compute(mnConfig, baseBlockIdx, expertIMM, tokensRowOffset, outRowOffset);
                whetherWaitMM2 = whetherSyncBeforeMM1;
            }
        }
    }

    /** Control mm2 WaitIterateAll and end.
     */
    __aicore__ inline void ControlMM2()
    {
        if (whetherWaitMM2) {
            mm2.WaitIterateAll();
            mm2.End();
            whetherWaitMM2 = false;
        }
    }

    /** int32 cast to float function computation for mm1, cast to bf16 for mm2
     * @param computeSize: the number of elements to be activated.
     */
    __aicore__ inline void CastCompute(bool isMM1)
    {
        // if mm1 :cast int32 to float, if mm2: cast int32 to bf16
        LocalTensor<c1T> mmResUb = vecInQueue.DeQue<c1T>();
        LocalTensor<dequantT> dequantUb = scaleQueue.DeQue<dequantT>();
        if (isMM1) {
            AscendDequant(dequantOut, mmResUb, dequantUb, actTmp, {1, dequantCountNum, dequantParamsNum});
            vecInQueue.FreeTensor(mmResUb);
            scaleQueue.FreeTensor(dequantUb);
        } else {
            LocalTensor<c2T> castUb = vecOutQueue.AllocTensor<c2T>();
            AscendDequant(castUb, mmResUb, dequantUb, actTmp, {1, dequantCountNum, dequantParamsNum});
            vecInQueue.FreeTensor(mmResUb);
            scaleQueue.FreeTensor(dequantUb);
            vecOutQueue.EnQue<c2T>(castUb);
        }
    }

    __aicore__ inline void CreateIndex(uint32_t indexNum)
    {
        LocalTensor<int32_t> tmpS81 = tmpBuff.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> tmpS82 = tmpS81[indexNum];
        LocalTensor<uint32_t> tmpU81 = tmpBuff.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tmpU82 = tmpU81[indexNum];
        int firstValue = 0;
        CreateVecIndex(tmpS81, firstValue, indexNum);
        PipeBarrier<PIPE_V>();
        uint32_t scalarValue = 1;
        ShiftRight(tmpU82, tmpU81, scalarValue, indexNum);
        PipeBarrier<PIPE_V>();
        int32_t scalar = 4;
        Muls(tmpS81, tmpS82, scalar, indexNum);
        PipeBarrier<PIPE_V>();
        gatherIndex = tmpS81.ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void CastDeqScale(uint32_t offset, uint32_t nLength, uint32_t singleM, uint32_t singleN,
                                        bool isMM1)
    {
        DataCopyPadParams padParams;
        // 8 = 2(uint64 need double length of float) * sizeof(float),
        // should be limited by the size of vecOutQueue;
        uint32_t baseNAlign = AlignDown(ubCalSize / 8, UB_BLOCK_UNIT_SIZE);

        uint32_t baseN = baseNAlign;
        for (uint32_t offsetN = 0; offsetN < singleN; offsetN += baseNAlign) {
            if ((singleN - offsetN) < baseNAlign) {
                baseN = singleN - offsetN;
                baseNAlign = AlignUp<UB_BLOCK_UNIT_SIZE>(baseN);
            }
            CreateIndex(2 * baseNAlign); // 2: double

            LocalTensor<uint32_t> dequantInitLocal = vecOutQueue.AllocTensor<uint32_t>();
            uint32_t scalarZeroValue = 0;
            Duplicate(dequantInitLocal, scalarZeroValue, 2 * baseN); // 2: double
            PipeBarrier<PIPE_V>();

            LocalTensor<uint32_t> dequantLocal = vecInQueue.AllocTensor<uint32_t>();
            DataCopyParams intriParams1;
            intriParams1.blockLen = baseN * sizeof(uint32_t);
            intriParams1.blockCount = 1;
            intriParams1.srcStride = 0;
            intriParams1.dstStride = 0;
            DataCopyPad(dequantLocal, isMM1 ? deqScale1GmVector[offset + offsetN] : deqScale2GmVector[offset + offsetN],
                        intriParams1, padParams);
            vecInQueue.EnQue(dequantLocal);

            uint64_t mask[2] = {0x5555555555555555, 0}; // 0x5555555555555555: mask bits, 1 is to enable computation
            LocalTensor<uint32_t> dequantSrcLocal = vecInQueue.DeQue<uint32_t>();
            Gather(dequantInitLocal, dequantSrcLocal, gatherIndex, 0, mask,
                   (uint8_t)(baseNAlign / (DATABLOCK_NUM_IN_GATHER * sizeof(dequantT))), (uint16_t)(sizeof(uint64_t)));
            PipeBarrier<PIPE_ALL>();
            vecInQueue.FreeTensor(dequantSrcLocal);

            DataCopyParams intriParams2;
            intriParams2.blockLen = 2 * baseN * sizeof(uint32_t); // 2: double
            intriParams2.blockCount = 1;
            intriParams2.srcStride = 0;
            intriParams2.dstStride = 0;

            vecOutQueue.EnQue(dequantInitLocal);
            LocalTensor<uint32_t> dequantDstLocal = vecOutQueue.DeQue<uint32_t>();
            if (isMM1) {
                uint64_t deqscale1Offset =
                    (workspace1Size + workspace2Size) / sizeof(uint32_t) + (offset + offsetN) * 2;
                DataCopyPad(deqScale1FloatGm[deqscale1Offset], dequantDstLocal, intriParams2);
            } else {
                uint64_t deqscale2Offset =
                    (workspace1Size + workspace2Size + expertNum * n1 * sizeof(uint64_t)) / sizeof(uint32_t) +
                    (offset + offsetN) * 2;
                DataCopyPad(deqScale2FloatGm[deqscale2Offset], dequantDstLocal, intriParams2);
            }
            vecOutQueue.FreeTensor(dequantDstLocal);
        }
    }

    __aicore__ inline void DequantDataCopyForZeroN1(uint32_t curBaseN2, uint32_t offset,
                                                    const DataCopyPadParams &padParams)
    {
        LocalTensor<dequantT> dequantLocal = scaleQueue.AllocTensor<dequantT>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curBaseN2 * sizeof(dequantT);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(dequantLocal, deqScale2Gm[offset], copyParams, padParams);
        scaleQueue.EnQue(dequantLocal);
    }

    __aicore__ inline void ComputeZeroN1WithoutBias(uint32_t expertIdx)
    {
        singleM1 = Ceil(tokens, tilingData->ffnBaseParams.coreNum);
        uint32_t tokensByPrevCores = singleM1 * coreIdx;
        if (tokens <= tokensByPrevCores) {
            return;
        }
        uint32_t tokensRemaining = tokens - tokensByPrevCores;
        uint32_t curSingleM = (tokensRemaining > singleM1) ? singleM1 : tokensRemaining;
        InitOutput<c2T>(yGm[(tokensOffset + tokensByPrevCores) * n2], curSingleM * n2, 0);
    }

    __aicore__ inline void ComputeZeroN1WithBias(uint32_t expertIdx)
    {
        singleN2 = Ceil(n2, tilingData->ffnBaseParams.coreNum);
        if (singleN2 < baseN2) {
            singleN2 = baseN2;
        }
        if (n2 <= singleN2 * coreIdx) {
            return;
        }
        uint32_t offset = singleN2 * coreIdx;
        uint32_t n2Remaining = n2 - singleN2 * coreIdx;
        uint32_t curSingleN2 = (n2Remaining > singleN2) ? singleN2 : n2Remaining;
        uint32_t n2Loops = Ceil(curSingleN2, baseN2);
        uint32_t curBaseN2 = baseN2;
        DataCopyPadParams padParams;
        for (uint32_t n2InnerIdx = 0; n2InnerIdx < n2Loops; n2InnerIdx++) {
            if (n2InnerIdx == n2Loops - 1) {
                curBaseN2 = curSingleN2 - n2InnerIdx * baseN2;
            }

            uint32_t n2InnerOffset = expertIdx * n2 + offset + n2InnerIdx * baseN2;
            DequantDataCopyForZeroN1(curBaseN2, n2InnerOffset, padParams);

            LocalTensor<biasT> inLocalBias2 = vecInQueue.AllocTensor<biasT>();
            DataCopyParams copyParamsBias2{1, static_cast<uint16_t>(curBaseN2 * sizeof(biasT)), 0, 0};
            DataCopyPad(inLocalBias2, bias2Gm[n2InnerOffset], copyParamsBias2, padParams);
            vecInQueue.EnQue<biasT>(inLocalBias2);

            inLocalBias2 = vecInQueue.DeQue<biasT>();
            LocalTensor<dequantT> dequantUb = scaleQueue.DeQue<dequantT>();
            LocalTensor<c2T> outLocalBias2 = vecOutQueue.AllocTensor<c2T>();
            if constexpr (IsSameType<c2T, half>::value && IsSameType<dequantT, float>::value) {
                AscendDequant(dequantTmpOut, inLocalBias2, dequantUb, {1, baseN2, curBaseN2});
                Cast(outLocalBias2, dequantTmpOut, RoundMode::CAST_NONE, curBaseN2);
            } else {
                AscendDequant(outLocalBias2, inLocalBias2, dequantUb, {1, baseN2, curBaseN2});
            }
            scaleQueue.FreeTensor(dequantUb);
            vecInQueue.FreeTensor(inLocalBias2);
            vecOutQueue.EnQue<c2T>(outLocalBias2);

            outLocalBias2 = vecOutQueue.DeQue<c2T>();
            DataCopyParams copyParamsOut{1, static_cast<uint16_t>(curBaseN2 * sizeof(c2T)), 0, 0};
            for (uint32_t tokensIdx = 0; tokensIdx < tokens; tokensIdx++) {
                DataCopyPad(yGm[(tokensOffset + tokensIdx) * n2 + offset + n2InnerIdx * baseN2], outLocalBias2,
                            copyParamsOut);
            }
            vecOutQueue.FreeTensor(outLocalBias2);
        }
    }

    __aicore__ inline bool ProcessZeroN1()
    {
        if (likely(this->n1 > 0)) {
            return false;
        }
        if (subBlockIdx != 0) {
            return true;
        }
        tokensOffset = 0;
        for (uint32_t expertIdx = 0; expertIdx < expertNum; ++expertIdx) {
            tokens = ubTokens.GetValue(expertIdx);
            if (tokens == 0) {
                continue;
            }
            if (hasBias2) {
                ComputeZeroN1WithBias(expertIdx);
            } else {
                ComputeZeroN1WithoutBias(expertIdx);
            }
            tokensOffset += tokens;
        }
        return true;
    }
};
} // namespace FFN

#endif // ASCENDC_FFN_QUANT_H
