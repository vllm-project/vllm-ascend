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
 * \file ffn_antiquant_msd.h
 * \brief
 */

#ifndef ASCENDC_FFN_ANTIQUANT_MSD_H
#define ASCENDC_FFN_ANTIQUANT_MSD_H

#include "ffn_base.h"
#include "ffn_kernel.h"

namespace FFN {
struct TilingConfig {
    uint32_t mCube;
    uint32_t mVec;
    uint32_t k;
    uint32_t n;
    uint32_t vecBaseK;
    uint32_t vecBaseM;
    uint32_t vecSingleM;
    uint32_t vecSingleMTail;
    uint32_t vecBlockDimK;
    uint32_t vecBlockDimM;
    uint32_t cubeBaseM;
    uint32_t cubeBaseN;
    uint32_t aicNumPerExpert;
    uint32_t aivNumPerExpert;
    uint32_t cubeBlockDimM;
    uint32_t cubeBlockDimN;
    uint32_t cubeSingleM;
    uint32_t cubeSingleN;
    uint32_t cubeSingleNTail;
    bool hasBias;

    __aicore__ inline void SetBaseParams(const uint32_t mCube_, const uint32_t mVec_, const uint32_t n_,
                                         const uint32_t k_, bool hasBias_)
    {
        mCube = mCube_;
        mVec = mVec_;
        n = n_;
        k = k_;
        hasBias = hasBias_;
    }

    __aicore__ inline void SetTilingParams(const uint32_t baseM_, const uint32_t baseN_,
                                           const uint32_t aicNumPerExpert_, const uint32_t aivNumPerExpert_)
    {
        cubeBaseM = baseM_;
        cubeBaseN = baseN_;
        aicNumPerExpert = aicNumPerExpert_;
        aivNumPerExpert = aivNumPerExpert_;
    }
};

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t fullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (fullDim0 - dim0) * sizeof(T);
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    if (dim0 % (UB_BLOCK_UNIT_SIZE / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = matmul::CeilAlign(dim0, static_cast<uint32_t>(32 / sizeof(T))) - dim0;
        padParams.paddingValue = 0;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcFullDim0 - dim0) * sizeof(T) / static_cast<uint32_t>(ONE_BLK_SIZE);
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
class FFNAntiQuantMSD {
public:
    __aicore__ inline FFNAntiQuantMSD(mm1Type &mm1_, mm2Type &mm2_)
        : mm1(mm1_),
          mm2(mm2_)
    {
    }
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                                __gm__ uint8_t *antiQuantScale1, __gm__ uint8_t *antiQuantScale2,
                                __gm__ uint8_t *antiQuantOffset1, __gm__ uint8_t *antiQuantOffset2, __gm__ uint8_t *y,
                                __gm__ uint8_t *mm1Workspace, const FFNTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitTilingData();
    __aicore__ inline void GetMaxToken(__gm__ uint8_t *expertTokens);
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workSpace);
    __aicore__ inline void AllocLocalTensor(TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void PreProcessMM1(TilingConfig tilingParams, ExpertParallInfo mmExpertParallInfo);
    __aicore__ inline void PreProcessMM2(uint32_t offsetM, uint32_t curBaseK, TilingConfig tilingParams,
                                         uint32_t &syncCount, ExpertParallInfo mmExpertParallInfo);
    __aicore__ inline void MM1VectorTiling(TilingConfig &tilingParams);
    __aicore__ inline void MM2VectorTiling(uint32_t curV2BaseN, uint32_t blockDimK, uint32_t v2BaseM,
                                           TilingConfig &tilingParams);
    __aicore__ inline void CalcReduceMax(uint32_t gmReduceOffset, uint32_t curBaseM, uint32_t curBaseK,
                                         GlobalTensor<float> reduceMaxWorkspaceGm);
    __aicore__ inline void CalcReduceSum(uint32_t gmReduceOffset, uint32_t curBaseM, uint32_t curBaseK,
                                         GlobalTensor<float> reduceSumWorkspaceGm);
    __aicore__ inline void CalcA1A2(const TilingConfig &tilingParams, uint32_t offsetM, uint32_t curBaseM,
                                    uint32_t curBaseK, uint32_t aOffsetGm, GlobalTensor<int8_t> workspaceAMatrixGm);
    __aicore__ inline void MM1Compute(uint32_t curCubeSingleCoreN, uint32_t offsetN, const TilingConfig &tilingParams,
                                      ExpertParallInfo mmExpertParallInfo);
    __aicore__ inline void MM2Compute(uint32_t curCubeSingleCoreN, uint32_t offsetN, TilingConfig tilingParams,
                                      ExpertParallInfo mmExpertParallInfo, uint32_t curIterCount);
    __aicore__ inline void SaveAmaxInUb(uint32_t m, GlobalTensor<float> reduceMaxWorkspaceGm,
                                        ExpertParallInfo mmExpertParallInfo, uint32_t expertIdxInParaGroupMM);
    __aicore__ inline void CopyInBias(uint32_t curV2BaseN, uint32_t offsetAndScaleOffset, TilingConfig tilingParam,
                                      GlobalTensor<biasT> biasGm);
    __aicore__ inline void CopyOutFinalResult(uint32_t curV2BaseM, uint32_t curV2BaseN, uint32_t yOffset, uint32_t n);
    __aicore__ inline void ComputeExpert(ExpertParallInfo mm1ExpertParallInfo, ExpertParallInfo mm2ExpertParallInfo);
    __aicore__ inline void SyncAllByCount(uint32_t count);
    __aicore__ inline void ComputeExpertMM1(TilingConfig &tilingParams1, TilingConfig &tilingParams2,
                                            ExpertParallInfo mm1ExpertParallInfo);
    __aicore__ inline void ComputeExpertMM2(TilingConfig &tilingParams, ExpertParallInfo mm2ExpertParallInfo);
    __aicore__ inline void CubeResultPostProcess(TilingConfig &tilingParams2, ExpertParallInfo mm2ExpertParallInfo,
                                                 const uint32_t curIterCount, const uint32_t preExpertUsedCoreInIter,
                                                 const uint32_t idx);
    __aicore__ inline void ActivationCompute(uint32_t computeSize);
    __aicore__ inline void Elewise1(uint32_t curV2BaseM, uint32_t curV2BaseN);
    __aicore__ inline void CubeTiling(TilingConfig &tilingParams);
    __aicore__ inline void ComputeExpertParallNum(const uint32_t expertI, ExpertParallInfo &expertParallInfo);
    __aicore__ inline void CalcTailBlock(TilingConfig tilingParams, uint32_t &curSingleM, uint32_t &curBaseK);
    __aicore__ inline void CopyOriginInput(const TilingConfig &tilingParams, uint32_t curBaseM, uint32_t curBaseK,
                                           uint32_t xGmOffset);
    __aicore__ inline void CalcResDataUsedInPostProcessMM1(uint32_t offsetM, uint32_t vecOffsetN,
                                                           const TilingConfig &tilingParams1,
                                                           ExpertParallInfo mm1ExpertParallInfo);
    __aicore__ inline void CalcResDataUsedInPostProcessMM2(uint32_t offsetM, uint32_t vecOffsetN,
                                                           ExpertParallInfo mm1ExpertParallInfo,
                                                           TilingConfig tilingParams2);
    __aicore__ inline void ProcessC1C2(uint32_t curV2BaseM, uint32_t curV2BaseN, uint32_t MMOutputOffset,
                                       const TilingConfig &tilingParams, GlobalTensor<int32_t> workspaceMMOutputGm);
    __aicore__ inline void CalcCMatrix(uint32_t curV2BaseM, uint32_t curV2BaseN, uint32_t offsetM);
    __aicore__ inline void AddOffsetAndMulScale(uint32_t curV2BaseM, uint32_t curV2BaseN,
                                                const TilingConfig &tilingParams);
    __aicore__ inline void CalcAMax(const TilingConfig &tilingParams, uint32_t gmReduceOffset, uint32_t curBaseM,
                                    uint32_t offsetM, GlobalTensor<float> reduceMaxWorkspaceGm);
    __aicore__ inline void CopyInReduceSum(uint32_t curV2BaseM, uint32_t reduceSumOffset,
                                           GlobalTensor<float> reduceSumWorkspaceGm);
    __aicore__ inline void OffsetProcess(uint32_t curV2BaseM, uint32_t curV2BaseN, uint32_t offsetAndScaleOffset,
                                         TilingConfig tilingParams, GlobalTensor<xT> offsetWorkspaceGm);
    __aicore__ inline void ScaleProcess(uint32_t curV2BaseN, uint32_t offsetAndScaleOffset, TilingConfig tilingParams,
                                        GlobalTensor<xT> scaleWorkspaceGm);

    TPipe *pipe;
    const FFNTilingData *tilingData;
    mm1Type &mm1;
    mm2Type &mm2;

    GlobalTensor<xT> xGm;
    GlobalTensor<int8_t> workspaceMM1AMatrixGm_;
    GlobalTensor<int32_t> workspaceMM1OutputGm_;
    GlobalTensor<int8_t> workspaceMM2AMatrixGm_;
    GlobalTensor<int32_t> workspaceMM2OutputGm_;
    GlobalTensor<int8_t> weight1Gm;
    GlobalTensor<int8_t> weight2Gm;
    GlobalTensor<xT> scale1WorkspaceGm;
    GlobalTensor<xT> scale2WorkspaceGm;
    GlobalTensor<xT> offset1WorkspaceGm;
    GlobalTensor<xT> offset2WorkspaceGm;
    GlobalTensor<biasT> bias1Gm;
    GlobalTensor<biasT> bias2Gm;
    GlobalTensor<int64_t> expertTokensGm;
    GlobalTensor<yT> yGm;

    GlobalTensor<float> reduceMax1WorkspaceGm_;
    GlobalTensor<float> reduceSum1WorkspaceGm_;
    GlobalTensor<float> reduceMax2WorkspaceGm_;
    GlobalTensor<float> reduceSum2WorkspaceGm_;

    TQue<QuePosition::VECIN, 1> inQueueX_, inQueueReduceMax_;
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TBuf<QuePosition::VECCALC> eTokens64Buf;
    TBuf<> tmpBuff_;

    LocalTensor<float> middleResult1;
    LocalTensor<float> middleResult2;
    LocalTensor<float> middleResult3;
    LocalTensor<float> aMax;
    LocalTensor<float> tmpScaleAndOffset;
    LocalTensor<half> middleResultFP16;
    LocalTensor<int64_t> ubTokens;

    uint32_t ubCalcShape_ = MSD_EACH_UB_BLOCK_SIZR;
    uint32_t dtypeSizeFP16_ = sizeof(half);
    uint32_t dtypeSizeFP32_ = sizeof(float);
    uint32_t curVecBlockIdx_;
    uint32_t vec1BlockKIdx_;
    uint32_t vec1BlockMIdx_;
    uint32_t curCubeNIdx;

    uint32_t curBlockIdx_;
    uint32_t coreIdx;
    uint32_t currentTokensMM1;
    uint32_t currentTokensMM1Pre;
    uint32_t currentTokensMM2;
    uint32_t expertIdxInParaGroupMM1;
    uint32_t expertIdxInParaGroupMM1Pre;
    uint32_t expertIdxInParaGroupMM2;

    // tiling data
    uint32_t totalTokens;
    int64_t maxTokens;
    uint32_t k1;
    uint32_t n1;
    uint32_t k2;
    uint32_t n2;
    uint32_t expertNum;
    uint32_t coreNum;
    uint32_t cubeCoreNum;
    uint32_t activeType;
    uint32_t baseM1;
    uint32_t baseN1;
    uint32_t baseN2;
    uint32_t ubSize;
    uint32_t mm1DataTypeSize;
    uint32_t workspace1Size;
    uint32_t workspace2Size;
    uint32_t mm2ResUbSize;
    uint32_t maxTokenInParallelGroup = 0;

    bool hasBias1 = false;
    bool hasBias2 = false;
    uint32_t subBlockIdx_;
    uint32_t tokens;
    uint32_t tokensOffset; // tokensOffset = tilingData->ffnBaseParams.tokensArr[0...i-1];
    bool mm2WaitStatue = false;
    uint32_t currentExpertMM1 = 0;
    uint32_t currentExpertMM1Pre = 0;
    uint32_t currentExpertMM2 = 0;
    uint32_t bestCopySize = FP16_INT8_BEST_DATACOPY_BASE_SIZE;
    uint32_t tmpBuffSize;
    uint32_t curV2BaseM1;
    uint32_t curV2BaseN1;
    uint32_t curV2BaseM2;
    uint32_t curV2BaseN2;
    uint32_t countOfFloatUbCalcShape = 0;
    uint32_t preExpertUsedCoreInIter;
};

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::Init(
    __gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2, __gm__ uint8_t *expertTokens,
    __gm__ uint8_t *bias1, __gm__ uint8_t *bias2, __gm__ uint8_t *antiQuantScale1, __gm__ uint8_t *antiQuantScale2,
    __gm__ uint8_t *antiQuantOffset1, __gm__ uint8_t *antiQuantOffset2, __gm__ uint8_t *y, __gm__ uint8_t *workSpace,
    const FFNTilingData *__restrict tiling, TPipe *tPipe)
{
    tilingData = tiling;
    pipe = tPipe;
    InitTilingData();
    GetMaxToken(expertTokens);

    // init global buffer
    xGm.SetGlobalBuffer((__gm__ xT *)x);
    weight1Gm.SetGlobalBuffer((__gm__ int8_t *)weight1);
    if (bias1 != nullptr) {
        hasBias1 = true;
        bias1Gm.SetGlobalBuffer((__gm__ biasT *)bias1);
    }
    weight2Gm.SetGlobalBuffer((__gm__ int8_t *)weight2);
    if (bias2 != nullptr) {
        hasBias2 = true;
        bias2Gm.SetGlobalBuffer((__gm__ biasT *)bias2);
    }

    scale1WorkspaceGm.SetGlobalBuffer((__gm__ xT *)antiQuantScale1);
    scale2WorkspaceGm.SetGlobalBuffer((__gm__ xT *)antiQuantScale2);
    offset1WorkspaceGm.SetGlobalBuffer((__gm__ xT *)antiQuantOffset1);
    offset2WorkspaceGm.SetGlobalBuffer((__gm__ xT *)antiQuantOffset2);
    yGm.SetGlobalBuffer((__gm__ yT *)y);

    InitWorkspace(workSpace);
    AllocLocalTensor(tPipe);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::InitTilingData()
{
    curBlockIdx_ = GetBlockIdx();
    subBlockIdx_ = GetSubBlockIdx();
    coreIdx = curBlockIdx_ / GetTaskRation();

    totalTokens = tilingData->ffnBaseParams.totalTokens;
    k1 = tilingData->ffnBaseParams.k1;
    n1 = tilingData->ffnBaseParams.n1;
    k2 = n1;
    n2 = tilingData->ffnBaseParams.n2;
    expertNum = tilingData->ffnBaseParams.expertNum;
    cubeCoreNum = tilingData->ffnBaseParams.coreNum;
    coreNum = cubeCoreNum;
    activeType = tilingData->ffnBaseParams.activeType;

    baseM1 = tilingData->ffnSingleCoreParams.baseM1;
    baseN1 = tilingData->ffnSingleCoreParams.baseN1;
    baseN2 = tilingData->ffnSingleCoreParams.baseN2;
    ubSize = tilingData->ffnSingleCoreParams.ubRestBytes;
    workspace1Size = tilingData->ffnBaseParams.workspace1Size;
    workspace2Size = tilingData->ffnBaseParams.workspace2Size;
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::GetMaxToken(__gm__ uint8_t *expertTokens)
{
    uint32_t expertTokensUbSize = AlignUp<UB_BLOCK_UNIT_SIZE>(expertNum * sizeof(int64_t));
    pipe->InitBuffer(eTokens64Buf, expertTokensUbSize); // 32Byte alignment
    ubTokens = eTokens64Buf.Get<int64_t>();
    if (likely(expertTokens != nullptr)) {
        // copy tokens array from GM
        expertTokensGm.SetGlobalBuffer((__gm__ int64_t *)expertTokens);
        DataCopy(ubTokens, expertTokensGm, AlignUp<EXPERT_NUM_ALIGN>(expertNum)); // 32Byte alignment
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (tilingData->ffnBaseParams.tokensIndexFlag) {
            TokensIndicesToValues(ubTokens, expertNum);
        }
        maxTokens = 0;
        for (uint32_t i = 0; i < expertNum; i++) {
            int64_t curToken = ubTokens.GetValue(i);
            maxTokens = Max(curToken, maxTokens);
        }
    } else {
        ubTokens.SetValue(0, static_cast<int64_t>(tilingData->ffnBaseParams.maxTokens));
        maxTokens = tilingData->ffnBaseParams.maxTokens;
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::InitWorkspace(__gm__ uint8_t *workSpace)
{
    // init global buffer
    uint32_t maxParallelExpertNum1 = Max<uint32_t>(cubeCoreNum / Ceil(n1, tilingData->mm1TilingData.baseN), 1);
    uint32_t maxParallelExpertNum2 = Max<uint32_t>(cubeCoreNum / Ceil(n2, tilingData->mm2TilingData.baseN), 1);
    uint32_t usedWorkspaceSize = 0;
    workspaceMM1AMatrixGm_.SetGlobalBuffer((__gm__ int8_t *)(workSpace),
                                           ANTIQUANT_MSD_STEP * maxTokens * k1 * maxParallelExpertNum1);
    usedWorkspaceSize += ANTIQUANT_MSD_STEP * maxTokens * k1 * sizeof(int8_t) * maxParallelExpertNum1;
    workspaceMM1OutputGm_.SetGlobalBuffer((__gm__ int32_t *)(workSpace + usedWorkspaceSize),
                                          ANTIQUANT_MSD_STEP * maxTokens * n1 * maxParallelExpertNum1);
    usedWorkspaceSize += ANTIQUANT_MSD_STEP * maxTokens * n1 * sizeof(int32_t) * maxParallelExpertNum1;
    workspaceMM2AMatrixGm_.SetGlobalBuffer((__gm__ int8_t *)(workSpace + usedWorkspaceSize),
                                           ANTIQUANT_MSD_STEP * maxTokens * k2 * maxParallelExpertNum1);
    usedWorkspaceSize += ANTIQUANT_MSD_STEP * maxTokens * k2 * sizeof(int8_t) * maxParallelExpertNum1;
    workspaceMM2OutputGm_.SetGlobalBuffer((__gm__ int32_t *)(workSpace + usedWorkspaceSize),
                                          ANTIQUANT_MSD_STEP * maxTokens * n2 * maxParallelExpertNum2);
    usedWorkspaceSize += ANTIQUANT_MSD_STEP * maxTokens * n2 * sizeof(int32_t) * maxParallelExpertNum2;

    uint32_t reduceMaxWorkspaceGmSize = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * totalTokens * sizeof(float);
    reduceMax1WorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workSpace + usedWorkspaceSize));
    InitOutput(reduceMax1WorkspaceGm_, reduceMaxWorkspaceGmSize); // clear workspace
    usedWorkspaceSize += reduceMaxWorkspaceGmSize;
    reduceSum1WorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workSpace + usedWorkspaceSize));
    InitOutput(reduceSum1WorkspaceGm_, reduceMaxWorkspaceGmSize); // clear workspace
    usedWorkspaceSize += reduceMaxWorkspaceGmSize;
    reduceMax2WorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workSpace + usedWorkspaceSize));
    InitOutput(reduceMax2WorkspaceGm_, reduceMaxWorkspaceGmSize); // clear workspace
    usedWorkspaceSize += reduceMaxWorkspaceGmSize;
    reduceSum2WorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workSpace + usedWorkspaceSize));
    InitOutput(reduceSum2WorkspaceGm_, reduceMaxWorkspaceGmSize); // clear workspace
    SyncAll<true>();
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::AllocLocalTensor(TPipe *tPipe)
{
    pipe = tPipe;
    // init ub Tbuf
    uint32_t reduceMaxUbSize = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * maxTokens * sizeof(float);
    uint32_t expertTokensUbSize = AlignUp<UB_BLOCK_UNIT_SIZE>(expertNum * sizeof(int64_t));

    tmpBuffSize = ubSize - ubCalcShape_ * (sizeof(float) + sizeof(half)) - reduceMaxUbSize - expertTokensUbSize;
    pipe->InitBuffer(tmpBuff_, tmpBuffSize);

    // init ub TQue
    pipe->InitBuffer(inQueueX_, 1, ubCalcShape_ * sizeof(float));
    pipe->InitBuffer(outQueueY_, 1, ubCalcShape_ * sizeof(half));
    pipe->InitBuffer(inQueueReduceMax_, 1, reduceMaxUbSize);

    // scale should bigger than singleN, 32 alignment is required
    middleResult1 = tmpBuff_.GetWithOffset<float>(ubCalcShape_, 0);
    countOfFloatUbCalcShape += 1;
    middleResult2 = tmpBuff_.GetWithOffset<float>(ubCalcShape_, ubCalcShape_ * sizeof(float));
    countOfFloatUbCalcShape += 1;
    middleResult3 = tmpBuff_.GetWithOffset<float>(ubCalcShape_, countOfFloatUbCalcShape * ubCalcShape_ * sizeof(float));
    countOfFloatUbCalcShape += 1;
    tmpScaleAndOffset =
        tmpBuff_.GetWithOffset<float>(ubCalcShape_, countOfFloatUbCalcShape * ubCalcShape_ * sizeof(float));
    countOfFloatUbCalcShape += 1;
    middleResultFP16 =
        tmpBuff_.GetWithOffset<half>(ubCalcShape_, countOfFloatUbCalcShape * ubCalcShape_ * sizeof(float));
    aMax = tmpBuff_.GetWithOffset<float>(maxTokens * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE,
                                         countOfFloatUbCalcShape * ubCalcShape_ * sizeof(float) +
                                             ubCalcShape_ * sizeof(half));
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::MM1VectorTiling(TilingConfig &tilingParams)
{
    uint32_t vecBlockDimK_ = tilingParams.aivNumPerExpert;
    uint32_t vecBaseK_ = Ceil(tilingParams.k, vecBlockDimK_);
    // baseK 128 align up
    vecBaseK_ = (vecBaseK_ + NUM_ALIGN_TO_ONE_HUNDRED_TWEENTY_EIGHT) & (~NUM_ALIGN_TO_ONE_HUNDRED_TWEENTY_EIGHT);
    vecBlockDimK_ = Ceil(tilingParams.k, vecBaseK_); // recompute coreNum in K-axis
    uint32_t vecBlockDimM_ =
        tilingParams.aivNumPerExpert / vecBlockDimK_; // recompute coreNum in M-axis
                                                      // recompute singleM and M-axis coreNum
    uint32_t vecSingleM_ = Ceil(tilingParams.mVec, vecBlockDimM_);
    vecBlockDimM_ = Ceil(tilingParams.mVec, vecSingleM_);
    uint32_t vecSingleMTail_ = tilingParams.mVec - (vecBlockDimM_ - 1) * vecSingleM_;
    uint32_t vecBaseM_ = ubCalcShape_ / vecBaseK_;
    vecBaseM_ = vecBaseM_ < vecSingleM_ ? vecBaseM_ : vecSingleM_;
    tilingParams.vecBaseM = vecBaseM_;
    tilingParams.vecBaseK = vecBaseK_;
    tilingParams.vecBlockDimK = vecBlockDimK_;
    tilingParams.vecBlockDimM = vecBlockDimM_;
    tilingParams.vecSingleM = vecSingleM_;
    tilingParams.vecSingleMTail = vecSingleMTail_;
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::MM2VectorTiling(uint32_t curV2BaseN, uint32_t blockDimK,
                                                                           uint32_t v2BaseM, TilingConfig &tilingParams)
{
    tilingParams.vecBaseK = curV2BaseN;
    tilingParams.vecBaseM = v2BaseM;
    tilingParams.vecBlockDimK = blockDimK;
    tilingParams.vecBlockDimM = 1;
    tilingParams.vecSingleM = tilingParams.mVec;
    tilingParams.vecSingleMTail = tilingParams.mVec;
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CubeTiling(TilingConfig &tilingParams)
{
    tilingParams.cubeSingleM = ANTIQUANT_MSD_STEP * tilingParams.mCube; // M-axis is twice size ori m in msd-method
    tilingParams.cubeSingleM = (tilingParams.cubeSingleM + NUM_ALIGN_TO_SIXTEEN) & (~NUM_ALIGN_TO_SIXTEEN);
    tilingParams.cubeSingleM =
        tilingParams.cubeSingleM > tilingParams.cubeBaseM ? tilingParams.cubeSingleM : tilingParams.cubeBaseM;
    tilingParams.cubeBlockDimM = 1;
    tilingParams.cubeBlockDimN = tilingParams.aicNumPerExpert;
    tilingParams.cubeSingleN = Ceil(tilingParams.n, tilingParams.cubeBlockDimN);
    tilingParams.cubeSingleN = (tilingParams.cubeSingleN + NUM_ALIGN_TO_SIXTEEN) & (~NUM_ALIGN_TO_SIXTEEN);
    tilingParams.cubeSingleN =
        tilingParams.cubeSingleN > tilingParams.cubeBaseN ? tilingParams.cubeSingleN : tilingParams.cubeBaseN;
    tilingParams.cubeBlockDimN = Ceil(tilingParams.n, tilingParams.cubeSingleN);
    tilingParams.cubeSingleNTail = tilingParams.n - (tilingParams.cubeBlockDimN - 1) * tilingParams.cubeSingleN;
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CopyOriginInput(
    const TilingConfig &tilingParams, uint32_t curBaseM, uint32_t curBaseK, uint32_t xGmOffset)
{
    uint32_t alignedBaseK = (curBaseK + NUM_ALIGN_TO_SIXTEEN) & (~NUM_ALIGN_TO_SIXTEEN);
    LocalTensor<xT> xLocal = inQueueX_.AllocTensor<xT>();
    DataCopyPadExtParams<xT> padParams;
    DataCopyExtParams xParams;
    xParams.blockLen = curBaseK * dtypeSizeFP16_;
    xParams.blockCount = curBaseM;
    xParams.srcStride = (tilingParams.k - curBaseK) * dtypeSizeFP16_;
    xParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[xGmOffset], xParams, padParams);
    inQueueX_.EnQue(xLocal);
    LocalTensor<xT> xFP16InUb = inQueueX_.DeQue<xT>();
    Cast(middleResult1, xFP16InUb, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    inQueueX_.FreeTensor(xFP16InUb);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcTailBlock(TilingConfig tilingParams,
                                                                         uint32_t &curSingleM, uint32_t &curBaseK)
{
    if (vec1BlockMIdx_ == tilingParams.vecBlockDimM - 1) {
        curSingleM = tilingParams.vecSingleMTail;
    }
    if (vec1BlockKIdx_ == tilingParams.vecBlockDimK - 1) {
        curBaseK = tilingParams.k - (tilingParams.vecBlockDimK - 1) * tilingParams.vecBaseK;
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::PreProcessMM1(TilingConfig tilingParams,
                                                                         ExpertParallInfo mmExpertParallInfo)
{
    MM1VectorTiling(tilingParams);
    vec1BlockKIdx_ = (coreIdx - expertIdxInParaGroupMM1Pre * tilingParams.aivNumPerExpert) % tilingParams.vecBlockDimK;
    vec1BlockMIdx_ = (coreIdx - expertIdxInParaGroupMM1Pre * tilingParams.aivNumPerExpert) / tilingParams.vecBlockDimK;

    uint32_t curBaseK = tilingParams.vecBaseK;
    uint32_t curBaseM = tilingParams.vecBaseM;
    uint32_t curSingleM = tilingParams.vecSingleM;
    CalcTailBlock(tilingParams, curSingleM, curBaseK);

    uint32_t globalOffset = mmExpertParallInfo.GlobalOffset[expertIdxInParaGroupMM1Pre];
    uint32_t localOffset = mmExpertParallInfo.LocalOffset[expertIdxInParaGroupMM1Pre] * ANTIQUANT_MSD_STEP;
    uint32_t coreIdxThreshold = expertIdxInParaGroupMM1Pre * tilingParams.aivNumPerExpert +
                                tilingParams.vecBlockDimM * tilingParams.vecBlockDimK;

    for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += tilingParams.vecBaseM) {
        if (offsetM + tilingParams.vecBaseM >= curSingleM) {
            curBaseM = curSingleM - offsetM;
        }
        if (coreIdx < coreIdxThreshold && subBlockIdx_ == 0) {
            uint32_t offsetBase = globalOffset + vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM;
            uint32_t xGmOffset = offsetBase * tilingParams.k + vec1BlockKIdx_ * tilingParams.vecBaseK;
            CopyOriginInput(tilingParams, curBaseM, curBaseK, xGmOffset);
            uint32_t gmReduceOffset = offsetBase * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
            CalcReduceMax(gmReduceOffset, curBaseM, curBaseK, reduceMax1WorkspaceGm_);
            CalcReduceSum(gmReduceOffset, curBaseM, curBaseK, reduceSum1WorkspaceGm_);
        }
    }
    SyncAll<true>();
    curBaseM = tilingParams.vecBaseM;
    for (uint32_t offsetM = 0; offsetM < curSingleM; offsetM += tilingParams.vecBaseM) {
        if (offsetM + tilingParams.vecBaseM >= curSingleM) {
            curBaseM = curSingleM - offsetM;
        }
        if (coreIdx < coreIdxThreshold && subBlockIdx_ == 0) {
            uint32_t offsetBase = globalOffset + vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM;
            uint32_t xGmOffset = offsetBase * tilingParams.k + vec1BlockKIdx_ * tilingParams.vecBaseK;
            uint32_t gmReduceOffset = offsetBase * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
            uint32_t aOffsetGm = (localOffset + vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM) * tilingParams.k +
                                 vec1BlockKIdx_ * tilingParams.vecBaseK;
            CopyOriginInput(tilingParams, curBaseM, curBaseK, xGmOffset);
            CalcAMax(tilingParams, gmReduceOffset, curBaseM, offsetM, reduceMax1WorkspaceGm_);
            CalcA1A2(tilingParams, offsetM, curBaseM, curBaseK, aOffsetGm, workspaceMM1AMatrixGm_);
        }
    }
    SyncAll<true>();
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::PreProcessMM2(
    uint32_t offsetM, uint32_t curBaseK, TilingConfig tilingParams, uint32_t &syncCount,
    ExpertParallInfo mmExpertParallInfo)
{
    uint32_t curBaseM = tilingParams.vecBaseM;
    uint32_t aOffsetGm;
    uint32_t gmReduceOffset;

    if (coreIdx < expertIdxInParaGroupMM1 * tilingParams.aivNumPerExpert +
                      tilingParams.vecBlockDimM * tilingParams.vecBlockDimK &&
        subBlockIdx_ == 0) {
        vec1BlockKIdx_ = (coreIdx - expertIdxInParaGroupMM1 * tilingParams.aivNumPerExpert) % tilingParams.vecBlockDimK;
        vec1BlockMIdx_ = (coreIdx - expertIdxInParaGroupMM1 * tilingParams.aivNumPerExpert) / tilingParams.vecBlockDimK;
        gmReduceOffset = (mmExpertParallInfo.GlobalOffset[expertIdxInParaGroupMM1] +
                          vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM) *
                         FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
        CalcReduceMax(gmReduceOffset, curBaseM, curBaseK, reduceMax2WorkspaceGm_);
        CalcReduceSum(gmReduceOffset, curBaseM, curBaseK, reduceSum2WorkspaceGm_);
    }
    SyncAll<true>();
    syncCount -= 1;
    if (coreIdx < expertIdxInParaGroupMM1 * tilingParams.aivNumPerExpert +
                      tilingParams.vecBlockDimM * tilingParams.vecBlockDimK &&
        subBlockIdx_ == 0) {
        aOffsetGm = (mmExpertParallInfo.LocalOffset[expertIdxInParaGroupMM1] * ANTIQUANT_MSD_STEP +
                     vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM) *
                        tilingParams.k +
                    vec1BlockKIdx_ * tilingParams.vecBaseK;
        CalcAMax(tilingParams, gmReduceOffset, curBaseM, offsetM, reduceMax2WorkspaceGm_);
        CalcA1A2(tilingParams, offsetM, curBaseM, curBaseK, aOffsetGm, workspaceMM2AMatrixGm_);
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcReduceMax(
    uint32_t gmReduceOffset, uint32_t curBaseM, uint32_t curBaseK, GlobalTensor<float> reduceMaxWorkspaceGm)
{
    uint32_t alignedBaseK = (curBaseK + NUM_ALIGN_TO_SIXTEEN) & (~NUM_ALIGN_TO_SIXTEEN);
    Abs(middleResult2, middleResult1, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();

    // calc ReduceMax
    LocalTensor<float> blockReduceMaxInUb = outQueueY_.AllocTensor<float>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        ReduceMax(blockReduceMaxInUb[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], middleResult2[idxM * curBaseK],
                  middleResult3[idxM * curBaseK], curBaseK, false);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        float aLocalMax = blockReduceMaxInUb.GetValue(idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(blockReduceMaxInUb[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], aLocalMax,
                  FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
    }
    outQueueY_.EnQue<float>(blockReduceMaxInUb);
    LocalTensor<float> blockReduceMax = outQueueY_.DeQue<float>();
    SetAtomicMax<float>();
    DataCopyExtParams aMaxOutParams;
    aMaxOutParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * dtypeSizeFP32_;
    aMaxOutParams.blockCount = curBaseM;
    aMaxOutParams.srcStride = 0;
    aMaxOutParams.dstStride = 0;
    DataCopyPad(reduceMaxWorkspaceGm[gmReduceOffset], blockReduceMax, aMaxOutParams);
    SetAtomicNone();
    outQueueY_.FreeTensor(blockReduceMax);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcReduceSum(
    uint32_t gmReduceOffset, uint32_t curBaseM, uint32_t curBaseK, GlobalTensor<float> reduceSumWorkspaceGm)
{
    // calc ReduceSum
    LocalTensor<float> blockReduceSumInUb = outQueueY_.AllocTensor<float>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        ReduceSum(blockReduceSumInUb[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], middleResult1[idxM * curBaseK],
                  middleResult3[idxM * curBaseK], curBaseK);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        float aLocalSum = blockReduceSumInUb.GetValue(idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(blockReduceSumInUb[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], aLocalSum,
                  FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
    }
    outQueueY_.EnQue<float>(blockReduceSumInUb);
    LocalTensor<float> blockReduceSum = outQueueY_.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopyExtParams aSumOutParams;
    aSumOutParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * dtypeSizeFP32_;
    aSumOutParams.blockCount = curBaseM;
    aSumOutParams.srcStride = 0;
    aSumOutParams.dstStride = 0;
    DataCopyPad(reduceSumWorkspaceGm[gmReduceOffset], blockReduceSum, aSumOutParams);
    SetAtomicNone();
    outQueueY_.FreeTensor(blockReduceSum);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcAMax(
    const TilingConfig &tilingParams, uint32_t gmReduceOffset, uint32_t curBaseM, uint32_t offsetM,
    GlobalTensor<float> reduceMaxWorkspaceGm)
{
    // Calc Amax
    LocalTensor<float> aMaxLocal = inQueueReduceMax_.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams aMaxInParams;
    aMaxInParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * dtypeSizeFP32_;
    aMaxInParams.blockCount = curBaseM;
    aMaxInParams.srcStride = 0;
    aMaxInParams.dstStride = 0;
    DataCopyPad(aMaxLocal, reduceMaxWorkspaceGm[gmReduceOffset], aMaxInParams, padParams);
    inQueueReduceMax_.EnQue(aMaxLocal);
    LocalTensor<float> aMaxInUb = inQueueReduceMax_.DeQue<float>();

    Muls(aMax[(vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM) * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], aMaxInUb,
         static_cast<float>(1.001), curBaseM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    inQueueReduceMax_.FreeTensor(aMaxInUb);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcA1A2(
    const TilingConfig &tilingParams, uint32_t offsetM, uint32_t curBaseM, uint32_t curBaseK, uint32_t aOffsetGm,
    GlobalTensor<int8_t> workspaceAMatrixGm)
{
    uint32_t alignedBaseK = (curBaseK + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO);

    // A1 : A/AMax
    for (uint32_t idxM = 0; idxM < curBaseM; ++idxM) {
        float invertAMaxPerRow =
            1.0f / aMax((vec1BlockMIdx_ * tilingParams.vecSingleM + offsetM + idxM) * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Muls(middleResult3[idxM * curBaseK], middleResult1[idxM * curBaseK], invertAMaxPerRow, alignedBaseK);
    }

    // A1 : floor(128 * A/AMax)
    LocalTensor<int8_t> a1Int8InUb = outQueueY_.AllocTensor<int8_t>();
    Muls(middleResult2, middleResult3, static_cast<float>(128), curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Cast(middleResult1, middleResult2, RoundMode::CAST_FLOOR, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Cast(middleResultFP16, middleResult1, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Cast(a1Int8InUb, middleResultFP16, RoundMode::CAST_NONE, curBaseM * alignedBaseK);

    // A1 : DATACOPY a1(int8) ub->gm
    outQueueY_.EnQue<int8_t>(a1Int8InUb);
    LocalTensor<int8_t> a1Int8 = outQueueY_.DeQue<int8_t>();

    DataCopyExtParams aOutParams;
    aOutParams.blockLen = curBaseK;
    aOutParams.blockCount = curBaseM;
    aOutParams.srcStride = 0;
    aOutParams.dstStride = tilingParams.k - curBaseK;
    DataCopyPad(workspaceAMatrixGm[aOffsetGm], a1Int8, aOutParams);

    // A2 : floor((A/AMax - A1/128)*128*128)
    Sub(middleResult3, middleResult2, middleResult1, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Muls(middleResult1, middleResult3, static_cast<float>(128), curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Cast(middleResult1, middleResult1, RoundMode::CAST_FLOOR, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    Cast(middleResultFP16, middleResult1, RoundMode::CAST_NONE, curBaseM * alignedBaseK);
    PipeBarrier<PIPE_V>();
    outQueueY_.FreeTensor(a1Int8);
    LocalTensor<int8_t> a2Int8InUb = outQueueY_.AllocTensor<int8_t>();
    Cast(a2Int8InUb, middleResultFP16, RoundMode::CAST_NONE, curBaseM * alignedBaseK);

    // A2 : DATACOPY a2(int8) ub->gm
    outQueueY_.EnQue<int8_t>(a2Int8InUb);
    LocalTensor<int8_t> a2Int8 = outQueueY_.DeQue<int8_t>();
    DataCopyPad(workspaceAMatrixGm[tilingParams.mVec * tilingParams.k + aOffsetGm], a2Int8, aOutParams);
    outQueueY_.FreeTensor(a2Int8);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::MM1Compute(uint32_t curCubeSingleCoreN, uint32_t offsetN,
                                                                      const TilingConfig &tilingParams,
                                                                      ExpertParallInfo mmExpertParallInfo)
{
    uint32_t AMatrixMM1Offset =
        mmExpertParallInfo.LocalOffset[expertIdxInParaGroupMM1] * ANTIQUANT_MSD_STEP * tilingParams.k;
    uint32_t w1CoreOffset = currentExpertMM1 * tilingParams.n * tilingParams.k + offsetN;
    uint32_t outOffset =
        mmExpertParallInfo.LocalOffset[expertIdxInParaGroupMM1] * ANTIQUANT_MSD_STEP * tilingParams.n + offsetN;
    mm1.SetOrgShape(2 * tilingParams.mCube, tilingParams.n, tilingParams.k);
    mm1.SetSingleShape(2 * tilingParams.mCube, curCubeSingleCoreN, tilingParams.k);
    mm1.SetTensorA(workspaceMM1AMatrixGm_[AMatrixMM1Offset]);
    mm1.SetTensorB(weight1Gm[w1CoreOffset]);
    mm1.template IterateAll<false>(workspaceMM1OutputGm_[outOffset], 0, false, true);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::MM2Compute(
    uint32_t curCubeSingleCoreN, uint32_t offsetN, TilingConfig tilingParams, ExpertParallInfo mmExpertParallInfo,
    uint32_t curIterCount)
{
    uint32_t AMatrixMM2Offset =
        mmExpertParallInfo.LocalOffset[expertIdxInParaGroupMM2] * ANTIQUANT_MSD_STEP * tilingParams.k;
    uint32_t w2CoreOffset = currentExpertMM2 * tilingParams.n * tilingParams.k + offsetN;
    uint32_t outOffset = (mmExpertParallInfo.LocalOffset[expertIdxInParaGroupMM2] -
                          mmExpertParallInfo.LocalOffset[curIterCount * mmExpertParallInfo.expertParallelism]) *
                             tilingParams.n * ANTIQUANT_MSD_STEP +
                         offsetN;
    mm2.SetOrgShape(2 * tilingParams.mCube, tilingParams.n, tilingParams.k);
    mm2.SetSingleShape(2 * tilingParams.mCube, curCubeSingleCoreN, tilingParams.k);
    mm2.SetTensorA(workspaceMM2AMatrixGm_[AMatrixMM2Offset]);
    mm2.SetTensorB(weight2Gm[w2CoreOffset]);
    mm2.template IterateAll<false>(workspaceMM2OutputGm_[outOffset], 0, false, true);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::SaveAmaxInUb(
    uint32_t m, GlobalTensor<float> reduceMaxWorkspaceGm, ExpertParallInfo mmExpertParallInfo,
    uint32_t expertIdxInParaGroupMM)
{
    LocalTensor<float> aMaxLocal = inQueueReduceMax_.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams aMaxInParams;
    aMaxInParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * dtypeSizeFP32_;
    aMaxInParams.blockCount = m;
    aMaxInParams.srcStride = 0;
    aMaxInParams.dstStride = 0;
    uint32_t reduceMaxWorkspaceOffset =
        mmExpertParallInfo.GlobalOffset[expertIdxInParaGroupMM] * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    DataCopyPad(aMaxLocal, reduceMaxWorkspaceGm[reduceMaxWorkspaceOffset], aMaxInParams, padParams);
    inQueueReduceMax_.EnQue(aMaxLocal);
    LocalTensor<float> aMaxInUb = inQueueReduceMax_.DeQue<float>();
    Muls(aMax, aMaxInUb, static_cast<float>(1.001), m * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE);
    PipeBarrier<PIPE_V>();
    inQueueReduceMax_.FreeTensor(aMaxInUb);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CopyInReduceSum(
    uint32_t curV2BaseM, uint32_t reduceSumOffset, GlobalTensor<float> reduceSumWorkspaceGm)
{
    LocalTensor<float> aSumLocal = inQueueReduceMax_.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams aSumInParams;
    aSumInParams.blockLen = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE * dtypeSizeFP32_;
    aSumInParams.blockCount = curV2BaseM;
    aSumInParams.srcStride = 0;
    aSumInParams.dstStride = 0;
    DataCopyPad(aSumLocal, reduceSumWorkspaceGm[reduceSumOffset], aSumInParams, padParams);
    inQueueReduceMax_.EnQue(aSumLocal);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::OffsetProcess(
    uint32_t curV2BaseM, uint32_t curV2BaseN, uint32_t offsetAndScaleOffset, TilingConfig tilingParams,
    GlobalTensor<xT> offsetWorkspaceGm)
{
    LocalTensor<xT> offsetF16 = inQueueX_.AllocTensor<xT>();
    LocalTensor<float> v2Asum = inQueueReduceMax_.DeQue<float>();

    DataCopyPad2D(offsetF16, offsetWorkspaceGm[offsetAndScaleOffset], 1, curV2BaseN, tilingParams.n);
    inQueueX_.EnQue(offsetF16);
    LocalTensor<xT> offsetF16InUb = inQueueX_.DeQue<xT>();
    Cast(tmpScaleAndOffset, offsetF16InUb, RoundMode::CAST_NONE, curV2BaseN);
    PipeBarrier<PIPE_V>();
    inQueueX_.FreeTensor(offsetF16InUb);
    // (m, 8) (1, n) -> (m, n)
    uint32_t mask = DATASIZE_EACH_REPEAT_TIME / sizeof(float);
    uint32_t mainRepeatN = curV2BaseN / mask;
    uint32_t tailN = curV2BaseN % mask;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    repeatParams.src0RepStride = 0;
    repeatParams.src1RepStride = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    for (uint32_t idxM = 0; idxM < curV2BaseM; ++idxM) {
        Mul(middleResult2[idxM * curV2BaseN], v2Asum[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], tmpScaleAndOffset, mask,
            mainRepeatN, repeatParams);
    }
    if (tailN > 0) {
        for (uint32_t idxM = 0; idxM < curV2BaseM; ++idxM) {
            Mul(middleResult2[idxM * curV2BaseN + mainRepeatN * mask], v2Asum[idxM * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE],
                tmpScaleAndOffset[mainRepeatN * mask], tailN, 1, repeatParams);
        }
    }
    inQueueReduceMax_.FreeTensor(v2Asum);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ScaleProcess(
    uint32_t curV2BaseN, uint32_t offsetAndScaleOffset, TilingConfig tilingParams, GlobalTensor<xT> scaleWorkspaceGm)
{
    LocalTensor<xT> scaleF16 = inQueueX_.AllocTensor<xT>();
    DataCopyPad2D(scaleF16, scaleWorkspaceGm[offsetAndScaleOffset], 1, curV2BaseN, tilingParams.n);
    inQueueX_.EnQue(scaleF16);
    LocalTensor<xT> scaleF16InUb = inQueueX_.DeQue<xT>();
    Cast(tmpScaleAndOffset, scaleF16InUb, RoundMode::CAST_NONE, curV2BaseN);
    PipeBarrier<PIPE_V>();
    inQueueX_.FreeTensor(scaleF16InUb);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ProcessC1C2(
    uint32_t curV2BaseM, uint32_t curV2BaseN, uint32_t MMOutputOffset, const TilingConfig &tilingParams,
    GlobalTensor<int32_t> workspaceMMOutputGm)
{
    uint32_t curBaseNAligned = (curV2BaseN + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO);

    // k * (m, cubeN) -> k * (m, vecN)
    //            cubeBaseN
    //      vecBaseN   vecBaseN
    // m0
    // m1

    // process C1
    LocalTensor<int32_t> c1S32InUb = tmpBuff_.GetWithOffset<int32_t>(ubCalcShape_, 0);
    LocalTensor<int32_t> c2S32InUb = tmpBuff_.GetWithOffset<int32_t>(ubCalcShape_, 2 * ubCalcShape_ * sizeof(float));
    event_t eventIdMte2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte2ToV1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    DataCopyPad2D(c1S32InUb, workspaceMMOutputGm[MMOutputOffset], curV2BaseM, curV2BaseN, tilingParams.n);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV0);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV0);
    Cast(middleResult1, c1S32InUb, RoundMode::CAST_NONE, curV2BaseM * curBaseNAligned);
    PipeBarrier<PIPE_V>();
    Muls(middleResult1, middleResult1, static_cast<float>(1.0 / 128), curV2BaseM * curBaseNAligned);

    // process C2
    DataCopyPad2D(c2S32InUb, workspaceMMOutputGm[tilingParams.mCube * tilingParams.n + MMOutputOffset], curV2BaseM,
                  curV2BaseN, tilingParams.n);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV1);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV1);
    Cast(middleResult3, c2S32InUb, RoundMode::CAST_NONE, curV2BaseM * curBaseNAligned);
    PipeBarrier<PIPE_V>();
    Muls(middleResult3, middleResult3, static_cast<float>(1.0 / (128 * 128)), curV2BaseM * curBaseNAligned);
    PipeBarrier<PIPE_V>();
    // process C1+C2
    Add(middleResult1, middleResult3, middleResult1, curV2BaseM * curBaseNAligned);
    PipeBarrier<PIPE_V>();
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcCMatrix(uint32_t curV2BaseM,
                                                                                              uint32_t curV2BaseN,
                                                                                              uint32_t offsetM)
{
    // multiply with Amax to get C (m, n) * (m, 8) -> (m, n)
    uint32_t mask = DATASIZE_EACH_REPEAT_TIME / sizeof(float); // 64
    uint32_t mainRepeatN = curV2BaseN / mask;
    uint32_t tailN = curV2BaseN % mask;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    repeatParams.src0RepStride = FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    repeatParams.src1RepStride = 0;
    for (uint32_t idxM = 0; idxM < curV2BaseM; ++idxM) {
        Mul(middleResult3[idxM * curV2BaseN], middleResult1[idxM * curV2BaseN],
            aMax[(offsetM + idxM) * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], mask, mainRepeatN, repeatParams);
    }
    if (tailN > 0) {
        for (uint32_t idxM = 0; idxM < curV2BaseM; ++idxM) {
            Mul(middleResult3[idxM * curV2BaseN + mainRepeatN * mask],
                middleResult1[idxM * curV2BaseN + mainRepeatN * mask],
                aMax[(offsetM + idxM) * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE], tailN, 1, repeatParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::AddOffsetAndMulScale(
    uint32_t curV2BaseM, uint32_t curV2BaseN, const TilingConfig &tilingParams)
{
    uint32_t curBaseNAligned = (curV2BaseN + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO);
    // add with processed offset
    Add(middleResult2, middleResult2, middleResult3, curV2BaseM * curBaseNAligned);
    PipeBarrier<PIPE_V>();

    // multiply with scale (m, n) * (1, n) -> (m, n)
    for (uint32_t idxM = 0; idxM < curV2BaseM; ++idxM) {
        Mul(middleResult3[idxM * curV2BaseN], middleResult2[idxM * curV2BaseN], tmpScaleAndOffset, curBaseNAligned);
    }
    PipeBarrier<PIPE_V>();

    // add bias
    if (tilingParams.hasBias) {
        uint32_t usedTmpBufferSize =
            (countOfFloatUbCalcShape * ubCalcShape_ + maxTokens * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE) * sizeof(float) +
            ubCalcShape_ * sizeof(half);
        LocalTensor<float> tmpLocal =
            tmpBuff_.GetWithOffset<float>((tmpBuffSize - usedTmpBufferSize) / sizeof(float), usedTmpBufferSize);
        for (uint32_t idxM = 0; idxM < curV2BaseM; ++idxM) {
            Add(middleResult3[idxM * curV2BaseN], middleResult3[idxM * curV2BaseN], tmpLocal, curBaseNAligned);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CopyOutFinalResult(uint32_t curV2BaseM, uint32_t curV2BaseN,
                                                                              uint32_t yOffset, uint32_t n)
{
    uint32_t curBaseNAligned = (curV2BaseN + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO);
    LocalTensor<yT> outputInUb = outQueueY_.AllocTensor<yT>();
    if constexpr (IsSameType<xT, half>::value) {
        Cast(outputInUb, middleResult3, RoundMode::CAST_NONE, curV2BaseM * curBaseNAligned);
    } else {
        Cast(outputInUb, middleResult3, RoundMode::CAST_RINT, curV2BaseM * curBaseNAligned);
    }
    outQueueY_.EnQue(outputInUb);
    LocalTensor<yT> output = outQueueY_.DeQue<yT>();
    // (m, n)
    uint32_t curV2BaseNAlign16 = matmul::CeilAlign(curV2BaseN, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(half)));
    DataCopyPad2D(yGm[yOffset], output, curV2BaseM, curV2BaseN, curV2BaseNAlign16, n);
    outQueueY_.FreeTensor(output);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ActivationCompute(uint32_t computeSize)
{
    uint32_t usedTmpBufferSize =
        (countOfFloatUbCalcShape * ubCalcShape_ + maxTokens * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE) * sizeof(float) +
        ubCalcShape_ * sizeof(half);
    LocalTensor<uint8_t> tmpLocal =
        tmpBuff_.GetWithOffset<uint8_t>((tmpBuffSize - usedTmpBufferSize), usedTmpBufferSize);

    ActiveType active = ActiveType(activeType);
    if (active == ActiveType::FASTGELU) {
        FasterGelu(middleResult1, middleResult3, tmpLocal, computeSize);
    } else if (active == ActiveType::RELU) {
        Relu(middleResult1, middleResult3, computeSize);
        PipeBarrier<PIPE_V>();
    } else if (active == ActiveType::SILU) {
        Silu(middleResult1, middleResult3, computeSize);
    } else if (active == ActiveType::GELU) {
        Gelu(middleResult1, middleResult3, tmpLocal, computeSize);
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CopyInBias(
    uint32_t curV2BaseN, uint32_t offsetAndScaleOffset, TilingConfig tilingParams, GlobalTensor<biasT> biasGm)
{
    uint32_t usedTmpBufferSize =
        (countOfFloatUbCalcShape * ubCalcShape_ + maxTokens * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE) * sizeof(float) +
        ubCalcShape_ * sizeof(half);
    LocalTensor<float> tmpLocal =
        tmpBuff_.GetWithOffset<float>((tmpBuffSize - usedTmpBufferSize) / sizeof(float), usedTmpBufferSize);
    // add bias, firstly convert to float32 if xT is float16
    if constexpr (IsSameType<xT, half>::value) {
        LocalTensor<biasT> bias = inQueueX_.AllocTensor<biasT>();
        DataCopyPad2D(bias, biasGm[offsetAndScaleOffset], 1, curV2BaseN, tilingParams.n);
        inQueueX_.EnQue(bias);
        LocalTensor<biasT> biasInUb = inQueueX_.DeQue<biasT>();
        Cast(tmpLocal, biasInUb, RoundMode::CAST_NONE,
             (curV2BaseN + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO));
        PipeBarrier<PIPE_V>();
        inQueueX_.FreeTensor(biasInUb);
    } else {
        DataCopyPad2D(tmpLocal, biasGm[offsetAndScaleOffset], 1, curV2BaseN, tilingParams.n);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::Elewise1(uint32_t curV2BaseM,
                                                                                           uint32_t curV2BaseN)
{
    uint32_t computeBaseN1;
    uint32_t computeSize;

    // mm1 is float16 and 32-byte aligned. mm1 is float32 and 64-byte aligned.
    computeBaseN1 = AlignUp<GetNumInUbBlock<xT>()>(curV2BaseN);
    computeSize = curV2BaseM * computeBaseN1;
    ActivationCompute(computeSize);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcResDataUsedInPostProcessMM1(
    uint32_t offsetM, uint32_t vecOffsetN, const TilingConfig &tilingParams1, ExpertParallInfo mm1ExpertParallInfo)
{
    uint32_t reduceSumOffset =
        (mm1ExpertParallInfo.GlobalOffset[expertIdxInParaGroupMM1] + offsetM) * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    uint32_t offsetAndScaleOffset = currentExpertMM1 * tilingParams1.n + vecOffsetN;
    CopyInReduceSum(curV2BaseM1, reduceSumOffset, reduceSum1WorkspaceGm_);
    OffsetProcess(curV2BaseM1, curV2BaseN1, offsetAndScaleOffset, tilingParams1, offset1WorkspaceGm);
    ScaleProcess(curV2BaseN1, offsetAndScaleOffset, tilingParams1, scale1WorkspaceGm);
    if (hasBias1) {
        CopyInBias(curV2BaseN1, offsetAndScaleOffset, tilingParams1, bias1Gm);
    }
    if (offsetM == 0) {
        SaveAmaxInUb(tilingParams1.mCube, reduceMax1WorkspaceGm_, mm1ExpertParallInfo, expertIdxInParaGroupMM1);
        mm1.WaitIterateAll();
        mm1.End();
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CalcResDataUsedInPostProcessMM2(
    uint32_t offsetM, uint32_t vecOffsetN, ExpertParallInfo mm2ExpertParallInfo, TilingConfig tilingParams2)
{
    uint32_t reduceSumOffset =
        (mm2ExpertParallInfo.GlobalOffset[expertIdxInParaGroupMM2] + offsetM) * FACTOR_FOR_FLOAT_ALIGN_TO_32BYTE;
    uint32_t offsetAndScaleOffset = currentExpertMM2 * tilingParams2.n + vecOffsetN;
    CopyInReduceSum(curV2BaseM2, reduceSumOffset, reduceSum2WorkspaceGm_);
    OffsetProcess(curV2BaseM2, curV2BaseN2, offsetAndScaleOffset, tilingParams2, offset2WorkspaceGm);
    ScaleProcess(curV2BaseN2, offsetAndScaleOffset, tilingParams2, scale2WorkspaceGm);
    if (hasBias2) {
        CopyInBias(curV2BaseN2, offsetAndScaleOffset, tilingParams2, bias2Gm);
    }
    if (offsetM == 0) {
        SaveAmaxInUb(tilingParams2.mCube, reduceMax2WorkspaceGm_, mm2ExpertParallInfo, expertIdxInParaGroupMM2);
        mm2.WaitIterateAll();
        mm2.End();
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::SyncAllByCount(uint32_t count)
{
    while (count > 0) {
        SyncAll<true>();
        count--;
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ComputeExpertMM1(
    TilingConfig &tilingParams1, TilingConfig &tilingParams2, ExpertParallInfo mm1ExpertParallInfo)
{
    curCubeNIdx = (coreIdx - expertIdxInParaGroupMM1 * tilingParams1.aicNumPerExpert) % tilingParams1.cubeBlockDimN;
    uint32_t curCubeSingleCoreN = tilingParams1.cubeSingleN;
    if (curCubeNIdx == tilingParams1.cubeBlockDimN - 1) {
        curCubeSingleCoreN = tilingParams1.n - (tilingParams1.cubeBlockDimN - 1) * tilingParams1.cubeSingleN;
    }
    // preprocess MM1's first matrix
    uint32_t cubeOffsetN = curCubeNIdx * tilingParams1.cubeSingleN;
    PreProcessMM1(tilingParams1, mm1ExpertParallInfo);
    event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    uint32_t v2BaseN = (tilingParams1.cubeSingleN + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO);
    uint32_t v2BaseM = ubCalcShape_ / v2BaseN;
    uint32_t coreIdxThreshold = expertIdxInParaGroupMM1 * tilingParams1.aicNumPerExpert + tilingParams1.cubeBlockDimN;
    // MM1 matrix multiplication
    if (coreIdx < coreIdxThreshold && subBlockIdx_ == 0) {
        MM1Compute(curCubeSingleCoreN, cubeOffsetN, tilingParams1, mm1ExpertParallInfo);
    }
    curV2BaseM1 = v2BaseM;
    curV2BaseN1 = curCubeSingleCoreN;
    uint32_t syncCount2 = Ceil(maxTokenInParallelGroup, v2BaseM);
    // MM1 cube computation and scale/offset process in parallel
    for (uint32_t offsetM = 0; offsetM < tilingParams1.mCube; offsetM += v2BaseM) {
        if (offsetM + curV2BaseM1 >= tilingParams1.mCube) {
            curV2BaseM1 = tilingParams1.mCube - offsetM;
        }
        if (offsetM > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        }
        uint32_t vecOffsetN = curCubeNIdx * tilingParams1.cubeSingleN;
        uint32_t localOffset = mm1ExpertParallInfo.LocalOffset[expertIdxInParaGroupMM1] * ANTIQUANT_MSD_STEP;
        uint32_t MMOutputOffset = (localOffset + offsetM) * tilingParams1.n + vecOffsetN;
        if (coreIdx < coreIdxThreshold && subBlockIdx_ == 0) {
            CalcResDataUsedInPostProcessMM1(offsetM, vecOffsetN, tilingParams1, mm1ExpertParallInfo);
        }
        if (offsetM == 0) {
            SyncAll<true>();
        }
        if (coreIdx < coreIdxThreshold && subBlockIdx_ == 0) {
            // mm1 result postprocess
            ProcessC1C2(curV2BaseM1, curV2BaseN1, MMOutputOffset, tilingParams1, workspaceMM1OutputGm_);
            CalcCMatrix(curV2BaseM1, curV2BaseN1, offsetM);
            AddOffsetAndMulScale(curV2BaseM1, curV2BaseN1, tilingParams1);
            // activation
            Elewise1(curV2BaseM1, curV2BaseN1);
        }
        // preprocess MM2's first matrix
        MM2VectorTiling(tilingParams1.cubeSingleN, tilingParams1.cubeBlockDimN, curV2BaseM1, tilingParams2);
        PreProcessMM2(offsetM, curV2BaseN1, tilingParams2, syncCount2, mm1ExpertParallInfo);
        if (offsetM + curV2BaseM1 < tilingParams1.mCube) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        }
    }
    SyncAllByCount(syncCount2);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ComputeExpertMM2(TilingConfig &tilingParams2,
                                                                            ExpertParallInfo mm2ExpertParallInfo)
{
    uint32_t curIterCount = 0;
    for (uint32_t i = mm2ExpertParallInfo.start; i < mm2ExpertParallInfo.size;
         i += mm2ExpertParallInfo.expertParallelism) {
        expertIdxInParaGroupMM2 = Min<uint32_t>(coreIdx / tilingParams2.aicNumPerExpert,
                                                mm2ExpertParallInfo.expertParallelism - 1);
        expertIdxInParaGroupMM2 += curIterCount * mm2ExpertParallInfo.expertParallelism;
        expertIdxInParaGroupMM2 = Min<uint32_t>(expertIdxInParaGroupMM2, mm2ExpertParallInfo.size - 1);
        currentExpertMM2 = mm2ExpertParallInfo.expertIdxBuf[expertIdxInParaGroupMM2];
        currentTokensMM2 = ubTokens.GetValue(currentExpertMM2);
        tilingParams2.mCube = currentTokensMM2;
        CubeTiling(tilingParams2);
        // coreNum used by previous expert in this loop
        preExpertUsedCoreInIter = (expertIdxInParaGroupMM2 - curIterCount * mm2ExpertParallInfo.expertParallelism) *
                                  tilingParams2.aicNumPerExpert;
        curCubeNIdx = (coreIdx - preExpertUsedCoreInIter) % tilingParams2.cubeBlockDimN;
        uint32_t curCubeSingleCoreN = tilingParams2.cubeSingleN;
        if (curCubeNIdx == tilingParams2.cubeBlockDimN - 1) {
            curCubeSingleCoreN = tilingParams2.n - (tilingParams2.cubeBlockDimN - 1) * tilingParams2.cubeSingleN;
        }
        SyncAll<true>();
        uint32_t cubeOffsetN = curCubeNIdx * tilingParams2.cubeSingleN;
        uint32_t v2BaseN = curCubeSingleCoreN;
        v2BaseN = (v2BaseN + NUM_ALIGN_TO_THIRTYTWO) & (~NUM_ALIGN_TO_THIRTYTWO);
        uint32_t v2BaseM = ubCalcShape_ / v2BaseN;
        // MM2 cube multiplication
        if (coreIdx < preExpertUsedCoreInIter + tilingParams2.cubeBlockDimN && subBlockIdx_ == 0) {
            MM2Compute(curCubeSingleCoreN, cubeOffsetN, tilingParams2, mm2ExpertParallInfo, curIterCount);
        }
        curV2BaseM2 = v2BaseM;
        curV2BaseN2 = curCubeSingleCoreN;
        // mm2 result postprocess
        CubeResultPostProcess(tilingParams2, mm2ExpertParallInfo, curIterCount, v2BaseM, i);
        curIterCount += 1;
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::CubeResultPostProcess(
    TilingConfig &tilingParams2, ExpertParallInfo mm2ExpertParallInfo, const uint32_t curIterCount,
    const uint32_t v2BaseM, const uint32_t idx)
{
    event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (uint32_t offsetM = 0; offsetM < tilingParams2.mCube; offsetM += v2BaseM) {
        if (offsetM + curV2BaseM2 >= tilingParams2.mCube) {
            curV2BaseM2 = tilingParams2.mCube - offsetM;
        }
        if (idx > mm2ExpertParallInfo.start || offsetM > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        }
        uint32_t vecOffsetN = curCubeNIdx * tilingParams2.cubeSingleN;
        uint32_t MMOutputOffset =
            ((mm2ExpertParallInfo.LocalOffset[expertIdxInParaGroupMM2] -
              mm2ExpertParallInfo.LocalOffset[curIterCount * mm2ExpertParallInfo.expertParallelism]) *
                 ANTIQUANT_MSD_STEP +
             offsetM) *
                tilingParams2.n +
            vecOffsetN;
        if (coreIdx < preExpertUsedCoreInIter + tilingParams2.cubeBlockDimN && subBlockIdx_ == 0) {
            CalcResDataUsedInPostProcessMM2(offsetM, vecOffsetN, mm2ExpertParallInfo, tilingParams2);
        }
        if (offsetM == 0) {
            SyncAll<true>();
        }
        if (coreIdx < preExpertUsedCoreInIter + tilingParams2.cubeBlockDimN && subBlockIdx_ == 0) {
            ProcessC1C2(curV2BaseM2, curV2BaseN2, MMOutputOffset, tilingParams2, workspaceMM2OutputGm_);
            CalcCMatrix(curV2BaseM2, curV2BaseN2, offsetM);
            AddOffsetAndMulScale(curV2BaseM2, curV2BaseN2, tilingParams2);
            uint32_t yOffset =
                (mm2ExpertParallInfo.GlobalOffset[expertIdxInParaGroupMM2] + offsetM) * tilingParams2.n + vecOffsetN;
            CopyOutFinalResult(curV2BaseM2, curV2BaseN2, yOffset, tilingParams2.n);
        }
        if (idx + mm2ExpertParallInfo.expertParallelism < mm2ExpertParallInfo.size ||
            offsetM + curV2BaseM2 < tilingParams2.mCube) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        }
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ComputeExpert(ExpertParallInfo mm1ExpertParallInfo,
                                                                         ExpertParallInfo mm2ExpertParallInfo)
{
    TilingConfig tilingParams1;
    TilingConfig tilingParams2;
    uint32_t coreNumEachExpert1 = cubeCoreNum / mm1ExpertParallInfo.expertParallelism;
    uint32_t coreNumEachExpert2 = cubeCoreNum / mm2ExpertParallInfo.expertParallelism;
    uint32_t coreNumEachExpert1Vec = coreNum / mm1ExpertParallInfo.expertParallelism;
    // get expertIdx and corresponding expert tokens for each cube core
    expertIdxInParaGroupMM1 = Min<uint32_t>(coreIdx / coreNumEachExpert1, mm1ExpertParallInfo.expertParallelism - 1);
    currentExpertMM1 = mm1ExpertParallInfo.expertIdxBuf[expertIdxInParaGroupMM1];
    currentTokensMM1 = ubTokens.GetValue(currentExpertMM1);
    // get expertIdx and corresponding expert tokens for each vector core
    expertIdxInParaGroupMM1Pre =
        Min<uint32_t>(coreIdx / coreNumEachExpert1Vec, mm1ExpertParallInfo.expertParallelism - 1);
    currentExpertMM1Pre = mm1ExpertParallInfo.expertIdxBuf[expertIdxInParaGroupMM1Pre];
    currentTokensMM1Pre = ubTokens.GetValue(currentExpertMM1Pre);
    tilingParams1.SetBaseParams(currentTokensMM1, currentTokensMM1Pre, n1, k1, hasBias1);
    tilingParams1.SetTilingParams(tilingData->mm1TilingData.baseM, tilingData->mm1TilingData.baseN, coreNumEachExpert1,
                                  coreNumEachExpert1Vec);

    tilingParams2.SetBaseParams(currentTokensMM1, currentTokensMM1, n2, k2, hasBias2);
    tilingParams2.SetTilingParams(tilingData->mm2TilingData.baseM, tilingData->mm2TilingData.baseN, coreNumEachExpert2,
                                  coreNumEachExpert1);
    CubeTiling(tilingParams1);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    ComputeExpertMM1(tilingParams1, tilingParams2, mm1ExpertParallInfo);
    ComputeExpertMM2(tilingParams2, mm2ExpertParallInfo);
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void
FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::ComputeExpertParallNum(const uint32_t expertI,
                                                                                  ExpertParallInfo &expertParallInfo)
{
    if (expertI == expertNum) {
        expertParallInfo.expertParallelism = Min(expertParallInfo.size, expertParallInfo.maxExpertParallelism);
        expertParallInfo.start = 0;
        tokens = ubTokens.GetValue(expertParallInfo.expertIdxBuf[0]);
        return;
    }
    bool isFull = expertParallInfo.AddExpert(expertI, tokens, tokensOffset);
    if (isFull) {
        // the buffer is full, it's time to compute experts parallelly
        expertParallInfo.expertParallelism = expertParallInfo.maxExpertParallelism;
        expertParallInfo.start = 0;
    } else {
        expertParallInfo.expertParallelism = 0; // store this expert information, not to compute the expert
    }
}

template <typename xT, typename wT, typename mm1Type, typename mm2Type, typename c1T, typename yT, typename biasT>
__aicore__ inline void FFNAntiQuantMSD<xT, wT, mm1Type, mm2Type, c1T, yT, biasT>::Process()
{
    ExpertParallInfo mm1ExpertParallInfo(cubeCoreNum, Ceil(n1, tilingData->mm1TilingData.baseN));
    ExpertParallInfo mm2ExpertParallInfo(cubeCoreNum, Ceil(n2, tilingData->mm2TilingData.baseN));
    if (mm2ExpertParallInfo.maxSize < mm1ExpertParallInfo.maxSize) {
        // MM1 first computes experts, then MM2. If an expert is not processed by mm1, it cannot be processed by mm2.
        // Expertsin MM1 buffer are all unprocessed, so the buffer of MM2 should hold these experts too.
        // This requires MM2's maxSize >= MM1's maxSize, no matter what relative value of both of maxExpertParallelism.
        mm2ExpertParallInfo.maxSize = mm1ExpertParallInfo.maxSize;
    }
    if (mm2ExpertParallInfo.maxExpertParallelism > mm1ExpertParallInfo.maxExpertParallelism) {
        // Now MM2's expert parallelism is not supported to be larger than MM1's.
        // If it happens, one should consider adjusting workspace1Size and workspace2Size.
        mm2ExpertParallInfo.maxExpertParallelism = mm1ExpertParallInfo.maxExpertParallelism;
        mm2ExpertParallInfo.maxSize = mm1ExpertParallInfo.maxSize;
    }
    uint32_t tokensThisLoop = 0;
    tokensOffset = 0;
    tokens = 0;

    for (uint32_t expertI(0); expertI < expertNum || mm1ExpertParallInfo.size > 0 || mm2ExpertParallInfo.size > 0;
         ++expertI) {
        tokensOffset += tokensThisLoop; // cannot ignore Step5
        if (likely(expertI < expertNum)) {
            tokensThisLoop = ubTokens.GetValue(expertI);
            if (tokensThisLoop == 0) {
                continue;
            }
            maxTokenInParallelGroup = Max(tokensThisLoop, maxTokenInParallelGroup);
            tokens = tokensThisLoop;
        }
        // Step0: determine expert parallalism and core number for each expert.
        ComputeExpertParallNum(expertI, mm1ExpertParallInfo);
        ComputeExpertParallNum(expertI, mm2ExpertParallInfo);

        // Step1: mm
        if (mm1ExpertParallInfo.expertParallelism > 0) {
            ComputeExpert(mm1ExpertParallInfo, mm2ExpertParallInfo);
            maxTokenInParallelGroup = 0;
        }

        // Step2: post-process
        if (mm1ExpertParallInfo.expertParallelism > 0) {
            mm1ExpertParallInfo.Clear(mm1ExpertParallInfo.start);
        }
        if (mm2ExpertParallInfo.expertParallelism > 0) {
            mm2ExpertParallInfo.Clear(mm2ExpertParallInfo.start);
        }
    }
}

} // namespace FFN

#endif // ASCENDC_FFN_ANTIQUANT_MSD_H
