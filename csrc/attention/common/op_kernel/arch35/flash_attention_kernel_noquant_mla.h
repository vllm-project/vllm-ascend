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
 * \file flash_attention_kernel_noquant_mla.h
 * \brief
 */

#ifndef FLASH_ATTENTION_KERNEL_NOQUANT_MLA_H_
#define FLASH_ATTENTION_KERNEL_NOQUANT_MLA_H_

#include "attenmask.h"
#include "pse.h"
#include "flash_attention_block_cube_noquant_mla.h"
#include "flash_attention_noquant_block_vec_infer.h"

using namespace fa_base_matmul;

template <typename CubeBlockType, typename VecBlockType>
class FAKernelNoquantMla {
public:
    ARGS_TRAITS;
    __aicore__ inline FAKernelNoquantMla() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                                     __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                     __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                     const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void SetFlag3Buffer();
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndx, int64_t gS1Index, int64_t &multiCoreInnerIdx, RunParamStr<isInfer>& runParam);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam);
    __aicore__ inline bool IsLastBN(uint32_t bnStartIdx, uint32_t bnEndIdx);

private:
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = nullptr;
    TPipe *tPipe = nullptr;

    static constexpr uint32_t dTemplateAlign64 = CubeBlockType::dTemplateAlign64;
    static constexpr uint32_t s1BaseSize = CubeBlockType::s1BaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr bool splitD = CubeBlockType::splitD;
    static constexpr uint32_t PRELOAD_N = 1;
    static constexpr bool POST_QUANT = VecBlockType::POST_QUANT;

    /* Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    /* Public Variable */
    __gm__ uint8_t *currentKey;
    __gm__ uint8_t *currentValue;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;

    static constexpr bool useDn = false;

    /*========核Index信息========*/
    int32_t aicIdx;

    ConstInfo<isInfer, hasRope> constInfo;
    PseInfo pseInfo;
    AttenMaskInfo attenMaskInfo;
    CVSharedParams<isInfer, isPa> sharedParams;

    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    uint64_t s1SizeAcc = 0;
    uint64_t s2SizeAcc = 0;
    uint64_t b1SSOffset = 0;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    BufferManager<BufferType::L1> l1BufferManager;
    uint32_t l1BuffSize = 524288;  // 512k
    // mm1和mm2右矩阵，在L1上复用，其中K_rope内存空间与bmm2的左矩阵p复用
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm12Bmm2AL1Buffers;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    this->tilingData = tiling;
    constInfo.aivIdx = GetBlockIdx();
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIV {
        this->aicIdx = constInfo.aivIdx >> 1;
    }
    if ASCEND_IS_AIC {
        this->aicIdx = constInfo.aivIdx;
    }
    this->tPipe = tPipe;

    // Init Vec
    this->vecBlock.InitVecBlock(this->tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, 
        attenMaskInfo, pseInfo);
    this->vecBlock.CleanOutput(softmaxLse, attentionOut, constInfo);
    InitMMResBuf();

    if ASCEND_IS_AIC {
        this->cubeBlock.InitCubeBlock(this->tPipe, query, key, value, blockTable, queryRope, keyRope, tiling, &l1BufferManager, &mm12Bmm2AL1Buffers);
    }

    ComputeConstexpr();
    InitInput(query, key, value, pse, attenMask, actualSeqLengths, actualSeqLengthsKv, blockTable, postQuantScale,
        postQuantOffset, queryRope, keyRope, softmaxLse, attentionOut, workspace, tiling, tPipe);
    InitBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    constexpr uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    uint32_t mm12RightSize = max(static_cast<uint32_t>(dTemplateType), static_cast<uint32_t>(dVTemplateType)) * s2BaseSize * sizeof(INPUT_T);

    // L1
    l1BufferManager.Init(tPipe, l1BuffSize); // 512k
    // 3Buffer
    mm12Bmm2AL1Buffers.Init(l1BufferManager, mm12RightSize);    // L1: 144k * 3 = 432k
    if ASCEND_IS_AIC {
        SetFlag3Buffer();
    }
    // UB  现在最大情况是250k，不包括pse
    ubBufferManager.Init(tPipe, mm1ResultSize * 2 + mm2ResultSize * 2);     // ub: 16k * 2 + 64k * 2 = 160k
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize); // ub: 32 * 128 * 4 = 16k, 16k * 2 = 32k
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::SetFlag3Buffer()
{
    mm12Bmm2AL1Buffers.Get().SetEventID();
    mm12Bmm2AL1Buffers.Get().SetEventID();
    mm12Bmm2AL1Buffers.Get().SetEventID();
    SetFlag<HardEvent::MTE1_MTE2>(mm12Bmm2AL1Buffers.Get().GetEventID<HardEvent::MTE1_MTE2>());
    SetFlag<HardEvent::MTE1_MTE2>(mm12Bmm2AL1Buffers.Get().GetEventID<HardEvent::MTE1_MTE2>());
    SetFlag<HardEvent::MTE1_MTE2>(mm12Bmm2AL1Buffers.Get().GetEventID<HardEvent::MTE1_MTE2>());
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::InitInput(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    // init global buffer
    ListTensorDesc keyListTensorDescInit((__gm__ void*)key);
    ListTensorDesc valueListTensorDescInit((__gm__ void*)value);
    currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }

    if (this->tilingData->inputParamsRegbase.isActualSeqLengthsNull != 1) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
    }
    if (this->tilingData->inputParamsRegbase.isActualSeqLengthsKVNull != 1) {
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    }

    this->vecBlock.InitGlobalBuffer(pse, nullptr, nullptr, nullptr, nullptr, postQuantScale, postQuantOffset,
        nullptr, attenMask, nullptr, nullptr, nullptr, nullptr, nullptr, workspace, 0, this->aicIdx, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::InitBuffer()
{
    if ASCEND_IS_AIV {
        this->vecBlock.InitLocalBuffer(this->tPipe, constInfo);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;

    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;

    constInfo.bSize = inputParamsRegbase.bSize;
    constInfo.t1Size = inputParamsRegbase.t1Size;
    constInfo.n2Size = inputParamsRegbase.n2Size;
    constInfo.s1Size = inputParamsRegbase.s1Size;
    constInfo.s2Size = inputParamsRegbase.s2Size;
    constInfo.dSize = inputParamsRegbase.dSize;
    constInfo.dSizeV = inputParamsRegbase.dSizeV;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    if constexpr (hasRope) {
        constInfo.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = inputParamsRegbase.gSize;
    if (inputParamsRegbase.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_NBSD) ||
        inputParamsRegbase.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_NBSD) ||
        inputParamsRegbase.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_NBSD)) {
        constInfo.t1Size = constInfo.bSize * constInfo.s1Size;
    }
    constInfo.s1OuterSize = this->tilingData->multiCoreParamsRegbase.s1OuterSize;

    constInfo.s1D = constInfo.s1Size * constInfo.dSize;
    constInfo.s2D = constInfo.s2Size * constInfo.dSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2D = constInfo.n2Size * constInfo.dSize;
    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    int64_t bSize = inputParamsRegbase.bSize;
    constInfo.bN2D = bSize * constInfo.n2D;
    constInfo.bN2Dv = bSize * constInfo.n2Dv;
    constInfo.gS1D = constInfo.gSize * constInfo.s1D;
    constInfo.n2S2D = constInfo.n2Size * constInfo.s2D;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.bN2GD = bSize * constInfo.n2GD;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.bN2GDv = bSize * constInfo.n2GDv;
    constInfo.n2GS1D = constInfo.n2Size * constInfo.gS1D;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;

    // 计算切分轴的乘积
    constInfo.s2BaseN2D = s2BaseSize * constInfo.n2D;
    constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
    constInfo.n2S2D /= inputParamsRegbase.headNumRatio;
    constInfo.n2S2Dv /= inputParamsRegbase.headNumRatio;
    constInfo.s2BaseN2D /= inputParamsRegbase.headNumRatio;
    constInfo.s2BaseN2Dv /= inputParamsRegbase.headNumRatio;

    if constexpr (hasRope) {
        constInfo.s1DR = constInfo.s1Size * constInfo.dSizeRope;
        constInfo.s2DR = constInfo.s2Size * constInfo.dSizeRope;
        constInfo.gDR = constInfo.gSize * constInfo.dSizeRope;
        constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
        constInfo.bN2DR = bSize * constInfo.n2DR;
        constInfo.gS1DR = constInfo.gSize * constInfo.s1DR;
        constInfo.n2S2DR = constInfo.n2Size * constInfo.s2DR;
        constInfo.n2GDR = constInfo.n2Size * constInfo.gDR;
        constInfo.bN2GDR = bSize * constInfo.n2GDR;
        constInfo.n2GS1DR = constInfo.n2Size * constInfo.gS1DR;
        constInfo.s2BaseN2DR = s2BaseSize * constInfo.n2DR;
    }
    constInfo.layoutType = inputParamsRegbase.layoutType;
    constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        constInfo.s1BaseN2GD = s1BaseSize * constInfo.n2GD;
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
            constInfo.mm1RopeKa = constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.n2DR;
        }
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2D;
        constInfo.mm2Kb = constInfo.n2Dv;
        constInfo.mm1Kb /= inputParamsRegbase.headNumRatio;
        constInfo.mm2Kb /= inputParamsRegbase.headNumRatio;
        constInfo.attentionOutStride = 0;
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        constInfo.s1BaseD = s1BaseSize * constInfo.dSize;
        constInfo.s2BaseD = s2BaseSize * constInfo.dSize;
        constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
        constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
        if constexpr (hasRope) {
            constInfo.s1BaseDR = s1BaseSize * constInfo.dSizeRope;
            constInfo.s2BaseDR = s2BaseSize * constInfo.dSizeRope;
            constInfo.mm1RopeKa = constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.dSizeRope;
        }
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.dSize;
        constInfo.mm2Kb = constInfo.dSizeV;
        constInfo.attentionOutStride = 0;
    }

    if constexpr (hasAtten) {
        this->attenMaskInfo.preTokens = inputParamsRegbase.preTokens;
        this->attenMaskInfo.nextTokens = inputParamsRegbase.nextTokens;
        this->attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
        this->attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
        this->attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
        this->attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
        this->attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
    }

    constInfo.isRowInvalid = inputParamsRegbase.isRowInvalid;
    constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
    constInfo.dSizeRope = inputParamsRegbase.ropeHeadSize;
    constInfo.isGqa = inputParamsRegbase.isGqa;
    constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    constInfo.n2GDR = constInfo.gSize * constInfo.n2Size * constInfo.dSizeRope;
    constInfo.n2DR = constInfo.n2Size * constInfo.dSizeRope;
    constInfo.isKvContinuous = inputParamsRegbase.isKvContinuous;
    constInfo.actualSeqLenSize = inputParamsRegbase.actualSeqLengthsSize;
    constInfo.actualSeqLenKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    constInfo.isActualLenDimsNull = (inputParamsRegbase.isActualSeqLengthsNull == 1) ? true : false;
    constInfo.isActualLenDimsKVNull = (inputParamsRegbase.isActualSeqLengthsKVNull == 1) ? true : false;
    constInfo.isQHasLeftPadding = (inputParamsRegbase.isQHasLeftPadding == 1) ? true : false;
    constInfo.isKVHasLeftPadding = (inputParamsRegbase.isKVHasLeftPadding == 1) ? true : false;
    // pageAttention
    constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
    constInfo.blockSize = inputParamsRegbase.blockSize;
    constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
    constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    constInfo.keyBnStride = inputParamsRegbase.keyStrides.bnStride;
    constInfo.keyN2Stride = inputParamsRegbase.keyStrides.n2Stride;
    constInfo.valueBnStride = inputParamsRegbase.valueStrides.bnStride;
    constInfo.valueN2Stride = inputParamsRegbase.valueStrides.n2Stride;
    constInfo.keyRopeBnStride = inputParamsRegbase.kRopeStrides.bnStride;
    constInfo.keyRopeN2Stride = inputParamsRegbase.kRopeStrides.n2Stride;

    // service vector2
    constInfo.transposeLayout = inputParamsRegbase.transposeLayout;
    if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND)) {
        constInfo.attentionOutStride = 
            (constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
    } else if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::TND_NTD)) {
        constInfo.attentionOutStride = 0;
    } 

    // lse output
    constInfo.isSoftmaxLseEnable = inputParamsRegbase.isSoftMaxLseEnable;

    if constexpr (POST_QUANT) {
        constInfo.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        constInfo.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::Process()
{
    int32_t actualCoreNums = this->tilingData->multiCoreParamsRegbase.coreNum;
    // SyncAll Cube和Vector都需要调用
    if (this->tilingData->initOutputParams.needInit) {
        SyncAll<false>();
    }
    if (aicIdx >= actualCoreNums) {
        return;
    }

    // 确定核内切分起点
    int64_t gS1StartIdx = 0;
    int64_t gS1EndIdx = 1;
    uint32_t bnStartIdx = 0;
    uint32_t bnEndIdx = 1;
    int64_t s2LoopStart = 0;
    int64_t s2LoopLimit = 0;

    if constexpr (!isFd) {
        bnStartIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx];
        gS1StartIdx = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx];
        if (likely((this->tilingData->multiCoreParamsRegbase.coreNum - 1) > aicIdx)) {
            bnEndIdx = this->tilingData->multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
            // 下一个核从0开始gs1循环，当前核bn不需要多计算一个，否则需要多计算一个bn
            if (this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1] != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->tilingData->inputParamsRegbase.bSize * constInfo.n2Size;
        }
    }
    int64_t taskId = 0;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[NUM_4];
    RunParamStr<isInfer> runParam;

    int64_t multiCoreInnerIdx = 0;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; bnIdx++) {
        bool lastBN = IsLastBN(bnIdx, bnEndIdx);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / constInfo.n2Size;
            runParam.n2oIdx = bnIdx % constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, this->attenMaskInfo,
            keyGm, actualSeqQlenAddr, actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, lastBN,
            this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1]);

        gS1EndIdx = runParam.s1LoopTimes;
        if (lastBN && runParam.s1LoopTimes == 0) {
            runParam.s1LoopTimes += 1;
        }
        for (int64_t gS1Index = gS1StartIdx; gS1Index <runParam.s1LoopTimes; gS1Index++) {
            s2LoopLimit = 0;
            this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, multiCoreInnerIdx, runParam);
            bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo,
                gS1Index, actualSeqQlenAddr, this->pseInfo);
            bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo);
            bool lastLoopThisCore = lastBN && (gS1Index == runParam.s1LoopTimes - 1);
            bool lastBnNoNeedCalc = ComputeLastBN<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo,
                actualSeqQlenAddr);
            if (((s1NoNeedCalc || s2NoNeedCalc) && !lastLoopThisCore) || lastBnNoNeedCalc) {
                continue;
            }
            multiCoreInnerIdx++;
            // s2轴循环计数，支持sparse和非sparse场景
            s2LoopLimit = runParam.s2LoopEndIdx - 1;
            if (lastLoopThisCore) {
                isLastBmm1 = true;
                s2LoopLimit += PRELOAD_N;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
                if (s2LoopCount < runParam.s2LoopEndIdx) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, runParam.s2LoopEndIdx - 1,
                        multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, runParam, isLastBmm1 &&
                            (s2LoopCount == (runParam.s2LoopEndIdx - 1)), constInfo);
                    }
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec1(this->mm12Bmm2AL1Buffers.Get(), this->bmm1Buffers.Get(), runInfo1,
                            this->constInfo);
                    }
                }
                if (taskId >= PRELOAD_N) {
                    RunInfo<isInfer> &runInfo2 = runInfo[(taskId - PRELOAD_N) & 3];
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), runInfo2, constInfo);
                    }
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo2, this->constInfo);
                    }
                }
                taskId++;
            }
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndx, int64_t gS1Index, int64_t &multiCoreInnerIdx, RunParamStr<isInfer>& runParam)
{
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (runParam.boIdx == 0) {
            this->s1SizeAcc = 0;
            this->s2SizeAcc = 0;
        } else {
            this->s1SizeAcc = actualSeqQlenAddr[runParam.boIdx - 1];
            if constexpr (isPa) {
                this->s2SizeAcc = 0;
                for (uint32_t boIdx = 0; boIdx < runParam.boIdx; boIdx++) {
                    this->s2SizeAcc += actualSeqKvlenAddr[boIdx];
                }
            } else {
                this->s2SizeAcc = actualSeqKvlenAddr[runParam.boIdx - 1];
            }
        }
    }
    runParam.goIdx = gS1Index / constInfo.s1OuterSize;
    runParam.s1oIdx = gS1Index % constInfo.s1OuterSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::SetRunInfo(RunInfo<isInfer> &runInfo,
    RunParamStr<isInfer>& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx % 2;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId % 2;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, constInfo.s1Size, constInfo.s2Size);
        runInfo.b1SSOffset = this->b1SSOffset;
    } else {
        runInfo.b1SSOffset = runInfo.boIdx * constInfo.s1S2;
    }

    runInfo.actualS1Size = constInfo.s1Size;
    runInfo.actualS2Size = constInfo.s2Size;

    this->ComputeBmm1Tail(runInfo, runParam);
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, useDn, enableKVPrefix>(runParam, constInfo, s2LoopCount + runInfo.s2StartIdx
        / s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBoidx(
    int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvLen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    if constexpr (isPa) {
        // TND + PA 时 act_seq_kv 是非累加的
        actualSeqKvLen = actualSeqKvlenAddr[boIdx];
    } else {
        actualSeqKvLen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FAKernelNoquantMla<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer>& runParam)
{
    // -----------S1 Base Related----------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;
    runInfo.vecCoreOffset = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;

    // -----------S2 Base Related-----------------
    runInfo.s2RealSize = s2BaseSize;
    if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool FAKernelNoquantMla<CubeBlockType, VecBlockType>::IsLastBN(uint32_t bnStartIdx, uint32_t bnEndIdx)
{
    if constexpr(layout != LayOutTypeEnum::LAYOUT_TND) {
        if (bnStartIdx == bnEndIdx - 1) {
            return true;
        }
        if (constInfo.isActualLenDimsNull) {
            return false;
        } else {
            for (uint32_t bnIdx = bnStartIdx + 1; bnIdx < bnEndIdx; bnIdx++) {
                uint32_t boIdx = bnIdx / constInfo.n2Size;
                if (constInfo.actualSeqLenSize == 1 && actualSeqQlenAddr[0] != 0) {
                    return false;
                } else if (actualSeqQlenAddr[boIdx] != 0) {
                    return false;
                }
            }
            return true;
        }
    }
    // TND
    if (bnStartIdx != bnEndIdx - 1) {
        for (uint32_t bnIdx = bnStartIdx + 1; bnIdx < bnEndIdx; bnIdx++) {
            uint32_t boIdx = bnIdx / constInfo.n2Size;
            uint32_t boStart = bnStartIdx / constInfo.n2Size;
            if (actualSeqQlenAddr[boIdx] != actualSeqQlenAddr[boStart]) {
                return false;
            }
        }
    }
    return true;
}

#endif