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
 * \file flash_attention_score_block_vec_infer.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_H_
#include "flash_attention_score_block_vec_base.h"
#include "util_regbase.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_flash_decode.h"
#include "vf/vf_post_quant.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;


namespace BaseApi {
TEMPLATES_DEF
class FABlockVecInfer
    : public FABlockVecBase<FABlockVecInfer<TEMPLATE_ARGS>, TEMPLATE_ARGS> {
public:
    using BaseClass = FABlockVecBase<FABlockVecInfer<TEMPLATE_ARGS>, TEMPLATE_ARGS>;
public:
    /* ================编译期常量信息======================= */
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    static constexpr uint32_t preloadTimes = 3;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;

    /* =====================GM变量========================== */
    GlobalTensor<float> softmaxLseGm;

    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;
    FDGmType accumOutGm;
    FDGmType softmaxFDMaxGm;
    FDGmType softmaxFDSumGm;

    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    postQuantBf16GmType postQuantScaleBf16Gm;
    postQuantBf16GmType postQuantOffsetBf16Gm;

    /* =====================UB变量========================== */
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue; // FD
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue; // FD
    TQue<QuePosition::VECIN, 1> postQuantScaleQue; // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue; // postQuant

    __aicore__ inline FABlockVecInfer() {};
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *deqScaleQK, __gm__ uint8_t *quantScaleP,
        __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *dropMask, 
        __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<uint8_t> &dropMaskUb) {}
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr);

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t dSizeAligned64);

    __aicore__ inline void FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount, uint32_t dSizeAligned64);

    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);

private:
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm, 
        __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqKvLen);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CombineSplitKVRes(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, 
        ConstInfo<isInfer, hasRope> &constInfo, uint32_t splitSize, uint64_t lseOffset);

    __aicore__ inline void Bmm2FDOut(LocalTensor<T> &vec2ResUb, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, int64_t vec2CalcSize);

    __aicore__ inline void CopyLseIn(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void CopyFinalResOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow,
                                           uint32_t dealRowCount, uint64_t perChannelQuantOffset);

    __aicore__ inline void CopyAccumOutIn(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ReduceFinalRes(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                          uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void ReduceFDDataCopyOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb,
                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount);

};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    sharedParams.bSize = inputParamsRegbase.bSize;
    sharedParams.n2Size = inputParamsRegbase.n2Size;
    sharedParams.gSize = inputParamsRegbase.gSize;
    sharedParams.s1Size = inputParamsRegbase.s1Size;
    sharedParams.s2Size = inputParamsRegbase.s2Size;
    sharedParams.dSize = inputParamsRegbase.dSize;
    sharedParams.dSizeV = inputParamsRegbase.dSizeV;
    if constexpr (hasRope) {
        sharedParams.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        sharedParams.dSizeRope = 0;
    }
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.implMode = inputParamsRegbase.implMode;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
 
    if constexpr (isFd) {
        sharedParams.splitKVNum = inputParamsRegbase.kvSplitPart;
    }
    sharedParams.isBSNDOut = inputParamsRegbase.isBSNDOut;
    sharedParams.fromFused = inputParamsRegbase.fromFused;
    sharedParams.isRowInvalid = inputParamsRegbase.isRowInvalid;
    sharedParams.headNumRatio = inputParamsRegbase.headNumRatio;
    sharedParams.isGqa = inputParamsRegbase.isGqa;
    sharedParams.isKvContinuous = inputParamsRegbase.isKvContinuous;
    sharedParams.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
    sharedParams.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    sharedParams.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
    sharedParams.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;
    sharedParams.isQHasLeftPadding = inputParamsRegbase.isQHasLeftPadding;
    sharedParams.isKVHasLeftPadding = inputParamsRegbase.isKVHasLeftPadding;
    // pageAttention
    if constexpr (isPa) {
        sharedParams.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        sharedParams.blockSize = inputParamsRegbase.blockSize;
        sharedParams.paLayoutType = inputParamsRegbase.paLayoutType;
        sharedParams.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }
 
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    /* 多核切分偏移计算 */
    sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];
    sharedParams.needInit = this->tilingData->initOutputParams.needInit;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfo) 
{
    this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
    if constexpr (POST_QUANT) {
        this->attentionOutInitGm.SetGlobalBuffer((__gm__ half *)attentionOut);
    }
    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    constInfo.isSoftmaxLseEnable = this->tilingData->inputParamsRegbase.isSoftMaxLseEnable;
    if (this->tilingData->initOutputParams.needInit == 1) {
        InitOutputSingleCore(constInfo);
        // lse output
        if (constInfo.isSoftmaxLseEnable) {
            SyncAll();
            InitLseOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *deqScaleQK, __gm__ uint8_t *quantScaleP,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,  __gm__ uint8_t *dropMask, 
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    BaseClass::InitCommonGlobalBuffer(pse, deqScaleQ, deqScaleK, deqScaleV, quantScaleP, prefix, attenMask, workspace, constInfo);
    if constexpr (isFd) {
        workspace -= singleCoreOffset * preloadTimes * aicIdx;             // 让当前的workspace地址回到基地址
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        int32_t actualCoreNums = inputParamsRegbase.bSize * constInfo.n2Size * constInfo.splitKVNum;
        workspace += actualCoreNums * singleCoreOffset * preloadTimes;     // 针对所有核跳过其前面的所有workspace

        uint64_t accumOutSize = this->tilingData->inputParamsRegbase.accumOutSize;
        uint64_t logSumExpSize = this->tilingData->inputParamsRegbase.logSumExpSize;
        accumOutGm.SetGlobalBuffer((__gm__ T *)(workspace));
        workspace += accumOutSize * sizeof(float);
        softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
        softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
    }
    if constexpr (POST_QUANT) {
        this->InitPostQuant(constInfo, postQuantScale, postQuantOffset);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
{
    if (postQuantScale != nullptr) {
        constInfo.postQuantScaleValue = *(reinterpret_cast<__gm__ float*>(postQuantScale));
    }
    if (postQuantOffset != nullptr) {
        constInfo.isPostQuantOffsetExist = true;
        constInfo.postQuantOffsetValue = *(reinterpret_cast<__gm__ float*>(postQuantOffset));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, 2048); // 2K
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, 2048); // 2K
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo)
{
    this->tPipe->Reset();
    this->tPipe->InitBuffer(lseTmpBuff, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxMaxInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxSumInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(FDResOutputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(accumOutInputQue, 1, bufferSizeByte32K);
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, bufferSizeByte32K);
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, bufferSizeByte32K);
        }
    }
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND, 每行结果存为8个重复lse元素(32B对齐)
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo,
    GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr)
{
    int64_t bIdx = constInfo.aivIdx / constInfo.n2Size;
    int64_t n2Idx = constInfo.aivIdx % constInfo.n2Size;
    int64_t batchSize = this->tilingData->inputParamsRegbase.bSize;
    if (constInfo.aivIdx >= batchSize * constInfo.n2Size) {
        return;
    }
    int64_t actualSeqLen;
    GetActualSeqLenKV(constInfo, keyGm, actualSeqKvlenAddr, bIdx, actualSeqLen);
    if (actualSeqLen == 0) {
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * constInfo.n2GDv + n2Idx * constInfo.gDv;
    constInfo.actualCombineLoopSize = (actualSeqLen + constInfo.sInnerLoopSize - 1) / constInfo.sInnerLoopSize;
    CombineSplitKVRes(constInfo, attenOutOffset, bIdx, n2Idx);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUb)
{
    if constexpr (isFd) {
        ComputeLogSumExpAndCopyToGm(runInfo, constInfo);
        return;
    }
    SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    if constexpr (isFd) {
        Bmm2FDOut(vec2ResUb, runInfo, constInfo, vec2CalcSize);
    } else {
        this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, 
    GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqLen)
{
    int64_t s2InCurrentBatch = constInfo.s2Size;
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, boIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2InCurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            s2InCurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    if (constInfo.isActualLenDimsKVNull) {
        actualSeqLen = s2InCurrentBatch;
    } else {
        actualSeqLen = (constInfo.actualSeqLenKVSize == 1) ? actualSeqKvlenAddr[0] :
                                                             actualSeqKvlenAddr[boIdx];
    }
    if (constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualSeqLen - constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLen = 0;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    if (!constInfo.isSoftmaxLseEnable) {
        return;
    }
    LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
    ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.halfS1RealSize);
    softmaxLseQueue.template EnQue(lseUb);
    softmaxLseQueue.DeQue<float>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = runInfo.halfS1RealSize;
    intriParams1.srcStride = 0;
    if (layout == LayOutTypeEnum::LAYOUT_TND) {
        if (constInfo.isGqa) {
            intriParams1.dstStride = 0;
        } else {
            intriParams1.dstStride = sizeof(float) * (constInfo.n2G - 1);
        }
    } else {
        intriParams1.dstStride = 0;
    }
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParams1);
    softmaxLseQueue.FreeTensor(lseUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    if constexpr (POST_QUANT) {
        InitOutput<half>(this->attentionOutInitGm[constInfo.aivIdx * initParams.singleCoreSize / 2], singleInitOutputSize / 2, 0.0);
    } else {
        InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize], singleInitOutputSize, 0.0);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (constInfo.aivIdx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)], 
            singleCoreLseSize, 3e+99); // 3e+99:set the value of invalid batch to inf
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::CombineSplitKVRes(
    ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse =
        bufferSizeByte32K / (FA_BYTE_BLOCK * constInfo.splitKVNum); // 32K / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = bufferSizeByte32K / sizeof(float) / (uint32_t)dVTemplateType;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (constInfo.gSize > gSplitMax) {
        gSplitSize = (gSplitSize > gSplitMax) ? gSplitMax : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > constInfo.gSize) ? constInfo.gSize : gSplitSize;
    }
    uint32_t loopCount = CeilDivision(constInfo.gSize, gSplitSize);
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>(); // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, fp32BaseSize};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape

    uint64_t perChannelQuantOffset = n2Idx * constInfo.dSizeV * constInfo.gSize;
    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(constInfo, bIdx, n2Idx, startRow, gSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + i * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, constInfo, gSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(constInfo, bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(constInfo, attenOutOffset, tmp1, startRow, gSplitSize, perChannelQuantOffset);
    }
    // 尾块处理
    if (tailSplitSize > 0) {
        uint32_t startRow = (loopCount - 1) * gSplitSize;
        CopyLseIn(constInfo, bIdx, n2Idx, startRow, tailSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + (loopCount - 1) * gSplitSize;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, constInfo, tailSplitSize, lseOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(constInfo, bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, tailSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(constInfo, attenOutOffset, tmp1, startRow, tailSplitSize, perChannelQuantOffset);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::ComputeScaleValue(
    LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, ConstInfo<isInfer, hasRope> &constInfo, 
    uint32_t splitSize, uint64_t lseOffset)
{
    LocalTensor<T> lseOutputUb;
    if (constInfo.isSoftmaxLseEnable) {
        lseOutputUb = softmaxLseQueue.template AllocTensor<T>();
    }
    ComputeScaleValue_VF(lseMaxUb, lseSumUb, lseOutputUb, splitSize, constInfo.actualCombineLoopSize,
                         constInfo.isSoftmaxLseEnable);
    if (constInfo.isSoftmaxLseEnable) {
        softmaxLseQueue.template EnQue<T>(lseOutputUb);
        softmaxLseQueue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = splitSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[lseOffset], lseOutputUb, intriParams1);
        softmaxLseQueue.FreeTensor(lseOutputUb);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::Bmm2FDOut(LocalTensor<T> &vec2ResUb,
    RunInfo<isInfer> &runInfo,  ConstInfo<isInfer, hasRope> &constInfo, int64_t vec2CalcSize)
{
    LocalTensor<T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    SetFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    WaitFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    attenOut = vec2ResUb;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.halfS1RealSize;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t base = (runInfo.boIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV +
                   runInfo.n2oIdx * constInfo.gSize * constInfo.dSizeV) *
                      constInfo.splitKVNum +
                  mStart * constInfo.dSizeV;

    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * constInfo.gSize * constInfo.dSizeV],
                attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::CopyLseIn(ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = constInfo.splitKVNum;
    copyInParams.blockLen = dealRowCount * fp32BaseSize * sizeof(T);
    copyInParams.srcStride = (constInfo.gSize - dealRowCount) * fp32BaseSize * sizeof(T);
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    uint64_t combineLseOffset =
        ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum + n2Idx * constInfo.splitKVNum) *
            constInfo.gSize * fp32BaseSize +
        startRow * fp32BaseSize;

    DataCopyPad(softmaxMaxLocal, softmaxFDMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxFDSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::CopyFinalResOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, 
    LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t perChannelQuantOffset)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = FDResOutputQue.AllocTensor<OUTPUT_T>();
    uint32_t dSizeAligned64 = (uint32_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dSizeAligned64};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape
    if constexpr (!POST_QUANT) {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dSizeAligned64);
    } else {
        FDPostQuant(constInfo, tmpBmm2ResCastTensor, accumOutLocal, perChannelQuantOffset + startRow * constInfo.dSizeV, dealRowCount, dSizeAligned64);
    }

    FDResOutputQue.EnQue(tmpBmm2ResCastTensor);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(constInfo, attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeAligned64,
                        constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::ReduceFinalRes(ConstInfo<isInfer, hasRope> &constInfo, 
    uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow, uint32_t dealRowCount)
{
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    for (uint32_t j = 0; j < constInfo.actualCombineLoopSize; ++j) {
        // 第一次，mul结果直接放到dst里
        CopyAccumOutIn(constInfo, bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_VF<T>(dst, lseLocal, accumOutLocal, dealRowCount, dSizeAligned64, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::CopyAccumOutIn(ConstInfo<isInfer, hasRope> &constInfo, 
    uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD){
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dSizeAligned64 - constInfo.dSizeV) / 8; // 8 for align factor

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (dSizeAligned64 - constInfo.dSizeV) % 8; // 8 for align factor
    copyInPadParams.paddingValue = 0;

    uint64_t combineAccumOutOffset = ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum +
                                      n2Idx * constInfo.splitKVNum + splitKVIndex) *
                                         constInfo.gSize * constInfo.dSizeV +
                                     startRow * constInfo.dSizeV;
    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FABlockVecInfer<TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = constInfo.n2G * runInfo.s1SizeAcc;
        n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset = runInfo.s1oIdx * this->s1BaseSize + constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t gmOffset =
        runInfo.boIdx * constInfo.n2Size * constInfo.splitKVNum * constInfo.gSize * fp32BaseSize +
        runInfo.n2oIdx * constInfo.splitKVNum * constInfo.gSize * fp32BaseSize +
        runInfo.flashDecodeS2Idx * constInfo.gSize * fp32BaseSize + mStart * fp32BaseSize;
    // Copy sum to gm
    this->BroadCastAndCopyOut(runInfo, softmaxFDSumGm, softmaxFDMaxGm, gmOffset, calculateSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::ReduceFDDataCopyOut(ConstInfo<isInfer, hasRope> &constInfo,
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FA_BYTE_BLOCK / sizeof(OUTPUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void
FABlockVecInfer<TEMPLATE_ARGS>::PostQuant(ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo,
                                          LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                          int64_t vec2S1Idx, int64_t dSizeAligned64)
{
    AscendQuant(attenOut, vec2ResUb, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue,
                runInfo.vec2S1RealSize * dSizeAligned64);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInfer<TEMPLATE_ARGS>::FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo,
                                                                   LocalTensor<OUTPUT_T> &attenOut,
                                                                   LocalTensor<T> &accumOutLocal,
                                                                   uint64_t perChannelQuantOffset,
                                                                   uint32_t dealRowCount, uint32_t dSizeAligned64)
{
    PostQuantPerTensorVF<T, OUTPUT_T, true>(attenOut, accumOutLocal, constInfo.postQuantScaleValue,
                                            constInfo.postQuantOffsetValue, dealRowCount, constInfo.dSizeV,
                                            dSizeAligned64);
}
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_H_