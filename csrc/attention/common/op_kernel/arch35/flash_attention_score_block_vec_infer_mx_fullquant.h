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
 * \file flash_attention_score_block_vec_infer_mx_fullquant.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_MX_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_MX_FULLQUANT_H_
#include "flash_attention_score_block_vec_base_fullquant.h"
#include "util_regbase.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_post_quant.h"
#include "vf/vf_flash_decode.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF
class FABlockVecInferMxFullquant
    : public FABlockVecBaseFullquant<FABlockVecInferMxFullquant<TEMPLATE_ARGS>, TEMPLATE_ARGS> {
public:
    using BaseClass = FABlockVecBaseFullquant<FABlockVecInferMxFullquant<TEMPLATE_ARGS>, TEMPLATE_ARGS>;
public:
    /* ================编译期常量信息======================= */
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    static constexpr uint32_t preloadTimes = 3;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value &&
                                       !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    /* =====================GM变量========================== */
    GlobalTensor<float> softmaxLseGm;
    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;
    FDGmType accumOutGm;
    FDGmType softmaxFDSumGm;
    FDGmType softmaxFDMaxGm;

    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    postQuantBf16GmType postQuantOffsetBf16Gm;
    postQuantBf16GmType postQuantScaleBf16Gm;

    /* =====================UB变量========================== */
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;; // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;; // postQuant
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue; // FD
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue; // FD
    TQue<QuePosition::VECIN, 1> sinkQue; // AttentionSink

    __aicore__ inline FABlockVecInferMxFullquant() {};
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams,
        int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitDropOut(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace) {}
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *queryPaddingSize,
        __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitPostQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale,
        __gm__ uint8_t *postQuantOffset);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<uint8_t> &dropMaskUb) {}
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);
    __aicore__ inline void SoftmaxDataCopyOutFp8(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                                 LocalTensor<half> &sumUb, LocalTensor<half> &maxUb) {}
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo,
        GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr);

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo,
        LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t dSizeAligned64);

    __aicore__ inline void FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
        LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount, uint32_t dSizeAligned64);

    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize,
        uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
        GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);

private:
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetActualSeqLenKV(ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm,
        __gm__ int64_t *actualSeqKvlenAddr, int64_t boIdx, int64_t &actualSeqKvLen);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CombineSplitKVRes(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset,
                                             uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb,
        ConstInfo<isInfer, hasRope> &constInfo, uint32_t splitSize, uint64_t lseOffset, uint64_t sinkOffset);

    __aicore__ inline void Bmm2FDOut(LocalTensor<T> &vec2ResUb, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, int64_t vec2S1Idx, int64_t vec2CalcSize);

    __aicore__ inline void CopyLseIn(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx,
        uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void CopyFinalResOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset,
        LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t perChannelQuantOffset);

    __aicore__ inline void CopyAccumOutIn(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx,
        uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo,
                                                       ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void CopySinkIn(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void CopySinkFDIn(uint32_t splitSize, uint64_t sinkOffset);

    __aicore__ inline void Vec1SinkCompute(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);

    __aicore__ inline void Vec1SinkComputeGSFused(RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb);

    __aicore__ inline void ReduceFinalRes(ConstInfo<isInfer, hasRope> &constInfo, uint32_t bIdx, uint32_t n2Idx,
        LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void ReduceFDDataCopyOut(ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset,
        LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
        uint32_t actualColumnCount);

};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitCubeVecSharedParams(
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
    sharedParams.t1Size = inputParamsRegbase.t1Size;
    sharedParams.t2Size = inputParamsRegbase.t2Size;
    if constexpr (hasRope) {
        sharedParams.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        sharedParams.dSizeRope = 0;
    }
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.implMode = inputParamsRegbase.implMode;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;
    sharedParams.compressMode = inputParamsRegbase.attenMaskCompressMode;
    sharedParams.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
    sharedParams.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    if constexpr (isFd) {
        sharedParams.splitKVNum = inputParamsRegbase.kvSplitPart;
    }
    if constexpr (POST_QUANT) {
        sharedParams.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        sharedParams.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
    }
    sharedParams.transposeLayout = inputParamsRegbase.transposeLayout;
    sharedParams.fromFused = inputParamsRegbase.fromFused;
    sharedParams.headNumRatio = inputParamsRegbase.headNumRatio;
    sharedParams.isRowInvalid = inputParamsRegbase.isRowInvalid;
    sharedParams.isGqa = inputParamsRegbase.isGqa;
    sharedParams.isPfaGS1Merge = (inputParamsRegbase.isGqa && sharedParams.s1Size > 1);
    sharedParams.isKvContinuous = inputParamsRegbase.isKvContinuous;
    sharedParams.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
    sharedParams.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
    sharedParams.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
    sharedParams.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;
    sharedParams.isKVHasLeftPadding = inputParamsRegbase.isKVHasLeftPadding;
    sharedParams.isQHasLeftPadding = inputParamsRegbase.isQHasLeftPadding;
    // pageAttention
    if constexpr (isPa) {
        sharedParams.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        sharedParams.blockSize = inputParamsRegbase.blockSize;
        sharedParams.paLayoutType = inputParamsRegbase.paLayoutType;
        sharedParams.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }
    // prefix
    if constexpr (enableKVPrefix) {
        sharedParams.kvPrefixSize = inputParamsRegbase.prefixSeqInnerSize;
        sharedParams.isActualSharedPrefixLenNull = inputParamsRegbase.isActualSharedPrefixLenNull;
    }
    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    sharedParams.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[aicIdx];    /* 多核切分偏移计算 */
    sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    sharedParams.needInit = this->tilingData->initOutputParams.needInit;
    sharedParams.bnStartIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx];
    sharedParams.bnEndIdx = multiCoreParamsRegbase.bnStartIdx[aicIdx + 1];


    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            int loopNumTmp = sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t);
#pragma unroll
            for (int i = 0; i < loopNumTmp; ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }

}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfos)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if constexpr (POST_QUANT) {
            this->attentionOutInitGm.SetGlobalBuffer((__gm__ half *)attentionOut);
        }
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        constInfos.isSoftmaxLseEnable = this->tilingData->inputParamsRegbase.isSoftMaxLseEnable;
        if (this->tilingData->initOutputParams.needInit == 1) {
            InitOutputSingleCore(constInfos);
            // lse output
            if (constInfos.isSoftmaxLseEnable) {
                SyncAll();
                InitLseOutputSingleCore(constInfos);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
    __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *queryPaddingSize,
    __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
    ConstInfo<isInfer, hasRope> &constParam)
{
    BaseClass::InitCommonGlobalBuffer(pse, deqScaleQ, deqScaleK, deqScaleV, pScale, postQuantScale, prefix,
                                      attenMask, learnableSink, workspace, constParam);
    if constexpr (isFd) {
        // 让当前的workspace地址回到基地址, workspace偏移了totalOffset + mm2Offset * 3 + ve2offset * 3
        workspace -= singleCoreOffset * preloadTimes * (aicIdx + 1);
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        int32_t actualCoreNums = inputParamsRegbase.bSize * constParam.n2Size * constParam.splitKVNum;
        workspace += actualCoreNums * singleCoreOffset * preloadTimes;  // 针对所有核跳过其前面的所有workspace

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
        this->InitPostQuant(constParam, postQuantScale, postQuantOffset);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitPostQuant(
    ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
{
    if constexpr (POST_QUANT) {
        constInfo.isPostQuantOffsetExist = false;
        if (!constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
                constInfo.postQuantScaleValue = postQuantScaleGm.GetValue(0);
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
                constInfo.postQuantOffsetValue = postQuantOffsetGm.GetValue(0);
            } else {
                constInfo.postQuantOffsetValue = 0.0;
            }
        }
        if (constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                this->postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
            }
        }
        if (!constInfo.isPostQuantPerChnl && constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
                constInfo.postQuantScaleValue = ToFloat(postQuantScaleBf16Gm.GetValue(0));
            }
            if (postQuantOffset != nullptr) {
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
                constInfo.postQuantOffsetValue = ToFloat(postQuantOffsetBf16Gm.GetValue(0));
            } else {
                constInfo.postQuantOffsetValue = 0.0;
            }
        }
        if (constInfo.isPostQuantPerChnl && constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                postQuantScaleBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetBf16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)postQuantOffset);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitUniqueLocalBuffer(
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
        this->tPipe->InitBuffer(softmaxLseQueue, 1,
                                (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    if (constInfo.learnableSinkFlag) {
        this->tPipe->InitBuffer(sinkQue, 1, 256); // buffer size = 256 bytes
    }
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, 2048); // 2K
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, 2048); // 2K
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo)
{
    this->tPipe->Reset();
    this->tPipe->InitBuffer(lseTmpBuff, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxMaxInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxSumInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(FDResOutputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(accumOutInputQue, 1, bufferSizeByte32K);
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND, 每行结果存为8个重复lse元素(32B对齐)
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (BaseClass::s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, bufferSizeByte32K);
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, bufferSizeByte32K);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo,
    GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr)
{
    int64_t bIdx = constInfo.aivIdx / constInfo.n2Size;
    int64_t n2Idx = constInfo.aivIdx % constInfo.n2Size;
    int64_t batchSize = this->tilingData->inputParamsRegbase.bSize;
    if (constInfo.aivIdx >= batchSize * constInfo.n2Size) {
        return;
    }
    int64_t actualSeqLength;
    GetActualSeqLenKV(constInfo, keyGm, actualSeqKvlenAddr, bIdx, actualSeqLength);
    if (actualSeqLength == 0) {
        return;
    }
    uint64_t attenOutOffset = (uint64_t)bIdx * constInfo.n2GDv + n2Idx * constInfo.gDv;
    constInfo.actualCombineLoopSize = (actualSeqLength + constInfo.sInnerLoopSize - 1) /
                                      constInfo.sInnerLoopSize;
    CombineSplitKVRes(constInfo, attenOutOffset, bIdx, n2Idx);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
    LocalTensor<float> &maxUbuffer)
{
    if constexpr (isFd) {
        ComputeLogSumExpAndCopyToGm(runInfo, constInfo);
        return;
    }
    if (constInfo.learnableSinkFlag) {
        if (constInfo.isGqa) {
            this->Vec1SinkComputeGSFused(runInfo, constInfo, sumUb, maxUbuffer);
        } else {
            this->Vec1SinkCompute(runInfo, constInfo, sumUb, maxUbuffer);
        }
    }
    SoftmaxLseCopyOut(sumUb, maxUbuffer, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::Vec1SinkCompute(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
{
    int64_t sinkOffset = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx;
    auto sinkSrc = this->sinkGm.GetValue(sinkOffset);
    float sinkValue;
    if constexpr (IsSameType<decltype(sinkSrc), half>::value) {
        sinkValue = static_cast<float>(sinkSrc);
    } else {
        sinkValue = ToFloat(sinkSrc);
    }
    SinkSubExpAddVF<float>(sumUb, maxUb, sinkValue, runInfo.halfS1RealSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::Vec1SinkComputeGSFused(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
{
    CopySinkIn(runInfo, constInfo);
    LocalTensor<INPUT_T> sinkUb = sinkQue.DeQue<INPUT_T>();
    SinkSubExpAddGSFusedVF<float, INPUT_T>(sinkUb, sumUb, maxUb, runInfo.halfS1RealSize);
    sinkQue.FreeTensor(sinkUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CopySinkIn(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<INPUT_T> sinkUbBf16 = sinkQue.AllocTensor<INPUT_T>();
    int64_t sinkOffset = runInfo.n2oIdx * constInfo.gSize + constInfo.subBlockIdx * runInfo.halfS1RealSize;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1; // 进行一次连续拷贝
    sinkCopyParams.blockLen = runInfo.halfS1RealSize * sizeof(INPUT_T); // 实际需要拷贝的字节数
    sinkCopyParams.dstStride = 0; // 目的地址连续
    sinkCopyParams.srcStride = 0; // 源地址连续
    DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
    DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
    sinkQue.EnQue(sinkUbBf16);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    if constexpr (isFd) {
        Bmm2FDOut(vec2ResUb, runInfo, constInfo, vec2S1Idx, vec2CalcSize);
    } else {
        this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::GetActualSeqLenKV(
    ConstInfo<isInfer, hasRope> &constInfo, GlobalTensor<INPUT_T> &keyGm, __gm__ int64_t *actualSeqKvlenAddr,
    int64_t boIdx, int64_t &actualSeqLength)
{
    int64_t s2CurrentBatch = constInfo.s2Size;
    if (constInfo.isKvContinuous == 0) {
        ListTensorDesc keyListTensorDesc((__gm__ void *)keyGm.GetPhyAddr());
        AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;
        uint64_t dimInfo[4];
        kvTensorDesc.SetShapeAddr(&dimInfo[0]);
        keyListTensorDesc.GetDesc(kvTensorDesc, boIdx);
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            s2CurrentBatch = kvTensorDesc.GetShape(2);
        } else {
            s2CurrentBatch = kvTensorDesc.GetShape(1);
        }
    }
    // 赋值 actualSeqLength
    if (constInfo.isActualLenDimsKVNull) {
        actualSeqLength = s2CurrentBatch;
    } else {
        actualSeqLength = (constInfo.actualSeqLenKVSize == 1) ? actualSeqKvlenAddr[0] :
                                                             actualSeqKvlenAddr[boIdx];
    }
    if (constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualSeqLength - constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLength = 0;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
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
    DataCopyExtParams intriParamsExt;
    intriParamsExt.blockLen = sizeof(float);
    intriParamsExt.blockCount = runInfo.halfS1RealSize;
    intriParamsExt.srcStride = 0;
    if (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
        intriParamsExt.dstStride = constInfo.isGqa ? 0 : sizeof(float) * (constInfo.n2G - 1);
    } else {
        intriParamsExt.dstStride = 0;
    }
    DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], lseUb, intriParamsExt);
    softmaxLseQueue.FreeTensor(lseUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitOutputSingleCore(
    ConstInfo<isInfer, hasRope> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    uint32_t tailSize = (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) > 0 ?
        (initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize) : 0;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    if constexpr (POST_QUANT) {
        InitOutput<half>(this->attentionOutInitGm[constInfo.aivIdx * initParams.singleCoreSize / 2],  // 2: CV
                         singleInitOutputSize / 2, 0.0);  // 2: CV
    } else {
        InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize],
                             singleInitOutputSize, 0.0);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::InitLseOutputSingleCore(
    ConstInfo<isInfer, hasRope> &constInfo)
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParam = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParam.totalSoftMaxLseOutputSize / coreNum;
        if (constInfo.aivIdx == coreNum - 1) {
            singleCoreLseSize += initParam.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParam.totalSoftMaxLseOutputSize / coreNum)],
            singleCoreLseSize, 3e+99); // 3e+99:set the value of invalid batch to inf
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CombineSplitKVRes(
    ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx)
{
    uint32_t gSplitSizeLse = bufferSizeByte32K /
                            (FA_BYTE_BLOCK * constInfo.splitKVNum); // 32K / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = bufferSizeByte32K / sizeof(float) / (uint32_t)dVTemplateType;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitNum = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (constInfo.gSize > gSplitMax) {
        gSplitNum = (gSplitNum > gSplitMax) ? gSplitMax : gSplitNum;
    } else {
        gSplitNum = (gSplitNum > constInfo.gSize) ? constInfo.gSize : gSplitNum;
    }
    uint32_t loopCount = CeilDiv(constInfo.gSize, gSplitNum);
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitNum;
    uint64_t lseOffset = 0;
    uint64_t sinkOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>(); // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitNum, fp32BaseSize};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape

    uint64_t perChannelQuantOffset = n2Idx * constInfo.dSizeV * constInfo.gSize;
    // 非尾块处理
    for (uint32_t i = 0; i < loopCount - 1; i++) {
        uint32_t startRow = i * gSplitNum;
        CopyLseIn(constInfo, bIdx, n2Idx, startRow, gSplitNum);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + i * gSplitNum;
        sinkOffset = n2Idx * constInfo.gSize + i * gSplitNum;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, constInfo, gSplitNum, lseOffset, sinkOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(constInfo, bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitNum);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(constInfo, attenOutOffset, tmp1, startRow, gSplitNum, perChannelQuantOffset);
    }
    // 尾块处理
    if (tailSplitSize > 0) {
        uint32_t startRow = (loopCount - 1) * gSplitNum;
        CopyLseIn(constInfo, bIdx, n2Idx, startRow, tailSplitSize);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();

        lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + (loopCount - 1) * gSplitNum;
        sinkOffset = n2Idx * constInfo.gSize + (loopCount - 1) * gSplitNum;
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, constInfo, tailSplitSize, lseOffset, sinkOffset);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(constInfo, bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, tailSplitSize);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(constInfo, attenOutOffset, tmp1, startRow, tailSplitSize, perChannelQuantOffset);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::ComputeScaleValue(
    LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t splitSize, uint64_t lseOffset, uint64_t sinkOffset)
{
    LocalTensor<T> lseOutputUbuffer;
    if (constInfo.isSoftmaxLseEnable) {
        lseOutputUbuffer = softmaxLseQueue.template AllocTensor<T>();
    }
    LocalTensor<INPUT_T> tmpSinkUb;
    if (constInfo.learnableSinkFlag) {
        CopySinkFDIn(splitSize, sinkOffset);
        tmpSinkUb = sinkQue.DeQue<INPUT_T>();
    }
    ComputeScaleValue_VF(tmpSinkUb, lseMaxUb, lseSumUb, lseOutputUbuffer, splitSize, constInfo.actualCombineLoopSize,
                         constInfo.isSoftmaxLseEnable, constInfo.learnableSinkFlag);
    if (constInfo.isSoftmaxLseEnable) {
        softmaxLseQueue.template EnQue<T>(lseOutputUbuffer);
        softmaxLseQueue.DeQue<T>();
        DataCopyExtParams intriParams1;
        intriParams1.blockLen = sizeof(float);
        intriParams1.blockCount = splitSize;
        intriParams1.srcStride = 0;
        intriParams1.dstStride = 0;
        DataCopyPad(softmaxLseGm[lseOffset], lseOutputUbuffer, intriParams1);
        softmaxLseQueue.FreeTensor(lseOutputUbuffer);
    }
    if (constInfo.learnableSinkFlag) {
        sinkQue.FreeTensor(tmpSinkUb);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CopySinkFDIn(uint32_t splitSize, uint64_t sinkOffset)
{
    LocalTensor<INPUT_T> sinkUbBf16 = sinkQue.AllocTensor<INPUT_T>();
    DataCopyExtParams sinkBf16CopyParams;
    sinkBf16CopyParams.blockCount = 1; // 进行一次连续拷贝
    sinkBf16CopyParams.blockLen = splitSize * sizeof(INPUT_T); // 实际需要拷贝的字节数
    sinkBf16CopyParams.srcStride = 0; // 源地址连续
    sinkBf16CopyParams.dstStride = 0; // 目的地址连续

    DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
    DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkBf16CopyParams, sinkCopyPadParams);
    sinkQue.EnQue(sinkUbBf16);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::Bmm2FDOut(LocalTensor<T> &vec2ResUb,
    RunInfo<isInfer> &runInfo,  ConstInfo<isInfer, hasRope> &constInfo, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<T> attenOut;
    int64_t dvSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD) {
        dvSizeAligned64 = constInfo.dBasicBlock;
    }
    SetFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    WaitFlag<HardEvent::V_MTE3>(this->vToMte3Id[runInfo.taskIdMod2]);
    attenOut = vec2ResUb;

    // 地址偏移计算
    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    uint64_t base = (runInfo.boIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV +
                   runInfo.n2oIdx * constInfo.gSize * constInfo.dSizeV) *
                      constInfo.splitKVNum +
                  mStart * constInfo.dSizeV + vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV;

    // 拷贝参数赋值
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dvSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * constInfo.gSize * constInfo.dSizeV],
                attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CopyLseIn(ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.AllocTensor<T>();
    LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    copyInParams.blockCount = constInfo.splitKVNum;
    copyInParams.blockLen = dealRowCount * fp32BaseSize * sizeof(T);
    copyInParams.srcStride = (constInfo.gSize - dealRowCount) * fp32BaseSize * sizeof(T);
    copyInParams.dstStride = 0;

    DataCopyPadExtParams<T> copyInPadParams;
    copyInPadParams.isPad = false;
    copyInPadParams.paddingValue = 0;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    uint64_t combineLseOffset =
        ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum + n2Idx * constInfo.splitKVNum) *
        constInfo.gSize * fp32BaseSize + startRow * fp32BaseSize;
    DataCopyPad(softmaxMaxLocal, softmaxFDMaxGm[combineLseOffset], copyInParams, copyInPadParams);
    DataCopyPad(softmaxSumLocal, softmaxFDSumGm[combineLseOffset], copyInParams, copyInPadParams);
    softmaxMaxInputQue.EnQue(softmaxMaxLocal);
    softmaxSumInputQue.EnQue(softmaxSumLocal);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CopyFinalResOut(
    ConstInfo<isInfer, hasRope> &constInfo, uint64_t attenOutOffset,
    LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount, uint64_t perChannelQuantOffset)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCasTBuff = FDResOutputQue.AllocTensor<OUTPUT_T>();
    uint32_t dvSizeAligned64 = (uint32_t)dVTemplateType;
    if constexpr (BaseClass::splitD) {
        dvSizeAligned64 = constInfo.dBasicBlock;
    }
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dvSizeAligned64};
    tmpBmm2ResCasTBuff.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape
    if constexpr (!POST_QUANT) {
        Cast(tmpBmm2ResCasTBuff, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dvSizeAligned64);
    } else {
        FDPostQuant(constInfo, tmpBmm2ResCasTBuff, accumOutLocal,
            perChannelQuantOffset + startRow * constInfo.dSizeV, dealRowCount, dvSizeAligned64);
    }

    FDResOutputQue.EnQue(tmpBmm2ResCasTBuff);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(constInfo, attenOutOffset, tmpBmm2ResCasTBuff, startRow, dealRowCount, dvSizeAligned64,
                        constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCasTBuff);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::ReduceFinalRes(ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow, uint32_t dealRowCount)
{
    int64_t dvSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD) {
        dvSizeAligned64 = constInfo.dBasicBlock;
    }
    // mul 结果拷贝
    for (uint32_t j = 0; j < constInfo.actualCombineLoopSize; ++j) {
        CopyAccumOutIn(constInfo, bIdx, n2Idx, j, startRow, dealRowCount);
        LocalTensor<T> accumOut = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_VF<T>(dst, lseLocal, accumOut, dealRowCount, dvSizeAligned64, j);
        accumOutInputQue.FreeTensor(accumOut);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::CopyAccumOutIn(ConstInfo<isInfer, hasRope> &constInfo,
    uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();
    int64_t dvSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (BaseClass::splitD) {
        dvSizeAligned64 = constInfo.dBasicBlock;
    }

    // 地址偏移计算
    uint64_t combineAccumOutOffset = ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum +
                                    n2Idx * constInfo.splitKVNum + splitKVIndex) *
                                        constInfo.gSize * constInfo.dSizeV +
                                    startRow * constInfo.dSizeV;

    DataCopyExtParams copyInParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dvSizeAligned64 - constInfo.dSizeV) / 8; // 8 for align factor

    DataCopyPadExtParams<T> copyPadParams;
    copyPadParams.isPad = true;
    copyPadParams.leftPadding = 0;
    copyPadParams.rightPadding = (dvSizeAligned64 - constInfo.dSizeV) % 8; // 8 for align factor
    copyPadParams.paddingValue = 0;

    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t gmOffset =
        runInfo.boIdx * constInfo.n2Size * constInfo.splitKVNum * constInfo.gSize * fp32BaseSize +
        runInfo.n2oIdx * constInfo.splitKVNum * constInfo.gSize * fp32BaseSize +
        runInfo.flashDecodeS2Idx * constInfo.gSize * fp32BaseSize + mStart * fp32BaseSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    // Copy sum to gm
    this->BroadCastAndCopyOut(runInfo, softmaxFDSumGm, softmaxFDMaxGm, gmOffset, calculateSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::ReduceFDDataCopyOut(ConstInfo<isInfer, hasRope> &constInfo,
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = dealRowCount;
    copyExtParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    copyExtParams.srcStride = (columnCount - actualColumnCount) / (FA_BYTE_BLOCK / sizeof(OUTPUT_T));
    copyExtParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, copyExtParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::PostQuantPerChnl(
    ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb,
    uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = gSplitSize;
    copyParams.blockLen = constInfo.dSizeV * sizeof(POSTQUANT_PARAMS_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = (dSizeAligned64 - constInfo.dSizeV) / (32 / sizeof(POSTQUANT_PARAMS_T));

    LocalTensor<POSTQUANT_PARAMS_T> postQuantScaleUb =
        this->postQuantScaleQue.template AllocTensor<POSTQUANT_PARAMS_T>();
    DataCopyPadExtParams<POSTQUANT_PARAMS_T> copyPadParams;
    DataCopyPad(postQuantScaleUb, postQuantScaleGm[perChannelQuantOffset], copyParams, copyPadParams);
    // 同步
    this->postQuantScaleQue.template EnQue(postQuantScaleUb);
    this->postQuantScaleQue.template DeQue<POSTQUANT_PARAMS_T>();
    if (constInfo.isPostQuantOffsetExist) {
        LocalTensor<POSTQUANT_PARAMS_T> postQuantOffsetUb =
            this->postQuantOffsetQue.template AllocTensor<POSTQUANT_PARAMS_T>();
        DataCopyPad(postQuantOffsetUb, postQuantOffsetGm[perChannelQuantOffset], copyParams, copyPadParams);
        this->postQuantOffsetQue.template EnQue(postQuantOffsetUb);
        this->postQuantOffsetQue.template DeQue<POSTQUANT_PARAMS_T>();
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, postQuantOffsetUb,
            gSplitSize, s1RowCount, constInfo.dSizeV, dSizeAligned64);
        this->postQuantOffsetQue.FreeTensor(postQuantOffsetUb);
    } else {
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, gSplitSize, s1RowCount,
            constInfo.dSizeV, dSizeAligned64);
    }
    this->postQuantScaleQue.FreeTensor(postQuantScaleUb);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::PostQuant(ConstInfo<isInfer, hasRope> &constInfo,
                                                                      RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                                                      LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                      int64_t vec2S1Idx, int64_t dSizeAlign64)
{
    uint32_t s1RowCnt = constInfo.isGqa ? 1U : runInfo.vec2S1RealSize; // s1=1, gS合轴, bn2分核
    uint32_t gRowCnts = constInfo.isGqa ? runInfo.vec2S1RealSize : 1U;  // s1>1, bn1分核
    if (constInfo.isPostQuantPerChnl) {
        uint64_t perChannelQuantGQAOffset = runInfo.n2oIdx * constInfo.gDv + vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV +
                                            runInfo.sOuterOffset * constInfo.dSizeV;
        uint64_t perChannelQuantOffset = constInfo.isGqa ?
                                                perChannelQuantGQAOffset :
                                                runInfo.n2oIdx * constInfo.gDv + runInfo.goIdx * constInfo.dSizeV;
        uint32_t gSplitSize = constInfo.isPostQuantBF16 ? (2048U / ((uint32_t)dSizeAlign64 * sizeof(bfloat16_t))) :
                                                            (2048U / ((uint32_t)dSizeAlign64 * sizeof(float)));
        gSplitSize = gSplitSize > gRowCnts ? gRowCnts : gSplitSize;
        uint32_t loopCount = (gRowCnts + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gRowCnts - (loopCount - 1) * gSplitSize;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if (i + 1 == loopCount) {
                gSplitSize = tailSplitSize;
            }
            uint32_t splitOffset = startRow * dSizeAlign64;
            if (constInfo.isPostQuantBF16) {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                    gSplitSize, s1RowCnt, splitOffset, dSizeAlign64, postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
            } else {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                    gSplitSize, s1RowCnt, splitOffset, dSizeAlign64, postQuantScaleGm, postQuantOffsetGm);
            }
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(attenOut, vec2ResUb, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue,
                                                 runInfo.vec2S1RealSize, constInfo.dSizeV, dSizeAlign64);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecInferMxFullquant<TEMPLATE_ARGS>::FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo,
    LocalTensor<OUTPUT_T> &attenOut,
    LocalTensor<T> &accumOutLocal,
    uint64_t perChannelQuantOffset,
    uint32_t dealRowCount,
    uint32_t dSizeAlign64)
{
    if (constInfo.isPostQuantPerChnl) {
        if (constInfo.isPostQuantBF16) {
            PostQuantPerChnl(constInfo, attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount,
                1U, 0U, dSizeAlign64, postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
        } else {
            PostQuantPerChnl(constInfo, attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount,
                1U, 0U, dSizeAlign64, postQuantScaleGm, postQuantOffsetGm);
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(
            attenOut, accumOutLocal, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue, dealRowCount,
            constInfo.dSizeV, dSizeAlign64);
    }
}
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_VEC_INFER_H_