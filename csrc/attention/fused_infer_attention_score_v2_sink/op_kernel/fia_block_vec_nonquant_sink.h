/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_block_vec_nonquant_sink.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_NONQUANT_SINK_H
#define FIA_BLOCK_VEC_NONQUANT_SINK_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#if __has_include("../common/op_kernel/fia_public_define.h")
#include "fia_public_define.h"
#include "memory_copy.h"
#else
#include "fia_public_define.h"
#include "memory_copy.h"
#endif

using namespace AiInfraInferenceAttentionCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT> class FiaBlockVecNonQuant {
  public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using ORIGIN_T = typename FIAT::orginalType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;
    static constexpr bool SOFTMAX_WITH_BRC = FIAT::softmaxWithBrc;

    using UPDATE_T = T;
    using TMP_T = T;
    using COMPUTE_T = T;
    using SOFTMAX_TYPE = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline FiaBlockVecNonQuant(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo);
    __aicore__ inline void Init(__gm__ uint8_t *query,
                                __gm__ uint8_t *key,
                                __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift,
                                __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *deqScale1,
                                __gm__ uint8_t *quantScale1,
                                __gm__ uint8_t *deqScale2,
                                __gm__ uint8_t *quantScale2,
                                __gm__ uint8_t *quantOffset2,
                                __gm__ uint8_t *antiquantScale,
                                __gm__ uint8_t *antiquantOffset,
                                __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryPaddingSize,
                                __gm__ uint8_t *kvPaddingSize,
                                __gm__ uint8_t *keyAntiquantScale,
                                __gm__ uint8_t *keyAntiquantOffset,
                                __gm__ uint8_t *valueAntiquantScale,
                                __gm__ uint8_t *valueAntiquantOffset,
                                __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *keyRopeAntiquantScale,
                                __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM1_OUT_T> mm1ResGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm);
    __aicore__ inline void
    InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void SetMSplitInfo(uint32_t mDealSize);
    // V1
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info,
                                                uint32_t startRow,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info,
                                          LocalTensor<MM1_OUT_T> &mmResUb,
                                          TBuf<> &tmpBuf,
                                          uint32_t startRow,
                                          uint32_t dealRowCount,
                                          uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info,
                                                 LocalTensor<MM1_OUT_T> &mmResUb,
                                                 LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow,
                                                 uint32_t dealRowCount,
                                                 uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                       const MSplitInfo &mSplitInfo,
                                                       LocalTensor<COMPUTE_T> &softmaxSumUb,
                                                       LocalTensor<COMPUTE_T> &softmaxMaxUb);
    // V2
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info,
                                                uint32_t startRow,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info,
                                          LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                          uint32_t wsMStart,
                                          uint32_t startRow,
                                          uint32_t dealRowCount,
                                          uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info,
                                             LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                             uint32_t wsMStart,
                                             uint32_t startRow,
                                             uint32_t dealRowCount,
                                             uint32_t columnCount,
                                             uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info,
                                              LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                              uint32_t wsMStart,
                                              uint32_t startRow,
                                              uint32_t dealRowCount,
                                              uint32_t columnCount,
                                              uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info,
                                                LocalTensor<OUT_T> &attenOutUb,
                                                uint32_t wsMStart,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidMaskRows(const RunInfo &info,
                                               LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                               uint32_t wsMStart,
                                               uint32_t startRow,
                                               uint32_t dealRowCount,
                                               uint32_t columnCount,
                                               uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidRows(const RunInfo &info,
                                           LocalTensor<MM2_OUT_T> &attenOutUb,
                                           uint32_t wsMStart,
                                           uint32_t dealRowCount,
                                           uint32_t columnCount,
                                           uint32_t actualColumnCount);
    __aicore__ inline void Vec1SinkCompute(const RunInfo &info,
                                           LocalTensor<COMPUTE_T> &tmpSinkResUb,
                                           LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                           uint32_t idx,
                                           uint32_t wsMStart,
                                           uint32_t dealRowCount);
    __aicore__ inline void Vec1SinkSoftmaxProc(const RunInfo &info,
                                               LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                               uint32_t offset,
                                               uint32_t dealRowCountBrcb);
    __aicore__ inline void Vec1GetSinkValue(const RunInfo &info,
                                            LocalTensor<COMPUTE_T> &tmpSinkResUb,
                                            uint32_t wsMStart,
                                            uint32_t dealRowCount);

  protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ; // 需确认后续是否会用到
    GlobalTensor<uint64_t> actualSeqLengthsGm; // 需确认后续是否会用到
    GlobalTensor<bfloat16_t> sinkGm;

    __gm__ uint8_t *actualSequenceLengthsQ = nullptr;

    // =================================常量区=================================
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    T SOFTMAX_MIN_NUM = T(-1.0 / 0.0); // -inf
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_2K;
    static constexpr uint32_t LSE_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_8K;
    static constexpr uint32_t DATA_BLOCK_NUM = 8;

    ConstInfo constInfo = {};
    MSplitInfo mSplitInfo = {};

    static constexpr uint16_t brcbNum = (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(COMPUTE_T));
    bool learnableSinkFlag = false;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // ================================Local Buffer区====================================
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TQue<QuePosition::VECIN, 1> inputQue2;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
    TQue<QuePosition::VECOUT, 1> outputQue2;

    // 临时tbuf
    TBuf<> tmpBuff1;
    TBuf<> softmaxMaxBuff;
    TBuf<> softmaxExpBuff;
    TBuf<> softmaxSumBuff;
    TBuf<> softmaxMaxDefaultBuff;
    TBuf<> softmaxSumDefaultBuff;

    // ================================LocalTensor区====================================
    // 常驻
    LocalTensor<COMPUTE_T> softmaxMaxDefaultUb;
    LocalTensor<COMPUTE_T> softmaxSumDefaultUb;
    LocalTensor<COMPUTE_T> softmaxMaxUb;
    LocalTensor<COMPUTE_T> softmaxSumUb;
    LocalTensor<COMPUTE_T> softmaxExpUb;
};

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitParams(const struct ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Init(__gm__ uint8_t *query,
                                                       __gm__ uint8_t *key,
                                                       __gm__ uint8_t *value,
                                                       __gm__ uint8_t *pseShift,
                                                       __gm__ uint8_t *attenMask,
                                                       __gm__ uint8_t *actualSeqLengthsQ,
                                                       __gm__ uint8_t *actualSeqLengths,
                                                       __gm__ uint8_t *deqScale1,
                                                       __gm__ uint8_t *quantScale1,
                                                       __gm__ uint8_t *deqScale2,
                                                       __gm__ uint8_t *quantScale2,
                                                       __gm__ uint8_t *quantOffset2,
                                                       __gm__ uint8_t *antiquantScale,
                                                       __gm__ uint8_t *antiquantOffset,
                                                       __gm__ uint8_t *blockTable,
                                                       __gm__ uint8_t *queryPaddingSize,
                                                       __gm__ uint8_t *kvPaddingSize,
                                                       __gm__ uint8_t *keyAntiquantScale,
                                                       __gm__ uint8_t *keyAntiquantOffset,
                                                       __gm__ uint8_t *valueAntiquantScale,
                                                       __gm__ uint8_t *valueAntiquantOffset,
                                                       __gm__ uint8_t *queryRope,
                                                       __gm__ uint8_t *keyRope,
                                                       __gm__ uint8_t *keyRopeAntiquantScale,
                                                       __gm__ uint8_t *learnableSink,
                                                       __gm__ uint8_t *attentionOut,
                                                       __gm__ uint8_t *softmaxLse)
{
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    if (constInfo.softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
    attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
    this->actualSequenceLengthsQ = actualSeqLengthsQ;
    if (learnableSink != nullptr) {
        learnableSinkFlag = true;
        sinkGm.SetGlobalBuffer((__gm__ bfloat16_t *)learnableSink);
    }
    qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
    kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitVec1GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                                                       GlobalTensor<MM1_OUT_T> mm1ResGm)
{
    this->vec1ResGm = vec1ResGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm,
                                                                       GlobalTensor<MM2_OUT_T> mm2ResGm)
{
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm,
                                                                              GlobalTensor<T> lseMaxFdGm,
                                                                              GlobalTensor<T> lseSumFdGm)
{
    this->accumOutGm = accumOutGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitBuffers(TPipe *pipe)
{
    // in queue
    pipe->InitBuffer(inputQue1, 2, ConstInfo::BUFFER_SIZE_BYTE_32K); // 2:pingpong
    pipe->InitBuffer(inputQue2, 2, ConstInfo::BUFFER_SIZE_BYTE_16K); // 2:pingpong

    // out queue
    pipe->InitBuffer(outputQue1, 1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputQue2, 1, ConstInfo::BUFFER_SIZE_BYTE_8K);

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);

    // 1. [M,8]场景: 2K/32B = 64, 即单个VEC上可以缓存64行, 整个AICORE上有2个VEC，所以此时分核的MBaseSize<=128
    // 2. [M,1]场景: 2K/sizeof(float) = 512, 即单个VEC上可以缓存512行, 所以此时分核的MBaseSize<=512*2=1024
    pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, SOFTMAX_TMP_BUFFER_SIZE * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, SOFTMAX_TMP_BUFFER_SIZE);
    pipe->InitBuffer(softmaxSumDefaultBuff, SOFTMAX_TMP_BUFFER_SIZE);

    softmaxMaxUb = softmaxMaxBuff.Get<COMPUTE_T>();
    softmaxSumUb = softmaxSumBuff.Get<COMPUTE_T>();
    softmaxExpUb = softmaxExpBuff.Get<COMPUTE_T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<COMPUTE_T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<COMPUTE_T>();

    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T));
    Duplicate(softmaxSumDefaultUb, (COMPUTE_T)0.0, SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T));
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::AllocEventID() {}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::FreeEventID() {}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::SetMSplitInfo(uint32_t mDealSize)
{
    mSplitInfo.nBufferIdx = 0U;
    mSplitInfo.nBufferStartM = 0U;
    mSplitInfo.nBufferDealM = mDealSize;
    // VEC0处理的M大小
    if (mSplitInfo.nBufferDealM <= 16) {
        mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
    } else {
        mSplitInfo.vecDealM = ((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16;
    }
    mSplitInfo.vecStartM = 0;
    if (GetBlockIdx() % 2 == 1) {
        // VEC1处理的M大小
        mSplitInfo.vecStartM = mSplitInfo.vecDealM;
        mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::ProcessVec1SingleBuf(const RunInfo &info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(COMPUTE_T);
        // 向下8/16对齐是因为UB操作起始地址需32B对齐
        mSplitSize = mSplitSize / alignVal * alignVal;
    }
    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info,
                             i * mSplitSize,
                             dealSize,
                             info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
    }

    if (info.isLastS2Loop) {
        uint32_t outIdx = info.loop % (constInfo.preLoadNum);
        if (unlikely(learnableSinkFlag)) {
            LocalTensor<COMPUTE_T> tmpSinkResUb = tmpBuff1.GetWithOffset<COMPUTE_T>(SOFTMAX_TMP_BUFFER_SIZE, 0);
            LocalTensor<COMPUTE_T> tmpSinkResUbBrcb =
                tmpBuff1.GetWithOffset<COMPUTE_T>(SOFTMAX_TMP_BUFFER_SIZE, SOFTMAX_TMP_BUFFER_SIZE);
            Vec1SinkCompute(info,
                            tmpSinkResUb,
                            tmpSinkResUbBrcb,
                            outIdx,
                            mSplitInfo.nBufferStartM + mSplitInfo.vecStartM,
                            mSplitInfo.vecDealM);
        }

        auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)];
        auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T)];

        if (info.tndIsS2SplitCore) {
            if constexpr (FLASH_DECODE) {
                ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
        } else if (constInfo.softmaxLseFlag) {
            LocalTensor<COMPUTE_T> totalLseUb = tmpBuff1.Get<COMPUTE_T>(LSE_TMP_BUFFER_SIZE);
            if constexpr (!SOFTMAX_WITH_BRC) {
                LocalTensor<COMPUTE_T> lseSumUb =
                    tmpBuff1.GetWithOffset<COMPUTE_T>(LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE);
                LocalTensor<COMPUTE_T> lseMaxUb =
                    tmpBuff1.GetWithOffset<COMPUTE_T>(LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE * 2);
                Brcb(lseSumUb,
                     sumTensor[mSplitInfo.nBufferStartM / 2],
                     (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum,
                     {1, brcbNum});
                AscendC::PipeBarrier<PIPE_V>();
                Brcb(lseMaxUb,
                     maxTensor[mSplitInfo.nBufferStartM / 2],
                     (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum,
                     {1, brcbNum});
                AscendC::PipeBarrier<PIPE_V>();
                AiInfraInferenceCommonFaBaseVector::ComputeSoftMaxLse(
                    totalLseUb, lseSumUb, lseMaxUb, mSplitInfo.vecDealM);
            } else {
                AiInfraInferenceCommonFaBaseVector::ComputeSoftMaxLse(
                    totalLseUb, sumTensor, maxTensor, mSplitInfo.vecDealM);
            }

            bool isInvalidRows = AiInfraInferenceCommonFaBaseVector::IsExistInvalidRows(info.nextTokensPerBatch,
                                                                    info.preTokensPerBatch,
                                                                    constInfo.sparseMode,
                                                                    constInfo.attenMaskFlag,
                                                                    constInfo.isRowInvalid);

            if (isInvalidRows) { // 存在行无效场景
                SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(mSplitInfo.vecDealM),
                                                  static_cast<uint32_t>(brcbNum),
                                                  static_cast<uint32_t>(mSplitInfo.vecDealM),
                                                  static_cast<uint32_t>(brcbNum)};
                if constexpr (SOFTMAX_WITH_BRC) {
                    AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T>(
                        totalLseUb, maxTensor, negativeIntScalar, (COMPUTE_T)3e+99, softmaxShapeInfo);
                } else {
                    AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T, false, 1>(
                        totalLseUb, maxTensor, negativeIntScalar, (COMPUTE_T)3e+99, softmaxShapeInfo);
                }
            }

            LocalTensor<T> tmpLseResCastTensor = outputQue2.AllocTensor<T>();
            DataCopy(tmpLseResCastTensor, totalLseUb, mSplitInfo.vecDealM * brcbNum);
            outputQue2.EnQue(tmpLseResCastTensor);
            outputQue2.DeQue<T>();
            uint32_t mOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
            if (LAYOUT_T == FIA_LAYOUT::TND) {
                uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
                uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + info.n2Idx * constInfo.gSize;
                DataCopySoftmaxLseTND(
                    softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo);
            } else if (LAYOUT_T == FIA_LAYOUT::NTD) {
                uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
                uint32_t s1Size = info.bIdx == 0U ? actualSeqLengthsGmQ.GetValue(0U)
                                                  : actualSeqLengthsGmQ.GetValue(info.bIdx) -
                                                        actualSeqLengthsGmQ.GetValue(info.bIdx - 1U);
                uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + info.n2Idx * constInfo.gSize;
                DataCopySoftmaxLseNTD(
                    softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo, s1Size);
            } else if (LAYOUT_T == FIA_LAYOUT::BSND || LAYOUT_T == FIA_LAYOUT::BSH) {
                uint64_t bN2Offset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                                     info.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                DataCopySoftmaxLseBSND(softmaxLseGm,
                                       tmpLseResCastTensor,
                                       bN2Offset,
                                       mOffset,
                                       mSplitInfo.vecDealM,
                                       constInfo,
                                       qActSeqLensParser,
                                       info.bIdx);
            } else { // BNSD
                uint64_t bN2Offset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                                     info.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                DataCopySoftmaxLseBNSD<T, Q_MODE>(softmaxLseGm,
                                                  tmpLseResCastTensor,
                                                  bN2Offset,
                                                  mOffset,
                                                  mSplitInfo.vecDealM,
                                                  constInfo,
                                                  qActSeqLensParser,
                                                  info.bIdx);
            }
            outputQue2.FreeTensor(tmpLseResCastTensor);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;

    if (likely(constInfo.isSkipSmallBlock)) {
        bool isSkip = !constInfo.batchInvariant &&
                      FusedInferAttentionScoreV2SinkBaseKernel::IsSkipCal<LAYOUT_T>(
                        info.gS1Idx + mSplitInfo.nBufferStartM +
                        mSplitInfo.vecStartM + startRow,
                        dealRowCount,
                        info.s2Idx * constInfo.s2BaseSize,
                        info.actualSingleProcessSInnerSize,
                        info,
                        constInfo);
        // 由于sinkTensor存在，所以首块不可能跳过，所以没有处理首块。后续放开需要考虑首块处理
        // 此处也没有考虑FD，放开FD时需额外考虑
        if (isSkip) {
            uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
            uint32_t actDealRowCount = Align(dealRowCount, 8U);
            if constexpr (SOFTMAX_WITH_BRC) {
                baseOffset = baseOffset * this->brcbNum;
                actDealRowCount = actDealRowCount * this->brcbNum;
            }
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset;

            uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
            uint32_t softmaxInOffset = inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset;

            DataCopy(softmaxSumUb[softmaxOutOffset], softmaxSumUb[softmaxInOffset], actDealRowCount);
            DataCopy(softmaxMaxUb[softmaxOutOffset], softmaxMaxUb[softmaxInOffset], actDealRowCount);

            Duplicate(softmaxExpUb[softmaxOutOffset], static_cast<COMPUTE_T>(1), actDealRowCount);

            LocalTensor<KV_T> vec1ResUbTmp = outputQue1.AllocTensor<KV_T>();
            AscendC::Duplicate(vec1ResUbTmp, static_cast<KV_T>(0), computeSize);
            outputQue1.EnQue(vec1ResUbTmp);
            outputQue1.DeQue<KV_T>();
            DataCopy(vec1ResGm[inOutGmOffset], vec1ResUbTmp, computeSize);
            outputQue1.FreeTensor(vec1ResUbTmp);
            return;
        }
    }

    LocalTensor<MM1_OUT_T> mmResUb = inputQue1.AllocTensor<MM1_OUT_T>();
    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    inputQue1.EnQue(mmResUb);
    inputQue1.DeQue<MM1_OUT_T>();

    ElewiseCompute(info, mmResUb, tmpBuff1, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpBuff1.Get<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> vec1ResUb = outputQue1.AllocTensor<KV_T>();
    Cast(vec1ResUb, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    outputQue1.EnQue(vec1ResUb);
    outputQue1.DeQue<KV_T>();
    DataCopy(vec1ResGm[inOutGmOffset], vec1ResUb, computeSize);
    outputQue1.FreeTensor(vec1ResUb);

    inputQue1.FreeTensor(mmResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::ElewiseCompute(const RunInfo &info,
                                                                 LocalTensor<MM1_OUT_T> &mmResUb,
                                                                 TBuf<> &tmpBuf,
                                                                 uint32_t startRow,
                                                                 uint32_t dealRowCount,
                                                                 uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<MM1_OUT_T>(constInfo.scaleValue), dealRowCount * columnCount);
    if (unlikely(info.isSinkTensor)) {
        return;
    }

    if (constInfo.attenMaskFlag == 1) {
        AscendC::PipeBarrier<PIPE_V>();
        AiInfraInferenceCommonFaBaseVector::MaskInfo maskInfo;
        maskInfo.gs1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
        maskInfo.gs1dealNum = dealRowCount;
        maskInfo.s1Size = info.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = info.s2Idx * constInfo.s2BaseSize;
        maskInfo.s2dealNum = info.actualSingleProcessSInnerSize;
        maskInfo.s2Size = info.actS2Size;
        maskInfo.preToken = constInfo.preToken;
        maskInfo.nextToken = constInfo.nextToken;
        maskInfo.sparseMode = static_cast<AiInfraInferenceCommonFaBaseVector::SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = info.bIdx;
        maskInfo.batchOffset = constInfo.attenMaskSize;
        maskInfo.attenMaskStride = constInfo.attenMaskStride;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.sinkNumber = constInfo.sinkNumber;
        maskInfo.isEnableSinkNumber = maskInfo.sinkNumber > 0;

        if (constInfo.qSeqSize == 1) {
            maskInfo.layout = AiInfraInferenceCommonFaBaseVector::S1_EQUAL1;
        } else if (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::BSH) {
            maskInfo.layout = AiInfraInferenceCommonFaBaseVector::SG;
        } else {
            maskInfo.layout = AiInfraInferenceCommonFaBaseVector::GS;
        }

        maskInfo.attenMaskType = AiInfraInferenceCommonFaBaseVector::MASK_BOOL; // compatible with int8/uint8
        LocalTensor<bool> maskUb;
        LocalTensor<bool> attenMaskTmpUb;
        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();

        if (!AiInfraInferenceCommonFaBaseVector::IsSkipAttentionmask(maskInfo)) {
            maskUb = inputQue2.AllocTensor<bool>();
            attenMaskTmpUb = maskUb[ConstInfo::BUFFER_SIZE_BYTE_16K / 2];
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCopyIn(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo);
            AscendC::PipeBarrier<PIPE_V>();
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCompute<MM1_OUT_T>(
                mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo);
            inputQue2.FreeTensor(maskUb);
        }
        if (!AiInfraInferenceCommonFaBaseVector::IsSkipAttentionmaskForPre(maskInfo)) {
            maskUb = inputQue2.AllocTensor<bool>();
            attenMaskTmpUb = maskUb[ConstInfo::BUFFER_SIZE_BYTE_16K / 2];
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCopyIn(
                maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo, true);
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCompute<MM1_OUT_T>(
                mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo, true);
            inputQue2.FreeTensor(maskUb);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::SoftmaxFlashV2Compute(const RunInfo &info,
                                                                        LocalTensor<MM1_OUT_T> &mmResUb,
                                                                        LocalTensor<uint8_t> &softmaxTmpUb,
                                                                        uint32_t startRow,
                                                                        uint32_t dealRowCount,
                                                                        uint32_t columnCount,
                                                                        uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(COMPUTE_T), sizeof(COMPUTE_T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<COMPUTE_T> inSumTensor;
    LocalTensor<COMPUTE_T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * this->brcbNum;
    }
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
    }
    if constexpr (SOFTMAX_WITH_BRC) {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG>(mmResUb,
                                                                                        softmaxSumUb[softmaxOutOffset],
                                                                                        softmaxMaxUb[softmaxOutOffset],
                                                                                        mmResUb,
                                                                                        softmaxExpUb[softmaxOutOffset],
                                                                                        inSumTensor,
                                                                                        inMaxTensor,
                                                                                        softmaxTmpUb,
                                                                                        newTiling,
                                                                                        srcShape);
    } else {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb,
            softmaxSumUb[softmaxOutOffset],
            softmaxMaxUb[softmaxOutOffset],
            mmResUb,
            softmaxExpUb[softmaxOutOffset],
            inSumTensor,
            inMaxTensor,
            softmaxTmpUb,
            newTiling,
            srcShape);
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::ProcessVec2SingleBuf(const RunInfo &info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(COMPUTE_T);
        // 向下8/16对齐是因为UB操作起始地址需32B对齐
        mSplitSize = mSplitSize / alignVal * alignVal;
    }
    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, i * mSplitSize, dealSize, constInfo.headDimAlign, constInfo.headDim);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealBmm2ResBaseBlock(
    const RunInfo &info, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * this->brcbNum;
    }
    uint64_t inOutBaseOffset = mStart * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> bmm2ResUb = inputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(bmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    inputQue1.EnQue(bmm2ResUb);
    inputQue1.DeQue<MM2_OUT_T>();

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<COMPUTE_T> bmm2ResPreUb = inputQue1.AllocTensor<COMPUTE_T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
        inputQue1.EnQue(bmm2ResPreUb);

        inputQue1.DeQue<COMPUTE_T>();
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % (constInfo.preLoadNum);

        if constexpr (SOFTMAX_WITH_BRC) {
            AiInfraInferenceCommonFaBaseVector::RowMuls<COMPUTE_T>(
                bmm2ResPreUb,
                bmm2ResPreUb,
                softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                dealRowCount,
                columnCount,
                actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpExpBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpExpBrcbResUb,
                 softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                 (dealRowCount + this->brcbNum - 1) / this->brcbNum,
                 {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            AiInfraInferenceCommonFaBaseVector::RowMuls<COMPUTE_T>(
                bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue1.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.isLastS2Loop) {
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % (constInfo.preLoadNum);

        if constexpr (SOFTMAX_WITH_BRC) {
            AiInfraInferenceCommonFaBaseVector::RowDivs<COMPUTE_T>(
                bmm2ResUb,
                bmm2ResUb,
                softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                dealRowCount,
                columnCount,
                actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpSumBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpSumBrcbResUb,
                 softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                 (dealRowCount + this->brcbNum - 1) / this->brcbNum,
                 {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            AiInfraInferenceCommonFaBaseVector::RowDivs<COMPUTE_T>(
                bmm2ResUb, bmm2ResUb, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<COMPUTE_T> tmpBmm2Res = outputQue1.AllocTensor<COMPUTE_T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<COMPUTE_T>();
        uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }

    inputQue1.FreeTensor(bmm2ResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2ResCopyOut(const RunInfo &info,
                                                                 LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                                                 uint32_t wsMStart,
                                                                 uint32_t startRow,
                                                                 uint32_t dealRowCount,
                                                                 uint32_t columnCount,
                                                                 uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2FDDataCopyOut(const RunInfo &info,
                                                                    LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                                                    uint32_t wsMStart,
                                                                    uint32_t startRow,
                                                                    uint32_t dealRowCount,
                                                                    uint32_t columnCount,
                                                                    uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    DealInvalidMaskRows(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<MM2_OUT_T> tmp = outputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    outputQue1.EnQue(tmp);
    outputQue1.DeQue<T>();
    uint64_t offset = info.accumTmpOutNum * constInfo.mBaseSize * constInfo.headDim + // taskoffset
                      info.tndCoreStartKVSplitPos * constInfo.mBaseSize * constInfo.headDim + // 份数offset
                      wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount)
        / (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(dst, tmp, dataCopyParams);
    outputQue1.FreeTensor(tmp);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealInvalidMaskRows(const RunInfo &info,
                                                                      LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                                                      uint32_t wsMStart,
                                                                      uint32_t startRow,
                                                                      uint32_t dealRowCount,
                                                                      uint32_t columnCount,
                                                                      uint32_t actualColumnCount)
{
    if (!constInfo.isRowInvalid || !constInfo.attenMaskFlag) {
        return;
    }
    if (constInfo.sparseMode != AiInfraInferenceCommonFaBaseVector::DEFAULT_MASK &&
        constInfo.sparseMode != AiInfraInferenceCommonFaBaseVector::ALL_MASK) {
        return;
    }
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T));
    }

    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(T) + baseOffset;

    AiInfraInferenceCommonFaBaseVector::InvalidMaskRows<MM2_OUT_T, T, SOFTMAX_WITH_BRC>(
        softmaxOutOffset, dealRowCount, columnCount, softmaxMaxUb, negativeIntScalar, bmm2ResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2CastAndCopyOut(const RunInfo &info,
                                                                     LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                                                     uint32_t wsMStart,
                                                                     uint32_t startRow,
                                                                     uint32_t dealRowCount,
                                                                     uint32_t columnCount,
                                                                     uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    DealInvalidMaskRows(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }
    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2DataCopyOutTrans(const RunInfo &info,
                                                                       LocalTensor<OUT_T> &attenOutUb,
                                                                       uint32_t wsMStart,
                                                                       uint32_t dealRowCount,
                                                                       uint32_t columnCount,
                                                                       uint32_t actualColumnCount)
{
    FaUbTensor<OUT_T> ubTensor{
        .tensor = attenOutUb,
        .rowCount = dealRowCount,
        .colCount = columnCount,
    };
    GmCoord gmCoord{.bIdx = info.bIdx,
                    .n2Idx = info.n2Idx,
                    .gS1Idx = info.gS1Idx + wsMStart,
                    .dIdx = 0,
                    .gS1DealSize = dealRowCount,
                    .dDealSize = (uint32_t)constInfo.headDim};

    if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize,
                                          constInfo.kvHeadNum,
                                          constInfo.gSize,
                                          constInfo.qSeqSize,
                                          constInfo.headDim,
                                          actualSeqLengthsGmQ,
                                          constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize,
                                          constInfo.kvHeadNum,
                                          constInfo.gSize,
                                          constInfo.qSeqSize,
                                          constInfo.headDim,
                                          actualSeqLengthsGmQ,
                                          constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize,
                                          constInfo.kvHeadNum,
                                          constInfo.gSize,
                                          constInfo.qSeqSize,
                                          constInfo.headDim,
                                          actualSeqLengthsGmQ,
                                          constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
        constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(
            constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum,
                                          constInfo.gSize,
                                          constInfo.headDim,
                                          actualSeqLengthsGmQ,
                                          constInfo.actualLenQDims,
                                          constInfo.qTSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                              const MSplitInfo &mSplitInfo,
                                                                              LocalTensor<COMPUTE_T> &softmaxSumUb,
                                                                              LocalTensor<COMPUTE_T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    //  src-Shape  { gsizeV, S1, AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM }
    //  dst-Shape  { B  N2, splitKV s1, G, AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM}
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * brcbNum;
    uint64_t offset = (info.accumTmpOutNum * constInfo.mBaseSize + // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                      AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if constexpr (SOFTMAX_WITH_BRC) {
        DataCopy(lseSumFdGm[offset], softmaxSumUb[baseOffset], size);
        DataCopy(lseMaxFdGm[offset], softmaxMaxUb[baseOffset], size);
    } else {
        LocalTensor<T> tmp = outputQue2.AllocTensor<T>();
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseSumFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
        tmp = outputQue2.AllocTensor<T>();
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseMaxFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealInvalidRows(const RunInfo &info,
                                                                  LocalTensor<MM2_OUT_T> &attenOutUb,
                                                                  uint32_t wsMStart,
                                                                  uint32_t dealRowCount,
                                                                  uint32_t columnCount,
                                                                  uint32_t actualColumnCount)
{
    if (!constInfo.attenMaskFlag) {
        return;
    }

    if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::ALL_MASK ||
        constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::LEFT_UP_CAUSAL) {
        return;
    }

    AiInfraInferenceCommonFaBaseVector::InvalidRowParams params{
        .actS1Size = info.actS1Size,
        .gSize = constInfo.gSize,
        .gS1Idx = info.gS1Idx + wsMStart,
        .dealRowCount = dealRowCount,
        .columnCount = columnCount,
        .preTokensPerBatch = info.preTokensPerBatch,
        .nextTokensPerBatch = info.nextTokensPerBatch,
    };

    AiInfraInferenceCommonFaBaseVector::InvalidRows<
        T,
        AiInfraInferenceCommonFaBaseVector::GeInputUbFormat<LAYOUT_T>()
    > invalidRows;
    invalidRows(attenOutUb, params);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Vec1GetSinkValue(const RunInfo &info,
                                                                   LocalTensor<COMPUTE_T> &tmpSinkResUb,
                                                                   uint32_t wsMStart,
                                                                   uint32_t dealRowCount)
{
    constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    int64_t gIdx = 0;
    int64_t n1Offset = 0;
    int64_t s1Idx = 0;
    int64_t s1BottomTok = info.actS1Size + info.preTokensPerBatch;
    int64_t s1Tok = -info.nextTokensPerBatch;
    const COMPUTE_T minValue = *((COMPUTE_T *)&negativeIntScalar);

    for (int64_t loop = 0; loop < dealRowCount; ++loop) {
        if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { // 内存按照S1G排布
            gIdx = (info.gS1Idx + wsMStart + loop) % constInfo.gSize;
            n1Offset = info.n2Idx * constInfo.gSize + gIdx;
            s1Idx = (info.gS1Idx + wsMStart + loop) / constInfo.gSize;
        } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { // 内存按照GS1排布
            gIdx = (info.gS1Idx + wsMStart + loop) / info.actS1Size;
            n1Offset = info.n2Idx * constInfo.gSize + gIdx;
            s1Idx = (info.gS1Idx + wsMStart + loop) % info.actS1Size;
        }
        tmpSinkResUb.SetValue(loop, ToFloat(this->sinkGm.GetValue(n1Offset)));

        if (unlikely(info.preTokensPerBatch < 0)) { // 下方存在行无效
            if (s1Idx >= s1BottomTok && s1Idx < info.actS1Size) {
                tmpSinkResUb.SetValue(loop, minValue);
            }
        }
        if (unlikely(info.nextTokensPerBatch < 0)) { // 上方存在行无效
            if (s1Idx < s1Tok) {
                tmpSinkResUb.SetValue(loop, minValue);
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Vec1SinkSoftmaxProc(const RunInfo &info,
                                                                      LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                                                      uint32_t offset,
                                                                      uint32_t dealRowCountBrcb)
{
    AscendC::PipeBarrier<PIPE_V>();
    Sub(tmpSinkResUbBrcb, tmpSinkResUbBrcb, softmaxMaxUb[offset], dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tmpSinkResUbBrcb, tmpSinkResUbBrcb, dealRowCountBrcb);
    if (info.tndIsS2SplitCore) {
        SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(dealRowCountBrcb),
                                          static_cast<uint32_t>(brcbNum),
                                          static_cast<uint32_t>(dealRowCountBrcb),
                                          static_cast<uint32_t>(brcbNum)};
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (SOFTMAX_WITH_BRC) {
            AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T>(
                tmpSinkResUbBrcb, softmaxMaxUb[offset], negativeIntScalar, (COMPUTE_T)0.0, softmaxShapeInfo);
        } else {
            AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T, false, 1>(
                tmpSinkResUbBrcb, softmaxMaxUb[offset], negativeIntScalar, (COMPUTE_T)0.0, softmaxShapeInfo);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxSumUb[offset], softmaxSumUb[offset], tmpSinkResUbBrcb, dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Vec1SinkCompute(const RunInfo &info,
                                                                  LocalTensor<COMPUTE_T> &tmpSinkResUb,
                                                                  LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                                                  uint32_t idx,
                                                                  uint32_t wsMStart,
                                                                  uint32_t dealRowCount)
{
    bool sinkProcFlag = true;
    bool fdS2RealLastLoopFlag =
        info.actS2Size == (info.s2Idx * constInfo.s2BaseSize + info.actualSingleProcessSInnerSize);
    bool isInvalidRows = AiInfraInferenceCommonFaBaseVector::IsExistInvalidRows(info.nextTokensPerBatch,
                                                            info.preTokensPerBatch,
                                                            constInfo.sparseMode,
                                                            constInfo.attenMaskFlag,
                                                            constInfo.isRowInvalid);
    bool fdInvalidRowS2TailFlag = (isInvalidRows && constInfo.tailS2Split);

    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            if (fdS2RealLastLoopFlag || fdInvalidRowS2TailFlag) {
                sinkProcFlag = true;
            } else {
                sinkProcFlag = false;
            }
        }
    }
    if (unlikely(sinkProcFlag == false)) {
        return;
    }

    Vec1GetSinkValue(info, tmpSinkResUb, wsMStart, dealRowCount);

    uint32_t offset = idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + mSplitInfo.nBufferStartM / 2;
    if constexpr (SOFTMAX_WITH_BRC) {
        Brcb(tmpSinkResUbBrcb, tmpSinkResUb, (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
        uint32_t dealRowCountBrcb = dealRowCount * this->brcbNum;
        Vec1SinkSoftmaxProc(info, tmpSinkResUbBrcb, offset, dealRowCountBrcb);
    } else {
        Vec1SinkSoftmaxProc(info, tmpSinkResUb, offset, dealRowCount);
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::ComputeVec1(const RunInfo &info)
{
    SetMSplitInfo(info.actMBaseSize);
    CrossCoreWaitFlag(constInfo.syncC1V1);
    ProcessVec1SingleBuf(info);
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::ComputeVec2(const RunInfo &info)
{
    SetMSplitInfo(info.actMBaseSize);
    CrossCoreWaitFlag(constInfo.syncC2V2);
    ProcessVec2SingleBuf(info);
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV2C2);
}

#endif