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
 * \file fia_block_vec_nonquant_mla_sink.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_NONQUANT_MLA_SINK_H
#define FIA_BLOCK_VEC_NONQUANT_MLA_SINK_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#if __has_include("../common/op_kernel/fia_public_define.h")
#include "fia_public_define.h"
#include "vector_common.h"
#include "memory_copy.h"
#else
#include "fia_public_define.h"
#include "vector_common.h"
#include "memory_copy.h"
#endif

using namespace AiInfraInferenceAttentionCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT> class FiaBlockVecNonQuantMla {
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
    static constexpr float FLOAT_ZERO = 0;

    static constexpr bool SOFTMAX_WITH_BRC = FIAT::softmaxWithBrc;
    using UPDATE_T = T;
    using TMP_T = T;
    using COMPUTE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline FiaBlockVecNonQuantMla(){};
    __aicore__ inline void ProcessVec1L(const AiInfraInferenceAttentionCommon::RunInfo &info);
    __aicore__ inline void ProcessVec2L(const AiInfraInferenceAttentionCommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
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
                                __gm__ uint8_t *metadata,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *softmaxMax,
                                __gm__ uint8_t *softmaxSum,
                                const FusedInferAttentionScoreV2SinkTilingData *__restrict tilingData);
    __aicore__ inline void InitParams(const struct AiInfraInferenceAttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm);
    __aicore__ inline void
    InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                const MSplitInfo &mSplitInfo);
    __aicore__ inline void CopySoftmaxLseMaxSumToGmByLayout(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                          GlobalTensor<float> &gmDst,
                                                          LocalTensor<T> &srcUb,
                                                          uint32_t mOffset,
                                                          const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                const MSplitInfo &mSplitInfo,
                                                uint32_t startRow,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                 const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb,
                                                 LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow,
                                                 uint32_t dealRowCount,
                                                 uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void AmlaVecCompute(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                          const MSplitInfo &mSplitInfo,
                                          LocalTensor<T> &mmResUb,
                                          LocalTensor<uint8_t> &softmaxTmpUb,
                                          uint32_t startRow,
                                          uint32_t dealRowCount,
                                          uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                          const MSplitInfo &mSplitInfo,
                                          LocalTensor<T> &mmResUb,
                                          TBuf<> &tmpBuf,
                                          uint32_t startRow,
                                          uint32_t dealRowCount,
                                          uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void ProcessAmlaNupdate(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                              const MSplitInfo &mSplitInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                       const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb,
                                                       LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void CopySumMaxToGm(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                          const MSplitInfo &mSplitInfo,
                                          LocalTensor<T> &softmaxSumUb,
                                          LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlockBatchInvariant(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                              const MSplitInfo &mSplitInfo,
                                                              uint32_t startRow,
                                                              uint32_t dealRowCount,
                                                              uint32_t columnCount,
                                                              uint32_t actualColumnCount);
    __aicore__ inline void DealBmm2ResBaseBlockAmla(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                    const MSplitInfo &mSplitInfo,
                                                    uint32_t startRow,
                                                    uint32_t dealRowCount,
                                                    uint32_t columnCount,
                                                    uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                            const MSplitInfo &mSplitInfo,
                                            uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                                LocalTensor<OUT_T> &attenOutUb,
                                                uint32_t wsMStart,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                          const MSplitInfo &mSplitInfo,
                                          LocalTensor<T> &bmm2ResUb,
                                          uint32_t wsMStart,
                                          uint32_t startRow,
                                          uint32_t dealRowCount,
                                          uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                              const MSplitInfo &mSplitInfo,
                                              LocalTensor<T> &bmm2ResUb,
                                              uint32_t wsMStart,
                                              uint32_t startRow,
                                              uint32_t dealRowCount,
                                              uint32_t columnCount,
                                              uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                             const MSplitInfo &mSplitInfo,
                                             LocalTensor<T> &bmm2ResUb,
                                             uint32_t wsMStart,
                                             uint32_t startRow,
                                             uint32_t dealRowCount,
                                             uint32_t columnCount,
                                             uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void DealInvalidRows(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                           LocalTensor<T> &attenOutUb,
                                           uint32_t wsMStart,
                                           uint32_t dealRowCount,
                                           uint32_t columnCount,
                                           uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidMaskRows(const AiInfraInferenceAttentionCommon::RunInfo &info,
                                               const MSplitInfo &mSplitInfo,
                                               LocalTensor<T> &bmm2ResUb,
                                               uint32_t startRow,
                                               uint32_t dealRowCount,
                                               uint32_t columnCount,
                                               uint32_t actualColumnCount);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR,
                                                       int64_t numC,
                                                       const uint32_t stackBufferSize,
                                                       const uint32_t typeSize,
                                                       ConfusionTransposeTiling &tiling);

  protected:
    uint32_t pingpongFlag = 0U;
    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    GlobalTensor<KV_T> vec1ResGm; // 存放A1, A2
    GlobalTensor<T> lseSumFdGm; // no
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;

    // =================================常量区=================================
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t INPUT2_BUFFER_OFFSET = AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET =
        AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
    static constexpr uint32_t BMM2_PRE_PART_NUM = 2;
    static constexpr uint32_t LSE_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_8K;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM =
        AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    AiInfraInferenceAttentionCommon::ConstInfo constInfo = {};
    uint16_t brcbNum = (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(COMPUTE_T));

    T SOFTMAX_MIN_NUM = T(-1.0 / 0.0); // -inf
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr uint64_t headDimAll = 576ULL;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;
    GlobalTensor<uint32_t> metadataGm;

  private:
    // ================================Local Buffer区====================================
    // queue
    TBuf<> inputBuff1; // 32K
    TBuf<> inputBuff2; // 16K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> attenMaskTmpBuff; // 8K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> bmm2PreBuff; // 16K, batchInvariant==true 时分两次处理 bmm2ResPreUb
    TBuf<> softmaxMaxBuff; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff; // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    uint64_t mSizeVector = 0ULL;
    uint64_t mSizeVStart = 0ULL;

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;

    // attention mask
    uint32_t attenMaskSizeAlign = 0U;

    const FusedInferAttentionScoreV2SinkTilingData *__restrict tilingData = nullptr;
};

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuantMla<FIAT>::Init(__gm__ uint8_t *query,
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
                                   __gm__ uint8_t *metadata,
                                   __gm__ uint8_t *attentionOut,
                                   __gm__ uint8_t *softmaxLse,
                                   __gm__ uint8_t *softmaxMax,
                                   __gm__ uint8_t *softmaxSum,
                                   const FusedInferAttentionScoreV2SinkTilingData *__restrict tilingData)
{
    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    // attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask); // 差异
    if (constInfo.softmaxLseFlag) {
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }
    if (constInfo.softmaxMaxSumFlag) {
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    }
    if (constInfo.attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }
    metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
    qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
    kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);

    this->tilingData = tilingData;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitBuffers(TPipe *pipe)
{
    // queue
    pipe->InitBuffer(inputBuff1, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K * 2); // 2:pingpong
    pipe->InitBuffer(outputBuff1, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K);
    if (constInfo.subBlockNum == 1) { // CV1:1场景，vecDealM 变大一倍，需要的buffer变大一倍
        pipe->InitBuffer(outputBuff2, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K);
    } else {
        pipe->InitBuffer(outputBuff2, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K);
    }

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(attenMaskTmpBuff, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff,
                     AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff,
                     AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff,
                     AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(bmm2PreBuff, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);

    pipe->InitBuffer(softmaxMaxBuff,
                     AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff,
                     AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff,
                     AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitParams(
    const struct AiInfraInferenceAttentionCommon::ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                          GlobalTensor<KV_T> vec1ResGm,
                                                                          GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                          GlobalTensor<KV_T> vec1ResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm,
                                                                          GlobalTensor<MM2_OUT_T> mm2ResGm)
{
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm,
                                                                                 GlobalTensor<T> lseMaxFdGm,
                                                                                 GlobalTensor<T> lseSumFdGm)
{
    this->accumOutGm = accumOutGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::FreeEventID()
{
    if (!constInfo.batchInvariant) {
        CrossCoreWaitFlag(constInfo.syncC2V1);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ComputeLogSumExpAndCopyToGm(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &softmaxSumUb,
    LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    // workspace同步修改
    //  src-Shape  { gsizeV, S1, AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM }
    //  dst-Shape  { B  N2, splitKV s1, G, AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM}
    // 这里的offset计算，后续FD切G改切M时，同步改掉
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * kvHeadNum * constInfo.mBaseSize + // taskoffset
                       info.tndCoreStartKVSplitPos * kvHeadNum * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                      AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM; // m轴offset

    LocalTensor<T> tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(lseSumFdGm[offset], tmp, size);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

    tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopy(lseMaxFdGm[offset], tmp, size);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::CopySumMaxToGm(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &softmaxSumUb,
    LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    bool isInvalidRows = AiInfraInferenceCommonFaBaseVector::IsExistInvalidRows(info.nextTokensPerBatch,
                                                                                info.preTokensPerBatch,
                                                                                constInfo.sparseMode,
                                                                                constInfo.attenMaskFlag,
                                                                                constInfo.isRowInvalid);
    SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(mSplitInfo.vecDealM),
                                      static_cast<uint32_t>(brcbNum),
                                      static_cast<uint32_t>(mSplitInfo.vecDealM),
                                      static_cast<uint32_t>(brcbNum)};
    // 由于UB操作首地址必须32字节对齐，所以处理对于lseSum、lseMax必须进行Brcb，使得每一行起始都是32字节对齐
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    uint32_t mOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;

    LocalTensor<T> tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp,
         softmaxSumUb[baseOffset],
         (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum,
         {1, brcbNum});
    if (isInvalidRows) {
        AscendC::PipeBarrier<PIPE_V>();
        AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T, false, 1>(
                            tmp, softmaxMaxUb, negativeIntScalar, (COMPUTE_T)FLOAT_ZERO, softmaxShapeInfo);
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    CopySoftmaxLseMaxSumToGmByLayout(info, softmaxSumGm, tmp, mOffset, mSplitInfo);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);


    tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Brcb(tmp,
         softmaxMaxUb[baseOffset],
         (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum,
         {1, brcbNum});
    if (isInvalidRows) {
        AscendC::PipeBarrier<PIPE_V>();
        AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T, false, 1>(
                            tmp, softmaxMaxUb, negativeIntScalar, (COMPUTE_T)-3e99, softmaxShapeInfo);
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    CopySoftmaxLseMaxSumToGmByLayout(info, softmaxMaxGm, tmp, mOffset, mSplitInfo);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ElewiseCompute(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &mmResUb,
    TBuf<> &tmpBuf,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);

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
        maskInfo.sinkNumber = (constInfo.sinkNumber > 0) ? constInfo.sinkNumber : constInfo.keySinkNumber;
        maskInfo.isEnableSinkNumber = maskInfo.sinkNumber > 0;
        if (constInfo.qSeqSize == 1) {
            maskInfo.layout = AiInfraInferenceCommonFaBaseVector::S1_EQUAL1;
        } else if (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::BSH) {
            maskInfo.layout = AiInfraInferenceCommonFaBaseVector::SG;
        } else {
            maskInfo.layout = AiInfraInferenceCommonFaBaseVector::GS;
        }

        maskInfo.attenMaskType = AiInfraInferenceCommonFaBaseVector::MASK_BOOL; // compatible with int8/uint8
        LocalTensor<bool> maskUb = inputBuff2.Get<bool>();
        maskUb = maskUb[pingpongFlag * INPUT2_BUFFER_OFFSET / sizeof(bool)];
        LocalTensor<bool> attenMaskTmpUb = attenMaskTmpBuff.Get<bool>();
        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
        if (!AiInfraInferenceCommonFaBaseVector::IsSkipAttentionmask(maskInfo)) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCopyIn(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo);
            AscendC::PipeBarrier<PIPE_V>();
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCompute<MM1_OUT_T>(
                mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
        }
        if (!AiInfraInferenceCommonFaBaseVector::IsSkipAttentionmaskForPre(maskInfo)) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
            attenMaskTmpUb = attenMaskTmpBuff.Get<bool>(); // 后续改成attenMaskTmpBuff，加上pingpong
            maskUb = inputBuff2.Get<bool>();
            maskUb = maskUb[pingpongFlag * INPUT2_BUFFER_OFFSET / sizeof(bool)];
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCopyIn(
                maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo, true);
            AiInfraInferenceCommonFaBaseVector::AttentionmaskCompute<MM1_OUT_T>(
                mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo, true);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG + pingpongFlag);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::SoftmaxFlashV2Compute(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &mmResUb,
    LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    SoftmaxFlashV2<T, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(mmResUb,
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

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::AmlaVecCompute(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &mmResUb,
    LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = dealRowCount;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (info.loop - 1) % (constInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (info.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, FLOAT_ZERO, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount); // fp32->fp16/bf16
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    AscendC::PipeBarrier<PIPE_V>();
    if (info.isLastS2Loop) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }

    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AiInfraInferenceCommonFaBaseVector::RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (info.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount); // cof_0=1
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount); // 1 / cof(i)
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    AscendC::PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount); // cof(i - 1) / cof(i) - 1
    AscendC::PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount); // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    AscendC::PipeBarrier<PIPE_V>();
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    AscendC::PipeBarrier<PIPE_V>();

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow]; // 缓存nUpdate

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::DealInvalidRows(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    LocalTensor<T> &attenOutUb,
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

    AiInfraInferenceCommonFaBaseVector::InvalidRows<T,
        AiInfraInferenceCommonFaBaseVector::GeInputUbFormat<LAYOUT_T>()> invalidRows;
    invalidRows(attenOutUb, params);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::DealInvalidMaskRows(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &bmm2ResUb,
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
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;

    AiInfraInferenceCommonFaBaseVector::InvalidMaskRows<MM2_OUT_T, T, SOFTMAX_WITH_BRC>(
        softmaxOutOffset, dealRowCount, columnCount, softmaxMaxUb, negativeIntScalar, bmm2ResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::DealBmm1ResBaseBlock(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mSplitInfo, mmResUb, tmpBuff1, startRow, dealRowCount, columnCount, actualColumnCount);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(
        info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);

    if (!constInfo.batchInvariant) {
        AscendC::PipeBarrier<PIPE_V>();
        AmlaVecCompute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    }

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessAmlaNupdate(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (info.isFirstSInnerLoop) {
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize =
        64U; // tmpQue size 32KB，一次只能处理64个N，最大保存的数据大小：64*128*sizeof(int32)
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B
    uint32_t subMSize = Align(mSplitInfo.vecDealM, 16U);
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t); // 单个datablock的元素数，int32_t类型的为32/4=8

    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize; // 尾块
    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }

        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // (m,1)单次brcb扩充成(m,8), 重复16次, 扩充为(m,128)
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock],
                 nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(
                      dGroupSize / elementPerBlock), // 单次迭代内，目的操作数不同datablock间地址步长,单位为datablock
                  static_cast<uint16_t>(dGroupSize)}); // 相邻迭代间，目的操作数相同datablock地址步长
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                              (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + loop * mSplitSize) * headDim;

        AscendC::PipeBarrier<PIPE_MTE3>();
        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE; // 每个block是128个元素，单位为32B
        dataCopyParams.srcStride = 0; // 前面一个数据块的尾与后面数据块的头的间隔
        dataCopyParams.dstStride =
            static_cast<uint16_t>((headDim - dGroupSize) * sizeof(int32_t) / ONE_BLOCK_SIZE); // 单位为32B
        for (uint32_t i = 0; i < headDim / dGroupSize; i++) { // 4=512/128
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize], tmpQue, dataCopyParams);
        }

        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec1SingleBuf(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }

    const uint32_t MAX_M_SPLIT_SIZE = 128; // CV1:1场景，M轴方向最大支持128
    mSplitSize = (constInfo.subBlockNum == 1 && mSplitSize > MAX_M_SPLIT_SIZE) ? MAX_M_SPLIT_SIZE : mSplitSize;

    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info,
                             mSplitInfo,
                             i * mSplitSize,
                             dealSize,
                             info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::CopySoftmaxLseMaxSumToGmByLayout(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    GlobalTensor<float> &gmDst,
    LocalTensor<T> &srcUb,
    uint32_t mOffset,
    const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (LAYOUT_T == FIA_LAYOUT::TND) {
        uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
        uint64_t bN2Offset =
            static_cast<uint64_t>(prefixBS1) * constInfo.qHeadNum + static_cast<uint64_t>(info.n2Idx) * constInfo.gSize;
        DataCopySoftmaxLseTND(gmDst, srcUb, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo);
    } else if (LAYOUT_T == FIA_LAYOUT::BSND || LAYOUT_T == FIA_LAYOUT::BSH) {
        uint64_t bN2Offset = static_cast<uint64_t>(info.bIdx) * constInfo.qHeadNum * constInfo.qSeqSize +
                             static_cast<uint64_t>(info.n2Idx) * constInfo.gSize * constInfo.qSeqSize;
        DataCopySoftmaxLseBSND(
            gmDst, srcUb, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo, qActSeqLensParser, info.bIdx);
    } else {
        uint64_t bN2Offset = static_cast<uint64_t>(info.bIdx) * constInfo.qHeadNum * constInfo.qSeqSize +
                             static_cast<uint64_t>(info.n2Idx) * constInfo.gSize * constInfo.qSeqSize;
        DataCopySoftmaxLseBNSD<COMPUTE_T, Q_MODE>(
            gmDst, srcUb, bN2Offset, mOffset, mSplitInfo.vecDealM, constInfo, qActSeqLensParser, info.bIdx);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec1L(const AiInfraInferenceAttentionCommon::RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM
                                                              : (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (constInfo.subBlockNum == 1) { // CV1:1场景
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
        } else if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<AiInfraInferenceAttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);

        if (!constInfo.batchInvariant) {
            CrossCoreWaitFlag(constInfo.syncC2V1);
            // add nUpdate to mm2ResGm
            ProcessAmlaNupdate(info, mSplitInfo);
            CrossCoreSetFlag<AiInfraInferenceAttentionCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                constInfo.syncV1NupdateC2);
        }

        // move lse for flash decode
        if (info.isLastS2Loop) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            LocalTensor sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(COMPUTE_T)];
            LocalTensor maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(COMPUTE_T)];

            // softmaxMaxSumFlag和softmaxLseFlag互斥，同时仅支持1个flag打开
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            } else if (constInfo.softmaxLseFlag) {
                if (mSplitInfo.vecDealM == 0) {
                    continue;
                }

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

                LocalTensor<COMPUTE_T> tmpLseResCastTensor = outputBuff2.Get<COMPUTE_T>();
                WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
                DataCopy(tmpLseResCastTensor, totalLseUb, mSplitInfo.vecDealM * brcbNum);

                SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
                WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

                uint32_t mOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
                CopySoftmaxLseMaxSumToGmByLayout(info, softmaxLseGm, tmpLseResCastTensor, mOffset, mSplitInfo);
                SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
            }

            if (constInfo.softmaxMaxSumFlag) {
                CopySumMaxToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline uint64_t FiaBlockVecNonQuantMla<FIAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    uint32_t bN2IdxOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t gS1IdxOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t s2SplitNumOfFdHead_local[optiling::FIA_MAX_AIC_CORE_NUM] = {0};
    uint32_t usedCoreNum = metadataGm.GetValue(optiling::GetBaseMetaAbsIndex(optiling::USED_CORE_NUM_INDEX));
    for (uint32_t i = 0; i < usedCoreNum; ++i) {
        bN2IdxOfFdHead_local[i] =
            metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::BN2_IDX_OF_FD_HEAD_INDEX));
        gS1IdxOfFdHead_local[i] =
            metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::GS1_IDX_OF_FD_HEAD_INDEX));
        s2SplitNumOfFdHead_local[i] =
            metadataGm.GetValue(optiling::GetAICMetaAbsIndex(i, optiling::S2_SPLIT_NUM_OF_FD_HEAD_INDEX));
    }
    uint64_t accumTmpOutNum = 0;
    int taskId = 0;
    while (taskId < usedCoreNum &&
           (bN2IdxOfFdHead_local[taskId] != bN2Idx || gS1IdxOfFdHead_local[taskId] * constInfo.mBaseSize != gS1Idx)) {
        accumTmpOutNum += s2SplitNumOfFdHead_local[taskId]; // 计算前面的workspace数
        taskId++;
    }
    return accumTmpOutNum;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec2SingleBuf(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo)
{
    if (!constInfo.batchInvariant) {
        if (!info.isLastS2Loop) {
            return;
        }
    }
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec2L(const AiInfraInferenceAttentionCommon::RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM
                                                              : (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (constInfo.subBlockNum == 1) { // CV1:1场景
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM;
        } else if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
    if (constInfo.batchInvariant) {
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV2C2);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::ProcessVec2Inner(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    uint32_t mStartRow,
    uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        if (constInfo.batchInvariant) {
            DealBmm2ResBaseBlockBatchInvariant(info,
                                               mSplitInfo,
                                               i * mSplitSize + mStartRow,
                                               dealSize,
                                               headDimAlign,
                                               headDim);
        } else {
            DealBmm2ResBaseBlockAmla(info,
                                     mSplitInfo,
                                     i * mSplitSize + mStartRow,
                                     dealSize,
                                     headDimAlign,
                                     headDim);
        }
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::GetConfusionTransposeTiling(int64_t numR,
                                                                                 int64_t numC,
                                                                                 const uint32_t stackBufferSize,
                                                                                 const uint32_t typeSize,
                                                                                 ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::Bmm2FDDataCopyOut(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &bmm2ResUb,
    uint32_t wsMStart,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    DealInvalidMaskRows(info, mSplitInfo, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * kvHeadNum * constInfo.mBaseSize * headDim + // taskoffset
                      info.tndCoreStartKVSplitPos * kvHeadNum * constInfo.mBaseSize * headDim + // 份数offset
                      wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) /
        (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(dst, tmp, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::Bmm2DataCopyOutTrans(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
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
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::Bmm2CastAndCopyOut(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &bmm2ResUb,
    uint32_t wsMStart,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    DealInvalidMaskRows(info, mSplitInfo, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::Bmm2ResCopyOut(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    LocalTensor<T> &bmm2ResUb,
    uint32_t wsMStart,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(
                info, mSplitInfo, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(
                info, mSplitInfo, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(
            info, mSplitInfo, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::DealBmm2ResBaseBlockBatchInvariant(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t inOutBaseOffset = mStart * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;

    if (!info.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        uint32_t idx = info.loop % (constInfo.preLoadNum);
        uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        // bmm2PreBuff 16K，按行分两次处理，每次最多 halfRows 行
        // halfRows 向上对齐到 8（8 * sizeof(float) = 32B），保证 partStartRow = halfRows 满足 Brcb src 32B 对齐要求
        // 若 dealRowCount < 8，halfRows 取 dealRowCount，此时 part=1 的 partRows=0 会 continue，partStartRow 不被使用
        uint32_t halfRows = Align((dealRowCount + 1) / 2, 8U);
        halfRows = (halfRows > dealRowCount) ? dealRowCount : halfRows;
        uint32_t halfRowSize = halfRows * columnCount;
        LocalTensor<MM2_OUT_T> bmm2ResPreUb = bmm2PreBuff.Get<MM2_OUT_T>();
        for (uint32_t part = 0; part < BMM2_PRE_PART_NUM; part++) {
            uint32_t partRows = (part == 0) ? halfRows : (dealRowCount - halfRows);
            if (partRows == 0) {
                continue;
            }
            uint32_t partSize = partRows * columnCount;
            uint32_t partOffset = part * halfRowSize;
            uint32_t partStartRow = part * halfRows;
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
            DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + partOffset], partSize);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
            LocalTensor<T> expUb = attenMaskTmpBuff.Get<T>();
            Brcb(expUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset + partStartRow],
                (partRows + 7) / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AiInfraInferenceCommonFaBaseVector::RowMuls(
                bmm2ResPreUb, bmm2ResPreUb, expUb, partRows, columnCount, actualColumnCount);
            AscendC::PipeBarrier<PIPE_V>();
            Add(bmm2ResUb[partOffset], bmm2ResUb[partOffset], bmm2ResPreUb, partSize);
            AscendC::PipeBarrier<PIPE_V>();
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        }
    }

    if (info.isLastS2Loop) {
        uint32_t idx = info.loop % (constInfo.preLoadNum);
        LocalTensor<T> tmpSumUb = attenMaskTmpBuff.Get<T>();
        Brcb(tmpSumUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset],
            (dealRowCount + 7) / 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AiInfraInferenceCommonFaBaseVector::RowDivs(
            bmm2ResUb, bmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
        AscendC::PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, mSplitInfo, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        LocalTensor<T> outUb = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(outUb, bmm2ResUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], outUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuantMla<FIAT>::DealBmm2ResBaseBlockAmla(
    const AiInfraInferenceAttentionCommon::RunInfo &info,
    const MSplitInfo &mSplitInfo,
    uint32_t startRow,
    uint32_t dealRowCount,
    uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t inOutBaseOffset = mStart * columnCount;
    uint64_t srcGmOffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    // 将绝对值大于 1e10 的数置为 0
    LocalTensor<T> absBmm2ResUb = bmm2ResUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, FLOAT_ZERO, SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = info.loop % (constInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = attenMaskTmpBuff.Get<T>();
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AiInfraInferenceCommonFaBaseVector::RowDivs(
        bmm2ResUb, tmpBmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(info, mSplitInfo, bmm2ResUb, mStart, startRow, dealRowCount, columnCount, actualColumnCount);
}

#endif
