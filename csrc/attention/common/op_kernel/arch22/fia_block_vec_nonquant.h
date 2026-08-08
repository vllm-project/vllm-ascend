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
 * \file fia_block_vec_nonquant.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_NONQUANT_H
#define FIA_BLOCK_VEC_NONQUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../memory_copy.h"
#include "../post_quant.h"
#include "../const_def.h"

using namespace AttentionCommon;
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
    static constexpr GmFormat PostQuant_FORMAT = GmFormat::NGD;
    static constexpr bool ENABLE_TREE = FIAT::enableTree;

    using UPDATE_T = T;
    using TMP_T = T;
    using COMPUTE_T =  T;
    using SOFTMAX_TYPE =  T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using SINK_T = bfloat16_t;
    using PSE_T = typename AscendC::Conditional<IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    __aicore__ inline FiaBlockVecNonQuant(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM1_OUT_T> mm1ResGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm);
    __aicore__ inline void InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<T> lseMaxFdGm,
        GlobalTensor<T> lseSumFdGm);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void DealZeroActSeqLenWithPostQuant(uint32_t bIdx, uint32_t n2Idx);
protected:
    __aicore__ inline void SetMSplitInfo(uint32_t mDealSize);
    // V1
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, uint32_t startRow, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb, TBuf<> &tmpBuf,
        uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb,
        LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<COMPUTE_T> &softmaxSumUb, LocalTensor<COMPUTE_T> &softmaxMaxUb);
    // V2
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                              uint32_t actualColumnCount);
    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUT_T> &ubTensor, GmCoord &gmCoord);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidMaskRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart,
                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                               uint32_t actualColumnCount);
    __aicore__ inline void DealInvalidRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &attenOutUb, uint32_t wsMStart,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Vec1SinkCompute(const RunInfo &info, uint32_t idx, uint32_t wsMStart, uint32_t dealRowCount);
    __aicore__ inline void Vec1SinkSoftmaxProc(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                            uint32_t offset, uint32_t dealRowCountBrcb);
    __aicore__ inline void Vec1GetSinkValue(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                            uint32_t wsMStart, uint32_t dealRowCount);
    __aicore__ inline void SinkCopyIn(const RunInfo &info, LocalTensor<COMPUTE_T> &sinkBuf);
    __aicore__ inline void SinkValueNoBrc(LocalTensor<COMPUTE_T> tmpSinkResUb,
                                            LocalTensor<COMPUTE_T> tmpSinkResUbBrcb, uint32_t dealRowCount);
    __aicore__ inline void SinkInvalidRow(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                            int64_t s1Idx, int64_t row, int64_t dealRowCount);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline void DealPostQuantOutPerChn(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
                                                  uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void DealPostQuantOutPerTensor(LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<PSE_T> pseShiftGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ; // 需确认后续是否会用到
    GlobalTensor<uint64_t> actualSeqLengthsGm; // 需确认后续是否会用到
    GlobalTensor<SINK_T> sinkGm;

    //postquant
    FaGmTensor<T, PostQuant_FORMAT> quantScale2GmTensor;
    FaGmTensor<T, PostQuant_FORMAT> quantOffset2GmTensor;
    FaGmTensor<bfloat16_t, PostQuant_FORMAT> quantScale2Bf16GmTensor;
    FaGmTensor<bfloat16_t, PostQuant_FORMAT> quantOffset2Bf16GmTensor;
    PostQuant<GetOutUbFormat<LAYOUT_T>()> postQuantProcesser;

    __gm__ uint8_t *actualSequenceLengthsQ = nullptr;

    // =================================常量区=================================
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    T SOFTMAX_MIN_NUM = T(-1.0/0.0); // -inf
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_2K;
    static constexpr uint32_t LSE_TMP_BUFFER_SIZE = ConstInfo::BUFFER_SIZE_BYTE_8K;
    static constexpr uint32_t DATA_BLOCK_NUM = 8;
    static constexpr uint16_t brcbNum = (AttentionCommon::BYTE_BLOCK / sizeof(COMPUTE_T));

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    T scale2Value = 0;
    T offset2Value = 0;
    bool isQuantOffset2Exit = false;
    bool isQuant2PerChn = false;
    bool isQuant2Bf16 = false;

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

    // ================================其他成员区========================================
    ConstInfo constInfo = {};
    MSplitInfo mSplitInfo = {};

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>(); 
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>(); 
    ActualSeqLensParser<Q_MODE> qActSeqLensParser; 
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // PSE仅在Q的lauot为BSH/BSND/BNSD时支持
    static constexpr bool IS_SUPPORT_PSE = IsSupportPse<LAYOUT_T>();
    static constexpr UbFormat PSE_UB_FORMAT = GetPseUbFormat<LAYOUT_T>();
    FaGmTensor<PSE_T, GmFormat::BN2GS1S2> pseShiftGmTensor;
    CopyPSEGmToUb<PSE_T, GmFormat::BN2GS1S2, PSE_UB_FORMAT> copyPSEGmToUb;
    bool pseHasBatch = true;

    bool learnableSinkFlag = false;
};


template <typename FIAT> class FiaBlockVecNonQuantDummy {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;

    using UPDATE_T = T;
    using MM1_OUT_T = T;
    using MM2_OUT_T = T;

    __aicore__ inline FiaBlockVecNonQuantDummy(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM1_OUT_T> mm1ResGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm);
    __aicore__ inline void InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<T> lseMaxFdGm,
        GlobalTensor<T> lseSumFdGm);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void DealZeroActSeqLenWithPostQuant(uint32_t bIdx, uint32_t n2Idx);
};

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuant<FIAT>::InitParams(const struct ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse)
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
        sinkGm.SetGlobalBuffer((__gm__ SINK_T *)learnableSink);
    }
    qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize); 
    kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);

    if constexpr (IS_SUPPORT_PSE) {
        if (constInfo.pseShiftFlag) {
            pseShiftGm.SetGlobalBuffer((__gm__ PSE_T *)pseShift);
            pseShiftGmTensor.gmTensor = pseShiftGm;
            pseShiftGmTensor.offsetCalculator.Init(
                constInfo.pseShiftByBatch ? constInfo.batchSize : 1, constInfo.kvHeadNum, constInfo.gSize,
                constInfo.pseShiftS1, constInfo.pseShiftS2, this->actualSeqLengthsGmQ, constInfo.actualLenQDims);
        }
    }
    if constexpr (POST_QUANT) {
        InitPostQuant(quantScale2, quantOffset2);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitVec1GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM1_OUT_T> mm1ResGm)
{
    this->vec1ResGm = vec1ResGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitVec2GlobalTensor(GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm)
{
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
}


template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitFlashDecodeGlobalTensor(GlobalTensor<T> accumOutGm,
    GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm)
{
    this->accumOutGm = accumOutGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitBuffers(TPipe *pipe)
{
    // in queue
    pipe->InitBuffer(inputQue1, 2, ConstInfo::BUFFER_SIZE_BYTE_32K); // 2:pingpong
    pipe->InitBuffer(inputQue2, 2, ConstInfo::BUFFER_SIZE_BYTE_16K);  // 2:pingpong

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

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::AllocEventID()
{
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::FreeEventID()
{
}

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
        uint32_t alignVal = AttentionCommon::BYTE_BLOCK / sizeof(COMPUTE_T);
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
        DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
    }

    if (info.isLastS2Loop) {
        uint32_t outIdx = info.loop % constInfo.preLoadNum;
        if (unlikely(learnableSinkFlag)) {
            Vec1SinkCompute(info, outIdx, mSplitInfo.nBufferStartM + mSplitInfo.vecStartM, mSplitInfo.vecDealM);
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
                LocalTensor<COMPUTE_T> lseSumUb = tmpBuff1.GetWithOffset<COMPUTE_T>(LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE);
                LocalTensor<COMPUTE_T> lseMaxUb = tmpBuff1.GetWithOffset<COMPUTE_T>(
                    LSE_TMP_BUFFER_SIZE, LSE_TMP_BUFFER_SIZE  *2);
                Brcb(lseSumUb, sumTensor[mSplitInfo.nBufferStartM / 2], (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum, 
                    {1, brcbNum});
                AscendC::PipeBarrier<PIPE_V>();
                Brcb(lseMaxUb, maxTensor[mSplitInfo.nBufferStartM / 2], (mSplitInfo.vecDealM + brcbNum - 1) / brcbNum, 
                    {1, brcbNum});
                AscendC::PipeBarrier<PIPE_V>();
                fa_base_vector::ComputeSoftMaxLse(totalLseUb, lseSumUb, lseMaxUb, mSplitInfo.vecDealM);
            } else {
                fa_base_vector::ComputeSoftMaxLse(totalLseUb, sumTensor, maxTensor, mSplitInfo.vecDealM);
            }

            bool isInvalidRows = fa_base_vector::IsExistInvalidRows(info.nextTokensPerBatch, info.preTokensPerBatch, 
                constInfo.sparseMode, constInfo.attenMaskFlag, constInfo.isRowInvalid);

            if (isInvalidRows) { // 存在行无效场景
                SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(mSplitInfo.vecDealM), static_cast<uint32_t>(brcbNum),
                static_cast<uint32_t>(mSplitInfo.vecDealM), static_cast<uint32_t>(brcbNum)};

                if constexpr (SOFTMAX_WITH_BRC) {
                    AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T>(totalLseUb, maxTensor, negativeIntScalar, 
                        FLOAT_INF, softmaxShapeInfo);
                } else {
                    AdjustSoftMaxRes<COMPUTE_T, COMPUTE_T, false, 1>(totalLseUb, maxTensor, negativeIntScalar, 
                        FLOAT_INF, softmaxShapeInfo);
                }
            }

            LocalTensor<T> tmpLseResCastTensor = outputQue2.AllocTensor<T>();
            DataCopy(tmpLseResCastTensor, totalLseUb, mSplitInfo.vecDealM * brcbNum);
            outputQue2.EnQue(tmpLseResCastTensor);
            outputQue2.DeQue<T>();
            uint32_t mOffset = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
            if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
                uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
                uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + info.n2Idx * constInfo.gSize;
                DataCopySoftmaxLseTND<T>(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM,
                                         constInfo);
            } else if constexpr (LAYOUT_T == FIA_LAYOUT::NTD) {
                uint32_t prefixBS1 = info.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(info.bIdx - 1);
                uint32_t s1Size = info.bIdx == 0U ? actualSeqLengthsGmQ.GetValue(0U) :
                                                    actualSeqLengthsGmQ.GetValue(info.bIdx) -
                                                        actualSeqLengthsGmQ.GetValue(info.bIdx - 1U);
                uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + info.n2Idx * constInfo.gSize;
                DataCopySoftmaxLseNTD<T>(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset, mSplitInfo.vecDealM,
                                         constInfo, s1Size);
            } else if constexpr (LAYOUT_T == FIA_LAYOUT::BSND || LAYOUT_T == FIA_LAYOUT::BSH) {
                uint64_t bN2Offset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                                     info.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                DataCopySoftmaxLseBSND<T, Q_MODE>(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset,
                                                  mSplitInfo.vecDealM, constInfo, qActSeqLensParser, info.bIdx);
            } else { // BNSD
                uint64_t bN2Offset = info.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                                     info.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                DataCopySoftmaxLseBNSD<T, Q_MODE>(softmaxLseGm, tmpLseResCastTensor, bN2Offset, mOffset,
                                                  mSplitInfo.vecDealM, constInfo, qActSeqLensParser, info.bIdx);
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
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::ElewiseCompute(
    const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<MM1_OUT_T>(constInfo.scaleValue), dealRowCount * columnCount);

    if constexpr (IS_SUPPORT_PSE) {
        if (constInfo.pseShiftFlag) {
            LocalTensor<PSE_T> pseShiftB16 = inputQue2.AllocTensor<PSE_T>();
            FaUbTensor<PSE_T> pseShiftUbTensor {
                .tensor = pseShiftB16,
                .rowCount = dealRowCount,
                .colCount = columnCount
            };
            GmPseCoord pseCoord = {
                .bIdx = constInfo.pseShiftByBatch ? info.bIdx : 0,
                .n2Idx = info.n2Idx,
                .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow,
                .s2Idx = info.s2Idx * constInfo.s2BaseSize,
                .gS1DealSize = dealRowCount,
                .s2DealSize = actualColumnCount,
                .s1LeftPaddingSize = info.qPaddingBeginOffset,
                .s2LeftPaddingSize = info.kvPaddingBeginOffset,
                .actualBIdx = info.bIdx
            };
            bool qsEqualOne = (constInfo.qSeqSize == 1);
            copyPSEGmToUb(pseShiftUbTensor, pseShiftGmTensor, pseCoord, qsEqualOne);
            inputQue2.EnQue(pseShiftB16);
            inputQue2.DeQue<PSE_T>();
            LocalTensor<T> pseShiftUbFP32 = tmpBuf.Get<T>();
            AscendC::Cast(pseShiftUbFP32, pseShiftB16, AscendC::RoundMode::CAST_NONE, dealRowCount * columnCount);
            inputQue2.FreeTensor(pseShiftB16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(mmResUb, mmResUb, pseShiftUbFP32, dealRowCount * columnCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    if (constInfo.attenMaskFlag == 1) {
        AscendC::PipeBarrier<PIPE_V>();
        fa_base_vector::MaskInfo maskInfo;
        maskInfo.gs1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
        maskInfo.gs1dealNum = dealRowCount;
        maskInfo.s1Size = info.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = info.s2Idx * constInfo.s2BaseSize;
        maskInfo.s2dealNum = info.actualSingleProcessSInnerSize;
        maskInfo.s2Size = info.actS2Size;
        maskInfo.preToken = constInfo.preToken;
        maskInfo.nextToken = constInfo.nextToken;
        maskInfo.sparseMode = static_cast<fa_base_vector::SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = info.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskBatchStride;
        maskInfo.attenMaskStride = constInfo.attenMaskStride;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = info.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = info.kvPaddingBeginOffset;
        if (constInfo.qSeqSize == 1) {
            maskInfo.layout = fa_base_vector::S1_EQUAL1;
        } else if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::BSH) {
            maskInfo.layout = fa_base_vector::SG;
        } else {
            maskInfo.layout = fa_base_vector::GS;
        }

        maskInfo.attenMaskType = fa_base_vector::MASK_BOOL; // compatible with int8/uint8

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
        if constexpr (ENABLE_TREE) {
            // TREE模式：提前分配并初始化为0，保持占用直到处理完成
            LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
            LocalTensor<bool> attenMaskTmpUb = maskUb[BUFFER_SIZE_BYTE_16K / 2];
            event_t eventIdVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            LocalTensor<int16_t> mask16 = maskUb.template ReinterpretCast<int16_t>();
            Duplicate(mask16, static_cast<int16_t>(0), BUFFER_SIZE_BYTE_8K / sizeof(int16_t));
            maskUb = mask16.template ReinterpretCast<bool>();
            SetFlag<HardEvent::V_MTE2>(eventIdVMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVMte2);
            maskInfo.attenMaskBatchStride *= maskInfo.batchIdx;
            if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
                maskInfo.attenMaskStride = info.actS1Size;
                maskInfo.attenMaskBatchStride = 0;
                for (int32_t i = 0; i < maskInfo.batchIdx; i++) {
                    maskInfo.attenMaskBatchStride += qActSeqLensParser.GetActualSeqLength(i) * qActSeqLensParser.GetActualSeqLength(i);
                }
            }
            if (!fa_base_vector::IsSkipAttentionmask<ENABLE_TREE>(maskInfo)) {
                fa_base_vector::AttentionmaskCopyIn<bool, bool, true>(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo);
                AscendC::PipeBarrier<PIPE_V>();
                fa_base_vector::AttentionMaskCompute<MM1_OUT_T>(mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo);
            }
            inputQue2.FreeTensor(maskUb);
        } else {
            if (!fa_base_vector::IsSkipAttentionmask<ENABLE_TREE>(maskInfo)) {
                LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
                LocalTensor<bool> attenMaskTmpUb = maskUb[BUFFER_SIZE_BYTE_16K / 2];
                fa_base_vector::AttentionmaskCopyIn(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo);
                AscendC::PipeBarrier<PIPE_V>();
                fa_base_vector::AttentionMaskCompute<MM1_OUT_T>(mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo);
                inputQue2.FreeTensor(maskUb);
            }
            if (!fa_base_vector::IsSkipAttentionmaskForPre(maskInfo)) {
                LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
                LocalTensor<bool> attenMaskTmpUb = maskUb[BUFFER_SIZE_BYTE_16K / 2];
                fa_base_vector::AttentionmaskCopyIn(maskUb, attenMaskBoolGm, attenMaskTmpUb, maskInfo, true);
                fa_base_vector::AttentionMaskCompute<MM1_OUT_T>(mmResUb, mmResUb, maskUb, ubWorkSpace, maskInfo, true);
                inputQue2.FreeTensor(maskUb);
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, LocalTensor<MM1_OUT_T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
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
    uint32_t outIdx = info.loop % constInfo.preLoadNum;
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % constInfo.preLoadNum;
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset];
    }
    if constexpr (SOFTMAX_WITH_BRC) {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecNonQuant<FIAT>::ProcessVec2SingleBuf(const RunInfo &info)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;
    // 为防止后续处理函数溢出或异常，将 mSplitSize 的最大值限制在255
    if (mSplitSize > fa_base_vector::MAX_REPEAT_TIMES) {
        mSplitSize = fa_base_vector::MAX_REPEAT_TIMES;
    }
    if constexpr (!SOFTMAX_WITH_BRC) {
        uint32_t alignVal = AttentionCommon::BYTE_BLOCK / sizeof(COMPUTE_T);
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
        uint32_t idx = info.loop % constInfo.preLoadNum;

        if constexpr (SOFTMAX_WITH_BRC) {
            RowMuls<COMPUTE_T>(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpExpBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpExpBrcbResUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            RowMuls<COMPUTE_T>(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
        }

        AscendC::PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue1.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.isLastS2Loop) {
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % constInfo.preLoadNum;

        if constexpr (SOFTMAX_WITH_BRC) {
            fa_base_vector::RowDivs<COMPUTE_T>(bmm2ResUb, bmm2ResUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                dealRowCount, columnCount, actualColumnCount);
        } else {
            LocalTensor<COMPUTE_T> tmpSumBrcbResUb = tmpBuff1.Get<COMPUTE_T>();
            Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + baseOffset],
                (dealRowCount + this->brcbNum - 1) / this->brcbNum, {1, this->brcbNum});
            AscendC::PipeBarrier<PIPE_V>();
            fa_base_vector::RowDivs<COMPUTE_T>(bmm2ResUb, bmm2ResUb, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
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
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
    uint32_t wsMStart, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
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
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
    uint32_t wsMStart, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<MM2_OUT_T> tmp = outputQue1.AllocTensor<MM2_OUT_T>();
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    outputQue1.EnQue(tmp);
    outputQue1.DeQue<T>();
    uint64_t offset = info.accumTmpOutNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
                      info.tndCoreStartKVSplitPos * constInfo.mBaseSize * constInfo.headDim + // 份数offset
                      wsMStart * actualColumnCount;                                             // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;      
    DataCopyPad(dst, tmp, dataCopyParams);
    outputQue1.FreeTensor(tmp);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealInvalidMaskRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb,
    uint32_t wsMStart, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if (!constInfo.isRowInvalid || !constInfo.attenMaskFlag) {
        return;
    }
    if (constInfo.sparseMode != fa_base_vector::DEFAULT_MASK && constInfo.sparseMode != fa_base_vector::ALL_MASK) {
        return;
    }
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    if constexpr (SOFTMAX_WITH_BRC) {
        baseOffset = baseOffset * (AttentionCommon::BYTE_BLOCK / sizeof(T));
    }

    uint32_t outIdx = info.loop % constInfo.preLoadNum;
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(T) + baseOffset;

    fa_base_vector::InvalidMaskRows<MM2_OUT_T, T, SOFTMAX_WITH_BRC>(softmaxOutOffset, dealRowCount, columnCount,
        softmaxMaxUb, negativeIntScalar, bmm2ResUb);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Bmm2CastAndCopyOut(const RunInfo &info, 
    LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t wsMStart, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DealInvalidRows(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    DealInvalidMaskRows(info, bmm2ResUb, wsMStart, startRow, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (POST_QUANT) {
        if (isQuant2PerChn) {
            DealPostQuantOutPerChn(info, bmm2ResUb, startRow, dealRowCount, columnCount);
        } else {
            DealPostQuantOutPerTensor(bmm2ResUb, startRow, dealRowCount, columnCount);
        }
    }
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (POST_QUANT) {
        LocalTensor<half> quant2ResHalf = tmpBuff1.Get<half>();
        Cast(quant2ResHalf, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tmpBmm2ResCastTensor, quant2ResHalf, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    } else {
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
        } else {
            Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
        }
    }

    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuant<FIAT>::CopyAttentionOut(FaUbTensor<OUT_T> &ubTensor, GmCoord &gmCoord)
{
    if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
            constInfo.qSeqSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims,
            constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
            constInfo.qSeqSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims,
            constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
            constInfo.qSeqSize, constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
        constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim,
            actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim,
            actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecNonQuant<FIAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    FaUbTensor<OUT_T> ubTensor {
        .tensor = attenOutUb,
        .rowCount = dealRowCount,
        .colCount = columnCount,
    };
    GmCoord gmCoord {
        .bIdx = info.bIdx,
        .n2Idx = info.n2Idx,
        .gS1Idx = info.gS1Idx + wsMStart,
        .dIdx = 0,
        .gS1DealSize = dealRowCount,
        .dDealSize = (uint32_t)constInfo.headDim
    };

    CopyAttentionOut(ubTensor, gmCoord);
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
    //  src-Shape  { gsizeV, S1, AttentionCommon::FP32_BLOCK_ELEMENT_NUM }
    //  dst-Shape  { B  N2, splitKV s1, G, AttentionCommon::FP32_BLOCK_ELEMENT_NUM}
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * brcbNum;
    uint64_t offset = (info.accumTmpOutNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       AttentionCommon::FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if constexpr (SOFTMAX_WITH_BRC) {              
        DataCopy(lseSumFdGm[offset], softmaxSumUb[baseOffset], size);
        DataCopy(lseMaxFdGm[offset], softmaxMaxUb[baseOffset], size);       
    } else {
        LocalTensor<T> tmp = outputQue2.AllocTensor<T>();   
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / brcbNum, {1, brcbNum});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseSumFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
        tmp = outputQue2.AllocTensor<T>(); 
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / brcbNum, {1, brcbNum});
        outputQue2.EnQue(tmp);
        outputQue2.DeQue<T>();
        DataCopy(lseMaxFdGm[offset], tmp, size);
        outputQue2.FreeTensor(tmp);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealInvalidRows(const RunInfo &info, LocalTensor<MM2_OUT_T> &attenOutUb,
                                                      uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                      uint32_t actualColumnCount)
{
    if (!constInfo.attenMaskFlag) {
        return;
    }

    if (constInfo.sparseMode == fa_base_vector::ALL_MASK || constInfo.sparseMode == fa_base_vector::LEFT_UP_CAUSAL) {
        return;
    }

    fa_base_vector::InvalidRowParams params {
        .actS1Size = info.actS1Size,
        .gSize = constInfo.gSize,
        .gS1Idx = info.gS1Idx + wsMStart,
        .dealRowCount = dealRowCount,
        .columnCount = columnCount,
        .preTokensPerBatch = info.preTokensPerBatch,
        .nextTokensPerBatch = info.nextTokensPerBatch,
    };

    fa_base_vector::InvalidRows<T, fa_base_vector::GeInputUbFormat<LAYOUT_T>()> invalidRows;
    invalidRows(attenOutUb, params);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::SinkCopyIn(const RunInfo &info, LocalTensor<COMPUTE_T> &sinkBuf)
{
    uint64_t sinkGmOffset = info.n2Idx * constInfo.gSize;
    LocalTensor<SINK_T> sinkCopyInBuf = inputQue1.AllocTensor<SINK_T>();
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = constInfo.gSize * sizeof(SINK_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;
    DataCopyPadExtParams<SINK_T> sinkPadParams;
    sinkPadParams.isPad = true;
    sinkPadParams.paddingValue = static_cast<SINK_T>(0);
    DataCopyPad(sinkCopyInBuf, sinkGm[sinkGmOffset], sinkCopyParams, sinkPadParams);
    inputQue1.EnQue(sinkCopyInBuf);
    inputQue1.DeQue<SINK_T>();

    LocalTensor<COMPUTE_T> tmpSinkCastUb = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, BUFFER_SIZE_BYTE_8K * 3);
    Cast(tmpSinkCastUb, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo.gSize);
    AscendC::PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(sinkCopyInBuf);

    Brcb(sinkBuf, tmpSinkCastUb, (constInfo.gSize + brcbNum - 1) / brcbNum, {1, brcbNum});
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::SinkInvalidRow(const RunInfo &info, LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
    int64_t s1Idx, int64_t row, int64_t dealRowCount)
{
    int64_t s1BottomTok = info.actS1Size + info.preTokensPerBatch;
    int64_t s1Tok = -info.nextTokensPerBatch;
    const COMPUTE_T minValue = *((COMPUTE_T *)&negativeIntScalar);

    if (unlikely(info.nextTokensPerBatch < 0)) { // 上方存在行无效
        if (s1Idx < s1Tok) {
            Duplicate(tmpSinkResUbBrcb[row * brcbNum], minValue, brcbNum * dealRowCount);
        }
    }

    if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) { // sparse = 3时，不存在下方行无效，直接返回
        return;
    }

    if (unlikely(info.preTokensPerBatch < 0)) { // 下方存在行无效
        if (s1Idx >= s1BottomTok && s1Idx < info.actS1Size) {
            Duplicate(tmpSinkResUbBrcb[row * brcbNum], minValue, brcbNum * dealRowCount);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Vec1GetSinkValue(const RunInfo &info,
                                                                   LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb,
                                                                   uint32_t wsMStart, uint32_t dealRowCount)
{
    constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();


    LocalTensor<COMPUTE_T> sinkBuf = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, BUFFER_SIZE_BYTE_8K * 2);
    SinkCopyIn(info, sinkBuf);

    bool isInvalidRows = fa_base_vector::IsExistInvalidRows(info.nextTokensPerBatch, info.preTokensPerBatch, constInfo.sparseMode,
                                           constInfo.attenMaskFlag, constInfo.isRowInvalid);

    if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) {
        int64_t s1IdxStart = (info.gS1Idx + static_cast<int64_t>(wsMStart)) / constInfo.gSize;
        int64_t gIdxStart = (info.gS1Idx + static_cast<int64_t>(wsMStart)) % constInfo.gSize;
        int64_t s1IdxEnd = (info.gS1Idx + wsMStart + dealRowCount) / constInfo.gSize;
        int64_t gIdxEnd = (info.gS1Idx + wsMStart + dealRowCount) % constInfo.gSize;
        int64_t gStartIdx = 0; // 循环当前s1中的g的起点
        int64_t dealCount = 0;
        int64_t curDealRows = 0;
        for (int64_t i = s1IdxStart; i <= s1IdxEnd; i++) {
            if (i == s1IdxStart && s1IdxEnd == s1IdxStart) {
                curDealRows = gIdxEnd - gIdxStart;
            } else if (i == s1IdxStart && s1IdxEnd != s1IdxStart) {
                curDealRows = constInfo.gSize - gIdxStart;
            } else if (i == s1IdxEnd) {
                curDealRows = gIdxEnd;
            } else {
                curDealRows = constInfo.gSize;
            }
            if (i == s1IdxStart) {
                gStartIdx = gIdxStart; // 只有第一块的g的起点不是0
            } else {
                gStartIdx = 0;
            }
            if (curDealRows == 0) {
                continue;
            }
            DataCopy(tmpSinkResUbBrcb[dealCount * brcbNum], sinkBuf[gStartIdx * brcbNum], brcbNum * curDealRows);
            if (unlikely(isInvalidRows)) { // 行无效处理
                SinkInvalidRow(info, tmpSinkResUbBrcb, i, dealCount, curDealRows);
            }
            dealCount += curDealRows;
        }
    } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) {
        int64_t gIdx = 0;
        int64_t s1Idx = 0;
        for (uint32_t row = 0; row < dealRowCount; ++row) {
            gIdx = (info.gS1Idx + wsMStart + row) / info.actS1Size;
            s1Idx = (info.gS1Idx + wsMStart + row) % info.actS1Size;
            DataCopy(tmpSinkResUbBrcb[row * brcbNum], sinkBuf[gIdx * brcbNum], brcbNum);
            if (unlikely(isInvalidRows)) { // 行无效处理
                SinkInvalidRow(info, tmpSinkResUbBrcb, s1Idx, row, 1);
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Vec1SinkSoftmaxProc(const RunInfo &info,
        LocalTensor<COMPUTE_T> &tmpSinkResUbBrcb, uint32_t offset, uint32_t dealRowCountBrcb)
{
    AscendC::PipeBarrier<PIPE_V>();
    Sub(tmpSinkResUbBrcb, tmpSinkResUbBrcb, softmaxMaxUb[offset], dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(tmpSinkResUbBrcb, tmpSinkResUbBrcb, dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
    Add(softmaxSumUb[offset], softmaxSumUb[offset], tmpSinkResUbBrcb, dealRowCountBrcb);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::SinkValueNoBrc(LocalTensor<COMPUTE_T> tmpSinkResUb,
        LocalTensor<COMPUTE_T> tmpSinkResUbBrcb, uint32_t dealRowCount)
{
    // 不带brcb，需要把sink按行取最大值，RowMax后变为1*m的shape
    uint64_t repeatTimesOnce = 128;  //由于WholeReduceMax接口中repeatTimes支持范围（0,255），因此需要分多次调用WholeReduceMax，这里就使用每次repeatTime=128
    uint64_t loopTimes = (dealRowCount + repeatTimesOnce - 1) / repeatTimesOnce;
    uint64_t repeatTimes = repeatTimesOnce;

    for (uint64_t loop = 0; loop < loopTimes; ++loop) {
        if (loop == loopTimes - 1) {
            repeatTimes = dealRowCount - loop * repeatTimesOnce;
        }
        WholeReduceMax(tmpSinkResUb[loop * repeatTimesOnce], tmpSinkResUbBrcb[loop * brcbNum * repeatTimesOnce],
            brcbNum, repeatTimes, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::Vec1SinkCompute(const RunInfo &info, uint32_t idx, uint32_t wsMStart, uint32_t dealRowCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {  // sink叠加FD规约场景，在FD规约流程中处理，该处不处理
            return;
        }
    }

    LocalTensor<COMPUTE_T> tmpSinkResUb = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, 0);
    LocalTensor<COMPUTE_T> tmpSinkResUbBrcb = tmpBuff1.GetWithOffset<COMPUTE_T>(BUFFER_SIZE_BYTE_8K, BUFFER_SIZE_BYTE_8K);

    Vec1GetSinkValue(info, tmpSinkResUbBrcb, wsMStart, dealRowCount);

    uint32_t offset = idx * SOFTMAX_TMP_BUFFER_SIZE / sizeof(COMPUTE_T) + mSplitInfo.nBufferStartM / 2;
    if constexpr (SOFTMAX_WITH_BRC) {
        uint32_t dealRowCountBrcb = dealRowCount * brcbNum;
        Vec1SinkSoftmaxProc(info, tmpSinkResUbBrcb, offset, dealRowCountBrcb);
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        SinkValueNoBrc(tmpSinkResUb, tmpSinkResUbBrcb, dealRowCount);
        Vec1SinkSoftmaxProc(info, tmpSinkResUb, offset, dealRowCount);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::InitPostQuant(__gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
{
    isQuant2PerChn = constInfo.isPostQuantPerChn;
    isQuant2Bf16 = constInfo.isPostQuantTypeBf16;
    if (quantScale2 != nullptr) {
        if (isQuant2PerChn) {
            if (isQuant2Bf16) {
                postQuantProcesser.InitPerChannel(quantScale2Bf16GmTensor,
                    quantScale2, constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim);
            } else {
                postQuantProcesser.InitPerChannel(quantScale2GmTensor,
                    quantScale2, constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim);
            }
        } else {
            postQuantProcesser.InitPerTensor(scale2Value, quantScale2, isQuant2Bf16);
        }
    }

    if (quantOffset2 != nullptr) {
        isQuantOffset2Exit = true;
        if (isQuant2PerChn) {
            if (isQuant2Bf16) {
                postQuantProcesser.InitPerChannel(quantOffset2Bf16GmTensor,
                    quantOffset2, constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim);
            } else {
                postQuantProcesser.InitPerChannel(quantOffset2GmTensor,
                    quantOffset2, constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim);
            }
        } else {
            postQuantProcesser.InitPerTensor(offset2Value, quantOffset2, isQuant2Bf16);
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealPostQuantOutPerChn(
    const RunInfo &info, LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount)
{
    PostQuantInfo_V2 postQuantInfo;
    postQuantInfo.gSize = constInfo.gSize;
    postQuantInfo.dSize = constInfo.headDim;
    postQuantInfo.s1Size = info.actS1Size;
    postQuantInfo.n2Idx = info.n2Idx;
    postQuantInfo.gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    postQuantInfo.gS1DealSize = dealRowCount;
    postQuantInfo.colCount = columnCount;

    if (isQuant2Bf16) {
        uint32_t computeSize = dealRowCount * columnCount;
        LocalTensor<T> tempFp32Ub = tmpBuff1.Get<T>(computeSize);

        LocalTensor<bfloat16_t> quantScale2Ub = inputQue2.AllocTensor<bfloat16_t>();
        postQuantProcesser.CopyParamsGmToUb(quantScale2Ub, quantScale2Bf16GmTensor, postQuantInfo);
        inputQue2.EnQue(quantScale2Ub);
        inputQue2.DeQue<bfloat16_t>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tempFp32Ub, quantScale2Ub, RoundMode::CAST_NONE, computeSize);
        inputQue2.FreeTensor(quantScale2Ub);

        AscendC::PipeBarrier<PIPE_V>();
        postQuantProcesser.MulScale(bmm2ResUb, bmm2ResUb, tempFp32Ub, postQuantInfo);

        if(isQuantOffset2Exit){
            LocalTensor<bfloat16_t> quantOffset2Ub = inputQue2.AllocTensor<bfloat16_t>();
            postQuantProcesser.CopyParamsGmToUb(quantOffset2Ub, quantOffset2Bf16GmTensor, postQuantInfo);
            inputQue2.EnQue(quantOffset2Ub);
            inputQue2.DeQue<bfloat16_t>();
            AscendC::PipeBarrier<PIPE_V>();
            Cast(tempFp32Ub, quantOffset2Ub, RoundMode::CAST_NONE, computeSize);
            inputQue2.FreeTensor(quantOffset2Ub);

            AscendC::PipeBarrier<PIPE_V>();
            postQuantProcesser.AddOffset(bmm2ResUb, bmm2ResUb, tempFp32Ub, postQuantInfo);
        }
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        // 此处使用了单buffer(另一块被bmm2ResUb占用了), MTE2 bound时, 可以在调用MulScale和AddOffset之前将拷入的参数拷入tmpBuff1后提前释放buffer
        LocalTensor<T> quantScale2Ub = inputQue1.AllocTensor<T>();
        postQuantProcesser.CopyParamsGmToUb(quantScale2Ub, quantScale2GmTensor, postQuantInfo);
        inputQue1.EnQue(quantScale2Ub);
        inputQue1.DeQue<T>();
        AscendC::PipeBarrier<PIPE_V>();
        postQuantProcesser.MulScale(bmm2ResUb, bmm2ResUb, quantScale2Ub, postQuantInfo);
        inputQue1.FreeTensor(quantScale2Ub);

        if(isQuantOffset2Exit){
            LocalTensor<T> quantOffset2Ub = inputQue1.AllocTensor<T>();
            postQuantProcesser.CopyParamsGmToUb(quantOffset2Ub, quantOffset2GmTensor, postQuantInfo);
            inputQue1.EnQue(quantOffset2Ub);
            inputQue1.DeQue<T>();
            AscendC::PipeBarrier<PIPE_V>();
            postQuantProcesser.AddOffset(bmm2ResUb, bmm2ResUb, quantOffset2Ub, postQuantInfo);
            inputQue1.FreeTensor(quantOffset2Ub);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealPostQuantOutPerTensor(
    LocalTensor<MM2_OUT_T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(bmm2ResUb, bmm2ResUb, scale2Value, dealRowCount * columnCount);
    AscendC::PipeBarrier<PIPE_V>();
    if (isQuantOffset2Exit) {
        Adds(bmm2ResUb, bmm2ResUb, offset2Value, dealRowCount * columnCount);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecNonQuant<FIAT>::DealZeroActSeqLenWithPostQuant(uint32_t bIdx, uint32_t n2Idx)
{
    if (!isQuantOffset2Exit) {
        return;
    }
    // 兼容性考虑: actual_seq_lens为0场景, PFA输出0; IFA场景attentionInt8 = attention * quantScale + quantOffset
    if (constInfo.qSeqSize != 1) {
        return;
    }
    // query的actual_seq_len为0时没有输出，不需要处理
    uint64_t actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
    if (actSeqLensQ == 0) {
        return;
    }

    uint32_t gSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDimAlign;
    if (gSplitSize > constInfo.gSize) {
        gSplitSize = constInfo.gSize;
    }
    uint32_t loopCount = (constInfo.gSize + gSplitSize - 1) / gSplitSize;
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0, dealSize = gSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        uint32_t startRow = gSplitSize * i;
        uint32_t dealRowCount = dealSize;
        uint32_t columnCount = constInfo.headDimAlign;
        uint32_t actualColumnCount = constInfo.headDim;

        // 拷入quantOffset2
        LocalTensor<T> quantOffset2Ub = inputQue1.AllocTensor<T>();
        if (isQuant2PerChn) {
            PostQuantInfo_V2 postQuantInfo;
            postQuantInfo.gSize = constInfo.gSize;
            postQuantInfo.dSize = constInfo.headDim;
            postQuantInfo.s1Size = 1;
            postQuantInfo.n2Idx = n2Idx;
            postQuantInfo.gS1Idx = startRow;
            postQuantInfo.gS1DealSize = dealRowCount;
            postQuantInfo.colCount = columnCount;

            if (isQuant2Bf16) {
                LocalTensor<bfloat16_t> quantOffset2Bf16Ub =
                    quantOffset2Ub[BASE_BLOCK_MAX_ELEMENT_NUM / 2].template ReinterpretCast<bfloat16_t>();
                postQuantProcesser.CopyParamsGmToUb(quantOffset2Bf16Ub, quantOffset2Bf16GmTensor, postQuantInfo);
                inputQue1.EnQue(quantOffset2Ub);
                inputQue1.DeQue<T>();
                Cast(quantOffset2Ub, quantOffset2Bf16Ub, RoundMode::CAST_NONE, dealRowCount * columnCount);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                postQuantProcesser.CopyParamsGmToUb(quantOffset2Ub, quantOffset2GmTensor, postQuantInfo);
                inputQue1.EnQue(quantOffset2Ub);
                inputQue1.DeQue<T>();
            }
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(quantOffset2Ub, offset2Value, quantOffset2Ub.GetSize());
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Cast为INT8
        LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
        LocalTensor<half> quant2ResHalf = tmpBmm2ResCastTensor.template ReinterpretCast<half>();
        Cast(quant2ResHalf, quantOffset2Ub, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
        inputQue1.FreeTensor(quantOffset2Ub);

        AscendC::PipeBarrier<PIPE_V>();
        Cast(tmpBmm2ResCastTensor, quant2ResHalf, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2ResCastTensor);

        // 拷出
        outputQue1.DeQue<OUT_T>();
        {
            FaUbTensor<OUT_T> ubTensor {
                .tensor = tmpBmm2ResCastTensor,
                .rowCount = dealRowCount,
                .colCount = columnCount,
            };
            GmCoord gmCoord {
                .bIdx = bIdx,
                .n2Idx = n2Idx,
                .gS1Idx = startRow,
                .dIdx = 0,
                .gS1DealSize = dealRowCount,
                .dDealSize = (uint32_t)constInfo.headDim
            };
            CopyAttentionOut(ubTensor, gmCoord);
        }
        outputQue1.FreeTensor(tmpBmm2ResCastTensor);
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
}

#endif
