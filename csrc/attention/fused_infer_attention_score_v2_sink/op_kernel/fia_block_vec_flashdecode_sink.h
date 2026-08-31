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
 * \file fia_block_vec_flashdecode_sink.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FLASHDECODE_SINK_H
#define FIA_BLOCK_VEC_FLASHDECODE_SINK_H

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
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <typename FIAT> class FiaBlockVecFlashDecode {
  public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename FIAT::outputType;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;

    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm,
                                            GlobalTensor<T> lseSumFdGm,
                                            GlobalTensor<T> accumOutGm,
                                            GlobalTensor<OUT_T> attentionOutGm,
                                            GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                            GlobalTensor<uint64_t> actualSeqLengthsGm);
    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm);
    __aicore__ inline void InitParams(const AiInfraInferenceAttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FlashDecode(FDparams &fd);

  protected:
    __aicore__ inline void
    CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM);
    __aicore__ inline void
    ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t startRow, uint32_t dealRowCount, uint32_t cntM);
    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb,
                                                uint32_t startRow,
                                                uint32_t dealRowCount,
                                                uint32_t columnCount);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset,
                                           LocalTensor<OUT_T> &attenOutUb,
                                           uint32_t startRow,
                                           uint32_t dealRowCount,
                                           uint32_t columnCount,
                                           uint32_t actualColumnCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut,
                                          LocalTensor<T> &mm2Res,
                                          LocalTensor<T> &lseLocal,
                                          uint32_t cntKV,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CalaPreNextTokens();

  private:
    // =================================常量区=================================
    static constexpr uint64_t SYNC_LSE_SUM_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_LSE_SUM_BUF2_FLAG = 7;
    static constexpr uint64_t SYNC_LSE_MAX_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 7;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T); // 32/4=8

  protected:
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    // =======================获取实际Act_S，用于行无效处理===========================
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    AiInfraInferenceAttentionCommon::ConstInfo constInfo{};
    TaskInfo taskInfo{};

  private:
    // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1; // 1.5k: 16*24*4
    TBuf<> fdSumBuf2; // 1.5k: 16*24*4
    TBuf<> fdMaxBuf1; // 1.5k: 16*24*4
    TBuf<> fdMaxBuf2; // 1.5k: 16*24*4
    TBuf<> fdLseExpBuf; // 1.5k: 16*24*4
    TBuf<> fdMm2ResBuf1; // 32k: 16*512*4
    TBuf<> fdMm2ResBuf2; // 32k: 16*512*4
    TBuf<> fdReduceBuf; // 32k: 16*512*4
    TBuf<> fdOutputBuf; // 32k: 16*512*4

    TBuf<> fdLseMaxUbBuf; // 64B: 16*4
    TBuf<> fdLseSumUbBuf; // 64B: 16*4
    TBuf<> fdLseUbBuf; // 64B: 16*4
};

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::InitGlobalTensor(GlobalTensor<T> lseMaxFdGm,
                                                                      GlobalTensor<T> lseSumFdGm,
                                                                      GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm,
                                                                      GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                                                      GlobalTensor<uint64_t> actualSeqLengthsGm)
{
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->accumOutGm = accumOutGm;
    this->attentionOutGm = attentionOutGm;
    this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
    this->actualSeqLengthsGm = actualSeqLengthsGm;

    qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
    kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
{
    this->softmaxLseGm = softmaxLseGm;
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::InitParams(
    const AiInfraInferenceAttentionCommon::ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT> __aicore__ inline void FiaBlockVecFlashDecode<FIAT>::InitDecodeParams()
{
    this->blockIdx = GetBlockIdx();
}

template <typename FIAT> __aicore__ inline void FiaBlockVecFlashDecode<FIAT>::InitBuffers(TPipe *pipe)
{
    if ASCEND_IS_AIV {
        pipe->Reset();
        pipe->InitBuffer(fdSumBuf1,
                         AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K +
                             AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdSumBuf2,
                         AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K +
                             AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf1,
                         AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K +
                             AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf2,
                         AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K +
                             AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdLseExpBuf,
                         AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K +
                             AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMm2ResBuf1, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdMm2ResBuf2, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdReduceBuf, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdOutputBuf, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdLseMaxUbBuf, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseUbBuf, AiInfraInferenceAttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
    }
}

template <typename FIAT> __aicore__ inline void FiaBlockVecFlashDecode<FIAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecFlashDecode<FIAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::CopyAccumOutIn(LocalTensor<T> &accumOutLocal,
                                                                    uint32_t splitKVIndex,
                                                                    uint32_t startRow,
                                                                    uint32_t dealRowCount)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (constInfo.headDimAlign - constInfo.headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (constInfo.headDimAlign - constInfo.headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;
    uint64_t combineAccumOutOffset = startRow * constInfo.headDim + // taskoffset + g轴offset
                                     splitKVIndex * constInfo.mBaseSize * constInfo.headDim; // 份数offset

    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);

    uint64_t combineLseOffset = (baseOffset + startRow) * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t combineLoopOffset = constInfo.mBaseSize * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign],
                 lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(
            lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset], dealRowCountAlign);
    }

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_SUM_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_BUF1_FLAG + cntM % 2);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::ComputeScaleValue(LocalTensor<T> &lseExp,
                                                                       uint32_t startRow,
                                                                       uint32_t dealRowCount,
                                                                       uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    LocalTensor<T> lseMaxUb = fdLseMaxUbBuf.Get<T>();
    LocalTensor<T> lseSumUb = fdLseSumUbBuf.Get<T>();
    uint64_t dealRowCountAlign = dealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    Duplicate(lseMaxUb, -AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_MAX, dealRowCountAlign);
    Duplicate(lseSumUb, AiInfraInferenceAttentionCommon::ConstInfo::FLOAT_ZERO, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    AiInfraInferenceCommonFaBaseVector::ColMax(
        lseMaxUb, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    AiInfraInferenceCommonFaBaseVector::RowSub(
        lseExp, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Exp(lseExp, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Mul(lseExp, lseSum, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    AiInfraInferenceCommonFaBaseVector::ColAdd(
        lseSumUb, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    AiInfraInferenceCommonFaBaseVector::MatDivsVec(
        lseExp, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb,
                                                                          uint32_t startRow,
                                                                          uint32_t dealRowCount,
                                                                          uint32_t columnCount)
{
    FaUbTensor<OUT_T> ubTensor{
        .tensor = attenOutUb,
        .rowCount = dealRowCount,
        .colCount = columnCount,
    };
    GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                    .n2Idx = taskInfo.n2Idx,
                    .gS1Idx = taskInfo.gS1Idx + startRow,
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
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::Bmm2DataCopyOut(uint64_t attenOutOffset,
                                                                     LocalTensor<OUT_T> &attenOutUb,
                                                                     uint32_t startRow,
                                                                     uint32_t dealRowCount,
                                                                     uint32_t columnCount,
                                                                     uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride =
        (columnCount - actualColumnCount) / (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::ReduceFinalRes(
    LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal, uint32_t cntKV, uint32_t dealRowCount)
{
    uint32_t dealRowCountAlign = dealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<T> tmpRst =
        cntKV == 0 ? reduceOut : mm2Res; // 第一次mul结果直接写入reduceOut，否则在mm2Res原地进行mul，再加到reduceOut

    AiInfraInferenceCommonFaBaseVector::RowMuls(
        tmpRst, mm2Res, lseLocal[cntKV * dealRowCountAlign], dealRowCount, constInfo.headDimAlign, constInfo.headDim);

    if (cntKV != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        Add(reduceOut, reduceOut, tmpRst, dealRowCount * constInfo.headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT>
__aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * constInfo.headDimAlign);
    } else {
        Cast(
            tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * constInfo.headDimAlign);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, constInfo.headDimAlign);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename FIAT> __aicore__ inline void FiaBlockVecFlashDecode<FIAT>::CalaPreNextTokens()
{
    actSeqLensQ = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
    actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);

    if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::BAND) {
        preTokensPerBatch = constInfo.preToken;
        nextTokensPerBatch =
            static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(actSeqLensQ) + constInfo.nextToken;
        if (constInfo.sinkNumber > 0) {
            nextTokensPerBatch = static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(constInfo.sinkNumber) -
                                 static_cast<int32_t>(actSeqLensQ) + static_cast<int32_t>(constInfo.nextToken);
        }
    } else if ((constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::DEFAULT_MASK) && constInfo.attenMaskFlag) {
        nextTokensPerBatch = constInfo.nextToken;
        preTokensPerBatch = static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(actSeqLensQ) + constInfo.preToken;
    } else {
        nextTokensPerBatch = static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(actSeqLensQ);
        if (constInfo.sinkNumber > 0) {
            if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::RIGHT_DOWN_CAUSAL &&
                static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(constInfo.sinkNumber) <
                    static_cast<int32_t>(actSeqLensQ)) {
                nextTokensPerBatch = static_cast<int32_t>(actSeqLensKv) - static_cast<int32_t>(constInfo.sinkNumber) -
                                     static_cast<int32_t>(actSeqLensQ);
            } else if (constInfo.sparseMode == AiInfraInferenceCommonFaBaseVector::LEFT_UP_CAUSAL) {
                nextTokensPerBatch = constInfo.nextToken;
            }
        }
    }
}
template <typename FIAT> __aicore__ inline void FiaBlockVecFlashDecode<FIAT>::FlashDecode(FDparams &fd)
{
    if (blockIdx >= fd.usedVecNumOfFd) {
        return;
    }
    uint32_t fdTaskPrevEnd = (blockIdx > 0) ? fd.gS1IdxEndOfFdHead[blockIdx - 1] : 0; // 上一个核末尾是第几个规约
    uint32_t fdS1gOuterMPrevEnd =
        (blockIdx > 0) ? fd.gS1IdxEndOfFdHeadSplit[blockIdx - 1] : 0; // 上一个核末尾是该规约的第几个base行
    uint32_t fdTaskEnd = fd.gS1IdxEndOfFdHead[blockIdx];                 // 当前核的末尾是第几个规约任务
    uint32_t fdS1gOuterMEnd = fd.gS1IdxEndOfFdHeadSplit[blockIdx]; // 当前核的末尾是该规约的第几个base行
    uint32_t tmpFdS1gOuterMStart = (blockIdx > 0) ? fdS1gOuterMPrevEnd + 1 : 0; // 当前核从第几个base行开始
    uint32_t tmpFdS1gOuterMEnd = 0;
    uint32_t reduceGlobaLoop = 0;
    uint32_t reduceMLoop = 0;

    for (uint32_t fdTaskId = fdTaskPrevEnd; fdTaskId <= fdTaskEnd; fdTaskId++) {
        tmpFdS1gOuterMEnd = (fdTaskId == fdTaskEnd) ? fdS1gOuterMEnd : (fd.gS1SplitNumOfFdHead[fdTaskId] - 1);
        taskInfo.bIdx = fd.bN2IdxOfFdHead[fdTaskId] / constInfo.kvHeadNum;
        taskInfo.n2Idx = fd.bN2IdxOfFdHead[fdTaskId] % constInfo.kvHeadNum;
        taskInfo.gS1Idx = fd.gS1IdxOfFdHead[fdTaskId] * constInfo.mBaseSize;
        taskInfo.actualCombineLoopSize = fd.s2SplitNumOfFdHead[fdTaskId]; // 当前规约任务kv方向有几份

        uint64_t combineTaskPrefixSum = 0;
        for (int i = 0; i < fdTaskId; i++) {
            // 计算此前规约数据的累计份数，每一份的数据大小为 kvHeadNum * constInfo.tndSgBasicSize
            // |Task0-0|Task0-1|Task0-3|Task1-0|Task1-2|...|
            combineTaskPrefixSum += fd.s2SplitNumOfFdHead[i];
        }

        uint64_t taskOffset = combineTaskPrefixSum * constInfo.mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭

            uint32_t actualGSplitSize = fd.gS1BaseSizeOfFd;
            if (fdS1gOuterMIdx == fd.gS1SplitNumOfFdHead[fdTaskId] - 1) {
                actualGSplitSize = fd.gS1LastPartSizeOfFdHead[fdTaskId];
            }
            uint32_t startRow = fdS1gOuterMIdx * fd.gS1BaseSizeOfFd;

            LocalTensor<T> lseExp = fdLseExpBuf.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf.Get<T>();
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);

            LocalTensor<T> mm2Res;
            for (uint32_t preLoadIdx = 0; preLoadIdx < constInfo.preLoadNum; preLoadIdx++) {
                mm2Res = (reduceGlobaLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop + preLoadIdx) % 2);
            }

            ComputeScaleValue(lseExp, startRow, actualGSplitSize, reduceMLoop);
            CalaPreNextTokens();
            //****************************************************************************************************** */
            if (constInfo.softmaxLseFlag) {
                // 新增 计算规约后的max和sum
                LocalTensor<T> lseMaxUb = fdLseMaxUbBuf.Get<T>();
                LocalTensor<T> lseSumUb = fdLseSumUbBuf.Get<T>();
                // 新增 最终LSE的计算lse = log(sum) + max

                WaitFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
                LocalTensor<T> lseftMaxLseUb = fdLseUbBuf.Get<T>();
                AiInfraInferenceCommonFaBaseVector::ComputeSoftMaxLse(
                    lseftMaxLseUb, lseSumUb, lseMaxUb, actualGSplitSize);
                // 判断是否行无效
                bool isInValidRowsFlag = AiInfraInferenceCommonFaBaseVector::IsExistInvalidRows(nextTokensPerBatch,
                                                                            preTokensPerBatch,
                                                                            constInfo.sparseMode,
                                                                            constInfo.attenMaskFlag,
                                                                            constInfo.isRowInvalid);
                if (isInValidRowsFlag) {
                    SoftMaxShapeInfo softmaxShapeInfo{
                        static_cast<uint32_t>(actualGSplitSize),
                        static_cast<uint32_t>(AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM),
                        static_cast<uint32_t>(actualGSplitSize),
                        static_cast<uint32_t>(AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM)
                    };
                    AdjustSoftMaxRes<T, T>(lseftMaxLseUb, lseMaxUb, negativeIntScalar, (T)3e+99, softmaxShapeInfo);
                }
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if (LAYOUT_T == FIA_LAYOUT::TND) {
                    uint32_t prefixBS1 = taskInfo.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
                    uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseTND(softmaxLseGm, lseftMaxLseUb, bN2Offset, mOffset, actualGSplitSize, constInfo);
                } else if (LAYOUT_T == FIA_LAYOUT::NTD) {
                    uint32_t prefixBS1 = taskInfo.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
                    uint32_t s1Size = taskInfo.bIdx == 0U ? actualSeqLengthsGmQ.GetValue(0U)
                                                          : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx) -
                                                                actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1U);
                    uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseNTD(
                        softmaxLseGm, lseftMaxLseUb, bN2Offset, mOffset, actualGSplitSize, constInfo, s1Size);
                } else if (LAYOUT_T == FIA_LAYOUT::BSND || LAYOUT_T == FIA_LAYOUT::BSH) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                    DataCopySoftmaxLseBSND(softmaxLseGm,
                                           lseftMaxLseUb,
                                           bN2Offset,
                                           mOffset,
                                           actualGSplitSize,
                                           constInfo,
                                           qActSeqLensParser,
                                           taskInfo.bIdx);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                    DataCopySoftmaxLseBNSD<T, Q_MODE>(softmaxLseGm,
                                                      lseftMaxLseUb,
                                                      bN2Offset,
                                                      mOffset,
                                                      actualGSplitSize,
                                                      constInfo,
                                                      qActSeqLensParser,
                                                      taskInfo.bIdx);
                }
                SetFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_SUM_BUF1_FLAG + reduceMLoop % 2);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_BUF1_FLAG + reduceMLoop % 2);
            //****************************************************************************************************** */

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                mm2Res = reduceGlobaLoop % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                if (i >= constInfo.preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                }

                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobaLoop % 2);
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize);
            reduceMLoop += 1;
        }
        tmpFdS1gOuterMStart = 0;
    }
}
#endif