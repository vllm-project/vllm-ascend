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
 * \file fia_block_vec_flashdecode.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FLASHDECODE_H
#define FIA_BLOCK_VEC_FLASHDECODE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../vector_common.h"
#include "../memory_copy.h"
#include "../post_quant.h"
#include "../const_def.h"
using namespace AttentionCommon;
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <typename FIAT> 
class FiaBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename FIAT::outputType;  
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout; 
    using SINK_T = bfloat16_t;
    static constexpr GmFormat PostQuant_FORMAT = GmFormat::NGD;

    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm, GlobalTensor<T> accumOutGm, 
         GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<uint64_t> actualSeqLengthsGmQ, GlobalTensor<uint64_t> actualSeqLengthsGm,
         __gm__ uint8_t *key, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm);
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink);
    __aicore__ inline void InitParams(const AttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();   
    __aicore__ inline void FlashDecode(FDparams &fd);
protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount);                            
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t cntM);
    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                 uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal, 
                                          uint32_t cntKV, uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM);
    __aicore__ inline void CalcPreNextTokens();
    __aicore__ inline void CopySinkIn(uint32_t cntM);
    __aicore__ inline void SinkMax(LocalTensor<T> lseMaxUb, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void SinkExpSumUpdate(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint64_t dealRowCountAlign);
    __aicore__ inline void DealInvalidRows(LocalTensor<T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount);
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                               uint32_t columnCount, uint32_t cntM);

    __aicore__ inline void InitPostQuant(__gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline void DealPostQuantOutPerChn(LocalTensor<T> &bmm2ResUb, uint32_t startRow, 
                                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM);
    __aicore__ inline void DealPostQuantOutPerTensor(LocalTensor<T> &bmm2ResUb, uint32_t startRow, 
                                                    uint32_t dealRowCount, uint32_t columnCount);

private:
// =================================常量区=================================
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 7;
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 12;
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 13;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = AttentionCommon::BYTE_BLOCK / sizeof(T); // 32/4=8

protected:
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<SINK_T> sinkGm;

    //postquant
    FaGmTensor<T, PostQuant_FORMAT> quantScale2GmTensor;
    FaGmTensor<T, PostQuant_FORMAT> quantOffset2GmTensor;
    FaGmTensor<bfloat16_t, PostQuant_FORMAT> quantScale2Bf16GmTensor;
    FaGmTensor<bfloat16_t, PostQuant_FORMAT> quantOffset2Bf16GmTensor;
    PostQuant<GetOutUbFormat<LAYOUT_T>()> postQuantProcesser;

    // =======================获取实际Act_S，用于行无效处理===========================
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_T>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_T, PAGE_ATTENTION>();
    // tensorlist
    __gm__ uint8_t *keyPtr = nullptr;
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    
    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;
    
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    bool learnableSinkFlag = false;

    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    T scale2Value = 0;
    T offset2Value = 0;
    bool isQuantOffset2Exit = false;
    bool isQuant2PerChn = false;
    bool isQuant2Bf16 = false;
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    AttentionCommon::ConstInfo constInfo{};
    TaskInfo taskInfo{};
private:    
 // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1;
    TBuf<> fdSumBuf2;
    TBuf<> fdMaxBuf1;
    TBuf<> fdMaxBuf2;
    TBuf<> fdLseExpBuf;
    TBuf<> fdMm2ResBuf1;
    TBuf<> fdMm2ResBuf2;
    TBuf<> fdReduceBuf;
    TBuf<> fdOutputBuf;
    TBuf<> fdSinkCopyInBuf;
    TBuf<> fdSinkValueBuf;
    TBuf<> fdSinkExpBuf;
    TBuf<> fdSinkTmpBuf;

    TBuf<> fdLseMaxUbBuf1;
    TBuf<> fdLseMaxUbBuf2;
    TBuf<> fdLseSumUbBuf1;
    TBuf<> fdLseSumUbBuf2;
    TBuf<> quant2TmpBuf1;
    TBuf<> quant2TmpBuf2;
    TBuf<> fdLseUbBuf;
};

template <typename FIAT> 
class FiaBlockVecFlashDecodeDummy {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using OUT_T = typename FIAT::outputType;  
    using SINK_T = bfloat16_t;

    __aicore__ inline void InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm, GlobalTensor<T> accumOutGm, 
         GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<uint64_t> actualSeqLengthsGmQ, GlobalTensor<uint64_t> actualSeqLengthsGm,
         __gm__ uint8_t *key, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2);
    __aicore__ inline void InitParams(const AttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm);
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink);
    __aicore__ inline void FlashDecode(FDparams &fd);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitDecodeParams();
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
};

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitGlobalTensor(GlobalTensor<T> lseMaxFdGm, 
                                                        GlobalTensor<T> lseSumFdGm, 
                                                        GlobalTensor<T> accumOutGm,
                                                        GlobalTensor<OUT_T> attentionOutGm,
                                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                                        GlobalTensor<uint64_t> actualSeqLengthsGm,
                                                        __gm__ uint8_t *key,
                                                        __gm__ uint8_t *quantScale2,
                                                        __gm__ uint8_t *quantOffset2)
{
   this->lseMaxFdGm = lseMaxFdGm;
   this->lseSumFdGm = lseSumFdGm;
   this->accumOutGm = accumOutGm;
   this->attentionOutGm = attentionOutGm;
   this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
   this->actualSeqLengthsGm = actualSeqLengthsGm;
   this->keyPtr = key;

   qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.qSeqSize);
   kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualLenDims, constInfo.kvSeqSize);

    if constexpr (POST_QUANT) {
        InitPostQuant(quantScale2, quantOffset2);
    }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
{
   this->softmaxLseGm = softmaxLseGm;
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
{
    learnableSinkFlag = true;
    this->sinkGm = learnableSink;
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitParams(const AttentionCommon::ConstInfo &constInfo)
{
   this->constInfo = constInfo;
}


template <typename FIAT>__aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitDecodeParams()
{
    this->blockIdx = GetBlockIdx();
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::InitBuffers(TPipe *pipe)
{
    if ASCEND_IS_AIV {
        pipe->Reset();
        // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
        pipe->InitBuffer(fdSumBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdSumBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(fdMaxBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        // TmpBuf
        pipe->InitBuffer(fdLseExpBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_4K + AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        // InQue, DB, SYNC_MM2RES_BUF1_FLAG SYNC_MM2RES_BUF2_FLAG
        pipe->InitBuffer(fdMm2ResBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(fdMm2ResBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        // TmpBuf
        pipe->InitBuffer(fdReduceBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        // OutQue, SYNC_FDOUTPUT_BUF_FLAG
        pipe->InitBuffer(fdOutputBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        // TmpBuf, UB开DB
        pipe->InitBuffer(fdLseMaxUbBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseMaxUbBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        pipe->InitBuffer(fdLseSumUbBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
        // OutQue, SYNC_LSEOUTPUT_BUF_FLAG
        pipe->InitBuffer(fdLseUbBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);

        // TmpBuf
        pipe->InitBuffer(quant2TmpBuf1, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(quant2TmpBuf2, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K);
        if (unlikely(learnableSinkFlag)) {
            // InQue, DB, SYNC_SINK_BUF1_FLAG SYNC_SINK_BUF2_FLAG
            pipe->InitBuffer(fdSinkCopyInBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
            // TmpBuf
            pipe->InitBuffer(fdSinkValueBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdSinkExpBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(fdSinkTmpBuf, AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_2K);
        }
     }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF2_FLAG);
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF2_FLAG);
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex,
    uint32_t startRow, uint32_t dealRowCount)
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
    uint64_t combineAccumOutOffset = startRow * constInfo.headDim +                          // taskoffset + g轴offset
                                      splitKVIndex * constInfo.mBaseSize * constInfo.headDim; // 份数offset

    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopyLseIn(uint32_t startRow,
    uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    uint64_t combineLseOffset = (baseOffset + startRow) * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    uint64_t combineLoopOffset = constInfo.mBaseSize * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    uint64_t dealRowCountAlign = dealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
        DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign); // 份数offset
        DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                 dealRowCountAlign);
    }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopySinkIn(uint32_t cntM)
{
    LocalTensor<SINK_T> sinkCopyInBuf = fdSinkCopyInBuf.GetWithOffset<SINK_T>(BUFFER_SIZE_BYTE_1K, (cntM % 2) * BUFFER_SIZE_BYTE_1K);

    uint64_t sinkGmOffset = taskInfo.n2Idx * constInfo.gSize;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = constInfo.gSize * sizeof(SINK_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;
    DataCopyPadExtParams<SINK_T> sinkPadParams;
    sinkPadParams.isPad = true;
    sinkPadParams.paddingValue = static_cast<SINK_T>(0);

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + cntM % 2);
    DataCopyPad(sinkCopyInBuf, sinkGm[sinkGmOffset], sinkCopyParams, sinkPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + cntM % 2);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + cntM % 2);

    LocalTensor<T> tmpSinkCastBuf = fdSinkTmpBuf.Get<T>();
    Cast(tmpSinkCastBuf, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo.gSize);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + cntM % 2);

    LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.Get<T>();
    Brcb(sinkBrcbBuf, tmpSinkCastBuf, (constInfo.gSize + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM, {1, BLOCK_ELEMENT_NUM});
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::SinkMax(LocalTensor<T> lseMaxUb, uint32_t startRow, uint32_t dealRowCount)
{
    constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    int64_t gIdx = 0;
    LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.Get<T>();

    for (int64_t row = 0; row < dealRowCount; ++row) {
        if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { //内存按照S1G排布
            gIdx = (taskInfo.gS1Idx + startRow + row) % constInfo.gSize;
        } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { //内存按照GS1排布
            int64_t actS1Size = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
            gIdx = (taskInfo.gS1Idx + startRow + row) / actS1Size;
        }
        DataCopy(lseMaxUb[row * BLOCK_ELEMENT_NUM], sinkBrcbBuf[gIdx * BLOCK_ELEMENT_NUM], BLOCK_ELEMENT_NUM);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // lseMaxUb是基于S1G/GS1排布原始sink value值，copy到sinkExpBuf中，后面计算sink的exp使用
    LocalTensor<T> sinkExpBuf = fdSinkExpBuf.Get<T>();
    DataCopy(sinkExpBuf, lseMaxUb, dealRowCount * BLOCK_ELEMENT_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::SinkExpSumUpdate(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint64_t dealRowCountAlign)
{
    LocalTensor<T> sinkExpBuf = fdSinkExpBuf.Get<T>();
    Sub(sinkExpBuf, sinkExpBuf, lseMaxUb, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(sinkExpBuf, sinkExpBuf, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
    Add(lseSumUb, lseSumUb, sinkExpBuf, dealRowCountAlign);  // 计算分母，累计sink exp
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::ComputeScaleValue(LocalTensor<T> &lseExp, 
                                                    uint32_t startRow,
                                                    uint32_t dealRowCount, 
                                                    uint32_t cntM)
{
    LocalTensor<T> lseSum = cntM % 2 == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
    LocalTensor<T> lseMax = cntM % 2 == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

    // 开双buff
    LocalTensor<T> lseMaxUb = cntM % 2 == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();
    LocalTensor<T> lseSumUb = cntM % 2 == 0 ? fdLseSumUbBuf1.Get<T>() : fdLseSumUbBuf2.Get<T>();
    uint64_t dealRowCountAlign = dealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;

    if (unlikely(learnableSinkFlag)) {
        SinkMax(lseMaxUb, startRow, dealRowCount);
    } else {
        Duplicate(lseMaxUb, -AttentionCommon::ConstInfo::FLOAT_MAX, dealRowCountAlign);
    }
    Duplicate(lseSumUb, AttentionCommon::ConstInfo::FLOAT_ZERO, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::ColMax(lseMaxUb, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::RowSub(lseExp, lseMax, lseMaxUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Exp(lseExp, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    Mul(lseExp, lseSum, lseExp, taskInfo.actualCombineLoopSize * dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    fa_base_vector::ColAdd(lseSumUb, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();

    if (unlikely(learnableSinkFlag)) {
        SinkExpSumUpdate(lseMaxUb, lseSumUb, dealRowCountAlign);
    }

    fa_base_vector::MatDivsVec(lseExp, lseExp, lseSumUb, taskInfo.actualCombineLoopSize, dealRowCountAlign, dealRowCountAlign);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::Bmm2DataCopyOutTrans(LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                                                      uint32_t dealRowCount, uint32_t columnCount)
{
    FaUbTensor<OUT_T> ubTensor {.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = columnCount};
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
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize,
                                          constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize,
                                          constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims, constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGBSD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize,
                                          constInfo.headDim, actualSeqLengthsGmQ, constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
        constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim, actualSeqLengthsGmQ,
                                          constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
        constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
        FaGmTensor<OUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        outGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDim, actualSeqLengthsGmQ,
                                          constInfo.actualLenQDims);
        CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                   uint32_t startRow, uint32_t dealRowCount,
                                                   uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename FIAT>__aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::ReduceFinalRes(LocalTensor<T> &reduceOut, 
                                                      LocalTensor<T> &mm2Res, 
                                                      LocalTensor<T> &lseLocal, 
                                                      uint32_t cntKV, 
                                                      uint32_t dealRowCount)
{
    uint32_t dealRowCountAlign = dealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<T> tmpRes =
        cntKV == 0 ? reduceOut : mm2Res; // 第一次mul结果直接写入reduceOut，否则在mm2Res原地进行mul，再加到reduceOut

    fa_base_vector::RowMuls(tmpRes, mm2Res, lseLocal[cntKV * dealRowCountAlign], dealRowCount, constInfo.headDimAlign, constInfo.headDim);

    if (cntKV != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        Add(reduceOut, reduceOut, tmpRes, dealRowCount * constInfo.headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT> __aicore__ inline 
void FiaBlockVecFlashDecode<FIAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal, 
                                                       uint32_t startRow,
                                                       uint32_t dealRowCount,
                                                       uint32_t cntM)
{
    DealInvalidRows(accumOutLocal, startRow, dealRowCount, constInfo.headDimAlign);
    DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, constInfo.headDimAlign, cntM);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (POST_QUANT) {
        if (isQuant2PerChn) {
            DealPostQuantOutPerChn(accumOutLocal, startRow, dealRowCount, constInfo.headDimAlign, cntM);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            DealPostQuantOutPerTensor(accumOutLocal, startRow, dealRowCount, constInfo.headDimAlign);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    if constexpr (POST_QUANT) {
        LocalTensor<half> quant2ResHalf = quant2TmpBuf1.Get<half>();
        Cast(quant2ResHalf, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * constInfo.headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tmpBmm2ResCastTensor, quant2ResHalf, AscendC::RoundMode::CAST_ROUND, dealRowCount * constInfo.headDimAlign);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.headDim};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * constInfo.headDimAlign);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * constInfo.headDimAlign);
        }
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, constInfo.headDimAlign);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
}

template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::CalcPreNextTokens()
{
    actSeqLensQ = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
    if (constInfo.actualLenDims == 0 && !constInfo.batchContinuous) {
        actSeqLensKv = fa_base_kernel::SeqLenFromTensorList<LAYOUT_T>(keyPtr, taskInfo.bIdx);
    } else {
        actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
    }
    actSeqLensKv += constInfo.systemPrefixLen;
    int64_t safePreToken = constInfo.preToken;
    int64_t safeNextToken = constInfo.nextToken;

    fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

    if (constInfo.sparseMode == fa_base_vector::BAND) {
        preTokensPerBatch = safePreToken;
        nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
    } else if ((constInfo.sparseMode == fa_base_vector::DEFAULT_MASK) && constInfo.attenMaskFlag) {
        nextTokensPerBatch = safeNextToken;
        preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
    } else {
        nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
        preTokensPerBatch = 0;
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::InitPostQuant(__gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
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
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::DealPostQuantOutPerChn(LocalTensor<T> &bmm2ResUb, uint32_t startRow, 
                                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
{
    PostQuantInfo_V2 postQuantInfo;
    postQuantInfo.gSize = constInfo.gSize;
    postQuantInfo.dSize = constInfo.headDim;
    postQuantInfo.s1Size = actSeqLensQ;
    postQuantInfo.n2Idx = taskInfo.n2Idx;
    postQuantInfo.gS1Idx = taskInfo.gS1Idx + startRow;
    postQuantInfo.gS1DealSize = dealRowCount;
    postQuantInfo.colCount = columnCount;

    if (isQuant2Bf16) {
        uint32_t computeSize = dealRowCount * columnCount;
        LocalTensor<T> tempFp32Ub = quant2TmpBuf2.Get<T>(computeSize);

        LocalTensor<bfloat16_t> quantScale2Ub = (cntM % 2 == 0) ? fdMm2ResBuf1.Get<bfloat16_t>() : fdMm2ResBuf2.Get<bfloat16_t>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        postQuantProcesser.CopyParamsGmToUb(quantScale2Ub, quantScale2Bf16GmTensor, postQuantInfo);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(tempFp32Ub, quantScale2Ub, RoundMode::CAST_NONE, computeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);

        AscendC::PipeBarrier<PIPE_V>();
        postQuantProcesser.MulScale(bmm2ResUb, bmm2ResUb, tempFp32Ub, postQuantInfo);

        if(isQuantOffset2Exit){
            LocalTensor<bfloat16_t> quantOffset2Ub = (cntM % 2 == 0) ? fdMm2ResBuf1.Get<bfloat16_t>() : fdMm2ResBuf2.Get<bfloat16_t>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
            postQuantProcesser.CopyParamsGmToUb(quantOffset2Ub, quantOffset2Bf16GmTensor, postQuantInfo);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(tempFp32Ub, quantOffset2Ub, RoundMode::CAST_NONE, computeSize);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);

            AscendC::PipeBarrier<PIPE_V>();
            postQuantProcesser.AddOffset(bmm2ResUb, bmm2ResUb, tempFp32Ub, postQuantInfo);
        }
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<T> quantScale2Ub = (cntM % 2 == 0) ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        postQuantProcesser.CopyParamsGmToUb(quantScale2Ub, quantScale2GmTensor, postQuantInfo);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        AscendC::PipeBarrier<PIPE_V>();
        postQuantProcesser.MulScale(bmm2ResUb, bmm2ResUb, quantScale2Ub, postQuantInfo);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);

        if(isQuantOffset2Exit){
            LocalTensor<T> quantOffset2Ub = (cntM % 2 == 0) ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
            postQuantProcesser.CopyParamsGmToUb(quantOffset2Ub, quantOffset2GmTensor, postQuantInfo);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
            AscendC::PipeBarrier<PIPE_V>();
            postQuantProcesser.AddOffset(bmm2ResUb, bmm2ResUb, quantOffset2Ub, postQuantInfo);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + cntM % 2);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockVecFlashDecode<FIAT>::DealPostQuantOutPerTensor(LocalTensor<T> &bmm2ResUb, uint32_t startRow, 
                                                    uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(bmm2ResUb, bmm2ResUb, scale2Value, dealRowCount * columnCount);
    AscendC::PipeBarrier<PIPE_V>();
    if (isQuantOffset2Exit) {
        Adds(bmm2ResUb, bmm2ResUb, offset2Value, dealRowCount * columnCount);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::DealInvalidRows(LocalTensor<T> &attenOutUb, uint32_t startRow,
                                              uint32_t dealRowCount, uint32_t columnCount)
{
    if (!constInfo.attenMaskFlag) {
        return;
    }

    if (constInfo.sparseMode == fa_base_vector::ALL_MASK || constInfo.sparseMode == fa_base_vector::LEFT_UP_CAUSAL) {
        return;
    }

    fa_base_vector::InvalidRowParams params {
        .actS1Size = actSeqLensQ,
        .gSize = constInfo.gSize,
        .gS1Idx = taskInfo.gS1Idx + startRow,
        .dealRowCount = dealRowCount,
        .columnCount = columnCount,
        .preTokensPerBatch = preTokensPerBatch,
        .nextTokensPerBatch = nextTokensPerBatch,
    };

    fa_base_vector::InvalidRows<T, fa_base_vector::GeInputUbFormat<LAYOUT_T>()> invalidRows;
    invalidRows(attenOutUb, params);
}

template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::DealInvalidMaskRows(LocalTensor<T> &attenOutUb, uint32_t startRow,
                                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
{
    if (!constInfo.isRowInvalid || !constInfo.attenMaskFlag) {
        return;
    }
    if (constInfo.sparseMode != fa_base_vector::DEFAULT_MASK && constInfo.sparseMode != fa_base_vector::ALL_MASK) {
        return;
    }
    LocalTensor<T> lseMaxUb = cntM % 2 == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();

    // 这里要找到lseMaxUb 最大值为-inf 与 attenOutUb的对应位置之间的关系
    // 由于到这里的lseMaxUb 和 attenOutUb都是经过偏移后的，所以offset = 0
    // 同时，这里的lseMaxUb是经过brcb后的，所以填写true

    fa_base_vector::InvalidMaskRows<T, T, true>(0, dealRowCount, columnCount,
            lseMaxUb, negativeIntScalar, attenOutUb);
}

template <typename FIAT> __aicore__ inline void
FiaBlockVecFlashDecode<FIAT>::FlashDecode(FDparams &fd)
{
    if (blockIdx >= fd.usedVecNumOfFd) {
        return;
    }
    uint32_t fdTaskPrevEnd = (blockIdx > 0) ? fd.gS1IdxEndOfFdHead[blockIdx - 1] : 0; // 上一个核末尾是第几个规约
    uint32_t fdS1gOuterMPrevEnd =
        (blockIdx > 0) ? fd.gS1IdxEndOfFdHeadSplit[blockIdx - 1] : 0; //上一个核末尾是该规约的第几个base行
    uint32_t fdTaskEnd = fd.gS1IdxEndOfFdHead[blockIdx];                 // 当前核的末尾是第几个规约任务
    uint32_t fdS1gOuterMEnd = fd.gS1IdxEndOfFdHeadSplit[blockIdx]; // 当前核的末尾是该规约的第几个base行
    uint32_t tmpFdS1gOuterMStart = (blockIdx > 0) ? fdS1gOuterMPrevEnd + 1 : 0; // 当前核从第几个base行开始
    uint32_t tmpFdS1gOuterMEnd = 0;
    uint32_t reduceGlobalLoop = 0;
    uint32_t reduceMLoop = 0;

    for (uint32_t fdTaskId = fdTaskPrevEnd; fdTaskId <= fdTaskEnd; fdTaskId++) {
        tmpFdS1gOuterMEnd = (fdTaskId == fdTaskEnd) ? fdS1gOuterMEnd : (fd.gS1SplitNumOfFdHead[fdTaskId] - 1);
        taskInfo.bIdx = fd.bN2IdxOfFdHead[fdTaskId] / constInfo.kvHeadNum;
        taskInfo.n2Idx = fd.bN2IdxOfFdHead[fdTaskId] % constInfo.kvHeadNum;
        taskInfo.gS1Idx = fd.gS1IdxOfFdHead[fdTaskId] * constInfo.mBaseSize;
        taskInfo.actualCombineLoopSize = fd.s2SplitNumOfFdHead[fdTaskId]; // 当前规约任务kv方向有几份

        uint64_t combineTaskPrefixSum = 0;
        for (uint32_t i = 0; i < fdTaskId; i++) {
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

            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            if (unlikely(learnableSinkFlag)) {
                CopySinkIn(reduceMLoop);
            }

            for (uint32_t preLoadIdx = 0; preLoadIdx < constInfo.preLoadNum; preLoadIdx++) {
                LocalTensor<T> mm2Res = (reduceGlobalLoop + preLoadIdx) % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobalLoop + preLoadIdx) % 2);
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobalLoop + preLoadIdx) % 2);
            }

            ComputeScaleValue(lseExp, startRow, actualGSplitSize, reduceMLoop);
            CalcPreNextTokens();
            //****************************************************************************************************** */
            if (constInfo.softmaxLseFlag) {
                // 新增 计算规约后的max和sum
                LocalTensor<T> lseMaxUb = reduceMLoop % 2 == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();
                LocalTensor<T> lseSumUb = reduceMLoop % 2 == 0 ? fdLseSumUbBuf1.Get<T>() : fdLseSumUbBuf2.Get<T>();
                // 新增 最终LSE的计算lse = log(sum) + max

                WaitFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
                LocalTensor<T> maxLseUb = fdLseUbBuf.Get<T>();
                fa_base_vector::ComputeSoftMaxLse(maxLseUb, lseSumUb, lseMaxUb, actualGSplitSize);
                // 判断是否行无效
                bool isInValidRowsFlag = fa_base_vector::IsExistInvalidRows(nextTokensPerBatch, preTokensPerBatch, constInfo.sparseMode,
                                          constInfo.attenMaskFlag, constInfo.isRowInvalid);
                if (isInValidRowsFlag) {
                    SoftMaxShapeInfo softmaxShapeInfo{static_cast<uint32_t>(actualGSplitSize),
                                                      static_cast<uint32_t>(AttentionCommon::FP32_BLOCK_ELEMENT_NUM),
                                                      static_cast<uint32_t>(actualGSplitSize),
                                                      static_cast<uint32_t>(AttentionCommon::FP32_BLOCK_ELEMENT_NUM)};
                    AdjustSoftMaxRes<T, T>(maxLseUb, lseMaxUb, negativeIntScalar, FLOAT_INF, softmaxShapeInfo);
                }
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
                    uint32_t prefixBS1 = taskInfo.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
                    uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseTND<T>(softmaxLseGm, maxLseUb, bN2Offset, mOffset, actualGSplitSize, constInfo);
                } else if constexpr (LAYOUT_T == FIA_LAYOUT::NTD) {
                    uint32_t prefixBS1 = taskInfo.bIdx == 0U ? 0U : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1);
                    uint32_t s1Size = taskInfo.bIdx == 0U ? 
                            actualSeqLengthsGmQ.GetValue(0U) : actualSeqLengthsGmQ.GetValue(taskInfo.bIdx) - actualSeqLengthsGmQ.GetValue(taskInfo.bIdx - 1U);
                    uint64_t bN2Offset = prefixBS1 * constInfo.qHeadNum + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseNTD<T>(softmaxLseGm, maxLseUb, bN2Offset, mOffset, actualGSplitSize, constInfo, s1Size);
                } else if constexpr (LAYOUT_T == FIA_LAYOUT::BSND || LAYOUT_T == FIA_LAYOUT::BSH) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize + taskInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                    DataCopySoftmaxLseBSND<T, Q_MODE>(softmaxLseGm, maxLseUb, bN2Offset, mOffset, actualGSplitSize, constInfo, qActSeqLensParser, taskInfo.bIdx);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.qHeadNum * constInfo.qSeqSize + taskInfo.n2Idx * constInfo.gSize * constInfo.qSeqSize;
                    DataCopySoftmaxLseBNSD<T, Q_MODE>(softmaxLseGm, maxLseUb, bN2Offset, mOffset, actualGSplitSize, constInfo, qActSeqLensParser, taskInfo.bIdx);
                }
                SetFlag<HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + reduceMLoop % 2);
            //****************************************************************************************************** */

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res = reduceGlobalLoop % 2 == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                if (i >= constInfo.preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                }

                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + reduceGlobalLoop % 2);
                reduceGlobalLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
        tmpFdS1gOuterMStart = 0;
    }
}
#endif