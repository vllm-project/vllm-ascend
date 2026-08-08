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
 * \file fia_block_vec_flashdecode_fullquant.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FLASHDECODE_FULLQUANT_H
#define FIA_BLOCK_VEC_FLASHDECODE_FULLQUANT_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "infer_flash_attention_comm.h"
#include "./vf/vf_flash_decode.h"
#include "fia_public_define_arch35.h"
#include "../memory_copy_arch35.h"
#include "./vf/vf_post_quant_gs1.h"

namespace BaseApi {
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FiaBlockVecFlashDecodeFullQuant {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr GmFormat PostQuant_FORMAT = GmFormat::NGD;
    using SINK_T = INPUT_T;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    struct TaskInfo {
        uint32_t bIdx;
        uint32_t n2Idx;
        uint32_t gS1Idx;
        uint32_t actualCombineLoopSize;
    };

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 0;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 1;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 3;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 4;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 5;
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 7;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;
    static constexpr bool attenMaskFlag = hasAtten;
    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    // 基本块大小
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;

protected:
    GlobalTensor<float> lseSumFdGm;
    GlobalTensor<float> lseMaxFdGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<SINK_T> sinkGm;

    // postquant
    using postQuantGmType = typename std::conditional<POST_QUANT, FaGmTensor<T, PostQuant_FORMAT>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT,
        FaGmTensor<bfloat16_t, PostQuant_FORMAT>, int8_t>::type;
    postQuantGmType quantScale2GmTensor;
    postQuantGmType quantOffset2GmTensor;
    postQuantBf16GmType quantScale2Bf16GmTensor;
    postQuantBf16GmType quantOffset2Bf16GmTensor;
    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();
    static constexpr bool isS1G = (UB_FORMAT == UbFormat::S1G);
    static constexpr bool isPa = KvLayoutType > 0;
    bool isQuantOffset2Exit = false;
    bool isQuant2PerChn = false;
    bool isQuant2Bf16 = false;
    float postQuantScaleValue = 0;
    float postQuantOffsetValue = 0;

    // =======================获取实际Act_S===========================
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, isPa>();
    // tensorlist
    __gm__ uint8_t *keyPtr = nullptr;
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    bool learnableSinkFlag = false;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX &constInfo;
    TaskInfo taskInfo{};

private:
    // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1;    // 1.5k: 16*24*4
    TBuf<> fdSumBuf2;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf1;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf2;    // 1.5k: 16*24*4
    TBuf<> fdLseExpBuf;  // 1.5k: 16*24*4
    TBuf<> fdMm2ResBuf1; // 32k: 16*512*4
    TBuf<> fdMm2ResBuf2; // 32k: 16*512*4
    TBuf<> fdReduceBuf;  // 32k: 16*512*4
    TBuf<> fdOutputBuf;  // 32k: 16*512*4
    TBuf<> fdSinkCopyInBuf;
    TBuf<> fdSinkValueBuf;
    TBuf<> fdSinkExpBuf;
    TBuf<> fdSinkTmpBuf;

    TBuf<> fdLseMaxUbBuf1;
    TBuf<> fdLseMaxUbBuf2;
    TBuf<> fdLseUbBuf;

public:
    __aicore__ inline FiaBlockVecFlashDecodeFullQuant(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                            GlobalTensor<uint64_t> actualSeqLengthsGm, __gm__ uint8_t *key,
                                            __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2)
    {
        this->lseMaxFdGm = lseMaxFdGm;
        this->lseSumFdGm = lseSumFdGm;
        this->accumOutGm = accumOutGm;
        this->attentionOutGm = attentionOutGm;
        this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->keyPtr = key;

        qActSeqLensParser.Init(this->actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);
        kvActSeqLensParser.Init(this->actualSeqLengthsGm, constInfo.actualSeqLenKVSize, constInfo.s2Size);

        if constexpr (POST_QUANT) {
            InitPostQuant(quantScale2, quantOffset2);
        }
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm = softmaxLseGm;
    }
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
    {
        learnableSinkFlag = true;
        this->sinkGm = learnableSink;
    }
    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = this->Align(constInfo.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }
    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            pipe->Reset();
            // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
            pipe->InitBuffer(fdSumBuf1, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);

            pipe->InitBuffer(fdSumBuf2, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdMaxBuf1, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdMaxBuf2, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // TmpBuf
            pipe->InitBuffer(fdLseExpBuf, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // InQue, DB, SYNC_MM2RES_BUF1_FLAG SYNC_MM2RES_BUF2_FLAG
            pipe->InitBuffer(fdMm2ResBuf1, BUFFER_SIZE_BYTE_16K);
            pipe->InitBuffer(fdMm2ResBuf2, BUFFER_SIZE_BYTE_16K);
            // TmpBuf
            pipe->InitBuffer(fdReduceBuf, BUFFER_SIZE_BYTE_16K);
            // OutQue, SYNC_FDOUTPUT_BUF_FLAG
            pipe->InitBuffer(fdOutputBuf, BUFFER_SIZE_BYTE_16K);
            pipe->InitBuffer(fdLseMaxUbBuf1, BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(fdLseMaxUbBuf2, BUFFER_SIZE_BYTE_256B);
            // OutQue, SYNC_LSEOUTPUT_BUF_FLAG
            pipe->InitBuffer(fdLseUbBuf, BUFFER_SIZE_BYTE_256B);

            // TmpBuf
            if (unlikely(learnableSinkFlag)) {
                // InQue, DB, SYNC_SINK_BUF1_FLAG SYNC_SINK_BUF2_FLAG
                pipe->InitBuffer(fdSinkCopyInBuf, BUFFER_SIZE_BYTE_2K);
                // TmpBuf
                pipe->InitBuffer(fdSinkValueBuf, BUFFER_SIZE_BYTE_2K);
                pipe->InitBuffer(fdSinkTmpBuf, BUFFER_SIZE_BYTE_2K);
            }
            // 后面要取地址，放到外面
            pipe->InitBuffer(fdSinkExpBuf, BUFFER_SIZE_BYTE_256B);
        }
    }
    __aicore__ inline void AllocEventID()
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
    __aicore__ inline void FreeEventID()
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

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align - constInfo.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +                 // taskoffset + g轴offset
                                         splitKVIndex * s1BaseSize * constInfo.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();

        uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = s1BaseSize * BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; ++i) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1.Get<T>() : fdSumBuf2.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1.Get<T>() : fdMaxBuf2.Get<T>();
        if (unlikely(learnableSinkFlag)) {
            SinkMax(startRow, dealRowCount);
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf.Get<T>();
        LocalTensor<T> maxLseUb = fdLseUbBuf.Get<T>();
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo.isSoftmaxLseEnable, learnableSinkFlag);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                        .n2Idx = taskInfo.n2Idx,
                        .gS1Idx = taskInfo.gS1Idx + startRow,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};

        if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize,
                                              false, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize,
                                              false, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align;
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        if constexpr (POST_QUANT) {
            if (isQuant2PerChn) {
                DealPostQuantOutPerChn(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                DealPostQuantOutPerTensor(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUTPUT_T>();
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (POST_QUANT) {
            DealInvalidRows(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
            DealInvalidMaskRows(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align, cntM);
        } else {
            DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
            DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
        }
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        if constexpr (!POST_QUANT) {
            uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
            tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
            if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
                Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT,
                     dealRowCount * this->dSizeV_Align);
            } else {
                Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                     dealRowCount * this->dSizeV_Align);
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
        if (constInfo.actualSeqLenKVSize == 0 && !constInfo.isKvContinuous) {
            actSeqLensKv = SeqLenFromTensorList<layout>(keyPtr, taskInfo.bIdx);
        } else {
            actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
        }
        int64_t safePreToken = constInfo.preTokens;
        int64_t safeNextToken = constInfo.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

        if (constInfo.sparseMode == BAND) {
            preTokensPerBatch = safePreToken;
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
        } else if ((constInfo.sparseMode == DEFAULT_MASK) && attenMaskFlag) {
            nextTokensPerBatch = safeNextToken;
            preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
        } else {
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
            preTokensPerBatch = 0;
        }
    }
    __aicore__ inline void CopySinkIn(uint32_t cntM)
    {
        LocalTensor<SINK_T> sinkCopyInBuf =
            fdSinkCopyInBuf.GetWithOffset<SINK_T>(BUFFER_SIZE_BYTE_1K, (cntM & 1) * BUFFER_SIZE_BYTE_1K);

        uint64_t sinkGmOffset = taskInfo.n2Idx * constInfo.gSize;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = constInfo.gSize * sizeof(SINK_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<SINK_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<SINK_T>(0);

        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + (cntM & 1));
        DataCopyPad(sinkCopyInBuf, sinkGm[sinkGmOffset], sinkCopyParams, sinkPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + (cntM & 1));
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINK_BUF1_FLAG + (cntM & 1));

        LocalTensor<T> tmpSinkCastBuf = fdSinkTmpBuf.Get<T>();
        Cast(tmpSinkCastBuf, sinkCopyInBuf, AscendC::RoundMode::CAST_NONE, constInfo.gSize);
        AscendC::PipeBarrier<PIPE_V>();

        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG + (cntM & 1));

        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.Get<T>();
        Brcb(sinkBrcbBuf, tmpSinkCastBuf, (constInfo.gSize + BLOCK_ELEMENT_NUM - 1) / BLOCK_ELEMENT_NUM,
             {1, BLOCK_ELEMENT_NUM});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SinkMax(uint32_t startRow, uint32_t dealRowCount)
    {
        constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
        int64_t gIdx = 0;
        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf.Get<T>();
        LocalTensor<T> sinkExpBuf = fdSinkExpBuf.Get<T>();

        for (int64_t row = 0; row < dealRowCount; ++row) {
            if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { // 内存按照S1G排布
                gIdx = (taskInfo.gS1Idx + startRow + row) % constInfo.gSize;
            } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { // 内存按照GS1排布
                int64_t actS1Size = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                gIdx = (taskInfo.gS1Idx + startRow + row) / actS1Size;
            }
            DataCopy(sinkExpBuf[row * BLOCK_ELEMENT_NUM], sinkBrcbBuf[gIdx * BLOCK_ELEMENT_NUM], BLOCK_ELEMENT_NUM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if (!attenMaskFlag) {
            return;
        }

        if (constInfo.sparseMode == ALL_MASK || constInfo.sparseMode == LEFT_UP_CAUSAL) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ,
            .gSize = static_cast<uint64_t>(constInfo.gSize),
            .gS1Idx = taskInfo.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch,
            .nextTokensPerBatch = nextTokensPerBatch,
        };

        fa_base_vector::InvalidRows<UBOUT_T, fa_base_vector::GeInputUbFormat<layout>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if (!constInfo.isRowInvalidOpen || !attenMaskFlag) {
            return;
        }
        if (constInfo.sparseMode != DEFAULT_MASK && constInfo.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1.Get<T>() : fdLseMaxUbBuf2.Get<T>();

        // 这里要找到lseMaxUb 最大值为-inf 与 attenOutUb的对应位置之间的关系
        // 由于到这里的lseMaxUb 和 attenOutUb都是经过偏移后的，所以offset = 0
        // 同时，这里的lseMaxUb是经过brcb后的，所以填写true

        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar,
                                                          attenOutUb);
    }

    __aicore__ inline void InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset)
    {
        return;
    }
    __aicore__ inline void DealPostQuantOutPerChn(LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                                  uint32_t columnCount, uint32_t cntM)
    {
        PostQuantInfo_V2 postQuantInfo;
        postQuantInfo.gSize = constInfo.gSize;
        postQuantInfo.dSize = constInfo.dSizeV;
        postQuantInfo.s1Size = actSeqLensQ;
        postQuantInfo.n2Idx = taskInfo.n2Idx;
        postQuantInfo.gS1Idx = taskInfo.gS1Idx + startRow;
        postQuantInfo.gS1DealSize = dealRowCount;
        postQuantInfo.colCount = columnCount;

        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUTPUT_T>();

        if (isQuant2Bf16) {
            uint32_t computeSize = dealRowCount * columnCount;
            LocalTensor<bfloat16_t> quantScale2Ub =
                ((cntM & 1) == 0) ? fdMm2ResBuf1.Get<bfloat16_t>() : fdMm2ResBuf2.Get<bfloat16_t>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
            CopyParamsGmToUb<bfloat16_t, PostQuant_FORMAT, UB_FORMAT>(quantScale2Ub, quantScale2Bf16GmTensor,
                                                                      postQuantInfo);
            if (isQuantOffset2Exit) {
                LocalTensor<bfloat16_t> quantOffset2Ub =
                    ((cntM & 1) == 0) ? fdMm2ResBuf1.Get<bfloat16_t>() : fdMm2ResBuf2.Get<bfloat16_t>();
                AscendC::PipeBarrier<PIPE_MTE2>();
                CopyParamsGmToUb<bfloat16_t, PostQuant_FORMAT, UB_FORMAT>(quantOffset2Ub, quantOffset2Bf16GmTensor,
                                                                          postQuantInfo);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                AscendC::PipeBarrier<PIPE_V>();
                FaVectorApi::PostQuantPerChnlOffsetImpl<T, OUTPUT_T, bfloat16_t, isS1G>(
                    tmpBmm2ResCastTensor, bmm2ResUb, quantScale2Ub, quantOffset2Ub, postQuantInfo);
            } else {
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                AscendC::PipeBarrier<PIPE_V>();
                FaVectorApi::PostQuantPerChnlNoOffsetImpl<T, OUTPUT_T, bfloat16_t, isS1G>(
                    tmpBmm2ResCastTensor, bmm2ResUb, quantScale2Ub, postQuantInfo);
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<T> quantScale2Ub = ((cntM & 1) == 0) ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
            CopyParamsGmToUb<T, PostQuant_FORMAT, UB_FORMAT>(quantScale2Ub, quantScale2GmTensor, postQuantInfo);
            if (isQuantOffset2Exit) {
                LocalTensor<T> quantOffset2Ub = ((cntM & 1) == 0) ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                AscendC::PipeBarrier<PIPE_MTE2>();
                CopyParamsGmToUb<T, PostQuant_FORMAT, UB_FORMAT>(quantOffset2Ub, quantOffset2GmTensor, postQuantInfo);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                AscendC::PipeBarrier<PIPE_V>();
                PostQuantPerChnlOffsetImpl<T, OUTPUT_T, T, isS1G>(tmpBmm2ResCastTensor, bmm2ResUb, quantScale2Ub,
                                                                  quantOffset2Ub, postQuantInfo);

            } else {
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
                AscendC::PipeBarrier<PIPE_V>();
                PostQuantPerChnlNoOffsetImpl<T, OUTPUT_T, T, isS1G>(tmpBmm2ResCastTensor, bmm2ResUb, quantScale2Ub,
                                                                    postQuantInfo);
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (cntM & 1));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void DealPostQuantOutPerTensor(LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf.Get<OUTPUT_T>();
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(tmpBmm2ResCastTensor, bmm2ResUb, postQuantScaleValue,
                                                  postQuantOffsetValue, dealRowCount, constInfo.dSizeV,
                                                  this->dSizeV_Align);
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX &fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        uint32_t fdBalanceMBaseSize = 8U;
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo.bIdx = fd.fdBN2Idx / constInfo.n2Size;
        taskInfo.n2Idx = fd.fdBN2Idx % constInfo.n2Size;
        taskInfo.gS1Idx = fd.fdMIdx * s1BaseSize;
        taskInfo.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             ++fdS1gOuterMIdx) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf.Get<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            if (unlikely(learnableSinkFlag)) {
                CopySinkIn(reduceMLoop);
            }
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; ++preLoadIdx) {
                LocalTensor<T> mm2Res =
                    ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
            CalcPreNextTokens();
            if (constInfo.isSoftmaxLseEnable) {
                // lse行无效在ComputeScaleValue的VF计算时已经进行了赋值inf处理
                LocalTensor<T> maxLseUb = fdLseUbBuf.Get<T>();
                // 判断是否行无效
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                    uint32_t prefixBS1 = qActSeqLensParser.GetTBase(taskInfo.bIdx);
                    uint64_t bN2Offset =
                        prefixBS1 * constInfo.gSize * constInfo.n2Size + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseTNDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                               actualGSplitSize, constInfo);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
                    uint32_t prefixBS1 = qActSeqLensParser.GetTBase(taskInfo.bIdx);
                    uint32_t s1Size = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                    uint64_t bN2Offset =
                        prefixBS1 * constInfo.gSize * constInfo.n2Size + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                               actualGSplitSize, constInfo, s1Size);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                    uint64_t s1LeftPaddingSize = 0;
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo, s1LeftPaddingSize);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(taskInfo.bIdx);
                    uint64_t s1LeftPaddingSize = 0;
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo, qActSeqLens,
                                                                s1LeftPaddingSize);
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1.Get<T>() : fdMm2ResBuf2.Get<T>();
                if (i >= preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
          bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0,
          bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FiaBlockVecFlashDecodeFullQuantDummy {
public:
    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    __aicore__ inline FiaBlockVecFlashDecodeFullQuantDummy(ConstInfoX &constInfo) {};
};

} // namespace BaseApi
#endif