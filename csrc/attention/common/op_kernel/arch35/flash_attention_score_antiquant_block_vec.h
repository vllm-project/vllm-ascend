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
 * \file flash_attention_score_antiquant_block_vec.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_ANTIQUANT_BLOCK_VEC_BASE_H_
#define FLASH_ATTENTION_SCORE_ANTIQUANT_BLOCK_VEC_BASE_H_
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "vf/vf_post_quant.h"
#include "flash_attention_score_antiquant_processor.h"
#include "flash_attention_score_tiling_regbase.h"
#include "util_regbase.h"
using namespace optiling;
using namespace FaVectorApi;
namespace BaseApi {
__aicore__ inline constexpr uint16_t Align64FuncAntiquantup(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

template <typename T>
__aicore__ inline T ALIGNAntiquant(T num, T rnd) {
    return ((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd));
}

static constexpr uint32_t FA_BYTE_BLOCK_ANTIQUANT = 32;
constexpr uint64_t BYTE_BLOCK_ANTIQUANT = 32UL;

ANTIQUANT_TEMPLATES_DEF
class FABlockVecAntiquant {
public:
    /*Compile Static Variable*/
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1ScmBlockTrue = s1BaseSize * (16 / 2);
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64FuncAntiquantup((uint16_t)dVTemplateType);
    static constexpr uint32_t BUFFER_SIZE_BYTE_5K_ANTIQUANT = 5120;  // 5K deal the tail
    static constexpr uint32_t pseInputSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t attenMaskSize = s1BaseSize / CV_RATIO * s2BaseSize * 1;
    static constexpr uint32_t kvInputSize = dTemplateAlign64 * s2BaseSize / 4;
    static constexpr uint32_t kvOutSize = dTemplateAlign64 * s2BaseSize * 2 *0.25 + 1024;
    static constexpr uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t kvAntiquantResSize = s2BaseSize * dTemplateAlign64 * sizeof(Q_T);
    static constexpr uint32_t stage1OutQueSize = (s1BaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t stage2OutQueSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t bufferSizeByte32K = 32768;
    static constexpr uint32_t gSplitMax = 16;
    static constexpr bool useDn = false;
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten;
    /*伪量化模式相关*/
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool KVFP4 = (IsSameType<KV_T, fp4x2_e1m2_t>::value || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool ANTIQUANT_PER_GROUP = (IsSameType<KV_T, fp4x2_e1m2_t>::value
        || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool KEY_ANTIQUANT_PER_TOKEN = (antiquantMode == AntiquantTypeEnum::K_PER_TOKEN);
    static constexpr bool VALUE_ANTIQUANT_PER_TOKEN = (antiquantMode == AntiquantTypeEnum::K_PER_TOKEN
        || antiquantMode == AntiquantTypeEnum::K_PER_CHANNEL_V_PER_TOKEN);
    using KEY_ANTIQ_PARAMS_T = typename std::conditional<KEY_ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    using VALUE_ANTIQ_PARAMS_T = typename std::conditional<VALUE_ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value
        && !IsSameType<OUTPUT_T, float>::value;
    /*Public Function*/
    __aicore__ inline FABlockVecAntiquant() {};
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *value, __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *pse, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *workspace,
        const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        TPipe *pipe, AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo);
    __aicore__ inline void SendCrossCoreFlag();
    __aicore__ inline void InitLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitAntiquantBuffer();
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void ClearOutput(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitQuant(__gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
        __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
        __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
        __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitPostQuant(__gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
        ConstInfo<isInfer, hasRope> &constInfo);
    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(ConstInfo<isInfer, hasRope> &constInfo, const RunInfo<isInfer> &runInfo,
        LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t dSizeAligned64);
    __aicore__ inline void FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
        LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount, uint32_t dSizeAligned64);
    template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
    __aicore__ inline void PostQuantPerChnl(ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut,
        LocalTensor<VEC2_RES_T> &vec2ResUb, uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount,
        uint32_t splitOffset, int64_t dSizeAligned64, GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm,
        GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm);

    __aicore__ inline void setConstAntiTaskParam(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void AntiquantKey(const RunInfo<isInfer> &runInfo, int64_t &subTaskId, bool &first,
        RunParamStr<isInfer> &runParam, Buffer<BufferType::L1> &outBufAntiKey,
        GlobalTensor<KV_T> &tempKeyGm, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void AntiquantValue(const RunInfo<isInfer> &runInfo, int64_t &subTaskId, bool &first,
        RunParamStr<isInfer> &runParam, Buffer<BufferType::L1> &outBufAntiValue, ConstInfo<isInfer, hasRope> &constInfo);
    /*FD相关*/
    __aicore__ inline void FlashDecode(ConstInfo<isInfer, hasRope> &constInfo, __gm__ int64_t *seqQlenAddr,
        __gm__ int64_t *seqKVlenAddr, int64_t s2InCurrentBatch);
    __aicore__ inline void ProcessVec1(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<T> &inputTensorVec, Buffer<BufferType::L1> &outBufVec1);
    __aicore__ inline void UpdateVec2Result(RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb,
        LocalTensor<T> &inputTensorVec, LocalTensor<T> &expUb);
    __aicore__ inline void ProcessVec2(RunInfo<isInfer> &runInfo, LocalTensor<T> &inputTensorVec,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
        const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    template<typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(const RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
        int64_t vec2CalcSize, ConstInfo<isInfer, hasRope> &constInfo);
    template<typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
        const RunInfo<isInfer> &runInfo, int64_t dSizeAligned64, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline int64_t ComputeOffsetForSoftmax(const RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void Bmm2FDOut(const RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void FlashDecodeCompute(ConstInfo<isInfer, hasRope> &constInfo,
        __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr, int64_t s2InCurrentBatch);
    __aicore__ inline void GetActualSeqLenKV(int64_t boIdx, int64_t &actualSeqLen, __gm__ int64_t *actualSeqKvlenAddr,
        int64_t s2InCurrentBatch, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetActualSeqLenQ(int64_t boIdx, int64_t &actualSeqLen, __gm__ int64_t *actualSeqQlenAddr);
    __aicore__ inline void CombineSplitKVRes(uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx,
        ConstInfo<isInfer, hasRope> &constInfo, __gm__ int64_t *actualSeqQlenAddr);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint32_t splitSize,
        uint64_t lseOffset, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CopyLseIn(uint32_t bIdx, uint32_t n2Idx, uint32_t startRow,
        uint32_t dealRowCount, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CopyFinalResOut(uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow,
        uint32_t dealRowCount,  uint64_t perChannelQuantOffset, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CopyAccumOutIn(uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
        uint32_t dealRowCount, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetKvByTensorList(const RunInfo<isInfer>& runInfo, GlobalTensor<KV_T>& keyValueGm,
        GlobalTensor<KV_T>& tempKeyValueGm, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ReduceFinalRes(uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
        uint32_t startRow, uint32_t dealRowCount, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ReduceFDDataCopyOut(uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
        uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxDataCopyOut(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo);
    bool antiquantPageAttentionFlag;
    bool antiquantPerTensorFlag;
    bool antiquantPerHeadFlag;
    uint32_t antiqSeqSize;
    event_t UbToL1Event;

protected:
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<KV_T> valueSharedPrefixGm;
    T negativeFloatScalar;
    T positiveFloatScalar;
    TQue<QuePosition::VECIN, 1> kvInputQue;
    TQue<QuePosition::VECOUT, 1> kvOutputQue;
    TQue<QuePosition::VECIN, 1> keyAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> keyAntiqOffsetInputQue;
    TQue<QuePosition::VECIN, 1> valueAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> valueAntiqOffsetInputQue;
    TBuf<> kvAntiqMxScaleRes;  // for w4
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> vselrIndexesBuf[4];
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    TBuf<> commonTBuf;
    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECIN, 1> softmaxMaxInputQue;
    TQue<QuePosition::VECIN, 1> softmaxSumInputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> pseInQue;
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECOUT, 1> stage2OutQue[2];
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue;
    /*伪量化参数*/
    GlobalTensor<KEY_ANTIQ_PARAMS_T> keyAntiquantOffsetGm;
    GlobalTensor<KEY_ANTIQ_PARAMS_T> keyAntiqScaleGm;
    GlobalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiquantOffsetGm;
    GlobalTensor<VALUE_ANTIQ_PARAMS_T> valueAntiqScaleGm;
    bool antiqOffsetExistFlag;
    bool isBeforeHalf;
    AntiquantTaskParamBaseAPI taskParam;

private:
    TPipe *tPipe;
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    __gm__ uint8_t *currentValue;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    __gm__ uint8_t *pseSlope;
    using pseShiftType = typename AscendC::Conditional<POST_QUANT, Q_T, OUTPUT_T>::type;
    using pseGmType = typename std::conditional<hasPse, GlobalTensor<pseShiftType>, int8_t>::type;
    pseGmType pseGm;
    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;
    GlobalTensor<float> softmaxLseGm;
    __gm__ uint8_t *blocktablePtr;
    GlobalTensor<int32_t> blockTableGm;
    using FDGmType = typename std::conditional<isFd, GlobalTensor<float>, int8_t>::type;
    FDGmType accumOutGm;
    FDGmType softmaxFDMaxGm;
    FDGmType softmaxFDSumGm;
    LocalTensor<pseShiftType> dummyPseTensor;
    LocalTensor<uint8_t> dummyAttenMaskTensor;
    PseInfo pseInfo;
    AttenMaskInfo attenMaskInfo;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    AntiquantProcessorBaseAPI<ANTIQUANT_PROCESSOR_ARGS, KEY_ANTIQUANT_PER_TOKEN> keyAntiquantProcessor;
    AntiquantProcessorBaseAPI<ANTIQUANT_PROCESSOR_ARGS, VALUE_ANTIQUANT_PER_TOKEN> valueAntiquantProcessor;
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;
    postQuantBf16GmType postQuantScaleBf16Gm;
    postQuantBf16GmType postQuantOffsetBf16Gm;
};

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitVecBlock(
    __gm__ uint8_t *value, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *pse, __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *workspace, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
    TPipe *pipe, AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo)
{
    this->tPipe = pipe;
    this->tilingData = tiling;
    ListTensorDesc valueListTensorDescInit((__gm__ void *)value);
    currentValue = (__gm__ uint8_t *)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->valueGm.SetGlobalBuffer((__gm__ KV_T *)currentValue);
    } else {
        this->valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
    }
    if constexpr (enableKVPrefix) {
        this->valueSharedPrefixGm.SetGlobalBuffer((__gm__ KV_T *)valueSharedPrefix);
    }
    this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
    if constexpr (POST_QUANT) {
        this->attentionOutInitGm.SetGlobalBuffer((__gm__ half *)attentionOut);
    }
    this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    if constexpr (hasPse) {
        pseGm.SetGlobalBuffer((__gm__ pseShiftType *)pse);
        pseSlope = pse;
        this->pseInfo = pseInfo;
    }
    if constexpr (hasAtten) {
        attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        this->attenMaskInfo = attenMaskInfo;
    }
    if constexpr (isPa) {
        blocktablePtr = blockTable;
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    if constexpr (isFd) {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        uint64_t accumOutSize = this->tilingData->inputParamsRegbase.accumOutSize;
        uint64_t logSumExpSize = this->tilingData->inputParamsRegbase.logSumExpSize;
        accumOutGm.SetGlobalBuffer((__gm__ T *)(workspace));
        workspace += accumOutSize * sizeof(float);
        softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
        softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)(workspace));
        workspace += logSumExpSize * sizeof(float);
    }
    this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
            positiveScalar = *((half *)&tmp2);
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SendCrossCoreFlag()
{
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM1RES_EVENT[0]);
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM1RES_EVENT[1]);
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[0]);
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[1]);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo)
{
    this->InitAntiquantBuffer();
    this->SoftmaxInitBuffer();
    this->tPipe->InitBuffer(commonTBuf, COMMON_TEMP_BUF_SIZE); // 0.5K 内存
    if constexpr (hasPseOuter) {
        this->tPipe->InitBuffer(pseInQue, 1, pseInputSize);
    }
    if constexpr (hasAtten) {
        this->tPipe->InitBuffer(attenMaskInQue[0], 1, attenMaskSize);
        this->tPipe->InitBuffer(attenMaskInQue[1], 1, attenMaskSize);
    }
    if (constInfo.isSoftmaxLseEnable) {
        // 每行结果存为8个重复lse元素(32B对齐)
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8);
    }
    this->tPipe->InitBuffer(this->stage1OutQue[0], 1, stage1OutQueSize);
    this->tPipe->InitBuffer(this->stage2OutQue[0], 1, stage2OutQueSize);
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, POST_QUANT_BUF_SIZE_KB * KB_TO_BYTES); // 2K Buffer
        this->tPipe->InitBuffer(postQuantOffsetQue, 1, POST_QUANT_BUF_SIZE_KB * KB_TO_BYTES); // 2K Buffer
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitAntiquantBuffer()
{
    this->tPipe->InitBuffer(kvInputQue, 3, kvInputSize); // 3 buffer
    this->tPipe->InitBuffer(kvOutputQue, 2, kvOutSize);  // 2 buffer
    this->tPipe->InitBuffer(keyAntiqScaleInputQue, 1, 3072); // 3072 is 3 * 1024, deal the tail
    this->tPipe->InitBuffer(keyAntiqOffsetInputQue, 1, 3072); // 3072 is 3 * 1024, deal the tail
    this->tPipe->InitBuffer(valueAntiqScaleInputQue, 1, 3072); // 3072 is 3 * 1024, deal the tail
    this->tPipe->InitBuffer(valueAntiqOffsetInputQue, 1, 3072); // 3072 is 3 * 1024, deal the tail
    if constexpr (KVFP4) {
        this->tPipe->InitBuffer(kvAntiqMxScaleRes, BUFFER_SIZE_BYTE_5K_ANTIQUANT);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    this->tPipe->InitBuffer(this->softmaxSumBuf[0], SOFTMAX_ROW_BUF_SIZE); // [64, 1] SOFTMAXSUMBUF_SIZE:256
    this->tPipe->InitBuffer(this->softmaxSumBuf[1], SOFTMAX_ROW_BUF_SIZE); // [64, 1] 1 is second buffer,
    this->tPipe->InitBuffer(this->softmaxSumBuf[2], SOFTMAX_ROW_BUF_SIZE); // [64, 1] 2 is third buffer
    this->tPipe->InitBuffer(this->maxBrdcst, 1, SOFTMAX_BRDCST_BUF_SIZE); // [64, 8] SUMBRDCST_SIZE:2048
    this->tPipe->InitBuffer(this->sumBrdcst, 1, SOFTMAX_BRDCST_BUF_SIZE); // [64, 8] SUMBRDCST_SIZE:2048
    this->tPipe->InitBuffer(this->softmaxMaxBuf[0], SOFTMAX_ROW_BUF_SIZE); // [64, 1] SOFTMAXMAXBUF_SIZE:256
    this->tPipe->InitBuffer(this->softmaxMaxBuf[1], SOFTMAX_ROW_BUF_SIZE); // [64, 1] 1 is second buffer
    this->tPipe->InitBuffer(this->softmaxMaxBuf[2], SOFTMAX_ROW_BUF_SIZE); // [64, 1] 2 is the third buffer
    this->tPipe->InitBuffer(this->softmaxExpBuf[0], SOFTMAX_ROW_BUF_SIZE); // [64, 1] SOFTMAXEXPBUF_SIZE:256
    this->tPipe->InitBuffer(this->softmaxExpBuf[1], SOFTMAX_ROW_BUF_SIZE); // [64, 1] 1 is second buffer
    this->tPipe->InitBuffer(this->softmaxExpBuf[2], SOFTMAX_ROW_BUF_SIZE); // [64, 1] 2 is the third buffer
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ClearOutput(ConstInfo<isInfer, hasRope> &constInfo)
{
    if (this->tilingData->initOutputParams.needInit == 1) {
        InitOutputSingleCore(constInfo);
        if (constInfo.isSoftmaxLseEnable) {
            SyncAll();
            InitLseOutputSingleCore(constInfo);
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    auto &initParams = this->tilingData->initOutputParams;
    if (initParams.totalOutputSize > constInfo.aivIdx * initParams.singleCoreSize) {
        uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
        uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
        if constexpr (POST_QUANT) {
            InitOutput<half>(this->attentionOutInitGm[constInfo.aivIdx * initParams.singleCoreSize / 2],
                singleInitOutputSize / 2, 0.0); // 2: 将内容一分为二
        } else {
            InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * initParams.singleCoreSize],
                singleInitOutputSize, 0.0);
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitLseOutputSingleCore(ConstInfo<isInfer, hasRope> &constInfo)
{
    int64_t coreNum = GetBlockNum() * GetTaskRation();
    auto &initParams = this->tilingData->initOutputParams;
    if (coreNum != 0 && constInfo.aivIdx < coreNum) {
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (constInfo.aivIdx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[constInfo.aivIdx * (initParams.totalSoftMaxLseOutputSize / coreNum)],
            singleCoreLseSize, 3e+99);  // 3e+99: set the value of invalid batch to inf
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitQuant(
    __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
    __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (ANTIQUANT) {
        if (keyAntiquantScale == nullptr) {
            int64_t antiValueOffsetInitPos = constInfo.n2D;
            if constexpr (KVFP4) {
                antiValueOffsetInitPos = (uint64_t)(this->tilingData->inputParamsRegbase.bSize) * constInfo.n2S2D >> SHIFT_NUM_6;
            }
            if constexpr (KEY_ANTIQUANT_PER_TOKEN) {
                antiValueOffsetInitPos = (uint64_t)(this->tilingData->inputParamsRegbase.bSize) * antiqSeqSize;
            }
            if (antiquantPerTensorFlag) {
                antiValueOffsetInitPos = 1;
            }
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)antiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer(((__gm__ VALUE_ANTIQ_PARAMS_T*)antiquantScale) + antiValueOffsetInitPos);
            antiqOffsetExistFlag = (antiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)antiquantOffset);
                valueAntiquantOffsetGm.SetGlobalBuffer(((__gm__ VALUE_ANTIQ_PARAMS_T*)antiquantOffset) + antiValueOffsetInitPos);
            }
        } else {
            keyAntiqScaleGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)keyAntiquantScale);
            valueAntiqScaleGm.SetGlobalBuffer((__gm__ VALUE_ANTIQ_PARAMS_T*)valueAntiquantScale);
            antiqOffsetExistFlag = (keyAntiquantOffset != nullptr);
            if (antiqOffsetExistFlag) {
                keyAntiquantOffsetGm.SetGlobalBuffer((__gm__ KEY_ANTIQ_PARAMS_T*)keyAntiquantOffset);
                valueAntiquantOffsetGm.SetGlobalBuffer((__gm__ VALUE_ANTIQ_PARAMS_T*)valueAntiquantOffset);
            }
        }
    }

    if constexpr (POST_QUANT) {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        constInfo.isPostQuantPerChnl = inputParamsRegbase.isPostQuantPerChnl;
        constInfo.isPostQuantBF16 = inputParamsRegbase.isPostQuantBF16;
        InitPostQuant(postQuantScale, postQuantOffset, constInfo);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitPostQuant(__gm__ uint8_t *postQuantScale,
    __gm__ uint8_t *postQuantOffset, ConstInfo<isInfer, hasRope> &constInfo)
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

        if (constInfo.isPostQuantPerChnl && !constInfo.isPostQuantBF16) {
            if (postQuantScale != nullptr) {
                this->postQuantScaleGm.SetGlobalBuffer((__gm__ float *)postQuantScale);
            }
            if (postQuantOffset != nullptr) {
                constInfo.isPostQuantOffsetExist = true;
                postQuantOffsetGm.SetGlobalBuffer((__gm__ float *)postQuantOffset);
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


ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::setConstAntiTaskParam(ConstInfo<isInfer, hasRope> &constInfo)
{
    isBeforeHalf = ((constInfo.aivIdx & 1) == 0);
    taskParam.batchSize = this->tilingData->inputParamsRegbase.bSize;
    taskParam.seqSize = constInfo.s2Size;
    taskParam.kvHeadNum = constInfo.n2Size;
    taskParam.headDim = constInfo.dSize;
    taskParam.headDimAlignBlock = ALIGNAntiquant((uint64_t)taskParam.headDim, BYTE_BLOCK_ANTIQUANT / sizeof(Q_T));
    if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
        taskParam.kvStep = constInfo.dSize;
    } else {
        taskParam.kvStep = taskParam.kvHeadNum * constInfo.dSize;
    }
    taskParam.copySplitS = kvInputSize / sizeof(KV_T) / dTemplateAlign64;
    taskParam.isPageAttentionAntiquant = antiquantPageAttentionFlag;
    taskParam.isPertensor = antiquantPerTensorFlag;
    taskParam.isPerHead = antiquantPerHeadFlag;
    taskParam.kvCacheBlockSize = constInfo.blockSize;
    taskParam.maxBlockNumPerSeq = constInfo.blockTableDim2;
    taskParam.paKvShapeType = constInfo.paLayoutType;
    taskParam.isExistOffset = antiqOffsetExistFlag;
    taskParam.singleSInnerSize = constInfo.s2BaseSize;
    taskParam.sInnerLoopSize = constInfo.sInnerLoopSize;
    taskParam.antiqSeqSize = antiqSeqSize;
    taskParam.isKvCacheNz = false;
    if constexpr (isPa) {
        taskParam.isKvCacheNz = taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_NZ);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::AntiquantKey(
    const RunInfo<isInfer> &runInfo, int64_t &subTaskId, bool &first, RunParamStr<isInfer> &runParam,
    Buffer<BufferType::L1> &outBufAntiKey, GlobalTensor<KV_T> &tempKeyGm, ConstInfo<isInfer, hasRope> &constInfo)
{
    SetAntiqParamCommon(runInfo, runInfo.keyOffset, constInfo);
    if constexpr (KVFP4) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = false;
    } else if constexpr (KEY_ANTIQUANT_PER_TOKEN) {
        taskParam.isLoadAntiqParam = true;
        taskParam.isFreeAntiqParam = true;
    } else {
        if (antiquantPerTensorFlag) {
            taskParam.isLoadAntiqParam = (first);
            taskParam.isFreeAntiqParam = (first);
        } else {
            taskParam.isLoadAntiqParam = unlikely(runInfo.s2LoopCount == 0);
            taskParam.isFreeAntiqParam = unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit);
        }
    }
    keyAntiquantProcessor.ProcessBaseAPI(outBufAntiKey, tempKeyGm, keyAntiqScaleGm,
                              keyAntiquantOffsetGm, blockTableGm, kvInputQue, kvOutputQue, keyAntiqScaleInputQue,
                              keyAntiqOffsetInputQue, kvAntiqMxScaleRes, taskParam, subTaskId, isBeforeHalf, runInfo.s2RealSize);
}


ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SetAntiqParamCommon(
    const RunInfo<isInfer> &runInfo, int64_t kvOffset, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (isBeforeHalf) {
        taskParam.copyTotalS = GetRealDealSize(runInfo.s2RealSize);  // 2 is Vec num
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (GetRealDealSize(runInfo.s2RealSize));  // 2 is Vec num
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize +
        constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize);
    if constexpr (isInfer) {
        curSequence += runInfo.s2StartIdx;
    }
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    if constexpr(isFd) {
        curSequence += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }
    taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx *
        GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;
    taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim;
    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    if constexpr (isInfer) {
        taskParam.s2Idx += runInfo.s2StartIdx / constInfo.s2BaseSize;
        if constexpr (enableKVPrefix) {
            taskParam.isPrefixLoop = ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) <
                constInfo.prefixLoopCount); // 判断当前s2LoopCount是否处于prefixLoopCount
            if (taskParam.isPrefixLoop) {
                taskParam.kvGmOffset = runInfo.prefixOffset + constInfo.subBlockIdx *
                    GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
            } else {
                taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx *
                    GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
                taskParam.s2Idx -= constInfo.prefixLoopCount;
            }
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ProcessVec1(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
    LocalTensor<T> &inputTensorVec, Buffer<BufferType::L1> &outBufVec1)
{
    if (runInfo.actualS2Size == 0) {
        return;
    }
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(CV_MM1RES_EVENT[runInfo.taskIdMod2]);
    LocalTensor<pseShiftType> pseUb;
    if constexpr (hasPseOuter == true) {
        PseCopyIn<T, pseShiftType, hasPseOuter>(this->pseInQue, this->pseGm, runInfo, constInfo, pseInfo);
        pseUb = this->pseInQue.template DeQue<pseShiftType>();
    } else {
        pseUb = dummyPseTensor;
    }
    float slopes = 0.0f;
    float posShift = 0.0f;
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            if (this->tilingData->inputParamsRegbase.sparseType ==
                    static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL) &&
                runInfo.boIdx != 0) {
                pseInfo.qStartIdx = 0;
                pseInfo.kvStartIdx = 0;
            }
        }
        ComputeInnerPseOffset<T, Q_T, hasPse>(slopes, posShift, runInfo, constInfo, pseInfo, this->pseSlope);
    }
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        AttenMaskCopyIn<hasAtten, isFd, enableKVPrefix>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
            this->attenMaskGmInt, runInfo, constInfo, attenMaskInfo);
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    } else {
        attenMaskUb = dummyAttenMaskTensor;
    }

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
    LocalTensor<uint8_t> apiTmpBuffer;
    apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    LocalTensor<uint8_t> dropMaskUb;
    auto stage1CastTensor = this->stage1OutQue[0].template AllocTensor<Q_T>();
    if (unlikely(runInfo.s2LoopCount == 0)) {
        if (runInfo.s2RealSize == 128) {  // 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) { // 64 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 128 && runInfo.s2RealSize > 64) {  // 64 and 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 256 && runInfo.s2RealSize > 128) {  // 128 and 256 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 512 && runInfo.s2RealSize > 256) {  // 256 and 512 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_256_AND_LTE_512, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 1024 && runInfo.s2RealSize > 512) {  // 512 and 1024 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_512_AND_LTE_1024, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        }
    } else {
         if (runInfo.s2RealSize == 128) { // 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 64) { // 64 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 128 && runInfo.s2RealSize > 64) { // 64 and 128 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 256 && runInfo.s2RealSize > 128) { // 128 and 256 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 512 && runInfo.s2RealSize > 256) { // 256 and 512 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_256_AND_LTE_512, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        } else if (runInfo.s2RealSize <= 1024 && runInfo.s2RealSize > 512) { // 512 and 1024 is s2RealSize
            ProcessVec1Vf<T, Q_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_512_AND_LTE_1024, hasAtten, pseMode, false>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, inputTensorVec, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfo.pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), 1.0, negativeFloatScalar,
                constInfo.keepProb);
        }
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM1RES_EVENT[runInfo.taskIdMod2]);
    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    if constexpr (hasPseOuter) {
        this->pseInQue.template FreeTensor(pseUb);
    }

    // ===============DataCopy to L1===============
    this->stage1OutQue[0].template EnQue(stage1CastTensor);
    this->stage1OutQue[0].template DeQue<Q_T>();
    LocalTensor<Q_T> mm2AL1Tensor = outBufVec1.GetTensor<Q_T>();

    SetFlag<HardEvent::V_MTE3>(this->UbToL1Event);
    WaitFlag<HardEvent::V_MTE3>(this->UbToL1Event);

    if (likely(runInfo.halfS1RealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1ScmBlockTrue], stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                  (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
    }
    this->stage1OutQue[0].template FreeTensor(stage1CastTensor);
    // =======================================================
    if (runInfo.s2LoopCount != 0) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<Q_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, sumUb, maxUb);
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        SoftmaxDataCopyOut(runInfo, constInfo);
        if constexpr (isFd) {
            ComputeLogSumExpAndCopyToGm(runInfo, constInfo);
            return;
        }
        SoftmaxLseCopyOut(sumUb, maxUb, runInfo, constInfo);
    }
    return;
}

//fd
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(
    const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
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
    int64_t s1Offset = (runInfo.s1oIdx * s1BaseSize +
        constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;
    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t gmOffset = runInfo.boIdx * constInfo.n2Size * constInfo.splitKVNum * constInfo.gSize * FP32_ONE_BLOCK_SIZE +
                        runInfo.n2oIdx * constInfo.splitKVNum * constInfo.gSize * FP32_ONE_BLOCK_SIZE +
                        runInfo.flashDecodeS2Idx * constInfo.gSize * FP32_ONE_BLOCK_SIZE + mStart * FP32_ONE_BLOCK_SIZE +
                        runInfo.goIdx * constInfo.s1BaseSize * FP32_ONE_BLOCK_SIZE;
    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor =sumBrdcst.AllocTensor<float>();
    FaVectorApi::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.EnQue(sumOutTensor);
    sumBrdcst.DeQue<float>();
    DataCopy(this->softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
    this->sumBrdcst.template FreeTensor(sumOutTensor);

    // Copy max to gm
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    if constexpr (!IsSameType<Q_T, float>::value || !containAllOptionalInput) {
        LocalTensor<float> maxOutTensor = maxBrdcst.AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        maxBrdcst.EnQue(maxOutTensor);
        maxBrdcst.DeQue<float>();
        DataCopy(this->softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->maxBrdcst.template FreeTensor(maxOutTensor);
    } else {
        LocalTensor<float> maxOutTensor = sumBrdcst.AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        sumBrdcst.EnQue(maxOutTensor);
        sumBrdcst.DeQue<float>();
        DataCopy(this->softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->sumBrdcst.template FreeTensor(maxOutTensor);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SoftmaxLseCopyOut(
    LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
    const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }

    if constexpr (isInfer) {
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
        intriParams1.dstStride = 0;
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
    } else {
        return;
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::AntiquantValue(
    const RunInfo<isInfer> &runInfo, int64_t &subTaskId, bool &first, RunParamStr<isInfer> &runParam,
    Buffer<BufferType::L1> &outBufAntiValue, ConstInfo<isInfer, hasRope> &constInfo)
{
    GlobalTensor<KV_T> tempValueGm;
    if constexpr (enableKVPrefix) {
        taskParam.isPrefixLoop = ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) < constInfo.prefixLoopCount); // 判断当前s2LoopCount是否处于prefixLoopCount
        if (taskParam.isPrefixLoop) {
            tempValueGm = this->valueSharedPrefixGm;
        } else {
            tempValueGm = this->valueGm;
            GetKvByTensorList(runInfo, this->valueGm, tempValueGm, constInfo);
        }
    } else {
        tempValueGm = this->valueGm;
        GetKvByTensorList(runInfo, this->valueGm, tempValueGm, constInfo);
    }
    SetAntiqParamCommon(runInfo, runInfo.valueOffset, constInfo);
    if constexpr (VALUE_ANTIQUANT_PER_TOKEN || ANTIQUANT_PER_GROUP) {
        taskParam.isFreeAntiqParam = true;
        taskParam.isLoadAntiqParam = true;
    } else {
        if (antiquantPerTensorFlag) {
            taskParam.isLoadAntiqParam = (first);
            taskParam.isFreeAntiqParam = (first);
        } else {
            taskParam.isLoadAntiqParam = unlikely(runInfo.s2LoopCount == 0);
            taskParam.isFreeAntiqParam = unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit);
        }
    }
    valueAntiquantProcessor.ProcessBaseAPI(outBufAntiValue, tempValueGm,
                                valueAntiqScaleGm, valueAntiquantOffsetGm, blockTableGm, kvInputQue, kvOutputQue, valueAntiqScaleInputQue,
                                valueAntiqOffsetInputQue, kvAntiqMxScaleRes, taskParam, subTaskId, isBeforeHalf, runInfo.s2RealSize);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::GetKvByTensorList(
    const RunInfo<isInfer>& runInfo, GlobalTensor<KV_T>& keyValueGm, GlobalTensor<KV_T>& tempKeyValueGm,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.isKvContinuous != 0) {
        return;
    }
    ListTensorDesc keyValueListTensorDesc((__gm__ void*)keyValueGm.GetPhyAddr());
    __gm__ uint8_t* tempKeyValueGmPtr =
        (__gm__ uint8_t*)keyValueListTensorDesc.GetDataPtr<__gm__ uint8_t>(runInfo.boIdx);
    tempKeyValueGm.SetGlobalBuffer((__gm__ KV_T*)tempKeyValueGmPtr);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::UpdateVec2Result(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, LocalTensor<T> &inputTensorVec, LocalTensor<T> &expUb)
{
    float deSCalePreVValue = 1.0f;
    if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
        if (runInfo.s2LoopCount == 1) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true, false>(
                vec2ResUb, inputTensorVec, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                1.0, deSCalePreVValue);
        } else {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, inputTensorVec, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                1.0, deSCalePreVValue);
        }
    } else {
        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
        if (runInfo.s2LoopCount == 1) {
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, true, false>(
                vec2ResUb, inputTensorVec, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                1.0, deSCalePreVValue);
        } else {
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, inputTensorVec, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                1.0, deSCalePreVValue);
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ProcessVec2(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &inputTensorVec, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (runInfo.actualS2Size == 0) {
        return;
    }
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(CV_MM2RES_EVENT[runInfo.taskIdMod2]);
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    if (unlikely(runInfo.vec2S1RealSize == 0)) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[runInfo.taskIdMod2]);
        return;
    }
    LocalTensor<T> vec2ResUb = this->stage2OutQue[0].template AllocTensor<T>();
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, inputTensorVec, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        UpdateVec2Result(runInfo, vec2ResUb, inputTensorVec, expUb);
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(VC_MM2RES_EVENT[runInfo.taskIdMod2]);
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64, 1.0);
        }

        this->stage2OutQue[0].template EnQue(vec2ResUb);
        this->stage2OutQue[0].template DeQue<OUTPUT_T>();
        if constexpr (isFd) {
            Bmm2FDOut(runInfo, vec2ResUb, vec2CalcSize, constInfo);
        } else {
            Bmm2DataCopyOut(runInfo, vec2ResUb, 0, vec2CalcSize, constInfo);
        }
    }
    this->stage2OutQue[0].template FreeTensor(vec2ResUb);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::Bmm2DataCopyOut(
    const RunInfo<isInfer> &runInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
    int64_t vec2CalcSize, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<OUTPUT_T> attenOut;
    attenOut = vec2ResUb.template ReinterpretCast<OUTPUT_T>();
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (!IsSameType<Q_T, VEC2_RES_T>::value) {
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<Q_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
                InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
                    dSizeAligned64, this->negativeFloatScalar, 0.0);
            }
        }
        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, vec2S1Idx, runInfo, dSizeAligned64, constInfo);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        } else {
            PostQuant(constInfo, runInfo, attenOut, vec2ResUb, vec2S1Idx, dSizeAligned64);
            RowInvalid(vec2ResUb, vec2S1Idx, runInfo, dSizeAligned64, constInfo);
        }
        stage2OutQue[0].EnQue(attenOut);
        stage2OutQue[0].template DeQue<OUTPUT_T>();
    } else {
        stage2OutQue[runInfo.taskIdMod2].EnQue(attenOut);
        stage2OutQue[runInfo.taskIdMod2].template DeQue<OUTPUT_T>();
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    if constexpr (IsSameType<Q_T, float>::value) {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 3;
    } else if constexpr (POST_QUANT) {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 5;
    } else {
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4;
    }
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2S1RealSize;

    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
        if constexpr (isInfer) {
            if (constInfo.isPfaGS1Merge) {
                dataCopyParams.blockLen *= constInfo.gSize;
                attenOutOffset = 0;
                dataCopyParams.blockCount /= constInfo.gSize;
            } else if (constInfo.isGqa) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            attenOutOffset = constInfo.n2GDv;
            if constexpr (isInfer) {
                if (constInfo.isPfaGS1Merge) {
                    attenOutOffset = 0;
                    dataCopyParams.blockLen *= constInfo.gSize;
                    dataCopyParams.blockCount /= constInfo.gSize;
                } else if (constInfo.isGqa) {
                    attenOutOffset = constInfo.dSizeV;
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            attenOutOffset = constInfo.bN2GDv;
        }
        if constexpr (isInfer) {
            if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND)) {
                attenOutOffset = constInfo.n2GDv;
            }
        }
    }

    if constexpr (isInfer) {
        if (constInfo.isPfaGS1Merge && dSizeAligned64 - constInfo.dSizeV != 0 && (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH) || constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_TND))) {
            for (int64_t i = 0; i < runInfo.vec2S1BaseSize / constInfo.gSize; i++) {
                attenOutOffset = i * constInfo.dSizeV * constInfo.gSize * constInfo.n2Size;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
                dataCopyParams.blockCount = constInfo.gSize;
                dataCopyParams.dstStride = 0;
                DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + attenOutOffset],
                    attenOut[i * constInfo.gSize * dSizeAligned64], dataCopyParams);
            }
        } else {
            DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
                attenOut, dataCopyParams);
        }
    } else {
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ComputeOffsetForSoftmax(
        const RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
template<typename VEC2_RES_T>
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::RowInvalid(
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, const RunInfo<isInfer> &runInfo, int64_t dSizeAligned64,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer && hasAtten) {
        if (!constInfo.isRowInvalid) {
            return;
        }
        int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
        LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        bool isRowInvalidNeedUpdate = false;
        for (uint32_t i = 0; i < runInfo.vec2S1RealSize; i++) {
            float maxValue = maxTensor.GetValue(i);
            uint32_t checkValue = *(uint32_t*)&maxValue;
            if (checkValue == NEGATIVE_MIN_VALUE_FP32) {
                isRowInvalidNeedUpdate = true;
                break;
            }
        }
        if (isRowInvalidNeedUpdate) {
            if constexpr (!POST_QUANT) {
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, runInfo.vec2S1RealSize, constInfo.dSizeV, dSizeAligned64);
            } else {
                uint32_t dStride = CeilDiv(static_cast<uint32_t>(dSizeAligned64), sizeof(float));
                uint16_t dSize = CeilDiv(constInfo.dSizeV, sizeof(float)); // w8后量化的处理长度
                RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, runInfo.vec2S1RealSize, dSize, dStride);
            }
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
template <typename POSTQUANT_PARAMS_T, typename VEC2_RES_T>
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::PostQuantPerChnl(
    ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut, LocalTensor<VEC2_RES_T> &vec2ResUb,
    uint64_t perChannelQuantOffset, uint32_t gSplitSize, uint32_t s1RowCount, uint32_t splitOffset, int64_t dSizeAligned64,
    GlobalTensor<POSTQUANT_PARAMS_T> postQuantScaleGm, GlobalTensor<POSTQUANT_PARAMS_T> postQuantOffsetGm)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<POSTQUANT_PARAMS_T> copyInPadParams;
    copyInParams.blockCount = gSplitSize;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(POSTQUANT_PARAMS_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (dSizeAligned64 - constInfo.dSizeV) / (32 / sizeof(POSTQUANT_PARAMS_T)); // 32: DATA BLOCK SIZE

    LocalTensor<POSTQUANT_PARAMS_T> postQuantScaleUb =
        this->postQuantScaleQue.template AllocTensor<POSTQUANT_PARAMS_T>();
    DataCopyPad(postQuantScaleUb, postQuantScaleGm[perChannelQuantOffset], copyInParams, copyInPadParams);
    this->postQuantScaleQue.template EnQue(postQuantScaleUb);
    this->postQuantScaleQue.template DeQue<POSTQUANT_PARAMS_T>();
    if (constInfo.isPostQuantOffsetExist) {
        LocalTensor<POSTQUANT_PARAMS_T> postQuantOffsetUb =
            this->postQuantOffsetQue.template AllocTensor<POSTQUANT_PARAMS_T>();
        DataCopyPad(postQuantOffsetUb, postQuantOffsetGm[perChannelQuantOffset], copyInParams, copyInPadParams);
        this->postQuantOffsetQue.template EnQue(postQuantOffsetUb);
        this->postQuantOffsetQue.template DeQue<POSTQUANT_PARAMS_T>();

        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, postQuantOffsetUb, gSplitSize, s1RowCount,
            constInfo.dSizeV, dSizeAligned64);
        this->postQuantOffsetQue.FreeTensor(postQuantOffsetUb);
    } else {
        PostQuantPerChnlImpl<T, OUTPUT_T, POSTQUANT_PARAMS_T>(
            attenOut[splitOffset], vec2ResUb[splitOffset], postQuantScaleUb, gSplitSize, s1RowCount, constInfo.dSizeV, dSizeAligned64);
    }
    this->postQuantScaleQue.FreeTensor(postQuantScaleUb);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::PostQuant(ConstInfo<isInfer, hasRope> &constInfo,
                                                                        const RunInfo<isInfer> &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                                                        LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                        int64_t vec2S1Idx, int64_t dSizeAligned64)
{
    uint32_t s1RowCount = constInfo.isGqa ? 1U : runInfo.vec2S1RealSize; // s1=1, gS合轴, bn2分核
    uint32_t gRowCount = constInfo.isGqa ? runInfo.vec2S1RealSize : 1U;  // s1>1, bn1分核
    if (constInfo.isPostQuantPerChnl) {
        uint64_t perChannelQuantGQAOffset = runInfo.n2oIdx * constInfo.gDv + runInfo.vec2S1BaseSize * vec2S1Idx * constInfo.dSizeV +
                                            constInfo.subBlockIdx * runInfo.firstHalfS1RealSize * constInfo.dSizeV +
                                            runInfo.goIdx * constInfo.s1BaseSize * constInfo.dSizeV;
        uint64_t perChannelQuantOffset = constInfo.isGqa ?
                                            perChannelQuantGQAOffset :
                                            runInfo.n2oIdx * constInfo.gDv + runInfo.goIdx * constInfo.dSizeV;
        uint32_t gSplitSize = constInfo.isPostQuantBF16 ? (2048U / ((uint32_t)dSizeAligned64 * sizeof(bfloat16_t))) :
                                                          (2048U / ((uint32_t)dSizeAligned64 * sizeof(float)));
        gSplitSize = gSplitSize > gRowCount ? gRowCount : gSplitSize;
        uint32_t loopCount = (gRowCount + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gRowCount - (loopCount - 1) * gSplitSize;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if (i + 1 == loopCount) {
                gSplitSize = tailSplitSize;
            }
            uint32_t splitOffset = startRow * dSizeAligned64;
            if (constInfo.isPostQuantBF16) {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                                 gSplitSize, s1RowCount, splitOffset, dSizeAligned64, postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
            } else {
                PostQuantPerChnl(constInfo, attenOut, vec2ResUb, perChannelQuantOffset + startRow * constInfo.dSizeV,
                    gSplitSize, s1RowCount, splitOffset, dSizeAligned64, postQuantScaleGm, postQuantOffsetGm);
            }
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(
            attenOut, vec2ResUb, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue, runInfo.vec2S1RealSize,
            constInfo.dSizeV, dSizeAligned64);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::FDPostQuant(ConstInfo<isInfer, hasRope> &constInfo,
    LocalTensor<OUTPUT_T> &attenOut, LocalTensor<T> &accumOutLocal, uint64_t perChannelQuantOffset, uint32_t dealRowCount, uint32_t dSizeAligned64)
{
    if (constInfo.isPostQuantPerChnl) {
        if (constInfo.isPostQuantBF16) {
            PostQuantPerChnl(constInfo, attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, dSizeAligned64,
                postQuantScaleBf16Gm, postQuantOffsetBf16Gm);
        } else {
            PostQuantPerChnl(constInfo, attenOut, accumOutLocal, perChannelQuantOffset, dealRowCount, 1U, 0U, dSizeAligned64,
                postQuantScaleGm, postQuantOffsetGm);
        }
    } else {
        PostQuantPerTensorImpl<T, OUTPUT_T, true>(
            attenOut, accumOutLocal, constInfo.postQuantScaleValue, constInfo.postQuantOffsetValue, dealRowCount,
            constInfo.dSizeV, dSizeAligned64);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::Bmm2FDOut(
    const RunInfo<isInfer> &runInfo, LocalTensor<T> &vec2ResUb, int64_t vec2CalcSize,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;

    stage2OutQue[runInfo.taskIdMod2].EnQue(vec2ResUb);
    stage2OutQue[runInfo.taskIdMod2].template DeQue<OUTPUT_T>();
    attenOut = vec2ResUb;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = runInfo.halfS1RealSize;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (BYTE_BLOCK_ANTIQUANT / sizeof(T));
    dataCopyParams.dstStride = 0;

    uint32_t mStart = constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
    size_t base = (runInfo.boIdx * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV +
                  runInfo.n2oIdx * constInfo.gSize * constInfo.dSizeV) * constInfo.splitKVNum + mStart * constInfo.dSizeV +
                  runInfo.goIdx * constInfo.s1BaseSize * constInfo.dSizeV;
    DataCopyPad(this->accumOutGm[base + runInfo.flashDecodeS2Idx * constInfo.gSize * constInfo.dSizeV],
                attenOut, dataCopyParams);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::FlashDecode(
    ConstInfo<isInfer, hasRope> &constInfo, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr, int64_t s2InCurrentBatch)
{
    SyncAll();
    InitFDBuffers(constInfo);
    FlashDecodeCompute(constInfo, actualSeqQlenAddr, actualSeqKvlenAddr, s2InCurrentBatch);
}

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitFDBuffers(ConstInfo<isInfer, hasRope> &constInfo)
{
    this->tPipe->Reset();
    this->tPipe->InitBuffer(lseTmpBuff, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxMaxInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(softmaxSumInputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(FDResOutputQue, 1, bufferSizeByte32K);
    this->tPipe->InitBuffer(accumOutInputQue, 1, bufferSizeByte32K);
    if constexpr (POST_QUANT) {
        this->tPipe->InitBuffer(postQuantScaleQue, 1, BUFFER_SIZE_BYTE_32K);
        if (constInfo.isPostQuantOffsetExist) {
            this->tPipe->InitBuffer(postQuantOffsetQue, 1, BUFFER_SIZE_BYTE_32K);
        }
    }
    if (constInfo.isSoftmaxLseEnable) {
        // 8: 适配TND, 每行结果存为8个重复lse元素(32B对齐)
        this->tPipe->InitBuffer(softmaxLseQueue, 1, (s1BaseSize >> 1U) * sizeof(float) * 8);
    }
}

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::FlashDecodeCompute(
    ConstInfo<isInfer, hasRope> &constInfo, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr, int64_t s2InCurrentBatch)
{
    int64_t bIdx = constInfo.aivIdx / constInfo.n2Size;
    int64_t n2Idx = constInfo.aivIdx % constInfo.n2Size;
    int64_t batchSize = this->tilingData->inputParamsRegbase.bSize;
    if (constInfo.aivIdx >= batchSize * constInfo.n2Size) {
        return;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {  // FD Only support IFA, IFA (except TND) dosen't support actualSeqLenQ
        int64_t actualSeqLenQ;
        GetActualSeqLenQ(bIdx, actualSeqLenQ, actualSeqQlenAddr);
        if (actualSeqLenQ == 0) {
            return;
        }
    }
    int64_t actualSeqLenKV;
    GetActualSeqLenKV(bIdx, actualSeqLenKV, actualSeqKvlenAddr, s2InCurrentBatch, constInfo);
    if (actualSeqLenKV == 0) {
        return;
    }
    uint64_t attenOutOffset;
    if constexpr(layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = (bIdx == 0) ? 0 : actualSeqQlenAddr[bIdx - 1] * constInfo.n2GDv;
        attenOutOffset += n2Idx * constInfo.gDv;
    } else {
        attenOutOffset = (uint64_t)bIdx * constInfo.n2GDv + n2Idx * constInfo.gDv;
    }
    constInfo.actualCombineLoopSize = (actualSeqLenKV + constInfo.sInnerLoopSize - 1) / constInfo.sInnerLoopSize;
    CombineSplitKVRes(attenOutOffset, bIdx, n2Idx, constInfo, actualSeqQlenAddr);
}

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::GetActualSeqLenKV(
    int64_t boIdx, int64_t &actualSeqLen, __gm__ int64_t *actualSeqKvlenAddr, int64_t s2InCurrentBatch,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.isActualLenDimsKVNull) {
        actualSeqLen = s2InCurrentBatch;
    } else {
        actualSeqLen = (constInfo.actualSeqLenKVSize == 1) ? actualSeqKvlenAddr[0] : actualSeqKvlenAddr[boIdx];
        if constexpr ((layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) && (!isPa)) {
            if (boIdx > 0) {
                actualSeqLen -= actualSeqKvlenAddr[boIdx - 1];
            }
        }
    }
    if (constInfo.isKVHasLeftPadding) {
        int64_t kvLeftPaddingSize = constInfo.s2Size - actualSeqLen - constInfo.kvRightPaddingSize;
        if (kvLeftPaddingSize < 0) {
            actualSeqLen = 0;
        }
    }
}

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::GetActualSeqLenQ(
    int64_t boIdx, int64_t &actualSeqLen, __gm__ int64_t *actualSeqQlenAddr)
{
    actualSeqLen = (boIdx == 0) ? actualSeqQlenAddr[0] : actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
}

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::CombineSplitKVRes(
    uint64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx, ConstInfo<isInfer, hasRope> &constInfo,
    __gm__ int64_t *actualSeqQlenAddr)
{
    uint32_t gSplitSizeLse =
        bufferSizeByte32K / (FA_BYTE_BLOCK_ANTIQUANT * constInfo.splitKVNum); // 32K / (splitKVNum * 32B)
    uint32_t gSplitSizeAccumOut = bufferSizeByte32K / sizeof(float) / (uint32_t)dVTemplateType;
    // 取两者较小的，用来切g，保证ub够用
    uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
    if (constInfo.gSize > gSplitMax) {
        gSplitSize = (gSplitSize > gSplitMax) ? gSplitMax : gSplitSize;
    } else {
        gSplitSize = (gSplitSize > constInfo.gSize) ? constInfo.gSize : gSplitSize;
    }
    uint32_t loopCount = CeilDiv(constInfo.gSize, gSplitSize);
    uint32_t tailSplitSize = constInfo.gSize - (loopCount - 1) * gSplitSize;
    uint64_t lseOffset = 0;

    // 尾块与非尾块都使用这些ub，减少处理次数
    LocalTensor<T> lseMaxUb = lseTmpBuff.Get<T>(); // 复用内存
    uint32_t shapeArray[] = {(uint32_t)gSplitSize, fp32BaseSize};
    lseMaxUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape

    uint64_t perChannelQuantOffset = n2Idx * constInfo.dSizeV * constInfo.gSize;
    // 融合处理: 尾块处理 + 非尾块处理
    for (uint32_t i = 0; i <= loopCount - 1; i++) {
        uint32_t gSplitSizeTail = gSplitSize; // 初始化为非尾块
        if (tailSplitSize > 0 && i == (loopCount - 1)) {
            gSplitSizeTail = tailSplitSize; // 判断是否是尾块
        }
        uint32_t startRow = i * gSplitSize;
        CopyLseIn(bIdx, n2Idx, startRow, gSplitSizeTail, constInfo);
        LocalTensor<T> softmaxMaxLocal = softmaxMaxInputQue.DeQue<T>();
        // 内存复用，同时作为输出 scale 值
        LocalTensor<T> softmaxSumLocal = softmaxSumInputQue.DeQue<T>();
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint64_t batchoffset = bIdx == 0 ? 0 : actualSeqQlenAddr[bIdx - 1] * constInfo.n2G;
            lseOffset = batchoffset + n2Idx * constInfo.gSize + i * gSplitSize;
        } else {
            lseOffset = (bIdx * constInfo.n2Size + n2Idx) * constInfo.gSize + i * gSplitSize;
        }
        ComputeScaleValue(softmaxMaxLocal, softmaxSumLocal, gSplitSizeTail, lseOffset, constInfo);

        LocalTensor<T> tmp1 = lseMaxUb;
        ReduceFinalRes(bIdx, n2Idx, tmp1, softmaxSumLocal, startRow, gSplitSizeTail, constInfo);

        softmaxMaxInputQue.FreeTensor(softmaxMaxLocal);
        softmaxSumInputQue.FreeTensor(softmaxSumLocal);
        CopyFinalResOut(attenOutOffset, tmp1, startRow, gSplitSizeTail, perChannelQuantOffset, constInfo);
    }
}

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::CopyLseIn(
    uint32_t bIdx, uint32_t n2Idx, uint32_t startRow, uint32_t dealRowCount,
    ConstInfo<isInfer, hasRope> &constInfo)
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

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ComputeScaleValue(
    LocalTensor<T> lseMaxUb, LocalTensor<T> lseSumUb, uint32_t splitSize,
    uint64_t lseOffset, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<T> lseOutputUb;
    if (constInfo.isSoftmaxLseEnable) {
        lseOutputUb = softmaxLseQueue.template AllocTensor<T>();
    }
    LocalTensor<bfloat16_t> tmpSinkUb;
    ComputeScaleValue_VF(tmpSinkUb, lseMaxUb, lseSumUb, lseOutputUb, splitSize, constInfo.actualCombineLoopSize,
                         constInfo.isSoftmaxLseEnable, constInfo.learnableSinkFlag);
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

/*FD*/
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ReduceFinalRes(
    uint32_t bIdx, uint32_t n2Idx, LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
    uint32_t startRow, uint32_t dealRowCount, ConstInfo<isInfer, hasRope> &constInfo)
{
    for (uint32_t j = 0; j < constInfo.actualCombineLoopSize; ++j) {
        // 第一次，mul结果直接放到dst里
        CopyAccumOutIn(bIdx, n2Idx, j, startRow, dealRowCount, constInfo);
        LocalTensor<T> accumOutLocal = accumOutInputQue.DeQue<T>();
        ReduceFinalRes_const_VF<T, (uint32_t)dVTemplateType>(dst, lseLocal, accumOutLocal, dealRowCount, j);
        accumOutInputQue.FreeTensor(accumOutLocal);
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::CopyAccumOutIn(
    uint32_t bIdx, uint32_t n2Idx, uint32_t splitKVIndex, uint32_t startRow,
    uint32_t dealRowCount, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<T> accumOutLocal = accumOutInputQue.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = ((int64_t)dVTemplateType - constInfo.dSizeV) / 8; // 8 for align factor

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = ((int64_t)dVTemplateType - constInfo.dSizeV) % 8; // 8 for align factor
    copyInPadParams.paddingValue = 0;

    uint64_t combineAccumOutOffset = ((uint64_t)bIdx * constInfo.n2Size * constInfo.splitKVNum +
                                      n2Idx * constInfo.splitKVNum + splitKVIndex) *
                                         constInfo.gSize * constInfo.dSizeV +
                                     startRow * constInfo.dSizeV;
    DataCopyPad(accumOutLocal, this->accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    accumOutInputQue.EnQue(accumOutLocal);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::CopyFinalResOut(
    uint64_t attenOutOffset, LocalTensor<T> &accumOutLocal, uint32_t startRow,
    uint32_t dealRowCount, uint64_t perChannelQuantOffset, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = FDResOutputQue.AllocTensor<OUTPUT_T>();
    uint32_t dSizeAligned64 = (uint32_t)dVTemplateType;
    uint32_t shapeArray[] = {(uint32_t)dealRowCount, dSizeAligned64};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND)); // 2 for shape
    if constexpr (POST_QUANT) {
        FDPostQuant(constInfo, tmpBmm2ResCastTensor, accumOutLocal,
        perChannelQuantOffset + startRow * constInfo.dSizeV, dealRowCount, dSizeAligned64);
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * dSizeAligned64);
    }

    FDResOutputQue.EnQue(tmpBmm2ResCastTensor);
    FDResOutputQue.DeQue<OUTPUT_T>();
    ReduceFDDataCopyOut(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, dSizeAligned64,
                        constInfo.dSizeV);
    FDResOutputQue.FreeTensor(tmpBmm2ResCastTensor);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::ReduceFDDataCopyOut(
    uint64_t attenOutOffset, LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (FA_BYTE_BLOCK_ANTIQUANT / sizeof(OUTPUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(this->attentionOutGm[attenOutOffset + startRow * actualColumnCount], attenOutUb, dataCopyParams);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (constInfo.softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VALUE_FP32, softmaxShapeInfo);
        if (!res) {
            constInfo.softMaxCheckRes = false;
        } else {
            if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SoftmaxInvalidLineCheck(
    LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
        T maxValue = maxUb.GetValue(i);
        uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
        if (checkValue == negativeIntScalar) {
            isUpdateNeedCheck = true;
            break;
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer) {
        return;
    }
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = runInfo.s1SizeAcc * constInfo.n2G;
        n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset = (runInfo.s1oIdx * s1BaseSize +
        constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t gmOffset = (bOffset + n2Offset + gOffset + s1Offset) * fp32BaseSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;

    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> sumOutTensor = sumBrdcst.AllocTensor<float>();
    FaVectorApi::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
    sumBrdcst.EnQue(sumOutTensor);
    sumBrdcst.DeQue<float>();
    DataCopy(this->softmaxSumGm[gmOffset], sumOutTensor, calculateSize);
    this->sumBrdcst.template FreeTensor(sumOutTensor);

    // Copy max to gm
    LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    if constexpr (!IsSameType<Q_T, float>::value || !containAllOptionalInput) {
        LocalTensor<float> maxOutTensor = maxBrdcst.AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        maxBrdcst.EnQue(maxOutTensor);
        maxBrdcst.DeQue<float>();
        DataCopy(this->softmaxMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->maxBrdcst.template FreeTensor(maxOutTensor);
    } else {
        LocalTensor<float> maxOutTensor = sumBrdcst.AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        sumBrdcst.EnQue(maxOutTensor);
        sumBrdcst.DeQue<float>();
        DataCopy(this->softmaxMaxGm[gmOffset], maxOutTensor, calculateSize);
        this->sumBrdcst.template FreeTensor(maxOutTensor);
    }
}


ANTIQUANT_TEMPLATES_DEF
class FABlockVecAntiquantDummy {
public:
    __aicore__ inline void InitVecBlock(__gm__ uint8_t *value, __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *pse, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *workspace,
        const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        TPipe *pipe, AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {}
    __aicore__ inline void SendCrossCoreFlag() {}
    __aicore__ inline void setConstAntiTaskParam(ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void ClearOutput(ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitQuant(__gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
        __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
        __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
        __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void FlashDecode(ConstInfo<isInfer, hasRope> &constInfo, __gm__ int64_t *seqQlenAddr,
        __gm__ int64_t *seqKVlenAddr, int64_t s2InCurrentBatch) {}
};

}
#endif
