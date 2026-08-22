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
 * \file flash_attention_score_block_vec_base_fullquant.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_FULLQUANT_H_
#include "util_regbase.h"
#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "flash_attention_score_tiling_regbase.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF_BASE
class FABlockVecBaseFullquant {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = s1BaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = s1BaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = s1BaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = s1BaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool isMlaFullQuant = (isFp8 || isInt8) && hasRope;
    static constexpr bool isMxfp8FullQuant = s1BaseSize == 128 && s2BaseSize == 512;
    static constexpr bool useDn = IsDn(((IsSameType<INPUT_T, float>::value) || isFp8), ((isFp8 && s2BaseSize == 256) || isMxfp8FullQuant), pseMode, hasAtten, hasDrop,
                                       s1BaseSize == 64, dTemplateType, hasRope, enableKVPrefix, isInfer, IsSameType<INPUT_T, hifloat8_t>::value);
    static constexpr bool useNz = IsSameType<INPUT_T, hifloat8_t>::value && !isInfer;
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
                                        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten && hasDrop;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr TPosition bmm2OutPos = GetC2Position(
        dVTemplateType, UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop, hasRope,
        s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64), isMlaFullQuant);
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr bool isW8In = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value ||
                                  IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using pseShiftW8InType = typename AscendC::Conditional<isInfer, half, OUTPUT_T>::type;
    using pseShiftType = typename AscendC::Conditional<isW8In, pseShiftW8InType, INPUT_T>::type;
    static constexpr int64_t FP8_QUANT_KV_BLOCK_SIZE = 256;

    /*HIFLOAT8场景 K_BLOCK_SIZE = 256 V_BLOCK_SIZE = 512*/
    static constexpr int64_t FP8_QUANT_K_BLOCK_SIZE = 256;
    static constexpr int64_t FP8_QUANT_V_BLOCK_SIZE = 512;
    // ==================== Functions ======================
    __aicore__ inline FABlockVecBaseFullquant() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            GetDerived()->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
            attenMaskInfoPtr = &attenMaskInfo;
            pseInfoPtr = &pseInfo;
        }
    }
    __aicore__ inline void InitCommonGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
        __gm__ uint8_t * learnableSink, __gm__ uint8_t *&workspace, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, int32_t subLoop = 0);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    TPipe *tPipe;
    const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;

    /* =====================可选GM变量==================== */
    __gm__ uint8_t *pseSlope;
    using pseGmType = typename std::conditional<hasPse, GlobalTensor<pseShiftType>, int8_t>::type;
    pseGmType pseGm;
    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;
    using quantGmType = typename std::conditional<(isFp8 || isInt8), GlobalTensor<float>, int8_t>::type;
    quantGmType deScaleQGm;
    quantGmType deScaleKGm;
    quantGmType deScaleVGm;
    quantGmType pScaleGm;
    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;
    vec2ResGmType vec2ResGm[3];
    GlobalTensor<INPUT_T> sinkGm;

    /* =====================V侧UB变量==================== */
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TQue<QuePosition::VECIN, 1> pseInQue;
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[3];
    TBuf<> softmaxSumBuf[3];
    TBuf<> softmaxExpBuf[3];
    TBuf<> pScaleBuf[3];
    TBuf<> preLoopMaxBuf;
    TBuf<> preLoopSumBuf;
    TBuf<> firstLoopSumBuf;
    TQue<QuePosition::VECIN, 1> queryScaleQue[2];
    TBuf<> vselrIndexesBuf[4];
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    /* =================初始化后不变的信息================= */
    PseInfo *pseInfoPtr;
    AttenMaskInfo *attenMaskInfoPtr;
    T negativeFloatScalar;
    T positiveFloatScalar;
    // Bmm2阶段subblock在Gm上的偏移
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
protected:
    __aicore__ inline void BroadCastAndCopyOut(RunInfo<isInfer> &runInfo, GlobalTensor<T> &sumGm,
                                               GlobalTensor<T> &maxGm, int64_t gmOffset, int64_t calculateSize);

    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = INPUT_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
private:
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec1DnMxfp8(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, int32_t subLoop);
    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec1Nz(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, int64_t vec2CalcSize);

    __aicore__ inline bool SoftmaxInvalidLineCheck(LocalTensor<T> &maxUb, uint32_t negativeIntScalar, SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);

    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, int64_t dSizeAligned64);
    __aicore__ inline void MlaAttenMaskCopyIn(TQue<QuePosition::VECIN, 1> &attenMaskInQue,
        TQue<QuePosition::VECIN, 1> &attenMaskInQuePre, GlobalTensor<uint8_t> &srcTensor,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, AttenMaskInfo &attenMaskInfo);
    __aicore__ inline void MlaBoolCopyInRegbase(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
        int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize,
        ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo);
    __aicore__ inline void MlaTransposeDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
 	    LocalTensor<OUTPUT_T> &attenOut);
    __aicore__ inline void MlaTranspose2DataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<OUTPUT_T> &attenOut);
};

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::InitCommonGlobalBuffer(
    __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *learnableSink, __gm__ uint8_t *&workspace, ConstInfo<isInfer, hasRope> &constInfo)
{
    if ASCEND_IS_AIV {
        if constexpr (hasPse) {
            pseGm.SetGlobalBuffer((__gm__ pseShiftType *)pse);
            pseSlope = pse;
        }
        if constexpr (isFp8 || isInt8) {
            deScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
            deScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
            deScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
            if (pScale != nullptr) {
                pScaleGm.SetGlobalBuffer((__gm__ float *)pScale);
                constInfo.pScale = this->pScaleGm.GetValue(0);
            } else {
                constInfo.pScale = 1.0f;
            }
        }

        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (!bmm2Write2Ub) {
            int64_t bmm2ResBlock = tilingData->inputParamsRegbase.dSizeV;
            if constexpr (splitD) {
                bmm2ResBlock = (int64_t)dVTemplateType;
            }
            int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
            int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
            int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小， Gm上的数据按照实际的dSize存储
            if constexpr (splitD) {
                vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
                vec2ResGm[0].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[1].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
                vec2ResGm[2].SetGlobalBuffer((__gm__ T *)(workspace));
                workspace += vec2Offset;
            }
            bmm2SubBlockOffset = constInfo.subBlockIdx * mm2ResultSize >> 1; // s1BaseSize一定可以被2整除
        }
        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
            constInfo.learnableSinkFlag = true;
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo, int32_t subLoop)
{
    if constexpr (isMxfp8FullQuant) {
        ProcessVec1DnMxfp8(outputBuf, bmm1ResBuf, runInfo, constInfo, subLoop);
    } else if constexpr (useDn) {
        ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo, constInfo);
    } else if constexpr (useNz) {
        ProcessVec1Nz(outputBuf, bmm1ResBuf, runInfo, constInfo);
    } else {
        ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo, constInfo);
    }
}

// =================================Private Functions=================================
TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec1Nz(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<pseShiftType> pseUb;
    float slopes = 0.0f;
    float posShift = 0.0f;
    LocalTensor<uint8_t> attenMaskUb;
    LocalTensor<uint8_t> dropMaskUb;
    LocalTensor<half> pScaleUb;
    LocalTensor<half> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<half>();
    LocalTensor<half> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<half>();
    LocalTensor<half> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<half>();
    LocalTensor<uint8_t> apiTmpBuffer;
    apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    float descaleQK = 1.0;
    float deSCaleKValue = 1.0;

    LocalTensor<half> mmRes = bmm1ResBuf.template GetTensor<half>();
    auto stage1CastTensor = bmm1ResBuf.template GetTensor<INPUT_T>();

    if (runInfo.s2LoopCount == 0) {
        if (runInfo.s2RealSize <= 256) {
            ProcessVec1Vf<half, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_256, hasAtten, pseMode, hasDrop, false, false, useNz>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, pScaleUb, deSCaleKValue, constInfo.pScale);
        } else {
            ProcessVec1Vf<half, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_256_AND_LTE_512, hasAtten, pseMode, hasDrop, false, false, useNz>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, pScaleUb, deSCaleKValue, constInfo.pScale);
        }
    } else {
        if (runInfo.s2RealSize <= 256) {
            ProcessVec1Vf<half, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_256, hasAtten, pseMode, hasDrop, false, false, useNz>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, pScaleUb, deSCaleKValue, constInfo.pScale);
        } else {
            ProcessVec1Vf<half, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_256_AND_LTE_512, hasAtten, pseMode, hasDrop, false, false, useNz>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, pScaleUb, deSCaleKValue, constInfo.pScale);
        }
    }

    // ===================DataCopy to L1 ====================
    event_t v2Mte3Id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(v2Mte3Id);
    WaitFlag<HardEvent::V_MTE3>(v2Mte3Id);

    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    if (likely(runInfo.halfS1RealSize != 0)) {
        int32_t s2RealSizeAlign32 = (runInfo.s2RealSize + 31) >> 5 << 5;
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * (runInfo.s1RealSize - runInfo.halfS1RealSize)], stage1CastTensor,
            {(uint16_t)(s2RealSizeAlign32 / 32), (uint16_t)(runInfo.halfS1RealSize), (uint16_t)((s1BaseSize >> 1) - runInfo.halfS1RealSize),
            (uint16_t)(runInfo.halfS1RealSize)});
    }
    bmm1ResBuf.SetCrossCore<true>();
    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0) {
        UpdateExpSumAndExpMax<half, useNz>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        GetDerived()->SoftmaxDataCopyOutFp8(runInfo, constInfo, sumUb, maxUb);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec1DnMxfp8(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo, int32_t subLoop)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (isFp8 && hasAtten) {
        AttenMaskCopyInDn<hasAtten, hasRope, isInfer, isMxfp8FullQuant>(this->attenMaskInQue[0], this->attenMaskGmInt,
                                    runInfo, constInfo, *attenMaskInfoPtr,
                                    (runInfo.s2EndIdx - s1BaseSize < s2BaseSize) ||
                                    ((runInfo.s2EndIdx - s1BaseSize >= s2BaseSize) &&
                                     (runInfo.s2LoopCount == runInfo.s2LoopLimit)), subLoop);
        attenMaskUb = this->attenMaskInQue[0].template DeQue<uint8_t>();
    }

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];
    LocalTensor<float> preLoopMaxUb = this->preLoopMaxBuf.template Get<float>()[0];
    LocalTensor<float> preLoopSumUb = this->preLoopSumBuf.template Get<float>()[0];
    LocalTensor<float> firstLoopSumUb = this->firstLoopSumBuf.template Get<float>()[0];
    
    auto expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[0];
    int64_t stage1Offset = runInfo.taskIdMod2;
    float descaleQK = 1.0;
    
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    static constexpr int32_t mxfp8s2BaseSize = s2BaseSize / 2;

    uint32_t mxfp8s2RealSize = runInfo.s2RealSize;
    if (runInfo.s2RealSize > 256) {
        mxfp8s2RealSize = subLoop == 0 ? mxfp8s2BaseSize : runInfo.s2RealSize - mxfp8s2BaseSize;
    }

    if (unlikely(runInfo.s2LoopCount == 0)) {
        if constexpr (isFp8) {
            if (attenMaskInfoPtr->computeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, hasAtten, mxfp8s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                    ((runInfo.s1RealSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.s2AlignedSize / 2, mxfp8s2RealSize,
                    static_cast<T>(constInfo.scaleValue), descaleQK, constInfo.pScale,
                    negativeFloatScalar, constInfo.keepProb, true, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                    subLoop);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, false, false, mxfp8s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                    ((runInfo.s1RealSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.s2AlignedSize / 2, mxfp8s2RealSize,
                    static_cast<T>(constInfo.scaleValue), descaleQK, constInfo.pScale,
                    negativeFloatScalar, constInfo.keepProb, false, preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                    subLoop);
            }
        }
    } else {
        if constexpr (isFp8) {
            if (attenMaskInfoPtr->computeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, hasAtten, mxfp8s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                    ((runInfo.s1RealSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.s2AlignedSize  / 2, mxfp8s2RealSize,
                    static_cast<T>(constInfo.scaleValue), descaleQK, constInfo.pScale,
                    negativeFloatScalar, constInfo.keepProb, true, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop);
            } else {
                FaVectorApi::ProcessVec1VfDnMxfp8<T, INPUT_T, true, false, mxfp8s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                    ((runInfo.s1RealSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.s2AlignedSize  / 2, mxfp8s2RealSize,
                    static_cast<T>(constInfo.scaleValue), descaleQK, constInfo.pScale,
                    negativeFloatScalar, constInfo.keepProb, false, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop);
            }
        }
    }
    bmm1ResBuf.SetCrossCore();

    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    //-------------------------Data copy to l1-------------------------
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

    if constexpr (isFp8) {
        int64_t subLoopOffset = mxfp8s2BaseSize * runInfo.s1RealSizeAlign32 * subLoop;
        constexpr uint16_t elementSize = 32;
        uint32_t singleProcessSOuterSize = runInfo.s1RealSizeAlign32 >> 1;
        uint32_t actCopyCount = (singleProcessSOuterSize + elementSize - 1) / elementSize;
        uint32_t actVec0Align32 = runInfo.s1RealSizeAlign32 >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.s2RealSize + 63) >> 6) << 6);
        for (uint32_t i = 0; i < actCopyCount; i++) {
            uint32_t dstOffset = 32 * mxfp8s2BaseSize * i + subLoopOffset;
            if (constInfo.subBlockIdx == 1) {
                dstOffset += mxfp8s2BaseSize * actVec0Align32;
            }
            uint32_t srcOffset = i * (65 << 5);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor[srcOffset], {4, mxfp8s2BaseSize / 4, mxfp8s2BaseSize / 4 + 2, 0});
        }
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    if (((runInfo.s2RealSize > 256) && (subLoop % 2 == 1)) || (runInfo.s2RealSize <= 256)) {
        outputBuf.SetCrossCore();
    }

    if constexpr (isFp8 && hasAtten) {
        this->attenMaskInQue[0].template FreeTensor(attenMaskUb);
    }

    //-----------------------------------------------------------------
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        GetDerived()->SoftmaxDataCopyOut(runInfo, constInfo, sumUb, maxUb);
    }
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec1Dn(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (isFp8 && hasAtten) {
        AttenMaskCopyInDn<hasAtten>(this->attenMaskInQue[0], this->attenMaskGmInt,
                                    runInfo, constInfo, *attenMaskInfoPtr,
                                    (runInfo.s2EndIdx - s1BaseSize < s2BaseSize) ||
                                    ((runInfo.s2EndIdx - s1BaseSize >= s2BaseSize) &&
                                     (runInfo.s2LoopCount == runInfo.s2LoopLimit)));
        attenMaskUb = this->attenMaskInQue[0].template DeQue<uint8_t>();
    }
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];

    auto expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[0];
    int64_t stage1Offset = runInfo.taskIdMod2;
     
    float descaleQK = 1.0;
    if constexpr (isFp8) {
        int64_t deScaleQOffset = 0;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            int64_t s1BlockCnt = constInfo.t1Size / FP8_QUANT_BLOCK_SIZE + constInfo.bSize; // Q的反量化scale内容在Gm中的偏移 原始shape为 [N2, G,T // 128 + B, 1]
            int64_t s2BlockCnt = constInfo.t2Size / FP8_QUANT_KV_BLOCK_SIZE + constInfo.bSize; // KV的反量化scale内容在Gm中的偏移 原始shape为 [N2, G, T // 256 + B, 1]
            deScaleQOffset = runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
                                    runInfo.goIdx * s1BlockCnt + runInfo.s1ScaleNumAcc + runInfo.s1oIdx;
            runInfo.deScaleKvOffset = runInfo.n2oIdx * s2BlockCnt +
                                    runInfo.s2ScaleNumAcc + runInfo.s2LoopCount; // 8 ：按照256分块计算deScaleKv偏移
        } else {
            int64_t s1BlockCnt = CeilDiv(constInfo.s1Size, FP8_QUANT_BLOCK_SIZE); // Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1]
            int64_t s2BlockCnt = CeilDiv(constInfo.s2Size, FP8_QUANT_KV_BLOCK_SIZE); // KV的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S2, 256), 1]
            deScaleQOffset = runInfo.boIdx * constInfo.n2G * s1BlockCnt +
                                    runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
                                    runInfo.goIdx * s1BlockCnt + runInfo.s1oIdx;
            runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                    runInfo.n2oIdx * s2BlockCnt +
                                    (runInfo.s2StartIdx >> 8) + runInfo.s2LoopCount; // 8 ：按照256分块计算deScaleKv偏移
        }
        float deSCaleQValue = this->deScaleQGm.GetValue(deScaleQOffset);
        float deSCaleKValue = this->deScaleKGm.GetValue(runInfo.deScaleKvOffset);  // [0-128)
        descaleQK = deSCaleQValue * deSCaleKValue;
    }
 
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    if (unlikely(runInfo.s2LoopCount == 0)) {
        if constexpr (isFp8) {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, hasAtten, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                ((runInfo.s1RealSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.s2AlignedSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), descaleQK,
                negativeFloatScalar, constInfo.keepProb, runInfo.s2EndIdx - s1BaseSize < s2BaseSize, constInfo.pScale);
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.s1RealSizeAlign32 >> 1, runInfo.s2AlignedSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), descaleQK,
                negativeFloatScalar, constInfo.keepProb, false);
        }
    } else {
        if constexpr (isFp8) {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, hasAtten, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                ((runInfo.s1RealSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.s2AlignedSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), descaleQK,
                negativeFloatScalar, constInfo.keepProb, runInfo.s2LoopCount == runInfo.s2LoopLimit, constInfo.pScale);
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.s1RealSizeAlign32 >> 1, runInfo.s2AlignedSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.scaleValue), descaleQK,
                negativeFloatScalar, constInfo.keepProb, false);
        }
    }
    bmm1ResBuf.SetCrossCore();
    if constexpr (isFp8 && hasAtten) {
        this->attenMaskInQue[0].template FreeTensor(attenMaskUb);
    }
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    //-------------------------Data copy to l1-------------------------
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

    if constexpr (isFp8) {
        // 按照64对齐搬运
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * ((runInfo.s2RealSize + 63) >> 6 << 6)],
            stage1CastTensor, {static_cast<uint16_t>((runInfo.s2RealSize + 63) >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * ((runInfo.s2RealSize + 63) >> 6 << 6) +
            ((runInfo.s2RealSize + 63) >> 6 << 6) * 32], stage1CastTensor[65 << 5], {static_cast<uint16_t>((runInfo.s2RealSize + 63) >> 6), 64, 66, 0});
    } else {
        if (runInfo.s2RealSize > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.s2AlignedSize], stage1CastTensor,
                {vec1S2CopyCountDn, vec1S2CopyLenDn, 1, static_cast<uint16_t>(runInfo.s2AlignedSize - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.s2AlignedSize + vec1S2strideDn],
                stage1CastTensor[vec1ResOffsetDn],
                {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.s2AlignedSize - vec1S2CopyLenDn),
                static_cast<uint16_t>(s2BaseSize - runInfo.s2AlignedSize + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.s2AlignedSize], stage1CastTensor,
                {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.s2AlignedSize),
                static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.s2AlignedSize + 1), 0});
        }
    }
 
    outputBuf.SetCrossCore();
    //-----------------------------------------------------------------
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        GetDerived()->SoftmaxDataCopyOut(runInfo, constInfo, sumUb, maxUb);
    }
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline bool FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::SoftmaxInvalidLineCheck(
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

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::InvalidLineProcess(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb)
{
    if (constInfo.softMaxCheckRes) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1),
            static_cast<uint32_t>(runInfo.halfS1RealSize), static_cast<uint32_t>(1)};
        bool res = SoftmaxInvalidLineCheck(maxUb, NEGATIVE_MIN_VALUE_FP32, softmaxShapeInfo);
        if (!res) {
            constInfo.softMaxCheckRes = false;
        } else {
            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
                SoftmaxSumUpdate<T>(sumUb, maxUb, runInfo.halfS1RealSize, this->negativeFloatScalar,
                    this->positiveFloatScalar);
            }
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::BroadCastAndCopyOut(
    RunInfo<isInfer> &runInfo, GlobalTensor<T> &sumGm,
    GlobalTensor<T> &maxGm, int64_t gmOffset, int64_t calculateSize)
{
    // Copy sum to gm
    LocalTensor<float> sumTensor = softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    if constexpr (useNz) {
        event_t v2Mte3Id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(v2Mte3Id);
        WaitFlag<HardEvent::V_MTE3>(v2Mte3Id);
        DataCopy(sumGm[gmOffset], sumTensor, calculateSize);
    } else {
        LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumTensor, runInfo.halfS1RealSize);
        sumBrdcst.template EnQue(sumOutTensor);
        sumBrdcst.template DeQue<float>();
        DataCopy(sumGm[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst.template FreeTensor(sumOutTensor);
    }

    // Copy max to gm
    LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
    if constexpr (useNz) {
        LocalTensor<half> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<half>();
        Cast(maxOutTensor, maxTensor, RoundMode::CAST_NONE, calculateSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(maxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    } else {
        LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxTensor, runInfo.halfS1RealSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(maxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::MlaAttenMaskCopyIn(
    TQue<QuePosition::VECIN, 1> &attenMaskInQue, TQue<QuePosition::VECIN, 1> &attenMaskInQuePre, GlobalTensor<uint8_t> &srcTensor,
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, AttenMaskInfo &attenMaskInfo)
{
    if constexpr (hasAtten && isMlaFullQuant) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        attenMaskInfo.attenMaskS1Offset = 0;    // 0: 默认值
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten, enableKVPrefix, isFd, hasRope, isInfer, dTemplateType>(runInfo, constInfo, attenMaskInfo);
        this->MlaBoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
            attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo, runInfo);
        attenMaskInQue.template EnQue(attenMaskUb);
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            LocalTensor<uint8_t> attenMaskUbPre = attenMaskInQuePre.template AllocTensor<uint8_t>();
            this->MlaBoolCopyInRegbase(attenMaskUbPre, srcTensor, attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize, runInfo.s2RealSize,
                attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo, runInfo);
            attenMaskInQuePre.template EnQue(attenMaskUbPre);
            attenMaskInQuePre.template DeQue<uint8_t>();
            attenMaskInQue.template DeQue<uint8_t>();
            MergeBandModeMask<hasAtten>(attenMaskUbPre, attenMaskUb, runInfo.halfS1RealSize, constInfo.s2BaseSize);
            attenMaskInQuePre.template FreeTensor(attenMaskUbPre);
            attenMaskInQue.template EnQue(attenMaskUb);
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::MlaBoolCopyInRegbase(LocalTensor<uint8_t> &dstTensor,
    GlobalTensor<uint8_t> &srcTensor, int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize,
    ConstInfo<isInfer, hasRope> &constInfo, RunInfo<isInfer> &runInfo)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }

    if constexpr (isInfer == false) {
        return;
    }

    uint32_t neededSouterSize = s1Size;
    uint32_t s2BlockLenAlign = (s2Size + blockBytesU8 - 1) / blockBytesU8;
    DataCopyExtParams intriParams;
    intriParams.blockCount = s1Size;
    intriParams.blockLen = s2Size;
    intriParams.srcStride = totalS2Size - s2Size;
    intriParams.dstStride = s2BaseSize / blockBytesU8 - s2BlockLenAlign;
    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    padParams.rightPadding = 0;
    auto mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    if ((hasRope && (dTemplateType == DTemplateType::Aligned576)) &&
        (constInfo.layoutType != static_cast<uint32_t>(LayOutTypeEnum::LAYOUT_BNSD))) {
        intriParams.blockCount = 1;
        if (isMlaFullQuant && constInfo.gSize < 32) {
            for (int i = 0; i < s1Size; i++) {
                DataCopyPad(dstTensor[i * s2BaseSize], srcTensor[srcOffset], intriParams, padParams);
                // 下一行出现跨 G 时
                if ((runInfo.sOuterOffset + i + 1) % constInfo.gSize == 0) {
                    srcOffset += totalS2Size;
                }
            }
        } else {
            DataCopyPad(dstTensor, srcTensor[srcOffset], intriParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        return;
    } else if ((hasRope && (dTemplateType == DTemplateType::Aligned576)) &&
               (constInfo.layoutType == static_cast<uint32_t>(LayOutTypeEnum::LAYOUT_BNSD))) {
        int64_t s1OfMla = runInfo.actualS1Size / constInfo.gSize;
        int64_t firstS1Start = runInfo.sOuterOffset % s1OfMla;
        intriParams.blockCount = (s1OfMla - firstS1Start) < neededSouterSize ?
            (s1OfMla - firstS1Start) : neededSouterSize;
        DataCopyPad(dstTensor, srcTensor[srcOffset + firstS1Start * totalS2Size], intriParams, padParams);
        if (firstS1Start != 0 && (s1OfMla - firstS1Start) < neededSouterSize) {
            intriParams.blockCount = firstS1Start < (neededSouterSize - s1OfMla + firstS1Start) ?
                firstS1Start : (neededSouterSize - s1OfMla + firstS1Start);
            DataCopyPad(dstTensor[(s1OfMla - firstS1Start) * s2BaseSize],
                srcTensor[srcOffset], intriParams, padParams);
        }

        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        for (int64_t i = 1; (i + 1) * s1OfMla <= neededSouterSize; i++) {
            DataCopy(dstTensor[s1OfMla * i * s2BaseSize], dstTensor, s1OfMla * s2BaseSize);
        }
        if (neededSouterSize > s1OfMla && neededSouterSize % s1OfMla != 0) {
            DataCopy(dstTensor[s1OfMla * (neededSouterSize / s1OfMla) * s2BaseSize], dstTensor,
                (neededSouterSize % s1OfMla) * s2BaseSize);
        }
        return;
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec1Nd(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<pseShiftType> pseUb;
    if constexpr (hasPseOuter == true) {
        PseCopyIn<T, pseShiftType, hasPseOuter>(this->pseInQue, this->pseGm, runInfo, constInfo, *pseInfoPtr);
        pseUb = this->pseInQue.template DeQue<pseShiftType>();
    }
    float slopes = 0.0f;
    float posShift = 0.0f;
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
            if (this->tilingData->inputParamsRegbase.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL) &&
                runInfo.boIdx != 0) {
                pseInfoPtr->qStartIdx = 0;
                pseInfoPtr->kvStartIdx = 0;
            }
        }
        ComputeInnerPseOffset<T, INPUT_T, hasPse>(slopes, posShift, runInfo, constInfo, *pseInfoPtr, this->pseSlope);
    }

    LocalTensor<uint8_t> attenMaskUb;
    if constexpr (hasAtten == true) {
        if constexpr (isMlaFullQuant) {
            this->MlaAttenMaskCopyIn(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
                this->attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr);
        } else {
            AttenMaskCopyIn<hasAtten, isFd, enableKVPrefix>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
                this->attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr);
        }
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    }
    LocalTensor<uint8_t> dropMaskUb;
    GetDerived()->GenerateDropoutMask(runInfo, constInfo, dropMaskUb);

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
    LocalTensor<T> pScaleUb;
    LocalTensor<T> queryScaleUb;
    LocalTensor<uint8_t> apiTmpBuffer;
    if constexpr (IsSameType<INPUT_T, float>::value) {
        apiTmpBuffer = this->sumBrdcst.template AllocTensor<uint8_t>();
    } else {
        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    }

    int64_t stage1Offset = 0;
    if constexpr (!IsSameType<INPUT_T, float>::value) {
        stage1Offset = runInfo.taskIdMod2;
    }
    float descaleQK = 1.0;
    float deSCaleKValue = 1.0;
    if constexpr (isFp8 || isInt8) {
        int64_t deScaleQOffset = 0;
        int64_t deScaleKvOffset = 0;
        if constexpr (isMlaFullQuant) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                queryScaleUb = this->queryScaleQue[runInfo.multiCoreIdxMod2].template AllocTensor<T>();
                deScaleQOffset = runInfo.queryOffset / constInfo.dSize + constInfo.subBlockIdx * runInfo.halfS1RealSize;
                if (runInfo.halfS1RealSize >= 8) {  // datacopy参数count限制32字节对齐，8*sizeof(float)=32
                    DataCopy(queryScaleUb, this->deScaleQGm[deScaleQOffset], (runInfo.halfS1RealSize + 7) >> 3 << 3);
                } else if (runInfo.halfS1RealSize > 0) {
                    DataCopyExtParams dataCopyExtParams;
                    dataCopyExtParams.blockCount = 1;
                    dataCopyExtParams.blockLen = runInfo.halfS1RealSize * sizeof(float);
                    dataCopyExtParams.dstStride = 0;
                    dataCopyExtParams.srcStride = 0;
                    DataCopyPadExtParams<float> dataCopyPadParams;
                    DataCopyPad(queryScaleUb, this->deScaleQGm[deScaleQOffset], dataCopyExtParams, dataCopyPadParams);
                }
                this->queryScaleQue[runInfo.taskIdMod2].template EnQue(queryScaleUb);
                this->queryScaleQue[runInfo.taskIdMod2].template DeQue<T>();
            }
            pScaleUb = this->pScaleBuf[runInfo.taskIdMod3].template Get<T>();
        } else {
            // 训练模板：128*128 推理模板：128*256
            int64_t s1BlockCnt = CeilDiv(constInfo.s1Size, FP8_QUANT_BLOCK_SIZE);
            int64_t s2BlockCnt = CeilDiv(constInfo.s2Size, FP8_QUANT_KV_BLOCK_SIZE);
            /* Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1] */
            deScaleQOffset = runInfo.boIdx * constInfo.n2G * s1BlockCnt +
                                    runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
                                    runInfo.goIdx * s1BlockCnt + runInfo.s1oIdx;
            if constexpr (isInfer) {
                runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                        runInfo.n2oIdx * s2BlockCnt +
                                        (runInfo.s2StartIdx >> 8) + ((runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) >> 1); // 8 ：按照256分块计算deScaleKv偏移
            } else {
                runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                    runInfo.n2oIdx * s2BlockCnt +
                                    (runInfo.s2StartIdx >> 8) + (runInfo.s2LoopCount >> 1); // 8 ：按照256分块计算deScaleKv偏移
            }
            deScaleKvOffset = runInfo.deScaleKvOffset;
        }
        float deSCaleQValue = this->deScaleQGm.GetValue(deScaleQOffset);
        deSCaleKValue = this->deScaleKGm.GetValue(deScaleKvOffset);
        descaleQK = deSCaleQValue * deSCaleKValue;
    }

    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    constexpr bool useMlaSgdFlag = ((isMlaFullQuant) && layout != LayOutTypeEnum::LAYOUT_BNSD);
    bool isMlaFullQuantGsize32 = ((isMlaFullQuant) && layout != LayOutTypeEnum::LAYOUT_BNSD && constInfo.gSize >= 32);
    if (runInfo.s2LoopCount == 0) {
        if (likely(runInfo.s2RealSize == 128)) {
            if (isMlaFullQuantGsize32) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop, true, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            } else {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop, false, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            }
        } else if (runInfo.s2RealSize <= 64) {
            if (isMlaFullQuantGsize32) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop, true, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            } else {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop, false, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            }
        } else if (runInfo.s2RealSize < 128) {
            if (isMlaFullQuantGsize32) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop, true, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            } else {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop, false, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            }
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, hasDrop>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                    pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    constInfo.keepProb);
            }
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) {
            if (isMlaFullQuantGsize32) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop, true, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            } else {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop, false, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            }
        } else if (runInfo.s2RealSize <= 64) {
            if (isMlaFullQuantGsize32) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop, true, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            } else {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode, hasDrop, false, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            }
        } else if (runInfo.s2RealSize < 128) {
            if (isMlaFullQuantGsize32) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop, true, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            } else {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten, pseMode, hasDrop, false, isMlaFullQuant>(
                stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                constInfo.keepProb, queryScaleUb, deSCaleKValue);
            }
        } else {
            if constexpr (s2BaseSize == 256) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, s1BaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten, pseMode, hasDrop>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb,
                    attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.halfS1RealSize, runInfo.s2RealSize,
                    pseInfoPtr->pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    constInfo.keepProb);
            }
        }
    }
    bmm1ResBuf.SetCrossCore();
    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    if constexpr (hasPseOuter) {
        this->pseInQue.template FreeTensor(pseUb);
    }

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> mm2AL1Tensor;
    mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    if (likely(runInfo.halfS1RealSize != 0)) {
        if constexpr (isFp8 || isInt8) {
            if constexpr (isMlaFullQuant) {
                DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1ScmBlockFp8], stage1CastTensor,
                    {s2BaseSize / 32, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
            } else {
                DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) * (runInfo.s1RealSize - runInfo.halfS1RealSize)], stage1CastTensor,
                    {s2BaseSize / 32, (uint16_t)runInfo.halfS1RealSize, (uint16_t)(vec1Srcstride - runInfo.halfS1RealSize),
                    (uint16_t)(s1BaseSize - runInfo.halfS1RealSize)});
            }
        }
    }
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    if constexpr (isMlaFullQuant) {
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
            this->queryScaleQue[runInfo.taskIdMod2].template FreeTensor(queryScaleUb);
        }
    }
    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfS1RealSize);
    }
    if constexpr (IsSameType<INPUT_T, float>::value) {
        this->sumBrdcst.template FreeTensor(apiTmpBuffer);
    }
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, constInfo, sumUb, maxUb);
        }
    }
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        GetDerived()->SoftmaxDataCopyOut(runInfo, constInfo, sumUb, maxUb);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline int64_t FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ComputeOffsetForSoftmax(
    RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx)
{
    return vec2S1Idx * runInfo.vec2S1BaseSize;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec2OnUb(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    if constexpr (implMode != ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION && !(isFp8 || isInt8) && useDn) {
        if constexpr (isInfer) {
            if (constInfo.s2Size <= 128 && !constInfo.isRowInvalid && !POST_QUANT) { // 128: kv方向基本块大小
                ProcessVec2NoGlobalUpdate(runInfo, constInfo, bmm2ResBuf, (dTemplateAlign64 * sizeof(INPUT_T)) << 5);
                return;
            }
        } else {
            if (constInfo.s2Size <= 128) { // 128: kv方向基本块大小
                ProcessVec2NoGlobalUpdate(runInfo, constInfo, bmm2ResBuf, (dTemplateAlign64 * sizeof(INPUT_T)) << 5);
                return;
            }
        }
    }
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    float deSCaleVValue = 1.0f;
    if constexpr ((isFp8 || isInt8) && !isMxfp8FullQuant) {
        if constexpr (isMlaFullQuant) {
            deSCaleVValue = this->deScaleVGm.GetValue(0);
        } else if constexpr (useNz) {
            int64_t s2BlockCnt = CeilDivision(constInfo.s2Size, FP8_QUANT_V_BLOCK_SIZE);
            runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                                runInfo.n2oIdx * s2BlockCnt +
                                                (runInfo.s2StartIdx >> 9) + runInfo.s2LoopCount;
            deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
        } else {
            deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
        }
    }
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if constexpr (isInt8 && isMlaFullQuant) {
        LocalTensor<int32_t> mmResUb = bmm2ResBuf.template GetTensor<int32_t>();  // int8场景c2的输出为int32
        Cast(mmRes, mmResUb, RoundMode::CAST_ROUND, vec2CalcSize);  // 需要将int32转为fp32后参与后续的计算
    }
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>();
        LocalTensor<T> pScaleUb;
        if constexpr (isMlaFullQuant) {
            pScaleUb = pScaleBuf[runInfo.taskIdMod3].template Get<T>();
        }
        float deSCalePreVValue = 1.0f;
        if constexpr ((isFp8 || isInt8) && !isMxfp8FullQuant) {
            if constexpr (isInfer) {
                if constexpr (useDn) {
                    uint32_t deScaleKvOffset = (runInfo.deScaleKvOffset - 1 < 0) ? 0 : runInfo.deScaleKvOffset - 1;
                    deSCalePreVValue = this->deScaleVGm.GetValue(deScaleKvOffset);
                } else {
                    if constexpr (isMlaFullQuant) {
                        deSCalePreVValue = this->deScaleVGm.GetValue(0);
                    } else {
                        if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) & 1) {   // 7：KV基本块大小128，按照256分块计算deScaleKv偏移
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                        } else {
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                        }
                    }
                }
            } else {
                if constexpr (useDn) {
                    deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                } else {
                    if constexpr (isMlaFullQuant) {
                        deSCalePreVValue = this->deScaleVGm.GetValue(0);
                    } else {
                        if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount) & 1) {   // 7：KV基本块大小128，按照256分块计算deScaleKv偏移
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                        } else {
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                        }
                    }
                }
            }
        }
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            if constexpr (isFp8 || isInt8) {
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, isMlaFullQuant>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, isMlaFullQuant>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        deSCaleVValue, deSCalePreVValue);             
                }
            } else {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, isMlaFullQuant>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        1.0, 1.0);
            }
        } else {
            if constexpr (isFp8 || isInt8) {
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, isMlaFullQuant>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        deSCaleVValue, deSCalePreVValue);            
                } else {
                    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, isMlaFullQuant>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                        deSCaleVValue, deSCalePreVValue);
                }
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, isMlaFullQuant>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                    1.0, 1.0);
            }
        }
    }
    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, isMlaFullQuant>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
        }
        GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
    int64_t vec2CalcSize) {
    LocalTensor<INPUT_T> vec2ResUb = this->stage2OutBuf.template Get<INPUT_T>()[runInfo.taskIdMod2 * vec2CalcSize];
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    DivCast<T, INPUT_T, dTemplateAlign64>(vec2ResUb, bmm2ResBuf.template GetTensor<T>(), sumUb, runInfo.vec2S1RealSize);
    bmm2ResBuf.SetCrossCore();
    Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, 0);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[runInfo.taskIdMod2]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec2DSplit(
    GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {
    // bmm2 result is on GM and global update data on UB
    runInfo.vec2S1BaseSize = 8192 / constInfo.dBasicBlock;
    int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();

    event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
    event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
    event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    float deSCaleVValue;
    if constexpr (isFp8) {
        deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
    }
    for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
        runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
        if (vec2S1Idx == vec2LoopLimit - 1) {
            runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
        }

        int64_t vec2CalcSize = runInfo.vec2S1RealSize * constInfo.dBasicBlock;
        // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
        int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64;
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        if (constInfo.dSizeV == dTemplateAlign64) {
            if constexpr (useDn || isMlaFullQuant) {
                DataCopy(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
            } else {
                DataCopy(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * (int64_t)dVTemplateType + mm2ResInnerOffset], vec2CalcSize);
            }
        } else {
            DataCopyParams dataCopyParams;
            DataCopyPadParams dataCopyPadParams;
            dataCopyParams.blockCount = runInfo.vec2S1RealSize;
            dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
            dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
            dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
            if constexpr (useDn || isMlaFullQuant) {
                DataCopyPad(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset],
                        dataCopyParams, dataCopyPadParams);
            } else {
                DataCopyPad(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * (int64_t)dVTemplateType + mm2ResInnerOffset],
                        dataCopyParams, dataCopyPadParams);
            }
        }

        // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
        int64_t vec2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dBasicBlock;
        if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
            if constexpr (useDn || isMlaFullQuant) {
                DataCopy(vec2ResUb, this->vec2ResGm[runInfo.multiCoreIdxMod3][vec2SubBlockOffset + vec2ResInnerOffset], vec2CalcSize);
            } else {
                DataCopy(vec2ResUb, this->vec2ResGm[runInfo.multiCoreIdxMod3][constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dBasicBlock + vec2ResInnerOffset], vec2CalcSize);
            }
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        if (unlikely(runInfo.s2LoopCount == 0)) {
            DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
        } else {
            int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
            float deSCalePreVValue = 1.0f;
            if constexpr (isFp8) {
                if constexpr (isInfer) {
                    if constexpr (useDn) {
                        deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                    } else {
                        if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) & 1) {
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                        } else {
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                        }
                    }
                } else {
                    if constexpr (useDn) {
                        deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                    } else {
                        if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount) & 1) {
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                        } else {
                            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                        }
                    }
                }
            }
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, true, isMlaFullQuant>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, false, isMlaFullQuant>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            } else {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                if (unlikely(runInfo.s2LoopCount == 1)) {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, true, isMlaFullQuant>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                } else {
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false, isMlaFullQuant>(vec2ResUb, bmm2Ub,
                        vec2ResUb, expUb, expUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock,
                        deSCaleVValue, deSCalePreVValue);
                }
            }
        }

        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, constInfo.dBasicBlock, deSCaleVValue);
            }
            GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
        } else if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
            if constexpr (useDn || isMlaFullQuant) {
                DataCopy(this->vec2ResGm[runInfo.multiCoreIdxMod3][vec2SubBlockOffset + vec2ResInnerOffset], vec2ResUb, vec2CalcSize);
            } else {
                DataCopy(this->vec2ResGm[runInfo.multiCoreIdxMod3][constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dBasicBlock + vec2ResInnerOffset], vec2ResUb, vec2CalcSize);
            }   
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::ProcessVec2(
    mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    bmm2ResBuf.WaitCrossCore();

    if constexpr (bmm2Write2Ub) {
        ProcessVec2OnUb(bmm2ResBuf, runInfo, constInfo);
    } else if constexpr (splitD) {
        GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        ProcessVec2DSplit(mmRes, runInfo, constInfo);
    } else {
        // bmm2 result is on GM and global update data on UB
        runInfo.vec2S1BaseSize = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.halfS1RealSize, runInfo.vec2S1BaseSize);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();
        GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

        event_t mte2ToV = static_cast<event_t>(tPipe->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte2 = static_cast<event_t>(tPipe->FetchEventID(HardEvent::V_MTE2));
        float deSCaleVValue;
        if constexpr (isFp8) {
            deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
        }
        for (int64_t vec2S1Idx = 0; vec2S1Idx < vec2LoopLimit; vec2S1Idx++) {
            runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
            if (vec2S1Idx == vec2LoopLimit - 1) {
                runInfo.vec2S1RealSize = runInfo.halfS1RealSize - vec2S1Idx * runInfo.vec2S1BaseSize;
            }

            int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = vec2S1Idx * runInfo.vec2S1BaseSize * constInfo.dSizeV;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfo.dSizeV == dTemplateAlign64) {
                if constexpr (useDn || isMlaFullQuant) {
                    DataCopy(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
                } else {
                    DataCopy(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dSizeV + mm2ResInnerOffset], vec2CalcSize);
                }
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = runInfo.vec2S1RealSize;
                dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
                if constexpr (useDn || isMlaFullQuant) {
                    DataCopyPad(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset],
                            dataCopyParams, dataCopyPadParams);
                } else {
                    DataCopyPad(bmm2Ub, mmRes[constInfo.subBlockIdx * (runInfo.s1RealSize - runInfo.halfS1RealSize) * constInfo.dSizeV + mm2ResInnerOffset],
                            dataCopyParams, dataCopyPadParams);
                }
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            LocalTensor vec2ResInner = vec2ResUb[vec2S1Idx * runInfo.vec2S1BaseSize * dTemplateAlign64];

            if (unlikely(runInfo.s2LoopCount == 0)) {
                DataCopy(vec2ResInner, bmm2Ub, vec2CalcSize);
            } else {
                int64_t vec2ExpBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                float deSCalePreVValue = 1.0f;
                if constexpr (isFp8) {
                    if constexpr (isInfer) {
                        if constexpr (useDn) {
                            deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                        } else {
                            if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) & 1) {
                                deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                            } else {
                                deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                            }
                        }
                    } else {
                        if constexpr (useDn) {
                            deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                        } else {
                            if (((runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount) & 1) {
                                deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset);
                            } else {
                                deSCalePreVValue = deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
                            }
                        }
                    }
                }
                LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<T>()[vec2ExpBufOffset];
                if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                    if (unlikely(runInfo.s2LoopCount == 1)) {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, isMlaFullQuant>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    } else {
                        FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, isMlaFullQuant>(
                            vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64,
                            deSCaleVValue, deSCalePreVValue);
                    }
                } else {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    if (unlikely(runInfo.s2LoopCount == 1)) {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, isMlaFullQuant>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    } else {
                        FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, isMlaFullQuant>(vec2ResInner, bmm2Ub,
                            vec2ResInner, expUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue,
                            deSCalePreVValue);
                    }   
                }
            }

            if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
                if (unlikely(runInfo.s2LoopCount == 0)) {
                    int64_t vec2SumBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                        vec2ResInner, vec2ResInner, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64, deSCaleVValue);
                }
                GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResInner, vec2S1Idx, vec2CalcSize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, int64_t dSizeAligned64)
{
    if constexpr (isInfer && hasAtten) {
        if (!isMlaFullQuant && (!constInfo.isRowInvalid)) {
            return;
        }
        if (isMlaFullQuant && attenMaskInfoPtr->compressMode == static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
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
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor,  runInfo.vec2S1RealSize, constInfo.dSizeV, static_cast<uint32_t>(dSizeAligned64));
            } else {
                uint32_t dStride = CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                uint16_t dSize = CeilDiv(constInfo.dSizeV, sizeof(float)); // w8后量化后的处理长度
                RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, runInfo.vec2S1RealSize, dSize, dStride);
            }
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::MlaTransposeDataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut)
{
    int64_t s1DealSize = runInfo.vec2S1RealSize;
    int64_t headSize = 0;
    int64_t attenOutOffset = constInfo.dSizeV;
    int64_t headUbOffset = 0, headGmOffset = 0;
    int64_t curGIdx = runInfo.goIdx, curS1Idx = runInfo.s1oIdx;

    if (constInfo.gSize == 128) { // G为128时，基本块位于同一S1行
        curGIdx = (curS1Idx % 2 == 0) ? curGIdx : 64; // 64 s1Base基本块
        curS1Idx /= 2;
    } else if (constInfo.gSize <= 32) { // G<=32时，每64/G行为一个基本块
        curS1Idx *= (64 / constInfo.gSize); // 64 s1Base基本块
    }

    if (constInfo.subBlockIdx == 1) {
        int64_t firstCurGIdx = curGIdx;
        curGIdx = (firstCurGIdx + s1DealSize) % constInfo.gSize;
        curS1Idx += (firstCurGIdx + s1DealSize) / constInfo.gSize;
    }

    DataCopyExtParams dataCopyParams;
    if (curGIdx != 0 && constInfo.gSize != 1) { // 首块
        headSize = curGIdx + s1DealSize < constInfo.gSize ? s1DealSize : constInfo.gSize - curGIdx;
        headUbOffset = headSize * constInfo.dSizeV;
        headGmOffset = constInfo.dSizeV - curGIdx * constInfo.t1Size * constInfo.dSizeV;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.t1Size * constInfo.dSizeV - constInfo.dSizeV) * sizeof(OUTPUT_T);
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = headSize;

        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
    }

    s1DealSize -= headSize; // 中间块 & 尾块
    int64_t blocks = CeilDiv(s1DealSize, constInfo.gSize);
    bool hasTail = s1DealSize % constInfo.gSize != 0;
    for (int64_t i = 0; i < blocks; i++) {
        attenOutOffset = i * constInfo.dSizeV;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.t1Size * constInfo.dSizeV - constInfo.dSizeV) * sizeof(OUTPUT_T);
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
        if (i == blocks - 1 && hasTail) {
            dataCopyParams.blockCount = s1DealSize % constInfo.gSize;
        } else {
            dataCopyParams.blockCount = constInfo.gSize;
        }

        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + headGmOffset + attenOutOffset],
            attenOut[headUbOffset + i * constInfo.gSize * constInfo.dSizeV], dataCopyParams);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::MlaTranspose2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<OUTPUT_T> &attenOut)
{
    int64_t s1DealSize = runInfo.vec2S1RealSize;
    int64_t curGIdx = runInfo.sOuterOffset / constInfo.s1Size;
    int64_t curS1Idx = runInfo.sOuterOffset % constInfo.s1Size;
    bool hasHeadBlock = curS1Idx != 0;
    int headBlock = hasHeadBlock ? constInfo.s1Size - curS1Idx : 0;
    int gCount = hasHeadBlock ? (runInfo.vec2S1BaseSize - headBlock) / constInfo.s1Size : runInfo.vec2S1BaseSize / constInfo.s1Size;
    bool hasTailBlock = (runInfo.vec2S1BaseSize - headBlock) % constInfo.s1Size;
    int tailBlock = hasTailBlock ? runInfo.vec2S1BaseSize - gCount * constInfo.s1Size - headBlock : 0;
    uint32_t attenOutUbOffset = 0;

    if (curS1Idx != 0) { // 首块
        DataCopyExtParams dataCopyParams;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        dataCopyParams.blockLen = (constInfo.s1Size - curS1Idx) * constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParams.blockCount = 1;
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
        attenOutUbOffset += (constInfo.s1Size - curS1Idx) * constInfo.dSizeV;
    }
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = gCount; // 处理多少个G
    dataCopyParams.blockLen = constInfo.s1Size * constInfo.dSizeV * sizeof(OUTPUT_T); // 一个S1*D的大小
    dataCopyParams.srcStride = 0;                                                                    // 连读
    dataCopyParams.dstStride = (constInfo.bSize - 1) * constInfo.s1Size * constInfo.dSizeV * sizeof(OUTPUT_T); // 跳写
    runInfo.attentionOutOffset += int(hasHeadBlock) * constInfo.bSize * constInfo.s1Size * constInfo.dSizeV - curS1Idx * constInfo.dSizeV;
    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut[attenOutUbOffset], dataCopyParams);
    attenOutUbOffset += dataCopyParams.blockCount * (constInfo.s1Size * constInfo.dSizeV);

    if (hasTailBlock) { // 尾块单独一条DataCopy指令
        DataCopyExtParams dataCopyParamsTail;
        dataCopyParamsTail.blockCount = 1;
        dataCopyParamsTail.blockLen = tailBlock * constInfo.dSizeV * sizeof(OUTPUT_T);
        dataCopyParamsTail.srcStride = 0;
        dataCopyParamsTail.dstStride = 0;
        runInfo.attentionOutOffset += gCount * constInfo.bSize * constInfo.s1Size * constInfo.dSizeV;
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut[attenOutUbOffset], dataCopyParamsTail);
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = constInfo.dBasicBlock;
    }
    if constexpr (!IsSameType<INPUT_T, VEC2_RES_T>::value) {
        attenOut.SetAddr(vec2ResUb.address_);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
                LocalTensor<float> maxTensor = softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<float>()[vec2MaxBufOffset];
                InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
                    dSizeAligned64, this->negativeFloatScalar, 0.0);
            }
        }
        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, vec2S1Idx, runInfo, constInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
        } else {
            GetDerived()->PostQuant(constInfo, runInfo, attenOut, vec2ResUb, vec2S1Idx, dSizeAligned64);
            RowInvalid(vec2ResUb, vec2S1Idx, runInfo, constInfo, dSizeAligned64);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    } else {
        if constexpr (!POST_QUANT) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.taskIdMod2]);
            attenOut = vec2ResUb;
        }   
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    if constexpr (IsSameType<INPUT_T, float>::value) {
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
                attenOutOffset = 0;
                dataCopyParams.blockLen *= constInfo.gSize;
                dataCopyParams.blockCount /= constInfo.gSize;
            } else if (constInfo.isGqa) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    } else {
        if (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH)) {
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
            if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
 	            constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND)) {
                attenOutOffset = constInfo.n2GDv;
            }
        }
    }

    if constexpr (isInfer) {
        if (isMlaFullQuant) {
            if (isFp8 && constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_NBSD)) {
                MlaTranspose2DataCopyOut(runInfo, constInfo, attenOut);
            } else if (constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_NBSD) ||
                constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_NBSD) ||
                constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::TND_NTD)) {
                MlaTransposeDataCopyOut(runInfo, constInfo, attenOut);
            } else {
                DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
                    attenOut, dataCopyParams);
            }
        } else {
            if (constInfo.isPfaGS1Merge && dSizeAligned64 - constInfo.dSizeV != 0 && (constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH) || constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_TND))) {
                for(int64_t i = 0; i < runInfo.vec2S1BaseSize / constInfo.gSize; i++){
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
        }          
    } else {
        DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams);
    }
}
TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]
    if constexpr (isFd) {
        tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
        tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    }
    tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(preLoopMaxBuf, 256); // [64, 1]
    tPipe->InitBuffer(preLoopSumBuf, 256); // [64, 1]
    tPipe->InitBuffer(firstLoopSumBuf, 256); // [64, 1]
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo)
{
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    if constexpr (!bmm2Write2Ub) {
        tPipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
    }
    if constexpr (s2BaseSize == 256) { // s1BaseSize = 128
        if constexpr (s1BaseSize == 128) { // s1BaseSize = 128 s2BaseSize = 256
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            SoftmaxInitBuffer();
            if constexpr (isFp8 || isInt8) {
                tPipe->InitBuffer(stage1OutQue[0], 1, 16896);  // (32 + 1) * (256 / 32) * 64
                tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
                if constexpr (hasAtten) {
                    tPipe->InitBuffer(attenMaskInQue[0], 1, 16384); // 256 * 64
                }
            } else {
                tPipe->InitBuffer(stage1OutQue[0], 1, 33024);
                tPipe->InitBuffer(stage1OutQue[1], 1, 33024);
            }
        } else { // s1BaseSize = 64 s2BaseSize = 256
            SoftmaxInitBuffer();
            tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
            tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
            if constexpr (hasAtten) {
                tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
                tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
            }
            if constexpr (hasPseOuter) {
                tPipe->InitBuffer(pseInQue, 1, 16384);
            }
        }
    } else { 
        SoftmaxInitBuffer();
        if constexpr (isMlaFullQuant) {
            if constexpr (!useDn) {
                if constexpr (hasPseOuter) {
                    if constexpr (IsSameType<INPUT_T, float>::value) {
                        tPipe->InitBuffer(pseInQue, 1, 32768);
                    } else {
                        tPipe->InitBuffer(pseInQue, 1, 16384);
                    }
                }

                if constexpr (hasAtten) {
                    tPipe->InitBuffer(attenMaskInQue[0], 1, 4096); // 4096: GS1方向需要循环处理，一个vector计算softmax的数据量最大为32*128，对应mask(bool/int8/uint8)的数据量为4096Bytes
                    tPipe->InitBuffer(attenMaskInQue[1], 1, 4096); // 4096：同上
                }

                tPipe->InitBuffer(commonTBuf, 512); // 实际只需512Bytes
            }
            if constexpr (bmm2Write2Ub) {
                tPipe->InitBuffer(stage2OutBuf, 64 / CV_RATIO * dTemplateAlign64 * sizeof(T)); // 64: s1RealSize
            } else {
                tPipe->InitBuffer(stage2OutBuf, 32768);
            }
            tPipe->InitBuffer(stage1OutQue[0], 1, 4224); // 4224: (s1BaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(INPUT_T)
            tPipe->InitBuffer(stage1OutQue[1], 1, 4224); // 4224: 同上
        } else if constexpr (useNz) {
            tPipe->InitBuffer(commonTBuf, 512);
            tPipe->InitBuffer(stage2OutBuf, 32768);
        } else if constexpr (isMxfp8FullQuant) {
            tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            SoftmaxInitBuffer();
            tPipe->InitBuffer(stage1OutQue[0], 1, 16640);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16640);
            tPipe->InitBuffer(commonTBuf, 512);
            if constexpr (hasAtten) {
                tPipe->InitBuffer(attenMaskInQue[0], 1, 16384);
            }
        } else {
            if constexpr (!useDn) {
                if constexpr (hasPseOuter) {
                    if constexpr (IsSameType<INPUT_T, float>::value) {
                        tPipe->InitBuffer(pseInQue, 1, 32768);
                    } else {
                        tPipe->InitBuffer(pseInQue, 1, 16384);
                    }
                }

                if constexpr (hasAtten) {
                    tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
                    tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
                }

                if constexpr (!IsSameType<INPUT_T, float>::value) {
                    tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
                }
            }
            if constexpr (bmm2Write2Ub) {
                // 小于128Bmm2结果和Vec2结果都在UB
                tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            } else if constexpr (dTemplateAlign64 <= 256) {
                // bmm2结果在Gm，Vector2结果在UB，开启多层循环，每次处理32KB
                tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            } else {
                // bmm2结果在Gm，Vector2结果也在Gm，开启多层循环，每次处理32KB
                tPipe->InitBuffer(stage2OutBuf, 32768);
            }
            if constexpr (IsSameType<INPUT_T, float>::value) {
                tPipe->InitBuffer(stage1OutQue[0], 1, 33280);
            } else if constexpr (isFp8 || isInt8) {
                tPipe->InitBuffer(stage1OutQue[0], 1, 8320);
                tPipe->InitBuffer(stage1OutQue[1], 1, 8320);
            } else {
                tPipe->InitBuffer(stage1OutQue[0], 1, 16640);
                tPipe->InitBuffer(stage1OutQue[1], 1, 16640);
            }
        }
    }
    if constexpr (isFp8 || isInt8) {
        if constexpr (useDn) {
            tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)], 256);
            LocalTensor<uint8_t> vselrIndexesTensor =
                vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
            for (uint32_t i = 0; i < 4; i++) {
                for (uint32_t j = 0; j < (256 >> 2); j++) {
                    vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
                }
            }
        } else if constexpr (useNz) {
            tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::NZ_INDEX)], 256);
            LocalTensor<uint8_t> vselrIndexesTensor =
                vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::NZ_INDEX)].template Get<uint8_t>();
            uint32_t i1 = 0;
            uint32_t i2 = 1;
            for (uint32_t i = 0; i < 256; i += 32) {
                for (uint32_t j = i; j < i + 16; j++) {
                    vselrIndexesTensor.SetValue(j, i1);
                    i1 += 2;
                }
                for (uint32_t j = i + 16; j < i + 32; j++) {
                    vselrIndexesTensor.SetValue(j, i2);
                    i2 += 2;
                }
            }
        } else {
            tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)], 128);
            tPipe->InitBuffer(vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)], 64);
            LocalTensor<uint8_t> vselrIndexesTensor =
                vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
            for (uint32_t i = 0; i < 128; i++) {
                vselrIndexesTensor.SetValue(i, i << 1);
            }
            vselrIndexesTensor =
                vselrIndexesBuf[static_cast<uint32_t>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
            for (uint32_t i = 0; i < 64; i++) {
                vselrIndexesTensor.SetValue(i, i << 2);
            }
        }
    }

    GetDerived()->InitUniqueLocalBuffer(constInfo);

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}


TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBaseFullquant<TEMPLATE_BASE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    if constexpr (IsSameType<T, float>::value && !useNz) {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        negativeScalar = *((float *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
                positiveScalar = *((float *)&tmp2);
            }
        }
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VALUE_FP16;
        negativeScalar = *((half *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
                positiveScalar = *((half *)&tmp2);
            }
        }
    }
}

TEMPLATES_DEF
class FABlockVecFullquantDummy {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr TPosition bmm2OutPos = GetC2Position(
        dVTemplateType, BaseApi::UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop, hasRope,
        s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64), false);
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;

    __aicore__ inline FABlockVecFullquantDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, 
        ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const optiling::FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
        AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {};
    __aicore__ inline void InitDropOut(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace) {}
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
        __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,__gm__ uint8_t *prefix,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, 
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset,
        uint32_t aicIdx, ConstInfo<isInfer, hasRope> &constInfo) {}

    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
};
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_FULLQUANT_H_