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
 * \file flash_attention_score_block_vec_base.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_H_
#define FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_H_
#include "util_regbase.h"
#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_regbase_v2.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_regbase_v2.h"
#include "vf/vf_flashupdate_new_regbase_v2.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF_BASE
class FABlockVecBase {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = s1BaseSize; // __NPU_ARCH__ == 5102 does not have sameab
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
    static constexpr bool useDn = IsDn(((IsSameType<INPUT_T, float>::value) || isFp8), (isFp8 && (s2BaseSize == 256)), pseMode, hasAtten, hasDrop,
                                       s1BaseSize == 128, dTemplateType, hasRope);
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool hasPseOuter = (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) ||
                                        (pseMode == PseTypeEnum::PSE_OUTER_MUL_ADD_TYPE);
    static constexpr bool containAllOptionalInput = hasPse && hasAtten && hasDrop;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr TPosition bmm2OutPos = GetC2Position(
        dVTemplateType, UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
        s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64));
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr bool isW8In = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                  IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value ||
                                  IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using pseShiftW8InType = typename AscendC::Conditional<isInfer, half, OUTPUT_T>::type;
    using pseShiftType = typename AscendC::Conditional<isW8In, pseShiftW8InType, INPUT_T>::type;
    static constexpr int64_t FP8_QUANT_KV_BLOCK_SIZE = isInfer ? 256 : 128;
    // ==================== Functions ======================
    __aicore__ inline FABlockVecBase() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {
        tPipe = pipe;
        tilingData = tiling;
        GetDerived()->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
        this->GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
        attenMaskInfoPtr = &attenMaskInfo;
        pseInfoPtr = &pseInfo;
    }
    __aicore__ inline void InitCommonGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
        __gm__ uint8_t *deqScaleV, __gm__ uint8_t *quantScaleP,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *&workspace, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    TPipe *tPipe;
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
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
    quantGmType quantScalePGm;
    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;
    vec2ResGmType vec2ResGm[3];

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
    __aicore__ inline void ProcessVec1NdRegbaseV2(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        LocalTensor<T> mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec1DnRegbaseV2(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        LocalTensor<T> mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2OnUbRegbaseV2(LocalTensor<T> mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void ProcessVec2NoGlobalUpdate(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, int64_t vec2CalcSize);

    __aicore__ inline void InvalidLineProcess(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<T> &sumUb, LocalTensor<T> &maxUb);

    __aicore__ inline int64_t ComputeOffsetForSoftmax(RunInfo<isInfer> &runInfo, const int64_t vec2S1Idx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo, int64_t dSizeAligned64);
};

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::InitCommonGlobalBuffer(
    __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
    __gm__ uint8_t *deqScaleV, __gm__ uint8_t *quantScaleP,
    __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, 
    __gm__ uint8_t *&workspace, ConstInfo<isInfer, hasRope> &constInfo) 
{
    if constexpr (hasPse) {
        pseGm.SetGlobalBuffer((__gm__ pseShiftType *)pse);
        pseSlope = pse;
    }
    if constexpr (isInt8) {
        if (quantScaleP != nullptr) {
            quantScalePGm.SetGlobalBuffer((__gm__ float *)quantScaleP);
            constInfo.quantScalePValue = quantScalePGm.GetValue(0);
        }
    } else {
        constInfo.quantScalePValue = 1.0F;
    }
    if constexpr (isFp8) {
        deScaleQGm.SetGlobalBuffer((__gm__ float *)deqScaleQ);
        deScaleKGm.SetGlobalBuffer((__gm__ float *)deqScaleK);
        deScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
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
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    if constexpr (useDn) {
        ProcessVec1DnRegbaseV2(outputBuf, mmRes, runInfo, constInfo);
    } else { // IFA:use dn
        ProcessVec1NdRegbaseV2(outputBuf, mmRes, runInfo, constInfo);
    }
}

// =================================Private Functions=================================
TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1DnRegbaseV2(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf, LocalTensor<T> mmRes,
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<INPUT_T> tmpSoftmaxResUb[2];
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>()[0];
    LocalTensor<T> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<T>()[0];
 
    auto expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<float>()[0];
    int64_t stage1Offset = runInfo.taskIdMod2;
    AscendC::LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();

    if (unlikely(runInfo.s2LoopCount == 0)) {
        FaVectorApi::ProcessVec1VfDnRegbaseV2<T, INPUT_T, false, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, mmRes, expUb, runInfo.s1RealSizeAlign32, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), negativeFloatScalar, constInfo.quantScalePValue);
    } else {
        FaVectorApi::ProcessVec1VfDnRegbaseV2<T, INPUT_T, true, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, mmRes, expUb, runInfo.s1RealSizeAlign32, runInfo.s2RealSize,
            static_cast<T>(constInfo.scaleValue), negativeFloatScalar, constInfo.quantScalePValue);
    }
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    stage1CastTensor = this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    //-------------------------Data copy to l1-------------------------
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    DataCopy(mm2AL1Tensor[0], stage1CastTensor[0], {vec1S2CopyCountDn, vec1S2CopyLenDn, 0, vec1S2CopyLenDn});
    // 16:sinner / 2 * 32; 64:sinner / 2 * 32 * souter / 32
    DataCopy(mm2AL1Tensor[s2BaseSize * 16], stage1CastTensor[s2BaseSize * 64], {vec1S2CopyCountDn, vec1S2CopyLenDn, 0, vec1S2CopyLenDn});
    SetFlag<HardEvent::MTE3_MTE1>(SYNC_V1_C2_FLAG[runInfo.taskIdMod3]);
    //-----------------------------------------------------------------
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec1NdRegbaseV2(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf, LocalTensor<T> mmRes,
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    LocalTensor<pseShiftType> pseUb;
    if constexpr (hasPseOuter == true) {
        PseCopyIn<T, OUTPUT_T, hasPseOuter>(this->pseInQue, this->pseGm, runInfo, constInfo, *pseInfoPtr);
        pseUb = this->pseInQue.template DeQue<OUTPUT_T>();
    }
    float slopes = 0.0f;
    float posShift = 0.0f;
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                  pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
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
        AttenMaskCopyIn<hasAtten, isFd>(this->attenMaskInQue[runInfo.taskIdMod2], this->attenMaskInQue[1 - runInfo.taskIdMod2],
            this->attenMaskGmInt, runInfo, constInfo, *attenMaskInfoPtr);
        attenMaskUb = this->attenMaskInQue[runInfo.taskIdMod2].template DeQue<uint8_t>();
    }
    LocalTensor<uint8_t> dropMaskUb;
    GetDerived()->GenerateDropoutMask(runInfo, constInfo, dropMaskUb);
 
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
    LocalTensor<T> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod3].template Get<T>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod3].template Get<float>();
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
    if constexpr (isFp8) {
        int64_t s1BlockCnt = CeilDivision(constInfo.s1Size, FP8_QUANT_BLOCK_SIZE);
        int64_t s2BlockCnt = CeilDivision(constInfo.s2Size, FP8_QUANT_BLOCK_SIZE);
        /* Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1] */
        int64_t deScaleQOffset = runInfo.boIdx * constInfo.n2G * s1BlockCnt +
                                 runInfo.n2oIdx * constInfo.gSize * s1BlockCnt +
                                 runInfo.goIdx * s1BlockCnt + runInfo.s1oIdx;
        runInfo.deScaleKvOffset = runInfo.boIdx * constInfo.n2Size * s2BlockCnt +
                                  runInfo.n2oIdx * s2BlockCnt +
                                  (runInfo.s2StartIdx >> 7) + runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize; // 7 for multi factor 128
        float deSCaleQValue = this->deScaleQGm.GetValue(deScaleQOffset);
        float deSCaleKValue = this->deScaleKGm.GetValue(runInfo.deScaleKvOffset);
        descaleQK = deSCaleQValue * deSCaleKValue;
    }
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
    LocalTensor<float> null;
    if (runInfo.s2LoopCount == 0) {
        if (runInfo.s2RealSize == 128) {
            SoftmaxFlashV510_VF<T, INPUT_T, false, 1, s1BaseSize, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, expUb, mmRes, sumUb, maxUb, attenMaskUb, pseUb,
            apiTmpBuffer, runInfo.s1RealSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar, constInfo.quantScalePValue);
        } else {
            SoftmaxFlashV510_VF<T, INPUT_T, false, 2, s1BaseSize, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, expUb, mmRes, sumUb, maxUb, attenMaskUb, pseUb,
            apiTmpBuffer, runInfo.s1RealSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar, constInfo.quantScalePValue);
        }
    } else {
        if (runInfo.s2RealSize == 128) {
            SoftmaxFlashV510_VF<T, INPUT_T, true, 1, s1BaseSize, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, expUb, mmRes, sumUb, maxUb, attenMaskUb, pseUb,
                apiTmpBuffer, runInfo.s1RealSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar, constInfo.quantScalePValue);
        } else {
            SoftmaxFlashV510_VF<T, INPUT_T, true, 2, s1BaseSize, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, expUb, mmRes, sumUb, maxUb, attenMaskUb, pseUb,
            apiTmpBuffer, runInfo.s1RealSize, runInfo.s2RealSize, static_cast<T>(constInfo.scaleValue), negativeFloatScalar, constInfo.quantScalePValue);
        }
    }
    SetFlag<HardEvent::V_MTE3>(MM1_RES_INTRA_EVENT[runInfo.taskIdMod2]);
    WaitFlag<HardEvent::V_MTE3>(MM1_RES_INTRA_EVENT[runInfo.taskIdMod2]);
    if constexpr (hasAtten) {
        this->attenMaskInQue[runInfo.taskIdMod2].template FreeTensor(attenMaskUb);
    }
    if constexpr (hasPseOuter) {
        this->pseInQue.template FreeTensor(pseUb);
    }
 
    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    DataCopy(mm2AL1Tensor, stage1CastTensor, {s2BaseSize / 16, (uint16_t)(runInfo.s1RealSize), 
                    (uint16_t)(vec1Srcstride - runInfo.s1RealSize), 
                    (uint16_t)(s1BaseSize - runInfo.s1RealSize)});
    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
    SetFlag<HardEvent::MTE3_MTE1>(SYNC_V1_C2_FLAG[runInfo.taskIdMod3]);
    // ======================================================
    if constexpr (IsSameType<INPUT_T, float>::value) {
        this->sumBrdcst.template FreeTensor(apiTmpBuffer);
    }
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
        if (this->tilingData->inputParamsRegbase.implMode == static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
            this->InvalidLineProcess(runInfo, constInfo, sumUb, maxUb);
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2OnUbRegbaseV2(
    LocalTensor<T> mmRes, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {
    if (unlikely(runInfo.vec2S1BaseSize == 0)) {
        SetFlag<HardEvent::MTE3_V>(MM2_RES_INTRA_EVENT[runInfo.taskIdMod2]); // 疑问：此处是否是MTE3等V
        return;
    }
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize; // halfS1RealSize, 52没有除2
    int64_t vec2CalcSize = runInfo.vec2S1RealSize * dTemplateAlign64;
    float deSCaleVValue;
    if constexpr (isFp8) {
        deSCaleVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset);
    }
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<float> expUb = softmaxExpBuf[runInfo.taskIdMod3].template Get<float>();
        float deSCalePreVValue = 1.0f;
        if constexpr (isFp8) {
            deSCalePreVValue = this->deScaleVGm.GetValue(runInfo.deScaleKvOffset - 1);
        }
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateV510<T, T, T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2S1RealSize, dTemplateAlign64); // deq2在mm2fixp完成，quant2在搬出时完成。与flashupdate无关
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            FlashUpdateLastV510<T, T, T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2S1RealSize, dTemplateAlign64);
        }
    }
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod3].template Get<float>();
            FlashUpdateDivV510<T, T, T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2S1RealSize, (uint16_t)dTemplateAlign64);
        }
        GetDerived()->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::ProcessVec2(
    mm2ResPos &bmm2ResBuf, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (bmm2Write2Ub) {
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        ProcessVec2OnUbRegbaseV2(mmRes, runInfo, constInfo);
    }
    return;
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb,
    int64_t vec2S1Idx, RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, int64_t dSizeAligned64)
{
    if constexpr (isInfer && hasAtten) {
        if (!constInfo.isRowInvalid || \
            attenMaskInfoPtr->compressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
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
            if (checkValue == NEGATIVE_MIN_VAULE_FP32) {
                isRowInvalidNeedUpdate = true;
                break;
            }
        }
        if (isRowInvalidNeedUpdate) {
            if constexpr (!POST_QUANT) {
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor,  runInfo.vec2S1RealSize, constInfo.dSizeV, static_cast<uint32_t>(dSizeAligned64));
            } else {
                uint32_t dStride = CeilDivision(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                uint16_t dSize = CeilDivision(constInfo.dSizeV, sizeof(float)); // w8后量化后的处理长度
                RowInvalidUpdateVF<float>(*((LocalTensor<float>*)&vec2ResUb), maxTensor, runInfo.vec2S1RealSize, dSize, dStride);
            }
        }
    }
}

TEMPLATES_DEF_BASE_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::Bmm2DataCopyOut(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dVTemplateType;
    if constexpr (splitD) {
        dSizeAligned64 = constInfo.dBasicBlock;
    }
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
    int64_t attenOutOffset = constInfo.dSizeV;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        attenOutOffset = constInfo.n2GDv;
        if constexpr (isInfer) {
            if (constInfo.isGqa == 1) {
                attenOutOffset = constInfo.dSizeV;
            }
        }
    } else {
        if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_BSH) {
            attenOutOffset = constInfo.n2GDv;
            if constexpr (isInfer) {
                if (constInfo.isGqa == 1) {
                    attenOutOffset = constInfo.dSizeV;
                }
            }
        } else if (constInfo.layoutType == (uint8_t)LayOutTypeEnum::LAYOUT_SBH) {
            attenOutOffset = constInfo.bN2GDv;
        }
        if constexpr (isInfer) {
            if (constInfo.isBSNDOut == 1) {
                attenOutOffset = constInfo.n2GDv;
            }
        }
    }

    dataCopyParams.blockCount = runInfo.vec2S1RealSize;
    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset + vec2S1Idx * runInfo.vec2S1BaseSize * attenOutOffset],
                attenOut, dataCopyParams); // 
}
TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::SoftmaxInitBuffer()
{
    tPipe->InitBuffer(softmaxSumBuf[0], 512); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[1], 512); // [64, 1]
    tPipe->InitBuffer(softmaxSumBuf[2], 512); // [64, 1]
    tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
    tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
    tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[0], 512); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[1], 512); // [64, 1]
    tPipe->InitBuffer(softmaxExpBuf[2], 512); // [64, 1]
}

TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo<isInfer, hasRope> &constInfo)
{
    SoftmaxInitBuffer();
    tPipe->InitBuffer(stage2OutBuf, 128 * dTemplateAlign64 * sizeof(T));
    tPipe->InitBuffer(stage1OutQue[0], 1, 16640);
    tPipe->InitBuffer(stage1OutQue[1], 1, 16640);
    tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
    GetDerived()->InitUniqueLocalBuffer(constInfo);

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}


TEMPLATES_DEF_BASE_NO_DEFAULT
__aicore__ inline void FABlockVecBase<TEMPLATE_BASE_ARGS>::GetExtremeValue(
    T &negativeScalar, T &positiveScalar)
{
    if constexpr (IsSameType<T, float>::value) {
        uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
        negativeScalar = *((float *)&tmp1);
        if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || IsSameType<INPUT_T, float>::value) {
            if (this->tilingData->inputParamsRegbase.implMode ==
                static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
                uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
                positiveScalar = *((float *)&tmp2);
            }
        }
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
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
class FABlockVecDummy {
public:
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr TPosition bmm2OutPos = GetC2Position(
        dVTemplateType, BaseApi::UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
        s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64));
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;

    __aicore__ inline FABlockVecDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, 
        ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx,
        AttenMaskInfo &attenMaskInfo, PseInfo &pseInfo) {};
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *deqScaleQK, __gm__ uint8_t *quantScaleP,
        __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
        __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *dropMask, 
        __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
        ConstInfo<isInfer, hasRope> &constInfo) {}

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
#endif // FLASH_ATTENTION_SCORE_BLOCK_VEC_BASE_H_