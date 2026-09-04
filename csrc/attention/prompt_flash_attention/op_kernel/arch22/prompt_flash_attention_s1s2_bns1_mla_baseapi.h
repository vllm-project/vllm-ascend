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
* \file prompt_flash_attention_s1s2_bns1_mla_baseapi.h
* \brief
*/

#ifndef PROMPT_FLASH_ATTENTION_S1S2_BNS1_MLA_BASEAPI_H
#define PROMPT_FLASH_ATTENTION_S1S2_BNS1_MLA_BASEAPI_H

#include "util.h"
#include "prompt_flash_attention_mla_common.h"
#include "mla_custom_matmul_policy_d192.h"
#include "mla_custom_matmul_policy_d128.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "pfa_service_matmul_kv_nd.h"

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
    typename T = INPUT_T, CubeFormat bmm1Format = CubeFormat::ND,
    MmPolicyType mmPolicyType = MmPolicyType::NORMAL, bool pageAttention = false>
class MlaS1s2Bn2gs1SameABBaseApi {
public:
    __aicore__ inline MlaS1s2Bn2gs1SameABBaseApi(){};
    static constexpr bool PAGE_ATTENTION = pageAttention;
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable, __gm__ uint8_t *attentionOut, 
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const TILING_TYPE *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    PfaMatmulKvNd<INPUT_T, T, PAGE_ATTENTION> matmulService;

protected:
    static constexpr bool BMM1_OUT_ISFIXED_ND = false;  // false: 根据S2是否64字节对齐决定bmm1输出为ND/NZ. 对齐输出ND，否则NZ. true: 强制bmm1输出ND
    static constexpr bool BMM2_OUT_ISFIXED_ND = true;   // false: bmm2输出为NZ. true: bmm2输出ND
    static constexpr bool BMM2_A_ISFIXED_ND = true;     // false: bmm2.A矩阵输入为NZ. true: bmm2.A输入ND
    static constexpr bool DIS_NDNZ_CONVERT = (BMM1_OUT_ISFIXED_ND && BMM2_OUT_ISFIXED_ND && BMM2_A_ISFIXED_ND);
    static constexpr uint64_t STAGE2_TBUF_SIZE = 32 * 128;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t MLA_NEGATIVE_INF_VALUE_FP32 = 0xFF800000;

    static __aicore__ inline constexpr bool InputLayoutIsTNDLike()
    {
        return (layOutType == LayOutTypeEnum::LAYOUT_TND) || (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND);
    }

    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                     __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                     __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink,
                                     __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                    const TILING_TYPE *__restrict tiling, TPipe *tPipe);
    __aicore__ inline uint64_t CalcBmm2TensorBGmOffset(const SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
                                LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx);
    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void SetTiling(const TILING_TYPE *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void AllocEvent();
    __aicore__ inline void FreeEvent();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline uint64_t CalcBmm1TensorAGmOffset(const SplitSameABExtraInfo &extraInfo);
    __aicore__ inline uint64_t CalcBmm1QRopeGmOffset(const SplitSameABExtraInfo &extraInfo);
    __aicore__ inline uint64_t CalcBmm1TensorBGmOffset(const SplitSameABExtraInfo &extraInfo);
    __aicore__ inline uint64_t CalcBmm1KRopeGmOffset(const SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx, int64_t maskOffset,
                                        bool secondTime = false);
    __aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre, int64_t s1Offset,
                                                SplitSameABExtraInfo &extraInfo);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void GetBmm1Result(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo, LocalTensor<T> &bmm1ResUb,
                                            LocalTensor<uint8_t> &maskUb, const uint8_t maskType, event_t vWaitMte2);

    __aicore__ inline void SoftMaxCompute(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo, int64_t vec1S1realSplitN);
    __aicore__ inline void InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcTensor,
                                                    LocalTensor<T> &maxUb, int64_t loopIdx);
    __aicore__ inline void ProcessVec2(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultMul(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcBmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx, int64_t mm2ResCalcSize);
    __aicore__ inline void SoftmaxDataCopyOut(SplitSameABExtraInfo &extraInfo);

    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void GetS2LoopRange();

    uint32_t cubeS1BaseSize;
    uint32_t vecS1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    int64_t dSizeAlign16;
    uint32_t valueDSize;
    int64_t valueDSizeAlign16;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1OuterSize;

    // sparse 用参数
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t nextS2EndIdx;

    // BNG 外循环
    int64_t bngStartIdx;
    int64_t bngEndIdx;
    int64_t qCoreOffset;
    int64_t qRopeOffset;
    int64_t kRopeOffset;
    int32_t softmaxReduceSize = 8;

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> maskTBufPong;
    TBuf<> pseTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> sinkBuf;
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxLseBuf;
    TBuf<> commonTBuf; // common的复用空间
    TBuf<> softmaxTempBuf;
    TBuf<> outPBuf;
    TBuf<> outOBuf;

    LocalTensor<T> bmm1ResUb;
    LocalTensor<T> v1CalcUb;
    LocalTensor<T> bmm2ResUb;  // 上一轮V2计算的结果(update后的)
    LocalTensor<T> curBmm2ResUb;  // 当前bmm2的结果
    LocalTensor<T> bmm2ResOutUb;  // 用于将update后的V2结果copy到GM
    LocalTensor<T> softmaxLseTmpUb;  // 计算和搬出softmax lse结果
    event_t v1ReadyToInBmm1ResEvent;
    event_t v1ReadyToInPseMaskEvent;
    event_t v1Bmm1ResMte2EndEvent;
    event_t v1PseMaskMte2EndEvent;
    event_t v1VecEndEvent;
    event_t v1Mte3EndEvent;
    event_t v2VecFreeEvent;
    event_t v2VecEndEvent;
    event_t v2Mte2EndEvent;
    event_t v2Mte3EndEvent;

    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<half> pseAlibiGm;

    LocalTensor<float> sinkUb; // sinkUb

    constexpr static int32_t repeatMaxBytes = 256;
    constexpr static int32_t repeatMaxTimes = 255;
    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t n2GS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t bN2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2GD;
    int64_t gS2;
    int64_t n2GOuterSize;
    int64_t gBaseSize;

    // MLA 带rope, qRope
    int64_t gRopeD;
    int64_t n2S2RopeD;
    int64_t s2RopeD;
    int64_t s1BaseRopeD;
    int64_t s1RopeD;
    int64_t gS1RopeD;
    int64_t n2GS1RopeD;
    int64_t n2RopeD;
    int64_t bN2RopeD;
    int64_t n2GRopeD;
    int64_t bN2GRopeD;
    int64_t s1BaseN2GRopeD;
    int64_t s1BaseBN2GRopeD;
    int64_t s2BaseN2RopeD;
    int64_t s2BaseNratioN2RopeD;
    int64_t s2BaseNratioRopeD;
    int64_t ropeHeadSize;

    // MLA D不等长, V和OUT为D2
    int64_t gD2;
    int64_t n2S2D2;
    int64_t s2D2;
    int64_t s1BaseD2;
    int64_t s1D2;
    int64_t gS1D2;
    int64_t n2GS1D2;
    int64_t n2D2;
    int64_t bN2D2;
    int64_t n2GD2;
    int64_t bN2GD2;
    int64_t s1BaseN2GD2;
    int64_t s1BaseBN2GD2;

    // s2base*N之后的长度
    int64_t s2BaseNRatioSize;
    int64_t s2BaseN2D;
    int64_t s2BaseBN2D;
    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s2BaseNratioD;
    int64_t s2BaseNratioN2D;
    int64_t s2BaseNratioBN2D;
    int64_t bN2G;
    int64_t n2GS2;
    int64_t s2SizeSum;

    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm1RopeKa;
    int64_t mm1RopeKb;
    int64_t mm2Kb;
    // 当splitN大于16时，需要修改softMaxCheckRes数据类型
    uint16_t softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
    uint32_t negativeIntScalar = MLA_NEGATIVE_MIN_VAULE_FP32;
    T negativeFloatScalar;
    T positiveFloatScalar;

    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::NORMAL_MODE;

    int32_t vecBlockIdx;
    int32_t cubeBlockIdx;
    int32_t cubeSubIdx;
    const TILING_TYPE *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

    TPipe *pipe;

    __gm__ uint8_t* key_ptr;
    __gm__ uint8_t* value_ptr;
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> qRopeGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> kRopeGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    // pageAttention
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<bfloat16_t> sinkGm;
    int32_t kvCacheBlockSize = 0;
    int32_t maxBlockNumPerBatch = 0;

    int64_t attenMaskOffsetPre = 0;
    int64_t sinkN1Offset = -1;
    int64_t sinkS1Size = -1;
    bool hasSink = false;
    bool hasSparse4InvalidLine = false;
    bool notSplitG = false;
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
    typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, bmm1Format, mmPolicyType, pageAttention>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict tiling,
    TPipe *tPipe)
{
    this->InitInput(query, key, value, nullptr, nullptr, attenMask, nullptr, nullptr, attentionOut, softmaxLse, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
    typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *value, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict tiling,
    TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        this->vecBlockIdx = GetBlockIdx();
        this->cubeBlockIdx = this->vecBlockIdx / 2;
        this->cubeSubIdx = this->vecBlockIdx % 2;
    } else {
        this->cubeBlockIdx = GetBlockIdx();
        this->vecBlockIdx = this->cubeBlockIdx * 2;
        this->cubeSubIdx = 0;
    }
    this->pipe = tPipe;
    this->SetTiling(tiling);
    if (queryRope == nullptr) {
        this->ropeHeadSize = 0;
    }

    this->key_ptr = key;
    this->value_ptr = value;

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    if (this->tilingData->PFAinputParams.fromFused) {
        ListTensorDesc keyListTensorDescInit((__gm__ void*)key_ptr);
        ListTensorDesc valueListTensorDescInit((__gm__ void*)value_ptr);
        this->currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        this->currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)this->currentKey);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)this->currentValue);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }
    
    this->qRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
    this->kRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    if constexpr (PAGE_ATTENTION) {
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
    if (this->tilingData->PFAinputParams.isSoftMaxLseEnable) {
        this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
    }

    if (this->tilingData->PFAinputParams.hasLearnableSink) {
        this->hasSink = true;
        this->sinkGm.SetGlobalBuffer((__gm__ bfloat16_t *)learnableSink);
    }

    // 补齐到512， 统一按T处理
    int64_t mm1ResultSize = cubeS1BaseSize * s2BaseSize;
    int64_t mmNRatioOffset = CeilDiv(mm1ResultSize * this->tilingData->PFAcoreParams.nRatio, 128) * 128 * sizeof(T);
    int64_t mm2ResultSize = cubeS1BaseSize * valueDSizeAlign16;
    int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * 4;
    int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
    int64_t vector1OffsetPing = 0;
    int64_t vector1OffsetPong = mmNRatioOffset;

    // bmm1、bmm2 NZ场景，stage1Result不与bmm1Result共用空间，需要占用2倍mmNRatioOffset空间
    vector1OffsetPing = mmNRatioOffset * GM_DOUBLE_BUFFER;
    vector1OffsetPong = vector1OffsetPing + mmNRatioOffset / 2;
    bmm1AndVec1Ratio = GM_DOUBLE_BUFFER + 1;

    int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;

    // bmm1Result，占用2倍mmNRatioOffset空间
    this->mm1Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset));
    this->mm1Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset));

    // stage1Result，不占用/占用1倍/占用2倍mmNRatioOffset空间
    this->stage1Res[0].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + vector1OffsetPing));
    this->stage1Res[1].SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->cubeBlockIdx * totalOffset + vector1OffsetPong));

    // bmm2Result，占用2倍mmOffset空间
    this->mm2Res[0].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
    this->mm2Res[1].SetGlobalBuffer(
        (__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));

    // vec2阶段，占用2倍mmOffset空间，仅在D轴大于64的情况下出现
    this->vec2Res[0].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset *
                                                bmm1AndVec1Ratio + mm2Offset * 2));
    this->vec2Res[1].SetGlobalBuffer((__gm__ T *)(workspace + this->cubeBlockIdx * totalOffset + mmNRatioOffset *
                                                bmm1AndVec1Ratio + mm2Offset * 3));
    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = MLA_NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
    uint32_t kvCacheBlockSize = this->tilingData->PFAinputParams.blockSize;
    uint32_t maxBlockNumPerBatch = this->tilingData->PFAinputParams.blockTableDim2;

    if ASCEND_IS_AIC {
        matmulService.InitMm1GlobalTensor(queryGm, keyGm, qRopeGm, kRopeGm, mm1Res);
        matmulService.InitMm2GlobalTensor(stage1Res, valueGm, mm2Res, attentionOutGm);
        matmulService.InitPageAttentionInfo(blockTableGm, kvCacheBlockSize, maxBlockNumPerBatch);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::SetTiling(const TILING_TYPE *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->cubeS1BaseSize = this->tilingData->PFAcoreParams.s1BaseSize;
    this->vecS1BaseSize = this->cubeS1BaseSize / 2;
    this->s2BaseSize = this->tilingData->PFAcoreParams.s2BaseSize;
    this->dSize = this->tilingData->PFAinputParams.dSize;
    this->dSizeAlign16 = CeilDiv(this->tilingData->PFAinputParams.dSize, 16) * 16;
    this->valueDSize = this->tilingData->PFAinputParams.valueDSize;
    this->valueDSizeAlign16 = CeilDiv(this->tilingData->PFAinputParams.valueDSize, 16) * 16;
    this->ropeHeadSize = 64;
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::InitBuffer()
{
    if ASCEND_IS_AIV {
        constexpr uint64_t V1_BASE_TILE_SZ = 8 * 1024;

        // 计算中间结果buffer
        this->pipe->InitBuffer(this->stage1PingBuf, V1_BASE_TILE_SZ * sizeof(T));  // V1计算中间结果，32K

        // 输入buffer
        this->pipe->InitBuffer(this->pseTBuf, V1_BASE_TILE_SZ * sizeof(uint8_t));  // pse/second to,e atten mask/cast stage1 result, 8k

        // 输入buffer
        this->pipe->InitBuffer(this->stage2TBuf, 2 * STAGE2_TBUF_SIZE * sizeof(T));  // V32K

        // 临时空间
        this->pipe->InitBuffer(this->commonTBuf, V1_BASE_TILE_SZ * sizeof(T));  // api tmp，32K

        // 长生命周期空间
        this->pipe->InitBuffer(this->softmaxSumBuf[0], vecS1BaseSize * sizeof(float) * this->softmaxReduceSize);  // 1K if cubeS1BaseSize > 256 else 4K
        this->pipe->InitBuffer(this->softmaxSumBuf[1], vecS1BaseSize * sizeof(float) * this->softmaxReduceSize);  // 1K if cubeS1BaseSize > 256 else 4K
        this->pipe->InitBuffer(this->softmaxMaxBuf[0], vecS1BaseSize * sizeof(float) * this->softmaxReduceSize);  // 1K if cubeS1BaseSize > 256 else 4K
        this->pipe->InitBuffer(this->softmaxMaxBuf[1], vecS1BaseSize * sizeof(float) * this->softmaxReduceSize);  // 1K if cubeS1BaseSize > 256 else 4K
        this->pipe->InitBuffer(this->softmaxExpBuf[0], vecS1BaseSize * sizeof(float) * this->softmaxReduceSize);  // 1K if cubeS1BaseSize > 256 else 4K
        this->pipe->InitBuffer(this->softmaxExpBuf[1], vecS1BaseSize * sizeof(float) * this->softmaxReduceSize);  // 1K if cubeS1BaseSize > 256 else 4K

        if (this->hasSink) {
            this->pipe->InitBuffer(this->sinkBuf, vecS1BaseSize * sizeof(float) * this->softmaxReduceSize); // 1K if cubeS1BaseSize > 256 else 4K
            this->sinkUb = this->sinkBuf.template Get<float>();
        }

        // 输出空间
        this->pipe->InitBuffer(this->outPBuf, V1_BASE_TILE_SZ * sizeof(INPUT_T)); // 输出P到GM
        this->pipe->InitBuffer(this->outOBuf, STAGE2_TBUF_SIZE * sizeof(T));

        // 输入空间中的大小非规则部分
        this->pipe->InitBuffer(this->stage1PongBuf, (V1_BASE_TILE_SZ + (V1_BASE_TILE_SZ / 16)) * sizeof(T)); // load bmm1 result, 32K + 32K/16 = 34K +32k/16 是为了在Nz-Nd间搬运时间隔一个bank，避免bank冲突
        this->pipe->InitBuffer(this->maskTBufPing, (V1_BASE_TILE_SZ + (V1_BASE_TILE_SZ / 8)) * sizeof(uint8_t));  // atten mask 8K +8K/8 = 9K

        this->bmm1ResUb = this->stage1PongBuf.template Get<T>();
        this->v1CalcUb = this->stage1PingBuf.template Get<T>();
        this->curBmm2ResUb = this->stage2TBuf.template Get<T>();
        this->bmm2ResUb = curBmm2ResUb[STAGE2_TBUF_SIZE];
        this->bmm2ResOutUb = this->outOBuf.template Get<T>();
        this->softmaxLseTmpUb = this->commonTBuf.template Get<T>()[V1_BASE_TILE_SZ - vecS1BaseSize - (vecS1BaseSize * ONE_BLK_SIZE / sizeof(T))]; // 公用commTBuf，位置放在最后，避免和bmm2用到的commTBuf部分冲突。分为两半，分别用于计算log和广播后搬出结果
    } else {
        matmulService.InitBuffers(this->pipe);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::AllocEvent()
{
    if ASCEND_IS_AIV {
        this->v1ReadyToInBmm1ResEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        this->v1ReadyToInPseMaskEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        this->v1Bmm1ResMte2EndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->v1PseMaskMte2EndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->v1VecEndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        this->v1Mte3EndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

        SetFlag<HardEvent::V_MTE2>(this->v1ReadyToInBmm1ResEvent);
        SetFlag<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
        SetFlag<HardEvent::MTE3_V>(this->v1Mte3EndEvent);

        this->v2VecFreeEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        this->v2VecEndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        this->v2Mte2EndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->v2Mte3EndEvent = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

        SetFlag<HardEvent::V_MTE2>(this->v2VecFreeEvent);
        SetFlag<HardEvent::MTE3_V>(this->v2Mte3EndEvent);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::FreeEvent()
{
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::MTE3_V>(this->v1Mte3EndEvent);
        WaitFlag<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
        WaitFlag<HardEvent::V_MTE2>(this->v1ReadyToInBmm1ResEvent);

        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(this->v1ReadyToInBmm1ResEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->v1Bmm1ResMte2EndEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->v1PseMaskMte2EndEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(this->v1VecEndEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(this->v1Mte3EndEvent);

        WaitFlag<HardEvent::MTE3_V>(this->v2Mte3EndEvent);
        WaitFlag<HardEvent::V_MTE2>(this->v2VecFreeEvent);

        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(this->v2VecFreeEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(this->v2VecEndEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->v2Mte2EndEvent);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(this->v2Mte3EndEvent);
    }
}


template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ComputeConstexpr()
{
    // 计算轴的乘积
    this->s1OuterSize = this->tilingData->PFAcoreParams.s1OuterSize;
    this->s1D = this->tilingData->PFAinputParams.s1Size * dSize;
    this->s2D = this->tilingData->PFAinputParams.s2Size * dSize;
    this->gD = this->tilingData->PFAinputParams.gSize * dSize;
    this->n2D = this->tilingData->PFAinputParams.n2Size * dSize;
    this->s1S2 = this->tilingData->PFAinputParams.s1Size * this->tilingData->PFAinputParams.s2Size;
    this->gS1 = this->tilingData->PFAinputParams.gSize * this->tilingData->PFAinputParams.s1Size;
    this->n2G = this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAinputParams.gSize;
    this->gS1o = this->tilingData->PFAinputParams.gSize * this->s1OuterSize;

    this->bN2D = this->tilingData->PFAinputParams.bSize * n2D;
    this->n2GS1o = this->tilingData->PFAinputParams.n2Size * this->gS1o;
    this->gS1D = this->tilingData->PFAinputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->PFAinputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->PFAinputParams.n2Size * this->gD;
    this->bN2GD = this->tilingData->PFAinputParams.bSize * n2GD;
    this->gS1S2 = this->tilingData->PFAinputParams.gSize * this->s1S2;
    this->n2GS1 = this->tilingData->PFAinputParams.n2Size * this->gS1;

    this->n2GS1D = this->tilingData->PFAinputParams.n2Size * this->gS1D;
    this->n2GS1S2 = this->tilingData->PFAinputParams.n2Size * this->gS1S2;

    this->gRopeD = this->tilingData->PFAinputParams.gSize * ropeHeadSize;
    this->s2RopeD = this->tilingData->PFAinputParams.s2Size * ropeHeadSize;
    this->n2S2RopeD = this->tilingData->PFAinputParams.n2Size * this->s2RopeD;
    this->s1BaseRopeD = this->cubeS1BaseSize * ropeHeadSize;
    this->s1RopeD = this->tilingData->PFAinputParams.s1Size * ropeHeadSize;
    this->gS1RopeD = this->tilingData->PFAinputParams.gSize * this->s1RopeD;
    this->n2GS1RopeD = this->n2GS1 * ropeHeadSize;
    this->n2RopeD = this->tilingData->PFAinputParams.n2Size * ropeHeadSize;
    this->bN2RopeD = this->tilingData->PFAinputParams.bSize * this->n2RopeD;
    this->n2GRopeD = this->tilingData->PFAinputParams.n2Size * this->gRopeD;
    this->bN2GRopeD = this->tilingData->PFAinputParams.bSize * this->n2GRopeD;
    this->s1BaseN2GRopeD = this->cubeS1BaseSize * this->n2GRopeD;
    this->s1BaseBN2GRopeD = this->cubeS1BaseSize * this->bN2GRopeD;
    this->s2BaseNRatioSize = this->s2BaseSize * this->tilingData->PFAcoreParams.nRatio;
    this->s2BaseN2RopeD = this->s2BaseSize * this->n2RopeD;
    this->s2BaseNratioN2RopeD = this->s2BaseN2RopeD * this->tilingData->PFAcoreParams.nRatio;
    this->s2BaseNratioRopeD = this->s2BaseNRatioSize * ropeHeadSize;

    this->gD2 = this->tilingData->PFAinputParams.gSize * valueDSize;
    this->n2D2 = this->tilingData->PFAinputParams.n2Size * valueDSize;
    this->s1D2 = this->tilingData->PFAinputParams.s1Size * valueDSize;
    this->gS1D2 = this->tilingData->PFAinputParams.gSize * this->s1D2;
    this->n2GS1D2 = this->tilingData->PFAinputParams.n2Size * this->gS1D2;
    this->s2D2 = this->tilingData->PFAinputParams.s2Size * valueDSize;
    this->n2S2D2 = this->tilingData->PFAinputParams.n2Size * this->s2D2;
    this->bN2D2 = this->tilingData->PFAinputParams.bSize * this->n2D2;
    this->n2GD2 = this->tilingData->PFAinputParams.n2Size * this->gD2;
    this->bN2GD2 = this->tilingData->PFAinputParams.bSize * this->n2GD2;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        this->s1BaseD = this->cubeS1BaseSize * this->dSize;
        this->s1BaseD2 = this->cubeS1BaseSize * this->valueDSize;
        this->s2BaseNratioD = this->s2BaseNRatioSize * this->dSize;
        this->mm1Ka = this->dSize;
        this->mm1Kb = this->dSize;
        this->mm2Kb = this->valueDSize;
    }

    this->softmaxReduceSize = 1;

    this->kvCacheBlockSize = this->tilingData->PFAinputParams.blockSize;
    this->maxBlockNumPerBatch = this->tilingData->PFAinputParams.maxBlockNumPerBatch;

    if ASCEND_IS_AIC {
        matmulService.InitParams(dSize, valueDSize, ropeHeadSize, this->mm1Ka, this->mm1Kb, this->mm1RopeKa, this->mm1RopeKb, this->mm2Kb);
    }
}

// sparse functions
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit)
{
    // 计算sparse场景下s1的循环范围
    if constexpr (InputLayoutIsTNDLike()) {
        // sparse场景下负载均衡后每个核获取的结果
        multiCoreInnerOffset = this->tilingData->PFAmultiCoreParams.sparseStartIdx[this->cubeBlockIdx];
        if (likely((this->tilingData->PFAmultiCoreParams.coreNum - 2) > this->vecBlockIdx)) {
            multiCoreInnerLimit = this->tilingData->PFAmultiCoreParams.sparseStartIdx[this->cubeBlockIdx + 1];
        } else {
            multiCoreInnerLimit = this->tilingData->PFAmultiCoreParams.totalSize;
        }
    } else {
        if (this->tilingData->PFAinputParams.sparseType > 0) {
            // sparse场景下负载均衡后每个核获取的结果
            multiCoreInnerOffset = this->tilingData->PFAmultiCoreParams.sparseStartIdx[this->cubeBlockIdx];
            if (likely((this->tilingData->PFAmultiCoreParams.coreNum - 2) > this->vecBlockIdx)) {
                multiCoreInnerLimit = this->tilingData->PFAmultiCoreParams.sparseStartIdx[this->cubeBlockIdx + 1];
            } else {
                multiCoreInnerLimit = this->tilingData->PFAmultiCoreParams.totalSize;
            }
        }
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::GetS2LoopRange()
{
    // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
    if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, this->s2Size);
    } else if (this->tilingData->PFAinputParams.sparseType ==
            static_cast<uint8_t>(SparseModeEnum::BAND)) { // 对角线往外扩散场景, s1和s2可能不同
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize -
                            this->tilingData->PFAcoreParams.s1SparseValidSize, 0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + this->tilingData->PFAcoreParams.s2SparseValidSize,
                            this->s2Size);
    } else { // 其它场景, 如无attention mask
        this->s2StartIdx = 0;
        this->s2EndIdx = this->s2Size;
        return;
    }

    if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        // s1baseSize行都无效时, 将startIdx设置为0, endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min(this->s2Size, 128L);
        }
    }
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::Process()
{
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->s1OuterSize;
    this->s1Size = this->tilingData->PFAinputParams.s1Size;
    this->s2Size = this->tilingData->PFAinputParams.s2Size;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId,
    int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = multiCoreInnerIdx % 2;
    extraInfo.s1Size = this->tilingData->PFAinputParams.s1Size;
    extraInfo.s2Size = this->tilingData->PFAinputParams.s2Size;
    extraInfo.attenB1SSOffset = extraInfo.boIdx * this->s1S2;
    extraInfo.attenMaskS2Size = this->tilingData->PFAinputParams.attenMaskS2Size;
    extraInfo.s2SizeAcc = extraInfo.boIdx * extraInfo.s2Size;
    extraInfo.s1SizeAcc = extraInfo.boIdx * extraInfo.s1Size;
    extraInfo.cubeS1RealSize = Min(this->cubeS1BaseSize,
                                this->tilingData->PFAinputParams.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize;
    if (this->cubeSubIdx == 1) {
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
    }

    this->ComputeBmm1Tail(extraInfo);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
    typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s2RealSize = this->s2BaseNRatioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
    extraInfo.s2LastLoop = false;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
        extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
        extraInfo.s2LastLoop = true;
        extraInfo.duplicateMask = (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
            extraInfo.duplicateMask = 0xffffffffffffffff &
                (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        }
    }

    extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // 基本块为8 * 1024大小
    extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

    if (dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 1024 / valueDSizeAlign16 * 8;
    } else {
        extraInfo.vec2S1BaseSize = Min(128, extraInfo.s1RealSize);
    }
    if constexpr (bmm1Format == CubeFormat::NZ) {
        extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    } else {
        extraInfo.needNz2Nd = 0;
    }
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline uint64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::CalcBmm1TensorAGmOffset(const SplitSameABExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        bOffset = extraInfo.boIdx * this->n2GS1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // same as "ngbsd"
        bOffset = extraInfo.s1SizeAcc * this->n2GD;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * this->dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND) {
        // same as "ngbsd"
        n2Offset = extraInfo.n2oIdx * this->gS1D;
        gOffset = extraInfo.goIdx * this->s1D;
        bOffset = extraInfo.s1SizeAcc * this->dSize;
        s1Offset = extraInfo.s1oIdx * this->s1BaseD;
    }

    return bOffset + n2Offset + gOffset + s1Offset;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline uint64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::CalcBmm1QRopeGmOffset(const SplitSameABExtraInfo &extraInfo)
{
    int64_t ropeBOffset = 0;
    int64_t ropeS1Offset = 0;
    int64_t ropeN2Offset = 0;
    int64_t ropeGOffset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        ropeBOffset = extraInfo.s1SizeAcc * this->n2GRopeD;
        ropeS1Offset = extraInfo.s1oIdx * this->s1BaseN2GRopeD;
        ropeN2Offset = extraInfo.n2oIdx * this->gRopeD;
        ropeGOffset = extraInfo.goIdx * this->ropeHeadSize;
    } else {
        // NTD or NTD_TND
        ropeN2Offset = extraInfo.n2oIdx * this->gS1RopeD;
        ropeGOffset = extraInfo.goIdx * this->s1RopeD;
        ropeBOffset = extraInfo.s1SizeAcc * this->ropeHeadSize;
        ropeS1Offset = extraInfo.s1oIdx * this->s1BaseRopeD;
    }

    return ropeBOffset + ropeS1Offset + ropeN2Offset + ropeGOffset;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline uint64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, 
    bmm1Format, mmPolicyType, pageAttention>::CalcBmm1TensorBGmOffset(const SplitSameABExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = extraInfo.s2SizeAcc * this->n2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
        n2Offset = extraInfo.n2oIdx * this->dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND) {
        n2Offset = extraInfo.n2oIdx * this->s2D;
        bOffset = extraInfo.s2SizeAcc * this->dSize;
        s2Offset = extraInfo.s2StartIdx * this->dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    }

    return bOffset + n2Offset + s2Offset;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline uint64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::CalcBmm1KRopeGmOffset(const SplitSameABExtraInfo &extraInfo)
{
    int64_t ropeBOffset = 0;
    int64_t ropeS2Offset = 0;
    int64_t ropeN2Offset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        ropeBOffset = extraInfo.s2SizeAcc * this->n2RopeD;
        ropeS2Offset = extraInfo.s2StartIdx * this->n2RopeD + extraInfo.s2LoopCount * this->s2BaseNratioN2RopeD;
        ropeN2Offset =  extraInfo.n2oIdx * ropeHeadSize;
    } else {
        // NTD or NTD_TND
        ropeN2Offset = extraInfo.n2oIdx * this->s2RopeD;
        ropeBOffset = extraInfo.s2SizeAcc * ropeHeadSize;
        ropeS2Offset = extraInfo.s2StartIdx * ropeHeadSize + extraInfo.s2LoopCount * this->s2BaseNratioRopeD;
    }

    return ropeBOffset + ropeS2Offset + ropeN2Offset;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
    LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx)
{
    auto nzResUbTmp = nzResUb.template ReinterpretCast<half>();
    auto vec1ResUbTmp = vec1ResUb.template ReinterpretCast<half>();
    int64_t s21 = CeilDiv(extraInfo.s2AlignedSize, 16L);
    int64_t s2InnerLoop = s21 / 8L;
    int64_t s2InnerRemain = s21 % 8L;
    int64_t repeatMaxSize16 = this->repeatMaxBytes / sizeof(INPUT_T);
    CopyRepeatParams repeatParams;
    repeatParams.srcStride = 1;
    repeatParams.dstStride = extraInfo.vec1S1RealSize + 1; // 防止bank冲突，所以+1
    repeatParams.srcRepeatSize = extraInfo.s2RealSizeAlign64 / 16;
    repeatParams.dstRepeatSize = 1;
    int32_t s1OuterLoop = extraInfo.vec1S1RealSize / this->repeatMaxTimes;
    int32_t s1OuterRemain = extraInfo.vec1S1RealSize % this->repeatMaxTimes;
    int32_t s1OuterBmm1Offset = this->repeatMaxTimes * extraInfo.s2RealSizeAlign64;
    int32_t s1OuterTempOffset = this->repeatMaxTimes * 16;
    int64_t vec1Offset = extraInfo.vecCoreOffset * 16 + loopIdx * extraInfo.vec1S1BaseSize * 16;
    int64_t offsetJ = 8 * extraInfo.vec1S1RealSize * 16 + 128;
    PipeBarrier<PIPE_V>();
    for (int64_t outerIndex = 0; outerIndex < s1OuterLoop; ++outerIndex) {
        for (int64_t j = 0; j < s2InnerLoop; ++j) {
            AscendC::Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + j * offsetJ],
                    vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + j * 128],
                    repeatMaxSize16, this->repeatMaxTimes, repeatParams);
        }
        if (s2InnerRemain) {
            AscendC::Copy(nzResUbTmp[outerIndex * s1OuterTempOffset + s2InnerLoop * offsetJ],
                    vec1ResUbTmp[outerIndex * s1OuterBmm1Offset + s2InnerLoop * 128],
                    s2InnerRemain * 16, this->repeatMaxTimes, repeatParams);
        }
    }

    if (s1OuterRemain) {
        for (int64_t j = 0; j < s2InnerLoop; ++j) {
            AscendC::Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + j * offsetJ],
                    vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + j * 128],
                    repeatMaxSize16, s1OuterRemain, repeatParams);
        }
        if (s2InnerRemain) {
            AscendC::Copy(nzResUbTmp[s1OuterLoop * s1OuterTempOffset + s2InnerLoop * offsetJ],
                    vec1ResUbTmp[s1OuterLoop * s1OuterBmm1Offset + s2InnerLoop * 128],
                    s2InnerRemain * 16, s1OuterRemain, repeatParams);
        }
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s21;
    dataCopyParams.blockLen = extraInfo.vec1S1RealSize;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = CeilDiv(extraInfo.cubeS1RealSize, 16) * 16 - extraInfo.vec1S1RealSize;
    DataCopy(this->stage1Res[extraInfo.taskIdMod2][vec1Offset], nzResUb, dataCopyParams);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ProcessVec1(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }

    LocalTensor<T> &bmm1ResNdUb = (extraInfo.needNz2Nd == 0) ? this->bmm1ResUb : this->v1CalcUb; 

    extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
    for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; ++loopIdx) {
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vec1S1RealSize = extraInfo.s1RealSize * this->gBaseSize - loopIdx * extraInfo.vec1S1BaseSize;
        }

        WaitFlag<HardEvent::V_MTE2>(this->v1ReadyToInBmm1ResEvent);
        this->GetBmm1Result(extraInfo, bmm1ResNdUb, loopIdx);

        // mul或attenMask的计算依赖bmm1计算结果
        SetFlag<HardEvent::MTE2_V>(this->v1Bmm1ResMte2EndEvent);
        WaitFlag<HardEvent::MTE2_V>(this->v1Bmm1ResMte2EndEvent);

        if constexpr (hasAtten) {
            WaitFlag<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
            this->CopyInAttenMask(extraInfo, loopIdx, - 1);
        }

        Muls(this->v1CalcUb, bmm1ResNdUb, static_cast<T>(this->tilingData->PFAinputParams.scaleValue),
            extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
        SetFlag<HardEvent::V_MTE2>(this->v1ReadyToInBmm1ResEvent);
 
        if constexpr (hasAtten) {
            SelectWithBytesMaskShapeInfo shapeInfo;
            shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
            shapeInfo.srcLastAxis = extraInfo.s2RealSizeAlign64;
            shapeInfo.maskLastAxis = extraInfo.s2RealSizeAlign64;
            this->v1CalcUb.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
            if (this->attenMaskComputeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE &&
                this->attenMaskComputeMode != AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                uint8_t maskType = (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) ? 1 : 0;
                LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
                this->ComputeAttenMask(shapeInfo, this->v1CalcUb, attenMaskUb, maskType, this->v1PseMaskMte2EndEvent);
            }

            if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE ||
                this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                SetFlag<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
                WaitFlag<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
                this->CopyInAttenMask(extraInfo, loopIdx, this->attenMaskOffsetPre, true);
                LocalTensor<uint8_t> secondTimeMaskUb;
                uint8_t maskType;
                secondTimeMaskUb = this->pseTBuf.template Get<uint8_t>();
                maskType = 1;
                this->ComputeAttenMask(shapeInfo, this->v1CalcUb, secondTimeMaskUb, maskType, this->v1PseMaskMte2EndEvent);
            }
            SetFlag<HardEvent::V_MTE2>(this->v1ReadyToInPseMaskEvent);
        }

        this->SoftMaxCompute(extraInfo, this->v1CalcUb, loopIdx);
        WaitFlag<HardEvent::MTE3_V>(this->v1Mte3EndEvent);
        PipeBarrier<PIPE_V>();
        if constexpr (!IsSameType<T, INPUT_T>::value) {
            LocalTensor<INPUT_T> stage1CastTensor;
            stage1CastTensor = this->outPBuf.template Get<INPUT_T>();
            Cast(stage1CastTensor, this->v1CalcUb, RoundMode::CAST_ROUND,
                extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
            SetFlag<HardEvent::V_MTE3>(this->v1VecEndEvent);
            WaitFlag<HardEvent::V_MTE3>(this->v1VecEndEvent);
            if constexpr (! BMM2_A_ISFIXED_ND) {
                LocalTensor<INPUT_T> stage1NzTensor = this->commonTBuf.template Get<INPUT_T>();
                NdToNz(extraInfo, stage1NzTensor, stage1CastTensor, loopIdx);
            } else {
                DataCopyParams dataCopyParams;
                dataCopyParams.dstStride = 0;
                if (extraInfo.s2LastLoop) {
                    dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
                    dataCopyParams.blockLen = extraInfo.s2AlignedSize / 16;
                    dataCopyParams.srcStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2AlignedSize) / 16;
                } else {
                    dataCopyParams.blockCount = 1;
                    dataCopyParams.blockLen = extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize / 16;
                    dataCopyParams.srcStride = 0;
                }
                DataCopy(
                    this->stage1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset +
                        loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2AlignedSize],
                        stage1CastTensor, dataCopyParams);
            }
        } else {
            SetFlag<HardEvent::V_MTE3>(this->v1VecEndEvent);
            WaitFlag<HardEvent::V_MTE3>(this->v1VecEndEvent);
            DataCopyParams dataCopyParams;
            dataCopyParams.dstStride = 0;
            if (extraInfo.s2LastLoop) {
                dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
                dataCopyParams.blockLen = extraInfo.s2AlignedSize / 8;
                dataCopyParams.srcStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2AlignedSize) / 8;
            } else {
                dataCopyParams.blockCount = 1;
                dataCopyParams.blockLen = extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize / 8;
                dataCopyParams.srcStride = 0;
            }
            DataCopy(
                this->stage1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset +
                    loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2AlignedSize],
                    this->v1CalcUb, dataCopyParams);
        }

        SetFlag<HardEvent::MTE3_V>(this->v1Mte3EndEvent);
    }

    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx,
    int64_t maskOffset, bool secondTime)
{
    if constexpr (hasAtten) {
        LocalTensor<uint8_t> attenMaskUb;
        if (secondTime) {
            attenMaskUb = this->pseTBuf.template Get<uint8_t>();
        } else {
            attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
        }
        if (maskOffset == -1) {
            maskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);
        }
        if (this->attenMaskComputeMode == AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
            return;
        }
        if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE ||
            this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE) {
            maskOffset = this->attenMaskOffsetPre;
        }

        int64_t s2StrideSize = this->tilingData->PFAinputParams.attenMaskS2Size;
        if constexpr (InputLayoutIsTNDLike()) {
            if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskS1S2) {
                s2StrideSize = this->tilingData->PFAinputParams.s2Size;
            } else if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskTT) {
                s2StrideSize = this->s2SizeSum;
            }
            // band compress mode
            if (this->tilingData->PFAinputParams.attenMaskCompressMode !=
                static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
                s2StrideSize = this->tilingData->PFAinputParams.attenMaskS2Size;
            }
        }
        int64_t maskSize = extraInfo.vec1S1RealSize / extraInfo.gBaseSize * (CeilDiv(extraInfo.s2RealSize, 64) * 64);

        if (unlikely(this->notSplitG)) {
            for (int64_t loop = 0; loop < extraInfo.gBaseSize; ++loop) {
                LocalTensor<uint8_t> dst = attenMaskUb[loop * maskSize];
                BoolCopyIn(dst, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize / extraInfo.gBaseSize, extraInfo.s2RealSize,
                        s2StrideSize, 64);
            }
        } else {
            BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
                    s2StrideSize, 64);
        }
        return;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline int64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten) {
        if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
        }
        if constexpr (InputLayoutIsTNDLike()) {
            int64_t delta = 0;
            int64_t deltaPre = 0;
            int64_t deltaN = static_cast<int64_t>((extraInfo.s1Size)) - static_cast<int64_t>((extraInfo.s2Size));
            int64_t s1Offset = extraInfo.s1oIdx * this->cubeS1BaseSize + (extraInfo.vecCoreOffset  +
                            loopIdx * extraInfo.vec1S1BaseSize) / extraInfo.gBaseSize;
            int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNRatioSize;
            if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset;
            } else if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                    static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
            } else if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                    static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
                int64_t tmpPre = this->tilingData->PFAinputParams.preTokens;
                int64_t tmpNext = this->tilingData->PFAinputParams.nextTokens;
                int64_t transPreTokens = extraInfo.s1Size - Max(extraInfo.s2Size - tmpPre, 0);
                int64_t transNextTokens = extraInfo.s2Size - Max(extraInfo.s1Size - tmpNext, 0);
                deltaPre = s1Offset - s2Offset - transPreTokens - 1;
                int64_t maskOffsetPre =
                    ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize / extraInfo.gBaseSize, this->s2BaseNRatioSize,
                                        this->tilingData->PFAinputParams.attenMaskS2Size);
                this->attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask operation.
                delta = s1Offset - s2Offset + transNextTokens;
            } else if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                    static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE)) {
                if (extraInfo.boIdx == this->tilingData->PFAinputParams.bandIndex) {
                    delta = s1Offset - s2Offset - deltaN + this->tilingData->PFAinputParams.nextTokens;
                } else {
                    delta = s1Offset - s2Offset - deltaN;
                }
            } else if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                    static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE)) {
                if (extraInfo.boIdx == this->tilingData->PFAinputParams.bandIndex) {
                    delta = s1Offset - s2Offset + extraInfo.s2Size -
                            Max(extraInfo.s1Size - this->tilingData->PFAinputParams.nextTokens, 0);
                } else {
                    delta = s1Offset - s2Offset;
                }
            } else {
                return 0;
            }
            this->GetAttenMaskComputeMode(delta, deltaPre, s1Offset, extraInfo);
            return ComputeOffsetForCausal(delta, extraInfo.vec1S1BaseSize / extraInfo.gBaseSize, this->s2BaseNRatioSize,
                                        this->tilingData->PFAinputParams.attenMaskS2Size);
        }
        // compress mode
        int64_t deltaCausalOrNext = 0;
        int64_t deltaPre = 0;
        int64_t deltaN = extraInfo.s1Size - extraInfo.s2Size;
        int64_t s1Offset = extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                        loopIdx * extraInfo.vec1S1BaseSize;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNRatioSize;
        if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset;
        } else if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
        } else if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            deltaPre = s1Offset - s2Offset - this->tilingData->PFAinputParams.preTokens - 1;
            this->attenMaskOffsetPre =
                ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNRatioSize,
                                    this->tilingData->PFAinputParams.attenMaskS2Size);
            deltaCausalOrNext = s1Offset - s2Offset + this->tilingData->PFAinputParams.nextTokens;
        } else {
            return 0;
        }
        this->GetAttenMaskComputeMode(deltaCausalOrNext, deltaPre, s1Offset, extraInfo);
        return ComputeOffsetForCausal(deltaCausalOrNext, extraInfo.vec1S1BaseSize, s2BaseNRatioSize,
                                    this->tilingData->PFAinputParams.attenMaskS2Size);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre,
    int64_t s1Offset, SplitSameABExtraInfo &extraInfo)
{
    if constexpr (hasAtten) {
        int64_t causalOrNextFactor = deltaCausalOrNext - extraInfo.s2AlignedSize;
        if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
            this->tilingData->PFAinputParams.attenMaskCompressMode ==
                static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            if (causalOrNextFactor >= 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            }
            return;
        }
        if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            int64_t preFactor = deltaPre + 1 + extraInfo.vec1S1BaseSize;
            if (causalOrNextFactor >= 0 && preFactor <= 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else if (causalOrNextFactor < 0 && preFactor <= 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            } else if (causalOrNextFactor >= 0 && preFactor > 0) {
                this->attenMaskComputeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
            } else {
                this->attenMaskComputeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
            }
        }
        return;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline int64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                            loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2Size;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNRatioSize;
        if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskBN2GS1S2) {
            bOffset = extraInfo.attenB1SSOffset * this->n2G;
            n2Offset = extraInfo.n2oIdx * this->tilingData->PFAinputParams.gSize * extraInfo.s1Size * extraInfo.s2Size;
            gOffset = extraInfo.goIdx * extraInfo.s1Size * extraInfo.s2Size;
        } else if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskBS1S2) {
            bOffset = extraInfo.attenB1SSOffset;
        } else if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskS1S2) {
            s1Offset = (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                    loopIdx * extraInfo.vec1S1BaseSize) * this->tilingData->PFAinputParams.s2Size;
        } else if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskTT) {
            s1Offset = extraInfo.s1SizeAcc + extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                    loopIdx * extraInfo.vec1S1BaseSize;
            s1Offset = s1Offset * this->s2SizeSum;
            s2Offset = s2Offset + extraInfo.s2SizeAcc;
        }
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::GetBmm1Result(SplitSameABExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
{
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if (extraInfo.needNz2Nd == 1) {
            Nz2NdInfo nz2NdInfo;
            nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize * extraInfo.gBaseSize;
            nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec1S1BaseSize;
            nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec1S1RealSize;
            nz2NdInfo.ndLastAxis = extraInfo.s2RealSizeAlign64;
            nz2NdInfo.loopIdx = loopIdx;
            nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
            LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
            if constexpr (IsSameType<INPUT_T, float>::value) {
                // FP32场景，需要等待vec1上一轮输出搬完，释放stage1PingBUf/v1CalcUb
                event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            }
            NzToNd(nz2NdInfo, this->mm1Res[extraInfo.taskIdMod2], tempUb, bmm1ResUb);
            PipeBarrier<PIPE_V>();
            return;
        }
    }
    if (likely(extraInfo.s2RealSizeAlign64 == extraInfo.s2RealSize)) {
        DataCopy2D(bmm1ResUb,
                this->mm1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) *
                                                    extraInfo.s2RealSize],
                extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
    } else {
        DataCopyParams dataCopyParams;
        DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
        dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        if (extraInfo.s2LastLoop) {
            dataCopyParams.dstStride = (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSize) / 8;
            dataCopyPadParams.isPad = false;
        } else {
            dataCopyParams.dstStride = 0;
            dataCopyPadParams.isPad = true;
            dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
            if (dataCopyPadParams.rightPadding > blockSize) {
                dataCopyPadParams.rightPadding -= blockSize;
                dataCopyParams.dstStride = 1;
                int32_t s2BlockAlignedSize = CeilDiv(extraInfo.s2RealSize, blockSize) * blockSize;
                Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, blockSize, extraInfo.vec1S1RealSize, 0,
                            extraInfo.s2AlignedSize * sizeof(T) / blockBytes);
            }
            dataCopyPadParams.paddingValue = 0;
        }
        DataCopyPad(bmm1ResUb,
                    this->mm1Res[extraInfo.taskIdMod2][(extraInfo.vecCoreOffset + loopIdx * extraInfo.vec1S1BaseSize) *
                                                    extraInfo.s2RealSize], dataCopyParams, dataCopyPadParams);
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo,
    LocalTensor<T> &bmm1ResUb, LocalTensor<uint8_t> &attenMaskUb, const uint8_t maskType, event_t vWaitMte2)
{
    if constexpr (hasAtten) {
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
        attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);
        SetFlag<HardEvent::MTE2_V>(vWaitMte2);
        WaitFlag<HardEvent::MTE2_V>(vWaitMte2);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        if (maskType == 0) {
            SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
        } else {
            SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
        }
        return;
    }
}


template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo, int64_t vec1S1realSplitN)
{
    if constexpr (implMode == ImplModeEnum::AA_HIGH_PRECISION) {
        if (unlikely(extraInfo.realSplitN == 1)) {
            bool res = IsIncludeInvalidLine(this->softMaxCheckRes, vec1S1realSplitN);
            UpdateSoftMaxCheckRes(this->softMaxCheckRes, 0, res);
        } else {
            int64_t avg = CeilDiv(vec1S1realSplitN, extraInfo.realSplitN);
            for (int64_t i = 0; i < extraInfo.realSplitN; ++i) {
                int64_t endIdx = Min((i + 1) * avg, vec1S1realSplitN) - 1;
                if (IsIncludeInvalidLine(this->softMaxCheckRes, endIdx, i * avg)) {
                    UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, true);
                } else {
                    UpdateSoftMaxCheckRes(this->softMaxCheckRes, i, false);
                }
            }
        }
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo,
    LocalTensor<T> &srcTensor, LocalTensor<T> &maxUb, int64_t loopIdx)
{
    if constexpr (implMode == ImplModeEnum::AA_HIGH_PRECISION) {
        if (loopIdx == 0 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            int64_t vec1S1BaseSize = Min(8, extraInfo.s1RealSize);
            int64_t vec1S1realSplitN = CeilDiv(extraInfo.s1RealSize, vec1S1BaseSize);
            if (extraInfo.realSplitN != vec1S1realSplitN) {
                this->SoftMaxCheckResCompress(extraInfo, vec1S1realSplitN);
            }
        }

        if (hasInvalidLine(this->softMaxCheckRes, loopIdx)) {
            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
                static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
            bool res = false;
            if (this->softmaxReduceSize == 1) {
                res = AdjustSoftMaxRes<T, T, false, 1>(srcTensor, maxUb, this->negativeIntScalar,
                                                    0.0, softmaxShapeInfo);
            } else {
                res = AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
            }
            UpdateSoftMaxCheckRes(this->softMaxCheckRes, loopIdx, res);
        }

        if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit &&
            IsIncludeInvalidLine(this->softMaxCheckRes, extraInfo.realSplitN)) {
            LocalTensor<T> maxTensor = this->softmaxMaxBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();
            LocalTensor<T> sumTensor;
            sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

            SoftMaxShapeInfo softmaxShapeInfo{
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
            if (this->softmaxReduceSize == 1) {
                AdjustSoftMaxRes<T, T, false, 1>(sumTensor, maxTensor, this->negativeIntScalar,
                                                this->positiveFloatScalar, softmaxShapeInfo);
            } else {
                AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar,
                                    this->positiveFloatScalar, softmaxShapeInfo);
            }
        }
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::SoftMaxCompute(SplitSameABExtraInfo &extraInfo,
    LocalTensor<T> &srcTensor, int64_t loopIdx)
{
    int64_t vec1S1RealSize = CeilDiv(extraInfo.vec1S1RealSize, 8) * 8;
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(vec1S1RealSize),
                                static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(vec1S1RealSize),
                                    static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));
    srcTensor.SetSize(vec1S1RealSize * extraInfo.s2RealSizeAlign64);

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                            static_cast<uint32_t>(this->softmaxReduceSize)};
    LocalTensor<T> sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2]
                        .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];
    LocalTensor<T> maxUb = this->softmaxMaxBuf[extraInfo.multiCoreInnerIdxMod2]
                        .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    if (extraInfo.s2LastLoop) {
        if constexpr (hasAtten == false) {
            uint64_t mask[2] = {extraInfo.duplicateMask, 0};
            Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], *((float *)&MLA_NEGATIVE_INF_VALUE_FP32), mask, vec1S1RealSize,
                        1, extraInfo.s2RealSizeAlign64 * sizeof(T) / blockBytes);
        } else if (this->tilingData->PFAinputParams.attenMaskCompressMode !=
                static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE) || extraInfo.s2RealSize < 64) {
            uint64_t mask[2] = {extraInfo.duplicateMask, 0};
            Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], *((float *)&MLA_NEGATIVE_INF_VALUE_FP32), mask, vec1S1RealSize,
                        1, extraInfo.s2RealSizeAlign64 * sizeof(T) / blockBytes);
        }
    }

    LocalTensor<T> expUb = this->softmaxExpBuf[extraInfo.taskIdMod2]
                        .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];

    expUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    PipeBarrier<PIPE_V>();
    if (unlikely(extraInfo.s2LoopCount == 0)) {
        if (this->softmaxReduceSize == 1) {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                            extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            false, true, false, true);
            SoftmaxFlashV2<T, false, true, true, false, SOFTMAX_REDUCE_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                            extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            false, true);
            SoftmaxFlashV2<T, false, true, true, false, PFA_SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    } else {
        if (this->softmaxReduceSize == 1) {
                SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                                extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                                sizeof(T),
                                                                                apiTmpBuffer.GetSize() / sizeof(T),
                                                                                true, true, false, true);
                SoftmaxFlashV2<T, true, true, true, false, SOFTMAX_REDUCE_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        } else {
            SoftMaxTiling newTiling = AscendC::SoftMaxFlashV2TilingFuncImpl(vec1S1RealSize,
                                                                            extraInfo.s2RealSizeAlign64, sizeof(T),
                                                                            sizeof(T),
                                                                            apiTmpBuffer.GetSize() / sizeof(T),
                                                                            true, true);
            SoftmaxFlashV2<T, true, true, true, false, PFA_SOFTMAX_DEFAULT_CFG>(srcTensor, sumUb, maxUb, srcTensor, expUb,
                                                                            sumUb, maxUb, apiTmpBuffer, newTiling);
        }
    }

    if constexpr (implMode == ImplModeEnum::AA_HIGH_PRECISION) {
        if (unlikely(this->hasSparse4InvalidLine)) {
            if (extraInfo.s2Size > SPLIT_S2_SIZE_LIMIT) {
                this->InvalidLineSplitS2Process(extraInfo, srcTensor, maxUb, loopIdx);
            } else {
                SoftMaxShapeInfo softmaxShapeInfo{
                    static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64),
                    static_cast<uint32_t>(vec1S1RealSize), static_cast<uint32_t>(extraInfo.s2RealSizeAlign64)};
                if (this->softmaxReduceSize == 1) {
                    AdjustSoftMaxRes<T, T, false, 1>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
                } else {
                     AdjustSoftMaxRes<T, T>(srcTensor, maxUb, this->negativeIntScalar, 0.0, softmaxShapeInfo);
                }
            }
        }
    }
    if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        if (unlikely(hasSink)) {
            LocalTensor<T> sinkSoftmaxTemp = this->commonTBuf.template Get<T>();
            if (likely(!this->notSplitG)) {
                int64_t n1Offset = extraInfo.n2oIdx * this->tilingData->PFAinputParams.gSize + extraInfo.goIdx;
                Duplicate<float>(this->sinkUb, ToFloat(this->sinkGm.GetValue(n1Offset)), extraInfo.s1RealSize);
                PipeBarrier<PIPE_V>();
            } else {
                int64_t n1Offset;
                for (int64_t loop = extraInfo.gBaseSize - 1; loop >= 0; --loop) {
                    n1Offset = extraInfo.n2oIdx * this->tilingData->PFAinputParams.gSize + loop;
                    Duplicate(this->sinkUb, ToFloat(this->sinkGm.GetValue(n1Offset)), extraInfo.s1RealSize * (loop + 1));
                    PipeBarrier<PIPE_V>();
                }
            }
            auto size = extraInfo.s1RealSize * extraInfo.gBaseSize;
            LocalTensor<T> maxUb = this->softmaxMaxBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();
            LocalTensor<T> sumUb = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();
            Sub(sinkSoftmaxTemp, this->sinkUb, maxUb, size);
            PipeBarrier<PIPE_V>();
            Exp(sinkSoftmaxTemp, sinkSoftmaxTemp, size);
            PipeBarrier<PIPE_V>();
            Add(sumUb, sumUb, sinkSoftmaxTemp, size);
            PipeBarrier<PIPE_V>();
        }
        if (!this->tilingData->PFAinputParams.isSoftMaxLseEnable) {
            return;
        }
        extraInfo.softmaxLseOffset =
            (extraInfo.s1SizeAcc * this->n2G +
             extraInfo.n2oIdx * this->tilingData->PFAinputParams.gSize +
             extraInfo.goIdx + extraInfo.s1oIdx * this->cubeS1BaseSize *
             this->n2G + extraInfo.vecCoreOffset * this->n2G) * static_cast<int64_t>(1);
        LocalTensor<T> lseTensor = this->softmaxLseTmpUb[this->vecS1BaseSize * ONE_BLK_SIZE / sizeof(T)];
        LocalTensor<T> maxTensor = this->softmaxMaxBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();
        LocalTensor<T> sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();
        Log(lseTensor, sumTensor, extraInfo.s1RealSize);
        PipeBarrier<PIPE_V>();
        Add(lseTensor, lseTensor, maxTensor, extraInfo.s1RealSize);
        LocalTensor<T>& softmaxTemp = this->softmaxLseTmpUb;
        PipeBarrier<PIPE_V>();
        Brcb(softmaxTemp, lseTensor, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        if (unlikely(this->hasSparse4InvalidLine)) {
            SoftMaxShapeInfo softmaxFullShapeInfo{
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
            AdjustSoftMaxRes<T, T, false, 1>(softmaxTemp,  maxTensor, this->negativeIntScalar, 3e+99, softmaxFullShapeInfo); // 3e+99: 代表NAN
            PipeBarrier<PIPE_V>();
        }

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr() -> FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = extraInfo.s1RealSize;
        dataCopyParams.blockLen = sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (this->n2G - 1) * sizeof(T);
        DataCopyPad(this->softmaxLseGm[extraInfo.softmaxLseOffset], softmaxTemp, dataCopyParams);

        event_t lseEventMte3ToV = static_cast<event_t>(GetTPipePtr() -> FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(lseEventMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(lseEventMte3ToV);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline uint64_t
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::CalcBmm2TensorBGmOffset(const SplitSameABExtraInfo &extraInfo)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D2;
        n2Offset = extraInfo.n2oIdx * this->s2D2;
        s2Offset = extraInfo.s2StartIdx * valueDSize + extraInfo.s2LoopCount * s2BaseNRatioSize * valueDSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // TND
        bOffset = extraInfo.s2SizeAcc * this->n2D2;
        s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * this->s2BaseNRatioSize * this->n2D2;
        n2Offset = extraInfo.n2oIdx * this->valueDSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND) {
        // NTD or NTD_TND
        n2Offset = extraInfo.n2oIdx * this->s2D2;
        bOffset = extraInfo.s2SizeAcc * valueDSize;
        s2Offset = extraInfo.s2StartIdx * valueDSize + extraInfo.s2LoopCount * this->s2BaseNRatioSize * valueDSize;
    }

    return bOffset + n2Offset + s2Offset;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::ProcessVec2(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }
    // 获取缓存bmm2的计算结果
    auto& curBmm2ResTensor = unlikely(extraInfo.s2LoopCount == 0) ? this->bmm2ResUb :this->curBmm2ResUb;
    int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;

    for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; ++s1oIdx) {
        if (s1oIdx == vec2LoopLimit - 1) {
            extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
        }
        int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * this->valueDSizeAlign16 * this->gBaseSize;
        int64_t mm2ResOffset = (extraInfo.vecCoreOffset + s1oIdx * extraInfo.vec2S1BaseSize) * this->valueDSizeAlign16 * this->gBaseSize;
        WaitFlag<HardEvent::V_MTE2>(this->v2VecFreeEvent);

        Nz2NdInfo nz2NdInfo;
        nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
        nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec2S1BaseSize;
        nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec2S1RealSize;
        nz2NdInfo.ndLastAxis = this->valueDSizeAlign16;
        nz2NdInfo.loopIdx = s1oIdx;
        nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
 
        if constexpr (! BMM2_OUT_ISFIXED_ND) {
            LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
            NzToNd(nz2NdInfo, this->mm2Res[extraInfo.taskIdMod2], tempUb, curBmm2ResTensor);
            PipeBarrier<PIPE_V>();
        } else {
            int64_t bmmResOffset = nz2NdInfo.vecCoreOffset * this->valueDSize +
                                   nz2NdInfo.loopIdx * nz2NdInfo.ndFirstAxisBaseSize * this->valueDSize;
            DataCopy2D(curBmm2ResTensor,
                       this->mm2Res[extraInfo.taskIdMod2][bmmResOffset],
                       extraInfo.vec2S1RealSize * this->gBaseSize, this->valueDSizeAlign16, this->valueDSizeAlign16);                      
        }

        if (likely(extraInfo.s2LoopCount != 0)) {
            DataCopy(this->bmm2ResUb, this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], mm2ResCalcSize);
        }

        SetFlag<HardEvent::MTE2_V>(this->v2Mte2EndEvent);
        WaitFlag<HardEvent::MTE2_V>(this->v2Mte2EndEvent);

        if (likely(extraInfo.s2LoopCount != 0)) {
            this->Bmm2ResultMul(extraInfo, this->bmm2ResUb, s1oIdx);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(this->v2Mte3EndEvent);
            Add(this->bmm2ResOutUb, this->bmm2ResUb, curBmm2ResTensor, mm2ResCalcSize);
            SetFlag<HardEvent::V_MTE2>(this->v2VecFreeEvent);

            if (unlikely(extraInfo.s2LoopCount == extraInfo.s2LoopLimit)) {
                Bmm2ResultDiv(extraInfo, this->bmm2ResOutUb, s1oIdx);
            }
        } else {
            WaitFlag<HardEvent::MTE3_V>(this->v2Mte3EndEvent);
            if (unlikely(extraInfo.s2LoopCount == extraInfo.s2LoopLimit)) {
                Bmm2ResultDiv(extraInfo, curBmm2ResTensor, s1oIdx);
                SetFlag<HardEvent::V_MTE2>(this->v2VecFreeEvent);
            } 
        }

        if (likely(extraInfo.s2LoopCount < extraInfo.s2LoopLimit)) {
            SetFlag<HardEvent::V_MTE3>(this->v2VecEndEvent);
            WaitFlag<HardEvent::V_MTE3>(this->v2VecEndEvent);

            if (unlikely(extraInfo.s2LoopCount == 0)) {
                DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], curBmm2ResTensor, mm2ResCalcSize);
                SetFlag<HardEvent::V_MTE2>(this->v2VecFreeEvent);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            } else {
                DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], this->bmm2ResOutUb, mm2ResCalcSize);
            }
        } else {
            Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize);
        }

        SetFlag<HardEvent::MTE3_V>(this->v2Mte3EndEvent);
    }
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::Bmm2ResultMul(SplitSameABExtraInfo &extraInfo,
    LocalTensor<T> &bmm2ResUb, int64_t s1oIdx)
{
    PipeBarrier<PIPE_V>();
    LocalTensor<T> expUb = softmaxExpBuf[extraInfo.taskIdMod2].template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = this->valueDSizeAlign16 / blockSize;
    repeatParams.dstRepStride = this->valueDSizeAlign16 / blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = this->valueDSizeAlign16 / repeatMaxSize;
    int32_t remain = this->valueDSizeAlign16 % repeatMaxSize;
    int64_t softmaxTempOffset = s1oIdx * extraInfo.vec2S1BaseSize * 8;
    int64_t gsAxisSize = extraInfo.vec2S1RealSize * extraInfo.gBaseSize;

    if (this->softmaxReduceSize == 1) {
        LocalTensor<T> softmaxTemp = this->commonTBuf.template Get<T>();
        Brcb(softmaxTemp, expUb, (extraInfo.s1RealSize * extraInfo.gBaseSize + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; ++i) {
            Mul(bmm2ResUb[i * repeatMaxSize], softmaxTemp[softmaxTempOffset], bmm2ResUb[i * repeatMaxSize],
                repeatMaxSize, gsAxisSize, repeatParams);
        }
        if (remain) {
            Mul(bmm2ResUb[loop * repeatMaxSize], softmaxTemp[softmaxTempOffset],
                bmm2ResUb[loop * repeatMaxSize], remain, gsAxisSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; ++i) {
            Mul(bmm2ResUb[i * repeatMaxSize], expUb[softmaxTempOffset], bmm2ResUb[i * repeatMaxSize],
                repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (remain) {
            Mul(bmm2ResUb[loop * repeatMaxSize], expUb[softmaxTempOffset],
                bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::Bmm2ResultDiv(
    SplitSameABExtraInfo &extraInfo, LocalTensor<T> &srcBmm2ResUb, int64_t s1oIdx)
{
    LocalTensor<T> &bmm2ResOutUb = this->bmm2ResOutUb;
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = valueDSizeAlign16 / blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = valueDSizeAlign16 / blockSize;
    int32_t loop = valueDSizeAlign16 / repeatMaxSize;
    int32_t remain = valueDSizeAlign16 % repeatMaxSize;

    LocalTensor<float> sumUb = softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<float>();
    int32_t calcSize = sumUb.GetSize();
    PipeBarrier<PIPE_V>();
    if (this->softmaxReduceSize == 1) {
        LocalTensor<T> softmaxTemp = this->commonTBuf.template Get<T>();
        Brcb(softmaxTemp, sumUb, (extraInfo.s1RealSize * extraInfo.gBaseSize + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; ++i) {
            Div(bmm2ResOutUb[i * repeatMaxSize], srcBmm2ResUb[i * repeatMaxSize],
                softmaxTemp[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize * extraInfo.gBaseSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResOutUb[loop * repeatMaxSize], srcBmm2ResUb[loop * repeatMaxSize],
                softmaxTemp[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize * extraInfo.gBaseSize, repeatParams);
        }
    } else {
        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
            AscendC::Copy(commonBufTensor, sumUb, 64, calcSize / 64, {2, 1, 16, 8});
            AscendC::Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, {2, 1, 16, 8});
            LocalTensor<T> sumCastTensor =
                softmaxExpBuf[0].template Get<T>(this->tilingData->PFAtensorSizeParams.softmaxExpUbSize);

            Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
            for (int i = 0; i < loop; i++) {
                Div(bmm2ResOutUb[i * repeatMaxSize], srcBmm2ResUb[i * repeatMaxSize], sumCastTensor, repeatMaxSize,
                    extraInfo.vec2S1RealSize, repeatParams);
            }
            if (likely(remain)) {
                Div(bmm2ResOutUb[loop * repeatMaxSize], srcBmm2ResUb[loop * repeatMaxSize], sumCastTensor, remain,
                    extraInfo.vec2S1RealSize, repeatParams);
            }
        } else {
            for (int i = 0; i < loop; i++) {
                Div(bmm2ResOutUb[i * repeatMaxSize], srcBmm2ResUb[i * repeatMaxSize],
                    sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
            }
            if (likely(remain)) {
                Div(bmm2ResOutUb[loop * repeatMaxSize], srcBmm2ResUb[loop * repeatMaxSize],
                    sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
            }
        }
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType, pageAttention>::Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx,
    int64_t mm2ResCalcSize)
{
    static_assert(sizeof(INPUT_T) <= sizeof(T), "sizeof(INPUT_T) MUST <= sizeof(T)");
    LocalTensor<T> &bmm2ResOutUb = this->bmm2ResOutUb;
    LocalTensor<INPUT_T> &attenOut = reinterpret_cast<LocalTensor<INPUT_T>&>(bmm2ResOutUb);
    bmm2ResOutUb.SetSize(mm2ResCalcSize);
    PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResOutUb, RoundMode::CAST_ROUND, mm2ResCalcSize);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    int64_t dstStride = 0;
    int64_t attenOutOffset = this->valueDSize;
    int64_t datacopyOffset = this->valueDSize;
    int64_t bOffset = extraInfo.boIdx * this->n2GS1D2;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1D2;
    int64_t gOffset = extraInfo.goIdx * this->s1D2;
    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseD2;

    if constexpr ((layOutType == LayOutTypeEnum::LAYOUT_TND) || (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND)) {
        datacopyOffset = this->n2GD2;
        attenOutOffset = this->n2GD2;
        dstStride = (this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAinputParams.gSize - 1) * this->valueDSize *
                    sizeof(INPUT_T);
        bOffset = extraInfo.s1SizeAcc * this->n2GD2;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD2;
        n2Offset = extraInfo.n2oIdx * this->gD2;
        gOffset = extraInfo.goIdx * this->valueDSize;
    }
    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->valueDSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    int64_t vCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    if (likely(!this->notSplitG)) {
        if (likely(dstStride <= 65535)) {
            dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
            dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
            DataCopyPad(this->attentionOutGm[vCoreOffset +
                        (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset],
                        attenOut, dataCopyParams);
        } else {
            dataCopyParams.blockCount = 1;
            dataCopyParams.dstStride = 0;
    
            for (int32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
                DataCopyPad(this->attentionOutGm[vCoreOffset +
                            (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset +
                            i * datacopyOffset], attenOut[i * this->valueDSizeAlign16], dataCopyParams);
            }
        }
        return;
    }
    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    for (int32_t i = 0; i < extraInfo.gBaseSize; ++i) {
        if (likely(dstStride <= 65535)) {
            dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
            dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
            DataCopyPad(this->attentionOutGm[vCoreOffset +
                        (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset + i * this->valueDSize],
                        attenOut[i * dataCopyParams.blockCount * this->valueDSizeAlign16], dataCopyParams);
        } else {
            dataCopyParams.blockCount = 1;
            dataCopyParams.dstStride = 0;

            for (int32_t j = 0; j < extraInfo .vec2S1RealSize; j++) {
                DataCopyPad(this->attentionOutGm[vCoreOffset +
                            (s1oIdx * extraInfo.vec2S1BaseSize + extraInfo.vecCoreOffset) * attenOutOffset + i * this->valueDSize +
                            j * datacopyOffset], attenOut[i * dataCopyParams.blockCount * this->valueDSizeAlign16 + j * this->valueDSizeAlign16], dataCopyParams);
            }
        }
    }
}

#endif // PROMPT_FLASH_ATTENTION_S1S2_BNS1_MLA_BASEAPI_H
