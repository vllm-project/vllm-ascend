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
* \file prompt_flash_attention_s1s2_bns1_mla.h
* \brief
*/

#ifndef PROMPT_FLASH_ATTENTION_S1S2_BNS1_MLA_H
#define PROMPT_FLASH_ATTENTION_S1S2_BNS1_MLA_H

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

using matmul::MatmulType;

constexpr int64_t GM_DOUBLE_BUFFER = 2;
constexpr int64_t INVALID_OFFSET = INT64_MIN;
constexpr int64_t SPLIT_S2_SIZE_LIMIT = 1024;
constexpr AscendC::SoftmaxConfig PFA_SOFTMAX_DEFAULT_CFG = {false};
constexpr AscendC::SoftmaxConfig SOFTMAX_REDUCE_CFG = {false, 0, 0, AscendC::SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class MmPolicyType {
    NORMAL = 0,
    UNSPLITK = 1,
    PA_ND = 2,
    PA_NZ = 3
};

constexpr int64_t VEC_BUFFER_SIZE = 1024;

// L1 extension
template<MmPolicyType mmPolicyType>
struct Matmul1PolicySelector {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Matmul1PolicySelector<MmPolicyType::UNSPLITK> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MlaMatmulPolicyMM1HeadDim192<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<MmPolicyType mmPolicyType>
struct Matmul2PolicySelector {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

template<>
struct Matmul2PolicySelector<MmPolicyType::UNSPLITK> {
    template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
    struct Result : AscendC::Impl::Detail::MlaMatmulPolicyMM2HeadDim128<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>{};
};

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T = INPUT_T, CubeFormat bmm1Format = CubeFormat::ND,
        MmPolicyType mmPolicyType = MmPolicyType::NORMAL>
class MlaS1s2Bn2gs1SameAB {
public:
    __aicore__ inline MlaS1s2Bn2gs1SameAB(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const TILING_TYPE *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true,  LayoutMode::NONE, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED,
                matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                Matmul1PolicySelector<mmPolicyType>::template Result> bmm1;

    using c1NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    matmul::Matmul<a1Type, b1Type, c1NzType, bias1Type, CFG_EXCEED,
                matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                Matmul1PolicySelector<mmPolicyType>::template Result> bmm1Nz;

    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::NONE, true>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                Matmul2PolicySelector<mmPolicyType>::template Result> bmm2;

protected:
    static __aicore__ inline constexpr bool InputLayoutIsTNDLike()
    {
        return layOutType == LayOutTypeEnum::LAYOUT_TND || layOutType == LayOutTypeEnum::LAYOUT_NTD_TND;
    }

    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                    __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut,
                                    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                    const TILING_TYPE *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void WaitBmm2Result(SplitSameABExtraInfo &extraInfo);
    __aicore__ inline void IterateBmm2(SplitSameABExtraInfo &extraInfo,
                                    matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                                    matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                    Matmul2PolicySelector<mmPolicyType>::template Result> &bmm2);
    __aicore__ inline void NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
                                LocalTensor<INPUT_T> &vec1ResUb, int64_t loopIdx);
    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void SetTiling(const TILING_TYPE *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    template <typename T2>
    __aicore__ inline void IterateBmm1(SplitSameABExtraInfo &extraInfo,
                                    matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                    matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                    Matmul1PolicySelector<mmPolicyType>::template Result> &bmm1);
    template <typename T2>
    __aicore__ inline void Bmm1SetTensorA(SplitSameABExtraInfo &extraInfo,
                                        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                        matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                        Matmul1PolicySelector<mmPolicyType>::template Result> &bmm1);
    template <typename T2>
    __aicore__ inline void SetBmm1TensorB(SplitSameABExtraInfo &extraInfo,
                                        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                        matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                        Matmul1PolicySelector<mmPolicyType>::template Result> &bmm1);
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
    __aicore__ inline void Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx);
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
    TBuf<> softmaxMaxBuf;
    TBuf<> softmaxLseBuf;
    TBuf<> commonTBuf; // common的复用空间
    TBuf<> softmaxTempBuf;

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1Res[2];
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2Res[2];
    GlobalTensor<INPUT_T> stage1Res[2];
    GlobalTensor<half> pseAlibiGm;

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

    // MLA D不等长,V和OUT为D2
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
    int64_t s2BaseNratioSize;
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
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    int64_t attenMaskOffsetPre = 0;
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                                __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut,
                                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                                const TILING_TYPE *__restrict tiling,
                                                TPipe *tPipe)
{
    this->InitInput(query, key, value, attenMask, attentionOut, softmaxLse, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                        __gm__ uint8_t *value,
                                                        __gm__ uint8_t *attenMask,
                                                        __gm__ uint8_t *attentionOut,
                                                        __gm__ uint8_t *softmaxLse,
                                                        __gm__ uint8_t *workspace,
                                                        const TILING_TYPE *__restrict tiling,
                                                        TPipe *tPipe)
{
    this->vecBlockIdx = GetBlockIdx();
    this->cubeBlockIdx = this->vecBlockIdx / 2;
    this->cubeSubIdx = this->vecBlockIdx % 2;
    this->pipe = tPipe;
    this->SetTiling(tiling);

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
    
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
    if (this->tilingData->PFAinputParams.isSoftMaxLseEnable) {
        this->softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
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
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                            bmm1Format, mmPolicyType>::SetTiling(const TILING_TYPE *__restrict tilingData)
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
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                            bmm1Format, mmPolicyType>::InitBuffer()
{
    uint64_t stage1Size = 8 * 1024;
    uint64_t stage1AttenSize = 9 * 1024;
    uint64_t stage1PongSize = 34 * 1024;
    uint64_t stage2Size = 64 * 128;
    uint64_t maskTBufPongSize = 16 * 1024;

    // 可选输入的buffer空间，保持和stage1处理的size一致
    this->pipe->InitBuffer(this->maskTBufPing, stage1AttenSize); // 可以给attenmask 9k
    this->pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize); // 可以给dropoutmask 16k
    this->pipe->InitBuffer(this->pseTBuf, 16384); // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
    this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumBuf[0], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    this->pipe->InitBuffer(this->softmaxSumBuf[1], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    this->pipe->InitBuffer(this->softmaxMaxBuf, vecS1BaseSize * 4 * this->softmaxReduceSize);     // 2k
    if (this->tilingData->PFAinputParams.isSoftMaxLseEnable) {
        this->pipe->InitBuffer(this->softmaxLseBuf, vecS1BaseSize * 4 * this->softmaxReduceSize); // 2k
    }
    this->pipe->InitBuffer(this->softmaxExpBuf[0], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    this->pipe->InitBuffer(this->softmaxExpBuf[1], vecS1BaseSize * 4 * this->softmaxReduceSize);  // 2k
    if (this->softmaxReduceSize == 1) {
        this->pipe->InitBuffer(this->softmaxTempBuf, vecS1BaseSize * blockBytes);    // 16k
    }
    this->pipe->InitBuffer(this->stage1PongBuf, stage1PongSize); // i.a 34k
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType>::ComputeConstexpr()
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
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->PFAcoreParams.nRatio;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
        this->s1BaseN2GD2 = this->cubeS1BaseSize * this->n2GD2;
        this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->PFAcoreParams.nRatio;
        this->mm1Ka = this->n2GD;
        this->mm1Kb = this->n2D;
        this->mm2Kb = this->n2D2;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        this->bN2G = this->tilingData->PFAinputParams.bSize * this->n2G;
        this->s1BaseBN2GD = this->cubeS1BaseSize * this->tilingData->PFAinputParams.bSize * this->n2GD;
        this->s1BaseBN2GD2 = this->cubeS1BaseSize * this->tilingData->PFAinputParams.bSize * this->n2GD2;
        this->s2BaseBN2D = this->tilingData->PFAinputParams.bSize * this->s2BaseN2D;
        this->s2BaseNratioBN2D = this->s2BaseBN2D * this->tilingData->PFAcoreParams.nRatio;
        this->mm1Ka = this->bN2GD;
        this->mm1Kb = this->bN2D;
        this->mm2Kb = this->bN2D2;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        this->s1BaseD = this->cubeS1BaseSize * this->dSize;
        this->s1BaseD2 = this->cubeS1BaseSize * this->valueDSize;
        this->s2BaseNratioD = this->s2BaseNratioSize * this->dSize;
        this->mm1Ka = this->dSize;
        this->mm1Kb = this->dSize;
        this->mm2Kb = this->valueDSize;
    }
    if (this->cubeS1BaseSize > 256) {
        this->softmaxReduceSize = 1;
    }
}

// sparse functions
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::GetS1LoopRange(int64_t &multiCoreInnerOffset,
                                                            int64_t &multiCoreInnerLimit)
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::GetS2LoopRange()
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->PFAmultiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->PFAmultiCoreParams.splitFactorSize;
    if (this->tilingData->PFAmultiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->PFAmultiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);

    SplitSameABExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    bool notSecondLast = true;
    bool notLast = true;
    multiCoreInnerLimit += 2;
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
            ++multiCoreInnerIdx) {
        if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
            notLast = false;
        }

        int64_t s2LoopLimit;
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        bool notLastTwoLoop = notSecondLast && notLast;
        if (notLastTwoLoop) {
            this->ComputeAxisIdx(multiCoreInnerIdx);

            // s2轴循环计数, 支持sparse和非sparse场景
            this->GetS2LoopRange();
            s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, s2BaseNratioSize) - 1;
        } else {
            s2LoopLimit = 0;
        }
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
            if (taskId >= 1 && notLast) {
                WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
            }

            if (notLastTwoLoop) {
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                if constexpr (bmm1Format == CubeFormat::NZ) {
                    if (extraInfo[taskId % 3].needNz2Nd == 1) {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
                    } else {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
                    }
                } else {
                    this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
                }
            }

            if (taskId > 0 && notLast) {
                this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            if (taskId > 1) {
                WaitBmm2Result(extraInfo[(taskId + 1) % 3]);
            }

            if (taskId > 0 && notLast) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::WaitBmm1Result(SplitSameABExtraInfo &extraInfo)
{
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if (extraInfo.needNz2Nd == 1) {
            this->bmm1Nz.WaitIterateAll();
            this->bmm1Nz.End();
            return;
        }
    }
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId,
                                                        int64_t s2LoopCount, int64_t s2LoopLimit,
                                                        int64_t multiCoreInnerIdx)
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s2RealSize = this->s2BaseNratioSize;
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
template <typename T2>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::IterateBmm1(SplitSameABExtraInfo &extraInfo, 
                                                        matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                                        matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                                        Matmul1PolicySelector<mmPolicyType>::template Result> &bmm1)
{
    if constexpr (mmPolicyType == MmPolicyType::UNSPLITK) {
        bmm1.SetSelfDefineData(AscendC::MM1_INDEX);
    }
    bmm1.SetOrgShape(extraInfo.cubeS1RealSize, this->mm1Kb, this->mm1Ka, this->mm1Kb, extraInfo.s2RealSize);

    this->Bmm1SetTensorA(extraInfo, bmm1);
    this->SetBmm1TensorB(extraInfo, bmm1);
    bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2], false, false, true);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
template <typename T2>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::Bmm1SetTensorA(SplitSameABExtraInfo &extraInfo,
                                                            matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                                            matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                                            Matmul1PolicySelector<mmPolicyType>::template Result> &bmm1)
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
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD;
        bOffset = extraInfo.boIdx * this->n2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // bnsd
        bOffset = extraInfo.boIdx * this->n2GS1D;
        n2Offset = extraInfo.n2oIdx * this->gS1D;
        gOffset = extraInfo.goIdx * this->s1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseD;
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
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = this->qCoreOffset;
    bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
template <typename T2>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::SetBmm1TensorB(SplitSameABExtraInfo &extraInfo,
                                                            matmul::Matmul<a1Type, b1Type, T2, bias1Type, CFG_EXCEED,
                                                            matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                                            Matmul1PolicySelector<mmPolicyType>::template Result> &bmm1)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D + extraInfo.s2LoopCount * this->s2BaseNratioBN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // same as "BSND"
        bOffset = extraInfo.s2SizeAcc * this->n2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
        n2Offset = extraInfo.n2oIdx * this->dSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND) {
        // same as "NBSD"
        n2Offset = extraInfo.n2oIdx * this->s2D;
        bOffset = extraInfo.s2SizeAcc * this->dSize;
        s2Offset = extraInfo.s2StartIdx * this->dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    }
    int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
    bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    bmm1.SetTail(extraInfo.cubeS1RealSize, extraInfo.s2RealSize);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::NdToNz(SplitSameABExtraInfo &extraInfo, LocalTensor<INPUT_T> &nzResUb,
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::ProcessVec1(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }
    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    LocalTensor<T> &actualUseTensor = (extraInfo.needNz2Nd == 0) ? stage1PongTensor : stage1PingTensor;
    LocalTensor<T> commonTBuf = this->commonTBuf.template Get<T>(); // t.b 32k

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2C = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdDropMte3ToMte2;
    extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
    for (int32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; ++loopIdx) {
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
        }
        if (loopIdx > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        } else {
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }

        // FP32场景，需要等待vec1上一轮输出搬完
        if constexpr (IsSameType<INPUT_T, float>::value) {
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        this->GetBmm1Result(extraInfo, actualUseTensor, loopIdx);

        // mul或attenMask的计算依赖bmm1计算结果
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        this->CopyInAttenMask(extraInfo, loopIdx, -1);
        PipeBarrier<PIPE_V>();
        Muls(stage1PingTensor, actualUseTensor, static_cast<T>(this->tilingData->PFAinputParams.scaleValue),
            extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
        if constexpr (hasAtten) {
            SelectWithBytesMaskShapeInfo shapeInfo;
            shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
            shapeInfo.srcLastAxis = extraInfo.s2RealSizeAlign64;
            shapeInfo.maskLastAxis = extraInfo.s2RealSizeAlign64;
            stage1PingTensor.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
            if (this->attenMaskComputeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE &&
                this->attenMaskComputeMode != AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                uint8_t maskType = (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) ? 1 : 0;
                LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
                this->ComputeAttenMask(shapeInfo, stage1PingTensor, attenMaskUb, maskType, eventIdMte2ToV);
            }

            if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE ||
                this->attenMaskComputeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2C);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2C);
                this->CopyInAttenMask(extraInfo, loopIdx, this->attenMaskOffsetPre, true);
                LocalTensor<uint8_t> secondTimeMaskUb;
                uint8_t maskType;
                secondTimeMaskUb = this->pseTBuf.template Get<uint8_t>();
                maskType = 1;
                this->ComputeAttenMask(shapeInfo, stage1PingTensor, secondTimeMaskUb, maskType, eventIdMte2ToV);
            }
        }
        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }

        if (loopIdx > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }
        this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);

        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2A);
        }

        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (!IsSameType<T, INPUT_T>::value) {
            LocalTensor<INPUT_T> stage1CastTensor;
            stage1CastTensor = this->pseTBuf.template Get<INPUT_T>();
            Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                extraInfo.vec1S1RealSize * extraInfo.s2RealSizeAlign64);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            LocalTensor<INPUT_T> stage1NzTensor = this->stage2TBuf.template Get<INPUT_T>();
            NdToNz(extraInfo, stage1NzTensor, stage1CastTensor, loopIdx);
        } else {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
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
                stage1PingTensor, dataCopyParams);
        }

        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2C);
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::CopyInAttenMask(SplitSameABExtraInfo &extraInfo, int64_t loopIdx,
                                                            int64_t maskOffset, bool secondTime)
{
    if constexpr (hasAtten == true) {
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
        BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
                s2StrideSize, 64);
        return;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline int64_t MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType>::ComputeAttenMaskOffset(SplitSameABExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten == true) {
        if (this->tilingData->PFAinputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
        }
        if constexpr (InputLayoutIsTNDLike()) {
            int64_t delta = 0;
            int64_t deltaPre = 0;
            int64_t deltaN = static_cast<int64_t>((extraInfo.s1Size)) - static_cast<int64_t>((extraInfo.s2Size));
            int64_t s1Offset = extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                            loopIdx * extraInfo.vec1S1BaseSize;
            int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
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
                    ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
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
            return ComputeOffsetForCausal(delta, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                        this->tilingData->PFAinputParams.attenMaskS2Size);
        }
        // compress mode
        int64_t deltaCausalOrNext = 0;
        int64_t deltaPre = 0;
        int64_t deltaN = extraInfo.s1Size - extraInfo.s2Size;
        int64_t s1Offset = extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                        loopIdx * extraInfo.vec1S1BaseSize;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
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
                ComputeOffsetForCausal(deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                    this->tilingData->PFAinputParams.attenMaskS2Size);
            deltaCausalOrNext = s1Offset - s2Offset + this->tilingData->PFAinputParams.nextTokens;
        } else {
            return 0;
        }
        this->GetAttenMaskComputeMode(deltaCausalOrNext, deltaPre, s1Offset, extraInfo);
        return ComputeOffsetForCausal(deltaCausalOrNext, extraInfo.vec1S1BaseSize, s2BaseNratioSize,
                                    this->tilingData->PFAinputParams.attenMaskS2Size);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre,
                                                                    int64_t s1Offset, SplitSameABExtraInfo &extraInfo)
{
    if constexpr (hasAtten == true) {
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline int64_t MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
    bmm1Format, mmPolicyType>::ComputeOffsetForNoCompress(SplitSameABExtraInfo &extraInfo, int64_t loopIdx)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;
        int64_t s1Offset = (extraInfo.s1oIdx * this->cubeS1BaseSize + extraInfo.vecCoreOffset +
                            loopIdx * extraInfo.vec1S1BaseSize) * extraInfo.s2Size;
        int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                    bmm1Format, mmPolicyType>::GetBmm1Result(SplitSameABExtraInfo &extraInfo,
                                            LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
{
    if constexpr (bmm1Format == CubeFormat::NZ) {
        if (extraInfo.needNz2Nd == 1) {
            Nz2NdInfo nz2NdInfo;
            nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
            nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec1S1BaseSize;
            nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec1S1RealSize;
            nz2NdInfo.ndLastAxis = extraInfo.s2RealSizeAlign64;
            nz2NdInfo.loopIdx = loopIdx;
            nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
            LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
            NzToNd(nz2NdInfo, this->mm1Res[extraInfo.taskIdMod2], tempUb, bmm1ResUb);
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::ComputeAttenMask(SelectWithBytesMaskShapeInfo &shapeInfo,
                                                            LocalTensor<T> &bmm1ResUb,
                                                            LocalTensor<uint8_t> &attenMaskUb,
                                                            const uint8_t maskType, event_t vWaitMte2)
{
    if constexpr (hasAtten == true) {
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                            bmm1Format, mmPolicyType>::SoftMaxCheckResCompress(SplitSameABExtraInfo &extraInfo,
                                                                int64_t vec1S1realSplitN)
{
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::InvalidLineSplitS2Process(SplitSameABExtraInfo &extraInfo,
                                                                    LocalTensor<T> &srcTensor,
                                                                    LocalTensor<T> &maxUb, int64_t loopIdx)
{
    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
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
            LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                            bmm1Format, mmPolicyType>::SoftMaxCompute(SplitSameABExtraInfo &extraInfo,
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
    LocalTensor<T> maxUb = this->softmaxMaxBuf
                        .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->softmaxReduceSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    if (extraInfo.s2LastLoop) {
        if constexpr (hasAtten == false) {
            uint64_t mask[2] = {extraInfo.duplicateMask, 0};
            Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vec1S1RealSize,
                        1, extraInfo.s2RealSizeAlign64 * sizeof(T) / blockBytes);
        } else if (this->tilingData->PFAinputParams.attenMaskCompressMode !=
                static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE) || extraInfo.s2RealSize < 64) {
            uint64_t mask[2] = {extraInfo.duplicateMask, 0};
            Duplicate<T>(srcTensor[extraInfo.s2RealSizeFloorAlign8], this->negativeFloatScalar, mask, vec1S1RealSize,
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

    if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION) {
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
            if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
                LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
                LocalTensor<T> sumTensor;
                sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();

                SoftMaxShapeInfo softmaxFullShapeInfo{
                    static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize),
                    static_cast<uint32_t>(extraInfo.s1RealSize), static_cast<uint32_t>(fp32BaseSize)};
                if (this->softmaxReduceSize == 1) {
                    AdjustSoftMaxRes<T, T, false, 1>(sumTensor, maxTensor, this->negativeIntScalar,
                                                    this->positiveFloatScalar, softmaxFullShapeInfo);
                } else {
                    AdjustSoftMaxRes<T, T>(sumTensor, maxTensor, this->negativeIntScalar,
                                        this->positiveFloatScalar, softmaxFullShapeInfo);
                }
            }
        }
    }
    if (this->tilingData->PFAinputParams.isSoftMaxLseEnable &&
        loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        extraInfo.softmaxLseOffset =
            (extraInfo.s1SizeAcc * this->n2G +
             extraInfo.n2oIdx * this->tilingData->PFAinputParams.gSize +
             extraInfo.goIdx + extraInfo.s1oIdx * this->cubeS1BaseSize *
             this->n2G + extraInfo.vecCoreOffset * this->n2G) * static_cast<int64_t>(1);
        LocalTensor<T> lseTensor = this->softmaxLseBuf.template Get<T>();
        LocalTensor<T> maxTensor = this->softmaxMaxBuf.template Get<T>();
        LocalTensor<T> sumTensor = this->softmaxSumBuf[extraInfo.multiCoreInnerIdxMod2].template Get<T>();
        PipeBarrier<PIPE_V>();
        Log(lseTensor, sumTensor, extraInfo.s1RealSize);
        PipeBarrier<PIPE_V>();
        Add(lseTensor, lseTensor, maxTensor, extraInfo.s1RealSize);
        LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
        PipeBarrier<PIPE_V>();
        Brcb(softmaxTemp, lseTensor, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr() -> FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = extraInfo.s1RealSize;
        dataCopyParams.blockLen = sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (this->n2G - 1) * sizeof(T);
        DataCopyPad(this->softmaxLseGm[extraInfo.softmaxLseOffset], softmaxTemp, dataCopyParams);
    }
}


template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                    bmm1Format, mmPolicyType>::WaitBmm2Result(SplitSameABExtraInfo &extraInfo)
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                        bmm1Format, mmPolicyType>::IterateBmm2(SplitSameABExtraInfo &extraInfo,
                                            matmul::Matmul<a2Type, b2Type, c2NzType, bias2Type, CFG_EXCEED,
                                            matmul::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                                            Matmul2PolicySelector<mmPolicyType>::template Result> &bmm2)
{
    if constexpr (mmPolicyType == MmPolicyType::UNSPLITK) {
        bmm2.SetSelfDefineData(AscendC::MM2_INDEX);
    }
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D2;
        s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * s2BaseNratioSize * this->n2D2;
        n2Offset = extraInfo.n2oIdx * valueDSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D2 + extraInfo.s2LoopCount * s2BaseNratioSize * this->bN2D2;
        bOffset = extraInfo.boIdx * this->n2D2;
        n2Offset = extraInfo.n2oIdx * valueDSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BNSD) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D2;
        n2Offset = extraInfo.n2oIdx * this->s2D2;
        s2Offset = extraInfo.s2StartIdx * valueDSize + extraInfo.s2LoopCount * s2BaseNratioSize * valueDSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // TND
        bOffset = extraInfo.s2SizeAcc * this->n2D2;
        s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2D2;
        n2Offset = extraInfo.n2oIdx * this->valueDSize;
    } else if constexpr (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND) {
        // NTD or NTD_TND
        n2Offset = extraInfo.n2oIdx * this->s2D2;
        bOffset = extraInfo.s2SizeAcc * valueDSize;
        s2Offset = extraInfo.s2StartIdx * valueDSize + extraInfo.s2LoopCount * this->s2BaseNratioSize * valueDSize;        
    }

    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    bmm2.SetOrgShape(extraInfo.cubeS1RealSize, this->mm2Kb, extraInfo.s2AlignedSize, this->mm2Kb, this->valueDSize);

    bmm2.SetTensorA(this->stage1Res[extraInfo.taskIdMod2]);

    bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    bmm2.SetTail(extraInfo.cubeS1RealSize, this->valueDSize, extraInfo.s2RealSize);

    bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskIdMod2], false, false, true);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::ProcessVec2(SplitSameABExtraInfo &extraInfo)
{
    if (extraInfo.s1RealSize == 0) {
        return;
    }
    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
    int64_t vec2LoopLimit = CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;
    for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; ++s1oIdx) {
        if (s1oIdx == vec2LoopLimit - 1) {
            extraInfo.vec2S1RealSize = extraInfo.s1RealSize - s1oIdx * extraInfo.vec2S1BaseSize;
        }
        int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * this->valueDSizeAlign16;
        int64_t mm2ResOffset = (extraInfo.vecCoreOffset + s1oIdx * extraInfo.vec2S1BaseSize) * this->valueDSizeAlign16;
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        Nz2NdInfo nz2NdInfo;
        nz2NdInfo.ndFirstAxisRealSize = extraInfo.cubeS1RealSize;
        nz2NdInfo.ndFirstAxisBaseSize = extraInfo.vec2S1BaseSize;
        nz2NdInfo.ndFirstAxisLoopSize = extraInfo.vec2S1RealSize;
        nz2NdInfo.ndLastAxis = this->valueDSizeAlign16;
        nz2NdInfo.loopIdx = s1oIdx;
        nz2NdInfo.vecCoreOffset = extraInfo.vecCoreOffset;
        event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        LocalTensor<T> tempUb = this->stage1PongBuf.template Get<T>();
        NzToNd(nz2NdInfo, this->mm2Res[extraInfo.taskIdMod2], tempUb, stage2BufTensor);
        PipeBarrier<PIPE_V>();

        if (likely(extraInfo.s2LoopCount != 0)) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            DataCopy(bmm2ResUb, this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], mm2ResCalcSize);
        }

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        if (unlikely(extraInfo.s2LoopCount == 0)) {
            DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        } else {
            this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
            PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        }

        if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            Bmm2ResultDiv(extraInfo, s1oIdx);
            Bmm2DataCopyOut(extraInfo, s1oIdx, mm2ResCalcSize);
            event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        } else {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
        }
    }
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::Bmm2ResultMul(SplitSameABExtraInfo &extraInfo,
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
    if (this->softmaxReduceSize == 1) {
        LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
        Brcb(softmaxTemp, expUb, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; ++i) {
            Mul(bmm2ResUb[i * repeatMaxSize], softmaxTemp[softmaxTempOffset], bmm2ResUb[i * repeatMaxSize],
                repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (remain) {
            Mul(bmm2ResUb[loop * repeatMaxSize], softmaxTemp[softmaxTempOffset],
                bmm2ResUb[loop * repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
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
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::Bmm2ResultDiv(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

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
        LocalTensor<T> softmaxTemp = this->softmaxTempBuf.template Get<T>();
        Brcb(softmaxTemp, sumUb, (extraInfo.s1RealSize + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; ++i) {
            Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
                softmaxTemp[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
                softmaxTemp[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
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
                Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize], sumCastTensor, repeatMaxSize,
                    extraInfo.vec2S1RealSize, repeatParams);
            }
            if (likely(remain)) {
                Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize], sumCastTensor, remain,
                    extraInfo.vec2S1RealSize, repeatParams);
            }
        } else {
            for (int i = 0; i < loop; i++) {
                Div(bmm2ResUb[i * repeatMaxSize], bmm2ResUb[i * repeatMaxSize],
                    sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
            }
            if (likely(remain)) {
                Div(bmm2ResUb[loop * repeatMaxSize], bmm2ResUb[loop * repeatMaxSize],
                    sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
            }
        }
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T,
        typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                bmm1Format, mmPolicyType>::Bmm2DataCopyOut(SplitSameABExtraInfo &extraInfo, int64_t s1oIdx,
                                                            int64_t mm2ResCalcSize)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<INPUT_T> attenOut = this->stage2TBuf.template Get<INPUT_T>();
    bmm2ResUb.SetSize(mm2ResCalcSize);
    PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, mm2ResCalcSize);
    } else {
        DataCopy(attenOut, bmm2ResUb, mm2ResCalcSize);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->valueDSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    int64_t dstStride = 0;
    int64_t attenOutOffset = this->valueDSize;
    int64_t datacopyOffset = this->valueDSize;
    int64_t bOffset = extraInfo.boIdx * this->n2GS1D2;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1D2;
    int64_t gOffset = extraInfo.goIdx * this->s1D2;
    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseD2;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_BSH) {
        datacopyOffset = this->n2GD2;
        attenOutOffset = this->n2GD2;
        dstStride = (this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAinputParams.gSize - 1) * this->valueDSize *
                    sizeof(INPUT_T);
        bOffset = extraInfo.boIdx * this->n2GS1D2;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD2;
        n2Offset = extraInfo.n2oIdx * this->gD2;
        gOffset = extraInfo.goIdx * this->valueDSize;
    } else if constexpr ((layOutType == LayOutTypeEnum::LAYOUT_TND) || (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND)) {
        datacopyOffset = this->n2GD2;
        attenOutOffset = this->n2GD2;
        dstStride = (this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAinputParams.gSize - 1) * this->valueDSize *
                    sizeof(INPUT_T);
        bOffset = extraInfo.s1SizeAcc * this->n2GD2;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD2;
        n2Offset = extraInfo.n2oIdx * this->gD2;
        gOffset = extraInfo.goIdx * this->valueDSize;
    }
    int64_t vCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
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
}

#endif // PROMPT_FLASH_ATTENTION_S1S2_BNS1_MLA_H
