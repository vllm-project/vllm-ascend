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
 * \file vf_mul_sel_softmaxflashv2_cast_nz.h
 * \brief
 */
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H

#include "kernel_tensor.h"
#include "vf_basic_block_aligned128_no_update.h"
#include "vf_basic_block_aligned128_update.h"
#include "vf_basic_block_unaligned64_no_update.h"
#include "vf_basic_block_unaligned64_update.h"
#include "vf_basic_block_unaligned128_no_update.h"
#include "vf_basic_block_unaligned128_update.h"
#include "vf_basic_block_unaligned256_no_update.h"
#include "vf_basic_block_unaligned256_update.h"
#include "vf_basic_block_unaligned512_no_update.h"
#include "vf_basic_block_unaligned512_update.h"
#include "vf_basic_block_unaligned1024_no_update.h"
#include "vf_basic_block_unaligned1024_update.h"
#include "vf_basic_block_fullquant_unaligned256_no_update.h"
#include "vf_basic_block_fullquant_unaligned256_update.h"
#include "vf_basic_block_fullquant_unaligned512_no_update.h"
#include "vf_basic_block_fullquant_unaligned512_update.h"

#include "vf_basic_block_unaligned256_no_update_mx.h"
#include "vf_basic_block_unaligned256_update_mx.h"
#include "vf_basic_block_aligned128_no_update_mx.h"
#include "vf_basic_block_aligned128_update_mx.h"
#include "vf_basic_block_unaligned64_no_update_mx.h"
#include "vf_basic_block_unaligned64_update_mx.h"
#include "vf_basic_block_unaligned128_no_update_mx.h"
#include "vf_basic_block_unaligned128_update_mx.h"
using namespace regbaseutil;

namespace FaVectorApi {
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using AscendC::LocalTensor;

enum OriginNRange {
    GT_128_AND_LTE_256 = 0, // 128 < originN <= 256, support for non-alignment (s2BaseSize=256)
    GT_64_AND_LTE_128,      // 64 < originN <= 128, support for non-alignment (s2BaseSize=128)
    EQ_128,                 // originN == 128, better performance than GT_64_AND_LTE_128 (s2BaseSize=128)
    GT_0_AND_LTE_64,        // 0 < originN <= 64 (s2BaseSize <= 64 or tail s2)
    GT_256_AND_LTE_512,     // 256 < originN <= 512 (s2BaseSize <= 512 or tail s2)
    GT_512_AND_LTE_1024,    // 512 < originN <= 1024 (s2BaseSize <= 1024 or tail s2)
    GT_0_AND_LTE_256,       // 0 < originN <= 256 (s2BaseSize <= 256 or tail s2)
    N_INVALID
};
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1NoUpdate(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f, const float pScale = 1.0f,
    float sinkValue = 0.0f)
{
    if constexpr (useNz) {
        if constexpr (oriNRange == GT_256_AND_LTE_512) {
            LocalTensor<uint8_t> indexesTensor;
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)].template Get<uint8_t>();
            ProcessVec1NoUpdateGeneralImpl512GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize,
                hasAtten, pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, expSumTensor, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        } else {
            LocalTensor<uint8_t> indexesTensor;
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)].template Get<uint8_t>();
            ProcessVec1NoUpdateGeneralImpl256GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize,
                hasAtten, pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, expSumTensor, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        }
    } else {
        if constexpr (oriNRange == GT_128_AND_LTE_256) {
            ProcessVec1NoUpdateGeneralImpl256<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop, hasSink>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb, sinkValue);
        } else if constexpr (oriNRange == EQ_128) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
            }
            ProcessVec1NoUpdateImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes,
                posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
            }
            ProcessVec1NoUpdateImpl64<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes,
                posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
            ProcessVec1NoUpdateGeneralImpl512<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb);
        } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
            ProcessVec1NoUpdateGeneralImpl1024<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb);
        } else { // GT_64_AND_LTE_128
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
            }
            ProcessVec1NoUpdateGeneralImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride,
                slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue,
                sinkValue);
        }
    }
}

template <typename T, bool useNz = false>
__simd_vf__ inline void UpdateExpSumAndExpMaxImplVF(__ubuf__ T * maxUb, __ubuf__ T * inMaxUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * expSumUb, __ubuf__ T * inExpSumUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, const uint32_t m)
{
    if constexpr (useNz) {
        RegTensor<half> vreg_max;
        RegTensor<half> vreg_in_max;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_max_even_fp32;
        RegTensor<float> vreg_exp_max_odd_fp32;
        RegTensor<half> vreg_exp_max_even;
        RegTensor<half> vreg_exp_max_odd;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_exp_max_tmp;

        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_sum_update;
        MaskReg preg_all = CreateMask<half, MaskPattern::ALL>();
        // 注意：当m大于64的时候需要开启循环
        LoadAlign(vreg_max, tmpMaxUb);
        LoadAlign(vreg_in_max, inMaxUb);
        ExpSub<float, half, RegLayout::ZERO>(vreg_exp_max_even_fp32, vreg_in_max, vreg_max, preg_all);
        ExpSub<float, half, RegLayout::ONE>(vreg_exp_max_odd_fp32, vreg_in_max, vreg_max, preg_all);
        Interleave(vreg_exp_max, vreg_exp_max_tmp, vreg_exp_max_even_fp32, vreg_exp_max_odd_fp32);
        StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)maxUb, vreg_max, preg_all);

        // x_sum = exp_max * insum + x_sum
        LoadAlign(vreg_in_exp_sum, (__ubuf__ float *&)inExpSumUb);
        LoadAlign(vreg_exp_sum_brc, (__ubuf__ float *&)tmpExpSumUb);
        Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
        Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
        StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float *&)expSumUb, vreg_exp_sum_update, preg_all);
    } else {
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_exp_sum;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_sum_update;
        MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
        // 注意：当m大于64的时候需要开启循环
        LoadAlign(vreg_max, tmpMaxUb);
        LoadAlign(vreg_in_max, inMaxUb);
        ExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)maxUb, vreg_max, preg_all);

        // x_sum = exp_max * insum + x_sum
        LoadAlign(vreg_in_exp_sum, inExpSumUb);
        LoadAlign(vreg_exp_sum_brc, tmpExpSumUb);
        Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
        Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)expSumUb, vreg_exp_sum_update, preg_all);
    }
}

template <typename T, bool useNz = false>
__aicore__ inline void UpdateExpSumAndExpMaxImpl(
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor,  const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m)
{
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();

    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ T * inExpSumUb = (__ubuf__ T*)inExpSumTensor.GetPhyAddr();

    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    uint32_t repSize = floatRepSize;
    if constexpr (useNz) {
        repSize = halfRepSize;
    }
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + repSize;
    UpdateExpSumAndExpMaxImplVF<T, useNz>(maxUb, inMaxUb, expMaxUb, expSumUb, inExpSumUb, tmpExpSumUb, tmpMaxUb, m);
}

template <typename T, bool useNz = false>
__aicore__ inline void UpdateExpSumAndExpMax(const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m)
{
    UpdateExpSumAndExpMaxImpl<T, useNz>(expSumTensor, maxTensor, expMaxTensor,
                                 inExpSumTensor, inMaxTensor, sharedTmpBuffer, m);
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1Update(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& pScaleTensor, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const float dScaleQK, const T minValue, float keepProb, const LocalTensor<T>& queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float pScale = 1.0f, float sinkValue = 0.0f)
{
    if constexpr (useNz) {
        if constexpr (oriNRange == GT_256_AND_LTE_512) {
            LocalTensor<uint8_t> indexesTensor;
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)].template Get<uint8_t>();
            ProcessVec1UpdateGeneralImpl512GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, sharedTmpBuffer, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        } else {
            LocalTensor<uint8_t> indexesTensor;
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::NZ_INDEX)].template Get<uint8_t>();
            ProcessVec1UpdateGeneralImpl256GqaFullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
                pseMode, hasDrop>(
                dstTensor, srcTensor, maxTensor, inMaxTensor, sharedTmpBuffer, indexesTensor, m, originN, scale,
                dScaleQK, minValue, pScale);
        }
    } else {
        if constexpr (oriNRange == GT_128_AND_LTE_256) {
            ProcessVec1UpdateGeneralImpl256<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, hasSink>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
                maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN, pseStride, slopes, posShift,
                scale, minValue, keepProb, sinkValue);
        } else if constexpr (oriNRange == EQ_128) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
            }
            ProcessVec1UpdateImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor, m, originN, pseStride,
                slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
            }
            ProcessVec1UpdateImpl64<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor, m, originN, pseStride,
                slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue);
        } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
            ProcessVec1UpdateGeneralImpl512<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
                pseStride, slopes, posShift, scale, minValue, keepProb);
        } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
            ProcessVec1UpdateGeneralImpl1024<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop>(
                dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
                inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, m, originN,
                pseStride, slopes, posShift, scale, minValue, keepProb);
        } else { // GT_64_AND_LTE_128
            LocalTensor<uint8_t> indexesTensor;
            if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
                indexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
            }
            ProcessVec1UpdateGeneralImpl128<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
                hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
                dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
                inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor, m, originN,
                pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue,
                sinkValue);
        }
    }
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] maskTensor, atten mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] s2BaseSize, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 * @param [in] hasAtten, indicates whether there is atten_mask
 */
template <typename T, typename T2, typename pseShiftType, bool isUpdate = false, uint32_t s1BaseSize = 128,
          uint32_t s2BaseSize = 128, OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<T>& pScaleTensor, const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f,
    const float pScale = 1.0f, float sinkValue = 0.0f)
{
    if constexpr (useNz) {
        static_assert(IsSameType<T, half>::value, "VF mul_sel_softmaxflashv2_cast_nz, T must be half");
    } else {
        static_assert(IsSameType<T, float>::value, "VF mul_sel_softmaxflashv2_cast_nz, T must be float");
    }
    static_assert(IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value || IsSameType<T2, float>::value ||
                  IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value,
                  "VF mul_sel_softmaxflashv2_cast_nz, T2 must be half, bfloat16 or float or fp8 or int8");
    static_assert(oriNRange < N_INVALID, "VF mul_sel_softmaxflashv2_cast_nz, oriNRange is invalid");

    if constexpr (!isUpdate) {
        ProcessVec1NoUpdate<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb,
            deSCaleKValue, pScale, sinkValue);
    } else {
        ProcessVec1Update<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, sharedTmpBuffer, pScaleTensor,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb,
            deSCaleKValue, pScale, sinkValue);
    }
}


template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1NoUpdateMxfp8(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    uint32_t subLoop,  const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, float keepProb, const LocalTensor<T>& queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float pScale = 1.0f, float sinkValue = 0.0f)
{
    if constexpr (oriNRange == GT_128_AND_LTE_256) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateGeneralImpl256Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
            pseMode, hasDrop, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
            maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, subLoop,
            m, originN, pseStride, slopes, posShift,
            scale, dScaleQK, minValue, keepProb, sinkValue, pScale);
        
    } else if constexpr (oriNRange == EQ_128) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateImpl128Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor,
            sharedTmpBuffer, subLoop, m, originN, pseStride, slopes, posShift, scale,
            dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue, pScale);
    } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateImpl64Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor,
            sharedTmpBuffer, subLoop, m, originN, pseStride, slopes, posShift, scale,
            dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue, pScale);
    } else { // GT_64_AND_LTE_128
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1NoUpdateGeneralImpl128Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten,
            pseMode, hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor,
            sharedTmpBuffer, subLoop, m, originN, pseStride, slopes, posShift,
            scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue,
            sinkValue, pScale);
    }
}

template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1UpdateMxfp8(
    const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<float>& preLoopMaxTensor,
    const LocalTensor<float>& preLoopSumTensor, const LocalTensor<float>& firstLoopSumTensor, uint32_t subLoop,
    const uint16_t m, const uint32_t originN, const uint32_t pseStride, const float slopes,
    const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f,
    const float pScale = 1.0f, float sinkValue = 0.0f)
{
    if constexpr (oriNRange == GT_128_AND_LTE_256) {
         LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateGeneralImpl256Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
            hasDrop, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor, inMaxTensor,
            maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor,
            preLoopSumTensor, firstLoopSumTensor, subLoop, m, originN, pseStride, slopes, posShift,
            scale, dScaleQK, minValue, keepProb, sinkValue, pScale);
       
    } else if constexpr (oriNRange == EQ_128) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
            vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateImpl128Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor,
            preLoopSumTensor, firstLoopSumTensor, subLoop, m, originN, pseStride,
            slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue, pScale);
    } else if constexpr (oriNRange == GT_0_AND_LTE_64) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
            vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateImpl64Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor,
            preLoopSumTensor, firstLoopSumTensor, subLoop, m, originN, pseStride,
            slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue, sinkValue, pScale);
    } else { // GT_64_AND_LTE_128
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value) {
            indexesTensor =
                vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1UpdateGeneralImpl128Mxfp8Fullquant<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, hasSink>(
            dstTensor, indexesTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inExpSumTensor,
            inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer, preLoopMaxTensor,
            preLoopSumTensor, firstLoopSumTensor, subLoop, m, originN,
            pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb, deSCaleKValue,
            sinkValue, pScale);
    }
}

template <typename T, typename T2, typename pseShiftType, bool isUpdate = false, uint32_t s1BaseSize = 128,
          uint32_t s2BaseSize = 128, OriginNRange oriNRange = GT_64_AND_LTE_128, bool hasAtten = 0,
          PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
          bool isMlaFullQuant = false, bool useNz = false, bool hasSink = false>
__aicore__ inline void ProcessVec1VfMxfp8(const LocalTensor<T2>& dstTensor, TBuf<> *vselrIndexesBuf,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<pseShiftType>& pseTensor,
    const LocalTensor<uint8_t>& dropTensor, const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& pScaleTensor,
    const LocalTensor<float>& preLoopMaxTensor, const LocalTensor<float>& preLoopSumTensor,
    const LocalTensor<float>& firstLoopSumTensor, uint32_t subLoop,
    const uint16_t m, const uint32_t originN, const uint32_t pseStride,
    const float slopes, const float posShift, const T scale, const float dScaleQK, const T minValue, float keepProb,
    const LocalTensor<T>& queryScaleUb = LocalTensor<T>(), const float deSCaleKValue = 1.0f,
    const float pScale = 1.0f, float sinkValue = 0.0f)
{
    static_assert(IsSameType<T, float>::value, "VF mul_sel_softmaxflashv2_cast_nz, T must be float");
    static_assert(IsSameType<T2, half>::value || IsSameType<T2, bfloat16_t>::value || IsSameType<T2, float>::value ||
                  IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                  IsSameType<T2, hifloat8_t>::value || IsSameType<T2, int8_t>::value,
                  "VF mul_sel_softmaxflashv2_cast_nz, T2 must be half, bfloat16 or float or fp8 or int8");
    static_assert(oriNRange < N_INVALID, "VF mul_sel_softmaxflashv2_cast_nz, oriNRange is invalid");

    if constexpr (!isUpdate) {
        ProcessVec1NoUpdateMxfp8<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer,
            subLoop, m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb,
            deSCaleKValue, pScale, sinkValue);
    } else {
        ProcessVec1UpdateMxfp8<T, T2, pseShiftType, s1BaseSize, s2BaseSize, oriNRange, hasAtten, pseMode,
            hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>(
            dstTensor, vselrIndexesBuf, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            inExpSumTensor, inMaxTensor, maskTensor, pseTensor, dropTensor, pScaleSubLoop0Tensor, sharedTmpBuffer,
            preLoopMaxTensor, preLoopSumTensor, firstLoopSumTensor, subLoop,
            m, originN, pseStride, slopes, posShift, scale, dScaleQK, minValue, keepProb, queryScaleUb,
            deSCaleKValue, pScale, sinkValue);
    }
}

template <typename T>
__simd_vf__ inline void SoftmaxSumUpdateVF(__ubuf__ T * sumUb, __ubuf__ T * maxUb, const uint32_t m,
    const T minValue, const T maxValue)
{
    RegTensor<float> vreg_max_value;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_sum;
    RegTensor<float> vreg_sum_new;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_compare;

    Duplicate(vreg_max_value, maxValue);
    // 注意：当m大于64的时候需要开启循环
    LoadAlign(vreg_max, maxUb);
    LoadAlign(vreg_sum, sumUb);
    Compares<T, CMPMODE::EQ>(preg_compare, vreg_max, minValue, preg_all);
    Select(vreg_sum_new, vreg_max_value, vreg_sum, preg_compare);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)sumUb, vreg_sum_new, preg_all);
}

template <typename T>
__aicore__ inline void SoftmaxSumUpdate(const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor,
    const uint32_t m, const T minValue, const T maxValue)
{
    __ubuf__ T * sumUb = (__ubuf__ T*)sumTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();

    SoftmaxSumUpdateVF<T>(sumUb, maxUb, m, minValue, maxValue);
}
} // namespace

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_INTERFACE_H
