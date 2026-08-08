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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_dn.h
 * \brief
 */

#ifndef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#include "kernel_tensor.h"
namespace FaVectorApi {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnNoUpdateVF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint32_t *maskUb, __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, const T minValue, float keepProb,
    bool needAtten, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);


    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);

    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }
    
    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);    
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);    
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__aicore__ inline void ProcessVec1DnNoUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, const T minValue, float keepProb, bool needAtten, const float pScale = 1.0f,
    const float sinkValue = 0.0f)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnNoUpdateVF<T, T2, hasAtten, ubN, hasSink>(x_exp, input_x_local_UB, exp_max_fp32,
        new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN, scale, deScaleQK, minValue,
        keepProb, needAtten, dScale, blockStride, repeatStride, pScale, sinkValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__simd_vf__ inline void ProcessVec1DnUpdateVF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint32_t *maskUb, __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, const T minValue, float keepProb,
    bool needAtten, const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);     
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136); 
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136); 
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136); 
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool hasSink = false>
__aicore__ inline void ProcessVec1DnUpdate(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, const T minValue, float keepProb, bool needAtten, const float pScale = 1.0f,
    const float sinkValue = 0.0f)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnUpdateVF<T, T2, hasAtten, ubN, hasSink>(x_exp, input_x_local_UB, exp_max_fp32,
        new_global_sum, new_global_max, maskUb, indexesUb, m, n, originN, scale, deScaleQK, minValue,
        keepProb, needAtten, dScale, blockStride, repeatStride, pScale, sinkValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdateMxfp8VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint32_t *maskUb, __ubuf__ uint8_t *indexesUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
    float pScale, const T minValue, float keepProb, const float dScale, const uint32_t blockStride,
    const uint32_t repeatStride, int32_t subLoop)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;

    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3, preg_compare_all;
    preg_compare_all = CreateMask<T, MaskPattern::ALLF>();
    MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);

    if (subLoop == 1) {
        LoadAlign(vreg_x_max_f32_b, new_global_max);
        Max(max0, max0, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8; // ubN：s2BaseSize = 256
    } else {
        loopNum = ubN / 4;
    }

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134); // vreg_x_exp_i softmax分子
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }

    if (subLoop == 0) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_134);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(
            ((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0), vreg_p_scale_f8e8m0, preg_134);
    } else {
        RegTensor<float> first_loop_sum;
        LoadAlign(first_loop_sum, new_global_sum);
        Mul(first_loop_sum, vreg_x_max_f32_b, first_loop_sum, preg_134);
        Add(vreg_x_sum0, first_loop_sum, vreg_x_sum0, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_x_max_f32_b, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_x_max_f32_b, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
        Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
                (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_134);
        DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1,
            vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_0);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(
            ((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0), vreg_p_scale_f8e8m0_dst0, preg_134);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnNoUpdateMxfp8(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, float pScale,
    const T minValue, float keepProb, int32_t subLoop,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnNoUpdateMxfp8VF<T, T2, hasAtten, ubN>(x_exp, input_x_local_UB, exp_max_fp32,
        new_global_sum, new_global_max, maskUb, indexesUb, pScaleSubLoop0Ub, m, n, originN,
        scale, deScaleQK, pScale, minValue, keepProb, dScale, blockStride, repeatStride, subLoop);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdateMxfp8VF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint32_t *maskUb, __ubuf__ uint8_t *indexesUb, __ubuf__ fp8_e8m0_t *pScaleSubLoop0,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
    float pScale, const T minValue, float keepProb, const float dScale, const uint32_t blockStride,
    const uint32_t repeatStride, __ubuf__ float *pre_loop_max, __ubuf__ float *pre_loop_sum,
    __ubuf__ float *first_loop_sum, int32_t subLoop)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<float> vreg_subloop_update;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3, preg_compare_all;
    preg_compare_all = CreateMask<T, MaskPattern::ALLF>();
    MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    Duplicate(vreg_p_scale, static_cast<float>(pScale));
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
            LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
            Select(src0, src0, vreg_min, preg_compare0);
            Select(src1, src1, vreg_min, preg_compare1);
            Select(src2, src2, vreg_min, preg_compare2);
            Select(src3, src3, vreg_min, preg_compare3);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    Muls(max0, max0, dScale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);
    Muls(max0, max0, INV_LN2, preg_108);
    Truncate<T, RoundMode::CAST_CEIL>(max0, max0, preg_108);
    Muls(max0, max0, LN2, preg_108);

    if (subLoop == 0) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)pre_loop_max, vreg_x_max_f32_b, preg_108);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    } else {
        FusedExpSub(vreg_subloop_update, vreg_x_max_f32_b, max0, preg_134);
        LoadAlign(vreg_x_max_f32_b, pre_loop_max);
        FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    }
    
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)new_global_max, max0, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
        Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
        Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
        Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
            Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
            Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
            Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136); 
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    
    RegTensor<float> vreg_l0;
    if (subLoop == 0) {
        LoadAlign(vreg_l0, new_global_sum);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)pre_loop_sum, vreg_l0, preg_134);
        Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
        Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)first_loop_sum, vreg_x_sum0, preg_134);
        // pscale update
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;
        Duplicate(vreg_p_scale_f8e8m0, 0x7f, preg_134);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(
            ((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0), vreg_p_scale_f8e8m0, preg_134);
    } else {
        RegTensor<float> vreg_l1;
        LoadAlign(vreg_l0, first_loop_sum);
        LoadAlign(vreg_l1, pre_loop_sum);
        Mul(vreg_l0, vreg_subloop_update, vreg_l0, preg_134);
        Add(vreg_x_sum0, vreg_l0, vreg_x_sum0, preg_134);
        Mul(vreg_l1, vreg_x_max_f32_b, vreg_l1, preg_134);
        Add(vreg_l0, vreg_x_sum0, vreg_l1, preg_134);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
        // pscale update
        RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
        RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
        RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
        Cast<bfloat16_t, T, castTraitRintZero>(vreg_p_scale_bf16_0, vreg_subloop_update, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_0, vreg_p_scale_bf16_0, preg_108);
        Cast<bfloat16_t, T, castTraitRintOne>(vreg_p_scale_bf16_1, vreg_subloop_update, preg_135);
        Cast<fp8_e8m0_t, bfloat16_t, castTraitNoneZero>(vreg_p_scale_f8e8m0_1, vreg_p_scale_bf16_1, preg_108);
        Or((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
                (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_134);
        DeInterleave(vreg_p_scale_f8e8m0_dst0, vreg_p_scale_f8e8m0_dst1,
            vreg_p_scale_f8e8m0_0, vreg_p_scale_f8e8m0_0);
        StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(
            ((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0), vreg_p_scale_f8e8m0_dst0, preg_134);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__aicore__ inline void ProcessVec1DnUpdateMxfp8(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, float pScale, const T minValue, float keepProb,
    const LocalTensor<float>& preLoopMaxTensor, const LocalTensor<float>& preLoopSumTensor,
    const LocalTensor<float>& firstLoopSumTensor, int32_t subLoop,
    const LocalTensor<fp8_e8m0_t> &pScaleSubLoop0Tensor)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ float *pre_loop_max = (__ubuf__ T*)preLoopMaxTensor.GetPhyAddr();
    __ubuf__ float* pre_loop_sum = (__ubuf__ T*) preLoopSumTensor.GetPhyAddr();
    __ubuf__ float* first_loop_sum = (__ubuf__ T*) firstLoopSumTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0Tensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    __ubuf__ uint8_t * indexesPScaleUb = nullptr;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnUpdateMxfp8VF<T, T2, hasAtten, ubN>(x_exp, input_x_local_UB, exp_max_fp32,
        new_global_sum, new_global_max, maskUb, indexesUb, pScaleSubLoop0Ub, m, n, originN,
        scale, deScaleQK, pScale, minValue, keepProb, dScale, blockStride,
        repeatStride, pre_loop_max, pre_loop_sum, first_loop_sum, subLoop);
}

/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256>
__aicore__ inline void ProcessVec1VfDnMxfp8(const LocalTensor<T2>& dstTensor,
    const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    TBuf<> *vselrIndexesBuf, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<fp8_e8m0_t>& pScaleSubLoop0Tensor, const uint32_t m, const uint32_t n,
    const uint32_t originN, const T scale, float deScaleQK, float pScale,
    const T minValue, float keepProb, const LocalTensor<T>& preLoopMaxTensor,
    const LocalTensor<T>& preLoopSumTensor, const LocalTensor<T>& firstLoopSumTensor, int32_t subLoop)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdateMxfp8<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor,
            m, n, originN, scale, deScaleQK, pScale, minValue, keepProb, subLoop, pScaleSubLoop0Tensor);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdateMxfp8<T, T2, hasAtten, ubN>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor,
            m, n, originN, scale, deScaleQK, pScale, minValue, keepProb, preLoopMaxTensor, preLoopSumTensor,
            firstLoopSumTensor, subLoop, pScaleSubLoop0Tensor);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128,
          bool hasSink = false, bool hasInvalidLine = false>
__simd_vf__ inline void ProcessVec1DnCausalVF(__ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB,
    __ubuf__ float *exp_max_fp32, __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max,
    __ubuf__ uint32_t *maskUb, __ubuf__ T *tmpExpSumUb, __ubuf__ T *tmpMaxUb,
    __ubuf__ uint8_t *indexesUb, const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, const T minValue, float keepProb, bool needAtten,
    const float dScale, const uint32_t blockStride, const uint32_t repeatStride,
    const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + (m >> 2);
    __ubuf__ uint32_t *mask_ub2 = maskUb + (m >> 2) * 2;
    __ubuf__ uint32_t *mask_ub3 = maskUb + (m >> 2) * 3;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);
    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value || IsSameType<T2, bfloat16_t>::value ||
                    IsSameType<T2, half>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                if constexpr (hasInvalidLine) {
                    Muls(src0, src0, dScale, preg_108);
                    Muls(src1, src1, dScale, preg_108);
                    Muls(src2, src2, dScale, preg_108);
                    Muls(src3, src3, dScale, preg_108);
                }
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                if constexpr (hasInvalidLine) {
                    Muls(src0, src0, dScale, preg_108);
                    Muls(src1, src1, dScale, preg_108);
                    Muls(src2, src2, dScale, preg_108);
                    Muls(src3, src3, dScale, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                }
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            if constexpr (hasInvalidLine) {
                Muls(src0, src0, dScale, preg_108);
                Muls(src1, src1, dScale, preg_108);
                Muls(src2, src2, dScale, preg_108);
                Muls(src3, src3, dScale, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            }
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }
    if constexpr (isUpdate) {
        LoadAlign(vreg_x_max_f32_b, new_global_max);
    }
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);
    if constexpr (!hasInvalidLine) {
        Muls(max0, max0, dScale, preg_108);
    }
    if constexpr (hasSink) {
        Max(max0, max0, vreg_sink_input, preg_108);
    }

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    if constexpr (isUpdate) {
        Max(max0, max0, vreg_x_max_f32_b, preg_108);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)tmpMaxUb, max0, preg_108);
    } else {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            (__ubuf__ T *&)new_global_max, max0, preg_108);
    }
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }
    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }
        if constexpr (!hasInvalidLine) {
            Muls(vreg_x_f32_0, vreg_x_f32_0, dScale, preg_108);
            Muls(vreg_x_f32_1, vreg_x_f32_1, dScale, preg_108);
            Muls(vreg_x_f32_2, vreg_x_f32_2, dScale, preg_108);
            Muls(vreg_x_f32_3, vreg_x_f32_3, dScale, preg_108);
        }

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            if constexpr (!hasInvalidLine) {
                Muls(vreg_x_f32_4, vreg_x_f32_4, dScale, preg_108);
                Muls(vreg_x_f32_5, vreg_x_f32_5, dScale, preg_108);
                Muls(vreg_x_f32_6, vreg_x_f32_6, dScale, preg_108);
                Muls(vreg_x_f32_7, vreg_x_f32_7, dScale, preg_108);
            }

            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa, vreg_x_exp_even_bf16_1,
                         vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    if constexpr (isUpdate) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)tmpExpSumUb, vreg_x_sum0, preg_134);
    } else {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 128,
          bool hasSink = false, bool hasInvalidLine = false>
__aicore__ inline void ProcessVec1DnCausal(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
    const uint32_t n, const uint32_t originN, const T scale, float deScaleQK, const T minValue, float keepProb,
    bool needAtten, const float pScale = 1.0f, const float sinkValue = 0.0f)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    __ubuf__ T *tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T *tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    ProcessVec1DnCausalVF<T, T2, isUpdate, hasAtten, ubN, hasSink, hasInvalidLine>(x_exp, input_x_local_UB,
        exp_max_fp32, new_global_sum, new_global_max, maskUb, tmpExpSumUb, tmpMaxUb, indexesUb, m, n, originN,
        scale, deScaleQK, minValue, keepProb, needAtten, dScale, blockStride, repeatStride, pScale, sinkValue);
}
/*
 * @ingroup ProcessVec1Vf
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [in] srcTensor, input LocalTensor
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] n, input colums, should be 256 bytes aligned, the value is originN aligned to 64
 * @param [in] originN, input origin colums, support range: 0 < originN <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] isUpdate, enable flash mode
 * @param [in] oriNRange, originN range
 */

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
    bool hasSink = false>
__aicore__ inline void ProcessVec1VfDn(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, TBuf<> *vselrIndexesBuf,
                                       const LocalTensor<uint8_t>& maskTensor, const uint32_t m, const uint32_t n,
                                       const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                       float keepProb, bool needAtten, const float pScale = 1.0f,
                                       const float sinkValue = 0.0f)
{
    if constexpr (!isUpdate) {
        // 索引序列，用于在VF代码中量化cast后的gather指令
        // 在对计算结果进行cast之后，数据从fp32转换成了fp8，可能会得到一个乱序的非紧密排布的结果，需要定义一个索引序列来把结果调整正确
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdate<T, T2, hasAtten, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor,
            m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdate<T, T2, hasAtten, ubN, hasSink>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor,
            m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
    }
}
template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false, uint16_t ubN = 256,
    bool hasSink = false, bool hasInvalidLine = false>
__aicore__ inline void ProcessVec1VfDnCausal(const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor,
                                       const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& expMaxTensor, TBuf<> *vselrIndexesBuf,
                                       const LocalTensor<uint8_t>& maskTensor,
                                       const LocalTensor<uint8_t>& sharedTmpBuffer,
                                       const uint32_t m, const uint32_t n,
                                       const uint32_t originN, const T scale, float deScaleQK, const T minValue,
                                       float keepProb, bool needAtten, const float pScale = 1.0f,
                                       const float sinkValue = 0.0f)
{
    LocalTensor<uint8_t> indexesTensor;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
        IsSameType<T2, hifloat8_t>::value) {
        indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
    }
    ProcessVec1DnCausal<T, T2, isUpdate, hasAtten, ubN, hasSink, hasInvalidLine>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, indexesTensor, maskTensor, sharedTmpBuffer,
            m, n, originN, scale, deScaleQK, minValue, keepProb, needAtten, pScale, sinkValue);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdatePerTokenHeadVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32,
    __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb,
    __ubuf__ uint8_t *indexesUb, __ubuf__ float * qScaleUb, __ubuf__ float * kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
    const T minValue, float keepProb, bool needAtten, const float dScale,
    const uint32_t blockStride, const uint32_t repeatStride, const float pScale = 1.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);

                Mul(src0, src0, vreg_qscale_vec, preg_135);
                Mul(src1, src1, vreg_qscale_vec, preg_135);
                Mul(src2, src2, vreg_qscale_vec, preg_135);
                Mul(src3, src3, vreg_qscale_vec, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
                Mul(src0, src0, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
                Mul(src1, src1, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
                Mul(src2, src2, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
                Mul(src3, src3, vreg_kscale_val, preg_135);

                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);

                Mul(src0, src0, vreg_qscale_vec, preg_135);
                Mul(src1, src1, vreg_qscale_vec, preg_135);
                Mul(src2, src2, vreg_qscale_vec, preg_135);
                Mul(src3, src3, vreg_qscale_vec, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
                Mul(src0, src0, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
                Mul(src1, src1, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
                Mul(src2, src2, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
                Mul(src3, src3, vreg_kscale_val, preg_135);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);

            Mul(src0, src0, vreg_qscale_vec, preg_135);
            Mul(src1, src1, vreg_qscale_vec, preg_135);
            Mul(src2, src2, vreg_qscale_vec, preg_135);
            Mul(src3, src3, vreg_qscale_vec, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
            Mul(src0, src0, vreg_kscale_val, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
            Mul(src1, src1, vreg_kscale_val, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
            Mul(src2, src2, vreg_kscale_val, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
            Mul(src3, src3, vreg_kscale_val, preg_135);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnNoUpdatePerTokenHeadTailVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32,
    __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb,
    __ubuf__ uint8_t *indexesUb, __ubuf__ float * qScaleUb, __ubuf__ float * kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
    const T minValue, float keepProb, bool needAtten, const float dScale,
    const uint32_t blockStride, const uint32_t repeatStride, const float pScale = 1.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;

    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;
    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }

    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);

    // qscale和kscale应用循环
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m * 4);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, vreg_data_tmp3, preg_135);
    }
    mem_bar(VST_VLD);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ T *&)new_global_max, max0, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }

        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);
            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }

    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_x_sum0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdatePerTokenHeadVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32,
    __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb,
    __ubuf__ uint8_t *indexesUb, __ubuf__ float * qScaleUb, __ubuf__ float * kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
    const T minValue, float keepProb, bool needAtten, const float dScale,
    const uint32_t blockStride, const uint32_t repeatStride, const float pScale = 1.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);

    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);

    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);

                Mul(src0, src0, vreg_qscale_vec, preg_135);
                Mul(src1, src1, vreg_qscale_vec, preg_135);
                Mul(src2, src2, vreg_qscale_vec, preg_135);
                Mul(src3, src3, vreg_qscale_vec, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
                Mul(src0, src0, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
                Mul(src1, src1, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
                Mul(src2, src2, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
                Mul(src3, src3, vreg_kscale_val, preg_135);

                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);

                Mul(src0, src0, vreg_qscale_vec, preg_135);
                Mul(src1, src1, vreg_qscale_vec, preg_135);
                Mul(src2, src2, vreg_qscale_vec, preg_135);
                Mul(src3, src3, vreg_qscale_vec, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
                Mul(src0, src0, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
                Mul(src1, src1, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
                Mul(src2, src2, vreg_kscale_val, preg_135);
                LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
                Mul(src3, src3, vreg_kscale_val, preg_135);

                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);

            Mul(src0, src0, vreg_qscale_vec, preg_135);
            Mul(src1, src1, vreg_qscale_vec, preg_135);
            Mul(src2, src2, vreg_qscale_vec, preg_135);
            Mul(src3, src3, vreg_qscale_vec, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
            Mul(src0, src0, vreg_kscale_val, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
            Mul(src1, src1, vreg_kscale_val, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
            Mul(src2, src2, vreg_kscale_val, preg_135);
            LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
            Mul(src3, src3, vreg_kscale_val, preg_135);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128>
__simd_vf__ inline void ProcessVec1DnUpdatePerTokenHeadTailVF(
    __ubuf__ T2 *x_exp, __ubuf__ float *input_x_local_UB, __ubuf__ float *exp_max_fp32,
    __ubuf__ float *new_global_sum, __ubuf__ float *new_global_max, __ubuf__ uint32_t *maskUb,
    __ubuf__ uint8_t *indexesUb, __ubuf__ float * qScaleUb, __ubuf__ float * kScaleUb,
    const uint32_t m, const uint32_t n, const uint32_t originN, const T scale, float deScaleQK,
    const T minValue, float keepProb, bool needAtten, const float dScale,
    const uint32_t blockStride, const uint32_t repeatStride, const float pScale = 1.0f)
{
    RegTensor<float> vreg_x_sum_0;
    RegTensor<float> vreg_x_sum_1;
    RegTensor<float> vreg_x_sum_2;
    RegTensor<float> vreg_x_sum_3;
    RegTensor<float> vreg_x_sum_4;
    RegTensor<float> vreg_x_sum_5;
    RegTensor<float> vreg_x_sum_6;
    RegTensor<float> vreg_x_sum_7;
    RegTensor<float> vreg_x_sum0;
    RegTensor<float> vreg_x_sum1;
    RegTensor<float> vreg_x_sum2;
    RegTensor<float> vreg_x_sum3;
    RegTensor<half> vreg_x_exp_even_f16;
    RegTensor<half> vreg_x_exp_odd_f16;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16;
    RegTensor<float> vreg_sink_input;

    RegTensor<float> vreg_x_exp_0;
    RegTensor<float> vreg_x_exp_1;
    RegTensor<float> vreg_x_exp_2;
    RegTensor<float> vreg_x_exp_3;
    RegTensor<float> vreg_x_exp_4;
    RegTensor<float> vreg_x_exp_5;
    RegTensor<float> vreg_x_exp_6;
    RegTensor<float> vreg_x_exp_7;
    RegTensor<half> vreg_x_exp_even_f16_1;
    RegTensor<half> vreg_x_exp_odd_f16_1;
    RegTensor<bfloat16_t> vreg_x_exp_even_bf16_1;
    RegTensor<bfloat16_t> vreg_x_exp_odd_bf16_1;

    RegTensor<float> vreg_x_f32_0;
    RegTensor<float> vreg_x_f32_1;
    RegTensor<float> vreg_x_f32_2;
    RegTensor<float> vreg_x_f32_3;
    RegTensor<float> vreg_x_f32_4;
    RegTensor<float> vreg_x_f32_5;
    RegTensor<float> vreg_x_f32_6;
    RegTensor<float> vreg_x_f32_7;
    RegTensor<float> vreg_x_max_f32_b;
    RegTensor<half> vreg_x_exp_f16_pack;
    RegTensor<half> vreg_x_exp_f16_1_pack;
    RegTensor<half> vreg_x_exp_f16_packa;
    RegTensor<half> vreg_x_exp_f16_1_packa;

    RegTensor<bfloat16_t> vreg_x_exp_bf16_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_pack;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_packa;
    RegTensor<bfloat16_t> vreg_x_exp_bf16_1_packa;

    MaskReg preg_108;
    MaskReg preg_134;
    MaskReg preg_135;
    MaskReg preg_136;
    preg_108 = CreateMask<uint16_t, MaskPattern::ALL>();
    preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
    preg_135 = CreateMask<T, MaskPattern::ALL>();
    uint32_t sreg_92 = (uint32_t)128ULL;
    preg_136 = UpdateMask<uint16_t>(sreg_92);
    RegTensor<float> src0, src1, src2, src3;
    RegTensor<float> max0, max1, max2, max3;
    MaskReg preg_compare0, preg_compare1, preg_compare2, preg_compare3;
    RegTensor<float> vreg_min;

    RegTensor<float> vreg_p_scale;
    RegTensor<float> vreg_ln_p_scale;

    RegTensor<T2> vreg_x_exp_fp8_0, vreg_x_exp_f8_pack_0;
    RegTensor<T2> vreg_x_exp_fp8_1, vreg_x_exp_f8_pack_1;

    RegTensor<float> vreg_qscale_vec;
    RegTensor<float> vreg_kscale_val;
    RegTensor<float> vreg_data_tmp0, vreg_data_tmp1, vreg_data_tmp2, vreg_data_tmp3;

    __ubuf__ T2 *x_exp_1;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        x_exp_1 = x_exp + 32;
    } else {
        x_exp_1 = x_exp + (ubN * 4);
    }
    __ubuf__ float *src_ub0 = input_x_local_UB;
    __ubuf__ float *src_ub1 = src_ub0 + m;
    __ubuf__ float *src_ub2 = src_ub0 + m * 2;
    __ubuf__ float *src_ub3 = src_ub0 + m * 3;
    __ubuf__ uint32_t *mask_ub0 = maskUb;
    __ubuf__ uint32_t *mask_ub1 = maskUb + 16;
    __ubuf__ uint32_t *mask_ub2 = maskUb + 32;
    __ubuf__ uint32_t *mask_ub3 = maskUb + 48;

    Duplicate(max0, minValue);
    Duplicate(max1, minValue);
    Duplicate(max2, minValue);
    Duplicate(max3, minValue);
    Duplicate(vreg_min, minValue);

    // qscale和kscale应用循环
    LoadAlign(vreg_qscale_vec, qScaleUb);
    Muls(vreg_qscale_vec, vreg_qscale_vec, dScale, preg_108);
    for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
        LoadAlign(vreg_data_tmp0, src_ub0 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp1, src_ub1 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp2, src_ub2 + iter_m * m * 4);
        LoadAlign(vreg_data_tmp3, src_ub3 + iter_m * m * 4);

        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_qscale_vec, preg_135);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_qscale_vec, preg_135);

        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4);
        Mul(vreg_data_tmp0, vreg_data_tmp0, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 1);
        Mul(vreg_data_tmp1, vreg_data_tmp1, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 2);
        Mul(vreg_data_tmp2, vreg_data_tmp2, vreg_kscale_val, preg_135);
        LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_kscale_val, kScaleUb + iter_m * 4 + 3);
        Mul(vreg_data_tmp3, vreg_data_tmp3, vreg_kscale_val, preg_135);

        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, vreg_data_tmp0, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, vreg_data_tmp1, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, vreg_data_tmp2, preg_135);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, vreg_data_tmp3, preg_135);
    }
    mem_bar(VST_VLD);
    for (uint16_t i = originN; i < ubN; ++i) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)input_x_local_UB + i * m, vreg_min, preg_135);
    }
    mem_bar(VST_VLD);
    if constexpr ((IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                    IsSameType<T2, hifloat8_t>::value) && hasAtten) {
        if (needAtten) {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare0, mask_ub0 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare1, mask_ub1 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare2, mask_ub2 + iter_m * m);
                LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>(preg_compare3, mask_ub3 + iter_m * m);
                Select(src0, src0, vreg_min, preg_compare0);
                Select(src1, src1, vreg_min, preg_compare1);
                Select(src2, src2, vreg_min, preg_compare2);
                Select(src3, src3, vreg_min, preg_compare3);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub0 + iter_m * m * 4, src0, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub1 + iter_m * m * 4, src1, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub2 + iter_m * m * 4, src2, preg_108);
                StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(src_ub3 + iter_m * m * 4, src3, preg_108);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        } else {
            for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
                LoadAlign(src0, src_ub0 + iter_m * m * 4);
                LoadAlign(src1, src_ub1 + iter_m * m * 4);
                LoadAlign(src2, src_ub2 + iter_m * m * 4);
                LoadAlign(src3, src_ub3 + iter_m * m * 4);
                Max(max0, max0, src0, preg_108);
                Max(max1, max1, src1, preg_108);
                Max(max2, max2, src2, preg_108);
                Max(max3, max3, src3, preg_108);
            }
        }
    } else {
        for (uint16_t iter_m = 0; iter_m < uint16_t(ubN / 4); ++iter_m) {
            LoadAlign(src0, src_ub0 + iter_m * m * 4);
            LoadAlign(src1, src_ub1 + iter_m * m * 4);
            LoadAlign(src2, src_ub2 + iter_m * m * 4);
            LoadAlign(src3, src_ub3 + iter_m * m * 4);
            Max(max0, max0, src0, preg_108);
            Max(max1, max1, src1, preg_108);
            Max(max2, max2, src2, preg_108);
            Max(max3, max3, src3, preg_108);
        }
    }

    LoadAlign(vreg_x_max_f32_b, new_global_max);
    Max(max0, max0, max2, preg_108);
    Max(max1, max1, max3, preg_108);
    Max(max0, max0, max1, preg_108);

    Duplicate(vreg_p_scale, pScale);
    Ln(vreg_ln_p_scale, vreg_p_scale, preg_108);
    Sub(max0, max0, vreg_ln_p_scale, preg_108);
    Max(max0, max0, vreg_x_max_f32_b, preg_108);

    FusedExpSub(vreg_x_max_f32_b, vreg_x_max_f32_b, max0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)new_global_max, max0, preg_108);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
        (__ubuf__ T *&)exp_max_fp32, vreg_x_max_f32_b, preg_108);

    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_1, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_2, 0, preg_134);
    Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_3, 0, preg_134);
    RegTensor<uint8_t> idx_nd2nz;
    uint16_t loopNum;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_4, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_5, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_6, 0, preg_134);
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_7, 0, preg_134);
        LoadAlign(idx_nd2nz, indexesUb);
        loopNum = ubN / 8;
    } else {
        loopNum = ubN / 4;
    }

    if constexpr (hasAtten == true) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    for (uint16_t i0 = 0; i0 < loopNum; ++i0) {
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m * 2);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m * 2);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m * 2);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2);

            LoadAlign(vreg_x_f32_4, input_x_local_UB + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_5, input_x_local_UB + ubN * m / 4 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_6, input_x_local_UB + ubN * m / 2 + i0 * m * 2 + 64);
            LoadAlign(vreg_x_f32_7, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m * 2 + 64);
        } else {
            LoadAlign(vreg_x_f32_0, input_x_local_UB + i0 * m);
            LoadAlign(vreg_x_f32_1, input_x_local_UB + ubN * m / 4 + i0 * m);
            LoadAlign(vreg_x_f32_2, input_x_local_UB + ubN * m / 2 + i0 * m);
            LoadAlign(vreg_x_f32_3, input_x_local_UB + ubN * m / 2 + ubN * m / 4 + i0 * m);
        }

        FusedExpSub(vreg_x_exp_0, vreg_x_f32_0, max0, preg_134);
        FusedExpSub(vreg_x_exp_1, vreg_x_f32_1, max0, preg_134);
        FusedExpSub(vreg_x_exp_2, vreg_x_f32_2, max0, preg_134);
        FusedExpSub(vreg_x_exp_3, vreg_x_f32_3, max0, preg_134);

        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            FusedExpSub(vreg_x_exp_4, vreg_x_f32_4, max0, preg_134);
            FusedExpSub(vreg_x_exp_5, vreg_x_f32_5, max0, preg_134);
            FusedExpSub(vreg_x_exp_6, vreg_x_f32_6, max0, preg_134);
            FusedExpSub(vreg_x_exp_7, vreg_x_f32_7, max0, preg_134);
        }
        if constexpr (AscendC::IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_bf16_pack, vreg_x_exp_bf16_packa, vreg_x_exp_even_bf16, vreg_x_exp_odd_bf16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_bf16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_bf16_1, vreg_x_exp_3, preg_135);
            DeInterleave(vreg_x_exp_bf16_1_pack, vreg_x_exp_bf16_1_packa,
                         vreg_x_exp_even_bf16_1, vreg_x_exp_odd_bf16_1);
            /* vreg_x_exp_bf16_pack会不连续的存储156在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_bf16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_bf16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_bf16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        } else if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_0, vreg_x_exp_0, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_0, vreg_x_exp_1, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_1, vreg_x_exp_2, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_2, vreg_x_exp_3, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_0, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_1, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_0, (RegTensor<uint8_t>&)vreg_x_exp_fp8_0,
               (RegTensor<uint8_t>&)vreg_x_exp_2, preg_134);
            Gather(vreg_x_exp_f8_pack_0, vreg_x_exp_fp8_0, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f8_pack_0, blockStride, repeatStride, preg_134);

            // -----------------------------------------------------------------------------//
            Add(vreg_x_sum_4, vreg_x_exp_4, vreg_x_sum_4, preg_134);
            Add(vreg_x_sum_5, vreg_x_exp_5, vreg_x_sum_5, preg_134);
            Add(vreg_x_sum_6, vreg_x_exp_6, vreg_x_sum_6, preg_134);
            Add(vreg_x_sum_7, vreg_x_exp_7, vreg_x_sum_7, preg_134);
            if constexpr (IsSameType<T2, hifloat8_t>::value) {
                Cast<T2, T, castTraitZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            } else {
                Cast<T2, T, castTraitRintZero>(vreg_x_exp_fp8_1, vreg_x_exp_4, preg_135);
                Cast<T2, T, castTraitRintOne>((RegTensor<T2>&)vreg_x_exp_4, vreg_x_exp_5, preg_135);
                Cast<T2, T, castTraitRintTwo>((RegTensor<T2>&)vreg_x_exp_5, vreg_x_exp_6, preg_135);
                Cast<T2, T, castTraitRintThree>((RegTensor<T2>&)vreg_x_exp_6, vreg_x_exp_7, preg_135);
            }

            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_4, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_5, preg_134);
            Or((RegTensor<uint8_t>&)vreg_x_exp_fp8_1, (RegTensor<uint8_t>&)vreg_x_exp_fp8_1,
               (RegTensor<uint8_t>&)vreg_x_exp_6, preg_134);
            Gather(vreg_x_exp_f8_pack_1, vreg_x_exp_fp8_1, idx_nd2nz);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f8_pack_1, blockStride, repeatStride, preg_134);
        } else {
            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16, vreg_x_exp_0, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16, vreg_x_exp_2, preg_135);
            DeInterleave(vreg_x_exp_f16_pack, vreg_x_exp_f16_packa, vreg_x_exp_even_f16, vreg_x_exp_odd_f16);

            Cast<T2, T, castTraitZero>(vreg_x_exp_even_f16_1, vreg_x_exp_1, preg_135);
            Cast<T2, T, castTraitZero>(vreg_x_exp_odd_f16_1, vreg_x_exp_3, preg_135);

            DeInterleave(vreg_x_exp_f16_1_pack, vreg_x_exp_f16_1_packa, vreg_x_exp_even_f16_1, vreg_x_exp_odd_f16_1);

            /* vreg_x_exp_f16_pack会不连续的存储在x_exp上，shape为2*4*64*16， 其中每64*16个的head之间跳129 * 16
                个数，中间跳的部分就是vreg_x_exp_f16_1_pack的 */
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp), vreg_x_exp_f16_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_0, vreg_x_exp_0, vreg_x_sum_0, preg_134);
            Add(vreg_x_sum_2, vreg_x_exp_2, vreg_x_sum_2, preg_134);

            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)x_exp_1), vreg_x_exp_f16_1_pack, blockStride, repeatStride, preg_136);
            Add(vreg_x_sum_1, vreg_x_exp_1, vreg_x_sum_1, preg_134);
            Add(vreg_x_sum_3, vreg_x_exp_3, vreg_x_sum_3, preg_134);
        }
    }
    Add(vreg_x_sum0, vreg_x_sum_2, vreg_x_sum_0, preg_134);
    Add(vreg_x_sum1, vreg_x_sum_3, vreg_x_sum_1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum_6, vreg_x_sum_4, preg_134);
        Add(vreg_x_sum3, vreg_x_sum_7, vreg_x_sum_5, preg_134);
    }
    Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum1, preg_134);
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
        Add(vreg_x_sum2, vreg_x_sum2, vreg_x_sum3, preg_134);
        Add(vreg_x_sum0, vreg_x_sum0, vreg_x_sum2, preg_134);
    }
    RegTensor<float> vreg_l0;
    LoadAlign(vreg_l0, new_global_sum);
    Mul(vreg_l0, vreg_x_max_f32_b, vreg_l0, preg_134);
    Add(vreg_l0, vreg_l0, vreg_x_sum0, preg_134);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)new_global_sum, vreg_l0, preg_134);
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool isTail = false>
__aicore__ inline void ProcessVec1DnNoUpdatePerTokenHead(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<T>& qScaleUbTensor, const LocalTensor<T>& kScaleUbTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, float pScale, const T minValue, float keepProb, bool needAtten)
{
    __ubuf__ T2 *x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float *input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float *exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float *new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float *new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    __ubuf__ float * qScaleUb = (__ubuf__ float*)qScaleUbTensor.GetPhyAddr();
    __ubuf__ float * kScaleUb = (__ubuf__ float*)kScaleUbTensor.GetPhyAddr();
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    if constexpr (isTail) {
        ProcessVec1DnNoUpdatePerTokenHeadTailVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max,
            maskUb, indexesUb, qScaleUb, kScaleUb, m, n, originN, scale, deScaleQK, minValue,
            keepProb, needAtten, dScale, blockStride, repeatStride, pScale);
    } else {
        ProcessVec1DnNoUpdatePerTokenHeadVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max,
            maskUb, indexesUb, qScaleUb, kScaleUb, m, n, originN, scale, deScaleQK, minValue,
            keepProb, needAtten, dScale, blockStride, repeatStride, pScale);
    }
}

template <typename T, typename T2, bool hasAtten = false, uint16_t ubN = 128, bool isTail = false>
__aicore__ inline void ProcessVec1DnUpdatePerTokenHead(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<uint8_t> &vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<T>& qScaleUbTensor, const LocalTensor<T>& kScaleUbTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, float pScale, const T minValue, float keepProb, bool needAtten)
{
    __ubuf__ T2* x_exp = (__ubuf__ T2*) dstTensor.GetPhyAddr();
    __ubuf__ float* input_x_local_UB = (__ubuf__ T*) srcTensor.GetPhyAddr();
    __ubuf__ float* exp_max_fp32 = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* new_global_sum = (__ubuf__ T*) expSumTensor.GetPhyAddr();
    __ubuf__ float* new_global_max = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ uint32_t *maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = nullptr;
    __ubuf__ float * qScaleUb = (__ubuf__ float*)qScaleUbTensor.GetPhyAddr();
    __ubuf__ float * kScaleUb = (__ubuf__ float*)kScaleUbTensor.GetPhyAddr();
    float dScale;
    uint32_t blockStride;
    uint32_t repeatStride;
    if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                IsSameType<T2, hifloat8_t>::value) {
        dScale = scale * deScaleQK;
        blockStride = ubN >> 2 | 0x1;
        repeatStride = 2;
        indexesUb = (__ubuf__ uint8_t*)vselrIndexesBuf.GetPhyAddr();
    } else {
        dScale = scale;
        blockStride = ubN >> 1 | 0x1;
        repeatStride = 1;
    }

    if constexpr (isTail) {
        ProcessVec1DnUpdatePerTokenHeadTailVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max,
            maskUb, indexesUb, qScaleUb, kScaleUb, m, n, originN, scale, deScaleQK, minValue,
            keepProb, needAtten, dScale, blockStride, repeatStride, pScale);
    } else {
        ProcessVec1DnUpdatePerTokenHeadVF<T, T2, hasAtten, ubN>(
            x_exp, input_x_local_UB, exp_max_fp32, new_global_sum, new_global_max,
            maskUb, indexesUb, qScaleUb, kScaleUb, m, n, originN, scale, deScaleQK, minValue,
            keepProb, needAtten, dScale, blockStride, repeatStride, pScale);
    }
}

template <typename T, typename T2, bool isUpdate = false, bool hasAtten = false,
          uint16_t ubN = 256, bool isTail = false>
__aicore__ inline void ProcessVec1VfDnPerTokenHead(
    const LocalTensor<T2>& dstTensor, const LocalTensor<T>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, TBuf<> *vselrIndexesBuf,
    const LocalTensor<uint8_t>& maskTensor, const LocalTensor<T>& qScaleUbTensor, const LocalTensor<T>& kScaleUbTensor,
    const uint32_t m, const uint32_t n, const uint32_t originN,
    const T scale, float deScaleQK, float pScale, const T minValue, float keepProb, bool needAtten)
{
    if constexpr (!isUpdate) {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnNoUpdatePerTokenHead<T, T2, hasAtten, ubN, isTail>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            indexesTensor, maskTensor, qScaleUbTensor, kScaleUbTensor,
            m, n, originN, scale, deScaleQK, pScale, minValue, keepProb, needAtten);
    } else {
        LocalTensor<uint8_t> indexesTensor;
        if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            IsSameType<T2, hifloat8_t>::value) {
            indexesTensor = vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        }
        ProcessVec1DnUpdatePerTokenHead<T, T2, hasAtten, ubN, isTail>(
            dstTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
            indexesTensor, maskTensor, qScaleUbTensor, kScaleUbTensor,
            m, n, originN, scale, deScaleQK, pScale, minValue, keepProb, needAtten);
    }
}

template <typename T>
__simd_vf__ inline void BroadCastMaxSumVF(__ubuf__ float *out_ub, __ubuf__ float *ori_ub, const uint16_t loopM)
{
    RegTensor<float> broadcast_reg;
    MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopM; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(broadcast_reg, ori_ub + i * 8);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)out_ub + i * 64, broadcast_reg, preg_all);
    }
}

template <typename T>
__aicore__ inline void BroadcastMaxSum(const LocalTensor<T>& outTensor, const LocalTensor<T> &oriTensor,
                                       uint32_t vecS1RealSize)
{
    __ubuf__ float *out_ub = (__ubuf__ T*)outTensor.GetPhyAddr();
    __ubuf__ float *ori_ub = (__ubuf__ T*)oriTensor.GetPhyAddr();

    // Align8, broadcast one element to 8 elements, one register can store 64 elements,
    // so we can handle 64 / 8 = 8 elements per loop.
    uint16_t loopM = (vecS1RealSize + 7) >> 3;
    BroadCastMaxSumVF<T>(out_ub, ori_ub, loopM);
}
}
#endif // MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_