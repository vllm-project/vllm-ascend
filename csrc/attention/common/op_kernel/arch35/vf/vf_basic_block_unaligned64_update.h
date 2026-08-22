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
 * \file vf_basic_block_unaligned64_update.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED64_UPDATE_H
#define VF_BASIC_BLOCK_UNALIGNED64_UPDATE_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
    bool isMlaFullQuant = false, bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateImpl64VF(
    __ubuf__ T2 * expUb, __ubuf__ pseShiftType * pseUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * inMaxUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, __ubuf__ T * tmpMaxUb2,
    __ubuf__ T * qScaleUb, __ubuf__ T * pScaleUb, __ubuf__ uint8_t * indexesUb, __ubuf__ uint32_t * maskUb,
    __ubuf__ uint32_t * dropMaskUb, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride,
    const float dScale, uint32_t pltOriginalN, float divValue, uint32_t pltSrcN, uint32_t pltSrcN16, const uint16_t m,
    const uint32_t pseStride, const float slopes, const float posShift, const T scale, const float dScaleQK,
    const T minValue, const float deSCaleKValue = 1.0f, const float sinkValue = 0.0f)
{
    RegTensor<float> vreg_min;
    RegTensor<float> vreg_sel;
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_unroll;
    RegTensor<float> vreg_max_tmp;
    RegTensor<float> vreg_input_max;
    RegTensor<float> vreg_max_new;
    RegTensor<float> vreg_zero;
    RegTensor<float> vreg_exp;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_pse;
    RegTensor<float> vreg_alibi;
    RegTensor<float> vreg_sel_drop;
    RegTensor<float> vreg_rowmax_p;
    RegTensor<float> vreg_scale_qk;
    RegTensor<float> vreg_sink_input;

    // bfloat16_t
    RegTensor<bfloat16_t> vreg_exp_even_bf16;
    RegTensor<bfloat16_t> vreg_exp_bf16;
    RegTensor<bfloat16_t> vreg_dst_even_bf16;
    RegTensor<bfloat16_t> vreg_dst_odd_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_src;
    RegTensor<bfloat16_t> vreg_pse_bf16;
    RegTensor<bfloat16_t> vreg_pse_bf16_unroll;
    // half
    RegTensor<half> vreg_exp_even_f16;
    RegTensor<half> vreg_exp_f16;
    RegTensor<half> vreg_dst_even_f16;
    RegTensor<half> vreg_dst_odd_f16;
    RegTensor<half> vreg_pse_f16_src;
    RegTensor<half> vreg_pse_f16;
    RegTensor<half> vreg_pse_f16_unroll;

    UnalignRegForStore ureg_max;
    UnalignRegForStore ureg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();

    MaskReg preg_src_n = UpdateMask<T>(pltSrcN);
    MaskReg preg_src_n_b16 = UpdateMask<uint16_t>(pltSrcN16);
    MaskReg preg_ori_src_n = UpdateMask<T>(pltOriginalN);
    MaskReg preg_compare;
    MaskReg preg1;
    MaskReg preg2 = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg preg3;
    MaskReg preg4;

    if constexpr (hasSink) {
        Duplicate(vreg_sink_input, sinkValue);
    }
    if constexpr (hasAtten == 1) {
        Duplicate(vreg_min, minValue);
        if constexpr (isMlaSgd) {
            MicroAPI::LoadAlign<uint32_t, MicroAPI::MaskDist::DIST_DS>
                (preg_compare, ((__ubuf__ uint32_t*)(maskUb)));
        }
    }
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        Arange(vreg_alibi, posShift);
    }
    // x_max = max(src, axis=-1, keepdims=True)
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        if constexpr (isMlaFullQuant) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_scale_qk, qScaleUb + i);
            Muls(vreg_scale_qk, vreg_scale_qk, scale, preg_ori_src_n);
            Muls(vreg_scale_qk, vreg_scale_qk, deSCaleKValue, preg_ori_src_n);
            Mul(vreg_input_x, vreg_input_x, vreg_scale_qk, preg_ori_src_n);
        } else {
            if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                Muls(vreg_input_x, vreg_input_x, dScale, preg_ori_src_n);  // Muls(scale)
            } else {
                if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
                            IsSameType<T2, hifloat8_t>::value) {
                    Muls(vreg_input_x, vreg_input_x, dScaleQK, preg_ori_src_n);  // Muls(dScaleQK)
                }
            }
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                            pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                Abs(vreg_pse, vreg_alibi, preg_all);
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    Sqrt(vreg_pse, vreg_pse, preg_all);
                }
                Muls(vreg_pse, vreg_pse, slopes, preg_all);
                Adds(vreg_alibi, vreg_alibi, -1.0f, preg_all);
            } else {
                if constexpr (IsSameType<pseShiftType, float>::value) {
                    LoadAlign(vreg_pse, pseUb + i * pseStride);
                } else if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
                    LoadAlign(vreg_pse_bf16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_bf16, vreg_pse_bf16_unroll, vreg_pse_bf16_src, vreg_pse_bf16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_bf16, preg_all_b16);
                } else {
                    LoadAlign(vreg_pse_f16_src, pseUb + i * pseStride);
                    Interleave(vreg_pse_f16, vreg_pse_f16_unroll, vreg_pse_f16_src, vreg_pse_f16_src);
                    Cast<T, pseShiftType, castTraitZero>(vreg_pse, vreg_pse_f16, preg_all_b16);
                }
            }
            Add(vreg_input_x, vreg_input_x, vreg_pse, preg_ori_src_n);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            Muls(vreg_input_x, vreg_input_x, scale, preg_ori_src_n);  // Muls(scale)
        }
        if constexpr (hasAtten == 1) {
            // atten mask
            if constexpr (!isMlaSgd) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    preg_compare, (__ubuf__ uint32_t *&)maskUb, nPadding);
            }
            Select(vreg_sel, vreg_min, vreg_input_x, preg_compare);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_sel, preg_src_n);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_sel, preg_ori_src_n);
        } else {
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_src_n);
            Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                vreg_input_max, vreg_input_x, preg_ori_src_n);
        }
        if constexpr (hasSink) {
            Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_ori_src_n);
        }
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), vreg_input_max, ureg_max, 1);
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpMaxUb), ureg_max, 0);
    LoadAlign(vreg_in_max, inMaxUb);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(vreg_input_max, tmpMaxUb2);
    Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all); // 计算新、旧的最大值
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)tmpMaxUb2, vreg_max_new, preg_all);
    if constexpr (isMlaFullQuant) {
        ExpSub(vreg_rowmax_p, vreg_input_max, vreg_max_new, preg_all);
        Adds(vreg_rowmax_p, vreg_rowmax_p, floatEps, preg_all);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)pScaleUb, vreg_rowmax_p, preg_all);
    }

    if constexpr (hasDrop == 1) {
        Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg_max, tmpMaxUb2 + i);
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        ExpSub(vreg_exp, vreg_input_x, vreg_max, preg_ori_src_n);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_exp_sum, vreg_exp, preg_ori_src_n);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), vreg_exp_sum, ureg_exp_sum, 1);
        if constexpr (isMlaFullQuant) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_rowmax_p, pScaleUb + i);
            if constexpr (IsSameType<T2, int8_t>::value) {
                USE_MLA_FULLQUANT_V1_P_INT8(vreg_exp, vreg_rowmax_p, preg_ori_src_n);
            } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                USE_MLA_FULLQUANT_V1_P(vreg_exp, vreg_rowmax_p, preg_ori_src_n);
            } else {
                USE_MLA_FULLQUANT_V1_P_HIFP8(vreg_exp, vreg_rowmax_p, preg_ori_src_n);
            }
        }
        
        // dropmask compute
        if constexpr (hasDrop == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
                preg1, (__ubuf__ uint32_t *&)dropMaskUb, s2BaseSize >> 3);
            MaskInterleave<half>(preg3, preg4, preg1, preg2);
            Select(vreg_sel_drop, vreg_exp, vreg_zero, preg3);
            Muls(vreg_exp, vreg_sel_drop, divValue, preg_all);
        }

        if constexpr (IsSameType<T2, float>::value) {
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp, blockStride, repeatStride, preg_all);
        } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
            Cast<T2, T, castTraitZero>(vreg_exp_bf16, vreg_exp, preg_all_b16);
            DeInterleave(vreg_dst_even_bf16, vreg_dst_odd_bf16,
                    vreg_exp_bf16, vreg_exp_bf16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_dst_even_bf16, blockStride, repeatStride, preg_src_n_b16);
        }  else if constexpr (IsSameType<T2, fp8_e5m2_t>::value) {
            RegTensor<fp8_e5m2_t> vreg_exp_f8e5m2;
            RegTensor<uint8_t> vreg_exp_merge_f8e5m2_indexes;
            RegTensor<fp8_e5m2_t> vreg_exp_merge_f8e5m2;
            MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_f8e5m2, vreg_exp, preg_all);
            LoadAlign(vreg_exp_merge_f8e5m2_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e5m2, vreg_exp_f8e5m2, vreg_exp_merge_f8e5m2_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e5m2, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
            RegTensor<fp8_e4m3fn_t> vreg_exp_f8e4m3;
            RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
            RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3;
            MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitRintZero>(vreg_exp_f8e4m3, vreg_exp, preg_all);
            LoadAlign(vreg_exp_merge_f8e4m3_indexes, indexesUb);
            Gather(vreg_exp_merge_f8e4m3, vreg_exp_f8e4m3, vreg_exp_merge_f8e4m3_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_f8e4m3, blockStride, repeatStride, preg_all_b8_128);
        } else if constexpr (IsSameType<T2, int8_t>::value) {
            // 硬件不支持 float → int8 直接转换，需要分两步：float → half → int8
            RegTensor<int8_t> vreg_exp_merge_tmp_int8;
            RegTensor<int8_t> vreg_exp_merge_int8;
            RegTensor<half> vreg_cast_res;
            MaskReg preg_all_f16 = CreateMask<half, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            // float → half → Or → half → int8 → Gather
            static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            Cast<half, T, castTrait0>(vreg_exp_f16, vreg_exp, preg_all);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint16_t>&)vreg_cast_res,
                    (MicroAPI::RegTensor<uint32_t>&)vreg_exp_f16);     
            Cast<T2, half, castTrait0>(vreg_exp_merge_tmp_int8, vreg_cast_res, preg_all_f16);
            MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>((MicroAPI::RegTensor<uint8_t>&)vreg_exp_merge_int8,
                    (MicroAPI::RegTensor<uint16_t>&)vreg_exp_merge_tmp_int8);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ int8_t *&) expUb), vreg_exp_merge_int8, blockStride, repeatStride, preg_all_b8_128);                
        } else if constexpr (IsSameType<T2, hifloat8_t>::value) {
            RegTensor<hifloat8_t> vreg_exp_hif8;
            RegTensor<uint8_t> vreg_exp_merge_hif8_indexes;
            RegTensor<hifloat8_t> vreg_exp_merge_hif8;
            MaskReg preg_src_n_b8 = CreateMask<T2, MaskPattern::ALL>();
            uint32_t maskLen = 128;
            MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
            Cast<T2, T, castTraitZero>(vreg_exp_hif8, vreg_exp, preg_all);
            LoadAlign(vreg_exp_merge_hif8_indexes, indexesUb);
            Gather(vreg_exp_merge_hif8, vreg_exp_hif8, vreg_exp_merge_hif8_indexes);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_exp_merge_hif8, blockStride, repeatStride, preg_all_b8_128);
        } else {
            Cast<T2, T, castTraitZero>(vreg_exp_f16, vreg_exp, preg_all_b16);
            DeInterleave(vreg_dst_even_f16, vreg_dst_odd_f16, vreg_exp_f16, vreg_exp_f16);
            StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ T2 *&)expUb), vreg_dst_even_f16, blockStride, repeatStride, preg_src_n_b16);
        }
    }
    StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)tmpExpSumUb), ureg_exp_sum, 0);
}
// update, originN <= 64
template <typename T, typename T2, typename pseShiftType, uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE, bool hasDrop = 0, bool isMlaSgd = false,
    bool isMlaFullQuant = false, bool hasSink = false>
__aicore__ inline void ProcessVec1UpdateImpl64(
    const LocalTensor<T2>& dstTensor, const LocalTensor<uint8_t>& indexesTensor, const LocalTensor<T>& expSumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inExpSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& maskTensor,
    const LocalTensor<pseShiftType>& pseTensor, const LocalTensor<uint8_t>& dropTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& pScaleTensor, const uint16_t m,
    const uint32_t originN, const uint32_t pseStride, const float slopes, const float posShift, const T scale,
    const float dScaleQK, const T minValue, float keepProb, const LocalTensor<T>& queryScaleUb = LocalTensor<T>(),
    const float deSCaleKValue = 1.0f, const float sinkValue = 0.0f)
{
    const uint32_t nPadding = (s2BaseSize + blockBytesU8 - 1) / blockBytesU8 * blockBytesU8;
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    const float dScale = scale * dScaleQK;
    uint32_t pltOriginalN = originN;
    float divValue = 1.0f / keepProb;
    uint32_t pltSrcN = s2BaseSize;
    uint32_t pltSrcN16 = s2BaseSize;
    __ubuf__ T2 * expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ pseShiftType * pseUb = (__ubuf__ pseShiftType*)pseTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ T * inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ T * tmpExpSumUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T * tmpMaxUb = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * tmpMaxUb2 = (__ubuf__ T*)sharedTmpBuffer.GetPhyAddr() + 64;
    __ubuf__ T * qScaleUb = (__ubuf__ T*)queryScaleUb.GetPhyAddr();
    __ubuf__ T * pScaleUb = (__ubuf__ T*)pScaleTensor.GetPhyAddr();
    __ubuf__ uint8_t * indexesUb = (__ubuf__ uint8_t*)indexesTensor.GetPhyAddr();

    __ubuf__ uint32_t * maskUb = (__ubuf__ uint32_t*)maskTensor.GetPhyAddr();
    __ubuf__ uint32_t * dropMaskUb = (__ubuf__ uint32_t*)dropTensor.GetPhyAddr();

    ProcessVec1UpdateImpl64VF<T, T2, pseShiftType, s1BaseSize, s2BaseSize, hasAtten, pseMode, hasDrop,
        isMlaSgd, isMlaFullQuant, hasSink>(
        expUb, pseUb, maxUb, srcUb, expMaxUb, inMaxUb, tmpExpSumUb, tmpMaxUb, tmpMaxUb2, qScaleUb, pScaleUb,
        indexesUb, maskUb, dropMaskUb, nPadding, blockStride, repeatStride, dScale, pltOriginalN, divValue, pltSrcN,
        pltSrcN16, m, pseStride, slopes, posShift, scale, dScaleQK, minValue, deSCaleKValue, sinkValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED64_UPDATE_H
