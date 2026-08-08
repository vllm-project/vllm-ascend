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
 * \file vf_basic_block_unaligned1024_common.h
 * \brief shared __simd_callee__ helpers for unaligned1024 no_update / update VF kernels
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED1024_COMMON_H
#define VF_BASIC_BLOCK_UNALIGNED1024_COMMON_H

#include "vf_basic_block_utils.h"
#include "../pse.h"

using namespace regbaseutil;

namespace FaVectorApi {

#define VREG_FLOAT_PSE_ALIBI_16_DECL \
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2, \
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4, \
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6, \
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8, \
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10, \
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12, \
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14, \
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16, \
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2, \
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4, \
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6, \
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8, \
    RegTensor<float> &vreg_alibi9, RegTensor<float> &vreg_alibi10, \
    RegTensor<float> &vreg_alibi11, RegTensor<float> &vreg_alibi12, \
    RegTensor<float> &vreg_alibi13, RegTensor<float> &vreg_alibi14, \
    RegTensor<float> &vreg_alibi15, RegTensor<float> &vreg_alibi16, \
    const float slopes, MaskReg &preg_all

#define ARG_FLOAT_PSE_ALIBI_16 \
    vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4, \
    vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8, \
    vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12, \
    vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16, \
    vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4, \
    vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8, \
    vreg_alibi9, vreg_alibi10, vreg_alibi11, vreg_alibi12, \
    vreg_alibi13, vreg_alibi14, vreg_alibi15, vreg_alibi16, \
    slopes, preg_all

#define VREG_FLOAT_PSE_16_LOAD_DECL(T) \
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2, \
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4, \
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6, \
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8, \
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10, \
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12, \
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14, \
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16, \
    __ubuf__ T *&pseUb, const uint32_t i, const uint32_t pseStride, \
    MaskReg &preg_all_b16

#define ARG_FLOAT_PSE_16_LOAD \
    vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4, \
    vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8, \
    vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12, \
    vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16, \
    pseUb, i, pseStride, preg_all_b16

#define VREG_FLOAT_CAST_STORE_EXP_1024_DECL \
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1, \
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2, \
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3, \
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4, \
    RegTensor<float> &vreg_exp_even5, RegTensor<float> &vreg_exp_odd5, \
    RegTensor<float> &vreg_exp_even6, RegTensor<float> &vreg_exp_odd6, \
    RegTensor<float> &vreg_exp_even7, RegTensor<float> &vreg_exp_odd7, \
    RegTensor<float> &vreg_exp_even8, RegTensor<float> &vreg_exp_odd8, \
    __ubuf__ T2 *&expUb1, __ubuf__ T2 *&expUb2, \
    __ubuf__ T2 *&expUb3, __ubuf__ T2 *&expUb4, \
    __ubuf__ T2 *&expUb5, __ubuf__ T2 *&expUb6, \
    __ubuf__ T2 *&expUb7, __ubuf__ T2 *&expUb8, \
    const uint32_t blockStride, const uint32_t repeatStride, \
    MaskReg &preg_all, MaskReg &storeMask

#define ARG_FLOAT_CAST_STORE_EXP_1024 \
    vreg_exp_even1, vreg_exp_odd1, vreg_exp_even2, vreg_exp_odd2, \
    vreg_exp_even3, vreg_exp_odd3, vreg_exp_even4, vreg_exp_odd4, \
    vreg_exp_even5, vreg_exp_odd5, vreg_exp_even6, vreg_exp_odd6, \
    vreg_exp_even7, vreg_exp_odd7, vreg_exp_even8, vreg_exp_odd8, \
    expUb1, expUb2, expUb3, expUb4, expUb5, expUb6, expUb7, expUb8, \
    blockStride, repeatStride, preg_all, preg_all_b16

#define VREG_FLOAT_INPUT_X_16_DECL \
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2, \
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4, \
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6, \
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8, \
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10, \
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12, \
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14, \
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16, \
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize

#define ARG_FLOAT_INPUT_X_16 \
    vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4, \
    vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8, \
    vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12, \
    vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16, \
    srcUb, i, s2BaseSize

#define VREG_FLOAT_INPUT_X_16_NO_IDX_DECL \
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2, \
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4, \
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6, \
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8, \
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10, \
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12, \
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14, \
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16, \
    __ubuf__ T *&srcUb, const uint32_t s2BaseSize

#define ARG_FLOAT_INPUT_X_16_NO_IDX \
    vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4, \
    vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8, \
    vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12, \
    vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16, \
    srcUb, s2BaseSize

#define VREG_FLOAT_INPUT_X_NEW_8_DECL \
    RegTensor<float> &vreg_input_x9_new, RegTensor<float> &vreg_input_x10_new, \
    RegTensor<float> &vreg_input_x11_new, RegTensor<float> &vreg_input_x12_new, \
    RegTensor<float> &vreg_input_x13_new, RegTensor<float> &vreg_input_x14_new, \
    RegTensor<float> &vreg_input_x15_new, RegTensor<float> &vreg_input_x16_new

#define ARG_FLOAT_INPUT_X_NEW_8 \
    vreg_input_x9_new, vreg_input_x10_new, vreg_input_x11_new, vreg_input_x12_new, \
    vreg_input_x13_new, vreg_input_x14_new, vreg_input_x15_new, vreg_input_x16_new

#define VREG_FLOAT_PSE_16_DECL \
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2, \
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4, \
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6, \
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8, \
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10, \
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12, \
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14, \
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16

#define ARG_FLOAT_PSE_16 \
    vreg_pse1, vreg_pse2, vreg_pse3, vreg_pse4, \
    vreg_pse5, vreg_pse6, vreg_pse7, vreg_pse8, \
    vreg_pse9, vreg_pse10, vreg_pse11, vreg_pse12, \
    vreg_pse13, vreg_pse14, vreg_pse15, vreg_pse16

#define VREG_FLOAT_SEL_16_DECL \
    RegTensor<float> &vreg_sel1, RegTensor<float> &vreg_sel2, \
    RegTensor<float> &vreg_sel3, RegTensor<float> &vreg_sel4, \
    RegTensor<float> &vreg_sel5, RegTensor<float> &vreg_sel6, \
    RegTensor<float> &vreg_sel7, RegTensor<float> &vreg_sel8, \
    RegTensor<float> &vreg_sel9, RegTensor<float> &vreg_sel10, \
    RegTensor<float> &vreg_sel11, RegTensor<float> &vreg_sel12, \
    RegTensor<float> &vreg_sel13, RegTensor<float> &vreg_sel14, \
    RegTensor<float> &vreg_sel15, RegTensor<float> &vreg_sel16

#define ARG_FLOAT_SEL_16 \
    vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4, \
    vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8, \
    vreg_sel9, vreg_sel10, vreg_sel11, vreg_sel12, \
    vreg_sel13, vreg_sel14, vreg_sel15, vreg_sel16

#define VREG_FLOAT_SEL_NEW_8_DECL \
    RegTensor<float> &vreg_sel9_new, RegTensor<float> &vreg_sel10_new, \
    RegTensor<float> &vreg_sel11_new, RegTensor<float> &vreg_sel12_new, \
    RegTensor<float> &vreg_sel13_new, RegTensor<float> &vreg_sel14_new, \
    RegTensor<float> &vreg_sel15_new, RegTensor<float> &vreg_sel16_new

#define ARG_FLOAT_SEL_NEW_8 \
    vreg_sel9_new, vreg_sel10_new, vreg_sel11_new, vreg_sel12_new, \
    vreg_sel13_new, vreg_sel14_new, vreg_sel15_new, vreg_sel16_new

#define VREG_FLOAT_ALIBI_16_VREG_DECL \
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2, \
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4, \
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6, \
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8, \
    RegTensor<float> &vreg_alibi9, RegTensor<float> &vreg_alibi10, \
    RegTensor<float> &vreg_alibi11, RegTensor<float> &vreg_alibi12, \
    RegTensor<float> &vreg_alibi13, RegTensor<float> &vreg_alibi14, \
    RegTensor<float> &vreg_alibi15, RegTensor<float> &vreg_alibi16

#define ARG_FLOAT_ALIBI_16_VREG \
    vreg_alibi1, vreg_alibi2, vreg_alibi3, vreg_alibi4, \
    vreg_alibi5, vreg_alibi6, vreg_alibi7, vreg_alibi8, \
    vreg_alibi9, vreg_alibi10, vreg_alibi11, vreg_alibi12, \
    vreg_alibi13, vreg_alibi14, vreg_alibi15, vreg_alibi16

#define VREG_FLOAT_INPUT_X_16_VARS_DECL \
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2, \
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4, \
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6, \
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8, \
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10, \
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12, \
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14, \
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16

#define ARG_FLOAT_INPUT_X_16_VARS \
    vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4, \
    vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8, \
    vreg_input_x9, vreg_input_x10, vreg_input_x11, vreg_input_x12, \
    vreg_input_x13, vreg_input_x14, vreg_input_x15, vreg_input_x16

__simd_callee__ inline void ComputePseInnerMulAdd16(VREG_FLOAT_PSE_ALIBI_16_DECL)
{
    DO_ABS_16(vreg_pse, vreg_alibi, preg_all);
    DO_MULS_16(vreg_pse, slopes, preg_all);
    DO_ADDS_16(vreg_alibi, -1.0f, preg_all);
}

__simd_callee__ inline void ComputePseInnerMulAddSqrt16(VREG_FLOAT_PSE_ALIBI_16_DECL)
{
    DO_ABS_16(vreg_pse, vreg_alibi, preg_all);
    DO_SQRT_16(vreg_pse, vreg_pse, preg_all);
    DO_MULS_16(vreg_pse, slopes, preg_all);
    DO_ADDS_16(vreg_alibi, -1.0f, preg_all);
}

__simd_callee__ inline void LoadCastPseBf16_16(VREG_FLOAT_PSE_16_LOAD_DECL(bfloat16_t))
{
    RegTensor<bfloat16_t> vreg_pse_bf16_src1, vreg_pse_bf16_src2;
    RegTensor<bfloat16_t> vreg_pse_bf16_src3, vreg_pse_bf16_src4;
    RegTensor<bfloat16_t> vreg_pse_bf16_src5, vreg_pse_bf16_src6;
    RegTensor<bfloat16_t> vreg_pse_bf16_src7, vreg_pse_bf16_src8;
    RegTensor<bfloat16_t> vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse3_bf16, vreg_pse4_bf16;
    RegTensor<bfloat16_t> vreg_pse5_bf16, vreg_pse6_bf16, vreg_pse7_bf16, vreg_pse8_bf16;
    RegTensor<bfloat16_t> vreg_pse9_bf16, vreg_pse10_bf16, vreg_pse11_bf16, vreg_pse12_bf16;
    RegTensor<bfloat16_t> vreg_pse13_bf16, vreg_pse14_bf16, vreg_pse15_bf16, vreg_pse16_bf16;
    LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src4, pseUb + floatRepSize * 6 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src5, pseUb + floatRepSize * 8 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src6, pseUb + floatRepSize * 10 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src7, pseUb + floatRepSize * 12 + i * pseStride);
    LoadAlign(vreg_pse_bf16_src8, pseUb + floatRepSize * 14 + i * pseStride);
    DO_INTERLEAVE_PAIRS_8(vreg_pse, _bf16, vreg_pse_bf16_src);
    DO_CAST_16(float, bfloat16_t, castTraitZero, vreg_pse, vreg_pse, _bf16, preg_all_b16);
}

__simd_callee__ inline void LoadCastPseF16_16(VREG_FLOAT_PSE_16_LOAD_DECL(half))
{
    RegTensor<half> vreg_pse_f16_src1, vreg_pse_f16_src2;
    RegTensor<half> vreg_pse_f16_src3, vreg_pse_f16_src4;
    RegTensor<half> vreg_pse_f16_src5, vreg_pse_f16_src6;
    RegTensor<half> vreg_pse_f16_src7, vreg_pse_f16_src8;
    RegTensor<half> vreg_pse1_f16, vreg_pse2_f16, vreg_pse3_f16, vreg_pse4_f16;
    RegTensor<half> vreg_pse5_f16, vreg_pse6_f16, vreg_pse7_f16, vreg_pse8_f16;
    RegTensor<half> vreg_pse9_f16, vreg_pse10_f16, vreg_pse11_f16, vreg_pse12_f16;
    RegTensor<half> vreg_pse13_f16, vreg_pse14_f16, vreg_pse15_f16, vreg_pse16_f16;
    LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
    LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
    LoadAlign(vreg_pse_f16_src3, pseUb + floatRepSize * 4 + i * pseStride);
    LoadAlign(vreg_pse_f16_src4, pseUb + floatRepSize * 6 + i * pseStride);
    LoadAlign(vreg_pse_f16_src5, pseUb + floatRepSize * 8 + i * pseStride);
    LoadAlign(vreg_pse_f16_src6, pseUb + floatRepSize * 10 + i * pseStride);
    LoadAlign(vreg_pse_f16_src7, pseUb + floatRepSize * 12 + i * pseStride);
    LoadAlign(vreg_pse_f16_src8, pseUb + floatRepSize * 14 + i * pseStride);
    DO_INTERLEAVE_PAIRS_8(vreg_pse, _f16, vreg_pse_f16_src);
    DO_CAST_16(float, half, castTraitZero, vreg_pse, vreg_pse, _f16, preg_all_b16);
}

__simd_callee__ inline void MaxReduce16(
    RegTensor<float> &vreg_input_max,
    RegTensor<float> &vreg_a1, RegTensor<float> &vreg_a2,
    RegTensor<float> &vreg_a3, RegTensor<float> &vreg_a4,
    RegTensor<float> &vreg_a5, RegTensor<float> &vreg_a6,
    RegTensor<float> &vreg_a7, RegTensor<float> &vreg_a8,
    RegTensor<float> &vreg_b1, RegTensor<float> &vreg_b2,
    RegTensor<float> &vreg_b3, RegTensor<float> &vreg_b4,
    RegTensor<float> &vreg_b5, RegTensor<float> &vreg_b6,
    RegTensor<float> &vreg_b7, RegTensor<float> &vreg_b8,
    MaskReg &preg_all)
{
    RegTensor<float> vreg_max_tmp1;
    RegTensor<float> vreg_max_tmp2;
    RegTensor<float> vreg_max_tmp3;
    RegTensor<float> vreg_max_tmp4;
    RegTensor<float> vreg_max_tmp5;
    RegTensor<float> vreg_max_tmp6;
    RegTensor<float> vreg_max_tmp7;
    RegTensor<float> vreg_max_tmp8;
    Max(vreg_max_tmp1, vreg_a1, vreg_a2, preg_all);
    Max(vreg_max_tmp2, vreg_a3, vreg_a4, preg_all);
    Max(vreg_max_tmp3, vreg_a5, vreg_a6, preg_all);
    Max(vreg_max_tmp4, vreg_a7, vreg_a8, preg_all);
    Max(vreg_max_tmp5, vreg_b1, vreg_b2, preg_all);
    Max(vreg_max_tmp6, vreg_b3, vreg_b4, preg_all);
    Max(vreg_max_tmp7, vreg_b5, vreg_b6, preg_all);
    Max(vreg_max_tmp8, vreg_b7, vreg_b8, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp2, preg_all);
    Max(vreg_max_tmp3, vreg_max_tmp3, vreg_max_tmp4, preg_all);
    Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp6, preg_all);
    Max(vreg_max_tmp7, vreg_max_tmp7, vreg_max_tmp8, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp3, preg_all);
    Max(vreg_max_tmp5, vreg_max_tmp5, vreg_max_tmp7, preg_all);
    Max(vreg_max_tmp1, vreg_max_tmp1, vreg_max_tmp5, preg_all);
    Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_input_max, vreg_max_tmp1, preg_all);
}

template<typename T, typename T2>
__simd_callee__ inline void ExpSumReduceStore1024(
    RegTensor<float> &vreg_exp_sum1,
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    RegTensor<float> &vreg_exp_even5, RegTensor<float> &vreg_exp_odd5,
    RegTensor<float> &vreg_exp_even6, RegTensor<float> &vreg_exp_odd6,
    RegTensor<float> &vreg_exp_even7, RegTensor<float> &vreg_exp_odd7,
    RegTensor<float> &vreg_exp_even8, RegTensor<float> &vreg_exp_odd8,
    UnalignRegForStore &ureg_exp_sum, __ubuf__ T *&expSumUb,
    MaskReg &preg_all)
{
    RegTensor<float> vreg_exp_sum2, vreg_exp_sum3, vreg_exp_sum4;
    RegTensor<float> vreg_exp_sum5, vreg_exp_sum6, vreg_exp_sum7, vreg_exp_sum8;
    Add(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all);
    Add(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all);
    Add(vreg_exp_sum3, vreg_exp_even3, vreg_exp_odd3, preg_all);
    Add(vreg_exp_sum4, vreg_exp_even4, vreg_exp_odd4, preg_all);
    Add(vreg_exp_sum5, vreg_exp_even5, vreg_exp_odd5, preg_all);
    Add(vreg_exp_sum6, vreg_exp_even6, vreg_exp_odd6, preg_all);
    Add(vreg_exp_sum7, vreg_exp_even7, vreg_exp_odd7, preg_all);
    Add(vreg_exp_sum8, vreg_exp_even8, vreg_exp_odd8, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum2, preg_all);
    Add(vreg_exp_sum3, vreg_exp_sum3, vreg_exp_sum4, preg_all);
    Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum6, preg_all);
    Add(vreg_exp_sum7, vreg_exp_sum7, vreg_exp_sum8, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum3, preg_all);
    Add(vreg_exp_sum5, vreg_exp_sum5, vreg_exp_sum7, preg_all);
    Add(vreg_exp_sum1, vreg_exp_sum1, vreg_exp_sum5, preg_all);
    Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        vreg_exp_sum1, vreg_exp_sum1, preg_all);
    StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), vreg_exp_sum1, ureg_exp_sum, 1);
}

template<typename T2>
__simd_callee__ inline void CastStoreExpBf16_1024(VREG_FLOAT_CAST_STORE_EXP_1024_DECL)
{
    RegTensor<bfloat16_t> vreg_exp_even1_bf16, vreg_exp_odd1_bf16;
    RegTensor<bfloat16_t> vreg_exp_even2_bf16, vreg_exp_odd2_bf16;
    RegTensor<bfloat16_t> vreg_exp_even3_bf16, vreg_exp_odd3_bf16;
    RegTensor<bfloat16_t> vreg_exp_even4_bf16, vreg_exp_odd4_bf16;
    RegTensor<bfloat16_t> vreg_exp_even5_bf16, vreg_exp_odd5_bf16;
    RegTensor<bfloat16_t> vreg_exp_even6_bf16, vreg_exp_odd6_bf16;
    RegTensor<bfloat16_t> vreg_exp_even7_bf16, vreg_exp_odd7_bf16;
    RegTensor<bfloat16_t> vreg_exp_even8_bf16, vreg_exp_odd8_bf16;
    RegTensor<bfloat16_t> vreg_exp1_bf16, vreg_exp2_bf16, vreg_exp3_bf16, vreg_exp4_bf16;
    RegTensor<bfloat16_t> vreg_exp5_bf16, vreg_exp6_bf16, vreg_exp7_bf16, vreg_exp8_bf16;
    DO_CAST_EVEN_ODD_8(T2, float, vreg_exp_even, vreg_exp_odd, _bf16, preg_all);
    DO_OR_8(vreg_exp, vreg_exp_even, vreg_exp_odd, _bf16, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb3), vreg_exp3_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb4), vreg_exp4_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb5), vreg_exp5_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb6), vreg_exp6_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb7), vreg_exp7_bf16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb8), vreg_exp8_bf16, blockStride, repeatStride, storeMask);
}

template<typename T2>
__simd_callee__ inline void CastStoreExpF16_1024(VREG_FLOAT_CAST_STORE_EXP_1024_DECL)
{
    RegTensor<half> vreg_exp_even1_f16, vreg_exp_odd1_f16;
    RegTensor<half> vreg_exp_even2_f16, vreg_exp_odd2_f16;
    RegTensor<half> vreg_exp_even3_f16, vreg_exp_odd3_f16;
    RegTensor<half> vreg_exp_even4_f16, vreg_exp_odd4_f16;
    RegTensor<half> vreg_exp_even5_f16, vreg_exp_odd5_f16;
    RegTensor<half> vreg_exp_even6_f16, vreg_exp_odd6_f16;
    RegTensor<half> vreg_exp_even7_f16, vreg_exp_odd7_f16;
    RegTensor<half> vreg_exp_even8_f16, vreg_exp_odd8_f16;
    RegTensor<half> vreg_exp1_f16, vreg_exp2_f16, vreg_exp3_f16, vreg_exp4_f16;
    RegTensor<half> vreg_exp5_f16, vreg_exp6_f16, vreg_exp7_f16, vreg_exp8_f16;
    DO_CAST_EVEN_ODD_8(T2, float, vreg_exp_even, vreg_exp_odd, _f16, preg_all);
    DO_OR_8(vreg_exp, vreg_exp_even, vreg_exp_odd, _f16, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb3), vreg_exp3_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb4), vreg_exp4_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb5), vreg_exp5_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb6), vreg_exp6_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb7), vreg_exp7_f16, blockStride, repeatStride, storeMask);
    StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T2 *&)expUb8), vreg_exp8_f16, blockStride, repeatStride, storeMask);
}

__simd_callee__ inline void InitAlibiOffsets(
    RegTensor<float> &vreg_alibi1, RegTensor<float> &vreg_alibi2,
    RegTensor<float> &vreg_alibi3, RegTensor<float> &vreg_alibi4,
    RegTensor<float> &vreg_alibi5, RegTensor<float> &vreg_alibi6,
    RegTensor<float> &vreg_alibi7, RegTensor<float> &vreg_alibi8,
    RegTensor<float> &vreg_alibi9, RegTensor<float> &vreg_alibi10,
    RegTensor<float> &vreg_alibi11, RegTensor<float> &vreg_alibi12,
    RegTensor<float> &vreg_alibi13, RegTensor<float> &vreg_alibi14,
    RegTensor<float> &vreg_alibi15, RegTensor<float> &vreg_alibi16,
    const float posShift, MaskReg &preg_all)
{
    Arange(vreg_alibi1, posShift);
    Adds(vreg_alibi2, vreg_alibi1, floatRepSize, preg_all);
    Adds(vreg_alibi3, vreg_alibi2, floatRepSize, preg_all);
    Adds(vreg_alibi4, vreg_alibi3, floatRepSize, preg_all);
    Adds(vreg_alibi5, vreg_alibi4, floatRepSize, preg_all);
    Adds(vreg_alibi6, vreg_alibi5, floatRepSize, preg_all);
    Adds(vreg_alibi7, vreg_alibi6, floatRepSize, preg_all);
    Adds(vreg_alibi8, vreg_alibi7, floatRepSize, preg_all);
    Adds(vreg_alibi9, vreg_alibi8, floatRepSize, preg_all);
    Adds(vreg_alibi10, vreg_alibi9, floatRepSize, preg_all);
    Adds(vreg_alibi11, vreg_alibi10, floatRepSize, preg_all);
    Adds(vreg_alibi12, vreg_alibi11, floatRepSize, preg_all);
    Adds(vreg_alibi13, vreg_alibi12, floatRepSize, preg_all);
    Adds(vreg_alibi14, vreg_alibi13, floatRepSize, preg_all);
    Adds(vreg_alibi15, vreg_alibi14, floatRepSize, preg_all);
    Adds(vreg_alibi16, vreg_alibi15, floatRepSize, preg_all);
}

template<typename T>
__simd_callee__ inline void LoadInput16(VREG_FLOAT_INPUT_X_16_DECL)
{
    DO_LOADALIGN_16(vreg_input_x, srcUb, s2BaseSize, i);
}

__simd_callee__ inline void ScaleInput16(
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2,
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4,
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10,
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12,
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14,
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16,
    const float scale, MaskReg &preg_all,
    MaskReg &preg_ori_tail_n1, MaskReg &preg_ori_tail_n2,
    MaskReg &preg_ori_tail_n3, MaskReg &preg_ori_tail_n4,
    MaskReg &preg_ori_tail_n5, MaskReg &preg_ori_tail_n6,
    MaskReg &preg_ori_tail_n7, MaskReg &preg_ori_tail_n8)
{
    Muls(vreg_input_x1, vreg_input_x1, scale, preg_all);
    Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
    Muls(vreg_input_x3, vreg_input_x3, scale, preg_all);
    Muls(vreg_input_x4, vreg_input_x4, scale, preg_all);
    Muls(vreg_input_x5, vreg_input_x5, scale, preg_all);
    Muls(vreg_input_x6, vreg_input_x6, scale, preg_all);
    Muls(vreg_input_x7, vreg_input_x7, scale, preg_all);
    Muls(vreg_input_x8, vreg_input_x8, scale, preg_all);
    Muls(vreg_input_x9, vreg_input_x9, scale, preg_ori_tail_n1);
    Muls(vreg_input_x10, vreg_input_x10, scale, preg_ori_tail_n2);
    Muls(vreg_input_x11, vreg_input_x11, scale, preg_ori_tail_n3);
    Muls(vreg_input_x12, vreg_input_x12, scale, preg_ori_tail_n4);
    Muls(vreg_input_x13, vreg_input_x13, scale, preg_ori_tail_n5);
    Muls(vreg_input_x14, vreg_input_x14, scale, preg_ori_tail_n6);
    Muls(vreg_input_x15, vreg_input_x15, scale, preg_ori_tail_n7);
    Muls(vreg_input_x16, vreg_input_x16, scale, preg_ori_tail_n8);
}

__simd_callee__ inline void AddPseToInput16(
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2,
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4,
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10,
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12,
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14,
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16,
    RegTensor<float> &vreg_pse1, RegTensor<float> &vreg_pse2,
    RegTensor<float> &vreg_pse3, RegTensor<float> &vreg_pse4,
    RegTensor<float> &vreg_pse5, RegTensor<float> &vreg_pse6,
    RegTensor<float> &vreg_pse7, RegTensor<float> &vreg_pse8,
    RegTensor<float> &vreg_pse9, RegTensor<float> &vreg_pse10,
    RegTensor<float> &vreg_pse11, RegTensor<float> &vreg_pse12,
    RegTensor<float> &vreg_pse13, RegTensor<float> &vreg_pse14,
    RegTensor<float> &vreg_pse15, RegTensor<float> &vreg_pse16,
    MaskReg &preg_all, MaskReg &preg_ori_tail_n1, MaskReg &preg_ori_tail_n2,
    MaskReg &preg_ori_tail_n3, MaskReg &preg_ori_tail_n4,
    MaskReg &preg_ori_tail_n5, MaskReg &preg_ori_tail_n6,
    MaskReg &preg_ori_tail_n7, MaskReg &preg_ori_tail_n8)
{
    Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
    Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
    Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_all);
    Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_all);
    Add(vreg_input_x5, vreg_input_x5, vreg_pse5, preg_all);
    Add(vreg_input_x6, vreg_input_x6, vreg_pse6, preg_all);
    Add(vreg_input_x7, vreg_input_x7, vreg_pse7, preg_all);
    Add(vreg_input_x8, vreg_input_x8, vreg_pse8, preg_all);
    Add(vreg_input_x9, vreg_input_x9, vreg_pse9, preg_ori_tail_n1);
    Add(vreg_input_x10, vreg_input_x10, vreg_pse10, preg_ori_tail_n2);
    Add(vreg_input_x11, vreg_input_x11, vreg_pse11, preg_ori_tail_n3);
    Add(vreg_input_x12, vreg_input_x12, vreg_pse12, preg_ori_tail_n4);
    Add(vreg_input_x13, vreg_input_x13, vreg_pse13, preg_ori_tail_n5);
    Add(vreg_input_x14, vreg_input_x14, vreg_pse14, preg_ori_tail_n6);
    Add(vreg_input_x15, vreg_input_x15, vreg_pse15, preg_ori_tail_n7);
    Add(vreg_input_x16, vreg_input_x16, vreg_pse16, preg_ori_tail_n8);
}

__simd_callee__ inline void LoadAttenMask16(
    MaskReg &preg_compare1, MaskReg &preg_compare2,
    MaskReg &preg_compare3, MaskReg &preg_compare4,
    MaskReg &preg_compare5, MaskReg &preg_compare6,
    MaskReg &preg_compare7, MaskReg &preg_compare8,
    MaskReg &preg_compare9, MaskReg &preg_compare10,
    MaskReg &preg_compare11, MaskReg &preg_compare12,
    MaskReg &preg_compare13, MaskReg &preg_compare14,
    MaskReg &preg_compare15, MaskReg &preg_compare16,
    VREG_FLOAT_MASKUB_REF_16_DECL,
    const uint32_t nPadding)
{
    DO_LOADALIGN_MASK_16(preg_compare, maskUb, nPadding);
}

__simd_callee__ inline void ApplyAttenMaskSelect16(
    RegTensor<float> &vreg_sel1, RegTensor<float> &vreg_sel2,
    RegTensor<float> &vreg_sel3, RegTensor<float> &vreg_sel4,
    RegTensor<float> &vreg_sel5, RegTensor<float> &vreg_sel6,
    RegTensor<float> &vreg_sel7, RegTensor<float> &vreg_sel8,
    RegTensor<float> &vreg_sel9, RegTensor<float> &vreg_sel10,
    RegTensor<float> &vreg_sel11, RegTensor<float> &vreg_sel12,
    RegTensor<float> &vreg_sel13, RegTensor<float> &vreg_sel14,
    RegTensor<float> &vreg_sel15, RegTensor<float> &vreg_sel16,
    RegTensor<float> &vreg_min,
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2,
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4,
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10,
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12,
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14,
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16,
    MaskReg &preg_compare1, MaskReg &preg_compare2,
    MaskReg &preg_compare3, MaskReg &preg_compare4,
    MaskReg &preg_compare5, MaskReg &preg_compare6,
    MaskReg &preg_compare7, MaskReg &preg_compare8,
    MaskReg &preg_compare9, MaskReg &preg_compare10,
    MaskReg &preg_compare11, MaskReg &preg_compare12,
    MaskReg &preg_compare13, MaskReg &preg_compare14,
    MaskReg &preg_compare15, MaskReg &preg_compare16)
{
    DO_SELECT_16(vreg_sel, vreg_min, vreg_input_x, preg_compare);
}

template<typename T>
__simd_callee__ inline void StoreAlign16(
    RegTensor<float> &vreg_val1, RegTensor<float> &vreg_val2,
    RegTensor<float> &vreg_val3, RegTensor<float> &vreg_val4,
    RegTensor<float> &vreg_val5, RegTensor<float> &vreg_val6,
    RegTensor<float> &vreg_val7, RegTensor<float> &vreg_val8,
    RegTensor<float> &vreg_val9, RegTensor<float> &vreg_val10,
    RegTensor<float> &vreg_val11, RegTensor<float> &vreg_val12,
    RegTensor<float> &vreg_val13, RegTensor<float> &vreg_val14,
    RegTensor<float> &vreg_val15, RegTensor<float> &vreg_val16,
    __ubuf__ T *&srcUb, const uint16_t i, const uint32_t s2BaseSize,
    MaskReg &preg_all)
{
    DO_STOREALIGN_16(T, srcUb, vreg_val, s2BaseSize, i, preg_all);
}

template<typename T>
__simd_callee__ inline void LoadInputDinterleave8(VREG_FLOAT_INPUT_X_16_DECL)
{
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x1, vreg_input_x2, srcUb + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x3, vreg_input_x4, srcUb + floatRepSize * 2 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x5, vreg_input_x6, srcUb + floatRepSize * 4 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x7, vreg_input_x8, srcUb + floatRepSize * 6 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x9, vreg_input_x10, srcUb + floatRepSize * 8 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x11, vreg_input_x12, srcUb + floatRepSize * 10 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x13, vreg_input_x14, srcUb + floatRepSize * 12 + i * s2BaseSize);
    LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
        vreg_input_x15, vreg_input_x16, srcUb + floatRepSize * 14 + i * s2BaseSize);
}

__simd_callee__ inline void ExpSub16(
    RegTensor<float> &vreg_exp_even1, RegTensor<float> &vreg_exp_odd1,
    RegTensor<float> &vreg_exp_even2, RegTensor<float> &vreg_exp_odd2,
    RegTensor<float> &vreg_exp_even3, RegTensor<float> &vreg_exp_odd3,
    RegTensor<float> &vreg_exp_even4, RegTensor<float> &vreg_exp_odd4,
    RegTensor<float> &vreg_exp_even5, RegTensor<float> &vreg_exp_odd5,
    RegTensor<float> &vreg_exp_even6, RegTensor<float> &vreg_exp_odd6,
    RegTensor<float> &vreg_exp_even7, RegTensor<float> &vreg_exp_odd7,
    RegTensor<float> &vreg_exp_even8, RegTensor<float> &vreg_exp_odd8,
    RegTensor<float> &vreg_input_x1, RegTensor<float> &vreg_input_x2,
    RegTensor<float> &vreg_input_x3, RegTensor<float> &vreg_input_x4,
    RegTensor<float> &vreg_input_x5, RegTensor<float> &vreg_input_x6,
    RegTensor<float> &vreg_input_x7, RegTensor<float> &vreg_input_x8,
    RegTensor<float> &vreg_input_x9, RegTensor<float> &vreg_input_x10,
    RegTensor<float> &vreg_input_x11, RegTensor<float> &vreg_input_x12,
    RegTensor<float> &vreg_input_x13, RegTensor<float> &vreg_input_x14,
    RegTensor<float> &vreg_input_x15, RegTensor<float> &vreg_input_x16,
    RegTensor<float> &vreg_max, MaskReg &preg_all)
{
    DO_EXPSUB_PAIRS_8(vreg_exp_even, vreg_exp_odd, vreg_input_x, vreg_max, preg_all);
}

template<typename T, typename T2, bool hasAtten, PseTypeEnum pseMode>
__simd_callee__ inline void ProcessVec1MainLoop1024(
    VREG_FLOAT_INPUT_X_16_NO_IDX_DECL,
    VREG_FLOAT_INPUT_X_NEW_8_DECL,
    VREG_FLOAT_PSE_16_DECL,
    VREG_FLOAT_SEL_16_DECL,
    VREG_FLOAT_SEL_NEW_8_DECL,
    VREG_FLOAT_ALIBI_16_VREG_DECL,
    RegTensor<float> &vreg_input_max,
    MaskReg &preg_all, MaskReg &preg_all_b16,
    MaskReg &preg_ori_tail_n1, MaskReg &preg_ori_tail_n2,
    MaskReg &preg_ori_tail_n3, MaskReg &preg_ori_tail_n4,
    MaskReg &preg_ori_tail_n5, MaskReg &preg_ori_tail_n6,
    MaskReg &preg_ori_tail_n7, MaskReg &preg_ori_tail_n8,
    __ubuf__ T *&maxBuf,
    __ubuf__ T2 *&pseUb,
    VREG_FLOAT_MASKUB_REF_16_DECL,
    const float minValue, const uint16_t m,
    const float slopes, const float scale, const float posShift,
    const uint32_t pseStride, const uint32_t nPadding,
    UnalignRegForStore &ureg_max)
{
    RegTensor<float> vreg_min;

    VREG_PREG_COMPARE_DECL_16();

    Duplicate(vreg_min, minValue);
    if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                    pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        InitAlibiOffsets(ARG_FLOAT_ALIBI_16_VREG, posShift, preg_all);
    }
    for (uint16_t i = 0; i < m; ++i) {
        LoadInput16<T>(ARG_FLOAT_INPUT_X_16);

        if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput16(ARG_FLOAT_INPUT_X_16_VARS, scale, preg_all,
                preg_ori_tail_n1, preg_ori_tail_n2,
                preg_ori_tail_n3, preg_ori_tail_n4,
                preg_ori_tail_n5, preg_ori_tail_n6,
                preg_ori_tail_n7, preg_ori_tail_n8);
        }
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                    ComputePseInnerMulAddSqrt16(ARG_FLOAT_PSE_ALIBI_16);
                } else {
                    ComputePseInnerMulAdd16(ARG_FLOAT_PSE_ALIBI_16);
                }
            } else {
                if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    LoadCastPseBf16_16(ARG_FLOAT_PSE_16_LOAD);
                } else if constexpr (IsSameType<T2, half>::value) {
                    LoadCastPseF16_16(ARG_FLOAT_PSE_16_LOAD);
                }
            }
            AddPseToInput16(ARG_FLOAT_INPUT_X_16_VARS, ARG_FLOAT_PSE_16,
                preg_all, preg_ori_tail_n1, preg_ori_tail_n2,
                preg_ori_tail_n3, preg_ori_tail_n4,
                preg_ori_tail_n5, preg_ori_tail_n6,
                preg_ori_tail_n7, preg_ori_tail_n8);
        }
        if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            ScaleInput16(ARG_FLOAT_INPUT_X_16_VARS, scale, preg_all,
                preg_ori_tail_n1, preg_ori_tail_n2,
                preg_ori_tail_n3, preg_ori_tail_n4,
                preg_ori_tail_n5, preg_ori_tail_n6,
                preg_ori_tail_n7, preg_ori_tail_n8);
        }

        if constexpr (hasAtten == 1) {
            LoadAttenMask16(preg_compare1, preg_compare2, preg_compare3, preg_compare4,
                preg_compare5, preg_compare6, preg_compare7, preg_compare8,
                preg_compare9, preg_compare10, preg_compare11, preg_compare12,
                preg_compare13, preg_compare14, preg_compare15, preg_compare16,
                maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6, maskUb7, maskUb8,
                maskUb9, maskUb10, maskUb11, maskUb12, maskUb13, maskUb14, maskUb15, maskUb16,
                nPadding);
            ApplyAttenMaskSelect16(ARG_FLOAT_SEL_16, vreg_min, ARG_FLOAT_INPUT_X_16_VARS,
                preg_compare1, preg_compare2, preg_compare3, preg_compare4,
                preg_compare5, preg_compare6, preg_compare7, preg_compare8,
                preg_compare9, preg_compare10, preg_compare11, preg_compare12,
                preg_compare13, preg_compare14, preg_compare15, preg_compare16);

            Select(vreg_sel9_new, vreg_sel9, vreg_min, preg_ori_tail_n1);
            Select(vreg_sel10_new, vreg_sel10, vreg_min, preg_ori_tail_n2);
            Select(vreg_sel11_new, vreg_sel11, vreg_min, preg_ori_tail_n3);
            Select(vreg_sel12_new, vreg_sel12, vreg_min, preg_ori_tail_n4);
            Select(vreg_sel13_new, vreg_sel13, vreg_min, preg_ori_tail_n5);
            Select(vreg_sel14_new, vreg_sel14, vreg_min, preg_ori_tail_n6);
            Select(vreg_sel15_new, vreg_sel15, vreg_min, preg_ori_tail_n7);
            Select(vreg_sel16_new, vreg_sel16, vreg_min, preg_ori_tail_n8);

            StoreAlign16<T>(vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4,
                vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8,
                vreg_sel9_new, vreg_sel10_new, vreg_sel11_new, vreg_sel12_new,
                vreg_sel13_new, vreg_sel14_new, vreg_sel15_new, vreg_sel16_new,
                srcUb, i, s2BaseSize, preg_all);

            MaxReduce16(vreg_input_max,
                vreg_sel1, vreg_sel2, vreg_sel3, vreg_sel4,
                vreg_sel5, vreg_sel6, vreg_sel7, vreg_sel8,
                vreg_sel9_new, vreg_sel10_new, vreg_sel11_new, vreg_sel12_new,
                vreg_sel13_new, vreg_sel14_new, vreg_sel15_new, vreg_sel16_new,
                preg_all);
        } else {
            Select(vreg_input_x9_new, vreg_input_x9, vreg_min, preg_ori_tail_n1);
            Select(vreg_input_x10_new, vreg_input_x10, vreg_min, preg_ori_tail_n2);
            Select(vreg_input_x11_new, vreg_input_x11, vreg_min, preg_ori_tail_n3);
            Select(vreg_input_x12_new, vreg_input_x12, vreg_min, preg_ori_tail_n4);
            Select(vreg_input_x13_new, vreg_input_x13, vreg_min, preg_ori_tail_n5);
            Select(vreg_input_x14_new, vreg_input_x14, vreg_min, preg_ori_tail_n6);
            Select(vreg_input_x15_new, vreg_input_x15, vreg_min, preg_ori_tail_n7);
            Select(vreg_input_x16_new, vreg_input_x16, vreg_min, preg_ori_tail_n8);

            StoreAlign16<T>(vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9_new, vreg_input_x10_new, vreg_input_x11_new, vreg_input_x12_new,
                vreg_input_x13_new, vreg_input_x14_new, vreg_input_x15_new, vreg_input_x16_new,
                srcUb, i, s2BaseSize, preg_all);

            MaxReduce16(vreg_input_max,
                vreg_input_x1, vreg_input_x2, vreg_input_x3, vreg_input_x4,
                vreg_input_x5, vreg_input_x6, vreg_input_x7, vreg_input_x8,
                vreg_input_x9_new, vreg_input_x10_new, vreg_input_x11_new, vreg_input_x12_new,
                vreg_input_x13_new, vreg_input_x14_new, vreg_input_x15_new, vreg_input_x16_new,
                preg_all);
        }
        StoreUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxBuf), vreg_input_max, ureg_max, 1);
    }
}

} // namespace FaVectorApi

#endif // VF_BASIC_BLOCK_UNALIGNED1024_COMMON_H
