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
 * \file vf_basic_block_utils.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UTILS_H
#define VF_BASIC_BLOCK_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace FaVectorApi {
constexpr uint32_t floatRepSize = 64;
constexpr uint32_t halfRepSize = 128;
constexpr uint32_t blockBytesU8 = 32;
constexpr float fp8e4m3MaxValue = 448.0f;
constexpr float int8MaxValue = 127.0f;
constexpr float hifp8MaxValue = 32768.0f;
constexpr float floatEps = 2.220446049250313e-16;
constexpr float LN2 = static_cast<float>(0.6931471806f);
constexpr float INV_LN2 = static_cast<float>(1.4426950409F);
/* **************************************************************************************************
 * Muls + Select(optional) + SoftmaxFlashV2 + Cast(fp32->fp16/bf16) + ND2NZ
 * ************************************************************************************************* */
using namespace MicroAPI;

constexpr static AscendC::MicroAPI::CastTrait castTraitNoneZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};
 
constexpr static AscendC::MicroAPI::CastTrait castTraitRintZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
 
constexpr static AscendC::MicroAPI::CastTrait castTraitRintTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitRintThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

#define USE_MLA_FULLQUANT_V1_P(vreg_exp, vreg_rowmax_p, MaskReg)    \
    do {                                                            \
        Muls(vreg_exp, vreg_exp, fp8e4m3MaxValue, MaskReg);         \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg);            \
    } while (0)
    
#define USE_MLA_FULLQUANT_V1_P_INT8(vreg_exp, vreg_rowmax_p, MaskReg) \
    do {                                                              \
        Muls(vreg_exp, vreg_exp, int8MaxValue, MaskReg);              \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg);              \
    } while (0)

#define USE_MLA_FULLQUANT_V1_P_HIFP8(vreg_exp, vreg_rowmax_p, MaskReg) \
    do {                                                               \
        Muls(vreg_exp, vreg_exp, hifp8MaxValue, MaskReg);              \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg);               \
    } while (0)
/* **************************************************************************************************
 * Abs operation: Abs(dst_i, src_i, mask)
 * ************************************************************************************************* */
#define DO_ABS_2(dst, src, mask)                                  \
    do {                                                          \
        Abs(dst##1, src##1, mask); Abs(dst##2, src##2, mask);     \
    } while (0)

#define DO_ABS_4(dst, src, mask)                                  \
    do {                                                          \
        DO_ABS_2(dst, src, mask);                                 \
        Abs(dst##3, src##3, mask); Abs(dst##4, src##4, mask);     \
    } while (0)

#define DO_ABS_6(dst, src, mask)                                  \
    do {                                                          \
        DO_ABS_4(dst, src, mask);                                 \
        Abs(dst##5, src##5, mask); Abs(dst##6, src##6, mask);     \
    } while (0)

#define DO_ABS_8(dst, src, mask)                                  \
    do {                                                          \
        DO_ABS_6(dst, src, mask);                                 \
        Abs(dst##7, src##7, mask); Abs(dst##8, src##8, mask);     \
    } while (0)

#define DO_ABS_16(dst, src, mask)                                 \
    do {                                                          \
        DO_ABS_8(dst, src, mask);                                 \
        Abs(dst##9, src##9, mask);   Abs(dst##10, src##10, mask); \
        Abs(dst##11, src##11, mask); Abs(dst##12, src##12, mask); \
        Abs(dst##13, src##13, mask); Abs(dst##14, src##14, mask); \
        Abs(dst##15, src##15, mask); Abs(dst##16, src##16, mask); \
    } while (0)

/* **************************************************************************************************
 * Sqrt operation: Sqrt(dst_i, src_i, mask)
 * ************************************************************************************************* */
#define DO_SQRT_2(dst, src, mask)                                 \
    do {                                                          \
        Sqrt(dst##1, src##1, mask); Sqrt(dst##2, src##2, mask);   \
    } while (0)

#define DO_SQRT_4(dst, src, mask)                                 \
    do {                                                          \
        DO_SQRT_2(dst, src, mask);                                \
        Sqrt(dst##3, src##3, mask); Sqrt(dst##4, src##4, mask);   \
    } while (0)

#define DO_SQRT_6(dst, src, mask)                                 \
    do {                                                          \
        DO_SQRT_4(dst, src, mask);                                \
        Sqrt(dst##5, src##5, mask); Sqrt(dst##6, src##6, mask);   \
    } while (0)

#define DO_SQRT_8(dst, src, mask)                                 \
    do {                                                          \
        DO_SQRT_6(dst, src, mask);                                \
        Sqrt(dst##7, src##7, mask); Sqrt(dst##8, src##8, mask);   \
    } while (0)

#define DO_SQRT_16(dst, src, mask)                                  \
    do {                                                            \
        DO_SQRT_8(dst, src, mask);                                  \
        Sqrt(dst##9, src##9, mask);   Sqrt(dst##10, src##10, mask); \
        Sqrt(dst##11, src##11, mask); Sqrt(dst##12, src##12, mask); \
        Sqrt(dst##13, src##13, mask); Sqrt(dst##14, src##14, mask); \
        Sqrt(dst##15, src##15, mask); Sqrt(dst##16, src##16, mask); \
    } while (0)

/* **************************************************************************************************
 * Muls operation: Muls(dst_i, dst_i, val, mask)
 * ************************************************************************************************* */
#define DO_MULS_2(dst, val, mask)                                         \
    do {                                                                  \
        Muls(dst##1, dst##1, val, mask); Muls(dst##2, dst##2, val, mask); \
    } while (0)

#define DO_MULS_4(dst, val, mask)                                         \
    do {                                                                  \
        DO_MULS_2(dst, val, mask);                                        \
        Muls(dst##3, dst##3, val, mask); Muls(dst##4, dst##4, val, mask); \
    } while (0)

#define DO_MULS_6(dst, val, mask)                                         \
    do {                                                                  \
        DO_MULS_4(dst, val, mask);                                        \
        Muls(dst##5, dst##5, val, mask); Muls(dst##6, dst##6, val, mask); \
    } while (0)

#define DO_MULS_8(dst, val, mask)                                         \
    do {                                                                  \
        DO_MULS_6(dst, val, mask);                                        \
        Muls(dst##7, dst##7, val, mask); Muls(dst##8, dst##8, val, mask); \
    } while (0)

#define DO_MULS_16(dst, val, mask)                                            \
    do {                                                                      \
        DO_MULS_8(dst, val, mask);                                            \
        Muls(dst##9, dst##9, val, mask);   Muls(dst##10, dst##10, val, mask); \
        Muls(dst##11, dst##11, val, mask); Muls(dst##12, dst##12, val, mask); \
        Muls(dst##13, dst##13, val, mask); Muls(dst##14, dst##14, val, mask); \
        Muls(dst##15, dst##15, val, mask); Muls(dst##16, dst##16, val, mask); \
    } while (0)

/* **************************************************************************************************
 * Adds operation: Adds(dst_i, dst_i, val, mask)
 * ************************************************************************************************* */
#define DO_ADDS_2(dst, val, mask)                                         \
    do {                                                                  \
        Adds(dst##1, dst##1, val, mask); Adds(dst##2, dst##2, val, mask); \
    } while (0)

#define DO_ADDS_4(dst, val, mask)                                         \
    do {                                                                  \
        DO_ADDS_2(dst, val, mask);                                        \
        Adds(dst##3, dst##3, val, mask); Adds(dst##4, dst##4, val, mask); \
    } while (0)

#define DO_ADDS_6(dst, val, mask)                                         \
    do {                                                                  \
        DO_ADDS_4(dst, val, mask);                                        \
        Adds(dst##5, dst##5, val, mask); Adds(dst##6, dst##6, val, mask); \
    } while (0)

#define DO_ADDS_8(dst, val, mask)                                         \
    do {                                                                  \
        DO_ADDS_6(dst, val, mask);                                        \
        Adds(dst##7, dst##7, val, mask); Adds(dst##8, dst##8, val, mask); \
    } while (0)

#define DO_ADDS_16(dst, val, mask)                                            \
    do {                                                                      \
        DO_ADDS_8(dst, val, mask);                                            \
        Adds(dst##9, dst##9, val, mask);   Adds(dst##10, dst##10, val, mask); \
        Adds(dst##11, dst##11, val, mask); Adds(dst##12, dst##12, val, mask); \
        Adds(dst##13, dst##13, val, mask); Adds(dst##14, dst##14, val, mask); \
        Adds(dst##15, dst##15, val, mask); Adds(dst##16, dst##16, val, mask); \
    } while (0)

/* **************************************************************************************************
 * Cast operation: Cast<DstT, SrcT, trait>(dst_i, src_i_sfx, mask)
 * ************************************************************************************************* */
#define DO_CAST_2(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask)     \
    do {                                                                  \
        Cast<DstT, SrcT, trait>(dst_pre##1, src_pre##1##src_sfx, mask);   \
        Cast<DstT, SrcT, trait>(dst_pre##2, src_pre##2##src_sfx, mask);   \
    } while (0)

#define DO_CAST_4(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask)     \
    do {                                                                  \
        DO_CAST_2(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask);    \
        Cast<DstT, SrcT, trait>(dst_pre##3, src_pre##3##src_sfx, mask);   \
        Cast<DstT, SrcT, trait>(dst_pre##4, src_pre##4##src_sfx, mask);   \
    } while (0)

#define DO_CAST_6(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask)     \
    do {                                                                  \
        DO_CAST_4(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask);    \
        Cast<DstT, SrcT, trait>(dst_pre##5, src_pre##5##src_sfx, mask);   \
        Cast<DstT, SrcT, trait>(dst_pre##6, src_pre##6##src_sfx, mask);   \
    } while (0)

#define DO_CAST_8(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask)     \
    do {                                                                  \
        DO_CAST_6(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask);    \
        Cast<DstT, SrcT, trait>(dst_pre##7, src_pre##7##src_sfx, mask);   \
        Cast<DstT, SrcT, trait>(dst_pre##8, src_pre##8##src_sfx, mask);   \
    } while (0)

#define DO_CAST_16(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask)    \
    do {                                                                  \
        DO_CAST_8(DstT, SrcT, trait, dst_pre, src_pre, src_sfx, mask);    \
        Cast<DstT, SrcT, trait>(dst_pre##9, src_pre##9##src_sfx, mask);   \
        Cast<DstT, SrcT, trait>(dst_pre##10, src_pre##10##src_sfx, mask); \
        Cast<DstT, SrcT, trait>(dst_pre##11, src_pre##11##src_sfx, mask); \
        Cast<DstT, SrcT, trait>(dst_pre##12, src_pre##12##src_sfx, mask); \
        Cast<DstT, SrcT, trait>(dst_pre##13, src_pre##13##src_sfx, mask); \
        Cast<DstT, SrcT, trait>(dst_pre##14, src_pre##14##src_sfx, mask); \
        Cast<DstT, SrcT, trait>(dst_pre##15, src_pre##15##src_sfx, mask); \
        Cast<DstT, SrcT, trait>(dst_pre##16, src_pre##16##src_sfx, mask); \
    } while (0)

/* **************************************************************************************************
 * Cast even/odd alternating: Cast<DstT, SrcT, castTraitZero/One>(dst_i_sfx, src_i, mask)
 * ************************************************************************************************* */
#define DO_CAST_EVEN_ODD_2(DstT, SrcT, even_pre, odd_pre, dst_sfx, mask)          \
    do {                                                                          \
        Cast<DstT, SrcT, castTraitZero>(even_pre##1##dst_sfx, even_pre##1, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##1##dst_sfx, odd_pre##1, mask);    \
        Cast<DstT, SrcT, castTraitZero>(even_pre##2##dst_sfx, even_pre##2, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##2##dst_sfx, odd_pre##2, mask);    \
    } while (0)

#define DO_CAST_EVEN_ODD_4(DstT, SrcT, even_pre, odd_pre, dst_sfx, mask)          \
    do {                                                                          \
        DO_CAST_EVEN_ODD_2(DstT, SrcT, even_pre, odd_pre, dst_sfx, mask);         \
        Cast<DstT, SrcT, castTraitZero>(even_pre##3##dst_sfx, even_pre##3, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##3##dst_sfx, odd_pre##3, mask);    \
        Cast<DstT, SrcT, castTraitZero>(even_pre##4##dst_sfx, even_pre##4, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##4##dst_sfx, odd_pre##4, mask);    \
    } while (0)

#define DO_CAST_EVEN_ODD_8(DstT, SrcT, even_pre, odd_pre, dst_sfx, mask)          \
    do {                                                                          \
        DO_CAST_EVEN_ODD_4(DstT, SrcT, even_pre, odd_pre, dst_sfx, mask);         \
        Cast<DstT, SrcT, castTraitZero>(even_pre##5##dst_sfx, even_pre##5, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##5##dst_sfx, odd_pre##5, mask);    \
        Cast<DstT, SrcT, castTraitZero>(even_pre##6##dst_sfx, even_pre##6, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##6##dst_sfx, odd_pre##6, mask);    \
        Cast<DstT, SrcT, castTraitZero>(even_pre##7##dst_sfx, even_pre##7, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##7##dst_sfx, odd_pre##7, mask);    \
        Cast<DstT, SrcT, castTraitZero>(even_pre##8##dst_sfx, even_pre##8, mask); \
        Cast<DstT, SrcT, castTraitOne>(odd_pre##8##dst_sfx, odd_pre##8, mask);    \
    } while (0)

/* **************************************************************************************************
 * Interleave pairs: Interleave(dst_(2K-1)_sfx, dst_(2K)_sfx, src_K, src_K)
 * ************************************************************************************************* */
#define DO_INTERLEAVE_PAIRS_1(dst_pre, dst_sfx, src_pre)                              \
    do {                                                                              \
        Interleave(dst_pre##1##dst_sfx, dst_pre##2##dst_sfx, src_pre##1, src_pre##1); \
    } while (0)

#define DO_INTERLEAVE_PAIRS_2(dst_pre, dst_sfx, src_pre)                               \
    do {                                                                               \
        DO_INTERLEAVE_PAIRS_1(dst_pre, dst_sfx, src_pre);                              \
        Interleave(dst_pre##3##dst_sfx, dst_pre##4##dst_sfx, src_pre##2, src_pre##2);  \
    } while (0)

#define DO_INTERLEAVE_PAIRS_3(dst_pre, dst_sfx, src_pre)                               \
    do {                                                                               \
        DO_INTERLEAVE_PAIRS_2(dst_pre, dst_sfx, src_pre);                              \
        Interleave(dst_pre##5##dst_sfx, dst_pre##6##dst_sfx, src_pre##3, src_pre##3);  \
    } while (0)

#define DO_INTERLEAVE_PAIRS_4(dst_pre, dst_sfx, src_pre)                               \
    do {                                                                               \
        DO_INTERLEAVE_PAIRS_3(dst_pre, dst_sfx, src_pre);                              \
        Interleave(dst_pre##7##dst_sfx, dst_pre##8##dst_sfx, src_pre##4, src_pre##4);  \
    } while (0)

#define DO_INTERLEAVE_PAIRS_8(dst_pre, dst_sfx, src_pre)                                 \
    do {                                                                                 \
        DO_INTERLEAVE_PAIRS_4(dst_pre, dst_sfx, src_pre);                                \
        Interleave(dst_pre##9##dst_sfx, dst_pre##10##dst_sfx, src_pre##5, src_pre##5);   \
        Interleave(dst_pre##11##dst_sfx, dst_pre##12##dst_sfx, src_pre##6, src_pre##6);  \
        Interleave(dst_pre##13##dst_sfx, dst_pre##14##dst_sfx, src_pre##7, src_pre##7);  \
        Interleave(dst_pre##15##dst_sfx, dst_pre##16##dst_sfx, src_pre##8, src_pre##8);  \
    } while (0)
/* **************************************************************************************************
 * Declare N RegTensor<float> variables: name##1 .. name##N
 * ************************************************************************************************* */
#define VREG_FLOAT_DECL_6(name) \
    RegTensor<float> name##1; RegTensor<float> name##2; \
    RegTensor<float> name##3; RegTensor<float> name##4; \
    RegTensor<float> name##5; RegTensor<float> name##6

#define VREG_FLOAT_DECL_8(name) \
    VREG_FLOAT_DECL_6(name); \
    RegTensor<float> name##7; RegTensor<float> name##8

#define VREG_FLOAT_DECL_16(name) \
    VREG_FLOAT_DECL_8(name); \
    RegTensor<float> name##9;  RegTensor<float> name##10; \
    RegTensor<float> name##11; RegTensor<float> name##12; \
    RegTensor<float> name##13; RegTensor<float> name##14; \
    RegTensor<float> name##15; RegTensor<float> name##16

/* **************************************************************************************************
 * Declare N even/odd float pairs: name##_even1..N, name##_odd1..N
 * ************************************************************************************************* */
#define VREG_FLOAT_DECL_EXP_PAIRS_4(name) \
    RegTensor<float> name##_even1; RegTensor<float> name##_odd1; \
    RegTensor<float> name##_even2; RegTensor<float> name##_odd2; \
    RegTensor<float> name##_even3; RegTensor<float> name##_odd3; \
    RegTensor<float> name##_even4; RegTensor<float> name##_odd4

#define VREG_FLOAT_DECL_EXP_PAIRS_8(name) \
    VREG_FLOAT_DECL_EXP_PAIRS_4(name); \
    RegTensor<float> name##_even5; RegTensor<float> name##_odd5; \
    RegTensor<float> name##_even6; RegTensor<float> name##_odd6; \
    RegTensor<float> name##_even7; RegTensor<float> name##_odd7; \
    RegTensor<float> name##_even8; RegTensor<float> name##_odd8

/* **************************************************************************************************
 * Declare N MaskReg comparison variables: preg_compare1 .. preg_compareN
 * ************************************************************************************************* */
#define VREG_PREG_COMPARE_DECL_6() \
    MaskReg preg_compare1; MaskReg preg_compare2; \
    MaskReg preg_compare3; MaskReg preg_compare4; \
    MaskReg preg_compare5; MaskReg preg_compare6

#define VREG_PREG_COMPARE_DECL_8() \
    VREG_PREG_COMPARE_DECL_6(); \
    MaskReg preg_compare7; MaskReg preg_compare8

#define VREG_PREG_COMPARE_DECL_16() \
    VREG_PREG_COMPARE_DECL_8(); \
    MaskReg preg_compare9;  MaskReg preg_compare10; \
    MaskReg preg_compare11; MaskReg preg_compare12; \
    MaskReg preg_compare13; MaskReg preg_compare14; \
    MaskReg preg_compare15; MaskReg preg_compare16
/* **************************************************************************************************
 * Or even/odd pair: Or((uint16_t&)dst_K_sfx, (uint16_t&)even_K_sfx, (uint16_t&)odd_K_sfx, mask)
 * ************************************************************************************************* */
#define DO_OR_2(dst, even, odd, sfx, mask)                                                         \
    do {                                                                                           \
        Or((RegTensor<uint16_t>&)dst##1##sfx, (RegTensor<uint16_t>&)even##1##sfx,                 \
           (RegTensor<uint16_t>&)odd##1##sfx, mask);                                              \
        Or((RegTensor<uint16_t>&)dst##2##sfx, (RegTensor<uint16_t>&)even##2##sfx,                 \
           (RegTensor<uint16_t>&)odd##2##sfx, mask);                                              \
    } while (0)

#define DO_OR_4(dst, even, odd, sfx, mask)                                                         \
    do {                                                                                           \
        DO_OR_2(dst, even, odd, sfx, mask);                                                       \
        Or((RegTensor<uint16_t>&)dst##3##sfx, (RegTensor<uint16_t>&)even##3##sfx,                 \
           (RegTensor<uint16_t>&)odd##3##sfx, mask);                                              \
        Or((RegTensor<uint16_t>&)dst##4##sfx, (RegTensor<uint16_t>&)even##4##sfx,                 \
           (RegTensor<uint16_t>&)odd##4##sfx, mask);                                              \
    } while (0)

#define DO_OR_8(dst, even, odd, sfx, mask)                                                         \
    do {                                                                                           \
        DO_OR_4(dst, even, odd, sfx, mask);                                                       \
        Or((RegTensor<uint16_t>&)dst##5##sfx, (RegTensor<uint16_t>&)even##5##sfx,                 \
           (RegTensor<uint16_t>&)odd##5##sfx, mask);                                              \
        Or((RegTensor<uint16_t>&)dst##6##sfx, (RegTensor<uint16_t>&)even##6##sfx,                 \
           (RegTensor<uint16_t>&)odd##6##sfx, mask);                                              \
        Or((RegTensor<uint16_t>&)dst##7##sfx, (RegTensor<uint16_t>&)even##7##sfx,                 \
           (RegTensor<uint16_t>&)odd##7##sfx, mask);                                              \
        Or((RegTensor<uint16_t>&)dst##8##sfx, (RegTensor<uint16_t>&)even##8##sfx,                 \
           (RegTensor<uint16_t>&)odd##8##sfx, mask);                                              \
    } while (0)

/* **************************************************************************************************
 * ExpSub even/odd pairs: ExpSub(even_K ← inp_(2K-1)), ExpSub(odd_K ← inp_(2K))
 * ************************************************************************************************* */
#define DO_EXPSUB_PAIRS_4(even, odd, inp, maxreg, mask)            \
    do {                                                           \
        ExpSub(even##1, inp##1, maxreg, mask); ExpSub(odd##1, inp##2, maxreg, mask); \
        ExpSub(even##2, inp##3, maxreg, mask); ExpSub(odd##2, inp##4, maxreg, mask); \
        ExpSub(even##3, inp##5, maxreg, mask); ExpSub(odd##3, inp##6, maxreg, mask); \
        ExpSub(even##4, inp##7, maxreg, mask); ExpSub(odd##4, inp##8, maxreg, mask); \
    } while (0)

#define DO_EXPSUB_PAIRS_8(even, odd, inp, maxreg, mask)            \
    do {                                                           \
        DO_EXPSUB_PAIRS_4(even, odd, inp, maxreg, mask);          \
        ExpSub(even##5, inp##9, maxreg, mask); ExpSub(odd##5, inp##10, maxreg, mask); \
        ExpSub(even##6, inp##11, maxreg, mask); ExpSub(odd##6, inp##12, maxreg, mask); \
        ExpSub(even##7, inp##13, maxreg, mask); ExpSub(odd##7, inp##14, maxreg, mask); \
        ExpSub(even##8, inp##15, maxreg, mask); ExpSub(odd##8, inp##16, maxreg, mask); \
    } while (0)

/* **************************************************************************************************
 * Select operation: Select(sel_K, min, inp_K, cmp_K)
 * ************************************************************************************************* */
#define DO_SELECT_6(sel, mins, inp, cmp)                                                         \
    do {                                                                                         \
        Select(sel##1, mins, inp##1, cmp##1); Select(sel##2, mins, inp##2, cmp##2);              \
        Select(sel##3, mins, inp##3, cmp##3); Select(sel##4, mins, inp##4, cmp##4);              \
        Select(sel##5, mins, inp##5, cmp##5); Select(sel##6, mins, inp##6, cmp##6);              \
    } while (0)

#define DO_SELECT_8(sel, mins, inp, cmp)                                                         \
    do {                                                                                         \
        DO_SELECT_6(sel, mins, inp, cmp);                                                        \
        Select(sel##7, mins, inp##7, cmp##7); Select(sel##8, mins, inp##8, cmp##8);              \
    } while (0)

#define DO_SELECT_16(sel, mins, inp, cmp)                                                        \
    do {                                                                                         \
        DO_SELECT_8(sel, mins, inp, cmp);                                                        \
        Select(sel##9, mins, inp##9, cmp##9);   Select(sel##10, mins, inp##10, cmp##10);         \
        Select(sel##11, mins, inp##11, cmp##11); Select(sel##12, mins, inp##12, cmp##12);        \
        Select(sel##13, mins, inp##13, cmp##13); Select(sel##14, mins, inp##14, cmp##14);        \
        Select(sel##15, mins, inp##15, cmp##15); Select(sel##16, mins, inp##16, cmp##16);        \
    } while (0)

/* **************************************************************************************************
 * LoadAlign sequential: LoadAlign(reg_K, buf + floatRepSize*(K-1) + idx*stride)
 * ************************************************************************************************* */
#define DO_LOADALIGN_6(reg, buf, stride, idx)                                 \
    do {                                                                       \
        LoadAlign(reg##1, buf + floatRepSize * 0 + idx * stride);             \
        LoadAlign(reg##2, buf + floatRepSize * 1 + idx * stride);             \
        LoadAlign(reg##3, buf + floatRepSize * 2 + idx * stride);             \
        LoadAlign(reg##4, buf + floatRepSize * 3 + idx * stride);             \
        LoadAlign(reg##5, buf + floatRepSize * 4 + idx * stride);             \
        LoadAlign(reg##6, buf + floatRepSize * 5 + idx * stride);             \
    } while (0)

#define DO_LOADALIGN_8(reg, buf, stride, idx)                                 \
    do {                                                                       \
        DO_LOADALIGN_6(reg, buf, stride, idx);                                \
        LoadAlign(reg##7, buf + floatRepSize * 6 + idx * stride);             \
        LoadAlign(reg##8, buf + floatRepSize * 7 + idx * stride);             \
    } while (0)

#define DO_LOADALIGN_16(reg, buf, stride, idx)                                \
    do {                                                                       \
        DO_LOADALIGN_8(reg, buf, stride, idx);                                \
        LoadAlign(reg##9, buf + floatRepSize * 8 + idx * stride);             \
        LoadAlign(reg##10, buf + floatRepSize * 9 + idx * stride);            \
        LoadAlign(reg##11, buf + floatRepSize * 10 + idx * stride);           \
        LoadAlign(reg##12, buf + floatRepSize * 11 + idx * stride);           \
        LoadAlign(reg##13, buf + floatRepSize * 12 + idx * stride);           \
        LoadAlign(reg##14, buf + floatRepSize * 13 + idx * stride);           \
        LoadAlign(reg##15, buf + floatRepSize * 14 + idx * stride);           \
        LoadAlign(reg##16, buf + floatRepSize * 15 + idx * stride);           \
    } while (0)

/* **************************************************************************************************
 * LoadAlign for mask: LoadAlign<u32, POST_MODE_UPDATE, DIST_DS>(cmp_K, mub_K, pad)
 * ************************************************************************************************* */
#define DO_LOADALIGN_MASK_6(cmp, mub, pad)                                                         \
    do {                                                                                           \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##1, mub##1, pad);                                                                   \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##2, mub##2, pad);                                                                   \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##3, mub##3, pad);                                                                   \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##4, mub##4, pad);                                                                   \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##5, mub##5, pad);                                                                   \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##6, mub##6, pad);                                                                   \
    } while (0)

#define DO_LOADALIGN_MASK_8(cmp, mub, pad)                                                         \
    do {                                                                                           \
        DO_LOADALIGN_MASK_6(cmp, mub, pad);                                                        \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##7, mub##7, pad);                                                                   \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##8, mub##8, pad);                                                                   \
    } while (0)

#define DO_LOADALIGN_MASK_16(cmp, mub, pad)                                                        \
    do {                                                                                           \
        DO_LOADALIGN_MASK_8(cmp, mub, pad);                                                        \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##9, mub##9, pad);                                                                    \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##10, mub##10, pad);                                                                  \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##11, mub##11, pad);                                                                  \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##12, mub##12, pad);                                                                  \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##13, mub##13, pad);                                                                  \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##14, mub##14, pad);                                                                  \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##15, mub##15, pad);                                                                  \
        LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>( \
            cmp##16, mub##16, pad);                                                                  \
    } while (0)

/* **************************************************************************************************
 * StoreAlign sequential: StoreAlign<T, DIST_NORM_B32>(buf + floatRepSize*(K-1) + idx*stride, val_K, mask)
 * ************************************************************************************************* */
#define DO_STOREALIGN_6(T, buf, val, stride, idx, mask)                                \
    do {                                                                                \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 0 + idx * stride, val##1, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 1 + idx * stride, val##2, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 2 + idx * stride, val##3, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 3 + idx * stride, val##4, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 4 + idx * stride, val##5, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 5 + idx * stride, val##6, mask);       \
    } while (0)

#define DO_STOREALIGN_8(T, buf, val, stride, idx, mask)                                \
    do {                                                                                \
        DO_STOREALIGN_6(T, buf, val, stride, idx, mask);                               \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 6 + idx * stride, val##7, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 7 + idx * stride, val##8, mask);       \
    } while (0)

#define DO_STOREALIGN_16(T, buf, val, stride, idx, mask)                               \
    do {                                                                                \
        DO_STOREALIGN_8(T, buf, val, stride, idx, mask);                               \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 8 + idx * stride, val##9, mask);       \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 9 + idx * stride, val##10, mask);      \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 10 + idx * stride, val##11, mask);     \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 11 + idx * stride, val##12, mask);     \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 12 + idx * stride, val##13, mask);     \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 13 + idx * stride, val##14, mask);     \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 14 + idx * stride, val##15, mask);     \
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(                             \
            (__ubuf__ T *&)buf + floatRepSize * 15 + idx * stride, val##16, mask);     \
    } while (0)
/* **************************************************************************************************
 * maskUb reference-pointer declarations: __ubuf__ uint32_t *&maskUbN
 * ************************************************************************************************* */
#define VREG_FLOAT_MASKUB_REF_6_DECL \
    __ubuf__ uint32_t *&maskUb1, __ubuf__ uint32_t *&maskUb2, \
    __ubuf__ uint32_t *&maskUb3, __ubuf__ uint32_t *&maskUb4, \
    __ubuf__ uint32_t *&maskUb5, __ubuf__ uint32_t *&maskUb6

#define ARG_FLOAT_MASKUB_REF_6 \
    maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6

#define VREG_FLOAT_MASKUB_REF_8_DECL \
    VREG_FLOAT_MASKUB_REF_6_DECL, \
    __ubuf__ uint32_t *&maskUb7, __ubuf__ uint32_t *&maskUb8

#define ARG_FLOAT_MASKUB_REF_8 \
    ARG_FLOAT_MASKUB_REF_6, maskUb7, maskUb8

#define VREG_FLOAT_MASKUB_REF_16_DECL \
    VREG_FLOAT_MASKUB_REF_8_DECL, \
    __ubuf__ uint32_t *&maskUb9,  __ubuf__ uint32_t *&maskUb10, \
    __ubuf__ uint32_t *&maskUb11, __ubuf__ uint32_t *&maskUb12, \
    __ubuf__ uint32_t *&maskUb13, __ubuf__ uint32_t *&maskUb14, \
    __ubuf__ uint32_t *&maskUb15, __ubuf__ uint32_t *&maskUb16

#define ARG_FLOAT_MASKUB_REF_16 \
    ARG_FLOAT_MASKUB_REF_8, \
    maskUb9, maskUb10, maskUb11, maskUb12, maskUb13, maskUb14, maskUb15, maskUb16

/* **************************************************************************************************
 * maskUb plain pointer declarations: __ubuf__ uint32_t * maskUbN
 * ************************************************************************************************* */
#define VREG_FLOAT_MASKUB_6_DECL \
    __ubuf__ uint32_t * maskUb1, __ubuf__ uint32_t * maskUb2, \
    __ubuf__ uint32_t * maskUb3, __ubuf__ uint32_t * maskUb4, \
    __ubuf__ uint32_t * maskUb5, __ubuf__ uint32_t * maskUb6

#define ARG_FLOAT_MASKUB_6 \
    maskUb1, maskUb2, maskUb3, maskUb4, maskUb5, maskUb6

#define VREG_FLOAT_MASKUB_8_DECL \
    VREG_FLOAT_MASKUB_6_DECL, \
    __ubuf__ uint32_t * maskUb7, __ubuf__ uint32_t * maskUb8

#define ARG_FLOAT_MASKUB_8 \
    ARG_FLOAT_MASKUB_6, maskUb7, maskUb8

#define VREG_FLOAT_MASKUB_16_DECL \
    VREG_FLOAT_MASKUB_8_DECL, \
    __ubuf__ uint32_t * maskUb9,  __ubuf__ uint32_t * maskUb10, \
    __ubuf__ uint32_t * maskUb11, __ubuf__ uint32_t * maskUb12, \
    __ubuf__ uint32_t * maskUb13, __ubuf__ uint32_t * maskUb14, \
    __ubuf__ uint32_t * maskUb15, __ubuf__ uint32_t * maskUb16

#define ARG_FLOAT_MASKUB_16 \
    ARG_FLOAT_MASKUB_8, \
    maskUb9, maskUb10, maskUb11, maskUb12, maskUb13, maskUb14, maskUb15, maskUb16

/* **************************************************************************************************
 * maskUb local initialization: __ubuf__ uint32_t * maskUbN = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + offset)
 * ************************************************************************************************* */
#define DO_MASKUB_INIT_8(maskTensor) \
    __ubuf__ uint32_t * maskUb1 = (__ubuf__ uint32_t *)maskTensor.GetPhyAddr(); \
    __ubuf__ uint32_t * maskUb2 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize); \
    __ubuf__ uint32_t * maskUb3 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 2); \
    __ubuf__ uint32_t * maskUb4 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 3); \
    __ubuf__ uint32_t * maskUb5 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 4); \
    __ubuf__ uint32_t * maskUb6 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 5); \
    __ubuf__ uint32_t * maskUb7 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 6); \
    __ubuf__ uint32_t * maskUb8 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 7)

#define DO_MASKUB_INIT_16(maskTensor) \
    DO_MASKUB_INIT_8(maskTensor); \
    __ubuf__ uint32_t * maskUb9 =  (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 8); \
    __ubuf__ uint32_t * maskUb10 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 9); \
    __ubuf__ uint32_t * maskUb11 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 10); \
    __ubuf__ uint32_t * maskUb12 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 11); \
    __ubuf__ uint32_t * maskUb13 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 12); \
    __ubuf__ uint32_t * maskUb14 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 13); \
    __ubuf__ uint32_t * maskUb15 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 14); \
    __ubuf__ uint32_t * maskUb16 = (__ubuf__ uint32_t *)(maskTensor.GetPhyAddr() + floatRepSize * 15)

} // namespace

#endif // VF_BASIC_BLOCK_UTILS_H
