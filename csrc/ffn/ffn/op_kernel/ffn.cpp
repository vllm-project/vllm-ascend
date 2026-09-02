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
 * \file ffn.cpp
 * \brief
 */

#include "kernel_operator.h"
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#include "ffn_high_performence.h"
#include "ffn_quant.h"
#include "ffn_antiquant.h"
#include "ffn_antiquant_msd.h"
#include "ffn_glu.cpp"
#include "ffn_high_precision.h"
#else
#include "ffn_nonquant_nz.h"
#endif

using namespace FFN;
using namespace matmul;

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X) && FORMAT_X == FORMAT_FRACTAL_NZ
constexpr CubeFormat formatX = CubeFormat::NZ;
#else
constexpr CubeFormat formatX = CubeFormat::ND;
#endif

#if defined(FORMAT_WEIGHT1) && FORMAT_WEIGHT1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat formatW = CubeFormat::NZ;
#else
constexpr CubeFormat formatW = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat formatY = CubeFormat::NZ;
#else
constexpr CubeFormat formatY = CubeFormat::ND;
#endif

template <typename T, CubeFormat cubeFormatA> using aType = matmul::MatmulType<TPosition::GM, cubeFormatA, T, false>;
template <typename T, CubeFormat cubeFormatB> using bType = matmul::MatmulType<TPosition::GM, cubeFormatB, T, false>;
template <typename T> using biasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
template <typename T, TPosition tPositionC> using cType = matmul::MatmulType<tPositionC, formatY, T>;
// CFG_MDL: open large pack movement mode
template <typename T, CubeFormat cubeFormatA, CubeFormat cubeFormatB = CubeFormat::ND,
          TPosition tPositionC = TPosition::GM>
using mmType = MMType<aType<T, cubeFormatA>, bType<T, cubeFormatB>, cType<T, tPositionC>, biasType<T>, CFG_MDL>;

template <typename T, typename cT = T, typename biasT = T, CubeFormat cubeFormatA = CubeFormat::ND,
          TPosition tPositionC = TPosition::GM>
using mmWNDType =
    MMType<aType<T, cubeFormatA>, bType<T, CubeFormat::ND>, cType<cT, tPositionC>, biasType<biasT>, CFG_MDL>;
// MM_CFG_UNITFLAG: enable unitflag
constexpr MatmulConfig MM_CFG_UNITFLAG{false, false, true, 0, 0, 0, false, false, false, false,
                                       false, 0,     0,    0, 0, 0, 0,     0,     true};
static constexpr MatmulConfig MM_CFG_STEPN = GetSpecialMDLConfig();
template <typename T, typename cT = T, typename biasT = T, CubeFormat cubeFormatA = CubeFormat::ND,
          const MatmulConfig &MM_CFG = MM_CFG_UNITFLAG, TPosition tPositionC = TPosition::GM>
using mmQuantType =
    MMType<aType<T, cubeFormatA>, bType<T, CubeFormat::ND>, cType<cT, tPositionC>, biasType<biasT>, MM_CFG>;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename T> struct GetC1Type {
    using Type = T;
};

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <> struct GetC1Type<bfloat16_t> {
    using Type = float;
};
#endif

template <> struct GetC1Type<half> {
    using Type = half;
};
#endif

extern "C" __global__ __aicore__ void ffn(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                          __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                                          __gm__ uint8_t *scale, __gm__ uint8_t *offset, __gm__ uint8_t *deqScale1,
                                          __gm__ uint8_t *deqScale2, __gm__ uint8_t *antiquant_scale1,
                                          __gm__ uint8_t *antiquant_scale2, __gm__ uint8_t *antiquant_offset1,
                                          __gm__ uint8_t *antiquant_offset2, __gm__ uint8_t *y,
                                          __gm__ uint8_t *workSpace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    // 拆解TilingData 数据
    GET_TILING_DATA(tiling_data, tiling);
    const FFNTilingData *__restrict ffn_tiling_data = &tiling_data;
    const TCubeTiling *__restrict mm1Tiling = &(ffn_tiling_data->mm1TilingData);
    const TCubeTiling *__restrict mm2Tiling = &(ffn_tiling_data->mm2TilingData);
    // 获取Op可用WorkSpace空间
    __gm__ uint8_t *user1 = GetUserWorkspace(workSpace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    if (TILING_KEY_IS(2000)) { // One matmul, float16
        KERNEL_TASK_TYPE(2000, KERNEL_TYPE_MIX_AIC_1_1);
        using mt = mmType<half, CubeFormat::ND, CubeFormat::ND>;
        mt::MT mm;
        const TCubeTiling *__restrict mmTiling = mm1Tiling->isBias ? mm1Tiling : mm2Tiling;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, mmTiling);
        FFNHighPerformence<half, mt> op(mm, mm);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, y, user1, ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(0)) { // Two matmul, float16
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_1);
        using mt = mmType<half, CubeFormat::ND>;
        mt::MT mm1;
        mt::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNHighPerformence<half, mt> op(mm1, mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, y, user1, ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) { // QUANT
        mmQuantType<int8_t, half, int32_t>::MT mm1;
        mmQuantType<int8_t, half, int32_t>::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNQuant<int8_t, decltype(mm1), decltype(mm2), half, half, int32_t, half, uint64_t> op(mm1, mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
                ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(2001)) { // QUANT, One matmul
        mmQuantType<int8_t, half, int32_t>::MT mm;
        const TCubeTiling *__restrict mmTiling = mm1Tiling->isBias ? mm1Tiling : mm2Tiling;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, mmTiling);
        FFNQuant<int8_t, decltype(mm), decltype(mm), half, half, int32_t, half, uint64_t> op(mm, mm);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
                ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(2002)) { // QUANT, One matmul, stepN=2
        mmQuantType<int8_t, half, int32_t, CubeFormat::ND, MM_CFG_STEPN>::MT mm;
        const TCubeTiling *__restrict mmTiling = mm1Tiling->isBias ? mm1Tiling : mm2Tiling;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, mmTiling);
        FFNQuant<int8_t, decltype(mm), decltype(mm), half, half, int32_t, half, uint64_t> op(mm, mm);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
                ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // 2: glu fp16 high performance
        FFNGlu<half> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm1, mm1Tiling, op.mm2, mm2Tiling);
        op.Init(x, weight1, weight2, bias1, bias2, y, user1, ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) { // high precision
        using mt1 = mmWNDType<half, float>;
        mt1::MT mm1;
        using mt2 = mmWNDType<half>;
        mt2::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNHighPrecision<half, mt1, mt2, float, half, half> op(mm1, mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, y, user1, ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) { // ANTI_QUANT
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
                      (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
#else
        if constexpr (IsSameType<DTYPE_X, half>::value &&
                      (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
#endif
            using c1T = GetC1Type<DTYPE_X>::Type;
            mmWNDType<DTYPE_X, c1T, c1T>::MT mm1;     // bias dtype is the same as mm1 output dtype for antiquant
            mmWNDType<DTYPE_X, DTYPE_Y, c1T>::MT mm2; // bias dtype is the same as mm1 output dtype for antiquant
            FFNAntiQuant<DTYPE_X, DTYPE_WEIGHT1, decltype(mm1), decltype(mm2), c1T, DTYPE_Y, c1T, false> op(mm1, mm2);
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
            op.Init(x, weight1, weight2, expertTokens, bias1, bias2, antiquant_scale1, antiquant_scale2,
                    antiquant_offset1, antiquant_offset2, y, user1, ffn_tiling_data, &tPipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(12)) { // ANTI_QUANT_PERGROUP
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
                      (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
#else
        if constexpr (IsSameType<DTYPE_X, half>::value &&
                      (IsSameType<DTYPE_WEIGHT1, int8_t>::value || IsSameType<DTYPE_WEIGHT1, int4b_t>::value)) {
#endif
            using c1T = GetC1Type<DTYPE_X>::Type;
            // bias dtype is the same as mm1 output dtype for antiquant
            mmWNDType<DTYPE_X, c1T, c1T>::MT mm1;
            mmWNDType<DTYPE_X, DTYPE_Y, c1T>::MT mm2;
            FFNAntiQuant<DTYPE_X, DTYPE_WEIGHT1, decltype(mm1), decltype(mm2), c1T, DTYPE_Y, c1T, true> op(mm1, mm2);
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
            op.Init(x, weight1, weight2, expertTokens, bias1, bias2, antiquant_scale1, antiquant_scale2,
                    antiquant_offset1, antiquant_offset2, y, user1, ffn_tiling_data, &tPipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(7)) { // high precision
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        using mt1 = mmWNDType<bfloat16_t, float, float>;
        mt1::MT mm1;
        using mt2 = mmWNDType<bfloat16_t, bfloat16_t, float>;
        mt2::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNHighPrecision<bfloat16_t, mt1, mt2, float, bfloat16_t, float> op(mm1, mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, y, user1, ffn_tiling_data, &tPipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(11)) { // QUANT with output_type is bf16
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        mmQuantType<int8_t, int32_t, int32_t>::MT mm1;
        mmQuantType<int8_t, int32_t, int32_t>::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNQuant<int8_t, decltype(mm1), decltype(mm2), int32_t, bfloat16_t, int32_t, float, bfloat16_t> op(mm1, mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
                ffn_tiling_data, &tPipe);
        op.Process();
#endif
    } else if (TILING_KEY_IS(15)) { // ANTI_QUANT_MSD
        KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_1);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        if constexpr ((IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, bfloat16_t>::value) &&
                      (IsSameType<DTYPE_WEIGHT1, int8_t>::value)) {
#else
        if constexpr ((IsSameType<DTYPE_X, half>::value) && (IsSameType<DTYPE_WEIGHT1, int8_t>::value)) {
#endif
            mmQuantType<int8_t, int32_t>::MT mm1;
            mmQuantType<int8_t, int32_t>::MT mm2;
            using c1T = GetC1Type<DTYPE_X>::Type;
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
            // bias dtype is half for DTYPE_X half, float for DTYPE_X bfloat16
            FFNAntiQuantMSD<DTYPE_X, int8_t, decltype(mm1), decltype(mm2), int32_t, DTYPE_Y, c1T> op(mm1, mm2);
            op.Init(x, weight1, weight2, expertTokens, bias1, bias2, antiquant_scale1, antiquant_scale2,
                    antiquant_offset1, antiquant_offset2, y, user1, ffn_tiling_data, &tPipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(13)) { // QUANT decscale is float32
        mmQuantType<int8_t, half, int32_t>::MT mm1;
        mmQuantType<int8_t, half, int32_t>::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNQuant<int8_t, decltype(mm1), decltype(mm2), half, half, int32_t, half, float> op(mm1, mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
                ffn_tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(14)) { // QUANT SCALE PER-CHANNEL
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        using c1TQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, int32_t, half>;
        using actTQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, float, half>;
        using deqTQuant = std::conditional_t<IsSameType<DTYPE_Y, bfloat16_t>::value, bfloat16_t, uint64_t>;
        mmQuantType<int8_t, c1TQuant, int32_t>::MT mm1;
        mmQuantType<int8_t, c1TQuant, int32_t>::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNQuant<int8_t, decltype(mm1), decltype(mm2), c1TQuant, DTYPE_Y, int32_t, actTQuant, deqTQuant, true> op(mm1,
                                                                                                                  mm2);
        op.Init(x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, y, user1,
                ffn_tiling_data, &tPipe);
        op.Process();
#endif
    }
#else
    if (TILING_KEY_IS(0)) { // Two matmul, float16
        using mt1 = mmType<half, formatX, formatW, TPosition::VECIN>;
        using mt2 = mmType<half, formatX, formatW, TPosition::GM>;
        mt1::MT mm1;
        mt2::MT mm2;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm1, mm1Tiling, mm2, mm2Tiling);
        FFNCompute<half, mt1, mt2> computeOp(mm1, mm2);
        computeOp.Init(x, weight1, weight2, bias1, bias2, y, user1, ffn_tiling_data, &tPipe);
        FFNProcess<decltype(computeOp)> op(computeOp);
        op.Init(ffn_tiling_data);
        op.Process();
    }
#endif
}
