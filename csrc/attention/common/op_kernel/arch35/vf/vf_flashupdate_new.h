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
 * \file vf_flashupdate_new.h
 * \brief
 */
#ifndef MY_FLASH_UPDATE_NEW_INTERFACE_H
#define MY_FLASH_UPDATE_NEW_INTERFACE_H

#include "kernel_tensor.h"

namespace FaVectorApi {
// bf16->fp32
static constexpr MicroAPI::CastTrait castTraitFp16_32_update = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr uint16_t REDUCE_SIZE = 1;
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__simd_vf__ inline void FlashUpdateBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_row_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    // dstTensor = preTensor * expMaxTensor + curTensor
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);  // [m,8]
        if constexpr (isMlaFullQuant) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_row_max, rowMaxUb + i * reduceSize);
        }

        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            if constexpr (isMlaFullQuant) {
                Mul(vreg_input_cur, vreg_input_cur, vreg_row_max, preg_all);
            }
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_add, preg_all);
        }
    }
}
/* **************************************************************************************************
 * FlashUpdate, fp32
 * ************************************************************************************************* */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__aicore__ inline void FlashUpdateBasic(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * rowMaxUb = (__ubuf__ T*)rowMaxTensor.GetPhyAddr();

    FlashUpdateBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre, isMlaFullQuant>(
        dstUb, curUb, preUb, expMaxUb, rowMaxUb, m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateGeneralVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre, const uint32_t pltTailD, const uint16_t hasTail)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    uint32_t tmpTailD = pltTailD;
    MaskReg preg_tail_d = UpdateMask<float>(tmpTailD);
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d / floatRepSize;

    // dstTensor = preTensor * expMaxTensor + curTensor
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);  // [m,8]

        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_add, preg_all);
        }
        for (uint16_t t = 0; t < hasTail; ++t) {
            LoadAlign(vreg_input_pre, preUb + i * d + dLoops * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + dLoops * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_tail_d);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_tail_d);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + dLoops * floatRepSize, vreg_add, preg_tail_d);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateGeneral(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    FlashUpdateGeneralVF<T, INPUT_T, OUTPUT_T, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, m, d, deScaleV, deScaleVPre, pltTailD, hasTail);
}

/*
 * @ingroup FlashUpdate
 * @brief compute, dstTensor = preTensor * expMaxTensor + curTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input columns, should be 32 bytes aligned
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre, bool isMlaFullQuant>
__aicore__ inline void FlashUpdateNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& preTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdate, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr(srcD % floatRepSize == 0) {
        FlashUpdateBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre, isMlaFullQuant>(dstTensor, curTensor, preTensor, expMaxTensor, rowMaxTensor,
        m, d, deScaleV, deScaleVPre);
    } else {

        FlashUpdateGeneral<T, INPUT_T, OUTPUT_T, REDUCE_SIZE, isUpdatePre>(dstTensor, curTensor, preTensor, expMaxTensor, m, d,
        deScaleV, deScaleVPre);
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__simd_vf__ inline void FlashUpdateLastBasicVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * rowMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_row_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<half> vreg_cast;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        if constexpr (isMlaFullQuant) {
            LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_row_max, rowMaxUb + i * reduceSize);
        }
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            if constexpr (isMlaFullQuant) {
                Mul(vreg_input_cur, vreg_input_cur, vreg_row_max, preg_all);
            }
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);
            if constexpr (isMlaFullQuant) {
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value) {
                    Muls(vreg_div, vreg_div, fp8e4m3MaxValueRec, preg_all);
                } else if constexpr (IsSameType<INPUT_T, int8_t>::value) {
                    Muls(vreg_div, vreg_div, int8MaxValueRec, preg_all);
                } else {
                    Muls(vreg_div, vreg_div, hifp8MaxValueRec, preg_all);
                }
            }
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__aicore__ inline void FlashUpdateLastBasic(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();
    __ubuf__ float * rowMaxUb = (__ubuf__ T*)rowMaxTensor.GetPhyAddr();

    FlashUpdateLastBasicVF<T, INPUT_T, OUTPUT_T, srcD, reduceSize, isUpdatePre, isMlaFullQuant>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, rowMaxUb, m, d, deScaleV, deScaleVPre);
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__simd_vf__ inline void FlashUpdateLastGeneralVF(__ubuf__ float * dstUb, __ubuf__ float * curUb,
    __ubuf__ float * preUb, __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre, const uint32_t pltTailD, const uint16_t hasTail)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<half> vreg_cast;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    uint32_t tmpTailD = pltTailD;
    MaskReg preg_tail_d = UpdateMask<float>(tmpTailD);
    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = d / floatRepSize;

    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_max, expMaxUb + i * reduceSize);
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * reduceSize);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input_pre, preUb + i * d + j * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);

            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_all);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_all);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_all);
        }

        for (uint16_t t = 0; t < hasTail; ++t) {
            LoadAlign(vreg_input_pre, preUb + i * d + dLoops * floatRepSize);
            LoadAlign(vreg_input_cur, curUb + i * d + dLoops * floatRepSize);
            Mul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_tail_d);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
                if constexpr (isUpdatePre) {
                    Muls(vreg_mul, vreg_mul, deScaleVPre, preg_all);
                }
            }
            Add(vreg_add, vreg_mul, vreg_input_cur, preg_tail_d);
            Div(vreg_div, vreg_add, vreg_exp_sum, preg_tail_d);

            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + dLoops * floatRepSize, vreg_div, preg_tail_d);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t reduceSize, bool isUpdatePre>
__aicore__ inline void FlashUpdateLastGeneral(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& expSumTensor,
    const uint16_t m, const uint16_t d, const float deScaleV, const float deScaleVPre)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * preUb = (__ubuf__ T*)preTensor.GetPhyAddr();
    __ubuf__ float * expMaxUb = (__ubuf__ T*)expMaxTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = tailD;

    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    FlashUpdateLastGeneralVF<T, INPUT_T, OUTPUT_T, reduceSize, isUpdatePre>(
        dstUb, curUb, preUb, expMaxUb, expSumUb, m, d, deScaleV, deScaleVPre, pltTailD, hasTail);
}

/*
 * @ingroup FlashUpdateLast
 * @brief compute, dstTensor = (preTensor * expMaxTensor + curTensor) / expSumTensor
 * @param [out] dstTensor, output LocalTensor
 * @param [in] curTensor, input LocalTensor
 * @param [in] preTensor, input LocalTensor
 * @param [in] expMaxTensor, input LocalTensor
 * @param [in] expSumTensor, input LocalTensor
 * @param [in] m, input rows
 * @param [in] d, input columns, 32 bytes align
 */
template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, bool isUpdatePre, bool isMlaFullQuant>
__aicore__ inline void FlashUpdateLastNew(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& curTensor, const LocalTensor<T>& preTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& rowMaxTensor, const LocalTensor<T>& expSumTensor,
    uint16_t m, uint16_t d, const float deScaleV, const float deScaleVPre)
{
    static_assert(IsSameType<T, float>::value, "VF FlashUpdateLast, T must be float");

    constexpr uint16_t floatRepSize = 64;
    if constexpr(srcD % floatRepSize == 0) {
        FlashUpdateLastBasic<T, INPUT_T, OUTPUT_T, srcD, REDUCE_SIZE, isUpdatePre, isMlaFullQuant>(
            dstTensor, curTensor, preTensor, expMaxTensor, rowMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    } else {
        FlashUpdateLastGeneral<T, INPUT_T, OUTPUT_T, REDUCE_SIZE, isUpdatePre>(
            dstTensor, curTensor, preTensor, expMaxTensor, expSumTensor, m, d, deScaleV, deScaleVPre);
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD, bool isMlaFullQuant>
__simd_vf__ inline void LastDivNewVF(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * expSumUb,
    const uint16_t m, const uint16_t d, const float deScaleV)
{
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d >> 6;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    for (uint16_t i = 0; i < m; ++i) {
        uint32_t sreg_init = d;
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_exp_sum, expSumUb + i * REDUCE_SIZE);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MaskReg preg_update = UpdateMask<float>(sreg_init);

            LoadAlign(vreg_input_cur, curUb + i * d + j * floatRepSize);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                Muls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all);
            }
            Div(vreg_div, vreg_input_cur, vreg_exp_sum, preg_update);
            if constexpr (isMlaFullQuant) {
                if constexpr (IsSameType<INPUT_T, fp8_e4m3fn_t>::value) {
                    Muls(vreg_div, vreg_div, fp8e4m3MaxValueRec, preg_all);
                } else if constexpr (IsSameType<INPUT_T, int8_t>::value) {
                    Muls(vreg_div, vreg_div, int8MaxValueRec, preg_all);
                } else {
                    Muls(vreg_div, vreg_div, hifp8MaxValueRec, preg_all);
                }
            }
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_div, preg_update);
        }
    }
}

// dstTensor = curTensor / expSumTensor, curTensor: [64,128], expSumTensor: [64,8]
template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD, bool isMlaFullQuant>
__aicore__ inline void LastDivNew(const LocalTensor<T>& dstTensor, const LocalTensor<T>& curTensor,
    const LocalTensor<T>& expSumTensor, const uint16_t m, const uint16_t d, const float deScaleV)
{
    __ubuf__ float * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ float * curUb = (__ubuf__ T*)curTensor.GetPhyAddr();
    __ubuf__ float * expSumUb = (__ubuf__ T*)expSumTensor.GetPhyAddr();

    LastDivNewVF<T, INPUT_T, OUTPUT_T, srcD, isMlaFullQuant>(dstUb, curUb, expSumUb, m, d, deScaleV);
}

template <typename T, uint32_t srcD>
__simd_vf__ inline void InvalidLineUpdateVF(__ubuf__ T * dstUb, __ubuf__ T * srcUb, __ubuf__ T * maxUb,
    const uint16_t m, const uint16_t d, const T minValue, const T invalidValue)
{
    RegTensor<float> vreg_invalid_value;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_input;
    RegTensor<float> vreg_input_brc;

    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
    MaskReg preg_compare;
    const uint16_t dLoops = d >> 6;

    Duplicate(vreg_invalid_value, invalidValue);
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max, maxUb + i);
        Compares<T, CMPMODE::EQ>(preg_compare, vreg_max, minValue, preg_all);
        for (uint16_t j = 0; j < dLoops; ++j) {
            LoadAlign(vreg_input, srcUb + i * d + j * floatRepSize);
            Select(vreg_input_brc, vreg_invalid_value, vreg_input, preg_compare);
            StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ T *&)dstUb + i * d + j * floatRepSize, vreg_input_brc, preg_all);
        }
    }
}

template <typename T, uint32_t srcD>
__aicore__ inline void InvalidLineUpdate(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& maxTensor, const uint16_t m, const uint16_t d, const T minValue, const T invalidValue)
{
    __ubuf__ T * dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T * srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ T * maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    uint16_t dLoops = d >> 6;

    InvalidLineUpdateVF<T, srcD>(dstUb, srcUb, maxUb, m, d, minValue, invalidValue);
}

template <typename T>
__simd_vf__ inline void ComputeLseOutputVF(__ubuf__ T *srcSumUb, __ubuf__ T *srcMaxUb, __ubuf__ T *dstUb, const uint32_t dealCount)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregMax;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<T> vregResFinal;
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<float> vregInfValue;
    MicroAPI::MaskReg pregCompare;
    constexpr uint32_t dealRows = 8;
    constexpr uint32_t  floatRepSize = 64; // 64: 一个寄存器存64个float
    constexpr float infValue = 3e+99; // 3e+99 for float inf
    constexpr uint32_t tmpMin = 0xFF7FFFFF;
    float minValue = *((float*)&tmpMin);
    uint16_t updateLoops = dealCount / dealRows;
    uint16_t tailLSize = dealCount % dealRows * 8;
    uint32_t pltTail = static_cast<uint32_t>(tailLSize);

    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<T>(pltTail);
    MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
    MicroAPI::Duplicate<float, float>(vregInfValue, infValue);

    for (uint16_t i = 0; i < updateLoops; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + (i * dealRows));
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregMax, srcMaxUb + (i * dealRows));

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregAll);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregAll);

        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
        MicroAPI::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + (i * floatRepSize), vregResFinal, pregAll);
    }

    if (tailLSize != 0) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregSum, srcSumUb + dealRows * updateLoops);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_E2B_B32>(vregMax, srcMaxUb + dealRows * updateLoops);

        MicroAPI::Log<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTail);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTail);

        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTail);
        MicroAPI::Select<T>(vregResFinal, vregInfValue, vregRes, pregCompare);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + floatRepSize * updateLoops, vregResFinal, pregTail);
    }
}

template <typename T>
__aicore__ inline void ComputeLseOutputVF(const LocalTensor<T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
    const LocalTensor<T>& softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T * srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T * srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ T * dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();

    ComputeLseOutputVF<T>(srcSumUb, srcMaxUb, dstUb, dealCount);
}

template <typename T>
__simd_vf__ inline void SinkSubExpAddVF(__ubuf__ T *srcSumUb, __ubuf__ T *srcMaxUb, const T sinkValue, const uint32_t dealCount)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregMax;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<T> vregSink;

    constexpr uint32_t floatRepSize = 64;

    uint16_t updateLoops = dealCount / floatRepSize;
    uint16_t tailSize = dealCount % floatRepSize;
    uint32_t pltTail = static_cast<uint32_t>(tailSize);

    //mask
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<T>(pltTail);

    Duplicate(vregSink, sinkValue);

    for (uint16_t i = 0; i < updateLoops; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum, srcSumUb + (i * floatRepSize));
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax, srcMaxUb + (i * floatRepSize));

        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSink, vregMax, pregAll);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, pregAll);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregSum, vregRes, pregAll);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(srcSumUb + (i * floatRepSize), vregSum, pregAll);
    }

    for (uint16_t i = 0; i < tailSize; i = i + tailSize) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum, srcSumUb + (updateLoops * floatRepSize));
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax, srcMaxUb + (updateLoops * floatRepSize));

        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSink, vregMax, pregTail);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, pregTail);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregSum, vregRes, pregTail);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(srcSumUb + (updateLoops * floatRepSize), vregSum, pregTail);
    }
}

template <typename T>
__aicore__ inline void SinkSubExpAddVF(const LocalTensor<T>& softmaxSumTensor, const LocalTensor<T>& softmaxMaxTensor,
    const T sinkValue, uint32_t dealCount)
{
    __ubuf__ T * srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T * srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();

    SinkSubExpAddVF<T>(srcSumUb, srcMaxUb, sinkValue, dealCount);
}

template <typename T, typename SINK_T>
__simd_vf__ inline void SinkSubExpAddGSFusedVF(__ubuf__ T *srcSumUb, __ubuf__ T *srcMaxUb, __ubuf__ uint16_t *sinkUb, const uint32_t dealCount)
{
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregMax;
    MicroAPI::RegTensor<T> vregRes;
    MicroAPI::RegTensor<SINK_T> vregSink;
    MicroAPI::RegTensor<T> vregSinkCast;

    constexpr uint32_t floatRepSize = 64;

    uint16_t updateLoops = dealCount / floatRepSize;
    uint16_t tailSize = dealCount % floatRepSize;
    uint32_t pltTail = static_cast<uint32_t>(tailSize);

    //mask
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<T>(pltTail);
    MicroAPI::MaskReg pregSinkAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>((MicroAPI::RegTensor<uint16_t>&)vregSink, sinkUb);
    MicroAPI::Cast<T, SINK_T, castTraitFp16_32_update>(vregSinkCast, vregSink, pregSinkAll);

    for (uint16_t i = 0; i < updateLoops; ++i) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum, srcSumUb + (i * floatRepSize));
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax, srcMaxUb + (i * floatRepSize));

        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSinkCast, vregMax, pregAll);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, pregAll);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregSum, vregRes, pregAll);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(srcSumUb + (i * floatRepSize), vregSum, pregAll);
    }

    if (tailSize != 0) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregSum, srcSumUb + (updateLoops * floatRepSize));
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregMax, srcMaxUb + (updateLoops * floatRepSize));

        MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregSinkCast, vregMax, pregTail);
        MicroAPI::Exp<T, MicroAPI::MaskMergeMode::ZEROING>(vregRes, vregRes, pregTail);
        MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregSum, vregRes, pregTail);

        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(srcSumUb + (updateLoops * floatRepSize), vregSum, pregTail);
    }
}

template <typename T, typename SINK_T>
__aicore__ inline void SinkSubExpAddGSFusedVF(const LocalTensor<SINK_T>& dstTensor, const LocalTensor<T>& softmaxSumTensor,
    const LocalTensor<T>& softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T * srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T * srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ uint16_t * dstUb = (__ubuf__ uint16_t *)dstTensor.GetPhyAddr();

    SinkSubExpAddGSFusedVF<T, SINK_T>(srcSumUb, srcMaxUb, dstUb, dealCount);
}

template <typename T>
__simd_vf__ inline void RowInvalidUpdateVF(__ubuf__ T *finalUb, __ubuf__ float *maxUb, const uint16_t m,
    const uint16_t d, int64_t dSize, const uint32_t pltTailD, const uint16_t hasTail)
{
    constexpr uint16_t floatRepSize = 64; // 64: 一个寄存器可以存储64个float类型数据
    const uint16_t dLoops = d / floatRepSize;


    constexpr uint32_t tmpZero = 0x00000000; // zero value of fp16 and fp32
    const T zeroValue = *((T*)&tmpZero);
    constexpr uint32_t tmpMin = 0xFF7FFFFF; // min value of float
    const float minValue = *((float*)&tmpMin);
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<T> vregZeroValue;
    MicroAPI::RegTensor<float> vregMax;
    MicroAPI::RegTensor<T> vregFinal;
    MicroAPI::RegTensor<T> vregFinalNew;

    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t tmpTailD = pltTailD;
    MicroAPI::MaskReg pregTailD = MicroAPI::UpdateMask<T>(tmpTailD);
    MicroAPI::MaskReg pregCompare;

    MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
    MicroAPI::Duplicate<T, T>(vregZeroValue, zeroValue);
    for (uint16_t i = 0; i < m; ++i) {
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i);
        MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + j * floatRepSize);
            MicroAPI::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(finalUb + i * dSize + j * floatRepSize,
                vregFinalNew, pregAll);
        }
        for (uint16_t t = 0; t < hasTail; ++t) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + dLoops * floatRepSize);
            MicroAPI::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(finalUb + i * dSize + dLoops * floatRepSize,
                vregFinalNew, pregTailD);
        }
    }
}

template <typename T>
__aicore__ inline void RowInvalidUpdateVF(const LocalTensor<T>& finalTensor, const LocalTensor<float>& maxTensor,
    const uint16_t m, const uint16_t d, int64_t dSize)
{
    __ubuf__ T * finalUb = (__ubuf__ T*)finalTensor.GetPhyAddr();
    __ubuf__ float * maxUb = (__ubuf__ float*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    RowInvalidUpdateVF<T>(finalUb, maxUb, m, d, dSize, pltTailD, hasTail);
}
} // namespace

#endif // MY_FLASH_UPDATE_INTERFACE_H
