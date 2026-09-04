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
 * \file flash_attention_score_common.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_COMMON_REGBASE_H
#define FLASH_ATTENTION_SCORE_COMMON_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"
#include "pse.h"

using matmul::MatmulType;
using namespace AscendC;

constexpr uint64_t BLOCK_BYTE = 32;
constexpr int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t pse1NS1S2 = 2;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;
// 0级接口的block间隔范围需要满足32B对齐
constexpr int64_t attenMaskBN2GS1S2 = 0;
constexpr int64_t attenMaskBS1S2 = 1;
constexpr int64_t attenMaskS1S2 = 2;
constexpr int64_t attenMaskTT = 99;
constexpr uint16_t PREFIX_N_MAX_B = 32;
constexpr int32_t fp32BaseSize = 8;

constexpr uint32_t attenMaskNoCompress = 0;
constexpr uint32_t attenMaskLeftUpCausalCompress = 1;
constexpr uint32_t attenMaskRightDownCausalCompress = 2;
constexpr uint16_t SHIFT_NUM_2 = 2;
constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint32_t L0C_SHARED_SIZE_64K = 64 * 1024;
constexpr uint32_t L0C_SHARED_SIZE_128K = 128 * 1024;
constexpr uint32_t CV_RATIO = 1;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8}; // mm2ResIntraEvent
constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10}; //mm1ResIntraEvent
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint64_t BASE_SIZE_128 = 128;
constexpr uint64_t FLOAT_BYTES = 4;
enum class SparseModeEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9,
};

enum class ImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

static constexpr uint32_t FA_BYTE_BLOCK = 32;

__aicore__ constexpr uint16_t Align64Func(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

template <typename INPUT_T>
__aicore__ constexpr bool IsFp8OnlyWithAttenMask(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if constexpr (!IsSameType<INPUT_T, fp8_e5m2_t>::value &&
                  !IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
                  !IsSameType<INPUT_T, hifloat8_t>::value) {
        return false;
    }
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && hasAtten && !hasDrop) {
        return true;
    }
    return false;
}

__aicore__ constexpr bool ContainOptionalInput(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && !hasAtten && !hasDrop) {
        return false;
    } else {
        return true;
    }
}

__aicore__ constexpr bool IsDn(bool isFp32, bool isValidFp8, regbaseutil::PseTypeEnum pseMode, bool hasAtten,
                               bool hasDrop, bool isS1Base128, regbaseutil::DTemplateType dTemplateType, bool hasRope)
{
    if (!hasAtten && isS1Base128 && (uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned256) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool UbOutCondition(
    bool isFp32, regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool isS2Base64) {
    if (IsFp8OnlyWithAttenMask<INPUT_T>(pseMode, hasAtten, hasDrop)) {
        return true;
    }
    if (!ContainOptionalInput(pseMode, hasAtten, hasDrop)) {
        if (!isS2Base64 || isFp32) {
            return true;
        }
    }
    return false;
}

__aicore__ constexpr TPosition GetC2Position(regbaseutil::DTemplateType dTemplateType, bool ubOutCondition, bool isNdS2Size256) {
    if ((uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned128 ||
        (ubOutCondition && (uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned192) ||
        isNdS2Size256) {
        return TPosition::VECCALC;
    } else {
        return TPosition::GM;
    }
}
}
#endif // FLASH_ATTENTION_SCORE_COMMON_REGBASE_H