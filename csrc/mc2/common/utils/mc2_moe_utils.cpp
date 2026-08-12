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
 * \file mc2_moe_utils.cpp
 * \brief
 */

#include "mc2_moe_utils.h"
#include <algorithm>

namespace Mc2Moe {

enum class EpSize : int64_t {
    SIZE_2 = 2,
    SIZE_4 = 4,
    SIZE_8 = 8,
    SIZE_16 = 16,
    SIZE_32 = 32
};

enum class TpSize : int64_t {
    SIZE_2 = 2,
    SIZE_4 = 4,
    SIZE_8 = 8,
    SIZE_16 = 16,
    SIZE_32 = 32
};

static const std::vector<ge::DataType> DTYPE_SUPPORT_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};

static const std::vector<ge::DataType> BIAS_DTYPE_SUPPORT = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT};

// ep tp 值校验
bool EpTpSizeCheck(const int64_t epSize, const int64_t tpSize)
{
    if ((epSize != static_cast<int64_t>(EpSize::SIZE_2)) && (epSize != static_cast<int64_t>(EpSize::SIZE_4)) &&
        (epSize != static_cast<int64_t>(EpSize::SIZE_8)) && (epSize != static_cast<int64_t>(EpSize::SIZE_16)) &&
        (epSize != static_cast<int64_t>(EpSize::SIZE_32))) { // 当前 ep 仅支持 2 4 8 16 32
        return false;
    }
    if ((tpSize != static_cast<int64_t>(TpSize::SIZE_2)) && (tpSize != static_cast<int64_t>(TpSize::SIZE_4)) &&
        (tpSize != static_cast<int64_t>(TpSize::SIZE_8)) && (tpSize != static_cast<int64_t>(TpSize::SIZE_16)) &&
        (tpSize != static_cast<int64_t>(TpSize::SIZE_32))) { // 当前 tp 仅支持 2 4 8 16 32
        return false;
    }

    return true;
}

bool DimNumCheck(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape)
{
    if ((xShape->GetDimNum() != SUPPORT_DIM_NUM) || (weightShape->GetDimNum() != SUPPORT_DIM_NUM)) {
        OPS_LOG_E(nodeName, "Dim of input x and weight must be the same with %zu dims, but got dim x %zu, dim w %zu.",
                  SUPPORT_DIM_NUM, xShape->GetDimNum(), weightShape->GetDimNum());
        return false;
    }
    return true;
}

bool GroupCheck(const char *nodeName, const char *groupEp, const char *groupTp)
{
    const size_t maxGroupNameLength = 128UL;
    if ((groupEp == nullptr) || (strnlen(groupEp, maxGroupNameLength) == 0)) {
        OPS_LOG_E(nodeName, "groupEp is nullptr or empty.");
        return false;
    }

    if ((groupTp == nullptr) || (strnlen(groupTp, maxGroupNameLength) == 0)) {
        OPS_LOG_E(nodeName, "groupTp is nullptr or empty.");
        return false;
    }

    if (strncmp(groupEp, groupTp, maxGroupNameLength) == 0) {
        OPS_LOG_E(nodeName, "groupEp and groupTp can't be consistent.");
        return false;
    }
    return true;
}

void DynamicShapeCheck(const gert::Shape *xShape, const gert::Shape *weightShape, const size_t wDim,
                       OutShapeInfo &outShapeInfo)
{
    if (xShape->GetDim(DIM_E) == -1) {
        outShapeInfo.e = -1;
    }
    if (xShape->GetDim(X_DIM_C) == -1) {
        outShapeInfo.c = -1;
    }
    if (weightShape->GetDim(wDim) == -1) {
        outShapeInfo.h = -1;
    }
    return;
}

void EmptyShapeCheck(const gert::Shape *xShape, const gert::Shape *weightShape, const size_t wDim,
                     OutShapeInfo &outShapeInfo)
{
    if ((xShape->GetDim(DIM_E) == 0) || (weightShape->GetDim(DIM_E) == 0)) {
        outShapeInfo.e = 0;
    }
    if (xShape->GetDim(X_DIM_C) == 0) {
        outShapeInfo.c = 0;
    }
    if (weightShape->GetDim(wDim) == 0) {
        outShapeInfo.h = 0;
    }
    return;
}

bool CheckBiasDtype(const char *nodeName, const ge::DataType xType, const ge::DataType biasType)
{
    if (std::find(BIAS_DTYPE_SUPPORT.begin(), BIAS_DTYPE_SUPPORT.end(), biasType) == BIAS_DTYPE_SUPPORT.end()) {
        OPS_LOG_E(nodeName, "input bias support dtype is fp16 fp32, but got %u", static_cast<uint32_t>(biasType));
        return false;
    }

    // 两种情况: x = fp16, bias = fp16; x = bf16, bias = fp32
    if (xType == ge::DataType::DT_FLOAT16) {
        if (biasType != ge::DataType::DT_FLOAT16) {
            OPS_LOG_E(nodeName, "input x is fp16, bias must be fp16");
            return false;
        }
    } else if (xType == ge::DataType::DT_BF16) {
        if (biasType != ge::DataType::DT_FLOAT) {
            OPS_LOG_E(nodeName, "input x is bf16, bias must be fp32");
            return false;
        }
    }

    return true;
}

bool CheckTensorDtype(const char *nodeName, const ge::DataType xType, const ge::DataType weightType,
                      const ge::DataType biasType)
{
    if (std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(), xType) == DTYPE_SUPPORT_LIST.end()) {
        OPS_LOG_E(nodeName, "input x support dtype is fp16 bf16, but got %u", static_cast<uint32_t>(xType));
        return false;
    }
    if (std::find(DTYPE_SUPPORT_LIST.begin(), DTYPE_SUPPORT_LIST.end(), weightType) == DTYPE_SUPPORT_LIST.end()) {
        OPS_LOG_E(nodeName, "input weight support dtype is fp16 bf16, but got %u", static_cast<uint32_t>(weightType));
        return false;
    }
    if (xType != weightType) {
        OPS_LOG_E(nodeName, "input x and weight dtype not same");
        return false;
    }
    if (biasType != ge::DT_UNDEFINED) {
        OPS_LOG_D(nodeName, "need check bias type");
        if (!CheckBiasDtype(nodeName, xType, biasType)) {
            OPS_LOG_E(nodeName, "bias dtype check failed");
            return false;
        }
    }
    OPS_LOG_D(nodeName, "Dtype x [%u] weight [%u] bias [%u].", static_cast<uint32_t>(xType),
              static_cast<uint32_t>(weightType), static_cast<uint32_t>(biasType));
    OPS_LOG_I(nodeName, "check dtype success");
    return true;
}

void SetShape(gert::Shape *shape, const OutShapeInfo &outShapeInfo)
{
    shape->SetDimNum(SUPPORT_DIM_NUM);
    shape->SetDim(0, outShapeInfo.e);
    shape->SetDim(1, outShapeInfo.c);
    shape->SetDim(2, outShapeInfo.h); // 2 代表 shape 的第三维
    return;
}

void TransDimHMIdx(const bool isTransW, size_t &wDimH, size_t &wDimM)
{
    if (isTransW) {
        size_t tmp = wDimM;
        wDimM = wDimH;
        wDimH = tmp;
    }
    return;
}

void PrintTensorShape(const char *nodeName, const gert::Shape *shape, const char *shapeName)
{
    for (size_t i = 0; i < shape->GetDimNum(); i++) {
        OPS_LOG_D(nodeName, "%s %lu is %ld.", shapeName, i, shape->GetDim(i));
    }
    return;
}

} // namespace Mc2Moe
