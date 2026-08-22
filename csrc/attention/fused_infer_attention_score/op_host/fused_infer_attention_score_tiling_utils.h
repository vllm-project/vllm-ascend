/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_utils.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_UTILS_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_UTILS_H

#include "fused_infer_attention_score_tiling_constants.h"

namespace optiling {
template <typename T>
inline auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

template <typename T>
inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
inline auto increGcd(T a, T b) -> T
{
    if (b == 0) {
        return a;
    }
    if (a % b == 0) {
        return b;
    }
    return increGcd(b, a % b);
}

inline std::string ToString(ge::DataType type)
{
    const auto it = arch35FIA::DATATYPE_TO_STRING_MAP.find(type);
    if (it != arch35FIA::DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("FusedInferAttentionScore", "datatype %d is not supported", type);
        return "UNDEFINED";
    }
}

const std::map<ge::Format, std::string> kFormatToStringMap = {
    {ge::FORMAT_NCHW, "NCHW"},
    {ge::FORMAT_NHWC, "NHWC"},
    {ge::FORMAT_ND, "ND"},
    {ge::FORMAT_NCDHW, "NCDHW"},
};

inline std::string FormatToSerialString(const ge::Format format)
{
    const auto it = kFormatToStringMap.find(static_cast<ge::Format>(GetPrimaryFormat(format)));
    if (it != kFormatToStringMap.end()) {
        if (HasSubFormat(format)) {
            return std::string(it->second + ":" + std::to_string(GetSubFormat(format))).c_str();
        }
        return it->second.c_str();
    } else {
        OP_LOGW("FusedInferAttentionScore", "[Check][Param] Format not support %d", format);
        return "RESERVED";
    }
}

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
inline std::string ToString(ge::Format format)
{
    return FormatToSerialString(format);
}

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

template <typename T>
static std::string ToString(const std::vector<T> &v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

inline std::string ToString(const std::vector<const gert::Shape *> &v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << ToString(ToVector(*v[i])) << ", ";
        }
        oss << ToString(ToVector(*v[v.size() - 1]));
    }
    oss << "]";
    return oss.str();
}

inline std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

inline std::string ToString(const gert::Shape &shape)
{
    return ToString(ToVector(shape));
}

} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_UTILS_H