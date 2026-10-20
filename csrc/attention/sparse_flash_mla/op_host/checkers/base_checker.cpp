/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_checker_sparse_flash_mla.h"
#include <algorithm>
#include <sstream>
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *GetOpName(const CheckContext &context) { return context.opName == nullptr ? "SparseMla" : context.opName; }

std::string ShapeToString(const gert::Shape *shape)
{
    if (shape == nullptr) {
        return "nullptr";
    }
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape->GetDim(i);
    }
    oss << ")";
    return oss.str();
}

template <typename T>
std::string ValuesToString(std::initializer_list<T> values)
{
    std::ostringstream oss;
    bool first = true;
    for (const auto value : values) {
        if (!first) {
            oss << ", ";
        }
        oss << static_cast<int64_t>(value);
        first = false;
    }
    return oss.str();
}
} // namespace

ge::graphStatus BaseChecker::CheckSinglePara(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckParaExistence(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckFeature(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckMultiPara(const CheckContext &context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckTensorDesc(const CheckContext &context, const TensorParam &param, const char *name,
                                             std::initializer_list<ge::DataType> dtypes) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GetOpName(context), name, "Tensor desc cannot be null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        param.desc->GetOriginFormat() != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(GetOpName(context), name,
                                   std::to_string(static_cast<int32_t>(param.desc->GetOriginFormat())).c_str(), "ND"),
        return ge::GRAPH_FAILED);
    const ge::DataType actual = param.desc->GetDataType();
    const std::string expectedDtypes = ValuesToString(dtypes);
    OP_CHECK_IF(std::find(dtypes.begin(), dtypes.end(), actual) == dtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE(GetOpName(context), name,
                                          std::to_string(static_cast<int32_t>(actual)).c_str(), expectedDtypes.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckDimNum(const CheckContext &context, const TensorParam &param, const char *name,
                                         std::initializer_list<size_t> dimNums) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GetOpName(context), name, "Tensor shape cannot be null"),
                return ge::GRAPH_FAILED);
    const size_t actual = param.shape->GetDimNum();
    const std::string expectedDimNums = ValuesToString(dimNums);
    OP_CHECK_IF(
        std::find(dimNums.begin(), dimNums.end(), actual) == dimNums.end(),
        OP_LOGE_FOR_INVALID_SHAPEDIM(GetOpName(context), name, std::to_string(actual).c_str(), expectedDimNums.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckNoEmptyDim(const CheckContext &context, const TensorParam &param,
                                             const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GetOpName(context), name, "Tensor shape cannot be null"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < param.shape->GetDimNum(); ++i) {
        OP_CHECK_IF(param.shape->GetDim(i) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetOpName(context), name, ShapeToString(param.shape).c_str(),
                                                          "Each dimension must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckShape(const CheckContext &context, const TensorParam &param, const char *name,
                                        std::initializer_list<int64_t> expected) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.shape == nullptr || param.shape->GetDimNum() != expected.size(),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetOpName(context), name, ShapeToString(param.shape).c_str(),
                                                      "Shape dim number does not match the documented shape"),
                return ge::GRAPH_FAILED);
    size_t index = 0;
    for (const int64_t value : expected) {
        OP_CHECK_IF(param.shape->GetDim(index) != value,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        GetOpName(context), name, ShapeToString(param.shape).c_str(),
                        ("Dimension " + std::to_string(index) + " must be " + std::to_string(value)).c_str()),
                    return ge::GRAPH_FAILED);
        ++index;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckSameShape(const CheckContext &context, const TensorParam &left, const char *leftName,
                                            const TensorParam &right, const char *rightName) const
{
    OP_CHECK_IF(
        left.shape == nullptr || right.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            GetOpName(context), (std::string(leftName) + " and " + rightName).c_str(), "Tensor shape cannot be null"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(left.shape->GetDimNum() != right.shape->GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    GetOpName(context), (std::string(leftName) + " and " + rightName).c_str(),
                    (ShapeToString(left.shape) + " and " + ShapeToString(right.shape)).c_str(),
                    "Tensor dim numbers must be the same"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < left.shape->GetDimNum(); ++i) {
        OP_CHECK_IF(left.shape->GetDim(i) != right.shape->GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        GetOpName(context), (std::string(leftName) + " and " + rightName).c_str(),
                        (ShapeToString(left.shape) + " and " + ShapeToString(right.shape)).c_str(),
                        ("Dimension " + std::to_string(i) + " must be the same").c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

int64_t BaseChecker::GetDim(const TensorParam &param, size_t index) const
{
    if (param.shape == nullptr || index >= param.shape->GetDimNum()) {
        return -1;
    }
    return param.shape->GetDim(index);
}

bool BaseChecker::CanOmitSequsedOriKv(const CheckContext &context) const { return context.oriTopkLength.present; }

bool BaseChecker::CanOmitSequsedCmpKv(const CheckContext &context) const { return context.cmpTopkLength.present; }

} // namespace sparse_mla_checker
} // namespace optiling
