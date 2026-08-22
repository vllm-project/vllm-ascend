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
 * \file base_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "base_checker.h"
namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

ge::graphStatus BaseChecker::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE("FusedInferAttentionScore",
                            "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        if (std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end()) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BaseChecker::CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        if ((FORMAT_SUPPORT_SET.find(format) == FORMAT_SUPPORT_SET.end())) {
            OP_LOGE_FOR_INVALID_FORMAT("FusedInferAttentionScore", name.c_str(),
                ToString(format).c_str(), "ND, NCHW, NHWC or NCDHW");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 判断tensor是否连续
ge::graphStatus BaseChecker::CheckTensorContiguous(const uint32_t &tensorDimNum, const gert::Shape &inputShape,
        const gert::Stride *Strides, int32_t &index) const
{
    // 根据kv kvscale krope算出连续场景的strides，如果存在某一维不相等则表示不连续
    if (Strides == nullptr || Strides->GetDimNum() == 0){
        return ge::GRAPH_SUCCESS;
    }
    // 维度为1的tensor始终连续
    if (tensorDimNum == 0 || tensorDimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t preStride = 1; // 连续场景最后一维的stride默认为1
    for (index = tensorDimNum - 1; index >= 0; index--) {
        if (inputShape.GetDim(index) == 1) { // dim=1时步长不影响连续性
            continue;
        }
        if (preStride != Strides->GetStride(index)) {
            return ge::GRAPH_FAILED;
        }
        preStride *= inputShape.GetDim(index);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus BaseChecker::CheckValueSupport(const T value, const std::vector<T> &expectValList) const
{
    if (std::find(expectValList.begin(), expectValList.end(), value) == expectValList.end()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

std::string BaseChecker::DataTypeToSerialString(ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("FusedInferAttentionScore", "datatype %d is not supported", type);
        return "UNDEFINED";
    }
}

// explicit instantiation
template ge::graphStatus BaseChecker::CheckValueSupport(
    const std::tuple<ge::DataType, ge::DataType, ge::DataType> value,
    const std::vector<std::tuple<ge::DataType, ge::DataType, ge::DataType>> &expectValList) const;
template ge::graphStatus BaseChecker::CheckValueSupport(const int32_t value,
                                                        const std::vector<int32_t> &expectValList) const;
}
