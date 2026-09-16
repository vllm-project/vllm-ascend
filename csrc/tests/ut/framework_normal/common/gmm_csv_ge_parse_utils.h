/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_GE_PARSE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_GE_PARSE_UTILS_H

#include <map>
#include <string>
#include <vector>

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include <graph/types.h>

#include "gmm_csv_parse_utils.h"

namespace ops::ut {

inline ge::DataType ParseGeDtype(const std::string &dtype)
{
    static const std::map<std::string, ge::DataType> dtypeMap = {
        {"FLOAT", ge::DT_FLOAT},
        {"FLOAT16", ge::DT_FLOAT16},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"INT32", ge::DT_INT32},
        {"INT64", ge::DT_INT64},
        {"UINT64", ge::DT_UINT64},
        {"FLOAT8_E4M3FN", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
        {"HIFLOAT8", ge::DT_HIFLOAT8},
        {"FLOAT4_E2M1", ge::DT_FLOAT4_E2M1},
        {"FLOAT4_E1M2", ge::DT_FLOAT4_E1M2},
        {"UNDEFINED", ge::DT_UNDEFINED},
    };
    const auto it = dtypeMap.find(Trim(dtype));
    return it == dtypeMap.end() ? ge::DT_UNDEFINED : it->second;
}

inline ge::Format ParseGeFormat(const std::string &format)
{
    static const std::map<std::string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"NCL", ge::FORMAT_NCL},
        {"NCHW", ge::FORMAT_NCHW},
        {"NDC1HWC0", ge::FORMAT_NDC1HWC0},
        {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    };
    const auto it = formatMap.find(Trim(format));
    return it == formatMap.end() ? ge::FORMAT_ND : it->second;
}

inline gert::Shape BuildGertShape(const std::vector<int64_t> &dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

inline gert::StorageShape MakeGertStorageShape(const std::vector<int64_t> &origin_dims,
                                               const std::vector<int64_t> &storage_dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildGertShape(origin_dims);
    shape.MutableStorageShape() = BuildGertShape(storage_dims);
    return shape;
}

inline gert::StorageShape MakeGertStorageShape(const std::vector<int64_t> &dims)
{
    return MakeGertStorageShape(dims, dims);
}

inline gert::StorageShape MakeGertStorageShape(const std::string &dims)
{
    return MakeGertStorageShape(ParseDims(dims));
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_GE_PARSE_UTILS_H
