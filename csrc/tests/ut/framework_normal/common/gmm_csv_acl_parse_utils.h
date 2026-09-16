/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_ACL_PARSE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_ACL_PARSE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/common_types.h"
#include "gmm_csv_parse_utils.h"

namespace ops::ut {

inline aclDataType ParseAclDtype(const std::string &dtype)
{
    static const std::map<std::string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},
        {"FLOAT16", ACL_FLOAT16},
        {"BF16", ACL_BF16},
        {"INT8", ACL_INT8},
        {"INT4", ACL_INT4},
        {"INT32", ACL_INT32},
        {"INT64", ACL_INT64},
        {"UINT64", ACL_UINT64},
        {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"HIFLOAT8", ACL_HIFLOAT8},
        {"FLOAT4_E2M1", ACL_FLOAT4_E2M1},
        {"FLOAT4_E1M2", ACL_FLOAT4_E1M2},
        {"UNDEFINED", ACL_DT_UNDEFINED},
    };
    const auto it = dtypeMap.find(Trim(dtype));
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

inline aclnnStatus ParseAclnnStatus(const std::string &status)
{
    static const std::map<std::string, aclnnStatus> statusMap = {
        {"ACLNN_SUCCESS", ACLNN_SUCCESS},
        {"ACLNN_ERR_PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
        {"ACLNN_ERR_PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
        {"ACLNN_ERR_INNER_NULLPTR", ACLNN_ERR_INNER_NULLPTR},
        {"ACLNN_ERR_INNER", ACLNN_ERR_INNER},
    };
    const auto trimmed = Trim(status);
    const auto it = statusMap.find(trimmed);
    if (it != statusMap.end()) {
        return it->second;
    }
    return static_cast<aclnnStatus>(std::stoll(trimmed));
}

inline aclFormat ParseAclFormat(const std::string &format)
{
    static const std::map<std::string, aclFormat> formatMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
        {"FRACTAL_NZ_C0_2", ACL_FORMAT_FRACTAL_NZ_C0_2},
        {"FRACTAL_NZ_C0_4", ACL_FORMAT_FRACTAL_NZ_C0_4},
        {"UNDEFINED", ACL_FORMAT_UNDEFINED},
    };
    const auto it = formatMap.find(Trim(format));
    return it == formatMap.end() ? ACL_FORMAT_ND : it->second;
}

struct AclTensorSpec {
    std::vector<int64_t> viewDims;
    std::vector<int64_t> stride;
    std::vector<int64_t> storageDims;
};

inline AclTensorSpec ParseAclTensorSpec(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> sections;
    SplitStr2Vec(trimmed, "#", sections);

    AclTensorSpec spec;
    if (!sections.empty()) {
        spec.viewDims = ParseDims(sections[0]);
    }
    if (sections.size() > 1U) {
        spec.stride = ParseDims(sections[1]);
    }
    if (sections.size() > 2U) {
        spec.storageDims = ParseDims(sections[2]);
    }
    return spec;
}

inline std::vector<int64_t> ParseAclTensorViewDims(const std::string &value)
{
    return ParseAclTensorSpec(value).viewDims;
}

inline std::vector<std::string> ParseAclTensorListSpecs(const std::string &value)
{
    std::vector<std::string> specs;
    SplitStr2Vec(Trim(value), ";", specs);
    if (specs.empty()) {
        specs.emplace_back("NONE");
    }
    for (auto &spec : specs) {
        spec = Trim(spec);
        if (spec.empty()) {
            spec = "NONE";
        }
    }
    return specs;
}

inline TensorDesc MakeAclTensorDesc(const std::vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                                    const std::vector<int64_t> &storage_shape = {},
                                    const std::vector<int64_t> &stride = {})
{
    return TensorDesc(shape, dtype, format, stride, 0, storage_shape).ValueRange(-10, 10);
}

inline TensorDesc BuildAclTensorDescFromSpec(const std::string &shape_spec, const std::string &dtype,
                                             const std::string &format)
{
    const AclTensorSpec spec = ParseAclTensorSpec(shape_spec);
    return TensorDesc(spec.viewDims, ParseAclDtype(dtype), ParseAclFormat(format), spec.stride, 0, spec.storageDims);
}

inline std::vector<int64_t> GetOptionalDims(const std::vector<std::vector<int64_t>> &dims_list, size_t index)
{
    return index < dims_list.size() ? dims_list[index] : std::vector<int64_t>{};
}

inline TensorListDesc BuildAclTensorListDesc(const std::vector<std::vector<int64_t>> &shapes, aclDataType dtype,
                                             aclFormat format,
                                             const std::vector<std::vector<int64_t>> &storage_shapes = {},
                                             const std::vector<std::vector<int64_t>> &strides = {})
{
    std::vector<TensorDesc> tensor_vec;
    tensor_vec.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        tensor_vec.emplace_back(MakeAclTensorDesc(shapes[i], dtype, format, GetOptionalDims(storage_shapes, i),
                                                  GetOptionalDims(strides, i)));
    }
    return TensorListDesc(tensor_vec);
}

inline TensorListDesc BuildAclTensorListDesc(const std::string &shape_spec_list, const std::string &dtype,
                                             const std::string &format, bool with_value_range = true)
{
    std::vector<TensorDesc> tensors;
    for (const auto &shape_spec : ParseAclTensorListSpecs(shape_spec_list)) {
        TensorDesc tensor = BuildAclTensorDescFromSpec(shape_spec, dtype, format);
        if (with_value_range) {
            tensor.ValueRange(-10, 10);
        }
        tensors.emplace_back(tensor);
    }
    return TensorListDesc(tensors);
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_CSV_ACL_PARSE_UTILS_H
