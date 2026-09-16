/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_CSV_CASE_LOADER_H
#define OP_API_CSV_CASE_LOADER_H

#include "csv_case_load_utils.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

const std::unordered_map<std::string, aclDataType> ACL_DTYPE{{"UNDEFINED", ACL_DT_UNDEFINED},
                                                             {"FLOAT", ACL_FLOAT},
                                                             {"FLOAT16", ACL_FLOAT16},
                                                             {"INT8", ACL_INT8},
                                                             {"INT32", ACL_INT32},
                                                             {"UINT8", ACL_UINT8},
                                                             {"INT16", ACL_INT16},
                                                             {"UINT16", ACL_UINT16},
                                                             {"UINT32", ACL_UINT32},
                                                             {"INT64", ACL_INT64},
                                                             {"UINT64", ACL_UINT64},
                                                             {"DOUBLE", ACL_DOUBLE},
                                                             {"BOOL", ACL_BOOL},
                                                             {"STRING", ACL_STRING},
                                                             {"COMPLEX64", ACL_COMPLEX64},
                                                             {"COMPLEX128", ACL_COMPLEX128},
                                                             {"BF16", ACL_BF16},
                                                             {"INT4", ACL_INT4},
                                                             {"UINT1", ACL_UINT1},
                                                             {"COMPLEX32", ACL_COMPLEX32},
                                                             {"HIFLOAT8", ACL_HIFLOAT8},
                                                             {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
                                                             {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
                                                             {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
                                                             {"FLOAT6_E3M2", ACL_FLOAT6_E3M2},
                                                             {"FLOAT6_E2M3", ACL_FLOAT6_E2M3},
                                                             {"FLOAT4_E2M1", ACL_FLOAT4_E2M1},
                                                             {"FLOAT4_E1M2", ACL_FLOAT4_E1M2}};

const std::unordered_map<std::string, aclFormat> ACL_FORMAT{{"UNDEFINED", ACL_FORMAT_UNDEFINED},
                                                            {"NCHW", ACL_FORMAT_NCHW},
                                                            {"NHWC", ACL_FORMAT_NHWC},
                                                            {"ND", ACL_FORMAT_ND},
                                                            {"NC1HWC0", ACL_FORMAT_NC1HWC0},
                                                            {"FRACTAL_Z", ACL_FORMAT_FRACTAL_Z},
                                                            {"NC1HWC0_C04", ACL_FORMAT_NC1HWC0_C04},
                                                            {"HWCN", ACL_FORMAT_HWCN},
                                                            {"NDHWC", ACL_FORMAT_NDHWC},
                                                            {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
                                                            {"NCDHW", ACL_FORMAT_NCDHW},
                                                            {"NDC1HWC0", ACL_FORMAT_NDC1HWC0},
                                                            {"FRACTAL_Z_3D", ACL_FRACTAL_Z_3D},
                                                            {"NC", ACL_FORMAT_NC},
                                                            {"NCL", ACL_FORMAT_NCL},
                                                            {"FRACTAL_NZ_C0_16", ACL_FORMAT_FRACTAL_NZ_C0_16},
                                                            {"FRACTAL_NZ_C0_32", ACL_FORMAT_FRACTAL_NZ_C0_32},
                                                            {"FRACTAL_NZ_C0_2", ACL_FORMAT_FRACTAL_NZ_C0_2},
                                                            {"FRACTAL_NZ_C0_4", ACL_FORMAT_FRACTAL_NZ_C0_4},
                                                            {"FRACTAL_NZ_C0_8", ACL_FORMAT_FRACTAL_NZ_C0_8}};

const std::unordered_map<std::string, op::SocVersion> SOC_VERSION{
    {"Ascend910", op::SocVersion::ASCEND910},       {"Ascend910B", op::SocVersion::ASCEND910B},
    {"Ascend910_93", op::SocVersion::ASCEND910_93}, {"Ascend950", op::SocVersion::ASCEND950},
    {"Ascend910E", op::SocVersion::ASCEND910E},     {"Ascend310", op::SocVersion::ASCEND310},
    {"Ascend310P", op::SocVersion::ASCEND310P},     {"Ascend310B", op::SocVersion::ASCEND310B},
    {"Ascend310C", op::SocVersion::ASCEND310C},     {"Ascend610LITE", op::SocVersion::ASCEND610LITE},
    {"KirinX90", op::SocVersion::KIRINX90},         {"Kirin9030", op::SocVersion::KIRIN9030}};

const std::unordered_map<std::string, aclnnStatus> ACR_RET{{"SUCCESS", ACLNN_SUCCESS},
                                                           {"PARAM_NULLPTR", ACLNN_ERR_PARAM_NULLPTR},
                                                           {"PARAM_INVALID", ACLNN_ERR_PARAM_INVALID},
                                                           {"RUNTIME_ERROR", ACLNN_ERR_RUNTIME_ERROR}};

inline op::SocVersion GetCaseSocVersion(const csv_map &csvMap, const std::string &socKey)
{
    std::string socStr = ReadMap(csvMap, socKey);
    if (socStr.empty()) {
        return op::SocVersion::ASCEND910B;
    }

    return ReadMap(SOC_VERSION, socStr, op::SocVersion::ASCEND910B);
}

inline aclnnStatus GetAclnnRet(const csv_map &csvMap, const std::string &aclnnRetKey)
{
    std::string aclnnRetStr = ReadMap(csvMap, aclnnRetKey);
    if (aclnnRetStr.empty()) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ReadMap(ACR_RET, aclnnRetStr);
}

inline TensorDesc GetTensorACL(const csv_map &csvMap, const std::string &shapeKey, const std::string &dtypeKey,
                               const std::string &formatKey)
{
    aclDataType dtype = ReadMap(ACL_DTYPE, ReadMap(csvMap, dtypeKey), ACL_DT_UNDEFINED);
    aclFormat format = ReadMap(ACL_FORMAT, ReadMap(csvMap, formatKey), ACL_FORMAT_UNDEFINED);
    std::string shapeStr = ReadMap(csvMap, shapeKey);
    if (shapeStr.empty()) {
        return {{}, dtype, format};
    }
    return {GetShapeArr(shapeStr), dtype, format};
}

#endif // OP_API_CSV_CASE_LOADER_H
