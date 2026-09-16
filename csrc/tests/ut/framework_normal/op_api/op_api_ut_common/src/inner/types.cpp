/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <assert.h>
#include "op_api_ut_common/inner/types.h"

#include <map>

const std::string &String(aclDataType data_type)
{
    static std::map<aclDataType, std::string> dtype = {{ACL_DT_UNDEFINED, "undefined"},
                                                       {ACL_FLOAT, "float32"},
                                                       {ACL_FLOAT16, "float16"},
                                                       {ACL_INT8, "int8"},
                                                       {ACL_INT32, "int32"},
                                                       {ACL_UINT8, "uint8"},
                                                       {ACL_INT16, "int16"},
                                                       {ACL_UINT16, "uint16"},
                                                       {ACL_UINT32, "uint32"},
                                                       {ACL_INT64, "int64"},
                                                       {ACL_UINT64, "uint64"},
                                                       {ACL_DOUBLE, "double"},
                                                       {ACL_BOOL, "bool"},
                                                       {ACL_STRING, "string"},
                                                       {ACL_COMPLEX64, "complex64"},
                                                       {ACL_COMPLEX128, "complex128"},
                                                       {ACL_BF16, "bfloat16"}};
    auto iter = dtype.find(data_type);
    assert(iter != dtype.end());
    return iter->second;
}
