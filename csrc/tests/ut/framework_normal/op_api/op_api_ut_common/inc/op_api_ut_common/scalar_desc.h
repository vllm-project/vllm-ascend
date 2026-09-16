/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _OP_API_UT_COMMON_SCALAR_DESC_H_
#define _OP_API_UT_COMMON_SCALAR_DESC_H_

#include "nlohmann/json.hpp"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"

using namespace std;
using namespace nlohmann;

class ScalarDesc {
public:
    ScalarDesc(bool val = false)
        : data_type_(ACL_BOOL)
    {
        val_.b = val;
    }
    ScalarDesc(int8_t val = 0)
        : data_type_(ACL_INT8)
    {
        SetInt8Value(&val);
    }
    ScalarDesc(uint8_t val = 0)
        : data_type_(ACL_UINT8)
    {
        SetInt8Value(&val);
    }
    ScalarDesc(int16_t val = 0)
        : data_type_(ACL_INT16)
    {
        SetInt16Value(&val);
    }
    ScalarDesc(uint16_t val = 0)
        : data_type_(ACL_UINT16)
    {
        SetInt16Value(&val);
    }
    ScalarDesc(int32_t val = 0)
        : data_type_(ACL_INT32)
    {
        SetInt32Value(&val);
    }
    ScalarDesc(uint32_t val = 0)
        : data_type_(ACL_UINT32)
    {
        SetInt32Value(&val);
    }
    ScalarDesc(int64_t val = 0)
        : data_type_(ACL_INT64)
    {
        SetInt64Value(&val);
    }
    ScalarDesc(uint64_t val = 0)
        : data_type_(ACL_UINT64)
    {
        SetInt64Value(&val);
    }
    ScalarDesc(float val = 0, aclDataType data_type = ACL_FLOAT);
    ScalarDesc(double val = 0)
        : data_type_(ACL_DOUBLE)
    {
        val_.d = val;
    }
    ~ScalarDesc() {}

    unique_ptr<aclScalar, void (*)(aclScalar *)> ToAclType() const;
    aclScalar *ToAclTypeRawPtr() const;
    void ToJson(json &root, bool is_input = true) const;

private:
    void SetInt8Value(void *v);
    void SetInt16Value(void *v);
    void SetInt32Value(void *v);
    void SetInt64Value(void *v);

private:
    union {
        bool b;
        uint64_t i64;
        uint32_t i32;
        uint16_t i16;
        uint8_t i8;
        float f;
        double d;
    } val_;
    op::fp16_t f16_;
    aclDataType data_type_;
};

aclScalar *DescToAclContainer(const ScalarDesc &scalar_desc);
void Release(aclScalar *p);
aclScalar *InferAclType(const ScalarDesc &scalar_desc);
void DescToJson(json &root, const ScalarDesc &scalar_desc, bool is_input = true);

#endif
