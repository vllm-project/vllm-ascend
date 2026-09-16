/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/scalar_desc.h"

using namespace std;

static void AclScalarReleaseWrapper(aclScalar *p)
{
    Release(p);
}

void ScalarDesc::SetInt8Value(void *v)
{
    val_.i8 = *reinterpret_cast<const uint8_t *>(v);
}

void ScalarDesc::SetInt16Value(void *v)
{
    val_.i16 = *reinterpret_cast<const uint16_t *>(v);
}

void ScalarDesc::SetInt32Value(void *v)
{
    val_.i32 = *reinterpret_cast<const uint32_t *>(v);
}

void ScalarDesc::SetInt64Value(void *v)
{
    val_.i64 = *reinterpret_cast<const uint64_t *>(v);
}

ScalarDesc::ScalarDesc(float val, aclDataType data_type)
{
    assert(data_type == ACL_FLOAT || data_type == ACL_FLOAT16);
    data_type_ = data_type;
    if (data_type == ACL_FLOAT) {
        val_.f = val;
    } else {
        f16_ = val;
    }
}

unique_ptr<aclScalar, void (*)(aclScalar *)> ScalarDesc::ToAclType() const
{
    unique_ptr<aclScalar, decltype(&AclScalarReleaseWrapper)> p(this->ToAclTypeRawPtr(), AclScalarReleaseWrapper);
    return move(p);
}

aclScalar *ScalarDesc::ToAclTypeRawPtr() const
{
    switch (data_type_) {
        case ACL_BOOL:
            return aclCreateScalar((void *)&val_.b, data_type_);
        case ACL_FLOAT:
            return aclCreateScalar((void *)&val_.f, data_type_);
        case ACL_FLOAT16:
            return aclCreateScalar((void *)&f16_.val, data_type_);
        case ACL_INT8:
        case ACL_UINT8:
            return aclCreateScalar((void *)&val_.i8, data_type_);
        case ACL_INT16:
        case ACL_UINT16:
            return aclCreateScalar((void *)&val_.i16, data_type_);
        case ACL_INT32:
        case ACL_UINT32:
            return aclCreateScalar((void *)&val_.i32, data_type_);
        case ACL_INT64:
        case ACL_UINT64:
            return aclCreateScalar((void *)&val_.i64, data_type_);
        case ACL_DOUBLE:
            return aclCreateScalar((void *)&val_.d, data_type_);
        default:
            return nullptr;
    }
}

void ScalarDesc::ToJson(json &root, bool is_input) const
{
    (void)is_input;
    json x;
    stringstream ss;
    switch (data_type_) {
        case ACL_BOOL:
            ss << (val_.b ? "True" : "False");
            break;
        case ACL_FLOAT:
            ss << val_.f;
            break;
        case ACL_FLOAT16:
            ss << float(f16_);
            break;
        case ACL_INT8:
            ss << (int32_t)(static_cast<int8_t>(val_.i8));
            break;
        case ACL_UINT8:
            ss << static_cast<int32_t>(val_.i8);
            break;
        case ACL_INT16:
            ss << (static_cast<int16_t>(val_.i16));
            break;
        case ACL_UINT16:
            ss << val_.i16;
            break;
        case ACL_INT32:
            ss << (static_cast<int32_t>(val_.i32));
            break;
        case ACL_UINT32:
            ss << val_.i32;
            break;
        case ACL_INT64:
            ss << (static_cast<int64_t>(val_.i64));
            break;
        case ACL_UINT64:
            ss << val_.i64;
            break;
        case ACL_DOUBLE:
            ss << val_.d;
            break;
        default:
            break;
    }
    x["dtype"] = String(data_type_);
    x["value"] = ss.str();
    root.push_back(x);
}

//////////////////////////////////////

aclScalar *DescToAclContainer(const ScalarDesc &scalar_desc)
{
    return scalar_desc.ToAclTypeRawPtr();
}

void Release(aclScalar *p)
{
    if (p != nullptr) {
        delete p;
    }
}

aclScalar *InferAclType(const ScalarDesc &scalar_desc)
{
    (void)scalar_desc;
    return nullptr;
}

void DescToJson(json &root, const ScalarDesc &scalar_desc, bool is_input)
{
    scalar_desc.ToJson(root, is_input);
}
