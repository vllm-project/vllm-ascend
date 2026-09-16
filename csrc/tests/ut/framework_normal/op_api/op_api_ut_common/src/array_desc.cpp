/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/common_types.h"

#include "op_api_ut_common/array_desc.h"

using namespace std;

static void AclIntArrayReleaseWrapper(aclIntArray *p)
{
    Release(p);
}

unique_ptr<aclIntArray, void (*)(aclIntArray *)> IntArrayDesc::ToAclType() const
{
    unique_ptr<aclIntArray, decltype(&AclIntArrayReleaseWrapper)> p(this->ToAclTypeRawPtr(), AclIntArrayReleaseWrapper);
    return move(p);
}

aclIntArray *IntArrayDesc::ToAclTypeRawPtr() const
{
    return new aclIntArray(arr_.data(), arr_.size());
}

////////////////////////////////////

static void AclFloatArrayReleaseWrapper(aclFloatArray *p)
{
    Release(p);
}

unique_ptr<aclFloatArray, void (*)(aclFloatArray *)> FloatArrayDesc::ToAclType() const
{
    unique_ptr<aclFloatArray, decltype(&AclFloatArrayReleaseWrapper)> p(this->ToAclTypeRawPtr(),
                                                                        AclFloatArrayReleaseWrapper);
    return move(p);
}

aclFloatArray *FloatArrayDesc::ToAclTypeRawPtr() const
{
    return new aclFloatArray(arr_.data(), arr_.size());
}

////////////////////////////////////

static void AclBoolArrayReleaseWrapper(aclBoolArray *p)
{
    Release(p);
}

unique_ptr<aclBoolArray, void (*)(aclBoolArray *)> BoolArrayDesc::ToAclType() const
{
    unique_ptr<aclBoolArray, decltype(&AclBoolArrayReleaseWrapper)> p(this->ToAclTypeRawPtr(),
                                                                      AclBoolArrayReleaseWrapper);
    return move(p);
}

aclBoolArray *BoolArrayDesc::ToAclTypeRawPtr() const
{
    auto p = new bool[arr_.size()];
    for (size_t i = 0; i < arr_.size(); ++i) {
        p[i] = arr_[i];
    }
    auto aclPtr = new aclBoolArray(p, arr_.size());
    delete[] p;
    return aclPtr;
}
