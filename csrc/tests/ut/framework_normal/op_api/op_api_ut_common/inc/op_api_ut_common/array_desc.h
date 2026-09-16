/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _OP_API_UT_COMMON_ARRAY_DESC_H_
#define _OP_API_UT_COMMON_ARRAY_DESC_H_

#include <sstream>
#include <vector>
#include "nlohmann/json.hpp"

#include "opdev/common_types.h"

using namespace std;
using namespace nlohmann;

#define ARRAY_DESC(TB, TS) \
    inline void Release(acl##TB##Array *p) \
    { \
        if (p != nullptr) { \
            aclDestroy##TB##Array(p); \
        } \
    } \
    class TB##ArrayDesc { \
    public: \
        TB##ArrayDesc(const vector<TS> &val) \
        { \
            arr_ = val; \
        } \
        TB##ArrayDesc(int duplicate_cnt, TS single_val) \
        { \
            assert(duplicate_cnt > 0); \
            arr_ = move(vector<TS>(duplicate_cnt, single_val)); \
        } \
        ~TB##ArrayDesc() {} \
        unique_ptr<acl##TB##Array, void (*)(acl##TB##Array *)> ToAclType() const; \
        acl##TB##Array *ToAclTypeRawPtr() const; \
        void ToJson(json &root, bool is_input = true) const \
        { \
            (void)is_input; \
            json x; \
            if (0 == strcmp(#TS, "int64_t")) { \
                x["dtype"] = "list_int"; \
            } else { \
                x["dtype"] = "list_" #TS; \
            } \
            stringstream ss; \
            auto arr = Get(); \
            ss << "["; \
            for (size_t i = 0; i < arr.size(); ++i) { \
                if (0 == strcmp(#TS, "bool")) { \
                    ss << (arr[i] ? "True" : "False"); \
                } else { \
                    ss << arr[i]; \
                } \
                if (i < arr.size() - 1) { \
                    ss << ","; \
                } \
            } \
            ss << "]"; \
            x["value"] = ss.str(); \
            root.push_back(x); \
        } \
        const vector<TS> &Get() const \
        { \
            return arr_; \
        } \
\
    private: \
        vector<TS> arr_; \
    }; \
    inline acl##TB##Array *DescToAclContainer(const TB##ArrayDesc &array_desc) \
    { \
        return array_desc.ToAclTypeRawPtr(); \
    } \
    inline acl##TB##Array *InferAclType(const TB##ArrayDesc &array_desc) \
    { \
        (void)array_desc; \
        return nullptr; \
    } \
    inline void DescToJson(json &root, const TB##ArrayDesc &array_desc, bool is_input = true) \
    { \
        array_desc.ToJson(root, is_input); \
    }

ARRAY_DESC(Int, int64_t)
ARRAY_DESC(Float, float)
ARRAY_DESC(Bool, bool)

#endif
