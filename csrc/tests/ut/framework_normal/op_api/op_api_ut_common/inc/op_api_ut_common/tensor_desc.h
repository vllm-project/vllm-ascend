/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _OP_API_UT_COMMON_TENSOR_DESC_H_
#define _OP_API_UT_COMMON_TENSOR_DESC_H_

#include <sstream>
#include <set>

#include "nlohmann/json.hpp"
#include "opdev/common_types.h"

using namespace std;
using namespace nlohmann;

class TensorDesc {
public:
    TensorDesc(const vector<int64_t> &view_dims = {1, 1, 16, 16}, aclDataType data_type = ACL_FLOAT,
               aclFormat format = ACL_FORMAT_NCHW, const vector<int64_t> &stride_ = {}, int64_t offset = 0,
               const vector<int64_t> &storage_dims = {});
    TensorDesc(const TensorDesc &desc);
    ~TensorDesc();

    unique_ptr<aclTensor, void (*)(aclTensor *)> ToAclType() const;
    aclTensor *ToAclTypeRawPtr() const;
    void ToJson(json &root, bool is_input = true) const;

    TensorDesc &ViewDims(const vector<int64_t> &view_dims);
    TensorDesc &Format(aclFormat format);

    TensorDesc &Precision(float relative, float percentage, float absolute = 1e-08);

    template <typename T>
    TensorDesc &ValueRange(T low, T high)
    {
        assert(low <= high);
        stringstream ss;
        if (typeid(T) == typeid(uint8_t)) {
            ss << "[" << (uint)low << "," << (uint)high << "]";
        } else if (typeid(T) == typeid(int8_t)) {
            ss << "[" << (int)low << "," << (int)high << "]";
        } else {
            ss << "[" << low << "," << high << "]";
        }
        ss >> value_range_;
        return *this;
    }

    template <typename T>
    TensorDesc &Value(const vector<T> &v)
    {
        stringstream ss;
        ss << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (typeid(T) == typeid(uint8_t)) {
                ss << (uint)v[i];
            } else if (typeid(T) == typeid(int8_t)) {
                ss << (int)v[i];
            } else {
                ss << v[i];
            }

            if (i < v.size() - 1)
                ss << ",";
        }
        ss << "]";
        ss >> value_;
        return *this;
    }

    TensorDesc &ValueFile(const std::string &binary_file);

    TensorDesc &ValidCount(int32_t cnt);

    const std::vector<int64_t> &GetViewDims() const;

private:
    int64_t GetViewCount() const;
    int64_t GetStorageCount() const;

private:
    vector<int64_t> view_dims_;
    aclDataType data_type_;
    aclFormat format_;
    vector<int64_t> stride_;
    int64_t offset_;
    vector<int64_t> storage_dims_;

    string precision_;
    string value_range_ = "[-2,2]";
    string value_;

    int32_t valid_count_ = -1;
};

class TensorListDesc {
public:
    TensorListDesc(const vector<TensorDesc> &val);
    TensorListDesc(int duplicate_cnt, const TensorDesc &tensor_desc);

    unique_ptr<aclTensorList, void (*)(aclTensorList *)> ToAclType() const;
    aclTensorList *ToAclTypeRawPtr() const;
    void ToJson(json &root, bool is_input = true) const;

private:
    vector<TensorDesc> arr_;
};

aclTensor *DescToAclContainer(const TensorDesc &tensor_desc);
aclTensorList *DescToAclContainer(const TensorListDesc &tensor_list_desc);

void Release(aclTensor *p);
void Release(aclTensorList *p);

aclTensor *InferAclType(const TensorDesc &tensor_desc);
aclTensorList *InferAclType(const TensorListDesc &tensor_list_desc);

void DescToJson(json &root, const TensorDesc &tensor_desc, bool is_input = true);
void DescToJson(json &root, const TensorListDesc &tensor_list_desc, bool is_input = true);

int ReloadDataFromBinaryFile(aclTensor *p, size_t index, size_t total_input, const string &file_prefix);
int ReloadDataFromBinaryFile(aclTensorList *p, size_t index, size_t total_input, const string &file_prefix);

int SaveResultToBinaryFile(aclTensor *p, size_t index, size_t total_input, const string &file_prefix,
                           const set<short int> &inplace_output);
int SaveResultToBinaryFile(aclTensorList *p, size_t index, size_t total_input, const string &file_prefix,
                           const set<short int> &inplace_output);

int SaveResultToBinaryFile(aclTensor *p, size_t index, const string &file_prefix);
int SaveResultToBinaryFile(aclTensorList *p, size_t index, const string &file_prefix);

#endif
