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
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"

#include "op_api_ut_common/inner/c_shell.h"
#include "op_api_ut_common/inner/file_io.h"
#include "op_api_ut_common/inner/rts_interface.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace nlohmann;

static void AclTensorReleaseWrapper(aclTensor *p)
{
    Release(p);
}

static void AclTensorListReleaseWrapper(aclTensorList *p)
{
    Release(p);
}

TensorDesc::TensorDesc(const vector<int64_t> &view_dims, aclDataType data_type, aclFormat format,
                       const vector<int64_t> &stride, int64_t offset, const vector<int64_t> &storage_dims)
    : view_dims_(view_dims), data_type_(data_type), format_(format), stride_(stride), offset_(offset),
      storage_dims_(storage_dims)
{
}

TensorDesc::~TensorDesc()
{
}

TensorDesc::TensorDesc(const TensorDesc &desc)
{
    view_dims_ = desc.view_dims_;
    data_type_ = desc.data_type_;
    format_ = desc.format_;
    stride_ = desc.stride_;
    offset_ = desc.offset_;
    storage_dims_ = desc.storage_dims_;

    precision_ = desc.precision_;
    value_range_ = desc.value_range_;
    value_ = desc.value_;
    valid_count_ = desc.valid_count_;
}

unique_ptr<aclTensor, void (*)(aclTensor *)> TensorDesc::ToAclType() const
{
    unique_ptr<aclTensor, decltype(&AclTensorReleaseWrapper)> p(this->ToAclTypeRawPtr(), AclTensorReleaseWrapper);
    return move(p);
}

aclTensor *TensorDesc::ToAclTypeRawPtr() const
{
    void *storage_data = nullptr;
    const auto &storage_dims = storage_dims_.empty() ? view_dims_ : storage_dims_;
    aclTensor *acl_tensor =
        aclCreateTensor(view_dims_.data(), view_dims_.size(), data_type_, stride_.empty() ? nullptr : stride_.data(),
                        offset_, format_, storage_dims.data(), storage_dims.size(), storage_data);
    assert(acl_tensor != nullptr);
    return acl_tensor;
}

void TensorDesc::ToJson(json &root, bool is_input) const
{
    json x;
    x["dtype"] = String(data_type_);
    x["format"] = op::ToString(static_cast<op::Format>(format_)).GetString();
    x["view_shape"] = view_dims_;
    x["storage_shape"] = storage_dims_.empty() ? view_dims_ : storage_dims_;
    x["offset"] = offset_;

    if (!stride_.empty()) {
        x["stride"] = stride_;
    }

    if (is_input) {
        x["value_range"] = value_range_;
        if (!value_.empty()) {
            x["value"] = value_;
        }
    }

    if (!precision_.empty()) {
        x["precision"] = precision_; // in case of inplace
    }

    if (valid_count_ >= 0) {
        x["valid_count"] = valid_count_; // count in output need to check.
    }
    root.push_back(x);
}

TensorDesc &TensorDesc::ViewDims(const vector<int64_t> &view_dims)
{
    view_dims_ = view_dims;
    return *this;
}

TensorDesc &TensorDesc::Format(aclFormat format)
{
    format_ = format;
    return *this;
}

TensorDesc &TensorDesc::Precision(float relative, float percentage, float absolute)
{
    assert(relative >= 0 && percentage >= 0 && absolute >= 0);
    stringstream ss;
    ss << "(" << relative << "," << percentage << "," << absolute << ")";
    ss >> precision_;
    return *this;
}

TensorDesc &TensorDesc::ValueFile(const string &binary_file)
{
    static string suffix = ".bin";
    assert(binary_file.length() < suffix.length() &&
           binary_file.compare(binary_file.length() - suffix.length(), suffix.length(), suffix) == 0 &&
           "binary file must end with `.bin`");
    value_ = binary_file;
    return *this;
}

TensorDesc &TensorDesc::ValidCount(int32_t cnt)
{
    valid_count_ = cnt;
    return *this;
}

const std::vector<int64_t> &TensorDesc::GetViewDims() const
{
    return view_dims_;
}

int64_t TensorDesc::GetViewCount() const
{
    const auto &v = view_dims_;
    return accumulate(v.cbegin(), v.cend(), 1, multiplies<int64_t>());
}

int64_t TensorDesc::GetStorageCount() const
{
    const auto &v = storage_dims_.empty() ? view_dims_ : storage_dims_;
    return accumulate(v.cbegin(), v.cend(), 1, multiplies<int64_t>());
}

////////////////////////////////////////////////////////////////////////

TensorListDesc::TensorListDesc(const vector<TensorDesc> &val)
{
    assert(val.size() > 0);
    arr_ = val;
}

TensorListDesc::TensorListDesc(int duplicate_cnt, const TensorDesc &tensor_desc)
{
    assert(duplicate_cnt > 0);
    vector<TensorDesc> arr(duplicate_cnt, tensor_desc);
    arr_ = move(arr);
}

unique_ptr<aclTensorList, void (*)(aclTensorList *)> TensorListDesc::ToAclType() const
{
    unique_ptr<aclTensorList, decltype(&AclTensorListReleaseWrapper)> p(this->ToAclTypeRawPtr(),
                                                                        AclTensorListReleaseWrapper);
    return move(p);
}

aclTensorList *TensorListDesc::ToAclTypeRawPtr() const
{
    assert(arr_.size() > 0);
    aclTensor **ts = new aclTensor *[arr_.size()];
    for (size_t i = 0; i < arr_.size(); i++)
        ts[i] = arr_[i].ToAclTypeRawPtr();
    auto tensor_list = aclCreateTensorList(ts, arr_.size());
    return tensor_list;
}

void TensorListDesc::ToJson(json &root, bool is_input) const
{
    json x = json::array();
    for_each(arr_.cbegin(), arr_.cend(), [&x, is_input](const TensorDesc &c) { c.ToJson(x, is_input); });
    root.push_back(x);
}

/////////////////////////////////////////////////////////////////////

aclTensor *DescToAclContainer(const TensorDesc &tensor_desc)
{
    return tensor_desc.ToAclTypeRawPtr();
}

aclTensorList *DescToAclContainer(const TensorListDesc &tensor_list_desc)
{
    return tensor_list_desc.ToAclTypeRawPtr();
}

/////////////////////////////////////////////////////////////////////

void Release(aclTensor *p)
{
    if (p != nullptr) {
        if (p->GetPlacement() == op::TensorPlacement::kOnDeviceHbm) {
            FreeDeviceMemory(const_cast<void *>(reinterpret_cast<const void *>(p->GetStorageAddr())));
        }
        aclDestroyTensor(p);
    }
}

void Release(aclTensorList *p)
{
    if (p != nullptr) {
        auto tensors = p->GetData();
        auto sze = p->Size();
        if (tensors != nullptr && sze > 0) {
            for (size_t i = 0; i < sze; i++) {
                if (tensors[i]->GetPlacement() == op::TensorPlacement::kOnDeviceHbm) {
                    FreeDeviceMemory(const_cast<void *>(reinterpret_cast<const void *>(tensors[i]->GetStorageAddr())));
                }
            }
        }
        aclDestroyTensorList(p);
    }
}

//////////////////////////////////////////////////////////////////////////

aclTensor *InferAclType(const TensorDesc &tensor_desc)
{
    (void)tensor_desc;
    return nullptr;
}

aclTensorList *InferAclType(const TensorListDesc &tensor_list_desc)
{
    (void)tensor_list_desc;
    return nullptr;
}

///////////////////////////////////////////////////////////////////////////

void DescToJson(json &root, const TensorDesc &tensor_desc, bool is_input)
{
    tensor_desc.ToJson(root, is_input);
}

void DescToJson(json &root, const TensorListDesc &tensor_list_desc, bool is_input)
{
    tensor_list_desc.ToJson(root, is_input);
}


///////////////////////////////////////////////////////////////////////////

static void *MallocAndMemcpyDeviceMemory(void *host_mem, size_t size)
{
    void *dev_mem = MallocDeviceMemory(size);
    if (dev_mem == nullptr) {
        return nullptr;
    }

    auto ret = MemcpyToDevice(host_mem, dev_mem, size);
    if (ret != 0) {
        FreeDeviceMemory(dev_mem);
        dev_mem = nullptr;
    }
    return dev_mem;
}

void SetTensorStorageData(aclTensor *p, void *dev_mem)
{
    assert(p->GetPlacement() == op::TensorPlacement::kOnDeviceHbm);
    FreeDeviceMemory(const_cast<void *>(reinterpret_cast<const void *>(p->GetStorageAddr())));
    p->SetStorageAddr(dev_mem);
}

static int LoadBinaryFile(aclTensor *p, const string &binary_file)
{
    size_t size = 0;
    void *host_mem = ReadBinFile(binary_file, size);
    if (host_mem == nullptr) {
        return 1;
    }
    cout << "Reload binary file size: " << size << endl;

    unique_ptr<char[]> ptr(static_cast<char *>(host_mem));
    void *dev_mem = nullptr;
    if (size > 0) {
        dev_mem = MallocAndMemcpyDeviceMemory(host_mem, size);
        if (dev_mem == nullptr) {
            return 1;
        }
    }
    SetTensorStorageData(p, dev_mem);
    return 0;
}

static int MallocDeviceMemoryOnly(aclTensor *p)
{
    auto size = p->Size() * op::TypeSize(p->GetDataType());
    cout << "Malloc device memory size: " << size << endl;
    void *dev_mem = nullptr;
    if (size > 0) {
        dev_mem = MallocDeviceMemory(size);
        if (dev_mem == nullptr) {
            return 1;
        }
    }
    SetTensorStorageData(p, dev_mem);
    return 0;
}

int ReloadDataFromBinaryFile(aclTensor *p, size_t index, size_t total_input, const string &file_prefix)
{
    if (p == nullptr)
        return 0;
    if (index < total_input) {
        cout << "Reload input for tensor. Index: " << index << ". ";
        string file_name = file_prefix + "_input_" + to_string(index) + ".bin";
        return LoadBinaryFile(p, file_name);
    } else {
        cout << "Reload output for tensor. Index: " << index << ". ";
        return MallocDeviceMemoryOnly(p);
    }
}

int ReloadDataFromBinaryFile(aclTensorList *p, size_t index, size_t total_input, const string &file_prefix)
{
    if (p == nullptr)
        return 0;
    uint64_t size = p->Size();
    if (index < total_input) {
        cout << "Reload input for tensor list. Index: " << index << ". ";
        for (uint64_t i = 0; i < size; i++) {
            auto acl_tensor = (*p)[i];
            string file_name = file_prefix + "_input_" + to_string(index) + "_" + to_string(i) + ".bin";
            if (0 != LoadBinaryFile(acl_tensor, file_name)) {
                return 1;
            }
        }
    } else {
        cout << "Reload output for tensor list. Index: " << index << ". ";
        for (uint64_t i = 0; i < size; i++) {
            auto acl_tensor = (*p)[i];
            if (0 != MallocDeviceMemoryOnly(acl_tensor)) {
                return 1;
            }
        }
    }
    return 0;
}

/////////////////////////////////

static int SaveToBinaryFile(aclTensor *p, const string &binary_file)
{
    auto size = p->Size() * op::TypeSize(p->GetDataType());
    void *host_mem = static_cast<void *>(new (nothrow) char[size]);
    if (host_mem == nullptr) {
        cout << "alloc host memory failed. need memory: " << size << endl;
        return 1;
    }
    memset_s(host_mem, size, 0, size);

    unique_ptr<char[]> ptr(static_cast<char *>(host_mem));
    auto ret = 0;
    if (size > 0) {
        ret = MemcpyFromDevice(host_mem, p->GetStorageAddr(), size);
    }
    if (ret == 0) {
        ret = WriteBinFile(host_mem, binary_file, size);
    }
    return ret;
}

static short int CalcRealOutputIndex(short int index, const set<short int> &inplace_output)
{
    short int diff = 0;
    short int le = 0;
    do {
        diff = le;
        short int cmp = index + le;
        le = inplace_output.count(cmp) + distance(inplace_output.begin(), inplace_output.lower_bound(cmp));
    } while (diff < le);
    return index + le;
}

int SaveResultToBinaryFile(aclTensor *p, size_t index, const string &file_prefix)
{
    string file_name = file_prefix + "_output_" + to_string(index) + ".bin";
    return SaveToBinaryFile(p, file_name);
}

int SaveResultToBinaryFile(aclTensorList *p, size_t index, const string &file_prefix)
{
    uint64_t size = p->Size();
    for (uint64_t i = 0; i < size; i++) {
        auto acl_tensor = (*p)[i];
        string file_name = file_prefix + "_output_" + to_string(index) + "_" + to_string(i) + ".bin";
        if (0 != SaveToBinaryFile(acl_tensor, file_name)) {
            return 1;
        }
    }
    return 0;
}

int SaveResultToBinaryFile(aclTensor *p, size_t index, size_t total_input, const string &file_prefix,
                           const set<short int> &inplace_output)
{
    if (index < total_input) {
        return 0;
    }
    auto output_idx = CalcRealOutputIndex(index - total_input, inplace_output);
    return SaveResultToBinaryFile(p, output_idx, file_prefix);
}

int SaveResultToBinaryFile(aclTensorList *p, size_t index, size_t total_input, const string &file_prefix,
                           const set<short int> &inplace_output)
{
    if (index < total_input) {
        return 0;
    }
    auto output_idx = CalcRealOutputIndex(index - total_input, inplace_output);
    return SaveResultToBinaryFile(p, output_idx, file_prefix);
}
