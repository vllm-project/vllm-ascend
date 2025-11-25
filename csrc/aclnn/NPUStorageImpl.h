/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>
#include <c10/util/order_preserving_flat_hash_map.h>

#include "acl/acl_rt.h"
#include "acl/acl_base.h"

namespace vllm_ascend
{

    struct NPUStorageDesc
    {
    public:
        struct use_byte_size_t
        {
        };

        c10::SmallVector<int64_t, 5> base_sizes_;
        c10::SmallVector<int64_t, 5> base_strides_;
        c10::SmallVector<int64_t, 5> storage_sizes_;
        int64_t base_offset_ = 0;
        use_byte_size_t base_dtype_ = {};
        aclFormat origin_format_ = ACL_FORMAT_UNDEFINED;
        aclFormat npu_format_ = ACL_FORMAT_ND;
        // used to make CANN GE tensor from storagImpl
        caffe2::TypeMeta data_type_ = caffe2::TypeMeta::Make<uint8_t>();
    };

    struct NPUStorageImpl : public c10::StorageImpl
    {
        explicit NPUStorageImpl(
            use_byte_size_t use_byte_size,
            size_t size_bytes,
            at::DataPtr data_ptr,
            at::Allocator *allocator,
            bool resizable);
        ~NPUStorageImpl() override = default;

        void release_resources() override;

        NPUStorageDesc npu_desc_;

        NPUStorageDesc get_npu_desc() const
        {
            return npu_desc_;
        }
    };

    c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(
        c10::StorageImpl::use_byte_size_t,
        c10::SymInt size_bytes,
        c10::DataPtr data_ptr,
        c10::Allocator *allocator,
        bool resizable);

}
