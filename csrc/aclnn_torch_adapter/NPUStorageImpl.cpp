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

#include "NPUStorageImpl.h"

namespace vllm_ascend
{

    NPUStorageImpl::NPUStorageImpl(
        use_byte_size_t use_byte_size,
        size_t size_bytes,
        at::DataPtr data_ptr,
        at::Allocator *allocator,
        bool resizable) : c10::StorageImpl(use_byte_size,
                                           size_bytes,
                                           at::DataPtr(std::move(data_ptr)),
                                           allocator,
                                           resizable)
    {
    }

    void NPUStorageImpl::release_resources()
    {
        StorageImpl::release_resources();
    }

    c10::intrusive_ptr<c10::StorageImpl> make_npu_storage_impl(
        c10::StorageImpl::use_byte_size_t,
        c10::SymInt size_bytes,
        c10::DataPtr data_ptr,
        c10::Allocator *allocator,
        bool resizable)
    {
        if (data_ptr == nullptr)
        {
            data_ptr = allocator->allocate(size_bytes.as_int_unchecked());
        }
        // Correctly create NPUStorageImpl object.
        c10::intrusive_ptr<c10::StorageImpl> npu_storage_impl = c10::make_intrusive<NPUStorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            size_bytes.as_int_unchecked(),
            std::move(data_ptr),
            allocator,
            resizable);
        // There is no need to consider the NPUStorageDesc information, it will be carried out in the subsequent processing.
        return npu_storage_impl;
    }

}
