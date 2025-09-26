// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "tensor.h"
#include <acl/acl.h>
#include <cstring>
#include <iostream>

static memcpy_fun_t memcpy_ptr = aclrtMemcpy;

// only for UT test
aclError ut_mem_fun(void *dst, size_t destMax, const void *src, size_t count,
                    aclrtMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    memcpy(dst, src, count);
    return ACL_ERROR_NONE;
}

memcpy_fun_t get_memcpy_fun() { return memcpy_ptr; }
void set_memcpy_fun(memcpy_fun_t fun) { memcpy_ptr = fun; }

void set_ut_memcpy_fun() { memcpy_ptr = ut_mem_fun; }

void unset_ut_memcpy_fun() { memcpy_ptr = aclrtMemcpy; }

aclError Tensor::to_host(void *host_ptr) const {
    size_t tensor_size = get_total_size();
    if (host_ptr == nullptr || data_ptr_ == nullptr) {
        throw std::runtime_error(
            "Invalid pointers: npu_ptr or host_ptr is null");
    }

    aclError ret = (*memcpy_ptr)(host_ptr, tensor_size, data_ptr_, tensor_size,
                                 ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                 std::to_string(ret));
    }
    return ret;
}

aclError Tensor::to_device(void *host_ptr) const {
    size_t tensor_size = get_total_size();
    if (host_ptr == nullptr || data_ptr_ == nullptr) {
        throw std::runtime_error(
            "Invalid pointers: npu_ptr or host_ptr is null");
    }
    aclError ret = (*memcpy_ptr)(data_ptr_, tensor_size, host_ptr, tensor_size,
                                 ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                 std::to_string(ret));
    }
    return ret;
}

aclError Tensor::to_host(void *host_ptr, aclrtStream stream) const {
    size_t tensor_size = get_total_size();
    if (host_ptr == nullptr || data_ptr_ == nullptr) {
        throw std::runtime_error(
            "Invalid pointers: npu_ptr or host_ptr is null");
    }

    aclError ret =
        aclrtMemcpyAsync(host_ptr, tensor_size, data_ptr_, tensor_size,
                         ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                 std::to_string(ret));
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error(
            "aclrtMemcpy aclrtSynchronizeStream failed, error code: " +
            std::to_string(ret));
    }
    return ret;
}

aclError Tensor::to_device(void *host_ptr, aclrtStream stream) const {
    size_t tensor_size = get_total_size();
    if (host_ptr == nullptr || data_ptr_ == nullptr) {
        throw std::runtime_error(
            "Invalid pointers: npu_ptr or host_ptr is null");
    }
    aclError ret =
        aclrtMemcpyAsync(data_ptr_, tensor_size, host_ptr, tensor_size,
                         ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error("aclrtMemcpy failed, error code: " +
                                 std::to_string(ret));
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        throw std::runtime_error(
            "aclrtMemcpy aclrtSynchronizeStream failed, error code: " +
            std::to_string(ret));
    }
    return ret;
}