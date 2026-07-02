/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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

#include "zb_runtime.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <c10/util/Exception.h>
#include "torch_npu/csrc/aten/common/from_blob.h"

#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
#include "acl/acl_rt.h"
#include "shmem.h"
#endif

namespace vllm_ascend {
namespace {

std::mutex g_shmem_mutex;
bool g_initialized = false;
void *g_ext_info = nullptr;
std::vector<void *> g_tensor_ptrs;

#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
void throw_shmem_unavailable()
{
    TORCH_CHECK(false,
                "zero-buffer SHMEM runtime is not available in this build. "
                "Please install Ascend SHMEM and rebuild vllm_ascend_C.");
}
#else
int32_t fill_init_attr(int32_t rank, int32_t world_size, uint64_t local_mem_size,
                       const std::string &server_ip_port, aclshmemx_init_attr_t *attributes)
{
    aclshmemx_uniqueid_t default_flag_uid = {};
    size_t ip_len = 0;
    if (!server_ip_port.empty()) {
        ip_len = std::min(server_ip_port.size(), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
        std::copy_n(server_ip_port.data(), ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    return ACLSHMEM_SUCCESS;
}

bool zb_debug_enabled()
{
    const char *value = std::getenv("VLLM_ASCEND_ZB_DEBUG");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int32_t query_aclrt_device_or_neg1()
{
    int32_t device_id = -1;
    if (aclrtGetDevice(&device_id) != ACL_SUCCESS) {
        return -1;
    }
    return device_id;
}

int32_t query_logic_device_id(int32_t user_device_id)
{
    using RtGetLogicDevIdByUserDevIdFunc = int (*)(int32_t, int32_t *);
    static RtGetLogicDevIdByUserDevIdFunc fn =
        reinterpret_cast<RtGetLogicDevIdByUserDevIdFunc>(dlsym(RTLD_DEFAULT, "rtGetLogicDevIdByUserDevId"));
    if (fn == nullptr) {
        return -1;
    }
    int32_t logic_device_id = -1;
    if (fn(user_device_id, &logic_device_id) != 0) {
        return -1;
    }
    return logic_device_id;
}

void log_shmem_device_context(const char *stage, int64_t rank, int64_t world_size, int32_t logical_device_id,
                              int32_t user_device_id, int32_t aclrt_device_id, int32_t logic_device_id)
{
    if (!zb_debug_enabled()) {
        return;
    }
    const char *visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    std::cerr << "[ZB][device][" << stage << "]"
              << " rank=" << rank << "/" << world_size << " logical_device_id=" << logical_device_id
              << " user_device_id=" << user_device_id << " aclrtGetDevice=" << aclrt_device_id
              << " rtGetLogicDevIdByUserDevId=" << logic_device_id
              << " ASCEND_RT_VISIBLE_DEVICES="
              << (visible != nullptr && visible[0] != '\0' ? visible : "<unset>") << std::endl;
}

int32_t resolve_user_device_id(int32_t *logical_device_out)
{
    int32_t logical_device = 0;
    if (logical_device_out != nullptr) {
        *logical_device_out = -1;
    }
    if (aclrtGetDevice(&logical_device) != ACL_SUCCESS) {
        TORCH_CHECK(false, "aclrtGetDevice failed while resolving SHMEM user device id");
    }
    if (logical_device_out != nullptr) {
        *logical_device_out = logical_device;
    }

    const char *visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (visible == nullptr || visible[0] == '\0') {
        return logical_device;
    }

    std::vector<int32_t> devices;
    std::stringstream stream(visible);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token.erase(token.begin(),
                    std::find_if(token.begin(), token.end(),
                                 [](unsigned char ch) { return !std::isspace(ch); }));
        token.erase(std::find_if(token.rbegin(), token.rend(),
                                 [](unsigned char ch) { return !std::isspace(ch); })
                        .base(),
                    token.end());
        if (token.empty()) {
            continue;
        }
        devices.push_back(static_cast<int32_t>(std::stoi(token)));
    }

    if (logical_device >= 0 && logical_device < static_cast<int32_t>(devices.size())) {
        return devices[logical_device];
    }
    return logical_device;
}

void ensure_hybm_user_device_id(int64_t rank, int64_t world_size)
{
    // aclshmem init captures aclrtGetDevice() for hybm_init() P2P user ids.
    int32_t logical_device = -1;
    const int32_t physical_device_id = resolve_user_device_id(&logical_device);
    const int32_t aclrt_before = query_aclrt_device_or_neg1();
    const int32_t logic_before = aclrt_before >= 0 ? query_logic_device_id(aclrt_before) : -1;

    log_shmem_device_context("pre_set_user_device", rank, world_size, logical_device, physical_device_id,
                             aclrt_before, logic_before);

    // aclrtSetDevice expects the process-local logical id (0..N-1 within the
    // current ASCEND_RT_VISIBLE_DEVICES slice), not the physical device id.
    const aclError status = aclrtSetDevice(logical_device);
    TORCH_CHECK(status == ACL_SUCCESS,
                "aclrtSetDevice(logical_device_id=", logical_device,
                ") failed before SHMEM init, status=", static_cast<int>(status),
                ", physical_device_id=", physical_device_id,
                ", aclrtGetDevice(before)=", aclrt_before,
                ", ASCEND_RT_VISIBLE_DEVICES=",
                (std::getenv("ASCEND_RT_VISIBLE_DEVICES") != nullptr
                     ? std::getenv("ASCEND_RT_VISIBLE_DEVICES")
                     : "<unset>"));

    const int32_t aclrt_after = query_aclrt_device_or_neg1();
    const int32_t logic_after = aclrt_after >= 0 ? query_logic_device_id(aclrt_after) : -1;
    log_shmem_device_context("post_set_user_device", rank, world_size, logical_device, physical_device_id,
                             aclrt_after, logic_after);
}
#endif

int64_t checked_numel(c10::ArrayRef<int64_t> shape)
{
    TORCH_CHECK(!shape.empty(), "shape must not be empty");
    int64_t numel = 1;
    for (int64_t dim : shape) {
        TORCH_CHECK(dim > 0, "all shape dimensions must be positive, got ", dim);
        TORCH_CHECK(numel <= std::numeric_limits<int64_t>::max() / dim,
                    "shape is too large for zero-buffer SHMEM tensor allocation");
        numel *= dim;
    }
    return numel;
}

size_t checked_nbytes(c10::ArrayRef<int64_t> shape, at::ScalarType dtype)
{
    int64_t numel = checked_numel(shape);
    size_t element_size = c10::elementSize(dtype);
    TORCH_CHECK(element_size > 0, "invalid dtype element size");
    TORCH_CHECK(static_cast<uint64_t>(numel) <= std::numeric_limits<uint64_t>::max() / element_size,
                "byte size overflow for zero-buffer SHMEM tensor allocation");
    return static_cast<size_t>(numel) * element_size;
}

void free_tensor_buffers()
{
#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    for (void *ptr : g_tensor_ptrs) {
        if (ptr != nullptr) {
            aclshmem_free(ptr);
        }
    }
    g_tensor_ptrs.clear();
#endif
}

} // namespace

int64_t zb_init(int64_t rank, int64_t world_size, int64_t local_mem_size, const std::string &server_ip_port)
{
#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    throw_shmem_unavailable();
#else
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    TORCH_CHECK(rank >= 0, "rank must be non-negative, got ", rank);
    TORCH_CHECK(world_size > 0, "world_size must be positive, got ", world_size);
    TORCH_CHECK(rank < world_size, "rank must be smaller than world_size, got rank=", rank,
                ", world_size=", world_size);
    TORCH_CHECK(local_mem_size > 0, "local_mem_size must be positive, got ", local_mem_size);

    if (!g_initialized) {
        if (zb_debug_enabled()) {
            std::cerr << "[ZB][init] starting rank=" << rank << "/" << world_size
                      << " local_mem_size=" << local_mem_size << " uri=" << server_ip_port << std::endl;
        }

        ensure_hybm_user_device_id(rank, world_size);
        aclshmemx_set_conf_store_tls(false, nullptr, 0);
        aclshmemx_init_attr_t attributes = {};
        int32_t status = fill_init_attr(static_cast<int32_t>(rank), static_cast<int32_t>(world_size),
                                        static_cast<uint64_t>(local_mem_size), server_ip_port, &attributes);
        TORCH_CHECK(status == ACLSHMEM_SUCCESS, "failed to fill SHMEM init attributes, status=", status);

        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        if (status != ACLSHMEM_SUCCESS) {
            const int32_t aclrt_now = query_aclrt_device_or_neg1();
            const int32_t logic_now = aclrt_now >= 0 ? query_logic_device_id(aclrt_now) : -1;
            std::cerr << "[ZB][init] aclshmemx_init_attr failed rank=" << rank << "/" << world_size
                      << " status=" << status << " aclrtGetDevice=" << aclrt_now
                      << " logicDeviceId=" << logic_now << " uri=" << server_ip_port << std::endl;
        }
        TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmemx_init_attr failed, status=", status);
        TORCH_CHECK(aclshmemx_init_status() == ACLSHMEM_STATUS_IS_INITIALIZED,
                    "aclshmem runtime is not initialized after aclshmemx_init_attr");

        if (zb_debug_enabled()) {
            const int32_t aclrt_now = query_aclrt_device_or_neg1();
            const int32_t logic_now = aclrt_now >= 0 ? query_logic_device_id(aclrt_now) : -1;
            std::cerr << "[ZB][init] success rank=" << rank << "/" << world_size
                      << " my_pe=" << aclshmem_my_pe() << " aclrtGetDevice=" << aclrt_now
                      << " logicDeviceId=" << logic_now << std::endl;
        }
        g_initialized = true;
    } else if (zb_debug_enabled()) {
        std::cerr << "[ZB][init] skipped re-init rank=" << rank << "/" << world_size
                  << " runtime already initialized, my_pe=" << aclshmem_my_pe() << std::endl;
    }

    return static_cast<int64_t>(aclshmem_my_pe());
#endif
}

int64_t zb_alloc(int64_t element_count, int64_t element_size)
{
#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    throw_shmem_unavailable();
#else
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    TORCH_CHECK(g_initialized, "SHMEM runtime must be initialized before allocation");
    TORCH_CHECK(element_count > 0, "element_count must be positive, got ", element_count);
    TORCH_CHECK(element_size > 0, "element_size must be positive, got ", element_size);
    TORCH_CHECK(g_ext_info == nullptr, "zero-buffer SHMEM runtime currently supports one active allocation");

    g_ext_info = aclshmemx_calloc(static_cast<size_t>(element_count), static_cast<size_t>(element_size));
    TORCH_CHECK(g_ext_info != nullptr, "aclshmemx_calloc failed");
    return reinterpret_cast<int64_t>(g_ext_info);
#endif
}

at::Tensor zb_alloc_tensor(c10::ArrayRef<int64_t> shape, at::ScalarType dtype, const std::string &device)
{
#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    throw_shmem_unavailable();
#else
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    TORCH_CHECK(g_initialized, "SHMEM runtime must be initialized before tensor allocation");

    size_t nbytes = checked_nbytes(shape, dtype);
    void *tensor_ptr = aclshmem_malloc(nbytes);
    TORCH_CHECK(tensor_ptr != nullptr, "aclshmem_malloc failed");
    g_tensor_ptrs.push_back(tensor_ptr);

    std::vector<int64_t> tensor_shape(shape.begin(), shape.end());
    auto options = at::TensorOptions().dtype(dtype).device(at::Device(device));
    return at_npu::native::from_blob(tensor_ptr, c10::IntArrayRef(tensor_shape), [](void *) {}, options);
#endif
}

at::Tensor zb_alias_tensor(const at::Tensor &base, c10::ArrayRef<int64_t> shape, at::ScalarType dtype)
{
    TORCH_CHECK(base.defined(), "base tensor must be defined");
    TORCH_CHECK(base.is_contiguous(), "base tensor must be contiguous");

    size_t alias_nbytes = checked_nbytes(shape, dtype);
    size_t base_nbytes = static_cast<size_t>(base.numel()) * base.element_size();
    TORCH_CHECK(alias_nbytes <= base_nbytes, "alias tensor byte size ", alias_nbytes,
                " exceeds base tensor byte size ", base_nbytes);

    std::vector<int64_t> tensor_shape(shape.begin(), shape.end());
    auto options = at::TensorOptions().dtype(dtype).device(base.device());
    return at_npu::native::from_blob(base.data_ptr(), c10::IntArrayRef(tensor_shape), [](void *) {}, options);
}

void zb_free(int64_t ptr)
{
#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    void *raw_ptr = reinterpret_cast<void *>(ptr);
    if (raw_ptr != nullptr) {
        aclshmem_free(raw_ptr);
    }
    if (raw_ptr == g_ext_info) {
        g_ext_info = nullptr;
    }
#else
    (void)ptr;
    throw_shmem_unavailable();
#endif
}

void zb_finalize()
{
#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    free_tensor_buffers();
    if (g_ext_info != nullptr) {
        aclshmem_free(g_ext_info);
        g_ext_info = nullptr;
    }
    if (g_initialized) {
        int32_t status = aclshmem_finalize();
        TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmem_finalize failed, status=", status);
        g_initialized = false;
    }
#else
    throw_shmem_unavailable();
#endif
}

int64_t zb_get_ext_info()
{
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    return reinterpret_cast<int64_t>(g_ext_info);
}

bool zb_is_initialized()
{
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    return g_initialized;
}

} // namespace vllm_ascend
