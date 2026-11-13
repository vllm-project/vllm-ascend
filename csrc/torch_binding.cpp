/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include <dlfcn.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <ATen/core/Formatting.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "ops.h"
#include "utils.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#include "moe/rms_norm_cast/rms_norm_cast_torch_adpt.h"
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include "mla_preprocess/mla_preprocess_torch_adpt.h"
#endif
#include "mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "aclnn_torch_adapter/gbsa_torch_adpt.h"
#include "attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.h"
#include "attention/kv_compress_epilog_v2/kv_compress_epilog_v2_torch_adpt.h"
#include "attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h"
#include "attention/kv_quant_sparse_flash_attention_vllm/kv_quant_sparse_flash_attention_vllm_torch_adpt.h"
#include "attention/fused_sparse_attention_overlap/fused_sparse_attention_overlap_torch_adpt.h"
#if defined(VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD) || defined(VLLM_ASCEND_ENABLE_SELECTIVE_LIM)
#include "attention/fused_lightning_indexer_manage/fused_lightning_indexer_manage_torch_adpt.h"
#endif
#include "attention/fused_scatter_copy_sparse_flash_attention/fused_scatter_copy_sparse_flash_attention_torch_adpt.h"
#include "attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h"
#include "moe/causal_conv1d_v310/causal_conv1d_310_torch_adpt.h"
#include "attention/attn_res_fwd/attn_res_fwd_torch_adpt.h"
#include "attention/kda_gate_cumsum_vllm/kda_gate_cumsum_vllm_torch_adpt.h"
#include "attention/kda_layout_swap12/kda_layout_swap12_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule_v310/recurrent_gated_delta_rule_310_torch_adpt.h"
#include "attention/k2q_csr/k2q_csr_torch_adpt.h"
#include "attention/sparse_attention_score/sparse_attention_score_torch_adpt.h"
#include "moe/dequant_situ_quant/dequant_situ_quant_torch_adpt.h"
#include "moe/situ_mx_quant/situ_mx_quant_torch_adpt.h"
#include "moe/grouped_matmul_situ_quant/grouped_matmul_situ_quant_torch_adpt.h"
#include <c10/core/Device.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
#include <omp.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vllm_ascend {

c10::optional<at::Tensor> get_npu_view_from_cpu_tensor(const at::Tensor& cpu_tensor)
{
    TORCH_CHECK(cpu_tensor.defined(), "UVA view requires a defined CPU tensor");
    TORCH_CHECK(cpu_tensor.device().is_cpu(), "UVA view requires a CPU tensor");
    TORCH_CHECK(cpu_tensor.layout() == at::kStrided, "UVA view requires strided layout");

    const c10::Device npu_device(c10::DeviceType::PrivateUse1, c10_npu::current_device());
    const auto options = at::TensorOptions().dtype(cpu_tensor.scalar_type()).device(npu_device);
    if (cpu_tensor.numel() == 0) {
        return at::empty_strided(cpu_tensor.sizes(), cpu_tensor.strides(), options);
    }

    if (!cpu_tensor.is_pinned()) {
        return c10::nullopt;
    }
    // Query the registered storage base, then apply the logical tensor offset.
    // torch_npu's host allocator remains the sole register/unregister owner.
    const auto* host_base = cpu_tensor.storage().data_ptr().get();
    if (host_base == nullptr) {
        return c10::nullopt;
    }
    c10_npu::NPUGuard guard(npu_device);
    void* mapped_base = nullptr;
    const aclError ret = aclrtHostGetDevicePointer(const_cast<void*>(host_base), &mapped_base, 0);
    if (ret != ACL_SUCCESS || mapped_base == nullptr) {
        return c10::nullopt;
    }
    auto* mapped_data = static_cast<char*>(mapped_base) +
                        cpu_tensor.storage_offset() * cpu_tensor.element_size();
    auto keep_cpu_alive = [base = cpu_tensor](void*) mutable {};
    return at_npu::native::from_blob(mapped_data, cpu_tensor.sizes(), cpu_tensor.strides(),
                                     0, keep_cpu_alive, options, npu_device);
}

// user_device_id is the ordinal passed to torch.npu.set_device/aclrtSetDevice,
// not a vLLM local rank or an ASCEND_RT_VISIBLE_DEVICES entry.
int64_t get_physical_device_id(int64_t user_device_id)
{
    TORCH_CHECK(user_device_id >= 0 && user_device_id <= std::numeric_limits<int32_t>::max(),
                "Invalid NPU user device ID: ", user_device_id);
#ifdef CANN_DEVICE_ID_MAPPING
    int32_t physical_device_id = -1;
    auto ret = aclrtGetPhyDevIdByUserDevId(static_cast<int32_t>(user_device_id), &physical_device_id);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtGetPhyDevIdByUserDevId failed for user device ",
                user_device_id, ", error code: ", ret);
    TORCH_CHECK(physical_device_id >= 0, "CANN returned an invalid physical device ID: ", physical_device_id);
    return physical_device_id;
#else
    TORCH_CHECK(false, "NPU physical device lookup requires a vllm-ascend extension built with CANN "
                       "aclrtGetPhyDevIdByUserDevId support. "
                       "Upgrade CANN and rebuild vllm-ascend.");
#endif
}

// Required by EXEC_NPU_CMD hash helpers in aclnn_torch_adapter/op_api_common.h
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace {

constexpr int64_t DSA_SLOT_MAPPING_FLAT = 1;
constexpr int64_t DSA_SLOT_MAPPING_BLOCK_OFFSET = 2;

struct DevicePrintPayload {
    std::string message;
    at::Tensor host_tensor_snapshot;
};

std::mutex& get_device_print_mutex()
{
    static std::mutex device_print_mutex;
    return device_print_mutex;
}

std::mutex& get_pinned_host_mmap_mutex()
{
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<void*, uint64_t>& get_pinned_host_mmaps()
{
    static std::unordered_map<void*, uint64_t> regions;
    return regions;
}

void device_print_callback(void* args)
{
    // device_print is a debug-only helper. We intentionally do not reclaim the
    // callback payload here because aclgraph replay may re-execute the same host
    // callback payload multiple times. Freeing it on first execution would make
    // later replays dereference a dangling pointer.
    auto* payload = static_cast<DevicePrintPayload*>(args);
    if (payload == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> guard(get_device_print_mutex());
    if (!payload->message.empty()) {
        std::cout << payload->message;
    }

    if (payload->host_tensor_snapshot.defined()) {
        if (!payload->message.empty()) {
            std::cout << std::endl;
        }
        at::print(std::cout, payload->host_tensor_snapshot.contiguous(), 120);
    }

    std::cout << std::endl;
    std::cout.flush();
}

void enqueue_device_print(std::unique_ptr<DevicePrintPayload> payload,
                          aclrtStream stream)
{
    auto* raw_payload = payload.release();
    const aclError ret = aclrtLaunchHostFunc(stream, device_print_callback,
                                             raw_payload);
    if (ret != ACL_SUCCESS) {
        delete raw_payload;
    }
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtLaunchHostFunc failed, error code: ", ret);
}

}

void swap_blocks_batch(const torch::Tensor& src_ptrs,
                       const torch::Tensor& dst_ptrs,
                       const torch::Tensor& sizes,
                       int64_t direction) {

    TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
    TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
    TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
    TORCH_CHECK(src_ptrs.dtype() == torch::kInt64, "src_ptrs must be int64");
    TORCH_CHECK(dst_ptrs.dtype() == torch::kInt64, "dst_ptrs must be int64");
    TORCH_CHECK(sizes.dtype() == torch::kInt64, "sizes must be int64");

    const int64_t n = src_ptrs.size(0);
    TORCH_CHECK(dst_ptrs.size(0) == n, "dst_ptrs length must match src_ptrs");
    TORCH_CHECK(sizes.size(0) == n, "sizes length must match src_ptrs");

    if (n == 0) return;

    const int64_t* src_data = src_ptrs.data_ptr<int64_t>();
    const int64_t* dst_data = dst_ptrs.data_ptr<int64_t>();
    const int64_t* size_data = sizes.data_ptr<int64_t>();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    aclrtMemcpyKind memcpy_kind;
    switch (direction) {
        case 0:
            memcpy_kind = ACL_MEMCPY_HOST_TO_DEVICE;
            break;
        case 1:
            memcpy_kind = ACL_MEMCPY_DEVICE_TO_HOST;
            break;
        case 2:
            memcpy_kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            break;
        default:
            TORCH_CHECK(false,
                        "swap_blocks_batch: invalid direction ", direction,
                        " (expected 0=H2D, 1=D2H, 2=D2D)");
    }

    // =========================================================================
    // path 1: aclrtMemcpyBatchAsync (CANN 8.5+)
    // =========================================================================
#if defined(CANN_MEMCPY_BATCH_ASYNC)
    if (memcpy_kind != ACL_MEMCPY_DEVICE_TO_DEVICE) {
        static_assert(sizeof(void*) == sizeof(int64_t),
                      "void* and int64_t must be the same size");
        static_assert(sizeof(size_t) == sizeof(int64_t),
                      "size_t and int64_t must be the same size");

        void** dst_arr = reinterpret_cast<void**>(
            const_cast<int64_t*>(dst_data));
        void** src_arr = reinterpret_cast<void**>(
            const_cast<int64_t*>(src_data));
        size_t* size_arr = reinterpret_cast<size_t*>(
            const_cast<int64_t*>(size_data));
        size_t* dest_maxs = size_arr;

        // aclrtMemcpyBatchAttr uses srcLoc/dstLoc (aclrtMemLocation)
        // to specify memory locations, not aclrtMemcpyKind.
        int32_t device_id = 0;
        const aclError get_device_result = aclrtGetDevice(&device_id);
        TORCH_CHECK(get_device_result == ACL_SUCCESS,
                    "aclrtGetDevice failed with error code ",
                    get_device_result);

        aclrtMemLocation host_loc = {};
        host_loc.type = ACL_MEM_LOCATION_TYPE_HOST;
        host_loc.id = 0;

        aclrtMemLocation device_loc = {};
        device_loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        device_loc.id = device_id;

        aclrtMemcpyBatchAttr attr = {};
        if (memcpy_kind == ACL_MEMCPY_HOST_TO_DEVICE) {
            attr.srcLoc = host_loc;
            attr.dstLoc = device_loc;
        } else {  // ACL_MEMCPY_DEVICE_TO_HOST
            attr.srcLoc = device_loc;
            attr.dstLoc = host_loc;
        }

        size_t attrs_index = 0;
        size_t fail_index = 0;

        aclError result = aclrtMemcpyBatchAsync(
            dst_arr, dest_maxs, src_arr, size_arr,
            static_cast<size_t>(n),
            &attr, &attrs_index, 1,
            &fail_index, stream);

        TORCH_CHECK(result == ACL_SUCCESS,
                    "aclrtMemcpyBatchAsync failed at index ", fail_index,
                    " with error code ", result);
        return;
    }
#endif

    // =========================================================================
    // path 2: aclrtMemcpyAsync
    // =========================================================================
    for (int64_t i = 0; i < n; i++) {
        void* dst = reinterpret_cast<void*>(dst_data[i]);
        const void* src = reinterpret_cast<const void*>(src_data[i]);
        size_t copy_size = static_cast<size_t>(size_data[i]);

        aclError ret = aclrtMemcpyAsync(
            dst,
            copy_size,
            src,
            copy_size,
            memcpy_kind,
            stream);

        TORCH_CHECK(ret == ACL_SUCCESS,
                    "aclrtMemcpyAsync failed at index ", i,
                    " with error code ", ret,
                    ", src=", src_data[i],
                    ", dst=", dst_data[i],
                    ", size=", size_data[i]);
    }
}

bool supports_eccpu_offload()
{
#if defined(CANN_MEMCPY_BATCH_ASYNC) && defined(CANN_HOST_REGISTER_V2_PINNED)
    return true;
#else
    return false;
#endif
}

void register_pinned_host_mmap(const torch::Tensor& host_tensor)
{
    // mmap is pageable by default; request page locking explicitly.
    TORCH_CHECK(host_tensor.device().is_cpu(),
                "register_pinned_host_mmap expects a CPU tensor");
    TORCH_CHECK(host_tensor.is_contiguous(),
                "register_pinned_host_mmap expects a contiguous tensor");

    void* ptr = host_tensor.data_ptr();
    const uint64_t size = static_cast<uint64_t>(
        host_tensor.numel() * host_tensor.element_size());
    TORCH_CHECK(size > 0, "register_pinned_host_mmap expects a non-empty tensor");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(ptr) % 4096 == 0,
                "register_pinned_host_mmap requires a 4 KiB-aligned mmap address, got ",
                reinterpret_cast<std::uintptr_t>(ptr));

#if defined(CANN_HOST_REGISTER_V2_PINNED)
    std::lock_guard<std::mutex> lock(get_pinned_host_mmap_mutex());
    auto& regions = get_pinned_host_mmaps();
    // Track the address and size to reject duplicate or mismatched operations.
    TORCH_CHECK(regions.find(ptr) == regions.end(),
                "Host mmap address is already registered as pinned memory: ",
                reinterpret_cast<std::uintptr_t>(ptr));

    const aclError ret = aclrtHostRegisterV2(ptr, size, ACL_HOST_REG_PINNED);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclrtHostRegisterV2(ACL_HOST_REG_PINNED) failed with error code ",
                ret, ", address=", reinterpret_cast<std::uintptr_t>(ptr),
                ", size=", size);
    regions.emplace(ptr, size);
#else
    TORCH_CHECK(false,
                "Pinned shared mmap requires aclrtHostRegisterV2 with "
                "ACL_HOST_REG_PINNED in the CANN headers/runtime");
#endif
}

void unregister_pinned_host_mmap(const torch::Tensor& host_tensor)
{
    TORCH_CHECK(host_tensor.device().is_cpu(),
                "unregister_pinned_host_mmap expects a CPU tensor");
    TORCH_CHECK(host_tensor.is_contiguous(),
                "unregister_pinned_host_mmap expects a contiguous tensor");

#if defined(CANN_HOST_REGISTER_V2_PINNED)
    void* ptr = host_tensor.data_ptr();
    std::lock_guard<std::mutex> lock(get_pinned_host_mmap_mutex());
    auto& regions = get_pinned_host_mmaps();
    const auto it = regions.find(ptr);
    TORCH_CHECK(it != regions.end(),
                "Host mmap address is not registered as pinned memory: ",
                reinterpret_cast<std::uintptr_t>(ptr));
    const uint64_t size = static_cast<uint64_t>(
        host_tensor.numel() * host_tensor.element_size());
    TORCH_CHECK(it->second == size,
                "Pinned host mmap size changed before unregister: registered=",
                it->second, ", current=", size);

    const aclError ret = aclrtHostUnregister(ptr);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclrtHostUnregister failed with error code ", ret,
                ", address=", reinterpret_cast<std::uintptr_t>(ptr));
    regions.erase(it);
#else
    TORCH_CHECK(false,
                "Pinned shared mmap unregister is unavailable because this "
                "extension was built without aclrtHostRegisterV2 support");
#endif
}

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
// Direct kernel wrappers depend on vllm_ascend_kernels, which is skipped on
// 310P and A5 builds.
void swap_blocks_impl(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping, aclrtStream stream)
{
    torch::Device src_device = src.device();
    torch::Device dst_device = dst.device();
    aclrtMemcpyKind memcpy_type;

    if ((!src_device.is_cpu()) && (!dst_device.is_cpu())) {
        TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same npu");
        memcpy_type = ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else if ((!src_device.is_cpu()) && dst_device.is_cpu()) {
        memcpy_type = ACL_MEMCPY_DEVICE_TO_HOST;
    } else if (src_device.is_cpu() && (!dst_device.is_cpu())) {
        memcpy_type = ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
        TORCH_CHECK(false, "Invalid device combination, src tensor device: ", src_device, ", dst tensor device: ", dst_device);
    }

    TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    const int64_t block_size_in_bytes = src.element_size() * src.stride(0);

    const int64_t num_blocks = block_mapping.size(0);
    const int64_t max_src_block = src.size(0);
    const int64_t max_dst_block = dst.size(0);
    for (size_t i = 0; i < num_blocks; i++) {
        int64_t src_block_number = block_mapping[i][0].item<int64_t>();
        int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
        TORCH_CHECK(src_block_number >= 0 && src_block_number <= max_src_block,
                    "src block index ", src_block_number, " out of range (max: ", max_src_block, ")");
        TORCH_CHECK(dst_block_number >= 0 && dst_block_number <= max_dst_block,
                    "dst block index ", dst_block_number, " out of range (max: ", max_dst_block, ")");

        int64_t src_offset = src_block_number * block_size_in_bytes;
        int64_t dst_offset = dst_block_number * block_size_in_bytes;

        aclrtMemcpyAsync(dst_ptr + dst_offset, block_size_in_bytes,
                         src_ptr + src_offset, block_size_in_bytes,
                         memcpy_type, stream);
    }
}

void swap_blocks(torch::Tensor &x, torch::Tensor &y, const torch::Tensor &z)
{

    const c10_npu::OptionalNPUGuard npuGuard(
        (!x.device().is_cpu()) ? x.device() : y.device()
    );
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    swap_blocks_impl(x, y, z, stream);
    return;
}

AscendType get_dtype_from_torch(at::ScalarType scalarType)
{
    if (scalarType == at::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == at::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else {
        return AscendType::FP16;
    }
}

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, input_hidden_token,
                          lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, num_tokens_per_core,
                         input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                       int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size, lora_rank,
                          slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size,
                         num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                 at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size,
                          seq_len_ptr, seq_len_size, y_ptr,
                          batch_size, input_hidden_token, lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size,
                         y_ptr, batch_size,
                         num_tokens_per_core, input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                          batch_size, lora_rank, slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                         batch_size, num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}
#endif

at::Tensor npu_sign_bits_pack(const at::Tensor& input,
                                   const int64_t size) {
    int64_t ySize = (input.size(0) + 7) / 8;
    int64_t outDim = 0;
    if (size != 0) {
        outDim = ySize / size;
    }

    at::Tensor out = torch::empty({size, outDim}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    EXEC_NPU_CMD(aclnnSignBitsPack, input, size, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> npu_gemma_rms_norm(
    const at::Tensor& x,
    const at::Tensor& gamma,
    double epsilon)
{
    int64_t dim_x = x.dim();
    int64_t dim_gamma = gamma.dim();
    int64_t diff = dim_x - dim_gamma;
    std::vector<int64_t> new_shape;
    at::Tensor rstd;
    if (diff > 0) {
        new_shape.reserve(dim_x);
        auto x_sizes = x.sizes();
        for (int64_t i = 0; i < diff; ++i) {
            new_shape.push_back(x_sizes[i]);
        }
        for (int64_t i = 0; i < dim_gamma; ++i) {
            new_shape.push_back(1);
        }
    } else {
        new_shape.assign(dim_x, 1);
    }
    rstd = at::empty(new_shape, x.options().dtype(at::kFloat));
    at::Tensor y = at::empty(x.sizes(), x.options());
    EXEC_NPU_CMD(aclnnGemmaRmsNorm, x, gamma, epsilon, y, rstd);
    return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}

void transpose_kv_cache_by_block(
    const at::TensorList &kCache,
    const at::TensorList &vCache,
    const at::Tensor &blockIDs,
    int64_t blockSize,
    int64_t headNum,
    int64_t headDim,
    int64_t splitNum,
    int64_t layerNum)
{
    TORCH_CHECK(layerNum >= 0 && kCache.size() == layerNum && vCache.size() == layerNum,
                "K/V cache lists must match layerNum");
    TORCH_CHECK(blockSize > 0 && headNum > 0 && headDim > 0 && splitNum > 0 && headNum % splitNum == 0,
                "Invalid KV block dimensions or splitNum");
    TORCH_CHECK(blockIDs.dim() == 1 && blockIDs.scalar_type() == at::kLong && blockIDs.is_contiguous(),
                "blockIDs must be a contiguous 1D int64 tensor");
    if (layerNum == 0) {
        return;
    }

    // Only the block axis may have gaps. The kernel still transposes a dense
    // [blockSize, headNum, headDim] payload inside each selected block.
    auto checkCache = [&](const at::Tensor &cache) {
        TORCH_CHECK(cache.dim() == 4 && cache.size(1) == blockSize &&
                    cache.size(2) == headNum && cache.size(3) == headDim,
                    "KV caches must have shape [blocks, blockSize, headNum, headDim]");
        TORCH_CHECK(cache.device() == blockIDs.device() && cache.scalar_type() == kCache[0].scalar_type(),
                    "KV caches must have the same dtype and be on the blockIDs device");
        int64_t denseStride = 1;
        for (int64_t dim = 3; dim > 0; --dim) {
            TORCH_CHECK(cache.size(dim) == 1 || cache.stride(dim) == denseStride,
                        "KV caches must be contiguous within each block");
            denseStride *= cache.size(dim);
        }
        TORCH_CHECK(cache.size(0) <= 1 || cache.stride(0) >= denseStride,
                    "KV cache blocks must not overlap");
        TORCH_CHECK(cache.size(0) > 0 || blockIDs.numel() == 0,
                    "Nonempty blockIDs require nonempty KV caches");
    };
    for (int64_t layer = 0; layer < layerNum; ++layer) {
        checkCache(kCache[layer]);
        checkCache(vCache[layer]);
        TORCH_CHECK(kCache[layer].size(0) == vCache[layer].size(0),
                    "K/V caches must have the same number of blocks");
    }
    if (blockIDs.numel() == 0) {
        return;
    }

    // ACLNN tiling does not reliably expose dynamic-input strides. Pass the
    // physical strides explicitly, without copying or replacing the caches.
    // Consecutive layers with the same K/V strides retain one fused launch.
    auto getBlockStride = [&](const at::Tensor &cache) -> int64_t {
        // A singleton block axis never advances, so its stride is arbitrary.
        // Normalize it for both layer grouping and the tiling overlap check.
        return cache.size(0) > 1 ? cache.stride(0) : blockSize * headNum * headDim;
    };
    for (int64_t begin = 0; begin < layerNum;) {
        int64_t kBlockStride = getBlockStride(kCache[begin]);
        int64_t vBlockStride = getBlockStride(vCache[begin]);
        int64_t end = begin + 1;
        while (end < layerNum && getBlockStride(kCache[end]) == kBlockStride &&
               getBlockStride(vCache[end]) == vBlockStride) {
            ++end;
        }
        int64_t groupLayers = end - begin;
        auto kGroup = kCache.slice(begin, groupLayers);
        auto vGroup = vCache.slice(begin, groupLayers);
        EXEC_NPU_CMD(aclnnTransposeKvCacheByBlock, kGroup, vGroup, blockIDs,
                     blockSize, headNum, headDim, splitNum, groupLayers, kBlockStride, vBlockStride);
        begin = end;
    }
}

void npu_scatter_pa_kv_cache(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    c10::string_view cache_mode,
    c10::string_view scatter_mode)
{
    std::string cache_mode_str(cache_mode);
    std::string scatter_mode_str(scatter_mode);
    char* cache_mode_ptr = const_cast<char*>(cache_mode_str.c_str());
    char* scatter_mode_ptr = const_cast<char*>(scatter_mode_str.c_str());

    c10::optional<at::Tensor> optional_tensor = c10::nullopt;
    c10::optional<at::IntArrayRef> optional_int_array = c10::nullopt;

    // aclnnScatterPaKvCache uses a different argument order from the public
    // torch_npu wrapper. Call it directly so scatter_mode (for example,
    // NHSD) is forwarded instead of being fixed to None by op-plugin.
    EXEC_NPU_CMD(aclnnScatterPaKvCache,
                 key,
                 key_cache,
                 slot_mapping,
                 value,
                 value_cache,
                 optional_tensor,
                 optional_tensor,
                 optional_tensor,
                 cache_mode_ptr,
                 scatter_mode_ptr,
                 optional_int_array,
                 optional_int_array);
}

void device_print(c10::string_view msg)
{
    auto payload = std::make_unique<DevicePrintPayload>();
    payload->message = std::string(msg);
    enqueue_device_print(std::move(payload), c10_npu::getCurrentNPUStream().stream());
}

void device_print(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.defined(), "tensor must be defined");
    TORCH_CHECK(
        tensor.device().is_cpu() ||
            tensor.device().type() == c10::DeviceType::PrivateUse1,
        "device_print only supports CPU and NPU tensors, but got device ",
        tensor.device());

    auto payload = std::make_unique<DevicePrintPayload>();
    if (tensor.device().is_cpu()) {
        payload->host_tensor_snapshot = tensor.contiguous().clone();
        enqueue_device_print(std::move(payload),
                             c10_npu::getCurrentNPUStream().stream());
        return;
    }

    const c10_npu::OptionalNPUGuard npu_guard(tensor.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at::Tensor contiguous_tensor = tensor.contiguous();
    payload->host_tensor_snapshot = at::empty_like(
        contiguous_tensor,
        contiguous_tensor.options().device(at::kCPU).pinned_memory(true));

    const size_t num_bytes = contiguous_tensor.numel() *
                             contiguous_tensor.element_size();
    const aclError memcpy_ret = aclrtMemcpyAsync(
        payload->host_tensor_snapshot.data_ptr(), num_bytes,
        contiguous_tensor.data_ptr(), num_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    TORCH_CHECK(memcpy_ret == ACL_SUCCESS,
                "aclrtMemcpyAsync failed, error code: ", memcpy_ret);

    // The D2H copy and host callback are queued on the same stream so the
    // callback prints only after the host snapshot is ready.
    enqueue_device_print(std::move(payload), stream);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_copy_and_expand_eagle_inputs(
    const at::Tensor &target_token_ids,
    const at::Tensor &target_positions,
    const at::Tensor &next_token_ids,
    const at::Tensor &query_start_loc,
    const at::Tensor &query_end_loc,
    int64_t padding_token_id,
    int64_t parallel_drafting_token_id,
    int64_t num_padding_slots_per_request,
    bool shift_input_ids,
    int64_t total_draft_tokens)
{
    int64_t total_input_tokens = target_token_ids.size(0);
    int64_t num_reqs = query_start_loc.size(0) - 1;

    auto device = target_token_ids.device();
    at::Tensor out_input_ids = at::zeros({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_positions = at::zeros({total_draft_tokens}, at::dtype(at::kInt).device(device));
    at::Tensor out_is_rejected_token_mask = at::zeros({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_is_masked_token_mask = at::zeros({total_draft_tokens}, at::dtype(at::kChar).device(device));
    at::Tensor out_new_token_indices = at::zeros({num_reqs * num_padding_slots_per_request}, at::dtype(at::kInt).device(device));
    at::Tensor out_hidden_state_mapping = at::zeros({total_input_tokens}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnCopyAndExpandEagleInputs,
        target_token_ids, target_positions, next_token_ids, query_start_loc, query_end_loc,
        padding_token_id, parallel_drafting_token_id, num_padding_slots_per_request,
        shift_input_ids, total_input_tokens,
        out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
        out_new_token_indices, out_hidden_state_mapping);

    return {out_input_ids, out_positions, out_is_rejected_token_mask, out_is_masked_token_mask,
            out_new_token_indices, out_hidden_state_mapping};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& input_ids_opt,
    const c10::optional<at::Tensor>& tid2eid_opt,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    const c10::optional<at::Tensor>& bias_vl_opt,
    int64_t image_sentinel_lo,
    int64_t image_sentinel_count)
{

    TORCH_CHECK(x.dim() == 2, "x must be 2D, but got dim=", x.dim());
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "x dtype must be float16/float32/bfloat16, but got ", x.scalar_type());

    TORCH_CHECK(k > 0, "k must be > 0, but got k=", k);
    TORCH_CHECK(k_group >= 1, "k_group must be >= 1, but got k_group=", k_group);
    TORCH_CHECK(group_count >= 1, "group_count must be >= 1, but got group_count=", group_count);

    TORCH_CHECK(group_select_mode == 0 || group_select_mode == 1,
                "group_select_mode must be 0 or 1, but got ", group_select_mode);
    TORCH_CHECK(renorm == 0,
                "renorm can only be 0 currently, but got ", renorm);
    TORCH_CHECK(norm_type == 0 || norm_type == 1 || norm_type ==2,
                "norm_type must be 0 (softmax) or 1 (sigmoid) or 2 (softplus), but got ", norm_type);

    TORCH_CHECK(eps > 0.0, "eps must be > 0, but got ", eps);
    TORCH_CHECK(routed_scaling_factor > 0.0,
                "routed_scaling_factor must be > 0, but got ", routed_scaling_factor);

    const auto sizes = x.sizes();
    const int64_t rows = sizes[0];
    const int64_t expert_num = sizes[1];

    TORCH_CHECK(expert_num > 0, "expert_num must be > 0");
    TORCH_CHECK(expert_num <= 2048,
                "expert_num (E) must be <= 2048, but got ", expert_num);

    if (bias_opt.has_value() && bias_opt->defined()) {
        const auto& bias = *bias_opt;
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D, but got dim=", bias.dim());
        TORCH_CHECK(bias.size(0) == expert_num,
                    "bias.size(0) must equal expert_num. bias.size(0)=",
                    bias.size(0), ", expert_num=", expert_num);
        TORCH_CHECK(bias.scalar_type() == x.scalar_type(),
                    "bias dtype must equal x dtype. x=", x.scalar_type(),
                    ", bias=", bias.scalar_type());
    }

    if (input_ids_opt.has_value() && input_ids_opt->defined()) {
        const auto& input_ids = *input_ids_opt;
        TORCH_CHECK(input_ids.scalar_type() == at::kInt || input_ids.scalar_type() == at::kLong,
                    "input_ids dtype must be int32 or int64, but got ", input_ids.scalar_type());
        TORCH_CHECK(input_ids.numel() == rows,
                    "input_ids.numel() must equal x.size(0). input_ids.numel()=",
                    input_ids.numel(), ", rows=", rows);
    }

    if (tid2eid_opt.has_value() && tid2eid_opt->defined()) {
        const auto& tid2eid = *tid2eid_opt;
        TORCH_CHECK(tid2eid.scalar_type() == at::kInt || tid2eid.scalar_type() == at::kLong,
                    "tid2eid dtype must be int32 or int64, but got ", tid2eid.scalar_type());
        TORCH_CHECK(tid2eid.dim() >= 1, "tid2eid must have dim>=1, but got dim=", tid2eid.dim());
    }

    if (bias_vl_opt.has_value() && bias_vl_opt->defined()) {
        const auto& bias_vl = *bias_vl_opt;
        TORCH_CHECK(input_ids_opt.has_value() && input_ids_opt->defined(),
                    "input_ids is required when bias_vl is present");
        TORCH_CHECK(bias_vl.dim() == 1, "bias_vl must be 1D, but got dim=", bias_vl.dim());
        TORCH_CHECK(bias_vl.size(0) == expert_num,
                    "bias_vl.size(0) must equal expert_num. bias_vl.size(0)=",
                    bias_vl.size(0), ", expert_num=", expert_num);
        TORCH_CHECK(bias_vl.scalar_type() == x.scalar_type(),
                    "bias_vl dtype must equal x dtype. x=", x.scalar_type(),
                    ", bias_vl=", bias_vl.scalar_type());
        TORCH_CHECK(image_sentinel_count > 0,
                    "image_sentinel_count must be > 0, but got ", image_sentinel_count);
    }

    const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor& input_ids = c10::value_or_else(input_ids_opt, [] { return at::Tensor(); });
    const at::Tensor& tid2eid = c10::value_or_else(tid2eid_opt, [] { return at::Tensor(); });
    const at::Tensor& bias_vl = c10::value_or_else(bias_vl_opt, [] { return at::Tensor(); });

    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnMoeGatingTopKHash,
                 x,
                 bias,
                 input_ids,
                 tid2eid,
                 bias_vl,
                 k,
                 k_group,
                 group_count,
                 group_select_mode,
                 renorm,
                 norm_type,
                 out_flag,
                 routed_scaling_factor,
                 eps,
                 image_sentinel_lo,
                 image_sentinel_count,
                 y,
                 expert_idx,
                 out);

    return {y, expert_idx, out};
}

std::vector<bool> is_contiguous_axes(const at::Tensor &tensor)
{
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int64_t ndim = sizes.size();

    if (ndim == 0) {
        return {};
    }
    std::vector<bool> result(ndim, false);

    std::vector<int64_t> contiguous_stride(ndim, 1);
    for (int64_t i = ndim - 2; i >= 0; i--) {
        contiguous_stride[i] = contiguous_stride[i + 1] * sizes[i + 1];
    }

    for (int64_t i = 0; i < ndim; i++) {
        result[i] = (strides[i] == contiguous_stride[i]);
    }
    return result;
}

std::tuple<at::Tensor> construct_compressor_output_tensor(const at::Tensor &x, const at::Tensor &norm_weight,
                                                          const at::Tensor &rope_sin, int64_t cmp_ratio, int64_t coff)
{
    constexpr int DIM_3 = 3;
    auto x_dim = x.dim();
    at::SmallVector<int64_t, 8> cmp_kv_size;
    at::Tensor cmp_kv;
    auto cmp_s = 0;
    if (x_dim == DIM_3) {
        cmp_s = (x.size(1) + cmp_ratio - 1) / cmp_ratio;
        cmp_kv_size = {x.size(0), cmp_s, norm_weight.size(0)};
    } else {
        cmp_s = rope_sin.size(0);
        cmp_kv_size = {cmp_s, norm_weight.size(0)};
    }

    cmp_kv = at::empty(cmp_kv_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor>(cmp_kv);
}

std::tuple<at::Tensor> compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                                  at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                                  const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                                  const c10::optional<at::Tensor> &state_block_table,
                                  const c10::optional<at::Tensor> &cu_seqlens, const c10::optional<at::Tensor> &seqused,
                                  const c10::optional<at::Tensor> &start_pos, int64_t rope_head_dim, int64_t cmp_ratio,
                                  int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode)
{
    constexpr int CONTINUOUS = 1;
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;
    constexpr int32_t DIM_3 = 3;
    constexpr int32_t VALUE_0 = 0;
    auto x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_2 || x_dim == DIM_3, "x dim num[", x_dim, "] should be 2 or 3");

    TORCH_CHECK(norm_weight.defined(), "Check norm_weight != nullptr failed");
    auto norm_weight_dim = norm_weight.dim();
    TORCH_CHECK(norm_weight_dim == DIM_1, "norm_weight dim num[", norm_weight_dim, "] should be 1");

    TORCH_CHECK(rope_sin.defined(), "Check rope_sin != nullptr failed");
    auto rope_sin_dim = rope_sin.dim();
    TORCH_CHECK(rope_sin_dim == x_dim, "rope_sin dim num[", rope_sin_dim, "] should be equal to x dim num[", x_dim,
                "]");

    TORCH_CHECK(cmp_ratio > VALUE_0, "cmp_ratio should be greater than 0");

    std::tuple<at::Tensor> output = construct_compressor_output_tensor(x, norm_weight, rope_sin, cmp_ratio, coff);
    at::Tensor cmp_kv = std::get<0>(output);

    auto state_cache_dim = state_cache.dim();
    TORCH_CHECK(state_cache_dim == DIM_3, "state_cache dim num[", state_cache_dim, "] should be 3");
    auto contiguous_axes_result = is_contiguous_axes(state_cache);
    // if (cache_mode == CONTINUOUS) {
    //     TORCH_CHECK(contiguous_axes_result[0] && contiguous_axes_result[1] && contiguous_axes_result[2],
    //                 "when cache_mode == ", cache_mode, ", state_cache must be contiguous on all axes");
    // }
    int64_t state_cache_stride_dim0 = state_cache.stride(0);

    EXEC_NPU_CMD(aclnnCompressor, x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                    state_block_table, cu_seqlens, seqused, start_pos, rope_head_dim, cmp_ratio, coff, norm_eps,
                    rotary_mode, cache_mode, state_cache_stride_dim0, cmp_kv);

    return std::tuple<at::Tensor>(cmp_kv);
}

void check_compressor_metadata_common(
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &cu_seqlens,
    const at::Tensor &start_pos, const at::Tensor &kv_block_table, int64_t kv_block_size,
    int64_t slot_mapping_format, int64_t compress_ratio, int64_t num_reqs_actual)
{
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t VALUE_0 = 0;

    TORCH_CHECK(rope_cos.defined() && rope_sin.defined(), "rope_cos and rope_sin should be defined");
    TORCH_CHECK(rope_cos.dim() == DIM_2 && rope_sin.dim() == DIM_2,
                "rope_cos and rope_sin should be 2D tensors");
    TORCH_CHECK(rope_cos.scalar_type() == rope_sin.scalar_type(),
                "rope_cos and rope_sin should have same dtype");
    TORCH_CHECK(rope_cos.size(0) == rope_sin.size(0) && rope_cos.size(1) == rope_sin.size(1),
                "rope_cos and rope_sin should have same shape");
    TORCH_CHECK(rope_cos.size(0) > VALUE_0 && rope_cos.size(1) > VALUE_0,
                "rope_cos shape should be non-empty");
    TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.dim() == 1, "cu_seqlens should be a 1D tensor");
    TORCH_CHECK(start_pos.defined() && start_pos.dim() == 1, "start_pos should be a 1D tensor");
    TORCH_CHECK(kv_block_table.defined() && kv_block_table.dim() == DIM_2, "kv_block_table should be a 2D tensor");
    TORCH_CHECK(kv_block_size > VALUE_0, "kv_block_size should be greater than 0");
    TORCH_CHECK(compress_ratio > VALUE_0, "compress_ratio should be greater than 0");
    TORCH_CHECK(slot_mapping_format == DSA_SLOT_MAPPING_BLOCK_OFFSET || slot_mapping_format == DSA_SLOT_MAPPING_FLAT,
                "slot_mapping_format should be 1(flat) or 2(block_offset), but got ", slot_mapping_format);
    TORCH_CHECK(num_reqs_actual > VALUE_0, "num_reqs_actual should be greater than 0");
    TORCH_CHECK(cu_seqlens.size(0) > num_reqs_actual,
                "cu_seqlens dim0 should be greater than num_reqs_actual");
    TORCH_CHECK(start_pos.size(0) >= num_reqs_actual,
                "start_pos dim0 should be greater than or equal to num_reqs_actual");
    TORCH_CHECK(kv_block_table.size(0) >= num_reqs_actual,
                "kv_block_table dim0 should be greater than or equal to num_reqs_actual");
}

void check_compressor_metadata_outputs(
    const at::Tensor &rope_cos, const at::Tensor &compress_cos, const at::Tensor &compress_sin,
    const at::Tensor &slot_mapping, int64_t slot_mapping_format)
{
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t VALUE_0 = 0;

    TORCH_CHECK(compress_cos.defined() && compress_sin.defined() && slot_mapping.defined(),
                "compress_cos, compress_sin, and slot_mapping should be defined");
    TORCH_CHECK(compress_cos.dim() >= DIM_2, "compress_cos dim num should be at least 2");
    TORCH_CHECK(compress_sin.dim() == compress_cos.dim(), "compress_cos and compress_sin should have same dim num");
    TORCH_CHECK(compress_cos.size(0) > VALUE_0, "compress_cos dim0 should be greater than 0");
    TORCH_CHECK(compress_cos.size(compress_cos.dim() - 1) == rope_cos.size(1),
                "compress_cos last dim should match rope dim");
    for (int64_t dim_idx = 0; dim_idx < compress_cos.dim(); ++dim_idx) {
        TORCH_CHECK(compress_sin.size(dim_idx) == compress_cos.size(dim_idx),
                    "compress_cos and compress_sin should have same shape");
    }
    TORCH_CHECK(compress_cos.scalar_type() == rope_cos.scalar_type() &&
                    compress_sin.scalar_type() == rope_cos.scalar_type(),
                "compress outputs should have same dtype as rope_cos");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping dtype should be int32");
    if (slot_mapping_format == DSA_SLOT_MAPPING_BLOCK_OFFSET) {
        TORCH_CHECK(slot_mapping.dim() == DIM_2 && slot_mapping.size(0) == compress_cos.size(0) &&
                        slot_mapping.size(1) == DIM_2,
                    "block_offset slot_mapping should have shape [num_rows, 2]");
    } else {
        TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == compress_cos.size(0),
                    "flat slot_mapping should have shape [num_rows]");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compressor_metadata(
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &cu_seqlens,
    const at::Tensor &start_pos, const at::Tensor &kv_block_table, int64_t kv_block_size,
    int64_t slot_mapping_format, int64_t compress_ratio, int64_t num_compressed_tokens, int64_t num_reqs_actual)
{
    constexpr int64_t VALUE_0 = 0;

    check_compressor_metadata_common(
        rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table, kv_block_size, slot_mapping_format, compress_ratio,
        num_reqs_actual);
    TORCH_CHECK(num_compressed_tokens > VALUE_0, "num_compressed_tokens should be greater than 0");

    at::SmallVector<int64_t, 4> rope_output_size = {num_compressed_tokens, 1, 1, rope_cos.size(1)};
    at::Tensor compress_cos = at::empty(rope_output_size, rope_cos.options());
    at::Tensor compress_sin = at::empty(rope_output_size, rope_sin.options());

    at::SmallVector<int64_t, 2> slot_mapping_size;
    if (slot_mapping_format == DSA_SLOT_MAPPING_BLOCK_OFFSET) {
        slot_mapping_size = {num_compressed_tokens, 2};
    } else {
        slot_mapping_size = {num_compressed_tokens};
    }
    at::Tensor slot_mapping = at::empty(slot_mapping_size, kv_block_table.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnCompressorMetadata, rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table,
                 kv_block_size, slot_mapping_format, compress_ratio, num_reqs_actual, compress_cos, compress_sin,
                 slot_mapping);
    return std::make_tuple(compress_cos, compress_sin, slot_mapping);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compressor_metadata_out(
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, const at::Tensor &cu_seqlens,
    const at::Tensor &start_pos, const at::Tensor &kv_block_table, int64_t kv_block_size,
    int64_t slot_mapping_format, int64_t compress_ratio, int64_t num_reqs_actual, at::Tensor &compress_cos,
    at::Tensor &compress_sin, at::Tensor &slot_mapping)
{
    check_compressor_metadata_common(
        rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table, kv_block_size, slot_mapping_format, compress_ratio,
        num_reqs_actual);
    check_compressor_metadata_outputs(rope_cos, compress_cos, compress_sin, slot_mapping, slot_mapping_format);

    EXEC_NPU_CMD(aclnnCompressorMetadata, rope_cos, rope_sin, cu_seqlens, start_pos, kv_block_table,
                 kv_block_size, slot_mapping_format, compress_ratio, num_reqs_actual, compress_cos, compress_sin,
                 slot_mapping);
    return std::make_tuple(compress_cos, compress_sin, slot_mapping);
}

std::tuple<at::Tensor, at::Tensor> construct_quant_lightning_indexer_v2_output_tensor(const at::Tensor& query, const at::Tensor& key,
                                                           int64_t sparse_count, std::string query_layout_str,
                                                           std::string key_layout_str, int64_t return_value)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND") ? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = at::empty(output_size, query.options().dtype(at::kBFloat16));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kBFloat16));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer_v2_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    int64_t topk, int64_t quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &output_idx_offset,
    const c10::optional<at::Tensor> &metadata,
    int64_t max_seqlen_q, c10::string_view layout_q, c10::string_view layout_k,
    int64_t mask_mode, int64_t cmp_ratio, int64_t return_value)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    std::string query_layout_str = std::string(layout_q);
    std::string key_layout_str = std::string(layout_k);

    std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer_output = construct_quant_lightning_indexer_v2_output_tensor(
            query, key, topk, query_layout_str, key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(quant_lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(quant_lightning_indexer_output);
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    if (key_layout_str == "PA_BSND" || key_layout_str == "PA_BBND") {
        auto contiguous_axes_result_key = is_contiguous_axes(key);
        TORCH_CHECK(contiguous_axes_result_key[1] && contiguous_axes_result_key[2],
                    "key must be contiguous on all axes except axis 0");
        auto contiguous_axes_result_key_scale = is_contiguous_axes(key_dequant_scale);
        TORCH_CHECK(contiguous_axes_result_key_scale[1] && contiguous_axes_result_key_scale[2],
                    "key_dequant_scale must be contiguous on all axes except axis 0");
    }

    EXEC_NPU_CMD(aclnnQuantLightningIndexerV2, query, key, weights, query_dequant_scale, key_dequant_scale,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table, output_idx_offset, metadata,
        topk, quant_mode, max_seqlen_q, query_layout_ptr, key_layout_ptr, mask_mode, cmp_ratio, return_value,
        sparse_indices_out, sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer_v2_compat_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    int64_t topk, int64_t quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &output_idx_offset,
    const c10::optional<at::Tensor> &metadata,
    int64_t max_seqlen_q, c10::string_view layout_q, c10::string_view layout_k,
    int64_t mask_mode, int64_t cmp_ratio, int64_t return_value)
{
    TORCH_CHECK(return_value == 0, "npu_quant_lightning_indexer_v2 only supports return_value=0");
    auto outputs = qli_v2::QuantLightningIndexerCandidate(
        query, key, weights, query_dequant_scale, key_dequant_scale, topk, quant_mode,
        c10::nullopt, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        block_table, output_idx_offset, metadata, max_seqlen_q, layout_q, layout_k,
        mask_mode, cmp_ratio, 3, 2048, 8);
    return {std::get<0>(outputs), std::get<1>(outputs)};
}

std::tuple<at::Tensor, at::Tensor> construct_output_tensor(const at::Tensor &q, std::string layout,
    bool return_softmax_lse)
{
    for (size_t i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0,
            "All values within query's shape should be greater "
            "than 0, but shape[",
            i,
            "] is ",
            q.size(i));
    }
    at::Tensor output = at::empty(q.sizes(), q.options().dtype(q.dtype()));
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        std::vector<int64_t> lse_sizes(q.sizes().begin(), q.sizes().end());
        lse_sizes.back() = 1;
        softmax_lse = at::empty(lse_sizes, q.options().dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv_npu(const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    std::tuple<at::Tensor, at::Tensor> output = construct_output_tensor(q, layout_q_str, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(output);
    at::Tensor softmax_lse = std::get<1>(output);
    int64_t ori_kv_stride = 0;
    int64_t cmp_kv_stride = 0;
    if (ori_kv.has_value()){
        const at::Tensor& tmp_kv = *ori_kv;
        ori_kv_stride = tmp_kv.stride(0);
    }
    if (cmp_kv.has_value()){
        const at::Tensor& tmp_kv = *cmp_kv;
        cmp_kv_stride = tmp_kv.stride(0);
    }

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    EXEC_NPU_CMD(aclnnSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv, sinks,
        metadata, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_kv_stride, cmp_kv_stride, ori_win_left, ori_win_right, layout_q_ptr,
        layout_kv_ptr, return_softmax_lse, attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}

auto get_valid_tensor = [](const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor npu_sparse_attn_sharedkv_metadata_npu(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    }
    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(aclnnSparseAttnSharedkvMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val, cu_seqlens_cmp_kv_val, seqused_q_val,
                    seqused_kv_val, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk,
                    cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr,
                    layout_kv_ptr, has_ori_kv, has_cmp_kv, output);
    return output;
}

at::Tensor npu_quant_lightning_indexer_v2_metadata_npu(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t topk, int64_t quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k,
    const c10::string_view layout_q, c10::string_view layout_k, int64_t mask_mode, int64_t cmp_ratio,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (seqused_k.has_value()) {
        output_device = seqused_k.value().device();
    } else if (cu_seqlens_k.has_value()) {
        output_device = cu_seqlens_k.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (cmp_residual_k.has_value()) {
        output_device = cmp_residual_k.value().device();
    }

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_k_val = get_valid_tensor(cu_seqlens_k, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_k_val = get_valid_tensor(seqused_k, output_device);
    auto cmp_residual_k_val = get_valid_tensor(cmp_residual_k, output_device);

    std::string layout_q_str = std::string(layout_q);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    std::string layout_k_str = std::string(layout_k);
    char *layout_k_ptr = const_cast<char *>(layout_k_str.c_str());

    EXEC_NPU_CMD(aclnnVllmAscendQuantLightningIndexerV2Metadata,
                 cu_seqlens_q_val, cu_seqlens_k_val, seqused_q_val, seqused_k_val,
                 cmp_residual_k_val, num_heads_q, num_heads_k, head_dim, topk, quant_mode, batch_size, max_seqlen_q,
                 max_seqlen_k, layout_q_ptr, layout_k_ptr, mask_mode, cmp_ratio, output);

    return output;
}

at::Tensor construct_hc_post_output_tensor(const at::Tensor& residual)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    at::SmallVector<int64_t, SIZE> output_size = {residual.size(DIM_0), residual.size(DIM_1), residual.size(DIM_2), residual.size(DIM_3)};
    at::Tensor out = at::empty(output_size, residual.options().dtype(residual.dtype()));
    return out;
}

// step1，工具函数，检查输入shape
void check_hc_post_shape_and_dtype(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post, const at::Tensor& com) {
    // check x shape: [b, s, d]
    TORCH_CHECK(x.dim() == 3, "Input tensor x's dim num should be 3, actual ", x.dim(), ".");
    for (size_t i = 0; i < 3; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is :", x.size(i), ".");
    }
    auto batch = x.size(0);
    auto sequence = x.size(1);
    auto d = x.size(2);
    // check residual: [b, s, hc, d]
    TORCH_CHECK(residual.dim() == 4, "Input tensor residual's dim num should be 4, actual ", residual.dim(), ".");
    auto hc = residual.size(2);
    TORCH_CHECK(hc > 0, "The hc of residual should be positive, actual ", hc, ".");
    TORCH_CHECK(residual.size(0) == batch, "The residual.shape[0] should be batch, actual residual.shape[0] is ", residual.size(0), ", batch is ", batch, ".");
    TORCH_CHECK(residual.size(1) == sequence, "The residual.shape[1] should be sequence, actual residual.shape[1] is ", residual.size(1), ", sequence is ", sequence, ".");
    TORCH_CHECK(residual.size(3) == d, "The residual.shape[3] should be d, actual residual.shape[3] is ", residual.size(3), ", d is ", d, ".");
    // check post [b, s, hc]
    TORCH_CHECK(post.dim() == 3, "Input tensor post's dim num should be 3, actual ", post.dim(), ".");
    TORCH_CHECK(post.size(0) == batch, "The post.shape[0] should be batch, actual post.shape[0] is ", post.size(0), ", batch is ", batch, ".");
    TORCH_CHECK(post.size(1) == sequence, "The post.shape[1] should be sequence, actual post.shape[1] is ", post.size(1), ", sequence is ", sequence, ".");
    TORCH_CHECK(post.size(2) == hc, "The post.shape[2] should be hc, actual post.shape[2] is ", post.size(2), ", hc is ", hc, ".");
    // check com: [b, s, hc, hc]
    TORCH_CHECK(com.dim() == 4, "Input tensor com's dim num should be 4, actual ", com.dim(), ".");
    TORCH_CHECK(com.size(0) == batch, "The com.shape[0] should be batch, actual com.shape[0] is ", com.size(0), ", batch is ", batch, ".");
    TORCH_CHECK(com.size(1) == sequence, "The com.shape[1] should be sequence, actual com.shape[1] is ", com.size(1), ", sequence is ", sequence, ".");
    TORCH_CHECK(com.size(2) == hc, "The com.shape[2] should be hc, actual com.shape[2] is ", com.size(2), ", hc is ", hc, ".");
    TORCH_CHECK(com.size(3) == hc, "The com.shape[3] should be hc, actual com.shape[3] is ", com.size(3), ", hc is ", hc, ".");
    // check dtype
    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");
    TORCH_CHECK(residual.dtype() == x.dtype(), "x's dtype should be equal to residual's dtype.");
    TORCH_CHECK(post.dtype() == at::kFloat || post.dtype() == at::kHalf || post.dtype() == at::kBFloat16,
                "post should be FLOAT16, BFLOAT16, or FLOAT32.");
    TORCH_CHECK(com.dtype() == post.dtype(), "com's dtype should be equal to post's dtype.");
}

at::Tensor npu_hc_post_npu(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb)
{
    check_hc_post_shape_and_dtype(x, residual, post, comb);
    // construct the output tensor
    at::Tensor out = construct_hc_post_output_tensor(residual);
    EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out);
    return out;
}

constexpr int64_t HC_PRE_HC_LIMIT = 4;
constexpr int64_t HC_PRE_MIX_HC_LIMIT = 24;

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(const at::Tensor& x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3){
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

at::Tensor construct_hc_pre_pre_output_tensor(const at::Tensor& x, int64_t hc_mult)
{
    at::SmallVector<int64_t, 8> pre_size;
    if (x.dim() == 4) {
        pre_size = {x.size(0), x.size(1), hc_mult};
    } else if (x.dim() == 3) {
        pre_size = {x.size(0), hc_mult};
    }
    return at::empty(pre_size, x.options().dtype(at::kFloat));
}

void check_hc_pre_shape_and_dtype(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const c10::optional<at::Tensor>& pre_mix,
    int64_t hc_mult)
{
    constexpr int64_t HC_SCALE_SIZE = 3;
    auto x_dims = x.dim();
    TORCH_CHECK(x_dims == 3 || x_dims == 4, "Input tensor x's dim num should be 3 or 4, actual ", x_dims, ".");
    for (auto i = 0; i < x_dims; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is ",
                    x.size(i), ".");
    }

    auto hc = x_dims == 4 ? x.size(2) : x.size(1);
    auto d = x_dims == 4 ? x.size(3) : x.size(2);
    TORCH_CHECK(hc_mult == HC_PRE_HC_LIMIT, "hc_mult only supports ", HC_PRE_HC_LIMIT, ", actual ", hc_mult, ".");
    TORCH_CHECK(hc == HC_PRE_HC_LIMIT, "The hc of x only supports ", HC_PRE_HC_LIMIT, ", actual ", hc, ".");
    TORCH_CHECK(hc_fn.dim() == 2, "Input tensor hc_fn's dim num should be 2, actual ", hc_fn.dim(), ".");
    TORCH_CHECK(hc_fn.size(0) == HC_PRE_MIX_HC_LIMIT, "The hc_fn.shape[0] only supports ",
                HC_PRE_MIX_HC_LIMIT, ", actual ", hc_fn.size(0), ".");
    TORCH_CHECK(hc_fn.size(1) == hc * d, "The hc_fn.shape[1] should be hc * d, actual hc_fn.shape[1] is ",
                hc_fn.size(1), ", hc is ", hc, ", d is ", d, ".");
    TORCH_CHECK(hc_scale.dim() == 1, "Input tensor hc_scale's dim num should be 1, actual ", hc_scale.dim(), ".");
    TORCH_CHECK(hc_scale.size(0) == HC_SCALE_SIZE, "Input tensor hc_scale's shape should be [", HC_SCALE_SIZE,
                "], actual [", hc_scale.size(0), "].");
    TORCH_CHECK(hc_base.dim() == 1, "Input tensor hc_base's dim num should be 1, actual ", hc_base.dim(), ".");
    TORCH_CHECK(hc_base.size(0) == HC_PRE_MIX_HC_LIMIT, "The hc_base.shape[0] only supports ",
                HC_PRE_MIX_HC_LIMIT, ", actual ", hc_base.size(0), ".");

    TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
    TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
    TORCH_CHECK(hc_scale.dtype() == at::kFloat, "hc_scale's dtype should be FLOAT32.");
    TORCH_CHECK(hc_base.dtype() == at::kFloat, "hc_base's dtype should be FLOAT32.");
    if (pre_mix.has_value() && pre_mix->defined()) {
        TORCH_CHECK(pre_mix->dtype() == at::kFloat, "pre_mix's dtype should be FLOAT32.");
        TORCH_CHECK(pre_mix->dim() == x_dims - 1, "pre_mix's dim num should be ", x_dims - 1, ", actual ",
                    pre_mix->dim(), ".");
        for (auto i = 0; i < x_dims - 1; i++) {
            TORCH_CHECK(pre_mix->size(i) == x.size(i), "pre_mix.shape[", i, "] should equal x.shape[", i,
                        "], actual ", pre_mix->size(i), " vs ", x.size(i), ".");
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> run_hc_pre_fusion(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    const c10::optional<at::Tensor>& pre_mix, int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
    double hc_eps)
{
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    at::Tensor pre = construct_hc_pre_pre_output_tensor(x, hc_mult);
    EXEC_NPU_CMD(aclnnHcPre, x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps,
                 y, post, comb_frag, pre);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag, pre);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    const c10::optional<at::Tensor> pre_mix = c10::nullopt;
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult);
    auto outputs = run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps,
                                     hc_eps);
    return {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v3_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    const c10::optional<at::Tensor>& pre_mix, int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
    double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult);
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

void inplace_partial_rotary_mul_npu(at::Tensor & x, const at::Tensor &r1, const at::Tensor &r2, c10::string_view rotary_mode, at::IntArrayRef partial_slice, bool negate_sin)
{
    constexpr int BSND_DIM_NUM = 4;
    static const std::unordered_map<std::string, int> mode_map = {
        {"half", 0},
        {"interleave", 1},
        {"quarter", 2},
        {"interleave-half", 3}
    };
    std::string rotary_mode_str = std::string(rotary_mode);
    auto it = mode_map.find(rotary_mode_str);
    if (it == mode_map.end())
    {
        return;
    }
    auto origin_dim_num = x.dim();
    TORCH_CHECK(origin_dim_num == BSND_DIM_NUM, "Input tensor x's dim num should be 4, actual ", origin_dim_num, ".");
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice, negate_sin);
}

std::tuple<at::Tensor, at::Tensor> npu_rms_norm_dynamic_quant_npu(
    const at::Tensor& x,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& smooth_scale,
    const c10::optional<at::Tensor>& beta,
    double epsilon)
{
    constexpr int32_t SIZE = 8;
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(gamma.numel() > 0, "Input tensor gamma should not be empty.");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x.size(-1), "gamma dim are not equal to last dim of x shape.");
    TORCH_CHECK(epsilon > 0, "epsilon should be greater than 0.");
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16, "x should be FLOAT16, BFLOAT16.");

    at::Tensor smooth_scale2{nullptr};
    auto options = x.options();
    at::Tensor y_out = at::empty_like(x, options.dtype(at::kChar));
    at::Tensor y2_out = at::empty({1}, options.dtype(at::kChar));

    c10::SmallVector<int64_t, SIZE> scale_out_shape;
    for (size_t i = 0; i < x.sizes().size() - 1; i++) {
        scale_out_shape.push_back(x.sizes()[i]);
    }
    at::Tensor scale_out = at::empty(scale_out_shape, options.dtype(at::kFloat));
    at::Tensor scale2_out = at::empty_like(scale_out);
    std::array<bool, 2>* output_mask = nullptr;
    int64_t* dst_type = nullptr;

    EXEC_NPU_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth_scale, smooth_scale2, beta, epsilon, output_mask, dst_type,
                 y_out, y2_out, scale_out, scale2_out);

    return std::make_tuple(y_out, scale_out);
}

std::tuple<at::Tensor, at::Tensor> npu_kv_quant_sparse_attn_sharedkv_npu(
    const at::Tensor& q,
    int64_t kv_quant_mode,
    const c10::optional<at::Tensor>& ori_kv,
    const c10::optional<at::Tensor>& cmp_kv,
    const c10::optional<at::Tensor>& ori_sparse_indices,
    const c10::optional<at::Tensor>& cmp_sparse_indices,
    const c10::optional<at::Tensor>& ori_block_table,
    const c10::optional<at::Tensor>& cmp_block_table,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& metadata,
    int64_t tile_size,
    int64_t rope_head_dim,
    double softmax_scale,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    auto output = construct_output_tensor(q, layout_q_str, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(output);
    at::Tensor softmax_lse = std::get<1>(output);

    char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
    char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());
    int64_t ori_kv_stride0 = 0;
    int64_t cmp_kv_stride0 = 0;
    if (ori_kv.has_value() && ori_kv.value().defined()) {
        ori_kv_stride0 = ori_kv.value().stride(0);
    }
    if (cmp_kv.has_value() && cmp_kv.value().defined()) {
        cmp_kv_stride0 = cmp_kv.value().stride(0);
    }

    EXEC_NPU_CMD(aclnnKvQuantSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
                 ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                 seqused_q, seqused_kv, sinks, metadata, kv_quant_mode, tile_size, rope_head_dim,
                 softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                 layout_q_ptr, layout_kv_ptr, ori_kv_stride0, cmp_kv_stride0, return_softmax_lse,
                 attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}

at::Tensor npu_kv_quant_sparse_attn_sharedkv_metadata_npu(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    int64_t kv_quant_mode,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t tile_size,
    int64_t rope_head_dim,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv,
    const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    }
    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
    char* layout_kv_ptr = const_cast<char*>(layout_kv_str.c_str());

    EXEC_NPU_CMD(aclnnKvQuantSparseAttnSharedkvMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val,
                 cu_seqlens_cmp_kv_val, seqused_q_val, seqused_kv_val, num_heads_q, num_heads_kv,
                 head_dim, batch_size, max_seqlen_q, max_seqlen_kv, ori_topk, cmp_topk, kv_quant_mode,
                 tile_size, rope_head_dim, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
                 ori_win_right, layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv, output);
    return output;
}

int64_t get_type_code(at::ScalarType dst_type)
{
    switch (dst_type) {
        case at::ScalarType::Float8_e5m2:
            return 35;
        case at::ScalarType::Float8_e4m3fn:
            return 36;
        case at::ScalarType::Float4_e2m1fn_x2:
            return 40;
        case at::ScalarType::Half:
            return 1;
        case at::ScalarType::BFloat16:
            return 27;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dst_type);
    }
    return 0;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_swiglu_group_quant_output_tensor(
    const at::Tensor& x,
    int64_t dst_type,
    int64_t quant_mode,
    bool ue8m0_scale)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t SWIGLU_FACTOR = 2;
    constexpr int64_t PER_BLOCK_FP16 = 128;
    constexpr int64_t PER_MX_FP16 = 32;
    constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;
    constexpr int64_t GROUP_QUANT = 1;
    constexpr int64_t MX_QUANT = 2;
    constexpr int64_t FP8_QUANT = 3;

    at::SmallVector<int64_t, SIZE> y_size(x.sizes().begin(), x.sizes().end());
    for (size_t i = 0; i < x.sizes().size(); i++) {
        TORCH_CHECK(x.size(i) >= 0, "All values within x's shape should be non-negative, but shape[",
                    i, "] is ", x.size(i));
    }
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16 or BFLOAT16.");
    int64_t x_last_dim = x.sizes().back();
    TORCH_CHECK(quant_mode == GROUP_QUANT || quant_mode == MX_QUANT || quant_mode == FP8_QUANT,
                "Unsupported quant mode, only support ", GROUP_QUANT, " or ", MX_QUANT, " or ", FP8_QUANT, ".");
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        TORCH_CHECK(x_last_dim % 256 == 0,
                    "In group quant, the last dim of x should be divisible by 256, actual ", x_last_dim, ".");
    } else {
        TORCH_CHECK(x_last_dim % 128 == 0,
                    "In mx quant, the last dim of x should be divisible by 128, actual ", x_last_dim, ".");
    }

    y_size.back() = y_size.back() / SWIGLU_FACTOR;
    int64_t y_last_dim = y_size.back();
    auto y_dtype = dst_type == 35 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));

    at::SmallVector<int64_t, SIZE> scale_size(y_size.begin(), y_size.end());
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        scale_size.back() = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
    } else if (quant_mode == MX_QUANT) {
        int64_t scale_last_dim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_last_dim = (scale_last_dim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_size.back() = scale_last_dim;
        scale_size.push_back(MX_SCALE_ALIGN_FACTOR);
    }

    auto scale_type = at::kFloat;
    if (quant_mode == MX_QUANT || (quant_mode == FP8_QUANT && ue8m0_scale)) {
        scale_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale = at::empty(scale_size, x.options().dtype(scale_type));
    at::Tensor y_origin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swiglu_group_quant_npu(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& topk_weight,
    const c10::optional<at::Tensor>& group_index,
    at::ScalarType dst_type = at::ScalarType::Float8_e4m3fn,
    int64_t quant_mode = 1,
    int64_t group_size = 128,
    bool round_scale = false,
    bool ue8m0_scale = false,
    bool output_origin = false,
    int64_t group_list_type = 0,
    double clamp_value = 0.0)
{
    int64_t dst_type_code = get_type_code(dst_type);
    auto output_tensors = construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode, ue8m0_scale);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor scale = std::get<1>(output_tensors);
    at::Tensor y_origin = std::get<2>(output_tensors);

    EXEC_NPU_CMD(aclnnSwigluGroupQuant, x, topk_weight, group_index, dst_type_code, quant_mode, group_size,
                 round_scale, ue8m0_scale, output_origin, group_list_type, clamp_value, y, scale, y_origin);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

void indexer_compress_epilog_v2_npu(
    at::Tensor& indexer_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t layout = 2)
{
    int64_t indexer_compress_cache_stride = indexer_compress_cache.stride(0);
    EXEC_NPU_CMD(aclnnIndexerCompressEpilogV2, indexer_compress_cache, x, slot_mapping, layout,
                 indexer_compress_cache_stride);
}

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode,
    int64_t swiglu_mode,
    double clamp_limit,
    double glu_alpha,
    double glu_bias)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1");
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1, but got ", quant_mode);
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode);
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit >= 0.0, "clamp_limit should be positive finite");
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite");
    TORCH_CHECK(std::isfinite(glu_bias), "glu_bias should be finite");
    TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even");

    c10::SmallVector<int64_t, 8> y_size;
    c10::SmallVector<int64_t, 8> scale_size;
    for (int64_t i = 0; i < x.dim() - 1; ++i) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    y_size.push_back(x.size(x.dim() - 1) / 2);

    at::Tensor y = at::empty(y_size, x.options().dtype(c10::ScalarType::Char));
    at::Tensor scale = at::empty(scale_size, x.options().dtype(c10::ScalarType::Float));

    std::string quant_mode_str = quant_mode == 1 ? "dynamic" : "static";
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& weight_scale_value = c10::value_or_else(weight_scale, [] { return at::Tensor(); });
    const at::Tensor& activation_scale_opt = c10::value_or_else(activation_scale, [] { return at::Tensor(); });
    const at::Tensor& bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });
    const at::Tensor& quant_scale_opt = c10::value_or_else(quant_scale, [] { return at::Tensor(); });
    const at::Tensor& quant_offset_opt = c10::value_or_else(quant_offset, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    static const bool is_v2_available =
        GetOpApiFuncAddr("aclnnDequantSwigluQuantV2") != nullptr &&
        GetOpApiFuncAddr("aclnnDequantSwigluQuantV2GetWorkspaceSize") != nullptr;

    if (swiglu_mode == 0 && !is_v2_available) {
        EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_value, activation_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y, scale);
    } else {
        int64_t dst_type = 2;
        char* round_mode = const_cast<char*>("rint");
        int64_t activate_dim = -1;
        EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_value, activation_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, dst_type, round_mode,
                     activate_dim, swiglu_mode, clamp_limit, glu_alpha, glu_bias, y, scale);
    }

    return std::make_tuple(y, scale);
}

void npu_scatter_nd_update_sk(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    // construct the output tensor
    at::IntArrayRef var_stride = var.strides();
    EXEC_NPU_CMD(aclnnScatterNdUpdateSk, var, indices, update, var_stride);
    return;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h_vllm(
    const at::Tensor & k,
    const at::Tensor & w,
    const at::Tensor & u,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & gk,
    const c10::optional<at::Tensor> & initial_state,
    c10::optional<bool> output_final_state,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> save_new_value,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<bool> use_exp2,
    c10::optional<bool> transpose_state_layout)
{
    bool output_final_state_ = output_final_state.has_value() ? output_final_state.value() : false;
    const at::Tensor &initial_state_ = c10::value_or_else(initial_state, [] { return at::Tensor(); });
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    const at::Tensor &gk_ = c10::value_or_else(gk, [] { return at::Tensor(); });

    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int K = k_sizes[3];
    int B = k_sizes[0];
    int T = k_sizes[2];
    int HV = u_sizes[1];
    int V = u_sizes[3];

    int NT = 0;
    if (chunk_indices.has_value()) {
        auto chunk_indices_ref = chunk_indices.value();
        NT = chunk_indices_ref.size() / 2;
    } else {
        NT = (T + chunk_size_ - 1) / chunk_size_;
    }

    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::zeros(u.sizes(), u.options());
    at::Tensor final_state_out;
    if (output_final_state_) {
        int N = cu_seqlens.has_value() ? cu_seqlens->size() - 1 : B;
        auto state_options = initial_state.has_value() ? initial_state->options() : h_out.options();
        final_state_out = at::empty({N, HV, K, V}, state_options);
    } else {
        final_state_out = at::empty({1}, k.options());
    }

    bool save_new_value_ = save_new_value.value_or(true);
    bool use_exp2_ = use_exp2.value_or(false);
    bool transpose_state_layout_ = transpose_state_layout.value_or(false);

    EXEC_NPU_CMD(
        aclnnChunkGatedDeltaRuleFwdHVllm,
        k, w, u, g_,
        gk_, initial_state_, output_final_state_, chunk_size_, save_new_value_,
        cu_seqlens, chunk_indices, use_exp2_, transpose_state_layout_,
        h_out, v_new_out, final_state_out
    );

    if (output_final_state_) {
        return std::make_tuple(h_out, v_new_out, final_state_out);
    } else {
        return std::make_tuple(h_out, v_new_out, at::Tensor());
    }
}

at::Tensor chunk_fwd_o_vllm(
    const at::Tensor & q,
    const at::Tensor & k,
    const at::Tensor & v,
    const at::Tensor & h,
    double scale,
    const c10::optional<at::Tensor> & g,
    const c10::optional<at::Tensor> & g_gamma,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<at::IntArrayRef> chunk_indices,
    c10::optional<int64_t> chunk_size,
    c10::optional<bool> transpose_state_layout)
{
    at::Tensor o = at::zeros(v.sizes(), v.options());
    int64_t chunk_size_ = chunk_size.has_value() ? chunk_size.value() : 64;
    const at::Tensor &g_ = c10::value_or_else(g, [] { return at::Tensor(); });
    (void)g_gamma;
    (void)transpose_state_layout;

    EXEC_NPU_CMD(
        aclnnChunkFwdOVllm,
        q, k, v, h, g_,
        cu_seqlens, chunk_indices, scale, chunk_size_,
        o
    );
    return o;
}

at::Tensor npu_sparse_attention_score_prefill(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &block_table,
    const at::Tensor &k2q_row_ptr,
    const at::Tensor &k2q_q_indices,
    const at::Tensor &k2q_slot_indices,
    int64_t num_key_value_heads, double scale_value, int64_t block_size,
    int64_t top_k, int64_t inner_precise,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv)
{
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    at::ScalarType out_dtype = query.scalar_type();
    if (query.scalar_type() == at::kFloat8_e4m3fn) {
        out_dtype = at::kBFloat16;
    }
    at::Tensor output = at::empty(query.sizes(), query.options().dtype(out_dtype));
    // MinimaxSparseAttentionSplitKv always exposes softmaxLse as its second
    // ACLNN output. Prefill does not consume it, so keep the flag disabled and
    // pass the empty FP32 placeholder required by the operator interface.
    at::Tensor softmax_lse = at::empty({0}, query.options().dtype(at::kFloat));
    bool softmax_lse_flag = false;
    std::string input_layout = "TND";
    char *input_layout_ptr = const_cast<char *>(input_layout.c_str());

    EXEC_NPU_CMD(
        aclnnMinimaxSparseAttentionSplitKv,
        query,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        num_key_value_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise,
        softmax_lse_flag,
        input_layout_ptr,
        output,
        softmax_lse
    );

    return output;
}

bool is_minimax_sparse_attention_split_kv_available()
{
    static const bool is_available = []() {
        auto run = GetOpApiFuncAddr("aclnnMinimaxSparseAttentionSplitKv");
        auto workspace = GetOpApiFuncAddr("aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize");
        if (run == nullptr || workspace == nullptr) {
            return false;
        }
        Dl_info run_info{}, workspace_info{};
        if (dladdr(run, &run_info) == 0 || dladdr(workspace, &workspace_info) == 0 ||
            run_info.dli_fbase != workspace_info.dli_fbase) {
            return false;
        }
        // The metadata-enabled experimental ABI adds metadataOptional to
        // GetWorkspaceSize without renaming it. This binding uses the earlier
        // signature. Reject that known incompatible ABI before invoking it.
        // Inspect the selected provider, not a lower-priority vendor library.
        if (workspace_info.dli_fname == nullptr) {
            return false;
        }
        auto handle = dlopen(workspace_info.dli_fname, RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) {
            return false;
        }
        const bool has_metadata =
            dlsym(handle, "aclnnMinimaxSparseAttentionSplitKvMetadataGetWorkspaceSize") != nullptr;
        dlclose(handle);
        return !has_metadata;
    }();
    return is_available;
}

std::vector<int64_t> get_npu_storage_shape(const at::Tensor& tensor)
{
    TORCH_CHECK(
        tensor.is_privateuseone(),
        "get_npu_storage_shape only supports NPU tensors, but got device ",
        tensor.device());
    const auto& desc = NPUBridge::GetNpuStorageImplDesc(tensor);
    return std::vector<int64_t>(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
}

#if defined(VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD) || defined(VLLM_ASCEND_ENABLE_SELECTIVE_LIM)
namespace {

struct HostCallbackGraphPayloadRegistry {
  aclmdlRI model_ri;
  std::vector<std::shared_ptr<void>> payloads;
};

std::mutex& host_callback_graph_registry_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<aclmdlRI, std::unique_ptr<HostCallbackGraphPayloadRegistry>>&
host_callback_graph_registries() {
  static auto* registries = new std::unordered_map<aclmdlRI, std::unique_ptr<HostCallbackGraphPayloadRegistry>>();
  return *registries;
}

void delete_host_callback_graph_payloads(void* args) noexcept {
  auto* registry = static_cast<HostCallbackGraphPayloadRegistry*>(args);
  std::unique_ptr<HostCallbackGraphPayloadRegistry> owned_registry;
  {
    std::lock_guard<std::mutex> lock(host_callback_graph_registry_mutex());
    auto& registries = host_callback_graph_registries();
    const auto it = registries.find(registry->model_ri);
    if (it != registries.end() && it->second.get() == registry) {
      owned_registry = std::move(it->second);
      registries.erase(it);
    }
  }
}

template <typename Payload>
Payload* retain_host_callback_graph_payload(
    aclmdlRI model_ri, std::unique_ptr<Payload> payload) {
  auto* raw_payload = payload.get();
  std::lock_guard<std::mutex> lock(host_callback_graph_registry_mutex());
  auto& registries = host_callback_graph_registries();
  auto it = registries.find(model_ri);
  if (it == registries.end()) {
    auto registry = std::make_unique<HostCallbackGraphPayloadRegistry>();
    registry->model_ri = model_ri;
    auto* raw_registry = registry.get();
    it = registries.emplace(model_ri, std::move(registry)).first;
    const aclError register_ret =
        aclmdlRIDestroyRegisterCallback(model_ri, delete_host_callback_graph_payloads, raw_registry);
    if (register_ret != ACL_SUCCESS) {
      registries.erase(it);
    }
    TORCH_CHECK(
        register_ret == ACL_SUCCESS,
        "failed to bind Host callback payload to graph lifetime, error code: ", register_ret);
  }
  it->second->payloads.emplace_back(std::move(payload));
  return raw_payload;
}

}  // namespace
#endif

#if defined(VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD) || defined(VLLM_ASCEND_ENABLE_SELECTIVE_LIM)
#if defined(__GNUC__) || defined(__clang__)
  #define HOT_FUNCTION __attribute__((hot))
  #define FORCE_INLINE inline __attribute__((always_inline))
  #define LIKELY(x) __builtin_expect(!!(x), 1)
  #define UNLIKELY(x) __builtin_expect(!!(x), 0)
  #define RESTRICT __restrict__
#else
  #define HOT_FUNCTION
  #define FORCE_INLINE inline
  #define LIKELY(x) (x)
  #define UNLIKELY(x) (x)
  #define RESTRICT
#endif

constexpr int32_t EPOCH_RESET_THRESHOLD = 1 << 30;
constexpr int16_t DIRECT_SELECTION_LAYOUT_MARKER = 0x5A44;

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
FORCE_INLINE int choose_lru_resident_threads(const int num_reqs, const int workspace_threads,
                                             const int requested_threads) {
  if (num_reqs <= 1) {
    return 1;
  }
  int active_threads = num_reqs;
  if (workspace_threads > 0) {
    active_threads = std::min(active_threads, workspace_threads);
  }
  if (requested_threads > 0) {
    active_threads = std::min(active_threads, requested_threads);
  }
  return std::max(active_threads, 1);
}

int64_t warmup_lru_resident_threads(const int64_t requested_threads) {
  const int active_threads = std::max(1, std::min(static_cast<int>(requested_threads), omp_get_max_threads()));
  if (active_threads == 1) {
    return 1;
  }
  int observed_threads = 0;
#pragma omp parallel num_threads(active_threads) reduction(+ : observed_threads)
  {
    observed_threads += 1;
  }
  TORCH_CHECK(observed_threads == active_threads,
              "OpenMP planner warmup created an unexpected thread count: expected=", active_threads,
              ", actual=", observed_threads);
  return observed_threads;
}
#endif

FORCE_INLINE bool is_valid_lru_resident_token(const int32_t token, const int32_t max_token) {
  return token >= 0 && token < max_token;
}

FORCE_INLINE void reset_lru_resident_row(int32_t* RESTRICT slot_to_token_row, int32_t* RESTRICT lru_slots_row,
                                         const int32_t capacity) {
  std::fill(slot_to_token_row, slot_to_token_row + capacity, -1);
  for (int32_t slot = 0; slot < capacity; ++slot) {
    lru_slots_row[slot] = slot;
  }
}

FORCE_INLINE int32_t next_lru_resident_epoch(int32_t* RESTRICT token_mark, int32_t* RESTRICT token_pos,
                                             int32_t* RESTRICT epoch, const int32_t max_token) {
  int32_t base = epoch[0] + 1;
  if (UNLIKELY(base >= EPOCH_RESET_THRESHOLD)) {
    std::fill(token_mark, token_mark + max_token, 0);
    std::fill(token_pos, token_pos + max_token, -1);
    base = 1;
  }
  epoch[0] = base;
  return base;
}

FORCE_INLINE void process_one_lru_resident_row(
    const int logical_row, const int physical_row, const int32_t topk, const int32_t capacity, const int32_t max_token,
    const int64_t* RESTRICT req_ids, int64_t* RESTRICT last_req_ids, const int32_t* RESTRICT topk_indices,
    const int32_t* RESTRICT stable_prefix_lens, int32_t* RESTRICT slot_to_token, int32_t* RESTRICT lru_slots,
    int32_t* RESTRICT current_slots, int32_t* RESTRICT miss_count, int32_t* RESTRICT miss_tokens_out,
    int32_t* RESTRICT miss_slots_out, int32_t* RESTRICT token_mark, int32_t* RESTRICT token_pos,
    int32_t* RESTRICT slot_workspace, int32_t* RESTRICT miss_positions, int32_t* RESTRICT epoch,
    int32_t* RESTRICT current_token_slots, int16_t* RESTRICT encoded_plan, const int32_t encoded_plan_stride,
    const bool encode_physical_row, const int32_t* RESTRICT visible_seq_lens) {
  int32_t* RESTRICT slot_to_token_row = slot_to_token + static_cast<int64_t>(physical_row) * capacity;
  int32_t* RESTRICT lru_slots_row = lru_slots + static_cast<int64_t>(physical_row) * capacity;
  int32_t* RESTRICT current_slots_row = current_slots + static_cast<int64_t>(logical_row) * topk;
  int32_t* RESTRICT miss_tokens_row = miss_tokens_out + static_cast<int64_t>(logical_row) * topk;
  int32_t* RESTRICT miss_slots_row = miss_slots_out + static_cast<int64_t>(logical_row) * topk;
  const int32_t* RESTRICT topk_row = topk_indices + static_cast<int64_t>(logical_row) * topk;
  int32_t* RESTRICT hit_slots = slot_workspace;
  int32_t* RESTRICT evictable_slots = slot_workspace + capacity;
  int32_t* RESTRICT assigned_miss_slots = slot_workspace + static_cast<int64_t>(capacity) * 2;
  int16_t* RESTRICT encoded_plan_row =
      encoded_plan == nullptr ? nullptr : encoded_plan + static_cast<int64_t>(logical_row) * encoded_plan_stride;
  const int32_t resident_capacity = current_token_slots == nullptr ? capacity : capacity - 1;
  if (UNLIKELY(resident_capacity <= 0)) {
    return;
  }

  std::fill(current_slots_row, current_slots_row + topk, -1);
  std::fill(miss_tokens_row, miss_tokens_row + topk, -1);
  std::fill(miss_slots_row, miss_slots_row + topk, -1);
  if (encoded_plan_row != nullptr) {
    std::fill(encoded_plan_row, encoded_plan_row + topk, static_cast<int16_t>(0));
    if (encode_physical_row) {
      encoded_plan_row[topk + 4] = static_cast<int16_t>(physical_row);
      encoded_plan_row[topk + 5] = DIRECT_SELECTION_LAYOUT_MARKER;
      encoded_plan_row[topk + 6] = static_cast<int16_t>(capacity);
    }
  }
  miss_count[logical_row] = 0;

  const int64_t req_id = req_ids[logical_row];
  if (UNLIKELY(last_req_ids[physical_row] != req_id)) {
    reset_lru_resident_row(slot_to_token_row, lru_slots_row, resident_capacity);
    last_req_ids[physical_row] = req_id;
  }
  const int32_t stable_prefix_len = std::clamp(stable_prefix_lens[logical_row], 0, max_token);
  const int32_t visible_seq_len =
      visible_seq_lens == nullptr ? max_token : std::clamp(visible_seq_lens[logical_row], 0, max_token);
  int32_t current_token_slot = resident_capacity;

  const int32_t base = next_lru_resident_epoch(token_mark, token_pos, epoch, max_token);
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = topk_row[pos];
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token < visible_seq_len && token_mark[token] != base) {
      token_mark[token] = base;
      token_pos[token] = pos;
    }
  }

  int32_t hit_count = 0;
  int32_t evictable_count = 0;
  for (int32_t order = 0; order < resident_capacity; ++order) {
    const int32_t slot = lru_slots_row[order];
    if (UNLIKELY(slot < 0 || slot >= resident_capacity)) {
      continue;
    }
    int32_t token = slot_to_token_row[slot];
    // The speculative suffix may have been overwritten in the CPU KV pool.
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && UNLIKELY(token >= stable_prefix_len)) {
      slot_to_token_row[slot] = -1;
      token = -1;
    }
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] == base) {
      const int32_t pos = token_pos[token];
      current_slots_row[pos] = slot;
      if (token == visible_seq_len - 1) {
        current_token_slot = slot;
      }
      if (encoded_plan_row != nullptr) {
        encoded_plan_row[pos] = static_cast<int16_t>(slot + 1);
      }
      hit_slots[hit_count] = slot;
      ++hit_count;
    } else {
      evictable_slots[evictable_count] = slot;
      ++evictable_count;
    }
  }

  int32_t local_miss_count = 0;
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = topk_row[pos];
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token < visible_seq_len &&
        current_slots_row[pos] < 0) {
      miss_tokens_row[local_miss_count] = token;
      miss_positions[local_miss_count] = pos;
      ++local_miss_count;
    }
  }

  const int32_t assign_count = std::min(local_miss_count, evictable_count);
  for (int32_t miss_idx = 0; miss_idx < assign_count; ++miss_idx) {
    const int32_t slot = evictable_slots[miss_idx];
    const int32_t token = miss_tokens_row[miss_idx];
    const int32_t pos = miss_positions[miss_idx];
    slot_to_token_row[slot] = token;
    current_slots_row[pos] = slot;
    if (token == visible_seq_len - 1) {
      current_token_slot = slot;
    }
    if (encoded_plan_row != nullptr) {
      const bool is_current_token = visible_seq_lens != nullptr && token == visible_seq_len - 1;
      encoded_plan_row[pos] = static_cast<int16_t>(is_current_token ? slot + 1 : -(slot + 1));
    }
    miss_slots_row[miss_idx] = slot;
    assigned_miss_slots[miss_idx] = slot;
    miss_count[logical_row] = miss_idx + 1;
  }

  int32_t write_pos = 0;
  for (int32_t idx = assign_count; idx < evictable_count; ++idx) {
    lru_slots_row[write_pos] = evictable_slots[idx];
    ++write_pos;
  }
  for (int32_t idx = 0; idx < assign_count; ++idx) {
    lru_slots_row[write_pos] = assigned_miss_slots[idx];
    ++write_pos;
  }
  for (int32_t idx = 0; idx < hit_count; ++idx) {
    lru_slots_row[write_pos] = hit_slots[idx];
    ++write_pos;
  }
  if (current_token_slots != nullptr) {
    current_token_slots[logical_row] = physical_row * capacity + current_token_slot;
  }
}

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
HOT_FUNCTION void lru_resident_compact_impl(
    uintptr_t req_ids_ptr, uintptr_t last_req_ids_ptr, uintptr_t topk_indices_ptr, uintptr_t stable_prefix_lens_ptr,
    uintptr_t slot_to_token_ptr, uintptr_t lru_slots_ptr, uintptr_t current_slots_ptr, uintptr_t miss_count_ptr,
    uintptr_t miss_tokens_ptr, uintptr_t miss_slots_ptr, uintptr_t token_mark_workspace_ptr,
    uintptr_t token_pos_workspace_ptr, uintptr_t slot_workspace_ptr, uintptr_t miss_position_workspace_ptr,
    uintptr_t epochs_ptr, int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token,
    int64_t workspace_threads, int64_t requested_threads, uintptr_t encoded_plan_ptr, int64_t encoded_plan_stride,
    uintptr_t physical_row_workspace_ptr, int64_t physical_row_capacity, uintptr_t visible_seq_lens_ptr) {
  if (num_reqs <= 0 || topk <= 0 || capacity <= 0 || max_token <= 0 || physical_row_capacity < num_reqs) {
    return;
  }

  auto* RESTRICT req_ids = reinterpret_cast<int64_t*>(req_ids_ptr);
  auto* RESTRICT last_req_ids = reinterpret_cast<int64_t*>(last_req_ids_ptr);
  auto* RESTRICT topk_indices = reinterpret_cast<int32_t*>(topk_indices_ptr);
  auto* RESTRICT stable_prefix_lens = reinterpret_cast<int32_t*>(stable_prefix_lens_ptr);
  auto* RESTRICT slot_to_token = reinterpret_cast<int32_t*>(slot_to_token_ptr);
  auto* RESTRICT lru_slots = reinterpret_cast<int32_t*>(lru_slots_ptr);
  auto* RESTRICT current_slots = reinterpret_cast<int32_t*>(current_slots_ptr);
  auto* RESTRICT miss_count = reinterpret_cast<int32_t*>(miss_count_ptr);
  auto* RESTRICT miss_tokens = reinterpret_cast<int32_t*>(miss_tokens_ptr);
  auto* RESTRICT miss_slots = reinterpret_cast<int32_t*>(miss_slots_ptr);
  auto* RESTRICT token_mark_workspace = reinterpret_cast<int32_t*>(token_mark_workspace_ptr);
  auto* RESTRICT token_pos_workspace = reinterpret_cast<int32_t*>(token_pos_workspace_ptr);
  auto* RESTRICT slot_workspace = reinterpret_cast<int32_t*>(slot_workspace_ptr);
  auto* RESTRICT miss_position_workspace = reinterpret_cast<int32_t*>(miss_position_workspace_ptr);
  auto* RESTRICT epochs = reinterpret_cast<int32_t*>(epochs_ptr);
  auto* RESTRICT encoded_plan = reinterpret_cast<int16_t*>(encoded_plan_ptr);
  auto* RESTRICT physical_row_workspace = reinterpret_cast<int32_t*>(physical_row_workspace_ptr);

  const int num_reqs_int = static_cast<int>(num_reqs);
  const int topk_int = static_cast<int>(topk);
  const int capacity_int = static_cast<int>(capacity);
  const int max_token_int = static_cast<int>(max_token);
  const int physical_row_capacity_int = static_cast<int>(physical_row_capacity);
  int32_t* RESTRICT logical_to_physical = physical_row_workspace;
  int32_t* RESTRICT physical_row_used =
      physical_row_workspace == nullptr ? nullptr : physical_row_workspace + physical_row_capacity_int;
  int32_t* RESTRICT current_token_slots =
      physical_row_workspace == nullptr ? nullptr
                                        : physical_row_workspace + static_cast<int64_t>(physical_row_capacity_int) * 2;
  const int32_t* RESTRICT visible_seq_lens = reinterpret_cast<int32_t*>(visible_seq_lens_ptr);
  if (logical_to_physical == nullptr) {
    physical_row_capacity = num_reqs;
  } else {
    std::fill(logical_to_physical, logical_to_physical + num_reqs_int, -1);
    std::fill(physical_row_used, physical_row_used + physical_row_capacity_int, 0);
    for (int logical_row = 0; logical_row < num_reqs_int; ++logical_row) {
      for (int physical_row = 0; physical_row < physical_row_capacity_int; ++physical_row) {
        if (physical_row_used[physical_row] == 0 && last_req_ids[physical_row] == req_ids[logical_row]) {
          logical_to_physical[logical_row] = physical_row;
          physical_row_used[physical_row] = 1;
          break;
        }
      }
    }
    for (int logical_row = 0; logical_row < num_reqs_int; ++logical_row) {
      if (logical_to_physical[logical_row] >= 0) {
        continue;
      }
      int physical_row = -1;
      for (int candidate = 0; candidate < physical_row_capacity_int; ++candidate) {
        if (physical_row_used[candidate] == 0 && last_req_ids[candidate] == -1) {
          physical_row = candidate;
          break;
        }
      }
      if (physical_row < 0) {
        for (int candidate = 0; candidate < physical_row_capacity_int; ++candidate) {
          if (physical_row_used[candidate] == 0) {
            physical_row = candidate;
            break;
          }
        }
      }
      if (physical_row < 0) {
        return;
      }
      logical_to_physical[logical_row] = physical_row;
      physical_row_used[physical_row] = 1;
    }
  }
  const int active_threads = choose_lru_resident_threads(num_reqs_int, static_cast<int>(workspace_threads),
                                                         static_cast<int>(requested_threads));

  if (active_threads == 1) {
    for (int row = 0; row < num_reqs_int; ++row) {
      const int physical_row = logical_to_physical == nullptr ? row : logical_to_physical[row];
      process_one_lru_resident_row(row, physical_row, topk_int, capacity_int, max_token_int, req_ids, last_req_ids,
                                   topk_indices, stable_prefix_lens, slot_to_token, lru_slots, current_slots,
                                   miss_count, miss_tokens, miss_slots, token_mark_workspace, token_pos_workspace,
                                   slot_workspace, miss_position_workspace, epochs, current_token_slots, encoded_plan,
                                   static_cast<int32_t>(encoded_plan_stride), logical_to_physical != nullptr,
                                   visible_seq_lens);
    }
    return;
  }

#pragma omp parallel num_threads(active_threads)
  {
    const int thread_id = omp_get_thread_num();
    int32_t* RESTRICT token_mark = token_mark_workspace + static_cast<int64_t>(thread_id) * max_token_int;
    int32_t* RESTRICT token_pos = token_pos_workspace + static_cast<int64_t>(thread_id) * max_token_int;
    int32_t* RESTRICT slots = slot_workspace + static_cast<int64_t>(thread_id) * capacity_int * 3;
    int32_t* RESTRICT miss_positions = miss_position_workspace + static_cast<int64_t>(thread_id) * topk_int;
    int32_t* RESTRICT epoch = epochs + thread_id;
    for (int row = thread_id; row < num_reqs_int; row += active_threads) {
      const int physical_row = logical_to_physical == nullptr ? row : logical_to_physical[row];
      process_one_lru_resident_row(row, physical_row, topk_int, capacity_int, max_token_int, req_ids, last_req_ids,
                                   topk_indices, stable_prefix_lens, slot_to_token, lru_slots, current_slots,
                                   miss_count, miss_tokens, miss_slots, token_mark, token_pos, slots, miss_positions,
                                   epoch, current_token_slots, encoded_plan, static_cast<int32_t>(encoded_plan_stride),
                                   logical_to_physical != nullptr, visible_seq_lens);
    }
  }
}
#endif

// This low-level experimental interface is not safe as a standalone API: the
// worker must bind one control and all state buffers to one group/physical row,
// keep them alive, and call finish only after every transfer has completed.
// Different controls must never alias the same state or be paired with another
// row's identity during finish/retire. No such worker is connected yet.
struct SelectivePlanReservation {
  int64_t* control;
  int64_t* identity;
  bool handed_off = false;
  ~SelectivePlanReservation() {
    if (!handed_off) {
      std::fill_n(identity, 9, -1);
      __atomic_store_n(control, 0, __ATOMIC_RELEASE);
    }
  }
};

void selective_check_disjoint(const std::vector<at::Tensor>& tensors) {
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  for (const auto& tensor : tensors) {
    if (tensor.numel() == 0) continue;
    const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(static_cast<uint64_t>(tensor.numel()) <= UINTPTR_MAX / tensor.element_size(),
                "selective tensor size overflow");
    const auto size = static_cast<uintptr_t>(tensor.numel()) * tensor.element_size();
    TORCH_CHECK(begin <= UINTPTR_MAX - size, "selective tensor range overflow");
    const auto end = begin + size;
    for (const auto& range : ranges)
      TORCH_CHECK(end <= range.first || range.second <= begin, "selective tensors overlap");
    ranges.emplace_back(begin, end);
  }
}

std::vector<at::Tensor> selective_group_lru_plan_cpu(
    const at::Tensor& topk, const at::Tensor& page_tables, at::Tensor slot_to_token,
    at::Tensor lru_slots, at::Tensor resident_pages, at::Tensor last_identity, at::Tensor token_mark,
    at::Tensor token_pos, at::Tensor epoch, at::Tensor control, const std::vector<int64_t>& meta) {
  auto check = [](const at::Tensor& t, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() && t.scalar_type() == dtype,
                name, " must be contiguous CPU memory with the requested dtype");
  };
  check(topk, at::kInt, "topk");
  check(page_tables, at::kInt, "page_tables");
  check(slot_to_token, at::kInt, "slot_to_token");
  check(lru_slots, at::kInt, "lru_slots");
  check(resident_pages, at::kInt, "resident_pages");
  check(last_identity, at::kLong, "last_identity");
  check(token_mark, at::kInt, "token_mark");
  check(token_pos, at::kInt, "token_pos");
  check(epoch, at::kInt, "epoch");
  check(control, at::kLong, "control");
  // uid, generation, data version, hot-buffer generation, visible length,
  // tail row, tail offset, block size, number of Host blocks, stable prefix,
  // physical row, total hot rows per request (resident plus tail).
  TORCH_CHECK(meta.size() == 12 && topk.dim() == 1 && page_tables.dim() == 2,
              "invalid selective group inputs");
  const auto width64 = topk.numel(), cap64 = slot_to_token.numel(), max64 = token_mark.numel();
  TORCH_CHECK(width64 > 0 && width64 <= INT32_MAX && cap64 > 0 && cap64 <= INT32_MAX &&
              max64 > 0 && max64 <= INT32_MAX && lru_slots.numel() == cap64 &&
              token_pos.numel() == max64 && last_identity.numel() == 9 && epoch.numel() == 1 &&
              resident_pages.dim() == 2 && resident_pages.size(0) == page_tables.size(0) &&
              resident_pages.size(1) == cap64 &&
              control.numel() == 2 &&
              page_tables.size(0) > 0 && meta[0] > 0 && meta[4] >= 0 && meta[4] <= max64 &&
              meta[5] >= -1 && meta[6] >= 0 && meta[7] > 0 && meta[7] <= INT32_MAX &&
              meta[6] <= meta[7] && meta[8] > 0 && meta[8] <= INT32_MAX &&
              meta[9] >= 0 && meta[9] <= meta[4] && meta[10] >= 0 &&
              cap64 % meta[7] == 0 && meta[11] >= cap64 + meta[7] && meta[11] % meta[7] == 0,
              "invalid selective group geometry");
  const int64_t blocks_per_row = meta[11] / meta[7];
  const int64_t tail_row = meta[5] < 0 ? meta[10] : meta[5];
  const int64_t block_count = cap64 / meta[7];
  TORCH_CHECK(meta[10] <= (INT32_MAX - block_count) / blocks_per_row &&
              tail_row <= (INT32_MAX - block_count) / blocks_per_row,
              "selective block table overflows");
  selective_check_disjoint({topk, page_tables, slot_to_token, lru_slots, resident_pages, last_identity,
                            token_mark, token_pos, epoch, control});
  auto* busy = control.data_ptr<int64_t>();
  int64_t expected = 0;
  TORCH_CHECK(__atomic_compare_exchange_n(busy, &expected, -1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED),
              "selective row already has a plan in flight");
  auto* identity = last_identity.data_ptr<int64_t>();
  SelectivePlanReservation reservation{busy, identity};
  TORCH_CHECK(busy[1] >= 0 && busy[1] < INT64_MAX, "selective ticket overflow");
  const auto ticket = ++busy[1];
  const auto width = static_cast<int32_t>(width64), cap = static_cast<int32_t>(cap64);
  const auto max_token = static_cast<int32_t>(max64), seq = static_cast<int32_t>(meta[4]);
  const auto block_size = static_cast<int32_t>(meta[7]), prefix = static_cast<int32_t>(meta[9]);
  const auto* input = topk.data_ptr<int32_t>();
  const auto* pages = page_tables.data_ptr<int32_t>();
  const int64_t cols = page_tables.size(1);
  std::vector<int32_t> unique(width, -1), original_to_unique(width, -1), tail_slots(width, -1);
  std::unordered_map<int32_t, int32_t> unique_pos;
  int32_t count = 0;
  for (int32_t i = 0; i < width; ++i) {
    const int32_t token = input[i];
    if (token < 0 || token >= seq) continue;
    const int32_t block = token / block_size, offset = token % block_size;
    if (meta[5] >= 0 && block == (seq - 1) / block_size && offset >= meta[6]) {
      TORCH_CHECK(static_cast<int64_t>(cap) + offset <= INT32_MAX, "tail slot overflows");
      tail_slots[i] = cap + offset;
      continue;
    }
    TORCH_CHECK(token != seq - 1, "current token requires a valid NPU tail");
    TORCH_CHECK(block < cols, "historical token is missing a Host page");
    for (int64_t layer = 0; layer < page_tables.size(0); ++layer) {
      const auto page = pages[layer * cols + block];
      TORCH_CHECK(page >= 0 && page < meta[8], "historical token has an invalid Host page");
    }
    auto found = unique_pos.find(token);
    if (found == unique_pos.end()) {
      TORCH_CHECK(count < cap, "more unique historical tokens than resident slots");
      found = unique_pos.emplace(token, count).first;
      unique[count++] = token;
    }
    original_to_unique[i] = found->second;
  }
  auto physical = at::full({width}, -1, topk.options());
  auto miss_tokens = at::full({width}, -1, topk.options());
  auto miss_slots = at::full({width}, -1, topk.options());
  auto block_table = at::empty({block_count + 1}, topk.options());
  auto* table = block_table.data_ptr<int32_t>();
  for (int64_t b = 0; b < block_count; ++b) table[b] = static_cast<int32_t>(meta[10] * blocks_per_row + b);
  table[block_count] = static_cast<int32_t>(tail_row * blocks_per_row + block_count);
  auto ticket_tensor = at::full({1}, ticket, last_identity.options());
  std::vector<int32_t> current(width, -1), workspace(static_cast<size_t>(cap) * 3), positions(width);
  int64_t last_req = std::equal(identity, identity + 4, meta.begin()) && identity[4] == meta[10] &&
                     identity[5] == meta[7] && identity[6] == meta[11] && identity[7] == cap64 &&
                     identity[8] == page_tables.size(0) ? meta[0] : -1;
  auto* epoch_value = epoch.data_ptr<int32_t>();
  TORCH_CHECK(*epoch_value >= 0 && *epoch_value < EPOCH_RESET_THRESHOLD, "invalid selective epoch");
  if (last_req == meta[0]) {
    std::vector<bool> seen_slots(cap, false);
    std::unordered_map<int32_t, bool> seen_tokens;
    const auto* lru = lru_slots.data_ptr<int32_t>();
    auto* resident = slot_to_token.data_ptr<int32_t>();
    auto* resident_page_data = resident_pages.data_ptr<int32_t>();
    auto* marks = token_mark.data_ptr<int32_t>();
    bool reset_marks = false;
    for (int32_t i = 0; i < cap; ++i) {
      TORCH_CHECK(lru[i] >= 0 && lru[i] < cap && !seen_slots[lru[i]], "invalid selective LRU slots");
      seen_slots[lru[i]] = true;
      const auto token = resident[i];
      TORCH_CHECK(token >= -1 && token < max_token, "invalid selective resident token");
      if (token >= 0) {
        TORCH_CHECK(seen_tokens.emplace(token, true).second, "duplicate selective resident token");
        const auto block = token / block_size;
        bool remapped = block >= cols;
        if (!remapped) for (int64_t layer = 0; layer < page_tables.size(0); ++layer) {
          const int32_t page = pages[layer * cols + block];
          if (page < 0 || page >= meta[8] || resident_page_data[layer * cap + i] != page) {
            remapped = true;
            break;
          }
        }
        if (remapped) {
          resident[i] = -1;
          for (int64_t layer = 0; layer < page_tables.size(0); ++layer)
            resident_page_data[layer * cap + i] = -1;
          continue;
        }
        reset_marks |= marks[token] == *epoch_value + 1;
      }
    }
    if (reset_marks) {
      std::fill_n(marks, max_token, 0);
      std::fill_n(token_pos.data_ptr<int32_t>(), max_token, -1);
      *epoch_value = 0;
    }
  }
  int32_t miss_count = 0;
  process_one_lru_resident_row(0, 0, width, cap, max_token, meta.data(), &last_req, unique.data(), &prefix,
      slot_to_token.data_ptr<int32_t>(), lru_slots.data_ptr<int32_t>(), current.data(), &miss_count,
      miss_tokens.data_ptr<int32_t>(), miss_slots.data_ptr<int32_t>(), token_mark.data_ptr<int32_t>(),
      token_pos.data_ptr<int32_t>(), workspace.data(), positions.data(), epoch.data_ptr<int32_t>(),
      nullptr, nullptr, 0, false, &seq);
  auto* resident_page_data = resident_pages.data_ptr<int32_t>();
  const auto* resident = slot_to_token.data_ptr<int32_t>();
  for (int32_t slot = 0; slot < cap; ++slot) {
    const int32_t token = resident[slot];
    for (int64_t layer = 0; layer < page_tables.size(0); ++layer) {
      resident_page_data[layer * cap + slot] = token < 0 ? -1 : pages[layer * cols + token / block_size];
    }
  }
  for (int32_t i = 0; i < width; ++i) {
    if (tail_slots[i] >= 0) physical.data_ptr<int32_t>()[i] = tail_slots[i];
    else if (original_to_unique[i] >= 0) {
      const auto slot = current[original_to_unique[i]];
      TORCH_CHECK(slot >= 0 && slot < cap, "resident planning left a valid token unmapped");
      physical.data_ptr<int32_t>()[i] = slot;
    }
  }
  std::copy_n(meta.begin(), 4, identity);
  identity[4] = meta[10];
  identity[5] = meta[7];
  identity[6] = meta[11];
  identity[7] = cap64;
  identity[8] = page_tables.size(0);
  // Construct every potentially throwing return component before publishing
  // the ticket. The reservation still invalidates/releases the row if any
  // tensor view or vector allocation throws.
  std::vector<at::Tensor> result{physical, miss_tokens.narrow(0, 0, miss_count),
                                 miss_slots.narrow(0, 0, miss_count), block_table, ticket_tensor};
  static_assert(std::is_nothrow_move_constructible<std::vector<at::Tensor>>::value,
                "selective plan return must not throw after ticket publication");
  __atomic_store_n(busy, ticket, __ATOMIC_RELEASE);
  reservation.handed_off = true;
  return result;
}

// Each rank publishes a bounded serialized value and retains its slot until
// all peers have copied it.  The first int64 is a sticky fail-closed flag.
std::vector<std::string> selective_tp_exchange_cpu(at::Tensor storage, int64_t rank, int64_t world,
                                                   int64_t epoch, const std::string& value) {
  TORCH_CHECK(storage.device().is_cpu() && storage.is_contiguous() && storage.scalar_type() == at::kByte &&
              world > 1 && rank >= 0 && rank < world && epoch > 0, "invalid selective TP exchange");
  constexpr int64_t capacity = 1024 * 1024;
  TORCH_CHECK(world <= (INT64_MAX - 8) / (capacity + 32), "selective TP exchange world overflow");
  const auto header = (1 + 3 * world) * int64_t{8};
  TORCH_CHECK(storage.numel() == header + world * (capacity + 8), "invalid selective TP exchange size");
  auto* base = storage.data_ptr<uint8_t>();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(base) % alignof(int64_t) == 0, "unaligned selective TP exchange");
  auto* control = reinterpret_cast<int64_t*>(base);
  auto load = [&](int64_t index) { return __atomic_load_n(control + index, __ATOMIC_ACQUIRE); };
  auto store = [&](int64_t index, int64_t item) { __atomic_store_n(control + index, item, __ATOMIC_RELEASE); };
  try {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    auto wait = [&](int64_t index, int64_t expected) {
      for (;;) {
        TORCH_CHECK(load(0) == 0, "selective TP exchange peer failed");
        const auto observed = load(index);
        TORCH_CHECK(observed <= expected, "selective TP exchange epoch mismatch");
        if (observed == expected) return;
        TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "selective TP exchange timed out");
        std::this_thread::yield();
      }
    };
    for (int64_t i = 0; i < world; ++i) wait(1 + world + i, epoch - 1);
    auto* slot = base + header + rank * (capacity + 8);
    const int64_t length = value.size() > static_cast<size_t>(capacity) ? -1 : static_cast<int64_t>(value.size());
    std::memcpy(slot, &length, sizeof(length));
    if (length >= 0) std::memcpy(slot + 8, value.data(), static_cast<size_t>(length));
    store(1 + rank, epoch);
    for (int64_t i = 0; i < world; ++i) wait(1 + i, epoch);
    std::vector<std::string> values;
    values.reserve(world);
    bool overflow = false;
    for (int64_t i = 0; i < world; ++i) {
      const auto* peer = base + header + i * (capacity + 8);
      int64_t peer_length;
      std::memcpy(&peer_length, peer, sizeof(peer_length));
      TORCH_CHECK(peer_length >= -1 && peer_length <= capacity, "invalid selective TP exchange payload");
      if (peer_length < 0) overflow = true;
      else values.emplace_back(reinterpret_cast<const char*>(peer + 8), static_cast<size_t>(peer_length));
    }
    store(1 + world + rank, epoch);
    if (overflow) return {};  // All ranks use the existing collective for this phase.
    return values;
  } catch (...) {
    store(0, 1);
    throw;
  }
}

void selective_tp_exchange_confirm_cpu(at::Tensor storage, int64_t rank, int64_t world, int64_t epoch) {
  TORCH_CHECK(storage.device().is_cpu() && storage.is_contiguous() && storage.scalar_type() == at::kByte &&
              world > 1 && rank >= 0 && rank < world && epoch > 0,
              "invalid selective TP exchange confirmation");
  TORCH_CHECK(world <= (INT64_MAX / 8 - 1) / 3, "selective TP exchange world overflow");
  TORCH_CHECK(storage.numel() >= (1 + 3 * world) * 8 &&
              reinterpret_cast<uintptr_t>(storage.data_ptr<uint8_t>()) % 8 == 0,
              "invalid selective TP exchange confirmation storage");
  auto* control = reinterpret_cast<int64_t*>(storage.data_ptr<uint8_t>());
  auto load = [&](int64_t index) { return __atomic_load_n(control + index, __ATOMIC_ACQUIRE); };
  try {
    TORCH_CHECK(load(0) == 0, "selective TP exchange peer failed");
    __atomic_store_n(control + 1 + 2 * world + rank, epoch, __ATOMIC_RELEASE);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (int64_t peer = 0; peer < world; ++peer) {
      for (;;) {
        TORCH_CHECK(load(0) == 0, "selective TP exchange peer failed");
        const auto observed = load(1 + 2 * world + peer);
        TORCH_CHECK(observed <= epoch, "selective TP exchange confirmation epoch mismatch");
        if (observed == epoch) break;
        TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "selective TP exchange confirmation timed out");
        std::this_thread::yield();
      }
    }
  } catch (...) {
    __atomic_store_n(control, int64_t{1}, __ATOMIC_RELEASE);
    throw;
  }
}

void selective_tp_exchange_fail_cpu(at::Tensor storage) {
  TORCH_CHECK(storage.device().is_cpu() && storage.is_contiguous() && storage.scalar_type() == at::kByte &&
              storage.numel() >= 8 && reinterpret_cast<uintptr_t>(storage.data_ptr<uint8_t>()) % 8 == 0,
              "invalid selective TP exchange storage");
  __atomic_store_n(reinterpret_cast<int64_t*>(storage.data_ptr<uint8_t>()), int64_t{1}, __ATOMIC_RELEASE);
}

void selective_group_lru_finish_cpu(at::Tensor last_identity, at::Tensor control, int64_t ticket, bool success) {
  TORCH_CHECK(last_identity.device().is_cpu() && last_identity.is_contiguous() &&
              last_identity.scalar_type() == at::kLong && last_identity.numel() == 9 &&
              control.device().is_cpu() && control.is_contiguous() && control.scalar_type() == at::kLong &&
              control.numel() == 2 && ticket > 0, "invalid selective finish arguments");
  selective_check_disjoint({last_identity, control});
  auto* busy = control.data_ptr<int64_t>();
  int64_t expected = ticket;
  TORCH_CHECK(__atomic_compare_exchange_n(busy, &expected, -ticket, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED),
              "selective ticket is not in flight or is already finishing");
  if (!success) std::fill_n(last_identity.data_ptr<int64_t>(), 9, -1);
  __atomic_store_n(busy, 0, __ATOMIC_RELEASE);
}

bool selective_group_lru_retire_cpu(at::Tensor last_identity, at::Tensor control,
                                    int64_t request, int64_t generation) {
  TORCH_CHECK(last_identity.device().is_cpu() && last_identity.is_contiguous() &&
              last_identity.scalar_type() == at::kLong && last_identity.numel() == 9 && request > 0 &&
              control.device().is_cpu() && control.is_contiguous() && control.scalar_type() == at::kLong &&
              control.numel() == 2,
              "invalid selective group retirement arguments");
  selective_check_disjoint({last_identity, control});
  auto* busy = control.data_ptr<int64_t>();
  int64_t expected = 0;
  TORCH_CHECK(__atomic_compare_exchange_n(busy, &expected, -1, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED),
              "cannot retire a row with a plan in flight");
  auto* identity = last_identity.data_ptr<int64_t>();
  const bool matches = identity[0] == request && identity[1] == generation;
  if (matches) std::fill_n(identity, 9, -1);
  __atomic_store_n(busy, 0, __ATOMIC_RELEASE);
  return matches;
}

// Native-worker-only transfer buffers. The owner must hold these tensors, the
// actual MemFabric mapping and the pinned registration until all H2D and graph
// scatter work completes; a tensor fabricated from a remote GVA is not proof of
// a CPU mapping. This type is deliberately not registered as a Python op.
struct SelectiveGroupLayerTransfer {
  at::Tensor host_k, host_v, hot_k, hot_v;
  at::Tensor stage_host_k, stage_host_v, stage_device_k, stage_device_v;
  at::Tensor destinations_host, destinations_device, lengths_host, lengths_device;
  at::Tensor pages;
};

struct SelectiveGroupCopyRow {
  size_t layer, part, staged_offset, host_offset, hot_offset, bytes;
};

size_t selective_tensor_bytes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.defined() && tensor.is_contiguous(), "selective transfer requires contiguous tensors");
  TORCH_CHECK(static_cast<uint64_t>(tensor.numel()) <= SIZE_MAX / tensor.element_size(),
              "selective transfer tensor size overflow");
  return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool selective_is_pinned_or_registered(const at::Tensor& tensor) {
  if (tensor.is_pinned()) return true;
  const auto size = selective_tensor_bytes(tensor);
  const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  if (!size || base > UINTPTR_MAX - size) return false;
  std::lock_guard<std::mutex> lock(get_pinned_host_mmap_mutex());
  for (const auto& entry : get_pinned_host_mmaps()) {
    const auto start = reinterpret_cast<uintptr_t>(entry.first);
    if (start <= UINTPTR_MAX - entry.second && base >= start && base + size <= start + entry.second) return true;
  }
  return false;
}

void selective_check_read_write_disjoint(const std::vector<at::Tensor>& inputs,
                                         const std::vector<at::Tensor>& outputs) {
  std::vector<std::pair<uintptr_t, uintptr_t>> written;
  for (const auto& tensor : outputs) {
    const size_t size = selective_tensor_bytes(tensor);
    if (!size) continue;
    const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(base <= UINTPTR_MAX - size, "selective writable range overflow");
    for (const auto& range : written)
      TORCH_CHECK(base + size <= range.first || range.second <= base, "selective writable buffers overlap");
    written.emplace_back(base, base + size);
  }
  for (const auto& tensor : inputs) {
    const size_t size = selective_tensor_bytes(tensor);
    if (!size) continue;
    const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(base <= UINTPTR_MAX - size, "selective input range overflow");
    for (const auto& range : written)
      TORCH_CHECK(base + size <= range.first || range.second <= base, "selective input and output overlap");
  }
}

// Produces compact payloads and fixed-capacity scatter metadata for one stable
// request row. This function only runs on a host worker, never an ACL callback.
// It requires the mapping and page-locked staging registration to have already
// been validated by the owner. Descriptor validation precedes any gather copy.
void selective_gather_group_rows(
    std::vector<SelectiveGroupLayerTransfer>& layers, const at::Tensor& miss_tokens,
    const at::Tensor& miss_slots, int64_t physical_row, int64_t row_stride,
    int64_t block_size, int64_t capacity, size_t k_bytes, size_t v_bytes, bool gather_host,
    size_t logical_row = 0, size_t batch_rows = 1) {
  TORCH_CHECK(!layers.empty() && miss_tokens.device().is_cpu() && miss_slots.device().is_cpu() &&
              miss_tokens.scalar_type() == at::kInt && miss_slots.scalar_type() == at::kInt &&
              miss_tokens.is_contiguous() && miss_slots.is_contiguous() &&
              miss_tokens.numel() == miss_slots.numel() && miss_tokens.numel() <= capacity &&
              physical_row >= 0 && row_stride > 0 && block_size > 0 && capacity > 0 &&
              capacity <= row_stride && capacity <= INT64_MAX / 2 && batch_rows > 0 &&
              logical_row < batch_rows && batch_rows <= static_cast<size_t>(INT64_MAX) /
                                                             (2 * static_cast<size_t>(capacity)) &&
              k_bytes > 0 && v_bytes > 0 &&
              k_bytes <= INT32_MAX && v_bytes <= INT32_MAX &&
              static_cast<uint64_t>(physical_row) <= SIZE_MAX / static_cast<uint64_t>(row_stride),
              "invalid selective group transfer geometry");
  const size_t count = static_cast<size_t>(miss_tokens.numel());
  const size_t cap = static_cast<size_t>(capacity);
  TORCH_CHECK(cap <= SIZE_MAX / 2 && cap <= SIZE_MAX / k_bytes && cap <= SIZE_MAX / v_bytes,
              "selective group staging size overflow");
  const size_t row_base = static_cast<size_t>(physical_row) * static_cast<size_t>(row_stride);
  TORCH_CHECK(row_base <= SIZE_MAX - cap, "selective group hot row overflow");
  const auto* tokens = miss_tokens.data_ptr<int32_t>();
  const auto* slots = miss_slots.data_ptr<int32_t>();
  std::unordered_map<int32_t, bool> unique_tokens, unique_slots;
  for (size_t i = 0; i < count; ++i) {
    TORCH_CHECK(tokens[i] >= 0 && slots[i] >= 0 && slots[i] < capacity &&
                unique_tokens.emplace(tokens[i], true).second && unique_slots.emplace(slots[i], true).second,
                "invalid or duplicate selective miss");
  }
  std::vector<SelectiveGroupCopyRow> rows;
  std::vector<at::Tensor> inputs{miss_tokens, miss_slots}, outputs;
  TORCH_CHECK(layers.size() <= SIZE_MAX / 2 && layers.size() * 2 <= SIZE_MAX / std::max<size_t>(count, 1),
              "selective descriptor count overflow");
  rows.reserve(layers.size() * 2 * count);
  for (size_t layer = 0; layer < layers.size(); ++layer) {
    auto& item = layers[layer];
    if (gather_host) {
      TORCH_CHECK(item.pages.device().is_cpu() && item.pages.is_contiguous() &&
                  item.pages.scalar_type() == at::kInt, "invalid selective Host page table");
    }
    TORCH_CHECK(item.destinations_host.device().is_cpu() && item.destinations_host.scalar_type() == at::kLong &&
                item.lengths_host.device().is_cpu() && item.lengths_host.scalar_type() == at::kInt &&
                item.destinations_host.numel() >= static_cast<int64_t>(cap * 2 * batch_rows) &&
                item.lengths_host.numel() >= static_cast<int64_t>(cap * 2 * batch_rows),
                "invalid selective scatter metadata");
    selective_tensor_bytes(item.destinations_host);
    selective_tensor_bytes(item.lengths_host);
    outputs.insert(outputs.end(), {item.destinations_host, item.lengths_host});
    if (gather_host) {
      inputs.insert(inputs.end(), {item.pages, item.host_k, item.host_v});
      outputs.insert(outputs.end(), {item.stage_host_k, item.stage_host_v});
    } else {
      inputs.insert(inputs.end(), {item.stage_host_k, item.stage_host_v});
    }
    const at::Tensor host[] = {item.host_k, item.host_v};
    const at::Tensor hot[] = {item.hot_k, item.hot_v};
    const at::Tensor staging[] = {item.stage_host_k, item.stage_host_v};
    const size_t bytes[] = {k_bytes, v_bytes};
    for (size_t part = 0; part < 2; ++part) {
      TORCH_CHECK(staging[part].device().is_cpu() && hot[part].device().is_privateuseone() &&
                  (!gather_host || host[part].device().is_cpu()), "invalid selective Host or NPU tensor");
      const size_t host_size = gather_host ? selective_tensor_bytes(host[part]) : 0;
      const size_t hot_size = selective_tensor_bytes(hot[part]);
      TORCH_CHECK(selective_tensor_bytes(staging[part]) >= cap * bytes[part] &&
                  row_base + cap <= hot_size / bytes[part], "selective staging or hot buffer too small");
      const auto* pages = gather_host ? item.pages.data_ptr<int32_t>() : nullptr;
      for (size_t i = 0; i < count; ++i) {
        size_t source_row = 0;
        if (gather_host) {
          const auto block = static_cast<int64_t>(tokens[i]) / block_size;
          const auto offset = static_cast<int64_t>(tokens[i]) % block_size;
          TORCH_CHECK(block < item.pages.numel() && pages[block] >= 0 &&
                      static_cast<uint64_t>(pages[block]) <= (SIZE_MAX - static_cast<uint64_t>(offset)) /
                                                               static_cast<uint64_t>(block_size),
                      "selective missing or invalid Host page");
          source_row = static_cast<size_t>(pages[block]) * static_cast<size_t>(block_size) +
                       static_cast<size_t>(offset);
          TORCH_CHECK(source_row < host_size / bytes[part], "selective Host page exceeds mapped buffer");
          TORCH_CHECK(reinterpret_cast<uintptr_t>(host[part].data_ptr()) <=
                          UINTPTR_MAX - source_row * bytes[part], "selective Host address overflow");
        }
        const auto hot_offset = (row_base + static_cast<size_t>(slots[i])) * bytes[part];
        TORCH_CHECK(reinterpret_cast<uintptr_t>(hot[part].data_ptr()) <= UINTPTR_MAX - hot_offset,
                    "selective destination address overflow");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(staging[part].data_ptr()) <= UINTPTR_MAX - i * bytes[part],
                    "selective staging address overflow");
        rows.push_back({layer, part, i * bytes[part], source_row * bytes[part],
                        hot_offset, bytes[part]});
      }
    }
  }
  selective_check_read_write_disjoint(inputs, outputs);
  for (auto& item : layers) {
    for (size_t part = 0; part < 2; ++part) {
      const auto offset = (part * batch_rows + logical_row) * cap;
      std::fill_n(item.destinations_host.data_ptr<int64_t>() + offset, cap, 0);
      std::fill_n(item.lengths_host.data_ptr<int32_t>() + offset, cap, 0);
    }
  }
  for (const auto& row : rows) {
    auto& item = layers[row.layer];
    auto& host = row.part == 0 ? item.host_k : item.host_v;
    auto& hot = row.part == 0 ? item.hot_k : item.hot_v;
    auto& stage = row.part == 0 ? item.stage_host_k : item.stage_host_v;
    const size_t i = row.staged_offset / row.bytes;
    if (gather_host) {
      std::memcpy(static_cast<uint8_t*>(stage.data_ptr()) + row.staged_offset,
                  static_cast<const uint8_t*>(host.data_ptr()) + row.host_offset, row.bytes);
    }
    const auto base = reinterpret_cast<uintptr_t>(hot.data_ptr());
    const auto index = (row.part * batch_rows + logical_row) * cap + i;
    item.destinations_host.data_ptr<int64_t>()[index] =
        static_cast<int64_t>(base + row.hot_offset);
    item.lengths_host.data_ptr<int32_t>()[index] = static_cast<int32_t>(row.bytes);
  }
}

struct SelectiveNativeOwner {
  std::vector<std::pair<uintptr_t, size_t>> state, hot;
  int device = -1;
  int64_t tp_rank = 0, tp_world = 1;
  size_t references = 0;
  bool in_flight = false;
};
struct SelectiveNativeSpans {
  uintptr_t owner = 0;
  int device = -1;
  std::vector<std::pair<uintptr_t, size_t>> host_reads, host_writes, device_reads, device_writes;
};
constexpr int64_t SELECTIVE_TP_COMMIT_EPOCH = 27;
constexpr int64_t SELECTIVE_TP_ACK_BASE = 28;
std::mutex& selective_owner_mutex() { static auto* mutex = new std::mutex(); return *mutex; }
std::unordered_map<uintptr_t, SelectiveNativeOwner>& selective_owners() {
  static auto* owners = new std::unordered_map<uintptr_t, SelectiveNativeOwner>();
  return *owners;
}
std::unordered_map<const void*, SelectiveNativeSpans>& selective_owner_spans() {
  static auto* spans = new std::unordered_map<const void*, SelectiveNativeSpans>();
  return *spans;
}

// Experimental group native graph handle. The normal
// selective path does not call this entry until its version, graph-completion
// and TP lifecycle contracts are validated. All tensor owners are retained by
// the handle through graph destruction and worker shutdown.
class SelectiveNativeGraph {
 public:
  explicit SelectiveNativeGraph(std::vector<at::Tensor> buffers, std::vector<at::Tensor> tp = {},
                                int64_t rank = 0, int64_t world = 1)
      : b_(std::move(buffers)), tp_(std::move(tp)), tp_rank_(rank), tp_world_(world) {
    TORCH_CHECK(tp_world_ > 0 && tp_world_ <= (INT64_MAX - SELECTIVE_TP_ACK_BASE) / 2 &&
                tp_rank_ >= 0 && tp_rank_ < tp_world_ &&
                (tp_world_ == 1 ? tp_.empty() : tp_.size() == 8), "invalid selective TP configuration");
    TORCH_CHECK(b_.size() >= 30 && (b_.size() - 30) % 14 == 0, "invalid selective native tensor list");
    layer_count_ = 1 + (b_.size() - 30) / 14;
    const int device_indices[] = {0, 2, 4, 8, 9, 12, 13, 15, 17, 19, 21};
    for (int i : device_indices)
      TORCH_CHECK(b_[i].defined() && b_[i].device().is_privateuseone() && b_[i].is_contiguous() &&
                  b_[i].device() == b_[0].device(),
                  "selective native device buffer invalid at ", i);
    const int host_indices[] = {1, 3, 5, 6, 7, 10, 11, 14, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29};
    for (int i : host_indices)
      TORCH_CHECK(b_[i].defined() && b_[i].device().is_cpu() && b_[i].is_contiguous(),
                  "selective native host buffer invalid at ", i);
    const int pinned_indices[] = {1, 3, 5, 10, 11, 14, 16, 18, 20};
    for (int i : pinned_indices)
      TORCH_CHECK(selective_is_pinned_or_registered(b_[i]), "selective native buffer is not pinned at ", i);
    TORCH_CHECK(b_[0].scalar_type() == at::kInt && b_[1].scalar_type() == at::kInt &&
                b_[0].numel() == b_[1].numel() && b_[0].numel() > 0 &&
                b_[2].scalar_type() == at::kInt && b_[3].scalar_type() == at::kInt &&
                b_[2].numel() == b_[3].numel() && b_[4].scalar_type() == at::kLong &&
                b_[5].scalar_type() == at::kLong && b_[4].numel() > 0 && b_[4].numel() % 12 == 0 &&
                b_[5].numel() == b_[4].numel(),
                "invalid selective native metadata buffers");
    rows_ = b_[4].numel() / 12;
    TORCH_CHECK(b_[0].numel() % rows_ == 0 && b_[2].numel() % rows_ == 0,
                "invalid selective native request rows");
    width_ = b_[0].numel() / rows_;
    page_columns_ = b_[2].numel() / rows_;
    TORCH_CHECK(width_ > 0 && width_ <= 2048 && b_[0].numel() <= INT64_MAX / 2 &&
                b_[22].numel() > 0 && b_[22].numel() % width_ == 0 &&
                static_cast<uint64_t>(layer_count_) <= static_cast<uint64_t>(INT64_MAX) /
                                                         static_cast<uint64_t>(b_[22].numel()) &&
                b_[22].numel() / width_ <= INT64_MAX / 9,
                "selective native tensor geometry overflows");
    TORCH_CHECK(width_ > 0 && width_ <= 2048 && page_columns_ > 0 && b_[22].numel() % width_ == 0 &&
                b_[22].numel() >= rows_ * width_ && b_[23].numel() == b_[22].numel() &&
                b_[24].numel() == (b_[22].numel() / width_) * static_cast<int64_t>(layer_count_) * width_ &&
                b_[25].numel() == (b_[22].numel() / width_) * 9 &&
                b_[14].numel() == 2 * rows_ * width_ && b_[15].numel() == b_[14].numel() &&
                b_[16].numel() == b_[14].numel() && b_[17].numel() == b_[14].numel() &&
                b_[18].numel() == rows_ * width_ && b_[19].numel() == b_[18].numel() &&
                b_[20].numel() >= rows_ && b_[20].numel() == b_[21].numel() && b_[20].numel() % rows_ == 0 &&
                b_[26].numel() > 0 && b_[26].numel() == b_[27].numel() &&
                b_[26].numel() % (b_[22].numel() / width_) == 0 &&
                b_[28].numel() == b_[22].numel() / width_ && b_[29].numel() == 2 * b_[28].numel(),
                "invalid selective native buffer geometry");
    const int int32_indices[] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28};
    for (int i : int32_indices) TORCH_CHECK(b_[i].scalar_type() == at::kInt, "invalid int32 buffer ", i);
    const int int64_indices[] = {14, 15, 25, 29};
    for (int i : int64_indices) TORCH_CHECK(b_[i].scalar_type() == at::kLong, "invalid int64 buffer ", i);
    for (size_t layer = 1; layer < layer_count_; ++layer) {
      const size_t n = 30 + (layer - 1) * 14;
      for (size_t i : {size_t{0}, size_t{4}, size_t{5}, size_t{8}, size_t{9}, size_t{11}, size_t{13}})
        TORCH_CHECK(b_[n + i].defined() && b_[n + i].device() == b_[0].device() && b_[n + i].is_contiguous(),
                    "invalid selective layer device tensor");
      for (size_t i : {size_t{1}, size_t{2}, size_t{3}, size_t{6}, size_t{7}, size_t{10}, size_t{12}})
        TORCH_CHECK(b_[n + i].defined() && b_[n + i].device().is_cpu() && b_[n + i].is_contiguous(),
                    "invalid selective layer host tensor");
      for (size_t i : {size_t{1}, size_t{6}, size_t{7}, size_t{10}, size_t{12}})
        TORCH_CHECK(selective_is_pinned_or_registered(b_[n + i]), "selective layer host tensor is not pinned");
      TORCH_CHECK(b_[n].scalar_type() == at::kInt && b_[n + 1].scalar_type() == at::kInt &&
                  b_[n].numel() == rows_ * page_columns_ && b_[n + 1].numel() == b_[n].numel() &&
                  b_[n + 10].scalar_type() == at::kLong && b_[n + 11].scalar_type() == at::kLong &&
                  b_[n + 12].scalar_type() == at::kInt && b_[n + 13].scalar_type() == at::kInt &&
                  b_[n + 10].numel() == 2 * rows_ * width_ && b_[n + 11].numel() == b_[n + 10].numel() &&
                  b_[n + 12].numel() == b_[n + 10].numel() && b_[n + 13].numel() == b_[n + 10].numel(),
                  "invalid selective layer geometry");
    }
    if (tp_world_ > 1) {
      for (auto& tensor : tp_) TORCH_CHECK(tensor.defined() && tensor.device().is_cpu() && tensor.is_contiguous(),
                                           "invalid selective TP shared tensor");
      for (size_t i : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{7}})
        TORCH_CHECK(selective_is_pinned_or_registered(tp_[i]), "selective TP shared tensor is not registered");
      TORCH_CHECK(reinterpret_cast<uintptr_t>(tp_[0].data_ptr()) % alignof(int64_t) == 0 &&
                  reinterpret_cast<uintptr_t>(tp_[2].data_ptr()) % alignof(int64_t) == 0 &&
                  reinterpret_cast<uintptr_t>(tp_[1].data_ptr()) % alignof(int32_t) == 0 &&
                  reinterpret_cast<uintptr_t>(tp_[3].data_ptr()) % alignof(int32_t) == 0 &&
                  reinterpret_cast<uintptr_t>(tp_[4].data_ptr()) % alignof(int32_t) == 0 &&
                  reinterpret_cast<uintptr_t>(tp_[6].data_ptr()) % alignof(int64_t) == 0 &&
                  reinterpret_cast<uintptr_t>(tp_[7].data_ptr()) % alignof(int64_t) == 0,
                  "selective TP shared tensor alignment is invalid");
      const auto q = b_[20].numel() / rows_;
      TORCH_CHECK(tp_[2].numel() > 0 && tp_[2].numel() % 12 == 0, "invalid selective TP row capacity");
      tp_capacity_rows_ = tp_[2].numel() / 12;
      TORCH_CHECK(tp_capacity_rows_ >= rows_ && q <= INT64_MAX - 3 * width_ - 1 &&
                  tp_capacity_rows_ <= INT64_MAX / (3 * width_ + q + 1) &&
                  tp_capacity_rows_ <= INT64_MAX / width_ &&
                  static_cast<uint64_t>(layer_count_) <= static_cast<uint64_t>(INT64_MAX) /
                      (static_cast<uint64_t>(tp_capacity_rows_) * static_cast<uint64_t>(width_)),
                  "selective TP plan geometry overflows");
      TORCH_CHECK(tp_[0].scalar_type() == at::kLong && tp_[0].numel() == SELECTIVE_TP_ACK_BASE + 2 * tp_world_ &&
                  tp_[1].scalar_type() == at::kInt && tp_[1].numel() == tp_capacity_rows_ * (3 * width_ + q + 1) &&
                  tp_[2].scalar_type() == at::kLong &&
                  tp_[3].scalar_type() == at::kInt && tp_[3].numel() == tp_capacity_rows_ * width_ &&
                  tp_[4].scalar_type() == at::kInt &&
                  tp_[4].numel() == static_cast<int64_t>(layer_count_) * tp_capacity_rows_ * width_ &&
                  tp_[6].scalar_type() == at::kLong && tp_[7].scalar_type() == at::kLong &&
                  tp_[6].numel() == static_cast<int64_t>(layer_count_) + 1 && tp_[7].numel() == tp_[6].numel(),
                  "invalid selective TP shared layout");
      const auto owner_base = reinterpret_cast<uintptr_t>(tp_[5].data_ptr());
      const auto owner_bytes = selective_tensor_bytes(tp_[5]);
      TORCH_CHECK(owner_bytes && owner_bytes <= static_cast<size_t>(INT64_MAX) &&
                  owner_base <= UINTPTR_MAX - owner_bytes, "invalid selective TP mapping owner");
      for (size_t i : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{7}}) {
        const auto base = reinterpret_cast<uintptr_t>(tp_[i].data_ptr());
        const auto bytes = selective_tensor_bytes(tp_[i]);
        TORCH_CHECK(base >= owner_base && bytes <= owner_bytes && base - owner_base <= owner_bytes - bytes,
                    "selective TP tensor is outside mapping owner");
      }
      for (int i : {10, 11}) {
        const auto base = reinterpret_cast<uintptr_t>(b_[i].data_ptr());
        const auto bytes = selective_tensor_bytes(b_[i]);
        TORCH_CHECK(base >= owner_base && bytes <= owner_bytes && base - owner_base <= owner_bytes - bytes,
                    "selective TP staging is outside mapping owner");
      }
      for (size_t layer = 1; layer < layer_count_; ++layer) {
        const size_t n = 30 + (layer - 1) * 14;
        TORCH_CHECK(b_[n + 6].data_ptr() == b_[10].data_ptr() && b_[n + 6].numel() == b_[10].numel() &&
                    b_[n + 7].data_ptr() == b_[11].data_ptr() && b_[n + 7].numel() == b_[11].numel(),
                    "selective TP layers must share exactly one K/V Host staging pair");
      }
    }
    std::vector<at::Tensor> host_reads{b_[6], b_[7]};
    std::vector<at::Tensor> host_writes{b_[1], b_[3], b_[5], b_[10], b_[11], b_[14], b_[16], b_[18], b_[20],
                                        b_[22], b_[23], b_[24], b_[25], b_[26], b_[27], b_[28], b_[29]};
    std::vector<at::Tensor> device_reads{b_[0], b_[2], b_[4]};
    std::vector<at::Tensor> device_writes{b_[8], b_[9], b_[12], b_[13], b_[15], b_[17], b_[19], b_[21]};
    for (size_t layer = 1; layer < layer_count_; ++layer) {
      const size_t n = 30 + (layer - 1) * 14;
      host_reads.insert(host_reads.end(), {b_[n + 2], b_[n + 3]});
      host_writes.insert(host_writes.end(), {b_[n + 1], b_[n + 10], b_[n + 12]});
      if (tp_world_ == 1) host_writes.insert(host_writes.end(), {b_[n + 6], b_[n + 7]});
      device_reads.push_back(b_[n]);
      device_writes.insert(device_writes.end(), {b_[n + 4], b_[n + 5], b_[n + 8], b_[n + 9], b_[n + 11], b_[n + 13]});
    }
    if (tp_world_ > 1) {
      host_reads.push_back(tp_[6]);
      host_writes.insert(host_writes.end(), {tp_[0], tp_[1], tp_[2], tp_[3], tp_[4], tp_[7]});
    }
    selective_check_read_write_disjoint(host_reads, host_writes);
    selective_check_read_write_disjoint(device_reads, device_writes);
    c10_npu::NPUGuard guard(b_[0].device());
    TORCH_CHECK(aclrtGetCurrentContext(&context_) == ACL_SUCCESS && context_, "selective native context unavailable");
    create_acl_resources();
    try { register_owner(); } catch (...) {
      if (!destroy_acl_resources()) fatal_cleanup("selective native constructor cleanup failed");
      throw;
    }
  }
  ~SelectiveNativeGraph() {
    // A registered graph or running thread cannot be safely destroyed here.
    if (!released_ && (captured_ || thread_.joinable())) std::abort();
    if (!released_) {
      unregister_owner();
      if (!destroy_acl_resources()) fatal_cleanup("selective native destructor cleanup failed");
    }
  }
  void capture(const at::Tensor& anchor) {
    std::lock_guard<std::mutex> lock(mu_);
    TORCH_CHECK(!released_ && !closing_ && anchor.device() == b_[0].device() && !captured_,
                "invalid selective native capture");
    c10_npu::NPUGuard guard(anchor.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model = nullptr;
    TORCH_CHECK(aclmdlRICaptureGetInfo(stream, &status, &model) == ACL_SUCCESS &&
                status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && model, "selective native requires active capture");
    auto* callback_owner = new std::shared_ptr<std::atomic<bool>>(destroyed_);
    const auto callback_result = aclmdlRIDestroyRegisterCallback(model, mark_graph_destroyed, callback_owner);
    if (callback_result != ACL_SUCCESS) delete callback_owner;
    TORCH_CHECK(callback_result == ACL_SUCCESS, "selective native graph lifetime registration failed");
    captured_ = true;
    model_ri_ = model;
    capture_stream_ = stream;
    copy_capture(b_[1], b_[0], stream);
    copy_capture(b_[3], b_[2], stream);
    copy_capture(b_[5], b_[4], stream);
    for (size_t layer = 1; layer < layer_count_; ++layer) {
      const size_t n = 30 + (layer - 1) * 14;
      copy_capture(b_[n + 1], b_[n], stream);
    }
    check_acl(aclrtRecordNotify(ready_, stream), "capture ready");
    check_acl(aclrtWaitAndResetNotify(done_, stream, 0), "capture done wait");
  }
  void record_scatter(const at::Tensor& anchor) {
    std::lock_guard<std::mutex> lock(mu_);
    TORCH_CHECK(!released_ && !closing_ && captured_ && !scatter_captured_ && anchor.device() == b_[0].device(),
                "invalid selective scatter capture");
    c10_npu::NPUGuard guard(anchor.device());
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    TORCH_CHECK(stream == capture_stream_, "selective scatter marker must follow scatter on the capture stream");
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model = nullptr;
    TORCH_CHECK(aclmdlRICaptureGetInfo(stream, &status, &model) == ACL_SUCCESS &&
                status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE && model == model_ri_,
                "selective scatter marker belongs to a different graph");
    check_acl(aclrtRecordNotify(scatter_, stream), "capture scatter done");
    scatter_captured_ = true;
  }
  void start() {
    std::lock_guard<std::mutex> lock(mu_);
    TORCH_CHECK(!released_ && !closing_ && captured_ && scatter_captured_ && !thread_.joinable() && !stopping_ && !stopped_ &&
                !destroyed_->load(std::memory_order_acquire), "invalid selective native start");
    thread_ = std::thread([this] { run(); });
  }
  int64_t graph_identity() {
    std::lock_guard<std::mutex> lock(mu_);
    const auto value = reinterpret_cast<uintptr_t>(model_ri_);
    TORCH_CHECK(!released_ && !closing_ && captured_ && !destroyed_->load(std::memory_order_acquire) &&
                value <= static_cast<uintptr_t>(INT64_MAX), "selective graph identity unavailable");
    return static_cast<int64_t>(value);
  }
  void notify_peer_error() noexcept { tp_mark_error(); }
  void begin(int64_t epoch, const at::Tensor& anchor) {
    TORCH_CHECK(anchor.device() == b_[0].device(), "selective begin requires matching NPU anchor");
    c10_npu::NPUGuard guard(anchor.device());
    std::lock_guard<std::mutex> lock(mu_);
    TORCH_CHECK(!released_ && !closing_ && thread_.joinable() && !stopping_ && !inflight_ && epoch > last_epoch_ &&
                !destroyed_->load(std::memory_order_acquire),
                "selective native graph already in flight or invalid epoch");
    if (tp_world_ > 1) {
      TORCH_CHECK(tp_load(0) == 0, "selective TP group is in an error state");
    }
    {
      std::lock_guard<std::mutex> owner_lock(selective_owner_mutex());
      auto it = selective_owners().find(owner_key_);
      TORCH_CHECK(it != selective_owners().end() && !it->second.in_flight,
                  "selective group hot cache is already in use by another graph");
      it->second.in_flight = true;
    }
    launch_stream_ = c10_npu::getCurrentNPUStream().stream();
    inflight_ = true;
    group_done_ = false;
    completion_recorded_ = false;
    completion_waiting_ = false;
    last_epoch_ = epoch;
  }
  // Called on the actual replay launch stream after replay. The graph's
  // multi-stream completion semantics must be validated for the runtime.
  void record_complete(const at::Tensor& anchor) {
    TORCH_CHECK(anchor.device() == b_[0].device(), "selective completion requires matching NPU anchor");
    c10_npu::NPUGuard guard(anchor.device());
    std::lock_guard<std::mutex> lock(mu_);
    TORCH_CHECK(!released_ && !closing_ && inflight_ && !completion_recorded_, "invalid selective completion record");
    const auto stream = c10_npu::getCurrentNPUStream().stream();
    TORCH_CHECK(stream == launch_stream_, "selective completion must use the replay launch stream");
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model = nullptr;
    check_acl(aclmdlRICaptureGetInfo(stream, &status, &model), "query replay stream capture");
    TORCH_CHECK(status == ACL_MODEL_RI_CAPTURE_STATUS_NONE, "graph completion must be recorded outside capture");
    check_acl(aclrtRecordNotify(graph_done_, stream), "record graph completion");
    completion_recorded_ = true;
  }
  void retire(int64_t epoch) {
    c10_npu::NPUGuard guard(b_[0].device());
    {
      std::lock_guard<std::mutex> lock(mu_);
      TORCH_CHECK(!released_ && !closing_ && inflight_ && epoch == last_epoch_ && completion_recorded_ && !completion_waiting_,
                  "selective native graph lacks completion record");
      check_acl(aclrtWaitAndResetNotify(graph_done_, monitor_, 0), "wait graph completion");
      completion_waiting_ = true;
    }
    // A timeout is not proof of retirement: keep the handle and buffers live.
    const auto completion_result = aclrtSynchronizeStreamWithTimeout(monitor_, 15000);
    if (completion_result != ACL_SUCCESS) tp_mark_error();
    check_acl(completion_result, "synchronize graph completion");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (group_done_) break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        tp_mark_error();
        TORCH_CHECK(false, "selective scatter completion timed out");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    try {
      if (tp_world_ > 1) {
        tp_store(SELECTIVE_TP_ACK_BASE + tp_world_ + tp_rank_, epoch);
        if (tp_rank_ == 0) {
          for (int64_t rank = 0; rank < tp_world_; ++rank) tp_wait(SELECTIVE_TP_ACK_BASE + tp_world_ + rank, epoch);
          const auto max_rows = b_[22].numel() / width_;
          auto identity = b_[25].view({max_rows, 9});
          auto controls = b_[29].view({max_rows, 2});
          for (const auto& ticket : pending_tickets_)
            selective_group_lru_finish_cpu(identity[ticket.first], controls[ticket.first], ticket.second, true);
          pending_tickets_.clear();
          tp_store(SELECTIVE_TP_COMMIT_EPOCH, epoch);
        }
        // No rank may report a successful replay until the owner has committed
        // all rows.  A pre-commit peer error makes the whole group fail closed.
        tp_wait(SELECTIVE_TP_COMMIT_EPOCH, epoch);
        TORCH_CHECK(tp_load(0) == 0, "selective TP peer failed before commit");
      }
      std::lock_guard<std::mutex> lock(mu_);
      std::lock_guard<std::mutex> owner_lock(selective_owner_mutex());
      auto it = selective_owners().find(owner_key_);
      TORCH_CHECK(it != selective_owners().end() && it->second.in_flight,
                  "selective group ownership lost during replay");
      it->second.in_flight = false;
      inflight_ = false;
    } catch (...) { tp_mark_error(); throw; }
  }
  void stop() {
    c10_npu::NPUGuard guard(b_[0].device());
    {
      std::lock_guard<std::mutex> lock(mu_);
      TORCH_CHECK(!released_ && !closing_ && !inflight_ && !stopping_, "cannot stop selective native graph in this state");
      if (!thread_.joinable()) return;
      if (tp_world_ > 1) {
        TORCH_CHECK(tp_load(0) == 0, "cannot stop a selective TP group in an error state");
        const auto plan = tp_load(1), stage = tp_load(3);
        TORCH_CHECK(tp_load(SELECTIVE_TP_COMMIT_EPOCH) == plan,
                    "cannot stop selective TP group before commit");
        for (int64_t rank = 0; rank < tp_world_; ++rank)
          TORCH_CHECK(tp_load(SELECTIVE_TP_ACK_BASE + rank) == stage &&
                      tp_load(SELECTIVE_TP_ACK_BASE + tp_world_ + rank) == plan,
                      "cannot stop selective TP group while another rank is in flight");
      }
      stopping_ = true;
    }
    if (aclrtRecordNotify(ready_, control_) != ACL_SUCCESS ||
        aclrtSynchronizeStreamWithTimeout(control_, 3000) != ACL_SUCCESS)
      fatal_cleanup("selective native worker wake failed");
    try { thread_.join(); } catch (...) { fatal_cleanup("selective native worker join failed"); }
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = false;
    stopped_ = true;
  }
  void release() {
    c10_npu::NPUGuard guard(b_[0].device());
    std::lock_guard<std::mutex> lock(mu_);
    TORCH_CHECK(!released_ && !stopping_ && !thread_.joinable() && !inflight_ &&
                (!captured_ || destroyed_->load(std::memory_order_acquire)),
                "cannot release selective native resources before graph destruction and worker stop");
    std::lock_guard<std::mutex> owner_lock(selective_owner_mutex());
    auto& owners = selective_owners();
    auto it = owners.find(owner_key_);
    TORCH_CHECK(owner_registered_ && it != owners.end() && !it->second.in_flight && it->second.references > 0,
                "cannot release selective native resources while a group graph is in flight");
    closing_ = true;
    TORCH_CHECK(destroy_acl_resources(), "selective native resource destruction failed");
    selective_owner_spans().erase(this);
    if (--it->second.references == 0) owners.erase(it);
    owner_registered_ = false;
    released_ = true;
  }
 private:
  [[noreturn]] static void fatal_cleanup(const char* message) noexcept {
    std::fprintf(stderr, "%s; terminating rank to quarantine native resources\n", message);
    std::abort();
  }
  static std::pair<uintptr_t, size_t> tensor_span(const at::Tensor& tensor) {
    const auto size = selective_tensor_bytes(tensor);
    const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(size && base <= UINTPTR_MAX - size, "invalid selective owner tensor span");
    return {base, size};
  }
  void register_owner() {
    SelectiveNativeOwner candidate;
    candidate.device = b_[0].device().index();
    candidate.tp_rank = tp_rank_;
    candidate.tp_world = tp_world_;
    for (int i = 22; i <= 29; ++i) candidate.state.push_back(tensor_span(b_[i]));
    if (tp_world_ > 1) for (size_t i = 0; i < 5; ++i) candidate.state.push_back(tensor_span(tp_[i]));
    if (tp_world_ > 1) { candidate.state.push_back(tensor_span(tp_[6])); candidate.state.push_back(tensor_span(tp_[7])); }
    candidate.hot.push_back(tensor_span(b_[8])); candidate.hot.push_back(tensor_span(b_[9]));
    for (size_t layer = 1; layer < layer_count_; ++layer) {
      const size_t n = 30 + (layer - 1) * 14;
      candidate.hot.push_back(tensor_span(b_[n + 4])); candidate.hot.push_back(tensor_span(b_[n + 5]));
    }
    owner_key_ = reinterpret_cast<uintptr_t>(b_[29].data_ptr());
    SelectiveNativeSpans spans;
    spans.owner = owner_key_;
    spans.device = candidate.device;
    for (int i : {6, 7}) if (b_[i].numel()) spans.host_reads.push_back(tensor_span(b_[i]));
    for (int i : {1, 3, 5, 10, 11, 14, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29})
      spans.host_writes.push_back(tensor_span(b_[i]));
    for (int i : {0, 2, 4}) spans.device_reads.push_back(tensor_span(b_[i]));
    for (int i : {8, 9, 12, 13, 15, 17, 19, 21}) spans.device_writes.push_back(tensor_span(b_[i]));
    for (size_t layer = 1; layer < layer_count_; ++layer) {
      const size_t n = 30 + (layer - 1) * 14;
      for (size_t i : {size_t{2}, size_t{3}}) if (b_[n + i].numel()) spans.host_reads.push_back(tensor_span(b_[n + i]));
      for (size_t i : {size_t{1}, size_t{6}, size_t{7}, size_t{10}, size_t{12}})
        spans.host_writes.push_back(tensor_span(b_[n + i]));
      spans.device_reads.push_back(tensor_span(b_[n]));
      for (size_t i : {size_t{4}, size_t{5}, size_t{8}, size_t{9}, size_t{11}, size_t{13}})
        spans.device_writes.push_back(tensor_span(b_[n + i]));
    }
    if (tp_world_ > 1) {
      for (size_t i = 0; i < 5; ++i) spans.host_writes.push_back(tensor_span(tp_[i]));
      spans.host_reads.push_back(tensor_span(tp_[6]));
      spans.host_writes.push_back(tensor_span(tp_[7]));
    }
    std::lock_guard<std::mutex> lock(selective_owner_mutex());
    auto& owners = selective_owners();
    auto& registered_spans = selective_owner_spans();
    const auto disjoint = [](const auto& a, const auto& b) {
      return a.first + a.second <= b.first || b.first + b.second <= a.first;
    };
    for (const auto& registered : registered_spans) {
      const auto& old = registered.second;
      if (old.owner == spans.owner) continue;  // Same owner uses one graph lease.
      for (const auto& write : spans.host_writes) {
        for (const auto& previous : old.host_writes) TORCH_CHECK(disjoint(write, previous), "overlapping selective Host writes");
        for (const auto& previous : old.host_reads) TORCH_CHECK(disjoint(write, previous), "selective Host write overlaps another group's input");
      }
      for (const auto& read : spans.host_reads) for (const auto& previous : old.host_writes)
        TORCH_CHECK(disjoint(read, previous), "selective Host input overlaps another group's write");
      if (old.device == spans.device) {
        for (const auto& write : spans.device_writes) {
          for (const auto& previous : old.device_writes) TORCH_CHECK(disjoint(write, previous), "overlapping selective device writes");
          for (const auto& previous : old.device_reads) TORCH_CHECK(disjoint(write, previous), "selective device write overlaps another group's input");
        }
        for (const auto& read : spans.device_reads) for (const auto& previous : old.device_writes)
          TORCH_CHECK(disjoint(read, previous), "selective device input overlaps another group's write");
      }
    }
    auto existing = owners.find(owner_key_);
    if (existing != owners.end()) {
      TORCH_CHECK(existing->second.device == candidate.device && existing->second.tp_rank == candidate.tp_rank &&
                  existing->second.tp_world == candidate.tp_world && existing->second.state == candidate.state &&
                  existing->second.hot == candidate.hot,
                  "selective graph handles sharing a group must share identical state and hot tensors");
      TORCH_CHECK(!existing->second.in_flight, "cannot register a graph while its selective group is in flight");
    } else {
      for (const auto& owner : owners) {
        for (const auto& old_state : owner.second.state) for (const auto& new_state : candidate.state)
          TORCH_CHECK(old_state.first + old_state.second <= new_state.first ||
                      new_state.first + new_state.second <= old_state.first,
                      "overlapping selective CPU state uses different group ownership");
        if (owner.second.device == candidate.device) for (const auto& old_hot : owner.second.hot)
          for (const auto& new_hot : candidate.hot)
            TORCH_CHECK(old_hot.first + old_hot.second <= new_hot.first ||
                        new_hot.first + new_hot.second <= old_hot.first,
                        "overlapping selective hot buffers use different group state");
      }
    }
    registered_spans.emplace(this, std::move(spans));
    try {
      if (existing != owners.end()) ++existing->second.references;
      else { candidate.references = 1; owners.emplace(owner_key_, std::move(candidate)); }
    } catch (...) { registered_spans.erase(this); throw; }
    owner_registered_ = true;
  }
  void unregister_owner() noexcept {
    if (!owner_registered_) return;
    std::lock_guard<std::mutex> lock(selective_owner_mutex());
    auto& owners = selective_owners();
    auto it = owners.find(owner_key_);
    if (it == owners.end() || it->second.in_flight || it->second.references == 0) std::abort();
    selective_owner_spans().erase(this);
    if (--it->second.references == 0) owners.erase(it);
    owner_registered_ = false;
  }
  static void mark_graph_destroyed(void* ptr) noexcept {
    std::unique_ptr<std::shared_ptr<std::atomic<bool>>> flag(static_cast<std::shared_ptr<std::atomic<bool>>*>(ptr));
    (*flag)->store(true, std::memory_order_release);
  }
  static void check_acl(aclError result, const char* op) { TORCH_CHECK(result == ACL_SUCCESS, op, ": ", result); }
  void create_acl_resources() {
    try {
      check_acl(aclrtCreateStream(&worker_), "selective worker stream");
      check_acl(aclrtCreateStream(&control_), "selective control stream");
      check_acl(aclrtCreateStream(&monitor_), "selective monitor stream");
      check_acl(aclrtCreateNotify(&ready_, ACL_NOTIFY_DEFAULT), "selective ready");
      check_acl(aclrtCreateNotify(&done_, ACL_NOTIFY_DEFAULT), "selective done");
      check_acl(aclrtCreateNotify(&scatter_, ACL_NOTIFY_DEFAULT), "selective scatter");
      check_acl(aclrtCreateNotify(&graph_done_, ACL_NOTIFY_DEFAULT), "selective graph completion");
    } catch (...) {
      if (!destroy_acl_resources()) fatal_cleanup("selective native ACL creation cleanup failed");
      throw;
    }
  }
  bool destroy_acl_resources() noexcept {
    bool ok = true;
    if (ready_) { if (aclrtDestroyNotify(ready_) == ACL_SUCCESS) ready_ = nullptr; else ok = false; }
    if (done_) { if (aclrtDestroyNotify(done_) == ACL_SUCCESS) done_ = nullptr; else ok = false; }
    if (scatter_) { if (aclrtDestroyNotify(scatter_) == ACL_SUCCESS) scatter_ = nullptr; else ok = false; }
    if (graph_done_) { if (aclrtDestroyNotify(graph_done_) == ACL_SUCCESS) graph_done_ = nullptr; else ok = false; }
    if (worker_) { if (aclrtDestroyStream(worker_) == ACL_SUCCESS) worker_ = nullptr; else ok = false; }
    if (control_) { if (aclrtDestroyStream(control_) == ACL_SUCCESS) control_ = nullptr; else ok = false; }
    if (monitor_) { if (aclrtDestroyStream(monitor_) == ACL_SUCCESS) monitor_ = nullptr; else ok = false; }
    return ok;
  }
  static void copy_capture(at::Tensor& host, at::Tensor& device, aclrtStream stream) {
    const size_t bytes = selective_tensor_bytes(device);
    TORCH_CHECK(selective_tensor_bytes(host) >= bytes, "selective native D2H buffer too small");
    check_acl(aclrtMemcpyAsync(host.data_ptr(), selective_tensor_bytes(host), device.data_ptr(), bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream), "capture metadata D2H");
  }
  void wait_notify(aclrtNotify notify) {
    check_acl(aclrtWaitAndResetNotify(notify, worker_, 0), "selective native notify wait");
    // This worker is woken explicitly during idle teardown. An error or a
    // failed graph is quarantined rather than inferred from timeout codes.
    check_acl(aclrtSynchronizeStream(worker_), "selective native stream wait");
  }
  int64_t tp_load(int64_t index) const {
    return __atomic_load_n(tp_[0].data_ptr<int64_t>() + index, __ATOMIC_ACQUIRE);
  }
  void tp_store(int64_t index, int64_t value) {
    __atomic_store_n(tp_[0].data_ptr<int64_t>() + index, value, __ATOMIC_RELEASE);
  }
  void tp_mark_error() noexcept {
    if (tp_world_ > 1) __atomic_store_n(tp_[0].data_ptr<int64_t>(), int64_t{1}, __ATOMIC_RELEASE);
  }
  void tp_wait(int64_t index, int64_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
      TORCH_CHECK(tp_load(0) == 0, "selective TP peer failed");
      const auto value = tp_load(index);
      TORCH_CHECK(value <= expected, "selective TP observed a later epoch");
      if (value == expected) return;
      TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "selective TP synchronization timed out");
      std::this_thread::yield();
    }
  }
  int32_t tp_selected_page(const int32_t* pages, int64_t row, int64_t position) {
    const auto* meta = b_[5].data_ptr<int64_t>() + row * 12;
    const int32_t token = b_[1].data_ptr<int32_t>()[row * width_ + position];
    if (meta[0] == 0 || token < 0 || token >= meta[4] || meta[7] <= 0) return -1;
    const int64_t block = token / meta[7];
    if (block >= page_columns_) return -1;
    return pages[row * page_columns_ + block];
  }
  void tp_check_or_publish_inputs(int64_t epoch) {
    const auto* meta = b_[5].data_ptr<int64_t>();
    const auto* topk = b_[1].data_ptr<int32_t>();
    auto* shared_meta = tp_[2].data_ptr<int64_t>();
    auto* shared_topk = tp_[3].data_ptr<int32_t>();
    auto* shared_pages = tp_[4].data_ptr<int32_t>();
    const auto* local_ids = tp_[6].data_ptr<int64_t>();
    auto* shared_ids = tp_[7].data_ptr<int64_t>();
    std::vector<const int32_t*> layer_pages;
    layer_pages.reserve(layer_count_);
    for (size_t layer = 0; layer < layer_count_; ++layer)
      layer_pages.push_back(layer_binding(layer).pages.data_ptr<int32_t>());
    const auto owner_base = reinterpret_cast<uintptr_t>(tp_[5].data_ptr());
    const auto owner_bytes = selective_tensor_bytes(tp_[5]);
    const at::Tensor views[] = {tp_[0], tp_[1], tp_[2], tp_[3], tp_[4], tp_[7], b_[10], b_[11]};
    std::array<int64_t, 16> view_layout{};
    for (size_t i = 0; i < 8; ++i) {
      const auto base = reinterpret_cast<uintptr_t>(views[i].data_ptr());
      view_layout[i * 2] = static_cast<int64_t>(base - owner_base);
      view_layout[i * 2 + 1] = static_cast<int64_t>(selective_tensor_bytes(views[i]));
    }
    std::unordered_map<int64_t, bool> layer_ids;
    TORCH_CHECK(local_ids[0] >= 0, "invalid selective TP group identity");
    for (size_t layer = 0; layer < layer_count_; ++layer)
      TORCH_CHECK(local_ids[layer + 1] >= 0 && layer_ids.emplace(local_ids[layer + 1], true).second,
                  "invalid or duplicate selective TP layer identity");
    const auto stage_rows = static_cast<size_t>(rows_) * static_cast<size_t>(width_);
    TORCH_CHECK(stage_rows && selective_tensor_bytes(b_[10]) % stage_rows == 0 &&
                selective_tensor_bytes(b_[11]) % stage_rows == 0, "invalid selective TP payload geometry");
    const auto k_bytes = selective_tensor_bytes(b_[10]) / stage_rows;
    const auto v_bytes = selective_tensor_bytes(b_[11]) / stage_rows;
    TORCH_CHECK(k_bytes > 0 && k_bytes <= INT32_MAX && v_bytes > 0 && v_bytes <= UINT32_MAX,
                "invalid selective TP row bytes");
    const auto layout = static_cast<int64_t>((static_cast<uint64_t>(k_bytes) << 32) | v_bytes);
    if (tp_rank_ == 0) {
      const auto previous = tp_load(1);
      TORCH_CHECK(previous >= 0 && epoch > previous, "invalid selective TP plan epoch");
      for (int64_t rank = 0; rank < tp_world_; ++rank) tp_wait(SELECTIVE_TP_ACK_BASE + tp_world_ + rank, previous);
      tp_wait(SELECTIVE_TP_COMMIT_EPOCH, previous);
      std::copy_n(meta, rows_ * 12, shared_meta);
      std::copy_n(topk, rows_ * width_, shared_topk);
      std::copy_n(local_ids, layer_count_ + 1, shared_ids);
      tp_store(2, layout);
      tp_store(5, rows_);
      tp_store(6, static_cast<int64_t>(layer_count_));
      tp_store(7, width_);
      tp_store(8, b_[20].numel() / rows_);
      tp_store(9, tp_capacity_rows_);
      tp_store(10, static_cast<int64_t>(owner_bytes));
      for (size_t i = 0; i < view_layout.size(); ++i) tp_store(11 + i, view_layout[i]);
      for (size_t layer = 0; layer < layer_count_; ++layer)
        for (int64_t row = 0; row < rows_; ++row) for (int64_t i = 0; i < width_; ++i)
          shared_pages[(layer * tp_capacity_rows_ + row) * width_ + i] = tp_selected_page(layer_pages[layer], row, i);
    } else {
      tp_wait(1, epoch);
      TORCH_CHECK(tp_load(2) == layout, "selective TP row layout differs");
      TORCH_CHECK(tp_load(5) == rows_ && tp_load(6) == static_cast<int64_t>(layer_count_) &&
                  tp_load(7) == width_ && tp_load(8) == b_[20].numel() / rows_ &&
                  tp_load(9) == tp_capacity_rows_ && tp_load(10) == static_cast<int64_t>(owner_bytes) &&
                  std::equal(local_ids, local_ids + layer_count_ + 1, shared_ids),
                  "selective TP group identity or batch geometry differs");
      for (size_t i = 0; i < view_layout.size(); ++i)
        TORCH_CHECK(tp_load(11 + i) == view_layout[i], "selective TP mapping view layout differs");
      TORCH_CHECK(std::equal(meta, meta + rows_ * 12, shared_meta) &&
                  std::equal(topk, topk + rows_ * width_, shared_topk), "selective TP request metadata differs");
      for (size_t layer = 0; layer < layer_count_; ++layer)
        for (int64_t row = 0; row < rows_; ++row) for (int64_t i = 0; i < width_; ++i)
          TORCH_CHECK(shared_pages[(layer * tp_capacity_rows_ + row) * width_ + i] == tp_selected_page(layer_pages[layer], row, i),
                      "selective TP selected page differs");
    }
  }
  void tp_publish_plans(const std::vector<std::vector<at::Tensor>>& plans, int64_t epoch) {
    const int64_t q = b_[20].numel() / rows_, stride = 3 * width_ + q + 1;
    auto* shared = tp_[1].data_ptr<int32_t>();
    const auto* physical = b_[18].data_ptr<int32_t>();
    const auto* blocks = b_[20].data_ptr<int32_t>();
    for (int64_t row = 0; row < rows_; ++row) {
      auto* out = shared + row * stride;
      std::fill_n(out, stride, -1);
      std::copy_n(physical + row * width_, width_, out);
      std::copy_n(blocks + row * q, q, out + width_);
      const auto& plan = plans[static_cast<size_t>(row)];
      const auto count = plan.empty() ? 0 : plan[1].numel();
      TORCH_CHECK(count >= 0 && count <= width_, "invalid selective TP miss count");
      out[width_ + q] = static_cast<int32_t>(count);
      if (count) {
        std::copy_n(plan[1].data_ptr<int32_t>(), count, out + width_ + q + 1);
        std::copy_n(plan[2].data_ptr<int32_t>(), count, out + 2 * width_ + q + 1);
      }
    }
    tp_store(1, epoch);  // Publishes exact input snapshot and plan together.
  }
  std::vector<at::Tensor> tp_read_plan(int64_t row, bool padding) {
    const int64_t q = b_[20].numel() / rows_, stride = 3 * width_ + q + 1;
    auto shared = tp_[1].view({tp_capacity_rows_, stride})[row];
    const auto* values = shared.data_ptr<int32_t>();
    const int32_t count = values[width_ + q];
    TORCH_CHECK(count >= 0 && count <= width_ && (!padding || count == 0), "invalid shared selective plan");
    if (padding) {
      for (int64_t i = 0; i < width_; ++i) TORCH_CHECK(values[i] == -1, "invalid shared padding row");
      return {};
    }
    const auto* meta = b_[5].data_ptr<int64_t>() + row * 12;
    TORCH_CHECK(meta[7] > 0 && meta[7] <= INT32_MAX - width_, "invalid selective TP block size");
    const auto* topk = b_[1].data_ptr<int32_t>() + row * width_;
    std::unordered_map<int32_t, int32_t> selected, slot_owners;
    selected.reserve(static_cast<size_t>(width_));
    slot_owners.reserve(static_cast<size_t>(width_));
    for (int64_t i = 0; i < width_; ++i) {
      TORCH_CHECK(values[i] >= -1 && values[i] < width_ + meta[7],
                  "invalid shared selective physical slot");
      if (topk[i] < 0 || topk[i] >= meta[4]) {
        TORCH_CHECK(values[i] == -1, "invalid shared selective padding slot");
      } else {
        TORCH_CHECK(values[i] >= 0, "valid selective token is missing a slot");
        const auto inserted = selected.emplace(topk[i], values[i]);
        TORCH_CHECK(inserted.second || inserted.first->second == values[i], "duplicate token maps to different slots");
        const auto owner = slot_owners.emplace(values[i], topk[i]);
        TORCH_CHECK(owner.second || owner.first->second == topk[i], "different tokens share a selective slot");
      }
    }
    for (int32_t i = 0; i < count; ++i) {
      const auto token = values[width_ + q + 1 + i], slot = values[2 * width_ + q + 1 + i];
      TORCH_CHECK(token >= 0 && token < meta[4] && slot >= 0 && slot < width_, "invalid shared selective miss");
      const auto found = selected.find(token);
      TORCH_CHECK(found != selected.end() && found->second == slot, "shared selective miss is absent from physical plan");
    }
    for (size_t layer = 0; layer < layer_count_; ++layer) {
      const auto entry = layer_binding(layer);
      TORCH_CHECK(selective_tensor_bytes(entry.stage_host_k) % (rows_ * width_) == 0 &&
                  selective_tensor_bytes(entry.stage_host_v) % (rows_ * width_) == 0,
                  "invalid selective TP stage row size");
      const auto kb = selective_tensor_bytes(entry.stage_host_k) / (rows_ * width_);
      const auto vb = selective_tensor_bytes(entry.stage_host_v) / (rows_ * width_);
      TORCH_CHECK(kb && vb && kb <= SIZE_MAX / meta[7] && vb <= SIZE_MAX / meta[7],
                  "invalid selective TP hot geometry");
      const auto kblocks = selective_tensor_bytes(entry.hot_k) / (kb * meta[7]);
      const auto vblocks = selective_tensor_bytes(entry.hot_v) / (vb * meta[7]);
      for (int64_t i = 0; i < q; ++i)
        TORCH_CHECK(values[width_ + i] >= 0 && static_cast<size_t>(values[width_ + i]) < kblocks &&
                    static_cast<size_t>(values[width_ + i]) < vblocks, "shared selective block table out of range");
    }
    return {shared.narrow(0, 0, width_), shared.narrow(0, width_ + q + 1, count),
            shared.narrow(0, 2 * width_ + q + 1, count), shared.narrow(0, width_, q)};
  }
  int64_t tp_begin_stage(size_t layer, int64_t epoch) {
    TORCH_CHECK(tp_load(1) == epoch, "selective TP stage has wrong plan epoch");
    if (tp_rank_ == 0) {
      const auto previous = tp_load(3);
      TORCH_CHECK(previous >= 0 && previous < INT64_MAX, "selective TP stage epoch overflow");
      for (int64_t rank = 0; rank < tp_world_; ++rank) tp_wait(SELECTIVE_TP_ACK_BASE + rank, previous);
      return previous + 1;
    }
    const auto previous = tp_load(SELECTIVE_TP_ACK_BASE + tp_rank_);
    TORCH_CHECK(previous >= 0 && previous < INT64_MAX, "selective TP acknowledgement overflow");
    const auto next = previous + 1;
    tp_wait(3, next);
    TORCH_CHECK(tp_load(1) == epoch && tp_load(4) == static_cast<int64_t>(layer),
                "selective TP layer or epoch mismatch");
    return next;
  }
  void tp_publish_stage(int64_t stage, size_t layer) {
    if (tp_rank_ != 0) return;
    tp_store(4, static_cast<int64_t>(layer));
    tp_store(3, stage);
  }
  SelectiveGroupLayerTransfer layer_binding(size_t layer) {
    if (layer == 0) return {b_[6], b_[7], b_[8], b_[9], b_[10], b_[11], b_[12], b_[13],
                            b_[14], b_[15], b_[16], b_[17], b_[3]};
    const size_t n = 30 + (layer - 1) * 14;
    return {b_[n + 2], b_[n + 3], b_[n + 4], b_[n + 5], b_[n + 6], b_[n + 7], b_[n + 8], b_[n + 9],
            b_[n + 10], b_[n + 11], b_[n + 12], b_[n + 13], b_[n + 1]};
  }
  void run() noexcept {
    try {
      // Inference mode is thread-local and is not inherited from the caller.
      c10::InferenceMode inference_guard(true);
      check_acl(aclrtSetCurrentContext(context_), "selective native worker context");
      for (;;) {
        wait_notify(ready_);
        int64_t replay_epoch;
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (stopping_) return;
          TORCH_CHECK(inflight_, "selective native ready without replay");
          replay_epoch = last_epoch_;
        }
        if (tp_world_ > 1) tp_check_or_publish_inputs(replay_epoch);
        const auto* metadata = b_[5].data_ptr<int64_t>();
        const auto max_rows = b_[22].numel() / width_;
        const auto max_seq = b_[26].numel() / max_rows;
        const auto table_width = b_[20].numel() / rows_;
        auto topk = b_[1].view({rows_, width_});
        auto physical = b_[18].view({rows_, width_});
        auto block_table = b_[20].view({rows_, table_width});
        auto keys = b_[22].view({max_rows, width_});
        auto lru = b_[23].view({max_rows, width_});
        auto resident = b_[24].view({max_rows, static_cast<int64_t>(layer_count_), width_});
        auto identity = b_[25].view({max_rows, 9});
        auto marks = b_[26].view({max_rows, max_seq});
        auto positions = b_[27].view({max_rows, max_seq});
        auto controls = b_[29].view({max_rows, 2});
        std::vector<std::vector<at::Tensor>> plans(static_cast<size_t>(rows_));
        std::vector<int64_t> physical_rows(static_cast<size_t>(rows_), -1);
        std::vector<std::vector<int64_t>> metas(static_cast<size_t>(rows_));
        std::unordered_map<int64_t, bool> used_rows;
        for (int64_t row = 0; row < rows_; ++row) {
          const auto* values = metadata + 12 * row;
          auto& meta = metas[static_cast<size_t>(row)];
          meta.assign(values, values + 12);
          if (meta[0] == 0) { physical[row].fill_(-1); block_table[row].zero_(); continue; }
          const int64_t slot_row = meta[10];
          TORCH_CHECK(slot_row >= 0 && slot_row < max_rows && used_rows.emplace(slot_row, true).second,
                      "invalid or duplicate selective physical row");
          physical_rows[static_cast<size_t>(row)] = slot_row;
          if (tp_world_ > 1 && tp_rank_ != 0) {
            auto result = tp_read_plan(row, false);
            physical[row].copy_(result[0]);
            block_table[row].copy_(result[3]);
            plans[static_cast<size_t>(row)] = std::move(result);
            continue;
          }
          auto pages = at::empty({static_cast<int64_t>(layer_count_), page_columns_}, b_[3].options());
          for (size_t layer = 0; layer < layer_count_; ++layer) {
            auto entry = layer_binding(layer);
            pages[static_cast<int64_t>(layer)].copy_(entry.pages.view({rows_, page_columns_})[row]);
          }
          auto epoch = b_[28].view({max_rows}).narrow(0, slot_row, 1);
          auto result = selective_group_lru_plan_cpu(topk[row], pages, keys[slot_row], lru[slot_row],
              resident[slot_row], identity[slot_row], marks[slot_row], positions[slot_row], epoch,
              controls[slot_row], meta);
          physical[row].copy_(result[0]);
          block_table[row].copy_(result[3]);
          plans[static_cast<size_t>(row)] = std::move(result);
        }
        if (tp_world_ > 1 && tp_rank_ == 0) tp_publish_plans(plans, replay_epoch);
        for (size_t layer_index = 0; layer_index < layer_count_; ++layer_index) {
          auto layer = layer_binding(layer_index);
          const int64_t stage = tp_world_ > 1 ? tp_begin_stage(layer_index, replay_epoch) : 0;
          layer.destinations_host.zero_();
          layer.lengths_host.zero_();
          for (int64_t row = 0; row < rows_; ++row) {
            const auto& result = plans[static_cast<size_t>(row)];
            if (result.empty()) continue;
            const auto& meta = metas[static_cast<size_t>(row)];
            const auto host_rows = static_cast<uint64_t>(meta[8]) * static_cast<uint64_t>(meta[7]);
            const auto stage_rows = static_cast<uint64_t>(rows_) * static_cast<uint64_t>(width_);
            TORCH_CHECK(host_rows > 0 && selective_tensor_bytes(layer.stage_host_k) % stage_rows == 0 &&
                        selective_tensor_bytes(layer.stage_host_v) % stage_rows == 0, "invalid selective Host row sizes");
            const auto k_bytes = selective_tensor_bytes(layer.stage_host_k) / stage_rows;
            const auto v_bytes = selective_tensor_bytes(layer.stage_host_v) / stage_rows;
            if (tp_rank_ == 0)
              TORCH_CHECK(selective_tensor_bytes(layer.host_k) / host_rows == k_bytes &&
                          selective_tensor_bytes(layer.host_v) / host_rows == v_bytes &&
                          selective_tensor_bytes(layer.host_k) % host_rows == 0 &&
                          selective_tensor_bytes(layer.host_v) % host_rows == 0, "invalid selective Host KV geometry");
            TORCH_CHECK(k_bytes > 0 && v_bytes > 0 &&
                        k_bytes <= static_cast<uint64_t>(INT64_MAX) / (rows_ * width_) &&
                        v_bytes <= static_cast<uint64_t>(INT64_MAX) / (rows_ * width_),
                        "selective staging row offsets overflow");
            TORCH_CHECK(layer.stage_host_k.scalar_type() == at::kByte && layer.stage_host_v.scalar_type() == at::kByte &&
                        layer.stage_device_k.scalar_type() == at::kByte && layer.stage_device_v.scalar_type() == at::kByte,
                        "selective native staging must be bytes");
            auto one = layer;
            one.pages = layer.pages.view({rows_, page_columns_})[row];
            one.stage_host_k = layer.stage_host_k.view({-1}).narrow(0, row * width_ * k_bytes, width_ * k_bytes);
            one.stage_host_v = layer.stage_host_v.view({-1}).narrow(0, row * width_ * v_bytes, width_ * v_bytes);
            std::vector<SelectiveGroupLayerTransfer> one_layer{one};
            selective_gather_group_rows(one_layer, result[1], result[2], meta[10], meta[11], meta[7],
                                        width_, k_bytes, v_bytes, tp_rank_ == 0, row, rows_);
            const auto count = static_cast<size_t>(result[1].numel());
            if (count) {
              auto dk = layer.stage_device_k.view({-1}).narrow(0, row * width_ * k_bytes, width_ * k_bytes);
              auto dv = layer.stage_device_v.view({-1}).narrow(0, row * width_ * v_bytes, width_ * v_bytes);
              check_acl(aclrtMemcpyAsync(dk.data_ptr(), selective_tensor_bytes(dk), one.stage_host_k.data_ptr(),
                                         count * k_bytes, ACL_MEMCPY_HOST_TO_DEVICE, worker_), "group K H2D");
              check_acl(aclrtMemcpyAsync(dv.data_ptr(), selective_tensor_bytes(dv), one.stage_host_v.data_ptr(),
                                         count * v_bytes, ACL_MEMCPY_HOST_TO_DEVICE, worker_), "group V H2D");
            }
          }
          if (tp_world_ > 1) tp_publish_stage(stage, layer_index);
          copy_h2d(layer.destinations_device, layer.destinations_host);
          copy_h2d(layer.lengths_device, layer.lengths_host);
          if (tp_world_ > 1) {
            check_acl(aclrtSynchronizeStreamWithTimeout(worker_, 3000), "selective TP H2D completion");
            tp_store(SELECTIVE_TP_ACK_BASE + tp_rank_, stage);
          }
        }
        copy_h2d(b_[19], b_[18]);
        copy_h2d(b_[21], b_[20]);
        check_acl(aclrtSynchronizeStreamWithTimeout(worker_, 3000), "selective native H2D completion");
        check_acl(aclrtRecordNotify(done_, worker_), "selective native done");
        wait_notify(scatter_);
        std::vector<std::pair<int64_t, int64_t>> tickets;
        if (tp_rank_ == 0) for (int64_t row = 0; row < rows_; ++row) {
          const auto& result = plans[static_cast<size_t>(row)];
          if (result.empty()) continue;
          const auto slot_row = physical_rows[static_cast<size_t>(row)];
          if (tp_world_ == 1) selective_group_lru_finish_cpu(identity[slot_row], controls[slot_row],
                                                             result[4].item<int64_t>(), true);
          else tickets.emplace_back(slot_row, result[4].item<int64_t>());
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          pending_tickets_ = std::move(tickets);
          group_done_ = true;
        }
      }
    } catch (const std::exception& error) {
      tp_mark_error();
      std::fprintf(stderr, "selective native worker fatal: %s\n", error.what());
      std::abort();
    } catch (...) { tp_mark_error(); std::abort(); }
  }
  void copy_h2d(at::Tensor& device, at::Tensor& host) {
    const auto bytes = selective_tensor_bytes(host);
    TORCH_CHECK(selective_tensor_bytes(device) >= bytes, "selective native output too small");
    check_acl(aclrtMemcpyAsync(device.data_ptr(), selective_tensor_bytes(device), host.data_ptr(), bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE, worker_), "selective native output H2D");
  }
  std::vector<at::Tensor> b_;
  std::vector<at::Tensor> tp_;
  int64_t tp_rank_ = 0, tp_world_ = 1, tp_capacity_rows_ = 0;
  std::vector<std::pair<int64_t, int64_t>> pending_tickets_;
  size_t layer_count_ = 0;
  int64_t rows_ = 0, width_ = 0, page_columns_ = 0;
  aclrtContext context_ = nullptr;
  aclrtStream worker_ = nullptr, control_ = nullptr, monitor_ = nullptr, launch_stream_ = nullptr, capture_stream_ = nullptr;
  aclrtNotify ready_ = nullptr, done_ = nullptr, scatter_ = nullptr, graph_done_ = nullptr;
  aclmdlRI model_ri_ = nullptr;
  std::shared_ptr<std::atomic<bool>> destroyed_ = std::make_shared<std::atomic<bool>>(false);
  std::thread thread_;
  std::mutex mu_;
  int64_t last_epoch_ = 0;
  bool captured_ = false, scatter_captured_ = false, stopping_ = false, inflight_ = false;
  bool group_done_ = false, released_ = false, closing_ = false, completion_recorded_ = false;
  bool completion_waiting_ = false, stopped_ = false;
  uintptr_t owner_key_ = 0;
  bool owner_registered_ = false;
};

namespace {
std::mutex& selective_native_registry_mutex() { static auto* mutex = new std::mutex(); return *mutex; }
std::unordered_map<int64_t, std::shared_ptr<SelectiveNativeGraph>>& selective_native_registry() {
  static auto* registry = new std::unordered_map<int64_t, std::shared_ptr<SelectiveNativeGraph>>();
  return *registry;
}
std::shared_ptr<SelectiveNativeGraph> selective_native_get(int64_t handle) {
  std::lock_guard<std::mutex> lock(selective_native_registry_mutex());
  const auto it = selective_native_registry().find(handle);
  TORCH_CHECK(it != selective_native_registry().end(), "unknown selective native handle");
  return it->second;
}
}  // namespace

int64_t selective_native_register(std::shared_ptr<SelectiveNativeGraph> graph) {
  std::lock_guard<std::mutex> lock(selective_native_registry_mutex());
  static int64_t next = 0;
  TORCH_CHECK(next < INT64_MAX, "selective native handle overflow");
  const int64_t handle = ++next;
  selective_native_registry().emplace(handle, std::move(graph));
  return handle;
}
int64_t selective_native_create(const std::vector<at::Tensor>& buffers) {
  return selective_native_register(std::make_shared<SelectiveNativeGraph>(buffers));
}
std::vector<at::Tensor> selective_group_native_buffers(const std::vector<at::Tensor>& shared,
                                                       const std::vector<at::Tensor>& layers, int64_t count) {
  TORCH_CHECK(shared.size() == 16 && count > 0 && count <= INT64_MAX / 14 &&
              layers.size() == static_cast<size_t>(count) * 14, "invalid selective group tensor list");
  std::vector<at::Tensor> buffers{
      shared[0], shared[1], layers[0], layers[1], shared[2], shared[3],
      layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[8], layers[9],
      layers[10], layers[11], layers[12], layers[13], shared[5], shared[4], shared[7], shared[6],
      shared[8], shared[9], shared[10], shared[11], shared[12], shared[13], shared[14], shared[15]};
  buffers.insert(buffers.end(), layers.begin() + 14, layers.end());
  return buffers;
}
int64_t selective_group_native_create(const std::vector<at::Tensor>& shared,
                                      const std::vector<at::Tensor>& layers, int64_t count) {
  return selective_native_create(selective_group_native_buffers(shared, layers, count));
}
int64_t selective_group_tp_native_create(const std::vector<at::Tensor>& shared,
                                         const std::vector<at::Tensor>& layers, int64_t count,
                                         const std::vector<at::Tensor>& tp, int64_t rank, int64_t world) {
  return selective_native_register(std::make_shared<SelectiveNativeGraph>(
      selective_group_native_buffers(shared, layers, count), tp, rank, world));
}
template <typename Fn> void selective_native_call(int64_t handle, Fn fn) {
  auto graph = selective_native_get(handle);
  try { fn(*graph); } catch (...) { graph->notify_peer_error(); throw; }
}
void selective_native_capture(int64_t h, const at::Tensor& anchor) {
  selective_native_call(h, [&](auto& graph) { graph.capture(anchor); });
}
void selective_native_scatter_done(int64_t h, const at::Tensor& anchor) {
  selective_native_call(h, [&](auto& graph) { graph.record_scatter(anchor); });
}
void selective_native_start(int64_t h) { selective_native_call(h, [&](auto& graph) { graph.start(); }); }
int64_t selective_native_graph_identity(int64_t h) { return selective_native_get(h)->graph_identity(); }
void selective_native_begin(int64_t h, int64_t epoch, const at::Tensor& anchor) {
  selective_native_call(h, [&](auto& graph) { graph.begin(epoch, anchor); });
}
void selective_native_record_complete(int64_t h, const at::Tensor& anchor) {
  selective_native_call(h, [&](auto& graph) { graph.record_complete(anchor); });
}
void selective_native_retire(int64_t h, int64_t epoch) {
  selective_native_call(h, [&](auto& graph) { graph.retire(epoch); });
}
void selective_native_stop(int64_t h) { selective_native_call(h, [&](auto& graph) { graph.stop(); }); }
void selective_native_release(int64_t h) {
  auto graph = selective_native_get(h);
  graph->release();
  std::lock_guard<std::mutex> lock(selective_native_registry_mutex());
  selective_native_registry().erase(h);
}

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
HOT_FUNCTION void lru_resident_compact_with_plan_stable_rows(
    int64_t req_ids_ptr, int64_t last_req_ids_ptr, int64_t topk_indices_ptr, int64_t stable_prefix_lens_ptr,
    int64_t slot_to_token_ptr, int64_t lru_slots_ptr, int64_t current_slots_ptr, int64_t miss_count_ptr,
    int64_t miss_tokens_ptr, int64_t miss_slots_ptr, int64_t token_mark_workspace_ptr,
    int64_t token_pos_workspace_ptr, int64_t slot_workspace_ptr, int64_t miss_position_workspace_ptr,
    int64_t epochs_ptr, int64_t physical_row_workspace_ptr, int64_t physical_row_capacity,
    int64_t encoded_plan_ptr, int64_t encoded_plan_stride, int64_t num_reqs, int64_t topk, int64_t capacity,
    int64_t max_token, int64_t workspace_threads, int64_t requested_threads, int64_t visible_seq_lens_ptr) {
  TORCH_CHECK(physical_row_workspace_ptr != 0, "stable-row planner requires physical_row_workspace_ptr");
  TORCH_CHECK(physical_row_capacity >= num_reqs, "stable-row planner capacity must cover all logical rows");
  TORCH_CHECK(physical_row_capacity <= std::numeric_limits<int16_t>::max(),
              "stable-row planner capacity exceeds int16 row encoding");
  TORCH_CHECK(encoded_plan_ptr != 0, "stable-row planner requires encoded_plan_ptr");
  TORCH_CHECK(encoded_plan_stride >= topk + 7, "stable-row encoded_plan_stride must include control row");
  TORCH_CHECK(capacity <= std::numeric_limits<int16_t>::max(), "resident capacity exceeds int16 plan encoding");
  TORCH_CHECK(visible_seq_lens_ptr != 0, "stable-row planner requires visible_seq_lens_ptr");
  lru_resident_compact_impl(req_ids_ptr, last_req_ids_ptr, topk_indices_ptr, stable_prefix_lens_ptr, slot_to_token_ptr,
                            lru_slots_ptr, current_slots_ptr, miss_count_ptr, miss_tokens_ptr, miss_slots_ptr,
                            token_mark_workspace_ptr, token_pos_workspace_ptr, slot_workspace_ptr,
                            miss_position_workspace_ptr, epochs_ptr, num_reqs, topk, capacity, max_token,
                            workspace_threads, requested_threads, encoded_plan_ptr, encoded_plan_stride,
                            physical_row_workspace_ptr, physical_row_capacity, visible_seq_lens_ptr);
}

namespace {

struct LruResidentCompactWithPlanPayload {
  uintptr_t req_ids_ptr;
  uintptr_t last_req_ids_ptr;
  uintptr_t topk_indices_ptr;
  uintptr_t stable_prefix_lens_ptr;
  uintptr_t slot_to_token_ptr;
  uintptr_t lru_slots_ptr;
  uintptr_t current_slots_ptr;
  uintptr_t miss_count_ptr;
  uintptr_t miss_tokens_ptr;
  uintptr_t miss_slots_ptr;
  uintptr_t token_mark_workspace_ptr;
  uintptr_t token_pos_workspace_ptr;
  uintptr_t slot_workspace_ptr;
  uintptr_t miss_position_workspace_ptr;
  uintptr_t epochs_ptr;
  uintptr_t encoded_plan_ptr;
  int64_t encoded_plan_stride;
  uintptr_t physical_row_workspace_ptr;
  int64_t physical_row_capacity;
  uintptr_t visible_seq_lens_ptr;
  int64_t num_reqs;
  int64_t topk;
  int64_t capacity;
  int64_t max_token;
  int64_t workspace_threads;
  int64_t requested_threads;
  bool delete_after_run;
};

void lru_resident_compact_with_plan_callback(void* args) noexcept {
  auto* payload = static_cast<LruResidentCompactWithPlanPayload*>(args);
  lru_resident_compact_impl(
      payload->req_ids_ptr, payload->last_req_ids_ptr, payload->topk_indices_ptr, payload->stable_prefix_lens_ptr,
      payload->slot_to_token_ptr, payload->lru_slots_ptr, payload->current_slots_ptr, payload->miss_count_ptr,
      payload->miss_tokens_ptr, payload->miss_slots_ptr, payload->token_mark_workspace_ptr,
      payload->token_pos_workspace_ptr, payload->slot_workspace_ptr, payload->miss_position_workspace_ptr,
      payload->epochs_ptr, payload->num_reqs, payload->topk, payload->capacity, payload->max_token,
      payload->workspace_threads, payload->requested_threads, payload->encoded_plan_ptr, payload->encoded_plan_stride,
      payload->physical_row_workspace_ptr, payload->physical_row_capacity, payload->visible_seq_lens_ptr);
  if (payload->delete_after_run) {
    delete payload;
  }
}

}  // namespace

void enqueue_lru_resident_compact_with_plan_stable_rows(
    int64_t req_ids_ptr, int64_t last_req_ids_ptr, int64_t topk_indices_ptr, int64_t stable_prefix_lens_ptr,
    int64_t slot_to_token_ptr, int64_t lru_slots_ptr, int64_t current_slots_ptr, int64_t miss_count_ptr,
    int64_t miss_tokens_ptr, int64_t miss_slots_ptr, int64_t token_mark_workspace_ptr,
    int64_t token_pos_workspace_ptr, int64_t slot_workspace_ptr, int64_t miss_position_workspace_ptr,
    int64_t epochs_ptr, int64_t physical_row_workspace_ptr, int64_t physical_row_capacity,
    int64_t encoded_plan_ptr, int64_t encoded_plan_stride, int64_t num_reqs, int64_t topk, int64_t capacity,
    int64_t max_token, int64_t workspace_threads, int64_t requested_threads, int64_t visible_seq_lens_ptr) {
  TORCH_CHECK(physical_row_workspace_ptr != 0, "stable-row planner requires physical_row_workspace_ptr");
  TORCH_CHECK(physical_row_capacity >= num_reqs, "stable-row planner capacity must cover all logical rows");
  TORCH_CHECK(physical_row_capacity <= std::numeric_limits<int16_t>::max(),
              "stable-row planner capacity exceeds int16 row encoding");
  TORCH_CHECK(encoded_plan_ptr != 0, "stable-row planner requires encoded_plan_ptr");
  TORCH_CHECK(encoded_plan_stride >= topk + 7, "stable-row encoded_plan_stride must include control row");
  TORCH_CHECK(capacity <= std::numeric_limits<int16_t>::max(), "resident capacity exceeds int16 plan encoding");
  TORCH_CHECK(visible_seq_lens_ptr != 0, "stable-row planner requires visible_seq_lens_ptr");

  const auto stream = c10_npu::getCurrentNPUStream().stream();
  aclmdlRICaptureStatus capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI model_ri = nullptr;
  const aclError capture_ret = aclmdlRICaptureGetInfo(stream, &capture_status, &model_ri);
  TORCH_CHECK(capture_ret == ACL_SUCCESS,
              "aclmdlRICaptureGetInfo for stable-row external LRU plan failed, error code: ", capture_ret);
  TORCH_CHECK(capture_status != ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
              "stable-row external LRU plan cannot enqueue on an invalidated graph capture");
  const bool graph_lifetime = capture_status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
  TORCH_CHECK(!graph_lifetime || model_ri != nullptr, "active graph capture returned a null model runtime instance");

  auto payload =
      std::make_unique<LruResidentCompactWithPlanPayload>(LruResidentCompactWithPlanPayload{
          static_cast<uintptr_t>(req_ids_ptr),
          static_cast<uintptr_t>(last_req_ids_ptr),
          static_cast<uintptr_t>(topk_indices_ptr),
          static_cast<uintptr_t>(stable_prefix_lens_ptr),
          static_cast<uintptr_t>(slot_to_token_ptr),
          static_cast<uintptr_t>(lru_slots_ptr),
          static_cast<uintptr_t>(current_slots_ptr),
          static_cast<uintptr_t>(miss_count_ptr),
          static_cast<uintptr_t>(miss_tokens_ptr),
          static_cast<uintptr_t>(miss_slots_ptr),
          static_cast<uintptr_t>(token_mark_workspace_ptr),
          static_cast<uintptr_t>(token_pos_workspace_ptr),
          static_cast<uintptr_t>(slot_workspace_ptr),
          static_cast<uintptr_t>(miss_position_workspace_ptr),
          static_cast<uintptr_t>(epochs_ptr),
          static_cast<uintptr_t>(encoded_plan_ptr),
          encoded_plan_stride,
          static_cast<uintptr_t>(physical_row_workspace_ptr),
          physical_row_capacity,
          static_cast<uintptr_t>(visible_seq_lens_ptr),
          num_reqs,
          topk,
          capacity,
          max_token,
          workspace_threads,
          requested_threads,
          !graph_lifetime});
  auto* raw_payload = payload.get();
  if (graph_lifetime) {
    raw_payload = retain_host_callback_graph_payload(model_ri, std::move(payload));
  }
  const aclError ret = aclrtLaunchHostFunc(stream, lru_resident_compact_with_plan_callback, raw_payload);
  if (ret == ACL_SUCCESS && !graph_lifetime) {
    payload.release();
  }
  TORCH_CHECK(ret == ACL_SUCCESS, "aclrtLaunchHostFunc for stable-row external LRU plan failed, error code: ", ret);
}

at::Tensor restore_tensor(uintptr_t ptr_val, const std::vector<int64_t>& shape,
                          torch::ScalarType dtype = torch::kBFloat16) {
  if (ptr_val == 0) {
    return at::Tensor();
  }
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  return torch::from_blob(reinterpret_cast<void*>(ptr_val), shape, options);
}

#endif
#endif

#if defined(VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD) || defined(VLLM_ASCEND_ENABLE_SELECTIVE_LIM)
namespace {

void check_selective_cpu_tensor(const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() && tensor.scalar_type() == dtype,
              name, " must be a contiguous CPU tensor with the required dtype");
}

struct SelectiveHostGatherPayload {
  std::array<at::Tensor, 8> owners;
  const uint64_t* sources = nullptr;
  const uint64_t* destinations = nullptr;
  const uint32_t* lengths = nullptr;
  uint8_t* packed_k = nullptr;
  uint8_t* packed_v = nullptr;
  uint64_t* control = nullptr;
  uint64_t* epoch = nullptr;
  std::size_t entries = 0, k_bytes = 0, v_bytes = 0;
  int64_t rank = 0, world_size = 1;
  bool to_device = false, complete_only = false, delete_after_run = false;
};

void wait_selective_epoch(const uint64_t* value, uint64_t epoch) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (__atomic_load_n(value, __ATOMIC_ACQUIRE) < epoch) {
    if (std::chrono::steady_clock::now() >= deadline) {
      std::fputs("Selective Host gather peer synchronization timed out\n", stderr);
      std::abort();
    }
    std::this_thread::yield();
  }
}

void selective_copy_rows(const SelectiveHostGatherPayload& plan) noexcept {
  for (std::size_t component = 0; component < 2; ++component) {
    const auto row_bytes = component == 0 ? plan.k_bytes : plan.v_bytes;
    auto* packed = component == 0 ? plan.packed_k : plan.packed_v;
    for (std::size_t row = 0; row < plan.entries; ++row) {
      const auto index = component * plan.entries + row;
      const auto length = plan.lengths[index];
      if (length == 0) continue;
      const auto address = plan.to_device ? plan.sources[index] : plan.destinations[index];
      if (length > row_bytes || address == 0) {
        std::fputs("Selective Host gather has an invalid row descriptor\n", stderr);
        std::abort();
      }
      if (plan.to_device) {
        std::memcpy(packed + row * row_bytes, reinterpret_cast<const void*>(address), length);
      } else {
        std::memcpy(reinterpret_cast<void*>(address), packed + row * row_bytes, length);
      }
    }
  }
}

void selective_host_gather_callback(void* data) noexcept {
  auto& plan = *static_cast<SelectiveHostGatherPayload*>(data);
  if (plan.complete_only) {
    if (*plan.epoch == 0) {
      std::fputs("Selective Host completion ran before its gather\n", stderr);
      std::abort();
    }
    // The caller acknowledges only after every SDMA read of the shared payload has completed.
    __atomic_store_n(plan.control + 6 + plan.rank, *plan.epoch, __ATOMIC_RELEASE);
  } else if (!plan.to_device) {
    selective_copy_rows(plan);
  } else {
    auto* control = plan.control;
    const uint64_t epoch = ++*plan.epoch;
    uint64_t fingerprint = 14695981039346656037ULL;
    uint64_t skip = 1;
    for (std::size_t i = 0; i < 2 * plan.entries; ++i) {
      if (plan.lengths[i] != 0) {
        skip = 0;
        fingerprint = (fingerprint ^ plan.sources[i]) * 1099511628211ULL;
        fingerprint = (fingerprint ^ i) * 1099511628211ULL;
        fingerprint = (fingerprint ^ plan.lengths[i]) * 1099511628211ULL;
      }
    }
    if (plan.rank == 0) {
      for (int64_t rank = 1; rank < plan.world_size; ++rank) {
        wait_selective_epoch(control + 6 + rank, epoch - 1);
      }
      selective_copy_rows(plan);
      control[1] = fingerprint;
      control[2] = plan.entries;
      control[3] = plan.k_bytes;
      control[4] = plan.v_bytes;
      control[5] = skip;
      __atomic_store_n(control, epoch, __ATOMIC_RELEASE);
    } else {
      wait_selective_epoch(control, epoch);
      if (__atomic_load_n(control, __ATOMIC_ACQUIRE) != epoch || control[1] != fingerprint ||
          control[2] != plan.entries || control[3] != plan.k_bytes || control[4] != plan.v_bytes ||
          control[5] != skip) {
        std::fputs("Selective Host gather TP plans disagree\n", stderr);
        std::abort();
      }
    }
  }
  if (plan.delete_after_run) delete &plan;
}

void enqueue_selective_payload(std::unique_ptr<SelectiveHostGatherPayload> payload) {
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI model = nullptr;
  auto result = aclmdlRICaptureGetInfo(stream, &status, &model);
  TORCH_CHECK(result == ACL_SUCCESS && status != ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
              "Selective Host callback capture query failed, error code: ", result);
  const bool captured = status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
  TORCH_CHECK(!captured || model != nullptr, "Selective Host callback capture returned a null model");
  payload->delete_after_run = !captured;
  auto* raw = payload.get();
  if (captured) raw = retain_host_callback_graph_payload(model, std::move(payload));
  result = aclrtLaunchHostFunc(stream, selective_host_gather_callback, raw);
  if (result == ACL_SUCCESS && !captured) payload.release();
  TORCH_CHECK(result == ACL_SUCCESS, "Selective Host callback enqueue failed, error code: ", result);
}

}  // namespace

void enqueue_selective_host_gather(
    const at::Tensor& anchor, const at::Tensor& sources, const at::Tensor& destinations,
    const at::Tensor& lengths, const at::Tensor& packed_k, const at::Tensor& packed_v,
    const at::Tensor& control, const at::Tensor& epoch, int64_t entries,
    int64_t k_bytes, int64_t v_bytes, int64_t rank, int64_t world_size, bool to_device) {
  TORCH_CHECK(anchor.device().is_privateuseone(), "Selective Host gather requires an NPU anchor");
  const c10_npu::NPUGuard guard(anchor.device());
  check_selective_cpu_tensor(sources, at::kLong, "sources");
  check_selective_cpu_tensor(destinations, at::kLong, "destinations");
  check_selective_cpu_tensor(lengths, at::kInt, "lengths");
  check_selective_cpu_tensor(packed_k, at::kByte, "packed_k");
  check_selective_cpu_tensor(packed_v, at::kByte, "packed_v");
  check_selective_cpu_tensor(control, at::kLong, "control");
  check_selective_cpu_tensor(epoch, at::kLong, "epoch");
  TORCH_CHECK(entries > 0 && entries <= std::numeric_limits<int64_t>::max() / 2 &&
                  k_bytes > 0 && v_bytes > 0 && k_bytes <= UINT32_MAX && v_bytes <= UINT32_MAX,
              "Selective Host gather requires positive, representable entries and row widths");
  TORCH_CHECK(sources.numel() >= 2 * entries && destinations.numel() >= 2 * entries &&
                  lengths.numel() >= 2 * entries && packed_k.numel() / k_bytes >= entries &&
                  packed_v.numel() / v_bytes >= entries,
              "Selective Host gather descriptors or packed buffers are too small");
  TORCH_CHECK(world_size > 0 && world_size <= std::numeric_limits<int64_t>::max() - 6 &&
                  rank >= 0 && rank < world_size,
              "Selective Host gather has an invalid TP rank/world size");
  TORCH_CHECK(!to_device || (control.numel() >= 6 + world_size && epoch.numel() == 1),
              "Selective Host gather requires 6+world_size control words and one local epoch");
  auto payload = std::make_unique<SelectiveHostGatherPayload>();
  payload->owners = {anchor, sources, destinations, lengths, packed_k, packed_v, control, epoch};
  payload->sources = reinterpret_cast<const uint64_t*>(sources.data_ptr<int64_t>());
  payload->destinations = reinterpret_cast<const uint64_t*>(destinations.data_ptr<int64_t>());
  payload->lengths = reinterpret_cast<const uint32_t*>(lengths.data_ptr<int32_t>());
  payload->packed_k = packed_k.data_ptr<uint8_t>();
  payload->packed_v = packed_v.data_ptr<uint8_t>();
  payload->control = reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>());
  payload->epoch = reinterpret_cast<uint64_t*>(epoch.data_ptr<int64_t>());
  payload->entries = entries;
  payload->k_bytes = k_bytes;
  payload->v_bytes = v_bytes;
  payload->rank = rank;
  payload->world_size = world_size;
  payload->to_device = to_device;
  enqueue_selective_payload(std::move(payload));
}

void enqueue_selective_host_gather_done(
    const at::Tensor& anchor, const at::Tensor& control, const at::Tensor& epoch, int64_t rank) {
  TORCH_CHECK(anchor.device().is_privateuseone(), "Selective Host completion requires an NPU anchor");
  const c10_npu::NPUGuard guard(anchor.device());
  check_selective_cpu_tensor(control, at::kLong, "control");
  check_selective_cpu_tensor(epoch, at::kLong, "epoch");
  TORCH_CHECK(rank >= 0 && control.numel() > 6 && rank < control.numel() - 6 && epoch.numel() == 1,
              "Selective Host completion has an invalid control/epoch/rank");
  auto payload = std::make_unique<SelectiveHostGatherPayload>();
  payload->owners[0] = anchor;
  payload->owners[6] = control;
  payload->owners[7] = epoch;
  payload->control = reinterpret_cast<uint64_t*>(control.data_ptr<int64_t>());
  payload->epoch = reinterpret_cast<uint64_t*>(epoch.data_ptr<int64_t>());
  payload->rank = rank;
  payload->complete_only = true;
  enqueue_selective_payload(std::move(payload));
}

#endif

// Adapted from ops-transformer attention/lightning_indexer_v2/torch_extension.
// Preserve the upstream ACLNN argument order and output protocol.
namespace li_v2 {
constexpr int64_t LI_V2_METADATA_SIZE = 2048;
inline at::Tensor li_v2_optional_tensor(const c10::optional<at::Tensor>& value, at::Device device)
{
    return value.has_value() ? *value : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
}

at::Tensor LightningIndexerMetadata(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                    const c10::optional<at::Tensor>& cuSeqlensQ,
                                    const c10::optional<at::Tensor>& cuSeqlensK,
                                    const c10::optional<at::Tensor>& sequsedQ,
                                    const c10::optional<at::Tensor>& sequsedK,
                                    const c10::optional<at::Tensor>& cmpResidualK, int64_t batchSize,
                                    int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ,
                                    c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio)
{
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensK.has_value()) {
        outputDevice = cuSeqlensK.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (cmpResidualK.has_value()) {
        outputDevice = cmpResidualK.value().device();
    }

    TORCH_CHECK(outputDevice.type() == c10::DeviceType::PrivateUse1, "LI metadata inputs must be on NPU");
    for (const auto* tensor : {&cuSeqlensQ, &cuSeqlensK, &sequsedQ, &sequsedK, &cmpResidualK}) {
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == outputDevice,
                    "LI metadata inputs must be on the same NPU");
    }
    const c10::OptionalDeviceGuard device_guard(outputDevice);
    at::Tensor output = torch::empty({LI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = li_v2_optional_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = li_v2_optional_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = li_v2_optional_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = li_v2_optional_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = li_v2_optional_tensor(cmpResidualK, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char* layoutQPtr = const_cast<char*>(layoutQStr.c_str());
    char* layoutKPtr = const_cast<char*>(layoutKStr.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal, cmpResidualKVal,
              numHeadsQ, numHeadsK, headDim, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQPtr, layoutKPtr, maskMode,
              cmpRatio, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> ConstructLightningIndexerOutputTensor(const at::Tensor& query, const at::Tensor& key,
                                                                         int64_t topk, std::string queryLayoutStr,
                                                                         std::string keyLayoutStr, bool returnValue)
{
    TORCH_CHECK(queryLayoutStr == "TND" || queryLayoutStr == "BSND", "Unsupported query layout");
    TORCH_CHECK(keyLayoutStr == "TND" || keyLayoutStr == "BSND" || keyLayoutStr == "PA_BBND", "Unsupported key layout");
    TORCH_CHECK(query.dim() == (queryLayoutStr == "TND" ? 3 : 4), "Invalid query rank");
    TORCH_CHECK(key.dim() == (keyLayoutStr == "TND" ? 3 : 4), "Invalid key rank");
    at::SmallVector<int64_t, 4> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(1) : key.size(2);
    if (queryLayoutStr == "BSND") {
        outputSize = {query.size(0), query.size(1), keyHeadNum, topk};
    } else {
        int nDimIndex = 0;
        nDimIndex = (keyLayoutStr == "TND") ? 1 : 2;
        outputSize = {query.size(0), key.size(nDimIndex), topk};
    }
    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        sparseValuesOut = at::empty(outputSize, query.options().dtype(at::kFloat));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor> LightningIndexer(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& w, int64_t topk,
    const c10::optional<at::Tensor>& cuSeqlensQ, const c10::optional<at::Tensor>& cuSeqlensK,
    const c10::optional<at::Tensor>& sequsedQ, const c10::optional<at::Tensor>& sequsedK,
    const c10::optional<at::Tensor>& cmpResidualK, const c10::optional<at::Tensor>& blockTable,
    const c10::optional<at::Tensor>& outputIdxOffset, const c10::optional<at::Tensor>& metadata, int64_t maxSeqlenQ,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t returnValue)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "LI query must be on NPU");
    TORCH_CHECK(k.device() == q.device() && w.device() == q.device(), "LI Q/K/weights must be on the same NPU");
    for (const auto* tensor : {&cuSeqlensQ, &cuSeqlensK, &sequsedQ, &sequsedK, &cmpResidualK, &blockTable,
                               &outputIdxOffset, &metadata}) {
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == q.device(),
                    "LI optional inputs must be on the query NPU");
    }
    const c10::OptionalDeviceGuard device_guard(q.device());
    TORCH_CHECK(returnValue == 0 || returnValue == 1, "return_value must be 0 or 1");
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> lightningIndexerOutput =
        ConstructLightningIndexerOutputTensor(q, k, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(lightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(lightningIndexerOutput);
    // convert str
    char* queryLayoutPtr = const_cast<char*>(queryLayoutStr.c_str());
    char* keyLayoutPtr = const_cast<char*>(keyLayoutStr.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexerV2, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable,
              outputIdxOffset, metadata, topk, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode, cmpRatio,
              returnValue, sparseIndicesOut, sparseValuesOut);
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

}  // namespace li_v2

} // namespace vllm_ascend

#ifdef ASCEND_PLATFORM_310P
// Pybind on Ascend 310P
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
    ops.def("get_npu_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor?");
    ops.impl("get_npu_view_from_cpu_tensor", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_npu_view_from_cpu_tensor);
    ops.def("get_physical_device_id(int user_device_id) -> int");
    ops.impl("get_physical_device_id", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_physical_device_id);
    ops.def(
        "npu_causal_conv1d_310(Tensor x, "
        "                         Tensor weight, "
        "                         Tensor? bias, "
        "                         Tensor conv_states, "
        "                         Tensor? query_start_loc, "
        "                         Tensor? cache_indices, "
        "                         Tensor? initial_state_mode, "
        "                         Tensor? num_accepted_tokens, "
        "                         int activation_mode, "
        "                         int pad_slot_id, "
        "                         int run_mode) -> (Tensor output)");
    ops.impl("npu_causal_conv1d_310", torch::kPrivateUse1, &vllm_ascend::npu_causal_conv1d_310);

    ops.def(
        "npu_recurrent_gated_delta_rule_310(Tensor query, "
        "                                   Tensor key, "
        "                                   Tensor value, "
        "                                   Tensor beta, "
        "                                   Tensor state, "
        "                                   Tensor actual_seq_lengths, "
        "                                   Tensor ssm_state_indices, "
        "                                   Tensor? g, "
        "                                   Tensor? gk, "
        "                                   Tensor? num_accepted_tokens, "
        "                                   float scale_value=1.0) -> (Tensor output)");
    ops.impl("npu_recurrent_gated_delta_rule_310", torch::kPrivateUse1, &vllm_ascend::npu_recurrent_gated_delta_rule_310);

    ops.def(
        "chunk_gated_delta_rule_fwd_h_vllm(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h_vllm", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h_vllm);

    ops.def(
        "chunk_fwd_o_vllm(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o_vllm", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o_vllm);

    ops.def(
        "kda_gate_cumsum_vllm(Tensor g, int chunk_size, *, Tensor? A_log=None, Tensor? dt_bias=None, int[]? cu_seqlens=None, bool? use_gate_in_kernel=False, bool? safe_gate=False, float? lower_bound=-5.0, str layout=\"BSND\") -> Tensor"
    );
    ops.impl("kda_gate_cumsum_vllm", torch::kPrivateUse1, &vllm_ascend::kda_gate_cumsum_vllm);

    ops.def(
        "kda_layout_swap12(Tensor x, *, Tensor? dependency=None) -> Tensor"
    );
    ops.impl("kda_layout_swap12", torch::kPrivateUse1, &vllm_ascend::kda_layout_swap12);
}
#else
// Pybind on other platform
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
    ops.def("get_npu_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor?");
    ops.impl("get_npu_view_from_cpu_tensor", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_npu_view_from_cpu_tensor);
    ops.def("get_physical_device_id(int user_device_id) -> int");
    ops.impl("get_physical_device_id", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_physical_device_id);

    // vLLM-Ascend custom ops
    // Gemma RmsNorm
    ops.def(
        "npu_gemma_rms_norm(Tensor x, "
                            "Tensor gamma, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd)"
        );
    ops.impl("npu_gemma_rms_norm", torch::kPrivateUse1, &vllm_ascend::npu_gemma_rms_norm);

    ops.def(
        "attn_res_fwd(Tensor(b) prefix_sum, Tensor? addend, Tensor(a!) block_residual, "
        "Tensor proj_weight, Tensor norm_weight, float norm_eps, int num_valid_blocks, "
        "Tensor? output_norm_weight=None, float output_norm_eps=1e-5, int block_write_idx=-1, "
        "bool return_materialized=False, bool mix=True, bool optimize_prefill=False) -> (Tensor(c), Tensor(b), Tensor(c))");
    ops.impl("attn_res_fwd", torch::kPrivateUse1, &vllm_ascend::attn_res_fwd);

    ops.def(
        "dequant_situ_quant(Tensor x, "
        "                   *, Tensor? weight_scale=None, "
        "                   Tensor? activation_scale=None, "
        "                   Tensor? bias=None, "
        "                   Tensor? quant_scale=None, "
        "                   Tensor? quant_offset=None, "
        "                   Tensor? group_index=None, "
        "                   float beta=4.0, "
        "                   float linear_beta=25.0, "
        "                   bool activate_left=True, "
        "                   str quant_mode=\"dynamic\") -> (Tensor y, Tensor scale)");
    ops.impl("dequant_situ_quant", torch::kPrivateUse1, &vllm_ascend::dequant_situ_quant);

    ops.def(
        "situ_mx_quant(Tensor x, "
        "              float beta=1.0, "
        "              float linear_beta=0.0, "
        "              bool activate_left=False, "
        "              int dst_type=36) -> (Tensor y, Tensor mxscale)");
    ops.impl("situ_mx_quant", torch::kPrivateUse1, &vllm_ascend::situ_mx_quant);

    ops.def(
        "grouped_matmul_situ_quant(Tensor x, Tensor weight, Tensor weight_scale, Tensor? weight_assist_matrix, "
        "Tensor? bias, Tensor x_scale, Tensor? smooth_scale, Tensor group_list, int dequant_mode, "
        "int dequant_dtype, int quant_mode, int group_list_type, int[]? tuning_config, float beta, "
        "float linear_beta) -> (Tensor output, Tensor output_scale)");
    ops.impl("grouped_matmul_situ_quant", torch::kPrivateUse1,
             &ascend_kernel::gmm_situ_quant_v2_nd);
    ops.def(
        "grouped_matmul_situ_quant.list(Tensor x, Tensor[] weight, Tensor[] weight_scale, "
        "Tensor[]? weight_assist_matrix, Tensor? bias, Tensor x_scale, Tensor? smooth_scale, "
        "Tensor group_list, int dequant_mode, int dequant_dtype, int quant_mode, int group_list_type, "
        "int[]? tuning_config, float beta, float linear_beta) -> (Tensor output, Tensor output_scale)");
    ops.impl("grouped_matmul_situ_quant.list", torch::kPrivateUse1,
             &ascend_kernel::gmm_situ_quant_v2_nd_list);
    ops.def(
        "grouped_matmul_situ_quant_weight_nz(Tensor x, Tensor weight, Tensor weight_scale, "
        "Tensor? weight_assist_matrix, Tensor? bias, Tensor x_scale, Tensor? smooth_scale, "
        "Tensor group_list, int dequant_mode, int dequant_dtype, int quant_mode, int group_list_type, "
        "int[]? tuning_config, float beta, float linear_beta) -> (Tensor output, Tensor output_scale)");
    ops.impl("grouped_matmul_situ_quant_weight_nz", torch::kPrivateUse1,
             &ascend_kernel::gmm_situ_quant_v2_nz);
    ops.def(
        "grouped_matmul_situ_quant_weight_nz.list(Tensor x, Tensor[] weight, Tensor[] weight_scale, "
        "Tensor[]? weight_assist_matrix, Tensor? bias, Tensor x_scale, Tensor? smooth_scale, "
        "Tensor group_list, int dequant_mode, int dequant_dtype, int quant_mode, int group_list_type, "
        "int[]? tuning_config, float beta, float linear_beta) -> (Tensor output, Tensor output_scale)");
    ops.impl("grouped_matmul_situ_quant_weight_nz.list", torch::kPrivateUse1,
             &ascend_kernel::gmm_situ_quant_v2_nz_list);

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
    // Direct kernel custom ops
    ops.def("bgmv_shrink(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y, float scale) -> ()");
    ops.impl("bgmv_shrink", torch::kPrivateUse1, &vllm_ascend::bgmv_shrink);

    ops.def(
        "bgmv_expand(Tensor! x, Tensor! weight, Tensor! indices, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("bgmv_expand", torch::kPrivateUse1, &vllm_ascend::bgmv_expand);

    ops.def("sgmv_shrink(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y, float scale) -> ()");
    ops.impl("sgmv_shrink", torch::kPrivateUse1, &vllm_ascend::sgmv_shrink);

    ops.def(
        "sgmv_expand(Tensor! x, Tensor! weight, Tensor! lora_indices, Tensor! seq_len, Tensor! y,"
        "            int slice_offset, int slice_size) -> Tensor");
    ops.impl("sgmv_expand", torch::kPrivateUse1, &vllm_ascend::sgmv_expand);

    ops.def(
        "mla_preprocess(Tensor hiddenState, Tensor wdqkv,"
        "               Tensor? descale0, Tensor gamma1, Tensor? beta1, Tensor wuq, Tensor? descale1,"
        "               Tensor gamma2, Tensor? cos, Tensor? sin, Tensor wuk, Tensor kv_cache,"
        "               Tensor kv_cache_rope, Tensor slotmapping, Tensor? quant_scale0,"
        "               Tensor? quant_offset0, Tensor? bias0, Tensor? quant_scale1, Tensor? quant_offset1,"
        "               Tensor? bias1, Tensor? ctkv_scale, Tensor? q_nope_scale, str? cache_mode,"
        "               str? quant_mode, bool? enable_inner_out, Tensor! q_out0, Tensor! kv_cache_out0, Tensor! q_out1,"
        "               Tensor! kv_cache_out1, Tensor! inner_out) -> (Tensor q_out0, Tensor kv_cache_out0,"
        "                                          Tensor q_out1, Tensor kv_cache_out1, Tensor inner_out)"
    );
    ops.impl("mla_preprocess", torch::kPrivateUse1, &vllm_ascend::mla_preprocess);

    ops.def("swap_blocks(Tensor! x, Tensor! y, Tensor z) -> ()");
    ops.impl("swap_blocks", torch::kPrivateUse1, &vllm_ascend::swap_blocks);
#endif

    // swap_blocks_batch takes CPU tensors (int64 pointer/size arrays), not NPU
    // tensors, so dispatch must be registered on the CPU backend. The function
    // internally submits async memcpy on the current NPU stream.
    ops.def("swap_blocks_batch(Tensor x, Tensor y, Tensor z, int direction) -> ()");
    ops.impl("swap_blocks_batch", torch::kCPU, &vllm_ascend::swap_blocks_batch);

#if defined(VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD) || defined(VLLM_ASCEND_ENABLE_SELECTIVE_LIM)
    ops.def("enqueue_selective_host_gather(Tensor anchor, Tensor sources, Tensor destinations, Tensor lengths, "
            "Tensor(a!) packed_k, Tensor(b!) packed_v, Tensor(c!) control, Tensor(d!) epoch, "
            "int entries, int k_bytes, int v_bytes, int rank, int world_size, bool to_device) -> ()");
    ops.impl("enqueue_selective_host_gather", torch::kPrivateUse1, &vllm_ascend::enqueue_selective_host_gather);
    ops.def("enqueue_selective_host_gather_done(Tensor anchor, Tensor(a!) control, Tensor epoch, int rank) -> ()");
    ops.impl("enqueue_selective_host_gather_done", torch::kPrivateUse1,
             &vllm_ascend::enqueue_selective_host_gather_done);

    ops.def("selective_group_lru_plan_cpu(Tensor topk, Tensor page_tables, Tensor(a!) slot_to_token, "
            "Tensor(b!) lru_slots, Tensor(c!) resident_pages, Tensor(d!) last_identity, Tensor(e!) token_mark, "
            "Tensor(f!) token_pos, Tensor(g!) epoch, Tensor(h!) control, int[] meta) -> Tensor[]");
    ops.impl("selective_group_lru_plan_cpu", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_group_lru_plan_cpu);
    ops.def("selective_tp_exchange_cpu(Tensor(a!) storage, int rank, int world, int epoch, str value) -> str[]");
    ops.impl("selective_tp_exchange_cpu", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_tp_exchange_cpu);
    ops.def("selective_tp_exchange_confirm_cpu(Tensor(a!) storage, int rank, int world, int epoch) -> ()");
    ops.impl("selective_tp_exchange_confirm_cpu", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_tp_exchange_confirm_cpu);
    ops.def("selective_tp_exchange_fail_cpu(Tensor(a!) storage) -> ()");
    ops.impl("selective_tp_exchange_fail_cpu", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_tp_exchange_fail_cpu);
    ops.def("selective_group_lru_finish_cpu(Tensor(a!) last_identity, Tensor(b!) control, int ticket, bool success) -> ()");
    ops.impl("selective_group_lru_finish_cpu", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_group_lru_finish_cpu);
    ops.def("selective_group_lru_retire_cpu(Tensor(a!) last_identity, Tensor(b!) control, "
            "int request, int generation) -> bool");
    ops.impl("selective_group_lru_retire_cpu", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_group_lru_retire_cpu);
    ops.def("selective_native_create(Tensor[] buffers) -> int");
    ops.impl("selective_native_create", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_create);
    ops.def("selective_group_native_create(Tensor[] shared, Tensor[] layers, int count) -> int");
    ops.impl("selective_group_native_create", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_group_native_create);
    ops.def("selective_group_tp_native_create(Tensor[] shared, Tensor[] layers, int count, "
            "Tensor[] tp, int rank, int world) -> int");
    ops.impl("selective_group_tp_native_create", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_group_tp_native_create);
    ops.def("selective_native_capture(int handle, Tensor anchor) -> ()");
    ops.impl("selective_native_capture", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_capture);
    ops.def("selective_native_scatter_done(int handle, Tensor anchor) -> ()");
    ops.impl("selective_native_scatter_done", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_scatter_done);
    ops.def("selective_native_start(int handle) -> ()");
    ops.impl("selective_native_start", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_start);
    ops.def("selective_native_graph_identity(int handle) -> int");
    ops.impl("selective_native_graph_identity", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_graph_identity);
    ops.def("selective_native_begin(int handle, int epoch, Tensor anchor) -> ()");
    ops.impl("selective_native_begin", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_begin);
    ops.def("selective_native_record_complete(int handle, Tensor anchor) -> ()");
    ops.impl("selective_native_record_complete", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_record_complete);
    ops.def("selective_native_retire(int handle, int epoch) -> ()");
    ops.impl("selective_native_retire", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_retire);
    ops.def("selective_native_stop(int handle) -> ()");
    ops.impl("selective_native_stop", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_stop);
    ops.def("selective_native_release(int handle) -> ()");
    ops.impl("selective_native_release", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::selective_native_release);
#endif

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
    ops.def("sparse_kv_warmup_lru_resident_threads(int requested_threads) -> int");
    ops.impl("sparse_kv_warmup_lru_resident_threads", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::warmup_lru_resident_threads);

    ops.def("sparse_kv_lru_resident_compact_with_plan_stable_rows(int req_ids_ptr, int last_req_ids_ptr, "
            "int topk_indices_ptr, int stable_prefix_lens_ptr, int slot_to_token_ptr, int lru_slots_ptr, "
            "int current_slots_ptr, int miss_count_ptr, int miss_tokens_ptr, int miss_slots_ptr, "
            "int token_mark_workspace_ptr, int token_pos_workspace_ptr, int slot_workspace_ptr, "
            "int miss_position_workspace_ptr, int epochs_ptr, int physical_row_workspace_ptr, "
            "int physical_row_capacity, int encoded_plan_ptr, int encoded_plan_stride, int num_reqs, "
            "int topk, int capacity, int max_token, int workspace_threads, int requested_threads, "
            "int visible_seq_lens_ptr) -> ()");
    ops.impl("sparse_kv_lru_resident_compact_with_plan_stable_rows", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::lru_resident_compact_with_plan_stable_rows);

    ops.def("sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows(int req_ids_ptr, int last_req_ids_ptr, "
            "int topk_indices_ptr, int stable_prefix_lens_ptr, int slot_to_token_ptr, int lru_slots_ptr, "
            "int current_slots_ptr, int miss_count_ptr, int miss_tokens_ptr, int miss_slots_ptr, "
            "int token_mark_workspace_ptr, int token_pos_workspace_ptr, int slot_workspace_ptr, "
            "int miss_position_workspace_ptr, int epochs_ptr, int physical_row_workspace_ptr, "
            "int physical_row_capacity, int encoded_plan_ptr, int encoded_plan_stride, int num_reqs, "
            "int topk, int capacity, int max_token, int workspace_threads, int requested_threads, "
            "int visible_seq_lens_ptr) -> ()");
    ops.impl("sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows",
             c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::enqueue_lru_resident_compact_with_plan_stable_rows);

    ops.def("sparse_kv_restore_bfloat16_tensor(int ptr_val, int[] shape) -> Tensor");
    ops.impl("sparse_kv_restore_bfloat16_tensor", c10::DispatchKey::CompositeExplicitAutograd,
             [](int64_t ptr_val, const std::vector<int64_t>& shape) {
                 TORCH_CHECK(ptr_val != 0, "restore_bfloat16_tensor requires a non-zero pointer");
                 auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
                 return at::from_blob(reinterpret_cast<void*>(ptr_val), shape, options);
             });

    ops.def("sparse_kv_restore_int16_tensor(int ptr_val, int[] shape) -> Tensor");
    ops.impl("sparse_kv_restore_int16_tensor", c10::DispatchKey::CompositeExplicitAutograd,
             [](int64_t ptr_val, const std::vector<int64_t>& shape) {
                 TORCH_CHECK(ptr_val != 0, "restore_int16_tensor requires a non-zero pointer");
                 auto options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
                 return at::from_blob(reinterpret_cast<void*>(ptr_val), shape, options);
             });
#endif

    ops.def("supports_eccpu_offload() -> bool");
    ops.impl("supports_eccpu_offload", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::supports_eccpu_offload);
    ops.def("register_pinned_host_mmap(Tensor x) -> ()");
    ops.impl("register_pinned_host_mmap", torch::kCPU,
             &vllm_ascend::register_pinned_host_mmap);
    ops.def("unregister_pinned_host_mmap(Tensor x) -> ()");
    ops.impl("unregister_pinned_host_mmap", torch::kCPU,
             &vllm_ascend::unregister_pinned_host_mmap);
    ops.def("device_print(str msg) -> ()");
    ops.impl("device_print", c10::DispatchKey::CompositeExplicitAutograd,
             static_cast<void (*)(c10::string_view)>(&vllm_ascend::device_print));

    ops.def("device_print_tensor(Tensor tensor) -> ()");
    ops.impl("device_print_tensor", c10::DispatchKey::CompositeExplicitAutograd,
             static_cast<void (*)(const at::Tensor&)>(&vllm_ascend::device_print));

    ops.def("get_npu_storage_shape(Tensor tensor) -> int[]");
    ops.impl("get_npu_storage_shape", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::get_npu_storage_shape);

    ops.def(
        "grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "                            Tensor group_list, *, Tensor? bias=None,"
        "                            Tensor? offset=None, float swiglu_limit=0.0) ->"
        "                            (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant);

    ops.def(
        "grouped_matmul_swiglu_quant_weight_nz(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "                                      Tensor group_list, *, Tensor? bias=None,"
        "                                      Tensor? offset=None, float swiglu_limit=0.0) -> "
        "                                      (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant_weight_nz", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz);

    ops.def(
        "grouped_matmul_swiglu_quant_weight_nz_tensor_list(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,"
        "                                                  Tensor group_list, *,"
        "                                                  Tensor? bias=None, Tensor? offset=None, float swiglu_limit=0.0) ->"
        "                                                  (Tensor output, Tensor output_scale, Tensor output_offset)"
    );
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz_tensor_list);

    ops.def(
        "grouped_matmul_swiglu_quant_v2(Tensor x, Tensor[] weight, Tensor[] weight_scale, Tensor x_scale,  Tensor group_list,  Tensor? smooth_scale=None,"
        "                                                   Tensor[]? weight_assist_matrix=None, Tensor? bias=None, int? dequant_mode=0, int? dequant_dtype=0, int? quant_mode=0,"
        "                                                 int? quant_dtype=0, bool transpose_weight=False, int group_list_type=0, int[2] tuning_config=[],float swiglu_limit=0.0) ->"
        "                                                  (Tensor output, Tensor output_scale)"
    );
    ops.impl("grouped_matmul_swiglu_quant_v2", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_v2);

    // fused_quant_lightning_indexer_manage (nanovllm fused_li_manage_mtp port, quantized C8 variant)
    ops.def(
        "npu_fused_quant_lightning_indexer_manage(Tensor index_weights, Tensor query_dequant_scale, "
        "Tensor query, Tensor index_key_dequant_scale, Tensor index_key_cache, "
        "Tensor index_block_table, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key, "
        "Tensor num_cache_tokens, Tensor request_state, Tensor req_pool_entries, "
        "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
        "Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, "
        "Tensor(e!) miss_src_ids, Tensor(f!) miss_dst_slots, "
        "Tensor(g!) miss_counts) -> ()"
    );
    ops.impl("npu_fused_quant_lightning_indexer_manage", torch::kPrivateUse1,
             &vllm_ascend::npu_fused_quant_lightning_indexer_manage);

    // k2q_csr: q2k -> k2q CSR (Meta/Hist/RowPrefix/TilePrefix/Scatter)
    ops.def(
        "npu_k2q_csr(Tensor q2k, Tensor cu_seqlens, Tensor cu_block_lens, "
        "int order_method=0, int total_rows=-1, int max_kv=-1, int use_simt=0, "
        "int q_global_offset=0) "
        "-> (Tensor row_ptr, Tensor q_ind, Tensor slot)"
    );
    ops.impl("npu_k2q_csr", torch::kPrivateUse1, &vllm_ascend::npu_k2q_csr);

    ops.def(
        "npu_sparse_attention_score_prefill(Tensor query, Tensor key, Tensor value,"
        "                           Tensor block_table,Tensor k2q_row_ptr,Tensor k2q_q_indices,Tensor k2q_slot_indices,"
        "                           int num_key_value_heads, float scale_value,"
        "                           int block_size, int top_k, int inner_precise, *,"
        "                           Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None"
        "                           ) -> Tensor "
    );
    ops.impl("npu_sparse_attention_score_prefill", torch::kPrivateUse1, &vllm_ascend::npu_sparse_attention_score_prefill);

    ops.def("is_minimax_sparse_attention_split_kv_available() -> bool");
    ops.impl("is_minimax_sparse_attention_split_kv_available", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::is_minimax_sparse_attention_split_kv_available);

    ops.def(
        "npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                           Tensor sparse_indices, float scale_value, *,"
        "                           Tensor? block_table=None, Tensor? actual_seq_lengths_query=None,"
        "                           Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None,"
        "                           Tensor? key_rope=None, int sparse_block_size=1,"
        "                           str layout_query='BSND', str layout_kv='BSND',"
        "                           int sparse_mode=3, int pre_tokens=9223372036854775807,"
        "                           int next_tokens=9223372036854775807, int attention_mode=2,"
        "                           bool return_softmax_lse=False) -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)"
    );
    ops.impl("npu_sparse_flash_attention", torch::kPrivateUse1, &vllm_ascend::npu_sparse_flash_attention);

    ops.def(
        "npu_lightning_indexer_v2_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, *, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
        "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int batch_size=0, int max_seqlen_q=-1, "
        "int max_seqlen_k=-1, str layout_q=\"BSND\", str layout_k=\"BSND\", int mask_mode=0, "
        "int cmp_ratio=1) -> Tensor"
    );
    ops.impl("npu_lightning_indexer_v2_metadata", torch::kPrivateUse1, &vllm_ascend::li_v2::LightningIndexerMetadata);
    ops.def(
        "npu_lightning_indexer_v2(Tensor q, Tensor k, Tensor w, int topk, *, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
        "Tensor? cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, "
        "Tensor? metadata=None, int max_seqlen_q=-1, str layout_q=\"BSND\", str layout_k=\"BSND\", "
        "int mask_mode=0, int cmp_ratio=1, int return_value=0) -> (Tensor, Tensor)"
    );
    ops.impl("npu_lightning_indexer_v2", torch::kPrivateUse1, &vllm_ascend::li_v2::LightningIndexer);

    ops.def(
        "npu_generic_block_sparse_attention_metadata(Tensor sparse_block_idx, Tensor sparse_block_count, "
        "int num_heads_q, int num_heads_kv, int head_dim, int[] block_shape, *, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, "
        "Tensor? seqused_kv=None, int max_seqlen_q=-1, int max_seqlen_kv=-1, "
        "str layout_q='TND', str layout_kv='PA_BBND', int layout_sparse_pattern=4, "
        "int mask_mode=1, int quant_mode=0, int softmax_precision=1, int win_left=-1, "
        "int win_right=-1, int residual_block_mode=0, bool is_consistent_topk=False) -> Tensor"
    );
    ops.impl("npu_generic_block_sparse_attention_metadata", torch::kPrivateUse1,
             &vllm_ascend::npu_generic_block_sparse_attention_metadata);

    ops.def(
        "npu_generic_block_sparse_attention(Tensor q, Tensor k, Tensor v, Tensor sparse_block_idx, "
        "Tensor sparse_block_count, int[] block_shape, *, Tensor? metadata=None, Tensor? attn_mask=None, "
        "Tensor? q_dequant_scale=None, Tensor? k_dequant_scale=None, Tensor? v_dequant_scale=None, "
        "Tensor? p_quant_scale=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? block_table=None, str layout_q='TND', "
        "str layout_kv='PA_BBND', int layout_sparse_pattern=4, float softmax_scale=0.0, int mask_mode=1, "
        "int quant_mode=0, float dst_type_max=0.0, int softmax_precision=1, int win_left=-1, "
        "int win_right=-1, bool return_softmax_lse=False, int residual_block_mode=0, "
        "bool is_consistent_topk=False, ScalarType? attention_out_dtype=None) -> (Tensor, Tensor)"
    );
    ops.impl("npu_generic_block_sparse_attention", torch::kPrivateUse1,
             &vllm_ascend::npu_generic_block_sparse_attention);

    ops.def(
        "npu_quant_lightning_indexer_v2_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, "
        "int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
        "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int batch_size=0, int max_seqlen_q=0, "
        "int max_seqlen_k=0, str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, int cmp_ratio=1, "
        "str device='npu') -> Tensor"
    );
    ops.impl("npu_quant_lightning_indexer_v2_metadata", torch::kPrivateUse1,
             &vllm_ascend::npu_quant_lightning_indexer_v2_metadata_npu);
    ops.def(
        "npu_quant_lightning_indexer_v3(Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, "
        "Tensor? candidate_topk_index=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
        "Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, "
        "int max_seqlen_q=-1, str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, "
        "int cmp_ratio=1, int candidate_mode=3, int candidate_topk_blocks=2048, "
        "int candidate_block_size=8) -> (Tensor, Tensor, Tensor)"
    );
    ops.impl("npu_quant_lightning_indexer_v3", torch::kPrivateUse1,
             &vllm_ascend::qli_v2::QuantLightningIndexerCandidate);
    ops.impl("npu_quant_lightning_indexer_v3", torch::kMeta,
             &vllm_ascend::qli_v2::QuantLightningIndexerCandidate);

    ops.def(
        "npu_sparse_flash_mla_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
        "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_ori_kv=0, int max_seqlen_cmp_kv=0, "
        "int ori_topk=0, int cmp_topk=0, int cmp_ratio=0, int ori_mask_mode=0, int cmp_mask_mode=0, "
        "int ori_win_left=-1, int ori_win_right=-1, str layout_q='BSND', str layout_kv='BSND', "
        "bool has_ori_kv=True, bool has_cmp_kv=True) -> Tensor"
    );
    ops.impl("npu_sparse_flash_mla_metadata", torch::kPrivateUse1,
             &vllm_ascend::npu_sparse_flash_mla_metadata);

    ops.def(
        "npu_sparse_flash_mla(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, "
        "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
        "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
        "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
        "Tensor? sinks=None, Tensor? metadata=None, float softmax_scale=1.0, int cmp_ratio=0, "
        "int ori_mask_mode=0, int cmp_mask_mode=0, int ori_win_left=-1, int ori_win_right=-1, "
        "str layout_q='BSND', str layout_kv='BSND', int topk_value_mode=1, "
        "bool return_softmax_lse=False) -> (Tensor, Tensor)"
    );
    ops.impl("npu_sparse_flash_mla", torch::kPrivateUse1,
             &vllm_ascend::npu_sparse_flash_mla);

    ops.def(
        "npu_kv_quant_sparse_flash_attention_vllm(Tensor query, Tensor key, Tensor value,"
        "                                    Tensor sparse_indices, float scale_value, *,"
        "                                    int key_quant_mode=1, int value_quant_mode=1,"
        "                                    Tensor? key_dequant_scale=None,"
        "                                    Tensor? value_dequant_scale=None,"
        "                                    Tensor? block_table=None,"
        "                                    Tensor? actual_seq_lengths_query=None,"
        "                                    Tensor? actual_seq_lengths_kv=None,"
        "                                    int sparse_block_size=1,"
        "                                    str layout_query='BSND', str layout_kv='BSND',"
        "                                    int sparse_mode=3,"
        "                                    int pre_tokens=9223372036854775807,"
        "                                    int next_tokens=9223372036854775807,"
        "                                    int attention_mode=2,"
        "                                    int quant_scale_repo_mode=1,"
        "                                    int tile_size=128,"
        "                                    int rope_head_dim=64,"
        "                                    bool return_softmax_lse=False)"
        " -> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)"
    );
    ops.impl("npu_kv_quant_sparse_flash_attention_vllm", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_flash_attention_vllm);

    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,"
        "                     Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, Tensor[] bias2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, float swiglu_limit=1000000.0, int world_size=0) -> (Tensor out, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_ffn_combine", torch::kPrivateUse1, &vllm_ascend::dispatch_ffn_combine);

    // vLLM-Ascend custom ops
    ops.def(
        "moe_gating_top_k(Tensor x, "
                            "int k, "
                            "int k_group, "
                            "int group_count, "
                            "int group_select_mode, "
                            "int renorm, "
                            "int norm_type, "
                            "bool out_flag, "
                            "float routed_scaling_factor, "
                            "float eps,"
                            "Tensor? bias_opt=None)"

        "-> (Tensor y ,Tensor expert_idx, Tensor out)"
        );
    ops.impl("moe_gating_top_k", torch::kPrivateUse1,&vllm_ascend::moe_gating_top_k);

    ops.def(
        "npu_add_rms_norm_bias(Tensor x1, "
                            "Tensor x2, "
                            "Tensor gamma, "
                            "Tensor? beta=None, "
                            "float epsilon=1e-6)"
        "-> (Tensor y ,Tensor rstd, Tensor x)"
        );
    ops.impl("npu_add_rms_norm_bias", torch::kPrivateUse1, &vllm_ascend::npu_add_rms_norm_bias);

    ops.def(
        "npu_rms_norm_cast(Tensor x, Tensor gamma, float epsilon=1e-6)"
        " -> (Tensor y, Tensor y_fp32)"
    );
    ops.impl("npu_rms_norm_cast", torch::kPrivateUse1, &vllm_ascend::npu_rms_norm_cast);

    ops.def("npu_sign_bits_pack(Tensor input, int size) -> Tensor");
    ops.impl("npu_sign_bits_pack", torch::kPrivateUse1, &vllm_ascend::npu_sign_bits_pack);

    ops.def(
        "transpose_kv_cache_by_block(Tensor[] kCache, Tensor[] vCache, Tensor blockIDs, int blockSize, int headNum, int headDim, int splitNum, int layerNum) -> ()"
    );
    ops.impl("transpose_kv_cache_by_block", torch::kPrivateUse1, &vllm_ascend::transpose_kv_cache_by_block);

    ops.def(
        "npu_scatter_pa_kv_cache(Tensor key, Tensor value, "
        "Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor slot_mapping, *, "
        "str cache_mode='Norm', str scatter_mode='None') -> ()"
    );
    ops.impl("npu_scatter_pa_kv_cache", torch::kPrivateUse1,
             &vllm_ascend::npu_scatter_pa_kv_cache);

    ops.def(
        "npu_copy_and_expand_eagle_inputs(Tensor target_token_ids, Tensor target_positions, "
        "Tensor next_token_ids, Tensor query_start_loc, Tensor query_end_loc, "
        "int padding_token_id, int parallel_drafting_token_id, int num_padding_slots_per_request, "
        "bool shift_input_ids, int total_draft_tokens) -> "
        "(Tensor out_input_ids, Tensor out_positions, Tensor out_is_rejected_token_mask, "
        "Tensor out_is_masked_token_mask, Tensor out_new_token_indices, Tensor out_hidden_state_mapping)"
    );
    ops.impl("npu_copy_and_expand_eagle_inputs", torch::kPrivateUse1, &vllm_ascend::npu_copy_and_expand_eagle_inputs);
    ops.def(
        "moe_gating_top_k_hash("
        "Tensor x, "
        "int k, "
        "Tensor? bias=None, "
        "Tensor? input_ids=None, "
        "Tensor? tid2eid=None, "
        "int k_group=1, "
        "int group_count=1, "
        "float routed_scaling_factor=1.0, "
        "float eps=1e-20, "
        "int group_select_mode=0, "
        "int renorm=0, "
        "int norm_type=0, "
        "bool out_flag=False, "
        "Tensor? bias_vl=None, "
        "int image_sentinel_lo=129257, "
        "int image_sentinel_count=5"
        ") -> (Tensor y, Tensor expert_idx, Tensor out)"
        );
    ops.impl("moe_gating_top_k_hash", torch::kPrivateUse1,&vllm_ascend::moe_gating_top_k_hash);

    ops.def(
        "compressor("
            "Tensor x, Tensor wkv, Tensor wgate, "
            "Tensor(a!) state_cache, Tensor ape, Tensor norm_weight, "
            "Tensor rope_sin, Tensor rope_cos, "
            "Tensor? state_block_table, Tensor? cu_seqlens, "
            "Tensor? seqused, Tensor? start_pos, "
            "int rope_head_dim, int cmp_ratio, int coff, "
            "float norm_eps, int rotary_mode, int cache_mode"
        ") -> Tensor"
        );
    ops.impl("compressor", torch::kPrivateUse1, &vllm_ascend::compressor);

    ops.def(
        "compressor_metadata("
            "Tensor rope_cos, Tensor rope_sin, "
            "Tensor cu_seqlens, Tensor start_pos, Tensor kv_block_table, "
            "int kv_block_size, int slot_mapping_format, int compress_ratio, int num_compressed_tokens, "
            "int num_reqs_actual"
        ") -> (Tensor, Tensor, Tensor)"
        );
    ops.impl("compressor_metadata", torch::kPrivateUse1, &vllm_ascend::compressor_metadata);

    ops.def(
        "compressor_metadata_out("
            "Tensor rope_cos, Tensor rope_sin, "
            "Tensor cu_seqlens, Tensor start_pos, Tensor kv_block_table, "
            "int kv_block_size, int slot_mapping_format, int compress_ratio, int num_reqs_actual, "
            "Tensor(a!) compress_cos, Tensor(b!) compress_sin, Tensor(c!) slot_mapping"
        ") -> (Tensor(a!), Tensor(b!), Tensor(c!))"
        );
    ops.impl("compressor_metadata_out", torch::kPrivateUse1, &vllm_ascend::compressor_metadata_out);

    ops.def(
        "npu_quant_lightning_indexer_v2("
            "Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, "
            "int topk, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_k=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, "
            "Tensor? cmp_residual_k=None, "
            "Tensor? block_table=None, "
            "Tensor? output_idx_offset=None, "
            "Tensor? metadata=None, "
            "int max_seqlen_q=-1, "
            "str layout_q=\"TND\", str layout_k=\"PA_BBND\", "
            "int mask_mode=3, int cmp_ratio=4, int return_value=0"
        ") -> (Tensor sparse_indices, Tensor sparse_values)"
        );
    ops.impl("npu_quant_lightning_indexer_v2", torch::kPrivateUse1,
             &vllm_ascend::npu_quant_lightning_indexer_v2_compat_npu);

    ops.def(
        "npu_sparse_attn_sharedkv("
            "Tensor q, *, "
            "Tensor? ori_kv=None, "
            "Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, "
            "Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, "
            "Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "Tensor? sinks=None, "
            "Tensor? metadata=None, "
            "float softmax_scale=0, "
            "int cmp_ratio=0, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=128, "
            "int ori_win_right=0, "
            "str layout_q=\"BSND\", "
            "str layout_kv=\"PA_ND\", "
            "bool return_softmax_lse=False"
        ") -> (Tensor out, Tensor softmax_lse)"
        );
    ops.impl("npu_sparse_attn_sharedkv", torch::kPrivateUse1, &vllm_ascend::npu_sparse_attn_sharedkv_npu);

    ops.def(
        "npu_sparse_attn_sharedkv_metadata("
            "int num_heads_q, "
            "int num_heads_kv, "
            "int head_dim, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "int batch_size=0, "
            "int max_seqlen_q=0, "
            "int max_seqlen_kv=0, "
            "int ori_topk=0, "
            "int cmp_topk=0, "
            "int cmp_ratio=4, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=128, "
            "int ori_win_right=0, "
            "str layout_q=\"BSND\", "
            "str layout_kv=\"PA_ND\", "
            "bool has_ori_kv=True, "
            "bool has_cmp_kv=True, "
            "str device=\"npu\""
        ") -> (Tensor metadata)"
        );
    ops.impl("npu_sparse_attn_sharedkv_metadata", torch::kPrivateUse1, &vllm_ascend::npu_sparse_attn_sharedkv_metadata_npu);

    ops.def(
          "npu_hc_post("
            "Tensor x, "
            "Tensor residual, "
            "Tensor post, "
            "Tensor comb"
        ") -> (Tensor out)"
        );
    ops.impl("npu_hc_post", torch::kPrivateUse1, &vllm_ascend::npu_hc_post_npu);

    ops.def(
        "npu_hc_pre_v2("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor y, Tensor post, Tensor comb_frag)"
        );
    ops.impl("npu_hc_pre_v2", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v2_npu);

    ops.def(
        "npu_hc_pre_v3("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, Tensor? pre_mix=None, *, "
            "int hc_mult=4, int hc_sinkhorn_iters=20, "
            "float norm_eps=1e-6, float hc_eps=1e-6"
        ") -> (Tensor y, Tensor post, Tensor comb_frag, Tensor pre)"
        );
    ops.impl("npu_hc_pre_v3", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v3_npu);

    ops.def(
        "inplace_partial_rotary_mul("
            "Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice, bool negate_sin=False"
        ") -> ()"
    );
    ops.impl("inplace_partial_rotary_mul", torch::kPrivateUse1, &vllm_ascend::inplace_partial_rotary_mul_npu);

    ops.def(
        "npu_rms_norm_dynamic_quant("
            "Tensor x, "
            "Tensor gamma, "
            "Tensor? smooth_scale=None, "
            "Tensor? beta=None, "
            "float epsilon=1e-6"
        ") -> (Tensor y_out, Tensor scale_out)"
        );
    ops.impl("npu_rms_norm_dynamic_quant", torch::kPrivateUse1, &vllm_ascend::npu_rms_norm_dynamic_quant_npu);

    ops.def(
        "kv_compress_epilog_v2(Tensor(a!) cache, Tensor x, Tensor slot_mapping, *, "
        "int quant_group_size=32, str quant_mode='mxfp8_bf16', "
        "bool round_scale=True, float x_scale=1.0) -> ()"
    );
    ops.impl("kv_compress_epilog_v2", torch::kPrivateUse1,
             &vllm_ascend::kv_compress_v2::KvCompressEpilogV2Npu);

    ops.def(
        "npu_kv_quant_sparse_attn_sharedkv("
            "Tensor q, "
            "int kv_quant_mode, "
            "Tensor? ori_kv=None, "
            "Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, "
            "Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, "
            "Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "Tensor? sinks=None, "
            "Tensor? metadata=None, "
            "int tile_size=0, "
            "int rope_head_dim=0, "
            "float softmax_scale=0.0, "
            "int cmp_ratio=0, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=127, "
            "int ori_win_right=0, "
            "str layout_q='BSND', "
            "str layout_kv='PA_ND', "
            "bool return_softmax_lse=False"
        ") -> (Tensor out, Tensor softmax_lse)"
    );
    ops.impl("npu_kv_quant_sparse_attn_sharedkv", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_attn_sharedkv_npu);

    ops.def(
        "npu_kv_quant_sparse_attn_sharedkv_metadata("
            "int num_heads_q, "
            "int num_heads_kv, "
            "int head_dim, "
            "int kv_quant_mode, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "int batch_size=0, "
            "int max_seqlen_q=0, "
            "int max_seqlen_kv=0, "
            "int ori_topk=0, "
            "int cmp_topk=0, "
            "int tile_size=0, "
            "int rope_head_dim=0, "
            "int cmp_ratio=-1, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=127, "
            "int ori_win_right=0, "
            "str layout_q='BSND', "
            "str layout_kv='PA_ND', "
            "bool has_ori_kv=True, "
            "bool has_cmp_kv=True, "
            "str device='npu'"
        ") -> Tensor"
    );
    ops.impl("npu_kv_quant_sparse_attn_sharedkv_metadata", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_attn_sharedkv_metadata_npu);

    ops.def(
        "npu_swiglu_group_quant(Tensor x, Tensor? topk_weight, Tensor? group_index, "
        "                       ScalarType dst_type=39, "
        "                       int quant_mode=1, int group_size=128, "
        "                       bool round_scale=False, bool ue8m0_scale=False, "
        "                       bool output_origin=False, int group_list_type=0, "
        "                       float clamp_value=0.0) "
        "-> (Tensor y, Tensor scale, Tensor y_origin)");
    ops.impl("npu_swiglu_group_quant", torch::kPrivateUse1, &vllm_ascend::npu_swiglu_group_quant_npu);

    ops.def(
        "indexer_compress_epilog_v2("
            "Tensor(a!) indexer_compress_cache, "
            "Tensor x, "
            "Tensor slot_mapping, "
            "int layout=2"
        ") -> ()"
    );
    ops.impl("indexer_compress_epilog_v2", torch::kPrivateUse1,
             &vllm_ascend::indexer_compress_epilog_v2_npu);

    ops.def(
        "npu_dequant_swiglu_quant("
            "Tensor x, *, "
            "Tensor? weight_scale=None, "
            "Tensor? activation_scale=None, "
            "Tensor? bias=None, "
            "Tensor? quant_scale=None, "
            "Tensor? quant_offset=None, "
            "Tensor? group_index=None, "
            "bool activate_left=True, "
            "int quant_mode=0, "
            "int swiglu_mode=0, "
            "float clamp_limit=0.0, "
            "float glu_alpha=1.0, "
            "float glu_bias=0.0"
        ") -> (Tensor y, Tensor scale)"
    );
    ops.impl("npu_dequant_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::npu_dequant_swiglu_quant);

    ops.def(
        "npu_scatter_nd_update_sk("
                "Tensor(a!) var, Tensor indices, Tensor update"
            ") -> ()"
    );
    ops.impl("npu_scatter_nd_update_sk", torch::kPrivateUse1, &vllm_ascend::npu_scatter_nd_update_sk);

    ops.def(
        "chunk_gated_delta_rule_fwd_h_vllm(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h_vllm", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h_vllm);

    ops.def(
        "chunk_fwd_o_vllm(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o_vllm", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o_vllm);

    ops.def(
        "kda_gate_cumsum_vllm(Tensor g, int chunk_size, *, Tensor? A_log=None, Tensor? dt_bias=None, int[]? cu_seqlens=None, bool? use_gate_in_kernel=False, bool? safe_gate=False, float? lower_bound=-5.0, str layout=\"BSND\") -> Tensor"
    );
    ops.impl("kda_gate_cumsum_vllm", torch::kPrivateUse1, &vllm_ascend::kda_gate_cumsum_vllm);

    ops.def(
        "kda_layout_swap12(Tensor x, *, Tensor? dependency=None) -> Tensor"
    );
    ops.impl("kda_layout_swap12", torch::kPrivateUse1, &vllm_ascend::kda_layout_swap12);

    ops.def(
        "npu_sparse_attention_score("
        "Tensor query, Tensor key, Tensor value, Tensor select_idx, "
        "Tensor block_table, *, "
        "Tensor? select_num_idx=None, "
        "Tensor? q_dequant_scale=None, Tensor? k_dequant_scale=None, "
        "Tensor? v_dequant_scale=None, "
        "Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, "
        "str q_input_layout=\"TND\", str kv_input_layout=\"BNSD\", "
        "int num_key_value_heads=1, float scale_value=1.0, "
        "int block_size=128, int top_k=16, int inner_precise=0, "
        "ScalarType? attention_out_dtype=None) -> Tensor"
    );
    ops.impl("npu_sparse_attention_score", torch::kPrivateUse1,
             &vllm_ascend::npu_sparse_attention_score);

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
    ops.def(
        "npu_fused_lightning_indexer_manage("
        "Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
        "Tensor index_key_dequant_scale, Tensor index_key_cache, "
        "Tensor index_block_table, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key, "
        "Tensor num_cache_tokens, Tensor request_state, Tensor req_pool_entries, "
        "Tensor(a!) cache_slots_pool, Tensor(b!) topk_src_ids, "
        "Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, "
        "Tensor(e!) miss_src_ids, Tensor(f!) miss_dst_slots, "
        "Tensor(g!) miss_counts) -> ()"
    );
    ops.impl("npu_fused_lightning_indexer_manage", torch::kPrivateUse1,
             &vllm_ascend::npu_fused_lightning_indexer_manage);

#endif

#if defined(VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD) || defined(VLLM_ASCEND_ENABLE_SELECTIVE_LIM)
    ops.def("npu_precomputed_topk_manage(Tensor topk, Tensor layer_descriptors, int host_blocks, "
            "int block_size, int hot, int row_stride, int k_bytes, int v_bytes) -> ()");
    ops.impl("npu_precomputed_topk_manage", torch::kPrivateUse1, &vllm_ascend::npu_precomputed_topk_manage);

#endif

    ops.def(
        "npu_fused_scatter_copy_sparse_flash_attention("
        "Tensor query_rope, Tensor query, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_kv, Tensor num_cache_tokens, "
        "Tensor topk_dst_slots, Tensor topk_src_ids, Tensor topk_miss_counts, "
        "Tensor miss_src_ids, Tensor miss_dst_slots, Tensor miss_counts, "
        "Tensor hbm_block_table, Tensor dram_block_table, "
        "Tensor(a!) hbm_k_rope, Tensor(b!) hbm_kv_cache, "
        "Tensor dram_k_rope, Tensor dram_kv_cache, float scale_value, "
        "Tensor(c!) attention_out) -> ()"
    );
    ops.impl("npu_fused_scatter_copy_sparse_flash_attention", torch::kPrivateUse1,
             &vllm_ascend::npu_fused_scatter_copy_sparse_flash_attention);

    ops.def(
        "npu_fused_sparse_attention_overlap(Tensor query, Tensor(a!) selection_k_rope, "
        "Tensor(b!) selection_kv_cache, Tensor(c!) selection_kv_block_table, "
        "Tensor(d!) selection_kv_block_status, Tensor(e!) selection_membership_map, "
        "Tensor selection_topk_indices, "
        "Tensor full_k_rope, Tensor full_kv_cache, Tensor full_kv_block_table, "
        "Tensor full_kv_actual_seq, Tensor full_q_actual_seq, "
        "float scale_value, int sparse_block_size, int selection_topk_block_size, *, "
        "str layout_query='TND', str layout_kv='PA_BSND', int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_fused_sparse_attention_overlap", torch::kPrivateUse1,
             &vllm_ascend::npu_fused_sparse_attention_overlap);
}
#endif
