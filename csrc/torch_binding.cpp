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
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "ops.h"
#include "utils.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "mla_preprocess/mla_preprocess_torch_adpt.h"
#endif
#include "mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h"
#include "attention/lightning_indexer/lightning_indexer_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "attention/kv_quant_sparse_flash_attention/kv_quant_sparse_flash_attention_torch_adpt.h"
#include "attention/fused_sparse_attention_overlap/fused_sparse_attention_overlap_torch_adpt.h"
#include "attention/lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
#include "moe/causal_conv1d_v310/causal_conv1d_310_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "attention/recurrent_kda/recurrent_kda_torch_adpt.h"
#include "attention/chunk_kda_fwd/chunk_kda_fwd_torch_adpt.h"
#include "attention/kda_gate_cumsum/kda_gate_cumsum_torch_adpt.h"
#include "attention/kda_layout_swap12/kda_layout_swap12_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule_v310/recurrent_gated_delta_rule_310_torch_adpt.h"
#include "attention/k2q_csr/k2q_csr_torch_adpt.h"
#include "attention/msa_index_score/msa_index_score_torch_adpt.h"
#include "attention/sparse_attention_score/sparse_attention_score_torch_adpt.h"
#include "attention/store_kv_block/store_kv_block_torch_adpt.h"
#include "attention/store_kv_block_metadata/store_kv_block_metadata_torch_adpt.cpp"
#include "moe/dequant_situ_quant/dequant_situ_quant_torch_adpt.h"
#include "moe/situ_mx_quant/situ_mx_quant_torch_adpt.h"
#include "attention/mla_prolog_v3/mla_prolog_v3_torch_adpt.h"
#include <c10/core/Device.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
#include <omp.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace vllm_ascend {

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
        aclrtGetDevice(&device_id);

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

    EXEC_NPU_CMD(aclnnTransposeKvCacheByBlock, kCache, vCache, blockIDs,
                 blockSize, headNum, headDim, splitNum, layerNum);

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

at::Tensor npu_causal_conv1d_custom(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& query_start_loc_opt,
    const c10::optional<at::Tensor>& cache_indices_opt,
    const c10::optional<at::Tensor>& initial_state_mode_opt,
    const c10::optional<at::Tensor>& num_accepted_tokens_opt,
    int64_t  activation_mode,
    int64_t  pad_slot_id,
    int64_t  run_mode)
{
    EXEC_NPU_CMD(aclnnCausalConv1d,
                    x,
                    weight,
                    bias_opt,
                    conv_state,
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode,
                    output
                );

    return output;
}

// It is expected that further improvements will be made after it is incorporated into CANN on June 30th.
std::vector<at::Tensor> moe_grouped_matmul(
    at::Tensor x,
    at::Tensor weight,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type
)
{
    bool transpose_weight = false;
    bool weight_nz = true;

    at::TensorList x_list = at::TensorList(x);
    at::TensorList weight_list = at::TensorList(weight);
    std::vector<at::Tensor> y;
    c10::TensorOptions options = x_list[0].options().dtype(x[0].scalar_type());
    auto m = x_list[0].sizes()[0];
    auto n = weight_list[0].sizes()[1];
    if (!transpose_weight) {
        n = weight_list[0].sizes()[2];
    }
    at::Tensor y_0 = at::empty(at::IntArrayRef{m, n}, options);
    y.emplace_back(y_0);
    at::TensorList result = at::TensorList(y);

    EXEC_NPU_CMD(aclnnMoeGroupedMatmulWeightNz,
                x_list, weight_list, group_list, transpose_weight, result);

    return y;
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
    bool out_flag)
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

    const at::Tensor& bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    const at::Tensor& input_ids = c10::value_or_else(input_ids_opt, [] { return at::Tensor(); });
    const at::Tensor& tid2eid = c10::value_or_else(tid2eid_opt, [] { return at::Tensor(); });

    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnMoeGatingTopKHash,
                 x,
                 bias,
                 input_ids,
                 tid2eid,
                 k,
                 k_group,
                 group_count,
                 routed_scaling_factor,
                 eps,
                 group_select_mode,
                 renorm,
                 norm_type,
                 out_flag,
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

std::tuple<at::Tensor, at::Tensor> construct_quant_lightning_indexer_output_tensor(const at::Tensor& query, const at::Tensor& key,
                                                           int64_t sparse_count, std::string query_layout_str,
                                                           std::string key_layout_str, bool return_value)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
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
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor sparse_indices_out = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = at::empty(output_size, query.options().dtype(at::kFloat));
    } else {
        sparse_values_out = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_vllm_quant_lightning_indexer_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &metadata,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_value)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer_output = construct_quant_lightning_indexer_output_tensor(
            query, key, sparse_count, query_layout_str, key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(quant_lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(quant_lightning_indexer_output);
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());
    int64_t stride = key.stride(0);
    int64_t scale_stride = key_dequant_scale.stride(0);

    if (key_layout_str == "PA_BSND") {
        auto contiguous_axes_result_key = is_contiguous_axes(key);
        TORCH_CHECK(contiguous_axes_result_key[1] && contiguous_axes_result_key[2],
                    "key must be contiguous on all axes except axis 0");
        auto contiguous_axes_result_key_scale = is_contiguous_axes(key_dequant_scale);
        TORCH_CHECK(contiguous_axes_result_key_scale[1] && contiguous_axes_result_key_scale[2],
                    "key_dequant_scale must be contiguous on all axes except axis 0");
    }

    EXEC_NPU_CMD(aclnnVllmQuantLightningIndexer, query,
        key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key,
        block_table, metadata, query_quant_mode, key_quant_mode, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
        pre_tokens, next_tokens, cmp_ratio, return_value, stride, scale_stride, sparse_indices_out, sparse_values_out);


    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
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

at::Tensor npu_vllm_quant_lightning_indexer_metadata_npu(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim, int64_t query_quant_mode, int64_t key_quant_mode,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_key, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, const c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, const c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (actual_seq_lengths_query.has_value()) {
        output_device = actual_seq_lengths_query.value().device();
    } else if (actual_seq_lengths_key.has_value()) {
        output_device = actual_seq_lengths_key.value().device();
    }

    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    auto actual_seq_lengths_query_val = get_valid_tensor(actual_seq_lengths_query, output_device);
    auto actual_seq_lengths_key_val = get_valid_tensor(actual_seq_lengths_key, output_device);

    std::string layout_query_str = std::string(layout_query);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    std::string layout_key_str = std::string(layout_key);
    char *layout_key_ptr = const_cast<char *>(layout_key_str.c_str());

    EXEC_NPU_CMD(aclnnVllmQuantLightningIndexerMetadata, actual_seq_lengths_query_val, actual_seq_lengths_key_val,
                    num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, batch_size,
                    max_seqlen_q, max_seqlen_k, layout_query_ptr, layout_key_ptr, sparse_count,
                    sparse_mode, pre_tokens, next_tokens, cmp_ratio, output);

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
constexpr int64_t HC_PRE_D_LIMIT = 4096;
constexpr int64_t HC_PRE_D_LIMIT_EXTEND = 7168;
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

void check_hc_pre_shape_and_dtype(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
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
    TORCH_CHECK(d == HC_PRE_D_LIMIT || d == HC_PRE_D_LIMIT_EXTEND, "The d of x only supports ", HC_PRE_D_LIMIT,
                " or ", HC_PRE_D_LIMIT_EXTEND, ", actual ", d, ".");
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
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_hc_pre_fusion(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    EXEC_NPU_CMD(aclnnHcPre, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps,
                 y, post, comb_frag);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

void inplace_partial_rotary_mul_npu(at::Tensor & x, const at::Tensor &r1, const at::Tensor &r2, c10::string_view rotary_mode, at::IntArrayRef partial_slice)
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
    EXEC_NPU_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice);
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

void validate_kv_compress_epilog_inputs(
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_compress_cache)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D tensor, but got dimensions: ", x.dim());
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
                "x dimensions must be positive, but got: [", x.size(0), ", ", x.size(1), "]");
    TORCH_CHECK(slot_mapping.dim() == 1,
                "slot_mapping must be 1D tensor, but got dimensions: ", slot_mapping.dim());
    TORCH_CHECK(slot_mapping.size(0) == x.size(0),
                "slot_mapping size must equal x's first dimension, but got slot_mapping_size=",
                slot_mapping.size(0), ", x.dim(0)=", x.size(0));
    if (kv_compress_cache.dim() == 4) {
        TORCH_CHECK(kv_compress_cache.size(2) == 1,
                    "kv_compress_cache 4D tensor requires headnum (dim 2) == 1, but got ",
                    kv_compress_cache.size(2));
    }
    TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be BF16, but got ", x.dtype());
    TORCH_CHECK(slot_mapping.dtype() == at::kInt || slot_mapping.dtype() == at::kLong,
                "slot_mapping must be INT32 or INT64, but got ", slot_mapping.dtype());
    TORCH_CHECK(kv_compress_cache.dtype() == at::ScalarType::Float8_e5m2 ||
                kv_compress_cache.dtype() == at::ScalarType::Float8_e4m3fn,
                "kv_compress_cache must be FP8_E5M2 or FP8_E4M3, but got ", kv_compress_cache.dtype());
}

void kv_compress_epilog_npu(
    at::Tensor& kv_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_group_size,
    int64_t quant_mode,
    bool round_scale_flag,
    int64_t layout)
{
    validate_kv_compress_epilog_inputs(x, slot_mapping, kv_compress_cache);

    at::Tensor cache = kv_compress_cache;
    if (cache.dim() == 4) {
        cache = cache.squeeze(2);
    }

    int64_t round_scale = round_scale_flag ? 1 : 0;
    int64_t cache_stride = cache.stride(0);
    EXEC_NPU_CMD(aclnnKvCompressEpilog, cache, x, slot_mapping, quant_group_size, quant_mode, round_scale,
                 layout, cache_stride);
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

void npu_scatter_nd_update_v2(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    // construct the output tensor
    at::IntArrayRef var_stride = var.strides();
    EXEC_NPU_CMD(aclnnScatterNdUpdateV2, var, indices, update, var_stride);
    return;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h(
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
        aclnnChunkGatedDeltaRuleFwdH,
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

at::Tensor chunk_fwd_o(
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
        aclnnChunkFwdO,
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

std::vector<int64_t> get_npu_storage_shape(const at::Tensor& tensor)
{
    TORCH_CHECK(
        tensor.is_privateuseone(),
        "get_npu_storage_shape only supports NPU tensors, but got device ",
        tensor.device());
    const auto& desc = NPUBridge::GetNpuStorageImplDesc(tensor);
    return std::vector<int64_t>(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
}

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
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

HOT_FUNCTION void lru_resident_compact(int64_t req_ids_ptr, int64_t last_req_ids_ptr, int64_t topk_indices_ptr,
                                       int64_t stable_prefix_lens_ptr, int64_t slot_to_token_ptr,
                                       int64_t lru_slots_ptr, int64_t current_slots_ptr, int64_t miss_count_ptr,
                                       int64_t miss_tokens_ptr, int64_t miss_slots_ptr,
                                       int64_t token_mark_workspace_ptr, int64_t token_pos_workspace_ptr,
                                       int64_t slot_workspace_ptr, int64_t miss_position_workspace_ptr,
                                       int64_t epochs_ptr, int64_t num_reqs, int64_t topk, int64_t capacity,
                                       int64_t max_token, int64_t workspace_threads, int64_t requested_threads) {
  lru_resident_compact_impl(req_ids_ptr, last_req_ids_ptr, topk_indices_ptr, stable_prefix_lens_ptr, slot_to_token_ptr,
                            lru_slots_ptr, current_slots_ptr, miss_count_ptr, miss_tokens_ptr, miss_slots_ptr,
                            token_mark_workspace_ptr, token_pos_workspace_ptr, slot_workspace_ptr,
                            miss_position_workspace_ptr, epochs_ptr, num_reqs, topk, capacity, max_token,
                            workspace_threads, requested_threads, 0, 0, 0, num_reqs, 0);
}

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

int64_t compute_lru_resident_addrs(const at::Tensor& miss_count, const at::Tensor& miss_tokens,
                                   const at::Tensor& miss_slots, const at::Tensor& block_table,
                                   const int64_t block_size, const int64_t token_size_bytes_k,
                                   const int64_t token_size_bytes_v, const int64_t gvas_k_base,
                                   const int64_t gvas_v_base, const int64_t addr_k_base, const int64_t addr_v_base,
                                   const int64_t resident_capacity, const int64_t max_num_threads,
                                   at::Tensor& gvas_buffer, at::Tensor& addr_buffer, at::Tensor& size_buffer,
                                   at::Tensor& num_tokens_buffer) {
  const int32_t num_reqs = miss_tokens.size(0);
  const int32_t topk = miss_tokens.size(1);
  const int32_t max_num_tokens_to_load = num_reqs * topk;
  const int32_t max_num_blocks = block_table.size(1);
  const int64_t block_size_bytes_k = block_size * token_size_bytes_k;
  const int64_t block_size_bytes_v = block_size * token_size_bytes_v;
  TORCH_CHECK(miss_count.size(0) >= num_reqs, "miss_count size not enough for loading.");
  TORCH_CHECK(miss_slots.size(0) == num_reqs && miss_slots.size(1) == topk,
              "miss_slots shape should match miss_tokens.");
  TORCH_CHECK(gvas_buffer.size(0) >= max_num_tokens_to_load * 2, "gvas_buffer size not enough for loading.");
  TORCH_CHECK(addr_buffer.size(0) >= max_num_tokens_to_load * 2, "addr_buffer size not enough for loading.");
  TORCH_CHECK(size_buffer.size(0) >= max_num_tokens_to_load * 2, "size_buffer size not enough for loading.");
  TORCH_CHECK(miss_count.scalar_type() == at::kInt, "miss_count wrong dtype, should be int32.");
  TORCH_CHECK(miss_tokens.scalar_type() == at::kInt, "miss_tokens wrong dtype, should be int32.");
  TORCH_CHECK(miss_slots.scalar_type() == at::kInt, "miss_slots wrong dtype, should be int32.");
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table wrong dtype, should be int32.");
  TORCH_CHECK(gvas_buffer.scalar_type() == at::kLong, "gvas_buffer wrong dtype, should be int64.");
  TORCH_CHECK(addr_buffer.scalar_type() == at::kLong, "addr_buffer wrong dtype, should be int64.");
  TORCH_CHECK(size_buffer.scalar_type() == at::kInt, "size_buffer wrong dtype, should be int32.");

  const int32_t* miss_count_ptr = static_cast<int32_t*>(miss_count.data_ptr());
  const int32_t* miss_tokens_ptr = static_cast<int32_t*>(miss_tokens.data_ptr());
  const int32_t* miss_slots_ptr = static_cast<int32_t*>(miss_slots.data_ptr());
  const int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());
  int64_t* gvas_buffer_ptr = static_cast<int64_t*>(gvas_buffer.data_ptr());
  int64_t* addr_buffer_ptr = static_cast<int64_t*>(addr_buffer.data_ptr());
  int32_t* size_buffer_ptr = static_cast<int32_t*>(size_buffer.data_ptr());
  int32_t* num_tokens_buffer_ptr = static_cast<int32_t*>(num_tokens_buffer.data_ptr());

  int n_threads = static_cast<int>(std::min(static_cast<int64_t>(num_reqs), max_num_threads));
  n_threads = std::max(n_threads, 1);

  std::vector<int32_t> num_tokens_to_load_req(num_reqs);
#pragma omp parallel for num_threads(n_threads)
  for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int32_t count = miss_count_ptr[req_idx];
    count = std::max(count, 0);
    count = std::min(count, topk);
    num_tokens_to_load_req[req_idx] = count;
  }
  int32_t num_tokens_to_load_sum = std::accumulate(num_tokens_to_load_req.begin(), num_tokens_to_load_req.end(), 0);
  std::vector<int32_t> req_start_locs(num_reqs);
  int32_t cumsum = 0;
  for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    req_start_locs[req_idx] = cumsum;
    cumsum += num_tokens_to_load_req[req_idx];
  }

#pragma omp parallel for num_threads(n_threads)
  for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int32_t req_start_loc_k = req_start_locs[req_idx];
    int32_t req_start_loc_v = num_tokens_to_load_sum + req_start_loc_k;
    for (int32_t idx = 0; idx < num_tokens_to_load_req[req_idx]; ++idx) {
      const int32_t token = miss_tokens_ptr[req_idx * topk + idx];
      const int32_t slot = miss_slots_ptr[req_idx * topk + idx];
      if (token < 0 || slot < 0 || slot >= resident_capacity) {
        continue;
      }
      const int32_t block_id = token / static_cast<int32_t>(block_size);
      if (block_id < 0 || block_id >= max_num_blocks) {
        continue;
      }
      const int32_t offset_in_block = token % static_cast<int32_t>(block_size);
      const int32_t block_indice = block_table_ptr[req_idx * max_num_blocks + block_id];
      const int64_t gvas_k = gvas_k_base + block_indice * block_size_bytes_k + offset_in_block * token_size_bytes_k;
      const int64_t gvas_v = gvas_v_base + block_indice * block_size_bytes_v + offset_in_block * token_size_bytes_v;
      const int64_t addr_k = addr_k_base + (req_idx * resident_capacity + slot) * token_size_bytes_k;
      const int64_t addr_v = addr_v_base + (req_idx * resident_capacity + slot) * token_size_bytes_v;
      gvas_buffer_ptr[req_start_loc_k + idx] = gvas_k;
      gvas_buffer_ptr[req_start_loc_v + idx] = gvas_v;
      addr_buffer_ptr[req_start_loc_k + idx] = addr_k;
      addr_buffer_ptr[req_start_loc_v + idx] = addr_v;
    }
  }

  std::fill_n(size_buffer_ptr, num_tokens_to_load_sum, token_size_bytes_k);
  std::fill_n(&(size_buffer_ptr[num_tokens_to_load_sum]), num_tokens_to_load_sum, token_size_bytes_v);

  num_tokens_buffer_ptr[0] = num_tokens_to_load_sum * 2;
  return num_tokens_to_load_sum;
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

struct LruResidentCompactGraphPayloadRegistry {
  aclmdlRI model_ri;
  std::vector<std::unique_ptr<LruResidentCompactWithPlanPayload>> payloads;
};

std::mutex& lru_resident_compact_graph_registry_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<aclmdlRI, std::unique_ptr<LruResidentCompactGraphPayloadRegistry>>&
lru_resident_compact_graph_registries() {
  static auto* registries = new std::unordered_map<aclmdlRI, std::unique_ptr<LruResidentCompactGraphPayloadRegistry>>();
  return *registries;
}

void delete_lru_resident_compact_graph_payloads(void* args) noexcept {
  auto* registry = static_cast<LruResidentCompactGraphPayloadRegistry*>(args);
  std::unique_ptr<LruResidentCompactGraphPayloadRegistry> owned_registry;
  {
    std::lock_guard<std::mutex> lock(lru_resident_compact_graph_registry_mutex());
    auto& registries = lru_resident_compact_graph_registries();
    const auto it = registries.find(registry->model_ri);
    if (it != registries.end() && it->second.get() == registry) {
      owned_registry = std::move(it->second);
      registries.erase(it);
    }
  }
}

LruResidentCompactWithPlanPayload* retain_lru_resident_compact_graph_payload(
    aclmdlRI model_ri, std::unique_ptr<LruResidentCompactWithPlanPayload> payload) {
  auto* raw_payload = payload.get();
  std::lock_guard<std::mutex> lock(lru_resident_compact_graph_registry_mutex());
  auto& registries = lru_resident_compact_graph_registries();
  auto it = registries.find(model_ri);
  if (it == registries.end()) {
    auto registry = std::make_unique<LruResidentCompactGraphPayloadRegistry>();
    registry->model_ri = model_ri;
    auto* raw_registry = registry.get();
    it = registries.emplace(model_ri, std::move(registry)).first;
    const aclError register_ret =
        aclmdlRIDestroyRegisterCallback(model_ri, delete_lru_resident_compact_graph_payloads, raw_registry);
    if (register_ret != ACL_SUCCESS) {
      registries.erase(it);
    }
    TORCH_CHECK(
        register_ret == ACL_SUCCESS,
        "failed to bind stable-row external LRU payload registry to graph lifetime, error code: ", register_ret);
  }
  it->second->payloads.push_back(std::move(payload));
  return raw_payload;
}

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
    raw_payload = retain_lru_resident_compact_graph_payload(model_ri, std::move(payload));
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

} // namespace vllm_ascend

#ifdef ASCEND_PLATFORM_310P
// Pybind on Ascend 310P
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{
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
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h);

    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o);

    ops.def(
        "chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, int chunk_size, str layout=\"BSND\", *, Tensor? initial_state=None, bool? output_final_state=False, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? safe_gate=False, float? lower_bound=None, bool? use_gate_in_kernel=False, Tensor? A_log=None, Tensor? dt_bias=None, bool? disable_recompute=False, bool? return_intermediate_states=False, bool? state_v_first=False) -> (Tensor o, Tensor? final_state, Tensor? gk, Tensor aqk, Tensor akk, Tensor? w, Tensor? u, Tensor? qg, Tensor? kg, Tensor? v_new, Tensor? h, Tensor? initial_state_out)"
    );
    ops.impl("chunk_kda_fwd", torch::kPrivateUse1, &vllm_ascend::chunk_kda_fwd);

    ops.def(
        "kda_gate_cumsum(Tensor g, int chunk_size, *, Tensor? A_log=None, Tensor? dt_bias=None, int[]? cu_seqlens=None, bool? use_gate_in_kernel=False, bool? safe_gate=False, float? lower_bound=-5.0, str layout=\"BSND\") -> Tensor"
    );
    ops.impl("kda_gate_cumsum", torch::kPrivateUse1, &vllm_ascend::kda_gate_cumsum);

    ops.def(
        "kda_layout_swap12(Tensor x, *, Tensor? dependency=None) -> Tensor"
    );
    ops.impl("kda_layout_swap12", torch::kPrivateUse1, &vllm_ascend::kda_layout_swap12);
}
#else
// Pybind on other platform
TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{

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
        "npu_recurrent_gated_delta_rule(Tensor query, "
        "                               Tensor key, "
        "                               Tensor value, "
        "                               Tensor(a!) state, "
        "                               *, "
        "                               Tensor? beta=None, "
        "                               float? scale=None, "
        "                               Tensor? actual_seq_lengths=None, "
        "                               Tensor? ssm_state_indices=None, "
        "                               Tensor? num_accepted_tokens=None, "
        "                               Tensor? g=None, "
        "                               Tensor? gk=None) -> Tensor");
    ops.impl("npu_recurrent_gated_delta_rule", torch::kPrivateUse1, &vllm_ascend::npu_recurrent_gated_delta_rule);

    ops.def(
        "recurrent_kda(Tensor query, Tensor key, Tensor value, Tensor gate, Tensor beta, "
        "Tensor(a!) initial_state, Tensor cu_seqlens, Tensor ssm_state_indices, Tensor A_log, Tensor dt_bias, *, "
        "Tensor? num_accepted_tokens=None, float scale=0.08838834764831845, "
        "bool use_qk_l2norm_in_kernel=True, bool use_gate_in_kernel=True, "
        "bool use_beta_sigmoid_in_kernel=False, bool allow_neg_eigval=False, "
        "bool safe_gate=True, float lower_bound=-5.0) -> Tensor output");
    ops.impl("recurrent_kda", torch::kPrivateUse1, &vllm_ascend::recurrent_kda);

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

    //batch_matmul ops refer to sgl-kernel-npu
    ops.def(
            "batch_matmul_transpose(Tensor tensor_a, Tensor tensor_b, Tensor tensor_c, str? format_mode=None, str? quant_mode=None) -> ()");
    ops.impl("batch_matmul_transpose", torch::kPrivateUse1, &vllm_ascend::batch_matmul_transpose);

    ops.def("swap_blocks(Tensor! x, Tensor! y, Tensor z) -> ()");
    ops.impl("swap_blocks", torch::kPrivateUse1, &vllm_ascend::swap_blocks);
#endif

    // npu_mla_prolog_v3: aligned with torch_npu; underlying aclnn op is MlaPrologV3 (950-only).
    ops.def(
        "npu_mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk,"
        "           Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv,"
        "           Tensor rope_sin, Tensor rope_cos, Tensor(a!) kv_cache, Tensor(b!) kr_cache, *,"
        "           Tensor? cache_index=None, Tensor? dequant_scale_x=None,"
        "           Tensor? dequant_scale_w_dq=None, Tensor? dequant_scale_w_uq_qr=None,"
        "           Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None,"
        "           Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None,"
        "           Tensor? actual_seq_len=None, Tensor? k_nope_clip_alpha=None,"
        "           float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05,"
        "           str cache_mode=\"PA_BSND\", bool query_norm_flag=False,"
        "           int weight_quant_mode=0, int kv_cache_quant_mode=0, int query_quant_mode=0,"
        "           int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128,"
        "           float qc_qr_scale=1.0, float kc_scale=1.0)"
        " -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    ops.impl("npu_mla_prolog_v3", torch::kPrivateUse1, &vllm_ascend::npu_mla_prolog_v3);

    // swap_blocks_batch takes CPU tensors (int64 pointer/size arrays), not NPU
    // tensors, so dispatch must be registered on the CPU backend. The function
    // internally submits async memcpy on the current NPU stream.
    ops.def("swap_blocks_batch(Tensor x, Tensor y, Tensor z, int direction) -> ()");
    ops.impl("swap_blocks_batch", torch::kCPU, &vllm_ascend::swap_blocks_batch);

#ifdef VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD
    ops.def("sparse_kv_warmup_lru_resident_threads(int requested_threads) -> int");
    ops.impl("sparse_kv_warmup_lru_resident_threads", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::warmup_lru_resident_threads);

    ops.def("sparse_kv_lru_resident_compact(int req_ids_ptr, int last_req_ids_ptr, int topk_indices_ptr, "
            "int stable_prefix_lens_ptr, int slot_to_token_ptr, int lru_slots_ptr, int current_slots_ptr, "
            "int miss_count_ptr, int miss_tokens_ptr, int miss_slots_ptr, int token_mark_workspace_ptr, "
            "int token_pos_workspace_ptr, int slot_workspace_ptr, int miss_position_workspace_ptr, "
            "int epochs_ptr, int num_reqs, int topk, int capacity, int max_token, int workspace_threads, "
            "int requested_threads) -> ()");
    ops.impl("sparse_kv_lru_resident_compact", c10::DispatchKey::CompositeExplicitAutograd,
             &vllm_ascend::lru_resident_compact);

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

    ops.def("sparse_kv_compute_lru_resident_addrs(Tensor miss_count, Tensor miss_tokens, Tensor miss_slots, "
            "Tensor block_table, int block_size, int token_size_bytes_k, int token_size_bytes_v, "
            "int gvas_k_base, int gvas_v_base, int addr_k_base, int addr_v_base, int resident_capacity, "
            "int max_num_threads, Tensor(a!) gvas_buffer, Tensor(b!) addr_buffer, Tensor(c!) size_buffer, "
            "Tensor(d!) num_tokens_buffer) -> int");
    ops.impl("sparse_kv_compute_lru_resident_addrs", torch::kCPU,
             &vllm_ascend::compute_lru_resident_addrs);

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

    ops.def(
        "npu_lightning_indexer("
            "Tensor query, Tensor key, Tensor weights, "
            "*, "
            "Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_key=None, "
            "Tensor? block_table=None, "
            "str layout_query=\"BSND\", str layout_key=\"BSND\", "
            "int sparse_count=2048, int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, "
            "bool return_value=False"
        ") -> (Tensor sparse_indices, Tensor sparse_values)"
    );
    ops.impl("npu_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer);

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
        "npu_kv_quant_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
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
    ops.impl("npu_kv_quant_sparse_flash_attention", torch::kPrivateUse1,
             &vllm_ascend::npu_kv_quant_sparse_flash_attention);

    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,"
        "                     Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, Tensor[] bias2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)"
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

    ops.def("npu_sign_bits_pack(Tensor input, int size) -> Tensor");
    ops.impl("npu_sign_bits_pack", torch::kPrivateUse1, &vllm_ascend::npu_sign_bits_pack);

    ops.def(
        "transpose_kv_cache_by_block(Tensor[] kCache, Tensor[] vCache, Tensor blockIDs, int blockSize, int headNum, int headDim, int splitNum, int layerNum) -> ()"
    );
    ops.impl("transpose_kv_cache_by_block", torch::kPrivateUse1, &vllm_ascend::transpose_kv_cache_by_block);

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
        "npu_causal_conv1d_custom(Tensor output, Tensor x, "
        "                         Tensor weight, "
        "                         Tensor conv_state, "
        "                         Tensor? bias_opt, "
        "                         Tensor? query_start_loc_opt, "
        "                         Tensor? cache_indices_opt, "
        "                         Tensor? initial_state_mode_opt, "
        "                         Tensor? num_accepted_tokens_opt, "
        "                         int activation_mode, "
        "                         int pad_slot_id, "
        "                         int run_mode"
        ") -> (Tensor output)");
    ops.impl("npu_causal_conv1d_custom", torch::kPrivateUse1, &vllm_ascend::npu_causal_conv1d_custom);
    ops.def(
        "moe_grouped_matmul("
            "Tensor x,"
            "Tensor weight,"
            "Tensor group_list,"
            "int split_item,"
            "int group_type,"
            "int group_list_type)"

        "-> Tensor[]"
    );
    ops.impl("moe_grouped_matmul", torch::kPrivateUse1,&vllm_ascend::moe_grouped_matmul);

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
        "bool out_flag=False"
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
        "npu_vllm_quant_lightning_indexer("
            "Tensor query, Tensor key, Tensor weights, "
            "Tensor query_dequant_scale, Tensor key_dequant_scale, "
            "int query_quant_mode=0, int key_quant_mode=0, "
            "Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_key=None, "
            "Tensor? block_table=None, "
            "Tensor? metadata=None, "
            "str layout_query=\"BSND\", str layout_key=\"BSND\", "
            "int sparse_count=2048, int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, "
            "int cmp_ratio=1, bool return_value=False"
        ") -> (Tensor sparse_indices, Tensor sparse_values)"
        );
    ops.impl("npu_vllm_quant_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_vllm_quant_lightning_indexer_npu);

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
        "npu_vllm_quant_lightning_indexer_metadata("
            "int num_heads_q, "
            "int num_heads_k, "
            "int head_dim, "
            "int query_quant_mode, "
            "int key_quant_mode, "
            "Tensor? actual_seq_lengths_query=None, "
            "Tensor? actual_seq_lengths_key=None, "
            "int batch_size=0, "
            "int max_seqlen_q=0, "
            "int max_seqlen_k=0, "
            "str layout_query=\"BSND\", "
            "str layout_key=\"BSND\", "
            "int sparse_count=2048, "
            "int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, "
            "int next_tokens=9223372036854775807, "
            "int cmp_ratio=1, "
            "str device=\"npu\""
        ") -> (Tensor metadata)"
        );
    ops.impl("npu_vllm_quant_lightning_indexer_metadata", torch::kPrivateUse1, &vllm_ascend::npu_vllm_quant_lightning_indexer_metadata_npu);

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
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre_v2", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v2_npu);

    ops.def(
        "inplace_partial_rotary_mul("
            "Tensor(a!) x, Tensor r1, Tensor r2, str rotary_mode, int[] partial_slice"
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
        "kv_compress_epilog("
            "Tensor(a!) kv_compress_cache, "
            "Tensor x, "
            "Tensor slot_mapping, "
            "int quant_group_size, "
            "int quant_mode, "
            "bool round_scale_flag, "
            "int layout"
        ") -> ()"
    );
    ops.impl("kv_compress_epilog", torch::kPrivateUse1, &vllm_ascend::kv_compress_epilog_npu);

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
        "npu_scatter_nd_update_v2("
                "Tensor(a!) var, Tensor indices, Tensor update"
            ") -> ()"
    );
    ops.impl("npu_scatter_nd_update_v2", torch::kPrivateUse1, &vllm_ascend::npu_scatter_nd_update_v2);

    // This operator is planned to be integrated into PTA in the near future.
    // Once that happens, the implementation in csrc will be removed.
    ops.def(
        "npu_lightning_indexer_quant(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
        "                            Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, "
        "                            Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, "
        "                            int query_quant_mode=0, int key_quant_mode=0, "
        "                            str layout_query='BSND', str layout_key='BSND',"
        "                            int sparse_count=2048, int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_lightning_indexer_quant", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer_quant);

    ops.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h);

    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o);

    ops.def(
        "chunk_kda_fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, int chunk_size, str layout=\"BSND\", *, Tensor? initial_state=None, bool? output_final_state=False, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? safe_gate=False, float? lower_bound=None, bool? use_gate_in_kernel=False, Tensor? A_log=None, Tensor? dt_bias=None, bool? disable_recompute=False, bool? return_intermediate_states=False, bool? state_v_first=False) -> (Tensor o, Tensor? final_state, Tensor? gk, Tensor aqk, Tensor akk, Tensor? w, Tensor? u, Tensor? qg, Tensor? kg, Tensor? v_new, Tensor? h, Tensor? initial_state_out)"
    );
    ops.impl("chunk_kda_fwd", torch::kPrivateUse1, &vllm_ascend::chunk_kda_fwd);

    ops.def(
        "kda_gate_cumsum(Tensor g, int chunk_size, *, Tensor? A_log=None, Tensor? dt_bias=None, int[]? cu_seqlens=None, bool? use_gate_in_kernel=False, bool? safe_gate=False, float? lower_bound=-5.0, str layout=\"BSND\") -> Tensor"
    );
    ops.impl("kda_gate_cumsum", torch::kPrivateUse1, &vllm_ascend::kda_gate_cumsum);

    ops.def(
        "kda_layout_swap12(Tensor x, *, Tensor? dependency=None) -> Tensor"
    );
    ops.impl("kda_layout_swap12", torch::kPrivateUse1, &vllm_ascend::kda_layout_swap12);

    //store_kv_block
     ops.def(
        "store_kv_block_metadata(Tensor slot_mapping_npu, Tensor group_len, Tensor group_key_idx, Tensor group_key_cache_idx, int block_size=0)"
         "-> ()"
     );
    ops.impl("store_kv_block_metadata", torch::kPrivateUse1, &vllm_ascend::store_kv_block_metadata);

    ops.def(
        "store_kv_block(Tensor key_in, Tensor key_cache_in, Tensor group_len, Tensor group_key_idx,Tensor group_key_cache_idx, int block_size=0) -> ()"
    );
    ops.impl("store_kv_block", torch::kPrivateUse1, &vllm_ascend::store_kv_block);

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

    ops.def(
        "npu_msa_index_score("
        "Tensor query, Tensor key, Tensor block_table, Tensor start_loc, *, "
        "Tensor? scale=None, Tensor? atten_mask=None, "
        "Tensor? actual_seq_qlen=None, Tensor? actual_seq_klen=None, "
        "str layout_key=\"BBND\", int sparse_mode=3, "
        "int init_blocks=0, int local_blocks=0) -> Tensor"
    );
    ops.impl("npu_msa_index_score", torch::kPrivateUse1,
             &vllm_ascend::npu_msa_index_score);

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
