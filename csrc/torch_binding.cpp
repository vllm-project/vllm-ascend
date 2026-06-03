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
#include "moe/apply_top_k_top_p_custom/apply_top_k_top_p_custom_torch_adpt.h"
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include "batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h"
#include "mla_preprocess/mla_preprocess_torch_adpt.h"
#endif
#include "mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"
#include "mc2/dispatch_gmm_combine_decode/dispatch_gmm_combine_decode_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
#include "gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h"
#include "attention/lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h"
#include "mc2/matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h"
#include "moe/moe_gating_top_k/moe_gating_top_k_torch_adpt.h"
#include "moe/moe_init_routing_custom/moe_init_routing_custom_torch_adpt.h"
#include "attention/sparse_flash_attention/sparse_flash_attention_torch_adpt.h"
#include "attention/lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
#include "attention/ngram_spec_decode/ngram_spec_decode_torch_adpt.h"
#include "moe/causal_conv1d_v310/causal_conv1d_310_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
#include "attention/recurrent_gated_delta_rule_v310/recurrent_gated_delta_rule_310_torch_adpt.h"
#include "attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h"
#include <c10/core/Device.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace vllm_ascend {

namespace {

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

std::tuple<at::Tensor, at::Tensor> get_masked_input_and_mask(
    at::Tensor &input,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index)
    /*
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py#L161-L198
    Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |
    Parameters:
        org_vocab_start_index //base embeddings start
        org_vocab_end_index //base embeddings end
        num_org_vocab_padding //base embeddings padding
        added_vocab_start_index //LoRA embeddings start
        added_vocab_end_index //LoRA embeddings end
    */
{
    // Input validation
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    TORCH_CHECK(org_vocab_start_index >= 0, "org_vocab_start_index must be non-negative");
    TORCH_CHECK(org_vocab_end_index >= org_vocab_start_index, "org_vocab_end_index must be greater than org_vocab_start_index");
    TORCH_CHECK(num_org_vocab_padding >= 0, "num_org_vocab_padding must be non-negative");
    TORCH_CHECK(added_vocab_start_index >= org_vocab_end_index, "added_vocab_start_index must be greater than org_vocab_end_index");
    TORCH_CHECK(added_vocab_end_index >= added_vocab_start_index, "added_vocab_end_index must be greater than added_vocab_start_index");

    // Get total number of elements
    int64_t size = input.numel();

    // Create output tensors
    at::Tensor masked_input = at::empty_like(input);
    at::Tensor mask = at::empty_like(input).to(at::kBool);

    // Get data pointers
    void *input_ptr = input.data_ptr();
    void *masked_input_ptr = masked_input.data_ptr();
    void *mask_ptr = mask.data_ptr();

    // Get current stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // Get scalar type
    at::ScalarType scalar_type = input.scalar_type();

    // Create and configure OpCommand
    at_npu::native::OpCommand cmd;
    cmd.Name("get_masked_input_and_mask");
    cmd.SetCustomHandler([scalar_type, size, stream,
                         input_ptr, masked_input_ptr, mask_ptr,
                         org_vocab_start_index, org_vocab_end_index,
                         num_org_vocab_padding, added_vocab_start_index,
                         added_vocab_end_index]() -> int {
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (size + aiv_num - 1) / aiv_num;

        // Call implementation
        get_masked_input_and_mask_impl(
            stream,
            input_ptr,
            masked_input_ptr,
            mask_ptr,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            size,
            loop_cnt,
            aiv_num);

        return 0;
    });
    cmd.Run();
    return {masked_input, mask};
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

at::Tensor convert_hamming_dist_top_k_output(const at::Tensor &hashq,
                                             const at::Tensor &hashkCache,
                                             const c10::optional<at::Tensor>& indices) {
    if (indices.has_value()) {
        return indices.value();
    }
    uint32_t MAX_BLOCK_PER_REQ_INHSA = 512;

    auto n_bs = hashq.size(0);
    auto n_kv_heads = hashkCache.size(1);
    auto n_max_kv = MAX_BLOCK_PER_REQ_INHSA;
    at::Tensor res = at::empty({n_bs, n_kv_heads, n_max_kv}, torch::TensorOptions().dtype(torch::kInt32).device(hashq.device()));
    return res;
}

at::Tensor npu_hamming_dist_top_k(const at::Tensor &hashq,
                                       const at::Tensor &hashkCache,
                                       const at::Tensor& hashkCacheRope,
                                       const at::Tensor &topN,
                                       const at::Tensor &seqLen,
                                       const c10::optional<at::Tensor> &chunkSize,
                                       const c10::optional<int64_t> maxSeqLen,
                                       const c10::optional<int64_t> sink,
                                       const c10::optional<int64_t> recent,
                                       const c10::optional<int64_t> supportOffload,
                                       const c10::optional<at::Tensor> &blockTable,
                                       const c10::optional<at::Tensor> &mask,
                                       const c10::optional<at::Tensor>& indices) {

    auto&& maxSeqLen_ = maxSeqLen.value_or(0);
    auto&& sink_ = sink.value_or(0);
    auto&& recent_ = recent.value_or(0);
    auto&& supportOffload_ = supportOffload.value_or(0);

    at::Tensor out = convert_hamming_dist_top_k_output(hashq, hashkCache, indices);
    EXEC_NPU_CMD(aclnnHammingDistTopK, hashq, hashkCache, topN, seqLen, chunkSize, blockTable, indices, hashkCacheRope, mask, maxSeqLen_, sink_, recent_, supportOffload_, out);
    return out;
}

at::Tensor npu_reshape_and_cache_bnsd(const at::Tensor& hashq,
                                           const at::Tensor& hashkCache,
                                           const at::Tensor& slotMapping,
                                           const at::Tensor& seqLen,
                                           const at::Tensor& hashkCacheOut) {
    EXEC_NPU_CMD(aclnnReshapeAndCacheBnsd, hashq, hashkCache, slotMapping, seqLen, hashkCacheOut);
    return hashkCacheOut;
}

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
    at::IntArrayRef query_start_loc_opt,
    at::IntArrayRef cache_indices_opt,
    at::IntArrayRef initial_state_mode_opt,
    at::IntArrayRef num_accepted_tokens_opt,
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

std::tuple<at::Tensor, at::Tensor> npu_quant_lightning_indexer_npu(
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

    EXEC_NPU_CMD(aclnnQuantLightningIndexer, query,
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

at::Tensor npu_quant_lightning_indexer_metadata_npu(
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

    EXEC_NPU_CMD(aclnnQuantLightningIndexerMetadata, actual_seq_lengths_query_val, actual_seq_lengths_key_val,
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
constexpr int64_t HC_PRE_FUSION_BASE_BS = 8192;
constexpr int64_t HC_PRE_FUSION_SPLIT_K_MAX_BS = 512;
constexpr const char* ASCEND_950_PREFIX = "Ascend950";

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

at::Tensor construct_hc_pre_rsqrt_output_tensor(const at::Tensor& x, float epsilon=1e-6)
{
    constexpr int64_t SIZE = 8;
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    auto options = x.options();
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (size_t i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
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

int64_t get_hc_pre_batch_size(const at::Tensor& x)
{
    if (x.dim() == 4) {
        return x.size(0) * x.size(1);
    }
    return x.size(0);
}

bool is_ascend950()
{
    static const char* soc_name = aclrtGetSocName();
    return soc_name != nullptr && std::string(soc_name).find(ASCEND_950_PREFIX) == 0;
}

bool should_use_hc_pre_fusion(const at::Tensor& x)
{
    if (!is_ascend950()) {
        return true;
    }
    auto bs = get_hc_pre_batch_size(x);
    return bs <= HC_PRE_FUSION_SPLIT_K_MAX_BS || bs % HC_PRE_FUSION_BASE_BS == 0;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_hc_pre_composite(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto xDims = x.dim();
    auto rsqrt = construct_hc_pre_rsqrt_output_tensor(x, norm_eps);
    EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt);

    auto original_type = x.dtype();
    at::Tensor x_float = x.to(at::kFloat);
    at::Tensor x_flattened = x_float.flatten(2, -1);
    if (xDims == 3) {
        x_flattened = x_float.flatten(1, -1);
    }
    auto mixes = at::linear(x_flattened, hc_fn);

    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps,
                    y, post, comb_frag);
    y = y.to(original_type);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    return run_hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);
    if (!should_use_hc_pre_fusion(x)) {
        return run_hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
    }
    return run_hc_pre_fusion(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
}

at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x, float epsilon=1e-20)
{
    constexpr int64_t SIZE = 8;
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    auto options = x.options();
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (auto i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

at::Tensor npu_hc_pre_inv_rms_npu(const at::Tensor& x, double epsilon=1e-20)
{
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");

    at::Tensor yOut;
    yOut = construct_hc_pre_inv_rms_output_tensor(x, epsilon);

    EXEC_NPU_CMD(aclnnHcPreInvRms, x, epsilon, yOut);

    return yOut;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_sinkhorn_output_tensor(const at::Tensor& mixes, const at::Tensor& x, int64_t hc_mult)
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_sinkhorn_npu(
    const at::Tensor& mixes, const at::Tensor& rsqrt, const at::Tensor& hc_scale, const at::Tensor& hc_base,
    const at::Tensor& x, int64_t hc_mult, int64_t hc_sinkhorn_iters, double hc_eps)
{
    auto output_tensors = construct_hc_pre_sinkhorn_output_tensor(mixes, x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);

    EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps,
                    y, post, comb_frag);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
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

void indexer_compress_epilog_npu(
    at::Tensor& indexer_compress_cache,
    at::Tensor& indexer_compress_cache_scale,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_mode = 1,
    bool round_scale = true)
{
    EXEC_NPU_CMD(aclnnIndexerCompressEpilog, indexer_compress_cache, indexer_compress_cache_scale, x,
                 slot_mapping, quant_mode, round_scale);
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

std::tuple<at::Tensor, at::Tensor> construct_load_index_kv_cache_output_tensor(
    const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping)
{
    constexpr int64_t KV_LAST_DIM = 128;
    int64_t n = slot_mapping.size(0);

    at::Tensor kv = at::empty({n, KV_LAST_DIM}, kv_cache.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor kv_scale = at::empty({n}, kv_cache.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
}

std::tuple<at::Tensor, at::Tensor> npu_load_index_kv_cache_npu(
    const at::Tensor& kv_cache,
    const at::Tensor& slot_mapping)
{
    auto output_tensors = construct_load_index_kv_cache_output_tensor(kv_cache, slot_mapping);
    at::Tensor kv = std::get<0>(output_tensors);
    at::Tensor kv_scale = std::get<1>(output_tensors);

    int64_t kv_cache_stride = kv_cache.stride(0);
    EXEC_NPU_CMD(aclnnLoadIndexKvCache, kv_cache, slot_mapping, kv_cache_stride, kv, kv_scale);

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
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
    const at::Tensor& group_index_value = c10::value_or_else(group_index, [&x] {
        return at::empty({1}, x.options().dtype(c10::ScalarType::Long)).fill_(x.size(0));
    });

    static const bool is_v2_available =
        GetOpApiFuncAddr("aclnnDequantSwigluQuantV2") != nullptr &&
        GetOpApiFuncAddr("aclnnDequantSwigluQuantV2GetWorkspaceSize") != nullptr;

    if (swiglu_mode == 0 && !is_v2_available) {
        EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_value, activation_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_value, activate_left, quant_mode_ptr, y, scale);
    } else {
        int64_t dst_type = 2;
        char* round_mode = const_cast<char*>("rint");
        int64_t activate_dim = -1;
        EXEC_NPU_CMD(aclnnDequantSwigluQuantV2, x, weight_scale_value, activation_scale_opt, bias_opt, quant_scale_opt,
                     quant_offset_opt, group_index_value, activate_left, quant_mode_ptr, dst_type, round_mode,
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

std::vector<int64_t> get_npu_storage_shape(const at::Tensor& tensor)
{
    TORCH_CHECK(
        tensor.is_privateuseone(),
        "get_npu_storage_shape only supports NPU tensors, but got device ",
        tensor.device());
    const auto& desc = NPUBridge::GetNpuStorageImplDesc(tensor);
    return std::vector<int64_t>(desc.storage_sizes_.begin(), desc.storage_sizes_.end());
}


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

#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
    // Direct kernel custom ops
    ops.def(
        "get_masked_input_and_mask(Tensor input, "
        "                         int org_vocab_start_index, "
        "                         int org_vocab_end_index, "
        "                         int num_org_vocab_padding, "
        "                         int added_vocab_start_index, "
        "                         int added_vocab_end_index) -> (Tensor masked_input, Tensor mask)");
    ops.impl("get_masked_input_and_mask", torch::kPrivateUse1, &vllm_ascend::get_masked_input_and_mask);

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
        "               Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor kv_cache,"
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

    // swap_blocks_batch takes CPU tensors (int64 pointer/size arrays), not NPU
    // tensors, so dispatch must be registered on the CPU backend. The function
    // internally submits async memcpy on the current NPU stream.
    ops.def("swap_blocks_batch(Tensor x, Tensor y, Tensor z, int direction) -> ()");
    ops.impl("swap_blocks_batch", torch::kCPU, &vllm_ascend::swap_blocks_batch);
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
        "dispatch_gmm_combine_decode(Tensor x, Tensor expert_ids, Tensor[] gmm1_permuted_weight,"
        "                            Tensor[] gmm1_permuted_weight_scale,"
        "                            Tensor[] gmm2_weight, Tensor[] gmm2_weight_scale,"
        "                            Tensor expert_scales, Tensor? expert_smooth_scales=None,"
        "                            Tensor? x_active_mask=None,"
        "                            str group_ep='',"
        "                            int ep_rank_size=0, int ep_rank_id=0, int moe_expert_num=0,"
        "                            int shared_expert_num=1, int shared_expert_rank_num=0,"
        "                            int quant_mode=0,"
        "                            int global_bs=0) -> (Tensor output, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_gmm_combine_decode", torch::kPrivateUse1, &vllm_ascend::dispatch_gmm_combine_decode);

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
        "npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *,"
        "                      Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None,"
        "                      Tensor? block_table=None, str layout_query='BSND', str layout_key='BSND',"
        "                      int sparse_count=2048, int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_lightning_indexer);

    ops.def(
        "npu_sparse_flash_attention(Tensor query, Tensor key, Tensor value,"
        "                           Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
        "                           Tensor? block_table=None, Tensor? actual_seq_lengths_query=None,"
        "                           Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None,"
        "                           Tensor? key_rope=None, str layout_query='BSND', str layout_kv='BSND',"
        "                           int sparse_mode=3) -> Tensor"
    );
    ops.impl("npu_sparse_flash_attention", torch::kPrivateUse1, &vllm_ascend::npu_sparse_flash_attention);

    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, Tensor expert_idx,"
        "                     Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, Tensor[] bias2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, float swiglu_limit=1000000.0) -> (Tensor out, Tensor expert_token_nums)"
    );
    ops.impl("dispatch_ffn_combine", torch::kPrivateUse1, &vllm_ascend::dispatch_ffn_combine);

    ops.def("matmul_allreduce_add_rmsnorm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, \
        str groupTp, int tpRankSize, int tpRankId, float epsilon, bool isTransB, bool isGatherAddOut) -> (Tensor output, Tensor add_out)");
    ops.impl("matmul_allreduce_add_rmsnorm", torch::kPrivateUse1, &vllm_ascend::matmul_allreduce_add_rmsnorm);

    ops.def(
        "npu_moe_init_routing_custom(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, "
        "                            int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, "
        "                            bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], "
        "                            int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)"
    );
    ops.impl("npu_moe_init_routing_custom", torch::kPrivateUse1, &vllm_ascend::npu_moe_init_routing_custom);
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

    ops.def("npu_apply_top_k_top_p(Tensor logits, Tensor? p=None, Tensor? k=None) -> Tensor");
    ops.impl("npu_apply_top_k_top_p", torch::kPrivateUse1, &vllm_ascend::npu_apply_top_k_top_p);

    ops.def(
        "npu_hamming_dist_top_k(Tensor q, Tensor k_comp, Tensor k_comp_rope, Tensor k,"
        "                      Tensor seq_len, Tensor? chunk_size=None,"
        "                      int? max_seq_len=None, int? sink=None, int? recent=None, int? support_offload=None,"
        "                      Tensor? key_block_table=None, Tensor? mask=None, Tensor? indices=None) -> Tensor"
    );
    ops.impl("npu_hamming_dist_top_k", torch::kPrivateUse1, &vllm_ascend::npu_hamming_dist_top_k);

    ops.def(
        "npu_reshape_and_cache_bnsd(Tensor q, Tensor k_comp, Tensor slot_mapping, Tensor seq_len, Tensor k_out) -> Tensor"
    );
    ops.impl("npu_reshape_and_cache_bnsd", torch::kPrivateUse1, &vllm_ascend::npu_reshape_and_cache_bnsd);

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
        "                         int[] query_start_loc_opt, "
        "                         int[] cache_indices_opt, "
        "                         int[] initial_state_mode_opt, "
        "                         int[] num_accepted_tokens_opt, "
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
        "npu_quant_lightning_indexer("
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
    ops.impl("npu_quant_lightning_indexer", torch::kPrivateUse1, &vllm_ascend::npu_quant_lightning_indexer_npu);

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
        "npu_quant_lightning_indexer_metadata("
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
    ops.impl("npu_quant_lightning_indexer_metadata", torch::kPrivateUse1, &vllm_ascend::npu_quant_lightning_indexer_metadata_npu);

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
        "npu_hc_pre("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_npu);

    ops.def(
        "npu_hc_pre_v2("
            "Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
            "int hc_mult, int hc_sinkhorn_iters, "
            "float norm_eps, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre_v2", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_v2_npu);

    ops.def(
        "npu_hc_pre_inv_rms("
            "Tensor x, float epsilon=1e-20"
        ") -> (Tensor out)"
        );
    ops.impl("npu_hc_pre_inv_rms", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_inv_rms_npu);

    ops.def(
        "npu_hc_pre_sinkhorn("
            "Tensor mixes, Tensor rsqrt, Tensor hc_scale, Tensor hc_base, Tensor x, "
            "int hc_mult, int hc_sinkhorn_iters, float hc_eps"
        ") -> (Tensor out0, Tensor out1, Tensor out2)"
        );
    ops.impl("npu_hc_pre_sinkhorn", torch::kPrivateUse1, &vllm_ascend::npu_hc_pre_sinkhorn_npu);

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
        "indexer_compress_epilog("
            "Tensor(a!) indexer_compress_cache, "
            "Tensor(b!) indexer_compress_cache_scale, "
            "Tensor x, "
            "Tensor slot_mapping, "
            "int quant_mode=1, "
            "bool round_scale=True"
        ") -> ()"
    );
    ops.impl("indexer_compress_epilog", torch::kPrivateUse1, &vllm_ascend::indexer_compress_epilog_npu);

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
        "npu_load_index_kv_cache("
            "Tensor kv_cache, Tensor slot_mapping"
        ") -> (Tensor out, Tensor out_scale)"
    );
    ops.impl("npu_load_index_kv_cache", torch::kPrivateUse1, &vllm_ascend::npu_load_index_kv_cache_npu);

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
    // N-gram spec decode
    ops.def(
        "npu_ngram_spec_decode(Tensor(a!) token_ids, Tensor num_tokens_no_spec, "
        "Tensor sampled_token_ids, Tensor discard_request_mask, "
        "int vocab_size, int min_n, int max_n, int k) -> "
        "(Tensor token_ids, Tensor next_token_ids, Tensor draft_token_ids, Tensor num_valid_draft_tokens)"
    );
    ops.impl("npu_ngram_spec_decode", torch::kPrivateUse1,
             &vllm_ascend::npu_ngram_spec_decode);

    ops.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor? g=None, *, Tensor? gk=None, Tensor? initial_state=None, bool? output_final_state=False, int? chunk_size=None, bool? save_new_value=True, int[]? cu_seqlens=None, int[]? chunk_indices=None, bool? use_exp2=False, bool? transpose_state_layout=False) -> (Tensor h_out, Tensor v_new_out, Tensor final_state_out)"
    );
    ops.impl("chunk_gated_delta_rule_fwd_h", torch::kPrivateUse1, &vllm_ascend::chunk_gated_delta_rule_fwd_h);

    ops.def(
        "chunk_fwd_o(Tensor q, Tensor k, Tensor v, Tensor h, float scale, *, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, int[]? chunk_indices=None, int? chunk_size=None, bool? transpose_state_layout=False) -> Tensor"
    );
    ops.impl("chunk_fwd_o", torch::kPrivateUse1, &vllm_ascend::chunk_fwd_o);

    // Fused GDN gating.
    ops.def(
        "npu_fused_gdn_gating(Tensor A_log, "
        "                     Tensor a, "
        "                     Tensor b, "
        "                     Tensor dt_bias, "
        "                     float beta=1.0) -> (Tensor g, Tensor beta_output)");
    ops.impl("npu_fused_gdn_gating", torch::kPrivateUse1, &vllm_ascend::npu_fused_gdn_gating);
}
#endif
