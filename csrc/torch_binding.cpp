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
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "ops.h"
#include "utils.h"
#include "mla_preprocess/op_host/mla_preprocess.h"
#include "batch_matmul_transpose/op_host/batch_matmul_transpose.h"
#include "aclnn_torch_adapter/op_api_common.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

namespace vllm_ascend {
const int64_t INT4_NUMS_IN_INT32 = 8;
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

std::tuple<at::Tensor, at::Tensor> rotary_embedding(at::Tensor &positions, at::Tensor &query, at::Tensor &key,
    int64_t head_size, at::Tensor &cos_sin_cache,  bool is_neox)
{
    int32_t deviceId = 0;
    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    TORCH_CHECK(
        positions_ndim == 1 || positions_ndim == 2,
        "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
          "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) &&
              key.size(0) == positions.size(0) &&
              query.size(1) == positions.size(1) &&
              key.size(1) == positions.size(1),
          "query, key and positions must have the same batch_size and seq_len");
    }
    TORCH_CHECK(head_size % 32 == 0, "rotary_embedding: headSize should be divisible by 32");
    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;
    TORCH_CHECK(query_hidden_size % head_size == 0);
    TORCH_CHECK(key_hidden_size % head_size == 0);
    TORCH_CHECK(is_neox == true, "rotary_embedding: neox=false is not supported as custom kernel in vllm-ascend");

    // Make sure query and key have consistent number of heads
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    TORCH_CHECK(num_heads % num_kv_heads == 0);
    at::Tensor query_dst = at::empty({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty({num_tokens, num_kv_heads, head_size}, key.options());

    int rot_dim = cos_sin_cache.size(1);
    int seq_dim_idx = positions_ndim - 1;
    int64_t *position_ids_ptr = positions.data_ptr<int64_t>();
    void *query_dst_ptr = query_dst.data_ptr();
    void *key_dst_ptr = key_dst.data_ptr();
    void *query_ptr = query.data_ptr();
    void *key_ptr = key.data_ptr();
    void *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = key.stride(seq_dim_idx);
    int64_t dst_query_stride = query_dst.stride(0);
    int64_t dst_key_stride = key_dst.stride(0);
    at::ScalarType scalar_type = query.scalar_type();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("rotary_embedding");
    cmd.SetCustomHandler([scalar_type, is_neox, num_tokens, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr,
                          query_ptr, key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride,
                          dst_query_stride, dst_key_stride, num_heads, num_kv_heads, head_size]() -> int {
        auto dtype_num = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        uint32_t loop_cnt = (num_tokens + aiv_num - 1) / aiv_num;
        rotary_embedding_impl(dtype_num, is_neox, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr, query_ptr,
                                key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride, dst_query_stride,
                                dst_key_stride, num_heads, num_kv_heads, head_size, num_tokens, loop_cnt, aiv_num);
        return 0;
    });
    cmd.Run();
    return {query_dst, key_dst};
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv,
    const c10::optional<at::Tensor> &descale0, const at::Tensor &gamma1, const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &wuk, const at::Tensor &kv_cache, const at::Tensor &kv_cache_rope, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0, const c10::optional<at::Tensor> &bias0,
    const c10::optional<at::Tensor> &quant_scale1, const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    c10::optional<c10::string_view> cache_mode, c10::optional<c10::string_view> quant_mode, c10::optional<bool> enable_inner_out, at::Tensor &q_out0,
    at::Tensor &kv_cache_out0, at::Tensor &q_out1, at::Tensor &kv_cache_out1, at::Tensor &inner_out)
{
    at::Tensor Descale0 =
        descale0.has_value()
            ? descale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Descale1 =
        descale1.has_value()
            ? descale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Beta1 =
        beta1.has_value()
            ? beta1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale0 =
        quant_scale0.has_value()
            ? quant_scale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale1 =
        quant_scale1.has_value()
            ? quant_scale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset0 =
        quant_offset0.has_value()
            ? quant_offset0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset1 =
        quant_offset1.has_value()
            ? quant_offset1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias0 =
        bias0.has_value()
            ? bias0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias1 =
        bias1.has_value()
            ? bias1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor CtkvScale =
        ctkv_scale.has_value()
            ? ctkv_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor QnopeScale =
        q_nope_scale.has_value()
            ? q_nope_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    bool enableInnerOut =
        enable_inner_out.has_value()
            ? enable_inner_out.value()
            : false;
    
    auto [workspace_tensor, tiling, block_dim] = mlapo::mla_preprocess_tiling(
        hiddenState,
        wdqkv,
        wuk,
        cache_mode,
        quant_mode,
        enableInnerOut
    );

    void *hidden_state_ptr = hiddenState.data_ptr();
    void *quant_scale0_ptr = Quant_scale0.data_ptr();
    void *quant_offset0_ptr = Quant_offset0.data_ptr();
    void *wdqkv_ptr = wdqkv.data_ptr();
    void *bias0_ptr = Bias0.data_ptr();
    void *gamma1_ptr = gamma1.data_ptr();
    void *beta1_ptr = Beta1.data_ptr();
    void *quant_scale1_ptr = Quant_scale1.data_ptr();
    void *quant_offset1_ptr = Quant_offset1.data_ptr();
    void *gamma2_ptr = gamma2.data_ptr();
    void *sin_ptr = sin.data_ptr();
    void *cos_ptr = cos.data_ptr();
    void *kv_cache_ptr = kv_cache.data_ptr();
    void *slotmapping_ptr = slotmapping.data_ptr();
    void *wuq_ptr = wuq.data_ptr();
    void *bias1_ptr = Bias1.data_ptr();
    void *wuk_ptr = wuk.data_ptr();
    void *descale0_ptr = Descale0.data_ptr();
    void *descale1_ptr = Descale1.data_ptr();
    void *ctkv_scale_ptr = CtkvScale.data_ptr();
    void *qnope_scale_ptr = QnopeScale.data_ptr();
    void *q_out0_ptr = q_out0.data_ptr();
    void *kv_cache_out0_ptr = kv_cache_out0.data_ptr();
    void *q_out1_ptr = q_out1.data_ptr();
    void *kv_cache_out1_ptr = kv_cache_out1.data_ptr();
    void *inner_out_ptr = inner_out.data_ptr();
    void *workspace_ptr = workspace_tensor.data_ptr();
    void *tiling_ptr = tiling.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("mla_preprocess");

    cmd.SetCustomHandler([stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                          gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr,
                          kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                          qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, inner_out_ptr, workspace_ptr,
                          tiling_ptr, block_dim]() -> int {
        mla_preprocess_impl(stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr,
                            gamma1_ptr, beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, sin_ptr, cos_ptr,
                            kv_cache_ptr, slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                            qnope_scale_ptr, q_out0_ptr, kv_cache_out0_ptr, q_out1_ptr, kv_cache_out1_ptr, inner_out_ptr, workspace_ptr,
                            tiling_ptr, block_dim);
        return 0;
    });
    cmd.Run();
    return std::forward_as_tuple(q_out0, kv_cache_out0, q_out1, kv_cache_out1, inner_out);
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant(
    const at::Tensor &x, const at::Tensor &weight, const at::Tensor &weight_scale, const at::Tensor &x_scale,
    const at::Tensor &group_list, const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &offset)
{
    int m = x.sizes()[0];
    int n = weight.sizes()[2];
    bool is_a8w4 = x.dtype() == at::kChar && weight.dtype() == at::kInt;
    if (is_a8w4) {
        n *= INT4_NUMS_IN_INT32;
    }

    at::Tensor output = at::empty({m, n/2}, x.options().dtype(c10::ScalarType::Char));
    at::Tensor output_scale = at::empty({m}, x.options().dtype(c10::ScalarType::Float));
    at::Tensor output_offset = at::empty({}, x.options().dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNZ,
        x,
        weight,
        bias,
        offset,
        weight_scale,
        x_scale,
        group_list,
        output,
        output_scale,
        output_offset);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, output_scale, output_offset);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> grouped_matmul_swiglu_quant_weight_nz_tensor_list(
    const at::Tensor & x,
    const at::TensorList & weight,
    const at::TensorList & weight_scale,
    const at::Tensor & x_scale,
    const at::Tensor & group_list,
    const c10::optional<at::Tensor> & bias,
    const c10::optional<at::Tensor> & offset)
{
    auto x_size = x.sizes();
    int n = weight[0].sizes()[1];
    int m = x_size[0];
    int k = x_size[1];

    at::Tensor output = at::empty({m, n/2}, x.options().dtype(at::kChar));
    at::Tensor output_scale = at::empty({m}, x.options().dtype(at::kFloat));
    at::Tensor output_offset = at::empty({m}, x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluQuantWeightNzTensorList,
        x,
        weight,
        bias,
        offset,
        weight_scale,
        x_scale,
        group_list,
        output,
        output_scale,
        output_offset);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, output_scale, output_offset);
}

std::tuple<at::Tensor, at::Tensor> dispatch_gmm_combine_decode(
    const at::Tensor &x,
    const at::Tensor &expert_ids,
    const at::TensorList &gmm1_permuted_weight,
    const at::TensorList &gmm1_permuted_weight_scale,
    const at::TensorList &gmm2_weight,
    const at::TensorList &gmm2_weight_scale,
    const at::Tensor &expert_scales,
    const c10::optional<at::Tensor> &expert_smooth_scales,
    const c10::optional<at::Tensor> &x_active_mask,
    c10::string_view group_ep,
    int64_t ep_rank_size,
    int64_t ep_rank_id,
    int64_t moe_expert_num,
    int64_t shared_expert_num,
    int64_t shared_expert_rank_num,
    int64_t quant_mode,
    int64_t global_bs)
{
    auto x_shape = x.sizes();
    int bs = x_shape[0];
    int h = x_shape[1];

    at::Tensor output = at::empty({bs, h}, x.options());

    bool is_shared_expert = (ep_rank_id < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : moe_expert_num / (ep_rank_size - shared_expert_rank_num);
    auto opts = expert_ids.options().dtype(at::kLong);
    at::Tensor expert_token_nums = at::empty({num_local_experts}, opts);

    vector<char> group_ep_chrs(group_ep.begin(), group_ep.end());
    group_ep_chrs.push_back('\0');
    char *group_ep_ptr = &group_ep_chrs[0];
    EXEC_NPU_CMD(
        // op api
        aclnnDispatchGmmCombineDecode,
        // input tensors
        x,
        expert_ids,
        gmm1_permuted_weight,
        gmm1_permuted_weight_scale,
        gmm2_weight,
        gmm2_weight_scale,
        expert_scales,
        expert_smooth_scales,
        x_active_mask,
        //input attrs
        group_ep_ptr,
        ep_rank_size,
        ep_rank_id,
        moe_expert_num,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        // output tensors
        output,
        expert_token_nums);
    return {output, expert_token_nums};
}

void batch_matmul_transpose(const at::Tensor &tensor_a, const at::Tensor &tensor_b, at::Tensor &tensor_c,
                                    c10::optional<c10::string_view> format_mode,
                                    c10::optional<c10::string_view> quant_mode)
{
    auto [tiling_tensor, block_dim] = bmm_trans::batch_matmul_transpose_tiling(
        tensor_a,
        tensor_b,
        tensor_c,
        format_mode,
        quant_mode
    );

    void *gm_a = tensor_a.data_ptr();
    void *gm_b = tensor_b.data_ptr();
    void *gm_c = tensor_c.data_ptr();
    void *gm_tiling_data = tiling_tensor.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("batch_matmul_transpose");

    cmd.SetCustomHandler([stream, gm_a, gm_b, gm_c, gm_tiling_data,
                          block_dim]() -> int {
        batch_matmul_transpose_impl(stream, gm_a, gm_b, gm_c, gm_tiling_data,
                            block_dim);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor& dispatch_ffn_combine(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out
) {
    char *group_ep_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD(aclnnDispatchFFNCombine,
                 x,
                 weight1,
                 weight2,
                 expert_idx,
                 scale1,
                 scale2,
                 probs,
                 group_ep_ptr,
                 max_output_size,
                 out);
    return out;
}

at::Tensor npu_lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    // npu tensor max size
    constexpr int32_t SIZE = 8;
    constexpr int32_t DIM_0 = 0;
    constexpr int32_t DIM_1 = 1;
    constexpr int32_t DIM_2 = 2;
    constexpr int32_t DIM_3 = 3;

    TORCH_CHECK(query.numel() > 0, "Query is empty.");
    TORCH_CHECK(key.numel() > 0, "Key is empty.");
    TORCH_CHECK(weights.numel() > 0, "Weights is empty.");
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);

    at::SmallVector<int64_t, SIZE> output_size;
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    at::Tensor lightning_indexer_output = at::empty(output_size, query.options().dtype(at::kInt));
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());
    EXEC_NPU_CMD(
        aclnnLightningIndexer,
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        query_layout_ptr,
        key_layout_ptr,
        sparse_count,
        sparse_mode,
        lightning_indexer_output);
    return lightning_indexer_output;
}

at::Tensor npu_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }
    // construct the output tensor
    at::Tensor output = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        output);
    return output;
}
std::tuple<at::Tensor, at::Tensor> matmul_allreduce_add_rmsnorm(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_trans_b,
    bool is_gather_add_out)
    {
        at::Tensor output = at::empty_like(residual);
        at::Tensor add_out = at::empty_like(residual);

        std::string group_tp_str(group_tp);

        char *group_tp_ptr = group_tp_str.data();

        float epsilon_f = static_cast<float>(epsilon);
        EXEC_NPU_CMD(aclnnMatmulAllreduceAddRmsnorm,
            // input
            x1, x2, residual, gamma,
            // attr
            group_tp_ptr, tp_rank_size, tp_rank_id, epsilon_f, is_trans_b, is_gather_add_out,
            // output
            output, add_out);

        return {output, add_out};
    }

std::tuple<at::Tensor, at::Tensor, at::Tensor> get_dispatch_layout(const at::Tensor& topk_idx, int64_t num_experts,
                                                                   int64_t num_ranks) {
    TORCH_BIND_ASSERT(topk_idx.dim() == 2);
    TORCH_BIND_ASSERT(topk_idx.is_contiguous());
    TORCH_BIND_ASSERT(num_experts > 0);

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);

    auto device = topk_idx.device();
    auto num_tokens_per_expert = at::zeros({num_experts}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rank = at::zeros({num_ranks}, at::dtype(at::kInt).device(device));
    auto is_token_in_rank = at::zeros({num_tokens, num_ranks}, at::dtype(at::kInt).device(device));

    EXEC_NPU_CMD(aclnnDispatchLayout,
        topk_idx,
        num_tokens,
        num_ranks,
        num_experts,
        num_topk,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank);

    auto is_token_in_rank_bool = is_token_in_rank.to(at::kBool);

    return std::make_tuple(num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank_bool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch_prefill(
    const at::Tensor& x, const at::Tensor& topk_idx, const at::Tensor& topk_weights,
    const at::Tensor& num_tokens_per_rank, const at::Tensor& is_token_in_rank, at::Tensor& num_tokens_per_expert,
    int64_t num_worst_tokens, c10::string_view groupEp, int64_t rank, int64_t num_ranks) {
    std::vector<char> group_ep_chrs(groupEp.begin(), groupEp.end());
    group_ep_chrs.push_back('\0');
    char* group_ep_ptr = &group_ep_chrs[0];
    at::Tensor new_x = x;

    // Type checks
    TORCH_BIND_ASSERT(is_token_in_rank.scalar_type() == at::kBool);
    TORCH_BIND_ASSERT(num_tokens_per_expert.scalar_type() == at::kInt);
    TORCH_BIND_ASSERT(num_tokens_per_rank.scalar_type() == at::kInt);

    // Shape and contiguous checks
    TORCH_BIND_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    // TORCH_BIND_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    TORCH_BIND_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    TORCH_BIND_ASSERT(is_token_in_rank.size(0) == new_x.size(0) and is_token_in_rank.size(1) == num_ranks);
    TORCH_BIND_ASSERT(num_tokens_per_expert.dim() == 1 and num_tokens_per_expert.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens_per_expert.size(0) % num_ranks == 0);
    TORCH_BIND_ASSERT(num_tokens_per_rank.dim() == 1 and num_tokens_per_rank.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens_per_rank.size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0));
    auto hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert.size(0));
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);

    // Top-k checks
    int num_topk = 0;
    num_topk = static_cast<int>(topk_idx.size(1));
    TORCH_BIND_ASSERT(num_experts > 0);
    TORCH_BIND_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    TORCH_BIND_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    TORCH_BIND_ASSERT(num_tokens == topk_idx.size(0));
    TORCH_BIND_ASSERT(num_topk == topk_weights.size(1));
    TORCH_BIND_ASSERT(topk_weights.scalar_type() == at::kFloat);

    int send_per_group = 3;  // (send_to_expert_num, send_to_expert_offset, send_rank_tokens)

    auto send_data = at::empty({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));
    int64_t send_count = send_per_group * num_local_experts * num_ranks;

    auto send_data_offset = at::empty({num_experts}, at::dtype(at::kInt).device(x.device()));
    at::Tensor recv_data = at::empty({num_experts * send_per_group}, at::dtype(at::kInt).device(x.device()));

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;

    EXEC_NPU_CMD(aclnnNotifyDispatch,
        send_data,
        num_tokens_per_expert, 
        send_count,
        num_tokens,
        group_ep_ptr,  // commGroup
        num_ranks,     // rankSize
        rank,          // rankId
        local_rank_size,
        local_rank_id,
        send_data_offset,
        recv_data);

    auto options_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    std::vector<int32_t> local_expert_acc(num_experts, 0);
    auto send_token_idx_cpu = at::empty({num_tokens, num_topk}, options_cpu);
    auto send_token_idx_ptr = send_token_idx_cpu.data_ptr<int>();

    auto topk_idx_cpu = topk_idx.to(at::kCPU);
    auto topk_idx_ptr = topk_idx_cpu.data_ptr<int64_t>();
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < num_topk; ++j) {
            int64_t expert_idx = topk_idx_ptr[i * num_topk + j];
            if (expert_idx >= 0) {
                int32_t cnt = local_expert_acc[expert_idx];
                send_token_idx_ptr[i * num_topk + j] = cnt;
                local_expert_acc[expert_idx]++;
            }
        }
    }

    TORCH_BIND_ASSERT(recv_data.dim() == 1 and recv_data.is_contiguous());
    TORCH_BIND_ASSERT(recv_data.size(0) % num_experts == 0);
    at::Tensor recv_offset_cpu = at::empty({num_experts}, options_cpu);
    at::Tensor recv_count_cpu = at::empty({num_experts}, options_cpu);
    auto recv_data_cpu = recv_data.to(at::kCPU);
    auto recv_data_ptr = recv_data_cpu.data_ptr<int>();
    auto recv_count_ptr = recv_count_cpu.data_ptr<int>();
    auto recv_offset_ptr = recv_offset_cpu.data_ptr<int>();
    int64_t total_recv_tokens = 0;
    int64_t num_max_dispatch_tokens_per_rank = 0;
    std::vector<int64_t> num_recv_tokens_per_expert_list;

    for (int64_t local_e = 0; local_e < num_local_experts; ++local_e) {
        int64_t local_expert_recv_tokens = 0;
        for (int64_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
            int64_t index = local_e * num_ranks + src_rank;
            int64_t pair_idx = send_per_group * (src_rank * num_local_experts + local_e);

            int recv_cnt = recv_data_ptr[pair_idx];                 // count from this src_rank for
                                                                    // this global_expert
            int recv_off = recv_data_ptr[pair_idx + 1];             // offset in that src_rank's window
            int64_t send_num_tokens = recv_data_ptr[pair_idx + 2];  // all bs from rank

            total_recv_tokens += recv_cnt;
            recv_count_ptr[index] = total_recv_tokens;
            recv_offset_ptr[index] = recv_off;
            num_max_dispatch_tokens_per_rank = std::max(num_max_dispatch_tokens_per_rank, send_num_tokens);

            local_expert_recv_tokens += recv_cnt;
        }
        num_recv_tokens_per_expert_list.push_back(local_expert_recv_tokens);
    }
    auto option = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    at::Tensor num_recv_tokens_per_expert = torch::from_blob(
        num_recv_tokens_per_expert_list.data(), {static_cast<int64_t>(num_recv_tokens_per_expert_list.size())}, option)
        .clone();

    at::Tensor expert_ids = topk_idx.to(at::kInt);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = 0;
    int64_t global_bs = static_cast<int64_t>(
        std::max(num_max_dispatch_tokens_per_rank * num_ranks, static_cast<int64_t>(num_worst_tokens)));

    auto send_token_idx = send_token_idx_cpu.to(x.device());
    auto recv_offset = recv_offset_cpu.to(x.device());
    auto recv_count = recv_count_cpu.to(x.device());

    int total_cnt = total_recv_tokens;
    if (total_cnt == 0) {
        total_cnt = 1;
    }
    auto expandx_out = at::empty({total_cnt, hidden}, x.options());
    auto dynamic_scales_out = at::empty({total_cnt}, at::dtype(at::kFloat).device(x.device()));
    auto expand_idx_out = at::empty({total_cnt * 3}, at::dtype(at::kInt).device(x.device()));

    EXEC_NPU_CMD(aclnnMoeDispatchNormal,
        new_x,
        expert_ids,
        send_data_offset,
        send_token_idx,
        recv_offset,
        recv_count,
        group_ep_ptr,  // commGroup
        num_ranks,     // rankSize
        rank,          // rankId
        group_ep_ptr,
        tp_size,
        tp_rank,
        num_experts,
        quant_mode,
        global_bs,
        expandx_out,
        dynamic_scales_out,
        expand_idx_out);

    // Return values
    return {expandx_out, expand_idx_out, recv_count, num_recv_tokens_per_expert};
}

at::Tensor combine_prefill(const at::Tensor& x, const at::Tensor& topk_idx, const at::Tensor& topk_weights,
                           const at::Tensor& src_idx, const at::Tensor& send_head, c10::string_view groupEp,
                           int64_t rank, int64_t num_ranks) {
    std::vector<char> group_ep_chrs(groupEp.begin(), groupEp.end());
    group_ep_chrs.push_back('\0');
    char* group_ep_ptr = &group_ep_chrs[0];

    TORCH_BIND_ASSERT(x.dim() == 2 and x.is_contiguous());
    at::Tensor recv_x = x;

    at::Tensor topk_idx_p = topk_idx;

    auto topk_idx_int32 = topk_idx_p.to(at::kInt);
    at::Tensor expand_ids = topk_idx_int32;
    at::Tensor token_src_info = src_idx;
    at::Tensor ep_send_counts = send_head;
    auto device = x.device();

    const int num_tokens = topk_idx_p.size(0);
    const int num_topk = topk_idx_p.size(1);

    int64_t hidden = static_cast<int>(recv_x.size(1));
    at::Tensor tp_send_counts = at::empty({1}, at::dtype(at::kInt).device(device));
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t moe_expert_number = send_head.size(0);
    int64_t global_bs = topk_idx_p.size(0) * num_ranks;

    // Combine data
    auto combined_x = torch::empty({topk_weights.size(0), hidden}, x.options());

    EXEC_NPU_CMD(aclnnMoeCombineNormal,
        recv_x,
        token_src_info,
        ep_send_counts,
        topk_weights,
        tp_send_counts,
        group_ep_ptr,
        num_ranks,
        rank,
        group_ep_ptr,
        tp_world_size,
        tp_rankId,
        moe_expert_number,
        global_bs,
        combined_x);

    return combined_x;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_init_routing_custom(
    const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type)
{
    constexpr int64_t DIM_X = 2;
    constexpr int64_t DIM_EXPERT_IDX = 2;
    constexpr int64_t LENGTH_ACTIVE_EXPERT_RANGE = 2;
    constexpr int64_t EXPERT_TOKENS_COUNT = 1;
    constexpr int64_t EXPERT_TOKENS_KEY_VALUE = 2;
    constexpr int64_t QUANT_MODE_UNQUANT = -1;
    constexpr int64_t QUANT_MODE_DYNAMIC_QUANT = 1;
    constexpr int64_t CUMSUM = 0;
    constexpr int64_t COUNT = 1;
    constexpr int64_t KEY_VALUE = 2;

    if (active_expert_range.empty()) {
        active_expert_range =  at::IntArrayRef({0, expert_num});
    }

    int64_t x_dim = x.dim();
    TORCH_CHECK(x_dim == DIM_X, "The x should be ", DIM_X, 
                "-Dimension, current is ", x_dim, "-Dimension.");

    int64_t expert_idx_dim = expert_idx.dim();
    TORCH_CHECK(expert_idx_dim == DIM_EXPERT_IDX, "The expert_idx should be ", DIM_EXPERT_IDX, 
                "-Dimension, current is ", expert_idx_dim, "-Dimension.");

    int64_t active_expert_range_length = active_expert_range.size();
    TORCH_CHECK(active_expert_range_length == LENGTH_ACTIVE_EXPERT_RANGE, "The active_expert_range should be ", LENGTH_ACTIVE_EXPERT_RANGE, 
                "-Dimension, current is ", expert_idx_dim, "-Dimension.");

    int expert_length = active_expert_range[1] - active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();

    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];
    int64_t expanded_scale_len = 0;
    at::Tensor expanded_x;

    if (drop_pad_mode == 1) { // Drop/Pad
        if (quant_mode == QUANT_MODE_UNQUANT) {
            expanded_x = at::empty({expert_num, expert_capacity, h}, x.options());
        } else {
            expanded_x = at::empty({expert_num, expert_capacity, h}, x.options().dtype(at::kChar));
        }
        expanded_scale_len = expert_num * expert_capacity;
    } else { // Dropless / Active
        if (active_num > 0) { // Active
            int64_t num_out_tokens = std::min((int64_t)bs * k, active_num);
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({num_out_tokens, h}, x.options());
            } else {
                expanded_x = at::empty({num_out_tokens, h}, x.options().dtype(at::kChar));
            }
            expanded_scale_len = num_out_tokens;
        } else { // Dropless
            if (quant_mode == QUANT_MODE_UNQUANT) {
                expanded_x = at::empty({bs * k, h}, x.options());
            } else {
                expanded_x = at::empty({bs * k, h}, x.options().dtype(at::kChar));
            }
            expanded_scale_len = bs * k;
        }
    }

    at::Tensor expanded_row_idx = at::empty({bs * k}, expert_idx.options());
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type >= CUMSUM && expert_tokens_num_type <= COUNT) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum = at::empty({expert_length}, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum = at::empty({expert_num, 2}, x.options().dtype(at::kLong));
    }
    at::Tensor expanded_scale = at::empty({expanded_scale_len}, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMoeInitRoutingCustom,
                 x,
                 expert_idx,
                 scale,
                 offset,
                 active_num,
                 expert_capacity,
                 expert_num,
                 drop_pad_mode,
                 expert_tokens_num_type,
                 expert_tokens_num_flag,
                 quant_mode,
                 active_expert_range,
                 row_idx_type,
                 expanded_x,
                 expanded_row_idx,
                 expert_tokens_count_or_cumsum,
                 expanded_scale);
    return std::tie(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k(
    const at::Tensor& x,
    int64_t k,
    int64_t k_group,
    int64_t group_count,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag,
    double routed_scaling_factor,
    double eps,
    const c10::optional<at::Tensor>& bias_opt
    )
{
    TORCH_CHECK(x.dim() == 2, "The x should be 2D");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "float16、float32 or bfloat16 tensor expected but got a tensor with dtype: ",
        x.scalar_type());

    auto x_size = x.sizes();
    auto rows = x_size[0];
    auto expert_num = x_size[1];
    const at::Tensor &bias = c10::value_or_else(bias_opt, [] { return at::Tensor(); });
    if (bias.defined()) {
        TORCH_CHECK(x.scalar_type() == bias.scalar_type(), "The dtype of x and bias should be same");
        TORCH_CHECK(bias.dim() == 1, "The bias should be 1D");
        auto bias_size = bias.sizes();
        TORCH_CHECK(bias_size[0] == expert_num, "The bias first dim should be same as x second dim");
    }
    at::Tensor y = at::empty({rows, k}, x.options());
    at::Tensor expert_idx = at::empty({rows, k}, x.options().dtype(at::kInt));
    at::Tensor out = at::empty({rows, expert_num}, x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnMoeGatingTopK,
                    x,                 
                    bias,
                    k,                  
                    k_group,             
                    group_count,         
                    group_select_mode,   
                    renorm,              
                    norm_type,            
                    out_flag,            
                    routed_scaling_factor, 
                    eps,                
                    y,                
                    expert_idx,        
                    out
                ); 

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y,expert_idx,out);
}

} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)
{

    // vLLM-Ascend custom ops
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", torch::kPrivateUse1, &vllm_ascend::weak_ref_tensor);

    // Rotary embedding
    // Apply GPT-NeoX style rotary embedding to query and key.
    ops.def(
        "rotary_embedding(Tensor positions, Tensor! query,"
        "                 Tensor! key, int head_size,"
        "                 Tensor cos_sin_cache, bool is_neox) -> (Tensor query, Tensor key)");
    ops.impl("rotary_embedding", torch::kPrivateUse1, &vllm_ascend::rotary_embedding);

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

    ops.def(
        "grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale,"
        "                            Tensor group_list, *, Tensor? bias=None,"
        "                            Tensor? offset=None) -> (Tensor output, Tensor output_scale, Tensor output_offset)");
    ops.impl("grouped_matmul_swiglu_quant", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant);

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
        "                                                  Tensor? bias=None, Tensor? offset=None) ->"
        "                                                  (Tensor output, Tensor output_scale, Tensor output_offset)"
    );
    ops.impl("grouped_matmul_swiglu_quant_weight_nz_tensor_list", torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_weight_nz_tensor_list);

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
        "                     Tensor[] scale1, Tensor[] scale2, Tensor probs, str group,"
        "                     int max_output_size, Tensor! out) -> Tensor"
    );
    ops.impl("dispatch_ffn_combine", torch::kPrivateUse1, &vllm_ascend::dispatch_ffn_combine);

    ops.def("matmul_allreduce_add_rmsnorm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, \
        str groupTp, int tpRankSize, int tpRankId, float epsilon, bool isTransB, bool isGatherAddOut) -> (Tensor output, Tensor add_out)");
    ops.impl("matmul_allreduce_add_rmsnorm", torch::kPrivateUse1, &vllm_ascend::matmul_allreduce_add_rmsnorm);

    ops.def("get_dispatch_layout(Tensor topk_idx, int num_experts, int "
            "num_ranks) -> (Tensor num_tokens_per_rank, Tensor "
            "num_tokens_per_expert, Tensor is_token_in_rank_bool)");
    ops.impl("get_dispatch_layout", torch::kPrivateUse1,
             &vllm_ascend::get_dispatch_layout);

    ops.def(
        "dispatch_prefill(Tensor x, Tensor topk_idx, Tensor topk_weights, "
        "Tensor num_tokens_per_rank, Tensor is_token_in_rank, Tensor "
        "num_tokens_per_expert, int num_worst_tokens, str groupEp, int rank, "
        "int num_ranks) -> (Tensor expandx_out, Tensor expand_idx_out, Tensor "
        "recv_count, Tensor num_recv_tokens_per_expert)");
    ops.impl("dispatch_prefill", torch::kPrivateUse1,
             &vllm_ascend::dispatch_prefill);

    ops.def("combine_prefill(Tensor x, Tensor topk_idx, Tensor topk_weights, "
            "Tensor src_idx, Tensor send_head, str grouEp, int rank, int "
            "num_ranks) -> Tensor");
    ops.impl("combine_prefill", torch::kPrivateUse1,
             &vllm_ascend::combine_prefill);
    
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
}
