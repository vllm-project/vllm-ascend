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
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include <pybind11/pybind11.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnn/opdev/platform.h"
#include "ops.h"
#include "utils.h"

namespace vllm_ascend {

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
        fe::PlatFormInfos platform_infos;
        int device_id = 0;
        fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
        uint32_t aivNum = platform_infos.GetCoreNumByType("aiv");
        uint32_t loop_cnt = (num_tokens + aivNum - 1) / aivNum;
        rotary_embedding_impl(dtype_num, is_neox, stream, position_ids_ptr, query_dst_ptr, key_dst_ptr, query_ptr,
                                key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride, dst_query_stride,
                                dst_key_stride, num_heads, num_kv_heads, head_size, num_tokens, loop_cnt, aivNum);
        return 0;
    });
    cmd.Run();
    return {query_dst, key_dst};
}

void verify_tensor(std::string const& name, at::Tensor const& t,
                          int64_t const size_0, int64_t const size_1,
                          c10::ScalarType const type) {
    bool size_0_cond = true;
    if (size_0 != -1) {
        size_0_cond = t.size(0) == size_0;
    }

    bool size_1_cond = true;
    if (size_1 != -1) {
        size_1_cond = t.size(1) == size_1;
    }

    bool is_contiguous = t.is_contiguous();
    bool same_type = t.dtype() == type;

    bool pass = size_0_cond && size_1_cond && is_contiguous && same_type;
    if (!pass) {
        TORCH_CHECK(false, "tensor: name = ", name, ", shape = ", t.sizes(),
                " is_cont = ", t.is_contiguous(), ", type = ", t.dtype(),
                " is not as expected: shape = [", size_0, ", ", size_1,
                "], type = ", type);
    }
}


void advance_step_flashattn_ascendc(
    int64_t num_seqs, int64_t num_queries, int64_t block_size,
    at::Tensor& input_tokens,
    at::Tensor& sampled_token_ids,
    at::Tensor& input_positions,
    at::Tensor& seq_lens,
    at::Tensor& slot_mapping,
    at::Tensor& block_tables
){
    // Verify all tensors
    verify_tensor("input_tokens", input_tokens, num_seqs, -1, at::kLong);
    verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1,at::kLong);
    verify_tensor("input_positions", input_positions, num_seqs, -1, at::kLong);
    verify_tensor("seq_lens", seq_lens, num_seqs, -1, at::kInt);
    verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, at::kInt);
    verify_tensor("block_tables", block_tables, num_seqs, -1, at::kInt);


    int64_t* input_tokens_ptr = input_tokens.data_ptr<int64_t>();
    int64_t* sampled_token_ids_ptr = sampled_token_ids.data_ptr<int64_t>();
    int64_t* input_positions_ptr = input_positions.data_ptr<int64_t>();
    int32_t* seq_lens_ptr = seq_lens.data_ptr<int32_t>();
    int32_t* slot_mapping_ptr = slot_mapping.data_ptr<int32_t>();
    int32_t* block_tables_ptr =  block_tables.data_ptr<int32_t>();


    int32_t device_id;
    aclrtGetDevice(&device_id);
    auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
    aclrtStream stream = npu_stream.stream();

    // aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("advance_step_flashattn_ascendc");
    cmd.SetCustomHandler([stream, num_seqs, num_queries,
                          block_size, input_tokens_ptr, sampled_token_ids_ptr,
                          input_positions_ptr, seq_lens_ptr, slot_mapping_ptr,
                          block_tables_ptr, block_tables]() -> int {
        launch_advance_step_flashattn(stream,
                                    num_seqs,
                                    num_queries,
                                    block_size,
                                    input_tokens_ptr,
                                    sampled_token_ids_ptr,
                                    input_positions_ptr,
                                    seq_lens_ptr,
                                    slot_mapping_ptr,
                                    block_tables_ptr,
                                    block_tables.stride(0));
        return 0;
    });
    cmd.Run();
    return ;
}
} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(_C, ops)
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
        "advance_step_flashattn_ascendc(int num_seqs, int num_queries, int block_size,"
        "                               Tensor! input_tokens, Tensor! sampled_token_ids, Tensor! input_positions,"
        "                               Tensor! seq_lens, Tensor! slot_mapping, Tensor! block_tables) -> ()");
    ops.impl("advance_step_flashattn_ascendc", torch::kPrivateUse1, &vllm_ascend::advance_step_flashattn_ascendc);
}

REGISTER_EXTENSION(_C)
