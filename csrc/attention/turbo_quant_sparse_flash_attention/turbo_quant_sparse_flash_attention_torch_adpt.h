/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef TURBOQUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define TURBOQUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor> turboquant_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    double scale_value, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t sparse_block_size, c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t attention_mode,
    int64_t quant_scale_repo_mode, int64_t tile_size, int64_t rope_head_dim,
    bool return_softmax_lse)
{
    std::string layout_query_str(layout_query);
    std::string layout_kv_str(layout_kv);
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    auto query_shape = query.sizes();
    at::Tensor output = at::empty(
        {query_shape[0], query_shape[1], query_shape[2] - rope_head_dim}, query.options());

    // TQ SFA is TND-layout only; mirror kv_quant's TND softmax_lse shape so the DCP
    // decode merge can consume softmax_max/softmax_sum. Empty {0} when not requested.
    at::SmallVector<int64_t, 8> softmax_size;
    if (return_softmax_lse) {
        const int64_t kv_head_dim =
            layout_kv_str == "PA_BSND" ? key.size(2) : key.size(1);
        softmax_size = {kv_head_dim, query.size(0), query.size(1) / kv_head_dim};
    } else {
        softmax_size = {0};
    }
    at::Tensor softmax_max = at::empty(softmax_size, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum = at::empty(softmax_size, query.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnTurboQuantSparseFlashAttention,
                 query,
                 key,
                 value,
                 sparse_indices,
                 key_dequant_scale,
                 value_dequant_scale,
                 block_table,
                 actual_seq_lengths_query,
                 actual_seq_lengths_kv,
                 scale_value,
                 key_quant_mode,
                 value_quant_mode,
                 sparse_block_size,
                 layout_query_ptr,
                 layout_kv_ptr,
                 sparse_mode,
                 pre_tokens,
                 next_tokens,
                 attention_mode,
                 quant_scale_repo_mode,
                 tile_size,
                 rope_head_dim,
                 return_softmax_lse,
                 output,
                 softmax_max,
                 softmax_sum);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, softmax_max, softmax_sum);
}

} // namespace vllm_ascend
#endif
