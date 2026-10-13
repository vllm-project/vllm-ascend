/**
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

#pragma once

namespace vllm_ascend {

constexpr int64_t GBSA_METADATA_SIZE = 1024;
constexpr int64_t GBSA_NO_QUANT = 0;

at::Tensor npu_generic_block_sparse_attention_metadata(
    const at::Tensor &sparse_block_idx, const at::Tensor &sparse_block_count,
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    at::IntArrayRef block_shape,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    c10::string_view layout_q, c10::string_view layout_kv,
    int64_t layout_sparse_pattern, int64_t mask_mode, int64_t quant_mode,
    int64_t softmax_precision, int64_t win_left, int64_t win_right,
    int64_t residual_block_mode, bool is_consistent_topk)
{
    auto output = at::empty(
        {GBSA_METADATA_SIZE}, sparse_block_idx.options().dtype(at::kInt));
    std::string layout_q_str(layout_q);
    std::string layout_kv_str(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(aclnnGenericBlockSparseAttentionMetadata,
                 sparse_block_idx, sparse_block_count, cu_seqlens_q,
                 cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q,
                 max_seqlen_kv, num_heads_q, num_heads_kv, head_dim,
                 block_shape, layout_q_ptr, layout_kv_ptr,
                 layout_sparse_pattern, mask_mode, quant_mode,
                 softmax_precision, win_left, win_right,
                 residual_block_mode, is_consistent_topk, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> npu_generic_block_sparse_attention(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &sparse_block_idx, const at::Tensor &sparse_block_count,
    at::IntArrayRef block_shape,
    const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &q_dequant_scale,
    const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale,
    const c10::optional<at::Tensor> &p_quant_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &block_table,
    c10::string_view layout_q, c10::string_view layout_kv,
    int64_t layout_sparse_pattern, double softmax_scale,
    int64_t mask_mode, int64_t quant_mode, double dst_type_max,
    int64_t softmax_precision, int64_t win_left, int64_t win_right,
    bool return_softmax_lse, int64_t residual_block_mode,
    bool is_consistent_topk,
    c10::optional<at::ScalarType> attention_out_dtype)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.");
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.");
    TORCH_CHECK(v.numel() > 0, "Tensor v is empty.");
    TORCH_CHECK(sparse_block_idx.numel() > 0,
                "Tensor sparse_block_idx is empty.");
    TORCH_CHECK(sparse_block_count.numel() > 0,
                "Tensor sparse_block_count is empty.");
    TORCH_CHECK(block_shape.size() == 2,
                "block_shape must contain [query_block_size, kv_block_size].");

    at::ScalarType output_dtype;
    if (attention_out_dtype.has_value()) {
        output_dtype = attention_out_dtype.value();
    } else {
        TORCH_CHECK(quant_mode == GBSA_NO_QUANT,
                    "attention_out_dtype must be specified when quant_mode != 0");
        output_dtype = q.scalar_type();
    }

    auto attention_out = at::empty(q.sizes(), q.options().dtype(output_dtype));
    std::vector<int64_t> lse_shape{0};
    if (return_softmax_lse) {
        const auto layout_q_str = std::string(layout_q);
        if (layout_q_str == "TND") {
            lse_shape = {q.size(0), q.size(1), 1};
        } else if (layout_q_str == "BNSD") {
            lse_shape = {q.size(0), q.size(1), q.size(2), 1};
        } else {
            lse_shape = {q.size(0), q.size(2), q.size(1), 1};
        }
    }
    auto softmax_lse = at::empty(lse_shape, q.options().dtype(at::kFloat));

    std::string layout_q_str(layout_q);
    std::string layout_kv_str(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    const int64_t return_softmax_lse_flag = return_softmax_lse ? 1 : 0;
    EXEC_NPU_CMD(aclnnGenericBlockSparseAttention,
                 q, k, v, sparse_block_idx, sparse_block_count, metadata,
                 attn_mask, q_dequant_scale, k_dequant_scale, v_dequant_scale,
                 p_quant_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q,
                 seqused_kv, block_table, block_shape,
                 layout_q_ptr, layout_kv_ptr,
                 layout_sparse_pattern, softmax_scale, mask_mode, quant_mode,
                 dst_type_max, softmax_precision, win_left, win_right,
                 return_softmax_lse_flag, residual_block_mode,
                 is_consistent_topk, attention_out, softmax_lse);
    return {attention_out, softmax_lse};
}

}  // namespace vllm_ascend
