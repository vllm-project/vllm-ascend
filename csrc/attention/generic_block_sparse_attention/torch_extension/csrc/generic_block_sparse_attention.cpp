/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// Keep values in sync with Python QuantMode / aclnn quantType.
enum QuantMode : int64_t {
    NO_QUANT = 0,
    FP8_E4M3_STATIC_PER_GROUP = 1,
    FP8_E4M3_DYNAMIC_MX = 2,
    FP4_E2M1_DYNAMIC_OCP = 3,
    FP4_E2M1_DYNAMIC_CX = 4,
    FP8_E4M3_STATIC_CAST_P = 5,
};

at::Tensor GenericBlockSparseAttentionMetadata(
    const at::Tensor &sparse_block_idx, const at::Tensor &sparse_block_count,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv, int64_t max_seqlen_q,
    int64_t max_seqlen_kv, int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, at::IntArrayRef block_shape,
    std::string layout_q, std::string layout_kv, int64_t layout_sparse_pattern, int64_t mask_mode, int64_t quant_mode,
    int64_t softmax_precision, int64_t win_left, int64_t win_right, int64_t residual_block_mode,
    bool is_consistent_topk, const at::Tensor &output)
{
    ACLNN_CMD(aclnnGenericBlockSparseAttentionMetadata, sparse_block_idx, sparse_block_count, cu_seqlens_q,
              cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim,
              block_shape, layout_q, layout_kv, layout_sparse_pattern, mask_mode, quant_mode, softmax_precision,
              win_left, win_right, residual_block_mode, is_consistent_topk, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> GenericBlockSparseAttention(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &sparse_block_idx,
    const at::Tensor &sparse_block_count, std::vector<int64_t> block_shape, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &attn_mask, const c10::optional<at::Tensor> &q_dequant_scale,
    const c10::optional<at::Tensor> &k_dequant_scale, const c10::optional<at::Tensor> &v_dequant_scale,
    const c10::optional<at::Tensor> &p_quant_scale, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, const c10::optional<at::Tensor> &block_table, std::string layout_q,
    std::string layout_kv, int64_t layout_sparse_pattern, double softmax_scale, int64_t mask_mode, int64_t quant_mode,
    double dst_type_max, int64_t softmax_precision, int64_t win_left, int64_t win_right, bool return_softmax_lse,
    int64_t residual_block_mode, bool is_consistent_topk, c10::optional<at::ScalarType> attention_out_dtype)
{
    // pybind 按值收下 std::vector，再在本作用域构建 IntArrayRef，避免悬垂指针。
    at::IntArrayRef block_shape_ref(block_shape);

    // Resolve output dtype: an explicit value passes through unchanged.
    // A missing value (None) only falls back to the input dtype for
    // quant_mode=NO_QUANT; quantized paths require it explicitly.
    at::ScalarType out_dtype;
    if (attention_out_dtype.has_value()) {
        out_dtype = *attention_out_dtype;
    } else if (quant_mode == NO_QUANT) {
        out_dtype = q.scalar_type();
    } else {
        TORCH_CHECK(false, "attention_out_dtype must be specified when quant_mode != NO_QUANT");
    }

    at::Tensor attention_out = at::empty(q.sizes(), q.options().dtype(out_dtype));

    at::Tensor softmax_lse;
    auto opts_f32 = q.options().dtype(at::kFloat);
    if (return_softmax_lse) {
        if (layout_q == "TND") {
            softmax_lse = at::empty({q.size(0), q.size(1), 1}, opts_f32);
        } else if (layout_q == "BNSD") {
            softmax_lse = at::empty({q.size(0), q.size(1), q.size(2), 1}, opts_f32);
        } else {
            // BSND
            softmax_lse = at::empty({q.size(0), q.size(2), q.size(1), 1}, opts_f32);
        }
    } else {
        softmax_lse = at::empty({0}, opts_f32);
    }

    const int64_t return_softmax_lse_flag = return_softmax_lse ? 1 : 0;
    ACLNN_CMD(aclnnGenericBlockSparseAttention, q, k, v, sparse_block_idx, sparse_block_count, metadata, attn_mask,
              q_dequant_scale, k_dequant_scale, v_dequant_scale, p_quant_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q,
              seqused_kv, block_table, block_shape_ref, layout_q, layout_kv, layout_sparse_pattern, softmax_scale,
              mask_mode, quant_mode, dst_type_max, softmax_precision, win_left, win_right, return_softmax_lse_flag,
              residual_block_mode, is_consistent_topk, attention_out, softmax_lse);
    return std::make_tuple(attention_out, softmax_lse);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generic_block_sparse_attention_metadata", &op_api::GenericBlockSparseAttentionMetadata,
          "generic_block_sparse_attention_metadata");
    m.def("generic_block_sparse_attention", &op_api::GenericBlockSparseAttention, "generic_block_sparse_attention");
}
