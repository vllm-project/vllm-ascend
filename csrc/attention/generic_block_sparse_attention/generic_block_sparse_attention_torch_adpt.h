/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_TORCH_ADPT_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_TORCH_ADPT_H
#include "generic_block_sparse_attention_output.h"

namespace vllm_ascend {
namespace gbsa_detail {
// ACLNN executors may retain attribute pointers until graph capture ends.
// String literals keep the layouts alive beyond this adapter invocation.
inline char* LayoutAttr(c10::string_view layout)
{
    if (layout == "TND") return const_cast<char*>("TND");
    if (layout == "BNSD") return const_cast<char*>("BNSD");
    if (layout == "BSND") return const_cast<char*>("BSND");
    if (layout == "PA_BBND") return const_cast<char*>("PA_BBND");
    if (layout == "PA_BNBD") return const_cast<char*>("PA_BNBD");
    TORCH_CHECK(false, "unsupported GBSA layout: ", layout);
}
} // namespace gbsa_detail

at::Tensor npu_generic_block_sparse_attention_metadata(
    const at::Tensor& sparse_block_idx, const at::Tensor& sparse_block_count,
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, at::IntArrayRef block_shape,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& cu_seqlens_kv,
    const c10::optional<at::Tensor>& seqused_q, const c10::optional<at::Tensor>& seqused_kv, int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    c10::string_view layout_q, c10::string_view layout_kv, int64_t layout_sparse_pattern, int64_t mask_mode, int64_t quant_mode,
    int64_t softmax_precision, int64_t win_left, int64_t win_right, int64_t residual_block_mode,
    bool is_consistent_topk)
{
    auto output = at::empty({gbsa_detail::METADATA_SIZE}, sparse_block_idx.options().dtype(at::kInt));
    auto layout_q_attr = gbsa_detail::LayoutAttr(layout_q);
    auto layout_kv_attr = gbsa_detail::LayoutAttr(layout_kv);
    EXEC_NPU_CMD(aclnnGenericBlockSparseAttentionMetadata, sparse_block_idx, sparse_block_count, cu_seqlens_q,
              cu_seqlens_kv, seqused_q, seqused_kv, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim,
              block_shape, layout_q_attr, layout_kv_attr, layout_sparse_pattern, mask_mode, quant_mode, softmax_precision,
              win_left, win_right, residual_block_mode, is_consistent_topk, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> npu_generic_block_sparse_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& sparse_block_idx,
    const at::Tensor& sparse_block_count, at::IntArrayRef block_shape, const c10::optional<at::Tensor>& metadata,
    const c10::optional<at::Tensor>& attn_mask, const c10::optional<at::Tensor>& q_dequant_scale,
    const c10::optional<at::Tensor>& k_dequant_scale, const c10::optional<at::Tensor>& v_dequant_scale,
    const c10::optional<at::Tensor>& p_quant_scale, const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_kv, const c10::optional<at::Tensor>& seqused_q,
    const c10::optional<at::Tensor>& seqused_kv, const c10::optional<at::Tensor>& block_table, c10::string_view layout_q,
    c10::string_view layout_kv, int64_t layout_sparse_pattern, double softmax_scale, int64_t mask_mode, int64_t quant_mode,
    double dst_type_max, int64_t softmax_precision, int64_t win_left, int64_t win_right, bool return_softmax_lse,
    int64_t residual_block_mode, bool is_consistent_topk, c10::optional<at::ScalarType> attention_out_dtype)
{
    auto [attention_out, softmax_lse] = gbsa_detail::MakeOutputs(
        q, layout_q, quant_mode, return_softmax_lse, attention_out_dtype);
    auto layout_q_attr = gbsa_detail::LayoutAttr(layout_q);
    auto layout_kv_attr = gbsa_detail::LayoutAttr(layout_kv);
    const int64_t return_softmax_lse_flag = return_softmax_lse ? 1 : 0;
    EXEC_NPU_CMD(aclnnGenericBlockSparseAttention, q, k, v, sparse_block_idx, sparse_block_count, metadata, attn_mask,
              q_dequant_scale, k_dequant_scale, v_dequant_scale, p_quant_scale, cu_seqlens_q, cu_seqlens_kv, seqused_q,
              seqused_kv, block_table, block_shape, layout_q_attr, layout_kv_attr, layout_sparse_pattern, softmax_scale,
              mask_mode, quant_mode, dst_type_max, softmax_precision, win_left, win_right, return_softmax_lse_flag,
              residual_block_mode, is_consistent_topk, attention_out, softmax_lse);
    return std::make_tuple(attention_out, softmax_lse);
}
} // namespace vllm_ascend
#endif
