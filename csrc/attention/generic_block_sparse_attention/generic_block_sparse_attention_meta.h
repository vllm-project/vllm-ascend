/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_META_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_META_H
#include "generic_block_sparse_attention_output.h"

namespace vllm_ascend::meta {
at::Tensor npu_generic_block_sparse_attention_metadata_meta(
    const at::Tensor& sparse_block_idx, const at::Tensor& sparse_block_count,
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, at::IntArrayRef block_shape,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& cu_seqlens_kv,
    const c10::optional<at::Tensor>& seqused_q, const c10::optional<at::Tensor>& seqused_kv, int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    c10::string_view layout_q, c10::string_view layout_kv, int64_t layout_sparse_pattern, int64_t mask_mode, int64_t quant_mode,
    int64_t softmax_precision, int64_t win_left, int64_t win_right, int64_t residual_block_mode,
    bool is_consistent_topk)
{
    return at::empty({gbsa_detail::METADATA_SIZE}, sparse_block_idx.options().dtype(at::kInt));
}

std::tuple<at::Tensor, at::Tensor> npu_generic_block_sparse_attention_meta(
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
    return gbsa_detail::MakeOutputs(q, layout_q, quant_mode, return_softmax_lse, attention_out_dtype);
}
} // namespace vllm_ascend::meta
#endif
