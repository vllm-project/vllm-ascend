/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef SPARSE_FLASH_MLA_TORCH_ADPT_H
#define SPARSE_FLASH_MLA_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

std::tuple<at::Tensor, at::Tensor> construct_sparse_flash_mla_output(
    const at::Tensor &query, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, c10::string_view layout_q,
    c10::string_view layout_kv, bool return_softmax_lse)
{
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t QUERY_TND_DIM = 3;
    constexpr int64_t QUERY_BSND_DIM = 4;

    const std::string layout_q_str(layout_q);
    const std::string layout_kv_str(layout_kv);
    TORCH_CHECK(layout_q_str == "BSND" || layout_q_str == "TND",
                "layout_q must be BSND or TND, but got ", layout_q_str);
    TORCH_CHECK((layout_q_str == "TND" && query.dim() == QUERY_TND_DIM) ||
                    (layout_q_str == "BSND" && query.dim() == QUERY_BSND_DIM),
                "query dimension does not match layout_q ", layout_q_str,
                ", got dimension ", query.dim());
    TORCH_CHECK(ori_kv.has_value() || cmp_kv.has_value(),
                "At least one of ori_kv and cmp_kv must be provided.");

    at::Tensor attention_output = at::empty(query.sizes(), query.options());
    if (!return_softmax_lse) {
        return {attention_output,
                at::empty({0}, query.options().dtype(at::kFloat))};
    }

    const at::Tensor &kv = ori_kv.has_value() ? *ori_kv : *cmp_kv;
    TORCH_CHECK(kv.dim() > DIM_2,
                "KV tensor must have at least 3 dimensions, but got ",
                kv.dim());
    const int64_t kv_head_num =
        layout_kv_str == "TND" ? kv.size(DIM_1) : kv.size(DIM_2);
    TORCH_CHECK(kv_head_num > 0, "KV head count must be positive.");

    at::SmallVector<int64_t, 4> softmax_lse_size;
    if (query.dim() == QUERY_TND_DIM) {
        softmax_lse_size = {kv_head_num, query.size(DIM_0),
                            query.size(DIM_1) / kv_head_num};
    } else {
        softmax_lse_size = {query.size(DIM_0), kv_head_num,
                            query.size(DIM_1),
                            query.size(DIM_2) / kv_head_num};
    }
    return {attention_output,
            at::empty(softmax_lse_size,
                      query.options().dtype(at::kFloat))};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_mla(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv,
    const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length,
    const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, double softmax_scale,
    int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q,
    c10::string_view layout_kv, int64_t topk_value_mode,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.");
    auto outputs = construct_sparse_flash_mla_output(
        q, ori_kv, cmp_kv, layout_q, layout_kv, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(outputs);
    at::Tensor softmax_lse = std::get<1>(outputs);

    std::string layout_q_str(layout_q);
    std::string layout_kv_str(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnSparseFlashMla, q, ori_kv, cmp_kv, ori_sparse_indices,
        cmp_sparse_indices, ori_block_table, cmp_block_table, cu_seqlens_q,
        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv,
        seqused_cmp_kv, cmp_residual_kv, ori_topk_length, cmp_topk_length,
        sinks, metadata, softmax_scale, cmp_ratio, ori_mask_mode,
        cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr,
        layout_kv_ptr, topk_value_mode, return_softmax_lse, attn_out,
        softmax_lse);
    return {attn_out, softmax_lse};
}

}  // namespace vllm_ascend

#endif  // SPARSE_FLASH_MLA_TORCH_ADPT_H
