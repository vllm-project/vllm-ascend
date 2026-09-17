// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#pragma once

namespace vllm_ascend {

namespace {

constexpr int64_t FLASH_MLA_AIC_CORE_NUM = 36;
constexpr int64_t FLASH_MLA_AIV_CORE_NUM = 72;
constexpr int64_t FLASH_MLA_META_WORDS_PER_CORE = 16;
constexpr int64_t FLASH_MLA_META_ALIGNMENT = 4096;

int64_t flash_mla_metadata_numel(int64_t batch_size)
{
    TORCH_CHECK(batch_size > 0, "FlashMLA metadata requires a positive batch size, got ", batch_size);
    const int64_t words =
        ((FLASH_MLA_AIC_CORE_NUM + FLASH_MLA_AIV_CORE_NUM) * batch_size + 1) *
        FLASH_MLA_META_WORDS_PER_CORE;
    return ((words + FLASH_MLA_META_ALIGNMENT - 1) / FLASH_MLA_META_ALIGNMENT) *
           FLASH_MLA_META_ALIGNMENT;
}

std::tuple<at::Tensor, at::Tensor> construct_flash_mla_output(
    const at::Tensor &query,
    int64_t head_dim_v,
    c10::string_view layout_q,
    const c10::optional<c10::string_view> &layout_out,
    bool return_softmax_lse)
{
    const std::string layout_q_str(layout_q);
    const std::string layout_out_str = layout_out.has_value()
        ? std::string(layout_out.value())
        : layout_q_str;
    TORCH_CHECK(layout_out_str == layout_q_str ||
                (layout_q_str == "TND" && layout_out_str == "NTD"),
                "FlashMLA layout_out must equal layout_q (", layout_q_str,
                "), got ", layout_out_str);
    TORCH_CHECK(head_dim_v > 0, "FlashMLA head_dim_v must be positive, got ", head_dim_v);

    at::Tensor output;
    at::Tensor softmax_lse;
    const auto float_options = query.options().dtype(at::ScalarType::Float);
    if (layout_q_str == "TND") {
        TORCH_CHECK(query.dim() == 3, "FlashMLA TND query must be rank 3, got ", query.dim());
        output = layout_out_str == "NTD"
            ? at::empty_symint({query.sym_size(1), query.sym_size(0), c10::SymInt(head_dim_v)}, query.options())
            : at::empty_symint({query.sym_size(0), query.sym_size(1), c10::SymInt(head_dim_v)}, query.options());
        if (return_softmax_lse) {
            softmax_lse = at::empty_symint({query.sym_size(1), query.sym_size(0)}, float_options);
        }
    } else if (layout_q_str == "BSND") {
        TORCH_CHECK(query.dim() == 4, "FlashMLA BSND query must be rank 4, got ", query.dim());
        output = at::empty_symint(
            {query.sym_size(0), query.sym_size(1), query.sym_size(2), c10::SymInt(head_dim_v)},
            query.options());
        if (return_softmax_lse) {
            softmax_lse = at::empty_symint(
                {query.sym_size(0), query.sym_size(2), query.sym_size(1)}, float_options);
        }
    } else if (layout_q_str == "BNSD") {
        TORCH_CHECK(query.dim() == 4, "FlashMLA BNSD query must be rank 4, got ", query.dim());
        output = at::empty_symint(
            {query.sym_size(0), query.sym_size(1), query.sym_size(2), c10::SymInt(head_dim_v)},
            query.options());
        if (return_softmax_lse) {
            softmax_lse = at::empty_symint(
                {query.sym_size(0), query.sym_size(1), query.sym_size(2)}, float_options);
        }
    } else {
        TORCH_CHECK(false, "FlashMLA only supports TND, BSND, or BNSD query layout, got ", layout_q_str);
    }
    if (!return_softmax_lse) {
        softmax_lse = at::empty_symint({c10::SymInt(0)}, float_options);
    }
    return {output, softmax_lse};
}

} // namespace

at::Tensor flash_mla_with_kvcache_metadata(
    const at::Tensor &cache_seqlens,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &seqused_q,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t head_dim_qk,
    int64_t head_dim_v,
    int64_t mask_mode,
    c10::string_view layout_q)
{
    TORCH_CHECK(cache_seqlens.dim() == 1,
                "FlashMLA cache_seqlens must be rank 1, got ", cache_seqlens.dim());
    TORCH_CHECK(num_heads_kv == 1,
                "FlashMLA requires exactly one KV head, got ", num_heads_kv);
    at::Tensor metadata = at::empty(
        {flash_mla_metadata_numel(cache_seqlens.size(0))},
        cache_seqlens.options().dtype(at::ScalarType::Int));
    std::string layout_q_str(layout_q);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    EXEC_NPU_CMD(aclnnFlashMlaWithKvcacheMetadata,
                 cu_seqlens_q, cache_seqlens, seqused_q,
                 max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv,
                 head_dim_qk, head_dim_v, mask_mode, layout_q_ptr, metadata);
    return metadata;
}

at::Tensor flash_mla_with_kvcache_metadata_meta(
    const at::Tensor &cache_seqlens,
    int64_t num_heads_q,
    int64_t num_heads_kv,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &seqused_q,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t head_dim_qk,
    int64_t head_dim_v,
    int64_t mask_mode,
    c10::string_view layout_q)
{
    TORCH_CHECK(num_heads_kv == 1,
                "FlashMLA requires exactly one KV head, got ", num_heads_kv);
    return at::empty(
        {flash_mla_metadata_numel(cache_seqlens.size(0))},
        cache_seqlens.options().dtype(at::ScalarType::Int));
}

std::tuple<at::Tensor, at::Tensor> flash_mla_with_kvcache(
    const at::Tensor &query,
    const at::Tensor &k_cache,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &cache_seqlens,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata,
    int64_t head_dim_v,
    double softmax_scale,
    int64_t mask_mode,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    const c10::optional<c10::string_view> &layout_out,
    bool return_softmax_lse)
{
    auto outputs = construct_flash_mla_output(
        query, head_dim_v, layout_q, layout_out, return_softmax_lse);
    at::Tensor &attention_output = std::get<0>(outputs);
    at::Tensor &softmax_lse = std::get<1>(outputs);

    if (metadata.has_value() && metadata.value().defined()) {
        TORCH_CHECK(metadata.value().scalar_type() == at::ScalarType::Int,
                    "FlashMLA metadata must be int32");
        TORCH_CHECK(metadata.value().numel() >= FLASH_MLA_META_ALIGNMENT,
                    "FlashMLA metadata must be produced by flash_mla_with_kvcache_metadata");
    }
    std::string layout_q_str(layout_q);
    std::string layout_kv_str(layout_kv);
    std::string layout_out_str = layout_out.has_value()
        ? std::string(layout_out.value())
        : layout_q_str;
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());
    int64_t return_softmax_lse_i64 = return_softmax_lse ? 1 : 0;

    EXEC_NPU_CMD(aclnnFlashMlaWithKvcache,
                 query, k_cache, block_table, cache_seqlens, cu_seqlens_q,
                 seqused_q, attn_mask, metadata, head_dim_v, softmax_scale,
                 mask_mode, max_seqlen_q, max_seqlen_kv, layout_q_ptr,
                 layout_kv_ptr, layout_out_ptr, return_softmax_lse_i64,
                 attention_output, softmax_lse);
    return outputs;
}

std::tuple<at::Tensor, at::Tensor> flash_mla_with_kvcache_meta(
    const at::Tensor &query,
    const at::Tensor &k_cache,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &cache_seqlens,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata,
    int64_t head_dim_v,
    double softmax_scale,
    int64_t mask_mode,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    c10::string_view layout_q,
    c10::string_view layout_kv,
    const c10::optional<c10::string_view> &layout_out,
    bool return_softmax_lse)
{
    return construct_flash_mla_output(
        query, head_dim_v, layout_q, layout_out, return_softmax_lse);
}

} // namespace vllm_ascend
