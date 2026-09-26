// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace vllm_ascend {
inline std::tuple<at::Tensor, at::Tensor> flash_attn_c8_meta(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &query_rope, const at::Tensor &key_rope,
    const at::Tensor &dequant_scale_query, const at::Tensor &dequant_scale_key,
    const at::Tensor &dequant_scale_value, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_kv, const at::Tensor &metadata,
    double softmax_scale, int64_t mask_mode, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &attn_mask,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes(),
                "FlashAttnC8 requires TND q/k and a separate matching value tensor");
    TORCH_CHECK(q.size(2) == 128 && k.size(2) == 128 && q.size(1) == k.size(1),
                "FlashAttnC8 requires matching heads and non-RoPE/value dimension128");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "FlashAttnC8 q/k/v must be FP8 E4M3FN");
    TORCH_CHECK(query_rope.dim() == 3 && key_rope.dim() == 3 &&
                query_rope.size(0) == q.size(0) && key_rope.size(0) == k.size(0) &&
                query_rope.size(1) == q.size(1) && key_rope.size(1) == k.size(1) &&
                query_rope.size(2) == 64 && key_rope.size(2) == 64 &&
                query_rope.scalar_type() == at::ScalarType::BFloat16 &&
                key_rope.scalar_type() == at::ScalarType::BFloat16,
                "FlashAttnC8 requires BF16 per-head query/key RoPE64");
    TORCH_CHECK(dequant_scale_query.dim() == 2 &&
                dequant_scale_query.size(0) == q.size(0) && dequant_scale_query.size(1) == q.size(1) &&
                dequant_scale_key.dim() == 1 && dequant_scale_key.size(0) == q.size(1) &&
                dequant_scale_value.sizes() == dequant_scale_key.sizes() &&
                dequant_scale_query.scalar_type() == at::kFloat &&
                dequant_scale_key.scalar_type() == at::kFloat && dequant_scale_value.scalar_type() == at::kFloat,
                "FlashAttnC8 scales must be FP32 query[T,H], key[H], value[H]");
    TORCH_CHECK(mask_mode == 0 || mask_mode == 3, "FlashAttnC8 supports mask modes0/3");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.numel() >= 2 &&
                cu_seqlens_kv.sizes() == cu_seqlens_q.sizes() &&
                cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_kv.scalar_type() == at::kInt,
                "FlashAttnC8 requires matching INT32 cumulative lengths with leading zero");
    TORCH_CHECK(metadata.dim() == 1 && metadata.scalar_type() == at::kInt && metadata.numel() >= 4096,
                "FlashAttnC8 requires FlashAttnMetadata QK192/V128 M128/N128 metadata");
    auto output = at::empty_symint(q.sym_sizes(), q.options().dtype(at::kBFloat16));
    auto lse = return_softmax_lse
        ? at::empty_symint({q.sym_size(1), q.sym_size(0)}, q.options().dtype(at::kFloat))
        : at::empty({0}, q.options().dtype(at::kFloat));
    return {output, lse};
}

inline std::tuple<at::Tensor, at::Tensor> flash_attn_c8(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &query_rope, const at::Tensor &key_rope,
    const at::Tensor &dequant_scale_query, const at::Tensor &dequant_scale_key,
    const at::Tensor &dequant_scale_value, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_kv, const at::Tensor &metadata,
    double softmax_scale, int64_t mask_mode, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &attn_mask,
    bool return_softmax_lse)
{
    auto outputs = flash_attn_c8_meta(q, k, v, query_rope, key_rope, dequant_scale_query,
        dequant_scale_key, dequant_scale_value, cu_seqlens_q, cu_seqlens_kv, metadata,
        softmax_scale, mask_mode, max_seqlen_q, max_seqlen_kv, seqused_q, attn_mask, return_softmax_lse);
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "FlashAttnC8 requires NPU tensors");
    for (const auto *tensor : {&k, &v, &query_rope, &key_rope, &dequant_scale_query,
                              &dequant_scale_key, &dequant_scale_value, &cu_seqlens_q,
                              &cu_seqlens_kv, &metadata}) {
        TORCH_CHECK(tensor->device() == q.device(), "FlashAttnC8 tensors must share the same NPU");
    }
    if (seqused_q.has_value()) {
        TORCH_CHECK(seqused_q->device() == q.device() && seqused_q->scalar_type() == at::kInt &&
                    seqused_q->dim() == 1 && seqused_q->numel() + 1 == cu_seqlens_q.numel(),
                    "FlashAttnC8 seqused_q must be INT32[batch] on the same NPU");
    }
    if (mask_mode == 3) {
        TORCH_CHECK(attn_mask.has_value() && attn_mask->device() == q.device() &&
                    attn_mask->scalar_type() == at::kChar && attn_mask->dim() == 2 &&
                    attn_mask->size(0) == 2048 && attn_mask->size(1) == 2048,
                    "FlashAttnC8 causal attention requires INT8[2048,2048] mask");
    }
    if (max_seqlen_q < 0) max_seqlen_q = q.size(0);
    if (max_seqlen_kv < 0) max_seqlen_kv = k.size(0);
    EXEC_NPU_CMD(aclnnInnerFlashAttnC8, q, k, v, query_rope, key_rope, dequant_scale_query,
        dequant_scale_key, dequant_scale_value, cu_seqlens_q, cu_seqlens_kv, seqused_q,
        attn_mask, metadata, softmax_scale, mask_mode, max_seqlen_q, max_seqlen_kv,
        return_softmax_lse, std::get<0>(outputs), std::get<1>(outputs));
    return outputs;
}
}
