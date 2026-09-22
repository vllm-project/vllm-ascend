// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace vllm_ascend {
inline std::tuple<at::Tensor, at::Tensor> gather_mla_prefill_meta(
    const at::Tensor &latent_cache, const at::Tensor &rope_cache,
    const at::Tensor &block_table, const at::Tensor &cumulative_lengths,
    const at::Tensor &lengths, const at::Tensor &starts, const at::Tensor &scale,
    int64_t num_tokens, int64_t max_seq_len)
{
    TORCH_CHECK(latent_cache.dim() == 4 && rope_cache.dim() == 4 &&
        latent_cache.size(0) == rope_cache.size(0) && latent_cache.size(1) == 128 &&
        rope_cache.size(1) == 128 && latent_cache.size(2) == 1 && rope_cache.size(2) == 1 &&
        latent_cache.size(3) == 512 && rope_cache.size(3) == 64 &&
        latent_cache.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        rope_cache.scalar_type() == at::kBFloat16 &&
        latent_cache.stride(3) == 1 && rope_cache.stride(3) == 1,
        "GatherMlaPrefill requires page128 FP8 latent512/BF16 rope64 with dense final dimensions");
    TORCH_CHECK(block_table.dim() == 2 && cumulative_lengths.dim() == 1 &&
        lengths.dim() == 1 && starts.dim() == 1 && block_table.size(0) == lengths.numel() &&
        starts.numel() == lengths.numel() && cumulative_lengths.numel() == lengths.numel() + 1 &&
        block_table.scalar_type() == at::kInt && cumulative_lengths.scalar_type() == at::kInt &&
        lengths.scalar_type() == at::kInt && starts.scalar_type() == at::kInt &&
        scale.scalar_type() == at::kFloat && scale.numel() == 1,
        "GatherMlaPrefill requires INT32 request metadata and one FP32 cache scale");
    TORCH_CHECK(num_tokens >= 0 && max_seq_len >= 0 &&
        (num_tokens == 0 || (lengths.numel() > 0 && max_seq_len > 0)),
        "GatherMlaPrefill invalid output size or launch bound");
    return {at::empty({num_tokens, 1, 512}, latent_cache.options().dtype(at::kBFloat16)),
            at::empty({num_tokens, 1, 64}, latent_cache.options().dtype(at::kBFloat16))};
}

inline std::tuple<at::Tensor, at::Tensor> gather_mla_prefill(
    const at::Tensor &latent_cache, const at::Tensor &rope_cache,
    const at::Tensor &block_table, const at::Tensor &cumulative_lengths,
    const at::Tensor &lengths, const at::Tensor &starts, const at::Tensor &scale,
    int64_t num_tokens, int64_t max_seq_len)
{
    auto outputs = gather_mla_prefill_meta(latent_cache, rope_cache, block_table,
        cumulative_lengths, lengths, starts, scale, num_tokens, max_seq_len);
    TORCH_CHECK(latent_cache.device().type() == c10::DeviceType::PrivateUse1,
        "GatherMlaPrefill requires NPU tensors");
    for (const auto *tensor : {&rope_cache, &block_table, &cumulative_lengths, &lengths, &starts, &scale}) {
        TORCH_CHECK(tensor->device() == latent_cache.device(), "GatherMlaPrefill tensors must share one NPU");
    }
    if (num_tokens == 0) return outputs;
    const int64_t latent_page_stride = latent_cache.stride(0);
    const int64_t latent_row_stride = latent_cache.stride(1);
    const int64_t rope_page_stride = rope_cache.stride(0);
    const int64_t rope_row_stride = rope_cache.stride(1);
    EXEC_NPU_CMD(aclnnInnerGatherMlaPrefill, latent_cache, rope_cache, block_table,
        cumulative_lengths, lengths, starts, scale, num_tokens, max_seq_len,
        latent_page_stride, latent_row_stride, rope_page_stride, rope_row_stride,
        std::get<0>(outputs), std::get<1>(outputs));
    return outputs;
}
} // namespace vllm_ascend
