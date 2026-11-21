#ifndef KVTC_PREPARE_ROPE_TORCH_ADPT_H
#define KVTC_PREPARE_ROPE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor kvtc_prepare_rope(at::TensorList caches, const at::Tensor& block_ids, const at::Tensor& token_starts,
                             const at::Tensor& inv_cos_sin_cache)
{
    TORCH_CHECK(!caches.empty(), "kvtc_prepare_rope: caches must not be empty");
    const at::Tensor& first_cache = caches.front();
    TORCH_CHECK(first_cache.dim() == 4, "kvtc_prepare_rope: every cache must be [G,B,H,D]");
    TORCH_CHECK(first_cache.scalar_type() == at::kBFloat16, "kvtc_prepare_rope: caches must be BF16");
    TORCH_CHECK(first_cache.is_contiguous(), "kvtc_prepare_rope: caches must be contiguous ND tensors");
    TORCH_CHECK(first_cache.size(3) > 0 && first_cache.size(3) <= 128 && first_cache.size(3) % 16 == 0,
                "kvtc_prepare_rope: head size must be <=128 and divisible by 16");

    TORCH_CHECK(block_ids.dim() == 1 && block_ids.scalar_type() == at::kLong && block_ids.is_contiguous(),
                "kvtc_prepare_rope: block_ids must be contiguous int64 [S]");
    TORCH_CHECK(token_starts.dim() == 1 && token_starts.scalar_type() == at::kLong && token_starts.is_contiguous(),
                "kvtc_prepare_rope: token_starts must be contiguous int64 [S]");
    TORCH_CHECK(token_starts.size(0) == block_ids.size(0),
                "kvtc_prepare_rope: block_ids and token_starts must have equal length");
    TORCH_CHECK(inv_cos_sin_cache.dim() == 2 && inv_cos_sin_cache.scalar_type() == at::kFloat &&
                    inv_cos_sin_cache.is_contiguous(),
                "kvtc_prepare_rope: inv_cos_sin_cache must be contiguous FP32 [P,D]");
    TORCH_CHECK(inv_cos_sin_cache.size(1) == first_cache.size(3),
                "kvtc_prepare_rope: RoPE width must equal cache head size");

    TORCH_CHECK(block_ids.device() == first_cache.device() && token_starts.device() == first_cache.device() &&
                    inv_cos_sin_cache.device() == first_cache.device(),
                "kvtc_prepare_rope: every input must be on one device");

    for (const at::Tensor& cache : caches) {
        TORCH_CHECK(cache.dim() == 4 && cache.sizes() == first_cache.sizes(),
                    "kvtc_prepare_rope: every cache must have the same [G,B,H,D] shape");
        TORCH_CHECK(cache.scalar_type() == at::kBFloat16, "kvtc_prepare_rope: every cache must be BF16");
        TORCH_CHECK(cache.is_contiguous(), "kvtc_prepare_rope: every cache must be contiguous ND");
        TORCH_CHECK(cache.device() == first_cache.device(), "kvtc_prepare_rope: all caches must be on one device");
    }

    const int64_t num_tokens = block_ids.size(0) * first_cache.size(1);
    const int64_t feature_dim = static_cast<int64_t>(caches.size()) * first_cache.size(2) * first_cache.size(3);
    at::Tensor prepared = at::empty({num_tokens, feature_dim}, first_cache.options());
    EXEC_NPU_CMD(aclnnKvtcPrepareRope, caches, block_ids, token_starts, inv_cos_sin_cache, prepared);
    return prepared;
}

}  // namespace vllm_ascend

#endif  // KVTC_PREPARE_ROPE_TORCH_ADPT_H
