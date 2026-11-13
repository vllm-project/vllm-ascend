#ifndef KVTC_RESTORE_ROPE_TORCH_ADPT_H
#define KVTC_RESTORE_ROPE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor kvtc_restore_rope(const at::Tensor& reconstructed, const at::Tensor& dst_block_ids,
                             const at::Tensor& token_starts, const at::Tensor& cos_sin_cache, at::TensorList caches)
{
    TORCH_CHECK(!caches.empty(), "kvtc_restore_rope: caches must not be empty");
    const at::Tensor& first_cache = caches.front();
    TORCH_CHECK(first_cache.dim() == 4 && first_cache.scalar_type() == at::kBFloat16 && first_cache.is_contiguous(),
                "kvtc_restore_rope: caches must be contiguous BF16 [G,B,H,D]");
    TORCH_CHECK(first_cache.size(3) > 0 && first_cache.size(3) <= 128 && first_cache.size(3) % 16 == 0,
                "kvtc_restore_rope: head size must be <=128 and divisible by 16");
    TORCH_CHECK(
        reconstructed.dim() == 2 && reconstructed.scalar_type() == at::kBFloat16 && reconstructed.is_contiguous(),
        "kvtc_restore_rope: reconstructed must be contiguous BF16 [M,K]");
    TORCH_CHECK(dst_block_ids.dim() == 1 && dst_block_ids.scalar_type() == at::kLong && dst_block_ids.is_contiguous(),
                "kvtc_restore_rope: dst_block_ids must be contiguous int64 [S]");
    TORCH_CHECK(token_starts.dim() == 1 && token_starts.scalar_type() == at::kLong && token_starts.is_contiguous() &&
                    token_starts.size(0) == dst_block_ids.size(0),
                "kvtc_restore_rope: token_starts must be contiguous int64 [S]");
    TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.scalar_type() == at::kFloat &&
                    cos_sin_cache.is_contiguous() && cos_sin_cache.size(1) == first_cache.size(3),
                "kvtc_restore_rope: cos_sin_cache must be contiguous FP32 [P,D]");

    const int64_t num_tokens = dst_block_ids.size(0) * first_cache.size(1);
    const int64_t feature_dim = static_cast<int64_t>(caches.size()) * first_cache.size(2) * first_cache.size(3);
    TORCH_CHECK(reconstructed.size(0) == num_tokens && reconstructed.size(1) == feature_dim,
                "kvtc_restore_rope: reconstructed shape must be [S*B,L*H*D]");
    TORCH_CHECK(reconstructed.device() == first_cache.device() && dst_block_ids.device() == first_cache.device() &&
                    token_starts.device() == first_cache.device() && cos_sin_cache.device() == first_cache.device(),
                "kvtc_restore_rope: every input must be on one device");

    for (const at::Tensor& cache : caches) {
        TORCH_CHECK(cache.dim() == 4 && cache.sizes() == first_cache.sizes() && cache.scalar_type() == at::kBFloat16 &&
                        cache.is_contiguous() && cache.device() == first_cache.device(),
                    "kvtc_restore_rope: every cache must match the first_cache cache");
    }

    at::Tensor status = at::empty({1}, first_cache.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnKvtcRestoreRope, reconstructed, dst_block_ids, token_starts, cos_sin_cache, caches, status);
    return status;
}

}  // namespace vllm_ascend

#endif  // KVTC_RESTORE_ROPE_TORCH_ADPT_H
