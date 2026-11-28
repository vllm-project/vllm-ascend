#ifndef KVTC_RESTORE_TORCH_ADPT_H
#define KVTC_RESTORE_TORCH_ADPT_H

#include "../kvtc_shape.h"

namespace vllm_ascend {

at::Tensor kvtc_restore(const at::Tensor& reconstructed, const at::Tensor& dst_block_ids, at::TensorList caches)
{
    TORCH_CHECK(!caches.empty(), "kvtc_restore: caches must not be empty");
    const at::Tensor& first_cache = caches.front();
    TORCH_CHECK(first_cache.dim() == 4 && first_cache.scalar_type() == at::kBFloat16 && first_cache.is_contiguous(),
                "kvtc_restore: caches must be contiguous BF16 [G,B,H,D]");
    TORCH_CHECK(
        reconstructed.dim() == 2 && reconstructed.scalar_type() == at::kBFloat16 && reconstructed.is_contiguous(),
        "kvtc_restore: reconstructed must be contiguous BF16 [M,K]");
    TORCH_CHECK(dst_block_ids.dim() == 1 && dst_block_ids.scalar_type() == at::kLong && dst_block_ids.is_contiguous(),
                "kvtc_restore: dst_block_ids must be contiguous int64 [S]");

    const int64_t block_size = first_cache.size(1);
    const int64_t num_heads = first_cache.size(2);
    const int64_t head_size = first_cache.size(3);
    const int64_t selected_blocks = dst_block_ids.size(0);
    check_kvtc_cache_shape(first_cache.size(0), block_size, num_heads, head_size, selected_blocks,
                           false, "kvtc_restore");
    const auto [num_tokens, feature_dim] = kvtc_matrix_shape(static_cast<int64_t>(caches.size()), block_size,
        num_heads, head_size, selected_blocks, "kvtc_restore");
    TORCH_CHECK(reconstructed.size(0) == num_tokens && reconstructed.size(1) == feature_dim,
                "kvtc_restore: reconstructed shape must be [S*B,L*H*D]");
    TORCH_CHECK(reconstructed.device() == first_cache.device() && dst_block_ids.device() == first_cache.device(),
                "kvtc_restore: every input must be on one device");

    for (const at::Tensor& cache : caches) {
        TORCH_CHECK(cache.dim() == 4 && cache.sizes() == first_cache.sizes() && cache.scalar_type() == at::kBFloat16 &&
                        cache.is_contiguous() && cache.device() == first_cache.device(),
                    "kvtc_restore: every cache must match the first_cache cache");
    }

    at::Tensor status = at::empty({1}, first_cache.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnKvtcRestore, reconstructed, dst_block_ids, caches, status);
    return status;
}

}  // namespace vllm_ascend

#endif  // KVTC_RESTORE_TORCH_ADPT_H
