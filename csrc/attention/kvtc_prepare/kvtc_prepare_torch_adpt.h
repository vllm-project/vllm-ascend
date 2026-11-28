#ifndef KVTC_PREPARE_TORCH_ADPT_H
#define KVTC_PREPARE_TORCH_ADPT_H

#include "../kvtc_shape.h"

namespace vllm_ascend {

at::Tensor kvtc_prepare(at::TensorList caches, const at::Tensor& block_ids)
{
    TORCH_CHECK(!caches.empty(), "kvtc_prepare: caches must not be empty");
    const at::Tensor& first_cache = caches.front();
    TORCH_CHECK(first_cache.dim() == 4, "kvtc_prepare: every cache must be [G,B,H,D]");
    TORCH_CHECK(first_cache.scalar_type() == at::kBFloat16, "kvtc_prepare: caches must be BF16");
    TORCH_CHECK(first_cache.is_contiguous(), "kvtc_prepare: caches must be contiguous ND tensors");
    TORCH_CHECK(block_ids.dim() == 1, "kvtc_prepare: block_ids must be one-dimensional");
    TORCH_CHECK(block_ids.scalar_type() == at::kLong, "kvtc_prepare: block_ids must be int64");
    TORCH_CHECK(block_ids.is_contiguous(), "kvtc_prepare: block_ids must be contiguous");
    TORCH_CHECK(block_ids.device() == first_cache.device(), "kvtc_prepare: caches and block_ids must be on one device");

    for (const at::Tensor& cache : caches) {
        TORCH_CHECK(cache.dim() == 4 && cache.sizes() == first_cache.sizes(),
                    "kvtc_prepare: every cache must have the same [G,B,H,D] shape");
        TORCH_CHECK(cache.scalar_type() == at::kBFloat16, "kvtc_prepare: every cache must be BF16");
        TORCH_CHECK(cache.is_contiguous(), "kvtc_prepare: every cache must be contiguous ND");
        TORCH_CHECK(cache.device() == first_cache.device(), "kvtc_prepare: all caches must be on one device");
    }

    const int64_t block_size = first_cache.size(1);
    const int64_t num_heads = first_cache.size(2);
    const int64_t head_size = first_cache.size(3);
    const int64_t selected_blocks = block_ids.size(0);
    check_kvtc_cache_shape(first_cache.size(0), block_size, num_heads, head_size, selected_blocks,
                           false, "kvtc_prepare");
    const auto [num_tokens, feature_dim] = kvtc_matrix_shape(static_cast<int64_t>(caches.size()), block_size,
        num_heads, head_size, selected_blocks, "kvtc_prepare");
    at::Tensor prepared = at::empty({num_tokens, feature_dim}, first_cache.options());
    EXEC_NPU_CMD(aclnnKvtcPrepare, caches, block_ids, prepared);
    return prepared;
}

}  // namespace vllm_ascend

#endif  // KVTC_PREPARE_TORCH_ADPT_H
