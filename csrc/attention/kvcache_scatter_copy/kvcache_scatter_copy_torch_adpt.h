/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef KVCACHE_SCATTER_COPY_TORCH_ADPT_H
#define KVCACHE_SCATTER_COPY_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_kvcache_scatter_copy(
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& src_token_ids,
    const at::Tensor& dst_slots,
    const at::Tensor& copy_counts)
{
    constexpr int64_t block_size = 128;
    constexpr int64_t copy_capacity = 16384;
    TORCH_CHECK(hbm_k_rope.dim() == 3 && hbm_k_rope.size(1) == block_size &&
                    hbm_k_rope.size(2) == 64,
                "hbm_k_rope must have shape [blocks, 128, 64].");
    TORCH_CHECK(hbm_kv_cache.dim() == 3 &&
                    hbm_kv_cache.size(1) == block_size &&
                    hbm_kv_cache.size(2) == 512,
                "hbm_kv_cache must have shape [blocks, 128, 512].");
    TORCH_CHECK(dram_k_rope.dim() == 3 &&
                    dram_k_rope.size(1) == block_size &&
                    dram_k_rope.size(2) == 64,
                "dram_k_rope must have shape [blocks, 128, 64].");
    TORCH_CHECK(dram_kv_cache.dim() == 3 &&
                    dram_kv_cache.size(1) == block_size &&
                    dram_kv_cache.size(2) == 512,
                "dram_kv_cache must have shape [blocks, 128, 512].");
    TORCH_CHECK(hbm_k_rope.size(0) == hbm_kv_cache.size(0) &&
                    dram_k_rope.size(0) == dram_kv_cache.size(0),
                "RoPE and latent KV block counts must match per tier.");
    TORCH_CHECK(hbm_block_table.dim() == 2 &&
                    dram_block_table.dim() == 2,
                "HBM and DRAM block tables must be 2-D.");
    TORCH_CHECK(src_token_ids.dim() == 3 &&
                    src_token_ids.size(1) == 1 &&
                    src_token_ids.size(2) == copy_capacity,
                "src_token_ids must have shape [B, 1, 16384].");
    TORCH_CHECK(dst_slots.sizes() == src_token_ids.sizes(),
                "dst_slots must match src_token_ids shape.");
    TORCH_CHECK(copy_counts.dim() == 1,
                "copy_counts must have shape [B].");
    const int64_t batch_size = copy_counts.size(0);
    TORCH_CHECK(hbm_block_table.size(0) == batch_size &&
                    dram_block_table.size(0) == batch_size &&
                    src_token_ids.size(0) == batch_size,
                "all KSC row tensors must have the same batch size.");

    const auto cache_dtype = hbm_k_rope.scalar_type();
    TORCH_CHECK(cache_dtype == at::kHalf || cache_dtype == at::kBFloat16,
                "KSC caches must be fp16 or bf16.");
    TORCH_CHECK(hbm_kv_cache.scalar_type() == cache_dtype &&
                    dram_k_rope.scalar_type() == cache_dtype &&
                    dram_kv_cache.scalar_type() == cache_dtype,
                "all KSC caches must have the same dtype.");
    TORCH_CHECK(hbm_block_table.scalar_type() == at::kInt &&
                    dram_block_table.scalar_type() == at::kInt &&
                    src_token_ids.scalar_type() == at::kInt &&
                    dst_slots.scalar_type() == at::kInt &&
                    copy_counts.scalar_type() == at::kInt,
                "KSC metadata tensors must be int32.");
    const auto device = hbm_k_rope.device();
    TORCH_CHECK(hbm_kv_cache.device() == device &&
                    dram_k_rope.device() == device &&
                    dram_kv_cache.device() == device &&
                    hbm_block_table.device() == device &&
                    dram_block_table.device() == device &&
                    src_token_ids.device() == device &&
                    dst_slots.device() == device &&
                    copy_counts.device() == device,
                "all KSC tensors must be on the same device.");
    TORCH_CHECK(hbm_k_rope.is_contiguous() && hbm_kv_cache.is_contiguous() &&
                    dram_k_rope.is_contiguous() && dram_kv_cache.is_contiguous() &&
                    hbm_block_table.is_contiguous() &&
                    dram_block_table.is_contiguous() &&
                    src_token_ids.is_contiguous() && dst_slots.is_contiguous() &&
                    copy_counts.is_contiguous(),
                "all KSC tensors must be contiguous.");

    EXEC_NPU_CMD(aclnnKvcacheScatterCopy,
                 hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
                 hbm_block_table, dram_block_table, src_token_ids, dst_slots,
                 copy_counts);
}

}  // namespace vllm_ascend

#endif  // KVCACHE_SCATTER_COPY_TORCH_ADPT_H
