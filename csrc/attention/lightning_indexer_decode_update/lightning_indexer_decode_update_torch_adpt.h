/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef LIGHTNING_INDEXER_DECODE_UPDATE_TORCH_ADPT_H
#define LIGHTNING_INDEXER_DECODE_UPDATE_TORCH_ADPT_H

namespace vllm_ascend {

constexpr int64_t DSA_DECODE_UPDATE_OUTPUT_CAPACITY = 16384;
constexpr int64_t DSA_DECODE_UPDATE_BLOCK_SIZE = 128;
constexpr int64_t DSA_DECODE_UPDATE_TOKEN_CAPACITY = 1LL << 18;

inline void npu_lightning_indexer_decode_update_out(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots,
    const at::Tensor& row_modes,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& block_table,
    at::Tensor topk_index_out,
    at::Tensor topk_slots_out,
    at::Tensor miss_count_out,
    at::Tensor tail_info_out)
{
    TORCH_CHECK(query.dim() == 3,
                "query must have TND decode shape [B, N_idx, 128].");
    TORCH_CHECK(key.dim() == 4,
                "key must have PA_BSND shape [num_blocks, 128, 1, 128].");
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [B, N_idx].");
    TORCH_CHECK(req_pool_entries.dim() == 1,
                "req_pool_entries must have shape [B].");
    TORCH_CHECK(cache_slots.dim() == 2,
                "cache_slots must have shape [pool_size, W].");
    TORCH_CHECK(row_modes.dim() == 1, "row_modes must have shape [B].");
    TORCH_CHECK(actual_seq_lengths_key.dim() == 1,
                "actual_seq_lengths_key must have shape [B].");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must have shape [B, max_blocks].");

    const int64_t batch_size = query.size(0);
    const int64_t indexer_heads = query.size(1);
    TORCH_CHECK(batch_size > 0, "query batch size must be positive.");
    TORCH_CHECK(indexer_heads == 32 || indexer_heads == 64,
                "query N_idx must be 32 or 64.");
    TORCH_CHECK(query.size(2) == 128, "query head_dim must be 128.");
    TORCH_CHECK(key.size(0) > 0 &&
                    key.size(1) == DSA_DECODE_UPDATE_BLOCK_SIZE &&
                    key.size(2) == 1 && key.size(3) == 128,
                "key must have shape [num_blocks, 128, 1, 128].");
    TORCH_CHECK(weights.size(0) == batch_size &&
                    weights.size(1) == indexer_heads,
                "weights must match query batch and indexer heads.");
    TORCH_CHECK(req_pool_entries.size(0) == batch_size &&
                    row_modes.size(0) == batch_size &&
                    actual_seq_lengths_key.size(0) == batch_size &&
                    block_table.size(0) == batch_size,
                "all row metadata must match query batch size.");
    TORCH_CHECK(block_table.size(1) > 0,
                "block_table max_blocks must be positive.");
    TORCH_CHECK(cache_slots.size(0) > 0 && cache_slots.size(1) >= 2,
                "cache_slots must have pool_size > 0 and W >= 2.");

    const int64_t token_capacity = cache_slots.size(1) - 1;
    TORCH_CHECK(token_capacity <= DSA_DECODE_UPDATE_TOKEN_CAPACITY,
                "cache_slots token capacity must be <= 262144.");
    const int64_t padded_token_capacity =
        ((token_capacity + DSA_DECODE_UPDATE_BLOCK_SIZE - 1) /
         DSA_DECODE_UPDATE_BLOCK_SIZE) * DSA_DECODE_UPDATE_BLOCK_SIZE;
    TORCH_CHECK(block_table.size(1) * DSA_DECODE_UPDATE_BLOCK_SIZE <=
                    padded_token_capacity,
                "block_table token capacity exceeds cache_slots.");

    TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                    query.scalar_type() == weights.scalar_type(),
                "query, key and weights dtype must match.");
    TORCH_CHECK(query.scalar_type() == at::kHalf ||
                    query.scalar_type() == at::kBFloat16,
                "query, key and weights must be fp16 or bf16.");
    TORCH_CHECK(req_pool_entries.scalar_type() == at::kInt &&
                    cache_slots.scalar_type() == at::kInt &&
                    row_modes.scalar_type() == at::kInt &&
                    actual_seq_lengths_key.scalar_type() == at::kInt &&
                    block_table.scalar_type() == at::kInt,
                "LIDU metadata tensors must be int32.");

    TORCH_CHECK(topk_index_out.dim() == 3 &&
                    topk_index_out.size(0) == batch_size &&
                    topk_index_out.size(1) == 1 &&
                    topk_index_out.size(2) == DSA_DECODE_UPDATE_OUTPUT_CAPACITY,
                "topk_index_out must have shape [B, 1, 16384].");
    TORCH_CHECK(topk_slots_out.sizes() == topk_index_out.sizes(),
                "topk_slots_out must match topk_index_out shape.");
    TORCH_CHECK(miss_count_out.dim() == 1 &&
                    miss_count_out.size(0) == batch_size,
                "miss_count_out must have shape [B].");
    TORCH_CHECK(tail_info_out.dim() == 2 &&
                    tail_info_out.size(0) == batch_size &&
                    tail_info_out.size(1) == 2,
                "tail_info_out must have shape [B, 2].");
    TORCH_CHECK(topk_index_out.scalar_type() == at::kInt &&
                    topk_slots_out.scalar_type() == at::kInt &&
                    miss_count_out.scalar_type() == at::kInt &&
                    tail_info_out.scalar_type() == at::kInt,
                "LIDU output tensors must be int32.");

    const auto device = query.device();
    TORCH_CHECK(key.device() == device && weights.device() == device &&
                    req_pool_entries.device() == device &&
                    cache_slots.device() == device && row_modes.device() == device &&
                    actual_seq_lengths_key.device() == device &&
                    block_table.device() == device &&
                    topk_index_out.device() == device &&
                    topk_slots_out.device() == device &&
                    miss_count_out.device() == device &&
                    tail_info_out.device() == device,
                "all LIDU tensors must be on the same device.");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                    weights.is_contiguous() && req_pool_entries.is_contiguous() &&
                    cache_slots.is_contiguous() && row_modes.is_contiguous() &&
                    actual_seq_lengths_key.is_contiguous() &&
                    block_table.is_contiguous() && topk_index_out.is_contiguous() &&
                    topk_slots_out.is_contiguous() && miss_count_out.is_contiguous() &&
                    tail_info_out.is_contiguous(),
                "all LIDU tensors must be contiguous.");

    EXEC_NPU_CMD(aclnnLightningIndexerDecodeUpdate,
                 query, key, weights, req_pool_entries, cache_slots, row_modes,
                 actual_seq_lengths_key, block_table, topk_index_out,
                 topk_slots_out, miss_count_out, tail_info_out);
}

}  // namespace vllm_ascend

#endif  // LIGHTNING_INDEXER_DECODE_UPDATE_TORCH_ADPT_H
