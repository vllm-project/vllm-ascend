/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Ported from nanovllm-DSA-offload ops/fused_li_manage_mtp (unchanged semantics;
 * symbol-isolated namespaces to coexist with the existing fused_li_manage ops).
 */

#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_TORCH_ADPT_H_
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_TORCH_ADPT_H_

#include "op_kernel/fused_quant_lightning_indexer_manage_constants.h"

namespace vllm_ascend {

// MTP fused LightningIndexer + top-2048 union + hit/evict + request-pool
// update (nanovllm fused_li_manage_mtp port).  The caller pre-allocates all
// output buffers; the operator writes in place and returns void.
//
//   query                     - int8 [T, H, 128], H=32/64, T = sum(routes)
//   query_dequant_scale       - fp16 [T, H] (per-(token, head) dequant scale;
//                               folded with index_weights by the kernel)
//   index_weights             - bf16 [T, H]
//   index_key_cache           - int8 [blocks, 128, 1, 128]
//   index_key_dequant_scale   - fp16 [blocks, 128, 1] (per-token dequant
//                               scale, gathered via the block table)
//   index_block_table         - int32 [B, max_blocks]
//   actual_seq_lengths_query  - int32 [B] (prefix sum, routes per request in [1,14])
//   actual_seq_lengths_key    - int32 [B]
//   offload_seq_lengths_key   - int32 [B]
//   num_cache_tokens          - int32 [B]
//   request_state             - int32 [B] (-3 standard / -2 first decode / -1 steady)
//   req_pool_entries          - int32 [B]
//   cache_slots_pool          - int32 [POOL, max_blocks*128], read-write
//   topk_src_ids              - int32 [T, 1, 2048], write-only
//   topk_dst_slots            - int32 [T, 1, 2048], write-only
//   topk_miss_counts          - int32 [T], write-only
//   miss_src_ids              - int32 [B, 32768], write-only
//   miss_dst_slots            - int32 [B, 32768], write-only
//   miss_counts               - int32 [B], write-only
inline void npu_fused_quant_lightning_indexer_manage(
    const at::Tensor& index_weights, const at::Tensor& query_dequant_scale,
    const at::Tensor& query, const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_key_cache, const at::Tensor& index_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& offload_seq_lengths_key,
    const at::Tensor& num_cache_tokens, const at::Tensor& request_state,
    const at::Tensor& req_pool_entries, at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids, at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts, at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots, at::Tensor miss_counts) {
  constexpr int64_t kTopK = LIMQuantConfig::TOPK;
  constexpr int64_t kMissCapacity = LIMQuantConfig::MISS_CAPACITY;
  constexpr int64_t kBlockSize = LIMQuantConfig::CACHE_BLOCK_SIZE;
  TORCH_CHECK(query.dim() == 3 &&
                  (query.size(1) == LIMQuantConfig::QUERY_HEADS_SMALL ||
                   query.size(1) == LIMQuantConfig::QUERY_HEADS_LARGE) &&
                  query.size(2) == LIMQuantConfig::HEAD_DIM && query.size(0) > 0,
              "LIM-QUANT query must be [T, H, 128], H=32 or 64 and T>0.");
  TORCH_CHECK(query.device().is_privateuseone(), "LIM-QUANT tensors must be on NPU.");
  const int64_t total_queries = query.size(0);
  TORCH_CHECK(index_weights.dim() == 2 && index_weights.size(0) == total_queries &&
                  index_weights.size(1) == query.size(1),
              "LIM-QUANT index_weights must be [T, H].");
  TORCH_CHECK(query_dequant_scale.sizes() == index_weights.sizes(),
              "LIM-QUANT query_dequant_scale must be [T, H].");
  TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 &&
                  index_key_cache.size(1) == kBlockSize &&
                  index_key_cache.size(2) == LIMQuantConfig::KEY_HEADS &&
                  index_key_cache.size(3) == LIMQuantConfig::HEAD_DIM,
              "LIM-QUANT index_key_cache must be [blocks, 128, 1, 128].");
  TORCH_CHECK(index_key_dequant_scale.dim() == 3 &&
                  index_key_dequant_scale.size(0) == index_key_cache.size(0) &&
                  index_key_dequant_scale.size(1) == kBlockSize &&
                  index_key_dequant_scale.size(2) == LIMQuantConfig::KEY_HEADS,
              "LIM-QUANT index_key_dequant_scale must be [blocks, 128, 1].");
  TORCH_CHECK(index_block_table.dim() == 2 && index_block_table.size(0) > 0 &&
                  index_block_table.size(1) > 0 && index_block_table.size(1) <= LIMQuantConfig::MAX_BLOCKS_PER_REQUEST,
              "LIM-QUANT index_block_table must be non-empty [B, max_blocks], max_blocks<=16384.");
  const int64_t batch_size = index_block_table.size(0);
  TORCH_CHECK(total_queries >= batch_size &&
                  total_queries <= batch_size * LIMQuantConfig::MAX_ROUTES,
              "LIM-QUANT total_queries must satisfy B <= T <= 14B.");
  auto check_batch_vector = [batch_size](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == batch_size, name, " must be [B].");
  };
  check_batch_vector(actual_seq_lengths_query, "actual_seq_lengths_query");
  check_batch_vector(actual_seq_lengths_key, "actual_seq_lengths_key");
  check_batch_vector(offload_seq_lengths_key, "offload_seq_lengths_key");
  check_batch_vector(num_cache_tokens, "num_cache_tokens");
  check_batch_vector(request_state, "request_state");
  check_batch_vector(req_pool_entries, "req_pool_entries");
  TORCH_CHECK(cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
                  cache_slots_pool.size(1) == index_block_table.size(1) * kBlockSize,
              "LIM-QUANT cache_slots_pool width must equal max_blocks*128.");
  TORCH_CHECK(topk_src_ids.dim() == 3 && topk_src_ids.size(0) == total_queries &&
                  topk_src_ids.size(1) == 1 && topk_src_ids.size(2) == kTopK &&
                  topk_dst_slots.sizes() == topk_src_ids.sizes(),
              "LIM-QUANT topk outputs must be [T, 1, 2048].");
  TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.size(0) == total_queries,
              "LIM-QUANT topk_miss_counts must be [T].");
  TORCH_CHECK(miss_src_ids.dim() == 2 && miss_src_ids.size(0) == batch_size &&
                  miss_src_ids.size(1) == kMissCapacity &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes(),
              "LIM-QUANT miss outputs must be [B, 32768].");
  check_batch_vector(miss_counts, "miss_counts");

  TORCH_CHECK(query.scalar_type() == at::kChar &&
                  index_key_cache.scalar_type() == at::kChar,
              "LIM-QUANT query/index_key_cache must be int8.");
  TORCH_CHECK(index_weights.scalar_type() == at::kBFloat16,
              "LIM-QUANT index_weights must be bf16.");
  TORCH_CHECK(query_dequant_scale.scalar_type() == at::kHalf &&
                  index_key_dequant_scale.scalar_type() == at::kHalf,
              "LIM-QUANT dequant scales must be fp16.");
  const at::Tensor int_tensors[] = {
      index_block_table, actual_seq_lengths_query, actual_seq_lengths_key,
      offload_seq_lengths_key, num_cache_tokens, request_state, req_pool_entries,
      cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts,
      miss_src_ids, miss_dst_slots, miss_counts};
  for (const auto& tensor : int_tensors) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt, "LIM-QUANT metadata and outputs must be int32.");
  }
  const at::Tensor all_tensors[] = {
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts};
  const auto device = query.device();
  for (const auto& tensor : all_tensors) {
    TORCH_CHECK(tensor.device() == device, "all LIM-QUANT tensors must be on the same NPU.");
    TORCH_CHECK(tensor.is_contiguous(), "all LIM-QUANT tensors must be contiguous.");
  }

  EXEC_NPU_CMD(
      aclnnFusedQuantLightningIndexerManage,
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
      miss_counts, cache_slots_pool);
}

}  // namespace vllm_ascend
#endif  // FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_TORCH_ADPT_H_
