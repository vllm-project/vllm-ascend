/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H

#include <array>
#include <cmath>
#include "op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"

namespace vllm_ascend {

inline bool is_fused_quant_sfa_head_count_supported(int64_t heads) {
  return heads >= fused_quant_scatter_copy_sfa::MIN_QUERY_HEAD_COUNT &&
         heads <= fused_quant_scatter_copy_sfa::MAX_QUERY_HEAD_COUNT &&
         fused_quant_scatter_copy_sfa::IsSupportedHeadCount(
             static_cast<uint32_t>(heads));
}

inline void npu_fused_quant_scatter_copy_sparse_flash_attention(
    const at::Tensor& query, at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes, const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table, const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids, const at::Tensor& topk_miss_counts,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& miss_src_ids, const at::Tensor& miss_dst_slots,
    const at::Tensor& miss_counts, double scale_value,
    at::Tensor attention_out) {
  TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 &&
                  query.size(2) == fused_quant_scatter_copy_sfa::QUERY_DIM &&
                  is_fused_quant_sfa_head_count_supported(query.size(1)),
              "query must be [T, H, 576], H in {8, 16, 32, 64, 128}.");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 ||
                  query.scalar_type() == at::kHalf,
              "query must be bfloat16 or float16.");
  TORCH_CHECK(query.device().is_privateuseone(),
              "fused quant scatter-copy SFA tensors must be on NPU.");

  const int64_t batch_size = num_cache_tokens.numel();
  const int64_t total_queries = query.size(0);
  TORCH_CHECK(num_cache_tokens.dim() == 1 && batch_size > 0 &&
                  total_queries >= batch_size &&
                  total_queries <= fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT * batch_size,
              "num_cache_tokens must be [B] and B <= T <= 16B (MTP0..MTP15).");
  for (const auto* cache : std::array<const at::Tensor*, 2>{
           &hbm_kv_bytes, &dram_kv_bytes}) {
    TORCH_CHECK(cache->dim() == 4 && cache->size(0) > 0 &&
                    cache->size(1) == fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE &&
                    cache->size(2) == fused_quant_scatter_copy_sfa::KV_HEAD_COUNT &&
                    cache->size(3) == fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES &&
                    cache->scalar_type() == at::kChar,
                "KV caches must be int8 [blocks, 128, 1, 656] C8 byte views.");
  }
  for (const auto* table : {&hbm_block_table, &dram_block_table}) {
    TORCH_CHECK(table->dim() == 2 && table->size(0) == batch_size &&
                    table->size(1) > 0 &&
                    table->size(1) <= fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST,
                "block tables must be [B, 1..2048].");
  }
  for (const auto* ids : {&topk_dst_slots, &topk_src_ids}) {
    TORCH_CHECK(ids->dim() == 3 && ids->size(0) == total_queries &&
                    ids->size(1) == 1 &&
                    ids->size(2) == fused_quant_scatter_copy_sfa::TOPK_CAPACITY,
                "topk indices must be [T, 1, 2048].");
  }
  for (const auto* ids : {&miss_src_ids, &miss_dst_slots}) {
    TORCH_CHECK(ids->dim() == 3 && ids->size(0) == batch_size &&
                    ids->size(1) == 1 &&
                    ids->size(2) == fused_quant_scatter_copy_sfa::COPY_CAPACITY,
                "miss indices must be [B, 1, 32768].");
  }
  TORCH_CHECK(topk_miss_counts.dim() == 1 &&
                  topk_miss_counts.size(0) == total_queries &&
                  actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size &&
                  miss_counts.dim() == 1 &&
                  miss_counts.size(0) == batch_size,
              "topk_miss_counts must be [T], and query ends, actual KV "
              "lengths, num_cache_tokens and miss_counts must be [B].");

  for (const auto* metadata : std::array<const at::Tensor*, 11>{
           &hbm_block_table, &dram_block_table, &topk_dst_slots,
           &topk_src_ids, &topk_miss_counts, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &num_cache_tokens, &miss_src_ids,
           &miss_dst_slots, &miss_counts}) {
    TORCH_CHECK(metadata->scalar_type() == at::kInt,
                "all metadata tensors must be int32.");
  }
  TORCH_CHECK(attention_out.dim() == 3 &&
                  attention_out.size(0) == total_queries &&
                  attention_out.size(1) == query.size(1) &&
                  attention_out.size(2) == fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM &&
                  attention_out.scalar_type() == query.scalar_type(),
              "attention_out must be [T, H, 512] with the query dtype.");
  TORCH_CHECK(std::isfinite(scale_value), "scale_value must be finite.");

  const auto device = query.device();
  for (const auto* tensor : std::array<const at::Tensor*, 15>{
           &query, &hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table,
           &dram_block_table, &topk_dst_slots, &topk_src_ids,
           &topk_miss_counts, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &num_cache_tokens, &miss_src_ids,
           &miss_dst_slots, &miss_counts, &attention_out}) {
    TORCH_CHECK(tensor->device() == device && tensor->is_contiguous(),
                "all tensors must be contiguous on the same NPU.");
  }

  // The same OpDef name is used for the HBM input and output, so ACLNN exposes
  // hbm_kv_bytes as one mutable Ref argument. Pass it only once here.
  EXEC_NPU_CMD(
      aclnnFusedQuantScatterCopySparseFlashAttention, query, hbm_kv_bytes,
      dram_kv_bytes, hbm_block_table, dram_block_table, topk_dst_slots,
      topk_src_ids, topk_miss_counts, actual_seq_lengths_query,
      actual_seq_lengths_kv, num_cache_tokens, miss_src_ids, miss_dst_slots,
      miss_counts, scale_value, attention_out);
}

}  // namespace vllm_ascend

#endif  // FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
