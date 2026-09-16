#ifndef FUSED_COPY_SFA_MTP_TORCH_ADPT_H
#define FUSED_COPY_SFA_MTP_TORCH_ADPT_H

namespace vllm_ascend {

// Variable-width MTP source-aware copy + causal sparse Attention in one
// AscendC launch. actual_seq_lengths_query is the TND prefix sum and each
// request contributes between one and sixteen query rows (MTP0..MTP15).
// HBM caches are mutated in place and the caller owns attention_out; no cache
// aliases are exposed through the public torch.library schema.
inline void npu_fused_copy_sfa_mtp(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& topk_miss_counts,
    const at::Tensor& miss_src_ids,
    const at::Tensor& miss_dst_slots,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kMaxQueryCount = 16;
  constexpr int64_t kTopK = 2048;
  // MTP15 has sixteen query rows and each contributes up to TopK misses.
  constexpr int64_t kMissCapacity = 32768;
  constexpr int64_t kCkvDim = 512;
  constexpr int64_t kKpeDim = 64;

  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused MTP copy+Attention inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kCkvDim,
              "Fused MTP query must be [T, N, 512].");
  TORCH_CHECK(num_cache_tokens.dim() == 1 && num_cache_tokens.size(0) > 0,
              "Fused MTP cache_tokens must be [B] with B > 0.");
  const int64_t batch_size = num_cache_tokens.size(0);
  TORCH_CHECK(query.size(0) >= batch_size &&
                  query.size(0) <= batch_size * kMaxQueryCount,
              "Fused MTP T must satisfy B <= T <= 16B (MTP0..MTP15).");
  const int64_t num_query_heads = query.size(1);
  TORCH_CHECK(num_query_heads >= 8 && num_query_heads <= 128 &&
                  (num_query_heads & (num_query_heads - 1)) == 0,
              "Fused MTP query head count N must be one of "
              "{8, 16, 32, 64, 128}.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == query.size(0) &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == kKpeDim,
              "Fused MTP query_rope must be [T, N, 64].");
  TORCH_CHECK(topk_dst_slots.dim() == 3 &&
                  topk_dst_slots.size(0) == query.size(0) &&
                  topk_dst_slots.size(1) == 1 &&
                  topk_dst_slots.size(2) == kTopK,
              "Fused MTP topk_slots must be [T, 1, 2048].");
  TORCH_CHECK(topk_src_ids.sizes() == topk_dst_slots.sizes(),
              "Fused MTP topk_source_ids must match topk_slots.");
  TORCH_CHECK(topk_miss_counts.dim() == 1 &&
                  topk_miss_counts.size(0) == query.size(0),
              "Fused MTP topk_miss_counts must be [T].");
  TORCH_CHECK(miss_src_ids.dim() == 2 &&
                  miss_src_ids.size(0) == batch_size &&
                  miss_src_ids.size(1) == kMissCapacity &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes() &&
                  miss_counts.dim() == 1 &&
                  miss_counts.size(0) == batch_size,
              "Fused MTP first-fill metadata must be miss_src_ids/"
              "miss_dst_slots [B, 32768] and miss_counts [B].");
  TORCH_CHECK(hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == batch_size &&
                  hbm_block_table.size(1) > 0 &&
                  dram_block_table.dim() == 2 &&
                  dram_block_table.size(0) == batch_size &&
                  dram_block_table.size(1) > 0,
              "Fused MTP block tables must be [B, max_blocks].");
  TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size,
              "Fused MTP actual sequence lengths must be [B].");

  TORCH_CHECK(hbm_k_rope.dim() == 4 &&
                  hbm_k_rope.size(1) == kBlockSize &&
                  hbm_k_rope.size(2) == 1 &&
                  hbm_k_rope.size(3) == kKpeDim,
              "Fused MTP HBM KPE must be [blocks, 128, 1, 64].");
  TORCH_CHECK(hbm_kv_cache.dim() == 4 &&
                  hbm_kv_cache.size(0) == hbm_k_rope.size(0) &&
                  hbm_kv_cache.size(1) == kBlockSize &&
                  hbm_kv_cache.size(2) == 1 &&
                  hbm_kv_cache.size(3) == kCkvDim,
              "Fused MTP HBM CKV must be [blocks, 128, 1, 512].");
  TORCH_CHECK(dram_k_rope.dim() == 3 &&
                  dram_k_rope.size(1) == kBlockSize &&
                  dram_k_rope.size(2) == kKpeDim,
              "Fused MTP DRAM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(dram_kv_cache.dim() == 3 &&
                  dram_kv_cache.size(0) == dram_k_rope.size(0) &&
                  dram_kv_cache.size(1) == kBlockSize &&
                  dram_kv_cache.size(2) == kCkvDim,
              "Fused MTP DRAM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "Fused MTP attention_out must have the query shape.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Fused MTP copy+Attention supports fp16 or bf16.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 7>{
           &query_rope, &hbm_k_rope, &hbm_kv_cache, &dram_k_rope,
           &dram_kv_cache, &attention_out, &query}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "All fused MTP floating-point tensors must share one dtype.");
  }
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 11>{
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &num_cache_tokens, &topk_dst_slots, &topk_src_ids,
           &topk_miss_counts, &miss_src_ids, &miss_dst_slots,
           &miss_counts, &hbm_block_table, &dram_block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "All fused MTP metadata tensors must be int32.");
  }

  const auto device = query.device();
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 16>{
           &query_rope, &query, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &num_cache_tokens, &topk_dst_slots,
           &topk_src_ids, &topk_miss_counts, &miss_src_ids,
           &miss_dst_slots, &miss_counts,
           &hbm_block_table, &dram_block_table, &hbm_k_rope,
           &hbm_kv_cache,
           &attention_out}) {
    TORCH_CHECK(tensor->device() == device,
                "All fused MTP tensors must be on the same NPU.");
    TORCH_CHECK(tensor->is_contiguous(),
                "All fused MTP tensors must be contiguous.");
  }

  // Match the MTP0 offload ABI: the manager owns registered MemFabric GVA
  // allocations exposed as non-owning CPU tensors. ConvertType passes their
  // storage pointers directly to ACLNN without copying the history to HBM.
  // Ordinary CPU allocations are not device-accessible; callers must register
  // the backing allocation before passing a host view here.
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 2>{&dram_k_rope, &dram_kv_cache}) {
    TORCH_CHECK(tensor->device().is_cpu() || tensor->device() == device,
                "DRAM sources must be registered host views or on the query NPU.");
    TORCH_CHECK(tensor->is_contiguous(), "DRAM sources must be contiguous.");
  }

  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto first_fill_keepalive = std::make_tuple(
      hbm_k_rope, hbm_kv_cache, dram_k_rope, dram_kv_cache,
      hbm_block_table, dram_block_table, miss_src_ids, miss_dst_slots,
      miss_counts, num_cache_tokens);
  // This launch is a device-side conditional no-op for steady-state batches.
  // When any request is first-fill, it copies the request-level miss list to
  // HBM. The following Attention launch is ordered on the same NPU stream.
  EXEC_NPU_CMD_ORDERED(
      aclnnFirstFillScatterCopy,
      first_fill_keepalive,
      hbm_k_rope,
      hbm_kv_cache,
      dram_k_rope,
      dram_kv_cache,
      hbm_block_table,
      dram_block_table,
      miss_src_ids,
      miss_dst_slots,
      miss_counts,
      num_cache_tokens);

  auto keepalive = std::make_tuple(
      query_rope, query, actual_seq_lengths_query,
      actual_seq_lengths_kv, num_cache_tokens, topk_dst_slots,
      topk_src_ids, topk_miss_counts, miss_src_ids, miss_dst_slots,
      miss_counts,
      hbm_block_table, dram_block_table, hbm_k_rope,
      hbm_kv_cache, dram_k_rope, dram_kv_cache, attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnFusedCopySfaMtp,
      keepalive,
      query,
      hbm_kv_cache,
      hbm_kv_cache,
      topk_dst_slots,
      num_cache_tokens,
      hbm_block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      query_rope,
      hbm_k_rope,
      dram_k_rope,
      dram_kv_cache,
      dram_block_table,
      topk_src_ids,
      topk_miss_counts,
      miss_src_ids,
      miss_dst_slots,
      miss_counts,
      scale_value,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      attention_out);
}

}  // namespace vllm_ascend

#endif
