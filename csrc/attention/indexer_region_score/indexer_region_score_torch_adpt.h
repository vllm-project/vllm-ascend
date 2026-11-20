// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <cmath>
namespace vllm_ascend {
inline void npu_indexer_region_score_ids_out(
    const at::Tensor &query, const at::Tensor &weights, const at::Tensor &key,
    const at::Tensor &lengths, const at::Tensor &table, const at::Tensor &cu,
    int64_t regions, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode, const at::Tensor &selectedIds)
{
    constexpr int64_t MIN_HEADS = 4, MAX_HEADS = 128, DIM = 256, MAX_REGIONS = 16777216, MAX_TOP_K = 512;
    TORCH_CHECK(query.device().is_privateuseone(), "indexer_region_score expects NPU inputs");
    TORCH_CHECK(cacheMode == 1 && maskMode == 3 && sortMode == 1,
        "only cache_mode=1, mask_mode=3 and sort_mode=1 are supported");
    const at::Tensor *inputs[] = {&query, &weights, &key, &lengths, &table, &cu};
    for (const auto *tensor : inputs) {
        TORCH_CHECK(tensor->device() == query.device(), "all inputs must be on the same NPU");
    }
    TORCH_CHECK(query.dim() == 4 && query.size(0) > 0 && query.size(0) <= INT32_MAX &&
        query.size(1) > 0 && (query.size(1) == 1 || query.size(1) % MIN_HEADS == 0) &&
        query.size(2) >= MIN_HEADS && query.size(2) <= MAX_HEADS && query.size(2) % MIN_HEADS == 0 &&
        query.size(1) <= UINT32_MAX / query.size(0) && query.size(1) <= UINT32_MAX / (query.size(2) * DIM) &&
        query.size(3) == DIM, "query must be [T,G,H,256], G=1 or a positive multiple of 4, H=4..128 in steps of 4");
    TORCH_CHECK(weights.dim() == 3 && weights.size(0) == query.size(0) &&
        weights.size(1) == query.size(1) && weights.size(2) == query.size(2), "weights must have shape [T,G,H]");
    TORCH_CHECK((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 ||
        query.scalar_type() == at::ScalarType::Float8_e4m3fn) && key.scalar_type() == query.scalar_type(),
        "query and key_summary must have the same FP16, BF16 or FP8 E4M3FN dtype");
    TORCH_CHECK(weights.scalar_type() == at::kHalf || weights.scalar_type() == at::kBFloat16 ||
        weights.scalar_type() == at::kFloat, "weights must have FP16, BF16 or FP32 dtype");
    TORCH_CHECK(key.dim() == 4 && key.size(0) > 0 && key.size(0) <= UINT32_MAX &&
        key.size(1) > 0 && key.size(1) <= 1024 && key.size(2) == 1 && key.size(3) == DIM,
        "key_summary must have shape [P,S,1,256], 1 <= S <= 1024");
    TORCH_CHECK(table.dim() == 2 && table.size(0) > 0 && table.size(0) <= UINT32_MAX &&
        table.size(1) > 0 && table.size(1) <= UINT32_MAX / key.size(1) &&
        lengths.dim() == 1 && lengths.size(0) == table.size(0) &&
        cu.dim() == 1 && cu.size(0) == table.size(0) + 1,
        "expected block_table[N,C], actual_kv_lengths[N], cu_seq_lens[N+1]");
    TORCH_CHECK(lengths.scalar_type() == at::kInt && table.scalar_type() == at::kInt &&
        cu.scalar_type() == at::kInt, "request metadata must be int32");
    TORCH_CHECK(regions >= 0 && regions <= MAX_REGIONS && regions <= table.size(1) * key.size(1) &&
        topK > 0 && topK <= MAX_TOP_K && regionSize > 0 && regionSize <= INT32_MAX &&
        std::isfinite(headScale) && std::isfinite(static_cast<float>(headScale)), "invalid scalar attributes");
    TORCH_CHECK(selectedIds.dim() == 3 && selectedIds.size(0) == query.size(1) &&
        selectedIds.size(1) == query.size(0) && selectedIds.size(2) == topK &&
        selectedIds.scalar_type() == at::kInt, "selected_region_ids must be INT32 [G,T,top_k]");
    TORCH_CHECK(selectedIds.device() == query.device(), "selected_region_ids must be on the input NPU");
    at::assert_no_internal_overlap(selectedIds);
    for (const auto *input : inputs) at::assert_no_overlap(selectedIds, *input);
    const c10_npu::OptionalNPUGuard deviceGuard(query.device());
    const c10::optional<at::Tensor> scoreOutput = c10::nullopt;
    EXEC_NPU_CMD(aclnnIndexerRegionScore, query, weights, key, lengths, table, cu,
        regions, topK, regionSize, headScale, cacheMode, maskMode, sortMode, scoreOutput, selectedIds);
}
} // namespace vllm_ascend
