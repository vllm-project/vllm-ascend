// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <limits>
#include <type_traits>
#include "op_kernel/engram_hash_ids_constants.h"

namespace vllm_ascend {
inline void engram_hash_ids(
    const at::Tensor &input_ids, const at::Tensor &token_map, const at::Tensor &dead_mask,
    const at::Tensor &positions, const c10::optional<at::Tensor> &block_table,
    const at::Tensor &query_start_loc, const at::Tensor &req_ids,
    const at::Tensor &multipliers, const at::Tensor &primes, const at::Tensor &offsets,
    const c10::optional<at::Tensor> &cache, const at::Tensor &lookback_token_ids,
    const at::Tensor &lookback_dead_mask, const at::Tensor &output,
    int64_t num_tokens, int64_t num_slots, int64_t pad_id,
    int64_t input_stride, int64_t mask_stride, int64_t position_stride,
    int64_t table_stride, int64_t table_col_stride, int64_t query_stride,
    int64_t num_table_rows, int64_t max_blocks, int64_t cache_block_size,
    int64_t MAX_NGRAM, int64_t num_heads, int64_t dead_id, int64_t lookback_depth,
    int64_t lookback_row_stride, int64_t lookback_col_stride,
    int64_t lookback_mask_row_stride, int64_t lookback_mask_col_stride, int64_t num_layers)
{
    using namespace EngramHashIds;
    TORCH_CHECK(num_tokens >= 0 && num_layers > 0 && MAX_NGRAM >= 1 && num_heads > 0 && lookback_depth >= 0,
                "engram_hash_ids: invalid effective dimensions.");
    const int64_t parameters[]{
        num_tokens, num_slots, pad_id, input_stride, mask_stride, position_stride,
        table_stride, table_col_stride, query_stride, num_table_rows, max_blocks, cache_block_size,
        MAX_NGRAM, num_heads, dead_id, lookback_depth, lookback_row_stride, lookback_col_stride,
        lookback_mask_row_stride, lookback_mask_col_stride, num_layers};
    static_assert(std::extent_v<decltype(parameters)> == PARAMETER_COUNT);
    for (const auto index : {INPUT_STRIDE, MASK_STRIDE, POSITION_STRIDE, TABLE_STRIDE, TABLE_COL_STRIDE,
                              QUERY_STRIDE, LOOKBACK_ROW_STRIDE, LOOKBACK_COL_STRIDE,
                              LOOKBACK_MASK_ROW_STRIDE, LOOKBACK_MASK_COL_STRIDE}) {
        TORCH_CHECK(parameters[index] >= 0, "engram_hash_ids: strides must be nonnegative.");
    }
    const auto table = block_table.value_or(at::Tensor());
    const auto history = cache.value_or(at::Tensor());
    for (const auto &tensor : {token_map, dead_mask, positions, table, query_start_loc, req_ids,
                               multipliers, primes, offsets, history, lookback_token_ids, lookback_dead_mask, output}) {
        TORCH_CHECK(!tensor.defined() || tensor.device() == input_ids.device(),
                    "engram_hash_ids: all tensors must be on the same NPU.");
    }
    for (const auto &tensor : {token_map, req_ids, multipliers, primes, offsets, history, output}) {
        TORCH_CHECK(!tensor.defined() || tensor.is_contiguous(),
                    "engram_hash_ids: token_map, req_ids, parameters, cache and output must be contiguous.");
    }
    TORCH_CHECK(output.scalar_type() == at::kInt, "engram_hash_ids: output must be int32.");
    const auto product = [](int64_t left, int64_t right) {
        TORCH_CHECK(left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
                    "engram_hash_ids: effective dimensions overflow int64.");
        return left * right;
    };
    const int64_t columns = product(MAX_NGRAM - 1, num_heads);
    const int64_t param_elements = product(num_layers, columns);
    const int64_t output_elements = product(num_tokens, param_elements);
    if (output_elements == 0) {
        return;
    }
    const auto covers = [](const at::Tensor &tensor, int64_t count, int64_t stride) {
        const int64_t capacity = tensor.storage().nbytes() / tensor.element_size() - tensor.storage_offset();
        return count == 0 || (capacity > 0 && (stride == 0 || count - 1 <= (capacity - 1) / stride));
    };
    TORCH_CHECK(covers(input_ids, num_tokens, input_stride) && covers(dead_mask, num_tokens, mask_stride) &&
                    covers(positions, num_tokens, position_stride) && req_ids.numel() >= num_tokens &&
                    multipliers.numel() >= product(num_layers, MAX_NGRAM) &&
                    primes.numel() >= param_elements && offsets.numel() >= param_elements &&
                    output.numel() >= output_elements,
                "engram_hash_ids: buffers do not cover the effective dimensions.");
    if (history.defined()) {
        TORCH_CHECK(table.defined() && num_slots > 0 && num_slots <= history.numel() &&
                        num_table_rows > 0 && max_blocks > 0 && cache_block_size > 0,
                    "engram_hash_ids: cache requires a block table and positive effective dimensions.");
    }
    // Describe just the written prefix; retain the caller's allocation and untouched tail.
    auto output_view = output.as_strided({num_tokens, num_layers, columns}, {param_elements, columns, 1});
    const at::IntArrayRef parameter_array(parameters);
    static_assert(std::is_same_v<decltype(ConvertType(parameter_array)), aclIntArray *>);
    EXEC_NPU_CMD(aclnnEngramHashIds, input_ids, token_map, dead_mask, positions, table, query_start_loc,
                 req_ids, multipliers, primes, offsets, history, lookback_token_ids, lookback_dead_mask,
                 parameter_array, output_view);
}
} // namespace vllm_ascend
