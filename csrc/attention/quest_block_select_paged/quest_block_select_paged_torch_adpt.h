/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
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
#ifndef QUEST_BLOCK_SELECT_PAGED_TORCH_ADPT_H
#define QUEST_BLOCK_SELECT_PAGED_TORCH_ADPT_H

namespace vllm_ascend {
namespace {
constexpr int64_t QUEST_BLOCK_SELECT_BLOCK_SIZE = 128;
constexpr int64_t QUEST_BLOCK_SELECT_HEAD_DIM = 128;
constexpr int64_t QUEST_BLOCK_SELECT_MAX_MMBPR = 6;
constexpr int64_t QUEST_INDICES_BYTES = 4;
constexpr int64_t QUEST_DATA_BLOCK_BYTES = 32;

inline int64_t round_k_for_quest(int64_t k)
{
    const int64_t rounded_bytes =
        ((k * QUEST_INDICES_BYTES + QUEST_DATA_BLOCK_BYTES - 1) / QUEST_DATA_BLOCK_BYTES) *
        QUEST_DATA_BLOCK_BYTES;
    return rounded_bytes / QUEST_INDICES_BYTES;
}

inline void check_quest_block_select_paged_common(
    const at::Tensor &query,
    const at::Tensor &maxblocks,
    const at::Tensor &minblocks,
    const at::Tensor &metadata_block_tables,
    const at::Tensor &seq_lens,
    const at::Tensor &output,
    int64_t tokens_since_metadata_update,
    bool require_aligned_output)
{
    TORCH_CHECK(query.dim() == 3, "query must be 3D.");
    TORCH_CHECK(maxblocks.dim() == 4, "maxblocks must be 4D.");
    TORCH_CHECK(minblocks.dim() == 4, "minblocks must be 4D.");
    TORCH_CHECK(metadata_block_tables.dim() == 2, "metadata_block_tables must be 2D.");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D.");
    TORCH_CHECK(output.dim() == 3, "selected_indices must be 3D.");

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "quest_block_select_paged only supports float16 and bfloat16 query tensors.");
    TORCH_CHECK(maxblocks.scalar_type() == query.scalar_type() &&
                    minblocks.scalar_type() == query.scalar_type(),
                "query, maxblocks, and minblocks must share the same dtype.");
    TORCH_CHECK(metadata_block_tables.scalar_type() == at::kInt &&
                    seq_lens.scalar_type() == at::kInt &&
                    output.scalar_type() == at::kInt,
                "quest_block_select_paged expects int32 metadata_block_tables, seq_lens, and outputs.");

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_dim = query.size(2);
    const int64_t block_size = maxblocks.size(1);
    const int64_t num_kv_heads = maxblocks.size(2);
    const int64_t output_k = output.size(2);

    TORCH_CHECK(head_dim == QUEST_BLOCK_SELECT_HEAD_DIM,
                "quest_block_select_paged requires head_dim == 128, got ", head_dim);
    TORCH_CHECK(block_size == QUEST_BLOCK_SELECT_BLOCK_SIZE,
                "quest_block_select_paged requires block_size == 128, got ", block_size);
    TORCH_CHECK(maxblocks.size(0) == minblocks.size(0) &&
                    minblocks.size(1) == block_size &&
                    minblocks.size(2) == num_kv_heads &&
                    minblocks.size(3) == head_dim,
                "maxblocks and minblocks must have matching shapes.");
    TORCH_CHECK(metadata_block_tables.size(0) == batch_size,
                "Batch size mismatch between query and metadata_block_tables.");
    TORCH_CHECK(seq_lens.size(0) == batch_size,
                "Batch size mismatch between query and seq_lens.");
    TORCH_CHECK(output.size(0) == batch_size && output.size(1) == num_heads,
                "selected_indices must have shape [B, H, k].");
    TORCH_CHECK(output_k > 0, "k must be positive.");
    TORCH_CHECK(num_heads % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads.");
    TORCH_CHECK(metadata_block_tables.size(1) <= QUEST_BLOCK_SELECT_MAX_MMBPR,
                "metadata_block_tables.size(1) cannot exceed ",
                QUEST_BLOCK_SELECT_MAX_MMBPR, ".");
    TORCH_CHECK(tokens_since_metadata_update == -1 ||
                    (tokens_since_metadata_update >= 0 &&
                     tokens_since_metadata_update <= block_size),
                "tokens_since_metadata_update must be -1 or in [0, block_size].");
    if (require_aligned_output) {
        TORCH_CHECK(output_k == round_k_for_quest(output_k),
                    "The last dimension of the output tensor must be aligned to 8 int32 values.");
    }
}
} // namespace

inline at::Tensor npu_quest_block_select_paged(
    const at::Tensor &query,
    const at::Tensor &maxblocks,
    const at::Tensor &minblocks,
    const at::Tensor &metadata_block_tables,
    const at::Tensor &seq_lens,
    int64_t k,
    int64_t tokens_since_metadata_update)
{
    TORCH_CHECK(k > 0, "k must be positive.");
    const int64_t rounded_k = round_k_for_quest(k);
    at::Tensor output = at::empty(
        {query.size(0), query.size(1), rounded_k},
        query.options().dtype(at::kInt));
    check_quest_block_select_paged_common(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        output,
        tokens_since_metadata_update,
        false);

    EXEC_NPU_CMD(
        aclnnQuestBlockSelectPaged,
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        tokens_since_metadata_update,
        output);

    if (rounded_k != k) {
        output = output.slice(/*dim=*/2, /*start=*/0, /*end=*/k);
    }
    return output;
}

inline at::Tensor &npu_quest_block_select_paged_out(
    const at::Tensor &query,
    const at::Tensor &maxblocks,
    const at::Tensor &minblocks,
    const at::Tensor &metadata_block_tables,
    const at::Tensor &seq_lens,
    at::Tensor &output,
    int64_t tokens_since_metadata_update)
{
    check_quest_block_select_paged_common(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        output,
        tokens_since_metadata_update,
        true);

    EXEC_NPU_CMD(
        aclnnQuestBlockSelectPaged,
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        tokens_since_metadata_update,
        output);
    return output;
}
} // namespace vllm_ascend

#endif
