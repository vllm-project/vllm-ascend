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
#ifndef QUEST_PREFILL_METADATA_TORCH_ADPT_H
#define QUEST_PREFILL_METADATA_TORCH_ADPT_H

namespace vllm_ascend {
namespace {
constexpr int64_t QUEST_BLOCK_SIZE = 128;
constexpr int64_t QUEST_HEAD_DIM = 128;
}

inline void npu_quest_prefill_metadata(
    const at::Tensor &k_cache,
    const at::Tensor &block_tables,
    const at::Tensor &seq_lens,
    const at::Tensor &metadata_block_tables,
    at::Tensor &maxblocks,
    at::Tensor &minblocks)
{
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D.");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2D.");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D.");
    TORCH_CHECK(metadata_block_tables.dim() == 2, "metadata_block_tables must be 2D.");
    TORCH_CHECK(maxblocks.dim() == 4, "maxblocks must be 4D.");
    TORCH_CHECK(minblocks.dim() == 4, "minblocks must be 4D.");

    TORCH_CHECK(k_cache.scalar_type() == at::kHalf || k_cache.scalar_type() == at::kBFloat16,
                "quest_prefill_metadata only supports float16 and bfloat16 k_cache.");
    TORCH_CHECK(maxblocks.scalar_type() == k_cache.scalar_type() &&
                    minblocks.scalar_type() == k_cache.scalar_type(),
                "quest_prefill_metadata requires maxblocks and minblocks to match k_cache dtype.");
    TORCH_CHECK(block_tables.scalar_type() == at::kInt &&
                    seq_lens.scalar_type() == at::kInt &&
                    metadata_block_tables.scalar_type() == at::kInt,
                "quest_prefill_metadata expects int32 block tables and seq_lens.");

    const int64_t batch_size = seq_lens.size(0);
    // QUEST metadata remains KV-head aligned because the cache stores keys at
    // KV-head granularity even when decode-time selection is performed per q head.
    const int64_t num_kv_heads = k_cache.size(2);
    const int64_t block_size = k_cache.size(1);
    const int64_t head_dim = k_cache.size(3);

    TORCH_CHECK(head_dim == QUEST_HEAD_DIM,
                "quest_prefill_metadata requires head_dim == 128, got ", head_dim);
    TORCH_CHECK(block_size == QUEST_BLOCK_SIZE,
                "quest_prefill_metadata requires block_size == 128, got ", block_size);
    TORCH_CHECK(block_tables.size(0) == batch_size,
                "Batch size mismatch between seq_lens and block_tables.");
    TORCH_CHECK(metadata_block_tables.size(0) == batch_size,
                "Batch size mismatch between seq_lens and metadata_block_tables.");
    TORCH_CHECK(maxblocks.size(1) == block_size && minblocks.size(1) == block_size,
                "Metadata outputs must use the same block size as k_cache.");
    TORCH_CHECK(maxblocks.size(2) == num_kv_heads && minblocks.size(2) == num_kv_heads,
                "Metadata outputs must match the k_cache KV-head count.");
    TORCH_CHECK(maxblocks.size(3) == head_dim && minblocks.size(3) == head_dim,
                "Metadata outputs must match the k_cache head dimension.");
    TORCH_CHECK(maxblocks.is_contiguous() && minblocks.is_contiguous(),
                "quest_prefill_metadata expects contiguous output tensors.");

    EXEC_NPU_CMD(
        aclnnQuestPrefillMetadata,
        k_cache,
        block_tables,
        seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks);
}
} // namespace vllm_ascend

#endif
