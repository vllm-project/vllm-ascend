/**
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

/*******************************************************************************
 *  quest_block_select_paged_kernel - vector-core, 1 core = (request, q-head)
 *  Scores every page of the request with the QUEST upper bound, then selects
 *  the top-k pages (plus the page-0 and last-page anchors) via Sort + GatherMask
 *******************************************************************************/
#include "kernel_operator.h"
#include "quest_block_select_paged_tilingkey.h"

constexpr int32_t BYTES_UB_BLOCK = 32;
constexpr int32_t BYTES_DATA_BLOCK = 32;
constexpr int32_t NUM_FLOAT_ELEMS_PER_VECTOR = 64;
constexpr int32_t QUEST_PAGE_SIZE = 128;  // tokens per KV page
constexpr float QUEST_MIN_SCORE = -3.4028235e38f;
constexpr float QUEST_MAX_SCORE = 3.4028235e38f;
// The Sort API stores each element as a (score, index) pair == two words.
constexpr int32_t SORT_PAIR_WORDS = 2;
// The Sort high-level API emits one (score, index) pair per element; GatherMask
// pattern 2 keeps the index word of each pair, 32 pairs per 256-byte repeat.
constexpr int32_t NUM_SORT_PAIRS_PER_REPEAT = 32;
constexpr int32_t BYTES_VECTOR_REPEAT = 256;
constexpr uint8_t GATHER_INDEX_PATTERN = 2;

inline __aicore__ int32_t ceilDiv(int32_t x, int32_t d) { return (x + d - 1) / d; }
inline __aicore__ int32_t ceilDivMul(int32_t x, int32_t d) { return d * ceilDiv(x, d); }
inline __aicore__ int32_t alignUpUbBytes(int32_t bytes) { return ceilDivMul(bytes, BYTES_UB_BLOCK); }
inline __aicore__ int32_t numDataBlocks(int32_t bytes) { return ceilDiv(bytes, BYTES_DATA_BLOCK); }
inline __aicore__ int32_t minI32(int32_t a, int32_t b) { return a < b ? a : b; }

using namespace AscendC;

// QuestBlockSelectPagedTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.

__aicore__ inline void quest_apply_sequential_selection(
    LocalTensor<uint32_t> &selected_indices_lt,
    int32_t k)
{
    // The "select every page" case (k >= valid_page_count). paged_select_attention
    // reads only the first min(k, valid_page_count) entries, so a plain
    // [0, 1, ..., k - 1] page-index ramp is correct over whatever prefix it
    // consumes; the trailing entries are unused. k is a multiple of 8, so this
    // single deterministic vector op writes exactly k elements (no spill).
    AscendC::Arange<int32_t>(
        selected_indices_lt.ReinterpretCast<int32_t>(),
        static_cast<int32_t>(0), static_cast<int32_t>(1), k);
}

template <typename StorageT>
class KernelQuestBlockSelectPaged {
    using VecBufT = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    using ComputeT = float;

    struct LocalTensors {
        AscendC::LocalTensor<ComputeT> query;
        AscendC::LocalTensor<ComputeT> maxblock;
        AscendC::LocalTensor<ComputeT> minblock;
        AscendC::LocalTensor<ComputeT> block_scores;
        AscendC::LocalTensor<ComputeT> accumulated_scores;
        AscendC::LocalTensor<uint32_t> selected_indices;
        AscendC::LocalTensor<uint32_t> index_local;
        AscendC::LocalTensor<ComputeT> sort_tmp;
    };

public:
    __aicore__ inline KernelQuestBlockSelectPaged() {}

    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR maxblocks,
        GM_ADDR minblocks,
        GM_ADDR metadata_block_tables,
        GM_ADDR seq_lens,
        GM_ADDR selected_indices,
        int32_t batch_size,
        int32_t num_kv_heads,
        int32_t num_heads,
        int32_t block_size,
        int32_t head_dim,
        int32_t max_metadata_blocks_per_request,
        int32_t tokens_since_metadata_update,
        int32_t k)
    {
        AscendC::SetAtomicNone();

        batch_size_ = batch_size;
        num_kv_heads_ = num_kv_heads;
        num_heads_ = num_heads;
        pages_per_metadata_block_ = block_size;
        head_dim_ = head_dim;
        max_metadata_blocks_per_request_ = max_metadata_blocks_per_request;
        tokens_since_metadata_update_ = tokens_since_metadata_update;
        k_ = k;

        // A metadata row (head_dim elements) spans this many 32-byte data blocks,
        // and consecutive page rows are inter_kv_head_stride_blocks_ apart because
        // other KV heads sit between them in the metadata layout.
        head_dim_storage_blocks_ =
            ceilDiv(head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_DATA_BLOCK);
        inter_kv_head_stride_blocks_ = ceilDiv(
            (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(StorageT)),
            BYTES_DATA_BLOCK);

        query_gm_.SetGlobalBuffer((__gm__ StorageT *)query);
        maxblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)minblocks);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
        selected_indices_gm_.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);

        uint32_t input_storage_buf_size =
            alignUpUbBytes(pages_per_metadata_block_ * head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        pipe_.InitBuffer(input_storage_buf_, input_storage_buf_size);

        uint32_t query_buf_size =
            alignUpUbBytes(head_dim_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t block_buf_size =
            alignUpUbBytes(pages_per_metadata_block_ * head_dim_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t reduced_buf_size =
            alignUpUbBytes(pages_per_metadata_block_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t accumulated_scores_size = alignUpUbBytes(
            max_metadata_blocks_per_request_ * pages_per_metadata_block_ * static_cast<int32_t>(sizeof(ComputeT)));
        // GatherMask extracts a whole 32-index repeat at a time, so the output
        // buffer must hold ceil(k / 32) * 32 indices even when k is smaller.
        uint32_t selected_indices_buf_size = alignUpUbBytes(
            ceilDiv(k_, NUM_SORT_PAIRS_PER_REPEAT) * NUM_SORT_PAIRS_PER_REPEAT *
            static_cast<int32_t>(sizeof(uint32_t)));
        uint32_t index_local_buf_size =
            alignUpUbBytes(max_metadata_blocks_per_request_ * pages_per_metadata_block_ *
                         static_cast<int32_t>(sizeof(uint32_t)));
        uint32_t sort_tmp_buf_size = alignUpUbBytes(
            max_metadata_blocks_per_request_ * pages_per_metadata_block_ * SORT_PAIR_WORDS *
            static_cast<int32_t>(sizeof(ComputeT)));

        pipe_.InitBuffer(query_buf_, query_buf_size);
        pipe_.InitBuffer(maxblock_buf_, block_buf_size);
        pipe_.InitBuffer(minblock_buf_, block_buf_size);
        pipe_.InitBuffer(block_scores_buf_, reduced_buf_size);
        pipe_.InitBuffer(accumulated_scores_buf_, accumulated_scores_size);
        pipe_.InitBuffer(selected_indices_buf_, selected_indices_buf_size);
        pipe_.InitBuffer(index_local_buf_, index_local_buf_size);
        pipe_.InitBuffer(sort_tmp_buf_, sort_tmp_buf_size);
    }

    __aicore__ inline void Process()
    {
        int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        int32_t num_batch_heads = batch_size_ * num_heads_;
        int32_t query_heads_per_kv_head = num_heads_ / num_kv_heads_;

        LocalTensors tensors = GetLocalTensors();

        for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads;
             batch_head_idx += num_blocks) {
            int32_t batch_idx = batch_head_idx / num_heads_;
            int32_t query_head_idx = batch_head_idx % num_heads_;
            int32_t kv_head_idx = query_head_idx / query_heads_per_kv_head;

            int32_t query_offset = batch_idx * num_heads_ * head_dim_ + query_head_idx * head_dim_;
            int32_t output_offset = batch_head_idx * k_;

            int32_t seq_len = seq_lens_gm_.GetValue(batch_idx);
            int32_t valid_page_count = seq_len > 0 ? ceilDiv(seq_len, QUEST_PAGE_SIZE) : 0;
            bool use_fixed_anchors = tokens_since_metadata_update_ >= 0;
            // When the page budget covers every page, select all pages
            // sequentially. This degenerate case also satisfies the fixed
            // anchors (page 0 and the last page are always in [0, valid_page_count)).
            if (unlikely(valid_page_count <= 0 || k_ >= valid_page_count)) {
                quest_apply_sequential_selection(tensors.selected_indices, k_);
            } else {
                int32_t num_meta_blocks_in_request = ceilDiv(valid_page_count, pages_per_metadata_block_);
                int32_t sort_element_count = ceilDiv(valid_page_count, 32) * 32;

                LoadQuery(tensors, query_offset);
                DuplicateAccumulatedScores(tensors, sort_element_count);

                for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
                    int32_t meta_block_id = metadata_block_tables_gm_.GetValue(
                        batch_idx * max_metadata_blocks_per_request_ + meta_block);
                    int32_t meta_block_offset =
                        meta_block_id * pages_per_metadata_block_ * num_kv_heads_ * head_dim_ + kv_head_idx * head_dim_;

                    ScoreMetadataBlock(tensors, meta_block_offset);
                    CopyScoresToAccumulated(
                        tensors,
                        valid_page_count,
                        meta_block,
                        num_meta_blocks_in_request);
                }

                if (likely(use_fixed_anchors)) {
                    // Pin the anchor pages (page 0 and the last page) to +inf so
                    // they sort to the top and are always among the selected top-k.
                    PinAnchorScores(tensors, valid_page_count);
                }
                SortAndExtract(tensors, sort_element_count);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            uint16_t indices_copy_block_len = numDataBlocks(k_ * static_cast<int32_t>(sizeof(int32_t)));
            auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
            AscendC::DataCopy(selected_indices_gm_[output_offset], tensors.selected_indices, indices_copy_params);
            // selected_indices (UB) is reused by the next iteration's vector
            // writes (Duplicate/Arange or GatherMask), so wait for this GM copy
            // (MTE3) to finish reading it before those writes can start.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

private:
    __aicore__ inline LocalTensors GetLocalTensors()
    {
        return {
            query_buf_.Get<ComputeT>(),
            maxblock_buf_.Get<ComputeT>(),
            minblock_buf_.Get<ComputeT>(),
            block_scores_buf_.Get<ComputeT>(),
            accumulated_scores_buf_.Get<ComputeT>(),
            selected_indices_buf_.Get<uint32_t>(),
            index_local_buf_.Get<uint32_t>(),
            sort_tmp_buf_.Get<ComputeT>()};
    }

    __aicore__ inline void LoadQuery(
        LocalTensors &tensors,
        int32_t query_offset)
    {
        uint16_t query_copy_block_len =
            numDataBlocks(head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);

        AscendC::LocalTensor<StorageT> input_storage_lt =
            input_storage_buf_.Get<StorageT>(pages_per_metadata_block_ * head_dim_);
        AscendC::DataCopy(input_storage_lt, query_gm_[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.query,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            head_dim_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DuplicateAccumulatedScores(
        LocalTensors &tensors,
        int32_t sort_element_count)
    {
        AscendC::Duplicate(
            tensors.accumulated_scores,
            static_cast<ComputeT>(QUEST_MIN_SCORE),
            sort_element_count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Multiply every page's head_dim channels in `buf` by the query (broadcast
    // across pages), one NUM_FLOAT_ELEMS_PER_VECTOR-wide vector chunk at a time.
    // head_dim is a multiple of the vector width (128 / 64 == 2 chunks today).
    __aicore__ inline void MultiplyPagesByQuery(
        AscendC::LocalTensor<ComputeT> buf,
        AscendC::LocalTensor<ComputeT> query,
        const AscendC::BinaryRepeatParams &mul_repeat_params)
    {
        int32_t num_vector_chunks = head_dim_ / NUM_FLOAT_ELEMS_PER_VECTOR;
        for (int32_t chunk = 0; chunk < num_vector_chunks; chunk++) {
            int32_t offset = chunk * NUM_FLOAT_ELEMS_PER_VECTOR;
            AscendC::Mul(
                buf[offset],
                query[offset],
                buf[offset],
                NUM_FLOAT_ELEMS_PER_VECTOR,
                pages_per_metadata_block_,
                mul_repeat_params);
        }
    }

    __aicore__ inline void ScoreMetadataBlock(
        LocalTensors &tensors,
        int32_t meta_block_offset)
    {
        AscendC::LocalTensor<StorageT> input_storage_lt =
            input_storage_buf_.Get<StorageT>(pages_per_metadata_block_ * head_dim_);
        AscendC::DataCopyParams gm_ub_cp;
        gm_ub_cp.blockCount = pages_per_metadata_block_;
        gm_ub_cp.blockLen = head_dim_storage_blocks_;
        gm_ub_cp.srcStride = inter_kv_head_stride_blocks_;
        gm_ub_cp.dstStride = 0;

        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_head_dim = head_dim_ / mask;
        AscendC::BinaryRepeatParams mul_repeat_params = AscendC::BinaryRepeatParams(
            1,
            1,
            1,
            8 * masks_per_head_dim,
            0,
            8 * masks_per_head_dim);

        AscendC::DataCopy(input_storage_lt, maxblocks_gm_[meta_block_offset], gm_ub_cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.maxblock,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            pages_per_metadata_block_ * head_dim_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        MultiplyPagesByQuery(tensors.maxblock, tensors.query, mul_repeat_params);

        AscendC::DataCopy(input_storage_lt, minblocks_gm_[meta_block_offset], gm_ub_cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.minblock,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            pages_per_metadata_block_ * head_dim_);
        MultiplyPagesByQuery(tensors.minblock, tensors.query, mul_repeat_params);

        AscendC::Max(tensors.maxblock, tensors.maxblock, tensors.minblock, pages_per_metadata_block_ * head_dim_);

        // Reduce each page's head_dim upper bounds in NUM_FLOAT_ELEMS_PER_VECTOR
        // chunks; chunk c leaves one partial sum per page at minblock[c*pages_per_metadata_block_].
        // PairReduceSum below then collapses the per-chunk partials into one score.
        for (int32_t chunk = 0; chunk < static_cast<int32_t>(masks_per_head_dim); chunk++) {
            AscendC::RepeatReduceSum(
                tensors.minblock[chunk * pages_per_metadata_block_],
                tensors.maxblock[chunk * pages_per_metadata_block_ * NUM_FLOAT_ELEMS_PER_VECTOR],
                pages_per_metadata_block_,
                mask,
                0,
                1,
                1,
                8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(
            tensors.block_scores,
            tensors.minblock,
            masks_per_head_dim * pages_per_metadata_block_ / mask,
            mask,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyScoresToAccumulated(
        LocalTensors &tensors,
        int32_t valid_page_count,
        int32_t meta_block,
        int32_t num_meta_blocks_in_request)
    {
        int32_t start_page = meta_block * pages_per_metadata_block_;
        int32_t pages_in_meta_block = minI32(valid_page_count - start_page, pages_per_metadata_block_);
        if (pages_in_meta_block <= 0) {
            return;
        }

        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_head_dim = head_dim_ / mask;
        for (int32_t sub_meta_block_id = 0; sub_meta_block_id < static_cast<int32_t>(masks_per_head_dim);
             sub_meta_block_id++) {
            int32_t block_scores_offset = sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR;
            int32_t pages_remaining = pages_in_meta_block - block_scores_offset;
            if (pages_remaining <= 0) {
                break;
            }

            int32_t accumulated_offset = meta_block * pages_per_metadata_block_ + block_scores_offset;
            AscendC::Copy(
                tensors.accumulated_scores[accumulated_offset],
                tensors.block_scores[block_scores_offset],
                static_cast<uint64_t>(minI32(pages_remaining, NUM_FLOAT_ELEMS_PER_VECTOR)),
                1,
                {1, 1, 8, 8});
            if (meta_block < num_meta_blocks_in_request - 1) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
        }
    }

    __aicore__ inline void PinAnchorScores(LocalTensors &tensors, int32_t valid_page_count)
    {
        if (valid_page_count <= 0) {
            return;
        }

        // The page scores were just written by the vector Copy in
        // CopyScoresToAccumulated; wait for the vector unit before overwriting
        // the anchor scores with the scalar unit (vector -> scalar).
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        tensors.accumulated_scores.SetValue(0, static_cast<ComputeT>(QUEST_MAX_SCORE));
        tensors.accumulated_scores.SetValue(valid_page_count - 1, static_cast<ComputeT>(QUEST_MAX_SCORE));
        // These scalar writes feed the vector Sort/Concat in SortAndExtract, so
        // they must be made visible to the vector unit (scalar -> vector).
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void SortAndExtract(
        LocalTensors &tensors,
        int32_t sort_element_count)
    {
        int32_t repeat_times = sort_element_count / NUM_SORT_PAIRS_PER_REPEAT;

        // Build the [0, 1, 2, ...] page-index ramp with a vector op (Arange) so
        // it is naturally ordered before the Sort that consumes it -- no
        // scalar->vector sync needed. Arange has no uint32_t overload, but the
        // int32_t bit pattern is identical for these in-range page indices.
        AscendC::Arange<int32_t>(
            tensors.index_local.template ReinterpretCast<int32_t>(),
            static_cast<int32_t>(0),
            static_cast<int32_t>(1),
            sort_element_count);

        // Full descending sort of the raw scores (Sort concatenates the score
        // and index internally -- no separate Concat needed). The result in
        // `maxblock` holds one (score, index) pair per element; as uint32 words
        // it is [score0, index0, score1, index1, ...], 32 pairs per 256B repeat.
        AscendC::Sort<ComputeT, true>(
            tensors.maxblock,
            tensors.accumulated_scores,
            tensors.index_local,
            tensors.sort_tmp,
            repeat_times);
        AscendC::PipeBarrier<PIPE_V>();

        // Gather only the index word of each sorted pair (built-in pattern 2,
        // which keeps the second word of every pair). Each repeat extracts 32
        // indices, so ceil(k / 32) repeats cover the top k; the first k written
        // are the highest-scoring pages.
        AscendC::GatherMaskParams gather_params;
        gather_params.src0BlockStride = 1;
        gather_params.repeatTimes = ceilDiv(k_, NUM_SORT_PAIRS_PER_REPEAT);
        gather_params.src0RepeatStride = BYTES_VECTOR_REPEAT / BYTES_DATA_BLOCK;
        gather_params.src1RepeatStride = 0;
        uint64_t reserved_count = 0;
        uint8_t index_pattern = GATHER_INDEX_PATTERN;
        AscendC::GatherMask(
            tensors.selected_indices,
            tensors.maxblock.template ReinterpretCast<uint32_t>(),
            index_pattern,
            false,
            static_cast<uint32_t>(0),
            gather_params,
            reserved_count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe pipe_;
    VecBufT input_storage_buf_;
    VecBufT query_buf_;
    VecBufT maxblock_buf_;
    VecBufT minblock_buf_;
    VecBufT block_scores_buf_;
    VecBufT accumulated_scores_buf_;
    VecBufT selected_indices_buf_;
    VecBufT index_local_buf_;
    VecBufT sort_tmp_buf_;

    AscendC::GlobalTensor<StorageT> query_gm_;
    AscendC::GlobalTensor<StorageT> maxblocks_gm_;
    AscendC::GlobalTensor<StorageT> minblocks_gm_;
    AscendC::GlobalTensor<int32_t> metadata_block_tables_gm_;
    AscendC::GlobalTensor<int32_t> seq_lens_gm_;
    AscendC::GlobalTensor<uint32_t> selected_indices_gm_;

    int32_t batch_size_;
    int32_t num_kv_heads_;
    int32_t num_heads_;
    int32_t pages_per_metadata_block_;  // maxblocks dim 1 (== 128)
    int32_t head_dim_;
    int32_t max_metadata_blocks_per_request_;
    int32_t tokens_since_metadata_update_;
    int32_t k_;
    int32_t head_dim_storage_blocks_;
    int32_t inter_kv_head_stride_blocks_;
};

template <typename StorageT>
__aicore__ inline void RunQuestBlockSelectPaged(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    const QuestBlockSelectPagedTilingData *__restrict tiling_data)
{
    KernelQuestBlockSelectPaged<StorageT> op;
    op.Init(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        selected_indices,
        static_cast<int32_t>(tiling_data->batchSize),
        static_cast<int32_t>(tiling_data->numKvHeads),
        static_cast<int32_t>(tiling_data->numHeads),
        static_cast<int32_t>(tiling_data->blockSize),
        static_cast<int32_t>(tiling_data->headDim),
        static_cast<int32_t>(tiling_data->maxMetadataBlocksPerRequest),
        tiling_data->tokensSinceMetadataUpdate,
        static_cast<int32_t>(tiling_data->k));
    op.Process();
}

extern "C" __global__ __aicore__ void quest_block_select_paged(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    QUEST_BLOCK_SELECT_PAGED_COPY_TILING_DATA(QuestBlockSelectPagedTilingData, tiling);

    if (!TILING_KEY_IS(QUEST_BLOCK_SELECT_PAGED_TILING)) {
        ASSERT(false && "Unsupported quest_block_select_paged tiling key.");
        return;
    }

    if (tiling_data->dataType == QUEST_BLOCK_SELECT_PAGED_DTYPE_FP16) {
        RunQuestBlockSelectPaged<half>(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            tiling_data);
        return;
    }

    if (tiling_data->dataType == QUEST_BLOCK_SELECT_PAGED_DTYPE_BF16) {
        RunQuestBlockSelectPaged<bfloat16_t>(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            tiling_data);
        return;
    }

    ASSERT(false && "Unsupported quest_block_select_paged dtype.");
}
