/**
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
#include "quest_block_select_paged_tilingkey.h"

#define BYTES_UB_BLOCK 32
#define BYTES_DATA_BLOCK 32
#define NUM_FLOAT_ELEMS_PER_VECTOR 64
#define DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define DIV_ROUNDUP_MUL(bytes, bytes_per_block) (DIV_ROUNDUP(bytes, bytes_per_block) * (bytes_per_block))
#define NUM_UB_BYTES(bytes) (DIV_ROUNDUP_MUL(bytes, BYTES_UB_BLOCK))
#define NUM_DATA_BLOCKS(bytes) (DIV_ROUNDUP(bytes, BYTES_DATA_BLOCK))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MINFLOAT -3.4028235e38f

constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 = 2;

using namespace AscendC;

// QuestBlockSelectPagedTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.

__aicore__ inline void quest_apply_anchor_selection(
    LocalTensor<uint32_t> &selected_indices_lt,
    int32_t seq_len,
    int32_t block_size,
    int32_t k)
{
    if (seq_len <= 0 || k <= 0) {
        return;
    }

    int32_t num_valid_pages = DIV_ROUNDUP(seq_len, block_size);
    int32_t num_selected_pages = MIN(k, num_valid_pages);
    // Case 1: Nothing selected
    if (num_selected_pages <= 0) {
        return;
    }

    // Case 2: <= 2 blocks selected
    uint32_t last_anchor = static_cast<uint32_t>(num_valid_pages - 1);
    if (unlikely(num_selected_pages <= 2)) {
        for (int32_t idx = 0; idx < k; ++idx) {
            selected_indices_lt.SetValue(idx, 0U);
        }
        // If only one page is selected, last_anchor should win
        selected_indices_lt.SetValue(num_selected_pages - 1, last_anchor);
        return;
    }

    // Case 3: > 2 blocks selected
    bool has_first_anchor = false;
    bool has_last_anchor = false;
    for (int32_t idx = 0; idx < num_selected_pages; ++idx) {
        uint32_t selected_page = selected_indices_lt.GetValue(idx);
        has_first_anchor = has_first_anchor || selected_page == 0U;
        has_last_anchor = has_last_anchor || selected_page == last_anchor;
        if (has_first_anchor && has_last_anchor) {
            break;
        }
    }

    if (!has_first_anchor) {
        selected_indices_lt.SetValue(num_selected_pages - 1, 0U);
    }
    if (!has_last_anchor) {
        selected_indices_lt.SetValue(
            num_selected_pages - 1 - static_cast<int32_t>(!has_first_anchor),
            last_anchor);
    }

    for (int32_t idx = num_selected_pages; idx < k; ++idx) {
        selected_indices_lt.SetValue(idx, 0U);
    }
}

template <typename StorageT>
class KernelQuestBlockSelectPaged {
    using VecBufT = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    using ComputeT = float;
    static constexpr uint32_t REGION_SIZE = REGION_PROPOSAL_DATA_SIZE_FLOAT_V220;

    struct LocalTensors {
        AscendC::LocalTensor<ComputeT> query;
        AscendC::LocalTensor<ComputeT> maxblock;
        AscendC::LocalTensor<ComputeT> minblock;
        AscendC::LocalTensor<ComputeT> block_scores;
        AscendC::LocalTensor<ComputeT> accumulated_scores;
        AscendC::LocalTensor<uint32_t> selected_indices;
        AscendC::LocalTensor<ComputeT> selected_values;
        AscendC::LocalTensor<ComputeT> tmp_concat;
        AscendC::LocalTensor<ComputeT> concat;
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
        block_size_ = block_size;
        head_dim_ = head_dim;
        max_metadata_blocks_per_request_ = max_metadata_blocks_per_request;
        tokens_since_metadata_update_ = tokens_since_metadata_update;
        k_ = k;

        query_gm_.SetGlobalBuffer((__gm__ StorageT *)query);
        maxblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)minblocks);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
        selected_indices_gm_.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);

        uint32_t input_storage_buf_size =
            NUM_UB_BYTES(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        pipe_.InitBuffer(input_storage_buf_, input_storage_buf_size);

        uint32_t query_buf_size =
            NUM_UB_BYTES(head_dim_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t block_buf_size =
            NUM_UB_BYTES(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t reduced_buf_size =
            NUM_UB_BYTES(block_size_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t accumulated_scores_size = NUM_UB_BYTES(
            max_metadata_blocks_per_request_ * block_size_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t selected_indices_buf_size = NUM_UB_BYTES(k_ * static_cast<int32_t>(sizeof(uint32_t)));
        uint32_t selected_values_buf_size =
            NUM_UB_BYTES(k_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t tmp_concat_buf_size = NUM_UB_BYTES(
            max_metadata_blocks_per_request_ * block_size_ * REGION_SIZE *
            static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t concat_buf_size = NUM_UB_BYTES(
            (max_metadata_blocks_per_request_ * block_size_ +
             max_metadata_blocks_per_request_ * block_size_ * REGION_SIZE) *
            static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t index_local_buf_size =
            NUM_UB_BYTES(max_metadata_blocks_per_request_ * block_size_ *
                         static_cast<int32_t>(sizeof(uint32_t)));
        uint32_t sort_tmp_buf_size = NUM_UB_BYTES(
            max_metadata_blocks_per_request_ * block_size_ * REGION_SIZE *
            static_cast<int32_t>(sizeof(ComputeT)));

        pipe_.InitBuffer(query_buf_, query_buf_size);
        pipe_.InitBuffer(maxblock_buf_, block_buf_size);
        pipe_.InitBuffer(minblock_buf_, block_buf_size);
        pipe_.InitBuffer(block_scores_buf_, reduced_buf_size);
        pipe_.InitBuffer(accumulated_scores_buf_, accumulated_scores_size);
        pipe_.InitBuffer(selected_indices_buf_, selected_indices_buf_size);
        pipe_.InitBuffer(selected_values_buf_, selected_values_buf_size);
        pipe_.InitBuffer(tmp_concat_buf_, tmp_concat_buf_size);
        pipe_.InitBuffer(concat_buf_, concat_buf_size);
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
            int32_t valid_page_count = DIV_ROUNDUP(seq_len, block_size_);
            int32_t num_tokens_per_meta_block = block_size_ * block_size_;
            int32_t num_meta_blocks_in_request = DIV_ROUNDUP(seq_len, num_tokens_per_meta_block);
            bool use_fixed_anchors = tokens_since_metadata_update_ >= 0;
            if (unlikely(use_fixed_anchors &&
                         valid_page_count > 0 &&
                         valid_page_count <= 2 &&
                         k_ >= valid_page_count)) {
                // Catch case where number of selected blocks <= 2
                quest_apply_anchor_selection(
                    tensors.selected_indices,
                    seq_len,
                    block_size_,
                    k_);
            } else {
                LoadQuery(tensors, query_offset);
                DuplicateAccumulatedScores(tensors, num_meta_blocks_in_request);

                for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
                    int32_t meta_block_id = metadata_block_tables_gm_.GetValue(
                        batch_idx * max_metadata_blocks_per_request_ + meta_block);
                    int32_t meta_block_offset =
                        meta_block_id * block_size_ * num_kv_heads_ * head_dim_ + kv_head_idx * head_dim_;

                    ScoreMetadataBlock(tensors, meta_block_offset);
                    CopyScoresToAccumulated(
                        tensors,
                        seq_len,
                        num_tokens_per_meta_block,
                        meta_block,
                        num_meta_blocks_in_request);
                }

                SortAndExtract(tensors, num_meta_blocks_in_request);

                if (likely(use_fixed_anchors)) {
                    quest_apply_anchor_selection(
                        tensors.selected_indices,
                        seq_len,
                        block_size_,
                        k_);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            uint16_t indices_copy_block_len = NUM_DATA_BLOCKS(k_ * static_cast<int32_t>(sizeof(int32_t)));
            auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
            AscendC::DataCopy(selected_indices_gm_[output_offset], tensors.selected_indices, indices_copy_params);
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
            selected_values_buf_.Get<ComputeT>(),
            tmp_concat_buf_.Get<ComputeT>(),
            concat_buf_.Get<ComputeT>(),
            index_local_buf_.Get<uint32_t>(),
            sort_tmp_buf_.Get<ComputeT>()};
    }

    __aicore__ inline void LoadQuery(
        LocalTensors &tensors,
        int32_t query_offset)
    {
        uint16_t query_copy_block_len =
            NUM_DATA_BLOCKS(head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);

        AscendC::LocalTensor<StorageT> input_storage_lt =
            input_storage_buf_.Get<StorageT>(block_size_ * head_dim_);
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
        int32_t num_meta_blocks_in_request)
    {
        AscendC::Duplicate(
            tensors.accumulated_scores,
            static_cast<ComputeT>(MINFLOAT),
            max_metadata_blocks_per_request_ * num_meta_blocks_in_request);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScoreMetadataBlock(
        LocalTensors &tensors,
        int32_t meta_block_offset)
    {
        AscendC::LocalTensor<StorageT> input_storage_lt =
            input_storage_buf_.Get<StorageT>(block_size_ * head_dim_);
        AscendC::DataCopyParams gm_ub_cp;
        gm_ub_cp.blockCount = block_size_;
        gm_ub_cp.blockLen =
            DIV_ROUNDUP(head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_DATA_BLOCK);
        gm_ub_cp.srcStride = DIV_ROUNDUP(
            (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(StorageT)),
            BYTES_DATA_BLOCK);
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
            block_size_ * head_dim_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Mul(
            tensors.maxblock,
            tensors.query,
            tensors.maxblock,
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);
        AscendC::Mul(
            tensors.maxblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.query[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.maxblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);

        AscendC::DataCopy(input_storage_lt, minblocks_gm_[meta_block_offset], gm_ub_cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.minblock,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            block_size_ * head_dim_);
        AscendC::Mul(
            tensors.minblock,
            tensors.query,
            tensors.minblock,
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);
        AscendC::Mul(
            tensors.minblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.query[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.minblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);

        AscendC::Max(tensors.maxblock, tensors.maxblock, tensors.minblock, block_size_ * head_dim_);

        AscendC::RepeatReduceSum(
            tensors.minblock,
            tensors.maxblock,
            block_size_,
            mask,
            0,
            1,
            1,
            8);
        AscendC::RepeatReduceSum(
            tensors.minblock[block_size_],
            tensors.maxblock[block_size_ * head_dim_ / masks_per_head_dim],
            block_size_,
            mask,
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(
            tensors.block_scores,
            tensors.minblock,
            masks_per_head_dim * block_size_ / mask,
            mask,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyScoresToAccumulated(
        LocalTensors &tensors,
        int32_t seq_len,
        int32_t num_tokens_per_meta_block,
        int32_t meta_block,
        int32_t num_meta_blocks_in_request)
    {
        uint64_t seq_len_curr_meta_block =
            MIN(seq_len - (meta_block * num_tokens_per_meta_block), block_size_);
        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_head_dim = head_dim_ / mask;
        for (int32_t sub_meta_block_id = 0; sub_meta_block_id < static_cast<int32_t>(masks_per_head_dim);
             sub_meta_block_id++) {
            int32_t block_scores_offset = sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR;
            int32_t accumulated_offset = meta_block * block_size_ + block_scores_offset;
            AscendC::Copy(
                tensors.accumulated_scores[accumulated_offset],
                tensors.block_scores[block_scores_offset],
                seq_len_curr_meta_block - sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR,
                1,
                {1, 1, 8, 8});
            if (meta_block < num_meta_blocks_in_request - 1) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
        }
    }

    __aicore__ inline void SortAndExtract(
        LocalTensors &tensors,
        int32_t num_meta_blocks_in_request)
    {
        uint32_t total_elements = num_meta_blocks_in_request * block_size_;
        uint32_t repeat_times = DIV_ROUNDUP(total_elements, 32);

        for (uint32_t idx = 0; idx < total_elements; idx++) {
            tensors.index_local.SetValue(idx, idx);
        }

        AscendC::Concat(
            tensors.concat,
            tensors.accumulated_scores,
            tensors.tmp_concat,
            repeat_times);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort<ComputeT, true>(
            tensors.maxblock,
            tensors.concat,
            tensors.index_local,
            tensors.sort_tmp,
            repeat_times);
        AscendC::Extract(
            tensors.selected_values,
            tensors.selected_indices,
            tensors.maxblock,
            repeat_times);
    }

    AscendC::TPipe pipe_;
    VecBufT input_storage_buf_;
    VecBufT query_buf_;
    VecBufT maxblock_buf_;
    VecBufT minblock_buf_;
    VecBufT block_scores_buf_;
    VecBufT accumulated_scores_buf_;
    VecBufT selected_indices_buf_;
    VecBufT selected_values_buf_;
    VecBufT tmp_concat_buf_;
    VecBufT concat_buf_;
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
    int32_t block_size_;
    int32_t head_dim_;
    int32_t max_metadata_blocks_per_request_;
    int32_t tokens_since_metadata_update_;
    int32_t k_;
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
