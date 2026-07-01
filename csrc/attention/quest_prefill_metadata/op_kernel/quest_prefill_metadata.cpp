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
 *  quest_prefill_metadata_kernel - vector-core, 1 core = (batch, head)
 *  Loads each KV-block ONCE, keeps copy, reduces min & max logarithmically
 *******************************************************************************/
#include "kernel_operator.h"
#include "quest_prefill_metadata_tilingkey.h"

constexpr int32_t SINGLEBUFFER = 1;
constexpr int32_t DOUBLEBUFFER = 2;
constexpr int32_t BYTES_UB_BLOCK = 32;
constexpr int32_t BYTES_DATA_BLOCK = 32;
// Tokens reduced per FP32 cast chunk. A power of two (so ReducePageRows needs
// no tail handling) and < block_size, so a 128-token page reduces in two
// chunks, keeping the FP32 scratch tile small.
constexpr int32_t METADATA_REDUCE_CHUNK_TOKENS = 64;
constexpr uint64_t FP32_VECTOR_MASK = 64;
// A metadata block stores the summaries of this many KV pages (one row per page).
// It is numerically equal to the page size but is a distinct concept.
constexpr int32_t PAGES_PER_METADATA_BLOCK = 128;

inline __aicore__ int32_t ceilDiv(int32_t x, int32_t d) { return (x + d - 1) / d; }
inline __aicore__ int32_t ceilDivMul(int32_t x, int32_t d) { return d * ((x + d - 1) / d); }

using namespace AscendC;

// QuestPrefillMetadataTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.

template <typename StorageT>
class KernelQuestMetadata {
public:
    // Metadata reductions always run in FP32 regardless of the KV storage dtype
    // (FP16 or BF16); the per-channel result is cast back to StorageT at the end.
    using ComputeT = float;

    __aicore__ inline KernelQuestMetadata() {}

    __aicore__ void Init(
        GM_ADDR k_cache,
        GM_ADDR block_tables,
        GM_ADDR refresh_start_seq_lens,
        GM_ADDR refresh_end_seq_lens,
        GM_ADDR metadata_block_tables,
        GM_ADDR maxblocks,
        GM_ADDR minblocks,
        int32_t batch_size,
        int32_t num_kv_heads,
        int32_t block_size,
        int32_t head_dim,
        int32_t max_kv_blocks_per_request,
        int32_t max_metadata_blocks_per_request)
    {
        batch_size_ = batch_size;
        num_kv_heads_ = num_kv_heads;
        block_size_ = block_size;
        head_dim_ = head_dim;
        max_kv_blocks_per_request_ = max_kv_blocks_per_request;
        max_metadata_blocks_per_request_ = max_metadata_blocks_per_request;

        // A key row (head_dim elements) occupies this many 32-byte data blocks,
        // and consecutive tokens are inter_kv_head_stride_blocks_ apart in the
        // KV cache because other KV heads sit between them.
        head_dim_storage_blocks_ =
            ceilDiv(head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_DATA_BLOCK);
        inter_kv_head_stride_blocks_ = ceilDiv(
            (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(StorageT)),
            BYTES_DATA_BLOCK);

        k_cache_gm_.SetGlobalBuffer((__gm__ StorageT *)k_cache);
        block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)block_tables);
        refresh_start_seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)refresh_start_seq_lens);
        refresh_end_seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)refresh_end_seq_lens);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        maxblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)minblocks);

        int32_t storage_tile_bytes =
            ceilDivMul(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_UB_BLOCK);
        // FP32 reduction scratch: two accumulator rows (max, min) plus one
        // METADATA_REDUCE_CHUNK_TOKENS-token cast chunk.
        int32_t compute_rows = METADATA_REDUCE_CHUNK_TOKENS + 2;
        int32_t work_tile_bytes =
            ceilDivMul(compute_rows * head_dim_ * static_cast<int32_t>(sizeof(ComputeT)), BYTES_UB_BLOCK);
        pipe_.InitBuffer(k_block_in_q_, DOUBLEBUFFER, storage_tile_bytes);
        pipe_.InitBuffer(work_calc_buf_, work_tile_bytes);
        pipe_.InitBuffer(max_out_q_, SINGLEBUFFER, storage_tile_bytes);
        pipe_.InitBuffer(min_out_q_, SINGLEBUFFER, storage_tile_bytes);
    }

    __aicore__ void Process()
    {
        int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        int32_t num_batch_heads = batch_size_ * num_kv_heads_;

        for (int32_t batch_head_idx = GetBlockIdx(); batch_head_idx < num_batch_heads;
             batch_head_idx += num_blocks) {
            int32_t request_idx = batch_head_idx / num_kv_heads_;
            int32_t head_idx = batch_head_idx % num_kv_heads_;

            int32_t start_len = refresh_start_seq_lens_gm_.GetValue(request_idx);
            int32_t end_len = refresh_end_seq_lens_gm_.GetValue(request_idx);
            if (end_len <= start_len) {
                continue;
            }
            ASSERT(start_len >= 0 && "refresh_start_seq_lens must be non-negative.");

            // Metadata is materialized only for complete KV pages. The active
            // tail page is selected by the decode selector's fixed anchor, so
            // do not refresh partial-page metadata here.
            int32_t start_page = start_len / block_size_;
            int32_t end_page = end_len / block_size_;
            int32_t start_meta_block = start_page / PAGES_PER_METADATA_BLOCK;
            int32_t end_meta_block = ceilDiv(end_page, PAGES_PER_METADATA_BLOCK);

            for (int32_t meta_block = start_meta_block; meta_block < end_meta_block; meta_block++) {
                int32_t meta_block_start_page = meta_block * PAGES_PER_METADATA_BLOCK;
                int32_t first_page = start_page - meta_block_start_page;
                if (first_page < 0) {
                    first_page = 0;
                }
                int32_t last_page = end_page - meta_block_start_page;
                if (last_page > PAGES_PER_METADATA_BLOCK) {
                    last_page = PAGES_PER_METADATA_BLOCK;
                }
                int32_t pages_to_refresh = last_page - first_page;
                if (pages_to_refresh <= 0) {
                    continue;
                }

                LocalTensor<StorageT> max_lt = max_out_q_.AllocTensor<StorageT>();
                LocalTensor<StorageT> min_lt = min_out_q_.AllocTensor<StorageT>();

                for (int32_t page_offset = first_page; page_offset < last_page; ++page_offset) {
                    int32_t logical_page = meta_block_start_page + page_offset;
                    int32_t tokens_to_reduce = block_size_;

                    int32_t kv_block_id = block_tables_gm_.GetValue(
                        request_idx * max_kv_blocks_per_request_ + logical_page);
                    int32_t kv_block_offset_gm =
                        (kv_block_id * block_size_ * num_kv_heads_ * head_dim_) + head_idx * head_dim_;

                    LocalTensor<StorageT> k_block_lt = k_block_in_q_.AllocTensor<StorageT>();
                    DataCopyParams gm_ub_cp;
                    gm_ub_cp.blockCount = tokens_to_reduce;
                    gm_ub_cp.blockLen = head_dim_storage_blocks_;
                    gm_ub_cp.srcStride = inter_kv_head_stride_blocks_;
                    gm_ub_cp.dstStride = 0;
                    DataCopy(k_block_lt, k_cache_gm_[kv_block_offset_gm], gm_ub_cp);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    ReduceBlockToOutput<true>(
                        max_lt[(page_offset - first_page) * head_dim_],
                        k_block_lt,
                        tokens_to_reduce);
                    ReduceBlockToOutput<false>(
                        min_lt[(page_offset - first_page) * head_dim_],
                        k_block_lt,
                        tokens_to_reduce);
                    k_block_in_q_.FreeTensor(k_block_lt);
                }

                int32_t rows_to_write = pages_to_refresh;
                int32_t zero_rows = 0;
                if (first_page == 0 && last_page < PAGES_PER_METADATA_BLOCK) {
                    zero_rows = PAGES_PER_METADATA_BLOCK - last_page;
                    Duplicate<StorageT>(
                        max_lt[pages_to_refresh * head_dim_],
                        static_cast<StorageT>(0),
                        zero_rows * head_dim_);
                    Duplicate<StorageT>(
                        min_lt[pages_to_refresh * head_dim_],
                        static_cast<StorageT>(0),
                        zero_rows * head_dim_);
                    rows_to_write += zero_rows;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                int32_t meta_block_id = metadata_block_tables_gm_.GetValue(
                    request_idx * max_metadata_blocks_per_request_ + meta_block);
                int32_t meta_offset =
                    (meta_block_id * PAGES_PER_METADATA_BLOCK * num_kv_heads_ * head_dim_) +
                    first_page * num_kv_heads_ * head_dim_ + head_idx * head_dim_;

                DataCopyParams ub_gm_cp;
                ub_gm_cp.blockCount = rows_to_write;
                ub_gm_cp.blockLen = head_dim_storage_blocks_;
                ub_gm_cp.srcStride = 0;
                ub_gm_cp.dstStride = inter_kv_head_stride_blocks_;
                DataCopy(maxblocks_gm_[meta_offset], max_lt, ub_gm_cp);
                DataCopy(minblocks_gm_[meta_offset], min_lt, ub_gm_cp);
                // max_lt/min_lt are single-buffered and reused by the next
                // meta-block's reduce, so wait for these GM copies (MTE3) to
                // finish reading them before those vector writes can start.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

                max_out_q_.FreeTensor(max_lt);
                min_out_q_.FreeTensor(min_lt);
            }
        }
    }

private:
    template <bool isMax>
    __aicore__ inline void ReduceBlockToOutput(
        LocalTensor<StorageT> out_lt,
        LocalTensor<StorageT> k_block_lt,
        int32_t tokens_to_reduce)
    {
        LocalTensor<ComputeT> work_lt = work_calc_buf_.Get<ComputeT>();
        LocalTensor<ComputeT> acc_lt = work_lt[isMax ? 0 : head_dim_];
        LocalTensor<ComputeT> chunk_lt = work_lt[2 * head_dim_];

        // Reduce in FP32 regardless of the storage dtype: cast each chunk of
        // tokens up to FP32, reduce it, and combine into the FP32 accumulator;
        // cast the final per-channel result back to StorageT. For FP16 this is
        // bit-identical to reducing in FP16 (Max/Min pick the same element and
        // the exact value casts back losslessly) and it keeps BF16 accurate.
        for (int32_t token_offset = 0; token_offset < tokens_to_reduce;
             token_offset += METADATA_REDUCE_CHUNK_TOKENS) {
            int32_t chunk_tokens = tokens_to_reduce - token_offset;
            if (chunk_tokens > METADATA_REDUCE_CHUNK_TOKENS) {
                chunk_tokens = METADATA_REDUCE_CHUNK_TOKENS;
            }

            Cast(
                chunk_lt,
                k_block_lt[token_offset * head_dim_],
                RoundMode::CAST_NONE,
                chunk_tokens * head_dim_);
            AscendC::PipeBarrier<PIPE_V>();
            ReducePageRows<isMax>(chunk_lt, chunk_tokens);

            if (token_offset == 0) {
                CopyRow(acc_lt, chunk_lt);
            } else if (isMax) {
                Max(acc_lt, acc_lt, chunk_lt, head_dim_);
            } else {
                Min(acc_lt, acc_lt, chunk_lt, head_dim_);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        Cast(out_lt, acc_lt, RoundMode::CAST_RINT, head_dim_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyRow(LocalTensor<ComputeT> dst_lt, LocalTensor<ComputeT> src_lt)
    {
        uint64_t mask = FP32_VECTOR_MASK;
        uint8_t repeats = static_cast<uint8_t>(head_dim_ / FP32_VECTOR_MASK);
        Copy(dst_lt, src_lt, mask, repeats, {1, 1, 8, 8});
    }

    // Reduce `token_rows` rows of head_dim elements down to a single head_dim
    // row, taking the per-channel Max (isMax) or Min across the rows. Fixed
    // power-of-two halving stages -- token_rows must be a power of two (the
    // METADATA_REDUCE_CHUNK_TOKENS-token cast chunk), so there is no tail
    // handling and no scalar bookkeeping. Each stage reads what the previous
    // stage wrote (in-place RAW), so a per-stage PipeBarrier is required.
    template <bool isMax>
    __aicore__ inline void ReducePageRows(LocalTensor<ComputeT> vec_lt, int32_t token_rows)
    {
        for (int32_t half = token_rows >> 1; half > 0; half >>= 1) {
            int32_t count = half * head_dim_;
            if constexpr (isMax) {
                Max(vec_lt[0], vec_lt[0], vec_lt[count], count);
            } else {
                Min(vec_lt[0], vec_lt[0], vec_lt[count], count);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, DOUBLEBUFFER> k_block_in_q_;
    TBuf<TPosition::VECCALC> work_calc_buf_;
    TQue<TPosition::VECOUT, SINGLEBUFFER> max_out_q_;
    TQue<TPosition::VECOUT, SINGLEBUFFER> min_out_q_;

    GlobalTensor<StorageT> k_cache_gm_;
    GlobalTensor<StorageT> maxblocks_gm_;
    GlobalTensor<StorageT> minblocks_gm_;
    GlobalTensor<int32_t> block_tables_gm_;
    GlobalTensor<int32_t> refresh_start_seq_lens_gm_;
    GlobalTensor<int32_t> refresh_end_seq_lens_gm_;
    GlobalTensor<int32_t> metadata_block_tables_gm_;

    int32_t batch_size_;
    int32_t num_kv_heads_;
    int32_t block_size_;  // tokens per KV page (k_cache dim 1, == 128)
    int32_t head_dim_;
    int32_t max_kv_blocks_per_request_;
    int32_t max_metadata_blocks_per_request_;
    int32_t head_dim_storage_blocks_;
    int32_t inter_kv_head_stride_blocks_;
};

template <typename StorageT>
__aicore__ inline void RunQuestPrefillMetadata(
    GM_ADDR k_cache,
    GM_ADDR block_tables,
    GM_ADDR refresh_start_seq_lens,
    GM_ADDR refresh_end_seq_lens,
    GM_ADDR metadata_block_tables,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    const QuestPrefillMetadataTilingData *tiling_data)
{
    KernelQuestMetadata<StorageT> op;
    op.Init(
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
        static_cast<int32_t>(tiling_data->batchSize),
        static_cast<int32_t>(tiling_data->numKvHeads),
        static_cast<int32_t>(tiling_data->blockSize),
        static_cast<int32_t>(tiling_data->headDim),
        static_cast<int32_t>(tiling_data->maxKvBlocksPerRequest),
        static_cast<int32_t>(tiling_data->maxMetadataBlocksPerRequest));
    op.Process();
}

extern "C" __global__ __aicore__ void quest_prefill_metadata(
    GM_ADDR k_cache,
    GM_ADDR block_tables,
    GM_ADDR refresh_start_seq_lens,
    GM_ADDR refresh_end_seq_lens,
    GM_ADDR metadata_block_tables,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    QUEST_PREFILL_METADATA_COPY_TILING_DATA(QuestPrefillMetadataTilingData, tiling);

    if (!TILING_KEY_IS(QUEST_PREFILL_METADATA_TILING)) {
        ASSERT(false && "Unsupported quest_prefill_metadata tiling key.");
        return;
    }

    if (tiling_data->dataType == QUEST_PREFILL_METADATA_DTYPE_FP16) {
        RunQuestPrefillMetadata<half>(
            k_cache,
            block_tables,
            refresh_start_seq_lens,
            refresh_end_seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }

    if (tiling_data->dataType == QUEST_PREFILL_METADATA_DTYPE_BF16) {
        RunQuestPrefillMetadata<bfloat16_t>(
            k_cache,
            block_tables,
            refresh_start_seq_lens,
            refresh_end_seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }

    ASSERT(false && "Unsupported quest_prefill_metadata dtype.");
}
