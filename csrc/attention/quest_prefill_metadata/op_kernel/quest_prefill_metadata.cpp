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
constexpr int32_t BF16_METADATA_REDUCE_CHUNK_TOKENS = 64;
constexpr uint64_t FP32_VECTOR_MASK = 64;

inline __aicore__ int32_t ceilDiv(int32_t x, int32_t d) { return (x + d - 1) / d; }
inline __aicore__ int32_t ceilDivMul(int32_t x, int32_t d) { return d * ((x + d - 1) / d); }

using namespace AscendC;

// QuestPrefillMetadataTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.

template <typename A, typename B>
struct quest_is_same {
    static constexpr bool value = false;
};

template <typename A>
struct quest_is_same<A, A> {
    static constexpr bool value = true;
};

template <typename StorageT, typename ComputeT>
class KernelQuestMetadata {
public:
    __aicore__ inline KernelQuestMetadata() {}

    __aicore__ void Init(
        GM_ADDR k_cache,
        GM_ADDR block_tables,
        GM_ADDR seq_lens,
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

        k_cache_gm_.SetGlobalBuffer((__gm__ StorageT *)k_cache);
        block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)block_tables);
        seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        maxblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)minblocks);

        int32_t storage_tile_bytes =
            ceilDivMul(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_UB_BLOCK);
        int32_t compute_rows = block_size_;
        if constexpr (!quest_is_same<StorageT, ComputeT>::value) {
            // Keep the BF16 FP32 scratch tile the same size as a full FP16 tile.
            compute_rows = BF16_METADATA_REDUCE_CHUNK_TOKENS + 2;
        }
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

            int32_t seq_len = seq_lens_gm_.GetValue(request_idx);
            int32_t num_kv_blocks_in_request = ceilDiv(seq_len, block_size_);
            int32_t num_meta_blocks_in_request = ceilDiv(num_kv_blocks_in_request, block_size_);

            for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
                LocalTensor<StorageT> max_lt = max_out_q_.AllocTensor<StorageT>();
                LocalTensor<StorageT> min_lt = min_out_q_.AllocTensor<StorageT>();

                int32_t completed_kv_blocks = meta_block * block_size_;
                int32_t kv_blocks_this_iter = num_kv_blocks_in_request - completed_kv_blocks;
                if (kv_blocks_this_iter > block_size_) {
                    kv_blocks_this_iter = block_size_;
                }
                for (int32_t kv_block_offset = 0; kv_block_offset < kv_blocks_this_iter;
                     ++kv_block_offset) {
                    int32_t tokens_to_reduce;
                    if ((kv_block_offset == kv_blocks_this_iter - 1) &&
                        (meta_block == num_meta_blocks_in_request - 1)) {
                        int32_t reduced_tokens_so_far =
                            (meta_block * block_size_ + kv_block_offset) * block_size_;
                        tokens_to_reduce = seq_len - reduced_tokens_so_far;
                    } else {
                        tokens_to_reduce = block_size_;
                    }

                    int32_t kv_block_id = block_tables_gm_.GetValue(
                        request_idx * max_kv_blocks_per_request_ + completed_kv_blocks + kv_block_offset);
                    int32_t kv_block_offset_gm =
                        (kv_block_id * block_size_ * num_kv_heads_ * head_dim_) + head_idx * head_dim_;

                    LocalTensor<StorageT> k_block_lt = k_block_in_q_.AllocTensor<StorageT>();
                    DataCopyParams gm_ub_cp;
                    gm_ub_cp.blockCount = tokens_to_reduce;
                    gm_ub_cp.blockLen =
                        ceilDiv(head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_DATA_BLOCK);
                    gm_ub_cp.srcStride = ceilDiv(
                        (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(StorageT)),
                        BYTES_DATA_BLOCK);
                    gm_ub_cp.dstStride = 0;
                    DataCopy(k_block_lt, k_cache_gm_[kv_block_offset_gm], gm_ub_cp);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    ReduceBlockToOutput<true>(
                        max_lt[kv_block_offset * head_dim_],
                        k_block_lt,
                        tokens_to_reduce);
                    ReduceBlockToOutput<false>(
                        min_lt[kv_block_offset * head_dim_],
                        k_block_lt,
                        tokens_to_reduce);
                    k_block_in_q_.FreeTensor(k_block_lt);
                }

                int32_t unused_metadata_rows = block_size_ - kv_blocks_this_iter;
                if (unused_metadata_rows > 0) {
                    Duplicate<StorageT>(
                        max_lt[kv_blocks_this_iter * head_dim_],
                        static_cast<StorageT>(0),
                        unused_metadata_rows * head_dim_);
                    Duplicate<StorageT>(
                        min_lt[kv_blocks_this_iter * head_dim_],
                        static_cast<StorageT>(0),
                        unused_metadata_rows * head_dim_);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                int32_t meta_block_id = metadata_block_tables_gm_.GetValue(
                    request_idx * max_metadata_blocks_per_request_ + meta_block);
                int32_t meta_offset =
                    (meta_block_id * block_size_ * num_kv_heads_ * head_dim_) + head_idx * head_dim_;

                DataCopyParams ub_gm_cp;
                ub_gm_cp.blockCount = block_size_;
                ub_gm_cp.blockLen =
                    ceilDiv(head_dim_ * static_cast<int32_t>(sizeof(StorageT)), BYTES_UB_BLOCK);
                ub_gm_cp.srcStride = 0;
                ub_gm_cp.dstStride = ceilDiv(
                    (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(StorageT)),
                    BYTES_UB_BLOCK);
                DataCopy(maxblocks_gm_[meta_offset], max_lt, ub_gm_cp);
                DataCopy(minblocks_gm_[meta_offset], min_lt, ub_gm_cp);

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
        if constexpr (quest_is_same<StorageT, ComputeT>::value) {
            uint64_t mask = head_dim_;
            CopyRepeatParams ub_ub_cp = {1, 1, 8, 8};
            LocalTensor<ComputeT> work_lt = work_calc_buf_.Get<ComputeT>();
            Copy(work_lt, k_block_lt, mask, tokens_to_reduce, ub_ub_cp);
            ReduceTokenDim<ComputeT, isMax>(work_lt, tokens_to_reduce * head_dim_);
            Copy(out_lt, work_lt, mask, 1, ub_ub_cp);
        } else {
            ReduceCastBlockToOutput<isMax>(out_lt, k_block_lt, tokens_to_reduce);
        }
    }

    template <bool isMax>
    __aicore__ inline void ReduceCastBlockToOutput(
        LocalTensor<StorageT> out_lt,
        LocalTensor<StorageT> k_block_lt,
        int32_t tokens_to_reduce)
    {
        LocalTensor<ComputeT> work_lt = work_calc_buf_.Get<ComputeT>();
        LocalTensor<ComputeT> acc_lt = work_lt[isMax ? 0 : head_dim_];
        LocalTensor<ComputeT> chunk_lt = work_lt[2 * head_dim_];

        // BF16 chunks are reduced in FP32 and cast back to BF16 metadata rows.
        for (int32_t token_offset = 0; token_offset < tokens_to_reduce;
             token_offset += BF16_METADATA_REDUCE_CHUNK_TOKENS) {
            int32_t chunk_tokens = tokens_to_reduce - token_offset;
            if (chunk_tokens > BF16_METADATA_REDUCE_CHUNK_TOKENS) {
                chunk_tokens = BF16_METADATA_REDUCE_CHUNK_TOKENS;
            }

            Cast(
                chunk_lt,
                k_block_lt[token_offset * head_dim_],
                RoundMode::CAST_NONE,
                chunk_tokens * head_dim_);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceTokenDim<ComputeT, isMax>(chunk_lt, chunk_tokens * head_dim_);

            if (token_offset == 0) {
                CopyRow<ComputeT>(acc_lt, chunk_lt);
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

    template <typename ElementT>
    __aicore__ inline void CopyRow(LocalTensor<ElementT> dst_lt, LocalTensor<ElementT> src_lt)
    {
        if constexpr (quest_is_same<ElementT, float>::value) {
            uint64_t mask = FP32_VECTOR_MASK;
            uint8_t repeats = static_cast<uint8_t>(head_dim_ / FP32_VECTOR_MASK);
            Copy(dst_lt, src_lt, mask, repeats, {1, 1, 8, 8});
        } else {
            Copy(dst_lt, src_lt, head_dim_, 1, {1, 1, 8, 8});
        }
    }

    template <typename ElementT, bool isMax>
    __aicore__ inline void ReduceTokenDim(LocalTensor<ElementT> vec_lt, int32_t initial_length)
    {
        if (initial_length != block_size_ * head_dim_) {
            AscendC::PipeBarrier<PIPE_V>();
        }

        int32_t len = initial_length;
        while (len > head_dim_) {
            int32_t num_vec = len / head_dim_;
            int32_t pair_vec = num_vec >> 1;
            int32_t has_tail = num_vec & 1;
            int32_t reduce_len = pair_vec * head_dim_;

            if (reduce_len > 0) {
                if (isMax) {
                    Max(vec_lt[0], vec_lt[0], vec_lt[reduce_len], reduce_len);
                } else {
                    Min(vec_lt[0], vec_lt[0], vec_lt[reduce_len], reduce_len);
                }
            }

            if (has_tail) {
                CopyRow<ElementT>(vec_lt[reduce_len], vec_lt[(num_vec - 1) * head_dim_]);
                reduce_len += head_dim_;
            }

            len = reduce_len;
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
    GlobalTensor<int32_t> seq_lens_gm_;
    GlobalTensor<int32_t> metadata_block_tables_gm_;

    int32_t batch_size_;
    int32_t num_kv_heads_;
    int32_t block_size_;
    int32_t head_dim_;
    int32_t max_kv_blocks_per_request_;
    int32_t max_metadata_blocks_per_request_;
};

template <typename StorageT, typename ComputeT>
__aicore__ inline void RunQuestPrefillMetadata(
    GM_ADDR k_cache,
    GM_ADDR block_tables,
    GM_ADDR seq_lens,
    GM_ADDR metadata_block_tables,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    const QuestPrefillMetadataTilingData *tiling_data)
{
    KernelQuestMetadata<StorageT, ComputeT> op;
    op.Init(
        k_cache,
        block_tables,
        seq_lens,
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
    GM_ADDR seq_lens,
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
        RunQuestPrefillMetadata<half, half>(
            k_cache,
            block_tables,
            seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }

    if (tiling_data->dataType == QUEST_PREFILL_METADATA_DTYPE_BF16) {
        RunQuestPrefillMetadata<bfloat16_t, float>(
            k_cache,
            block_tables,
            seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }

    ASSERT(false && "Unsupported quest_prefill_metadata dtype.");
}
