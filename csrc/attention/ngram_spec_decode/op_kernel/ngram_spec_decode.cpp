// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"
#include "ngram_spec_decode.h"

const uint32_t ELEM_SIZE = 4;  // int32
constexpr uint32_t SAFE_CHUNK = 8192u;

// ----------------------------------------------------------------------------
// GM -> UB copy macro with automatic alignment padding.
// Handles unaligned GM addresses via DataCopyPad.
// ----------------------------------------------------------------------------
#define COPY_GM_TO_UB(dst, src, src_offset, count, T)                          \
    do {                                                                      \
        if ((count) > 0) {                                                    \
            constexpr uint32_t __align_elem = 8u;                             \
            uint32_t __c = static_cast<uint32_t>(count);                    \
            uint32_t __aligned = ((__c + __align_elem - 1u) / __align_elem) \
                                 * __align_elem;                              \
            uint8_t __pad = static_cast<uint8_t>(__aligned - __c);           \
            AscendC::DataCopyExtParams __p{                                   \
                1,                                                            \
                static_cast<uint32_t>(__c * sizeof(T)),                       \
                0, 0, 0};                                                     \
            AscendC::DataCopyPadExtParams<T> __pp{true, 0, __pad, -1};       \
            AscendC::DataCopyPad((dst), (src)[(src_offset)], __p, __pp);    \
        }                                                                     \
    } while (0)

// ----------------------------------------------------------------------------
// UB -> GM copy macro with chunking for large transfers.
// Ascend UB store has a single-transfer limit (16383 elements).
// ----------------------------------------------------------------------------
#define COPY_UB_TO_GM(dst, dst_offset, src, src_offset, count, T)           \
    do {                                                                      \
        if ((count) > 0) {                                                    \
            constexpr uint32_t __store_max = 16383u;                          \
            uint32_t __c = static_cast<uint32_t>(count);                    \
            for (uint32_t __off = 0; __off < __c; __off += __store_max) {    \
                uint32_t __chunk = (__off + __store_max <= __c)              \
                                       ? __store_max                          \
                                       : (__c - __off);                       \
                AscendC::DataCopyParams __p{                                  \
                    1, static_cast<uint16_t>(__chunk * sizeof(T)), 0, 0};  \
                AscendC::DataCopyPad(                                         \
                    (dst)[(dst_offset) + __off],                              \
                    (src)[(src_offset) + __off], __p);                        \
            }                                                                 \
        }                                                                     \
    } while (0)

class KernelNgramSpecDecode {
public:
    __aicore__ inline KernelNgramSpecDecode() {}

    __aicore__ inline void Init(
        GM_ADDR token_ids_gm, GM_ADDR num_tokens_gm, GM_ADDR sampled_gm,
        GM_ADDR discard_gm, GM_ADDR next_tokens_gm, GM_ADDR draft_tokens_gm,
        GM_ADDR num_valid_gm, GM_ADDR workspace, GM_ADDR tiling)
    {
        REGISTER_TILING_DEFAULT(NgramSpecDecodeTilingData);
        GET_TILING_DATA_WITH_STRUCT(NgramSpecDecodeTilingData, tilingData, tiling);

        // --------------------------------------------------------------------
        // Extract tiling parameters
        // --------------------------------------------------------------------
        this->batch_size      = tilingData.ngramInfo.batchSize;
        this->max_seq_len     = tilingData.ngramInfo.maxSeqLen;
        this->max_new_tokens  = tilingData.ngramInfo.maxNewTokens;
        this->vocab_size_val  = tilingData.ngramInfo.vocabSize;
        this->min_n_val       = tilingData.ngramInfo.minN;
        this->max_n_val       = tilingData.ngramInfo.maxN;
        this->k_val           = tilingData.ngramInfo.k;
        this->former_num      = tilingData.ngramInfo.formerNum;
        this->rows_per_core   = tilingData.ngramInfo.rowsPerCore;

        // --------------------------------------------------------------------
        // Compute alignment paddings (32-byte aligned for vector engine)
        // --------------------------------------------------------------------
        int32_t align_elems = 32 / ELEM_SIZE;
        this->max_seq_len_align    = ((this->max_seq_len    + align_elems - 1) / align_elems) * align_elems;
        this->max_new_tokens_align = ((this->max_new_tokens + align_elems - 1) / align_elems) * align_elems;
        this->k_align              = ((this->k_val          + align_elems - 1) / align_elems) * align_elems;

        // --------------------------------------------------------------------
        // Compute row distribution for this AI Core
        // --------------------------------------------------------------------
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < static_cast<uint32_t>(this->former_num)) {
            this->my_row_count = static_cast<uint32_t>(this->rows_per_core) + 1;
            this->my_row_offset = (static_cast<uint32_t>(this->rows_per_core) + 1) * blockIdx;
        } else {
            this->my_row_count  = static_cast<uint32_t>(this->rows_per_core);
            this->my_row_offset = static_cast<uint32_t>(this->rows_per_core + 1)
                            * static_cast<uint32_t>(this->former_num)
                            + this->my_row_count * (blockIdx - static_cast<uint32_t>(this->former_num));
        }

        // --------------------------------------------------------------------
        // Bind GM buffers
        // --------------------------------------------------------------------
        tokenGm.SetGlobalBuffer((__gm__ int32_t *)token_ids_gm,
                                static_cast<uint64_t>(this->batch_size) * this->max_seq_len);
        numTokensGm.SetGlobalBuffer((__gm__ int32_t *)num_tokens_gm,
                                    static_cast<uint64_t>(this->batch_size));
        sampledGm.SetGlobalBuffer((__gm__ int32_t *)sampled_gm,
                                  static_cast<uint64_t>(this->batch_size) * this->max_new_tokens);
        discardGm.SetGlobalBuffer((__gm__ int32_t *)discard_gm,
                                static_cast<uint64_t>(this->batch_size));
        nextTokensGm.SetGlobalBuffer((__gm__ int32_t *)next_tokens_gm,
                                static_cast<uint64_t>(this->batch_size));
        draftTokensGm.SetGlobalBuffer((__gm__ int32_t *)draft_tokens_gm,
                                static_cast<uint64_t>(this->batch_size) * this->k_val);
        numValidGm.SetGlobalBuffer((__gm__ int32_t *)num_valid_gm,
                                static_cast<uint64_t>(this->batch_size));

        // --------------------------------------------------------------------
        // Compute UB buffer sizes
        // --------------------------------------------------------------------
        uint32_t mnta_aligned = static_cast<uint32_t>(this->max_new_tokens_align);
        uint32_t k_aligned    = static_cast<uint32_t>(this->k_align);
        uint32_t my_rows_aligned = ((this->my_row_count + 7u) / 8u) * 8u;

        uint32_t chunk_ub       = SAFE_CHUNK + static_cast<uint32_t>(this->max_n_val);
        uint32_t chunk_ub_align = ((chunk_ub + 7u) / 8u) * 8u;
        uint32_t token_buf_size = chunk_ub_align * ELEM_SIZE;

        // --------------------------------------------------------------------
        // Initialize TQue / TBuf pipelines
        // --------------------------------------------------------------------
        // Input queues
        pipe.InitBuffer(tokenInQue,    2, token_buf_size);          // Double-buffered tokens
        pipe.InitBuffer(sampledInQue,  1, this->my_row_count * mnta_aligned * ELEM_SIZE);
        pipe.InitBuffer(numTokensInQue,1, my_rows_aligned * ELEM_SIZE);
        pipe.InitBuffer(discardInQue,  1, my_rows_aligned * ELEM_SIZE);
        pipe.InitBuffer(suffixInQue,   1, static_cast<uint32_t>(this->max_n_val) * ELEM_SIZE);

        // Output queues
        pipe.InitBuffer(nextOutQue,    1, my_rows_aligned * ELEM_SIZE);
        pipe.InitBuffer(draftOutQue,   1, this->my_row_count * k_aligned * ELEM_SIZE);
        pipe.InitBuffer(numValidOutQue,1, my_rows_aligned * ELEM_SIZE);

        // Calculation buffers
        pipe.InitBuffer(validCountBuf,  my_rows_aligned * ELEM_SIZE);
        pipe.InitBuffer(ngramCalcBuf,   token_buf_size);
        pipe.InitBuffer(ngramTempBuf,   token_buf_size);
        pipe.InitBuffer(ngramGatherBuf, token_buf_size);

        // ReduceMin requires extra scratch space for multi-iteration reduction
        uint32_t reduce_count    = SAFE_CHUNK;
        uint32_t reduce_tmp_elems = CalcReduceMinTmpSize(reduce_count, ELEM_SIZE);
        uint32_t reduce_tmp_bytes = ((reduce_tmp_elems * ELEM_SIZE + 31) / 32) * 32;
        pipe.InitBuffer(ngramReduceBuf, reduce_tmp_bytes);

    }

    // ------------------------------------------------------------------------
    // Main processing pipeline
    // ------------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        // Initialize gather indices for n-gram recurrence (byte offsets)
        auto gatherLocal = ngramGatherBuf.Get<int32_t>();
        const uint32_t gather_size = (((SAFE_CHUNK + static_cast<uint32_t>(this->max_n_val)) + 7u) / 8u) * 8u;
        AscendC::Arange<int32_t>(gatherLocal,
                                static_cast<int32_t>(sizeof(int32_t)),
                                static_cast<int32_t>(sizeof(int32_t)),
                                gather_size);

        // Stage 1: Load per-row metadata from GM to UB
        CopyInMetadata();

        auto sampledLocal   = sampledInQue.DeQue<int32_t>();
        auto numTokensLocal = numTokensInQue.DeQue<int32_t>();
        auto discardLocal   = discardInQue.DeQue<int32_t>();

        auto nextLocal     = nextOutQue.AllocTensor<int32_t>();
        auto draftLocal    = draftOutQue.AllocTensor<int32_t>();
        auto numValidLocal = numValidOutQue.AllocTensor<int32_t>();
        auto validCountLocal = validCountBuf.Get<int32_t>();

        // Stage 2: Validate sampled tokens and clamp valid counts by avail space
        for (uint32_t r = 0; r < this->my_row_count; ++r) {
            int32_t valid_count = ValidateTokens(r, this->my_row_offset + r,
                                                sampledLocal, numTokensLocal,
                                                discardLocal, nextLocal);
            validCountLocal.SetValue(r, valid_count);
        }

        // Stage 3: Synchronize MTE3 (store) and MTE2 (load)
        AscendC::TQueSync<PIPE_MTE3, PIPE_MTE2> sync_a;
        auto event = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        sync_a.SetFlag(event);
        sync_a.WaitFlag(event);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(event);

        // Stage 4: N-gram matching per row with double-buffered token loading
        int32_t preloaded_row = -1;  // Row index preloaded into tokenInQue's second slot
        for (uint32_t r = 0; r < this->my_row_count; ++r) {
            uint32_t global_row = this->my_row_offset + r;
            int32_t valid_count = validCountLocal.GetValue(r);
            int32_t seq_len     = numTokensLocal.GetValue(r);
            int32_t total_len   = seq_len + valid_count;
            bool is_short       = (total_len > 0 && total_len <= static_cast<int32_t>(SAFE_CHUNK));

            // Prefetch metadata for the next row to overlap memory with compute
            int32_t next_valid_count = -1;
            int32_t next_seq_len     = -1;
            int32_t next_total_len   = -1;
            if (r + 1 < this->my_row_count) {
                next_valid_count = validCountLocal.GetValue(r + 1);
                next_seq_len     = numTokensLocal.GetValue(r + 1);
                next_total_len   = next_seq_len + next_valid_count;
            }

            uint32_t draft_offset = r * static_cast<uint32_t>(this->k_align);
            AscendC::Duplicate(draftLocal[draft_offset], static_cast<int32_t>(-1),
                            static_cast<uint32_t>(this->k_align));

            // Skip rows that have no valid draft space or are shorter than min_n
            if (valid_count <= 0 || total_len < this->min_n_val) {
                numValidLocal.SetValue(r, 0);
                continue;
            }

            bool this_row_preloaded = (preloaded_row == static_cast<int32_t>(r));
            preloaded_row = -1;

            int32_t best_match_pos  = -1;
            int32_t best_ngram_len  = 0;
            int32_t draft_load      = 0;

            if (is_short) {
                // Short path: entire sequence fits in UB
                AscendC::LocalTensor<int32_t> tokenLocal;
                if (this_row_preloaded) {
                    tokenLocal = tokenInQue.DeQue<int32_t>();
                } else {
                    tokenLocal = LoadOneRowToken(global_row, total_len);
                }

                // Preload next row into the second buffer slot
                if (r + 1 < this->my_row_count && next_valid_count > 0 && next_total_len >= this->min_n_val) {
                    uint32_t next_global_row = this->my_row_offset + r + 1;
                    if (next_total_len <= static_cast<int32_t>(SAFE_CHUNK)) {
                        LaunchPreloadNextRow(next_global_row, next_total_len);
                    } else {
                        LaunchPreloadNextLongFirstChunk(next_global_row, next_total_len);
                    }
                    preloaded_row = static_cast<int32_t>(r + 1);
                }

                NgramMatchRowShort(total_len, tokenLocal, best_match_pos, best_ngram_len);

                if (best_match_pos >= 0) {
                    int32_t draft_start = best_match_pos + best_ngram_len;
                    int32_t avail       = total_len - draft_start;
                    draft_load = (avail < this->k_val) ? avail : this->k_val;
                    for (int32_t j = 0; j < draft_load; ++j) {
                        draftLocal.SetValue(draft_offset + static_cast<uint32_t>(j),
                                            tokenLocal.GetValue(static_cast<uint32_t>(draft_start + j)));
                    }
                }
                tokenInQue.FreeTensor(tokenLocal);
            } else {
                // Long path: chunk-based matching with double buffering
                int32_t next_global_row = (r + 1 < this->my_row_count)
                                        ? static_cast<int32_t>(this->my_row_offset + r + 1)
                                        : -1;
                int32_t preloaded_out = -1;
                NgramMatchRowLong(global_row, total_len, this_row_preloaded,
                                next_global_row, next_total_len, next_valid_count,
                                preloaded_out, best_match_pos, best_ngram_len);
                if (preloaded_out >= 0) {
                    preloaded_row = static_cast<int32_t>(r + 1);
                }

                if (best_match_pos >= 0) {
                    int32_t avail = total_len - (best_match_pos + best_ngram_len);
                    draft_load = (avail < this->k_val) ? avail : this->k_val;
                    if (draft_load > 0) {
                        uint64_t gm_row_offset = static_cast<uint64_t>(global_row)
                                                 * static_cast<uint32_t>(this->max_seq_len);
                        COPY_GM_TO_UB(draftLocal[draft_offset], tokenGm,
                                    gm_row_offset + best_match_pos + best_ngram_len,
                                    draft_load, int32_t);
                    }
                }
            }

            numValidLocal.SetValue(r, draft_load > 0 ? draft_load : 0);
        }

        // Discard any remaining preloaded tensor that was never consumed
        if (preloaded_row >= 0) {
            auto unused_token = tokenInQue.DeQue<int32_t>();
            tokenInQue.FreeTensor(unused_token);
        }

        // Release metadata and enqueue outputs
        sampledInQue.FreeTensor(sampledLocal);
        numTokensInQue.FreeTensor(numTokensLocal);
        discardInQue.FreeTensor(discardLocal);
        nextOutQue.EnQue(nextLocal);
        draftOutQue.EnQue(draftLocal);
        numValidOutQue.EnQue(numValidLocal);

        // Stage 5: Write results back to GM
        CopyOutMetadata();
    }

private:
    // ------------------------------------------------------------------------
    // Load metadata tensors from GM into UB input queues
    // ------------------------------------------------------------------------
    __aicore__ inline void CopyInMetadata()
    {
        uint32_t mnta_aligned = static_cast<uint32_t>(this->max_new_tokens_align);

        // sampled tokens: [my_row_count, max_new_tokens] -> padded to mnta_aligned
        auto sampledTensor = sampledInQue.AllocTensor<int32_t>();
        uint32_t src_row_bytes = static_cast<uint32_t>(this->max_new_tokens) * ELEM_SIZE;
        AscendC::DataCopyExtParams sampledParams{
            static_cast<uint16_t>(this->my_row_count), src_row_bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{
            true, 0, static_cast<uint8_t>(mnta_aligned - this->max_new_tokens), 0};
        AscendC::DataCopyPad(
            sampledTensor,
            sampledGm[static_cast<uint64_t>(this->my_row_offset) * this->max_new_tokens],
            sampledParams, padParams);
        sampledInQue.EnQue(sampledTensor);

        // num_tokens and discard: 1D int32 arrays
        auto numTokensTensor = numTokensInQue.AllocTensor<int32_t>();
        uint32_t meta_bytes = static_cast<uint32_t>(this->my_row_count) * ELEM_SIZE;
        AscendC::DataCopyExtParams metaParams{1, meta_bytes, 0, meta_bytes, 0};
        AscendC::DataCopyPadExtParams<int32_t> no_pad{false, 0, 0, 0};
        AscendC::DataCopyPad(numTokensTensor, numTokensGm[this->my_row_offset], metaParams, no_pad);
        numTokensInQue.EnQue(numTokensTensor);

        auto discardTensor = discardInQue.AllocTensor<int32_t>();
        AscendC::DataCopyPad(discardTensor, discardGm[this->my_row_offset], metaParams, no_pad);
        discardInQue.EnQue(discardTensor);
    }

    // ------------------------------------------------------------------------
    // Write output metadata from UB queues back to GM
    // ------------------------------------------------------------------------
    __aicore__ inline void CopyOutMetadata()
    {
        auto nextLocal     = nextOutQue.DeQue<int32_t>();
        auto draftLocal    = draftOutQue.DeQue<int32_t>();
        auto numValidLocal = numValidOutQue.DeQue<int32_t>();

        uint16_t meta_bytes16 = static_cast<uint16_t>(this->my_row_count) * ELEM_SIZE;
        AscendC::DataCopyParams nextParams{1, meta_bytes16, 0, 0};
        AscendC::DataCopyPad(nextTokensGm[this->my_row_offset], nextLocal, nextParams);
        AscendC::DataCopyPad(numValidGm[this->my_row_offset], numValidLocal, nextParams);

        // Draft tokens: write k_val elements per row (each row has k_align stride in UB)
        uint32_t k_aligned = static_cast<uint32_t>(this->k_align);
        for (uint32_t r = 0; r < this->my_row_count; ++r) {
            uint32_t k_bytes = static_cast<uint32_t>(this->k_val) * ELEM_SIZE;
            AscendC::DataCopyExtParams rowParams{1, k_bytes, 0, 0, 0};
            AscendC::DataCopyPad(
                draftTokensGm[static_cast<uint64_t>(this->my_row_offset + r) * this->k_val],
                draftLocal[r * k_aligned], rowParams);
        }

        nextOutQue.FreeTensor(nextLocal);
        draftOutQue.FreeTensor(draftLocal);
        numValidOutQue.FreeTensor(numValidLocal);
    }

    // ------------------------------------------------------------------------
    // Validate sampled tokens for one row.
    // - If discard flag is set, invalidate all sampled tokens.
    // - Clamp valid_count to available sequence space.
    // - Write the "next" token (last valid sampled, or last context token).
    // - Append valid sampled tokens back to the context token GM buffer.
    // ------------------------------------------------------------------------
    __aicore__ inline int32_t ValidateTokens(
        uint32_t local_idx, uint32_t global_row,
        AscendC::LocalTensor<int32_t>& sampledLocal,
        AscendC::LocalTensor<int32_t>& numTokensLocal,
        AscendC::LocalTensor<int32_t>& discardLocal,
        AscendC::LocalTensor<int32_t>& nextLocal)
    {
        uint32_t msl          = static_cast<uint32_t>(this->max_seq_len);
        uint32_t mnta_aligned = static_cast<uint32_t>(this->max_new_tokens_align);
        uint64_t gm_row       = static_cast<uint64_t>(global_row) * msl;
        uint32_t sampled_off  = local_idx * mnta_aligned;

        int32_t seq_len   = numTokensLocal.GetValue(local_idx);
        int32_t discard   = discardLocal.GetValue(local_idx);
        int32_t valid_count = 0;

        if (discard != 0) {
            AscendC::Duplicate(sampledLocal[sampled_off], -1, mnta_aligned);
        } else {
            for (int32_t j = 0; j < this->max_new_tokens; ++j) {
                int32_t val = sampledLocal.GetValue(sampled_off + j);
                if (val == -1 || val >= this->vocab_size_val) {
                    // Invalidate remainder on first invalid token
                    for (int32_t k = j; k < this->max_new_tokens; ++k) {
                        sampledLocal.SetValue(sampled_off + k, -1);
                    }
                    break;
                }
                valid_count++;
            }
        }

        // Clamp valid count by remaining sequence capacity
        int32_t avail_space = this->max_seq_len - seq_len;
        avail_space = (avail_space < 0) ? 0 : avail_space;
        valid_count = (valid_count > avail_space) ? avail_space : valid_count;
        int32_t total_len = seq_len + valid_count;

        // Determine next token for this position
        if (valid_count > 0) {
            int32_t last_sampled = sampledLocal.GetValue(
                sampled_off + static_cast<uint32_t>(valid_count - 1));
            nextLocal.SetValue(local_idx, last_sampled);
        } else {
            int32_t backtrack_pos = (total_len > 0) ? (total_len - 1) : 0;
            nextLocal.SetValue(local_idx, tokenGm.GetValue(gm_row + backtrack_pos));
        }

        // Append valid sampled tokens to context GM
        if (valid_count > 0) {
            COPY_UB_TO_GM(tokenGm, gm_row + seq_len,
                        sampledLocal, sampled_off, valid_count, int32_t);
        }
        return valid_count;
    }

    // ------------------------------------------------------------------------
    // Load one full row of context tokens from GM to UB (short sequences).
    // ------------------------------------------------------------------------
    __aicore__ inline AscendC::LocalTensor<int32_t> LoadOneRowToken(
        uint32_t global_row, int32_t count)
    {
        auto tensor = tokenInQue.AllocTensor<int32_t>();
        uint64_t gm_row = static_cast<uint64_t>(global_row) * static_cast<uint32_t>(this->max_seq_len);
        COPY_GM_TO_UB(tensor, tokenGm, gm_row, count, int32_t);
        tokenInQue.EnQue(tensor);
        return tokenInQue.DeQue<int32_t>();
    }

    // ------------------------------------------------------------------------
    // Preload an entire short row into tokenInQue for double buffering.
    // ------------------------------------------------------------------------
    __aicore__ inline void LaunchPreloadNextRow(uint32_t global_row, int32_t count)
    {
        auto tensor = tokenInQue.AllocTensor<int32_t>();
        uint64_t gm_row = static_cast<uint64_t>(global_row) * static_cast<uint32_t>(this->max_seq_len);
        COPY_GM_TO_UB(tensor, tokenGm, gm_row, count, int32_t);
        tokenInQue.EnQue(tensor);
    }

    // ------------------------------------------------------------------------
    // Preload only the first chunk of a long row (SAFE_CHUNK + max_n_val).
    // ------------------------------------------------------------------------
    __aicore__ inline void LaunchPreloadNextLongFirstChunk(uint32_t global_row, int32_t total_len)
    {
        int32_t load_count = SAFE_CHUNK + this->max_n_val;
        if (load_count > total_len) load_count = total_len;

        auto tensor = tokenInQue.AllocTensor<int32_t>();
        uint64_t gm_row = static_cast<uint64_t>(global_row) * static_cast<uint32_t>(this->max_seq_len);
        COPY_GM_TO_UB(tensor, tokenGm, gm_row, load_count, int32_t);
        tokenInQue.EnQue(tensor);
    }

    // ------------------------------------------------------------------------
    // Short-sequence n-gram match (entire row fits in UB).
    // Uses vectorized recurrence: diff_n = diff_{n-1} | (token - suffix[n]).
    // ------------------------------------------------------------------------
    __aicore__ inline void NgramMatchRowShort(
        int32_t total_len,
        AscendC::LocalTensor<int32_t>& tokenLocal,
        int32_t& best_match_pos,
        int32_t& best_ngram_len)
    {
        best_match_pos = -1;
        best_ngram_len = 0;

        auto ngramResult = ngramCalcBuf.Get<int32_t>();
        auto ngramTemp   = ngramTempBuf.Get<int32_t>();
        auto ngramTempF  = ngramTempBuf.Get<float>();
        auto ngramGather = ngramGatherBuf.Get<uint32_t>();
        auto ngramReduce = ngramReduceBuf.Get<float>();

        for (int32_t n = 1; n <= this->max_n_val; ++n) {
            int32_t valid_len = total_len - n;
            if (valid_len <= 0) break;

            // Recurrence: shift previous comparison result by one position
            if (n > 1) {
                AscendC::Gather<int32_t>(ngramTemp, ngramResult, ngramGather, 0, valid_len);
            }

            // Compare all positions against the n-th suffix token
            AscendC::Adds<int32_t>(ngramResult, tokenLocal,
                                -tokenLocal.GetValue(static_cast<uint32_t>(total_len - n)),
                                valid_len);

            // Merge with previous length-(n-1) match mask
            if (n > 1) {
                AscendC::Or<uint16_t>(ngramResult.ReinterpretCast<uint16_t>(),
                                    ngramResult.ReinterpretCast<uint16_t>(),
                                    ngramTemp.ReinterpretCast<uint16_t>(),
                                      static_cast<uint32_t>(valid_len * 2));
            }

            if (n < this->min_n_val) continue;

            // Reduce: find if any position has exact match (min abs diff == 0)
            AscendC::Cast<float, int32_t>(ngramTempF, ngramResult,
                                        AscendC::RoundMode::CAST_CEIL, valid_len);
            AscendC::Abs<float>(ngramTempF, ngramTempF, valid_len);
            AscendC::ReduceMin<float>(ngramReduce, ngramTempF, ngramReduce,
                                    static_cast<uint32_t>(valid_len), true);

            float min_val_f = ngramReduce.GetValue(0);
            if (min_val_f == 0.0f) {
                float min_idx_f = ngramReduce.GetValue(1);
                best_match_pos = static_cast<int32_t>(
                    *reinterpret_cast<uint32_t*>(&min_idx_f));
                best_ngram_len = n;
            } else {
                break;  // No match for length n; longer n-grams impossible
            }
        }
    }

    // ------------------------------------------------------------------------
    // Long-sequence n-gram match (chunked with double-buffered token loading).
    //
    // Algorithm:
    //   1. Load suffix (last max_n_val tokens) as query anchor.
    //   2. Iterate over the row in SAFE_CHUNK chunks.
    //   3. For each chunk, evaluate n = 1 .. max_n_val via vectorized recurrence.
    //   4. Overlap computation with preloading of next chunk or next row.
    // ------------------------------------------------------------------------
    __aicore__ inline void NgramMatchRowLong(
        uint32_t global_row,
        int32_t total_len,
        bool first_chunk_preloaded,
        int32_t next_global_row,
        int32_t next_total_len,
        int32_t next_valid_count,
        int32_t& preloaded_row_out,
        int32_t& best_match_pos,
        int32_t& best_ngram_len)
    {
        best_match_pos   = -1;
        best_ngram_len   = 0;
        preloaded_row_out = -1;

        // Load suffix: the trailing max_n_val tokens of the sequence
        int32_t suffix_gm_start = total_len - this->max_n_val;
        if (suffix_gm_start < 0) suffix_gm_start = 0;
        int32_t suffix_load = this->max_n_val;
        if (suffix_gm_start + suffix_load > total_len) {
            suffix_load = total_len - suffix_gm_start;
        }

        uint64_t gm_row_offset = static_cast<uint64_t>(global_row)
                                 * static_cast<uint32_t>(this->max_seq_len);
        auto suffixTensor = suffixInQue.AllocTensor<int32_t>();
        COPY_GM_TO_UB(suffixTensor, tokenGm, gm_row_offset + suffix_gm_start,
                    suffix_load, int32_t);
        suffixInQue.EnQue(suffixTensor);
        auto suffixLocal = suffixInQue.DeQue<int32_t>();

        // Vector calculation buffers
        auto ngramResult = ngramCalcBuf.Get<int32_t>();
        auto ngramTemp   = ngramTempBuf.Get<int32_t>();
        auto ngramTempF  = ngramTempBuf.Get<float>();
        auto ngramGather = ngramGatherBuf.Get<uint32_t>();
        auto ngramReduce = ngramReduceBuf.Get<float>();

        // Determine if next row is eligible for preloading
        bool has_next_row  = (next_total_len >= this->min_n_val) && (next_valid_count > 0);
        bool next_is_short = (next_total_len > 0 && next_total_len <= static_cast<int32_t>(SAFE_CHUNK));
        bool found_global_max = false;
        int32_t search_limit = total_len - this->min_n_val;  // Last valid n-gram start index

        // Initialize first chunk
        AscendC::LocalTensor<int32_t> tokenLocal;
        int32_t chunk_start  = 0;
        int32_t chunk_count  = (SAFE_CHUNK <= search_limit) ? SAFE_CHUNK : search_limit;
        int32_t load_count   = chunk_count + this->max_n_val;
        if (chunk_start + load_count > total_len) {
            load_count = total_len - chunk_start;
        }

        if (first_chunk_preloaded) {
            tokenLocal = tokenInQue.DeQue<int32_t>();
        } else {
            auto t0 = tokenInQue.AllocTensor<int32_t>();
            COPY_GM_TO_UB(t0, tokenGm, gm_row_offset + chunk_start, load_count, int32_t);
            tokenInQue.EnQue(t0);
            tokenLocal = tokenInQue.DeQue<int32_t>();
        }

        bool has_preloaded    = false;   // Next chunk tensor is waiting in tokenInQue
        bool preloaded_next_row = false; // The waiting tensor belongs to the next row

        // Chunked matching loop
        while (!found_global_max) {
            int32_t next_chunk_start = chunk_start + SAFE_CHUNK;
            bool has_next_chunk = (next_chunk_start < search_limit);

            // Preload priority: next chunk of current row > next row's first chunk
            if (has_next_chunk) {
                int32_t next_chunk_count = (next_chunk_start + SAFE_CHUNK <= search_limit)
                                        ? SAFE_CHUNK
                                        : (search_limit - next_chunk_start);
                int32_t next_load_count = next_chunk_count + this->max_n_val;
                if (next_chunk_start + next_load_count > total_len) {
                    next_load_count = total_len - next_chunk_start;
                }
                auto tn = tokenInQue.AllocTensor<int32_t>();
                COPY_GM_TO_UB(tn, tokenGm, gm_row_offset + next_chunk_start,
                            next_load_count, int32_t);
                tokenInQue.EnQue(tn);
                has_preloaded = true;
            } else if (has_next_row) {
                uint32_t next_gr = static_cast<uint32_t>(next_global_row);
                if (next_is_short) {
                    LaunchPreloadNextRow(next_gr, next_total_len);
                } else {
                    LaunchPreloadNextLongFirstChunk(next_gr, next_total_len);
                }
                preloaded_next_row = true;
                preloaded_row_out  = 1;  // Signal caller that next row is ready
            }

            // Evaluate n-gram lengths 1..max_n_val on current chunk
            for (int32_t n = 1; n <= this->max_n_val; ++n) {
                int32_t valid_len = load_count - n;
                if (valid_len <= 0) break;

                // Recurrence: shift previous comparison result
                if (n > 1) {
                    AscendC::Gather<int32_t>(ngramTemp, ngramResult,
                                            ngramGather, 0, valid_len);
                }

                // Vectorized subtraction against n-th suffix token
                int32_t suffix_idx = suffix_load - n;
                AscendC::Adds<int32_t>(ngramResult, tokenLocal,
                                    -suffixLocal.GetValue(static_cast<uint32_t>(suffix_idx)),
                                    valid_len);

                // Merge match masks across n-gram lengths
                if (n > 1) {
                    AscendC::Or<uint16_t>(ngramResult.ReinterpretCast<uint16_t>(),
                                        ngramResult.ReinterpretCast<uint16_t>(),
                                        ngramTemp.ReinterpretCast<uint16_t>(),
                                          static_cast<uint32_t>(valid_len * 2));
                }

                if (n < this->min_n_val) continue;

                // Clamp evaluation to actual chunk boundary (exclude overflow tail)
                int32_t eval_count = (chunk_start + chunk_count <= total_len - n)
                                    ? chunk_count
                                    : (total_len - n - chunk_start);
                if (eval_count <= 0) break;

                // Reduce to check for exact match in this chunk
                AscendC::Cast<float, int32_t>(ngramTempF, ngramResult,
                                            AscendC::RoundMode::CAST_CEIL, eval_count);
                AscendC::Abs<float>(ngramTempF, ngramTempF, eval_count);
                AscendC::ReduceMin<float>(ngramReduce, ngramTempF, ngramReduce,
                                        static_cast<uint32_t>(eval_count), true);

                float min_val_f = ngramReduce.GetValue(0);
                if (min_val_f == 0.0f) {
                    if (n > best_ngram_len) {
                        float min_idx_f = ngramReduce.GetValue(1);
                        uint32_t pos_u  = *reinterpret_cast<uint32_t*>(&min_idx_f);
                        best_match_pos  = chunk_start + static_cast<int32_t>(pos_u);
                        best_ngram_len  = n;
                        if (n == this->max_n_val) {
                            found_global_max = true;
                            break;
                        }
                    }
                } else {
                    break;  // No match for this n; longer n-grams cannot match
                }
            }

            if (found_global_max || !has_next_chunk) break;

            // Advance to next chunk, consuming the preloaded tensor
            tokenInQue.FreeTensor(tokenLocal);
            tokenLocal   = tokenInQue.DeQue<int32_t>();
            has_preloaded = false;

            chunk_start = next_chunk_start;
            chunk_count = (chunk_start + SAFE_CHUNK <= search_limit)
                        ? SAFE_CHUNK
                        : (search_limit - chunk_start);
            load_count  = chunk_count + this->max_n_val;
            if (chunk_start + load_count > total_len) {
                load_count = total_len - chunk_start;
            }
        }

        // Clean up unconsumed preloaded chunk (if it was a chunk, not next row)
        if (has_preloaded && !preloaded_next_row) {
            auto extra = tokenInQue.DeQue<int32_t>();
            tokenInQue.FreeTensor(extra);
        }

        tokenInQue.FreeTensor(tokenLocal);
        suffixInQue.FreeTensor(suffixLocal);
    }

    // ------------------------------------------------------------------------
    // Compute scratch-space size for AscendC::ReduceMin (multi-iteration).
    // ------------------------------------------------------------------------
    __aicore__ inline uint32_t CalcReduceMinTmpSize(uint32_t count, uint32_t typeSize)
    {
        uint32_t elementsPerBlock  = 32 / typeSize;
        uint32_t elementsPerRepeat = 256 / typeSize;

        auto RoundUp = [](uint32_t x, uint32_t unit) -> uint32_t {
            return (x + unit - 1) / unit;
        };

        uint32_t firstMaxRepeat   = RoundUp(count, elementsPerRepeat);
        uint32_t iter1OutputCount = firstMaxRepeat * 2;
        uint32_t iter2AlignStart  = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;
        uint32_t iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;
        uint32_t iter3AlignStart  = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock;
        uint32_t iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;
        uint32_t iter3AlignEnd    = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;

        return iter2AlignStart + iter3AlignStart + iter3AlignEnd;
    }

private:
    // ------------------------------------------------------------------------
    // TPipe and TQue/TBuf declarations
    // ------------------------------------------------------------------------
    AscendC::TPipe pipe;

    // Input queues
    AscendC::TQue<AscendC::TPosition::VECIN, 2> tokenInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> sampledInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> numTokensInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> discardInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> suffixInQue;

    // Output queues
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> nextOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> draftOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> numValidOutQue;

    // Calculation buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramCalcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramTempBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramGatherBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramReduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> validCountBuf;

    // GM tensors
    AscendC::GlobalTensor<int32_t> tokenGm;
    AscendC::GlobalTensor<int32_t> numTokensGm;
    AscendC::GlobalTensor<int32_t> sampledGm;
    AscendC::GlobalTensor<int32_t> discardGm;
    AscendC::GlobalTensor<int32_t> nextTokensGm;
    AscendC::GlobalTensor<int32_t> draftTokensGm;
    AscendC::GlobalTensor<int32_t> numValidGm;

    // Tiling scalars
    int32_t batch_size;
    int32_t max_seq_len;
    int32_t max_seq_len_align;
    int32_t max_new_tokens;
    int32_t max_new_tokens_align;
    int32_t k_val;
    int32_t k_align;
    int32_t vocab_size_val;
    int32_t min_n_val;
    int32_t max_n_val;
    int32_t former_num;
    int32_t rows_per_core;
    uint32_t my_row_count;
    uint32_t my_row_offset;
};

extern "C" __global__ __aicore__ void ngram_spec_decode(
    GM_ADDR token_ids, GM_ADDR num_tokens, GM_ADDR sampled,
    GM_ADDR discard, GM_ADDR next_tokens, GM_ADDR draft_tokens,
    GM_ADDR num_valid, GM_ADDR workspace, GM_ADDR tiling)
{
    KernelNgramSpecDecode op;
    op.Init(token_ids, num_tokens, sampled, discard, next_tokens,
            draft_tokens, num_valid, workspace, tiling);
    op.Process();
}
