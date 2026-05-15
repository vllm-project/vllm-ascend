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

#define COPY_GM_TO_UB(dst, src, src_offset, count, T)                      \
    do {                                                                   \
        if ((count) > 0) {                                                 \
            constexpr uint32_t __align_elem = 8u;                          \
            uint32_t __c = static_cast<uint32_t>(count);                   \
            uint32_t __aligned = ((__c + __align_elem - 1u) / __align_elem) \
                                 * __align_elem;                           \
            uint8_t __pad = static_cast<uint8_t>(__aligned - __c);          \
            AscendC::DataCopyExtParams __p{                                \
                1,                                                         \
                static_cast<uint32_t>(__c * sizeof(T)),                    \
                0,                                                         \
                0,                                                         \
                0};                                                        \
            AscendC::DataCopyPadExtParams<T> __pp{true, 0, __pad, -1};     \
            AscendC::DataCopyPad((dst), (src)[(src_offset)], __p, __pp);    \
        }                                                                  \
    } while (0)

#define COPY_UB_TO_GM(dst, src, dst_offset, count, T)                      \
    do {                                                                   \
        if ((count) > 0) {                                                 \
            constexpr uint32_t __store_max = 16383u;                       \
            uint32_t __c = static_cast<uint32_t>(count);                   \
            for (uint32_t __off = 0; __off < __c; __off += __store_max) {  \
                uint32_t __chunk = (__off + __store_max <= __c)            \
                                    ? __store_max                          \
                                    : (__c - __off);                       \
                AscendC::DataCopyParams __p{                               \
                    1,                                                     \
                    static_cast<uint16_t>(__chunk * sizeof(T)),          \
                    0,                                                     \
                    0};                                                    \
                AscendC::DataCopyPad(                                      \
                    (dst)[(dst_offset) + __off], (src)[__off], __p);       \
            }                                                              \
        }                                                                  \
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

        this->batch_size = static_cast<int32_t>(tilingData.ngramInfo.batchSize);
        this->max_seq_len = static_cast<int32_t>(tilingData.ngramInfo.maxSeqLen);
        this->max_new_tokens = static_cast<int32_t>(tilingData.ngramInfo.maxNewTokens);
        this->vocab_size_val = static_cast<int32_t>(tilingData.ngramInfo.vocabSize);
        this->min_n_val = static_cast<int32_t>(tilingData.ngramInfo.minN);
        this->max_n_val = static_cast<int32_t>(tilingData.ngramInfo.maxN);
        this->k_val = static_cast<int32_t>(tilingData.ngramInfo.k);
        this->former_num = static_cast<int32_t>(tilingData.ngramInfo.formerNum);
        this->rows_per_core = static_cast<int32_t>(tilingData.ngramInfo.rowsPerCore);
        
        // Align dimensions to 32-byte boundaries (8 elements for int32)
        int32_t align_elems = 32 / ELEM_SIZE;
        this->max_seq_len_align = ((this->max_seq_len + align_elems - 1) / align_elems) * align_elems;
        this->max_new_tokens_align = ((this->max_new_tokens + align_elems - 1) / align_elems) * align_elems;
        this->k_align = ((this->k_val + align_elems - 1) / align_elems) * align_elems;

        this->is_large_row = (this->max_seq_len_align > static_cast<int32_t>(SAFE_CHUNK));

        // Compute row distribution across cores
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < static_cast<uint32_t>(this->former_num)) {
            this->my_rows = static_cast<uint32_t>(this->rows_per_core) + 1;
            this->row_offset = (static_cast<uint32_t>(this->rows_per_core) + 1) * blockIdx;
        } else {
            this->my_rows = static_cast<uint32_t>(this->rows_per_core);
            this->row_offset = static_cast<uint32_t>(this->rows_per_core + 1)
                             * static_cast<uint32_t>(this->former_num)
                             + this->my_rows * (blockIdx - static_cast<uint32_t>(this->former_num));
        }

        // Bind Global Memory tensors
        tokenGm.SetGlobalBuffer(
            (__gm__ int32_t *)token_ids_gm,
            static_cast<uint64_t>(this->batch_size) * this->max_seq_len);
        numTokensGm.SetGlobalBuffer(
            (__gm__ int32_t *)num_tokens_gm,
            static_cast<uint64_t>(this->batch_size));
        sampledGm.SetGlobalBuffer(
            (__gm__ int32_t *)sampled_gm,
            static_cast<uint64_t>(this->batch_size) * this->max_new_tokens);
        discardGm.SetGlobalBuffer(
            (__gm__ int32_t *)discard_gm,
            static_cast<uint64_t>(this->batch_size));
        nextTokensGm.SetGlobalBuffer(
            (__gm__ int32_t *)next_tokens_gm,
            static_cast<uint64_t>(this->batch_size));
        draftTokensGm.SetGlobalBuffer(
            (__gm__ int32_t *)draft_tokens_gm,
            static_cast<uint64_t>(this->batch_size) * this->k_val);
        numValidGm.SetGlobalBuffer(
            (__gm__ int32_t *)num_valid_gm,
            static_cast<uint64_t>(this->batch_size));

        // ============================================================
        // TQue Initialization (depth=1, single-buffer, EnQue/DeQue for event sync)
        // ============================================================
        uint32_t mnta = static_cast<uint32_t>(this->max_new_tokens_align);
        uint32_t ka = static_cast<uint32_t>(this->k_align);
        uint32_t my_rows_align = ((this->my_rows + 7u) / 8u) * 8u;

        uint32_t token_buf_size = 0;
        if (!this->is_large_row) {
            token_buf_size = static_cast<uint32_t>(this->max_seq_len_align) * ELEM_SIZE;
        } else {
            uint32_t chunk_ub = SAFE_CHUNK + static_cast<uint32_t>(this->max_n_val);
            uint32_t chunk_ub_align = ((chunk_ub + 7u) / 8u) * 8u;
            token_buf_size = chunk_ub_align * ELEM_SIZE;
        }

        // VECIN: Input data queues (GM -> UB, MTE2 direction)
        pipe.InitBuffer(tokenInQue, 1, token_buf_size);
        pipe.InitBuffer(sampledInQue, 1, this->my_rows * mnta * ELEM_SIZE);
        pipe.InitBuffer(numTokensInQue, 1, my_rows_align * ELEM_SIZE);
        pipe.InitBuffer(discardInQue, 1, my_rows_align * ELEM_SIZE);
        pipe.InitBuffer(suffixInQue, 1, static_cast<uint32_t>(this->max_n_val) * ELEM_SIZE);

        // VECOUT: Output data queues (UB -> GM, MTE3 direction)
        pipe.InitBuffer(nextOutQue, 1, my_rows_align * ELEM_SIZE);
        pipe.InitBuffer(draftOutQue, 1, this->my_rows * ka * ELEM_SIZE);
        pipe.InitBuffer(numValidOutQue, 1, my_rows_align * ELEM_SIZE);

        // Pure UB computation buffers (no GM traffic, keep as TBuf)
        pipe.InitBuffer(ngramCalcBuf, token_buf_size);
        pipe.InitBuffer(ngramTempBuf, token_buf_size);
        pipe.InitBuffer(ngramGatherBuf, token_buf_size);

        uint32_t reduce_count = this->is_large_row
                              ? SAFE_CHUNK
                              : (static_cast<uint32_t>(this->max_seq_len) - 1);
        uint32_t reduce_tmp_elems = CalcReduceMinTmpSize(reduce_count, ELEM_SIZE);
        uint32_t reduce_tmp_bytes = ((reduce_tmp_elems * ELEM_SIZE + 31) / 32) * 32;
        pipe.InitBuffer(ngramReduceBuf, reduce_tmp_bytes);
    }

    __aicore__ inline void Process()
    {
        // Step 1: Enqueue input data (MTE2 transfer in)
        CopyInMetadata();

        // Step 2: Dequeue input tensors before loop (sync point: ensure MTE2 done)
        auto sampledLocal = sampledInQue.DeQue<int32_t>();
        auto numTokensLocal = numTokensInQue.DeQue<int32_t>();
        auto discardLocal = discardInQue.DeQue<int32_t>();

        // Step 3: Allocate output tensors (not enqueued yet, only reserve UB space)
        auto nextTensor = nextOutQue.AllocTensor<int32_t>();
        auto draftTensor = draftOutQue.AllocTensor<int32_t>();
        auto numValidTensor = numValidOutQue.AllocTensor<int32_t>();

        // Step 4: Row-by-row computation (output tensors passed by reference, written directly)
        for (uint32_t r = 0; r < this->my_rows; ++r) {
            uint32_t global_row = this->row_offset + r;
            ProcessOneRow(
                r, global_row,
                sampledLocal, numTokensLocal, discardLocal,
                nextTensor, draftTensor, numValidTensor);
        }

        // Step 5: Free input tensors
        sampledInQue.FreeTensor(sampledLocal);
        numTokensInQue.FreeTensor(numTokensLocal);
        discardInQue.FreeTensor(discardLocal);

        // Step 6: Enqueue output data (sync point: mark vector compute done, allow MTE3 out)
        nextOutQue.EnQue(nextTensor);
        draftOutQue.EnQue(draftTensor);
        numValidOutQue.EnQue(numValidTensor);

        // Step 7: Dequeue output and copy to GM
        CopyOutMetadata();
    }

private:
    __aicore__ inline void CopyInMetadata()
    {
        uint32_t mnta = static_cast<uint32_t>(this->max_new_tokens_align);

        // sampled: Bulk copy of my_rows lines (repeat mode)
        auto sampledTensor = sampledInQue.AllocTensor<int32_t>();
        uint32_t srcRowBytes = static_cast<uint32_t>(this->max_new_tokens) * ELEM_SIZE;
        AscendC::DataCopyExtParams sampledParams{
            static_cast<uint16_t>(this->my_rows),  
            srcRowBytes,                            
            0,                                     
            0,                                     
            0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{
            true, 0, static_cast<uint8_t>(mnta - this->max_new_tokens), 0};
        AscendC::DataCopyPad(
            sampledTensor,
            sampledGm[static_cast<uint64_t>(this->row_offset) * this->max_new_tokens],
            sampledParams, padParams);
        sampledInQue.EnQue(sampledTensor);

        // numTokens / discard: 1D array copy (32-byte aligned)
        auto numTokensTensor = numTokensInQue.AllocTensor<int32_t>();
        uint32_t metaBytes = static_cast<uint32_t>(this->my_rows) * ELEM_SIZE;
        AscendC::DataCopyExtParams metaParams{1, metaBytes, 0, metaBytes, 0};
        AscendC::DataCopyPadExtParams<int32_t> noPadT{false, 0, 0, 0};
        AscendC::DataCopyPad(numTokensTensor, numTokensGm[this->row_offset], metaParams, noPadT);
        numTokensInQue.EnQue(numTokensTensor);

        auto discardTensor = discardInQue.AllocTensor<int32_t>();
        AscendC::DataCopyPad(discardTensor, discardGm[this->row_offset], metaParams, noPadT);
        discardInQue.EnQue(discardTensor);
    }

    __aicore__ inline void CopyOutMetadata()
    {
        // Dequeue output data (sync point: ensure vector computation completed)
        auto nextLocal = nextOutQue.DeQue<int32_t>();
        auto draftLocal = draftOutQue.DeQue<int32_t>();
        auto numValidLocal = numValidOutQue.DeQue<int32_t>();

        // nextToken / numValid: 1D array write-back
        uint16_t metaBytes16 = static_cast<uint16_t>(this->my_rows) * ELEM_SIZE;
        AscendC::DataCopyParams nextParams{1, metaBytes16, 0, 0};
        AscendC::DataCopyPad(nextTokensGm[this->row_offset], nextLocal, nextParams);
        AscendC::DataCopyPad(numValidGm[this->row_offset], numValidLocal, nextParams);

        // draftTokens: 2D array write-back (repeat mode, handle UB k_align padding)
        uint32_t kBytes = static_cast<uint32_t>(this->k_val) * ELEM_SIZE;
        AscendC::DataCopyParams draftParams{
            static_cast<uint16_t>(this->my_rows),  
            static_cast<uint16_t>(kBytes),          
            0,                                     
            0};                                 
        AscendC::DataCopyPad(
            draftTokensGm[static_cast<uint64_t>(this->row_offset) * this->k_val],
            draftLocal,
            draftParams);

        nextOutQue.FreeTensor(nextLocal);
        draftOutQue.FreeTensor(draftLocal);
        numValidOutQue.FreeTensor(numValidLocal);
    }

    __aicore__ inline void ProcessOneRow(
        uint32_t local_idx, uint32_t global_row,
        AscendC::LocalTensor<int32_t>& sampledLocal,
        AscendC::LocalTensor<int32_t>& numTokensLocal,
        AscendC::LocalTensor<int32_t>& discardLocal,
        AscendC::LocalTensor<int32_t>& nextTensor,
        AscendC::LocalTensor<int32_t>& draftTensor,
        AscendC::LocalTensor<int32_t>& numValidTensor)
    {
        uint32_t msl = static_cast<uint32_t>(this->max_seq_len);
        uint32_t mnta = static_cast<uint32_t>(this->max_new_tokens_align);
        uint32_t ka = static_cast<uint32_t>(this->k_align);
        uint64_t gmRow = static_cast<uint64_t>(global_row) * msl;

        // Step 1: Load pre-fetched metadata
        int32_t seq_len = numTokensLocal.GetValue(local_idx);
        int32_t discard = discardLocal.GetValue(local_idx);
        int32_t valid_count = 0;
        uint32_t sampled_offset = local_idx * mnta;

        // Step 2: Process sampled tokens (validation & truncation)
        if (discard != 0) {
            AscendC::Duplicate(sampledLocal[sampled_offset], -1, this->max_new_tokens_align);
        } else {
            for (int32_t j = 0; j < this->max_new_tokens; ++j) {
                int32_t val = sampledLocal.GetValue(sampled_offset + j);
                if (val == -1 || val >= this->vocab_size_val) {
                    for (int32_t k = j; k < this->max_new_tokens; ++k) {
                        sampledLocal.SetValue(sampled_offset + k, -1);
                    }
                    break;
                }
                valid_count++;
            }
        }

        int32_t avail_space = this->max_seq_len - seq_len;
        if (avail_space < 0) {
            avail_space = 0;
        }
        if (valid_count > avail_space) {
            valid_count = avail_space;
        }

        // Step 4: Append sampled tokens to history sequence (sampled -> tokenGm) — MTE3
        int32_t nt = seq_len + valid_count;
        if (valid_count > 0) {
            COPY_UB_TO_GM(tokenGm, sampledLocal[sampled_offset], gmRow + seq_len, valid_count, int32_t);
            AscendC::TQueSync<PIPE_MTE3, PIPE_S> sync;
            AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_S>();
            sync.SetFlag(eventID);
            sync.WaitFlag(eventID);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_S>(eventID);
        }

        // Leverage GetValue blocking semantics for implicit MTE3 -> MTE2 synchronization
        // Read the last token of current sequence as nextToken
        int32_t backup_pos = (nt > 0) ? (nt - 1) : 0;
        int32_t backup_token = tokenGm.GetValue(gmRow + backup_pos);
        nextTensor.SetValue(local_idx, backup_token);

        // Step 5: N-gram vectorized recursive matching
        int32_t best_match_pos = -1;
        int32_t best_ngram_len = 0;

        if (valid_count > 0 && nt >= this->min_n_val) {
            int32_t suffix_gm_start = nt - this->max_n_val;
            if (suffix_gm_start < 0) {
                suffix_gm_start = 0;
            }
            int32_t suffix_load = this->max_n_val;
            if (suffix_gm_start + suffix_load > nt) {
                suffix_load = nt - suffix_gm_start;
            }

            // Synchronous suffix load
            auto suffixTensor = suffixInQue.AllocTensor<int32_t>();
            COPY_GM_TO_UB(suffixTensor, tokenGm, gmRow + suffix_gm_start, suffix_load, int32_t);
            suffixInQue.EnQue(suffixTensor);
            auto suffixLocal = suffixInQue.DeQue<int32_t>();

            auto ngramResult = ngramCalcBuf.Get<int32_t>();
            auto ngramTemp = ngramTempBuf.Get<int32_t>();
            auto ngramTempF = ngramTempBuf.Get<float>();
            auto ngramGather = ngramGatherBuf.Get<int32_t>();
            auto ngramReduce = ngramReduceBuf.Get<float>();

            uint32_t max_gather_len = this->is_large_row
                                    ? (SAFE_CHUNK + static_cast<uint32_t>(this->max_n_val))
                                    : static_cast<uint32_t>(this->max_seq_len);
            if (this->max_n_val > 1) {
                AscendC::Arange<int32_t>(
                    ngramGather,
                    static_cast<int32_t>(sizeof(int32_t)),
                    static_cast<int32_t>(sizeof(int32_t)),
                    static_cast<int32_t>(max_gather_len));
            }

            bool found_global_max = false;

            for (int32_t chunk_start = 0;
                 chunk_start < nt - this->min_n_val && !found_global_max;
                 chunk_start += SAFE_CHUNK)
            {
                int32_t chunk_count = (chunk_start + SAFE_CHUNK <= nt - this->min_n_val)
                                      ? SAFE_CHUNK
                                      : (nt - this->min_n_val - chunk_start);
                if (chunk_count <= 0) {
                    break;
                }

                int32_t load_count = chunk_count + this->max_n_val;
                if (chunk_start + load_count > nt) {
                    load_count = nt - chunk_start;
                }
                if (load_count <= 0) {
                    continue;
                }

                // Synchronous chunk load
                auto chunkTensor = tokenInQue.AllocTensor<int32_t>();
                COPY_GM_TO_UB(chunkTensor, tokenGm, gmRow + chunk_start, load_count, int32_t);
                tokenInQue.EnQue(chunkTensor);
                auto tokenLocal = tokenInQue.DeQue<int32_t>();

                for (int32_t n = 1; n <= this->max_n_val; ++n) {
                    int32_t valid_len = load_count - n;
                    if (valid_len <= 0) {
                        break;
                    }

                    if (n > 1) {
                        AscendC::Gather<int32_t>(
                            ngramTemp, ngramResult,
                            ngramGather.ReinterpretCast<uint32_t>(), 0, valid_len);
                    }

                    int32_t s_val = suffixLocal.GetValue(static_cast<uint32_t>(suffix_load - n));
                    AscendC::Adds<int32_t>(ngramResult, tokenLocal, -s_val, valid_len);

                    if (n > 1) {
                        AscendC::Or<uint16_t>(
                            ngramResult.ReinterpretCast<uint16_t>(),
                            ngramResult.ReinterpretCast<uint16_t>(),
                            ngramTemp.ReinterpretCast<uint16_t>(),
                            valid_len * 2);
                    }

                    if (n >= this->min_n_val) {
                        int32_t check_count = (chunk_start + chunk_count <= nt - n)
                                            ? chunk_count
                                            : (nt - n - chunk_start);
                        if (check_count > 0) {
                            AscendC::Cast<float, int32_t>(
                                ngramTempF, ngramResult,
                                AscendC::RoundMode::CAST_CEIL, check_count);
                            AscendC::Abs<float>(ngramTempF, ngramTempF, check_count);
                            AscendC::ReduceMin<float>(
                                ngramReduce, ngramTempF, ngramReduce, check_count, true);

                            float min_val_f = ngramReduce.GetValue(0);
                            if (min_val_f == 0.0f) {
                                if (n > best_ngram_len) {
                                    float min_idx_f = ngramReduce.GetValue(1);
                                    uint32_t pos = *reinterpret_cast<uint32_t*>(&min_idx_f);
                                    best_match_pos = chunk_start + static_cast<int32_t>(pos);
                                    best_ngram_len = n;

                                    if (n == this->max_n_val) {
                                        found_global_max = true;
                                        break;
                                    }
                                }
                            } else {
                                // Current chunk failed for this n; larger n impossible
                                break;
                            }
                        }
                    }
                }

                tokenInQue.FreeTensor(tokenLocal);
            }

            suffixInQue.FreeTensor(suffixLocal);
        }

        // Step 6: Draft generation (write directly to VECOUT tensor)
        uint32_t draft_offset = local_idx * ka;
        int32_t valid_draft_count = 0;
        AscendC::Duplicate(draftTensor[draft_offset], -1, this->k_align);

        if (best_match_pos >= 0) {
            int32_t draft_start = best_match_pos + best_ngram_len;
            int32_t tokens_available = nt - draft_start;
            valid_draft_count = (tokens_available < this->k_val) ? tokens_available : this->k_val;
            if (valid_draft_count > 0) {
                COPY_GM_TO_UB(
                    draftTensor[draft_offset], tokenGm, gmRow + draft_start, valid_draft_count, int32_t);
            }
        }

        numValidTensor.SetValue(local_idx, valid_draft_count >= 0 ? valid_draft_count : 0);
    }

    __aicore__ inline uint32_t CalcReduceMinTmpSize(uint32_t count, uint32_t typeSize)
    {
        uint32_t elementsPerBlock = 32 / typeSize;
        uint32_t elementsPerRepeat = 256 / typeSize;
        auto RoundUp = [](uint32_t x, uint32_t unit) -> uint32_t {
            return (x + unit - 1) / unit;
        };
        uint32_t firstMaxRepeat = RoundUp(count, elementsPerRepeat);
        uint32_t iter1OutputCount = firstMaxRepeat * 2;
        uint32_t iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;
        uint32_t iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;
        uint32_t iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock;
        uint32_t iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;
        uint32_t iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;
        return iter2AlignStart + iter3AlignStart + iter3AlignEnd;
    }

private:
    AscendC::TPipe pipe;

    // Input queues (VECIN): EnQue after MTE2 in, DeQue guarantees transfer completion
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tokenInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> sampledInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> numTokensInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> discardInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> suffixInQue;

    // Output queues (VECOUT): EnQue after compute, DeQue guarantees MTE3 out sync
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> nextOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> draftOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> numValidOutQue;

    // Pure UB computation buffers (no GM traffic, kept as TBuf)
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramCalcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramTempBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramGatherBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramReduceBuf;

    AscendC::GlobalTensor<int32_t> tokenGm;
    AscendC::GlobalTensor<int32_t> numTokensGm;
    AscendC::GlobalTensor<int32_t> sampledGm;
    AscendC::GlobalTensor<int32_t> discardGm;
    AscendC::GlobalTensor<int32_t> nextTokensGm;
    AscendC::GlobalTensor<int32_t> draftTokensGm;
    AscendC::GlobalTensor<int32_t> numValidGm;

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
    uint32_t my_rows;
    uint32_t row_offset;
    bool is_large_row;
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
