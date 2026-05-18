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

constexpr int32_t ELEM_SIZE = sizeof(int32_t);  // 4 bytes
// Safety UB buffer size：32768(128KB)
constexpr uint32_t SAFE_CHUNK = 32768u;

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
        this->tail_rows = static_cast<int32_t>(tilingData.ngramInfo.tailRows);
        this->block_rows = static_cast<int32_t>(tilingData.ngramInfo.blockRows);

        int32_t align_elems = 32 / ELEM_SIZE;  // = 8
        this->max_seq_len_align = ((this->max_seq_len + align_elems - 1) / align_elems) * align_elems;
        this->max_new_tokens_align = ((this->max_new_tokens + align_elems - 1) / align_elems) * align_elems;
        this->k_align = ((this->k_val + align_elems - 1) / align_elems) * align_elems;

        this->is_large_row = (this->max_seq_len_align > static_cast<int32_t>(SAFE_CHUNK));

        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < static_cast<uint32_t>(this->former_num)) {
            this->my_rows = static_cast<uint32_t>(this->rows_per_core);
            this->row_offset = static_cast<uint32_t>(this->rows_per_core) * blockIdx;
        } else {
            this->my_rows = static_cast<uint32_t>(this->tail_rows);
            this->row_offset = static_cast<uint32_t>(this->rows_per_core) * static_cast<uint32_t>(this->former_num);
        }

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

        uint32_t br = static_cast<uint32_t>(this->block_rows);
        uint32_t br_align = ((br * ELEM_SIZE + 31) / 32) * 32 / ELEM_SIZE;

        if (!this->is_large_row) {
            pipe.InitBuffer(tokenTileBuf, br * static_cast<uint32_t>(this->max_seq_len_align) * ELEM_SIZE);
        } else {
            uint32_t chunk_ub = SAFE_CHUNK + static_cast<uint32_t>(this->max_n_val);
            uint32_t chunk_ub_align = ((chunk_ub + 7u) / 8u) * 8u;
            pipe.InitBuffer(tokenTileBuf, chunk_ub_align * ELEM_SIZE);
        }

        uint32_t mask_bytes = ((SAFE_CHUNK + 7u) / 8u);
        pipe.InitBuffer(maskBuf, mask_bytes);

        pipe.InitBuffer(sampledTileBuf, br * static_cast<uint32_t>(this->max_new_tokens_align) * ELEM_SIZE);
        pipe.InitBuffer(numTokensBuf, br_align * ELEM_SIZE);
        pipe.InitBuffer(discardTileBuf, br_align * ELEM_SIZE);
        pipe.InitBuffer(nextTokenBuf, br_align * ELEM_SIZE);
        pipe.InitBuffer(draftBuf, br * static_cast<uint32_t>(this->k_align) * ELEM_SIZE);
        pipe.InitBuffer(numValidBuf, br_align * ELEM_SIZE);
        pipe.InitBuffer(suffixBuf, static_cast<uint32_t>(this->max_n_val) * ELEM_SIZE);
    }

    __aicore__ inline void Process()
    {
        uint32_t remaining = this->my_rows;
        uint32_t cur_offset = 0;
        while (remaining > 0) {
            uint32_t cur_rows = (remaining > static_cast<uint32_t>(this->block_rows))
                                ? static_cast<uint32_t>(this->block_rows) : remaining;
            if (this->is_large_row) {
                ProcessChunkedRows(this->row_offset + cur_offset, cur_rows);
            } else {
                CopyIn(this->row_offset + cur_offset, cur_rows);
                Compute(cur_rows);
                CopyOut(this->row_offset + cur_offset, cur_rows);
            }
            cur_offset += cur_rows;
            remaining -= cur_rows;
        }
    }

private:

    __aicore__ inline void ProcessChunkedRows(uint32_t start_row, uint32_t rows)
    {
        uint32_t msl = static_cast<uint32_t>(this->max_seq_len);
        uint32_t mnta = static_cast<uint32_t>(this->max_new_tokens_align);
        uint32_t ka = static_cast<uint32_t>(this->k_align);

        auto sampledLocal = sampledTileBuf.Get<int32_t>();
        auto numTokensLocal = numTokensBuf.Get<int32_t>();
        auto discardLocal = discardTileBuf.Get<int32_t>();
        auto nextLocal = nextTokenBuf.Get<int32_t>();
        auto draftLocal = draftBuf.Get<int32_t>();
        auto numValidLocal = numValidBuf.Get<int32_t>();
        auto suffixLocal = suffixBuf.Get<int32_t>();
        auto tokenLocal = tokenTileBuf.Get<int32_t>();
        auto maskLocal = maskBuf.Get<uint8_t>();

        uint32_t metaBytes = rows * ELEM_SIZE;
        AscendC::DataCopyExtParams metaParams{1, metaBytes, 0, metaBytes, 0};
        AscendC::DataCopyPadExtParams<int32_t> noPadT{false, 0, 0, 0};
        AscendC::DataCopyPad(numTokensLocal, numTokensGm[start_row], metaParams, noPadT);
        AscendC::DataCopyPad(discardLocal, discardGm[start_row], metaParams, noPadT);

        uint32_t srcRowBytes2 = static_cast<uint32_t>(this->max_new_tokens) * ELEM_SIZE;
        uint32_t dstRowBytes2 = mnta * ELEM_SIZE;
        AscendC::DataCopyExtParams sampledParams{1, srcRowBytes2, 0, dstRowBytes2, 0};
        AscendC::DataCopyPadExtParams<int32_t> sampledPad{
            false, 0, static_cast<uint8_t>(mnta - this->max_new_tokens), 0};
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyPad(sampledLocal[static_cast<uint64_t>(r) * mnta],
                sampledGm[static_cast<uint64_t>(start_row + r) * this->max_new_tokens],
                sampledParams, sampledPad);
        }

        for (uint32_t i = 0; i < rows; ++i) {
            uint64_t gmRow = static_cast<uint64_t>(start_row + i) * msl;
            int32_t seq_len = numTokensLocal.GetValue(i);
            int32_t discard = discardLocal.GetValue(i);
            int32_t valid_count = 0;

            int32_t backup_pos = (seq_len > 0) ? (seq_len - 1) : 0;

            for (int32_t j = 0; j < this->max_new_tokens; ++j) {
                int32_t val = sampledLocal.GetValue(i * mnta + j);
                if (discard != 0) {
                    sampledLocal.SetValue(i * mnta + j, -1);
                } else if (val != -1 && val < this->vocab_size_val) {
                    valid_count++;
                } else {
                    sampledLocal.SetValue(i * mnta + j, -1);
                }
            }

            int32_t avail_space = this->max_seq_len - seq_len;
            if (avail_space < 0) avail_space = 0;
            if (valid_count > avail_space) valid_count = avail_space;

            LoadGmElements(gmRow + backup_pos, 1);
            int32_t backup_token = tokenLocal.GetValue(0);

            if (valid_count > 0) {
                nextLocal.SetValue(i, sampledLocal.GetValue(i * mnta + valid_count - 1));
            } else {
                nextLocal.SetValue(i, backup_token);
            }

            int32_t nt = seq_len + valid_count;
            if (valid_count > 0) {
                for (int32_t j = 0; j < valid_count; ++j) {
                    tokenLocal.SetValue(j, sampledLocal.GetValue(i * mnta + j));
                }
                StoreGmElements(gmRow + seq_len, valid_count);
            }

            int32_t best_match_pos = -1;
            int32_t best_ngram_len = 0;

            if (valid_count > 0 && nt >= this->min_n_val) {
                int32_t suffix_gm_start = nt - this->max_n_val;
                if (suffix_gm_start < 0) suffix_gm_start = 0;
                LoadGmElements(gmRow + suffix_gm_start, this->max_n_val);
                for (int32_t s = 0; s < this->max_n_val; ++s) {
                    suffixLocal.SetValue(static_cast<uint32_t>(s), tokenLocal.GetValue(static_cast<uint32_t>(s)));
                }

                for (int32_t ngram_len = this->min_n_val; ngram_len <= this->max_n_val; ++ngram_len) {
                    if (ngram_len > nt) break;
                    int32_t wc = nt - ngram_len;
                    if (wc <= 0) break;

                    int32_t suffix_offset = this->max_n_val - ngram_len;
                    int32_t suffix0 = suffixLocal.GetValue(static_cast<uint32_t>(suffix_offset));

                    for (int32_t chunk_start = 0; chunk_start < wc; chunk_start += SAFE_CHUNK) {
                        int32_t chunk_count = (chunk_start + SAFE_CHUNK <= wc) ? SAFE_CHUNK : (wc - chunk_start);
                        int32_t load_count = chunk_count + (ngram_len - 1);
                        if (chunk_start + load_count > nt) load_count = nt - chunk_start;
                        LoadGmElements(gmRow + chunk_start, load_count);

                        uint32_t cmp_count = ((static_cast<uint32_t>(chunk_count) + 63u) / 64u) * 64u;
                        uint32_t max_cmp = SAFE_CHUNK > 8192u ? 8192u : SAFE_CHUNK;
                        if (cmp_count > max_cmp) cmp_count = max_cmp;
                        if (cmp_count > static_cast<uint32_t>(load_count)) {
                            cmp_count = ((static_cast<uint32_t>(load_count) + 63u) / 64u) * 64u;
                        }

                        for (uint32_t cmp_off = 0; cmp_off < static_cast<uint32_t>(chunk_count); cmp_off += cmp_count) {
                            uint32_t rem = static_cast<uint32_t>(chunk_count) - cmp_off;
                            uint32_t elements = (rem >= cmp_count) ? cmp_count : rem;
                            uint32_t aligned = ((elements + 63u) / 64u) * 64u;

                            AscendC::CompareScalar<int32_t, uint8_t>(
                                maskLocal, tokenLocal[cmp_off],
                                suffix0, AscendC::CMPMODE::EQ, aligned);

                            for (uint32_t p = 0; p < elements; ++p) {
                                uint8_t bv = maskLocal.GetValue(p >> 3);
                                if (bv & (1u << (p & 7u))) {
                                    bool all_match = true;
                                    for (int32_t s = 1; s < ngram_len; ++s) {
                                        int32_t sv = suffixLocal.GetValue(static_cast<uint32_t>(suffix_offset + s));
                                        if (cmp_off + p + s < static_cast<uint32_t>(load_count)) {
                                            int32_t tv = tokenLocal.GetValue(cmp_off + p + static_cast<uint32_t>(s));
                                            if (tv != sv) { all_match = false; break; }
                                        } else {
                                            all_match = false; break;
                                        }
                                    }
                                    if (all_match) {
                                        best_match_pos = chunk_start + static_cast<int32_t>(cmp_off + p);
                                        best_ngram_len = ngram_len;
                                        break;
                                    }
                                }
                            }
                            if (best_match_pos >= 0) break;
                        }
                        if (best_match_pos >= 0) break;
                    }
                    if (best_match_pos >= 0) break;
                }
            }

            if (best_match_pos >= 0) {
                int32_t draft_start = best_match_pos + best_ngram_len;
                int32_t tokens_available = nt - draft_start;
                int32_t draft_load = (tokens_available < this->k_val) ? tokens_available : this->k_val;
                if (draft_load > 0) {
                    LoadGmElements(gmRow + draft_start, draft_load);
                    for (int32_t j = 0; j < this->k_val; ++j) {
                        if (j < draft_load) {
                            draftLocal.SetValue(i * ka + j, tokenLocal.GetValue(static_cast<uint32_t>(j)));
                        } else {
                            draftLocal.SetValue(i * ka + j, -1);
                        }
                    }
                } else {
                    for (int32_t j = 0; j < this->k_val; ++j) {
                        draftLocal.SetValue(i * ka + j, -1);
                    }
                }
            } else {
                for (int32_t j = 0; j < this->k_val; ++j) {
                    draftLocal.SetValue(i * ka + j, -1);
                }
            }

            int32_t valid_draft_count = 0;
            for (int32_t j = 0; j < this->k_val; ++j) {
                if (draftLocal.GetValue(i * ka + j) != -1) {
                    valid_draft_count++;
                } else {
                    break;
                }
            }
            numValidLocal.SetValue(i, valid_draft_count);
        }

        uint32_t metaBytes32 = static_cast<uint32_t>(rows) * ELEM_SIZE;
        AscendC::DataCopyExtParams nextParams{1, metaBytes32, 0, 0, 0};
        AscendC::DataCopyPad(nextTokensGm[start_row], nextLocal, nextParams);

        uint32_t kBytes = static_cast<uint32_t>(this->k_val) * ELEM_SIZE;
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyExtParams draftRowParams{1, kBytes, 0, 0, 0};
            AscendC::DataCopyPad(
                draftTokensGm[static_cast<uint64_t>(start_row + r) * this->k_val],
                draftLocal[static_cast<uint64_t>(r) * this->k_align], draftRowParams);
        }

        AscendC::DataCopyPad(numValidGm[start_row], numValidLocal, nextParams);
    }

    __aicore__ inline void LoadGmElements(uint64_t gm_offset, int32_t count)
    {
        if (count <= 0) return;
        auto tokenLocal = tokenTileBuf.Get<int32_t>();
        uint32_t c = static_cast<uint32_t>(count);
        uint32_t aligned = ((c + 7u) / 8u) * 8u;
        uint8_t pad = static_cast<uint8_t>(aligned - c);
        AscendC::DataCopyExtParams p{1, c * ELEM_SIZE, 0, aligned * ELEM_SIZE, 0};
        AscendC::DataCopyPadExtParams<int32_t> pp{false, 0, pad, 0};
        AscendC::DataCopyPad(tokenLocal[0], tokenGm[gm_offset], p, pp);
    }

    __aicore__ inline void StoreGmElements(uint64_t gm_offset, int32_t count)
    {
        if (count <= 0) return;
        auto tokenLocal = tokenTileBuf.Get<int32_t>();
        constexpr uint32_t STORE_MAX = 16383u;
        uint32_t c = static_cast<uint32_t>(count);
        for (uint32_t off = 0; off < c; off += STORE_MAX) {
            uint32_t chunk = (off + STORE_MAX <= c) ? STORE_MAX : (c - off);
            AscendC::DataCopyExtParams p{1, chunk * ELEM_SIZE, 0, 0, 0};
            AscendC::DataCopyPad(tokenGm[gm_offset + off], tokenLocal[off], p);
        }
    }


    __aicore__ inline void CopyIn(uint32_t start_row, uint32_t rows)
    {
        uint32_t msa = static_cast<uint32_t>(this->max_seq_len_align);
        uint32_t mnta = static_cast<uint32_t>(this->max_new_tokens_align);
        constexpr uint32_t MAX_CHUNK_ELEMS = 8192u;

        auto tokenLocal = tokenTileBuf.Get<int32_t>();
        uint32_t msl = static_cast<uint32_t>(this->max_seq_len);
        for (uint32_t r = 0; r < rows; ++r) {
            uint64_t gmRow = static_cast<uint64_t>(start_row + r) * msl;
            uint32_t ubRow = r * msa;
            for (uint32_t off = 0; off < msl; off += MAX_CHUNK_ELEMS) {
                uint32_t chunk = (off + MAX_CHUNK_ELEMS <= msl) ? MAX_CHUNK_ELEMS : (msl - off);
                uint32_t isLast = (off + chunk >= msl) ? 1u : 0u;
                uint32_t dstChunk = isLast ? (msa - off) : MAX_CHUNK_ELEMS;
                uint8_t pad = static_cast<uint8_t>(dstChunk - chunk);
                AscendC::DataCopyExtParams p{1, chunk * ELEM_SIZE, 0, dstChunk * ELEM_SIZE, 0};
                AscendC::DataCopyPadExtParams<int32_t> pp{false, 0, pad, 0};
                AscendC::DataCopyPad(tokenLocal[ubRow + off], tokenGm[gmRow + off], p, pp);
            }
        }

        auto sampledLocal = sampledTileBuf.Get<int32_t>();
        uint32_t srcRowBytes2 = static_cast<uint32_t>(this->max_new_tokens) * ELEM_SIZE;
        uint32_t dstRowBytes2 = mnta * ELEM_SIZE;
        AscendC::DataCopyExtParams sampledParams{1, srcRowBytes2, 0, dstRowBytes2, 0};
        AscendC::DataCopyPadExtParams<int32_t> sampledPad{
            false, 0, static_cast<uint8_t>(mnta - this->max_new_tokens), 0};
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyPad(sampledLocal[static_cast<uint64_t>(r) * mnta],
                sampledGm[static_cast<uint64_t>(start_row + r) * this->max_new_tokens],
                sampledParams, sampledPad);
        }

        auto numTokensLocal = numTokensBuf.Get<int32_t>();
        uint32_t metaBytes = static_cast<uint32_t>(rows) * ELEM_SIZE;
        AscendC::DataCopyExtParams metaParams{1, metaBytes, 0, metaBytes, 0};
        AscendC::DataCopyPadExtParams<int32_t> noPadT{false, 0, 0, 0};
        AscendC::DataCopyPad(numTokensLocal, numTokensGm[start_row], metaParams, noPadT);

        auto discardLocal = discardTileBuf.Get<int32_t>();
        AscendC::DataCopyPad(discardLocal, discardGm[start_row], metaParams, noPadT);
    }

    __aicore__ inline void Compute(uint32_t rows)
    {
        auto tokenLocal = tokenTileBuf.Get<int32_t>();
        auto sampledLocal = sampledTileBuf.Get<int32_t>();
        auto numTokensLocal = numTokensBuf.Get<int32_t>();
        auto discardLocal = discardTileBuf.Get<int32_t>();
        auto nextLocal = nextTokenBuf.Get<int32_t>();
        auto draftLocal = draftBuf.Get<int32_t>();
        auto numValidLocal = numValidBuf.Get<int32_t>();
        auto suffixLocal = suffixBuf.Get<int32_t>();
        auto maskLocal = maskBuf.Get<uint8_t>();

        for (uint32_t i = 0; i < rows; ++i) {
            ComputeOneRow(i, tokenLocal, sampledLocal, numTokensLocal,
                          discardLocal, nextLocal, draftLocal, numValidLocal,
                          suffixLocal, maskLocal);
        }
    }

    __aicore__ inline void ComputeOneRow(
        uint32_t idx,
        AscendC::LocalTensor<int32_t> &tokenLocal,
        AscendC::LocalTensor<int32_t> &sampledLocal,
        AscendC::LocalTensor<int32_t> &numTokensLocal,
        AscendC::LocalTensor<int32_t> &discardLocal,
        AscendC::LocalTensor<int32_t> &nextLocal,
        AscendC::LocalTensor<int32_t> &draftLocal,
        AscendC::LocalTensor<int32_t> &numValidLocal,
        AscendC::LocalTensor<int32_t> &suffixLocal,
        AscendC::LocalTensor<uint8_t> &maskLocal)
    {
        uint32_t msa = this->max_seq_len_align;
        uint32_t mnta = this->max_new_tokens_align;
        uint32_t ka = this->k_align;

        int32_t seq_len = numTokensLocal.GetValue(idx);
        int32_t discard = discardLocal.GetValue(idx);
        int32_t valid_count = 0;

        int32_t backup_pos = (seq_len > 0) ? (seq_len - 1) : 0;
        int32_t backup_token = tokenLocal.GetValue(idx * msa + backup_pos);

        for (int32_t j = 0; j < this->max_new_tokens; ++j) {
            int32_t val = sampledLocal.GetValue(idx * mnta + j);
            if (discard != 0) {
                sampledLocal.SetValue(idx * mnta + j, -1);
            } else if (val != -1 && val < this->vocab_size_val) {
                valid_count++;
            } else {
                sampledLocal.SetValue(idx * mnta + j, -1);
            }
        }

        int32_t avail_space = this->max_seq_len - seq_len;
        if (avail_space < 0) avail_space = 0;
        if (valid_count > avail_space) valid_count = avail_space;

        if (valid_count > 0) {
            nextLocal.SetValue(idx, sampledLocal.GetValue(idx * mnta + valid_count - 1));
        } else {
            nextLocal.SetValue(idx, backup_token);
        }

        int32_t num_tokens_tmp = seq_len + valid_count;
        for (int32_t j = 0; j < valid_count; ++j) {
            tokenLocal.SetValue(idx * msa + seq_len + j, sampledLocal.GetValue(idx * mnta + j));
        }

        int32_t best_match_pos = -1;
        int32_t best_ngram_len = 0;

        if (valid_count > 0 && num_tokens_tmp >= this->min_n_val) {
            if (this->block_rows <= 1) {
                int32_t nt = num_tokens_tmp;
                constexpr uint32_t CMP_MAX = 8192u;

                for (int32_t ngram_len = this->min_n_val; ngram_len <= this->max_n_val; ++ngram_len) {
                    if (ngram_len > nt) break;
                    int32_t wc = nt - ngram_len;
                    if (wc <= 0) break;

                    int32_t suffix0 = tokenLocal.GetValue(static_cast<uint32_t>(nt - ngram_len));
                    uint32_t msa_cmp = static_cast<uint32_t>(msa);

                    for (int32_t cmp_off = 0; cmp_off < wc; cmp_off += CMP_MAX) {
                        uint32_t remaining = static_cast<uint32_t>(wc - cmp_off);
                        uint32_t elements = (remaining >= CMP_MAX) ? CMP_MAX : remaining;
                        uint32_t count_aligned = ((elements + 63u) / 64u) * 64u;
                        uint32_t buf_avail = msa_cmp - static_cast<uint32_t>(cmp_off);
                        if (count_aligned > buf_avail) {
                            count_aligned = (buf_avail / 64u) * 64u;
                        }

                        if (count_aligned == 0) {
                            for (int32_t p = 0; p < static_cast<int32_t>(elements); ++p) {
                                if (tokenLocal.GetValue(static_cast<uint32_t>(cmp_off + p)) == suffix0) {
                                    bool all_match = true;
                                    for (int32_t s = 1; s < ngram_len; ++s) {
                                        int32_t sv = tokenLocal.GetValue(static_cast<uint32_t>(nt - ngram_len + s));
                                        int32_t tv = tokenLocal.GetValue(static_cast<uint32_t>(cmp_off + p + s));
                                        if (tv != sv) { all_match = false; break; }
                                    }
                                    if (all_match) {
                                        best_match_pos = cmp_off + p;
                                        best_ngram_len = ngram_len;
                                        break;
                                    }
                                }
                            }
                        } else {
                            AscendC::CompareScalar<int32_t, uint8_t>(
                                maskLocal, tokenLocal[static_cast<uint32_t>(cmp_off)],
                                suffix0, AscendC::CMPMODE::EQ, count_aligned);

                            for (int32_t p = 0; p < static_cast<int32_t>(elements); ++p) {
                                uint8_t byte_val = maskLocal.GetValue(static_cast<uint32_t>(p) >> 3);
                                if (byte_val & (1u << (static_cast<uint32_t>(p) & 7u))) {
                                    bool all_match = true;
                                    for (int32_t s = 1; s < ngram_len; ++s) {
                                        int32_t sv = tokenLocal.GetValue(static_cast<uint32_t>(nt - ngram_len + s));
                                        int32_t tv = tokenLocal.GetValue(static_cast<uint32_t>(cmp_off + p + s));
                                        if (tv != sv) { all_match = false; break; }
                                    }
                                    if (all_match) {
                                        best_match_pos = cmp_off + p;
                                        best_ngram_len = ngram_len;
                                        break;
                                    }
                                }
                            }
                        }
                        if (best_match_pos >= 0) break;
                    }
                }
            } else {
                int32_t row_base = static_cast<int32_t>(idx) * static_cast<int32_t>(msa);

                for (int32_t ngram_len = this->min_n_val; ngram_len <= this->max_n_val; ++ngram_len) {
                    if (ngram_len > num_tokens_tmp) break;

                    for (int32_t s = 0; s < ngram_len; ++s) {
                        suffixLocal.SetValue(static_cast<uint32_t>(s),
                            tokenLocal.GetValue(static_cast<uint32_t>(
                                row_base + num_tokens_tmp - ngram_len + s)));
                    }

                    int32_t max_pos = num_tokens_tmp - ngram_len - 1;
                    for (int32_t pos = 0; pos <= max_pos; ++pos) {
                        bool match = true;
                        for (int32_t s = 0; s < ngram_len; ++s) {
                            if (tokenLocal.GetValue(static_cast<uint32_t>(row_base + pos + s))
                                != suffixLocal.GetValue(static_cast<uint32_t>(s))) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            best_match_pos = pos;
                            best_ngram_len = ngram_len;
                            break;
                        }
                    }
                }
            }
        }

        if (best_match_pos >= 0) {
            int32_t draft_start = best_match_pos + best_ngram_len;
            int32_t tokens_available = num_tokens_tmp - draft_start;
            for (int32_t j = 0; j < this->k_val; ++j) {
                if (j < tokens_available) {
                    draftLocal.SetValue(idx * ka + j, tokenLocal.GetValue(idx * msa + draft_start + j));
                } else {
                    draftLocal.SetValue(idx * ka + j, -1);
                }
            }
        } else {
            for (int32_t j = 0; j < this->k_val; ++j) {
                draftLocal.SetValue(idx * ka + j, -1);
            }
        }

        int32_t valid_draft_count = 0;
        for (int32_t j = 0; j < this->k_val; ++j) {
            if (draftLocal.GetValue(idx * ka + j) != -1) {
                valid_draft_count++;
            } else {
                break;
            }
        }
        numValidLocal.SetValue(idx, valid_draft_count);
    }

    __aicore__ inline void CopyOut(uint32_t start_row, uint32_t rows)
    {
        uint32_t msa = static_cast<uint32_t>(this->max_seq_len_align);
        uint32_t msl = static_cast<uint32_t>(this->max_seq_len);
        constexpr uint32_t OUT_CHUNK_ELEMS = 8192u;

        auto tokenLocal = tokenTileBuf.Get<int32_t>();
        for (uint32_t r = 0; r < rows; ++r) {
            uint64_t gmRow = static_cast<uint64_t>(start_row + r) * msl;
            uint32_t ubRow = r * msa;
            for (uint32_t off = 0; off < msl; off += OUT_CHUNK_ELEMS) {
                uint32_t chunk = (off + OUT_CHUNK_ELEMS <= msl) ? OUT_CHUNK_ELEMS : (msl - off);
                AscendC::DataCopyExtParams p{1, chunk * ELEM_SIZE, 0, 0, 0};
                AscendC::DataCopyPad(tokenGm[gmRow + off], tokenLocal[ubRow + off], p);
            }
        }

        auto nextLocal = nextTokenBuf.Get<int32_t>();
        uint32_t metaBytes32 = static_cast<uint32_t>(rows) * ELEM_SIZE;
        AscendC::DataCopyExtParams nextParams{1, metaBytes32, 0, 0, 0};
        AscendC::DataCopyPad(nextTokensGm[start_row], nextLocal, nextParams);

        auto draftLocal = draftBuf.Get<int32_t>();
        uint32_t kBytes = static_cast<uint32_t>(this->k_val) * ELEM_SIZE;
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyExtParams draftRowParams{1, kBytes, 0, 0, 0};
            AscendC::DataCopyPad(
                draftTokensGm[static_cast<uint64_t>(start_row + r) * this->k_val],
                draftLocal[static_cast<uint64_t>(r) * this->k_align], draftRowParams);
        }

        auto numValidLocal = numValidBuf.Get<int32_t>();
        AscendC::DataCopyPad(numValidGm[start_row], numValidLocal, nextParams);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sampledTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> numTokensBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> discardTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nextTokenBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> draftBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> numValidBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> suffixBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;

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
    int32_t tail_rows;
    int32_t block_rows;
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
