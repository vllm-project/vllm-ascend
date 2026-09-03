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
#ifndef MSA_INDEX_SCORE_TORCH_ADPT_H
#define MSA_INDEX_SCORE_TORCH_ADPT_H

namespace vllm_ascend {
namespace msa_index_score_detail {

constexpr int64_t QUERY_DIM_NUM = 3;
constexpr int64_t KEY_CACHE_DIM_NUM = 3;
constexpr int64_t KEY_CACHE_WITH_HEAD_DIM_NUM = 4;
constexpr int64_t BLOCK_TABLE_DIM_NUM = 2;
constexpr int64_t SEQ_LEN_DIM_NUM = 1;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t SCORE_ALIGNMENT = 16;

void CheckMsaIndexScoreParams(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& blockTable, const at::Tensor& startLoc,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& attenMask,
    const c10::optional<at::Tensor>& actualSeqQlen,
    const c10::optional<at::Tensor>& actualSeqKlen,
    const std::string& layoutKey, int64_t sparseMode,
    int64_t initBlocks, int64_t localBlocks)
{
    TORCH_CHECK(query.dim() == QUERY_DIM_NUM,
                "query must use TND layout [T,N,D]");
    TORCH_CHECK(query.scalar_type() == at::kHalf ||
                    query.scalar_type() == at::kBFloat16,
                "query dtype must be float16 or bfloat16");
    TORCH_CHECK(key.dim() == KEY_CACHE_DIM_NUM ||
                    key.dim() == KEY_CACHE_WITH_HEAD_DIM_NUM,
                "key must be [block_num,128,D] or [block_num,128,1,D]");
    TORCH_CHECK(key.scalar_type() == query.scalar_type(),
                "non-quantized key dtype must match query dtype");
    TORCH_CHECK(layoutKey == "BBND", "only BBND key layout is supported");
    TORCH_CHECK(blockTable.dim() == BLOCK_TABLE_DIM_NUM &&
                    blockTable.scalar_type() == at::kInt,
                "block_table must be a 2D int32 tensor");
    TORCH_CHECK(startLoc.dim() == SEQ_LEN_DIM_NUM &&
                    startLoc.scalar_type() == at::kInt,
                "start_loc must be a 1D int32 tensor");
    TORCH_CHECK(actualSeqQlen.has_value() &&
                    actualSeqQlen.value().defined() &&
                    actualSeqQlen.value().dim() == SEQ_LEN_DIM_NUM &&
                    actualSeqQlen.value().scalar_type() == at::kInt,
                "actual_seq_qlen must be a 1D int32 tensor");
    TORCH_CHECK(actualSeqKlen.has_value() &&
                    actualSeqKlen.value().defined() &&
                    actualSeqKlen.value().dim() == SEQ_LEN_DIM_NUM &&
                    actualSeqKlen.value().scalar_type() == at::kInt,
                "actual_seq_klen must be a 1D int32 tensor");
    TORCH_CHECK(actualSeqQlen.value().size(0) ==
                    actualSeqKlen.value().size(0) + 1,
                "actual_seq_qlen size must equal batch_size + 1");
    TORCH_CHECK(blockTable.size(0) == actualSeqKlen.value().size(0) &&
                    startLoc.size(0) == actualSeqKlen.value().size(0),
                "block_table/start_loc batch size mismatch");
    TORCH_CHECK(key.size(1) == BLOCK_SIZE,
                "MSA index score requires block size 128");
    TORCH_CHECK(sparseMode == 0 || sparseMode == 3,
                "sparse_mode must be 0 or 3");
    if (sparseMode == 3) {
        TORCH_CHECK(attenMask.has_value() && attenMask.value().defined(),
                    "sparse_mode=3 requires atten_mask");
        TORCH_CHECK(attenMask.value().sizes() == at::IntArrayRef({2048, 2048}) &&
                        attenMask.value().scalar_type() == at::kChar,
                    "atten_mask must be int8 with shape [2048,2048]");
    } else {
        TORCH_CHECK(!attenMask.has_value() || !attenMask.value().defined(),
                    "sparse_mode=0 does not accept atten_mask");
    }
    TORCH_CHECK(!scale.has_value() || !scale.value().defined(),
                "scale is only valid for an int8 key cache");
    TORCH_CHECK(initBlocks >= 0 && localBlocks >= 0,
                "init_blocks and local_blocks must be non-negative");
}

}  // namespace msa_index_score_detail

at::Tensor npu_msa_index_score(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& blockTable, const at::Tensor& startLoc,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& attenMask,
    const c10::optional<at::Tensor>& actualSeqQlen,
    const c10::optional<at::Tensor>& actualSeqKlen,
    c10::string_view layoutKey, int64_t sparseMode,
    int64_t initBlocks, int64_t localBlocks)
{
    std::string layoutKeyStr(layoutKey);
    msa_index_score_detail::CheckMsaIndexScoreParams(
        query, key, blockTable, startLoc, scale, attenMask,
        actualSeqQlen, actualSeqKlen, layoutKeyStr, sparseMode,
        initBlocks, localBlocks);

    at::Tensor normalizedKey = key.dim() ==
            msa_index_score_detail::KEY_CACHE_DIM_NUM
        ? key.unsqueeze(2)
        : key;
    const int64_t scoreStride =
        ((blockTable.size(1) + msa_index_score_detail::SCORE_ALIGNMENT - 1) /
         msa_index_score_detail::SCORE_ALIGNMENT) *
        msa_index_score_detail::SCORE_ALIGNMENT;
    at::Tensor score = at::empty(
        {query.size(1), query.size(0), scoreStride},
        query.options().dtype(at::kFloat));

    // The aclnn executor can retain attribute pointers until ACL graph capture
    // is finalized.  A pointer into the stack-local std::string becomes
    // dangling after this adapter returns and crashes in
    // NnopbaseExecutorArgsGetDfxInfo during FULL_DECODE_ONLY capture.  This
    // adapter only supports BBND, so give the executor process-lifetime
    // storage, matching the capture-safe behavior of built-in ACLNN ops.
    static char layoutKeyBBND[] = "BBND";
    EXEC_NPU_CMD(aclnnMsaIndexScore, query, normalizedKey, blockTable,
                 scale, attenMask, actualSeqQlen, actualSeqKlen, startLoc,
                 layoutKeyBBND, sparseMode, initBlocks, localBlocks, score);
    return score;
}

}  // namespace vllm_ascend

#endif  // MSA_INDEX_SCORE_TORCH_ADPT_H
