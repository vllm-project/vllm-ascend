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
#ifndef SPARSE_ATTENTION_SCORE_TORCH_ADPT_H
#define SPARSE_ATTENTION_SCORE_TORCH_ADPT_H

namespace vllm_ascend {

namespace {

constexpr int64_t DIM_T = 0;
constexpr int64_t DIM_N = 1;
constexpr int64_t DIM_D = 2;
constexpr int64_t DIM_BLOCK_SIZE = 1;
constexpr int64_t DIM_KV_HEAD = 2;
constexpr int64_t DIM_KV_HEAD_SIZE = 3;
constexpr int64_t TND_DIM_NUM = 3;
constexpr int64_t BLOCK_KV_DIM_NUM = 4;
constexpr int64_t SELECT_IDX_DIM_NUM = 3;
constexpr int64_t SELECT_NUM_IDX_DIM_NUM = 2;
constexpr int64_t BLOCK_TABLE_DIM_NUM = 2;
constexpr int64_t DEQUANT_SCALE_DIM_NUM = 4;

void CheckFp8Tensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kFloat8_e4m3fn,
                name, " dtype must be float8_e4m3fn, got ", tensor.scalar_type());
}

void CheckDequantScaleTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be provided for float8_e4m3fn input.");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " dtype must be float32.");
    TORCH_CHECK(tensor.dim() == DEQUANT_SCALE_DIM_NUM, name, " must be a 4D tensor.");
}

void CheckTndTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.dim() == TND_DIM_NUM,
                name, " only supports TND layout [T,N,D], but got dim ", tensor.dim());
    TORCH_CHECK(tensor.numel() > 0, name, " should not be empty.");
}

void CheckBlockedKvTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.dim() == BLOCK_KV_DIM_NUM,
                name, " only supports blocked KV layout [blockNum,blockSize,KVHead,D], but got dim ",
                tensor.dim());
    TORCH_CHECK(tensor.numel() > 0, name, " should not be empty.");
}

void CheckParams(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                 const at::Tensor &selectIdx, const at::Tensor &blockTable,
                 const c10::optional<at::Tensor> &selectNumIdx,
                 const c10::optional<at::Tensor> &qDequantScale,
                 const c10::optional<at::Tensor> &kDequantScale,
                 const c10::optional<at::Tensor> &vDequantScale,
                 const c10::optional<at::Tensor> &actualSeqLengths,
                 const c10::optional<at::Tensor> &actualSeqLengthsKv,
                 int64_t numKeyValueHeads, int64_t blockSize, int64_t topK, int64_t innerPrecise)
{
    const bool isFp8 = query.scalar_type() == at::kFloat8_e4m3fn;
    if (isFp8) {
        CheckFp8Tensor(query, "query");
        CheckFp8Tensor(key, "key");
        CheckFp8Tensor(value, "value");
        TORCH_CHECK(qDequantScale.has_value(), "q_dequant_scale must be provided for float8_e4m3fn input.");
        TORCH_CHECK(kDequantScale.has_value(), "k_dequant_scale must be provided for float8_e4m3fn input.");
        TORCH_CHECK(vDequantScale.has_value(), "v_dequant_scale must be provided for float8_e4m3fn input.");
        CheckDequantScaleTensor(qDequantScale.value(), "q_dequant_scale");
        CheckDequantScaleTensor(kDequantScale.value(), "k_dequant_scale");
        CheckDequantScaleTensor(vDequantScale.value(), "v_dequant_scale");
        TORCH_CHECK(innerPrecise == 4,
                    "inner_precise must be 4 (LOW_HIGH_MIXED) for float8_e4m3fn, got ", innerPrecise);
    }
    CheckTndTensor(query, "query");
    CheckBlockedKvTensor(key, "key");
    CheckBlockedKvTensor(value, "value");

    TORCH_CHECK(selectIdx.dim() == SELECT_IDX_DIM_NUM,
                "select_idx must be [KVHead, maxQSeqlen, TopK], but got dim ", selectIdx.dim());
    TORCH_CHECK(selectIdx.scalar_type() == at::kInt, "select_idx dtype must be int32.");

    TORCH_CHECK(blockTable.dim() == BLOCK_TABLE_DIM_NUM,
                "block_table must be [batch, maxBlocksPerBatch], but got dim ", blockTable.dim());
    TORCH_CHECK(blockTable.scalar_type() == at::kInt, "block_table dtype must be int32.");

    TORCH_CHECK(numKeyValueHeads > 0, "num_key_value_heads must be positive.");
    TORCH_CHECK(key.size(DIM_KV_HEAD) == numKeyValueHeads && value.size(DIM_KV_HEAD) == numKeyValueHeads,
                "num_key_value_heads must match key/value KVHead dim.");
    TORCH_CHECK(selectIdx.size(DIM_T) == numKeyValueHeads,
                "select_idx dim0 must equal num_key_value_heads.");
    TORCH_CHECK(key.size(DIM_BLOCK_SIZE) == blockSize && value.size(DIM_BLOCK_SIZE) == blockSize,
                "key/value blockSize dim must match block_size.");
    TORCH_CHECK(query.size(DIM_N) % numKeyValueHeads == 0,
                "query heads must be divisible by num_key_value_heads.");
    TORCH_CHECK(query.size(DIM_D) == key.size(DIM_KV_HEAD_SIZE) &&
                    query.size(DIM_D) == value.size(DIM_KV_HEAD_SIZE),
                "query/key/value D dim must be equal.");
    TORCH_CHECK(blockSize > 0, "block_size must be positive.");
    TORCH_CHECK(topK > 0, "top_k must be positive.");

    if (selectNumIdx.has_value() && selectNumIdx.value().defined()) {
        const at::Tensor &snIdx = selectNumIdx.value();
        TORCH_CHECK(snIdx.dim() == SELECT_NUM_IDX_DIM_NUM,
                    "select_num_idx must be [KVHead, maxQSeqlen].");
        TORCH_CHECK(snIdx.scalar_type() == at::kInt, "select_num_idx dtype must be int32.");
    }

    TORCH_CHECK(actualSeqLengths.has_value() && actualSeqLengths.value().defined(),
                "actual_seq_lengths must be provided.");
    TORCH_CHECK(actualSeqLengthsKv.has_value() && actualSeqLengthsKv.value().defined(),
                "actual_seq_lengths_kv must be provided.");
}

}  // namespace

at::Tensor npu_sparse_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &selectIdx, const at::Tensor &blockTable,
    const c10::optional<at::Tensor> &selectNumIdx,
    const c10::optional<at::Tensor> &qDequantScale,
    const c10::optional<at::Tensor> &kDequantScale,
    const c10::optional<at::Tensor> &vDequantScale,
    const c10::optional<at::Tensor> &actualSeqLengths,
    const c10::optional<at::Tensor> &actualSeqLengthsKv,
    c10::string_view qInputLayout, c10::string_view kvInputLayout,
    int64_t numKeyValueHeads, double scaleValue, int64_t blockSize, int64_t topK,
    int64_t innerPrecise)
{
    TORCH_CHECK(std::string(qInputLayout) == "TND",
                "npu_sparse_attention_score only supports query TND layout");
    CheckParams(query, key, value, selectIdx, blockTable, selectNumIdx,
                qDequantScale, kDequantScale, vDequantScale,
                actualSeqLengths, actualSeqLengthsKv, numKeyValueHeads, blockSize, topK, innerPrecise);

    at::ScalarType outDtype = (query.scalar_type() == at::kFloat8_e4m3fn)
                                  ? at::kHalf
                                  : query.scalar_type();
    at::Tensor attentionOut = at::empty(query.sizes(), query.options().dtype(outDtype));
    at::Tensor softmaxLse;

    EXEC_NPU_CMD(aclnnSparseAttentionScore, query, key, value, selectIdx, blockTable,
                 selectNumIdx, actualSeqLengths, actualSeqLengthsKv,
                 qDequantScale, kDequantScale, vDequantScale,
                 numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise,
                 attentionOut, softmaxLse);

    return attentionOut;
}

}  // namespace vllm_ascend

#endif  // SPARSE_ATTENTION_SCORE_TORCH_ADPT_H
