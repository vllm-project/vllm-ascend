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

#include <array>

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
constexpr int64_t GBSA_METADATA_SIZE = 1024;

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

void CheckSeqTensor(const c10::optional<at::Tensor> &tensor, const char *name,
                    at::ScalarType dtype, int64_t expectedSize)
{
    TORCH_CHECK(tensor.has_value() && tensor.value().defined(), name, " must be provided.");
    TORCH_CHECK(tensor.value().dim() == 1, name, " must be a 1D tensor.");
    TORCH_CHECK(tensor.value().scalar_type() == dtype, name, " has an invalid dtype.");
    TORCH_CHECK(tensor.value().size(0) == expectedSize, name, " has an invalid length.");
}

void CheckParams(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                 const at::Tensor &selectIdx, const at::Tensor &blockTable,
                 const c10::optional<at::Tensor> &selectNumIdx,
                 const c10::optional<at::Tensor> &qDequantScale,
                 const c10::optional<at::Tensor> &kDequantScale,
                 const c10::optional<at::Tensor> &vDequantScale,
                 const c10::optional<at::Tensor> &actualSeqLengths,
                 const c10::optional<at::Tensor> &actualSeqLengthsKv,
                 const c10::optional<at::Tensor> &cuSeqLengthsQ,
                 const c10::optional<at::Tensor> &cuSeqLengthsKv,
                 int64_t numKeyValueHeads, int64_t blockSize, int64_t topK,
                 int64_t innerPrecise, int64_t softmaxPrecision)
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
                "select_idx must be [KVHead, totalQBlocks, TopK], but got dim ", selectIdx.dim());
    TORCH_CHECK(selectIdx.scalar_type() == at::kInt, "select_idx dtype must be int32.");
    TORCH_CHECK(selectIdx.size(1) == query.size(DIM_T),
                "select_idx totalQBlocks must equal the number of query tokens when blockShapeX is 1.");

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
    TORCH_CHECK(topK > 0 && topK <= 16, "top_k must be in [1, 16].");
    TORCH_CHECK(selectIdx.size(2) == topK, "select_idx last dim must match top_k.");
    TORCH_CHECK(softmaxPrecision == 0 || softmaxPrecision == 1,
                "softmax_precision must be 0 or 1.");

    TORCH_CHECK(selectNumIdx.has_value() && selectNumIdx.value().defined(),
                "select_num_idx must be provided for GenericBlockSparseAttention.");
    const at::Tensor &snIdx = selectNumIdx.value();
    TORCH_CHECK(snIdx.dim() == SELECT_NUM_IDX_DIM_NUM,
                "select_num_idx must be [KVHead, totalQBlocks].");
    TORCH_CHECK(snIdx.scalar_type() == at::kInt, "select_num_idx dtype must be int32.");
    TORCH_CHECK(snIdx.size(0) == selectIdx.size(0) && snIdx.size(1) == selectIdx.size(1),
                "select_num_idx shape must match select_idx without its last dim.");

    const int64_t batch = blockTable.size(0);
    CheckSeqTensor(actualSeqLengths, "actual_seq_lengths", at::kInt, batch);
    CheckSeqTensor(actualSeqLengthsKv, "actual_seq_lengths_kv", at::kInt, batch);
    CheckSeqTensor(cuSeqLengthsQ, "cu_seq_lengths_q", at::kLong, batch + 1);
    CheckSeqTensor(cuSeqLengthsKv, "cu_seq_lengths_kv", at::kLong, batch + 1);
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
    const c10::optional<at::Tensor> &cuSeqLengthsQ,
    const c10::optional<at::Tensor> &cuSeqLengthsKv,
    c10::string_view qInputLayout, c10::string_view kvInputLayout,
    int64_t numKeyValueHeads, double scaleValue, int64_t blockSize, int64_t topK,
    int64_t innerPrecise, int64_t softmaxPrecision)
{
    TORCH_CHECK(std::string(qInputLayout) == "TND",
                "npu_sparse_attention_score only supports query TND layout");
    TORCH_CHECK(std::string(kvInputLayout) == "PAGED_BBND",
                "npu_sparse_attention_score only supports PAGED_BBND KV layout");
    CheckParams(query, key, value, selectIdx, blockTable, selectNumIdx,
                qDequantScale, kDequantScale, vDequantScale,
                actualSeqLengths, actualSeqLengthsKv, cuSeqLengthsQ, cuSeqLengthsKv,
                numKeyValueHeads, blockSize, topK, innerPrecise, softmaxPrecision);

    const at::Tensor &sparseBlockCount = selectNumIdx.value();
    const at::Tensor &seqUsedQ = actualSeqLengths.value();
    const at::Tensor &seqUsedKv = actualSeqLengthsKv.value();
    const at::Tensor &cuQ = cuSeqLengthsQ.value();
    const at::Tensor &cuKv = cuSeqLengthsKv.value();
    at::Tensor metadata = at::empty({GBSA_METADATA_SIZE}, selectIdx.options().dtype(at::kInt));

    std::array<int64_t, 2> blockShapeStorage{1, blockSize};
    at::IntArrayRef blockShape(blockShapeStorage);
    std::string layoutQStr("TND");
    std::string layoutKvStr("PAGED_BBND");
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());
    const int64_t maxQSeqLen = selectIdx.size(1);
    const int64_t maxKvSeqLen = blockTable.size(1) * blockSize;
    const int64_t numQHeads = query.size(DIM_N);
    const int64_t headDim = query.size(DIM_D);
    const int64_t isPackedGqa = 1;
    const int64_t maskType = 1;
    const int64_t quantType = query.scalar_type() == at::kFloat8_e4m3fn ? 5 : 0;
    const int64_t winLeft = -1;
    const int64_t winRight = -1;

    EXEC_NPU_CMD(aclnnGenericBlockSparseAttentionMetadata,
                 selectIdx, sparseBlockCount, cuQ, cuKv, seqUsedQ, seqUsedKv,
                 maxQSeqLen, maxKvSeqLen, numQHeads, numKeyValueHeads, headDim,
                 blockShape, isPackedGqa, layoutQPtr, layoutKvPtr, maskType,
                 quantType, softmaxPrecision, winLeft, winRight, metadata);

    at::ScalarType outDtype = query.scalar_type() == at::kFloat8_e4m3fn
                                  ? at::kHalf
                                  : query.scalar_type();
    at::Tensor attentionOut = at::empty(query.sizes(), query.options().dtype(outDtype));
    at::Tensor softmaxLse = at::empty({0}, query.options().dtype(at::kFloat));
    const c10::optional<at::Tensor> noTensor = c10::nullopt;
    const int64_t returnSoftmaxLse = 0;
    const double dstTypeMax = 0.0;

    EXEC_NPU_CMD(aclnnGenericBlockSparseAttention,
                 query, key, value, selectIdx, sparseBlockCount, metadata,
                 noTensor, qDequantScale, kDequantScale, vDequantScale, noTensor,
                 cuQ, cuKv, seqUsedQ, seqUsedKv, blockTable, blockShape,
                 isPackedGqa, layoutQPtr, layoutKvPtr, scaleValue, maskType,
                 quantType, dstTypeMax, softmaxPrecision, winLeft, winRight,
                 returnSoftmaxLse, attentionOut, softmaxLse);

    return attentionOut;
}

}  // namespace vllm_ascend

#endif  // SPARSE_ATTENTION_SCORE_TORCH_ADPT_H
