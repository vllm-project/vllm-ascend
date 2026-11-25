/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "acl/acl_base_rt.h"

#include <limits>

namespace vllm_ascend::qli_v2 {
constexpr int SIZE = 8;
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
inline at::Tensor valid_tensor(const c10::optional<at::Tensor>& value, const at::Device& device) {
    return value.has_value() ? *value : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
}
constexpr int64_t QLI_V2_METADATA_SIZE = 1024;
constexpr int64_t QLI_LOCAL_CANDIDATES_MODE = 2;

// This entry keeps the legacy public QLI ABI intact. The private native mode
// interprets output_idx_offset as a per-query local visible PREFIX length.
// Return native comparator keys: BF16 on A5, FP32 on A3. Candidate packing
// performs the lossless FP32 transport conversion without a separate cast.
std::tuple<at::Tensor, at::Tensor> QuantLightningIndexerLocalCandidates(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &queryDequantScale, const at::Tensor &keyDequantScale,
    const at::Tensor &cuSeqlensQ, const at::Tensor &sequsedK,
    const at::Tensor &blockTable, const at::Tensor &localVisibleLengths,
    const at::Tensor &metadata, int64_t topk)
{
    TORCH_CHECK(query.dim() == 3 && query.size(2) == 128, "Local QLI requires TND head_dim=128");
    TORCH_CHECK(key.dim() == 4 && key.size(2) == 1 && key.size(3) == 128,
                "Local QLI requires PA_BBND key [blocks, block_size, 1, 128]");
    const bool isInt8 = query.scalar_type() == at::kChar;
    TORCH_CHECK((isInt8 || query.scalar_type() == at::kFloat8_e4m3fn) &&
                    key.scalar_type() == query.scalar_type(),
                "Local QLI requires matching INT8 or FP8 E4M3 query/key");
    const at::ScalarType scaleType = isInt8 ? at::kHalf : at::kFloat;
    TORCH_CHECK(weights.scalar_type() == scaleType && queryDequantScale.scalar_type() == scaleType &&
                    keyDequantScale.scalar_type() == scaleType,
                isInt8 ? "INT8 local QLI requires FP16 weights and dequant scales" :
                         "FP8 local QLI requires FP32 weights and dequant scales");
    TORCH_CHECK(localVisibleLengths.scalar_type() == at::kInt && localVisibleLengths.dim() == 2 &&
                    localVisibleLengths.size(0) == query.size(0) && localVisibleLengths.size(1) == 1,
                "local_visible_lengths must be INT32 [T,1]");
    TORCH_CHECK(localVisibleLengths.is_contiguous() && query.is_contiguous() && weights.is_contiguous(),
                "Local QLI query, weights and visible lengths must be contiguous");
    TORCH_CHECK(topk > 0 && topk <= 8192, "Local QLI topk must be in [1,8192]");
    TORCH_CHECK(!isInt8 || topk <= 2048, "INT8 local QLI topk must be in [1,2048]");
    // TND graph buffers may contain storage rows beyond cu_seqlens_q[-1].
    // Define those unused fixed-K rows without touching any KV history.
    at::Tensor indices = at::full({query.size(0), 1, topk}, -1, query.options().dtype(at::kInt));
    at::Tensor values = at::full({query.size(0), 1, topk}, -std::numeric_limits<float>::infinity(),
                               query.options().dtype(isInt8 ? at::kFloat : at::kBFloat16));
    at::Tensor unused = at::empty({0}, query.options().dtype(at::kInt));
    if (query.device().is_meta()) return {indices, values};
    char layoutQ[] = "TND";
    char layoutK[] = "PA_BBND";
    constexpr int64_t dynamicMaxSeqLen = -1;
    const int64_t quantMode = isInt8 ? 2 : 1;
    constexpr int64_t maskMode = 0;
    constexpr int64_t cmpRatio = 1;
    constexpr int64_t candidateMode = 3;
    constexpr int64_t candidateTopkBlocks = 2048;
    constexpr int64_t candidateBlockSize = 8;
    int64_t keyStride0 = key.stride(0);
    int64_t keyScaleStride0 = keyDequantScale.stride(0);
    c10::optional<at::Tensor> none = c10::nullopt;
    bool enableDcp = false;
    int64_t cpWorld = 1;
    int64_t cpRank = 0;
    int64_t cpInterleave = 1;
    EXEC_NPU_CMD(aclnnVllmAscendQuantLightningIndexerV2, query, key, weights, queryDequantScale,
              keyDequantScale, cuSeqlensQ, none, none, sequsedK, none, blockTable, localVisibleLengths,
              metadata, none, none, topk, quantMode, dynamicMaxSeqLen, layoutQ, layoutK, maskMode, cmpRatio,
              QLI_LOCAL_CANDIDATES_MODE, candidateMode, candidateTopkBlocks, candidateBlockSize,
              keyStride0, keyScaleStride0, enableDcp, cpWorld, cpRank, cpInterleave, indices, values, unused);
    return {indices, values};
}

at::Tensor QuantLightningIndexerMetadata(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                         int64_t quantMode, const c10::optional<at::Tensor> &cuSeqlensQ,
                                         const c10::optional<at::Tensor> &cuSeqlensK,
                                         const c10::optional<at::Tensor> &sequsedQ,
                                         const c10::optional<at::Tensor> &sequsedK,
                                         const c10::optional<at::Tensor> &cmpResidualK,
                                         const c10::optional<at::Tensor> &causalSeqlensKvGlobal, int64_t batchSize,
                                         int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ,
                                         c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio,
                                         int64_t enableDcp = 0, int64_t cpWorld = 1, int64_t cpRank = 0,
                                         int64_t cpInterleaveSize = 1)
{
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensK.has_value()) {
        outputDevice = cuSeqlensK.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (cmpResidualK.has_value()) {
        outputDevice = cmpResidualK.value().device();
    } else if (causalSeqlensKvGlobal.has_value()) {
        outputDevice = causalSeqlensKvGlobal.value().device();
    }

    at::Tensor output = torch::empty({QLI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = valid_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = valid_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = valid_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = valid_tensor(cmpResidualK, outputDevice);
    auto causalSeqlensKvGlobalVal = valid_tensor(causalSeqlensKvGlobal, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKPtr = const_cast<char *>(layoutKStr.c_str());

    if (output.device().is_meta()) return output;
    EXEC_NPU_CMD(aclnnVllmAscendQuantLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal,
              cmpResidualKVal, causalSeqlensKvGlobalVal, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize,
              maxSeqlenQ, maxSeqlenK, layoutQPtr, layoutKPtr, maskMode, cmpRatio, enableDcp, cpWorld, cpRank,
              cpInterleaveSize, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> ConstructQuantLightningIndexerOutputTensor(
    const at::Tensor &query, const at::Tensor &key, int64_t sparseCount, std::string queryLayoutStr,
    std::string keyLayoutStr, int64_t returnValue)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparseCount > 0, "sparse count should be greater than 0, but now is ", sparseCount);
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);
    if (queryLayoutStr == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparseCount};
    } else {
        int nDimIndex = 0;
        nDimIndex = (keyLayoutStr == "TND") ? DIM_1 : DIM_2;
        outputSize = {query.size(DIM_0), key.size(nDimIndex), sparseCount};
    }
    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        // arch22 (910B/910_93) 直接输出 FP32 score; 950/arch35 保持 BF16。
        const char *socName = aclrtGetSocName();
        const bool fp32Score =
            (socName == nullptr) || (std::string(socName).find("Ascend950") == std::string::npos);
        sparseValuesOut = at::empty(outputSize, query.options().dtype(fp32Score ? at::kFloat : at::kBFloat16));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kBFloat16));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> QuantLightningIndexerCandidate(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights, const at::Tensor &queryDequantScale,
    const at::Tensor &keyDequantScale, int64_t topk, int64_t quantMode,
    const c10::optional<at::Tensor> &candidateTopkIndexIn, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensK, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedK, const c10::optional<at::Tensor> &cmpResidualK,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &outputIdxOffset,
    const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ, c10::string_view layoutQ,
    c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t candidateMode,
    int64_t candidateTopkBlocks, int64_t candidateBlockSize, bool enableDcp, int64_t cpWorld, int64_t cpRank,
    int64_t cpKvCacheInterleaveSize, const c10::optional<at::Tensor> &causalSeqlensKvGlobal, int64_t returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    std::tuple<at::Tensor, at::Tensor> quantLightningIndexerOutput =
        ConstructQuantLightningIndexerOutputTensor(query, key, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(quantLightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(quantLightningIndexerOutput);
    if (enableDcp) {
        // The kernel visits only rows covered by cu_seqlens_q. Graph padding
        // must retain the local-candidate sentinel contract before exchange.
        sparseIndicesOut.fill_(-1);
        sparseValuesOut.fill_(-std::numeric_limits<float>::infinity());
    }

    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);
    at::Tensor candidateTopkIndexOut;
    if (candidateMode == 1) {
        at::SmallVector<int64_t, SIZE> candSize;
        if (queryLayoutStr == "BSND") {
            candSize = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, candidateTopkBlocks};
        } else {
            candSize = {query.size(DIM_0), keyHeadNum, candidateTopkBlocks};
        }
        candidateTopkIndexOut = at::empty(candSize, query.options().dtype(at::kInt));
    } else {
        candidateTopkIndexOut = at::empty({0}, query.options().dtype(at::kInt));
    }

    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    TORCH_CHECK(candidateMode >= 1 && candidateMode <= 3, "Invalid candidate_mode");
    TORCH_CHECK(candidateMode != 2 || candidateTopkIndexIn.has_value(), "Consumer requires candidate blocks");
    if (query.device().is_meta()) return {sparseIndicesOut, sparseValuesOut, candidateTopkIndexOut};

    // A11: key 0 轴非连续 — aclnn 动态调用下 tiling 拿不到 tensor stride (仅 TensorV2/图模式可见),
    // 从 key/k_scale 的 torch stride(0) 自动显式传入 (紧凑存储时等于紧凑值, 走 kernel 兜底语义)
    int64_t keyStride0 = key.stride(0);
    int64_t keyScaleStride0 = keyDequantScale.stride(0);

    EXEC_NPU_CMD(aclnnVllmAscendQuantLightningIndexerV2, query, key, weights, queryDequantScale, keyDequantScale,
              cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset,
              metadata, candidateTopkIndexIn, causalSeqlensKvGlobal, topk, quantMode, maxSeqlenQ, queryLayoutPtr,
              keyLayoutPtr, maskMode, cmpRatio, returnValue, candidateMode, candidateTopkBlocks, candidateBlockSize,
              keyStride0, keyScaleStride0, enableDcp, cpWorld, cpRank, cpKvCacheInterleaveSize, sparseIndicesOut,
              sparseValuesOut, candidateTopkIndexOut);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut, candidateTopkIndexOut);
}

}  // namespace vllm_ascend::qli_v2
