/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

inline TensorWrapper MakeWrapper(const at::Tensor &tensor)
{
    return {tensor, ConvertToAclDataType(tensor.scalar_type())};
}

inline bool IsMxQuantMode(int64_t quantMode) { return quantMode == 3 || quantMode == 5; }

inline bool IsE8M0Tensor(const at::Tensor &tensor) { return tensor.scalar_type() == at::kFloat8_e8m0fnu; }

inline bool IsFp4CompatibleTensor(const at::Tensor &tensor)
{
    return tensor.scalar_type() == at::kFloat4_e2m1fn_x2 || tensor.scalar_type() == at::kByte;
}

constexpr int64_t E8M0_SCALE_PACK_NUM = 2;

inline void FixQLIV2AclDtypes(int64_t quantMode, TensorWrapper &queryWrapper, TensorWrapper &keyWrapper,
                              TensorWrapper &queryScaleWrapper, TensorWrapper &keyScaleWrapper)
{
    if (quantMode == 4) {
        TORCH_CHECK(queryWrapper.tensor_.scalar_type() == at::kByte, "When quant_mode is 4, query must be hifp8 type");
        TORCH_CHECK(keyWrapper.tensor_.scalar_type() == at::kByte, "When quant_mode is 4, key must be hifp8 type");
        queryWrapper.dtype = ACL_HIFLOAT8;
        keyWrapper.dtype = ACL_HIFLOAT8;
        return;
    }

    if (quantMode == 5) {
        TORCH_CHECK(IsFp4CompatibleTensor(queryWrapper.tensor_),
                    "When quant_mode is 5, query must be torch.float4_e2m1fn_x2 or packed torch.uint8");
        TORCH_CHECK(IsFp4CompatibleTensor(keyWrapper.tensor_),
                    "When quant_mode is 5, key must be torch.float4_e2m1fn_x2 or packed torch.uint8");
        queryWrapper.dtype = ACL_FLOAT4_E2M1;
        keyWrapper.dtype = ACL_FLOAT4_E2M1;
    }

    if (IsMxQuantMode(quantMode)) {
        TORCH_CHECK(IsE8M0Tensor(queryScaleWrapper.tensor_),
                    "When quant_mode is 3 or 5, query_dequant_scale must be torch.float8_e8m0fnu");
        TORCH_CHECK(IsE8M0Tensor(keyScaleWrapper.tensor_),
                    "When quant_mode is 3 or 5, key_dequant_scale must be torch.float8_e8m0fnu");
        // Cube loads E8M0 scales in two-byte groups through a bfloat16_t view.
        TORCH_CHECK(queryScaleWrapper.tensor_.storage_offset() % E8M0_SCALE_PACK_NUM == 0,
                    "When quant_mode is 3 or 5, query_dequant_scale storage offset must satisfy 2-element E8M0 packing "
                    "alignment, but got ",
                    queryScaleWrapper.tensor_.storage_offset());
        TORCH_CHECK(keyScaleWrapper.tensor_.storage_offset() % E8M0_SCALE_PACK_NUM == 0,
                    "When quant_mode is 3 or 5, key_dequant_scale storage offset must satisfy 2-element E8M0 packing "
                    "alignment, but got ",
                    keyScaleWrapper.tensor_.storage_offset());
        queryScaleWrapper.dtype = ACL_FLOAT8_E8M0;
        keyScaleWrapper.dtype = ACL_FLOAT8_E8M0;
    }
}

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

constexpr int64_t QLI_V2_METADATA_SIZE = 1024;

at::Tensor QuantLightningIndexerMetadata(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                         int64_t quantMode, const c10::optional<at::Tensor> &cuSeqlensQ,
                                         const c10::optional<at::Tensor> &cuSeqlensK,
                                         const c10::optional<at::Tensor> &sequsedQ,
                                         const c10::optional<at::Tensor> &sequsedK,
                                         const c10::optional<at::Tensor> &cmpResidualK, int64_t batchSize,
                                         int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ,
                                         c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio)
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
    }

    at::Tensor output = torch::empty({QLI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = get_valid_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = get_valid_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = get_valid_tensor(cmpResidualK, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKPtr = const_cast<char *>(layoutKStr.c_str());

    ACLNN_CMD(aclnnQuantLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal,
              cmpResidualKVal, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK,
              layoutQPtr, layoutKPtr, maskMode, cmpRatio, output);
    return output;
}

// 工具函数，推导输出shape
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
        sparseValuesOut = at::empty(outputSize, query.options().dtype(at::kBFloat16));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kBFloat16));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor> QuantLightningIndexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights, const at::Tensor &queryDequantScale,
    const at::Tensor &keyDequantScale, int64_t topk, int64_t quantMode, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensK, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedK, const c10::optional<at::Tensor> &cmpResidualK,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &outputIdxOffset,
    const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ, c10::string_view layoutQ, c10::string_view layoutK,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> quantLightningIndexerOutput =
        ConstructQuantLightningIndexerOutputTensor(query, key, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(quantLightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(quantLightningIndexerOutput);
    // convert str
    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    auto queryWrapper = MakeWrapper(query);
    auto keyWrapper = MakeWrapper(key);
    auto queryScaleWrapper = MakeWrapper(queryDequantScale);
    auto keyScaleWrapper = MakeWrapper(keyDequantScale);
    FixQLIV2AclDtypes(quantMode, queryWrapper, keyWrapper, queryScaleWrapper, keyScaleWrapper);

    int64_t keyStride0Disabled = 0;   // A11: 旧入口不启用 stride 显式属性 (保持现网行为)
    int64_t keyScaleStride0Disabled = 0;
    ACLNN_CMD(aclnnQuantLightningIndexerV2, queryWrapper, keyWrapper, weights, queryScaleWrapper, keyScaleWrapper,
              cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset, metadata, topk,
              quantMode, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode, cmpRatio, returnValue,
              keyStride0Disabled, keyScaleStride0Disabled, sparseIndicesOut,
              sparseValuesOut);

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

// 两级TopK candidate 接口 (O1 方案b: 新增入口, 旧接口不变)
// candidate_mode: 1=source 输出 candidate_topk_index; 2=consumer 输入候选块; 3=关闭
std::tuple<at::Tensor, at::Tensor, at::Tensor> QuantLightningIndexerCandidate(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights, const at::Tensor &queryDequantScale,
    const at::Tensor &keyDequantScale, int64_t topk, int64_t quantMode,
    const c10::optional<at::Tensor> &candidateTopkIndexIn, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensK, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedK, const c10::optional<at::Tensor> &cmpResidualK,
    const c10::optional<at::Tensor> &blockTable, const c10::optional<at::Tensor> &outputIdxOffset,
    const c10::optional<at::Tensor> &metadata, int64_t maxSeqlenQ, c10::string_view layoutQ,
    c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t candidateMode,
    int64_t candidateTopkBlocks, int64_t candidateBlockSize)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    std::tuple<at::Tensor, at::Tensor> quantLightningIndexerOutput =
        ConstructQuantLightningIndexerOutputTensor(query, key, topk, queryLayoutStr, keyLayoutStr, 0);
    at::Tensor sparseIndicesOut = std::get<0>(quantLightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(quantLightningIndexerOutput);

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
    int64_t returnValue = 0;

    auto queryWrapper = MakeWrapper(query);
    auto keyWrapper = MakeWrapper(key);
    auto queryScaleWrapper = MakeWrapper(queryDequantScale);
    auto keyScaleWrapper = MakeWrapper(keyDequantScale);
    FixQLIV2AclDtypes(quantMode, queryWrapper, keyWrapper, queryScaleWrapper, keyScaleWrapper);

    // A11: key 0 轴非连续 — aclnn 动态调用下 tiling 拿不到 tensor stride (仅 TensorV2/图模式可见),
    // 从 key/k_scale 的 torch stride(0) 自动显式传入 (紧凑存储时等于紧凑值, 走 kernel 兜底语义)
    int64_t keyStride0 = key.stride(0);
    int64_t keyScaleStride0 = keyDequantScale.stride(0);

    ACLNN_CMD(aclnnQuantLightningIndexerV2, queryWrapper, keyWrapper, weights, queryScaleWrapper, keyScaleWrapper,
              cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset,
              metadata, candidateTopkIndexIn, topk, quantMode, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode,
              cmpRatio, returnValue, candidateMode, candidateTopkBlocks, candidateBlockSize, keyStride0,
              keyScaleStride0, sparseIndicesOut, sparseValuesOut, candidateTopkIndexOut);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut, candidateTopkIndexOut);
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_lightning_indexer_metadata", &QuantLightningIndexerMetadata, "quant_lightning_indexer_metadata");
    m.def("quant_lightning_indexer", &QuantLightningIndexer, "quant_lightning_indexer");
    m.def("quant_lightning_indexer_candidate", &QuantLightningIndexerCandidate,
          "quant_lightning_indexer_candidate");
}
} // namespace op_api
