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

// Adapted from ops-transformer attention/lightning_indexer_v2/torch_extension.
// Preserve the upstream ACLNN argument order and output protocol.
namespace vllm_ascend::li_v2 {
constexpr int64_t LI_V2_METADATA_SIZE = 2048;
inline at::Tensor li_v2_optional_tensor(const c10::optional<at::Tensor>& value, at::Device device)
{
    return value.has_value() ? *value : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
}

at::Tensor LightningIndexerMetadata(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                    const c10::optional<at::Tensor>& cuSeqlensQ,
                                    const c10::optional<at::Tensor>& cuSeqlensK,
                                    const c10::optional<at::Tensor>& sequsedQ,
                                    const c10::optional<at::Tensor>& sequsedK,
                                    const c10::optional<at::Tensor>& cmpResidualK, int64_t batchSize,
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

    TORCH_CHECK(outputDevice.type() == c10::DeviceType::PrivateUse1, "LI metadata inputs must be on NPU");
    const c10::OptionalDeviceGuard device_guard(outputDevice);
    at::Tensor output = torch::empty({LI_V2_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = li_v2_optional_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = li_v2_optional_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = li_v2_optional_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = li_v2_optional_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = li_v2_optional_tensor(cmpResidualK, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char* layoutQPtr = const_cast<char*>(layoutQStr.c_str());
    char* layoutKPtr = const_cast<char*>(layoutKStr.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal, cmpResidualKVal,
              numHeadsQ, numHeadsK, headDim, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQPtr, layoutKPtr, maskMode,
              cmpRatio, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> ConstructLightningIndexerOutputTensor(const at::Tensor& query, const at::Tensor& key,
                                                                         int64_t topk, std::string queryLayoutStr,
                                                                         std::string keyLayoutStr, bool returnValue)
{
    TORCH_CHECK(queryLayoutStr == "TND" || queryLayoutStr == "BSND", "Unsupported query layout");
    TORCH_CHECK(keyLayoutStr == "TND" || keyLayoutStr == "BSND" || keyLayoutStr == "PA_BBND", "Unsupported key layout");
    TORCH_CHECK(query.dim() == (queryLayoutStr == "TND" ? 3 : 4), "Invalid query rank");
    TORCH_CHECK(key.dim() == (keyLayoutStr == "TND" ? 3 : 4), "Invalid key rank");
    at::SmallVector<int64_t, 4> outputSize;
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
    TORCH_CHECK(topk > 0, "topk should be greater than 0, but now is ", topk);
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(1) : key.size(2);
    if (queryLayoutStr == "BSND") {
        outputSize = {query.size(0), query.size(1), keyHeadNum, topk};
    } else {
        int nDimIndex = 0;
        nDimIndex = (keyLayoutStr == "TND") ? 1 : 2;
        outputSize = {query.size(0), key.size(nDimIndex), topk};
    }
    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        sparseValuesOut = at::empty(outputSize, query.options().dtype(at::kFloat));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kFloat));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor> LightningIndexer(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& w, int64_t topk,
    const c10::optional<at::Tensor>& cuSeqlensQ, const c10::optional<at::Tensor>& cuSeqlensK,
    const c10::optional<at::Tensor>& sequsedQ, const c10::optional<at::Tensor>& sequsedK,
    const c10::optional<at::Tensor>& cmpResidualK, const c10::optional<at::Tensor>& blockTable,
    const c10::optional<at::Tensor>& outputIdxOffset, const c10::optional<at::Tensor>& metadata, int64_t maxSeqlenQ,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t returnValue)
{
    const c10::OptionalDeviceGuard device_guard(q.device());
    TORCH_CHECK(returnValue == 0 || returnValue == 1, "return_value must be 0 or 1");
    TORCH_CHECK(q.numel() > 0, "Tensor q is empty.")
    TORCH_CHECK(k.numel() > 0, "Tensor k is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> lightningIndexerOutput =
        ConstructLightningIndexerOutputTensor(q, k, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(lightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(lightningIndexerOutput);
    // convert str
    char* queryLayoutPtr = const_cast<char*>(queryLayoutStr.c_str());
    char* keyLayoutPtr = const_cast<char*>(keyLayoutStr.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexerV2, q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable,
              outputIdxOffset, metadata, topk, maxSeqlenQ, queryLayoutPtr, keyLayoutPtr, maskMode, cmpRatio,
              returnValue, sparseIndicesOut, sparseValuesOut);
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}


}  // namespace vllm_ascend::li_v2
