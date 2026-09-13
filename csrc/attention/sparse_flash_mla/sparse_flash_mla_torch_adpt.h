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
 * \file sparse_flash_mla.cpp
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_TORCH_ADPT_H
#define SPARSE_FLASH_MLA_TORCH_ADPT_H

namespace vllm_ascend {
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

constexpr int64_t SMLA_METADATA_SIZE = 1024;

inline at::Tensor GetValidSparseFlashMlaTensor(
    const c10::optional<at::Tensor> &tensor, const at::Device &device)
{
    return tensor.has_value()
               ? tensor.value()
               : at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
}

at::Tensor npu_sparse_flash_mla_metadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &cuSeqlensOriKv, const c10::optional<at::Tensor> &cuSeqlensCmpKv,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedOriKv,
    const c10::optional<at::Tensor> &sequsedCmpKv, const c10::optional<at::Tensor> &cmpResidualKv,
    const c10::optional<at::Tensor> &oriTopkLength, const c10::optional<at::Tensor> &cmpTopkLength, int64_t batchSize,
    int64_t maxSeqlenQ, int64_t maxSeqlenOriKv, int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk,
    int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
    c10::string_view layoutQ, c10::string_view layoutKv, bool hasOriKv, bool hasCmpKv)
{
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensOriKv.has_value()) {
        outputDevice = cuSeqlensOriKv.value().device();
    } else if (cuSeqlensCmpKv.has_value()) {
        outputDevice = cuSeqlensCmpKv.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedOriKv.has_value()) {
        outputDevice = sequsedOriKv.value().device();
    } else if (sequsedCmpKv.has_value()) {
        outputDevice = sequsedCmpKv.value().device();
    } else if (cmpResidualKv.has_value()) {
        outputDevice = cmpResidualKv.value().device();
    } else if (oriTopkLength.has_value()) {
        outputDevice = oriTopkLength.value().device();
    } else if (cmpTopkLength.has_value()) {
        outputDevice = cmpTopkLength.value().device();
    }

    at::Tensor output = torch::empty({SMLA_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = GetValidSparseFlashMlaTensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensOriKvVal = GetValidSparseFlashMlaTensor(cuSeqlensOriKv, outputDevice);
    auto cuSeqlensCmpKvVal = GetValidSparseFlashMlaTensor(cuSeqlensCmpKv, outputDevice);
    auto sequsedQVal = GetValidSparseFlashMlaTensor(sequsedQ, outputDevice);
    auto sequsedOriKvVal = GetValidSparseFlashMlaTensor(sequsedOriKv, outputDevice);
    auto sequsedCmpKvVal = GetValidSparseFlashMlaTensor(sequsedCmpKv, outputDevice);
    auto cmpResidualKvVal = GetValidSparseFlashMlaTensor(cmpResidualKv, outputDevice);
    auto oriTopkLengthVal = GetValidSparseFlashMlaTensor(oriTopkLength, outputDevice);
    auto cmpTopkLengthVal = GetValidSparseFlashMlaTensor(cmpTopkLength, outputDevice);

    // convert str
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    EXEC_NPU_CMD(aclnnSparseFlashMlaMetadata, cuSeqlensQVal, cuSeqlensOriKvVal, cuSeqlensCmpKvVal, sequsedQVal,
              sequsedOriKvVal, sequsedCmpKvVal, cmpResidualKvVal, oriTopkLengthVal, cmpTopkLengthVal, numHeadsQ,
              numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio,
              oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr, hasOriKv, hasCmpKv, output);
    return output;
}

void CheckQueryShape(const at::Tensor &q, const std::string &layoutQStr)
{
    TORCH_CHECK(layoutQStr == "BSND" || layoutQStr == "TND", "The layout of query only support BSND and TND, but got ",
                layoutQStr);
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.");
    for (int64_t i = 0; i < q.dim(); i++) {
        TORCH_CHECK(q.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", q.size(i));
    }
    if (layoutQStr == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4, "When the layout of query is BSND, the query dimension must be 4, but got ",
                    q.dim());
    } else {
        TORCH_CHECK(q.dim() == DIM_3, "When the layout of query is TND, the query dimension must be 3, but got ",
                    q.dim());
    }
}

int64_t GetKvHeadNum(const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
                     const std::string &layoutKvStr)
{
    TORCH_CHECK(oriKv.has_value() || cmpKv.has_value(),
                "ori_kv or cmp_kv is required when return_softmax_lse is true.");
    const at::Tensor &kv = oriKv.has_value() ? oriKv.value() : cmpKv.value();
    if (layoutKvStr == "TND") {
        return kv.size(DIM_1);
    }
    return kv.size(DIM_2);
}

std::tuple<at::Tensor, at::Tensor> MakeSparseFlashMlaOutputs(const at::Tensor &q,
                                                             const c10::optional<at::Tensor> &oriKv,
                                                             const c10::optional<at::Tensor> &cmpKv,
                                                             const std::string &layoutQStr,
                                                             const std::string &layoutKvStr, bool returnSoftmaxLse)
{
    CheckQueryShape(q, layoutQStr);
    at::Tensor attenOut = at::empty_like(q);
    at::Tensor softmaxLse;
    if (!returnSoftmaxLse) {
        softmaxLse = at::empty({0}, q.options().dtype(torch::kFloat32));
        return {attenOut, softmaxLse};
    }

    int64_t kvHeadNum = GetKvHeadNum(oriKv, cmpKv, layoutKvStr);
    TORCH_CHECK(kvHeadNum > 0, "head num of ori_kv or cmp_kv must be greater than 0, but got ", kvHeadNum);
    if (layoutQStr == "BSND") {
        softmaxLse = at::empty({q.size(DIM_0), kvHeadNum, q.size(DIM_1), q.size(DIM_2) / kvHeadNum},
                               q.options().dtype(torch::kFloat32));
    } else {
        softmaxLse =
            at::empty({kvHeadNum, q.size(DIM_0), q.size(DIM_1) / kvHeadNum}, q.options().dtype(torch::kFloat32));
    }
    return {attenOut, softmaxLse};
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_mla(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, c10::string_view layoutQ, c10::string_view layoutKv,
    int64_t topkValueMode, bool returnSoftmaxLse)
{
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    // convert str
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> sparseFlashMlaAttenOut =
        MakeSparseFlashMlaOutputs(q, oriKv, cmpKv, layoutQStr, layoutKvStr, returnSoftmaxLse);
    at::Tensor attenOut = std::get<0>(sparseFlashMlaAttenOut);
    at::Tensor softmaxLse = std::get<1>(sparseFlashMlaAttenOut);

    EXEC_NPU_CMD(aclnnSparseFlashMla, q, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
              cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv,
              oriTopkLength, cmpTopkLength, sinks, metadata, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
              oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr, topkValueMode, returnSoftmaxLse, attenOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

} // namespace vllm_ascend

#endif // SPARSE_FLASH_MLA_TORCH_ADPT_H
