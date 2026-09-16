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
 * \file mixed_quant_sparse_flash_mla.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

constexpr int64_t MQSMLA_METADATA_SIZE = 1024;

inline TensorWrapper MqsmlaMakeWrapper(const at::Tensor &tensor, aclDataType tensorAcltype)
{
    return {tensor, tensorAcltype};
}

at::Tensor MixedQuantSparseFlashMlaMetadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t quantMode,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
    int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t ropeHeadDim, int64_t cmpRatio,
    int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, c10::string_view layoutQ,
    c10::string_view layoutKv, bool hasOriKv, bool hasCmpKv)
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

    at::Tensor output = torch::empty({MQSMLA_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensOriKvVal = get_valid_tensor(cuSeqlensOriKv, outputDevice);
    auto cuSeqlensCmpKvVal = get_valid_tensor(cuSeqlensCmpKv, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedOriKvVal = get_valid_tensor(sequsedOriKv, outputDevice);
    auto sequsedCmpKvVal = get_valid_tensor(sequsedCmpKv, outputDevice);
    auto cmpResidualKvVal = get_valid_tensor(cmpResidualKv, outputDevice);
    auto oriTopkLengthVal = get_valid_tensor(oriTopkLength, outputDevice);
    auto cmpTopkLengthVal = get_valid_tensor(cmpTopkLength, outputDevice);

    // convert str
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    ACLNN_CMD(aclnnMixedQuantSparseFlashMlaMetadata, cuSeqlensQVal, cuSeqlensOriKvVal, cuSeqlensCmpKvVal, sequsedQVal,
              sequsedOriKvVal, sequsedCmpKvVal, cmpResidualKvVal, oriTopkLengthVal, cmpTopkLengthVal, numHeadsQ,
              numHeadsKv, headDim, quantMode, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk,
              ropeHeadDim, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr,
              hasOriKv, hasCmpKv, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> ConstructMixedQuantSparseFlashMlaAttenOutTensor(
    const at::Tensor &q, const at::Tensor &oriKv, std::string layoutQStr, std::string layoutKvStr,
    const uint64_t &ropeHeadDim, bool returnSoftmaxLse)
{
    TORCH_CHECK(layoutQStr == "BSND" || layoutQStr == "TND", "The layout of query only support BSND and TND, but got ",
                layoutQStr);
    for (auto i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", q.size(i));
    }
    at::SmallVector<int64_t, SIZE> attenOutSize;
    at::SmallVector<int64_t, SIZE> softmaxLseSize;
    if (layoutQStr == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4, "When the layout of query is BSND, the query dimension must be 4, but got ",
                    q.dim());
        // atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3) - rope_head_dim};
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3)};
    } else {
        TORCH_CHECK(q.dim() == DIM_3, "When the layout of query is TND, the query dimension must be 3, but got ",
                    q.dim());
        // atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2) - rope_head_dim};
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2)};
    }
    at::Tensor attenOut = at::empty(attenOutSize, q.options().dtype(q.dtype()));

    if (returnSoftmaxLse) {
        TORCH_CHECK(oriKv.size(DIM_1) > 0, "oriKv.size(DIM_1) must be greater than 0, but got ", oriKv.size(DIM_1));
        TORCH_CHECK(oriKv.size(DIM_2) > 0, "oriKv.size(DIM_2) must be greater than 0, but got ", oriKv.size(DIM_2));
        if (layoutQStr == "BSND") {
            int64_t dim0 = static_cast<int64_t>(q.size(DIM_0));
            int64_t dim1 = static_cast<int64_t>(oriKv.size(DIM_2));
            int64_t dim2 = static_cast<int64_t>(q.size(DIM_1));
            int64_t dim3 = static_cast<int64_t>(q.size(DIM_2)) / static_cast<int64_t>(oriKv.size(DIM_2));
            softmaxLseSize = {dim0, dim1, dim2, dim3};
        } else {
            if (layoutKvStr == "PA_BBND") {
                int64_t dim0 = static_cast<int64_t>(oriKv.size(DIM_2));
                int64_t dim1 = static_cast<int64_t>(q.size(DIM_0));
                int64_t dim2 = static_cast<int64_t>(q.size(DIM_1)) / static_cast<int64_t>(oriKv.size(DIM_2));
                softmaxLseSize = {dim0, dim1, dim2};
            } else {
                int64_t dim0 = static_cast<int64_t>(oriKv.size(DIM_1));
                int64_t dim1 = static_cast<int64_t>(q.size(DIM_0));
                int64_t dim2 = static_cast<int64_t>(q.size(DIM_1)) / static_cast<int64_t>(oriKv.size(DIM_1));
                softmaxLseSize = {dim0, dim1, dim2};
            }
        }
    } else {
        // 不返回时tensor传空
        softmaxLseSize = {0};
    }
    at::Tensor softmaxLse = at::empty(softmaxLseSize, q.options().dtype(torch::kFloat32));

    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

std::tuple<at::Tensor, at::Tensor> MixedQuantSparseFlashMla(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, int64_t quantMode, int64_t ropeHeadDim, double softmaxScale,
    int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
    c10::string_view layoutQ, c10::string_view layoutKv, int64_t topkValueMode, bool returnSoftmaxLse,
    c10::optional<int64_t> keyDtype, c10::optional<int64_t> valueDtype)
{
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.")

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    const at::Tensor &oriKvVal = *oriKv;
    // convert str
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> mixedQuantSparseFlashMlaAttenOut =
        op_api::ConstructMixedQuantSparseFlashMlaAttenOutTensor(q, oriKvVal, layoutQStr, layoutKvStr, ropeHeadDim,
                                                                returnSoftmaxLse);
    at::Tensor attenOut = std::get<0>(mixedQuantSparseFlashMlaAttenOut);
    at::Tensor softmaxLse = std::get<1>(mixedQuantSparseFlashMlaAttenOut);

    at::Tensor nullTensor;
    auto oriKvValue = oriKv.has_value() ? oriKv.value() : nullTensor;
    auto cmpKvValue = cmpKv.has_value() ? cmpKv.value() : nullTensor;
    TensorWrapper oriKvWrapper = MqsmlaMakeWrapper(oriKvValue, ACL_FLOAT8_E4M3FN);
    TensorWrapper cmpKvWrapper = MqsmlaMakeWrapper(cmpKvValue, ACL_FLOAT8_E4M3FN);
    ACLNN_CMD(aclnnMixedQuantSparseFlashMla, q, oriKvWrapper, cmpKvWrapper, oriSparseIndices, cmpSparseIndices,
              oriBlockTable, cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv,
              sequsedCmpKv, cmpResidualKv, oriTopkLength, cmpTopkLength, sinks, metadata, quantMode, ropeHeadDim,
              softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr,
              topkValueMode, returnSoftmaxLse, attenOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mixed_quant_sparse_flash_mla_metadata", &MixedQuantSparseFlashMlaMetadata,
          "mixed_quant_sparse_flash_mla_metadata");
    m.def("mixed_quant_sparse_flash_mla", &MixedQuantSparseFlashMla, "mixed_quant_sparse_flash_mla");
}
} // namespace op_api
