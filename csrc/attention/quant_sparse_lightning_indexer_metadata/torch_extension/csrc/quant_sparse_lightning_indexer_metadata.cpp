/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// QSLI（Quant Sparse Lightning Indexer）metadata 的 JIT 测试入口。
// 与 QLI v2 metadata 分开：QLI metadata 不感知候选；本算子的工作量按 max(L)×8 token、
// 候选 tile=256 计算，输出为 QSLI 私有语义的 int32[1024]（不能与 QLI metadata 混用）。
#include <torch/extension.h>
#include "aclnn_common.h"

namespace qsli_metadata {

constexpr int64_t METADATA_SIZE = 1024;

at::Tensor QuantSparseLightningIndexerMetadata(
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t quantMode,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensK,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedK,
    const c10::optional<at::Tensor> &cmpResidualK, const at::Tensor &candidateBlockLength, int64_t batchSize,
    int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ, c10::string_view layoutK,
    int64_t maskMode, int64_t cmpRatio)
{
    at::Device outputDevice = candidateBlockLength.device();
    at::Tensor output = torch::empty({METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));

    auto cuSeqlensQVal = cuSeqlensQ.has_value() ? cuSeqlensQ.value() : at::empty({0}, candidateBlockLength.options());
    auto cuSeqlensKVal = cuSeqlensK.has_value() ? cuSeqlensK.value() : at::empty({0}, candidateBlockLength.options());
    auto sequsedQVal = sequsedQ.has_value() ? sequsedQ.value() : at::empty({0}, candidateBlockLength.options());
    auto sequsedKVal = sequsedK.has_value() ? sequsedK.value() : at::empty({0}, candidateBlockLength.options());
    auto cmpResidualKVal = cmpResidualK.has_value() ? cmpResidualK.value() : at::empty({0}, candidateBlockLength.options());

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKPtr = const_cast<char *>(layoutKStr.c_str());

    ACLNN_CMD(aclnnVllmAscendQuantSparseLightningIndexerMetadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal,
              sequsedKVal, cmpResidualKVal, candidateBlockLength, numHeadsQ, numHeadsK, headDim, topk, quantMode,
              batchSize, maxSeqlenQ, maxSeqlenK, layoutQPtr, layoutKPtr, maskMode, cmpRatio, output);
    return output;
}

} // namespace qsli_metadata

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_sparse_lightning_indexer_metadata", &qsli_metadata::QuantSparseLightningIndexerMetadata,
          "QSLI metadata (candidate-driven scheduling)", py::arg("num_heads_q"), py::arg("num_heads_k"),
          py::arg("head_dim"), py::arg("topk"), py::arg("quant_mode"), py::arg("cu_seqlens_q"),
          py::arg("cu_seqlens_k"), py::arg("seqused_q"), py::arg("seqused_k"), py::arg("cmp_residual_k"),
          py::arg("candidate_block_length"), py::arg("batch_size"), py::arg("max_seqlen_q"),
          py::arg("max_seqlen_k"), py::arg("layout_q"), py::arg("layout_k"), py::arg("mask_mode"),
          py::arg("cmp_ratio"));
}
