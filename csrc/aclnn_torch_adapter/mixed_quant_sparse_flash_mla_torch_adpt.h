/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Adapted from ops-transformer c6240b268a6818ff343c721b8508217d4cfb9812.
#ifndef VLLM_MIXED_QUANT_SPARSE_FLASH_MLA_TORCH_ADPT_H
#define VLLM_MIXED_QUANT_SPARSE_FLASH_MLA_TORCH_ADPT_H

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>
#include <tuple>

namespace vllm_ascend::mqsmla {

inline constexpr int64_t MQSMLA_METADATA_SIZE = 1024;

// Preserve the eager bridge's full query-shaped output and its empty [0] LSE.
// SymInt keeps this helper usable by the Meta implementation as well.
inline std::tuple<at::Tensor, at::Tensor> ConstructOutputs(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv,
    c10::string_view layoutQ, c10::string_view layoutKv, bool returnSoftmaxLse)
{
    TORCH_CHECK(layoutQ == "BSND" || layoutQ == "TND", "query layout must be BSND or TND");
    TORCH_CHECK(q.dim() == (layoutQ == "BSND" ? 4 : 3), "query rank does not match layout_q");
    for (const auto &dim : q.sym_sizes()) {
        TORCH_CHECK(dim > 0, "All dimensions of query must be positive");
    }
    at::Tensor output = at::empty_symint(q.sym_sizes(), q.options());
    c10::SymDimVector lseShape{c10::SymInt(0)};
    if (returnSoftmaxLse) {
        TORCH_CHECK(oriKv.has_value() && oriKv->defined(), "ori_kv is required when return_softmax_lse=True");
        const auto &kv = oriKv.value();
        TORCH_CHECK(kv.dim() >= 3 && kv.sym_size(1) > 0 && kv.sym_size(2) > 0,
                    "ori_kv dimensions 1 and 2 must be positive");
        const auto kvHeads = kv.sym_size(layoutQ == "BSND" || layoutKv == "PA_BBND" ? 2 : 1);
        if (layoutQ == "BSND") {
            lseShape = {q.sym_size(0), kvHeads, q.sym_size(1), q.sym_size(2) / kvHeads};
        } else {
            lseShape = {kvHeads, q.sym_size(0), q.sym_size(1) / kvHeads};
        }
    }
    return {output, at::empty_symint(lseShape, q.options().dtype(at::kFloat))};
}

at::Tensor MixedQuantSparseFlashMlaMetadata(
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t quantMode,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, c10::optional<int64_t> batchSize, c10::optional<int64_t> maxSeqlenQ, c10::optional<int64_t> maxSeqlenOriKv,
    c10::optional<int64_t> maxSeqlenCmpKv, c10::optional<int64_t> oriTopk, c10::optional<int64_t> cmpTopk, c10::optional<int64_t> ropeHeadDim, c10::optional<int64_t> cmpRatio,
    c10::optional<int64_t> oriMaskMode, c10::optional<int64_t> cmpMaskMode, c10::optional<int64_t> oriWinLeft, c10::optional<int64_t> oriWinRight, c10::optional<c10::string_view> layoutQ,
    c10::optional<c10::string_view> layoutKv, c10::optional<bool> hasOriKv, c10::optional<bool> hasCmpKv);

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
    c10::optional<int64_t> keyDtype, c10::optional<int64_t> valueDtype);

}  // namespace vllm_ascend::mqsmla

#endif
