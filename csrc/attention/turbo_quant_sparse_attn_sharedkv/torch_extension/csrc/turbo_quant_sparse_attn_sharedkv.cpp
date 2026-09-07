/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
std::tuple<at::Tensor, at::Tensor> ConstructOutputs(const at::Tensor &q, bool returnSoftmaxLse)
{
    TORCH_CHECK(q.dim() == 3, "query must use TND layout");
    for (int64_t dim = 1; dim < q.dim(); ++dim) {
        TORCH_CHECK(q.size(dim) > 0, "query's N and D dimensions must be greater than 0");
    }
    at::Tensor attnOut = at::empty(q.sizes(), q.options());
    at::Tensor softmaxLse;
    if (returnSoftmaxLse) {
        std::vector<int64_t> lseShape(q.sizes().begin(), q.sizes().end());
        lseShape.back() = 1;
        softmaxLse = at::empty(lseShape, q.options().dtype(at::kFloat));
    } else {
        softmaxLse = at::empty({0}, q.options().dtype(at::kFloat));
    }
    return {attnOut, softmaxLse};
}
} // namespace

std::tuple<at::Tensor, at::Tensor> TurboQuantSparseAttnSharedkv(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedKv, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, c10::string_view layoutQ, c10::string_view layoutKv,
    bool returnSoftmaxLse, int64_t kvQuantMode)
{
    std::string layoutQStr(layoutQ);
    std::string layoutKvStr(layoutKv);
    auto outputs = ConstructOutputs(q, returnSoftmaxLse);
    at::Tensor attnOut = std::get<0>(outputs);
    at::Tensor softmaxLse = std::get<1>(outputs);
    int64_t oriKvStride = oriKv.has_value() ? oriKv.value().stride(0) : 0;
    int64_t cmpKvStride = cmpKv.has_value() ? cmpKv.value().stride(0) : 0;
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    ACLNN_CMD(aclnnTurboQuantSparseAttnSharedkv, q, oriKv, cmpKv, oriSparseIndices, cmpSparseIndices, oriBlockTable,
              cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedKv, sinks, metadata,
              softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriKvStride, cmpKvStride, oriWinLeft, oriWinRight,
              layoutQPtr, layoutKvPtr, returnSoftmaxLse, kvQuantMode, attnOut, softmaxLse);
    return {attnOut, softmaxLse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("turbo_quant_sparse_attn_sharedkv", &TurboQuantSparseAttnSharedkv, "turbo_quant_sparse_attn_sharedkv");
}
} // namespace op_api
