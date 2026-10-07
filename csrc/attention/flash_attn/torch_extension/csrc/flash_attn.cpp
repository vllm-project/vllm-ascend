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
 * \file flash_attn.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t DIM_FOUR = 4;
const int64_t DIM_FIVE = 5;
const int64_t MAX_DIM_SIZE = 8;

at::Tensor FlashAttnMetadata(const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensKv,
                             const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedKv,
                             int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t headDimV,
                             int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t maskMode,
                             int64_t winLeft, int64_t winRight, std::string layoutQ, std::string layoutKv,
                             std::string layoutOut, bool isGradEnabled, const at::Tensor &output)
{
    ACLNN_CMD(aclnnFlashAttnMetadata, cuSeqlensQ, cuSeqlensKv, sequsedQ, sequsedKv, batchSize, maxSeqlenQ, maxSeqlenKv,
              numHeadsQ, numHeadsKv, headDim, headDimV, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut,
              isGradEnabled, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> FlashAttn(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensKv,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &sequsedKv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attnMask,
    const c10::optional<at::Tensor> &metadata, double softmaxScale, int64_t maskMode, int64_t winLeft, int64_t winRight,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, string layoutQ, string layoutKv, string layoutOut,
    int64_t returnSoftmaxLse)
{
    int64_t tSize = 0;
    int64_t nSize = 0;
    int64_t dSize = 0;
    int64_t sSize = 0;
    int64_t bSize = 0;
    at::SmallVector<int64_t, MAX_DIM_SIZE> attentionOutSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> softmaxOutSize;
    if (layoutQ == "TND") {
        tSize = q.size(0);
        nSize = q.size(1);
        dSize = q.size(2);
    } else if (layoutQ == "BSND") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2);
        dSize = q.size(3);
    } else {
        bSize = q.size(0);
        nSize = q.size(1);
        sSize = q.size(2);
        dSize = q.size(3);
    }
    // attention_out 的 D 维取 v 的 head_dim（支持 qk != v）；布局与维度数严格匹配，不匹配回落 qk 维（畸形 shape 由算子
    // checker 拒绝）
    int64_t dSizeV = dSize;
    if ((layoutKv == "BSND" || layoutKv == "BNSD") && v.dim() == DIM_FOUR) {
        dSizeV = v.size(DIM_THREE);
    } else if (layoutKv == "TND" && v.dim() == DIM_THREE) {
        dSizeV = v.size(DIM_TWO);
    } else if ((layoutKv == "PA_BBND" || layoutKv == "PA_BNBD") && v.dim() == DIM_FOUR) {
        // PA_BBND (Bn,Bs,N2,D) / PA_BNBD (Bn,N2,Bs,D) 的 D 维在 index 3
        dSizeV = v.size(DIM_THREE);
    } else if (layoutKv == "PA_NZ" && v.dim() == DIM_FIVE) {
        // PA_NZ (Bn,N2,D/16,block_size,16) 的 D = dim2 * dim4
        dSizeV = v.size(DIM_TWO) * v.size(DIM_FOUR);
    }
    if (returnSoftmaxLse) {
        if (q.dim() == DIM_THREE) {
            softmaxOutSize = {nSize, tSize};
        } else {
            softmaxOutSize = {bSize, nSize, sSize};
        }
    } else {
        softmaxOutSize = {0};
    }
    at::Tensor softmaxLse = at::empty(softmaxOutSize, q.options().dtype(at::kFloat));

    if (layoutOut == "TND") {
        attentionOutSize = {tSize, nSize, dSizeV};
    } else if (layoutOut == "BNSD") {
        attentionOutSize = {bSize, nSize, sSize, dSizeV};
    } else {
        attentionOutSize = {bSize, sSize, nSize, dSizeV};
    }
    at::Tensor attentionOutput = at::empty(attentionOutSize, q.options().dtype(q.dtype()));

    char *layoutQPtr = const_cast<char *>(layoutQ.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKv.c_str());
    char *layoutOutPtr = const_cast<char *>(layoutOut.c_str());

    // Python bool → C++ int64_t: pybind11 auto-converts return_softmax_lse (True→1, False→0) before reaching here
    ACLNN_CMD(aclnnFlashAttn, q, k, v, blockTable, cuSeqlensQ, cuSeqlensKv, sequsedQ, sequsedKv, sinks, attnMask,
              metadata, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKv, layoutQPtr, layoutKvPtr,
              layoutOutPtr, returnSoftmaxLse, attentionOutput, softmaxLse);

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn", &FlashAttn, "flash_attn");
    m.def("flash_attn_metadata", &FlashAttnMetadata, "flash_attn_metadata");
}
} // namespace op_api
