/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_MLA_CHECKER_ADAPTER_H
#define SPARSE_MLA_CHECKER_ADAPTER_H

#include "checker_context.h"

namespace optiling {
namespace sparse_mla_checker {

template <typename OptionalParam>
void PopulateOptionalTensorParam(gert::TilingContext *context, uint32_t index, OptionalParam &param)
{
    param.desc = context->GetOptionalInputDesc(index);
    param.shape = context->GetOptionalInputShape(index);
    param.tensor = context->GetOptionalInputTensor(index);
}

template <typename RequiredParam>
TensorParam MakeRequiredTensor(const RequiredParam &param)
{
    return {param.desc, param.shape == nullptr ? nullptr : &param.shape->GetStorageShape(),
            param.desc != nullptr || param.shape != nullptr};
}

template <typename OptionalParam>
TensorParam MakeOptionalTensor(const OptionalParam &param)
{
    const gert::Shape *shape = param.shape == nullptr ? nullptr : &param.shape->GetStorageShape();
    if (shape == nullptr && param.tensor != nullptr) {
        shape = &param.tensor->GetStorageShape();
    }
    return {param.desc, shape, shape != nullptr};
}

inline int64_t GetLastDim(const TensorParam &param)
{
    return param.shape == nullptr || param.shape->GetDimNum() == 0 ? 0 :
                                                                     param.shape->GetDim(param.shape->GetDimNum() - 1);
}

template <typename TilingInfo>
void PopulateCommonContext(CheckContext &context, const TilingInfo &info)
{
    const auto &param = info.opParamInfo;
    context.opName = info.opName;
    context.q = MakeRequiredTensor(param.q);
    context.oriKv = MakeOptionalTensor(param.oriKv);
    context.cmpKv = MakeOptionalTensor(param.cmpKv);
    context.oriSparseIndices = MakeOptionalTensor(param.oriSparseIndices);
    context.cmpSparseIndices = MakeOptionalTensor(param.cmpSparseIndices);
    context.oriBlockTable = MakeOptionalTensor(param.oriBlockTable);
    context.cmpBlockTable = MakeOptionalTensor(param.cmpBlockTable);
    context.cuSeqlensQ = MakeOptionalTensor(param.cuSeqLensQ);
    context.cuSeqlensOriKv = MakeOptionalTensor(param.cuSeqLensOriKv);
    context.cuSeqlensCmpKv = MakeOptionalTensor(param.cuSeqLensCmpKv);
    context.sequsedQ = MakeOptionalTensor(param.seqUsedQ);
    context.sequsedOriKv = MakeOptionalTensor(param.sequsedOriKv);
    context.sequsedCmpKv = MakeOptionalTensor(param.sequsedCmpKv);
    context.cmpResidualKv = MakeOptionalTensor(param.cmpResidualKv);
    context.oriTopkLength = MakeOptionalTensor(param.oriTopkLength);
    context.cmpTopkLength = MakeOptionalTensor(param.cmpTopkLength);
    context.sinks = MakeOptionalTensor(param.sinks);
    context.metadata = MakeOptionalTensor(param.metadata);
    context.attentionOut = MakeRequiredTensor(param.attnOut);
    context.softmaxLse = MakeRequiredTensor(param.softmaxLse);

    context.qLayout = static_cast<Layout>(static_cast<uint32_t>(info.qLayout));
    context.kvLayout = static_cast<Layout>(static_cast<uint32_t>(info.kvLayout));
    context.bSize = info.bSize;
    context.qNumHeads = info.n1Size;
    context.kvNumHeads = info.n2Size;
    context.qSeqSize = info.s1Size;
    context.qTotalSize = info.qTSize;
    context.oriBlockSize = info.oriBlockSize;
    context.cmpBlockSize = info.cmpBlockSize;
    context.oriTopk = GetLastDim(context.oriSparseIndices);
    context.cmpTopk = GetLastDim(context.cmpSparseIndices);
    context.softmaxScale = info.softmaxScale;
    context.cmpRatio = info.cmpRatio;
    context.oriMaskMode = static_cast<int64_t>(info.oriMaskMode);
    context.cmpMaskMode = static_cast<int64_t>(info.cmpMaskMode);
    context.oriWinLeft = info.oriWinLeft;
    context.oriWinRight = info.oriWinRight;
    context.topkValueMode = info.topkValueMode;
    context.returnSoftmaxLse = info.returnSoftmaxLse;
}

} // namespace sparse_mla_checker
} // namespace optiling

#endif // SPARSE_MLA_CHECKER_ADAPTER_H
