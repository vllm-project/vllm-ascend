/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file turboquant_reshape_and_cache_v310_infershape.cpp
 * \brief Norm planes: [num_slots, 16]; num_slots = num_blocks * block_size.
 *        One whole 32B block per slot (only the first num_kv_heads lanes carry
 *        data) so each slot owns its block: the write is a plain aligned
 *        DataCopy, idempotent under slot reuse, with no 64B cache-line sharing.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;
namespace ops {

static ge::graphStatus InferShapeTurboquantReshapeAndCacheV310(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TurboquantReshapeAndCacheV310", "inference context is null");
        return ge::GRAPH_FAILED;
    }
    auto opName = context->GetNodeName();
    auto keyShape = context->GetInputShape(0);        // [num_tokens, num_kv_heads, head_dim]
    auto cacheShape = context->GetInputShape(2);      // (num_blocks, C1, block_size, 16)
    auto kNormOut = context->GetOutputShape(0);
    auto vNormOut = context->GetOutputShape(1);
    if (keyShape == nullptr || cacheShape == nullptr || kNormOut == nullptr || vNormOut == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }
    constexpr int64_t kNormLanes = 16;   // one 32B block per slot
    (void)keyShape->GetDim(1);           // num_kv_heads no longer sets the stride
    const int64_t numSlots = cacheShape->GetDim(0) * cacheShape->GetDim(2);
    kNormOut->SetDimNum(2);
    kNormOut->SetDim(0, numSlots);
    kNormOut->SetDim(1, kNormLanes);
    vNormOut->SetDimNum(2);
    vNormOut->SetDim(0, numSlots);
    vNormOut->SetDim(1, kNormLanes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTurboquantReshapeAndCacheV310(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboquantReshapeAndCacheV310)
    .InferShape(InferShapeTurboquantReshapeAndCacheV310)
    .InferDataType(InferDataTypeTurboquantReshapeAndCacheV310);
}  // namespace ops
