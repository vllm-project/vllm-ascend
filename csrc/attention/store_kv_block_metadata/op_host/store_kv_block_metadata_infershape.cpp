/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file store_kv_block_metadata_infershape.cpp
 * \brief InferShape implementation for StoreKvBlockMetadata
 */
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {

static constexpr int DIM_0 = 0;

static ge::graphStatus InferShape4StoreKvBlockMetadata(gert::InferShapeContext* context)
{
    // Output shapes match input slot_mapping shape (max possible groups = slot_mapping length)
    auto inputShape = context->GetInputShape(DIM_0);
    if (inputShape == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape* groupLenShape = context->GetOutputShape(0);
    gert::Shape* groupKeyIdxShape = context->GetOutputShape(1);
    gert::Shape* groupKeyCacheIdxShape = context->GetOutputShape(2);
    if (groupLenShape == nullptr || groupKeyIdxShape == nullptr || groupKeyCacheIdxShape == nullptr) {
        return GRAPH_FAILED;
    }

    *groupLenShape = *inputShape;
    *groupKeyIdxShape = *inputShape;
    *groupKeyCacheIdxShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4StoreKvBlockMetadata(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    context->SetOutputDataType(1, DT_INT32);
    context->SetOutputDataType(2, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StoreKvBlockMetadata)
    .InferShape(InferShape4StoreKvBlockMetadata)
    .InferDataType(InferDataType4StoreKvBlockMetadata);

} // namespace ops
