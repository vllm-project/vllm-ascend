/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

static constexpr size_t INDEX_IN_Q = 0;
static constexpr size_t INDEX_IN_K = 1;
static constexpr size_t INDEX_IN_V = 2;
static constexpr size_t INDEX_IN_SLP_RATE = 3;
static constexpr size_t INDEX_IN_KV_HIS = 4;
static constexpr size_t INDEX_IN_SLT_IDS = 5;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t INDEX_OUT_ATTN = 0;
static constexpr size_t INDEX_OUT_KV_CACHES = 1;    

static ge::graphStatus InferShapeLightningAttentionDecode(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(INDEX_IN_Q);
    gert::Shape* attn_out_shape = context->GetOutputShape(INDEX_OUT_ATTN);
    gert::Shape* kv_caches_shape = context->GetOutputShape(INDEX_OUT_KV_CACHES);
    *attn_out_shape = *q_shape;

    kv_caches_shape->SetDimNum(q_shape->GetDimNum());
    kv_caches_shape->SetDim(0, q_shape->GetDim(0));
    kv_caches_shape->SetDim(1, q_shape->GetDim(1));
    kv_caches_shape->SetDim(DIM_2, q_shape->GetDim(DIM_3));
    kv_caches_shape->SetDim(DIM_3, q_shape->GetDim(DIM_3));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningAttentionDecode(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(INDEX_IN_Q);
    context->SetOutputDataType(INDEX_OUT_ATTN, inputDataType);
    context->SetOutputDataType(INDEX_OUT_KV_CACHES, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningAttentionDecode)
    .InferShape(InferShapeLightningAttentionDecode)
    .InferDataType(InferDataTypeLightningAttentionDecode);

}
