/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_update_v2_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"


using namespace ge;
namespace ops {
constexpr int32_t LSE_INDEX = 0;
constexpr int32_t GO_INDEX = 1;
constexpr int32_t OUT_INDEX = 0;
constexpr int32_t LSEOUT_INDEX = 1;

static ge::graphStatus InferShape4DecodeUpdate(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4DecodeUpdate.");

    // lse: [sp, bsh],  go: [sp, bsh, hd]
    auto lseShape = context->GetInputShape(LSE_INDEX);
    auto goShape  = context->GetInputShape(GO_INDEX);
    auto outShape    = context->GetOutputShape(OUT_INDEX);
    auto lseOutShape = context->GetOutputShape(LSEOUT_INDEX);

    // out: [bsh, hd]
    outShape->SetDimNum(2);
    outShape->SetDim(0, goShape->GetDim(1));   // bsh
    outShape->SetDim(1, goShape->GetDim(2));   // hd

    // lseOut: [bsh]
    lseOutShape->SetDimNum(1);
    lseOutShape->SetDim(0, lseShape->GetDim(1));  // bsh

    OP_LOGI(context, "End to do InferShape4DecodeUpdate.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DecodeUpdate(gert::InferDataTypeContext* context)
{
    OP_LOGI(context, "Begin to do InferDataType4DecodeUpdate.");

    auto lseDtype      = context->GetInputDataType(LSE_INDEX);
    auto localOutDtype = context->GetInputDataType(GO_INDEX);
    context->SetOutputDataType(OUT_INDEX, localOutDtype);
    context->SetOutputDataType(LSEOUT_INDEX, lseDtype);

    OP_LOGI(context, "End to do InferDataType4DecodeUpdate.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AttentionUpdateV2)
    .InferShape(ops::InferShape4DecodeUpdate)
    .InferDataType(ops::InferDataType4DecodeUpdate);


} // namespace ops