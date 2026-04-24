/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hc_pre_sinkhorn_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;
namespace ops {
graphStatus InferShape4HcPreSinkhorn(gert::InferShapeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShape4HcPreSinkhorn.");

    const gert::Shape* mixShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, mixShape, ge::GRAPH_FAILED);
    auto mixDimNum = mixShape->GetDimNum();
    const gert::Shape* xShape = context->GetInputShape(4);
    OPS_LOG_E_IF_NULL(context, xShape, ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, yShape, ge::GRAPH_FAILED);
    gert::Shape* postShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, postShape, ge::GRAPH_FAILED);
    gert::Shape* combFragShape = context->GetOutputShape(2);
    OPS_LOG_E_IF_NULL(context, combFragShape, ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    auto *hcMult = attrs->GetAttrPointer<int>(0);

    yShape->SetDimNum(mixDimNum);
    postShape->SetDimNum(mixDimNum);
    combFragShape->SetDimNum(mixDimNum + 1);
    if (mixDimNum == 2) {
        yShape->SetDim(0, mixShape->GetDim(0));
        yShape->SetDim(1, xShape->GetDim(2));

        postShape->SetDim(0, mixShape->GetDim(0));
        postShape->SetDim(1, *hcMult);

        combFragShape->SetDim(0, mixShape->GetDim(0));
        combFragShape->SetDim(1, *hcMult);
        combFragShape->SetDim(2, *hcMult);
        
    } else {
        yShape->SetDim(0, mixShape->GetDim(0));
        yShape->SetDim(1, mixShape->GetDim(1));
        yShape->SetDim(2, xShape->GetDim(3));

        postShape->SetDim(0, mixShape->GetDim(0));
        postShape->SetDim(1, mixShape->GetDim(1));
        postShape->SetDim(2, *hcMult);

        combFragShape->SetDim(0, mixShape->GetDim(0));
        combFragShape->SetDim(1, mixShape->GetDim(1));
        combFragShape->SetDim(2, *hcMult);
        combFragShape->SetDim(3, *hcMult);
    }

    OPS_LOG_D(context->GetNodeName(), "End to do InferShape4HcPreSinkhorn");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4HcPreSinkhorn(gert::InferDataTypeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "InferDtype4HcPreSinkhorn enter");

    const auto xDataType = context->GetInputDataType(4);
    context->SetOutputDataType(0, xDataType);
    context->SetOutputDataType(1, DT_FLOAT);
    context->SetOutputDataType(2, DT_FLOAT);
    OPS_LOG_D(context->GetNodeName(), "InferDtype4HcPreSinkhorn end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcPreSinkhorn)
    .InferShape(InferShape4HcPreSinkhorn)
    .InferDataType(InferDtype4HcPreSinkhorn);
}  // namespace ops
