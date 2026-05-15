/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file hc_post_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_X = 0;
const int32_t INPUT_IDX_RESIDUAL = 1;
const int32_t INPUT_IDX_POST = 2;
const int32_t INPUT_IDX_COMB = 3;
const int32_t INDEX_OUTPUT_Y = 0;


static ge::graphStatus InferShape4HcPost(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4HcPost.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);

    const gert::Shape* residualShape = context->GetInputShape(INPUT_IDX_RESIDUAL);
    OPS_LOG_E_IF_NULL(context, residualShape, return ge::GRAPH_FAILED);

    const gert::Shape* postShape = context->GetInputShape(INPUT_IDX_POST);
    OPS_LOG_E_IF_NULL(context, postShape, return ge::GRAPH_FAILED);

    const gert::Shape* combShape = context->GetInputShape(INPUT_IDX_COMB);
    OPS_LOG_E_IF_NULL(context, combShape, return ge::GRAPH_FAILED);

    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    *yShape = *residualShape;

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4HcPost");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4HcPost(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4HcPost enter");
    const auto xDtype = context->GetInputDataType(INPUT_IDX_X);
    context->SetOutputDataType(INDEX_OUTPUT_Y, xDtype);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4HcPost end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcPost)
    .InferShape(InferShape4HcPost)
    .InferDataType(InferDtype4HcPost);
}  // namespace ops