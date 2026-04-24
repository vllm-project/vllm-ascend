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
 * \file hc_pre_inv_rms_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_X = 0;
const int32_t INDEX_OUTPUT_Y = 0;

static ge::graphStatus InferShape4HcPreInvRms(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4HcPreInvRms.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
    auto xDimNum = xShape->GetDimNum();

    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    yShape->SetDimNum(xDimNum);
    if (xDimNum == 4) {
        yShape->SetDim(0, xShape->GetDim(0));
        yShape->SetDim(1, xShape->GetDim(1));
        yShape->SetDim(2, 1);
    } else if (xDimNum == 3) {
        yShape->SetDim(0, xShape->GetDim(0));
        yShape->SetDim(1, 1);
    }
    
    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4HcPreInvRms");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4HcPreInvRms(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4HcHost enter");
    context->SetOutputDataType(INDEX_OUTPUT_Y, ge::DT_FLOAT);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4HcPreInvRms end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcPreInvRms)
    .InferShape(InferShape4HcPreInvRms)
    .InferDataType(InferDtype4HcPreInvRms);
}  // namespace ops