/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
ge::graphStatus InferShape4KvCompressEpilogV2(gert::InferShapeContext *context)
{
    const gert::Shape *cacheShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, cacheShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, outputShape, return ge::GRAPH_FAILED);
    *outputShape = *cacheShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4KvCompressEpilogV2(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCompressEpilogV2)
    .InferShape(InferShape4KvCompressEpilogV2)
    .InferDataType(InferDtype4KvCompressEpilogV2);
}  // namespace ops
