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
 * \file quant_lightning_indexer_metadata_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "../../quant_lightning_indexer/op_kernel/quant_lightning_indexer_metadata.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeQuantLightningIndexerMetaData(gert::InferShapeContext* context)
{
    gert::Shape* oShape = context->GetOutputShape(0);
    oShape->SetDimNum(1);
    oShape->SetDim(0, optiling::QLI_META_SIZE);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeQuantLightningIndexerMetaData(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantLightningIndexerMetadata)
    .InferShape(InferShapeQuantLightningIndexerMetaData)
    .InferDataType(InferDtypeQuantLightningIndexerMetaData);
} // namespace ops
