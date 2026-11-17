/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_metadata_update_proto.cpp
 * \brief InferShape and InferDataType for AttnMetadataUpdate.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeAttnMetadataUpdate(gert::InferShapeContext *context)
{
    // Output shapes are determined by the max_bs attr and block_table input.
    // Shape inference is handled at runtime via tiling; return success here.
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAttnMetadataUpdate(gert::InferDataTypeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AttnMetadataUpdate)
    .InferShape(InferShapeAttnMetadataUpdate)
    .InferDataType(InferDataTypeAttnMetadataUpdate);

}  // namespace ops
