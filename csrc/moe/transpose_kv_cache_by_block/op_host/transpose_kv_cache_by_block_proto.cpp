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
 * \file transpose_kv_cache_by_block_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTransposeKvCacheByBlock(gert::InferShapeContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTransposeKvCacheByBlock(gert::InferDataTypeContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TransposeKvCacheByBlock)
    .InferShape(InferShapeTransposeKvCacheByBlock)
    .InferDataType(InferDataTypeTransposeKvCacheByBlock);
} // namespace ops
