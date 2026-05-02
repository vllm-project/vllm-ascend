/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias_infershape.cpp
 * \brief
 */
#include "error/ops_error.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;

using namespace ge;
// using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4StoreKVBlock(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4StoreKVBlock(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StoreKVBlock).InferShape(InferShape4StoreKVBlock).InferDataType(InferDataType4StoreKVBlock);
} // namespace ops
