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
 * \file mixed_quant_sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "vendor/c6240b268/include/err/ops_err.h"
#include "mixed_quant_sparse_flash_mla_check.h"
#include "vendor/c6240b268/common/smla_infershape_common.h"

using namespace ge;
using namespace optiling;

namespace ops {
ge::graphStatus InferShapeMixedQuantSparseFlashMla(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MixedQuantSparseFlashMla", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    return MQSMLAC624InferShape(context, "MixedQuantSparseFlashMla", ATTR_RETURN_SOFTMAX_LSE_INDEX, ATTR_LAYOUT_Q_INDEX,
                          ATTR_LAYOUT_KV_INDEX);
}

ge::graphStatus InferDataTypeMixedQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    return MQSMLAC624InferDataTypeByInput(context, "MixedQuantSparseFlashMla");
}

IMPL_OP_INFERSHAPE(MixedQuantSparseFlashMla)
    .InferShape(InferShapeMixedQuantSparseFlashMla)
    .InferDataType(InferDataTypeMixedQuantSparseFlashMla);
} // namespace ops
