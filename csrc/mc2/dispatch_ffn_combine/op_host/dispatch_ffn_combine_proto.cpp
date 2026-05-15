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
 * \file dispatch_ffn_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
// #include "../../common/ophost/op_util.h"
// #include "../../common/ophost/hcom_topo_info.h"
// #include "log/ops_log.h"

using namespace ge;
namespace ops {
const size_t ATTR_GROUP = 0;
const size_t ATTR_RANK_SIZE = 1;
const size_t SUPPORT_DIM_SIZE = 2;

static ge::graphStatus InferShapeDispatchFFNCombine(gert::InferShapeContext* context) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDispatchFFNCombine(gert::InferDataTypeContext* context) {
  // auto d_type = context->GetInputDataType(0);
  // context->SetOutputDataType(0, d_type);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DispatchFFNCombine)
  .InferShape(InferShapeDispatchFFNCombine)
  .InferDataType(InferDataTypeDispatchFFNCombine);
}  // namespace ops
