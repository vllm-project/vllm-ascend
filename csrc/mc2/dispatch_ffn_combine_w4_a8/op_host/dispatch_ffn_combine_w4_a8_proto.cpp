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
 * \file dispatch_ffn_w4_a8_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {
constexpr uint32_t OUTPUT_PROFILING_DATA = 2;
constexpr uint32_t MAX_INFER_GETBLOCKNUM_UB = 128;
constexpr uint32_t MIX_AIC_1_2_SLOTS_PER_GROUP = 3;
constexpr uint32_t MAX_PROFILING_CORE_SLOTS = MAX_INFER_GETBLOCKNUM_UB * MIX_AIC_1_2_SLOTS_PER_GROUP;
constexpr uint32_t PROF_SIZE_PER_CORE_INFER = 2048;

static ge::graphStatus InferShapeDispatchFFNCombineW4A8(gert::InferShapeContext* context) {
  gert::Shape *profilingShape = context->GetOutputShape(OUTPUT_PROFILING_DATA);
  if (profilingShape == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  profilingShape->SetDimNum(1);
  profilingShape->SetDim(0, MAX_PROFILING_CORE_SLOTS * PROF_SIZE_PER_CORE_INFER);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDispatchFFNCombineW4A8(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(OUTPUT_PROFILING_DATA, ge::DT_INT64);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DispatchFFNCombineW4A8)
  .InferShape(InferShapeDispatchFFNCombineW4A8)
  .InferDataType(InferDataTypeDispatchFFNCombineW4A8);
}  // namespace ops
