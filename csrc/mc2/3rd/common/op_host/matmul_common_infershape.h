/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_common_infershape.h
 * \brief
 */

#ifndef MATMUL_COMMON_INFERSHAPE_H
#define MATMUL_COMMON_INFERSHAPE_H

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace Ops {
namespace Transformer {
ge::graphStatus InferShapeForBatchMatMul(gert::InferShapeContext *context, const int32_t attr_adj_idx,
                                         const size_t input_bias_index, const bool is_x2_packed = false);
ge::graphStatus InferShapeRangeForBatchMatMul(gert::InferShapeRangeContext *context, const int32_t attr_adj_idx,
                                              const size_t input_bias_index);
bool CheckIsUnknownDimNum(const gert::Shape &shape);
} // namespace Transformer
} // namespace Ops

#endif
