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
 * \file mc2_common_infershape.h
 * \brief
 */

#ifndef MC2_COMMON_INFERSHAPE_H_
#define MC2_COMMON_INFERSHAPE_H_

#include "mc2_log.h"
#include "register/op_impl_registry.h"
#if defined(ENABLE_BUILT_IN)
#include "mc2_hcom_topo_info.h"
#endif
namespace ops {
const size_t GROUP = 0;
const size_t AG_IS_TRANS_A = 1;
const size_t AG_IS_TRANS_B = 2;
const size_t RANK_SIZE = 5;
const size_t GATHER_OUT_V1 = 6;
const size_t GATHER_OUT_V2 = 8;
const size_t SUPPORT_DIM_SIZE = 2;
const size_t RS_IS_TRANS_A = 2;
const size_t RS_IS_TRANS_B = 3;
const size_t RS_IS_AMAX_OUT = 8;
const size_t AG_IS_AMAX_OUT = 9;

struct CommParas {
    const gert::Shape *x1MatrixShape;
    const gert::Shape *x2MatrixShape;
    int64_t dimM;
    int64_t dimKX1;
    int64_t dimKX2;
    int64_t dimN;
    int64_t rankSize;
};

ge::graphStatus AllGatherMatmulCommonInferShape(gert::InferShapeContext *context, const size_t gatherIndex);
ge::graphStatus InferMatmulReduceScatterCommon(gert::InferShapeContext *context);
} // namespace ops
#endif // MC2_COMMON_INFERSHAPE_H_
