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
 * \file mla_prolog_v3_infershape.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_INFERSHAPE_H
#define MLA_PROLOG_V3_INFERSHAPE_H

#include "mla_prolog_infershape.h"

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t WEIGHT_DQ_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INDEX = 2;
constexpr uint32_t WEIGHT_DKV_KR_INDEX = 4;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
// OUTPUT
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;
constexpr uint32_t QUERY_NORM_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_Q_NORM_INDEX = 6;
// ATTRIBUTE
constexpr uint32_t ATTR_QUERY_NORM_FLAG_INDEX = 3;
constexpr uint32_t ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX = 4;
constexpr uint32_t ATTR_KV_QUANT_MODE_FLAG_INDEX = 5;

constexpr uint32_t WEIGHT_QUANT_MODE_NO_QUANT = 0;
constexpr uint32_t WEIGHT_QUANT_MODE_PARTIAL_QUANT = 1;
constexpr uint32_t WEIGHT_QUANT_MODE_FULL_QUANT = 2;
constexpr uint32_t WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT = 3;
constexpr uint32_t WEIGHT_QUANT_MODE_FULL_QUANT_FP8 = 4;
constexpr uint32_t WEIGHT_QUANT_MODE_FULL_QUANT_HIF8 = 5;
constexpr uint32_t KV_QUANT_MODE_NO_QUANT = 0;
constexpr uint32_t KV_QUANT_MODE_PER_TENSOR = 1;
constexpr uint32_t KV_QUANT_MODE_PER_CHANNEL = 2;
constexpr uint32_t KV_QUANT_MODE_PER_TILE = 3;

ge::graphStatus GetMlaPrologV3ShapeDim(const gert::InferShapeContext *context, MlaPrologProtoShapeParam &shapeParam);
ge::graphStatus SetMlaPrologV3ShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context);

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext *context);


}  // namespace ops

#endif // MLA_PROLOG_V3_INFERSHAPE_H