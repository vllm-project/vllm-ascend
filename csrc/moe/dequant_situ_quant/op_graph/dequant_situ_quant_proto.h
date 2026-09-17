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
 * \file dequant_situ_quant_proto.h
 * \brief Kimi K3 routed-MoE Dequant + SiTU + dynamic quantization.
 */
#ifndef OPS_QUANT_DEQUANT_SITU_QUANT_PROTO_H_
#define OPS_QUANT_DEQUANT_SITU_QUANT_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Dequantize an INT32 grouped-matmul accumulator or consume an already
* dequantized BF16 WeightNz GMM output, apply Kimi K3 SiTU, and dynamically
* quantize each row.

* @par Inputs:
* Seven inputs, matching the grouped DequantSwigluQuant parameter order:
* @li x: Required INT32/BF16 tensor. Routed shape is [rows, 6144]. The shared-expert
* local gate-up width is 12288/6144/3072/1536/768 for TP1/2/4/8/16.
* @li weight_scale: FP32 tensor. Routed shape is [experts, 6144]; shared shape
* is [local_width] or [1, local_width]. Required when x is INT32.
* @li activation_scale: FP32 tensor containing one value per row. Required
* when x is INT32.
* @li bias: Optional FP32 tensor with the same grouped shape as weight_scale.
* @li quant_scale: Reserved optional input. It must be absent.
* @li quant_offset: Reserved optional input. It must be absent.
* @li group_index: Optional INT64 1-D tensor with shape [experts]. Each value
* is the number of consecutive valid routed rows belonging to that expert.
* For a pre-dequantized BF16 input, the counts are used to skip trailing
* dispatch-capacity rows. None selects the single-group shared-expert path.
* For BF16 x, GMM has already applied dequantization; weight_scale,
* activation_scale, and bias must all be absent.

* @par Outputs:
* @li y: INT8 tensor with shape [rows, x.shape[-1] / 2].
* @li scale: FP32 tensor with shape [rows].

* @par Attributes:
* @li beta: Must be 4.0.
* @li linear_beta: Must be 25.0.
* @li activate_left: Must be true.
* @li quant_mode: Must be "dynamic".

* @attention Constraints:
* @li Only the Kimi K3 A2/A3 W4A8 routed and TP-sharded shared contracts are supported.
* @li Non-empty group counts must cover x rows in expert order. Empty experts
* (zero counts) are supported.
*/
REG_OP(DequantSituQuant)
    .INPUT(x, TensorType({DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(beta, Float, 4.0)
    .ATTR(linear_beta, Float, 25.0)
    .ATTR(activate_left, Bool, true)
    .ATTR(quant_mode, String, "dynamic")
    .OP_END_FACTORY_REG(DequantSituQuant)
} // namespace ge

#endif // OPS_QUANT_DEQUANT_SITU_QUANT_PROTO_H_
