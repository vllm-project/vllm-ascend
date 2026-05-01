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
 * \file dequant_swiglu_quant_proto.h
 * \brief
 */
#ifndef OPS_QUANT_DEQUANT_SWIGLU_QUANT_PROTO_H_
#define OPS_QUANT_DEQUANT_SWIGLU_QUANT_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Combine Dequant + Swiglu + Quant.

* @par Inputs:
* Seven inputs, including:
* @li x: A tensor. Shape is (X..., H), dim must > 2, and H must be even. Type is int32, float16, bfloat16.
* @li weight_scale: Dequantization scale of weight. An optional tensor. Type is float32. Shape is (1..., H).
* @li activation_scale: Dequantization scale of activation. An optional tensor. Type is float32. Shape is  (X..., 1).
* @li bias: Bias for matmul. An optional tensor. Type is float16/bfloat16/int32/float32. Shape is (X..., H).
* @li quant_scale: Quantized scale. An optional tensor. Type is float16/bfloat16/float32. Shape is (1..., H).
* @li quant_offset: Quantized offset. An optional tensor. Type is float16/bfloat16/float32. Shape is (1..., H).
* @li group_index: Mean group index. An optional tensor. Type is int32/int64. Shape is (1,). \n

* @par Outputs:
* @li y: A tensor. Type is int8/fp8_e5m2/fp8_e4m3fn/fp4x2_e2m1/fp4x2_e1m2/hifloat8.
* @li scale: A tensor. Type is float32.

* @par Attributes:
* @li activate_left: Type is bool.
* The swi activate_left algorithm to use:
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
* @li quant_mode: Type is string. The quant mode to use: 'static' or 'dynamic', defalut is 'static'.
* @li dst_type: Type is Int32. Declare the output y dtype. Support 2:int8, 35:fp8_e5m2, 36:fp8_e4m3fn, 40:fp4x2_e2m1, 41:fp4x2_e1m2, Defaults to 2, only used for Ascend 950 AI Processors.
* @li round_mode: Type is String. The round mode to use: 'rint', 'round, 'floor', 'ceil', 'trunc', defalut is 'rint', only used for Ascend 950 AI Processors.
* @li activate_dim: Type is Int32. Describing the split dimension in Glu algorithm: value in [-xDim, xDim-1], defalut is -1, only used for Ascend 950 AI Processors.
* @li swiglu_mode: Type is int. Optional parameter, default is 0. The SWIGLU computation mode to use:
*     '0' (default) for standard SWIGLU, '1' for a variant using odd-even blocking, which requires support for clamp_limit, activation coefficient, and bias. This attribute is not supported in Ascend 950 AI Processors.
* @li clamp_limit: Type is float. Optional parameter, default is 7.0. The threshold limit for SWIGLU input. This attribute is not supported in Ascend 950 AI Processors.
* @li glu_alpha: Type is float. Optional parameter, default is 1.702. The activation coefficient for the GLU activation function. This attribute is not supported in Ascend 950 AI Processors.
* @li glu_bias: Type is float. Optional parameter, default is 1.0. The bias applied during SWIGLU linear computation. This attribute is not supported in Ascend 950 AI Processors.

* @attention Constraints:
* @li When the type of x is int32, weight_scale must be input.
* @li When the type of x is float16, bfloat16, weight_scale, activation_scale and bias must be None.
* @li When dst_type is int8, fp8_e5m2, fp8_e4m3fn, round_mode only supports 'rint'.
* @li When dst_type is fp4x2_e2m1 or fp4x2_e1m2, round_mode supports 'rint', 'round, 'floor', 'ceil' and 'trunc'.
* @li When dst_type is hifloat8, round_mode supports 'round'.
* @li The shape of activate_dim corresponding to x must be divisible by 2.
* @li When activate_dim is not the last dimension of x, group_index must be None.
* @li The input quant_offset is not supported in Ascend 950 AI Processors only.
* @li The type of output y is fp8_e5m2, fp8_e4m3fn, fp4x2_e2m1 and fp4x2_e1m2 only supported in Ascend 950 AI Processors.
* @li The attribute quant_mode is only supported 'dynamic' in Ascend 950 AI Processors.
* @li The attribute dst_type is only supported in Ascend 950 AI Processors.
* @li The attribute round_mode is only supported in Ascend 950 AI Processors.
* @li The attribute activate_dim is only supported in Ascend 950 AI Processors.
* @li The attribute swiglu_mode is not supported in Ascend 950 AI Processors.
* @li The attribute clamp_limit is not supported in Ascend 950 AI Processors.
* @li The attribute glu_alpha is not supported in Ascend 950 AI Processors.
* @li The attribute glu_bias is not supported in Ascend 950 AI Processors.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DequantSwigluQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8, DT_FP8_E4M3FN, DT_FP8_E5M2, DT_FP4X2_E2M1, DT_FP4X2_E1M2, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "static")
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(round_mode, String, "rint")
    .ATTR(activate_dim, Int, -1)
    .ATTR(swiglu_mode, Int, 0)
    .ATTR(clamp_limit, Float, 7.0)
    .ATTR(glu_alpha, Float, 1.702)
    .ATTR(glu_bias, Float, 1.0)
    .OP_END_FACTORY_REG(DequantSwigluQuant)
} // namespace ge

#endif // OPS_QUANT_DEQUANT_SWIGLU_QUANT_PROTO_H_