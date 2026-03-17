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
 * \file swi_glu_dynamic_quant_proto.h
 * \brief
 */
#ifndef OPS_QUANT_SWI_GLU_DYNAMIC_QUANT_PROTO_H_
#define OPS_QUANT_SWI_GLU_DYNAMIC_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief SwiGlu and DynamicQuant are integrated to implement quantization. Only MOE group quantization is supported.
*        The scale output stores the dequantization factor (inverse of quantization scale).

* @par Inputs:
* @li x: A matrix tensor. Must be one of the following types: float32,float16,bfloat16, has format ND.
* @li smooth_scales: A optional tensor. Describing the result of dynamic quantize scales.
        A tensor of type float32, has format ND.
* @li offsets: A optional tensor, describing the data of offsets, a tensor of type float32, has format ND.
* @li group_index: A optional tensor, described grouping data, a tensor of type int32, has format ND.

* @par Attributes:
* @li activate_left: A optional bool.
*     The SwiGlu activate_left algorithm to use:
*     'false' (activate right) or 'true' (activate left), defalut is 'false' (activate right).
* @li quant_mode: Optional parameter, which formula used for quantized computation.
      Type is String, the value must be "dynamic" or "static" or "dynamic_msd", "static" indicates static quantization,
      "dynamic" indicates dynamic quantization, and "dynamic_msd" indicates dynamic mean squared displacement quantization, defaults to dynamic.
      Now only support "dynamic" and "static" mode.
* @li group_list_type: Optional parameter, which used to describe group_index mode.
      Type is Int, the value must be 0 or 1, 0 indicates "cumsum" mode, 1 indicates "count" mode.
      Now only support "cumsum" and "count" mode.
* @li dst_type: Optional parameter, which used to describe quant output mode.
      Type is Int, the value must be 2(DT_INT8 enum value) or 29(DT_INT4 enum value), 2 indicates "int8 quant output" mode, 29 indicates "int4 quant output" mode.
      Now only support "int8 quant output" and "int4 quant output" mode.

* @par Outputs:
* @li y: A tensor ,type is int8 or int4, the size of the last dimension of output y is half of the size of input x.
       And the size of other dimensions is the same as that of input x, now only support DT_INT8.
* @li scale: A tensor. Type is float32. Stores the dequantization factor (max_abs / quant_range).
      The shape of scale matches the shape of x across all dimensions except for the last dimension.
*/
REG_OP(SwiGluDynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(offsets, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(group_list_type, Int, 0)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(SwiGluDynamicQuant)
}  // namespace ge

#endif  // OPS_QUANT_SWI_GLU_DYNAMIC_QUANT_PROTO_H_
