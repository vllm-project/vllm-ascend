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
 * \file op_transformer_proto_extend.h
 * \brief
 */
#ifndef OPS_OP_MATH_PROTO_EXTEND_H_
#define OPS_OP_MATH_PROTO_EXTEND_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief swin_transformer model specific structure.Operator only supports swin_transformer.

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be (B*W, N, S1, S2) or (B, W, N, S1, S2).
* @li atten_mask: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
                  the shape should be (W, S1, S2) or (W, 1, S1, S2) or (1, W, 1, S1, S2)
* @li relative_pos_bias: An ND Tensor. Must be one of the following types: float16, float, bfloat16.
                         the shape sholud be (N, S1, S2) or (1, N, S1, S2) or (1, 1, N, S1, S2)

* @par Attributes:
* @li scale_value: A optional attribute, the type is float. Defaults to 1.0.
* @li inner_precision_mode: A optional attribute, the type is int. Defaults to 0, reserved field.

* @par Outputs:
* One output, including:
* @li y: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be same with x.
*/
REG_OP(MaskedSoftmaxWithRelPosBias)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(relative_pos_bias, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(inner_precision_mode, Int, 0)
    .OP_END_FACTORY_REG(MaskedSoftmaxWithRelPosBias)

/**
* @brief AttentionScore's forward calculation.

* @par Inputs:
* six inputs, including:
* @li query: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li key: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li value: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li padding_mask: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li scale: A scalar. The type only support float16. Enter a 4D Tensor.
* @li drop_mask: A matrix Tensor. An optional input parameter. The type only support uint8. Enter a 4D Tensor.

* @par Attributes:
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li query_transpose: A bool. If True, changes the shape of "query" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li key_transpose: A bool. If True, changes the shape of "key" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li bmm_score_transpose_a: A bool. If True, changes the shape of "mid_data" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li bmm_score_transpose_b: A bool. If True, changes the shape of "value" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li softmax_axes: A list of int. The dimension softmax would be performed on. Defaults to "[-1]".

* @par Outputs:
* attention_score: The result matrix Tensor. The type only support float16. The output shape is the same as query.
* softmax_output: The result matrix Tensor. The type only support float16. The output shape is the same as query.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(padding_mask, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax_output, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(bmm_score_transpose_a, Bool, false)
    .ATTR(bmm_score_transpose_b, Bool, false)
    .ATTR(softmax_axes, ListInt, {-1})
    .OP_END_FACTORY_REG(AttentionScore)
}

#endif