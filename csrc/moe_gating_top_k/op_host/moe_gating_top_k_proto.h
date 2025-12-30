/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOEGATINGTOPK_H_
#define OPS_OP_PROTO_INC_MOEGATINGTOPK_H_

#include "graph/operator_reg.h"

namespace ge {

/**
   * @brief Compute renorm(sigmoid) and topk for moe input.
   *
   * @par Inputs:
   * @li x: A 2D tensor which moe gating topk is applied, The shape is: (B*S, E), format supports ND, and data type must be float16, float or bfloat16. E(Expert num) can not be greater than 2048. E(Expert num) should be divisible by group_count.
   * @li bias: A 1D tensor which is "bias" in moe gating topk. The shape is: (E), format supports ND, and data type must be the same as that of x.
   *
   * @par Outputs:
   * @li y: A 2D tensor which is the topk value result of moe gating topk, format supports ND, and data type must be the same as that of x.
         The size of the non-1 axis must be the same as that of the corresponding axis of x.
         The size of the -1 axis must be the same as that of k.
   * @li expert_idx: A 2D tensor which is the topk index result of moe gating topk, format supports ND, and data type must be int. The shape must be the same as that of y.
   * @li out: A 2D tensor which is the renorm result of moe gating topk, format supports ND, and data type must be float. The shape must be the same as that of x.
   *
   * @par Attributes:
   * @li k: A required attribute of type int. The value must greater than 0 and less than or equal to expert_num / group_count * k_group, idicating the topk value.
   * @li k_group: An optional attribute of type int. It can not be less than 1, and can not be greater than group_count, indicating the topk group value. The default value is 1.
   * @li group_count: An optional attribute of type int. It can not be less than 1, indicating the group count. The group_count * align_32(expert_num / group_count) can not be greater than 2048. The default value is 1.
   * @li group_select_mode: An optional attribute of type int. 0 indicating that sort group by max values, 1 indicating that sort group by sum of top-2 values. The default value is 0.
   * @li renorm: An optional attribute of type int. It can only be 0 now, indicating that norm firstly and then topk. The default value is 0.
   * @li norm_type: An optional attribute of type int. 0 indicating that the softmax function is used, 1 indicating that the sigmoid function is used. The default value is 0.
   * @li out_flag: An optional attribute of type bool. true indicating that has renorm output, false indicating that does not have renorm output. The default value is false.
   * @li routed_scaling_factor: An optional attribute of type float, indicating the routed_scaling_factor coefficient in use. The default value is 1.0.
   * @li eps: An optional attribute of type float, indicating the eps coefficient in use. The default value is 1e-20.
   */
REG_OP(MoeGatingTopK)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(expert_idx, TensorType({DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(k, Int)
    .ATTR(k_group, Int, 1)
    .ATTR(group_count, Int, 1)
    .ATTR(group_select_mode, Int, 0)
    .ATTR(renorm, Int, 0)
    .ATTR(norm_type, Int, 0)
    .ATTR(out_flag, Bool, false)
    .ATTR(routed_scaling_factor, Float, 1.0)
    .ATTR(eps, Float, 1e-20f)
    .OP_END_FACTORY_REG(MoeGatingTopK)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOEGATINGTOPK_H_