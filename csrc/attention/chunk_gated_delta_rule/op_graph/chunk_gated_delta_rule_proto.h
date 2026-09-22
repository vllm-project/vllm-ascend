/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_CHUNK_GATED_DELTA_RULE_OPS_H_
#define OPS_OP_PROTO_INC_CHUNK_GATED_DELTA_RULE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Chunk Gated Delta Rule.
 *
 * @par Inputs:
 * @li query: A 3D tensor. The type supports bfloat16.
 * @li key: A 3D tensor. The type supports bfloat16.
 * @li value: A 3D tensor. The type supports bfloat16.
 * @li beta: A tensor. The type supports bfloat16.
 * @li initial_state: A tensor. The type supports bfloat16.
 * @li actual_seq_lengths: A tensor. The type supports int32.
 * @li g: An optional tensor. The type supports float32.
 *
 * @par Attributes:
 * @li scale_value: An optional float attribute. Defaults to 1.0.
 *
 * @par Outputs:
 * @li out: A tensor. The type supports bfloat16.
 * @li final_state: A tensor. The type supports bfloat16.
 */
// 图模式算子注册（输入/输出/属性用于 GE 识别与校验）。
REG_OP(ChunkGatedDeltaRule)
    .INPUT(query, TensorType({DT_BF16}))
    .INPUT(key, TensorType({DT_BF16}))
    .INPUT(value, TensorType({DT_BF16}))
    .INPUT(beta, TensorType({DT_BF16}))
    .INPUT(initial_state, TensorType({DT_BF16, DT_FLOAT}))
    .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(g, TensorType({DT_FLOAT}))
    .OUTPUT(out, TensorType({DT_BF16}))
    .OUTPUT(final_state, TensorType({DT_BF16, DT_FLOAT}))
    .ATTR(scale_value, Float, 1.0)
    .OP_END_FACTORY_REG(ChunkGatedDeltaRule)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CHUNK_GATED_DELTA_RULE_OPS_H_
