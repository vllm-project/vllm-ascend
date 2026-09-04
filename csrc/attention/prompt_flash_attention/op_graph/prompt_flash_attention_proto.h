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
 * \file prompt_flash_attention_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PROMPT_FLASH_ATTENTION_OPS_H_
#define OPS_OP_PROTO_INC_PROMPT_FLASH_ATTENTION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Function PromptFlashAttention.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, int8.
* @li key: A matrix Tensor. The type support float16, bf16, int8.
* @li value: A matrix Tensor. The type support float16, bf16, int8.
* @li pse_shift: A matrix Tensor. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. The type support float16, bool, int8, uint8.
* @li actual_seq_lengths: A Tensor. The type support int64.
* @li actual_seq_lengths_kv: A Tensor. The type support int64.
* @li deq_scale1: A Tensor. The type support uint64, float32.
* @li quant_scale1: A Tensor. The type support float32.
* @li deq_scale2: A Tensor. The type support uint64, float32.
* @li quant_scale2: A Tensor. The type support float32, bf16.
* @li quant_offset2: A Tensor. The type support float32, bf16.

* @par Attributes:
* @li num_heads: An int. The number of the heads.
* @li scale_value: A float. The scale value. Default: 1.0.
* @li pre_tokens: An int. Previous tokens. Default: 214748647.
* @li next_tokens: An int. Next tokens. Default: 0.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "BNSD", "BSND", "BNSD_BSND"]. Default: "BSH".
* @li num_key_value_heads: Key value num heads. Default: 0.
* @li sparse_mode: Sparse mode. Default: 0.
* @li inner_precise: An int. 0, float16 high precision. 1, high performance. Default: 1.

* @par Outputs:
* @li attention_out: A matrix Tensor. The type support float16, bf16, int8.

* @attention Constraints:
* @li Ensure CANN and PyTorch package version compatibility when using this interface with PyTorch.
* @li Handle empty input: If 'query' is empty, return directly. If 'query' is non-empty and 'key', 'value' are empty tensors (S2=0), fill 'attention_out' with zeros of the corresponding shape.
* If 'attention_out' is an empty tensor, AscendCLNN will process it.
* @li The 'sparseMode' parameter currently only supports values 0, 1, 2, 3, and 4; other values will cause an error.
* @li Output is INT8. 'quantOffset2' must be a non-empty pointer and tensor. 'sparseMode', 'preTokens', and 'nextTokens' must meet certain conditions.
* If some rows of the matrix do not participate in calculations, resulting in computational errors, this scenario will be blocked
* (solution: if you want this scenario not to be blocked, post-quantization operations should be performed outside the PFA interface, not enabled within).
* For `sparseMode = 0`, if `attenMask` is a non-empty pointer, the condition for interception is `actualSeqLengths - actualSeqLengthsKV - preTokens > 0` or `nextTokens < 0` per batch.
* For `sparseMode = 1` or `2`, no interception conditions are met.
* For `sparseMode = 3`, the condition for interception is `actualSeqLengthsKV - actualSeqLengths < 0` per batch.
* For `sparseMode = 4`, the condition for interception is `preTokens < 0` or `nextTokens + actualSeqLengthsKV - actualSeqLengths < 0` per batch.
*/
REG_OP(PromptFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(pre_tokens, Int, 214748647)
    .ATTR(next_tokens, Int, 0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(PromptFlashAttention)
}  // namespace ge

#endif // OPS_OP_PROTO_INC_PROMPT_FLASH_ATTENTION_OPS_H_