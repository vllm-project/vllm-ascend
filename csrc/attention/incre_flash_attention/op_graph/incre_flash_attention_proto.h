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
 * \file incre_flash_attention_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_INCRE_FLASH_ATTENTION_OPS_H_
#define OPS_OP_PROTO_INC_INCRE_FLASH_ATTENTION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Implement incremental inference based on full inference.

* @par Inputs:
* @li query: A matrix Tensor. The type support float16, bf16, int8.
* @li key: It's a dynamic input. A matrix Tensor. The type support float16, bf16, int8.
* @li value: It's a dynamic input. A matrix Tensor. The type support float16, bf16, int8.
* @li pse_shift: A matrix Tensor. Position coding inside the attention structure. The type support float16, bf16.
* @li atten_mask: A matrix Tensor. Mask the result of multiplying query by key to indicate whether to calculate the correlation between tokens.
* The type support bool, int8, uint8.
* @li actual_seq_lengths: A matrix Tensor. Indicates the valid sequence length of the key/value in different batches.
* The type support int64.
* @li dequant_scale1: A matrix Tensor. Dequantization factor after multiplying query by key.
* The type support uint64, float32.
* @li quant_scale1: A matrix Tensor. Indicates the quantization factor before multiplying query by key.
* The type support float32.
* @li dequant_scale2: A matrix Tensor. Dequantization factor after multiplying the result of softmax by value.
* The type support uint64, float32.
* @li quant_scale2: A matrix Tensor. Quantization factor of the output. The type support float32, bf16.
* @li quant_offset2: A matrix Tensor. Indicates the quantization offset of the output. The type support float32, bf16.
* @li antiquant_scale: A matrix Tensor. Indicates the antiquant factor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. Indicates the antiquant offset. The type support float16, bf16.
* @li block_table: A matrix Tensor. Indicates the block mapping table used by KV storage in PageAttention.
* The type support int32.
* @li kv_padding_size: A matrix Tensor. Indicates whether the data of each batch in the key/value is
* right-aligned and the number of right-aligned data.The type support int64.

* @par Attributes:
* @li num_heads: A required int. The number of the heads.
* @li scale_value: An optional float. The scale value. Default: 1.0.
* @li input_layout: An optional string. Specifies the layout of query, the value must be one of ["BSH", "BNSD", "BSND"]. Default: "BSH".
* @li num_key_value_heads: An optional int. Key value num heads. Default: 1.
* @li block_size: An optional int. Max length in pageattention's kv block. Default: 0.
* @li inner_precise: An optional int. When innerPrecise is 0, the high-precision mode is used.
* When innerPrecise is 1, the high-performance mode is used. Default: 1.

* @par Outputs:
* attention_out: A matrix Tensor. The type support float16, bf16, int8. \n

* @attention Constraints:
* - Constraints for empty Input:
* @li Direct return if query is empty.
* @li If query exists, key and value are empty: output a zero-filled tensor of corresponding shape.
* @li AscendCLNN framework handles if attention_out is an empty tensor.
* @li No processing for parameters marked as can pass nullptr if they are null pointers.
* @li Key and value tensor shapes must match; batch in non-continuous scenarios can only be 1.
*
* - Constraints for int8 quantization:
* @li Specific parameter existence and data format requirements based on input/output data formats.
* @li Both input and output int8: need deqScale1, quantScale1, deqScale2, quantScale2.
* @li Input int8, output float16: need deqScale1, quantScale1, deqScale2; error if quantOffset2 or quantScale2 not nullptr.
* @li Input float16/bf16, output int8: only quantScale2 needs to exist.
*
* - Constraints for antiquant:
* @li Support per-tensor/per-channel formats and float32/bf16 data types.
* @li Types and shape of quantScale2 and quantOffset2 need to be consistent.
* @li Specific recommendations for quantScale2 shape based on input data type and output layout.
* @li Support per-channel, per-tensor, and per-token modes, and symmetric/asymmetric quantization.
* @li Per-channel mode: shape supports (2, N, 1, D), (2, N, D), (2, H); data type matches query; antiquantMode set to 0.
* @li Per-tensor mode: shape (2), data type matches query; antiquantMode set to 0.
* @li Per-token mode: shape (2, B, S), data type float32; antiquantMode set to 1.
* @li Symmetric quantization: antiquantOffset can be empty; if empty, symmetric quantization is performed.
* @li Asymmetric quantization: both antiquantScale and antiquantOffset need to exist.
*/
REG_OP(IncreFlashAttention)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_INT8, DT_UINT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1.0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention)
}  // namespace ge

#endif // OPS_OP_PROTO_INC_INCRE_FLASH_ATTENTION_OPS_H_