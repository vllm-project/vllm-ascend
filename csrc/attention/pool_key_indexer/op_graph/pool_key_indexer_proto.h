/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_PoolKeyIndexer_H_
#define OPS_OP_PROTO_INC_PoolKeyIndexer_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief PoolKeyIndexer packs consecutive tokens into pools, computes attention
 *        relevance scores per pool, selects top-k pools and expands them into
 *        sparse token indices, so as to reduce indexing overhead while keeping
 *        the benefit of sparse attention. \n

 * @par Inputs:
 * Inputs including:
 * @li query: The Query tensor used for pool score computation. \n
 *   - Data types: float16, bfloat16, float8_e4m3fn.
 *   - format: ND
 * @li pool_key: The pooled Key tensor used for pool score computation. \n
 *   - Data types: float16, bfloat16, float8_e4m3fn.
 *   - format: ND
 * @li weights: The multi-head weight projection matrix for score aggregation. \n
 *   - Data types: float16, bfloat16.
 *   - format: ND
 * @li pool_tail_k: The number of valid tokens in the tail (incomplete) pool of each batch.
 *   Value range is [0, pool_size - 1]. \n
 *   - Data types: int64.
 *   - format: ND
 * @li actual_seq_q: Optional. The valid token count of Query per batch.
 *   In TND scenario, each element represents a prefix sum of token counts. \n
 *   - Data types: int64.
 *   - format: ND
 * @li actual_seq_k: Optional. The valid pool count of pool_key per batch
 *   (pool count = floor(valid token count / pool_size)).
 *   In PageAttention (PA_BBND) scenario, each element represents the pool count
 *   of the current batch (not a prefix sum). In TND scenario, each element
 *   represents a prefix sum of pool counts. \n
 *   - Data types: int64.
 *   - format: ND
 * @li block_table: Optional. The block mapping table used in PageAttention KV storage. \n
 *   - Data types: int32.
 *   - format: ND
 * @li q_descale: Optional. The dequantization scale of Query. \n
 *   - Data types: float32, float8_e8m0.
 *   - format: ND
 * @li k_descale: Optional. The dequantization scale of Key. \n
 *   - Data types: float32, float8_e8m0.
 *   - format: ND

 * @par Attributes:
 * @li topk: An optional int attribute. The number of tokens to keep after pool
 *   expansion. k_pool = floor(topk / pool_size) pools are selected.
 *   Defaults to 2048.
 * @li pool_size: An optional int attribute. The number of tokens packed into one pool.
 *   Defaults to 16.
 * @li layout_q: An optional string attribute. The layout of Query, "BSND" or "TND".
 *   Defaults to "BSND".
 * @li layout_k: An optional string attribute. The layout of Key, "BSND", "TND" or "PA_BBND".
 *   Defaults to "BSND".
 * @li mask_mode: An optional int attribute. The mask mode.
 *   - 0: defaultMask
 *   - 3: rightDownCausal
 *   Defaults to 3.
 * @li quant_mode: An optional int attribute. The quantization mode of Query/Key.
 *   - -1: no quantization (default)
 *   - 0: FP8 per-token-head quantization (scale stored as float32)
 *   - 1: MX-FP8 quantization (scale stored as float8_e8m0)
 * @li return_value: An optional bool attribute. Whether to output sparse_values.
 *   - false: do not output (default)
 *   - true: output sparse_values
 * @li key_stride0: An optional int attribute. The stride (in element unit) of
 *   pool_key on axis 0, used for block address calculation when pool_key is
 *   non-contiguous on axis 0 in the PA_BBND scenario. It must be provided by the
 *   caller (e.g. read from at::Tensor::stride(0)) because the eager/aclnn path
 *   does not report the runtime stride. -1 means not specified (contiguous
 *   input, stride derived from shape). Defaults to -1.

 * @par Outputs:
 * @li sparse_indices: The expanded sparse token indices.
 *   - Data types: int32.
 * @li sparse_values: The pool scores of the selected top-k pools.
 *   - Data types: float32.

 * @attention Constraints:
 * @code{.c}
 *  - topk must be divisible by pool_size (topk % pool_size == 0).
 *  - pool_size supports [1, 128]; topk supports [1, 2048] and 3072/4096/5120/6144/7168/8192.
 *  - pool_tail_k value range is [0, pool_size - 1].
 *  - head_dim supports 128; query head num (N1) supports [1, 64]; pool_key head num (N2) supports 1.
 *  - block_size must be a multiple of 16 and no greater than 1024.
 *  - Padding invalid data on S1/S2 of query/pool_key is not supported.
 *  - pool_key supports non-contiguous input on axis 0 in the PA_BBND scenario
 *    (the actual stride must be passed via the key_stride0 attribute); all other
 *    axes must be contiguous. Other inputs do not support non-contiguous tensors.
 *  - Ascend950PR/Ascend950DT supports FP8 (float8_e4m3fn) and MX-FP8 (float8_e8m0 scale)
 *    data types and quantization.
 *  - Atlas A3 training/inference products and Atlas A2 training/inference products
 *    do not support FLOAT8_E4M3FN and FLOAT8_E8M0 data types, and do not support
 *    quantization.
 * @endcode
 */
REG_OP(PoolKeyIndexer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(pool_key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(pool_tail_k, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_q, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_k, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(q_descale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(k_descale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(sparse_indices, TensorType({DT_INT32}))
    .OUTPUT(sparse_values, TensorType({DT_FLOAT}))
    .ATTR(topk, Int, 2048)
    .ATTR(pool_size, Int, 16)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .ATTR(mask_mode, Int, 3)
    .ATTR(quant_mode, Int, -1)
    .ATTR(return_value, Bool, false)
    .ATTR(key_stride0, Int, -1)
    .ATTR(k_descale_stride0, Int, -1)
    .OP_END_FACTORY_REG(PoolKeyIndexer)
} // namespace ge

#endif // OPS_OP_PROTO_INC_PoolKeyIndexer_H_
