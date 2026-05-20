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
 * \file quant_lightning_indexer_metadata_proto.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_METADATA_PROTO_H
#define QUANT_LIGHTNING_INDEXER_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

 /**
  * @brief Function QuantLightningIndexerMetadata.

  * @par Inputs:
  * @li actual_seq_lengths_query: A matrix tensor. The type support int32.
  * Effective sequence length of q in different batches.
  * @li actual_seq_lengths_key: A matrix tensor. The type support int32.
  * Effective sequence length of key/value in different batches.

  * @par Attributes:
  * @li aic_core_num: An int. Cube core num of device.
  * @li aiv_core_num: An int. Vector core num of device.
  * @li soc_version: A string. Version of SOC.
  * @li num_heads_q: An int. Heads num of query.
  * @li num_heads_k: An int. Heads num of key.
  * @li head_dim: An Int. Dim of head.
  * @li query_quant_mode: An int. Mode of query quant.
  * @li key_quant_mode: An int. Mode of query quant.
  * @li batch_size: An int. Size of batch.
  * @li max_seqlen_q: An int. Max sequence length of query.
  * @li max_seqlen_k: An int. Max sequence length of key.
  * @li layout_query: A string. Layout of query.
  * @li layout_key: A string. Layout of key.
  * @li sparse_count: An int. Sparse count.
  * @li sparse_mode: An int. Mode of sparse.
  * @li pre_tokens: An int. Num of pretokens.
  * @li next_tokens: An int. Num of nexttokens.
  * @li cmp_ratio: An int. Ratio of compressor.

  * @par Outputs:
  * @li metadata: A matrix tensor. The type support int32.
  * The output of attention structure.
  */
REG_OP(QuantLightningIndexerMetadata)
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .REQUIRED_ATTR(soc_version, String)
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_k, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(query_quant_mode, Int)
    .REQUIRED_ATTR(key_quant_mode, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(max_seqlen_q, Int, 0)
    .ATTR(max_seqlen_k, Int, 0)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "BSND")
    .ATTR(sparse_count, Int, 2048)
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(cmp_ratio, Int, 1)
    .OP_END_FACTORY_REG(QuantLightningIndexerMetadata)
} // namespace ge

#endif // QUANT_LIGHTNING_INDEXER_METADATA_PROTO_H

// FD