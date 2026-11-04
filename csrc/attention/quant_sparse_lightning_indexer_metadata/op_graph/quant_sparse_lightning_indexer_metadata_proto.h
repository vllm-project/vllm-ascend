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
 * \file quant_sparse_lightning_indexer_metadata_proto.h
 * \brief
 */
#ifndef QUANT_SPARSE_LIGHTNING_INDEXER_METADATA_PROTO_H
#define QUANT_SPARSE_LIGHTNING_INDEXER_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(VllmAscendQuantSparseLightningIndexerMetadata)
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cmp_residual_k, TensorType({DT_INT32}))
    // candidate_block_length（每行有效候选块数，DSL 的 candidate_block_length 语义）：
    // QSLI 调度的**必选**输入 —— 工作量按 max(L)×8 token、候选 tile=256 计算（与主算子
    // V1 的候选驱动路径一致）。注意：该 metadata 是 QSLI 私有语义，不能与 QLI metadata 混用。
    .REQUIRED_INPUT(candidate_block_length, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_k, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(topk, Int)
    .REQUIRED_ATTR(quant_mode, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(max_seqlen_q, Int, -1)
    .ATTR(max_seqlen_k, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .ATTR(mask_mode, Int, 0)
    .ATTR(cmp_ratio, Int, 1)
    .ATTR(aic_core_num, Int, 0)
    .ATTR(aiv_core_num, Int, 0)
    .ATTR(soc_version, String, "")
    .OP_END_FACTORY_REG(VllmAscendQuantSparseLightningIndexerMetadata)
}  // namespace ge

#endif
