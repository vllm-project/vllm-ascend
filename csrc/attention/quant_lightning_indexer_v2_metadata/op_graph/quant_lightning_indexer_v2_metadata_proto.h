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
 * \file quant_lightning_indexer_v2_metadata_proto.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_METADATA_PROTO_H
#define QUANT_LIGHTNING_INDEXER_V2_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(QuantLightningIndexerV2Metadata)
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cmp_residual_k, TensorType({DT_INT32}))
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
    .OP_END_FACTORY_REG(QuantLightningIndexerV2Metadata)
}  // namespace ge

#endif
