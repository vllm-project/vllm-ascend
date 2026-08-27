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
 * \file kv_quant_sparse_attn_sharedkv_metadata_proto.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_PROTO_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(KvQuantSparseAttnSharedkvMetadata)
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_ori_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_cmp_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_kv, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_kv, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(kv_quant_mode, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(max_seqlen_q, Int, 0)
    .ATTR(max_seqlen_kv, Int, 0)
    .ATTR(ori_topk, Int, 0)
    .ATTR(cmp_topk, Int, 0)
    .ATTR(tile_size, Int, 0)
    .ATTR(rope_head_dim, Int, 0)
    .ATTR(cmp_ratio, Int, -1)
    .ATTR(ori_mask_mode, Int, 4)
    .ATTR(cmp_mask_mode, Int, 3)
    .ATTR(ori_win_left, Int, 127)
    .ATTR(ori_win_right, Int, 0)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "PA_ND")
    .ATTR(has_ori_kv, Bool, true)
    .ATTR(has_cmp_kv, Bool, true)
    .REQUIRED_ATTR(soc_version, String)
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .OP_END_FACTORY_REG(KvQuantSparseAttnSharedkvMetadata)
} // namespace ge

#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_METADATA_PROTO_H
