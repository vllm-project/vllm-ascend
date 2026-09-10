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
 * \file sparse_flash_mla_metadata_proto.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_METADATA_PROTO_H
#define SPARSE_FLASH_MLA_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(SparseFlashMlaMetadata)
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_ori_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_cmp_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_ori_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_cmp_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cmp_residual_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(ori_topk_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cmp_topk_length, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_kv, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(max_seqlen_q, Int, 0)
    .ATTR(max_seqlen_ori_kv, Int, 0)
    .ATTR(max_seqlen_cmp_kv, Int, 0)
    .ATTR(ori_topk, Int, 0)
    .ATTR(cmp_topk, Int, 0)
    .ATTR(cmp_ratio, Int, 1)
    .ATTR(ori_mask_mode, Int, 0)
    .ATTR(cmp_mask_mode, Int, 0)
    .ATTR(ori_win_left, Int, -1)
    .ATTR(ori_win_right, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(has_ori_kv, Bool, false)
    .ATTR(has_cmp_kv, Bool, false)
    .REQUIRED_ATTR(soc_version, String)
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .OP_END_FACTORY_REG(SparseFlashMlaMetadata)

} // namespace ge

#endif // SPARSE_FLASH_MLA_METADATA_PROTO_H
