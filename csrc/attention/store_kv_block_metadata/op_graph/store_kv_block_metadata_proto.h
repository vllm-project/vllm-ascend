/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file store_kv_block_metadata_proto.h
 * \brief Operator registration for StoreKvBlockMetadata
 */
#ifndef STORE_KV_BLOCK_METADATA_PROTO_H
#define STORE_KV_BLOCK_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(StoreKvBlockMetadata)
    .INPUT(slot_mapping, TensorType({DT_INT32}))
    .INPUT(group_len, TensorType({DT_INT32}))
    .INPUT(group_key_idx, TensorType({DT_INT32}))
    .INPUT(group_key_cache_idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(StoreKvBlockMetadata)

} // namespace ge

#endif // STORE_KV_BLOCK_METADATA_PROTO_H
