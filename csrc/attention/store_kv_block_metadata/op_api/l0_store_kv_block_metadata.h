/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_STORE_KV_BLOCK_METADATA_H
#define L0_STORE_KV_BLOCK_METADATA_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* StoreKvBlockMetadata(
    const aclTensor* slotMapping,
    const aclTensor* groupLen,
    const aclTensor* groupKeyIdx,
    const aclTensor* groupKeyCacheIdx,
    int64_t blockSize,
    aclOpExecutor* executor);
} // namespace l0op

#endif // L0_STORE_KV_BLOCK_METADATA_H
