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
 * \file l0_store_kv_block_metadata.cpp
 * \brief L0 interface for StoreKvBlockMetadata, adds AICPU task to launcher list
 */

#include "l0_store_kv_block_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(StoreKvBlockMetadata);

const aclTensor *StoreKvBlockMetadata(
    const aclTensor *slotMapping,
    int64_t blockSize,
    const aclTensor *groupLen,
    const aclTensor *groupKeyIdx,
    const aclTensor *groupKeyCacheIdx,
    aclOpExecutor *executor)
{
    L0_DFX(StoreKvBlockMetadata, slotMapping, blockSize, groupLen, groupKeyIdx, groupKeyCacheIdx);

    static internal::AicpuTaskSpace space("StoreKvBlockMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        StoreKvBlockMetadata,
        OP_ATTR_NAMES({"block_size"}),
        OP_INPUT(slotMapping),
        OP_OUTPUT(groupLen, groupKeyIdx, groupKeyCacheIdx),
        OP_ATTR(blockSize));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "StoreKvBlockMetadata"
                                              " ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return groupLen;
}

} // namespace l0op
