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
 * \file store_kv_block_metadata_aicpu.cpp
 * \brief AICPU kernel implementation for StoreKvBlockMetadata
 *
 * Ports the logic from store_kv_block_pre: groups contiguous slot_mapping entries
 * that belong to the same block, producing group_len / group_key_idx / group_key_cache_idx.
 */

#include "log.h"
#include "status.h"
#include <cstring>
#include "store_kv_block_metadata_aicpu.h"

namespace aicpu {

uint32_t StoreKvBlockMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return GenMetaData() ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool StoreKvBlockMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    slotMapping_ = ctx.Input(static_cast<uint32_t>(ParamId::slotMapping));
    // outputs
    groupLen_ = ctx.Output(static_cast<uint32_t>(ParamId::groupLen));
    groupKeyIdx_ = ctx.Output(static_cast<uint32_t>(ParamId::groupKeyIdx));
    groupKeyCacheIdx_ = ctx.Output(static_cast<uint32_t>(ParamId::groupKeyCacheIdx));

    // attribute
    auto attr = ctx.GetAttr("block_size");
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr block_size is null");
        return false;
    }
    blockSize_ = static_cast<int32_t>(attr->GetInt());
    if (blockSize_ <= 0) {
        KERNEL_LOG_ERROR("block_size must be positive, got %d", blockSize_);
        return false;
    }
    return true;
}

bool StoreKvBlockMetadataCpuKernel::GenMetaData()
{
    if (slotMapping_ == nullptr || slotMapping_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("slot_mapping is empty");
        return false;
    }
    if (groupLen_ == nullptr || groupLen_->GetData() == nullptr ||
        groupKeyIdx_ == nullptr || groupKeyIdx_->GetData() == nullptr ||
        groupKeyCacheIdx_ == nullptr || groupKeyCacheIdx_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("output tensor is empty");
        return false;
    }

    int32_t *slotMappingData = static_cast<int32_t *>(slotMapping_->GetData());
    int32_t *groupLenData = static_cast<int32_t *>(groupLen_->GetData());
    int32_t *groupKeyIdxData = static_cast<int32_t *>(groupKeyIdx_->GetData());
    int32_t *groupKeyCacheIdxData = static_cast<int32_t *>(groupKeyCacheIdx_->GetData());

    // total elements in slot_mapping (1-D tensor)
    int64_t slotMappingLen = slotMapping_->GetTensorShape()->GetDimSize(0);

    // total capacity of output tensors (1-D, same shape as input)
    int64_t outCapacity = groupLen_->GetTensorShape()->GetDimSize(0);

    int32_t idxSlotmap = 0;
    int32_t idxGroups = 0;

    while (idxSlotmap < slotMappingLen) {
        // Skip dirty values (negative slots)
        int32_t cacheSlot = slotMappingData[idxSlotmap];
        if (cacheSlot < 0) {
            idxSlotmap++;
            continue;
        }

        int32_t blockId = cacheSlot / blockSize_;

        // Record group start: source index and destination cache index
        groupKeyIdxData[idxGroups] = idxSlotmap;
        groupKeyCacheIdxData[idxGroups] = cacheSlot;

        // Find the end of consecutive slots within the same block
        int32_t groupEndIdx = idxSlotmap;
        while (groupEndIdx + 1 < slotMappingLen
               && slotMappingData[groupEndIdx + 1] / blockSize_ == blockId
               && slotMappingData[groupEndIdx + 1] == slotMappingData[groupEndIdx] + 1) {
            groupEndIdx++;
        }
        groupEndIdx++;

        groupLenData[idxGroups] = groupEndIdx - idxSlotmap;
        printf("[StoreKvBlockMetadata][DBG] group[%d] slotStart=%d len=%d blockId=%d\n",
               idxGroups, idxSlotmap, groupEndIdx - idxSlotmap, blockId);
        fflush(stdout);
        idxSlotmap = groupEndIdx;
        idxGroups++;
    }

    // 0 fill the remaining output entries. store_kv_block kernel reads groupLen as uint32_t,
    // so negative fillers would be interpreted as huge positive values and bypass the
    // `groupLen <= 0` guard, causing out-of-range MTE writes. Use 0 so that guard works.
    if (idxGroups < outCapacity) {
        std::memset(groupLenData + idxGroups, 0,
                    static_cast<size_t>(outCapacity - idxGroups) * sizeof(int32_t));
        std::memset(groupKeyIdxData + idxGroups, 0,
                    static_cast<size_t>(outCapacity - idxGroups) * sizeof(int32_t));
        std::memset(groupKeyCacheIdxData + idxGroups, 0,
                    static_cast<size_t>(outCapacity - idxGroups) * sizeof(int32_t));
    }

    return true;
}

namespace {
static const char *kernelType = "StoreKvBlockMetadata";
REGISTER_CPU_KERNEL(kernelType, StoreKvBlockMetadataCpuKernel);
} // namespace

} // namespace aicpu
