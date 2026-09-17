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
 * \file store_kv_block_metadata_aicpu.h
 * \brief AICPU kernel for StoreKvBlockMetadata: groups contiguous slot_mapping entries
 */

#ifndef STORE_KV_BLOCK_METADATA_AICPU_H
#define STORE_KV_BLOCK_METADATA_AICPU_H

#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

class StoreKvBlockMetadataCpuKernel : public CpuKernel {
public:
    StoreKvBlockMetadataCpuKernel() = default;
    ~StoreKvBlockMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool GenMetaData();

private:
    // input tensor
    Tensor *slotMapping_ = nullptr;
    Tensor *groupLen_ = nullptr;
    Tensor *groupKeyIdx_ = nullptr;
    Tensor *groupKeyCacheIdx_ = nullptr;
    // attribute
    int32_t blockSize_ = 0;

private:
    enum class ParamId : uint32_t {
        // input
        slotMapping = 0,
        groupLen = 1,
        groupKeyIdx = 2,
        groupKeyCacheIdx = 3,
    };
};

} // namespace aicpu

#endif // STORE_KV_BLOCK_METADATA_AICPU_H
