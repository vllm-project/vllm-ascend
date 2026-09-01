/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

// ⚠️ Stream 同步反模式（详见 torch-ascendc-op-extension SKILL.md「Stream 同步模式」和 references/anti_patterns.md）：
//   ❌ stream(false) + 函数调用 → 乱序：不清 queue，kernel 先于之前操作执行
//   ❌ lambda 内传 NPUStream + OpCommand → 死锁：queue 等 lambda，lambda 等 queue 空
//   ❌ zeros_like 创建输出 → 乱序：zeros_like 入 queue 但 kernel 不入 queue，用 empty_like
//   ✅ stream(true) + 函数调用 → 清 queue，安全（本文件使用此模式）

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "../op_kernel/add_custom_tiling.h"

// [MODIFY] extern 声明 kernel 入口 - add_custom → <your_op>，保持 _kernel 后缀
// 从 add_custom_kernel.asc 中的核函数签名复制，去掉 __global__ __vector__，加 extern "C"
extern "C" void add_custom_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                   uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *tiling);

// [MODIFY] 算子 PyTorch 实现 - add_custom → <your_op>，保持 _torch 后缀
namespace ascend_kernel {

at::Tensor add_custom_torch(const at::Tensor& x1, const at::Tensor& x2)
{
    TORCH_CHECK(x1.scalar_type() == at::kFloat, "only FP32 supported");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "x1 and x2 must have the same dtype");
    TORCH_CHECK(x1.is_privateuseone(), "x1 must be on NPU");
    TORCH_CHECK(x2.is_privateuseone(), "x2 must be on NPU");

    at::Tensor y = at::empty_like(x1);

    int64_t totalElements = x1.numel();
    TORCH_CHECK(totalElements > 0, "input tensors must not be empty");

    // stream(true) 在返回 ACL stream 前会清 queue，确保与之前 NPU 操作的正确同步
    // 禁止使用 stream(false)：不清 queue + 直接调用 kernel = 乱序风险
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // [MODIFY] 查询核数 - 从 main() 中复制的 Tiling 计算逻辑
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t availableCoreNum = 0;
    auto ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    TORCH_CHECK(ret == ACL_SUCCESS && availableCoreNum > 0, "failed to get NPU core count");

    // [MODIFY] Tiling 计算 - 与 main() 中的逻辑一致
    AddTilingData tiling;
    tiling.totalLength = totalElements;
    uint64_t totalTiles = (totalElements + TILE_LENGTH - 1) / TILE_LENGTH;
    uint64_t tilesPerCore = (totalTiles + availableCoreNum - 1) / availableCoreNum;
    tiling.blockNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    tiling.numPerCore = tilesPerCore * TILE_LENGTH;
    tiling.tailNumLastCore = totalElements - tiling.numPerCore * (tiling.blockNum - 1);

    uint32_t blockNum = static_cast<uint32_t>(tiling.blockNum);

    // Tiling 数据搬到 device
    at::Tensor tilingTensor = at::empty(
        {static_cast<int64_t>(sizeof(AddTilingData))},
        x1.options().dtype(at::kByte));
    aclrtMemcpy(tilingTensor.mutable_data_ptr(), sizeof(AddTilingData),
        &tiling, sizeof(AddTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

    // [MODIFY] 调用 kernel - add_custom → <your_op>，保持 _kernel 后缀
    add_custom_kernel(blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x1.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(x2.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));

    return y;
}

} // namespace ascend_kernel
