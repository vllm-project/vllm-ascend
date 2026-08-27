/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "matmul_tiling_data.h"
#include "matmul_tiling_stub.h"

extern "C" void matmul_blaze_launch(
    aclrtStream stream,
    void* dA, void* dB, void* dScaleA, void* dScaleB, void* dC,
    const QuantMatmulTilingData tilingData,
    bool transA, bool transB);

namespace ascend_kernel {

at::Tensor matmul_blaze_torch(const at::Tensor& a, const at::Tensor& b,
                               const at::Tensor& scaleA, const at::Tensor& scaleB,
                               bool transA, bool transB)
{
    TORCH_CHECK(a.is_privateuseone(), "a must be on NPU");
    TORCH_CHECK(b.is_privateuseone(), "b must be on NPU");

    int64_t m = transA ? a.size(1) : a.size(0);
    int64_t k = transA ? a.size(0) : a.size(1);
    int64_t n = transB ? b.size(0) : b.size(1);

    at::Tensor c = at::empty({m, n}, a.options().dtype(at::kBFloat16));

    QuantMatmulTilingData tilingData;
    MatmulTilingStub tilingStub;
    tilingStub.GetTilingData(m, n, k, transA, transB, tilingData);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    matmul_blaze_launch(
        aclStream,
        a.data_ptr(), b.data_ptr(),
        scaleA.data_ptr(), scaleB.data_ptr(),
        c.data_ptr(),
        tilingData, transA, transB);

    return c;
}

} // namespace ascend_kernel
