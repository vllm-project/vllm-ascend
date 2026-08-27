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
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "acl/acl.h"

#include "common/acl_utils.h"
#include "common/common_utils.h"
#include "common/io_utils.h"
#include "op_tiling/matmul_tiling_data.h"
#include "op_tiling/matmul_tiling_stub.h"

extern "C" void matmul_blaze_launch(
    aclrtStream stream,
    void* dA, void* dB, void* dScaleA, void* dScaleB, void* dC,
    const QuantMatmulTilingData tilingData,
    bool transA, bool transB);

int RunMatmulBlaze(uint64_t m, uint64_t k, uint64_t n, bool transA, bool transB)
{
    constexpr int32_t deviceId = 0;
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI_BASE = 2;

    QuantMatmulTilingData tilingData;
    MatmulTilingStub tilingEngine;
    tilingEngine.GetTilingData(m, n, k, transA, transB, tilingData);

    AclRtSession aclSession(deviceId);
    aclSession.Init();
    aclrtStream stream = aclSession.GetStream();

    uint64_t sizeA = m * k;
    uint64_t sizeB = k * n;
    uint64_t sizeScaleA =
        (m * CeilDiv(k, MXFP_DIVISOR) * MXFP_MULTI_BASE) * sizeof(uint8_t);
    uint64_t sizeScaleB =
        (n * CeilDiv(k, MXFP_DIVISOR) * MXFP_MULTI_BASE) * sizeof(uint8_t);
    uint64_t sizeC = m * n * sizeof(bfloat16_t);

    ExampleIoPaths paths = GetExampleIoPaths();

    uint8_t* hA = nullptr;
    uint8_t* hB = nullptr;
    uint8_t* hScaleA = nullptr;
    uint8_t* hScaleB = nullptr;
    uint8_t* hC = nullptr;

    CHECK_COND(aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "Failed to allocate host buffer for A.");
    std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hB, sizeB) == ACL_SUCCESS, "Failed to allocate host buffer for B.");
    std::unique_ptr<void, aclError (*)(void*)> hostB(hB, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hScaleA, sizeScaleA) == ACL_SUCCESS, "Failed to allocate host buffer for scaleA.");
    std::unique_ptr<void, aclError (*)(void*)> hostScaleA(hScaleA, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hScaleB, sizeScaleB) == ACL_SUCCESS, "Failed to allocate host buffer for scaleB.");
    std::unique_ptr<void, aclError (*)(void*)> hostScaleB(hScaleB, aclrtFreeHost);
    CHECK_COND(aclrtMallocHost((void**)&hC, sizeC) == ACL_SUCCESS, "Failed to allocate host buffer for C.");
    std::unique_ptr<void, aclError (*)(void*)> hostC(hC, aclrtFreeHost);

    CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "Failed to read input_a.bin.");
    CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "Failed to read input_b.bin.");
    CHECK_COND(ReadExactFile(paths.inputDir + "/input_scaleA.bin", hScaleA, sizeScaleA), "Failed to read input_scaleA.bin.");
    CHECK_COND(ReadExactFile(paths.inputDir + "/input_scaleB.bin", hScaleB, sizeScaleB), "Failed to read input_scaleB.bin.");

    void* dAPtr = nullptr;
    CHECK_COND(aclrtMalloc(&dAPtr, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "aclrtMalloc A failed.");
    std::unique_ptr<void, aclError (*)(void*)> deviceA(dAPtr, aclrtFree);

    void* dBPtr = nullptr;
    CHECK_COND(aclrtMalloc(&dBPtr, sizeB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "aclrtMalloc B failed.");
    std::unique_ptr<void, aclError (*)(void*)> deviceB(dBPtr, aclrtFree);

    void* dScaleAPtr = nullptr;
    CHECK_COND(aclrtMalloc(&dScaleAPtr, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "aclrtMalloc scaleA failed.");
    std::unique_ptr<void, aclError (*)(void*)> deviceScaleA(dScaleAPtr, aclrtFree);

    void* dScaleBPtr = nullptr;
    CHECK_COND(aclrtMalloc(&dScaleBPtr, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "aclrtMalloc scaleB failed.");
    std::unique_ptr<void, aclError (*)(void*)> deviceScaleB(dScaleBPtr, aclrtFree);

    void* dCPtr = nullptr;
    CHECK_COND(aclrtMalloc(&dCPtr, sizeC, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "aclrtMalloc C failed.");
    std::unique_ptr<void, aclError (*)(void*)> deviceC(dCPtr, aclrtFree);

    CHECK_COND(aclrtMemcpyAsync(dAPtr, sizeA, hA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "memcpy A H2D failed.");
    CHECK_COND(aclrtMemcpyAsync(dBPtr, sizeB, hB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "memcpy B H2D failed.");
    CHECK_COND(aclrtMemcpyAsync(dScaleAPtr, sizeScaleA, hScaleA, sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "memcpy scaleA H2D failed.");
    CHECK_COND(aclrtMemcpyAsync(dScaleBPtr, sizeScaleB, hScaleB, sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS, "memcpy scaleB H2D failed.");

    matmul_blaze_launch(
        stream,
        dAPtr, dBPtr, dScaleAPtr, dScaleBPtr, dCPtr,
        tilingData, transA, transB);

    CHECK_COND(aclrtMemcpyAsync(hC, sizeC, dCPtr, sizeC, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS, "memcpy C D2H failed.");
    CHECK_COND(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "stream sync failed.");

    CHECK_COND(WriteFile(paths.outputDir + "/npu_out.bin", hC, sizeC), "Failed to write npu_out.bin.");
    return 0;
}

int main(int argc, char* argv[])
{
    uint64_t m = 16;
    uint64_t k = 128;
    uint64_t n = 16384;
    bool transA = false;
    bool transB = true;

    try {
        if (argc >= 4) {
            m = ParsePositiveUint64(argv[1], "m");
            k = ParsePositiveUint64(argv[2], "k");
            n = ParsePositiveUint64(argv[3], "n");
        }
        if (argc >= 6) {
            transA = (std::string(argv[4]) == "true");
            transB = (std::string(argv[5]) == "true");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        return RunMatmulBlaze(m, k, n, transA, transB);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
