/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#include "data_utils.h"
#include "add_custom_tiling.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling);
#endif

int32_t main(int32_t argc, char* argv[])
{
    uint32_t totalLength = 8 * 2048;
    size_t inputByteSize = totalLength * sizeof(uint16_t);
    size_t outputByteSize = totalLength * sizeof(uint16_t);

#ifdef ASCENDC_CPU_DEBUG
    // Kirin暂不支持该编译分支
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_custom, blockDim, x, y, z);

    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
#else
    CHECK_ACL(aclInit("./scripts/acl.json"));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint32_t availableCoreNum = 1;

    AddTilingData tiling;
    tiling.totalLength = totalLength;
    uint32_t totalTiles = (totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
    uint32_t tilesPerCore = (totalTiles + availableCoreNum - 1) / availableCoreNum;
    uint32_t blockNum = tiling.blockNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    tiling.numPerCore = tilesPerCore * TILE_LENGTH;
    tiling.tailNumLastCore = totalLength - tiling.numPerCore * (blockNum - 1);

    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;
    uint8_t *tilingDevice = nullptr;

    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&tilingDevice, sizeof(AddTilingData), ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, sizeof(AddTilingData), &tiling, sizeof(AddTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    add_custom<<<blockNum, nullptr, stream>>>(xDevice, yDevice, zDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
