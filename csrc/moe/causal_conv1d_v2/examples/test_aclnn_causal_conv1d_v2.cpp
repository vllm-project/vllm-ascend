/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file test_aclnn_causal_conv1d_v2.cpp
 * \brief Example of invoking the aclnnCausalConv1dV2 operator.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_v2.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        if (i >= 5) {
            break;
        }
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *convStatesDeviceAddr = nullptr;
    void *queryStartLocDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *convStates = nullptr;
    aclTensor *queryStartLoc = nullptr;
    aclTensor *y = nullptr;

    int32_t batchSize = 2;
    int32_t seqLen = 4;
    int32_t dim = 64;
    int32_t width = 3;
    int32_t stateLen = width;

    std::vector<int64_t> xShape = {batchSize * seqLen, dim};
    std::vector<int64_t> weightShape = {width, dim};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> convStatesShape = {batchSize, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batchSize + 1};
    std::vector<int64_t> yShape = {batchSize * seqLen, dim};

    std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int16_t> weightHostData(GetShapeSize(weightShape), 1);
    std::vector<int16_t> biasHostData(GetShapeSize(biasShape), 0);
    std::vector<int16_t> convStatesHostData(GetShapeSize(convStatesShape), 0);
    std::vector<int32_t> queryStartLocHostData = {0, seqLen, batchSize * seqLen};
    std::vector<int16_t> yHostData(GetShapeSize(yShape), 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(convStatesHostData, convStatesShape, &convStatesDeviceAddr, aclDataType::ACL_BF16, &convStates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryStartLocHostData, queryStartLocShape, &queryStartLocDeviceAddr,
                          aclDataType::ACL_INT32, &queryStartLoc);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    char activation[] = "none";
    int64_t padSlotId = -1;
    int64_t nullBlockId = -1;
    int64_t runMode = 0;
    int64_t headNum = 0;
    int64_t maxQueryLen = -1;

    ret = aclnnCausalConv1dV2GetWorkspaceSize(x, weight, bias, convStates, queryStartLoc, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr, activation, padSlotId, nullBlockId,
                                            runMode, headNum, maxQueryLen, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCausalConv1dV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dV2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(yShape, &yDeviceAddr);

    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(convStates);
    aclDestroyTensor(queryStartLoc);
    aclDestroyTensor(y);

    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(convStatesDeviceAddr);
    aclrtFree(queryStartLocDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
