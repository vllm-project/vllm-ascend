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
 * \file test_aclnn_quant_lightning_indexer_v2.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2_metadata.h"
#include "aclnn/opdev/platform.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct TensorResources {
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *qScaleDeviceAddr = nullptr;
    void *kScaleDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *sparseValuesDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *weightsTensor = nullptr;
    aclTensor *qScaleTensor = nullptr;
    aclTensor *kScaleTensor = nullptr;
    aclTensor *metadataTensor = nullptr;
    aclTensor *sparseIndicesTensor = nullptr;
    aclTensor *sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources &resources)
{
    int64_t B = 2;
    int64_t S1 = 4;
    int64_t S2 = 8;
    int64_t N1 = 64;
    int64_t N2 = 1;
    int64_t D = 128;
    int64_t topk = 512;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> qScaleShape = {B, S1, N1};
    std::vector<int64_t> kScaleShape = {B, S2, N2};
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t qScaleShapeSize = GetShapeSize(qScaleShape);
    int64_t kScaleShapeSize = GetShapeSize(kScaleShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint8_t> queryHostData(queryShapeSize, 0x38);
    std::vector<uint8_t> keyHostData(keyShapeSize, 0x38);
    std::vector<float> weightsHostData(weightsShapeSize, 0.01f);
    std::vector<float> qScaleHostData(qScaleShapeSize, 1.0f);
    std::vector<float> kScaleHostData(kScaleShapeSize, 1.0f);
    std::vector<int32_t> metadataHostData(metadataShapeSize, 0);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<uint16_t> sparseValuesHostData(sparseValuesShapeSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                          &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(qScaleHostData, qScaleShape, &resources.qScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.qScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(kScaleHostData, kScaleShape, &resources.kScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.kScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(metadataHostData, metadataShape, &resources.metadataDeviceAddr, aclDataType::ACL_INT32,
                          &resources.metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_BF16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    return ACL_SUCCESS;
}

int GenerateMetadata(TensorResources &resources, aclrtStream stream, int64_t B, int64_t S1, int64_t S2, int64_t N1,
                     int64_t N2, int64_t D, int64_t topk, int64_t quantMode, int64_t maskMode, int64_t cmpRatio)
{
    constexpr const char layoutQ[] = "BSND";
    constexpr const char layoutK[] = "BSND";
    constexpr size_t layoutLen = sizeof(layoutQ);
    char layoutQCopy[layoutLen];
    char layoutKCopy[layoutLen];
    errno_t memcpyRet = memcpy_s(layoutQCopy, sizeof(layoutQCopy), layoutQ, layoutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("metadata memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKCopy, sizeof(layoutKCopy), layoutK, layoutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("metadata memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }

    aclOpExecutor *executor;
    uint64_t workspaceSize = 0;
    int ret = aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, N1, N2, D, topk, quantMode, B, S1, S2, layoutQCopy, layoutKCopy,
        maskMode, cmpRatio, resources.metadataTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    void *metadataWsAddr = nullptr;
    if (workspaceSize > 0ULL) {
        ret = aclrtMalloc(&metadataWsAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("metadata allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2Metadata(metadataWsAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2Metadata failed. ERROR: %d\n", ret);
        if (metadataWsAddr) {
            (void)aclrtFree(metadataWsAddr);
        }
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("metadata synchronize stream failed. ERROR: %d\n", ret);
        if (metadataWsAddr) {
            (void)aclrtFree(metadataWsAddr);
        }
        return ret;
    }

    if (metadataWsAddr) {
        (void)aclrtFree(metadataWsAddr);
    }
    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexerV2(TensorResources &resources, aclrtStream stream, void **workspaceAddr,
                                   uint64_t *workspaceSize)
{
    int64_t topk = 512;
    int64_t quantMode = 1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 1;
    constexpr const char layoutQStr[] = "BSND";
    constexpr const char layoutKStr[] = "BSND";
    constexpr size_t layoutQLen = sizeof(layoutQStr);
    constexpr size_t layoutKLen = sizeof(layoutKStr);
    char layoutQ[layoutQLen];
    char layoutK[layoutKLen];
    errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor *executor;

    int ret = aclnnQuantLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, resources.qScaleTensor,
        resources.kScaleTensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, resources.metadataTensor,
        topk, quantMode, -1, layoutQ, layoutK, maskMode, cmpRatio, returnValue, resources.sparseIndicesTensor,
        resources.sparseValuesTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(const std::vector<int64_t> &shape, void *deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    LOG_PRINT("sparse_indices result (first 10 elements):\n");
    for (int64_t i = 0; i < size && i < 10; i++) {
        LOG_PRINT("  [%ld] = %d\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources &resources, void *workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
        aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
        aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.qScaleTensor) {
        aclDestroyTensor(resources.qScaleTensor);
    }
    if (resources.kScaleTensor) {
        aclDestroyTensor(resources.kScaleTensor);
    }
    if (resources.metadataTensor) {
        aclDestroyTensor(resources.metadataTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
        aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
        aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.qScaleDeviceAddr) {
        aclrtFree(resources.qScaleDeviceAddr);
    }
    if (resources.kScaleDeviceAddr) {
        aclrtFree(resources.kScaleDeviceAddr);
    }
    if (resources.metadataDeviceAddr) {
        aclrtFree(resources.metadataDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    if (stream) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main()
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return 0;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 4;
    int64_t S2 = 8;
    int64_t N1 = 64;
    int64_t N2 = 1;
    int64_t D = 128;
    int64_t topk = 512;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = GenerateMetadata(resources, stream, B, S1, S2, N1, N2, D, topk, 1, 0, 1);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("GenerateMetadata failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteQuantLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, resources.sparseIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
