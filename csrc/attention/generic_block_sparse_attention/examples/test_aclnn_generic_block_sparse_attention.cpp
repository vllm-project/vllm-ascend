/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_generic_block_sparse_attention.cpp
 * \brief GenericBlockSparseAttention + Metadata 算子调用示例
 *
 * Regular path: layoutQ=TND, layoutKv=PA_BBND, maskType=1, blockShape=[1,128], D=128.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

uint16_t FloatToFp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign << 15);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    }
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
}

float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (mant << 13);
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

void PrintOutResult(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultData[i]));
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
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

std::vector<uint16_t> MakeFp16Data(int64_t size, float value)
{
    std::vector<uint16_t> data(static_cast<size_t>(size), FloatToFp16(value));
    return data;
}

} // namespace

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Regular path smoke case: TND query + PA_BBND KV
    int64_t B = 1;
    int64_t S1 = 4;
    int64_t S2 = 256;
    int64_t N1 = 4;
    int64_t N2 = 1;
    int64_t D = 128;
    int64_t topK = 2;
    int64_t blockSize = 128;
    int64_t blockShapeX = 1;
    double scaleValue = 1.0 / sqrt(static_cast<double>(D));

    int64_t T = B * S1;
    int64_t maxBlocks = (S2 + blockSize - 1) / blockSize;
    int64_t totalQBlocks = T; // blockShapeX == 1

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> qShape = {T, N1, D};
    std::vector<int64_t> kvShape = {maxBlocks, blockSize, N2, D};
    std::vector<int64_t> sparseIdxShape = {N2, totalQBlocks, topK};
    std::vector<int64_t> sparseCountShape = {N2, totalQBlocks};
    std::vector<int64_t> blockTableShape = {B, maxBlocks};
    std::vector<int64_t> cuSeqQShape = {B + 1};
    std::vector<int64_t> sequsedKvShape = {B};
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> attnOutShape = {T, N1, D};

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *sparseIdxDeviceAddr = nullptr;
    void *sparseCountDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *cuSeqQDeviceAddr = nullptr;
    void *sequsedKvDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *sparseIdx = nullptr;
    aclTensor *sparseCount = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *cuSeqQ = nullptr;
    aclTensor *sequsedKv = nullptr;
    aclTensor *blockTable = nullptr;
    aclTensor *attnOut = nullptr;

    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);
    int64_t sparseIdxSize = GetShapeSize(sparseIdxShape);
    int64_t sparseCountSize = GetShapeSize(sparseCountShape);
    int64_t blockTableSize = GetShapeSize(blockTableShape);
    int64_t attnOutSize = GetShapeSize(attnOutShape);

    std::vector<uint16_t> qHostData = MakeFp16Data(qSize, 1.0f);
    std::vector<uint16_t> kHostData = MakeFp16Data(kvSize, 1.0f);
    std::vector<uint16_t> vHostData = MakeFp16Data(kvSize, 1.0f);
    std::vector<int32_t> sparseIdxHostData(sparseIdxSize, -1);
    std::vector<int32_t> sparseCountHostData(sparseCountSize, 0);
    std::vector<int32_t> blockTableHostData(blockTableSize);
    std::iota(blockTableHostData.begin(), blockTableHostData.end(), 0);
    std::vector<int64_t> cuSeqQHostData = {0, S1};
    std::vector<int32_t> sequsedKvHostData = {static_cast<int32_t>(S2)};
    std::vector<int32_t> metadataHostData(1024, 0);
    std::vector<uint16_t> attnOutHostData = MakeFp16Data(attnOutSize, 0.0f);

    // Causal-like sparse window: keep up to topK trailing KV blocks per Q token.
    int64_t history = S2 - S1;
    for (int64_t qBlock = 0; qBlock < totalQBlocks; qBlock++) {
        int64_t visible = history + qBlock + 1;
        int64_t lastKvBlock = std::min(maxBlocks - 1, (visible - 1) / blockSize);
        int64_t count = std::min(topK, lastKvBlock + 1);
        int64_t start = lastKvBlock - count + 1;
        for (int64_t i = 0; i < count; i++) {
            sparseIdxHostData[qBlock * topK + i] = static_cast<int32_t>(start + i);
        }
        sparseCountHostData[qBlock] = static_cast<int32_t>(count);
    }

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, kvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sparseIdxHostData, sparseIdxShape, &sparseIdxDeviceAddr, aclDataType::ACL_INT32, &sparseIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sparseCountHostData, sparseCountShape, &sparseCountDeviceAddr, aclDataType::ACL_INT32,
                          &sparseCount);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqQHostData, cuSeqQShape, &cuSeqQDeviceAddr, aclDataType::ACL_INT64, &cuSeqQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sequsedKvHostData, sequsedKvShape, &sequsedKvDeviceAddr, aclDataType::ACL_INT32, &sequsedKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &blockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t blockShapeData[] = {blockShapeX, blockSize};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    CHECK_RET(blockShape != nullptr, LOG_PRINT("aclCreateIntArray failed\n"); return -1);

    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";
    int64_t layoutSparse = 4; // KVN_TotalQB_KB

    // 3. 先调用 Metadata，再调用主算子
    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        sparseIdx, sparseCount, cuSeqQ, nullptr, nullptr, sequsedKv, S1, S2, N1, N2, D, blockShape, layoutQ, layoutKv,
        4, 1, 0, 1, -1, -1, 0, false, metadata, &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnGenericBlockSparseAttentionMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                   stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionMetadata failed. ERROR: %d\n", ret);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret);
              return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
        q, k, v, sparseIdx, sparseCount, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, cuSeqQ, nullptr,
        nullptr, sequsedKv, blockTable, blockShape, layoutQ, layoutKv, layoutSparse, scaleValue, 1, 0, 0.0, 1, -1, -1,
        0, 0, false, attnOut, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGenericBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnGenericBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttention failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(attnOutShape, &attnOutDeviceAddr);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyIntArray(blockShape);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(sparseIdx);
    aclDestroyTensor(sparseCount);
    aclDestroyTensor(metadata);
    aclDestroyTensor(cuSeqQ);
    aclDestroyTensor(sequsedKv);
    aclDestroyTensor(blockTable);
    aclDestroyTensor(attnOut);

    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(sparseIdxDeviceAddr);
    aclrtFree(sparseCountDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(cuSeqQDeviceAddr);
    aclrtFree(sequsedKvDeviceAddr);
    aclrtFree(blockTableDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    if (metadataWorkspaceSize > 0) {
        aclrtFree(metadataWorkspaceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
