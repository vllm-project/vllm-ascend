/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_nsa_compress.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_gated_delta_rule.h"

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
        if (i >= 5) { // print the first five data
            break;
        }
        LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // AscendCL初始化
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
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1.device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *gamaDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *stateRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *ssmStaIdDeviceAddr = nullptr;
    void *numAccTokDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;

    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *gama = nullptr;
    aclTensor *gamak = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *stateRef = nullptr;
    aclTensor *actSeqLen = nullptr;
    aclTensor *ssmStaId = nullptr;
    aclTensor *numAccTok = nullptr;
    aclTensor *attnOut = nullptr;

    // 自定义输入与属性
    int32_t batchSize = 2;
    int32_t mtp = 2;
    int32_t headKNum = 4;
    int32_t headVNum = 8;
    int32_t dimV = 32;
    int32_t dimK = 32;


    std::vector<int64_t> stateShape = {batchSize * mtp, headVNum, dimV, dimK};
    std::vector<int64_t> qkShape = {batchSize * mtp, headKNum, dimK};
    std::vector<int64_t> vShape = {batchSize * mtp, headVNum, dimV};
    std::vector<int64_t> gamaShape = {batchSize * mtp, headVNum};
    std::vector<int64_t> actSeqLenShape = {batchSize};
    std::vector<int64_t> ssmStaIdShape = {batchSize * mtp};
    std::vector<int16_t> stateRefHostData(GetShapeSize(stateShape));
    std::vector<int16_t> queryHostData(GetShapeSize(qkShape));
    std::vector<int16_t> keyHostData(GetShapeSize(qkShape));
    std::vector<int16_t> valueHostData(GetShapeSize(vShape));
    std::vector<float> gamaHostData(GetShapeSize(gamaShape));
    std::vector<int16_t> betaHostData(GetShapeSize(gamaShape));
    std::vector<int32_t> actSeqLenHostData(batchSize, mtp);
    std::vector<int32_t> ssmStaIdHostData(batchSize * mtp);
    std::vector<int32_t> numAccTokHostData(batchSize, 1);
    for (int i = 0; i < stateRefHostData.size(); i++) {
        stateRefHostData[i] = 1;
    }
    for (int i = 0; i < queryHostData.size(); i++) {
        queryHostData[i] = 1;
    }
    for (int i = 0; i < keyHostData.size(); i++) {
        keyHostData[i] = 1;
    }
    for (int i = 0; i < valueHostData.size(); i++) {
        valueHostData[i] = 0;
    }
    for (int i = 0; i < betaHostData.size(); i++) {
        betaHostData[i] = 0;
    }
    for (int i = 0; i < ssmStaIdHostData.size(); i++) {
        ssmStaIdHostData[i] = i;
    }

    std::vector<int16_t> attnOutHostData(valueHostData);

    ret = CreateAclTensor(stateRefHostData, stateShape, &stateRefDeviceAddr, aclDataType::ACL_BF16, &stateRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, qkShape, &queryDeviceAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkShape, &keyDeviceAddr, aclDataType::ACL_BF16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, vShape, &valueDeviceAddr, aclDataType::ACL_BF16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gamaHostData, gamaShape, &gamaDeviceAddr, aclDataType::ACL_FLOAT, &gama);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gamaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqLenHostData, actSeqLenShape, &actSeqLenDeviceAddr, aclDataType::ACL_INT32, &actSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ssmStaIdHostData, ssmStaIdShape, &ssmStaIdDeviceAddr, aclDataType::ACL_INT32, &ssmStaId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numAccTokHostData, actSeqLenShape, &numAccTokDeviceAddr, aclDataType::ACL_INT32, &numAccTok);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, vShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    float scale = 1.0;
    aclOpExecutor *executor;
    // 调用aclnnRecurrentGatedDeltaRuleGetWorkspaceSize第一段接口
    ret = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(query, key, value, beta, stateRef, actSeqLen, ssmStaId, gama,
                                                       gamak, numAccTok, scale, attnOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnRecurrentGatedDeltaRule第二段接口
    ret = aclnnRecurrentGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRecurrentGatedDeltaRule failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(stateShape, &stateRefDeviceAddr);
    PrintOutResult(vShape, &attnOutDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(gama);
    aclDestroyTensor(beta);
    aclDestroyTensor(stateRef);
    aclDestroyTensor(actSeqLen);
    aclDestroyTensor(ssmStaId);
    aclDestroyTensor(numAccTok);
    aclDestroyTensor(attnOut);

    // 7. 释放device资源
    aclrtFree(query);
    aclrtFree(key);
    aclrtFree(value);
    aclrtFree(gama);
    aclrtFree(beta);
    aclrtFree(stateRef);
    aclrtFree(actSeqLen);
    aclrtFree(ssmStaId);
    aclrtFree(numAccTok);
    aclrtFree(attnOut);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}