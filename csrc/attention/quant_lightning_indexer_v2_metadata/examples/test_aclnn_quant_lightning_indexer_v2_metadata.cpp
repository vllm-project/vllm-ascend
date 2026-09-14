/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_quant_lightning_indexer_v2_metadata.cpp
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...) \
    do { \
        if (!(cond)) { \
            printf(fmt "\n", ##__VA_ARGS__); \
            return (ret_val); \
        } \
    } while (0)

// 参考 quant_lightning_indexer_v2_metadata.h
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t QLI_V2_METADATA_TOTAL_SIZE = 1024;
constexpr uint32_t QLI_V2_METADATA_SIZE = 8;
constexpr uint32_t QLD_V2_METADATA_SIZE = 8;

// QLI Metadata Index Definitions
constexpr uint32_t QLI_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t QLI_V2_BN2_START_INDEX = 1;
constexpr uint32_t QLI_V2_M_START_INDEX = 2;
constexpr uint32_t QLI_V2_S2_START_INDEX = 3;
constexpr uint32_t QLI_V2_BN2_END_INDEX = 4;
constexpr uint32_t QLI_V2_M_END_INDEX = 5;
constexpr uint32_t QLI_V2_S2_END_INDEX = 6;
constexpr uint32_t QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// QLD Metadata Index Definitions
constexpr uint32_t QLD_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t QLD_V2_BN2_IDX_INDEX = 1;
constexpr uint32_t QLD_V2_M_IDX_INDEX = 2;
constexpr uint32_t QLD_V2_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t QLD_V2_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t QLD_V2_M_START_INDEX = 5;
constexpr uint32_t QLD_V2_M_NUM_INDEX = 6;

struct QliV2Metadata {
    uint32_t faData[AIC_CORE_MAX_NUM][QLI_V2_METADATA_SIZE];
    uint32_t fdData[AIV_CORE_MAX_NUM][QLD_V2_METADATA_SIZE];
};

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> onExitScope)
        : m_exitFunc(std::move(onExitScope)),
          m_isDismissed(false)
    {}
    // 禁止拷贝
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr{nullptr};
    void *deviceAddr{nullptr};
    aclTensor *data{nullptr};
};

struct ArgScenario {
    bool hasCuSeq{false};
    bool hasSeqused{false};
};

struct ArgContext {
    // required input
    int64_t numHeadsQ{0};
    int64_t numHeadsK{0};
    int64_t headDim{0};
    int64_t topk{0};
    int64_t quantMode{2};
    // optional input
    Tensor cuSeqlensQOptional{};
    Tensor cuSeqlensKOptional{};
    Tensor sequsedQOptional{};
    Tensor sequsedKOptional{};
    Tensor cmpResidualKOptional{};
    int64_t batchSize{0};
    int64_t maxSeqlenQ{0};
    int64_t maxSeqlenK{0};
    char *layoutQOptional{nullptr};
    char *layoutKOptional{nullptr};
    int64_t maskMode{0};
    int64_t cmpRatio{0};
    // output
    Tensor metadata{};
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMallocHost申请host侧内存
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // 调用aclCreateTensor接口创建aclTensor
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), tensor.deviceAddr);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensKOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKOptional != nullptr) {
        free(context.layoutKOptional);
        context.layoutKOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;
    int64_t batchSize = 4;

    context.numHeadsQ = 1;
    context.numHeadsK = 1;
    context.headDim = 128;
    context.topk = 0;
    context.quantMode = 2; // 2: per-token-head / 3: group-scaling
    ret = CreateTensor(aclDataType::ACL_INT32, {QLI_V2_METADATA_TOTAL_SIZE}, context.metadata); // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create meta failed. Error: %d", ret);

    context.maskMode = 0; // 0: no mask, 3: causal
    context.cmpRatio = 1; // [1, 128], 1: no compress
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, "BSND"); // BSND,TND
    strcpy(context.layoutKOptional, "BSND"); // BSND,TND,PA_BBND

    if (!scenario.hasCuSeq && !scenario.hasSeqused) {
        context.batchSize = batchSize;
        context.maxSeqlenK = 1024;
        context.maxSeqlenQ = 1024;
        argsGuard.Dismiss();
        return ACL_SUCCESS;
    }

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, {batchSize + 1}, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {batchSize + 1}, context.cuSeqlensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensKOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, {batchSize}, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, {batchSize}, context.sequsedKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedKOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario{};
    scenario.hasCuSeq = true;
    scenario.hasSeqused = true;
    ArgContext context{};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnLightningIndexerV2Metadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensKOptional.data, context.sequsedQOptional.data,
        context.sequsedKOptional.data, context.cmpResidualKOptional.data, context.numHeadsQ, context.numHeadsK,
        context.headDim, context.topk, context.quantMode, context.batchSize, context.maxSeqlenQ, context.maxSeqlenK,
        context.layoutQOptional, context.layoutKOptional, context.maskMode, context.cmpRatio, context.metadata.data,
        &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n",
                  ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // 调用aclnnLightningIndexerV2Metadata第二段接口
    ret = aclnnQuantLightningIndexerV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantLightningIndexerV2Metadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    QliV2Metadata result{};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_MAX_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faData[i][QLI_V2_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faData[i][QLI_V2_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faData[i][QLI_V2_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faData[i][QLI_V2_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faData[i][QLI_V2_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faData[i][QLI_V2_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faData[i][QLI_V2_S2_END_INDEX]);
        printf("    First Workspace Index : %u\n", result.faData[i][QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdData[i][QLD_V2_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdData[i][QLD_V2_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdData[i][QLD_V2_M_IDX_INDEX]);
        printf("    FD Task Workspace Idx   : %u\n", result.fdData[i][QLD_V2_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdData[i][QLD_V2_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdData[i][QLD_V2_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdData[i][QLD_V2_M_NUM_INDEX]);
    }

    return 0;
}
