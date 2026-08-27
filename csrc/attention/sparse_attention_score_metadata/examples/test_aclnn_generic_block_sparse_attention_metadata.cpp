/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/**
 * @file test_aclnn_generic_block_sparse_attention_metadata.cpp
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

#include "../sparse_attention_score_metadata.h"

#define CHECK_LOG_RET(condition, returnValue, format, ...)                                                             \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::printf(format "\n", ##__VA_ARGS__);                                                                   \
            return (returnValue);                                                                                      \
        }                                                                                                              \
    } while (0)

namespace {

namespace metadata_protocol = optiling::generic_block_sparse_attention_metadata;

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> callback) : callback_(std::move(callback))
    {
    }
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;
    ~ScopeGuard()
    {
        callback_();
    }

private:
    std::function<void()> callback_;
};

struct Tensor {
    void *hostAddr = nullptr;
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
    size_t byteSize = 0U;
};

struct CaseContext {
    std::string name;
    Tensor sparseBlockIdx;
    Tensor sparseBlockCount;
    Tensor cuSeqLengths;
    Tensor cuSeqLengthsKv;
    Tensor seqUsedQ;
    Tensor seqUsedKv;
    Tensor metadata;
    aclIntArray *blockShape = nullptr;
    int64_t maxQSeqLen = 0;
    int64_t maxKvSeqLen = 0;
    int64_t numQHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headDim = 0;
    const char *qInputLayout = nullptr;
    const char *kvInputLayout = nullptr;
    int32_t expectedSaTotalTaskNum = 0;
    bool expectDecodeSchedule = false;
};

int64_t GetElementNum(const std::vector<int64_t> &shape)
{
    int64_t elementNum = 1;
    for (const int64_t dim : shape) {
        elementNum *= dim;
    }
    return elementNum;
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, const void *hostData, Tensor &tensor)
{
    tensor.byteSize = static_cast<size_t>(GetElementNum(shape)) * aclDataTypeSize(dataType);
    aclError ret = aclrtMallocHost(&tensor.hostAddr, tensor.byteSize);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed, error: %d", ret);
    if (hostData == nullptr) {
        std::memset(tensor.hostAddr, 0, tensor.byteSize);
    } else {
        std::memcpy(tensor.hostAddr, hostData, tensor.byteSize);
    }

    ret = aclrtMalloc(&tensor.deviceAddr, tensor.byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed, error: %d", ret);
    tensor.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                                    shape.size(), tensor.deviceAddr);
    CHECK_LOG_RET(tensor.tensor != nullptr, ACL_ERROR_BAD_ALLOC, "aclCreateTensor failed");

    ret = aclrtMemcpy(tensor.deviceAddr, tensor.byteSize, tensor.hostAddr, tensor.byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "copy tensor to device failed, error: %d", ret);
    return ACL_SUCCESS;
}

template <typename T>
aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, const std::vector<T> &hostData,
                         Tensor &tensor)
{
    const int64_t elementNum = GetElementNum(shape);
    CHECK_LOG_RET(elementNum == static_cast<int64_t>(hostData.size()), ACL_ERROR_INVALID_PARAM,
                  "tensor shape and host data size do not match");
    return CreateTensor(dataType, shape, hostData.data(), tensor);
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.tensor != nullptr) {
        aclDestroyTensor(tensor.tensor);
        tensor.tensor = nullptr;
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

void DestroyCase(CaseContext &context)
{
    DestroyTensor(context.sparseBlockIdx);
    DestroyTensor(context.sparseBlockCount);
    DestroyTensor(context.cuSeqLengths);
    DestroyTensor(context.cuSeqLengthsKv);
    DestroyTensor(context.seqUsedQ);
    DestroyTensor(context.seqUsedKv);
    DestroyTensor(context.metadata);
    if (context.blockShape != nullptr) {
        aclDestroyIntArray(context.blockShape);
        context.blockShape = nullptr;
    }
}

std::vector<int32_t> MakeSparseBlockIndices(const std::vector<int32_t> &blockCounts, int64_t capacity)
{
    std::vector<int32_t> indices(blockCounts.size() * static_cast<size_t>(capacity), -1);
    for (size_t task = 0; task < blockCounts.size(); ++task) {
        for (int32_t block = 0; block < blockCounts[task]; ++block) {
            indices[task * static_cast<size_t>(capacity) + static_cast<size_t>(block)] = block;
        }
    }
    return indices;
}

aclnnStatus CreateCommonArgs(CaseContext &context)
{
    const std::vector<int64_t> blockShapeData = {1, 128};
    context.blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_LOG_RET(context.blockShape != nullptr, ACL_ERROR_BAD_ALLOC, "aclCreateIntArray failed");
    return CreateTensor(ACL_INT32, {metadata_protocol::METADATA_TOTAL_SIZE}, nullptr, context.metadata);
}

aclnnStatus CreateBsndDecodeCase(CaseContext &context)
{
    context.name = "BSND decode schedule";
    context.maxQSeqLen = 2;
    context.maxKvSeqLen = 2048;
    context.numQHeads = 4;
    context.numKvHeads = 1;
    context.headDim = 128;
    context.qInputLayout = "BSND";
    context.kvInputLayout = "BSND";
    context.expectedSaTotalTaskNum = 2;
    context.expectDecodeSchedule = true;

    const std::vector<int32_t> blockCounts = {12, 3};
    const std::vector<int32_t> blockIndices = MakeSparseBlockIndices(blockCounts, 12);
    aclnnStatus ret = CreateTensor(ACL_INT32, {1, 1, 2, 12}, blockIndices, context.sparseBlockIdx);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create BSND sparseBlockIdx failed, error: %d", ret);
    ret = CreateTensor(ACL_INT32, {1, 1, 2}, blockCounts, context.sparseBlockCount);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create BSND sparseBlockCount failed, error: %d", ret);
    const std::vector<int32_t> seqUsedQ = {2};
    ret = CreateTensor(ACL_INT32, {1}, seqUsedQ, context.seqUsedQ);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create BSND seqUsedQ failed, error: %d", ret);
    return CreateCommonArgs(context);
}

aclnnStatus CreateTndSeqUsedCase(CaseContext &context)
{
    context.name = "TND cuSeqLengths and seqUsedQ";
    context.maxQSeqLen = 4;
    context.maxKvSeqLen = 2048;
    context.numQHeads = 4;
    context.numKvHeads = 1;
    context.headDim = 128;
    context.qInputLayout = "TND";
    context.kvInputLayout = "BSND";
    context.expectedSaTotalTaskNum = 5;

    // The two batches occupy physical Q ranges [0, 4) and [4, 8). seqUsedQ selects [0, 2) and [4, 7).
    const std::vector<int64_t> cuSeqLengths = {0, 4, 8};
    const std::vector<int32_t> seqUsedQ = {2, 3};
    const std::vector<int32_t> blockCounts = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int32_t> blockIndices = MakeSparseBlockIndices(blockCounts, 8);
    aclnnStatus ret = CreateTensor(ACL_INT32, {1, 8, 8}, blockIndices, context.sparseBlockIdx);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND sparseBlockIdx failed, error: %d", ret);
    ret = CreateTensor(ACL_INT32, {1, 8}, blockCounts, context.sparseBlockCount);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND sparseBlockCount failed, error: %d", ret);
    ret = CreateTensor(ACL_INT64, {3}, cuSeqLengths, context.cuSeqLengths);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND cuSeqLengths failed, error: %d", ret);
    ret = CreateTensor(ACL_INT32, {2}, seqUsedQ, context.seqUsedQ);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND seqUsedQ failed, error: %d", ret);
    return CreateCommonArgs(context);
}

void PrintMetadata(const std::string &caseName, const int32_t *metadata)
{
    using namespace metadata_protocol;
    std::printf("\n[%s]\n", caseName.c_str());
    std::printf("header: magic=0x%08X, version=%d, usedSize=%d, saUsedCoreNum=%d, saTotalTaskNum=%d, "
                "fdActiveCoreNum=%d, decodePerCoreTaskNum=%d, combineTaskNum=%d\n",
                static_cast<uint32_t>(metadata[MAGIC_INDEX]), metadata[VERSION_INDEX],
                metadata[METADATA_USED_SIZE_INDEX], metadata[SA_USED_CORE_NUM_INDEX], metadata[SA_TOTAL_TASK_NUM_INDEX],
                metadata[FD_ACTIVE_CORE_NUM_INDEX], metadata[DECODE_PER_CORE_TASK_NUM_INDEX],
                metadata[COMBINE_TASK_NUM_INDEX]);

    const int32_t decodeCoreNum = std::min<int32_t>(metadata[FD_ACTIVE_CORE_NUM_INDEX], MAX_DECODE_CORE_NUM);
    for (int32_t core = 0; core < decodeCoreNum; ++core) {
        std::printf("decode[%d]: baseTask=[%d, %d), firstBlockStart=%d, lastBlockEnd=%d\n", core,
                    metadata[GetDecodeScheduleIndex(core, DECODE_BASE_TASK_START_INDEX)],
                    metadata[GetDecodeScheduleIndex(core, DECODE_BASE_TASK_END_INDEX)],
                    metadata[GetDecodeScheduleIndex(core, DECODE_FIRST_BLOCK_START_INDEX)],
                    metadata[GetDecodeScheduleIndex(core, DECODE_LAST_BLOCK_END_INDEX)]);
    }
    const int32_t combineTaskNum = std::min<int32_t>(metadata[COMBINE_TASK_NUM_INDEX], MAX_COMBINE_TASK_NUM);
    for (int32_t combine = 0; combine < combineTaskNum; ++combine) {
        std::printf("combine[%d]: baseTask=%d, firstCore=%d, partialStart=%d, partialCount=%d\n", combine,
                    metadata[GetCombineScheduleIndex(combine, COMBINE_BASE_TASK_INDEX)],
                    metadata[GetCombineScheduleIndex(combine, COMBINE_FIRST_CORE_INDEX)],
                    metadata[GetCombineScheduleIndex(combine, COMBINE_PARTIAL_START_INDEX)],
                    metadata[GetCombineScheduleIndex(combine, COMBINE_PARTIAL_COUNT_INDEX)]);
    }
}

aclnnStatus ValidateMetadata(const CaseContext &context, const int32_t *metadata)
{
    using namespace metadata_protocol;
    CHECK_LOG_RET(metadata[MAGIC_INDEX] == METADATA_MAGIC, ACL_ERROR_FAILURE, "%s: unexpected metadata magic",
                  context.name.c_str());
    CHECK_LOG_RET(metadata[VERSION_INDEX] == METADATA_VERSION, ACL_ERROR_FAILURE, "%s: unexpected metadata version",
                  context.name.c_str());
    CHECK_LOG_RET(metadata[METADATA_USED_SIZE_INDEX] == static_cast<int32_t>(METADATA_USED_SIZE), ACL_ERROR_FAILURE,
                  "%s: unexpected metadata used size", context.name.c_str());
    CHECK_LOG_RET(metadata[SA_TOTAL_TASK_NUM_INDEX] == context.expectedSaTotalTaskNum, ACL_ERROR_FAILURE,
                  "%s: expected saTotalTaskNum=%d, but got %d", context.name.c_str(), context.expectedSaTotalTaskNum,
                  metadata[SA_TOTAL_TASK_NUM_INDEX]);
    CHECK_LOG_RET(
        metadata[SA_USED_CORE_NUM_INDEX] > 0 && metadata[SA_USED_CORE_NUM_INDEX] <= metadata[SA_TOTAL_TASK_NUM_INDEX],
        ACL_ERROR_FAILURE, "%s: invalid saUsedCoreNum=%d", context.name.c_str(), metadata[SA_USED_CORE_NUM_INDEX]);
    if (context.expectDecodeSchedule) {
        CHECK_LOG_RET(metadata[FD_ACTIVE_CORE_NUM_INDEX] > 0, ACL_ERROR_FAILURE,
                      "%s: DecodeSchedule was not generated", context.name.c_str());
        CHECK_LOG_RET(metadata[COMBINE_TASK_NUM_INDEX] > 0, ACL_ERROR_FAILURE, "%s: CombineSchedule was not generated",
                      context.name.c_str());
    }
    return ACL_SUCCESS;
}

aclnnStatus RunCase(CaseContext &context, aclrtStream stream)
{
    uint64_t workspaceSize = 0U;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
        context.sparseBlockIdx.tensor, context.sparseBlockCount.tensor, context.cuSeqLengths.tensor,
        context.cuSeqLengthsKv.tensor, context.seqUsedQ.tensor, context.seqUsedKv.tensor, context.maxQSeqLen,
        context.maxKvSeqLen, context.numQHeads, context.numKvHeads, context.headDim, context.blockShape, 1,
        context.qInputLayout, context.kvInputLayout, 0, 0, 0, -1, -1, context.metadata.tensor, &workspaceSize,
        &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
                  "%s: aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize failed, error: %d",
                  context.name.c_str(), ret);

    void *workspace = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "%s: allocate workspace failed, error: %d", context.name.c_str(), ret);
    }
    ScopeGuard workspaceGuard([&workspace]() {
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
    });

    ret = aclnnGenericBlockSparseAttentionMetadata(workspace, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "%s: execute metadata failed, error: %d", context.name.c_str(), ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "%s: synchronize stream failed, error: %d", context.name.c_str(), ret);
    ret = aclrtMemcpy(context.metadata.hostAddr, context.metadata.byteSize, context.metadata.deviceAddr,
                      context.metadata.byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "%s: copy metadata to host failed, error: %d", context.name.c_str(), ret);

    const auto *metadata = static_cast<const int32_t *>(context.metadata.hostAddr);
    PrintMetadata(context.name, metadata);
    return ValidateMetadata(context, metadata);
}

aclnnStatus InitAcl(int32_t deviceId, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed, error: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed, error: %d", ret);
    ret = aclrtCreateStream(&stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed, error: %d", ret);
    return ACL_SUCCESS;
}

void FinalizeAcl(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main(int argc, char *argv[])
{
    const int32_t deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
    aclrtStream stream = nullptr;
    aclnnStatus ret = InitAcl(deviceId, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "initialize ACL failed, error: %d", ret);
    ScopeGuard aclGuard([&stream, deviceId]() { FinalizeAcl(deviceId, stream); });

    CaseContext bsndContext;
    ScopeGuard bsndGuard([&bsndContext]() { DestroyCase(bsndContext); });
    ret = CreateBsndDecodeCase(bsndContext);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create BSND example failed, error: %d", ret);
    ret = RunCase(bsndContext, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "run BSND example failed, error: %d", ret);

    CaseContext tndContext;
    ScopeGuard tndGuard([&tndContext]() { DestroyCase(tndContext); });
    ret = CreateTndSeqUsedCase(tndContext);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND example failed, error: %d", ret);
    ret = RunCase(tndContext, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "run TND example failed, error: %d", ret);

    std::printf("\nAll GenericBlockSparseAttentionMetadata examples passed.\n");
    return ACL_SUCCESS;
}
