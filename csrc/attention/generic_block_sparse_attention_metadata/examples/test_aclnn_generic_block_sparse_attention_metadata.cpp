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
 * @file test_aclnn_generic_block_sparse_attention_metadata.cpp
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"
#include "securec.h"

#include "../generic_block_sparse_attention_metadata_tiling.h"

#define CHECK_LOG_RET(condition, returnValue, format, ...) \
    do { \
        if (!(condition)) { \
            std::printf(format "\n", ##__VA_ARGS__); \
            return (returnValue); \
        } \
    } while (0)

namespace {

namespace metadata_protocol = optiling::generic_block_sparse_attention_metadata;

constexpr int32_t INVALID_BLOCK_INDEX = -1;
constexpr int32_t DEVICE_ID_ARG_INDEX = 1;
constexpr int64_t GBSA_BLOCK_SHAPE_X = 1;
constexpr int64_t GBSA_BLOCK_SHAPE_Y = 128;
constexpr int64_t GBSA_HEAD_DIM = 128;
constexpr int64_t GBSA_LAYOUT_SPARSE_PATTERN = 4; // KVN_TotalQB_KB
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE = 0;
constexpr bool GBSA_IS_CONSISTENT_TOPK = false;
constexpr int64_t GBSA_MASK_TYPE = 1;
constexpr int64_t GBSA_QUANT_TYPE = 0;
constexpr int64_t GBSA_SOFTMAX_PRECISION = 0;
constexpr int64_t GBSA_WINDOW_SIZE = -1;
constexpr int64_t GBSA_DEFAULT_MAX_SEQ_LEN = -1;

struct ScopeGuard {
    explicit ScopeGuard(std::function<void()> callback)
        : callback_(std::move(callback))
    {}
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
    int64_t maxQSeqLen = GBSA_DEFAULT_MAX_SEQ_LEN;
    int64_t maxKvSeqLen = GBSA_DEFAULT_MAX_SEQ_LEN;
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
        const errno_t memsetRet = memset_s(tensor.hostAddr, tensor.byteSize, 0, tensor.byteSize);
        CHECK_LOG_RET(memsetRet == EOK, ACL_ERROR_FAILURE, "memset_s tensor host memory failed, error: %d", memsetRet);
    } else {
        const errno_t memcpyRet = memcpy_s(tensor.hostAddr, tensor.byteSize, hostData, tensor.byteSize);
        CHECK_LOG_RET(memcpyRet == EOK, ACL_ERROR_FAILURE, "memcpy_s tensor host data failed, error: %d", memcpyRet);
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
    std::vector<int32_t> indices(blockCounts.size() * static_cast<size_t>(capacity), INVALID_BLOCK_INDEX);
    for (size_t task = 0; task < blockCounts.size(); ++task) {
        for (int32_t block = 0; block < blockCounts[task]; ++block) {
            indices[task * static_cast<size_t>(capacity) + static_cast<size_t>(block)] = block;
        }
    }
    return indices;
}

aclnnStatus CreateCommonArgs(CaseContext &context)
{
    const std::vector<int64_t> blockShapeData = {GBSA_BLOCK_SHAPE_X, GBSA_BLOCK_SHAPE_Y};
    context.blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_LOG_RET(context.blockShape != nullptr, ACL_ERROR_BAD_ALLOC, "aclCreateIntArray failed");
    return CreateTensor(ACL_INT32, {metadata_protocol::METADATA_TOTAL_SIZE}, nullptr, context.metadata);
}

aclnnStatus CreateTndDecodeCase(CaseContext &context)
{
    constexpr int64_t batchSize = 1;
    constexpr int64_t qStorageLength = 2;
    constexpr int64_t kvStorageLength = 2048;
    constexpr int64_t numQHeads = 4;
    constexpr int64_t numKvHeads = 1;
    constexpr int64_t sparseBlockCapacity = 12;
    constexpr int32_t firstQBlockCount = 12;
    constexpr int32_t secondQBlockCount = 3;

    context.name = "TND decode schedule";
    context.maxQSeqLen = GBSA_DEFAULT_MAX_SEQ_LEN;
    context.maxKvSeqLen = GBSA_DEFAULT_MAX_SEQ_LEN;
    context.numQHeads = numQHeads;
    context.numKvHeads = numKvHeads;
    context.headDim = GBSA_HEAD_DIM;
    context.qInputLayout = "TND";
    context.kvInputLayout = "PA_BBND";
    context.expectedSaTotalTaskNum = static_cast<int32_t>(qStorageLength * numKvHeads);
    context.expectDecodeSchedule = true;

    const std::vector<int32_t> blockCounts = {firstQBlockCount, secondQBlockCount};
    const std::vector<int32_t> blockIndices = MakeSparseBlockIndices(blockCounts, sparseBlockCapacity);
    aclnnStatus ret = CreateTensor(ACL_INT32, {numKvHeads, qStorageLength, sparseBlockCapacity}, blockIndices,
                                   context.sparseBlockIdx);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND sparseBlockIdx failed, error: %d", ret);
    ret = CreateTensor(ACL_INT32, {numKvHeads, qStorageLength}, blockCounts, context.sparseBlockCount);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND sparseBlockCount failed, error: %d", ret);
    const std::vector<int64_t> cuSeqLengths = {0, qStorageLength};
    ret = CreateTensor(ACL_INT64, {static_cast<int64_t>(cuSeqLengths.size())}, cuSeqLengths, context.cuSeqLengths);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND cuSeqLengths failed, error: %d", ret);
    const std::vector<int32_t> seqUsedKv = {static_cast<int32_t>(kvStorageLength)};
    ret = CreateTensor(ACL_INT32, {batchSize}, seqUsedKv, context.seqUsedKv);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create PA seqUsedKv failed, error: %d", ret);
    const std::vector<int32_t> seqUsedQ = {static_cast<int32_t>(qStorageLength)};
    ret = CreateTensor(ACL_INT32, {batchSize}, seqUsedQ, context.seqUsedQ);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND seqUsedQ failed, error: %d", ret);
    return CreateCommonArgs(context);
}

aclnnStatus CreateTndSeqUsedCase(CaseContext &context)
{
    constexpr int64_t batchSize = 2;
    constexpr int64_t qStorageLengthPerBatch = 4;
    constexpr int64_t totalQStorageLength = batchSize * qStorageLengthPerBatch;
    constexpr int64_t kvStorageLengthPerBatch = 2048;
    constexpr int64_t numQHeads = 4;
    constexpr int64_t numKvHeads = 1;
    constexpr int64_t sparseBlockCapacity = totalQStorageLength;
    constexpr int32_t firstBatchUsedQ = 2;
    constexpr int32_t secondBatchUsedQ = 3;
    constexpr int32_t firstBlockCount = 1;

    context.name = "TND cuSeqLengths and seqUsedQ";
    context.maxQSeqLen = GBSA_DEFAULT_MAX_SEQ_LEN;
    context.maxKvSeqLen = GBSA_DEFAULT_MAX_SEQ_LEN;
    context.numQHeads = numQHeads;
    context.numKvHeads = numKvHeads;
    context.headDim = GBSA_HEAD_DIM;
    context.qInputLayout = "TND";
    context.kvInputLayout = "PA_BBND";
    context.expectedSaTotalTaskNum = (firstBatchUsedQ + secondBatchUsedQ) * static_cast<int32_t>(numKvHeads);

    // The two batches occupy physical Q ranges [0, 4) and [4, 8). seqUsedQ selects [0, 2) and [4, 7).
    const std::vector<int64_t> cuSeqLengths = {0, qStorageLengthPerBatch, totalQStorageLength};
    const std::vector<int32_t> seqUsedQ = {firstBatchUsedQ, secondBatchUsedQ};
    std::vector<int32_t> blockCounts(static_cast<size_t>(totalQStorageLength));
    for (size_t index = 0; index < blockCounts.size(); ++index) {
        blockCounts[index] = firstBlockCount + static_cast<int32_t>(index);
    }
    const std::vector<int32_t> blockIndices = MakeSparseBlockIndices(blockCounts, sparseBlockCapacity);
    aclnnStatus ret = CreateTensor(ACL_INT32, {numKvHeads, totalQStorageLength, sparseBlockCapacity}, blockIndices,
                                   context.sparseBlockIdx);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND sparseBlockIdx failed, error: %d", ret);
    ret = CreateTensor(ACL_INT32, {numKvHeads, totalQStorageLength}, blockCounts, context.sparseBlockCount);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND sparseBlockCount failed, error: %d", ret);
    ret = CreateTensor(ACL_INT64, {static_cast<int64_t>(cuSeqLengths.size())}, cuSeqLengths, context.cuSeqLengths);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND cuSeqLengths failed, error: %d", ret);
    const std::vector<int32_t> seqUsedKv = {static_cast<int32_t>(kvStorageLengthPerBatch),
                                            static_cast<int32_t>(kvStorageLengthPerBatch)};
    ret = CreateTensor(ACL_INT32, {batchSize}, seqUsedKv, context.seqUsedKv);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create PA seqUsedKv failed, error: %d", ret);
    ret = CreateTensor(ACL_INT32, {batchSize}, seqUsedQ, context.seqUsedQ);
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
        CHECK_LOG_RET(metadata[FD_ACTIVE_CORE_NUM_INDEX] > 0, ACL_ERROR_FAILURE, "%s: DecodeSchedule was not generated",
                      context.name.c_str());
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
        context.maxKvSeqLen, context.numQHeads, context.numKvHeads, context.headDim, context.blockShape,
        context.qInputLayout, context.kvInputLayout, GBSA_LAYOUT_SPARSE_PATTERN, GBSA_MASK_TYPE, GBSA_QUANT_TYPE,
        GBSA_SOFTMAX_PRECISION, GBSA_WINDOW_SIZE, GBSA_WINDOW_SIZE, GBSA_RESIDUAL_BLOCK_MODE, GBSA_IS_CONSISTENT_TOPK,
        context.metadata.tensor, &workspaceSize, &executor);
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
    const int32_t deviceId = argc > DEVICE_ID_ARG_INDEX ? std::atoi(argv[DEVICE_ID_ARG_INDEX]) : 0;
    aclrtStream stream = nullptr;
    aclnnStatus ret = InitAcl(deviceId, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "initialize ACL failed, error: %d", ret);
    ScopeGuard aclGuard([&stream, deviceId]() { FinalizeAcl(deviceId, stream); });

    CaseContext decodeContext;
    ScopeGuard decodeGuard([&decodeContext]() { DestroyCase(decodeContext); });
    ret = CreateTndDecodeCase(decodeContext);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND decode example failed, error: %d", ret);
    ret = RunCase(decodeContext, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "run TND decode example failed, error: %d", ret);

    CaseContext tndContext;
    ScopeGuard tndGuard([&tndContext]() { DestroyCase(tndContext); });
    ret = CreateTndSeqUsedCase(tndContext);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "create TND example failed, error: %d", ret);
    ret = RunCase(tndContext, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "run TND example failed, error: %d", ret);

    std::printf("\nAll GenericBlockSparseAttentionMetadata examples passed.\n");
    return ACL_SUCCESS;
}
