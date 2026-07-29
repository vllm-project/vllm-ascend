/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef lightning_indexer_decode_update_KERNEL_H
#define lightning_indexer_decode_update_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_decode_update_common.h"
#include "lightning_indexer_decode_update_service_vector.h"
#include "lightning_indexer_decode_update_service_cube.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
using namespace matmul;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename LIT>
class LIPreload {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using MM1_OUT_T = float;

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                __gm__ uint8_t *rowModes,
                                __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndex,
                                __gm__ uint8_t *topkSlots, __gm__ uint8_t *missCount,
                                __gm__ uint8_t *tailInfo,
                                __gm__ uint8_t *workspace, const LIU2TilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t WS_DOUBLE = 2;

    LIMatmul<LIT> matmulService;
    LIVector<LIT> vectorService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> rowModesGm;
    GlobalTensor<int32_t> topkIndexGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<int32_t> tailInfoGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> scoresGm;

    uint32_t tmpBlockIdx = 0;
    uint32_t aiCoreIdx = 0;
    uint32_t requestStart = 0;
    uint32_t requestCount = 0;
    LICommon::ConstInfo constInfo{};

    __aicore__ inline void InitRequestRange(uint32_t requestedCoreNum);
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessChunk(const LICommon::RunInfo &runInfo);
    __aicore__ inline void CleanEmptyRequest(uint32_t bIdx);
    __aicore__ inline bool IsSupportedBudget(int32_t metadata) const;
};

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::InitRequestRange(uint32_t requestedCoreNum)
{
    uint32_t activeCoreNum = Min(requestedCoreNum, static_cast<uint32_t>(constInfo.batchSize));
    if (activeCoreNum == 0 || aiCoreIdx >= activeCoreNum) {
        requestCount = 0;
        return;
    }

    uint32_t requestsPerCore = static_cast<uint32_t>(constInfo.batchSize) / activeCoreNum;
    uint32_t extraRequestCores = static_cast<uint32_t>(constInfo.batchSize) % activeCoreNum;
    requestStart = aiCoreIdx * requestsPerCore + Min(aiCoreIdx, extraRequestCores);
    requestCount = requestsPerCore + (aiCoreIdx < extraRequestCores ? 1U : 0U);
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                            __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
                                            __gm__ uint8_t *rowModes,
                                            __gm__ uint8_t *actualSeqLengths,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndex,
                                            __gm__ uint8_t *topkSlots, __gm__ uint8_t *missCount,
                                            __gm__ uint8_t *tailInfo,
                                            __gm__ uint8_t *workspace, const LIU2TilingData *__restrict tiling,
                                            TPipe *tPipe)
{
    tmpBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        aiCoreIdx = tmpBlockIdx;
    }

    constInfo.batchSize = tiling->bSize;
    constInfo.kSeqSize = tiling->s2Size;
    constInfo.kCacheBlockSize = tiling->blockSize;
    constInfo.maxBlockNumPerBatch = tiling->maxBlockNumPerBatch;
    constInfo.poolSize = tiling->poolSize;
    constInfo.qHeadNum = tiling->qHeadNum;
    constInfo.cacheRowWidth = tiling->cacheRowWidth;
    constInfo.metadataIndex = tiling->cacheRowWidth - 1U;
    constInfo.tokenCapacity = tiling->cacheRowWidth - 1U;
    InitRequestRange(tiling->usedCoreNum);

    uint64_t singleCoreMm1ResSize =
        WS_DOUBLE * constInfo.mBaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    uint64_t scoresOffset = static_cast<uint64_t>(tiling->usedCoreNum) * singleCoreMm1ResSize;
    scoresGm.SetGlobalBuffer((__gm__ float *)(workspace + scoresOffset));
    actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.batchSize);
    reqPoolEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries, constInfo.batchSize);
    rowModesGm.SetGlobalBuffer((__gm__ int32_t *)rowModes, constInfo.batchSize);
    cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);

    if ASCEND_IS_AIV {
        vectorService.InitParams(static_cast<uint32_t>(constInfo.kSeqSize), constInfo.qHeadNum,
                                 constInfo.cacheRowWidth);
        weightsGm.SetGlobalBuffer((__gm__ K_T *)weights);
        topkIndexGm.SetGlobalBuffer((__gm__ int32_t *)topkIndex);
        topkSlotsGm.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
        missCountGm.SetGlobalBuffer((__gm__ int32_t *)missCount);
        tailInfoGm.SetGlobalBuffer((__gm__ int32_t *)tailInfo);
        vectorService.InitVec1GlobalTensor(mm1ResGm, weightsGm, cacheSlotsGm, topkIndexGm,
                                           topkSlotsGm, missCountGm, tailInfoGm, scoresGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
    }
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(tPipe);
    } else {
        matmulService.InitBuffers(tPipe);
    }
}

template <typename LIT>
__aicore__ inline bool LIPreload<LIT>::IsSupportedBudget(int32_t metadata) const
{
    int32_t budget = metadata < 0 ? -metadata : metadata;
    return budget == 6144 || budget == 10240 || budget == 12288;
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::CleanEmptyRequest(uint32_t bIdx)
{
    if ASCEND_IS_AIV {
        if ((tmpBlockIdx & 1U) == 0) {
            vectorService.CleanInvalidOutput(static_cast<int64_t>(bIdx) * constInfo.outputCapacity, bIdx);
        }
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::Process()
{
    if (requestCount == 0) {
        return;
    }
    ProcessMain();
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::ProcessMain()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    uint32_t loop = 0;
    for (uint32_t requestOffset = 0; requestOffset < requestCount; ++requestOffset) {
        uint32_t bIdx = requestStart + requestOffset;
        int32_t rowMode = rowModesGm.GetValue(bIdx);
        if (rowMode == LICommon::ROW_MODE_PAD) {
            CleanEmptyRequest(bIdx);
            continue;
        }
        if (rowMode != LICommon::ROW_MODE_DENSE && rowMode != LICommon::ROW_MODE_SPARSE) {
            CleanEmptyRequest(bIdx);
            continue;
        }

        uint32_t actualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
        if (actualSeqLen == 0 || actualSeqLen > constInfo.kSeqSize ||
            actualSeqLen > constInfo.tokenCapacity) {
            CleanEmptyRequest(bIdx);
            continue;
        }

        bool isSparse = rowMode == LICommon::ROW_MODE_SPARSE;
        int32_t cacheRowIdx = 0;
        int32_t cacheMetadata = 0;
        bool isFirstFill = false;
        uint32_t processSeqLen = actualSeqLen;
        if (isSparse) {
            cacheRowIdx = reqPoolEntriesGm.GetValue(bIdx);
            if (cacheRowIdx < 0 || static_cast<uint32_t>(cacheRowIdx) >= constInfo.poolSize) {
                CleanEmptyRequest(bIdx);
                continue;
            }
            int64_t metadataOffset = static_cast<int64_t>(cacheRowIdx) * constInfo.cacheRowWidth +
                                     constInfo.metadataIndex;
            cacheMetadata = cacheSlotsGm.GetValue(metadataOffset);
            if (cacheMetadata == 0 || !IsSupportedBudget(cacheMetadata)) {
                CleanEmptyRequest(bIdx);
                continue;
            }
            isFirstFill = cacheMetadata < 0;
            uint32_t tailTokenCount = (actualSeqLen - 1U) % constInfo.kCacheBlockSize + 1U;
            processSeqLen -= tailTokenCount;
            uint32_t targetCount = static_cast<uint32_t>(isFirstFill ? -cacheMetadata : cacheMetadata);
            if (targetCount > processSeqLen || targetCount > constInfo.outputCapacity) {
                CleanEmptyRequest(bIdx);
                continue;
            }
        }

        uint32_t chunkCount = CeilDiv(processSeqLen, constInfo.s2BaseSize);
        for (uint32_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
            LICommon::RunInfo runInfo{};
            runInfo.loop = loop++;
            runInfo.bIdx = bIdx;
            runInfo.s2Idx = chunkIdx;
            runInfo.actS2Size = processSeqLen;
            runInfo.originalSeqLen = actualSeqLen;
            runInfo.cacheRowIdx = static_cast<uint32_t>(cacheRowIdx);
            runInfo.cacheMetadata = cacheMetadata;
            uint32_t chunkStart = chunkIdx * constInfo.s2BaseSize;
            runInfo.actualSingleProcessSInnerSize =
                Min(constInfo.s2BaseSize, processSeqLen - chunkStart);
            runInfo.actualSingleProcessSInnerSizeAlign =
                LICommon::Align(runInfo.actualSingleProcessSInnerSize, LICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);
            runInfo.isFirstS2InnerLoop = chunkIdx == 0;
            runInfo.isLastS2InnerLoop = chunkIdx + 1 == chunkCount;
            runInfo.isSparse = isSparse;
            runInfo.isFirstFill = isFirstFill;
            ProcessChunk(runInfo);
        }
    }

    if ASCEND_IS_AIC {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void LIPreload<LIT>::ProcessChunk(const LICommon::RunInfo &runInfo)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

} // namespace LIKernel
#endif // lightning_indexer_decode_update_KERNEL_H
