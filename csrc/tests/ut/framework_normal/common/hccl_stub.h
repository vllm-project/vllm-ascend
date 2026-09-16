/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIB_HCCL_HCCL_H
#define LIB_HCCL_HCCL_H

#include <cstdint>

#ifndef __aicore__
#define __aicore__
#endif

#define __gm__

#define GM_ADDR uint8_t *

namespace AscendC {

using HcclHandle = int8_t;

enum HcclServerType {
    HCCL_SERVER_TYPE_AICPU,
    HCCL_SERVER_TYPE_END
};

enum HcclDataType {
    HCCL_DATA_TYPE_INT8 = 0,
    HCCL_DATA_TYPE_INT16 = 1,
    HCCL_DATA_TYPE_INT32 = 2,
    HCCL_DATA_TYPE_FP16 = 3,
    HCCL_DATA_TYPE_FP32 = 4,
    HCCL_DATA_TYPE_INT64 = 5,
    HCCL_DATA_TYPE_UINT64 = 6,
    HCCL_DATA_TYPE_UINT8 = 7,
    HCCL_DATA_TYPE_UINT16 = 8,
    HCCL_DATA_TYPE_UINT32 = 9,
    HCCL_DATA_TYPE_FP64 = 10,
    HCCL_DATA_TYPE_BFP16 = 11,
    HCCL_DATA_TYPE_INT128 = 12,
    HCCL_DATA_TYPE_HIF8 = 14,
    HCCL_DATA_TYPE_FP8E4M3 = 15,
    HCCL_DATA_TYPE_FP8E5M2 = 16,
    HCCL_DATA_TYPE_FP8E8M0 = 17,
    HCCL_DATA_TYPE_RESERVED
};

enum HcclReduceOp {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_PROD = 1,
    HCCL_REDUCE_MAX = 2,
    HCCL_REDUCE_MIN = 3,
    HCCL_REDUCE_RESERVED
};

enum class CoreType : uint8_t {
    DEFAULT,
    ON_AIV,
    ON_AIC
};

enum class ScopeType : uint8_t {
    ALL,
    QUEUE,
    BLOCK,
    INVALID_TYPE
};

struct HcclServerConfig {
    CoreType type;
    int64_t blockId;
};

struct MemDetails {
    uint64_t size;
    uint64_t addr;
    uint32_t key;
};

struct IbVerbsData {
    MemDetails remoteInput;
    MemDetails remoteOutput;
    MemDetails localInput;
    MemDetails localOutput;
    uint8_t res[24];
};

constexpr int8_t INVALID_HANDLE_ID = -1;
constexpr uint32_t HCCL_GROUP_ID_0 = 0;
constexpr uint32_t HCCL_MAX_RANK_NUM = 32U;

static constexpr HcclServerConfig DEFAULT_CFG = {CoreType::DEFAULT, 0};

template <HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU, const auto &config = DEFAULT_CFG>
class Hccl {
public:
    template <bool commit = false>
    __aicore__ HcclHandle AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count, HcclDataType dataType,
                                    HcclReduceOp op, uint8_t repeat = 1)
    {
        return 0;
    }

    template <bool commit = false>
    __aicore__ HcclHandle AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount, HcclDataType dataType,
                                    uint64_t strideCount, uint8_t repeat = 1)
    {
        return 0;
    }

    template <bool commit = false>
    __aicore__ HcclHandle AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount, HcclDataType dataType,
                                   uint64_t strideCount = 0, uint8_t repeat = 1)
    {
        return 0;
    }

    template <bool commit = false>
    __aicore__ HcclHandle AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls, HcclDataType sendType,
                                    GM_ADDR recvBuf, void *recvCounts, void *rdispls, HcclDataType recvType,
                                    uint8_t repeat = 1)
    {
        return 0;
    }

    template <bool commit = false>
    __aicore__ HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount, HcclDataType dataType,
                                        HcclReduceOp op, uint64_t strideCount, uint8_t repeat = 1)
    {
        return 0;
    }

    template <bool commit = false>
    __aicore__ HcclHandle BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum, uint16_t queueID = 0U)
    {
        return 0;
    }

    template <bool commit = false>
    __aicore__ HcclHandle AlltoAllvWrite(GM_ADDR usrIn, GM_ADDR sendOffsets, GM_ADDR sendSizes,
                                         uint64_t remoteWinOffset, uint64_t localDataSize)
    {
        return 0;
    }

    __aicore__ void Init(GM_ADDR context, __gm__ void *initTiling = nullptr) {}

    __aicore__ void InitV2(GM_ADDR context, const void *initTiling) {}

    __aicore__ int32_t SetCcTiling(__gm__ void *ccOpTilingData)
    {
        return 0;
    }

    __aicore__ int32_t SetCcTilingV2(uint64_t offset)
    {
        return 0;
    }

    __aicore__ void Commit(HcclHandle handleId) {}

    __aicore__ int32_t Wait(HcclHandle handleId)
    {
        return 0;
    }

    __aicore__ int32_t Query(HcclHandle handleId)
    {
        return 1;
    }

    __aicore__ void InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID) {}

    template <ScopeType type = ScopeType::ALL>
    __aicore__ void QueueBarrier(uint16_t queueID)
    {}

    template <bool sync = true>
    __aicore__ int32_t Iterate(HcclHandle handleId, uint16_t *seqSlices, uint16_t seqSliceLen)
    {
        return 0;
    }

    template <bool sync = true>
    __aicore__ void Finalize()
    {}

    __aicore__ GM_ADDR GetWindowsInAddr(uint32_t rankId)
    {
        return 0;
    }

    __aicore__ GM_ADDR GetWindowsOutAddr(uint32_t rankId)
    {
        return 0;
    }

    __aicore__ uint32_t GetRankId()
    {
        return 0;
    }

    __aicore__ uint32_t GetRankDim()
    {
        return 0;
    }

    __aicore__ uint16_t GetQueueNum()
    {
        return 0;
    }

    __aicore__ bool SetReduceDataTypeAbility(HcclReduceOp op, HcclDataType dstDataType, HcclDataType srcDataType)
    {
        return true;
    }
};
} // namespace AscendC

#endif // LIB_HCCL_HCCL_H
