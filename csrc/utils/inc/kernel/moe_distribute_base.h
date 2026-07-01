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
 * \file moe_distribute_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;                                  // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType; // TopoType
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false;
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512;
};

struct HcclMC2WorkSpace {
    uint64_t workSpace;
    uint64_t workSpaceSize;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HDCommunicateParams {
    uint64_t hostAddr { 0 };
    uint64_t deviceAddr { 0 };
    uint64_t readCacheAddr { 0 };
    uint32_t devMemSize{ 0 };
    uint32_t buffLen{ 0 };
    uint32_t flag{ 0 };
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId; // usrrankid
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];
    uint32_t zeroCopyDevicePhyId[16];

    bool utraceStatusFlag;
};

// Transport
enum class HcclAiRMAMemType : uint32_t {
    LOCAL_INPUT = 0,
    REMOTE_INPUT,
    LOCAL_OUTPUT,
    REMOTE_OUTPUT,
    MAX_NUM
};

struct HcclAiRMAMemInfo {
    uint32_t memMaxNum{0};
    uint32_t sizeOfMemDetails{0};
    uint64_t memDetailPtr{0};
};

// Transport QP/Mem
struct HcclAiRMAInfo {
    uint32_t curRankId{0};
    uint32_t rankNum{0};
    uint32_t qpNum{0};
    uint32_t sizeOfAiRMAWQ{0};  // sizeof(HcclAiRMAWQ)
    uint32_t sizeOfAiRMACQ{0};  // sizeof(HcclAiRMACQ)
    uint32_t sizeOfAiRMAMem{0};  // sizeof(HcclAiRMAMemInfo)
    uint64_t sqPtr{0};
    uint64_t scqPtr{0};
    uint64_t rqPtr{0};
    uint64_t rcqPtr{0};
    uint64_t memPtr{0};
};

struct HcclA2CombineOpParam {
    uint64_t workSpace;                         // Address for communication between client and server,
                                                // hccl requests and clears
    uint64_t workSpaceSize;                     // Space for communication between client and server
    uint32_t rankId;                            // id of this rank
    uint32_t rankNum;                           // num of ranks in this comm group
    uint64_t winSize;                           // size of each windows memory
    uint64_t windowsIn[AscendC::HCCL_MAX_RANK_NUM];      // windows address for input, windowsIn[rankId] corresponds
                                                // to the local card address,
                                                // and others are cross-card mapping addresses.
    uint64_t windowsOut[AscendC::HCCL_MAX_RANK_NUM];     // windows address for output, windowsOut[rankId] corresponds
                                                // to the local card address,
                                                // and others are cross-card mapping addresses.
    uint8_t res[8328];
    uint8_t multiFlag;
    __gm__ AscendC::IbVerbsData *data;
    uint64_t dataSize;

    uint64_t sizeOfAiRMAInfo;
    uint64_t aiRMAInfo;
};

enum class DBMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB
};

struct HcclAiRMAWQ {
    uint32_t wqn{0};
    uint64_t bufAddr{0};
    uint32_t wqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB}; // 0-hw/1-sw
    uint64_t dbAddr{0};
    uint32_t sl{0};
};

struct HcclAiRMACQ {
    uint32_t cqn{0};
    uint64_t bufAddr{0};
    uint32_t cqeSize{0};
    uint32_t depth{0};
    uint64_t headAddr{0};
    uint64_t tailAddr{0};
    DBMode dbMode{DBMode::INVALID_DB}; // 0-hw/1-sw
    uint64_t dbAddr{0};
};

struct hns_roce_rc_sq_wqe {
    uint32_t byte_4;
    uint32_t msg_len;
    uint32_t immtdata;
    uint32_t byte_16;
    uint32_t byte_20;
    uint32_t rkey;
    uint64_t remoteVA;
};


struct hns_roce_lite_wqe_data_seg {
    uint32_t len;
    uint32_t lkey;
    uint64_t localVA;
};

__aicore__ inline void cacheWriteThrough(__gm__ uint8_t* sourceAddr, uint64_t length) {
    __gm__ uint8_t* start =
        (__gm__ uint8_t*)((uint64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t* end =
        (__gm__ uint8_t*)(((uint64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
}

#endif // MOE_DISTRIBUTE_BASE_H