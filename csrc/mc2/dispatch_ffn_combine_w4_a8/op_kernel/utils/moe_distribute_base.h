/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_base.h
 * \brief Base definitions for MoE distribution operations
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

#include "kernel_operator.h"

/// Maximum number of local notify resources
constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
/// Maximum number of local stream resources
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19U;
/// Maximum number of AICPU operator notify resources
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
/// Maximum supported rank number for AICPU
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;
/// Base unit for converting system cycles to time, fixed value 50
constexpr uint32_t TIME_CYCLE = 50;

/// Structure defining HCCL signal information (event/notify)
struct HcclSignalInfo {
    uint64_t resId;     // Resource ID: event ID for event type, notify ID for notify type
    uint64_t addr;      // Resource address
    uint32_t devId;     // Device ID
    uint32_t tsId;      // TS unit ID
    uint32_t rankId;    // Rank ID
    uint32_t flag;      // Control flag
};

/// Common doubly linked list structure for host and device management
struct ListCommon {
    uint64_t nextHost;   // Next node address on host
    uint64_t preHost;    // Previous node address on host
    uint64_t nextDevice; // Next node address on device
    uint64_t preDevice;  // Previous node address on device
};

/// Stream configuration and ID information
struct HcclStreamInfo {
    int32_t streamIds;   // Stream ID
    uint32_t sqIds;      // Send queue ID
    uint32_t cqIds;      // Physical completion queue ID
    uint32_t logicCqids; // Logical completion queue ID
};

/// Local resource information version 2
struct LocalResInfoV2 {
    uint32_t streamNum;                             // Total number of streams
    uint32_t signalNum;                             // Total number of signals
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM]; // Local signal array
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];   // Local stream info array
    HcclStreamInfo mainStreamInfo;                  // Main stream configuration
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM]; // Collective communication AICPU expanded resources
    ListCommon nextTagRes;                          // Pointer to HccltagLocalResV2 structure
};

/// Floating-point overflow mode enumeration
enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0, // Saturation mode
    RT_OVERFLOW_MODE_INFNAN,         // INF/NAN mode
    RT_OVERFLOW_MODE_UNDEF,          // Undefined mode
};

/// Algorithm topology information for cluster communication
struct AlgoTopoInfo {
    uint32_t userRank;                             // Rank ID within communication domain
    uint32_t userRankSize;                         // Total rank count in communication domain
    int32_t deviceLogicId;                         // Logical device ID
    bool isSingleMeshAggregation;                  // Flag for single Mesh aggregation
    uint32_t deviceNumPerAggregation;              // Device count per aggregation module
    uint32_t superPodNum;                          // Total number of super pods in cluster
    uint32_t devicePhyId;                          // Physical device ID
    uint32_t topoType;                             // Topology type
    uint32_t deviceType;                           // Device type
    uint32_t serverNum;                            // Total server count
    uint32_t meshAggregationRankSize;             // Rank size for Mesh aggregation
    uint32_t multiModuleDiffDeviceNumMode;         // Mode for different device counts across modules
    uint32_t multiSuperPodDiffServerNumMode;       // Mode for different server counts across super pods
    uint32_t realUserRank;                         // Actual user rank ID
    bool isDiffDeviceModule;                       // Flag for modules with different devices
    bool isDiffDeviceType;                         // Flag for mixed device types
    uint32_t gcdDeviceNumPerAggregation;           // GCD of device counts per aggregation
    uint32_t moduleNum;                            // Total module count
    uint32_t isUsedRdmaRankPairNum;                // Number of RDMA-enabled rank pairs
    uint64_t isUsedRdmaRankPair;                   // Bitmap for RDMA-enabled rank pairs
    uint32_t pairLinkCounterNum;                   // Link counter count per rank pair
    uint64_t pairLinkCounter;                      // Link counter value
    uint32_t nicNum;                               // NIC device count
    uint64_t nicList;                              // Pointer to NIC list array
    uint64_t complanRankLength;                    // Byte length of complanRank array
    uint64_t complanRank;                          // Pointer to complanRank array
    uint64_t bridgeRankNum;                        // Element count of bridgeRank array
    uint64_t bridgeRank;                           // Pointer to bridgeRank array
    uint64_t serverAndsuperPodRankLength;          // Byte length of serverAndsuperPodRank array
    uint64_t serverAndsuperPodRank;                // Pointer to serverAndsuperPodRank array
};

/// HCCL operator configuration parameters
struct HcclOpConfig {
    uint8_t deterministic;         // Deterministic computation switch
    uint8_t retryEnable;           // Retry execution enable flag
    uint8_t highPerfEnable;        // High performance mode enable flag
    uint8_t padding[5];            // 64-byte alignment padding, reduce when adding new parameters
    uint8_t linkTimeOut[8];        // Link transmission timeout duration
    uint64_t notifyWaitTime;       // Wait timeout, same as HCCL_EXEC_TIMEOUT
    uint32_t retryHoldTime;        // Retry hold time
    uint32_t retryIntervalTime;    // Retry interval time
    bool interHccsDisable = false; // RDMA enable control flag
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF; // Float overflow mode
    uint32_t multiQpThreshold = 512; // Minimum data threshold per QP in multi-QP mode
};

/// MC2 workspace memory configuration
struct HcclMC2WorkSpace {
    uint64_t workSpace;      // Workspace memory address
    uint64_t workSpaceSize;  // Workspace memory size
};

/// Remote resource pointer structure
struct RemoteResPtr {
    uint64_t nextHostPtr;    // Next node pointer on host
    uint64_t nextDevicePtr;  // Next node pointer on device
};

/// Host-Device communication parameters
struct HDCommunicateParams {
    uint64_t hostAddr { 0 };      // Host memory address
    uint64_t deviceAddr { 0 };   // Device memory address
    uint64_t readCacheAddr { 0 }; // Read cache address
    uint32_t devMemSize{ 0 };    // Device memory size
    uint32_t buffLen{ 0 };       // Buffer length
    uint32_t flag{ 0 };          // Control flag
};

/// Customized rank relationship resource version 2
struct HcclRankRelationResV2Custom {
    uint32_t remoteUsrRankId;  // Remote user rank ID
    uint32_t remoteWorldRank; // Remote global rank ID
    uint64_t windowsIn;        // Input window address
    uint64_t windowsOut;       // Output window address
    uint64_t windowsExp;       // Expansion window address
    ListCommon nextTagRes;     // Linked list node
};

/// Customized HCCL operator resource parameters
struct HcclOpResParamCustom {
    // Local resources
    HcclMC2WorkSpace mc2WorkSpace;       // MC2 workspace
    uint32_t localUsrRankId;             // Local user rank ID
    uint32_t rankSize;                   // Total rank count in communication domain
    uint64_t winSize;                    // Single window size (0 for static graph)
    uint64_t localWindowsIn;             // Local input window (all F means invalid)
    uint64_t localWindowsOut;            // Local output window (all F means invalid)
    char hcomId[128];                    // HCCL communication ID
    // AICORE remote window identification
    uint64_t winExpSize;                 // Expansion window size
    uint64_t localWindowsExp;            // Local expansion window address
    uint32_t rWinStart;                  // Start offset of HcclRankRelationRes
    uint32_t rWinOffset;                 // Size of HcclRemoteRes
    uint64_t version;                    // Resource version
    LocalResInfoV2 localRes;             // Local resource information
    AlgoTopoInfo topoInfo;               // Topology information

    // External configuration parameters
    HcclOpConfig config;                 // Operator configuration
    uint64_t hostStateInfo;              // Host state information address
    uint64_t aicpuStateInfo;             // AICPU state information address
    uint64_t lockAddr;                   // Synchronization lock address
    uint32_t rsv[16];                    // Reserved field
    uint32_t notifysize;                 // RDMA notify size: 4B for 910B/910_93, 8B for other chips
    uint32_t remoteResNum;               // Valid remote resource count
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM]; // Pointer array to HcclRankRelationResV2, indexed by remote rank ID

    // Communication retry control
    HDCommunicateParams kfcControlTransferH2DParams; // Host-to-Device control transfer params
    HDCommunicateParams kfcStatusTransferD2HParams; // Device-to-Host status transfer params
    uint64_t tinyMem;                                // Temporary memory for All2All operation
    uint64_t tinyMemSize;                            // Temporary memory size
    // Zero-copy scenario parameters
    uint64_t zeroCopyHeadPtr;                        // Zero-copy ring buffer head pointer
    uint64_t zeroCopyTailPtr;                        // Zero-copy ring buffer tail pointer
    uint64_t zeroCopyRingBuffer;                      // Zero-copy ring buffer address
    uint64_t zeroCopyIpcPtrs[16];                    // I/O memory addresses for peer ranks in collective communication
    uint32_t zeroCopyDevicePhyId[16];                // Physical device IDs for each rank

    bool utraceStatusFlag;                            // UTrace status flag
};

/// Transport layer memory type enumeration
enum class HcclAiRMAMemType : uint32_t {
    LOCAL_INPUT = 0,    // Local input memory
    REMOTE_INPUT,       // Remote input memory

    LOCAL_OUTPUT,       // Local output memory
    REMOTE_OUTPUT,      // Remote output memory

    // Extensible memory types can be added before MAX_NUM
    // e.g., LOCAL_EXP, REMOTE_EXP
    MAX_NUM             // Maximum memory type count
};

/// Transport layer memory information structure
struct HcclAiRMAMemInfo {
    uint32_t memMaxNum{0};          // Total memory types, equals HcclAiRMAMemType::MAX_NUM
    uint32_t sizeOfMemDetails{0};   // Size of MemDetails for validation and offset calculation
    uint64_t memDetailPtr{0};       // Base address of MemDetails array, count = HcclAiRMAMemType::MAX_NUM
    // Extensible fields
};

/// Complete Transport QP/Memory information structure
struct HcclAiRMAInfo {
    uint32_t curRankId{0};         // Current rank ID
    uint32_t rankNum{0};           // Total rank count
    uint32_t qpNum{0};             // QP count per Transport

    uint32_t sizeOfAiRMAWQ{0};     // Size of HcclAiRMAWQ structure
    uint32_t sizeOfAiRMACQ{0};     // Size of HcclAiRMACQ structure
    uint32_t sizeOfAiRMAMem{0};   // Size of HcclAiRMAMemInfo structure

    // 2D array base address for HcclAiRMAWQ (Send Queue)
    // Total QP count: rankNum * qpNum
    // SQ pointer calculation: sqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t sqPtr{0};

    // 2D array base address for HcclAiRMACQ (Send Completion Queue)
    // Total QP count: rankNum * qpNum
    // SCQ pointer calculation: scqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t scqPtr{0};

    // 2D array base address for HcclAiRMAWQ (Receive Queue)
    // Total QP count: rankNum * qpNum
    // RQ pointer calculation: rqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMAWQ
    // 0 <= qpIndex < qpNum
    uint64_t rqPtr{0};

    // 2D array base address for HcclAiRMACQ (Receive Completion Queue)
    // Total QP count: rankNum * qpNum
    // RCQ pointer calculation: rcqPtr + (dstRankId * qpNum + qpIndex) * sizeOfAiRMACQ
    // 0 <= qpIndex < qpNum
    uint64_t rcqPtr{0};

    // 1D array base address for HcclAiRMAMemInfo
    // Memory info count: rankNum
    // Memory pointer calculation: memPtr + rankId * sizeOfAiRMAMem
    // srcRankId for local memory, dstRankId for transport remote memory
    uint64_t memPtr{0};
    // Extensible fields
};

/// Communication capability combination structure
struct CombinedCapability {
    uint64_t dataplaneModeBitmap;  // Dataplane mode capability bitmap
};

/// A2 Combine operator parameters
struct HcclA2CombineOpParam {
    uint64_t workSpace;                         // Client-server communication memory, managed by HCCL
    uint64_t workSpaceSize;                     // Client-server communication memory size
    uint32_t rankId;                            // Local rank ID
    uint32_t rankNum;                           // Total rank count in communication group
    uint64_t winSize;                           // Size of each window memory
    uint64_t windowsIn[AscendC::HCCL_MAX_RANK_NUM];      // Input window addresses: windowsIn[rankId] = local address, others = remote mapping addresses
    uint64_t windowsOut[AscendC::HCCL_MAX_RANK_NUM];     // Output window addresses: windowsOut[rankId] = local address, others = remote mapping addresses
    uint8_t res[8328];                         // Reserved memory
    uint8_t multiFlag;                          // Multi-mode flag
    __gm__ AscendC::IbVerbsData *data;          // IB verbs data pointer
    uint64_t dataSize;                          // IB verbs data size
    // Extended fields
    uint64_t sizeOfAiRMAInfo;                   // Size of HcclAiRMAInfo structure
    uint64_t aiRMAInfo;                         // Pointer to HcclAiRMAInfo structure

    CombinedCapability* capability;             // Device-side communication capability info pointer
    uint64_t capabilitySize;                    // Size of communication capability structure
};

/// Dataplane operation mode enumeration
enum class DataplaneMode : uint32_t {
    HOST = 0,   // Host dataplane mode
    AICPU = 1,  // AICPU dataplane mode
    AIV = 2,    // AIV dataplane mode
};

/// Doorbell mode enumeration
enum class DBMode : int32_t {
    INVALID_DB = -1,  // Invalid doorbell mode
    HW_DB = 0,        // Hardware doorbell mode
    SW_DB             // Software doorbell mode
};

/// AI RMA Work Queue structure
struct HcclAiRMAWQ {
    uint32_t wqn{0};            // Work queue number
    uint64_t bufAddr{0};        // Queue buffer address
    uint32_t wqeSize{0};        // WQE (Work Queue Element) size
    uint32_t depth{0};          // Queue depth
    uint64_t headAddr{0};       // Queue head pointer address
    uint64_t tailAddr{0};       // Queue tail pointer address
    DBMode dbMode{DBMode::INVALID_DB}; // Doorbell mode: 0-HW/1-SW
    uint64_t dbAddr{0};         // Doorbell register address
    uint32_t sl{0};             // Service level
};

/// AI RMA Completion Queue structure
struct HcclAiRMACQ {
    uint32_t cqn{0};            // Completion queue number
    uint64_t bufAddr{0};        // Queue buffer address
    uint32_t cqeSize{0};        // CQE (Completion Queue Element) size
    uint32_t depth{0};          // Queue depth
    uint64_t headAddr{0};       // Queue head pointer address
    uint64_t tailAddr{0};       // Queue tail pointer address
    DBMode dbMode{DBMode::INVALID_DB}; // Doorbell mode: 0-HW/1-SW
    uint64_t dbAddr{0};         // Doorbell register address
};

/// HNS RoCE RC SQ WQE structure
struct hns_roce_rc_sq_wqe {
    uint32_t byte_4;       // 4-byte control field
    uint32_t msg_len;      // Message length
    uint32_t immtdata;     // Immediate data
    uint32_t byte_16;      // 16-byte control field
    uint32_t byte_20;      // 20-byte control field
    uint32_t rkey;         // Remote key
    uint64_t remoteVA;     // Remote virtual address
};

/// HNS RoCE Lite WQE data segment structure
struct hns_roce_lite_wqe_data_seg {
    uint32_t len;         // Data segment length
    uint32_t lkey;        // Local key
    uint64_t localVA;     // Local virtual address
};

/// Cache write-through operation to flush data to memory
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

/// Get current dataplane mode from context
__aicore__ inline DataplaneMode GetDataplaneMode(GM_ADDR contextGM0) {
    __gm__ HcclA2CombineOpParam *winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
    CombinedCapability* capability = winContext_->capability;
    uint64_t capabilitySize = winContext_->capabilitySize;
    DataplaneMode dataplaneMode = DataplaneMode::AICPU;
    if (capability == 0) {
        return dataplaneMode;
    }
    uint64_t dataplaneModeBitmap = capability->dataplaneModeBitmap;
    if ((dataplaneModeBitmap & 0x04) == 0x04) {
        dataplaneMode = DataplaneMode::AIV;
    }
    return dataplaneMode;
}

/// Get current timestamp in microseconds
__aicore__ inline int64_t GetCurrentTimestampUs()
{
    return AscendC::GetSystemCycle() / TIME_CYCLE;
}

/// Record communication duration for specified rank
__aicore__ inline void RecordRankCommDuration(AscendC::LocalTensor<int32_t> performanceInfoU32Tensor, uint32_t rankId, int64_t startTime)
{
    int64_t endTime = GetCurrentTimestampUs();
    int32_t duration = static_cast<int32_t>(endTime - startTime); // int32_t supports up to 2^31 microseconds (~35 minutes), sufficient for practical scenarios
    performanceInfoU32Tensor.SetValue(rankId * sizeof(int64_t) / sizeof(int32_t), duration); // Use int32_t because atomicAdd does not support int64_t, only lower 32 bits of int64_t are assigned
}
#endif // MOE_DISTRIBUTE_BASE_H