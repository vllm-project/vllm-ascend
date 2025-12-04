/*!
 * \file moe_distribute_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

/* system tick: 50MHz */
#define CAL_US(tick) (((tick) * 2) / 100)

/* performance macro */
// #define USE_256_TO_1__ // Enable 256 to 1
#ifdef USE_256_TO_1__
    #pragma message("use 256 to 1")
#else // 256 to 1 is only used as baseline, not combined with other optimization points
    #define USE_FOR_OPT__ // Enable loop optimization loop optimization
    #define DISPATCH_USE_WRITE_SHUFFLE__ // Dispatch uses write shuffle
    #define USE_TOKEN_COUNT_SPLIT__ // Enable separation of token and count flags token and count flags
    #define USE_ONE_CORE_WAIT__ // Enable single core wait

    #ifdef USE_ONE_CORE_WAIT__
        #pragma message("use one core wait")
    // Enable single core cumsum calculation
        // #define USE_ONE_CORE_GETCUMSUM__ 
    #endif
    #ifdef USE_FOR_OPT__
        #pragma message("use for optimization")
        #define FOR_OPT_MAX_BS__ 64
        #define FOR_OPT_MAX_MOE_RANK__ 256
    #endif
    // #define COMBINE_USE_DYNAMIC_QUANT // Combine quantization is disabled by default
    #define OPT_RANK_OFFSET  512
    #define USE_WRITE_SHUFFLE
#endif

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

struct HcclSignalInfo {
    uint64_t resId; // EventId when representing event, notifyId when representing notify
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
    uint32_t cqIds;      // Record physical cqId
    uint32_t logicCqids; // Record logical cqId
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];  // Collective communication AICPU expanded resources
    ListCommon nextTagRes;                                  // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;     // Communication domain RankID
    uint32_t userRankSize; // Number of Ranks in communication domain
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;  // Number of Devices in each Module
    uint32_t superPodNum;              // Total number of super nodes in cluster
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
    uint64_t nicList;           // Pointer to niclist array
    uint64_t complanRankLength; // Bytes occupied by complanRank
    uint64_t complanRank;       // Pointer
    uint64_t bridgeRankNum;     // Number of bridgeRank entries
    uint64_t bridgeRank;        // Pointer
    uint64_t serverAndsuperPodRankLength; // Bytes occupied by serverAndsuperPodRank
    uint64_t serverAndsuperPodRank; // Pointer
};

struct HcclOpConfig {
    uint8_t deterministic; // Deterministic computation switch
    uint8_t retryEnable;   // Whether to retry execution
    uint8_t highPerfEnable;
    uint8_t padding[5];    // Size needs 64-byte alignment, reduce padding when adding parameters in future
    uint8_t linkTimeOut[8]; // Send timeout duration
    uint64_t notifyWaitTime; // Timeout duration, same as HCCL_EXEC_TIMEOUTas HCCL_EXEC_TIMEOUT
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false; // Enable RDMA switch
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512;  // Minimum data amount threshold for each QP in multi-QP mode
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
    // Local resources
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId; // usrrankid
    uint32_t rankSize;       // Total number of ranks in communication domain
    uint64_t winSize; // Size of each window, may be 0 for static graphs, may be non-zero if dynamic graphs exist in communication domain
    uint64_t localWindowsIn; // All F means invalid value
    uint64_t localWindowsOut; // All F means invalid value
    char hcomId[128];
    // AICore identifies remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart; // Start position for HcclRankRelationRes
    uint32_t rWinOffset; // Size of HcclRemoteRes
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // External configuration parameters
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;                         // Used in RDMA scenarios, 4B for 910B/910_93, 8B for other chips
    uint32_t remoteResNum;                       // Valid remoteResNum
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];  // Array pointer, points to HcclRankRelationResV2, index is remoteUserRankId

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    // Used in zero-copy scenarios
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];                // Save input/output memory addresses of each peer during collective communication
    uint32_t zeroCopyDevicePhyId[16];            // Save physical card ID corresponding to each rank

    bool utraceStatusFlag;
};

#endif // MOE_DISTRIBUTE_BASE_H