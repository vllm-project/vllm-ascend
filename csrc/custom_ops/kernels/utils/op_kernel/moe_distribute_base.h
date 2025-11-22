/*!
 * \file moe_distribute_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_BASE_H
#define MOE_DISTRIBUTE_BASE_H

/* system tick: 50MHz */
#define CAL_US(tick) (((tick) * 2) / 100)

/* performance macro */
// #define USE_256_TO_1__ // 启用256打1
#ifdef USE_256_TO_1__
    #pragma message("use 256 to 1")
#else // 256打1开启仅作为基线，不配合其他优化点使用
    #define USE_FOR_OPT__ // 启用循环优化
    #define DISPATCH_USE_WRITE_SHUFFLE__ // Dispatch使用write shuffle
    #define USE_TOKEN_COUNT_SPLIT__ // 启用token与count的flag分离
    #define USE_ONE_CORE_WAIT__ // 启用单核等待

    #ifdef USE_ONE_CORE_WAIT__
        #pragma message("use one core wait")
    //启用单核计算cumsum
        // #define USE_ONE_CORE_GETCUMSUM__ 
    #endif
    #ifdef USE_FOR_OPT__
        #pragma message("use for optimization")
        #define FOR_OPT_MAX_BS__ 64
        #define FOR_OPT_MAX_MOE_RANK__ 256
    #endif
    // #define COMBINE_USE_DYNAMIC_QUANT // 默认不开启Combine量化
    #define OPT_RANK_OFFSET  512
    #define USE_WRITE_SHUFFLE
#endif

constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr uint32_t AICPU_MAX_RANK_NUM = 128 * 1024;

struct HcclSignalInfo {
    uint64_t resId; // 在代表event时为eventid，notify时为notifyid
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
    uint32_t cqIds;      // 记录物理cqId
    uint32_t logicCqids; // 记录逻辑cqId
};

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];  // 集合通信AICPU展开资源
    ListCommon nextTagRes;                                  // HccltagLocalResV2
};

enum class rtFloatOverflowMode_t {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
};

struct AlgoTopoInfo {
    uint32_t userRank;     // 通信域 RankID
    uint32_t userRankSize; // 通信域的Rank数量
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;  // 每个Module中的Device数量
    uint32_t superPodNum;              // 集群中总的超节点数
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
    uint64_t nicList;           // niclist数组指针
    uint64_t complanRankLength; // complanRank占用的字节数
    uint64_t complanRank;       // 指针
    uint64_t bridgeRankNum;     // bridgeRank占用的个数
    uint64_t bridgeRank;        // 指针
    uint64_t serverAndsuperPodRankLength; // serverAndsuperPodRank占用的字节数
    uint64_t serverAndsuperPodRank; // 指针
};

struct HcclOpConfig {
    uint8_t deterministic; //确定性计算开关
    uint8_t retryEnable;   // 是否重执行
    uint8_t highPerfEnable;
    uint8_t padding[5];    // 大小需要64By对齐，未来添加参数时减小padding
    uint8_t linkTimeOut[8]; // 发送超时时长 
    uint64_t notifyWaitTime; // 超时时长，同HCCL_EXEC_TIMEOUT
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interHccsDisable = false; //使能rdma开关
    rtFloatOverflowMode_t floatOverflowMode = rtFloatOverflowMode_t::RT_OVERFLOW_MODE_UNDEF;
    uint32_t multiQpThreshold = 512;  // 多QP每个QP分担数据量最小阈值
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
    // 本地资源
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId; // usrrankid
    uint32_t rankSize;       // 通信域内total rank个数
    uint64_t winSize; // 每个win大小，静态图时，可能是0，如果通信域内也有动态图，则可能为非0
    uint64_t localWindowsIn; // 全F为无效值
    uint64_t localWindowsOut; // 全F为无效值
    char hcomId[128];
    // aicore识别remote window
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart; // 为HcclRankRelationRes起始位置
    uint32_t rWinOffset; // 为HcclRemoteRes的大小
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // 外部配置参数
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;                         // RDMA场景使用，910B/910_93为4B，其余芯片为8B
    uint32_t remoteResNum;                       // 有效的remoteResNum
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];  //数组指针，指向HcclRankRelationResV2，下标为remoteUserRankId

    // communicate retry
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    uint64_t tinyMem;  // for all2all
    uint64_t tinyMemSize;
    // 零拷贝场景使用
    uint64_t zeroCopyHeadPtr;
    uint64_t zeroCopyTailPtr;
    uint64_t zeroCopyRingBuffer;
    uint64_t zeroCopyIpcPtrs[16];                // 保存集合通信时每个对端的输入输出内存地址
    uint32_t zeroCopyDevicePhyId[16];            // 保存每个rank对应的物理卡Id

    bool utraceStatusFlag;
};

#endif // MOE_DISTRIBUTE_BASE_H