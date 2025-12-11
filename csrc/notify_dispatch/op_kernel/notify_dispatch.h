#ifndef NOTIFY_DISPATCH_H
#define NOTIFY_DISPATCH_H

#include <climits>
#include "kernel_operator.h"

#include "../common/comm_args.h"
#include "../common/data_copy.h"
#include "../common/sync_collectives.h"
#include "../common/moe_distribute_base.h"

using namespace AscendC;
using namespace Moe;

#define KERNELS_ARGS_FUN_ALL2ALL()                                                                                  \
    GM_ADDR sendDataInput, GM_ADDR tokenPerExpertDataInput, GM_ADDR sendDataOffsetOutput, GM_ADDR recvDataOutput,   \
        int64_t len, int64_t numTokens, int op, int root, int cycleCount, GM_ADDR scale, int64_t scaleCount,        \
        GM_ADDR offset, int localRank, int localRankSize, GM_ADDR commArgs, int magic

#define KERNELS_ARGS_CALL_ALL2ALL()                                                                         \
    sendDataInput, tokenPerExpertDataInput, sendDataOffsetOutput, recvDataOutput, len, numTokens, op, root, \
    cycleCount, scale, scaleCount, offset, localRank, localRankSize, commArgs, magic

template <typename T>
class NotifyDispatch {
    constexpr static int INVALID_RANK_NUM = 0xFFFFFFFF;   // Invalid rank
    constexpr static int64_t CORE_NUMS_PER_STAGE_X = 24;  // Maximum number of cores provided by the producer stage
    constexpr static int64_t CORE_NUMS_PER_STAGE_Y = 16;  // Maximum number of cores provided by the consumer stage
    constexpr static int64_t CORE_NUMS_PER_STAGE_Z = 16;  // Maximum number of cores provided by the consumer stage 2
    constexpr static int64_t SHARE_QUE_DEPTH = 1;         // Depth of a single shared queue
    constexpr static int64_t RANK_NUM_PER_NODE = 16;
    constexpr static int64_t SIO_NUM = 2;  // Depth of a single shared queue
    constexpr static int64_t MAX_CORE_NUM = 48;
    constexpr static int64_t MAX_RANK_PER_CORE = 8;
    constexpr static int64_t MULTI_RANK_SIZE = 48;
    constexpr static int64_t MAX_BUFFER_NUMBER = 10;

    constexpr static int64_t IDLER_CORE = 0;  // Idle core
    constexpr static int64_t PRODUCER_CORE = 1;  // Producer group, responsible for writing data to shared memory, input->share, or share->share
    constexpr static int64_t CONSUMER_CORE = 2;  // Consumer group, responsible for reading data from shared memory, share->output
    constexpr static int64_t CONSUMER_CORE2 = 3;

public:
    __aicore__ inline NotifyDispatch(int rank, int rankSize, uint32_t extraFlag)
        : rank(rank), rankSize(rankSize), extraFlag(extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN_ALL2ALL())
    {
        InitSmallFullMesh(KERNELS_ARGS_CALL_ALL2ALL());
        nodeNum = rankSize / localRankSize;
        localRankId = rank % localRankSize;
        localNodeId = rank / localRankSize;
        perNodeDataNum = GetDataCount(len, nodeNum);   // 128K/4 = 32K
        perRankDataNum = GetDataCount(len, rankSize);  // 128K/64 = 2K

        tokenPerExpertDataAlignLen = Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataOffsetAlignLen = Ceil(numExperts * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataAlignLen = Ceil(numExperts * sendPerGroup * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        // Initialize core grouping
        InitCoreGroup();
        // Initialize data slicing
        InitDataSlice();

        this->sendDataInput = (__gm__ T *)sendDataInput;
        this->tokenPerExpertDataInput = (__gm__ int32_t *)tokenPerExpertDataInput;
        this->sendDataOffsetOutput = (__gm__ T *)sendDataOffsetOutput;
        this->recvDataOutput = (__gm__ T *)recvDataOutput;
        sendDataInputGt.SetGlobalBuffer((__gm__ T *)sendDataInput);
        tokenPerExpertDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tokenPerExpertDataInput);
        sendDataOffsetOutputGt.SetGlobalBuffer((__gm__ T *)sendDataOffsetOutput);
        recvDataOutputGt.SetGlobalBuffer((__gm__ T *)recvDataOutput);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx < 1) {
            AssembleSendData();
        }
        SyncAll<true>();
        if (blockIdx < coreNumPerStageX) {
            InputToShareSlice();
        }
        if (blockIdx < coreNumPerStageY) {
            ShareToShareSlice();
        }
    }

private:
    __aicore__ inline void InitCoreGroup()
    {
        coreNumPerStageY = MAX_CORE_NUM;
        coreNumPerStageX = MAX_CORE_NUM;
        rankNumPerCore = (rankSize + MAX_CORE_NUM - 1) / MAX_CORE_NUM;
    }

    __aicore__ inline void InitDataSlice()
    {
        // The producer is responsible for moving the input data of this rank to shared memory, input-->share
        if (blockIdx < coreNumPerStageX) {
            ProducerDataSlice();
        }
    }

    __aicore__ inline void ProducerDataSlice()
    {
        // The ipcQue responsible for the current core
        writeGt.SetGlobalBuffer((__gm__ T *)(shareAddrs[rank] + IPC_DATA_OFFSET));
    }

    __aicore__ inline void AssembleSendData()
    {
        pipe.InitBuffer(tokenPerExpertDataBuf, tokenPerExpertDataAlignLen);
        pipe.InitBuffer(sendDataBuf, sendDataAlignLen);
        pipe.InitBuffer(sendDataOffsetBuf, sendDataOffsetAlignLen);

        __ubuf__ int32_t *tokenPerExpertUB = (__ubuf__ int32_t *)get_imm(96);
        CpGM2UB(tokenPerExpertUB, (__gm__ int32_t *)tokenPerExpertDataInputGt.GetPhyAddr(), tokenPerExpertDataAlignLen);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        __ubuf__ T *sendDataOffsetUB = (__ubuf__ T *)get_imm(96 + tokenPerExpertDataAlignLen);
        __ubuf__ T *sendDataUB = (__ubuf__ T *)get_imm(96 + tokenPerExpertDataAlignLen + sendDataOffsetAlignLen);

        int prefixSum = 0;
        for (int i = 0; i < numExperts; ++i) {
            int numTokensExpert = tokenPerExpertUB[i];
            sendDataUB[i * sendPerGroup] = numTokensExpert;
            sendDataUB[i * sendPerGroup + 1] = prefixSum;
            sendDataUB[i * sendPerGroup + 2] = numTokens;
            sendDataOffsetUB[i] = prefixSum;

            prefixSum += numTokensExpert;
        }
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

        CpUB2GM((__gm__ T *)sendDataInputGt.GetPhyAddr(), sendDataUB, sendDataAlignLen);
        CpUB2GM((__gm__ T *)sendDataOffsetOutputGt.GetPhyAddr(), sendDataOffsetUB, sendDataOffsetAlignLen);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // copy input to other rank share
    __aicore__ inline void InputToShareSlice()
    {
        __ubuf__ int64_t *inputUB = (__ubuf__ int64_t *)get_imm(0);
        int64_t copyOffset = blockIdx * rankNumPerCore;
        copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
        if (copyLen > 0) {
            readGt = sendDataInputGt[copyOffset * perRankDataNum];
            CpGM2GMPingPong<T>(
                copyLen * perRankDataNum * sizeof(T), readGt, writeGt[copyOffset * perRankDataNum], COPYONLY);
            int64_t v = MergeMagicWithValue(magic, 1);
            *inputUB = v;
            AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
                CpUB2GM((__gm__ int64_t *)(shareAddrs[i]) + rank * FLAG_UNIT_INT_NUM, inputUB, sizeof(int64_t));
            }
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        // magic as the high part, eventID as the low part, combined into a value for comparison
        return (static_cast<int64_t>(static_cast<uint32_t>(magic)) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline void ShareToShareSlice()
    {
        __ubuf__ T *inputUB = (__ubuf__ T *)get_imm(96);
        int64_t copyOffset = blockIdx * rankNumPerCore;
        copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
        if (copyLen > 0) {
            int checkRank[MAX_RANK_PER_CORE];
            for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
                checkRank[i - copyOffset] = i + rank % copyLen;
                if (checkRank[i - copyOffset] >= copyOffset + copyLen) {
                    checkRank[i - copyOffset] -= copyLen;
                }
            }
            for (int i = 0; i < copyLen; i++) {
                readGt1[i].SetGlobalBuffer((__gm__ T *)(shareAddrs[checkRank[i]] + IPC_DATA_OFFSET));
            }
            sync.WaitSyncFlag(magic, 1, copyOffset, rank, copyLen);
            for (int i = 0; i < copyLen; i++) {
                CpGM2GMPingPong<T>(perRankDataNum * sizeof(T),
                    readGt1[i][rank * perRankDataNum],
                    recvDataOutputGt[checkRank[i] * perRankDataNum],
                    COPYONLY);
            }
        }
    }

    FORCE_INLINE_AICORE int64_t GetDataCount(const int64_t dataLen, const int64_t useBlockNum);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId, uint8_t ctxIdx);
    __aicore__ inline int32_t GetMagicValue(void);
    FORCE_INLINE_AICORE void InitSmallFullMesh(KERNELS_ARGS_FUN_ALL2ALL());
    template <typename F>
    FORCE_INLINE_AICORE void SetAtomic(int op);
    FORCE_INLINE_AICORE void UnsetAtomic(int op);
    template<HardEvent eventType>
    FORCE_INLINE_AICORE void SetWaitEvent(event_t eventId);
    template <typename K, typename U = K>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& sendDataInputGt,
                                            const GlobalTensor<K>& recvDataOutputGT, int op);

    GlobalTensor<T> sendDataInputGt;
    GlobalTensor<int> tokenPerExpertDataInputGt;
    GlobalTensor<T> sendDataOffsetOutputGt;
    GlobalTensor<T> recvDataOutputGt;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    GlobalTensor<T> readGt1[MAX_BUFFER_NUMBER];
    GlobalTensor<T> ipcGT;
    GlobalTensor<int64_t> sendCountMatrixGm;
    __gm__ T *sendDataInput;
    __gm__ int *tokenPerExpertDataInput;
    __gm__ T *sendDataOffsetOutput;
    __gm__ T *recvDataOutput;
    int64_t isPad = 0;
    int64_t maxSliceNum;
    int64_t revLen = 0;
    int64_t sendLen = 0;
    int64_t sliceLen;
    int64_t perNodeDataNum;
    int64_t perRankDataNum;
    int64_t curRankDataNum;
    int64_t sendOffset[MULTI_RANK_SIZE];
    int64_t revOffset[MULTI_RANK_SIZE];
    int64_t inputDataLen[MULTI_RANK_SIZE];

    int64_t nodeNum;
    int64_t localRankId;
    int64_t localNodeId;
    int64_t targetNode;
    int64_t targetLocalRankIds[2];
    int64_t queLen;
    int64_t queSize;
    int64_t coreNumPerStageX;             // Number of cores used per stage
    int64_t coreNumPerStageY;             // Number of cores used per stage
    int64_t coreNumPerStageZ;             // Number of cores used per stage
    int64_t flagNumPerStage;              // Number of synchronization flags used per stage
    int64_t coreNumPerNode;               // Number of cores allocated per node
    int64_t coreNumPerRank;               // Number of cores allocated per rank
    int64_t rankNumPerCore;               // Number of ranks responsible per core
    int64_t coreGroup;                    // Functional group of the current core
    int64_t targetRank[MULTI_RANK_SIZE];  // Ranks responsible by the current core
    int64_t targetRankX;
    int64_t targetRankY;

    int64_t queElemLen;  // Size of each element in the shared memory queue (in terms of T)

    int64_t copyLen;  // Length of the current data slice being copied (in terms of T)

    // for coll
    int rank;
    int rankSize;
    int localRank = 0;
    int localRankSize = 0;
    int xRankSize = 0;
    int yRankSize = 0;
    int xRankIdx = 0;
    int yRankIdx = 0;
    uint32_t extraFlag;
    int numTokens;
    int sendPerGroup = 3;
    int root;
    int64_t len;
    int64_t numExperts;
    int64_t magic;
    int64_t blockIdx;  // Index of the current aicore
    int64_t blockNum;  // Total number of aicores for the current rank
    int32_t numRanks;
    int64_t timeout;
    uint16_t *rootRanks;
    GM_ADDR scale;
    GM_ADDR shareAddrs[CAM_MAX_RANK_SIZE];  // List of shared memory addresses
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GlobalTensor<GM_ADDR> peerMemsAddrGm_;
    GlobalTensor<int64_t> dfx;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<> tokenPerExpertDataBuf;
    TBuf<> sendDataOffsetBuf;
    TBuf<> sendDataBuf;

    uint32_t sendDataAlignLen{0};
    uint32_t tokenPerExpertDataAlignLen{0};
    uint32_t sendDataOffsetAlignLen{0};

    SyncCollectives sync;
};

template <typename T>
FORCE_INLINE_AICORE int64_t NotifyDispatch<T>::GetDataCount(const int64_t dataLen, const int64_t useBlockNum)
{
    return dataLen / useBlockNum;
}

template <typename T>
__aicore__ inline GM_ADDR NotifyDispatch<T>::GetWindAddrByRankId(const int32_t rankId, uint8_t ctxIdx)
{
    uint32_t curRankId = rank;
#ifdef OPT_RANK_OFFSET
#pragma message("use rank offset")
    if (curRankId == rankId) {
        return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn) + rankId * OPT_RANK_OFFSET;
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn) +
            rankId * OPT_RANK_OFFSET;
#else
    if (curRankId == rankId) {
        return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn);
#endif
}

// Assign values to winContext_[COMM_EP_IDX] and blockIdx before calling
template <typename T>
__aicore__ inline int32_t NotifyDispatch<T>::GetMagicValue(void)
{
    int32_t magic = 0;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[blockIdx * UB_ALIGN_SIZE]);
    magic = selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE);
    if (magic <= 0) {
        magic = 1;
    }
    selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE) = magic + 1;
    return magic;
}

template <typename T>
FORCE_INLINE_AICORE void NotifyDispatch<T>::InitSmallFullMesh(KERNELS_ARGS_FUN_ALL2ALL())
{
    this->root = root;
    this->len = len;
    this->numExperts = len / sendPerGroup;
    this->numTokens = numTokens;
    this->scale = scale;
    this->localRank = localRank;
    this->localRankSize = localRankSize;
    this->xRankSize = localRankSize;
    this->yRankSize = rankSize / localRankSize;
    this->xRankIdx = rank % localRankSize;
    this->yRankIdx = rank / localRankSize;
    blockIdx = GetBlockIdx();
    blockNum = GetBlockNum();
    uint8_t ctxIdx;

    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    this->magic = GetMagicValue();
    ctxIdx = COMM_EP_IDX;

    shareAddrs[rank] = GetWindAddrByRankId(rank, ctxIdx) +
                        (this->magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);

    int64_t rankNumPerCore = (rankSize + MAX_CORE_NUM - 1) / MAX_CORE_NUM;
    int64_t copyOffset = blockIdx * rankNumPerCore;
    int64_t copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
    if (copyLen > 0) {
        for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
            shareAddrs[i] = GetWindAddrByRankId(i, ctxIdx) +
                            (this->magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }
    }

    // When the number of cores is more than the number of ranks, each core is responsible for fetching data from a specified rank
    int coreNumPerRank = blockNum / rankSize;  // Calculate the number of cores assigned to read for each rank, e.g., 48 cores 4 ranks, each rank is assigned 12 cores
    int maxCore = coreNumPerRank * rankSize;  // Calculate the maximum number of cores that can be used for reading, cores exceeding this number will not take action
    if (blockIdx < maxCore) {
        int readRank = blockIdx / coreNumPerRank;  // Calculate the rank to be read based on the block, 48 cores divided into 4 groups
        shareAddrs[readRank] = GetWindAddrByRankId(readRank, ctxIdx) +
                                (this->magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
    }

    pipe.InitBuffer(tBuf, UB_SINGLE_TOTAL_SIZE_MAX);

    sync.Init(rank, rankSize, shareAddrs, tBuf);
}

/**
 * @brief Copy data from GM to GM with ping-pong method.
 * @tparam dataSizeRemain The remaining size of data to be copied.
 * @tparam K The type of output data.
 * @tparam U The type of input data.
 * @param sendDataInputGt The global tensor of send data.
 * @param recvDataOutputGT The global tensor of recv data.
 * @param op The operation to be performed during the copy.
 * @details This function copies data from global memory to global memory using a ping-pong method.
 * It first checks if the input and output types are the same. If they are, it uses a single buffer.
 * If they are not, it divides the buffer according to the size ratio of the types and aligns it to 32 bytes.
 * Then, it sets the atomic operation, waits for the flags, and performs the copy operation.
 */
template <typename T>
template <typename K, typename U>
FORCE_INLINE_AICORE void NotifyDispatch<T>::CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& sendDataInputGt,
                                                                const GlobalTensor<K>& recvDataOutputGT, int op)
{
    // General case (U = K), input/output are the same, share one UB
    // Only when conversion is needed (U->K), UB will be divided into two parts according to the ratio of sizeof(U):sizeof(K) and aligned to 32 bytes
    constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
    constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(K) + sizeof(U)) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    constexpr int32_t inputUbBlockSize = std::is_same_v<K, U> ? ubBlockSize : ubAlignNum * sizeof(U);
    constexpr int32_t outputUbBlockSize = std::is_same_v<K, U> ? ubBlockSize : ubAlignNum * sizeof(K);

    __gm__ U *input = const_cast<__gm__ U *>(sendDataInputGt.GetPhyAddr());
    __gm__ K *output = const_cast<__gm__ K *>(recvDataOutputGT.GetPhyAddr());
    __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET), (__ubuf__ U*)(UB_MID_OFFSET)};
    __ubuf__ K* outputUB[2] = {(__ubuf__ K*)inputUB[0], (__ubuf__ K*)inputUB[1]};
    if constexpr (!std::is_same_v<K, U>) {
        outputUB[0] = (__ubuf__ K*)(inputUB[0] + inputUbBlockSize / sizeof(U));
        outputUB[1] = (__ubuf__ K*)(inputUB[1] + inputUbBlockSize / sizeof(U));
    }
    int inputOffsetNum = 0;
    int outputOffsetNum = 0;
    if (dataSizeRemain <= 0) {
        return;
    }

    SetAtomic<K>(op);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2 waits for MTE3
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2 waits for MTE3
    for (int64_t i = 0; dataSizeRemain > 0; i++) {
        // size and dataSizeRemain both refer to the output size
        uint32_t size = dataSizeRemain > outputUbBlockSize ? outputUbBlockSize : dataSizeRemain;
        event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size / sizeof(K) * sizeof(U));
        if constexpr (!std::is_same_v<K, U>) {
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CastImpl((i & 1) ? outputUB[0] : outputUB[1], (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE,
                size / sizeof(K));
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
        }
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CpUB2GM(output + outputOffsetNum, (i & 1) ? outputUB[0] : outputUB[1], size);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

        dataSizeRemain -= size;
        inputOffsetNum += (size / sizeof(K));
        outputOffsetNum += (size / sizeof(K));
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); // MTE2 waits for MTE3
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1); // MTE2 waits for MTE3

    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3); // Scalar waits for MTE3
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);

    UnsetAtomic(op);
    return;
}

template <typename T>
template <typename F>
FORCE_INLINE_AICORE void NotifyDispatch<T>::SetAtomic(int op)
{
    PipeBarrier<PIPE_ALL>();
    if (op != -1) {
#ifdef __DAV_C220_VEC__
        SetAtomicOpType<F>(op);
#endif
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
FORCE_INLINE_AICORE void NotifyDispatch<T>::UnsetAtomic(int op)
{
    if (op != -1) {
        AscendC::SetAtomicNone();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template<HardEvent eventType>
FORCE_INLINE_AICORE void NotifyDispatch<T>::SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

#endif  // NOTIFY_DISPATCH_H
