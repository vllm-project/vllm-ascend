#ifndef SYNC_COLLECTIVES_H
#define SYNC_COLLECTIVES_H

#include "comm_args.h"

using namespace AscendC;
using namespace Moe;

// Synchronization flag occupies length
constexpr int64_t FLAG_UNIT_INT_NUM = 4;
// Memory size occupied by each synchronization unit (Bytes)
constexpr int64_t SYNC_UNIT_SIZE = FLAG_UNIT_INT_NUM * sizeof(int64_t);
// High-order offset when using magic as a comparison value
constexpr int64_t MAGIC_OFFSET = 32;
constexpr int64_t MAGIC_MASK = ~((1LL << MAGIC_OFFSET) - 1);

class SyncCollectives {
public:
    __aicore__ inline SyncCollectives() {}

    __aicore__ inline void Init(int rank, int rankSize, GM_ADDR *shareAddrs, TBuf<QuePosition::VECCALC> &tBuf)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->shareAddrs = shareAddrs;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        // Length of a single indicator segment
        segmentCount = GetBlockNum() * FLAG_UNIT_INT_NUM;
        // Initialize the intra-card/inter-card synchronization address corresponding to the current core.
        localSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]);
        basicSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        blockOuterSyncAddr = (__gm__ int64_t*)(shareAddrs[rank]) + segmentCount + GetBlockIdx() * FLAG_UNIT_INT_NUM;
        this->tBuf = tBuf;
    }

    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag(localSyncAddr + eventID * FLAG_UNIT_INT_NUM, v);
    }

    /**
     * @brief Set the flag for the specified eventID of the designated card, with the value being a combination of magic and value.
     * @param magic The operator batch, which will be combined into the high 32 bits of the flag value to be set.
     * @param value The specific value to be set, which will be the low 32 bits of the flag value to be set.
     * @param eventID Physically, it is an offset from the shared memory base address (requires scaling, not an absolute value).
     * @param rank This rank is the rankId corresponding to the peerMems array in the CommArgs structure, not a global or local id.
     *              (Local is not applicable in the 91093 scenario, and global is not applicable in the 910B multi-machine scenario.)
     */
    __aicore__ inline void SetSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, v);
    }

    __aicore__ inline int32_t CalEventIdByMulBlockNum(int32_t blockMultiplier, int32_t targetCoreId)
    {
        return (blockMultiplier * blockNum) + targetCoreId;
    }

    /**
     * @brief Wait for the flag of the specified eventID on the specified card to become a value
     *        composed of the combination of magic and value.
     * @param magic The operator batch, which will be combined into the high 32 bits of the flag
     *              value to be wait.
     * @param value The specific value to be wait, which will be the low 32 bits of the flag
     *              value to be wait.
     * @param eventID Physically, it is an offset from the shared memory base address (requires
     *                scaling, not an absolute value).
     * @param rank This rank is the rankId corresponding to the peerMems array in the CommArgs
     *              structure, not a global or local id. (Local is not applicable in the 91093
     *              scenario, and global is not applicable in the 910B multi-machine scenario.)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[this->rank]) + eventID * FLAG_UNIT_INT_NUM, 1, v);
    }

    /**
     * @brief Wait for the flags starting from the specified eventID on the specified card to become
     *        a value composed of the combination of magic and value.<br>
     *        Note: [eventID, eventID + flagNum)
     */
    __aicore__ inline void WaitSyncFlag(int32_t magic, int32_t value, int32_t eventID, int32_t rank, int64_t flagNum)
    {
        int64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum, v);
    }

    // Set inner-card synchronization flag (memory A)
    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(basicSyncAddr, value);
    }

    __aicore__ inline void SetInnerFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag((__gm__ int64_t*)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
    }

    // Wait for a single inner-card synchronization flag (memory A)
    __aicore__ inline void WaitInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
    }

    // Wait for all inner-card synchronization flags within the entire rank (memory A)
    __aicore__ inline void WaitRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // Check all inner-card synchronization flags within the entire rank (memory A)
    __aicore__ inline bool CheckRankInnerFlag(int32_t magic, int32_t eventID, int64_t waitRank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        return CheckOneRankAllFlag((__gm__ int64_t*)(shareAddrs[waitRank]), value);
    }

    // Set inter-card synchronization flag (memory B)
    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(blockOuterSyncAddr, value);
    }

    __aicore__ inline void SetOuterFlag(int32_t magic, int32_t eventID, int64_t setRank, int64_t setBlock)
    {
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(setRank, setBlock);
        int64_t value = MergeMagicWithValue(magic, eventID);
        SetFlag(flagAddr, value);
    }

    // Wait for a single inter-card synchronization flag (memory B)
    __aicore__ inline void WaitOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr = GetOuterFlagAddr(waitRank, waitBlock);
        WaitOneRankPartFlag(flagAddr, 1, value);
    }

    // Wait for all inter-card synchronization flags within the entire rank (memory B)
    __aicore__ inline void WaitOneRankOuterFlag(int32_t magic, int32_t eventID, int64_t rank)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(rank, 0);
        WaitOneRankPartFlag(flagAddr, blockNum, value);
    }

    // Wait for flagNum inter-card synchronization flags starting from startBlock for all ranks (memory B)
    __aicore__ inline void WaitAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // Offset reading of rank flags to prevent performance impact from concurrent copying by multiple cores
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            WaitOneRankPartFlag(flagAddr, flagNum, value);
        }
    }

    // Check flagNum inter-card synchronization flags starting from startBlock for all ranks (memory B)
    __aicore__ inline bool CheckAllRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t startBlock,
        int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        int waitRank;
        for (auto r = 0; r < rankSize; ++r) {
            waitRank = (rank + r) % rankSize;  // Offset reading of rank flags to prevent performance impact from concurrent copying by multiple cores
            flagAddr = GetOuterFlagAddr(waitRank, startBlock);
            if (!CheckOneRankPartFlag(flagAddr, flagNum, value)) {
                return false;
            }
        }
        return true;
    }

    // Wait for all inter-card synchronization flags for all ranks, full rank synchronization (memory B)
    __aicore__ inline void WaitAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        WaitAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // Check all inter-card synchronization flags for all ranks, full rank synchronization (memory B)
    __aicore__ inline bool CheckAllRankOuterFlag(int32_t magic, int32_t eventID)
    {
        return CheckAllRankPartOuterFlag(magic, eventID, 0, blockNum);
    }

    // Low-level interface, set synchronization flag
    __aicore__ inline void SetFlag(__gm__ int64_t* setAddr, int64_t setValue)
    {
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<int64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(1, 0);
        localSet.SetValue(0, setValue);

        // Copy global synchronization flag to local
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // Wait for SetValue to complete
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // Wait for UB->GM to complete
    }

    // Low-level interface, wait for synchronization flag
    __aicore__ inline void WaitFlag(__gm__ int64_t* waitAddr, int64_t waitValue)
    {
        WaitOneRankPartFlag(waitAddr, 1, waitValue);
    }

    // Read a flag, return an immediate number
    __aicore__ inline int64_t GetFlag(__gm__ int64_t* waitAddr)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(1, 0);
        // Copy global to local
        DataCopy(localWait, globalWait, FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

        int64_t res = localWait.GetValue(0);
        return res;
    }

    // Get multiple consecutive synchronization flags within a single card
    __aicore__ inline void WaitOneRankPartOuterFlag(int32_t magic, int32_t eventID, int64_t waitRank,
                                                    int64_t startBlock, int64_t flagNum)
    {
        int64_t value = MergeMagicWithValue(magic, eventID);
        __gm__ int64_t* flagAddr;
        flagAddr = GetOuterFlagAddr(waitRank, startBlock);
        WaitOneRankPartFlag(flagAddr, flagNum, value);
    }

    // Get synchronization flag within a single card (memory A)
    __aicore__ inline int64_t GetInnerFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM);
    }

    __aicore__ inline int64_t GetOuterFlag(int64_t waitRank, int64_t waitBlock)
    {
        return GetFlag((__gm__ int64_t*)(shareAddrs[waitRank]) + segmentCount + waitBlock * FLAG_UNIT_INT_NUM);
    }

    // In the rank Chunk Flag area, return success if the destRank chunk Flag value is 0, otherwise fail
    __aicore__ inline int64_t GetChunkFlag(int64_t rank, int64_t destRank, int64_t magic, int64_t timeout)
    {
        int64_t value = MergeMagicWithValue(magic, 0);
        int64_t status = GetChunkFlagValue((__gm__ int64_t*)(shareAddrs[rank]) +
                                            IPC_CHUNK_FLAG + destRank * FLAG_UNIT_INT_NUM, value, timeout);
        return status;
    }

    // Set the destRank chunk Flag value in the rank Chunk Flag area to value
    __aicore__ inline void SetChunkFlag(int64_t rank, int64_t destRank, int64_t magic, int64_t eventId)
    {
        int64_t value = MergeMagicWithValue(magic, eventId);
        SetFlag((__gm__ int64_t*)(shareAddrs[rank]) + IPC_CHUNK_FLAG + destRank * FLAG_UNIT_INT_NUM, value);
    }

    __aicore__ inline int64_t GetChunkRecvLen(int64_t rank, int64_t destRank, int64_t magic, int64_t timeout)
    {
        int64_t len = GetChunkFlagValue((__gm__ int64_t*)(shareAddrs[rank]) + IPC_CHUNK_FLAG +
                                        destRank * FLAG_UNIT_INT_NUM, 0, timeout, true, magic);
        return len;
    }

private:
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value)
    {
        // Merge magic as the high bits and eventID as the low bits into a value for comparison
        return (static_cast<int64_t>(static_cast<uint32_t>(magic)) << MAGIC_OFFSET) | static_cast<int64_t>(value);
    }

    __aicore__ inline __gm__ int64_t* GetInnerFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + flagBlock * FLAG_UNIT_INT_NUM;
    }

    __aicore__ inline __gm__ int64_t* GetOuterFlagAddr(int64_t flagRank, int64_t flagBlock)
    {
        return (__gm__ int64_t*)(shareAddrs[flagRank]) + segmentCount + flagBlock * FLAG_UNIT_INT_NUM;
    }

    // Wait for a part of synchronization flags within a rank
    __aicore__ inline void WaitOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        int64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            int64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }

    // Wait for all synchronization flags within a rank
    __aicore__ inline void WaitOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        WaitOneRankPartFlag(waitAddr, blockNum, checkValue);
    }

    // Check partial synchronization flags within a rank, copy only once
    __aicore__ inline bool CheckOneRankPartFlag(__gm__ int64_t* waitAddr, int64_t flagNum, int64_t checkValue)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        // Copy global synchronization flags to local
        DataCopy(localWait, globalWait, flagNum * FLAG_UNIT_INT_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB
        // Check if the synchronization flags are equal to checkValue
        bool isSync = true;
        for (auto i = 0; i < flagNum; ++i) {
            // Continue waiting if any core has not reached the checkValue phase
            int64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
            if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                isSync = false;
                break;
            }
        }
        return isSync;
    }

    __aicore__ inline int64_t GetChunkFlagValue(__gm__ int64_t* waitAddr, int64_t checkValue, int64_t timeout,
                                                bool checkNonZero = false, int64_t magic = 0)
    {
        GlobalTensor<int64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;

        int64_t waitTimes = 0;
        int64_t v = 0;

        do {
            // Copy global sync flag to local
            DataCopy(localWait, globalWait[0], FLAG_UNIT_INT_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            isSync = true;
            v = localWait.GetValue(0);
            if (checkNonZero) {
                // Non-zero check mode
                if (((v & MAGIC_MASK) == (static_cast<int64_t>(magic) << MAGIC_OFFSET)) && (v & 0xFFFFFFFF)) {
                    return v & 0xFFFFFFFF;  // Return lower 32 bits when non-zero
                }
            } else {
                // Exact value check mode
                if (v == checkValue) {
                    return WAIT_SUCCESS;
                }
            }

            isSync = false;
            waitTimes++;

            if (timeout > INT64_MAX / MAX_WAIT_ROUND_UNIT || waitTimes >= (timeout * MAX_WAIT_ROUND_UNIT)) {
                isSync = true;
                return v; // Return the read flag value
            }
        } while (!isSync);

        return checkNonZero ? 0 : v;
    }

    // Check all sync flags within a rank, copy only once
    __aicore__ inline bool CheckOneRankAllFlag(__gm__ int64_t* waitAddr, int64_t checkValue)
    {
        return CheckOneRankPartFlag(waitAddr, blockNum, checkValue);
    }
    int rank;
    int rankSize;
    int blockIdx;
    int blockNum;
    GM_ADDR *shareAddrs;
    int64_t segmentCount;  // Length of a single sync flag segment (count in int64_t)
    __gm__ int64_t* localSyncAddr;
    __gm__ int64_t* basicSyncAddr;  // Intra-card sync flag address for the current block
    __gm__ int64_t* blockOuterSyncAddr;  // Inter-card sync flag address for the current block
    TBuf<QuePosition::VECCALC> tBuf;
};

#endif // SYNC_COLLECTIVES_H