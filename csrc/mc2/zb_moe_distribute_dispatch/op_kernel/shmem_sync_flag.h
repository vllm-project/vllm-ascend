#ifndef SHMEM_SYNC_FLAG_H
#define SHMEM_SYNC_FLAG_H

#include "shmem.h"
#include "kernel_operator.h"

namespace ShmemSyncFlagImpl {

using namespace AscendC;

// ============================================================================
// Constants
// ============================================================================

/// Each flag slot occupies 32B (4 × int64_t), meeting vector 32B alignment.
constexpr uint32_t FLAG_SLOT_SIZE = 32U;
constexpr uint32_t FLAG_ELEM_NUM = FLAG_SLOT_SIZE / sizeof(int64_t);  // 4

/// Flag area starts at gva_gm + 100KB.
constexpr uint64_t FLAG_AREA_BASE = 100UL * 1024UL;

/// Magic slot alignment per core (512B per core).
constexpr uint32_t MAGIC_SLOT_ALIGN = 512U;
constexpr uint32_t MAX_CORE_NUM_SYNC = 48U;

/// Total magic area: 48 cores × 512B = 24KB.
constexpr uint64_t MAGIC_AREA_SIZE = static_cast<uint64_t>(MAX_CORE_NUM_SYNC) * MAGIC_SLOT_ALIGN;

/// Phase values are defined independently by each kernel.
/// Typical convention:
///   PHASE_ENTRY = 1  — kernel entered, prior kernel complete, input tensors ready
///   PHASE_DONE  = 2  — kernel compute/DMA complete, output tensors finalized

// ============================================================================
// Helper
// ============================================================================

/// Merge magic (high 32 bits) with value/phase (low 32 bits) into a 64-bit flag.
/// Monotonically increasing magic guarantees uniqueness across all kernel invocations.
__aicore__ inline int64_t MakeFlagValue(int32_t magic, int32_t value)
{
    return (static_cast<int64_t>(static_cast<uint32_t>(magic)) << 32) |
           static_cast<int64_t>(static_cast<uint32_t>(value));
}

// ============================================================================
// ShmemSyncFlag — Generic cross-card flag synchronization via shared memory
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                       Memory Layout (per rank)                          │
// │                                                                          │
// │  gva_gm + FLAG_AREA_BASE (100KB):                                       │
// │                                                                          │
// │  ┌──────────────────────────────────────────────────────┐               │
// │  │ Magic Area  (24 KB)                                  │               │
// │  │   slot[coreIdx] at coreIdx × 512B                    │               │
// │  │   Monotonically incremented once per IncrementMagic()│               │
// │  ├──────────────────────────────────────────────────────┤               │
// │  │ Barrier PingPong Buffer 0                            │               │
// │  │   slot[srcRank]  (32B each), epWorldSize slots       │               │
// │  ├──────────────────────────────────────────────────────┤               │
// │  │ Barrier PingPong Buffer 1  (same layout)             │               │
// │  └──────────────────────────────────────────────────────┘               │
// │                                                                          │
// │  Barrier buf size = epWorldSize × 32B  (relatively fixed)               │
// │                                                                          │
// │  ┌──────────────────────────────────────────────────────┐               │
// │  │ Flag PingPong Buffer 0                               │               │
// │  │   flag[srcRank][eventID]  (32B each)                 │               │
// │  │   srcRank  ∈ [0, epWorldSize)                        │               │
// │  │   eventID  ∈ [0, slotsPerRank)                       │               │
// │  ├──────────────────────────────────────────────────────┤               │
// │  │ Flag PingPong Buffer 1  (same layout)                │               │
// │  └──────────────────────────────────────────────────────┘               │
// │                                                                          │
// │  Flag buf size = epWorldSize × slotsPerRank × 32B                       │
// │  Total = Magic(24KB) + 2×BarrierBuf + 2×FlagBuf                         │
// └──────────────────────────────────────────────────────────────────────────┘
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                  Magic Management                                       │
// │                                                                          │
// │  Each kernel calls IncrementMagic() exactly once at entry.              │
// │  Magic increments monotonically: kernel K gets M, kernel K+1 gets M+1. │
// │                                                                          │
// │  Buffer selection: magic % 2 (ping-pong).                               │
// │  Flag value: (magic << 32) | phase.                                     │
// │  Monotonic magic ⇒ stale flags from ANY prior kernel are rejected.     │
// └──────────────────────────────────────────────────────────────────────────┘
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                  Synchronization Protocol                               │
// │                                                                          │
// │  Writer (rank A, core i):                                                │
// │    1. Complete data writes (DMA to remote shmem)                         │
// │    2. SetFlag(destRank=B, eventID=i, phase)                              │
// │       → MTE3 fence (flush all prior writes)                              │
// │       → Write flag to B's local shmem at slot [A][i]                     │
// │                                                                          │
// │  Reader (rank B, core j):                                                │
// │    1. WaitFlag(srcRank=A, eventID=i, phase)                              │
// │       → Poll own local shmem until flag matches (magic, phase)           │
// │    2. After return, data from rank A is guaranteed visible.              │
// │                                                                          │
// │  Ordering guarantee (DAG, no circular dependency):                       │
// │    DataWrite(A→B) ── MTE3 fence ──► SetFlag(A→B)                        │
// │                                       │                                  │
// │                                       ▼                                  │
// │                                   WaitFlag(B←A) ──► DataRead(B←A)       │
// │                                                                          │
// │  Key invariant:                                                          │
// │    Rank R setting flag with magic=M guarantees all kernels with          │
// │    magic < M have fully completed on rank R (including DMA flush),      │
// │    because kernels execute sequentially per rank.                        │
// └──────────────────────────────────────────────────────────────────────────┘
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                      Sync Granularity                                    │
// │                                                                          │
// │  Determined by the caller via slotsPerRank at Init():                   │
// │                                                                          │
// │  ● Per-core:   slotsPerRank = blockNum                                   │
// │      SetAllRankCoreFlag()   → each core signals all ranks               │
// │      WaitAllRankAllEvent()  → each core waits assigned rank slice       │
// │                                                                          │
// │  ● Per-slot:   slotsPerRank = N (arbitrary)                              │
// │      SetFlag(destRank, slotIdx, phase)                                   │
// │      WaitFlag(srcRank, slotIdx, phase)                                   │
// │                                                                          │
// │  ● Per-rank:   slotsPerRank = 1                                          │
// │      SetFlagBatch(0, epWorldSize, 0, phase)                              │
// │      WaitFlagBatch(0, epWorldSize, 0, phase)                             │
// └──────────────────────────────────────────────────────────────────────────┘
//
class ShmemSyncFlag
{
public:
    __aicore__ inline ShmemSyncFlag() {}

    // ========================================================================
    // Initialization
    // ========================================================================

    /// @param gvaGM         Base GVA symmetric address (identical layout on every rank)
    /// @param epRankId      This rank's ID [0, epWorldSize)
    /// @param epWorldSize   Number of ranks in the EP communication domain
    /// @param slotsPerRank  Event slots per source rank:
    ///                        blockNum             → per-core granularity
    ///                        moeExpertPerRankNum  → per-expert granularity
    ///                        1                    → per-rank granularity
    /// @param tBuf          UB scratch buffer (>= FLAG_SLOT_SIZE = 32 bytes)
    __aicore__ inline void Init(GM_ADDR gvaGM, uint32_t epRankId, uint32_t epWorldSize,
                                uint32_t slotsPerRank, TBuf<QuePosition::VECCALC> &tBuf)
    {
        gvaGM_ = gvaGM;
        epRankId_ = epRankId;
        epWorldSize_ = epWorldSize;
        slotsPerRank_ = slotsPerRank;
        coreIdx_ = GetBlockIdx();
        coreNum_ = GetBlockNum();
        tBuf_ = tBuf;

        flagBaseGM_ = gvaGM_ + FLAG_AREA_BASE;
        barrierBufSize_ = static_cast<uint64_t>(epWorldSize_) * FLAG_SLOT_SIZE;
        barrierBaseOffset_ = MAGIC_AREA_SIZE;
        bufSize_ = static_cast<uint64_t>(epWorldSize_) * slotsPerRank_ * FLAG_SLOT_SIZE;
        flagBufBaseOffset_ = MAGIC_AREA_SIZE + 2 * barrierBufSize_;
    }

    /// Read magic and increment — EVERY kernel calls this exactly once at entry.
    /// Magic is stored per-core at flagBase + coreIdx × 512B.
    /// Monotonically increments: kernel K gets M, kernel K+1 gets M+1, etc.
    /// Buffer selection: magic % 2 (ping-pong).
    __aicore__ inline void IncrementMagic()
    {
        GM_ADDR magicAddr = flagBaseGM_ + static_cast<uint64_t>(coreIdx_) * MAGIC_SLOT_ALIGN;
        GlobalTensor<uint64_t> magicGT;
        magicGT.SetGlobalBuffer((__gm__ uint64_t *)magicAddr);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(magicGT);
        magic_ = static_cast<int32_t>(magicGT(0));
        magicGT(0) = static_cast<uint64_t>(magic_) + 1ULL;
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(magicGT);
        PipeBarrier<PIPE_ALL>();
        barrierSeq_ = 0;
    }

    // ========================================================================
    // Core Flag Operations (naming follows sync_collectives.h)
    // ========================================================================

    /// Write a flag to destRank's local shmem at slot [epRankId_][eventID].
    ///
    /// Implicit MTE3 fence ensures ALL prior remote data writes
    /// are visible before the flag is written.
    ///
    /// @param destRank  Destination rank to signal
    /// @param eventID   Slot index [0, slotsPerRank_), typically coreIdx or a domain-specific index
    /// @param phase     PHASE_ENTRY (kernel entered) or PHASE_DONE (compute complete)
    __aicore__ inline void SetFlag(uint32_t destRank, uint32_t eventID, int32_t phase)
    {
        int64_t flagValue = MakeFlagValue(magic_, phase);

        auto remotePtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(flagBaseGM_, destRank));
        uint64_t slotOffset = GetSlotOffset(epRankId_, eventID, magic_);

        GlobalTensor<int64_t> remoteFlagGT;
        remoteFlagGT.SetGlobalBuffer((__gm__ int64_t *)(remotePtr + slotOffset), FLAG_ELEM_NUM);

        LocalTensor<int64_t> ubFlag = tBuf_.GetWithOffset<int64_t>(FLAG_ELEM_NUM, 0);

        // MTE3 fence: complete all prior UB→GM writes (token data)
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        // MTE2 fence: complete all prior GM→UB reads
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        // Prepare flag value in UB
        ubFlag.SetValue(0, flagValue);
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

        // Write flag from UB to remote GM
        DataCopy(remoteFlagGT, ubFlag, FLAG_ELEM_NUM);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    /// Poll local shmem for srcRank's eventID until flag matches (magic, phase).
    ///
    /// @param srcRank   Source rank whose flag to wait for
    /// @param eventID   Slot index [0, slotsPerRank_)
    /// @param phase     PHASE_ENTRY or PHASE_DONE
    __aicore__ inline void WaitFlag(uint32_t srcRank, uint32_t eventID, int32_t phase)
    {
        WaitFlagImpl(srcRank, eventID, magic_, phase);
    }

    /// Non-blocking check — returns true if the flag matches.
    __aicore__ inline bool CheckFlag(uint32_t srcRank, uint32_t eventID, int32_t phase)
    {
        return CheckFlagImpl(srcRank, eventID, magic_, phase);
    }

    /// Wait for a flag with an explicit magic value (cross-kernel sync).
    /// Use when the target kernel is not the immediately previous one.
    ///
    /// @param srcRank   Source rank
    /// @param eventID   Slot index
    /// @param magic     The magic value of the kernel whose flag to wait for
    /// @param phase     PHASE_ENTRY or PHASE_DONE
    __aicore__ inline void WaitFlagWithMagic(uint32_t srcRank, uint32_t eventID,
                                              int32_t magic, int32_t phase)
    {
        WaitFlagImpl(srcRank, eventID, magic, phase);
    }

    // ========================================================================
    // Batched Operations
    // ========================================================================

    /// Set flag on [startRank, endRank) with the same eventID.
    /// Skips self-rank. MTE3 fence before the batch ensures data ordering.
    __aicore__ inline void SetFlagBatch(uint32_t startRank, uint32_t endRank, uint32_t eventID,
                                        int32_t phase)
    {
        int64_t flagValue = MakeFlagValue(magic_, phase);
        LocalTensor<int64_t> ubFlag = tBuf_.GetWithOffset<int64_t>(FLAG_ELEM_NUM, 0);

        // Fence: ensure all prior data writes are complete
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        // Prepare flag in UB (once)
        ubFlag.SetValue(0, flagValue);
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

        uint64_t slotOffset = GetSlotOffset(epRankId_, eventID, magic_);

        // Write flag to each destination rank's shmem
        for (uint32_t r = startRank; r < endRank; r++) {
            if (r == epRankId_) {
                continue;
            }
            auto remotePtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(flagBaseGM_, r));

            GlobalTensor<int64_t> remoteFlagGT;
            remoteFlagGT.SetGlobalBuffer((__gm__ int64_t *)(remotePtr + slotOffset), FLAG_ELEM_NUM);
            DataCopy(remoteFlagGT, ubFlag, FLAG_ELEM_NUM);
        }

        // Fence: ensure all flag writes are complete before returning
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    /// Wait for flags from [startRank, endRank) at the same eventID.
    /// Skips self-rank.
    __aicore__ inline void WaitFlagBatch(uint32_t startRank, uint32_t endRank, uint32_t eventID,
                                          int32_t phase)
    {
        for (uint32_t r = startRank; r < endRank; r++) {
            if (r == epRankId_) {
                continue;
            }
            WaitFlagImpl(r, eventID, magic_, phase);
        }
    }

    /// Wait for consecutive eventIDs [startEvent, startEvent+count) from srcRank.
    /// Per-core sync: WaitFlagRange(srcRank, 0, blockNum) waits for all cores.
    __aicore__ inline void WaitFlagRange(uint32_t srcRank, uint32_t startEvent, uint32_t count,
                                          int32_t phase)
    {
        WaitFlagRangeImpl(srcRank, startEvent, count, magic_, phase);
    }

    /// Wait for flags from a specific kernel (explicit magic) across a range.
    __aicore__ inline void WaitFlagRangeWithMagic(uint32_t srcRank, uint32_t startEvent, uint32_t count,
                                                   int32_t magic, int32_t phase)
    {
        WaitFlagRangeImpl(srcRank, startEvent, count, magic, phase);
    }

    /// Wait for all events of ALL source ranks (current kernel).
    /// Splits source ranks across cores — no SyncAll needed.
    __aicore__ inline void WaitAllRankAllEvent(int32_t phase)
    {
        uint32_t perCore = 0U, startRank = 0U, endRank = 0U;
        SplitCoreCal(epWorldSize_, perCore, startRank, endRank);

        for (uint32_t r = startRank; r < endRank; r++) {
            if (r == epRankId_) {
                continue;
            }
            WaitFlagRangeImpl(r, 0, slotsPerRank_, magic_, phase);
        }
    }

    /// Set flag to ALL destination ranks from this core.
    /// Each core uses coreIdx as eventID — every core signals ALL ranks.
    __aicore__ inline void SetAllRankCoreFlag(int32_t phase)
    {
        SetFlagBatch(0, epWorldSize_, coreIdx_, phase);
    }

    // ========================================================================
    // Barrier
    // ========================================================================

    /// All-to-all barrier across the communication domain using shmem flags.
    ///
    /// Uses ShmemSyncFlag's internal magic for flag uniqueness and magic % 2
    /// for pingpong buffer selection. Barrier area is located after the flag
    /// pingpong buffers.
    ///
    /// Flow: SyncAll → Write flags to all ranks → Wait all ranks → SyncAll
    ///
    /// DAG guarantee: Write(Ri→Rj) → Wait(Rj←Ri), no reverse edges.
    /// Ranks are split across cores — no inter-core dependency.
    __aicore__ inline void BarrierAll()
    {
        // Step 1: intra-rank sync — ensure all local compute is complete
        SyncAll<true>();

        // Split ranks across cores
        uint32_t perCore = 0U, startRank = 0U, endRank = 0U;
        SplitCoreCal(epWorldSize_, perCore, startRank, endRank);

        // Barrier flag value: (magic << 32) | barrierSeq
        // Auto-incrementing barrierSeq_ ensures consecutive BarrierAll calls
        // within the same magic produce distinct flag values.
        const int64_t barrierFlag = MakeFlagValue(magic_, ++barrierSeq_);
        LocalTensor<int64_t> ubFlag = tBuf_.GetWithOffset<int64_t>(FLAG_ELEM_NUM, 0);

        // Select pingpong buffer for barrier: magic % 2
        uint64_t bufIdx = static_cast<uint64_t>(static_cast<uint32_t>(magic_) & 1U);
        uint64_t barrierBufOffset = barrierBaseOffset_ + bufIdx * barrierBufSize_;

        // Step 2a: Write — signal all assigned remote ranks
        ubFlag.SetValue(0, barrierFlag);
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

        for (uint32_t r = startRank; r < endRank; r++) {
            if (r == epRankId_) {
                continue;
            }
            auto remotePtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(flagBaseGM_, r));
            GlobalTensor<int64_t> remoteSlot;
            remoteSlot.SetGlobalBuffer(
                (__gm__ int64_t *)(remotePtr + barrierBufOffset + epRankId_ * FLAG_SLOT_SIZE),
                FLAG_ELEM_NUM);
            DataCopy(remoteSlot, ubFlag, FLAG_ELEM_NUM);
        }

        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        // Step 2b: Wait — poll local slots for all assigned remote ranks
        for (uint32_t r = startRank; r < endRank; r++) {
            if (r == epRankId_) {
                continue;
            }
            GlobalTensor<int64_t> localSlot;
            localSlot.SetGlobalBuffer(
                (__gm__ int64_t *)(flagBaseGM_ + barrierBufOffset + r * FLAG_SLOT_SIZE),
                FLAG_ELEM_NUM);

            int64_t observed;
            do {
                DataCopy(ubFlag, localSlot, FLAG_ELEM_NUM);
                AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                observed = ubFlag.GetValue(0);
            } while (observed != barrierFlag);
        }

        // Step 3: intra-rank sync — ensure all cores have completed barrier
        SyncAll<true>();
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum,
                                        uint32_t &startIdx, uint32_t &endIdx)
    {
        perCoreNum = totalNum / coreNum_;
        uint32_t remainder = totalNum % coreNum_;
        startIdx = perCoreNum * coreIdx_;
        if (coreIdx_ < remainder) {
            perCoreNum++;
            startIdx += coreIdx_;
        } else {
            startIdx += remainder;
        }
        endIdx = startIdx + perCoreNum;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    __aicore__ inline int32_t GetMagic() const { return magic_; }
    __aicore__ inline uint32_t GetCoreIdx() const { return coreIdx_; }
    __aicore__ inline uint32_t GetCoreNum() const { return coreNum_; }

private:
    /// Byte offset from flagBaseGM_ to slot [srcRank][eventID] in the buffer
    /// selected by the given magic value (magic % 2 for ping-pong).
    __aicore__ inline uint64_t GetSlotOffset(uint32_t srcRank, uint32_t eventID, int32_t magic)
    {
        uint64_t bufIdx = static_cast<uint64_t>(static_cast<uint32_t>(magic) & 1U);
        return flagBufBaseOffset_
             + bufIdx * bufSize_
             + static_cast<uint64_t>(srcRank) * slotsPerRank_ * FLAG_SLOT_SIZE
             + static_cast<uint64_t>(eventID) * FLAG_SLOT_SIZE;
    }

    /// Internal: poll local shmem until flag == (magic, phase).
    __aicore__ inline void WaitFlagImpl(uint32_t srcRank, uint32_t eventID, int32_t magic, int32_t phase)
    {
        int64_t expected = MakeFlagValue(magic, phase);
        uint64_t slotOffset = GetSlotOffset(srcRank, eventID, magic);

        GlobalTensor<int64_t> localFlagGT;
        localFlagGT.SetGlobalBuffer((__gm__ int64_t *)(flagBaseGM_ + slotOffset), FLAG_ELEM_NUM);

        LocalTensor<int64_t> ubFlag = tBuf_.GetWithOffset<int64_t>(FLAG_ELEM_NUM, 0);
        int64_t observed;
        do {
            DataCopy(ubFlag, localFlagGT, FLAG_ELEM_NUM);
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            observed = ubFlag.GetValue(0);
        } while (observed != expected);
    }

    /// Internal: non-blocking flag check.
    __aicore__ inline bool CheckFlagImpl(uint32_t srcRank, uint32_t eventID, int32_t magic, int32_t phase)
    {
        int64_t expected = MakeFlagValue(magic, phase);
        uint64_t slotOffset = GetSlotOffset(srcRank, eventID, magic);

        GlobalTensor<int64_t> localFlagGT;
        localFlagGT.SetGlobalBuffer((__gm__ int64_t *)(flagBaseGM_ + slotOffset), FLAG_ELEM_NUM);

        LocalTensor<int64_t> ubFlag = tBuf_.GetWithOffset<int64_t>(FLAG_ELEM_NUM, 0);

        DataCopy(ubFlag, localFlagGT, FLAG_ELEM_NUM);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        return ubFlag.GetValue(0) == expected;
    }

    /// Internal: wait for a range of event slots.
    __aicore__ inline void WaitFlagRangeImpl(uint32_t srcRank, uint32_t startEvent, uint32_t count,
                                              int32_t magic, int32_t phase)
    {
        int64_t expected = MakeFlagValue(magic, phase);
        LocalTensor<int64_t> ubFlag = tBuf_.GetWithOffset<int64_t>(FLAG_ELEM_NUM, 0);

        for (uint32_t i = 0; i < count; i++) {
            uint64_t slotOffset = GetSlotOffset(srcRank, startEvent + i, magic);
            GlobalTensor<int64_t> localFlagGT;
            localFlagGT.SetGlobalBuffer((__gm__ int64_t *)(flagBaseGM_ + slotOffset), FLAG_ELEM_NUM);

            int64_t observed;
            do {
                DataCopy(ubFlag, localFlagGT, FLAG_ELEM_NUM);
                AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                observed = ubFlag.GetValue(0);
            } while (observed != expected);
        }
    }

    GM_ADDR gvaGM_{0};
    GM_ADDR flagBaseGM_{0};          // gvaGM_ + FLAG_AREA_BASE
    uint32_t epRankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t coreIdx_{0};
    uint32_t coreNum_{0};
    uint32_t slotsPerRank_{0};       // event slots per source rank
    uint64_t barrierBufSize_{0};     // single pingpong barrier buffer size (epWorldSize × 32B)
    uint64_t barrierBaseOffset_{0};  // offset from flagBaseGM_ to barrier area (= MAGIC_AREA_SIZE)
    uint64_t bufSize_{0};            // single pingpong flag buffer size in bytes
    uint64_t flagBufBaseOffset_{0};  // offset from flagBaseGM_ to flag buffer area
    int32_t magic_{0};               // monotonically incrementing, +1 per kernel call
    int32_t barrierSeq_{0};          // per-magic barrier sequence, reset in IncrementMagic
    TBuf<QuePosition::VECCALC> tBuf_;
};

}  // namespace ShmemSyncFlagImpl

#endif  // SHMEM_SYNC_FLAG_H
