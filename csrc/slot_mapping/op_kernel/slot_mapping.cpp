/**
 * Maps token positions to physical KV-cache slot ids; replaces the upstream
 * Triton implementation on the Ascend NPU path. Output dtype is int32 to match
 * the slot_mapping buffer in `vllm_ascend/worker/block_table.py` (which the
 * downstream `reshape_and_cache_bnsd` kernel also reads as int32).
 *
 * Multi-core: each core processes a contiguous range [posStart, posEnd). The
 * range is rounded up to a 16-slot boundary (= one 64 B cache line of int32)
 * to avoid cross-core false sharing.
 *
 * Algorithm overview (cpWs == 1 fast path, the common case):
 *   blockIdx2 = pos >> log2(blockSize)              ; pos / blockSize
 *   offset    = pos & (blockSize - 1)               ; pos % blockSize
 *   slot_id   = (blockTable[req, blockIdx2] << log2(blockSize)) | offset
 * The general path (CP > 1 or non-power-of-2 blockSize) falls back to the
 * full div/mod expression.
 *
 * dav_c220 vector-API constraints we worked around:
 *   - Divs<int64> / Compare<int64> / Ands / Duplicate<int64> are unsupported.
 *   - int32 vec-op `count` parameter has unclear semantics for non-BLK-aligned
 *     sizes; not used here.
 * The inner loop is therefore scalar; one tile-wide scalar narrow casts the
 * int64 slot ids to int32 just before DataCopyPad writes them out.
 *
 * SlotMappingTilingData / GET_TILING_DATA are injected by CANN from
 * op_host/slot_mapping_tiling.h.
 */
#include "kernel_operator.h"

using namespace AscendC;

class SlotMappingKernel {
public:
    __aicore__ inline SlotMappingKernel() {}

    __aicore__ inline void Init(
        GM_ADDR queryStartLoc, GM_ADDR positions,
        GM_ADDR blockTable, GM_ADDR slotMapping,
        const SlotMappingTilingData* td)
    {
        // ---------- Tiling param ----------
        numReqs_ = td->numReqs;
        numTokens_ = td->numTokens;
        maxNumTokens_ = td->maxNumTokens;
        blockSize_ = td->blockSize;
        padId_ = td->padId;
        stride_ = td->blockTableStride;
        cpWs_ = td->totalCpWorldSize;
        cpRank_ = td->totalCpRank;
        ilv_ = td->cpKvCacheInterleaveSize;

        int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        int32_t blockNumCores = static_cast<int32_t>(GetBlockNum());
        if (blockNumCores < 1) blockNumCores = 1;

        // 16 × int32 = 64 B = one cache line. Per-core ranges are aligned so
        // that no two cores write to the same cache line (no false sharing).
        constexpr int32_t kSlotAlignment = 16;
        int32_t tilePerCore = (maxNumTokens_ + blockNumCores - 1) / blockNumCores;
        if (tilePerCore < kSlotAlignment) {
            tilePerCore = kSlotAlignment;
        } else {
            tilePerCore = ((tilePerCore + kSlotAlignment - 1) / kSlotAlignment) * kSlotAlignment;
        }
        posStart_ = blockIdx * tilePerCore;
        posEnd_ = posStart_ + tilePerCore;
        if (posEnd_ > maxNumTokens_) posEnd_ = maxNumTokens_;
        myNumPos_ = posEnd_ - posStart_;
        if (myNumPos_ <= 0) {
            myNumPos_ = 0;
            return;
        }

        gmQsl_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc), numReqs_ + 1);
        gmPos_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions), maxNumTokens_);
        gmBt_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable),
                              static_cast<uint64_t>(numReqs_) * static_cast<uint64_t>(stride_));
        gmSlot_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMapping), maxNumTokens_);

        int32_t qslElems = numReqs_ + 1;
        uint32_t qslBytes = AlignUp(static_cast<uint32_t>(qslElems) * sizeof(int32_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(bufQsl_, qslBytes);

        int64_t btElems64 = static_cast<int64_t>(numReqs_) * static_cast<int64_t>(stride_);
        btElems_ = static_cast<int32_t>(btElems64);
        uint32_t btBytes = AlignUp(static_cast<uint32_t>(btElems_) * sizeof(int32_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(bufBt_, btBytes);

        uint32_t posBytes = AlignUp(static_cast<uint32_t>(myNumPos_) * sizeof(int64_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(bufPos_, posBytes);

        pipe_.InitBuffer(bufSlot_, posBytes);

        // int32 output tile (half the byte count of bufSlot_): scalar narrow
        // casts the int64 slot ids into this buffer right before DataCopyPad.
        uint32_t slot32Bytes = AlignUp(static_cast<uint32_t>(myNumPos_) * sizeof(int32_t), ONE_BLK_SIZE);
        pipe_.InitBuffer(bufSlot32_, slot32Bytes);

        // The three input loads (qsl / blockTable / positions) are independent,
        // so we issue them back-to-back on MTE2 and use a single barrier
        // afterwards rather than one per copy.

        // ---------- Compute all alignment / validInTile_ first (host scalar) ----------
        constexpr int32_t kInt32PerBlk = ONE_BLK_SIZE / static_cast<int32_t>(sizeof(int32_t));  // 8
        constexpr int32_t kInt64PerBlk = ONE_BLK_SIZE / static_cast<int32_t>(sizeof(int64_t));  // 4
        int32_t qslAligned = (qslElems + kInt32PerBlk - 1) / kInt32PerBlk * kInt32PerBlk;
        int32_t btAligned = (btElems_ > 0) ? (btElems_ + kInt32PerBlk - 1) / kInt32PerBlk * kInt32PerBlk : 0;

        int32_t validInTile = posEnd_ < numTokens_ ? myNumPos_ : (numTokens_ - posStart_);
        if (validInTile < 0) validInTile = 0;
        if (validInTile > myNumPos_) validInTile = myNumPos_;
        validInTile_ = validInTile;
        int32_t posAligned = (validInTile_ > 0) ? (validInTile_ + kInt64PerBlk - 1) / kInt64PerBlk * kInt64PerBlk : 0;

        // ---------- Issue three DataCopies back-to-back on MTE2 ----------
        LocalTensor<int32_t> lqsl = bufQsl_.Get<int32_t>();
        DataCopy(lqsl, gmQsl_, qslAligned);

        if (btElems_ > 0) {
            LocalTensor<int32_t> lbt = bufBt_.Get<int32_t>();
            DataCopy(lbt, gmBt_, btAligned);
        }

        if (validInTile_ > 0) {
            LocalTensor<int64_t> lpos = bufPos_.Get<int64_t>();
            DataCopy(lpos, gmPos_[posStart_], posAligned);
        }

        // Single barrier waits for all MTE2 loads before Process() starts
        // reading via scalar GetValue.
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline int32_t FindReqIdx(int32_t tokIdx, LocalTensor<int32_t>& lqsl)
    {
        int32_t left = 0;
        int32_t right = numReqs_;
        while (left < right) {
            int32_t mid = (left + right) >> 1;
            if (lqsl.GetValue(mid) <= tokIdx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        int32_t r = left - 1;
        if (r < 0) r = 0;
        if (r >= numReqs_) r = numReqs_ - 1;
        return r;
    }

    static __aicore__ inline bool IsPow2(int32_t x)
    {
        return x > 0 && (x & (x - 1)) == 0;
    }

    static __aicore__ inline int32_t IntLog2(int32_t x)
    {
        int32_t r = 0;
        while ((1 << r) < x) ++r;
        return r;
    }

    __aicore__ inline void Process()
    {
        if (myNumPos_ <= 0) return;

        LocalTensor<int32_t> lqsl = bufQsl_.Get<int32_t>();
        LocalTensor<int32_t> lbt = bufBt_.Get<int32_t>();
        LocalTensor<int64_t> lpos = bufPos_.Get<int64_t>();
        LocalTensor<int64_t> lslot = bufSlot_.Get<int64_t>();

        const int64_t padIdI64 = static_cast<int64_t>(padId_);
        const int64_t vbs = static_cast<int64_t>(blockSize_) * static_cast<int64_t>(cpWs_);
        const int64_t intTimesCp = static_cast<int64_t>(cpWs_) * static_cast<int64_t>(ilv_);

        int32_t reqIdx = FindReqIdx(posStart_, lqsl);
        int32_t nextBoundary = (reqIdx + 1 < numReqs_) ? lqsl.GetValue(reqIdx + 1) : 2147483647;
        const int32_t lastReq = numReqs_ - 1;

        const bool fastPath = (cpWs_ == 1) && IsPow2(blockSize_);
        if (fastPath) {
            const int32_t bsShift = IntLog2(blockSize_);
            const int64_t bsMask = static_cast<int64_t>(blockSize_) - 1;

            for (int32_t i = 0; i < validInTile_; ++i) {
                int32_t globalIdx = posStart_ + i;
                while (reqIdx < lastReq && globalIdx >= nextBoundary) {
                    ++reqIdx;
                    nextBoundary = (reqIdx + 1 < numReqs_) ? lqsl.GetValue(reqIdx + 1) : 2147483647;
                }
                int64_t globalPos = lpos.GetValue(i);

                int64_t blockIdx2 = globalPos >> bsShift;
                int64_t offset = globalPos & bsMask;
                int32_t blockNum = lbt.GetValue(
                    static_cast<uint32_t>(reqIdx * stride_) +
                    static_cast<uint32_t>(blockIdx2));
                int64_t slotId = (static_cast<int64_t>(blockNum) << bsShift) | offset;
                lslot.SetValue(i, slotId);
            }
        } else {
            for (int32_t i = 0; i < validInTile_; ++i) {
                int32_t globalIdx = posStart_ + i;
                while (reqIdx < lastReq && globalIdx >= nextBoundary) {
                    ++reqIdx;
                    nextBoundary = (reqIdx + 1 < numReqs_) ? lqsl.GetValue(reqIdx + 1) : 2147483647;
                }
                int64_t globalPos = lpos.GetValue(i);

                int64_t blockIdx2 = globalPos / vbs;
                int32_t blockNum = lbt.GetValue(
                    static_cast<uint32_t>(reqIdx * stride_) +
                    static_cast<uint32_t>(blockIdx2));
                int64_t offset = globalPos - blockIdx2 * vbs;
                bool isLocal = ((offset / ilv_) % cpWs_) == cpRank_;
                int64_t localOffset = (offset / intTimesCp) * ilv_ + (offset % ilv_);
                int64_t slotId = static_cast<int64_t>(blockNum) * static_cast<int64_t>(blockSize_) +
                                 localOffset;
                lslot.SetValue(i, isLocal ? slotId : padIdI64);
            }
        }

        for (int32_t i = validInTile_; i < myNumPos_; ++i) {
            lslot.SetValue(i, padIdI64);
        }
        // No barrier between the padding loop and the cast loop below: both
        // run on the scalar pipe and are ordered within it.

        // Scalar narrow int64 -> int32. Production slot_id stays well below
        // 2^31 (block_num × block_size + offset typically ~10^7); the kernel
        // contract is int32 output. myNumPos_ ≤ 16 (kSlotAlignment), so this
        // loop is on the order of tens of cycles.
        LocalTensor<int32_t> lslot32 = bufSlot32_.Get<int32_t>();
        for (int32_t i = 0; i < myNumPos_; ++i) {
            lslot32.SetValue(i, static_cast<int32_t>(lslot.GetValue(i)));
        }
        // Cross-pipe sync (S -> MTE3): the cast loop writes lslot32 on the
        // scalar pipe; DataCopyPad reads it on MTE3.
        pipe_barrier(PIPE_ALL);

        // One DataCopyPad writes the whole tile back to GM (int32, half the
        // bytes of the original int64 layout).
        uint32_t totalBytes = static_cast<uint32_t>(myNumPos_) * sizeof(int32_t);
        DataCopyPad(gmSlot_[posStart_], lslot32, DataCopyExtParams(1, totalBytes, 0, 0, 0));
    }

private:
    static __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a)
    {
        return (x + a - 1) / a * a;
    }

    // GM Tensor
    GlobalTensor<int32_t> gmQsl_;
    GlobalTensor<int64_t> gmPos_;
    GlobalTensor<int32_t> gmBt_;
    GlobalTensor<int32_t> gmSlot_;

    // UB Buffers
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> bufQsl_;
    TBuf<QuePosition::VECCALC> bufBt_;
    TBuf<QuePosition::VECCALC> bufPos_;
    TBuf<QuePosition::VECCALC> bufSlot_;    // int64 internal slot ids
    TBuf<QuePosition::VECCALC> bufSlot32_;  // int32 output tile (DataCopyPad source)

    // Tiling param cache
    int32_t numReqs_ = 0;
    int32_t numTokens_ = 0;
    int32_t maxNumTokens_ = 0;
    int32_t blockSize_ = 1;
    int32_t padId_ = -1;
    int32_t stride_ = 1;
    int32_t cpWs_ = 1;
    int32_t cpRank_ = 0;
    int32_t ilv_ = 1;

    int32_t posStart_ = 0;
    int32_t posEnd_ = 0;
    int32_t myNumPos_ = 0;
    int32_t validInTile_ = 0;
    int32_t btElems_ = 0;
};

extern "C" __global__ __aicore__ void slot_mapping(
    GM_ADDR queryStartLoc, GM_ADDR positions,
    GM_ADDR blockTable, GM_ADDR slotMapping,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    SlotMappingKernel ker;
    ker.Init(queryStartLoc, positions, blockTable, slotMapping, &tilingData);
    ker.Process();
}
