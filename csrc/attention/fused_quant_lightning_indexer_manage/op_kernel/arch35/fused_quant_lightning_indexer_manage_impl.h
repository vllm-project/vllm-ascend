/**
 * A5 MTP0..13 C8 LightningIndexer with the standardized ABI.
 * request_state=-3 publishes identity slots; request_state=-2 performs the
 * correctness-first full-visible first-fill cache construction; and
 * request_state=-1 performs steady cache management, including normalization
 * of an identity row inherited directly from request_state=-3.
 */

#include "kernel_operator.h"
#include "lim_simt_config.h"
#include "fused_quant_lightning_indexer_manage_tiling.h"
#include "fused_quant_lightning_indexer_manage_qli.h"
#include "lim_cache_budget.h"
#include "lim_pool_contract.h"
#include "lim_payload_layout.h"
#include "lim_request_contract.h"
#include "lim_request_limits.h"
#include "fused_quant_lightning_indexer_manage_hit_sort_vf.h"
#include "fused_quant_lightning_indexer_manage_union.h"

namespace {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void SyncPipes()
{
    const event_t eventId = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

// Every AIV makes the same read-only decision before any owner changes a row.
// This keeps steady sparse batches out of the pre-QLI barrier entirely.
__aicore__ inline bool NeedsIdentityNormalization(
    GM_ADDR requestState, GM_ADDR numCacheTokens, GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool, const FusedQuantLightningIndexerManageTilingData *tiling,
    const lim_request_contract::Metadata &metadata)
{
    GlobalTensor<int32_t> states, budgets, entries, pool;
    states.SetGlobalBuffer((__gm__ int32_t *)requestState);
    budgets.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
    entries.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
    pool.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
    int32_t priorMaximum = 0;
    for (uint32_t batch = 0U; batch < tiling->batchSize; ++batch) {
        const int32_t end = metadata.QueryEnd(batch);
        const bool validSparse = states.GetValue(batch) == lim_request_limits::STEADY_STATE &&
            metadata.Valid(batch, priorMaximum);
        if (end > priorMaximum) priorMaximum = end;
        if (!validSparse) continue;
        const uint32_t budget = static_cast<uint32_t>(budgets.GetValue(batch));
        if (budget >= tiling->sourceCapacity) continue;
        const uint64_t base = static_cast<uint64_t>(
            entries.GetValue(batch)) * tiling->sourceCapacity;
        if (pool.GetValue(base + budget) >= static_cast<int32_t>(budget))
            return true;
    }
    return false;
}

// A3-style direct -3 -> -1 transition. The caller owns a valid identity row;
// normal steady rows avoid a full-row scan on every decode.
__aicore__ inline void NormalizeIdentityRows(
    GM_ADDR requestState, GM_ADDR numCacheTokens, GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool, const FusedQuantLightningIndexerManageTilingData *tiling,
    const lim_request_contract::Metadata &metadata, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if ((GetBlockIdx() & 1U) != 0U) return;
        constexpr uint32_t CHUNK = 512U;
        GlobalTensor<int32_t> states, budgets, entries, pool;
        states.SetGlobalBuffer((__gm__ int32_t *)requestState);
        budgets.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        entries.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        pool.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        TBuf<TPosition::VECCALC> invalidBuf;
        bool initialized = false;
        for (uint32_t batch = GetBlockIdx() / 2U;
             batch < tiling->batchSize; batch += tiling->usedCoreNum) {
            if (states.GetValue(batch) != lim_request_limits::STEADY_STATE ||
                !metadata.Valid(batch)) continue;
            const uint32_t budget = static_cast<uint32_t>(budgets.GetValue(batch));
            if (budget >= tiling->sourceCapacity) continue;
            const uint64_t base = static_cast<uint64_t>(
                entries.GetValue(batch)) * tiling->sourceCapacity;
            if (pool.GetValue(base + budget) < static_cast<int32_t>(budget))
                continue;
            const uint32_t probeLine = budget & ~31U;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(pool[base + probeLine]);
            if (!initialized) {
                pipe->InitBuffer(invalidBuf, CHUNK * sizeof(int32_t));
                // Preserve the exact INT32_MIN bit pattern. A float -0.0F
                // Duplicate may be canonicalized to +0.0F on A5.
                LocalTensor<int32_t> invalid = invalidBuf.Get<int32_t>();
                Duplicate(invalid, lim_pool_contract::INVALID_POOL_SLOT, CHUNK);
                PipeBarrier<PIPE_V>();
                initialized = true;
            }
            LocalTensor<int32_t> invalid = invalidBuf.Get<int32_t>();
            SyncPipes<HardEvent::V_MTE3>();
            for (uint32_t source = budget; source < tiling->sourceCapacity;
                 source += CHUNK) {
                const uint32_t valid = tiling->sourceCapacity - source < CHUNK
                    ? tiling->sourceCapacity - source : CHUNK;
                DataCopyPad(pool[base + source], invalid,
                    {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
                SyncPipes<HardEvent::MTE3_V>();
            }
        }
    }
}

constexpr uint32_t BLOCK_SIZE = lim_request_limits::BLOCK_SIZE;
constexpr uint32_t TOPK = lim_request_limits::TOPK;
constexpr uint32_t MISS_CAPACITY = lim_request_limits::MISS_CAPACITY;
constexpr uint32_t MAX_QUERIES_PER_REQUEST = lim_request_limits::MAX_ROUTES;
static_assert(MAX_QUERIES_PER_REQUEST * TOPK <= MISS_CAPACITY,
              "wide first-fill route staging exceeds the public miss list");

class NonOffloadPublisher {
public:
    __aicore__ inline void Init(
        TPipe *pipe,
        GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey,
        GM_ADDR requestState, GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
        GM_ADDR topkSrcIds, GM_ADDR topkDstSlots, GM_ADDR topkMissCounts,
        GM_ADDR missCounts,
        const FusedQuantLightningIndexerManageTilingData *tiling)
    {
        queryEnds_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        keyLengths_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
        states_.SetGlobalBuffer((__gm__ int32_t *)requestState);
        entries_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        pool_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        topkSources_.SetGlobalBuffer((__gm__ int32_t *)topkSrcIds);
        topkDestinations_.SetGlobalBuffer((__gm__ int32_t *)topkDstSlots);
        routeMissCounts_.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
        requestMissCounts_.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        pipe_ = pipe;
        tiling_ = tiling;
    }

    __aicore__ inline void Process(uint32_t owner, uint32_t owners)
    {
        for (uint32_t batch = owner; batch < tiling_->batchSize;
             batch += owners) {
            const int32_t state = states_.GetValue(batch);
            if (state == lim_request_limits::FIRST_FILL_STATE ||
                state == lim_request_limits::STEADY_STATE) {
                continue;
            }
            int32_t startValue = 0;
            int32_t endValue = 0;
            int32_t row = -1;
            const bool valid = LoadRequest(
                batch, startValue, endValue, row);
            uint32_t safeStart = 0U, safeEnd = 0U;
            SafeRouteRange(batch, safeStart, safeEnd);
            for (uint32_t route = safeStart; route < safeEnd; ++route) {
                const uint64_t base = static_cast<uint64_t>(route) * TOPK;
                if (!valid) {
                    for (uint32_t position = 0; position < TOPK; ++position) {
                        topkSources_.SetValue(base + position, -1);
                        topkDestinations_.SetValue(base + position, -1);
                    }
                    continue;
                }
                // The shared LI path has already published both dense rows.
            }
        }
    }

    // The standard lifecycle requires an identity pool on every -3 call.
    // Port the A3 standard operator's disjoint 4096-token AIV chunks instead
    // of issuing one scalar GM store per source on a single request owner.
    __aicore__ inline void FillIdentityChunks(uint32_t owner, uint32_t owners)
    {
        constexpr uint32_t CHUNK = 4096U;
        if (owner * CHUNK >= tiling_->sourceCapacity) return;
        bool initialized = false;
        for (uint32_t batch = 0U; batch < tiling_->batchSize; ++batch) {
            if (states_.GetValue(batch) != lim_request_limits::IDENTITY_STATE) continue;
            int32_t startValue = 0, endValue = 0, row = -1;
            if (!LoadRequest(batch, startValue, endValue, row)) {
                continue;
            }
            if (!initialized) {
                pipe_->InitBuffer(identityBaseBuf_, CHUNK * sizeof(int32_t));
                pipe_->InitBuffer(identityValuesBuf_, CHUNK * sizeof(int32_t));
                LocalTensor<int32_t> base = identityBaseBuf_.Get<int32_t>();
                Arange<int32_t>(base, 0, 1, static_cast<int32_t>(CHUNK));
                PipeBarrier<PIPE_V>();
                initialized = true;
            }
            LocalTensor<int32_t> base = identityBaseBuf_.Get<int32_t>();
            LocalTensor<int32_t> values = identityValuesBuf_.Get<int32_t>();
            const uint64_t poolBase =
                static_cast<uint64_t>(row) * tiling_->sourceCapacity;
            for (uint32_t source = owner * CHUNK;
                 source < tiling_->sourceCapacity;
                 source += owners * CHUNK) {
                const uint32_t valid = tiling_->sourceCapacity - source < CHUNK
                    ? tiling_->sourceCapacity - source : CHUNK;
                Adds(values, base, static_cast<int32_t>(source), valid);
                PipeBarrier<PIPE_V>();
                SyncPipes<HardEvent::V_MTE3>();
                DataCopyPad(pool_[poolBase + source], values,
                    {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
                SyncPipes<HardEvent::MTE3_V>();
            }
        }
    }

    // Keep compact count tensors single-writer. Adjacent requests share a GM
    // cache line, so scalar writes from multiple AIVs can otherwise lose the
    // first route count at a core boundary.
    __aicore__ inline void PublishCounts()
    {
        for (uint32_t batch = 0; batch < tiling_->batchSize; ++batch) {
            const int32_t state = states_.GetValue(batch);
            if (state == lim_request_limits::FIRST_FILL_STATE ||
                state == lim_request_limits::STEADY_STATE) {
                continue;
            }
            uint32_t safeStart = 0U, safeEnd = 0U;
            SafeRouteRange(batch, safeStart, safeEnd);
            for (uint32_t route = safeStart; route < safeEnd; ++route) {
                routeMissCounts_.SetValue(route, 0);
            }
            requestMissCounts_.SetValue(batch, 0);
        }
    }

    // As in the A3 standard operator, homogeneous non-offload counts can be
    // published with contiguous DMA instead of one scalar GM store per route.
    // This remains single-writer to avoid adjacent-request false sharing.
    __aicore__ inline void PublishHomogeneousCounts()
    {
        constexpr uint32_t CHUNK = 256U;
        pipe_->InitBuffer(zeroCountsBuf_, CHUNK * sizeof(int32_t));
        LocalTensor<int32_t> zeros = zeroCountsBuf_.Get<int32_t>();
        Duplicate(zeros, 0, CHUNK);
        PipeBarrier<PIPE_V>();
        for (uint32_t start = 0U; start < tiling_->totalQueryRows;
             start += CHUNK) {
            const uint32_t valid = tiling_->totalQueryRows - start < CHUNK
                ? tiling_->totalQueryRows - start : CHUNK;
            SyncPipes<HardEvent::V_MTE3>();
            DataCopyPad(routeMissCounts_[start], zeros,
                {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
            SyncPipes<HardEvent::MTE3_V>();
        }
        for (uint32_t start = 0U; start < tiling_->batchSize;
             start += CHUNK) {
            const uint32_t valid = tiling_->batchSize - start < CHUNK
                ? tiling_->batchSize - start : CHUNK;
            SyncPipes<HardEvent::V_MTE3>();
            DataCopyPad(requestMissCounts_[start], zeros,
                {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
            SyncPipes<HardEvent::MTE3_V>();
        }
    }

private:
    __aicore__ inline void SafeRouteRange(
        uint32_t batch, uint32_t &safeStart, uint32_t &safeEnd) const
    {
        int32_t maximum = 0;
        for (uint32_t previous = 0U; previous < batch; ++previous) {
            const int32_t end = queryEnds_.GetValue(previous);
            if (end > maximum) maximum = end;
        }
        const int32_t capacity = static_cast<int32_t>(tiling_->totalQueryRows);
        if (maximum > capacity) maximum = capacity;
        safeStart = static_cast<uint32_t>(maximum);
        const int32_t rawEnd = queryEnds_.GetValue(batch);
        safeEnd = rawEnd > maximum && rawEnd <= capacity
            ? static_cast<uint32_t>(rawEnd) : safeStart;
    }

    __aicore__ inline bool LoadRequest(
        uint32_t batch, int32_t &startValue, int32_t &endValue,
        int32_t &row)
    {
        endValue = queryEnds_.GetValue(batch);
        startValue = batch == 0U ? 0 : queryEnds_.GetValue(batch - 1U);
        const int32_t state = states_.GetValue(batch);
        row = entries_.GetValue(batch);
        const int32_t keyLength = keyLengths_.GetValue(batch);
        uint32_t safeStart = 0U, safeEnd = 0U;
        SafeRouteRange(batch, safeStart, safeEnd);
        const bool validRange = startValue >= 0 && endValue > startValue &&
            endValue <= static_cast<int32_t>(tiling_->totalQueryRows) &&
            static_cast<uint32_t>(startValue) == safeStart &&
            static_cast<uint32_t>(endValue) == safeEnd &&
            (batch + 1U != tiling_->batchSize ||
             endValue == static_cast<int32_t>(tiling_->totalQueryRows)) &&
            endValue - startValue <=
                static_cast<int32_t>(MAX_QUERIES_PER_REQUEST);
        return validRange && state == lim_request_limits::IDENTITY_STATE && row >= 0 &&
            row < static_cast<int32_t>(tiling_->poolSize) &&
            keyLength >= endValue - startValue &&
            keyLength <= static_cast<int32_t>(tiling_->sourceCapacity);
    }

    TPipe *pipe_ = nullptr;
    const FusedQuantLightningIndexerManageTilingData *tiling_ = nullptr;
    TBuf<TPosition::VECCALC> identityBaseBuf_;
    TBuf<TPosition::VECCALC> identityValuesBuf_;
    TBuf<TPosition::VECCALC> zeroCountsBuf_;
    GlobalTensor<int32_t> queryEnds_;
    GlobalTensor<int32_t> keyLengths_;
    GlobalTensor<int32_t> states_;
    GlobalTensor<int32_t> entries_;
    GlobalTensor<int32_t> pool_;
    GlobalTensor<int32_t> topkSources_;
    GlobalTensor<int32_t> topkDestinations_;
    GlobalTensor<int32_t> routeMissCounts_;
    GlobalTensor<int32_t> requestMissCounts_;
};

class FirstFillPublisher {
public:
    __aicore__ inline void Init(
        TPipe *pipe, GM_ADDR userWorkspace,
        GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey,
        GM_ADDR offloadSeqLengthsKey, GM_ADDR numCacheTokens,
        GM_ADDR requestState, GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
        GM_ADDR topkSrcIds, GM_ADDR topkDstSlots, GM_ADDR topkMissCounts,
        GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
        const FusedQuantLightningIndexerManageTilingData *tiling)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        hasLongSource_ = tiling_->sourceCapacity >
            lim_payload_layout::PACKED_SOURCE_CAPACITY;
        metadata_.Init(actualSeqLengthsQuery, actualSeqLengthsKey,
                       offloadSeqLengthsKey, numCacheTokens,
                       requestState, reqPoolEntries, tiling_);
        queryEnds_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        keyLengths_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
        prefixLengths_.SetGlobalBuffer((__gm__ int32_t *)offloadSeqLengthsKey);
        cacheTokens_.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        states_.SetGlobalBuffer((__gm__ int32_t *)requestState);
        entries_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        pool_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        topkSources_.SetGlobalBuffer((__gm__ int32_t *)topkSrcIds);
        topkDestinations_.SetGlobalBuffer((__gm__ int32_t *)topkDstSlots);
        routeMissCounts_.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
        missSources_.SetGlobalBuffer((__gm__ int32_t *)missSrcIds);
        missDestinations_.SetGlobalBuffer((__gm__ int32_t *)missDstSlots);
        requestMissCounts_.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        requestStatus_.SetGlobalBuffer(
            (__gm__ int32_t *)(userWorkspace + tiling_->managementWorkspaceOffset));
        routeStatus_.SetGlobalBuffer(
            (__gm__ int32_t *)(userWorkspace + tiling_->managementWorkspaceOffset +
                static_cast<uint64_t>(tiling_->batchSize) * 32U));
        pipe_->InitBuffer(topkBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(slotBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(sourceBuf_, TOPK * sizeof(int32_t));
        pipe_->InitBuffer(cacheChunkBuf_, CACHE_CHUNK * sizeof(int32_t));
        pipe_->InitBuffer(hashBuf_, 16384U * sizeof(uint32_t));
        // Two high-source tags per byte keep total UB below the 216 KiB
        // limit reserved by the SIMT configuration.
        pipe_->InitBuffer(hashTagBuf_, 8192U * sizeof(uint8_t));
        // Extra pair row permits a four-way hardware merge of up to 8192
        // request misses without touching caller-owned GM output buffers.
        pipe_->InitBuffer(residentBuf_, 16384U * sizeof(int32_t));
        pipe_->InitBuffer(victimBuf_, 8192U * sizeof(int32_t));
        pipe_->InitBuffer(scalarBuf_, 32U);
    }

    __aicore__ inline void Process(uint32_t owner, uint32_t owners)
    {
        for (uint32_t batch = owner; batch < tiling_->batchSize;
             batch += owners) {
            const int32_t state = states_.GetValue(batch);
            if (state == lim_request_limits::FIRST_FILL_STATE) {
                ProcessRequest(batch);
            }
        }
    }

    __aicore__ inline void PublishCounts()
    {
        for (uint32_t batch = 0; batch < tiling_->batchSize; ++batch) {
            const int32_t state = states_.GetValue(batch);
            if (state != lim_request_limits::FIRST_FILL_STATE) {
                continue;
            }
            uint32_t safeStart = 0U, safeEnd = 0U;
            metadata_.SafeRouteRange(batch, safeStart, safeEnd);
            for (uint32_t route = safeStart; route < safeEnd; ++route) {
                routeMissCounts_.SetValue(
                    route, routeStatus_.GetValue(route * 8U));
            }
            // A single-request launch has no compact-count false sharing.
            // Its owner already published the count directly; avoid reading
            // a staged scalar from workspace for this degenerate batch.
            if (tiling_->batchSize != 1U)
                requestMissCounts_.SetValue(
                    batch, requestStatus_.GetValue(batch * 8U));
        }
    }

private:
    // The reusable DMA staging buffer holds one TopK row at a time.
    static constexpr uint32_t CACHE_CHUNK = TOPK;
    static constexpr uint32_t HASH_CAPACITY = 16384U;
    static constexpr uint32_t HASH_MASK = HASH_CAPACITY - 1U;
    static constexpr uint32_t TOKEN_MASK = lim_payload_layout::SOURCE_MASK;
    static constexpr uint32_t SLOT_SHIFT = lim_payload_layout::SOURCE_BITS;
    static constexpr uint32_t MISS_SLOT = lim_payload_layout::PENDING_MISS_SLOT;
    static constexpr uint32_t HASH_EMPTY = 0xffffffffU;
    static constexpr uint32_t TAIL_CAPACITY = lim_request_limits::MAX_TAIL_TOKENS;
    static constexpr int32_t WIDE_NO_SOURCE = 0x7fffffff;

    __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline bool QueryRange(
        uint32_t batch, uint32_t &start, uint32_t &end) const
    {
        const int32_t endValue = queryEnds_.GetValue(batch);
        const int32_t startValue = batch == 0U
            ? 0 : queryEnds_.GetValue(batch - 1U);
        if (startValue < 0 || endValue <= startValue ||
            endValue > static_cast<int32_t>(tiling_->totalQueryRows) ||
            endValue - startValue > static_cast<int32_t>(MAX_QUERIES_PER_REQUEST)) {
            return false;
        }
        start = static_cast<uint32_t>(startValue);
        end = static_cast<uint32_t>(endValue);
        return true;
    }

    __aicore__ inline bool Validate(
        uint32_t batch, uint32_t queryCount, int32_t row,
        uint32_t &prefixLength, uint32_t &finalLength,
        uint32_t &budget) const
    {
        const int32_t prefix = prefixLengths_.GetValue(batch);
        const int32_t finalValue = keyLengths_.GetValue(batch);
        const int32_t cache = cacheTokens_.GetValue(batch);
        if (row < 0 || row >= static_cast<int32_t>(tiling_->poolSize) ||
            prefix < static_cast<int32_t>(TOPK) ||
            prefix % static_cast<int32_t>(BLOCK_SIZE) != 0 ||
            finalValue < static_cast<int32_t>(queryCount) ||
            finalValue > static_cast<int32_t>(tiling_->sourceCapacity) ||
            prefix > finalValue - static_cast<int32_t>(queryCount - 1U) ||
            finalValue - prefix < 0 ||
            finalValue - prefix > static_cast<int32_t>(TAIL_CAPACITY) ||
            cache <= 0 || !lim_cache_budget::Valid(
                static_cast<uint32_t>(cache),
                static_cast<uint32_t>(prefix), queryCount)) {
            return false;
        }
        prefixLength = static_cast<uint32_t>(prefix);
        finalLength = static_cast<uint32_t>(finalValue);
        budget = static_cast<uint32_t>(cache);
        return true;
    }

    __aicore__ inline uint32_t PackHashEntry(uint32_t token, uint32_t slot) const
    {
        return (slot << SLOT_SHIFT) | (token & TOKEN_MASK);
    }

    __aicore__ inline void StoreHashEntry(
        LocalTensor<uint32_t> hash, uint32_t position,
        uint32_t token, uint32_t slot)
    {
        hash.SetValue(position, PackHashEntry(token, slot));
        if (!hasLongSource_) return;
        LocalTensor<uint8_t> tags = hashTagBuf_.Get<uint8_t>();
        const uint32_t byteIndex = position >> 1U;
        const uint32_t shift = (position & 1U) * 4U;
        const uint8_t mask = static_cast<uint8_t>(0xfU << shift);
        const uint8_t prior = tags.GetValue(byteIndex);
        tags.SetValue(byteIndex, static_cast<uint8_t>(
            (prior & static_cast<uint8_t>(~mask)) |
            ((token >> SLOT_SHIFT) << shift)));
    }

    __aicore__ inline uint32_t HashToken(
        uint32_t position, uint32_t entry)
    {
        if (!hasLongSource_) return entry & TOKEN_MASK;
        const uint32_t shift = (position & 1U) * 4U;
        const uint32_t tag =
            (hashTagBuf_.Get<uint8_t>().GetValue(position >> 1U) >> shift) & 0xfU;
        return (tag << SLOT_SHIFT) |
            (entry & TOKEN_MASK);
    }

    __aicore__ inline void InitHash(LocalTensor<uint32_t> hash)
    {
        Duplicate(hash, HASH_EMPTY, HASH_CAPACITY);
        if (hasLongSource_) {
            Duplicate(hashTagBuf_.Get<uint8_t>(), static_cast<uint8_t>(0),
                      HASH_CAPACITY / 2U);
        }
        SyncPipes<HardEvent::V_S>();
    }

    __aicore__ inline bool FindHashPosition(
        LocalTensor<uint32_t> hash, int32_t token,
        uint32_t &position, bool &found)
    {
        position = (static_cast<uint32_t>(token) * 2654435761U) & HASH_MASK;
        for (uint32_t probe = 0; probe < HASH_CAPACITY; ++probe) {
            const uint32_t entry = hash.GetValue(position);
            if (entry != HASH_EMPTY &&
                HashToken(position, entry) == static_cast<uint32_t>(token)) {
                found = true;
                return true;
            }
            if (entry == HASH_EMPTY) {
                found = false;
                return true;
            }
            position = (position + 1U) & HASH_MASK;
        }
        found = false;
        return false;
    }

    __aicore__ inline void LoadTopk(
        uint32_t queryRow, LocalTensor<int32_t> topk)
    {
        DataCopyExtParams copy{1, TOPK * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(topk, topkSources_[static_cast<uint64_t>(queryRow) * TOPK],
                    copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    }

    __aicore__ inline void StoreRange(
        GlobalTensor<int32_t> destination, LocalTensor<int32_t> source,
        uint32_t count)
    {
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
        DataCopyPad<int32_t, PaddingMode::Normal>(destination, source, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void FillGlobalRange(
        GlobalTensor<int32_t> destination, uint32_t count, int32_t value)
    {
        LocalTensor<int32_t> scratch = cacheChunkBuf_.Get<int32_t>();
        Duplicate(scratch, value, CACHE_CHUNK);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
        for (uint32_t offset = 0; offset < count; offset += CACHE_CHUNK) {
            const uint32_t length = MinU32(CACHE_CHUNK, count - offset);
            DataCopyExtParams copy{
                1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(destination[offset], scratch, copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        }
    }

    // First-fill owns the complete row. Reuse one immutable invalid block for all
    // disjoint DMA writes and wait only once before installing resident slots.
    __aicore__ inline void ClearFirstFillPool(uint64_t poolBase)
    {
        constexpr uint32_t CHUNK = CACHE_CHUNK;
        LocalTensor<int32_t> invalid = cacheChunkBuf_.Get<int32_t>();
        Duplicate(invalid, lim_pool_contract::INVALID_POOL_SLOT, CHUNK);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
        for (uint32_t source = 0; source < tiling_->sourceCapacity;
             source += CHUNK) {
            const uint32_t count = MinU32(
                CHUNK, tiling_->sourceCapacity - source);
            DataCopyPad(pool_[poolBase + source], invalid,
                {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
        }
        SyncPipes<HardEvent::MTE3_S>();
    }

    // Destinations are 0..C-1. Sources are streamed separately because C
    // may exceed the resident scratch buffer.
    __aicore__ inline void PublishFirstFillDestinations(
        uint32_t batch, uint32_t budget)
    {
        constexpr uint32_t CHUNK = CACHE_CHUNK;
        const uint64_t missBase = static_cast<uint64_t>(batch) * MISS_CAPACITY;
        LocalTensor<int32_t> destinations = cacheChunkBuf_.Get<int32_t>();
        for (uint32_t slot = 0; slot < budget; slot += CHUNK) {
            const uint32_t count = MinU32(CHUNK, budget - slot);
            Arange<int32_t>(destinations, static_cast<int32_t>(slot), 1,
                            static_cast<int32_t>(CHUNK));
            PipeBarrier<PIPE_V>();
            SyncPipes<HardEvent::V_MTE3>();
            DataCopyPad(missDestinations_[missBase + slot], destinations,
                {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
            SyncPipes<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void WriteScalar(
        GlobalTensor<int32_t> destination, int32_t value)
    {
        LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
        scalar.SetValue(0, value);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(destination, scalar, one);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void WritePool(uint64_t offset, int32_t value)
    {
        // Q1-4 first-fill owns the row: install residents through scalar GM
        // stores, with cache handoffs at the row boundaries instead of one
        // 4-byte DMA and two event waits for every resident.
        pool_.SetValue(offset, value);
    }

    __aicore__ inline void SyncFirstFillPoolCache(uint64_t poolBase)
    {
        // Whole-cache maintenance ignores the address. OUT targets GM and
        // leaves the UB scratch buffers used by the publisher untouched.
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(pool_[poolBase]);
    }

    __aicore__ inline void WriteRequestStatus(uint32_t batch, int32_t value)
    {
        if (tiling_->batchSize == 1U)
            WriteScalar(requestMissCounts_[batch], value);
        else
            WriteScalar(requestStatus_[batch * 8U], value);
    }

    __aicore__ inline void WriteRouteStatus(uint32_t route, int32_t value)
    {
        WriteScalar(routeStatus_[route * 8U], value);
    }

    __aicore__ inline void MarkError(uint32_t batch)
    {
        uint32_t queryStart = 0U, queryEnd = 0U;
        metadata_.SafeRouteRange(batch, queryStart, queryEnd);
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        Duplicate(sources, -1, TOPK);
        Duplicate(slots, -1, TOPK);
        SyncPipes<HardEvent::V_S>();
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            StoreRange(topkSources_[base], sources, TOPK);
            StoreRange(topkDestinations_[base], slots, TOPK);
            WriteRouteStatus(route, 0);
        }
        WriteRequestStatus(batch, 0);
    }

    __aicore__ inline bool BuildUnion(
        uint32_t queryStart, uint32_t queryEnd,
        uint32_t prefixLength, uint32_t finalLength, uint32_t budget,
        LocalTensor<int32_t> topk, LocalTensor<uint32_t> hash)
    {
        InitHash(hash);
        uint32_t unionCount = 0;
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            const uint32_t visible = finalLength - (queryEnd - 1U - route);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 || static_cast<uint32_t>(token) >= visible) {
                    return false;
                }
                if (static_cast<uint32_t>(token) >= prefixLength) {
                    continue;
                }
                uint32_t position = 0;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found)) {
                    return false;
                }
                if (!found) {
                    if (unionCount >= budget) {
                        return false;
                    }
                    StoreHashEntry(hash, position,
                        static_cast<uint32_t>(token), MISS_SLOT);
                    ++unionCount;
                }
            }
        }
        return true;
    }

    __aicore__ inline void SortPairsTo(
        LocalTensor<int32_t> sources, LocalTensor<int32_t> slots,
        uint32_t count, LocalTensor<float> sortedPairs,
        LocalTensor<float> scratch)
    {
        const uint32_t aligned = (count + 63U) / 64U * 64U;
        LocalTensor<uint32_t> keys = topkBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> payloads = cacheChunkBuf_.Get<uint32_t>();
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::Pack>(
            (__ubuf__ uint32_t *)keys.GetPhyAddr(),
            (__ubuf__ uint32_t *)payloads.GetPhyAddr(),
            (__ubuf__ uint32_t *)sources.GetPhyAddr(),
            (__ubuf__ uint32_t *)slots.GetPhyAddr(), aligned / 64U);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_S>();
        for (uint32_t index = count; index < aligned; ++index)
            keys.SetValue(index, 0U);
        SyncPipes<HardEvent::S_V>();
        Sort<float, true>(sortedPairs, keys.ReinterpretCast<float>(),
            payloads, scratch, aligned / 32U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MergeSortedChunks(
        LocalTensor<float> input, LocalTensor<float> output,
        uint32_t count)
    {
        const uint32_t chunks = (count + TOPK - 1U) / TOPK;
        if (chunks == 1U) {
            DataCopy(output, input, (count + 31U) / 32U * 64U);
            PipeBarrier<PIPE_V>();
            return;
        }
        MrgSort4Info params;
        params.validBit = (1U << chunks) - 1U;
        params.repeatTimes = 1;
        params.ifExhaustedSuspension = false;
        for (uint32_t chunk = 0U; chunk < 4U; ++chunk)
            params.elementLengths[chunk] = chunk < chunks
                ? MinU32(TOPK, count - chunk * TOPK) : 0U;
        MrgSortSrcList<float> inputs;
        inputs.src1 = input;
        inputs.src2 = input[TOPK * 2U];
        inputs.src3 = input[TOPK * 4U];
        inputs.src4 = input[TOPK * 6U];
        MrgSort<float>(output, inputs, params);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SortSelectedSources(uint32_t count)
    {
        if (count == 0U) return;
        LocalTensor<int32_t> selected = victimBuf_.Get<int32_t>();
        LocalTensor<int32_t> resident = residentBuf_.Get<int32_t>();
        LocalTensor<int32_t> zeroSlots = slotBuf_.Get<int32_t>();
        LocalTensor<float> runs = resident.ReinterpretCast<float>();
        LocalTensor<float> merged = hashBuf_.Get<float>();
        Duplicate(zeroSlots, 0, TOPK);
        PipeBarrier<PIPE_V>();
        for (uint32_t offset = 0U; offset < count; offset += TOPK)
            SortPairsTo(selected[offset], zeroSlots,
                MinU32(TOPK, count - offset), runs[offset * 2U], merged);
        MergeSortedChunks(runs, merged, count);
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::Unpack>(
            (__ubuf__ uint32_t *)resident.GetPhyAddr(),
            (__ubuf__ uint32_t *)selected.GetPhyAddr(),
            (__ubuf__ uint32_t *)merged.GetPhyAddr(),
            (count + 63U) / 64U);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_S>();
    }

    __aicore__ inline void SortCopyList(uint32_t count)
    {
        if (count == 0U) return;
        LocalTensor<int32_t> sources = victimBuf_.Get<int32_t>();
        LocalTensor<int32_t> destinations = residentBuf_.Get<int32_t>();
        LocalTensor<float> runs = hashBuf_.Get<float>();
        LocalTensor<float> merged = destinations.ReinterpretCast<float>();
        LocalTensor<float> scratch = destinations[8192U].ReinterpretCast<float>();
        for (uint32_t offset = 0U; offset < count; offset += TOPK)
            SortPairsTo(sources[offset], destinations[offset],
                MinU32(TOPK, count - offset), runs[offset * 2U], scratch);
        MergeSortedChunks(runs, merged, count);
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::Unpack>(
            (__ubuf__ uint32_t *)sources.GetPhyAddr(),
            (__ubuf__ uint32_t *)hashBuf_.Get<uint32_t>().GetPhyAddr(),
            (__ubuf__ uint32_t *)merged.GetPhyAddr(),
            (count + 63U) / 64U);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
    }

    __aicore__ inline void PublishCopyList(uint32_t batch, uint32_t count)
    {
        if (count == 0U) return;
        SyncPipes<HardEvent::S_V>();
        SortCopyList(count);
        const uint64_t base = static_cast<uint64_t>(batch) * MISS_CAPACITY;
        StoreRange(missSources_[base], victimBuf_.Get<int32_t>(), count);
        StoreRange(missDestinations_[base], hashBuf_.Get<int32_t>(), count);
    }

    // Q5-Q14 cannot use the four-route 8192-entry hash and sort buffers.
    // Sort each dense LI row in UB, staging only its selected prefix IDs in
    // the caller's destination list. The subsequent merge reads at most one
    // head per route from GM, so no source/slot payload is truncated.
    __aicore__ inline bool StageWideSelectedRoutes(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        uint32_t prefixLength, uint32_t finalLength,
        uint32_t counts[MAX_QUERIES_PER_REQUEST])
    {
        const uint64_t missBase = static_cast<uint64_t>(batch) * MISS_CAPACITY;
        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        LocalTensor<float> sorted = residentBuf_.Get<float>();
        LocalTensor<float> scratch = hashBuf_.Get<float>();
        Duplicate(slots, 0, TOPK);
        PipeBarrier<PIPE_V>();
        for (uint32_t localRoute = 0U;
             localRoute < queryEnd - queryStart; ++localRoute) {
            const uint32_t route = queryStart + localRoute;
            const uint32_t visible =
                finalLength - (queryEnd - 1U - route);
            LoadTopk(route, topk);
            uint32_t selected = 0U;
            for (uint32_t index = 0U; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (token < 0 || static_cast<uint32_t>(token) >= visible)
                    return false;
                if (static_cast<uint32_t>(token) < prefixLength)
                    sources.SetValue(selected++, token);
            }
            counts[localRoute] = selected;
            if (selected == 0U) continue;
            SyncPipes<HardEvent::S_V>();
            SortPairsTo(sources, slots, selected, sorted, scratch);
            asc_vf_call<fused_quant_li_manage_hit_sort_vf::Unpack>(
                (__ubuf__ uint32_t *)sources.GetPhyAddr(),
                (__ubuf__ uint32_t *)slots.GetPhyAddr(),
                (__ubuf__ uint32_t *)sorted.GetPhyAddr(),
                (selected + 63U) / 64U);
            PipeBarrier<PIPE_V>();
            SyncPipes<HardEvent::V_MTE3>();
            StoreRange(missDestinations_[missBase + localRoute * TOPK],
                       sources, selected);
        }
        return true;
    }

    __aicore__ inline void AppendWideSource(
        uint64_t missBase, uint32_t source, uint32_t &written,
        uint32_t &buffered, LocalTensor<int32_t> buffer)
    {
        buffer.SetValue(buffered++, static_cast<int32_t>(source));
        ++written;
        if (buffered == TOPK) {
            StoreRange(missSources_[missBase + written - buffered],
                       buffer, buffered);
            buffered = 0U;
        }
    }

    // The selected union is assembled in source-ID order. The destination
    // list is temporary route storage and the source list becomes the final
    // union-followed-by-fillers copy list. Pool contents stay untouched until
    // the complete selection has been validated.
    __aicore__ inline bool BuildWideSelection(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        uint32_t prefixLength, uint32_t finalLength, uint32_t budget,
        uint32_t &unionCount)
    {
        const uint32_t routes = queryEnd - queryStart;
        if (routes <= 4U || routes > MAX_QUERIES_PER_REQUEST ||
            budget > MISS_CAPACITY)
            return false;
        uint32_t counts[MAX_QUERIES_PER_REQUEST] = {0U};
        uint32_t cursors[MAX_QUERIES_PER_REQUEST] = {0U};
        int32_t heads[MAX_QUERIES_PER_REQUEST];
        if (!StageWideSelectedRoutes(batch, queryStart, queryEnd,
                                     prefixLength, finalLength, counts))
            return false;

        const uint64_t missBase = static_cast<uint64_t>(batch) * MISS_CAPACITY;
        for (uint32_t route = 0U; route < routes; ++route)
            heads[route] = counts[route] == 0U ? WIDE_NO_SOURCE :
                missDestinations_.GetValue(missBase + route * TOPK);
        LocalTensor<int32_t> chunk = cacheChunkBuf_.Get<int32_t>();
        uint32_t written = 0U;
        uint32_t buffered = 0U;
        while (true) {
            int32_t next = WIDE_NO_SOURCE;
            for (uint32_t route = 0U; route < routes; ++route)
                if (heads[route] < next) next = heads[route];
            if (next == WIDE_NO_SOURCE) break;
            if (next < 0 || static_cast<uint32_t>(next) >= prefixLength ||
                written >= budget)
                return false;
            AppendWideSource(missBase, static_cast<uint32_t>(next),
                             written, buffered, chunk);
            for (uint32_t route = 0U; route < routes; ++route) {
                while (heads[route] == next) {
                    ++cursors[route];
                    heads[route] = cursors[route] == counts[route] ?
                        WIDE_NO_SOURCE : missDestinations_.GetValue(
                            missBase + route * TOPK + cursors[route]);
                }
            }
        }
        if (buffered != 0U) {
            StoreRange(missSources_[missBase + written - buffered],
                       chunk, buffered);
            buffered = 0U;
        }
        unionCount = written;

        // Select the smallest non-union source IDs as fillers. The union is
        // already materialized in GM, so appending fillers cannot overwrite
        // an unread union head.
        uint32_t unionCursor = 0U;
        int32_t unionHead = unionCount == 0U ? WIDE_NO_SOURCE :
            missSources_.GetValue(missBase);
        for (uint32_t source = 0U;
             source < prefixLength && written < budget; ++source) {
            if (unionHead == static_cast<int32_t>(source)) {
                ++unionCursor;
                unionHead = unionCursor == unionCount ? WIDE_NO_SOURCE :
                    missSources_.GetValue(missBase + unionCursor);
                continue;
            }
            AppendWideSource(missBase, source, written, buffered, chunk);
        }
        if (buffered != 0U)
            StoreRange(missSources_[missBase + written - buffered],
                       chunk, buffered);
        return written == budget;
    }

    __aicore__ inline void CommitWideFirstFill(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        uint32_t row, uint32_t prefixLength, uint32_t budget)
    {
        const uint64_t poolBase =
            static_cast<uint64_t>(row) * tiling_->sourceCapacity;
        const uint64_t missBase = static_cast<uint64_t>(batch) * MISS_CAPACITY;
        ClearFirstFillPool(poolBase);
        for (uint32_t slot = 0U; slot < budget; ++slot) {
            const int32_t source = missSources_.GetValue(missBase + slot);
            // Q5-Q14 may publish 32384 residents. A separate four-byte DMA
            // and event wait per slot can stall the other AIVs at SyncAll on
            // a 2M-source request. A3 uses scalar GM stores after the full
            // row clear has completed; retain the existing Q1-Q4 DMA path.
            pool_.SetValue(poolBase + static_cast<uint32_t>(source),
                           static_cast<int32_t>(slot));
        }
        // Scratch destination rows are no longer needed after selection.
        PublishFirstFillDestinations(batch, budget);

        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            uint32_t missCount = 0U;
            for (uint32_t index = 0U; index < TOPK; ++index) {
                if (static_cast<uint32_t>(topk.GetValue(index)) <
                    prefixLength) ++missCount;
            }
            uint32_t missCursor = 0U;
            uint32_t hitCursor = missCount;
            for (uint32_t index = 0U; index < TOPK; ++index) {
                const uint32_t source =
                    static_cast<uint32_t>(topk.GetValue(index));
                const bool isMiss = source < prefixLength;
                const uint32_t output = isMiss ? missCursor++ : hitCursor++;
                sources.SetValue(output, static_cast<int32_t>(source));
                slots.SetValue(output, isMiss ?
                    pool_.GetValue(poolBase + source) :
                    static_cast<int32_t>(budget + source % TAIL_CAPACITY));
            }
            SyncPipes<HardEvent::S_V>();
            SortTopkRow(sources, slots, missCount);
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            StoreRange(topkSources_[base], sources, TOPK);
            StoreRange(topkDestinations_[base], slots, TOPK);
            WriteRouteStatus(route, static_cast<int32_t>(missCount));
        }
        WriteRequestStatus(batch, static_cast<int32_t>(budget));
    }

    __aicore__ inline void SortTopkRow(
        LocalTensor<int32_t> sources, LocalTensor<int32_t> slots,
        uint32_t missCount)
    {
        LocalTensor<uint32_t> keys = topkBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> payloads = cacheChunkBuf_.Get<uint32_t>();
        LocalTensor<int32_t> resident = residentBuf_.Get<int32_t>();
        LocalTensor<float> scratch = resident[8192U].ReinterpretCast<float>();
        LocalTensor<float> sorted = resident[12288U].ReinterpretCast<float>();
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::PackRow>(
            (__ubuf__ uint32_t *)keys.GetPhyAddr(),
            (__ubuf__ uint32_t *)payloads.GetPhyAddr(),
            (__ubuf__ uint32_t *)sources.GetPhyAddr(),
            (__ubuf__ uint32_t *)slots.GetPhyAddr(),
            missCount, TOPK / 64U);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(sorted, keys.ReinterpretCast<float>(),
            payloads, scratch, TOPK / 32U);
        PipeBarrier<PIPE_V>();
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::Unpack>(
            (__ubuf__ uint32_t *)sources.GetPhyAddr(),
            (__ubuf__ uint32_t *)slots.GetPhyAddr(),
            (__ubuf__ uint32_t *)sorted.GetPhyAddr(), TOPK / 64U);
        PipeBarrier<PIPE_V>();
        SyncPipes<HardEvent::V_MTE3>();
    }

    __aicore__ inline bool CommitFirstFill(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd, uint32_t row,
        uint32_t prefixLength, uint32_t budget,
        LocalTensor<int32_t> topk, LocalTensor<uint32_t> hash)
    {
        const uint64_t poolBase = static_cast<uint64_t>(row) * tiling_->sourceCapacity;
        LocalTensor<int32_t> residentSources = residentBuf_.Get<int32_t>();
        LocalTensor<int32_t> selectedSources = victimBuf_.Get<int32_t>();
        uint32_t resident = 0;
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                if (static_cast<uint32_t>(token) >= prefixLength) {
                    continue;
                }
                uint32_t position = 0;
                bool found = false;
                if (!FindHashPosition(hash, token, position, found) || !found) {
                    return false;
                }
                if ((hash.GetValue(position) >> SLOT_SHIFT) == MISS_SLOT) {
                    // Mark this union source before visiting the next route.
                    // The final slot is assigned after sorting the unique IDs.
                    StoreHashEntry(hash, position,
                        static_cast<uint32_t>(token), resident);
                    selectedSources.SetValue(resident, token);
                    ++resident;
                }
            }
        }
        const uint32_t unionCount = resident;
        // The deduplicated IDs are complete. Hardware sort their chunks and
        // merge them before rebuilding the source->slot hash in sorted order.
        SyncPipes<HardEvent::S_V>();
        SortSelectedSources(unionCount);
        InitHash(hash);
        for (uint32_t index = 0; index < unionCount; ++index) {
            const int32_t token = residentSources.GetValue(index);
            uint32_t position = 0;
            bool found = false;
            if (!FindHashPosition(hash, token, position, found) || found)
                return false;
            StoreHashEntry(hash, position,
                static_cast<uint32_t>(token), index);
        }
        // The selected union is the only set needed by subsequent TopK
        // repair. Do not insert fillers into the 8192-entry union hash.
        if (unionCount > budget)
            return false;
        // prefixLength >= budget guarantees enough non-union fillers.
        const uint64_t missBase = static_cast<uint64_t>(batch) * MISS_CAPACITY;

        // Every fallible metadata/union/filler decision is complete.  Only
        // now replace the caller-owned row and publish the C copy pairs.
        // Retire old scalar cache lines before the DMA clear, so a partial
        // SetValue cannot restore stale nonresident entries from those lines.
        SyncFirstFillPoolCache(poolBase);
        ClearFirstFillPool(poolBase);
        StoreRange(missSources_[missBase], residentSources, unionCount);
        for (uint32_t index = 0; index < unionCount; ++index) {
            const int32_t token = residentSources.GetValue(index);
            WritePool(poolBase + static_cast<uint32_t>(token),
                      static_cast<int32_t>(index));
        }
        LocalTensor<int32_t> fillerChunk = cacheChunkBuf_.Get<int32_t>();
        uint32_t fillerCount = 0;
        uint32_t fillerSlot = unionCount;
        for (uint32_t token = 0; token < prefixLength && fillerSlot < budget; ++token) {
            uint32_t position = 0;
            bool found = false;
            FindHashPosition(hash, static_cast<int32_t>(token), position, found);
            if (found)
                continue;
            WritePool(poolBase + token, static_cast<int32_t>(fillerSlot));
            fillerChunk.SetValue(fillerCount++, static_cast<int32_t>(token));
            ++fillerSlot;
            if (fillerCount == CACHE_CHUNK) {
                StoreRange(missSources_[missBase + fillerSlot - fillerCount],
                           fillerChunk, fillerCount);
                fillerCount = 0;
            }
        }
        if (fillerCount != 0U)
            StoreRange(missSources_[missBase + fillerSlot - fillerCount],
                       fillerChunk, fillerCount);
        // Make all resident stores visible in GM before publishing success.
        SyncFirstFillPoolCache(poolBase);
        PublishFirstFillDestinations(batch, budget);

        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> slots = slotBuf_.Get<int32_t>();
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            LoadTopk(route, topk);
            uint32_t missCount = 0;
            for (uint32_t index = 0; index < TOPK; ++index) {
                if (static_cast<uint32_t>(topk.GetValue(index)) < prefixLength) {
                    ++missCount;
                }
            }
            Duplicate(sources, -1, TOPK);
            Duplicate(slots, -1, TOPK);
            SyncPipes<HardEvent::V_S>();
            uint32_t missCursor = 0;
            uint32_t hitCursor = missCount;
            for (uint32_t index = 0; index < TOPK; ++index) {
                const int32_t token = topk.GetValue(index);
                const bool prefix = static_cast<uint32_t>(token) < prefixLength;
                uint32_t slot = budget + static_cast<uint32_t>(token) % TAIL_CAPACITY;
                if (prefix) {
                    uint32_t position = 0;
                    bool found = false;
                    if (!FindHashPosition(hash, token, position, found) || !found) {
                        return false;
                    }
                    slot = hash.GetValue(position) >> SLOT_SHIFT;
                }
                const uint32_t output = prefix ? missCursor++ : hitCursor++;
                sources.SetValue(output, token);
                slots.SetValue(output, static_cast<int32_t>(slot));
            }
            SyncPipes<HardEvent::S_V>();
            SortTopkRow(sources, slots, missCount);
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            StoreRange(topkSources_[base], sources, TOPK);
            StoreRange(topkDestinations_[base], slots, TOPK);
            WriteRouteStatus(route, static_cast<int32_t>(missCount));
        }
        WriteRequestStatus(batch, static_cast<int32_t>(budget));
        return true;
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        uint32_t queryStart = 0;
        uint32_t queryEnd = 0;
        if (!QueryRange(batch, queryStart, queryEnd)) {
            MarkError(batch);
            return;
        }
        if (!metadata_.Valid(batch)) {
            MarkError(batch);
            return;
        }
        const int32_t rowValue = entries_.GetValue(batch);
        uint32_t prefixLength = 0;
        uint32_t finalLength = 0;
        uint32_t budget = 0;
        if (!Validate(batch, queryEnd - queryStart, rowValue,
                      prefixLength, finalLength, budget)) {
            MarkError(batch);
            return;
        }
        bool ok = false;
        if (queryEnd - queryStart <= 4U) {
            // Preserve the established four-route implementation and its UB
            // buffers exactly for the existing workload.
            LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
            LocalTensor<uint32_t> hash = hashBuf_.Get<uint32_t>();
            ok = BuildUnion(
                    queryStart, queryEnd, prefixLength, finalLength,
                    budget, topk, hash) &&
                CommitFirstFill(batch, queryStart, queryEnd,
                    static_cast<uint32_t>(rowValue), prefixLength,
                    budget, topk, hash);
        } else {
            uint32_t unionCount = 0U;
            ok = BuildWideSelection(batch, queryStart, queryEnd,
                                    prefixLength, finalLength, budget,
                                    unionCount);
            if (ok)
                CommitWideFirstFill(batch, queryStart, queryEnd,
                                    static_cast<uint32_t>(rowValue),
                                    prefixLength, budget);
        }
        if (!ok) {
            MarkError(batch);
        }
    }

    TPipe *pipe_ = nullptr;
    const FusedQuantLightningIndexerManageTilingData *tiling_ = nullptr;
    lim_request_contract::Metadata metadata_;
    GlobalTensor<int32_t> queryEnds_;
    GlobalTensor<int32_t> keyLengths_;
    GlobalTensor<int32_t> prefixLengths_;
    GlobalTensor<int32_t> cacheTokens_;
    GlobalTensor<int32_t> states_;
    GlobalTensor<int32_t> entries_;
    GlobalTensor<int32_t> pool_;
    GlobalTensor<int32_t> topkSources_;
    GlobalTensor<int32_t> topkDestinations_;
    GlobalTensor<int32_t> routeMissCounts_;
    GlobalTensor<int32_t> missSources_;
    GlobalTensor<int32_t> missDestinations_;
    GlobalTensor<int32_t> requestMissCounts_;
    bool hasLongSource_ = false;
    GlobalTensor<int32_t> requestStatus_;
    GlobalTensor<int32_t> routeStatus_;
    TBuf<TPosition::VECCALC> topkBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
    TBuf<TPosition::VECCALC> hashBuf_;
    TBuf<TPosition::VECCALC> hashTagBuf_;
    TBuf<TPosition::VECCALC> residentBuf_;
    TBuf<TPosition::VECCALC> victimBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};
} // namespace

__aicore__ inline void RunFusedQuantLightningIndexerManageArch35(
    GM_ADDR indexWeights, GM_ADDR queryDequantScale, GM_ADDR query,
    GM_ADDR indexKeyDequantScale, GM_ADDR indexKeyCache,
    GM_ADDR indexBlockTable, GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKey, GM_ADDR offloadSeqLengthsKey,
    GM_ADDR numCacheTokens, GM_ADDR requestState, GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool, GM_ADDR topkSrcIds, GM_ADDR topkDstSlots,
    GM_ADDR topkMissCounts, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts, GM_ADDR cacheSlotsPoolOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cacheSlotsPoolOut;

    REGISTER_TILING_DEFAULT(FusedQuantLightningIndexerManageTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    // Keep the historical B x 4 physical route rows for existing launches.
    // A wider request switches all cores to compact, public-query-row
    // indexing; the host reserved max(4B, T) rows for either layout.
    if (tilingData.totalQueryRows > 4U) {
        GlobalTensor<int32_t> queryEndsForLayout;
        queryEndsForLayout.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        int32_t previousQueryEnd = 0;
        for (uint32_t batch = 0U; batch < tilingData.batchSize; ++batch) {
            const int32_t queryEnd = queryEndsForLayout.GetValue(batch);
            if (queryEnd > previousQueryEnd &&
                static_cast<int64_t>(queryEnd) - previousQueryEnd > 4)
                tilingData.compactRouteRows = 1U;
            previousQueryEnd = queryEnd;
        }
    }

    bool needsNonOffload = false;
    bool needsFirstFill = false;
    bool needsSparse = false;
    bool needsIdentity = false;
    if ASCEND_IS_AIV {
        GlobalTensor<int32_t> states;
        states.SetGlobalBuffer((__gm__ int32_t *)requestState);
        for (uint32_t batch = 0U; batch < tilingData.batchSize; ++batch) {
            const int32_t state = states.GetValue(batch);
            needsNonOffload |= state != lim_request_limits::FIRST_FILL_STATE &&
                state != lim_request_limits::STEADY_STATE;
            needsFirstFill |= state == lim_request_limits::FIRST_FILL_STATE;
            needsSparse |= state == lim_request_limits::STEADY_STATE;
            needsIdentity |= state == lim_request_limits::IDENTITY_STATE;
        }
        if (needsSparse) {
            lim_request_contract::Metadata metadata;
            metadata.Init(actualSeqLengthsQuery, actualSeqLengthsKey,
                          offloadSeqLengthsKey, numCacheTokens, requestState,
                          reqPoolEntries, &tilingData);
            if (NeedsIdentityNormalization(
                    requestState, numCacheTokens, reqPoolEntries,
                    cacheSlotsPool, &tilingData, metadata)) {
                // First finish every AIV's probe so an early writer cannot
                // change another AIV's decision about the collective.
                AscendC::SyncAll();
                NormalizeIdentityRows(requestState, numCacheTokens,
                                      reqPoolEntries, cacheSlotsPool,
                                      &tilingData, metadata, &pipe);
                AscendC::SyncAll();
                pipe.Reset();
            }
        }
    }

    {
        const uint64_t batchSize = tilingData.batchSize;
        const uint64_t routeRows = tilingData.routeStorageRows;
        // Keep constexpr workspace arithmetic expanded in device code, as in
        // the A5 reference: host-only constexpr helpers are used by tiling.
        const uint64_t pairOffset = routeRows *
            tilingData.scoreRowStride * sizeof(uint16_t);
        const uint64_t thresholdOffset = pairOffset +
            routeRows * TOPK * 2U * sizeof(int32_t);
        const uint64_t countOffset = thresholdOffset +
            routeRows * 16U * sizeof(uint16_t);
        const uint64_t candidateScoreOffset =
            (countOffset + routeRows *
                fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE *
                sizeof(int32_t) + 31U) /
            32U * 32U;
        const uint64_t sourceBlocks =
            (static_cast<uint64_t>(tilingData.sourceCapacity) + 127U) / 128U;
        const uint64_t maxSegments = sourceBlocks / 18U;
        const uint64_t candidateSegments =
            tilingData.sourceCapacity < 64U * 1024U ? 1U :
            (maxSegments < tilingData.usedCoreNum
                ? maxSegments : tilingData.usedCoreNum);
        const uint64_t candidatePayloadOffset = candidateScoreOffset +
            routeRows * candidateSegments * TOPK * sizeof(uint16_t);
        const uint64_t candidateTagOffset = candidatePayloadOffset +
            routeRows * candidateSegments * TOPK * sizeof(uint32_t);
        GM_ADDR routePairRows = userWorkspace + pairOffset;
        GM_ADDR routeThresholds = userWorkspace + thresholdOffset;
        GM_ADDR routeMissCounts = userWorkspace + countOffset;
        GM_ADDR candidateScores = userWorkspace + candidateScoreOffset;
        GM_ADDR candidatePayloads = userWorkspace + candidatePayloadOffset;
        GM_ADDR candidateTags = userWorkspace + candidateTagOffset;

        fused_quant_li_manage::QuantLiMtpPhase qli(
            &pipe, &tilingData);
        qli.Init(
            indexWeights, query, queryDequantScale, actualSeqLengthsQuery,
            indexKeyCache, indexKeyDequantScale, actualSeqLengthsKey,
            reqPoolEntries, cacheSlotsPool, offloadSeqLengthsKey,
            numCacheTokens, requestState, indexBlockTable, routePairRows, topkDstSlots,
            topkSrcIds, routeThresholds, routeMissCounts, userWorkspace,
            candidateScores, candidatePayloads, candidateTags);
        qli.Process();

        if ASCEND_IS_AIV {
            AscendC::SyncAll();
            NonOffloadPublisher nonOffload;
            FirstFillPublisher firstFill;
            fused_quant_li_manage::OrderedMissUnion unionManager;
            if ((GetBlockIdx() & 1U) == 0U) {
                if (needsNonOffload) {
                    pipe.Reset();
                    nonOffload.Init(
                        &pipe,
                        actualSeqLengthsQuery, actualSeqLengthsKey,
                        requestState, reqPoolEntries, cacheSlotsPool,
                        topkSrcIds, topkDstSlots, topkMissCounts,
                        missCounts, &tilingData);
                    if (needsIdentity) {
                        nonOffload.FillIdentityChunks(
                            GetBlockIdx() / 2U, tilingData.usedCoreNum);
                    }
                    nonOffload.Process(GetBlockIdx() / 2U,
                                       tilingData.usedCoreNum);
                }
            }
            if (needsIdentity && (needsFirstFill || needsSparse)) {
                // An identity DMA can still be reading its UB source when
                // the same AIV reaches pipe.Reset() for a different request.
                // Finish those writes before reusing UB for first fill/union.
                if ((GetBlockIdx() & 1U) == 0U)
                    SyncPipes<HardEvent::MTE3_S>();
                AscendC::SyncAll();
            }
            if ((GetBlockIdx() & 1U) == 0U) {
                if (needsFirstFill || needsSparse) pipe.Reset();
                // Only -2 needs the first-fill manager. Pure -1 batches
                // proceed from QLI directly to the ordered union.
                if (needsFirstFill) {
                    firstFill.Init(
                        &pipe, userWorkspace, actualSeqLengthsQuery,
                        actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens,
                        requestState, reqPoolEntries, cacheSlotsPool, topkSrcIds,
                        topkDstSlots, topkMissCounts, missSrcIds, missDstSlots,
                        missCounts, &tilingData);
                    firstFill.Process(GetBlockIdx() / 2U,
                                      tilingData.usedCoreNum);
                    if (needsSparse) pipe.Reset();
                }
                if (needsSparse) {
                    unionManager.Init(
                        routePairRows, routeThresholds, routeMissCounts,
                        userWorkspace, actualSeqLengthsQuery,
                        offloadSeqLengthsKey, actualSeqLengthsKey,
                        requestState, reqPoolEntries, cacheSlotsPool,
                        numCacheTokens, missSrcIds, missDstSlots,
                        missCounts, topkDstSlots, topkSrcIds,
                        topkMissCounts,
                        userWorkspace + tilingData.managementWorkspaceOffset,
                        userWorkspace + tilingData.managementWorkspaceOffset +
                            batchSize * 32U,
                        tilingData.sourceCapacity, MISS_CAPACITY,
                        tilingData.scoreRowStride, tilingData.batchSize,
                        tilingData.poolSize, tilingData.totalQueryRows,
                        tilingData.sourceCapacity, &tilingData, &pipe);
                    unionManager.Process(GetBlockIdx() / 2U,
                                      tilingData.usedCoreNum);
                }
            }
            AscendC::SyncAll();
            if (GetBlockIdx() == 0U) {
                if (needsNonOffload) {
                    if (!needsFirstFill && !needsSparse) {
                        nonOffload.PublishHomogeneousCounts();
                    } else {
                        nonOffload.PublishCounts();
                    }
                }
                if (needsFirstFill) firstFill.PublishCounts();
                if (needsSparse) unionManager.PublishCounts();
            }
        }
    }
}
