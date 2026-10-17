/**
 * A5 C8 Stage 2-4 ordered union from vLLM-ascend-DSA li_manage_c8.
 * Adaptations are limited to the standardized ABI/lifecycle: no pool status
 * column, budget from num_cache_tokens, 32768-wide miss output, and compact
 * caller-owned counts staged to avoid cross-request false sharing.
 */

#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_UNION_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_UNION_H

#include "kernel_operator.h"
#include "fused_quant_lightning_indexer_manage_union_vf.h"
#include "fused_quant_lightning_indexer_manage_sort.h"
#include "fused_quant_lightning_indexer_manage_victim_vf.h"
#include "fused_quant_lightning_indexer_manage_wide_victim.h"
#include "fused_quant_lightning_indexer_manage_workspace.h"
#include "lim_request_contract.h"
#include "fused_quant_lightning_indexer_manage_victim_scan.h"
#include "lim_simt_postprocess.h"
#include "lim_simt_wide_postprocess.h"
#include "lim_pool_contract.h"
#include "lim_payload_layout.h"
#include "lim_cache_budget.h"
#include "arch35_payload_c8/payload/lim_normalize_pool_slots_vf.h"

namespace fused_quant_li_manage {
using namespace AscendC;

constexpr uint32_t UNION_ROUTES = 4U;
constexpr uint32_t UNION_MAX_ROUTES = lim_request_limits::MAX_ROUTES;
constexpr uint32_t UNION_TOPK = lim_request_limits::TOPK;
constexpr int32_t UNION_ROW_MODE_SPARSE = lim_request_limits::STEADY_STATE;
constexpr uint32_t UNION_CAPACITY = UNION_ROUTES * UNION_TOPK;
constexpr uint32_t UNION_PAIR_WORDS = UNION_TOPK * 2U;
// Match A3's generalized union: fixed-bucket SortAll up to 4096
// occurrences, then an ordered k-way union over the original route rows.
constexpr uint32_t UNION_SMALL_MISS_OCCURRENCES =
    fused_quant_li_manage_sort::CAPACITY;
static_assert(2U * UNION_SMALL_MISS_OCCURRENCES <= UNION_CAPACITY * 2U,
              "SortAll input and scratch must fit the pair buffers");
constexpr uint32_t UNION_SOURCE_MASK = lim_payload_layout::SOURCE_MASK;
constexpr uint32_t B32_VECTOR_ELEMENTS = 64U;
constexpr uint32_t VICTIM_SCAN_CHUNK = 2048U;
constexpr uint32_t VICTIM_SLOT_SHIFT = lim_payload_layout::SOURCE_BITS;
constexpr uint32_t VICTIM_SLOT_MASK = lim_payload_layout::SLOT_MASK;

template <HardEvent event>
__aicore__ inline void UnionSync(HardEvent value)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(value));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

class OrderedMissUnion {
public:
    __aicore__ inline void Init(
        GM_ADDR routePairs, GM_ADDR routeThresholds, GM_ADDR routeCounts,
        GM_ADDR scoreWorkspace, GM_ADDR actualSeqLengthsQuery,
        GM_ADDR offloadedLens,
        GM_ADDR finalSeqLengthsKv, GM_ADDR rowModes,
        GM_ADDR reqPoolEntries, GM_ADDR cacheSlots, GM_ADDR numCacheTokens,
        GM_ADDR unionSources, GM_ADDR unionDestinations,
        GM_ADDR unionCounts, GM_ADDR topkSlots, GM_ADDR topkIndex,
        GM_ADDR perQueryMissCounts,
        GM_ADDR requestStatus, GM_ADDR routeStatus,
        uint32_t tokenCapacity, uint32_t outputCapacity,
        uint32_t scoreRowStride, uint32_t batchSize,
        uint32_t poolSize, uint32_t totalQueryRows,
        uint32_t maxCandidateLen,
        const FusedQuantLightningIndexerManageTilingData *tiling,
        TPipe *pipe)
    {
        routePairsGm_.SetGlobalBuffer((__gm__ int32_t *)routePairs);
        routeThresholdsGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)routeThresholds);
        routeCountsGm_.SetGlobalBuffer((__gm__ int32_t *)routeCounts);
        scoreWorkspaceGm_.SetGlobalBuffer((__gm__ uint16_t *)scoreWorkspace);
        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        offloadedLensGm_.SetGlobalBuffer((__gm__ int32_t *)offloadedLens);
        finalSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)finalSeqLengthsKv);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        numCacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        unionSourcesGm_.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        unionDestinationsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)unionDestinations);
        unionCountsGm_.SetGlobalBuffer((__gm__ int32_t *)unionCounts);
        topkSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)topkSlots);
        topkIndexGm_.SetGlobalBuffer(
            (__gm__ int32_t *)topkIndex);
        perQueryMissCountsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)perQueryMissCounts);
        requestStatusGm_.SetGlobalBuffer((__gm__ int32_t *)requestStatus);
        routeStatusGm_.SetGlobalBuffer((__gm__ int32_t *)routeStatus);
        sourceCapacity_ = tokenCapacity;
        poolStride_ = tokenCapacity;
        outputCapacity_ = outputCapacity;
        scoreRowStride_ = scoreRowStride;
        batchSize_ = batchSize;
        poolSize_ = poolSize;
        totalQueryRows_ = totalQueryRows;
        maxCandidateLen_ = maxCandidateLen;
        tiling_ = tiling;
        metadata_.Init(actualSeqLengthsQuery, finalSeqLengthsKv,
                       offloadedLens, numCacheTokens, rowModes,
                       reqPoolEntries, tiling);
        pipe->InitBuffer(pairInputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutputBuf_,
                         UNION_CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(sourceBuf_,
                         UNION_CAPACITY * sizeof(int32_t));
        // Four independent route-status rows plus one request-status row
        // avoid compact-output false sharing across request owners.
        pipe->InitBuffer(countBuf_, 5U * 32U);
        pipe->InitBuffer(
            thresholdBuf_,
            UNION_ROUTES *
                fused_quant_li_manage_workspace::THRESHOLD_STRIDE *
                sizeof(uint16_t));
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize_; batch += stride) {
            ProcessRequest(batch);
        }
    }

    // The standardized compact count tensors can share a physical write
    // granule across requests. Stage each owner's counts in disjoint 32-byte
    // rows, then let one core publish the compact public tensors.
    __aicore__ inline void PublishCounts()
    {
        for (uint32_t batch = 0U; batch < batchSize_; ++batch) {
            if (rowModesGm_.GetValue(batch) != UNION_ROW_MODE_SPARSE)
                continue;
            uint32_t queryStart = 0U, queryEnd = 0U;
            metadata_.SafeRouteRange(batch, queryStart, queryEnd);
            for (uint32_t route = queryStart; route < queryEnd; ++route)
                perQueryMissCountsGm_.SetValue(
                    route, routeStatusGm_.GetValue(route * 8U));
            if (batchSize_ != 1U)
                unionCountsGm_.SetValue(
                    batch, requestStatusGm_.GetValue(batch * 8U));
        }
    }

private:
#if LIM_SIMT_REQUEST_META_CACHE
    struct SparseRequestMetadata {
        uint32_t candidate;
        uint32_t finalLen;
        uint32_t poolRow;
        uint32_t budget;
    };
#endif

    __aicore__ inline uint32_t Deduplicate(
        LocalTensor<float> merged, uint32_t total,
        LocalTensor<int32_t> output)
    {
        FusedQuantLiManageUnionVF::DeduplicateSortedPairs(
            (__ubuf__ uint32_t *)output.GetPhyAddr(),
            (__ubuf__ uint32_t *)merged.GetPhyAddr(), total);
        const uint32_t count = static_cast<uint32_t>(
            GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        // The following victim phase reuses the merged-pair UB allocation.
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        return count;
    }

    __aicore__ inline uint32_t ReadBudget(
        uint32_t batch, uint32_t poolRow)
    {
        (void)poolRow;
        return static_cast<uint32_t>(numCacheTokensGm_.GetValue(batch));
    }

    __aicore__ inline uint64_t WorkspaceRouteRow(
        uint32_t batch, uint32_t queryStart, uint32_t route) const
    {
        return tiling_->compactRouteRows != 0U
            ? static_cast<uint64_t>(queryStart) + route
            : static_cast<uint64_t>(batch) * UNION_ROUTES + route;
    }

    __aicore__ inline bool QueryRange(
        uint32_t batch, uint32_t &queryStart,
        uint32_t &queryEnd, uint32_t &routeCount)
    {
        const int32_t end = actualSeqLengthsQueryGm_.GetValue(batch);
        const int32_t start = batch == 0U
            ? 0 : actualSeqLengthsQueryGm_.GetValue(batch - 1U);
        if (start < 0 || end <= start ||
            end > static_cast<int32_t>(totalQueryRows_) ||
            end - start > static_cast<int32_t>(UNION_MAX_ROUTES)) {
            return false;
        }
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        routeCount = queryEnd - queryStart;
        return true;
    }

    __aicore__ inline void LoadThresholds(
        uint32_t batch, uint32_t queryStart, LocalTensor<uint16_t> local,
        uint32_t routeCount, uint16_t values[UNION_ROUTES])
    {
        constexpr uint32_t STRIDE =
            fused_quant_li_manage_workspace::THRESHOLD_STRIDE;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            DataCopyPad(
                local[route * STRIDE],
                routeThresholdsGm_[
                    WorkspaceRouteRow(batch, queryStart, route) * STRIDE],
                {1, STRIDE * static_cast<uint32_t>(sizeof(uint16_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
        }
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t route = 0U; route < routeCount; ++route) {
            values[route] = local.GetValue(route * STRIDE);
        }
        // The victim VF tests score < threshold for all four lanes.  An
        // absent speculative route must therefore be neutral, not a reason
        // to reject an otherwise safe victim.  Stage A does not publish an
        // unused score row; CompactSafeVictims supplies a zero row and this
        // maximal threshold makes that lane conservatively pass.  A real
        // score equal to 0xffff can only make the fast scan miss a victim;
        // AppendExactVictims still completes the request exactly.
        for (uint32_t route = routeCount; route < UNION_ROUTES; ++route) {
            values[route] = 0xffffU;
        }
    }

    __aicore__ inline uint32_t CompactSafeVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t required, uint32_t routeCount,
        uint32_t queryStart, uint32_t scanSeed,
        const uint16_t thresholds[UNION_ROUTES],
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        LocalTensor<uint8_t> scratch = pairInputBuf_.Get<uint8_t>();
        LocalTensor<uint16_t> score0 =
            scratch.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> score1 = score0[VICTIM_SCAN_CHUNK];
        LocalTensor<uint16_t> score2 =
            score0[VICTIM_SCAN_CHUNK * 2U];
        LocalTensor<uint16_t> score3 =
            score0[VICTIM_SCAN_CHUNK * 3U];
        LocalTensor<int32_t> slots32 =
            scratch[VICTIM_SCAN_CHUNK * UNION_ROUTES *
                    sizeof(uint16_t)].ReinterpretCast<int32_t>();
        LocalTensor<int16_t> slots16 =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t))]
                .ReinterpretCast<int16_t>();
        LocalTensor<uint32_t> compact =
            scratch[VICTIM_SCAN_CHUNK *
                    (UNION_ROUTES * sizeof(uint16_t) + sizeof(int32_t) +
                     sizeof(int16_t))].ReinterpretCast<uint32_t>();

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk = scanSeed % chunks;
        const uint64_t requestScoreBase =
            WorkspaceRouteRow(batch, queryStart, 0U) * scoreRowStride_;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        uint32_t written = 0U;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t chunkBase = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t chunkLen =
                chunkBase + VICTIM_SCAN_CHUNK > candidate
                    ? candidate - chunkBase
                    : VICTIM_SCAN_CHUNK;
            const uint32_t alignedLen =
                (chunkLen + B32_VECTOR_ELEMENTS - 1U) /
                B32_VECTOR_ELEMENTS * B32_VECTOR_ELEMENTS;
            const DataCopyExtParams scoreCopy{
                1, chunkLen * static_cast<uint32_t>(sizeof(uint16_t)),
                0, 0, 0};
            const DataCopyPadExtParams<uint16_t> scorePad{
                true, 0,
                static_cast<uint8_t>(alignedLen - chunkLen), 0U};
            if (routeCount > 0U) {
                DataCopyPad(
                    score0,
                    scoreWorkspaceGm_[requestScoreBase + chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score0, static_cast<uint16_t>(0), alignedLen);
            }
            if (routeCount > 1U) {
                DataCopyPad(
                    score1,
                    scoreWorkspaceGm_[requestScoreBase + scoreRowStride_ +
                                      chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score1, static_cast<uint16_t>(0), alignedLen);
            }
            if (routeCount > 2U) {
                DataCopyPad(
                    score2,
                    scoreWorkspaceGm_[requestScoreBase +
                                      scoreRowStride_ * 2U + chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score2, static_cast<uint16_t>(0), alignedLen);
            }
            if (routeCount > 3U) {
                DataCopyPad(
                    score3,
                    scoreWorkspaceGm_[requestScoreBase +
                                      scoreRowStride_ * 3U + chunkBase],
                    scoreCopy, scorePad);
            } else {
                Duplicate(score3, static_cast<uint16_t>(0), alignedLen);
            }
            DataCopyPad(
                slots32, cacheSlotsGm_[cacheBase + chunkBase],
                {1, chunkLen * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            asc_vf_call<lim_normalize_pool_slots_vf::Normalize>(
                (__ubuf__ uint32_t *)slots32.GetPhyAddr(),
                (chunkLen + 63U) / 64U);
            PipeBarrier<PIPE_V>();
            Cast(slots16, slots32, RoundMode::CAST_NONE, chunkLen);
            if (alignedLen > chunkLen) {
                Duplicate(
                    slots16[chunkLen], static_cast<int16_t>(-1),
                    alignedLen - chunkLen);
            }
            PipeBarrier<PIPE_V>();
            FusedQuantLiManageVictimVF::CompactEligiblePayloads(
                (__ubuf__ uint32_t *)compact.GetPhyAddr(),
                (__ubuf__ uint16_t *)score0.GetPhyAddr(),
                (__ubuf__ uint16_t *)score1.GetPhyAddr(),
                (__ubuf__ uint16_t *)score2.GetPhyAddr(),
                (__ubuf__ uint16_t *)score3.GetPhyAddr(),
                (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
                thresholds[0], thresholds[1], thresholds[2],
                thresholds[3], static_cast<uint16_t>(budget),
                chunkBase, alignedLen / B32_VECTOR_ELEMENTS);
            const uint32_t compactCount = static_cast<uint32_t>(
                GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
            PipeBarrier<PIPE_V>();
#if !LIM_SIMT_VICTIM_UNPACK
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
#endif
            const uint32_t remaining = required - written;
            const uint32_t keep =
                compactCount < remaining ? compactCount : remaining;
#if LIM_SIMT_VICTIM_UNPACK
            if (keep != 0U) {
                asc_vf_call<lim_simt::UnpackVictims>(
                    dim3(LIM_SIMT_THREADS),
                    (__ubuf__ uint32_t *)compact.GetPhyAddr(),
                    (__ubuf__ int32_t *)destinations.GetPhyAddr() + written,
                    (__ubuf__ int32_t *)victimSources.GetPhyAddr() + written,
                    keep, chunkBase & ~UNION_SOURCE_MASK);
            }
            // SIMD compaction and SIMT unpack share PIPE_V. Wait after
            // unpack (instead of before scalar unpack) before reusing the
            // scratch or letting AppendExactVictims read destinations.
            // This also completes compaction when keep==0; no empty VF.
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
#else
            for (uint32_t index = 0U; index < keep; ++index) {
                const uint32_t payload = compact.GetValue(index);
                destinations.SetValue(
                    written + index,
                    static_cast<int32_t>(
                        (payload >> VICTIM_SLOT_SHIFT) & VICTIM_SLOT_MASK));
                victimSources.SetValue(
                    written + index,
                    static_cast<int32_t>(
                        (chunkBase & ~UNION_SOURCE_MASK) |
                        (payload & UNION_SOURCE_MASK)));
            }
#endif
            written += keep;
        }
        return written;
    }

    __aicore__ inline uint32_t AppendExactVictims(
        uint32_t batch, uint32_t candidate, uint32_t poolRow,
        uint32_t budget, uint32_t written, uint32_t required,
        uint32_t routeCount, uint32_t queryStart,
        uint32_t scanSeed,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (written >= required) {
            return written;
        }
        // One byte per slot covers the complete 32640-slot C range in the
        // existing pair scratch buffer.
        LocalTensor<uint8_t> protectedSlots =
            pairInputBuf_.Get<uint8_t>();
        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        PipeBarrier<PIPE_V>();
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t index = 0U; index < written; ++index) {
            const int32_t slot = destinations.GetValue(index);
            if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
            }
        }
        // Stage 1 writes the selected prefix/tail union to [T,2048].
        // Only prefix slots (< budget) belong to the eviction domain.
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t topkBase =
                (static_cast<uint64_t>(queryStart) + route) *
                fused_quant_li_manage_workspace::ATTENTION_CAPACITY;
            for (uint32_t index = 0U; index < UNION_TOPK; ++index) {
                const int32_t slot =
                    topkSlotsGm_.GetValue(topkBase + index);
                if (slot >= 0 && static_cast<uint32_t>(slot) < budget) {
                    protectedSlots.SetValue(
                        static_cast<uint32_t>(slot), 1);
                }
            }
        }

        const uint32_t chunks =
            (candidate + VICTIM_SCAN_CHUNK - 1U) / VICTIM_SCAN_CHUNK;
        const uint32_t firstChunk = scanSeed % chunks;
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        for (uint32_t visit = 0U;
             visit < chunks && written < required; ++visit) {
            const uint32_t chunk = (firstChunk + visit) % chunks;
            const uint32_t begin = chunk * VICTIM_SCAN_CHUNK;
            const uint32_t end =
                begin + VICTIM_SCAN_CHUNK < candidate
                    ? begin + VICTIM_SCAN_CHUNK
                    : candidate;
            for (uint32_t source = begin;
                 source < end && written < required; ++source) {
                const int32_t slot =
                    cacheSlotsGm_.GetValue(cacheBase + source);
                if (slot < 0 || static_cast<uint32_t>(slot) >= budget ||
                    protectedSlots.GetValue(
                        static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                destinations.SetValue(written, slot);
                victimSources.SetValue(
                    written, static_cast<int32_t>(source));
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++written;
            }
        }
        return written;
    }

    __aicore__ inline uint32_t FindVictims(
        uint32_t batch, uint32_t count, uint32_t routeCount,
        uint32_t queryStart, uint32_t scanSeed,
#if LIM_SIMT_REQUEST_META_CACHE
        const SparseRequestMetadata &metadata,
#endif
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
        if (count == 0U) {
            return 0U;
        }
        Duplicate(destinations, static_cast<int32_t>(-1), count);
        Duplicate(victimSources, static_cast<int32_t>(-1), count);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> thresholdLocal = thresholdBuf_.Get<uint16_t>();
        uint16_t thresholds[UNION_ROUTES];
        LoadThresholds(batch, queryStart, thresholdLocal, routeCount, thresholds);
#if LIM_SIMT_REQUEST_META_CACHE
        const uint32_t candidate = metadata.candidate;
        const uint32_t poolRow = metadata.poolRow;
        const uint32_t budget = metadata.budget;
#else
        const uint32_t candidate = static_cast<uint32_t>(
            offloadedLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
#endif
        uint32_t written = CompactSafeVictims(
            batch, candidate, poolRow, budget, count, routeCount,
            queryStart, scanSeed, thresholds,
            destinations, victimSources);
        written = AppendExactVictims(
            batch, candidate, poolRow, budget, written, count,
            routeCount, queryStart, scanSeed,
            destinations, victimSources);
        return written;
    }

    __aicore__ inline uint32_t ApplyCacheUpdates(
        uint32_t batch, uint32_t count,
#if LIM_SIMT_REQUEST_META_CACHE
        const SparseRequestMetadata &metadata,
#endif
        LocalTensor<int32_t> sources,
        LocalTensor<int32_t> destinations,
        LocalTensor<int32_t> victimSources)
    {
#if LIM_SIMT_REQUEST_META_CACHE
        (void)batch;
        const uint32_t candidate = metadata.candidate;
        const uint32_t poolRow = metadata.poolRow;
        const uint32_t budget = metadata.budget;
#else
        const uint32_t candidate = static_cast<uint32_t>(
            offloadedLensGm_.GetValue(batch));
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(batch));
        const uint32_t budget = ReadBudget(batch, poolRow);
#endif
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRow) * poolStride_;
        uint32_t updated = 0U;
#if LIM_SIMT_POSTPROCESS
        // countBuf_[1] is idle until PublishFinalOutputs rewrites all count
        // rows. Keep it separate from the live source/destination arrays.
        LocalTensor<uint32_t> validPrefix = countBuf_.Get<uint32_t>()[1U];
        validPrefix.SetValue(0U, count);
        UnionSync<HardEvent::S_V>(HardEvent::S_V);
        asc_vf_call<lim_simt::ApplyUpdates>(
            dim3(LIM_SIMT_THREADS),
            (__gm__ int32_t *)cacheSlotsGm_.GetPhyAddr(), cacheBase,
            (__ubuf__ int32_t *)sources.GetPhyAddr(),
            (__ubuf__ int32_t *)destinations.GetPhyAddr(),
            (__ubuf__ int32_t *)victimSources.GetPhyAddr(),
            (__ubuf__ uint32_t *)validPrefix.GetPhyAddr(),
            count, candidate, budget);
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        updated = validPrefix.GetValue(0U);
#else
        for (uint32_t index = 0U; index < count; ++index) {
            const int32_t source = sources.GetValue(index);
            const int32_t destination = destinations.GetValue(index);
            const int32_t victimSource = victimSources.GetValue(index);
            if (source < 0 || victimSource < 0 || destination < 0 ||
                static_cast<uint32_t>(source) >= candidate ||
                static_cast<uint32_t>(victimSource) >= candidate ||
                static_cast<uint32_t>(destination) >= budget) {
                break;
            }
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(victimSource),
                lim_pool_contract::INVALID_POOL_SLOT);
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(source), destination);
            ++updated;
        }
#endif
        if (updated != 0U) {
            PipeBarrier<PIPE_ALL>();
        }
        return updated;
    }

    __aicore__ inline void PrepareTopkMissPrefixes(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount,
        const uint32_t lengths[UNION_ROUTES],
        uint32_t unionCount, LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> allDestinations)
    {
        LocalTensor<int32_t> routePairs = pairInputBuf_.Get<int32_t>();
#if LIM_SIMT_BATCHED_REPAIR
        static_assert(UNION_ROUTES == 4U && UNION_TOPK == lim_simt::REPAIR_TOPK,
            "batched repair must match the fixed route-pair workspace rows");
        // Each DMA owns a separate pairInput row. No route overwrites data
        // still being read by another route, so one MTE2_V wait is sufficient.
        const uint64_t requestPairBase =
            WorkspaceRouteRow(batch, queryStart, 0U) * UNION_PAIR_WORDS;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint32_t length = lengths[route];
            if (length != 0U) {
                const uint32_t pairOffset = route * UNION_PAIR_WORDS;
                DataCopyPad(
                    routePairs[pairOffset],
                    routePairsGm_[requestPairBase + pairOffset],
                    {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        asc_vf_call<lim_simt::RepairMissSlotsBatched>(
            dim3(LIM_SIMT_THREADS),
            (__ubuf__ int32_t *)routePairs.GetPhyAddr(),
            (__ubuf__ int32_t *)unionSources.GetPhyAddr(),
            (__ubuf__ int32_t *)unionDestinations.GetPhyAddr(),
            (__ubuf__ int32_t *)allDestinations.GetPhyAddr(),
            lengths[0], lengths[1], lengths[2], lengths[3], unionCount);
        // Complete pair reads before publication and the next request can
        // reuse their UB. Keep the existing final V_MTE3 ordering.
        UnionSync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
#else
        // pairInputBuf_ has 16384 int32 words. One 4096-word interleaved
        // route-pair row and four 2048-word destination prefixes are disjoint.
        const uint64_t requestPairBase =
            WorkspaceRouteRow(batch, queryStart, 0U) * UNION_PAIR_WORDS;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint32_t length = lengths[route];
            if (length == 0U) {
                continue;
            }
            const uint32_t pairOffset = route * UNION_PAIR_WORDS;
            DataCopyPad(
                routePairs,
                routePairsGm_[requestPairBase + pairOffset],
                {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                 0, 0, 0},
                {false, 0, 0, 0});
            LocalTensor<int32_t> rowDestinations =
                allDestinations[route * UNION_TOPK];
#if LIM_SIMT_POSTPROCESS
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            asc_vf_call<lim_simt::RepairMissSlots>(
                dim3(LIM_SIMT_THREADS),
                (__ubuf__ int32_t *)routePairs.GetPhyAddr(),
                (__ubuf__ int32_t *)unionSources.GetPhyAddr(),
                (__ubuf__ int32_t *)unionDestinations.GetPhyAddr(),
                (__ubuf__ int32_t *)rowDestinations.GetPhyAddr(),
                length, unionCount);
            // Next route reuses routePairs for MTE2. Do not overwrite it
            // while the current VF still reads it. Final MTE3 publication
            // retains the existing V_MTE3 dependency.
            UnionSync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
#else
            UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            uint32_t unionCursor = 0U;
            for (uint32_t miss = 0U; miss < length; ++miss) {
                // BuildSortedMissPairs emits source IDs in ascending order,
                // matching the deduplicated union and the public source
                // prefix already published once by Stage 1.
                const int32_t source = routePairs.GetValue(miss * 2U + 1U);
                while (unionCursor < unionCount &&
                       unionSources.GetValue(unionCursor) < source) {
                    ++unionCursor;
                }
                const int32_t destination =
                    unionCursor < unionCount &&
                            unionSources.GetValue(unionCursor) == source
                        ? unionDestinations.GetValue(unionCursor)
                        : -1;
                rowDestinations.SetValue(miss, destination);
            }
#endif
        }
#endif
    }

    __aicore__ inline void PublishFinalOutputs(
        uint32_t batch, uint32_t queryStart,
        uint32_t routeCount,
        const uint32_t lengths[UNION_ROUTES],
        uint32_t count, LocalTensor<int32_t> countLocal,
#if LIM_SIMT_REQUEST_META_CACHE
        const SparseRequestMetadata &metadata,
#endif
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<int32_t> topkMissDestinations)
    {
#if LIM_SIMT_REQUEST_META_CACHE
        (void)metadata;
#endif
        // Independent 32-byte scalar rows: no DMA source is overwritten
        // until the single completion wait at the end of this request.
        LocalTensor<int32_t> unionCountLocal = countLocal[32U];
        for (uint32_t route = 0U; route < routeCount; ++route) {
            countLocal.SetValue(route * 8U,
                                static_cast<int32_t>(lengths[route]));
        }
        unionCountLocal.SetValue(0U, static_cast<int32_t>(count));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        for (uint32_t route = 0U; route < routeCount; ++route)
            DataCopyPad(
                routeStatusGm_[(queryStart + route) * 8U],
                countLocal[route *
                    fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE],
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        if (batchSize_ == 1U)
            DataCopyPad(
                unionCountsGm_[batch], unionCountLocal,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        else
            DataCopyPad(
                requestStatusGm_[batch * 8U], unionCountLocal,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        if (count != 0U) {
            const uint64_t unionOffset =
                static_cast<uint64_t>(batch) * outputCapacity_;
            const uint16_t unionBytes = static_cast<uint16_t>(
                count * static_cast<uint32_t>(sizeof(int32_t)));
            DataCopyPad(
                unionSourcesGm_[unionOffset], unionSources,
                {1, unionBytes, 0, 0});
            DataCopyPad(
                unionDestinationsGm_[unionOffset], unionDestinations,
                {1, unionBytes, 0, 0});
        }
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t publicRow =
                (static_cast<uint64_t>(queryStart) + route) *
                fused_quant_li_manage_workspace::ATTENTION_CAPACITY;
            if (count != 0U && lengths[route] != 0U) {
                DataCopyPad(
                    topkSlotsGm_[publicRow],
                    topkMissDestinations[route * UNION_TOPK],
                    {1, static_cast<uint16_t>(lengths[route] * sizeof(int32_t)),
                     0, 0});
            }
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void FillAttentionRows(
        uint32_t queryStart, uint32_t routeCount, int32_t value)
    {
        LocalTensor<int32_t> row = sourceBuf_.Get<int32_t>();
        Duplicate(
            row, value,
            fused_quant_li_manage_workspace::ATTENTION_CAPACITY);
        UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t publicRow =
                (static_cast<uint64_t>(queryStart) + route) *
                fused_quant_li_manage_workspace::ATTENTION_CAPACITY;
            DataCopy(
                topkSlotsGm_[publicRow], row,
                fused_quant_li_manage_workspace::ATTENTION_CAPACITY);
            DataCopy(
                topkIndexGm_[publicRow], row,
                fused_quant_li_manage_workspace::ATTENTION_CAPACITY);
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void PublishRequestMetadata(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount,
        int32_t perQueryCount, int32_t copyCount)
    {
        LocalTensor<int32_t> values = countBuf_.Get<int32_t>();
        for (uint32_t route = 0U; route < routeCount; ++route) {
            values.SetValue(route * 8U, perQueryCount);
        }
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        for (uint32_t route = 0U; route < routeCount; ++route)
            DataCopyPad(
                routeStatusGm_[(queryStart + route) * 8U],
                values[route * 8U],
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        // Every UB source passed to MTE3 must be 32-byte aligned. Reuse the
        // first aligned scalar lane sequentially instead of addressing
        // values[4]/values[5], whose byte offsets are only 16/20.
        values.SetValue(0U, copyCount);
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (batchSize_ == 1U)
            DataCopyPad(
                unionCountsGm_[batch], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        else
            DataCopyPad(
                requestStatusGm_[batch * 8U], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void PublishError(uint32_t batch)
    {
        uint32_t safeStart = 0U, safeEnd = 0U;
        metadata_.SafeRouteRange(batch, safeStart, safeEnd);
        FillAttentionRows(safeStart, safeEnd - safeStart, -1);
        // An invalid Q can exceed the four-lane metadata scratch. Reuse one
        // aligned scalar source for every safe route, then publish the
        // request count through the same single-writer staging row.
        LocalTensor<int32_t> values = countBuf_.Get<int32_t>();
        values.SetValue(0U, 0);
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        for (uint32_t route = safeStart; route < safeEnd; ++route)
            DataCopyPad(routeStatusGm_[route * 8U], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        if (batchSize_ == 1U)
            DataCopyPad(unionCountsGm_[batch], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        else
            DataCopyPad(requestStatusGm_[batch * 8U], values,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline bool ValidateSparse(
        uint32_t batch, uint32_t routeCount,
        uint32_t &candidate, uint32_t &finalLen,
        uint32_t &poolRow, uint32_t &budget)
    {
        const int32_t candidateRaw = offloadedLensGm_.GetValue(batch);
        const int32_t finalRaw = finalSeqLengthsKvGm_.GetValue(batch);
        const int32_t poolRowRaw = reqPoolEntriesGm_.GetValue(batch);
        if (poolRowRaw < 0 ||
            poolRowRaw >= static_cast<int32_t>(poolSize_) ||
            candidateRaw < static_cast<int32_t>(UNION_TOPK) ||
            candidateRaw > static_cast<int32_t>(sourceCapacity_) ||
            finalRaw > static_cast<int32_t>(maxCandidateLen_) ||
            candidateRaw % lim_request_limits::BLOCK_SIZE != 0 ||
            finalRaw < static_cast<int32_t>(routeCount) ||
            finalRaw > static_cast<int32_t>(sourceCapacity_)) {
            return false;
        }
        candidate = static_cast<uint32_t>(candidateRaw);
        finalLen = static_cast<uint32_t>(finalRaw);
        poolRow = static_cast<uint32_t>(poolRowRaw);
        const uint32_t earliestVisible = finalLen - (routeCount - 1U);
        if (candidate > earliestVisible) {
            return false;
        }
        const int32_t budgetRaw = numCacheTokensGm_.GetValue(batch);
        if (budgetRaw <= 0) return false;
        budget = static_cast<uint32_t>(budgetRaw);
        return lim_cache_budget::Valid(
            budget, candidate, routeCount, finalLen - candidate);
    }

    __aicore__ inline void ProcessSparse(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount)
    {
        uint32_t candidate = 0U;
        uint32_t finalLen = 0U;
        uint32_t poolRow = 0U;
        uint32_t budget = 0U;
        if (!ValidateSparse(
                batch, routeCount, candidate, finalLen, poolRow, budget)) {
            PublishError(batch);
            return;
        }
#if LIM_SIMT_REQUEST_META_CACHE
        // ValidateSparse has already read and checked these four values.
        // Keep only a request-local snapshot, never a member / GM / UB cache:
        // the next request and every graph replay validate fresh metadata.
        // Steady updates change token columns, not the positive budget word.
        const SparseRequestMetadata metadata{candidate, finalLen, poolRow, budget};
#endif

        LocalTensor<float> pairs = pairInputBuf_.Get<float>();
        LocalTensor<float> merged = pairOutputBuf_.Get<float>();
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> countLocal = countBuf_.Get<int32_t>();
        const uint64_t routeBase =
            WorkspaceRouteRow(batch, queryStart, 0U) * UNION_PAIR_WORDS;
        // Internal workspaces retain four fixed lanes per request, while the
        // public ABI is compact TND.  Read only the actually present lanes.
        for (uint32_t route = 0U; route < routeCount; ++route) {
            DataCopyPad(
                countLocal[route *
                    fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE],
                routeCountsGm_[
                    WorkspaceRouteRow(batch, queryStart, route) *
                    fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE],
                {1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                {false, 0, 0, 0});
        }
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[UNION_ROUTES] = {0U, 0U, 0U, 0U};
        uint32_t total = 0U;
        LocalTensor<int32_t> pairWords = pairs.ReinterpretCast<int32_t>();
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const int32_t rawLength = countLocal.GetValue(
                route * fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE);
            if (rawLength < 0 || rawLength > static_cast<int32_t>(UNION_TOPK)) {
                PublishError(batch);
                return;
            }
            const uint32_t length = static_cast<uint32_t>(rawLength);
            lengths[route] = length;
            total += length;
            if (length != 0U) {
                const uint32_t pairOffset = route * UNION_PAIR_WORDS;
                DataCopyPad(
                    pairWords[pairOffset],
                    routePairsGm_[routeBase + pairOffset],
                    {1, static_cast<uint32_t>(
                            length * 2U * sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
            }
        }
        if (total == 0U) {
            PublishFinalOutputs(
                batch, queryStart, routeCount, lengths,
                0U, countLocal,
#if LIM_SIMT_REQUEST_META_CACHE
                metadata,
#endif
                sources, sources, sources);
            return;
        }

        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit =
            (lengths[0] > 0U ? 0b0001 : 0U) |
            (lengths[1] > 0U ? 0b0010 : 0U) |
            (lengths[2] > 0U ? 0b0100 : 0U) |
            (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> inputs;
        inputs.src1 = pairs;
        inputs.src2 = pairs[UNION_PAIR_WORDS];
        inputs.src3 = pairs[UNION_PAIR_WORDS * 2U];
        inputs.src4 = pairs[UNION_PAIR_WORDS * 3U];
        MrgSort<float>(merged, inputs, params);
        PipeBarrier<PIPE_V>();

        const uint32_t count = Deduplicate(merged, total, sources);
        if (count == 0U || count > total ||
            count > budget || count > outputCapacity_) {
            PublishError(batch);
            return;
        }
        LocalTensor<int32_t> victimStorage =
            pairOutputBuf_.Get<int32_t>();
        LocalTensor<int32_t> destinations = victimStorage;
        LocalTensor<int32_t> victimSources =
            victimStorage[UNION_CAPACITY];
        // Reuse the validated request-local length; both victim passes must
        // start from the same seed, refreshed on every invocation/replay.
        const uint32_t scanSeed =
            fused_quant_li_manage_victim_scan::HashVictimScanSeed(
                finalLen);
        const uint32_t found = FindVictims(
            batch, count, routeCount, queryStart, scanSeed,
#if LIM_SIMT_REQUEST_META_CACHE
            metadata,
#endif
            destinations, victimSources);
        if (found != count) {
            PublishError(batch);
            return;
        }
        const uint32_t updated = ApplyCacheUpdates(
            batch, count,
#if LIM_SIMT_REQUEST_META_CACHE
            metadata,
#endif
            sources, destinations, victimSources);
        if (updated != count) {
            PublishError(batch);
            return;
        }
#if LIM_SIMT_BATCHED_REPAIR
        // ApplyCacheUpdates has completed (V_S + PIPE_ALL). Victim sources
        // are now dead; reuse these 8192 words for four repaired prefixes.
        // The first half of pairOutput (union destinations) stays read-only
        // until PublishFinalOutputs finishes MTE3. No extra UB allocation.
        LocalTensor<int32_t> topkMissDestinations = victimSources;
#else
        LocalTensor<int32_t> topkScratch = pairInputBuf_.Get<int32_t>();
        LocalTensor<int32_t> topkMissDestinations =
            topkScratch[UNION_PAIR_WORDS];
#endif
        PrepareTopkMissPrefixes(
            batch, queryStart, routeCount, lengths, updated,
            sources, destinations, topkMissDestinations);
        PublishFinalOutputs(
            batch, queryStart, routeCount, lengths, updated,
            countLocal,
#if LIM_SIMT_REQUEST_META_CACHE
            metadata,
#endif
            sources, destinations, topkMissDestinations);
    }

    // A3 ProcessGeneralizedSteady concatenates misses before SortAll.
    // Encode full source IDs as finite descending keys, retaining the source
    // itself as the pair payload for the existing vector deduplicator.
    __aicore__ inline bool SortSmallWide(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount,
        const uint32_t lengths[UNION_MAX_ROUTES], uint32_t total,
        uint32_t candidate, uint32_t budget, uint32_t &unionCount)
    {
        const uint32_t sortLength =
            fused_quant_li_manage_sort::SortLength(total);
        LocalTensor<float> input = pairInputBuf_.Get<float>();
        LocalTensor<float> workspace = pairOutputBuf_.Get<float>();
        LocalTensor<int32_t> pairs = input.ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> words = workspace.ReinterpretCast<uint32_t>();
        Duplicate(words, 0U, sortLength * 2U);
        PipeBarrier<PIPE_V>();
        uint32_t offsets[UNION_MAX_ROUTES] = {0U};
        uint32_t storage = 0U;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            if (lengths[route] == 0U) continue;
            storage = (storage + 7U) & ~7U;
            offsets[route] = storage;
            storage += lengths[route];
        }
        if (storage <= sortLength) {
            // A3's aligned vector build keeps zero-key gaps between routes.
            // The public miss-prefix already contains the ordered full IDs,
            // so load half as many words as the private interleaved pair row.
            LocalTensor<int32_t> routeSources = sourceBuf_.Get<int32_t>();
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) continue;
                DataCopyPad(routeSources[offsets[route]],
                    topkIndexGm_[static_cast<uint64_t>(queryStart + route) * UNION_TOPK],
                    {1, lengths[route] * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0}, {false, 0, 0, 0});
            }
            UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) continue;
                const int32_t first = routeSources.GetValue(offsets[route]);
                const int32_t last = routeSources.GetValue(
                    offsets[route] + lengths[route] - 1U);
                if (first < 0 || last < first ||
                    static_cast<uint32_t>(last) >= candidate)
                    return false;
            }
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            LocalTensor<int32_t> keys = words.ReinterpretCast<int32_t>();
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) continue;
                Muls(keys[offsets[route]], routeSources[offsets[route]],
                     -1, lengths[route]);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) continue;
                const uint32_t offset = offsets[route];
                Adds(keys[offset], keys[offset], static_cast<int32_t>(
                    0x40000000U + lim_payload_layout::LONG_SOURCE_MASK),
                    lengths[route]);
                DataCopy(words[sortLength + offset],
                         routeSources[offset].ReinterpretCast<uint32_t>(),
                         (lengths[route] + 7U) & ~7U);
            }
            PipeBarrier<PIPE_V>();
        } else {
            // Exact compaction remains necessary when alignment gaps cross the
            // selected bucket (for example, 4095 occurrences across six routes).
            UnionSync<HardEvent::V_S>(HardEvent::V_S);
            uint32_t packed = 0U;
            for (uint32_t route = 0U; route < routeCount; ++route) {
                const uint32_t length = lengths[route];
                if (length == 0U) continue;
                DataCopyPad(
                    pairs, routePairsGm_[static_cast<uint64_t>(queryStart + route) *
                                         UNION_PAIR_WORDS],
                    {1, length * 2U * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0}, {false, 0, 0, 0});
                UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                for (uint32_t position = 0U; position < length; ++position) {
                    const int32_t source = pairs.GetValue(position * 2U + 1U);
                    if (source < 0 || static_cast<uint32_t>(source) >= candidate)
                        return false;
                    words.SetValue(packed,
                        0x40000000U + lim_payload_layout::LONG_SOURCE_MASK -
                            static_cast<uint32_t>(source));
                    words.SetValue(sortLength + packed,
                                   static_cast<uint32_t>(source));
                    ++packed;
                }
                // The next route reuses the local input row for MTE2.
                UnionSync<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            }
            UnionSync<HardEvent::S_V>(HardEvent::S_V);
        }
        fused_quant_li_manage_sort::SortAll(workspace, input, sortLength);
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        unionCount = Deduplicate(workspace, total, sources);
        if (unionCount == 0U || unionCount > total ||
            unionCount > budget || unionCount > outputCapacity_)
            return false;

        UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        const uint64_t missBase = static_cast<uint64_t>(batch) * outputCapacity_;
        for (uint32_t copied = 0U; copied < unionCount;) {
            const uint32_t chunk = unionCount - copied < UNION_TOPK
                ? unionCount - copied : UNION_TOPK;
            DataCopyPad(
                unionSourcesGm_[missBase + copied], sources[copied],
                {1, static_cast<uint16_t>(chunk * sizeof(int32_t)), 0, 0});
            copied += chunk;
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        return true;
    }

    // A3's large-occurrence fallback joins the already source-ordered route
    // prefixes. Keep one head per route in scalar storage and stream unique
    // sources through one UB row. The original pair rows are never rewritten
    // into intermediate four-route groups.
    __aicore__ inline bool MergeWideSources(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount,
        const uint32_t lengths[UNION_MAX_ROUTES], uint32_t candidate,
        uint32_t budget, uint32_t &unionCount)
    {
        constexpr uint32_t NO_SOURCE = 0xffffffffU;
        uint32_t cursors[UNION_MAX_ROUTES] = {0U};
        uint32_t heads[UNION_MAX_ROUTES];
        for (uint32_t route = 0U; route < routeCount; ++route) {
            heads[route] = NO_SOURCE;
            if (lengths[route] == 0U) continue;
            const int32_t source = routePairsGm_.GetValue(
                static_cast<uint64_t>(queryStart + route) * UNION_PAIR_WORDS + 1U);
            if (source < 0 || static_cast<uint32_t>(source) >= candidate)
                return false;
            heads[route] = static_cast<uint32_t>(source);
        }
        const uint64_t missBase = static_cast<uint64_t>(batch) * outputCapacity_;
        LocalTensor<int32_t> sources = sourceBuf_.Get<int32_t>();
        uint32_t buffered = 0U;
        unionCount = 0U;
        while (true) {
            uint32_t next = NO_SOURCE;
            for (uint32_t route = 0U; route < routeCount; ++route)
                if (heads[route] < next) next = heads[route];
            if (next == NO_SOURCE) break;
            if (unionCount >= budget || unionCount >= outputCapacity_)
                return false;
            sources.SetValue(buffered++, static_cast<int32_t>(next));
            ++unionCount;
            for (uint32_t route = 0U; route < routeCount; ++route) {
                while (heads[route] == next) {
                    ++cursors[route];
                    if (cursors[route] == lengths[route]) {
                        heads[route] = NO_SOURCE;
                        break;
                    }
                    const int32_t source = routePairsGm_.GetValue(
                        static_cast<uint64_t>(queryStart + route) * UNION_PAIR_WORDS +
                        static_cast<uint64_t>(cursors[route]) * 2U + 1U);
                    if (source < 0 || static_cast<uint32_t>(source) >= candidate ||
                        static_cast<uint32_t>(source) < next)
                        return false;
                    heads[route] = static_cast<uint32_t>(source);
                }
            }
            if (buffered == UNION_TOPK) {
                UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(unionSourcesGm_[missBase + unionCount - buffered],
                            sources, {1, static_cast<uint16_t>(
                                buffered * sizeof(int32_t)), 0, 0});
                UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                buffered = 0U;
            }
        }
        if (buffered != 0U) {
            UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(unionSourcesGm_[missBase + unionCount - buffered],
                        sources, {1, static_cast<uint16_t>(
                            buffered * sizeof(int32_t)), 0, 0});
            UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        return unionCount != 0U;
    }

    __aicore__ inline void LoadWideUpdateChunk(
        uint64_t missBase, uint64_t victimScratch, uint32_t offset,
        uint32_t length, LocalTensor<int32_t> incoming,
        LocalTensor<int32_t> victims, LocalTensor<int32_t> destinations,
        bool incomingResident = false)
    {
        // Complete the previous chunk's scalar/SIMT reads before reusing UB.
        UnionSync<HardEvent::S_MTE2>(HardEvent::S_MTE2);
#if LIM_SIMT_POSTPROCESS
        UnionSync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
#endif
        const DataCopyExtParams copy{
            1, length * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        if (!incomingResident)
            DataCopyPad(incoming, unionSourcesGm_[missBase + offset], copy, pad);
        DataCopyPad(victims, routePairsGm_[victimScratch + offset], copy, pad);
        DataCopyPad(destinations, unionDestinationsGm_[missBase + offset], copy, pad);
#if LIM_SIMT_POSTPROCESS
        UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
#else
        UnionSync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
#endif
    }

#if LIM_SIMT_POSTPROCESS
    __aicore__ inline bool RunWideCheckedUpdates(
        uint64_t cacheBase, uint32_t count, uint32_t candidate, uint32_t budget,
        LocalTensor<int32_t> incoming, LocalTensor<int32_t> victims,
        LocalTensor<int32_t> destinations, bool commit)
    {
        LocalTensor<uint32_t> validPrefix = countBuf_.Get<uint32_t>()[1U];
        validPrefix.SetValue(0U, count);
        UnionSync<HardEvent::S_V>(HardEvent::S_V);
        asc_vf_call<lim_simt::ApplyWideUpdatesChecked>(
            dim3(LIM_SIMT_THREADS),
            (__gm__ int32_t *)cacheSlotsGm_.GetPhyAddr(), cacheBase,
            (__ubuf__ int32_t *)incoming.GetPhyAddr(),
            (__ubuf__ int32_t *)destinations.GetPhyAddr(),
            (__ubuf__ int32_t *)victims.GetPhyAddr(),
            (__ubuf__ uint32_t *)validPrefix.GetPhyAddr(),
            count, candidate, budget, commit);
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        return validPrefix.GetValue(0U) == count;
    }
#endif

    __aicore__ inline bool ValidateWideUpdateRows(
        uint64_t cacheBase, uint32_t count, uint32_t candidate, uint32_t budget,
        LocalTensor<int32_t> incoming, LocalTensor<int32_t> victims,
        LocalTensor<int32_t> destinations)
    {
#if LIM_SIMT_POSTPROCESS
        return RunWideCheckedUpdates(cacheBase, count, candidate, budget,
                                     incoming, victims, destinations, false);
#else
        for (uint32_t index = 0U; index < count; ++index) {
            const int32_t source = incoming.GetValue(index);
            const int32_t victim = victims.GetValue(index);
            const int32_t slot = destinations.GetValue(index);
            if (source < 0 || static_cast<uint32_t>(source) >= candidate ||
                victim < 0 || static_cast<uint32_t>(victim) >= candidate ||
                slot < 0 || static_cast<uint32_t>(slot) >= budget ||
                source == victim)
                return false;
            if (cacheSlotsGm_.GetValue(
                    cacheBase + static_cast<uint32_t>(victim)) != slot)
                return false;
        }
        return true;
#endif
    }

    __aicore__ inline void CommitWideUpdateRows(
        uint64_t cacheBase, uint32_t count, LocalTensor<int32_t> incoming,
        LocalTensor<int32_t> victims, LocalTensor<int32_t> destinations)
    {
#if LIM_SIMT_POSTPROCESS
        asc_vf_call<lim_simt::CommitWideUpdates>(
            dim3(LIM_SIMT_THREADS),
            (__gm__ int32_t *)cacheSlotsGm_.GetPhyAddr(), cacheBase,
            (__ubuf__ int32_t *)incoming.GetPhyAddr(),
            (__ubuf__ int32_t *)destinations.GetPhyAddr(),
            (__ubuf__ int32_t *)victims.GetPhyAddr(), count);
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
#else
        for (uint32_t index = 0U; index < count; ++index) {
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(victims.GetValue(index)),
                lim_pool_contract::INVALID_POOL_SLOT);
            cacheSlotsGm_.SetValue(
                cacheBase + static_cast<uint32_t>(incoming.GetValue(index)),
                destinations.GetValue(index));
        }
#endif
    }

    __aicore__ inline bool ApplyWideUpdates(
        uint64_t missBase, uint64_t victimScratch, uint64_t cacheBase,
        uint32_t count, uint32_t candidate, uint32_t budget,
        bool incomingResident)
    {
        // MergeWideSources scalar-loads the original route-pair rows into
        // DCache. FindVictims then DMA-writes victim IDs over the same GM
        // range. PIPE_ALL/MTE3_S complete DMA but do not invalidate DCache:
        // a subsequent routePairsGm_.GetValue can return an old float key
        // (0x400xxxxx) and turn it into an out-of-range pool address.
        // Read update lists only from UB: retain a complete incoming union
        // when available, and reload every DMA-written victim/slot list.
        if (count <= UNION_CAPACITY) {
            // Victim selection releases both pair buffers; sourceBuf_ still
            // holds any complete incoming union retained by the union phase.
            // Keep <=8192 complete update rows in UB, validate once, then
            // commit without a second GM load. FindVictims never uses
            // sourceBuf_, so a complete union already held there stays live.
            LocalTensor<int32_t> incoming = sourceBuf_.Get<int32_t>();
            LocalTensor<int32_t> victims = pairInputBuf_.Get<int32_t>();
            LocalTensor<int32_t> destinations = pairOutputBuf_.Get<int32_t>();
            LoadWideUpdateChunk(missBase, victimScratch, 0U, count,
                                incoming, victims, destinations, incomingResident);
#if LIM_SIMT_POSTPROCESS
            return RunWideCheckedUpdates(cacheBase, count, candidate, budget,
                                         incoming, victims, destinations, true);
#else
            if (!ValidateWideUpdateRows(cacheBase, count, candidate, budget,
                                        incoming, victims, destinations))
                return false;
            CommitWideUpdateRows(cacheBase, count, incoming, victims, destinations);
            return true;
#endif
        }
        static_assert(3U * UNION_TOPK <= UNION_CAPACITY * 2U,
                      "wide update rows must fit pairInputBuf_");
        LocalTensor<int32_t> incoming = pairInputBuf_.Get<int32_t>();
        LocalTensor<int32_t> victims = incoming[UNION_TOPK];
        LocalTensor<int32_t> destinations = incoming[UNION_TOPK * 2U];

        // Validate every update before changing any pool entry. Apart from
        // bounding scalar addresses, this preserves request-level failure
        // atomicity if an internal source/slot list is malformed.
        for (uint32_t offset = 0U; offset < count; offset += UNION_TOPK) {
            const uint32_t length = count - offset < UNION_TOPK
                ? count - offset : UNION_TOPK;
            LoadWideUpdateChunk(missBase, victimScratch, offset, length,
                                incoming, victims, destinations);
            if (!ValidateWideUpdateRows(cacheBase, length, candidate, budget,
                                        incoming, victims, destinations))
                return false;
        }
        for (uint32_t offset = 0U; offset < count; offset += UNION_TOPK) {
            const uint32_t length = count - offset < UNION_TOPK
                ? count - offset : UNION_TOPK;
            LoadWideUpdateChunk(missBase, victimScratch, offset, length,
                                incoming, victims, destinations);
            CommitWideUpdateRows(cacheBase, length, incoming, victims, destinations);
        }
        return true;
    }

    __aicore__ inline void PublishWideCounts(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount,
        const uint32_t lengths[UNION_MAX_ROUTES], uint32_t count)
    {
        // Both pair buffers are dead after repair (also on the zero-miss
        // path). Give all Q<=14 rows independent 32-byte DMA sources without
        // changing the mature four-route countBuf_ allocation.
        LocalTensor<int32_t> value = pairInputBuf_.Get<int32_t>();
        Duplicate(value, 0, (routeCount + 1U) * 8U);
        UnionSync<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t route = 0U; route < routeCount; ++route) {
            value.SetValue(route * 8U, static_cast<int32_t>(lengths[route]));
        }
        value.SetValue(routeCount * 8U, static_cast<int32_t>(count));
        UnionSync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(routeStatusGm_[queryStart * 8U], value,
            {1, static_cast<uint16_t>(routeCount * 32U), 0, 0});
        if (batchSize_ == 1U) {
            DataCopyPad(unionCountsGm_[batch], value[routeCount * 8U],
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        } else {
            DataCopyPad(requestStatusGm_[batch * 8U], value[routeCount * 8U],
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

#if LIM_SIMT_POSTPROCESS
    __aicore__ inline void RepairWideMissPrefixes(
        uint32_t queryStart, uint32_t routeCount,
        const uint32_t lengths[UNION_MAX_ROUTES], uint64_t missBase,
        uint32_t unionCount)
    {
        // Four source rows and four destination rows exactly fit pairInput.
        // The <=8192 union sources/destinations retained by ApplyWideUpdates
        // stay live in sourceBuf_/pairOutputBuf_ throughout every group.
        static_assert(UNION_ROUTES * UNION_TOPK * 2U == UNION_CAPACITY * 2U,
                      "wide repair rows must fit pairInputBuf_");
        LocalTensor<int32_t> routeSources = pairInputBuf_.Get<int32_t>();
        LocalTensor<int32_t> output = routeSources[UNION_CAPACITY];
        for (uint32_t group = 0U; group < routeCount; group += UNION_ROUTES) {
            uint32_t groupLengths[UNION_ROUTES] = {0U};
            uint32_t total = 0U;
            for (uint32_t lane = 0U; lane < UNION_ROUTES; ++lane) {
                const uint32_t route = group + lane;
                if (route >= routeCount) break;
                const uint32_t length = lengths[route];
                groupLengths[lane] = length;
                total += length;
                if (length != 0U)
                    DataCopyPad(routeSources[lane * UNION_TOPK],
                        topkIndexGm_[static_cast<uint64_t>(queryStart + route) * UNION_TOPK],
                        {1, length * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                        {false, 0, 0, 0});
            }
            if (total == 0U) continue;
            UnionSync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            if (unionCount <= UNION_CAPACITY) {
                asc_vf_call<lim_simt::RepairWideMissSlots>(
                    dim3(LIM_SIMT_THREADS),
                    (__ubuf__ int32_t *)routeSources.GetPhyAddr(),
                    (__ubuf__ int32_t *)sourceBuf_.Get<int32_t>().GetPhyAddr(),
                    (__ubuf__ int32_t *)pairOutputBuf_.Get<int32_t>().GetPhyAddr(),
                    (__ubuf__ int32_t *)output.GetPhyAddr(),
                    groupLengths[0], groupLengths[1], groupLengths[2], groupLengths[3],
                    unionCount);
            } else {
                asc_vf_call<lim_simt::RepairWideMissSlotsGm>(
                    dim3(LIM_SIMT_THREADS),
                    (__ubuf__ int32_t *)routeSources.GetPhyAddr(),
                    (__gm__ int32_t *)unionSourcesGm_[missBase].GetPhyAddr(),
                    (__gm__ int32_t *)unionDestinationsGm_[missBase].GetPhyAddr(),
                    (__ubuf__ int32_t *)output.GetPhyAddr(),
                    groupLengths[0], groupLengths[1], groupLengths[2], groupLengths[3],
                    unionCount);
            }
            UnionSync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            for (uint32_t lane = 0U; lane < UNION_ROUTES; ++lane) {
                const uint32_t length = groupLengths[lane];
                if (length != 0U)
                    DataCopyPad(
                        topkSlotsGm_[static_cast<uint64_t>(queryStart + group + lane) * UNION_TOPK],
                        output[lane * UNION_TOPK],
                        {1, static_cast<uint16_t>(length * sizeof(int32_t)), 0, 0});
            }
            UnionSync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            UnionSync<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
        UnionSync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
#endif

    // The existing Q1-4 vector union stays unchanged. Wide requests use a
    // fixed-bucket SortAll for small totals and an ordered k-way union otherwise.
    __aicore__ inline void ProcessSparseWide(
        uint32_t batch, uint32_t queryStart, uint32_t routeCount)
    {
        uint32_t candidate = 0U, finalLen = 0U, poolRow = 0U, budget = 0U;
        if (!ValidateSparse(batch, routeCount, candidate, finalLen,
                            poolRow, budget)) {
            PublishError(batch);
            return;
        }
        uint32_t lengths[UNION_MAX_ROUTES] = {0U};
        uint32_t total = 0U;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const int32_t raw = routeCountsGm_.GetValue(
                WorkspaceRouteRow(batch, queryStart, route) *
                fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE);
            if (raw < 0 || raw > static_cast<int32_t>(UNION_TOPK)) {
                PublishError(batch);
                return;
            }
            lengths[route] = static_cast<uint32_t>(raw);
            total += lengths[route];
        }
        if (total == 0U) {
            PublishWideCounts(batch, queryStart, routeCount, lengths, 0U);
            return;
        }
        uint32_t unionCount = 0U;
        const uint64_t missBase = static_cast<uint64_t>(batch) * outputCapacity_;
        const bool unionReady = total <= UNION_SMALL_MISS_OCCURRENCES
            ? SortSmallWide(batch, queryStart, routeCount, lengths, total,
                            candidate, budget, unionCount)
            : MergeWideSources(batch, queryStart, routeCount, lengths,
                               candidate, budget, unionCount);
        if (!unionReady) {
            PublishError(batch);
            return;
        }

        // Q>4 uses the same strict score-threshold prefilter as Q1-4, in
        // 512-source chunks. Exact TopK-slot protection runs only if needed.
        const uint32_t scanSeed =
            fused_quant_li_manage_victim_scan::HashVictimScanSeed(finalLen);
        const uint32_t victimCount =
            fused_quant_li_manage_wide_victim::FindVictims(
                cacheSlotsGm_, scoreWorkspaceGm_, routeThresholdsGm_,
                topkSlotsGm_, routePairsGm_, unionDestinationsGm_,
                queryStart, routeCount,
                WorkspaceRouteRow(batch, queryStart, 0U), scoreRowStride_,
                candidate, poolRow, poolStride_, budget, unionCount,
                scanSeed, missBase,
                pairInputBuf_.Get<uint16_t>(), pairOutputBuf_.Get<uint8_t>());
        const uint64_t cacheBase = static_cast<uint64_t>(poolRow) * poolStride_;
        const uint64_t victimScratch =
            static_cast<uint64_t>(queryStart) * UNION_PAIR_WORDS;
        if (victimCount != unionCount) {
            PublishError(batch);
            return;
        }
        PipeBarrier<PIPE_ALL>();
        const bool incomingResident = total <= UNION_SMALL_MISS_OCCURRENCES ||
                                      unionCount <= UNION_TOPK;
        if (!ApplyWideUpdates(missBase, victimScratch, cacheBase,
                              unionCount, candidate, budget, incomingResident)) {
            PublishError(batch);
            return;
        }
        PipeBarrier<PIPE_ALL>();
#if LIM_SIMT_POSTPROCESS
        RepairWideMissPrefixes(queryStart, routeCount, lengths, missBase, unionCount);
#else
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const uint64_t topkBase =
                static_cast<uint64_t>(queryStart + route) * UNION_TOPK;
            for (uint32_t index = 0U; index < lengths[route]; ++index) {
                const int32_t source = topkIndexGm_.GetValue(topkBase + index);
                if (source >= 0 && static_cast<uint32_t>(source) < candidate)
                    topkSlotsGm_.SetValue(
                        topkBase + index,
                        cacheSlotsGm_.GetValue(cacheBase +
                            static_cast<uint32_t>(source)));
            }
        }
#endif
        PublishWideCounts(batch, queryStart, routeCount, lengths, unionCount);
    }

    __aicore__ inline void ProcessRequest(uint32_t batch)
    {
        // The standardized -3 identity and -2 first-fill lifecycle retain
        // their dedicated publishers. The ordered union owns only -1.
        if (rowModesGm_.GetValue(batch) != UNION_ROW_MODE_SPARSE)
            return;
        uint32_t queryStart = 0U;
        uint32_t queryEnd = 0U;
        uint32_t routeCount = 0U;
        if (!QueryRange(batch, queryStart, queryEnd, routeCount)) {
            PublishError(batch);
            return;
        }

        if (!metadata_.Valid(batch)) {
            PublishError(batch);
            return;
        }
        if (routeCount <= UNION_ROUTES) {
            ProcessSparse(batch, queryStart, routeCount);
        } else {
            ProcessSparseWide(batch, queryStart, routeCount);
        }
    }

    GlobalTensor<int32_t> routePairsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<int32_t> routeCountsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> offloadedLensGm_;
    GlobalTensor<int32_t> finalSeqLengthsKvGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    GlobalTensor<int32_t> numCacheTokensGm_;
    GlobalTensor<int32_t> unionSourcesGm_;
    GlobalTensor<int32_t> unionDestinationsGm_;
    GlobalTensor<int32_t> unionCountsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> topkIndexGm_;
    GlobalTensor<int32_t> perQueryMissCountsGm_;
    GlobalTensor<int32_t> requestStatusGm_;
    GlobalTensor<int32_t> routeStatusGm_;
    lim_request_contract::Metadata metadata_;
    const FusedQuantLightningIndexerManageTilingData *tiling_ = nullptr;
    TBuf<TPosition::VECCALC> pairInputBuf_;
    TBuf<TPosition::VECCALC> pairOutputBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> thresholdBuf_;
    uint32_t sourceCapacity_ = 0U;
    uint32_t poolStride_ = 0U;
    uint32_t outputCapacity_ = 0U;
    uint32_t scoreRowStride_ = 0U;
    uint32_t batchSize_ = 0U;
    uint32_t poolSize_ = 0U;
    uint32_t totalQueryRows_ = 0U;
    uint32_t maxCandidateLen_ = 0U;
};
} // namespace fused_quant_li_manage

#endif
