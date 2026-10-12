/** A5 C8 payload QLI for valid -3/-2/-1 requests. */

#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_QLI_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_QLI_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_quant_lightning_indexer_manage_tiling.h"
#include "fused_quant_lightning_indexer_manage_workspace.h"
#include "lim_cache_budget.h"
#include "lim_request_contract.h"

// The payload implementation keeps its private namespaces and include guards.
using VllmFastLIC8TilingData = FusedQuantLightningIndexerManageTilingData;
#define A5_MTP_CLASSIFY_ONLY 1
#define A5_MTP_SLOT_OUTPUT_STRIDE lim_request_limits::TOPK
#include "arch35_payload_c8/lightning_indexer_common.h"
#include "arch35_payload_c8/lightning_indexer_service_cube.h"
#include "arch35_payload_c8/lightning_indexer_service_vector.h"
#undef A5_MTP_SLOT_OUTPUT_STRIDE
#undef A5_MTP_CLASSIFY_ONLY

namespace fused_quant_li_manage {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = lim_request_limits::BLOCK_SIZE;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t TOPK = lim_request_limits::TOPK;
constexpr uint32_t ROUTES = 4U;
constexpr uint32_t MAX_ROUTES = lim_request_limits::MAX_ROUTES;
constexpr int32_t ROW_MODE_DENSE = lim_request_limits::IDENTITY_STATE;
constexpr int32_t ROW_MODE_FIRST_FILL = lim_request_limits::FIRST_FILL_STATE;
constexpr int32_t ROW_MODE_SPARSE = lim_request_limits::STEADY_STATE;
constexpr uint32_t THRESHOLD_STRIDE = 16U;
constexpr uint16_t INACTIVE_THRESHOLD = 0xffffU;

using C8MtpQliType = VllmFastQLICommon::QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true,
    VllmFastQLICommon::LI_LAYOUT::TND,
    VllmFastQLICommon::LI_LAYOUT::PA_BSND,
    bfloat16_t, float32_t, float32_t, uint16_t>;

// All valid requests use this LI phase. First-fill uses dense-direct TopK
// publication so its old pool is ignored. Metadata is checked per request.
class QuantLiMtpPhase {
public:
    __aicore__ inline QuantLiMtpPhase(
        TPipe *pipe, const FusedQuantLightningIndexerManageTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR indexWeights, GM_ADDR query, GM_ADDR queryDequantScale,
        GM_ADDR actualSeqLengthsQuery,
        GM_ADDR indexKeyCache, GM_ADDR indexKeyDequantScale,
        GM_ADDR finalSeqLengthsKv,
        GM_ADDR reqPoolEntries, GM_ADDR cacheSlotsPool,
        GM_ADDR offloadedLens, GM_ADDR numCacheTokens, GM_ADDR rowModes,
        GM_ADDR indexBlockTable,
        GM_ADDR routePairRows, GM_ADDR topkSlots,
        GM_ADDR topkSourceIds,
        GM_ADDR routeThresholds, GM_ADDR routeMissCounts,
        GM_ADDR userWorkspace,
        GM_ADDR candidateScores, GM_ADDR candidatePayloads,
        GM_ADDR candidateTags)
    {
        subBlockIdx_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            aiCoreIdx_ = subBlockIdx_;
        }

        queryEndsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery);
        finalLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)finalSeqLengthsKv);
        offloadedLensGm_.SetGlobalBuffer((__gm__ int32_t *)offloadedLens);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
        rowModesGm_.SetGlobalBuffer((__gm__ int32_t *)rowModes);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)indexBlockTable);
        metadata_.Init(actualSeqLengthsQuery, finalSeqLengthsKv,
                       offloadedLens, numCacheTokens, rowModes,
                       reqPoolEntries, tiling_);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.tSize = tiling_->totalQueryRows;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = HEAD_DIM;
        constInfo_.sparseCount = TOPK;
        constInfo_.kSeqSize = tiling_->scoreRowStride;
        constInfo_.qSeqSize = tiling_->compactRouteRows ? MAX_ROUTES : ROUTES;
        constInfo_.kCacheBlockSize = BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = VllmFastQLICommon::LI_LAYOUT::TND;
        // Both states scan the full final length and clip each route.
        constInfo_.attenMaskFlag = false;
        constInfo_.isAccumSeqS1 = true;
        constInfo_.s1BaseSize = tiling_->queryTileSize;
        constInfo_.mBaseSize =
            constInfo_.s1BaseSize * tiling_->indexHeads;
        constInfo_.s2BaseSize = BLOCK_SIZE;
        constInfo_.keyStride0 = tiling_->keyStride;
        constInfo_.keyDequantScaleStride0 = tiling_->scaleStride;
        constInfo_.poolSize = tiling_->poolSize;
        // Standard pool rows contain source mappings only; no state column.
        constInfo_.cacheSlotsSize =
            static_cast<uint64_t>(tiling_->sourceCapacity);
        // Match the official A5 QuantLI policy: when the average request fits
        // in one S1 tile, bypass L2 for the streaming Key and Key-scale reads.
        constInfo_.setL2DisableFlag =
            tiling_->totalQueryRows / tiling_->batchSize <=
            constInfo_.s1BaseSize;

        scoreWorkspaceBaseGm_.SetGlobalBuffer(
            (__gm__ uint16_t *)userWorkspace);
        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)indexWeights);
            queryScaleGm_.SetGlobalBuffer(
                (__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer(
                (__gm__ float *)indexKeyDequantScale);
            if (constInfo_.setL2DisableFlag) {
                keyScaleGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            routePairRowsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)routePairRows);
            topkSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)topkSlots);
            topkSourceIdsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)topkSourceIds);
            routeMissCountsGm_.SetGlobalBuffer(
                (__gm__ int32_t *)routeMissCounts);
            routeThresholdsGm_.SetGlobalBuffer(
                (__gm__ uint16_t *)routeThresholds);
            candidateScoresGm_.SetGlobalBuffer(
                (__gm__ uint16_t *)candidateScores);
            candidatePayloadsGm_.SetGlobalBuffer(
                (__gm__ uint32_t *)candidatePayloads);
            candidateTagsGm_.SetGlobalBuffer(
                (__gm__ uint16_t *)candidateTags);
            vectorService_.InitParams(constInfo_, tiling_);
            vectorService_.InitVecInputTensor(
                weightsGm_, queryScaleGm_, keyScaleGm_,
                routePairRowsGm_, blockTableGm_, cacheSlotsGm_,
                topkSlotsGm_, routeMissCountsGm_);
            // Dense publish writes the selected logical IDs to both public
            // TopK rows. The manager separately fills the identity pool.
            vectorService_.InitMtpTopkSourceTensor(topkSourceIdsGm_);
            vectorService_.InitMtpThresholdTensor(routeThresholdsGm_);
            vectorService_.InitCandidateTensor(
                candidateScoresGm_, candidatePayloadsGm_, candidateTagsGm_);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer(
                (__gm__ fp8_e4m3fn_t *)indexKeyCache);
            if (constInfo_.setL2DisableFlag) {
                keyGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(
                blockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        int32_t priorMaximum = 0;
        const uint32_t cached = tiling_->batchSize < CACHED_REQUESTS
            ? tiling_->batchSize : CACHED_REQUESTS;
        for (uint32_t batch = 0U; batch < cached; ++batch) {
            validRequests_[batch] = metadata_.Valid(batch, priorMaximum);
            const int32_t end = metadata_.QueryEnd(batch);
            if (end > priorMaximum) priorMaximum = end;
        }
        const uint32_t tile = tiling_->queryTileSize;
        if (tile != 2U && tile != ROUTES) {
            return;
        }
        const uint32_t queryTileCount = (ROUTES + tile - 1U) / tile;
        uint32_t taskCount = tiling_->batchSize * queryTileCount;
        if (tiling_->compactRouteRows) {
            // Wide batches use only the tiles needed by each valid request.
            // The four-route launch keeps its original task numbering.
            taskCount = 0U;
            for (uint32_t batch = 0U; batch < tiling_->batchSize; ++batch) {
                if (!(batch < CACHED_REQUESTS
                        ? validRequests_[batch] : metadata_.Valid(batch))) {
                    continue;
                }
                uint32_t start = 0U;
                uint32_t end = 0U;
                uint32_t routes = 0U;
                GetQueryRange(batch, start, end, routes);
                taskCount += (routes + tile - 1U) / tile;
            }
        }
        constexpr uint32_t S2_SPLIT_MIN_TOKENS = static_cast<uint32_t>(
            fused_quant_li_manage_workspace::S2_SPLIT_MIN_TOKENS);
        uint64_t totalBlocks = 0U;
        uint32_t activeTasks = 0U;
        bool allLong = true;
        for (uint32_t task = 0U; task < taskCount; ++task) {
            TaskInfo info{};
            if (LoadTaskInfo(task, queryTileCount, tile, info)) {
                totalBlocks += info.loopCount;
                ++activeTasks;
                allLong &= info.scanLen >= S2_SPLIT_MIN_TOKENS;
            }
        }
        const bool splitScores = activeTasks != 0U &&
            activeTasks < tiling_->usedCoreNum && allLong;
        uint64_t coreBoundaries[MAX_GLOBAL_SCHEDULE_CORES + 1U];
        uint32_t globalCoreCount = 0U;
        bool globalHasSplit = false;
        bool globalPool = activeTasks != 0U && allLong &&
            tiling_->usedCoreNum <= MAX_GLOBAL_SCHEDULE_CORES;
        if (globalPool) {
            const uint64_t possibleCores = totalBlocks / MIN_TOPK_BLOCKS;
            globalCoreCount = possibleCores < tiling_->usedCoreNum
                ? static_cast<uint32_t>(possibleCores) : tiling_->usedCoreNum;
            globalPool = globalCoreCount != 0U && BuildGlobalBoundaries(
                coreBoundaries, globalCoreCount, totalBlocks,
                taskCount, queryTileCount, tile, globalHasSplit);
        }
        if ASCEND_IS_AIV {
            vectorService_.AllocEventID();
            CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4,
                             PIPE_V>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4,
                             PIPE_V>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + 1U);
        } else {
            matmulService_.AllocEventID();
        }

        uint32_t globalLoop = 0U;
        if (globalPool) {
            // Like reference C8, each mix group owns one contiguous interval
            // in the flattened task/S2-block pool. It may cross task edges.
            const uint32_t sourceBlocks =
                (tiling_->sourceCapacity + BLOCK_SIZE - 1U) / BLOCK_SIZE;
            const uint32_t capacitySegments = sourceBlocks / MIN_TOPK_BLOCKS;
            const uint32_t candidateStride = capacitySegments < tiling_->usedCoreNum
                ? capacitySegments : tiling_->usedCoreNum;
            const uint64_t coreBegin = aiCoreIdx_ < globalCoreCount
                ? coreBoundaries[aiCoreIdx_] : totalBlocks;
            const uint64_t coreEnd = aiCoreIdx_ < globalCoreCount
                ? coreBoundaries[aiCoreIdx_ + 1U] : totalBlocks;
            uint64_t taskBegin = 0U;
            for (uint32_t task = 0U; task < taskCount; ++task) {
                TaskInfo info{};
                if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                    continue;
                }
                const uint64_t taskEnd = taskBegin + info.loopCount;
                const uint64_t overlapBegin = coreBegin > taskBegin
                    ? coreBegin : taskBegin;
                const uint64_t overlapEnd = coreEnd < taskEnd
                    ? coreEnd : taskEnd;
                if (overlapBegin < overlapEnd) {
                    uint32_t firstCore = 0U;
                    uint32_t lastCore = 0U;
                    TaskCoreSpan(coreBoundaries, globalCoreCount,
                                 taskBegin, taskEnd, firstCore, lastCore);
                    const uint32_t segmentCount = lastCore - firstCore + 1U;
                    const uint32_t s2Begin = static_cast<uint32_t>(
                        overlapBegin - taskBegin);
                    const uint32_t s2End = static_cast<uint32_t>(
                        overlapEnd - taskBegin);
                    if ASCEND_IS_AIV {
                        SetScoreWorkspace(info, tile);
                    }
                    for (uint32_t s2 = s2Begin; s2 < s2End;
                         ++s2, ++globalLoop) {
                        VllmFastQLICommon::RunInfo run{};
                        FillRunInfo(run, info, s2, s2Begin, s2End,
                                    globalLoop);
                        if ASCEND_IS_AIC {
                            matmulService_.ComputeMm1(run);
                        } else {
                            vectorService_.ProcessVec1(run);
                            if (run.isLastS2InnerLoop) {
                                if (segmentCount > 1U) {
                                    const uint32_t tokenBegin =
                                        s2Begin * BLOCK_SIZE;
                                    const uint32_t tokenEnd =
                                        s2End == info.loopCount
                                            ? info.scanLen : s2End * BLOCK_SIZE;
                                    vectorService_.ProcessTopK(
                                        run,
                                        info.sparse && overlapEnd == coreEnd,
                                        tokenBegin,
                                        tokenEnd - tokenBegin,
                                        aiCoreIdx_ - firstCore,
                                        candidateStride);
                                } else {
                                    vectorService_.ProcessTopK(
                                        run, info.sparse &&
                                            overlapEnd == coreEnd);
                                }
                            }
                        }
                    }
                }
                taskBegin = taskEnd;
            }
        } else if (!splitScores) {
            // Keep the original score/TopK order when a short tile or enough
            // independent tiles would make an S2 split unnecessary.
            for (uint32_t task = aiCoreIdx_; task < taskCount;
                 task += tiling_->usedCoreNum) {
                TaskInfo info{};
                if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                    continue;
                }
                if ASCEND_IS_AIV {
                    SetScoreWorkspace(info, tile);
                }
                for (uint32_t s2 = 0U; s2 < info.loopCount;
                     ++s2, ++globalLoop) {
                    VllmFastQLICommon::RunInfo run{};
                    FillRunInfo(run, info, s2, 0U, info.loopCount,
                                globalLoop);
                    if ASCEND_IS_AIC {
                        matmulService_.ComputeMm1(run);
                    } else {
                        vectorService_.ProcessVec1(run);
                        if (run.isLastS2InnerLoop) {
                            const bool allowFullSlotPrefetch = info.sparse &&
                                !HasLaterActiveTask(
                                    task, taskCount, queryTileCount, tile);
                            vectorService_.ProcessTopK(
                                run, allowFullSlotPrefetch);
                        }
                    }
                }
            }
        } else {
            // Give each active tile a contiguous range of mix groups, then
            // divide its S2 blocks into balanced contiguous intervals. The
            // number of groups follows the tile length, rather than a fixed
            // 16K-token candidate partition.
            const uint32_t extraCores = tiling_->usedCoreNum - activeTasks;
            const uint32_t sourceBlocks =
                (tiling_->sourceCapacity + BLOCK_SIZE - 1U) / BLOCK_SIZE;
            const uint32_t capacitySegments = sourceBlocks / MIN_TOPK_BLOCKS;
            const uint32_t candidateStride = capacitySegments < tiling_->usedCoreNum
                ? capacitySegments : tiling_->usedCoreNum;
            uint64_t assignedBlocks = 0U;
            uint32_t assignedCores = 0U;
            for (uint32_t task = 0U; task < taskCount; ++task) {
                TaskInfo info{};
                if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                    continue;
                }
                const uint32_t segments = SegmentCount(
                    info.loopCount, assignedBlocks, totalBlocks, extraCores);
                if (aiCoreIdx_ >= assignedCores &&
                    aiCoreIdx_ - assignedCores < segments) {
                    const uint32_t segment = aiCoreIdx_ - assignedCores;
                    uint32_t s2Begin = 0U;
                    uint32_t s2End = 0U;
                    SegmentBlockRange(info.loopCount, segments, segment,
                                      s2Begin, s2End);
                    if ASCEND_IS_AIV {
                        SetScoreWorkspace(info, tile);
                    }
                    for (uint32_t s2 = s2Begin; s2 < s2End;
                         ++s2, ++globalLoop) {
                        VllmFastQLICommon::RunInfo run{};
                        FillRunInfo(run, info, s2, s2Begin, s2End,
                                    globalLoop);
                        if ASCEND_IS_AIC {
                            matmulService_.ComputeMm1(run);
                        } else {
                            vectorService_.ProcessVec1(run);
                            if (run.isLastS2InnerLoop) {
                                const uint32_t tokenBegin =
                                    s2Begin * BLOCK_SIZE;
                                const uint32_t tokenEnd =
                                    s2End == info.loopCount
                                        ? info.scanLen : s2End * BLOCK_SIZE;
                                vectorService_.ProcessTopK(
                                    run, info.sparse, tokenBegin,
                                    tokenEnd - tokenBegin, segment,
                                    candidateStride);
                            }
                        }
                    }
                }
                assignedBlocks += info.loopCount;
                assignedCores += segments;
            }
        }

        if ASCEND_IS_AIV {
            vectorService_.FreeEventID();
            if (globalPool && globalHasSplit) {
                // All local candidates reach GM before any LD core reads
                // them. Distribute split route rows across the available
                // AIV cores, weighted by the number of candidate segments.
                SyncAll();
                vectorService_.AllocEventID();
                const uint32_t sourceBlocks =
                    (tiling_->sourceCapacity + BLOCK_SIZE - 1U) / BLOCK_SIZE;
                const uint32_t capacitySegments =
                    sourceBlocks / MIN_TOPK_BLOCKS;
                const uint32_t candidateStride =
                    capacitySegments < tiling_->usedCoreNum
                        ? capacitySegments : tiling_->usedCoreNum;
                uint64_t totalLdLoad = 0U;
                uint64_t taskBegin = 0U;
                for (uint32_t task = 0U; task < taskCount; ++task) {
                    TaskInfo info{};
                    if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                        continue;
                    }
                    const uint64_t taskEnd = taskBegin + info.loopCount;
                    uint32_t firstCore = 0U;
                    uint32_t lastCore = 0U;
                    TaskCoreSpan(coreBoundaries, globalCoreCount,
                                 taskBegin, taskEnd, firstCore, lastCore);
                    if (lastCore != firstCore) {
                        totalLdLoad += static_cast<uint64_t>(
                            lastCore - firstCore + 1U) * info.activeRoutes;
                    }
                    taskBegin = taskEnd;
                }
                uint64_t assignedLdLoad = 0U;
                taskBegin = 0U;
                for (uint32_t task = 0U; task < taskCount; ++task) {
                    TaskInfo info{};
                    if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                        continue;
                    }
                    const uint64_t taskEnd = taskBegin + info.loopCount;
                    uint32_t firstCore = 0U;
                    uint32_t lastCore = 0U;
                    TaskCoreSpan(coreBoundaries, globalCoreCount,
                                 taskBegin, taskEnd, firstCore, lastCore);
                    if (lastCore != firstCore) {
                        const uint32_t segments =
                            lastCore - firstCore + 1U;
                        for (uint32_t row = 0U; row < info.activeRoutes;
                             ++row) {
                            const uint32_t ldCore = static_cast<uint32_t>(
                                assignedLdLoad *
                                (tiling_->usedCoreNum * 2U) / totalLdLoad);
                            if (ldCore == subBlockIdx_) {
                                SetScoreWorkspace(info, tile);
                                VllmFastQLICommon::RunInfo run{};
                                FillRunInfo(run, info,
                                            info.loopCount - 1U, 0U,
                                            info.loopCount, 0U);
                                vectorService_.ReduceLocalTopK(
                                    run, segments, candidateStride,
                                    row, row + 1U);
                            }
                            assignedLdLoad += segments;
                        }
                    }
                    taskBegin = taskEnd;
                }
                vectorService_.FreeEventID();
            } else if (splitScores && !globalPool) {
                // All local score/TopK producers finish before the reducers
                // read candidate scores and packed payloads from GM.
                SyncAll();
                vectorService_.AllocEventID();
                const uint32_t sourceBlocks =
                    (tiling_->sourceCapacity + BLOCK_SIZE - 1U) / BLOCK_SIZE;
                const uint32_t capacitySegments = sourceBlocks / MIN_TOPK_BLOCKS;
                const uint32_t candidateStride = capacitySegments < tiling_->usedCoreNum
                    ? capacitySegments : tiling_->usedCoreNum;
                const uint32_t extraCores = tiling_->usedCoreNum - activeTasks;
                uint64_t assignedBlocks = 0U;
                for (uint32_t task = 0U; task < taskCount; ++task) {
                    TaskInfo info{};
                    if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                        continue;
                    }
                    const uint32_t segments = SegmentCount(
                        info.loopCount, assignedBlocks, totalBlocks, extraCores);
                    if (task % tiling_->usedCoreNum == aiCoreIdx_) {
                        SetScoreWorkspace(info, tile);
                        VllmFastQLICommon::RunInfo run{};
                        FillRunInfo(run, info, info.loopCount - 1U, 0U,
                                    info.loopCount, 0U);
                        const uint32_t firstHalf =
                            (info.activeRoutes + 1U) / 2U;
                        const uint32_t rowBegin = (subBlockIdx_ % 2U)
                            ? firstHalf : 0U;
                        const uint32_t rowEnd = (subBlockIdx_ % 2U)
                            ? info.activeRoutes : firstHalf;
                        vectorService_.ReduceLocalTopK(
                            run, segments, candidateStride,
                            rowBegin, rowEnd);
                    }
                    assignedBlocks += info.loopCount;
                }
                vectorService_.FreeEventID();
            }
        } else {
            matmulService_.FreeEventID();
            CrossCoreWaitFlag<
                VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT);
            CrossCoreWaitFlag<
                VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + 1U);
        }
    }

private:
    // Even the earliest causal route retains at least 2048 valid tokens in
    // an 18-block interval: 18*128 - 127 (partial tail) - 13 (route clip).
    static constexpr uint32_t MIN_TOPK_BLOCKS = static_cast<uint32_t>(
        fused_quant_li_manage_workspace::MIN_TOPK_BLOCKS);
    // Scalar scheduling scratch, not a fixed launch size. Larger launches
    // retain the existing per-task scheduler.
    static constexpr uint32_t MAX_GLOBAL_SCHEDULE_CORES = 64U;
    static constexpr uint32_t CACHED_REQUESTS = 64U;
    static_assert(
        MIN_TOPK_BLOCKS * BLOCK_SIZE - (BLOCK_SIZE - 1U) -
            (MAX_ROUTES - 1U) >= TOPK,
        "each local candidate interval must contain at least TOPK visible tokens");

    __aicore__ inline uint32_t SegmentCount(
        uint32_t loopCount, uint64_t assignedBlocks, uint64_t totalBlocks,
        uint32_t extraCores) const
    {
        const uint32_t requested = 1U + static_cast<uint32_t>(
            (assignedBlocks + loopCount) * extraCores / totalBlocks -
            assignedBlocks * extraCores / totalBlocks);
        const uint32_t maxSegments = loopCount / MIN_TOPK_BLOCKS;
        return requested < maxSegments ? requested : maxSegments;
    }

    __aicore__ inline void SegmentBlockRange(
        uint32_t loopCount, uint32_t segments, uint32_t segment,
        uint32_t &begin, uint32_t &end) const
    {
        const uint32_t blocksPerSegment = loopCount / segments;
        const uint32_t extraBlocks = loopCount % segments;
        begin = segment * blocksPerSegment +
            (segment < extraBlocks ? segment : extraBlocks);
        const uint32_t next = segment + 1U;
        end = next * blocksPerSegment +
            (next < extraBlocks ? next : extraBlocks);
    }

    struct TaskInfo {
        uint32_t batch;
        uint32_t queryStart;
        uint32_t routeCount;
        uint32_t gS1;
        uint32_t scanLen;
        uint32_t loopCount;
        uint32_t activeRoutes;
        uint32_t poolRow;
        uint32_t budget;
        uint32_t offloadedLen;
        bool sparse;
    };

    __aicore__ inline bool LoadTaskInfo(
        uint32_t task, uint32_t queryTileCount, uint32_t tile,
        TaskInfo &info) const
    {
        if (tiling_->compactRouteRows) {
            // Map a compact task ordinal to its request and local LI tile.
            // Invalid requests have no LI task and are published separately.
            uint32_t firstTask = 0U;
            info.batch = tiling_->batchSize;
            for (uint32_t batch = 0U; batch < tiling_->batchSize; ++batch) {
                if (!(batch < CACHED_REQUESTS
                        ? validRequests_[batch] : metadata_.Valid(batch))) {
                    continue;
                }
                uint32_t start = 0U;
                uint32_t end = 0U;
                uint32_t routes = 0U;
                GetQueryRange(batch, start, end, routes);
                const uint32_t tiles = (routes + tile - 1U) / tile;
                if (task < firstTask + tiles) {
                    info.batch = batch;
                    info.gS1 = task - firstTask;
                    break;
                }
                firstTask += tiles;
            }
            if (info.batch == tiling_->batchSize) return false;
        } else {
            info.batch = task / queryTileCount;
            info.gS1 = task % queryTileCount;
        }
        if (!(info.batch < CACHED_REQUESTS
                ? validRequests_[info.batch] : metadata_.Valid(info.batch)))
            return false;
        uint32_t queryEnd = 0U;
        GetQueryRange(info.batch, info.queryStart, queryEnd,
                      info.routeCount);
        const uint32_t routeBase = info.gS1 * tile;
        if (!TaskNeedsQli(info.batch, routeBase, tile,
                          info.routeCount)) {
            return false;
        }
        info.scanLen = static_cast<uint32_t>(
            finalLengthsGm_.GetValue(info.batch));
        info.loopCount = (info.scanLen + BLOCK_SIZE - 1U) / BLOCK_SIZE;
        if (info.loopCount == 0U) {
            return false;
        }
        info.activeRoutes = routeBase + tile > info.routeCount
            ? info.routeCount - routeBase : tile;
        info.sparse = rowModesGm_.GetValue(info.batch) == ROW_MODE_SPARSE;
        info.poolRow = !info.sparse ? 0U : static_cast<uint32_t>(
            reqPoolEntriesGm_.GetValue(info.batch));
        info.budget = !info.sparse ? 0U : static_cast<uint32_t>(
            cacheTokensGm_.GetValue(info.batch));
        info.offloadedLen = !info.sparse ? 0U : static_cast<uint32_t>(
            offloadedLensGm_.GetValue(info.batch));
        return true;
    }

    __aicore__ inline uint64_t RawCoreBoundary(
        uint32_t core, uint32_t coreCount, uint64_t totalBlocks) const
    {
        const uint64_t blocksPerCore = totalBlocks / coreCount;
        const uint32_t extraBlocks = static_cast<uint32_t>(
            totalBlocks % coreCount);
        return static_cast<uint64_t>(core) * blocksPerCore +
            (core < extraBlocks ? core : extraBlocks);
    }

    __aicore__ inline void TaskCoreSpan(
        const uint64_t *boundaries, uint32_t coreCount,
        uint64_t taskBegin, uint64_t taskEnd,
        uint32_t &firstCore, uint32_t &lastCore) const
    {
        firstCore = 0U;
        while (firstCore + 1U < coreCount &&
               boundaries[firstCore + 1U] <= taskBegin) {
            ++firstCore;
        }
        lastCore = firstCore;
        while (lastCore + 1U < coreCount &&
               boundaries[lastCore + 1U] < taskEnd) {
            ++lastCore;
        }
    }

    __aicore__ inline bool BuildGlobalBoundaries(
        uint64_t *boundaries, uint32_t coreCount, uint64_t totalBlocks,
        uint32_t taskCount, uint32_t queryTileCount, uint32_t tile,
        bool &hasSplit) const
    {
        // Follow the reference C8 flattened S2-block pool. The fused TopK
        // also carries a cache-slot payload and requires at least TOPK valid
        // tokens per local candidate, so snap a boundary that would leave
        // fewer than MIN_TOPK_BLOCKS at either end of a query tile.
        boundaries[0] = 0U;
        boundaries[coreCount] = totalBlocks;
        uint32_t nextBoundary = 1U;
        uint64_t taskBegin = 0U;
        for (uint32_t task = 0U; task < taskCount; ++task) {
            TaskInfo info{};
            if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                continue;
            }
            const uint64_t taskEnd = taskBegin + info.loopCount;
            while (nextBoundary < coreCount &&
                   RawCoreBoundary(nextBoundary, coreCount, totalBlocks) <=
                       taskEnd) {
                const uint64_t raw = RawCoreBoundary(
                    nextBoundary, coreCount, totalBlocks);
                uint64_t adjusted = raw;
                if (raw > taskBegin && raw < taskEnd) {
                    if (raw - taskBegin < MIN_TOPK_BLOCKS) {
                        adjusted = taskBegin;
                    } else if (taskEnd - raw < MIN_TOPK_BLOCKS) {
                        adjusted = taskEnd;
                    }
                }
                boundaries[nextBoundary] = adjusted;
                ++nextBoundary;
            }
            taskBegin = taskEnd;
        }
        if (nextBoundary != coreCount || taskBegin != totalBlocks) {
            return false;
        }
        for (uint32_t core = 0U; core < coreCount; ++core) {
            if (boundaries[core + 1U] <= boundaries[core] ||
                boundaries[core + 1U] - boundaries[core] <
                    MIN_TOPK_BLOCKS) {
                return false;
            }
        }
        hasSplit = false;
        taskBegin = 0U;
        for (uint32_t task = 0U; task < taskCount; ++task) {
            TaskInfo info{};
            if (!LoadTaskInfo(task, queryTileCount, tile, info)) {
                continue;
            }
            const uint64_t taskEnd = taskBegin + info.loopCount;
            uint32_t firstCore = 0U;
            uint32_t lastCore = 0U;
            TaskCoreSpan(boundaries, coreCount, taskBegin, taskEnd,
                         firstCore, lastCore);
            for (uint32_t core = firstCore; core <= lastCore; ++core) {
                const uint64_t begin = boundaries[core] > taskBegin
                    ? boundaries[core] : taskBegin;
                const uint64_t end = boundaries[core + 1U] < taskEnd
                    ? boundaries[core + 1U] : taskEnd;
                if (end <= begin || end - begin < MIN_TOPK_BLOCKS) {
                    return false;
                }
            }
            hasSplit |= lastCore != firstCore;
            taskBegin = taskEnd;
        }
        return true;
    }

    __aicore__ inline void SetScoreWorkspace(
        const TaskInfo &info, uint32_t tile)
    {
        if (tiling_->compactRouteRows) {
            const uint64_t row = static_cast<uint64_t>(info.queryStart) +
                static_cast<uint64_t>(info.gS1) * tile;
            vectorService_.InitVecWorkspaceTensor(
                scoreWorkspaceBaseGm_[row * tiling_->scoreRowStride]);
        } else {
            const uint64_t requestScoreBase =
                static_cast<uint64_t>(info.batch) *
                (tiling_->scoreWorkspaceStride / sizeof(uint16_t));
            const uint64_t routeScoreBase =
                static_cast<uint64_t>(info.gS1) * tile *
                tiling_->scoreRowStride;
            vectorService_.InitVecWorkspaceTensor(
                scoreWorkspaceBaseGm_[requestScoreBase + routeScoreBase]);
        }
    }

    __aicore__ inline void FillRunInfo(
        VllmFastQLICommon::RunInfo &run, const TaskInfo &info,
        uint32_t s2, uint32_t s2Begin, uint32_t s2End,
        uint32_t loop) const
    {
        run.loop = loop;
        run.bN2Idx = info.batch;
        run.bIdx = info.batch;
        run.n2Idx = 0U;
        run.gS1Idx = info.gS1;
        run.s2Idx = s2;
        run.s2Start = s2Begin;
        run.validS2Len = info.scanLen;
        run.kScaleLoop = (s2 - s2Begin) / 16U + 1U;
        run.cacheRowIdx = info.poolRow;
        run.cacheTokenCount = info.budget;
        run.offloadedLen = info.offloadedLen;
        run.actS1Size = info.routeCount;
        run.actS2Size = info.scanLen;
        run.actS2SizeOrig = info.scanLen;
        run.actMBaseSize = info.activeRoutes * tiling_->indexHeads;
        const uint32_t remaining = info.scanLen - s2 * BLOCK_SIZE;
        run.actualSingleProcessSInnerSize = remaining < BLOCK_SIZE
            ? remaining : BLOCK_SIZE;
        run.actualSingleProcessSInnerSizeAlign =
            VllmFastQLICommon::Align(
                run.actualSingleProcessSInnerSize, 32U);
        run.tensorQueryOffset =
            static_cast<uint64_t>(info.queryStart) *
                tiling_->indexHeads * HEAD_DIM +
            static_cast<uint64_t>(info.gS1) *
                constInfo_.mBaseSize * HEAD_DIM;
        run.tensorKeyOffset =
            static_cast<uint64_t>(s2) * BLOCK_SIZE * HEAD_DIM;
        run.tensorKeyScaleOffset =
            static_cast<uint64_t>(s2) * BLOCK_SIZE;
        run.tensorWeightsOffset =
            static_cast<uint64_t>(info.queryStart) *
            tiling_->weightStride;
        run.tensorQueryScaleOffset =
            static_cast<uint64_t>(info.queryStart) *
            tiling_->indexHeads;
        run.indiceOutOffset =
            static_cast<uint64_t>(info.queryStart) * TOPK;
        run.workspaceRow = tiling_->compactRouteRows
            ? info.queryStart : info.batch * ROUTES;
        run.causalClip = true;
        run.denseDirectPublish = !info.sparse;
        run.isFirstS2InnerLoop = s2 == s2Begin;
        run.isLastS2InnerLoop = s2 + 1U == s2End;
        run.isAllLoopEnd = false;
        run.isValid = true;
    }

    __aicore__ inline void GetQueryRange(
        uint32_t batch, uint32_t &queryStart, uint32_t &queryEnd,
        uint32_t &routeCount) const
    {
        const int32_t end = queryEndsGm_.GetValue(batch);
        const int32_t start = batch == 0U
            ? 0 : queryEndsGm_.GetValue(batch - 1U);
        queryStart = static_cast<uint32_t>(start);
        queryEnd = static_cast<uint32_t>(end);
        routeCount = queryEnd - queryStart;
    }

    __aicore__ inline bool TaskNeedsQli(
        uint32_t batch, uint32_t routeBase, uint32_t tile,
        uint32_t routeCount) const
    {
        if (routeBase >= routeCount) {
            return false;
        }
        const int32_t mode = rowModesGm_.GetValue(batch);
        if (mode != ROW_MODE_DENSE && mode != ROW_MODE_FIRST_FILL &&
            mode != ROW_MODE_SPARSE) {
            return false;
        }
        // The vector TopK already handles visible < TOPK with sequential
        // source IDs and -1 padding. Keep short dense tiles in this LI path.
        return true;
    }

    __aicore__ inline bool HasLaterActiveTask(
        uint32_t task, uint32_t taskCount, uint32_t queryTileCount,
        uint32_t tile) const
    {
        for (uint32_t next = task + tiling_->usedCoreNum;
             next < taskCount; next += tiling_->usedCoreNum) {
            if (tiling_->compactRouteRows) {
                TaskInfo info{};
                if (LoadTaskInfo(next, queryTileCount, tile, info)) {
                    return true;
                }
                continue;
            }
            const uint32_t nextBatch = next / queryTileCount;
            if (!(nextBatch < CACHED_REQUESTS
                    ? validRequests_[nextBatch] : metadata_.Valid(nextBatch)))
                continue;
            uint32_t queryStart = 0U;
            uint32_t queryEnd = 0U;
            uint32_t routeCount = 0U;
            GetQueryRange(
                nextBatch, queryStart, queryEnd, routeCount);
            const uint32_t routeBase =
                (next % queryTileCount) * tile;
            if (TaskNeedsQli(
                    nextBatch, routeBase, tile, routeCount)) {
                return true;
            }
        }
        return false;
    }

    TPipe *pipe_;
    const FusedQuantLightningIndexerManageTilingData *tiling_;
    uint32_t subBlockIdx_ = 0U;
    uint32_t aiCoreIdx_ = 0U;
    bool validRequests_[CACHED_REQUESTS] = {};
    VllmFastQLICommon::ConstInfo constInfo_{};
    VllmFastQLIKernel::QLIMatmul<C8MtpQliType> matmulService_;
    VllmFastQLIKernel::QLIVector<C8MtpQliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> queryEndsGm_;
    GlobalTensor<int32_t> finalLengthsGm_;
    GlobalTensor<int32_t> offloadedLensGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> rowModesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsGm_;
    lim_request_contract::Metadata metadata_;
    GlobalTensor<int32_t> routePairRowsGm_;
    GlobalTensor<int32_t> topkSlotsGm_;
    GlobalTensor<int32_t> topkSourceIdsGm_;
    GlobalTensor<int32_t> routeMissCountsGm_;
    GlobalTensor<uint16_t> routeThresholdsGm_;
    GlobalTensor<uint16_t> scoreWorkspaceBaseGm_;
    GlobalTensor<uint16_t> candidateScoresGm_;
    GlobalTensor<uint32_t> candidatePayloadsGm_;
    GlobalTensor<uint16_t> candidateTagsGm_;
};

} // namespace fused_quant_li_manage

#endif
