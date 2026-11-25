/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef INDEXER_REGION_SCORE_SHARD_PLAN_H
#define INDEXER_REGION_SCORE_SHARD_PLAN_H
#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
namespace IndexerRegionScoreHost {
constexpr uint32_t kProxyDim = 256;
constexpr uint32_t kCubeMRows = 128;
constexpr uint32_t kKeyTile = 512;
inline uint32_t SelectPagedQMicroTile(uint32_t heads)
{
    return std::max(1U, kCubeMRows / heads);
}
struct QkShardPlan {
    uint32_t qShards;
    uint32_t kShards;
};
inline QkShardPlan SelectQkShardPlan(uint32_t qRanges, uint32_t kRanges,
                              uint32_t pairCapacity, uint32_t groups,
                              uint32_t summariesPerPage, uint32_t qMicroTile,
                              uint32_t heads, bool pagedGather, uint32_t keyTile = kKeyTile)
{
    QkShardPlan trafficPlan{1U, 1U};
    QkShardPlan gatherPlan{1U, 1U};
    const auto maxCost = std::numeric_limits<uint64_t>::max();
    auto trafficCost = std::make_tuple(maxCost, maxCost, UINT32_MAX);
    auto gatherCost = std::make_tuple(maxCost, maxCost, maxCost, UINT32_MAX);
    uint64_t trafficPlanGatherCommands = maxCost;
    const uint32_t maxQShards = std::min(qRanges, pairCapacity);
    for (uint32_t qShards = 1U; qShards <= maxQShards; ++qShards) {
        const uint32_t maxKShards = std::min(kRanges, pairCapacity / qShards);
        for (uint32_t kShards = 1U; kShards <= maxKShards; ++kShards) {
            const uint64_t maxTiles =
                static_cast<uint64_t>((qRanges + qShards - 1U) / qShards) *
                ((kRanges + kShards - 1U) / kShards);
            const uint64_t qTraffic = static_cast<uint64_t>(kShards) * qRanges *
                groups * qMicroTile * heads * kProxyDim;
            const uint64_t kTraffic = static_cast<uint64_t>(qShards) * kRanges *
                keyTile * kProxyDim;
            const uint64_t traffic = qTraffic + kTraffic;
            const uint64_t gatherCommands = static_cast<uint64_t>(qShards) * kRanges *
                ((keyTile + summariesPerPage - 1U) / summariesPerPage);
            const uint32_t cores = qShards * kShards;
            const auto candidateTraffic = std::make_tuple(maxTiles, traffic, pairCapacity - cores);
            if (candidateTraffic < trafficCost) {
                trafficPlan = {qShards, kShards};
                trafficCost = candidateTraffic;
                trafficPlanGatherCommands = gatherCommands;
            }
            const auto candidateGather = std::make_tuple(maxTiles, gatherCommands, traffic, pairCapacity - cores);
            if (candidateGather < gatherCost) {
                gatherPlan = {qShards, kShards};
                gatherCost = candidateGather;
            }
        }
    }
    if (!pagedGather) {
        return trafficPlan;
    }
    const uint32_t minimumUsefulCores = (pairCapacity * 3U + 3U) / 4U;
    const bool controlsGatherBottleneck = std::get<1>(gatherCost) < trafficPlanGatherCommands &&
        gatherPlan.qShards >= 2U && gatherPlan.qShards * gatherPlan.kShards >= minimumUsefulCores &&
        std::get<2>(gatherCost) <= std::get<1>(trafficCost) * 2U;
    return controlsGatherBottleneck ? gatherPlan : trafficPlan;
}
struct PagedShardPlan {
    uint32_t qMicroTile;
    QkShardPlan shards;
};
struct RequestQueryWork {
    uint64_t microTiles = 0U;
    uint64_t requestSegments = 0U;
};
inline uint64_t ShardCeilDiv(uint64_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0U);
}
inline QkShardPlan FillPagedQueryShards(QkShardPlan original, uint32_t qRanges,
                                       uint32_t kRanges, uint32_t pairCapacity)
{
    if (qRanges == 0U || kRanges == 0U || pairCapacity == 0U ||
        original.qShards == 0U || original.kShards == 0U ||
        original.qShards > qRanges || original.kShards > kRanges ||
        original.qShards > pairCapacity / original.kShards) {
        return original;
    }
    const uint32_t availableQShards = std::min(qRanges, pairCapacity / original.kShards);
    if (availableQShards > original.qShards &&
        ShardCeilDiv(qRanges, availableQShards) == ShardCeilDiv(qRanges, original.qShards)) {
        return {availableQShards, original.kShards};
    }
    return original;
}
inline void AddUniformRequestWork(uint64_t begin, uint64_t end, uint32_t requestLength,
                                  uint32_t qMicroTile, RequestQueryWork &work)
{
    if (begin >= end || requestLength == 0U) {
        return;
    }
    const uint64_t first = begin / requestLength;
    const uint64_t last = (end - 1U) / requestLength;
    work.requestSegments += last - first + 1U;
    if (first == last) {
        work.microTiles += ShardCeilDiv(end - begin, qMicroTile);
        return;
    }
    work.microTiles += ShardCeilDiv(requestLength - begin % requestLength, qMicroTile);
    work.microTiles += ShardCeilDiv((end - 1U) % requestLength + 1U, qMicroTile);
    work.microTiles += (last - first - 1U) * ShardCeilDiv(requestLength, qMicroTile);
}
inline RequestQueryWork EstimateRequestQueryWork(uint32_t totalTokens, uint32_t requestCount,
                                                 uint32_t qMicroTile, uint64_t begin, uint64_t end)
{
    RequestQueryWork work;
    if (totalTokens == 0U || requestCount == 0U || qMicroTile == 0U) {
        return work;
    }
    const uint32_t nonempty = std::min(totalTokens, requestCount);
    const uint32_t shortLength = totalTokens / nonempty;
    const uint64_t longEnd = static_cast<uint64_t>(totalTokens % nonempty) * (shortLength + 1U);
    AddUniformRequestWork(begin, std::min(end, longEnd), shortLength + 1U, qMicroTile, work);
    if (end > longEnd) {
        AddUniformRequestWork(std::max(begin, longEnd) - longEnd, end - longEnd,
                              shortLength, qMicroTile, work);
    }
    return work;
}
inline PagedShardPlan SelectPagedRequestShardPlan(uint32_t totalTokens, uint32_t requestCount,
                                                  uint32_t kRanges, uint32_t pairCapacity,
                                                  uint32_t groups, uint32_t summariesPerPage,
                                                  uint32_t heads, uint32_t keyTile = kKeyTile)
{
    const uint32_t baseQTile = SelectPagedQMicroTile(heads);
    const uint32_t baseQRanges = static_cast<uint32_t>(ShardCeilDiv(totalTokens, baseQTile));
    const auto basePlan = SelectQkShardPlan(baseQRanges, std::max(1U, kRanges), pairCapacity,
                                           groups, summariesPerPage, baseQTile, heads, true, keyTile);
    const uint32_t nonempty = std::min(totalTokens, requestCount);
    if (nonempty <= 1U || kRanges == 0U || totalTokens / nonempty >= baseQTile) {
        return {baseQTile, basePlan};
    }
    const uint32_t qTile = std::max(1U, totalTokens / nonempty);
    const uint32_t qRanges = static_cast<uint32_t>(ShardCeilDiv(totalTokens, qTile));
    QkShardPlan best{1U, 1U};
    const auto maxCost = std::numeric_limits<uint64_t>::max();
    auto bestCost = std::make_tuple(maxCost, maxCost, UINT32_MAX);
    for (uint32_t qShards = 1U; qShards <= std::min(qRanges, pairCapacity); ++qShards) {
        uint64_t maxMicroTiles = 0U;
        uint64_t maxRequestSegments = 0U;
        uint64_t totalRequestSegments = 0U;
        for (uint32_t shard = 0U; shard < qShards; ++shard) {
            const uint64_t firstRange = static_cast<uint64_t>(shard) * (qRanges / qShards) +
                std::min(shard, qRanges % qShards);
            const uint64_t rangeCount = qRanges / qShards + (shard < qRanges % qShards);
            const auto work = EstimateRequestQueryWork(totalTokens, requestCount, qTile,
                firstRange * qTile, std::min<uint64_t>(totalTokens, (firstRange + rangeCount) * qTile));
            maxMicroTiles = std::max(maxMicroTiles, work.microTiles);
            maxRequestSegments = std::max(maxRequestSegments, work.requestSegments);
            totalRequestSegments += work.requestSegments;
        }
        for (uint32_t kShards = 1U; kShards <= std::min(kRanges, pairCapacity / qShards); ++kShards) {
            const uint64_t cost = maxMicroTiles * groups * ShardCeilDiv(kRanges, kShards) +
                maxRequestSegments;
            const uint64_t traffic = totalRequestSegments * kRanges * keyTile * kProxyDim +
                static_cast<uint64_t>(kShards) * totalTokens * groups * heads * kProxyDim;
            const uint32_t cores = qShards * kShards;
            const auto candidate = std::make_tuple(cost, traffic, pairCapacity - cores);
            if (candidate < bestCost) {
                best = {qShards, kShards};
                bestCost = candidate;
            }
        }
    }
    return {qTile, best};
}
} // namespace IndexerRegionScoreHost
#endif
