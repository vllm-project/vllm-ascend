#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_ARCH35_WORKSPACE_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_ARCH35_WORKSPACE_H

#include <cstdint>
#include "lim_request_limits.h"

namespace fused_quant_li_manage_workspace {
constexpr uint64_t ROUTES = 4U;
constexpr uint64_t TOPK = lim_request_limits::TOPK;
constexpr uint64_t UNION_CAPACITY = ROUTES * TOPK;
constexpr uint64_t THRESHOLD_STRIDE = 16U;
// Different AIV cores publish route counts concurrently. Give each scalar
// its own 32-byte GM transfer unit to avoid read-modify-write interference.
constexpr uint64_t ROUTE_COUNT_STRIDE = 8U; // int32 elements
constexpr uint64_t TAIL_CAPACITY = lim_request_limits::MAX_TAIL_TOKENS;
constexpr uint64_t ATTENTION_CAPACITY = TOPK;
constexpr uint64_t S2_BLOCK = lim_request_limits::BLOCK_SIZE;
constexpr uint64_t MIN_TOPK_BLOCKS = 18U;
constexpr uint64_t S2_SPLIT_MIN_TOKENS = 64U * 1024U;

constexpr uint64_t Align32(uint64_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}

constexpr uint64_t RoutePairOffset(
    uint64_t scoreRowStride, uint64_t routeStorageRows)
{
    return scoreRowStride * routeStorageRows * sizeof(uint16_t);
}

constexpr uint64_t RouteThresholdOffset(
    uint64_t scoreRowStride, uint64_t routeStorageRows)
{
    return RoutePairOffset(scoreRowStride, routeStorageRows) +
           routeStorageRows * TOPK * 2U * sizeof(int32_t);
}

constexpr uint64_t RouteCountOffset(
    uint64_t scoreRowStride, uint64_t routeStorageRows)
{
    return RouteThresholdOffset(scoreRowStride, routeStorageRows) +
           routeStorageRows * THRESHOLD_STRIDE * sizeof(uint16_t);
}

constexpr uint64_t MaxSegmentsForCapacity(uint64_t sourceCapacity)
{
    return (sourceCapacity + S2_BLOCK - 1U) / S2_BLOCK /
           MIN_TOPK_BLOCKS;
}

constexpr uint64_t CandidateSegmentCapacity(
    uint64_t sourceCapacity, uint64_t usedCores)
{
    return sourceCapacity < S2_SPLIT_MIN_TOKENS ? 1U :
        (MaxSegmentsForCapacity(sourceCapacity) < usedCores
            ? MaxSegmentsForCapacity(sourceCapacity) : usedCores);
}

constexpr uint64_t CandidateScoreOffset(
    uint64_t scoreRowStride, uint64_t routeStorageRows)
{
    return Align32(RouteCountOffset(scoreRowStride, routeStorageRows) +
                   routeStorageRows * ROUTE_COUNT_STRIDE * sizeof(int32_t));
}

constexpr uint64_t CandidatePayloadOffset(
    uint64_t scoreRowStride, uint64_t routeStorageRows,
    uint64_t candidateSegments)
{
    return CandidateScoreOffset(scoreRowStride, routeStorageRows) +
           routeStorageRows * candidateSegments * TOPK *
               sizeof(uint16_t);
}

constexpr uint64_t CandidateTagOffset(
    uint64_t scoreRowStride, uint64_t routeStorageRows,
    uint64_t candidateSegments)
{
    return CandidatePayloadOffset(
        scoreRowStride, routeStorageRows, candidateSegments) +
           routeStorageRows * candidateSegments * TOPK * sizeof(uint32_t);
}

constexpr uint64_t TotalBytes(
    uint64_t scoreRowStride, uint64_t routeStorageRows,
    uint64_t candidateSegments,
    bool longSource = false)
{
    return Align32(CandidateTagOffset(
        scoreRowStride, routeStorageRows, candidateSegments) +
        (longSource ? routeStorageRows * candidateSegments * TOPK *
            sizeof(uint16_t) : 0U));
}
} // namespace fused_quant_li_manage_workspace

#endif
