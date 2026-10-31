#ifndef LIM_SIMT_POSTPROCESS_ALGORITHMS_H
#define LIM_SIMT_POSTPROCESS_ALGORITHMS_H

#include <cstdint>
#include "lim_pool_contract.h"
#include "lim_payload_layout.h"
#include "lim_request_limits.h"

namespace lim_simt {

// Kept independent of the launcher so the exact indexing arithmetic can
// also be exercised by the small host-side developer test.
__simt_callee__ inline void UnpackVictim(
    uint32_t payload, __ubuf__ int32_t *destinations,
    __ubuf__ int32_t *victimSources, uint32_t index,
    uint32_t sourcePage)
{
    // The SIMD scan encodes slot[31:17] | token[16:0]. Keep uint32
    // arithmetic: legal slots >=16384 set the payload's sign bit.
    destinations[index] = static_cast<int32_t>(
        (payload >> lim_payload_layout::SOURCE_BITS) & lim_payload_layout::SLOT_MASK);
    victimSources[index] = static_cast<int32_t>(
        sourcePage | (payload & lim_payload_layout::SOURCE_MASK));
}

__simt_callee__ inline bool ValidPair(
    int32_t source, int32_t destination, int32_t victim,
    uint32_t candidate, uint32_t budget)
{
    return static_cast<uint32_t>(source) < candidate &&
        static_cast<uint32_t>(victim) < candidate &&
        static_cast<uint32_t>(destination) < budget;
}

__simt_callee__ inline void UpdatePair(
    __gm__ int32_t *pool, uint64_t rowBase,
    int32_t source, int32_t destination, int32_t victim)
{
    // Miss sources, victim sources and destinations are each unique and
    // miss/victim source sets are disjoint, as established by the union /
    // victim stages. No GM atomics or full-pool scan is needed here.
    pool[rowBase + static_cast<uint32_t>(victim)] =
        lim_pool_contract::INVALID_POOL_SLOT;
    pool[rowBase + static_cast<uint32_t>(source)] = destination;
}

__simt_callee__ inline int32_t LookupSlot(
    int32_t source, __ubuf__ const int32_t *unionSources,
    __ubuf__ const int32_t *unionDestinations, uint32_t unionCount)
{
    uint32_t left = 0U;
    uint32_t right = unionCount;
    while (left < right) {
        const uint32_t middle = left + (right - left) / 2U;
        if (unionSources[middle] < source) {
            left = middle + 1U;
        } else {
            right = middle;
        }
    }
    return left < unionCount && unionSources[left] == source
        ? unionDestinations[left] : -1;
}

constexpr uint32_t REPAIR_TOPK = lim_request_limits::TOPK;

__simt_callee__ inline void RepairBatchedMiss(
    __ubuf__ const int32_t *routePairs,
    __ubuf__ const int32_t *unionSources,
    __ubuf__ const int32_t *unionDestinations,
    __ubuf__ int32_t *output, uint32_t index,
    uint32_t end0, uint32_t end1, uint32_t end2, uint32_t unionCount)
{
    // index enumerates actual misses only. Repeated cumulative ends skip
    // zero-length / absent lanes; storage retains four fixed TopK rows.
    const uint32_t route = static_cast<uint32_t>(index >= end0) +
        static_cast<uint32_t>(index >= end1) +
        static_cast<uint32_t>(index >= end2);
    const uint32_t begin = route == 0U ? 0U :
        (route == 1U ? end0 : (route == 2U ? end1 : end2));
    const uint32_t offset = route * REPAIR_TOPK + index - begin;
    output[offset] = LookupSlot(
        routePairs[offset * 2U + 1U], unionSources, unionDestinations, unionCount);
}

} // namespace lim_simt
#endif
