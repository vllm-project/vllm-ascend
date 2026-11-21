#ifndef LIM_SIMT_WIDE_POSTPROCESS_ALGORITHMS_H
#define LIM_SIMT_WIDE_POSTPROCESS_ALGORITHMS_H

#include "lim_simt_postprocess_algorithms.h"

namespace lim_simt {

__simt_callee__ inline bool ValidWidePair(
    __gm__ const int32_t *pool, uint64_t rowBase,
    int32_t source, int32_t destination, int32_t victim,
    uint32_t candidate, uint32_t budget)
{
    // Preserve the wide path's checked-update contract. Short-circuit before
    // any GM access when an address or slot is outside the request's range.
    return ValidPair(source, destination, victim, candidate, budget) &&
        source != victim &&
        pool[rowBase + static_cast<uint32_t>(victim)] == destination;
}

__simt_callee__ inline uint32_t WideRepairOffset(
    uint32_t index, uint32_t end0, uint32_t end1, uint32_t end2)
{
    const uint32_t route = static_cast<uint32_t>(index >= end0) +
        static_cast<uint32_t>(index >= end1) +
        static_cast<uint32_t>(index >= end2);
    const uint32_t begin = route == 0U ? 0U :
        (route == 1U ? end0 : (route == 2U ? end1 : end2));
    return route * REPAIR_TOPK + index - begin;
}

__simt_callee__ inline int32_t LookupWideSlotGm(
    int32_t source, __gm__ const int32_t *unionSources,
    __gm__ const int32_t *unionDestinations, uint32_t unionCount)
{
    // Large unions exceed sourceBuf_. These immutable COPY lists are never
    // read by SIMT before their producer DMA completes; no updated pool or
    // overwritten victim-scratch row is consulted by the repair stage.
    uint32_t left = 0U;
    uint32_t right = unionCount;
    while (left < right) {
        const uint32_t middle = left + (right - left) / 2U;
        if (unionSources[middle] < source) left = middle + 1U;
        else right = middle;
    }
    return left < unionCount && unionSources[left] == source
        ? unionDestinations[left] : -1;
}

} // namespace lim_simt
#endif
