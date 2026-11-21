#ifndef LIM_SIMT_POSTPROCESS_H
#define LIM_SIMT_POSTPROCESS_H

#include "lim_simt_config.h"
#if LIM_SIMT_POSTPROCESS
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "lim_simt_postprocess_algorithms.h"

namespace lim_simt {

#if LIM_SIMT_VICTIM_UNPACK
__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void UnpackVictims(
    __ubuf__ const uint32_t *payloads,
    __ubuf__ int32_t *destinations,
    __ubuf__ int32_t *victimSources, uint32_t count,
    uint32_t sourcePage)
{
    // Disjoint output elements, no reduction/atomics or GM reads. The
    // caller supplies the original min(compactCount, remaining) prefix.
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        UnpackVictim(payloads[i], destinations, victimSources, i, sourcePage);
    }
}
#endif

__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void ApplyUpdates(
    __gm__ int32_t *pool, uint64_t rowBase,
    __ubuf__ const int32_t *sources,
    __ubuf__ const int32_t *destinations,
    __ubuf__ const int32_t *victims,
    __ubuf__ uint32_t *validPrefix,
    uint32_t count, uint32_t candidate, uint32_t budget)
{
    // The caller initializes validPrefix=count for EVERY invocation.
    // Keep the standardized request-level failure atomic: an invalid pair
    // suppresses the entire update, rather than committing a valid prefix.
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (!ValidPair(sources[i], destinations[i], victims[i], candidate, budget)) {
            asc_atomic_min(validPrefix, i);
        }
    }
    asc_syncthreads();
    if (threadIdx.x == 0U && validPrefix[0] != count)
        validPrefix[0] = 0U;
    asc_syncthreads();
    const uint32_t updated = validPrefix[0];
    for (uint32_t i = threadIdx.x; i < updated; i += blockDim.x) {
        UpdatePair(pool, rowBase, sources[i], destinations[i], victims[i]);
    }
    // Finish global stores before the main-scalar publishes copy metadata.
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void RepairMissSlots(
    __ubuf__ const int32_t *routePairs,
    __ubuf__ const int32_t *unionSources,
    __ubuf__ const int32_t *unionDestinations,
    __ubuf__ int32_t *output, uint32_t length, uint32_t unionCount)
{
    for (uint32_t i = threadIdx.x; i < length; i += blockDim.x) {
        // Binary search the immutable UB union, NOT the just-updated GM
        // reverse map. All routes sharing a miss consequently get one slot.
        output[i] = LookupSlot(
            routePairs[i * 2U + 1U], unionSources, unionDestinations, unionCount);
    }
}

#if LIM_SIMT_BATCHED_REPAIR
__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void RepairMissSlotsBatched(
    __ubuf__ const int32_t *routePairs,
    __ubuf__ const int32_t *unionSources,
    __ubuf__ const int32_t *unionDestinations,
    __ubuf__ int32_t *output, uint32_t length0, uint32_t length1,
    uint32_t length2, uint32_t length3, uint32_t unionCount)
{
    const uint32_t end0 = length0;
    const uint32_t end1 = end0 + length1;
    const uint32_t end2 = end1 + length2;
    const uint32_t total = end2 + length3;
    for (uint32_t index = threadIdx.x; index < total; index += blockDim.x) {
        RepairBatchedMiss(routePairs, unionSources, unionDestinations, output,
                         index, end0, end1, end2, unionCount);
    }
}
#endif

} // namespace lim_simt
#endif
#endif
