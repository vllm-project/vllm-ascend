#ifndef LIM_SIMT_WIDE_POSTPROCESS_H
#define LIM_SIMT_WIDE_POSTPROCESS_H

#include "lim_simt_postprocess.h"
#if LIM_SIMT_POSTPROCESS
#include "lim_simt_wide_postprocess_algorithms.h"

namespace lim_simt {

__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void ApplyWideUpdatesChecked(
    __gm__ int32_t *pool, uint64_t rowBase,
    __ubuf__ const int32_t *sources,
    __ubuf__ const int32_t *destinations,
    __ubuf__ const int32_t *victims,
    __ubuf__ uint32_t *validPrefix,
    uint32_t count, uint32_t candidate, uint32_t budget, bool commit)
{
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (!ValidWidePair(pool, rowBase, sources[i], destinations[i],
                           victims[i], candidate, budget))
            asc_atomic_min(validPrefix, i);
    }
    asc_syncthreads();
    if (threadIdx.x == 0U && validPrefix[0] != count) validPrefix[0] = 0U;
    asc_syncthreads();
    // For <=8192 rows this VF validates the whole request before writing.
    // Larger requests first validate every chunk with commit=false.
    if (commit) {
        for (uint32_t i = threadIdx.x; i < validPrefix[0]; i += blockDim.x)
            UpdatePair(pool, rowBase, sources[i], destinations[i], victims[i]);
        asc_threadfence();
    }
}

__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void CommitWideUpdates(
    __gm__ int32_t *pool, uint64_t rowBase,
    __ubuf__ const int32_t *sources,
    __ubuf__ const int32_t *destinations,
    __ubuf__ const int32_t *victims, uint32_t count)
{
    // Called only after all chunks have passed request-level validation.
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x)
        UpdatePair(pool, rowBase, sources[i], destinations[i], victims[i]);
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void RepairWideMissSlots(
    __ubuf__ const int32_t *routeSources,
    __ubuf__ const int32_t *unionSources,
    __ubuf__ const int32_t *unionDestinations,
    __ubuf__ int32_t *output, uint32_t length0, uint32_t length1,
    uint32_t length2, uint32_t length3, uint32_t unionCount)
{
    const uint32_t end0 = length0;
    const uint32_t end1 = end0 + length1;
    const uint32_t end2 = end1 + length2;
    const uint32_t total = end2 + length3;
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        const uint32_t offset = WideRepairOffset(i, end0, end1, end2);
        output[offset] = LookupSlot(
            routeSources[offset], unionSources, unionDestinations, unionCount);
    }
}

__simt_vf__ __launch_bounds__(LIM_SIMT_THREADS) inline void RepairWideMissSlotsGm(
    __ubuf__ const int32_t *routeSources,
    __gm__ const int32_t *unionSources,
    __gm__ const int32_t *unionDestinations,
    __ubuf__ int32_t *output, uint32_t length0, uint32_t length1,
    uint32_t length2, uint32_t length3, uint32_t unionCount)
{
    const uint32_t end0 = length0;
    const uint32_t end1 = end0 + length1;
    const uint32_t end2 = end1 + length2;
    const uint32_t total = end2 + length3;
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        const uint32_t offset = WideRepairOffset(i, end0, end1, end2);
        output[offset] = LookupWideSlotGm(
            routeSources[offset], unionSources, unionDestinations, unionCount);
    }
}

} // namespace lim_simt
#endif
#endif
