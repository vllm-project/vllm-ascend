#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_SORT_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_SORT_H

#include "kernel_operator.h"

namespace fused_quant_li_manage_sort {
using namespace AscendC;

constexpr uint32_t CAPACITY = 4096U;

// Match the buckets accepted by A3 LIServiceVec::SortAll. Each merge level
// contains complete, equally sized runs; only the last level uses 2/3/4 runs.
// The largest bucket merges two 2048-element runs at its final level.
__aicore__ inline uint32_t SortLength(uint32_t count)
{
    if (count <= 128U) return 128U;
    if (count <= 256U) return 256U;
    if (count <= 384U) return 384U;
    if (count <= 512U) return 512U;
    if (count <= 1024U) return 1024U;
    if (count <= 2048U) return 2048U;
    return CAPACITY;
}

// Adapted from ops_lim_standardization's in-place LIServiceVec::SortAll.
// input holds [keys, payloads] on entry and interleaved pairs on return.
// Both disjoint UB tensors reserve 2 * count float words. count is a bucket
// from SortLength, never an arbitrary number of 32-element runs.
__aicore__ inline void SortAll(
    LocalTensor<float> input, LocalTensor<float> scratch, uint32_t count)
{
    uint32_t groups = count / 32U;
    uint32_t runLength = 32U;
    Sort32<float>(scratch, input,
                  input[count].ReinterpretCast<uint32_t>(), groups);
    PipeBarrier<PIPE_V>();
    bool resultInScratch = true;
    while (groups > 1U) {
        LocalTensor<float> source = resultInScratch ? scratch : input;
        LocalTensor<float> destination = resultInScratch ? input : scratch;
        MrgSort4Info params;
        for (uint32_t lane = 0U; lane < 4U; ++lane)
            params.elementLengths[lane] = runLength;
        params.ifExhaustedSuspension = false;
        params.validBit = groups < 4U ? (1U << groups) - 1U : 0b1111U;
        params.repeatTimes = groups <= 4U ? 1U : groups / 4U;
        MrgSortSrcList<float> queues;
        queues.src1 = source;
        queues.src2 = source[runLength * 2U];
        queues.src3 = groups >= 3U ? source[runLength * 4U] : source;
        queues.src4 = groups >= 4U ? source[runLength * 6U] : source;
        MrgSort<float>(destination, queues, params);
        PipeBarrier<PIPE_V>();
        resultInScratch = !resultInScratch;
        groups = groups <= 4U ? 1U : groups / 4U;
        runLength *= 4U;
    }
    if (resultInScratch) {
        DataCopy(input, scratch, count * 2U);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace fused_quant_li_manage_sort

#endif
