#ifndef MC2_RECORD_EXPERT_LOAD_H
#define MC2_RECORD_EXPERT_LOAD_H

#include "kernel_operator.h"

// expertTokenNums contains per-expert counts after the rank-wise cumulative sum.
// Only vector core 0 updates this rank's disjoint physical-expert slice.
__aicore__ inline void RecordExpertLoad(GM_ADDR expertTokenNums, GM_ADDR expertLoad,
                                       GM_ADDR recordEnabled, uint32_t localExpertStart,
                                       uint32_t expertPerRank)
{
    if (expertLoad == nullptr || recordEnabled == nullptr) {
        return;
    }
    auto enabled = reinterpret_cast<__gm__ bool*>(recordEnabled);
    if (!*enabled) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    auto counts = reinterpret_cast<__gm__ int32_t*>(expertTokenNums);
    auto load = reinterpret_cast<__gm__ int32_t*>(expertLoad);
    for (uint32_t i = 0; i < expertPerRank; ++i) {
        load[localExpertStart + i] += counts[i];
    }
}

#endif
