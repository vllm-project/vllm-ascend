#ifndef MC2_RECORD_EXPERT_LOAD_H
#define MC2_RECORD_EXPERT_LOAD_H

#include "kernel_operator.h"

// BF16 still records on the expert owner until its routing path accepts an
// active-row mask.
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

// The routing stage has already counted this rank's outgoing tokens for every
// physical expert. Record them before the MC2 communication overwrites the
// shared count matrix. Only vector core 0 writes this rank's load buffer.
__aicore__ inline void RecordSourceExpertLoad(GM_ADDR sourceTokenCounts,
                                             GM_ADDR expertLoad,
                                             GM_ADDR recordEnabled,
                                             uint32_t expertCount)
{
    if (expertLoad == nullptr || recordEnabled == nullptr) {
        return;
    }
    auto enabled = reinterpret_cast<__gm__ bool*>(recordEnabled);
    if (!*enabled) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    auto counts = reinterpret_cast<__gm__ int32_t*>(sourceTokenCounts);
    auto load = reinterpret_cast<__gm__ int32_t*>(expertLoad);
    for (uint32_t i = 0; i < expertCount; ++i) {
        load[i] += counts[i];
    }
}

#endif
