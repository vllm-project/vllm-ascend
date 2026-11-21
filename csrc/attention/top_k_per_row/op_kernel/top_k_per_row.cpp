/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "top_k_per_row.h"

using namespace AscendC;

#define RUN_TOP_K_PER_ROW(TYPE)                                      \
    do {                                                             \
        TopKPerRowNS::KernelTopKPerRow<TYPE> op;                     \
        op.Init(logits, row_starts, row_ends, indices,               \
                &tiling_data, &pipe);                                \
        op.Process();                                                \
    } while (0)

extern "C" __global__ __aicore__ void top_k_per_row(
    GM_ADDR logits, GM_ADDR row_starts, GM_ADDR row_ends, GM_ADDR indices,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        RUN_TOP_K_PER_ROW(float);
    } else if (TILING_KEY_IS(2)) {
        RUN_TOP_K_PER_ROW(half);
    } else if (TILING_KEY_IS(3)) {
        RUN_TOP_K_PER_ROW(bfloat16_t);
    }
    (void)workspace;
}
