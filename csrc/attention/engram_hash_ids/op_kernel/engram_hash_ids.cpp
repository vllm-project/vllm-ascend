// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "engram_hash_ids_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void engram_hash_ids(
    GM_ADDR inputIds, GM_ADDR tokenMap, GM_ADDR deadMask, GM_ADDR positions,
    GM_ADDR blockTable, GM_ADDR queryStartLoc, GM_ADDR reqIds, GM_ADDR multipliers,
    GM_ADDR primes, GM_ADDR offsets, GM_ADDR cache, GM_ADDR lookbackTokenIds,
    GM_ADDR lookbackDeadMask, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    EngramHashIds::EngramHashIdsKernel<DTYPE_INPUT_IDS, DTYPE_TOKEN_MAP, DTYPE_POSITIONS,
                                     DTYPE_LOOKBACK_TOKEN_IDS> kernel;
    kernel.Init(inputIds, tokenMap, deadMask, positions, blockTable, queryStartLoc, reqIds,
                multipliers, primes, offsets, cache, lookbackTokenIds, lookbackDeadMask, output, tilingData, &pipe);
    kernel.Process();
}
