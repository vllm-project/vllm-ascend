#include "kernel_operator.h"
#include "full_load.h"
#include "general.h"


extern "C" __global__ __aicore__ void transpose_kv_cache_by_block(GM_ADDR KCache, GM_ADDR VCache, GM_ADDR blockIDs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(0)) {
        // full load not db
        TransposeKvCacheByBlockKernelFullLoad<DTYPE_KCACHE> kernel;
        kernel.Init(KCache, VCache, blockIDs, &tiling_data, &tPipe);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        // db \ align split blockSize
        TransposeKvCacheByBlockKernelGeneral<DTYPE_KCACHE, uint32_t(2), false> kernel;
        kernel.Init(KCache, VCache, blockIDs, &tiling_data, &tPipe);
        kernel.Process();
    } else if (TILING_KEY_IS(2)) {
        // not db \ align split blockSize
        TransposeKvCacheByBlockKernelGeneral<DTYPE_KCACHE, uint32_t(1), false> kernel;
        kernel.Init(KCache, VCache, blockIDs, &tiling_data, &tPipe);
        kernel.Process();
    } else if (TILING_KEY_IS(3)) {
        // db \ unalign split blockSize
        TransposeKvCacheByBlockKernelGeneral<DTYPE_KCACHE, uint32_t(2), true> kernel;
        kernel.Init(KCache, VCache, blockIDs, &tiling_data, &tPipe);
        kernel.Process();
    } else if (TILING_KEY_IS(4)) {
        // not db \ unalign split blockSize
        TransposeKvCacheByBlockKernelGeneral<DTYPE_KCACHE, uint32_t(1), true> kernel;
        kernel.Init(KCache, VCache, blockIDs, &tiling_data, &tPipe);
        kernel.Process();
    }
}