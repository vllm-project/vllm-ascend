#include "hamming_dist_top_k_split_s.h"
#include "hamming_dist_top_k_parallel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void hamming_dist_top_k(GM_ADDR query, GM_ADDR keyCompressed, GM_ADDR k,
        GM_ADDR seqLen, GM_ADDR chunkSize, GM_ADDR keyBlockTable, GM_ADDR indicesIn, GM_ADDR keyCompressedRope, GM_ADDR mask, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        HammingDistTopKParallelKernel op;
        op.Init(query, keyCompressed, keyCompressedRope, k, seqLen, chunkSize, keyBlockTable, mask, indices, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    } else if (TILING_KEY_IS(10)) {
        HammingDistTopKSplitSKernel op;
        op.Init(query, keyCompressed, keyCompressedRope, k, seqLen, chunkSize, keyBlockTable, mask, indices, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    }
}