/**
 * K2qCsrMeta apt（ascend950 / A5）。
 * 与 A2 共用 Meta 逻辑（无 SIMT 分支）。
 */
#include "kernel_operator.h"
#include "common/k2q_csr_tiling.h"
#include "common/k2q_csr_scratch.h"
#include "common/k2q_csr_stage_meta.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void k2q_csr_meta(GM_ADDR cu_seqlens, GM_ADDR cu_block_lens, GM_ADDR scratch,
                                                   GM_ADDR scratch_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(K2qCsrTilingData);
    GET_TILING_DATA_WITH_STRUCT(K2qCsrTilingData, tilingData, tiling);
    (void)scratch_out;
    (void)workspace;
    if (GetBlockIdx() != 0) {
        return;
    }
    GM_ADDR rowMap = nullptr;
    GM_ADDR tokenBatch = nullptr;
    GM_ADDR histScratch = nullptr;
    K2qCsrScratch::Split(scratch, &tilingData, rowMap, tokenBatch, histScratch);
    TPipe pipe;
    K2qCsrStageMeta meta;
    meta.Init(cu_seqlens, cu_block_lens, rowMap, tokenBatch, histScratch, &tilingData, &pipe);
    meta.Process();
}
