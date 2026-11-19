/**
 * K2qCsrScatter apt（ascend950 / A5）。
 * use_simt=1 → SIMT（Host fill_-1，无核内 Fill+SyncAll）；
 * use_simt=0 → 共用 MC（与 A2 / non-apt 一致）。
 */
#include "kernel_operator.h"
#include "common/k2q_csr_tiling.h"
#include "common/k2q_csr_scratch.h"
#include "common/k2q_csr_mc.h"
#include "common/arch35/k2q_csr_simt_scatter_arch35.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void k2q_csr_scatter(GM_ADDR q2k, GM_ADDR cu_seqlens, GM_ADDR scratch, GM_ADDR q_ind,
                                                      GM_ADDR slot, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(K2qCsrTilingData);
    GET_TILING_DATA_WITH_STRUCT(K2qCsrTilingData, tilingData, tiling);
    (void)workspace;

    GM_ADDR rowMap = nullptr;
    GM_ADDR tokenBatch = nullptr;
    GM_ADDR histScratch = nullptr;
    K2qCsrScratch::Split(scratch, &tilingData, rowMap, tokenBatch, histScratch);

    TPipe pipe;
    if (tilingData.useSimt != 0) {
        K2qCsrSimtScatter op;
        op.Init(q2k, cu_seqlens, rowMap, tokenBatch, q_ind, slot, histScratch, &tilingData, &pipe);
        op.Process();
        return;
    }
    K2qCsrPipelineMc op;
    op.Init(q2k, cu_seqlens, rowMap, tokenBatch, nullptr, q_ind, slot, histScratch, &tilingData, &pipe);
    op.ProcessPhase3Scatter();
}
