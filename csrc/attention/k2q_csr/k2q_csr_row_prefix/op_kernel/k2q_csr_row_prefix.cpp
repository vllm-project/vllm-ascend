/**
 * K2qCsrRowPrefix kernel（910b / A2）。
 */
#include "kernel_operator.h"
#include "common/k2q_csr_tiling.h"
#include "common/k2q_csr_scratch.h"
#include "common/k2q_csr_prefix.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void k2q_csr_row_prefix(GM_ADDR scratch, GM_ADDR row_ptr, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(K2qCsrTilingData);
    GET_TILING_DATA_WITH_STRUCT(K2qCsrTilingData, tilingData, tiling);
    (void)workspace;

    GM_ADDR rowMap = nullptr;
    GM_ADDR tokenBatch = nullptr;
    GM_ADDR histScratch = nullptr;
    K2qCsrScratch::Split(scratch, &tilingData, rowMap, tokenBatch, histScratch);
    (void)rowMap;
    (void)tokenBatch;

    TPipe pipe;
    K2qCsrRowPrefixKernel op;
    op.Init(row_ptr, nullptr, nullptr, histScratch, &tilingData, &pipe);
    op.Process();
}
