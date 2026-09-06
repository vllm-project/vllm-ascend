/*
 * gmm_situ_vcv_dev — graph-capturable device-group_list entry for the n3-A
 * fused gmm_situ_vcv datapath (round_3/parallel_4, n11_graph).
 *
 * Same single-launch fused kernel body as gmm_situ_vcv (shared controller in
 * op_kernel/gmsq_vcv_controller.h); differences are confined to the group
 * table source and the launch grid:
 *   - group_list arrives as a DEVICE tensor and is consumed on-device: an
 *     in-kernel preamble segment (per-core 28-entry running cumsum) builds the
 *     equivalent of the eager X14 active table — NOT an independent cumsum
 *     device kernel, so the contract red line is respected.
 *   - grid = full AIC core count (static "28-expert 满配" launch): no host
 *     dependence on the active-expert table, so the launch config is identical
 *     for every replay of a captured graph; cores with no basic blocks exit
 *     after the preamble (the +1~3% low-activity cost bucket).
 * Replay semantics: the kernel re-reads the group_list device buffer on every
 * launch, so an in-place updated router output is honored by each graph
 * replay (no frozen host copy).
 */
#include "gmsq_vcv_controller.h"

// Namespace-free alias so the REGISTER_TILING_DEFAULT metadata section name
// stays simple, matching the mla_prolog_v3 / situ_mx_quant pattern.
using GmmSituTilingData = gmm_situ::SituTilingHeader;

// NOTE: the kernel entry and its metadata macros (KERNEL_TASK_TYPE_DEFAULT /
// REGISTER_TILING_DEFAULT) must stay OUTSIDE any __NPU_ARCH__ guard — the
// AscendC feature-precompile pass that extracts the tiling struct does not
// run with __NPU_ARCH__==3510 and would miss them entirely ("do not registe
// tiling struct" / capacity-8 tiling buffer at runtime).
extern "C" __global__ __aicore__ void grouped_matmul_situ_quant(
    GM_ADDR x, GM_ADDR xScale, GM_ADDR w, GM_ADDR wScale, GM_ADDR groupList,
    GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GmmSituTilingData);
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
    AscendC::AscendCUtils::SetOverflow(1);
    TPipe tPipe;
    gmm_situ::GmmSituController op;
    op.InitDev(x, xScale, w, wScale, groupList, y, yScale, tiling);
    op.Process();
#endif
}
