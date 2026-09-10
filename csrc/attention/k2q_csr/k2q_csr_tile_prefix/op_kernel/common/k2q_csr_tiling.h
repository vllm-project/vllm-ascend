/**
 * k2q_csr AscendC tiling data.
 *
 * V1 (910b): Stage M + Map/Hist + RowPtr + device Emit
 * A5/arch35: Stage M + CUDA-like G 路 Hist/Prefix/Scatter
 */
#ifndef K2Q_CSR_TILING_H
#define K2Q_CSR_TILING_H

#include <cstdint>

struct K2qCsrTilingData {
    int64_t H;
    int64_t T;
    int64_t topk;
    int64_t N;           // T * topk
    int64_t B;
    int64_t totalRows;
    int64_t maxKv;
    int64_t rowMapElems; // B * maxKv

    // V1 edge-tile 字段（910b / 兼容）
    int64_t tileEdges;
    int64_t numTiles;
    int64_t totalUnits;
    int64_t workPer;
    int64_t maxIters;
    int64_t usedCores;

    int32_t orderMethod; // 0=Concat, 1=Round-robin
    int32_t useGather;
    int32_t emitTileEdges;

    // A5 / CUDA-like
    int32_t isArch35;    // 1 = Ascend950
    int32_t numGroups;   // G：q 维划分组数（≈ CTA 数），V1 下为 1
    int32_t qPerGroup;   // 每组负责的 q token 数（ceil(T/G)）
    int32_t reserved0;   // 阶段号（调试）：0 Meta / 1 Hist / 2 PR / 3 PT / 4 Scatter
    int32_t useSimt;     // 1 = Hist/Scatter 走 SIMT（仅 ascend950 生效）
    int32_t qGlobalOffset; // 1 = q_ind 写全局 Q 下标；0 = batch-local（默认）
};

// 共享实现位于 k2q_csr_common/；各命名算子薄入口 include 本头与对应 stage 头。

#endif
