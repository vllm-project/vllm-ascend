/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "kernel_operator.h"
#include "chunk_kda_fwd_prepare_tiling_key.h"
#include "chunk_kda_fwd_prepare_kernel.h"

template <int D_T_GATE, int D_T_BETA, uint32_t NORM_MODE,
          uint32_t BETA_MODE, uint32_t GATE_MODE,
          bool USE_EXP2, bool SAFE_GATE, uint32_t OUTPUT_MODE>
__global__ __aicore__ void chunk_kda_fwd_prepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
    GM_ADDR kg, GM_ADDR qg_scaled, GM_ADDR q_hat, GM_ADDR k_hat,
    GM_ADDR q_rstd, GM_ADDR k_rstd, GM_ADDR beta_eff, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(KdaPrepare::ChunkKdaFwdPrepareTilingData);
    GET_TILING_DATA_WITH_STRUCT(KdaPrepare::ChunkKdaFwdPrepareTilingData,
                                tilingData, tiling);

    KdaPrepare::PrepareKernelArgs args{};
    args.q = q;
    args.k = k;
    args.v = v;
    args.rawGate = g;
    args.beta = beta;
    args.aLog = a_log;
    args.dtBias = dt_bias;
    args.cuSeqlens = cu_seqlens;
    args.chunkIndices = chunk_indices;
    args.gk = gk;
    args.aqk = aqk;
    args.akk = akk;
    args.w = w;
    args.u = u;
    args.qg = qg;
    args.kg = kg;
    args.qgScaled = qg_scaled;
    args.qHat = q_hat;
    args.kHat = k_hat;
    args.qRstd = q_rstd;
    args.kRstd = k_rstd;
    args.betaEff = beta_eff;
    args.workspace = AscendC::GetUserWorkspace(workspace);
    if (args.workspace == nullptr) {
        return;
    }
    args.tiling.batch = static_cast<uint32_t>(tilingData.batch);
    args.tiling.seqNum = static_cast<uint32_t>(tilingData.seqNum);
    args.tiling.seqLen = static_cast<uint32_t>(tilingData.seqLen);
    args.tiling.qkHeadNum = static_cast<uint32_t>(tilingData.qkHeadNum);
    args.tiling.valueHeadNum =
        static_cast<uint32_t>(tilingData.valueHeadNum);
    args.tiling.totalChunks = static_cast<uint32_t>(tilingData.totalChunks);
    args.tiling.usedCoreNum = static_cast<uint32_t>(tilingData.usedCoreNum);
    args.tiling.headsPerPartition =
        static_cast<uint32_t>(tilingData.headsPerPartition);
    args.tiling.epsilon = tilingData.epsilon;
    args.tiling.lowerBound = tilingData.lowerBound;
    args.tiling.scale = tilingData.scale;
    args.tiling.isVarLen = tilingData.isVarLen;
    args.tiling.inputSequenceMajor = tilingData.inputSequenceMajor;
    args.tiling.hasDtBias = tilingData.hasDtBias;

    using Policy = KdaPrepare::PrepareCompilePolicy<
        static_cast<KdaPrepare::QkNormMode>(NORM_MODE),
        static_cast<KdaPrepare::BetaMode>(BETA_MODE),
        static_cast<KdaPrepare::GateMode>(GATE_MODE), USE_EXP2, SAFE_GATE,
        static_cast<KdaPrepare::OutputMode>(OUTPUT_MODE)>;
    using GateT = typename KdaPrepare::PrepareStorageType<D_T_GATE>::type;
    using BetaT = typename KdaPrepare::PrepareStorageType<D_T_BETA>::type;
    KdaPrepare::RunPrepare<GateT, BetaT, Policy>(args);
}
