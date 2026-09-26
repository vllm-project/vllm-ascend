/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "kernel_operator.h"
#include "chunk_kda_fwd_finalize_tiling_key.h"
#include "chunk_kda_fwd_finalize_kernel.h"

template <bool USE_AIV_INPUT_MOVER>
__global__ __aicore__ void chunk_kda_fwd_finalize(
    GM_ADDR qg_scaled, GM_ADDR aqk, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(KdaFinalize::ChunkKdaFwdFinalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(KdaFinalize::ChunkKdaFwdFinalizeTilingData,
                                tilingData, tiling);

    KdaFinalize::FinalizeArgs args{};
    args.qgScaled = qg_scaled;
    args.aqk = aqk;
    args.vNew = v_new;
    args.h = h;
    args.cuSeqlens = cu_seqlens;
    args.chunkIndices = chunk_indices;
    args.attnOut = attn_out;
    (void)workspace;
    args.tiling.batch = static_cast<uint32_t>(tilingData.batch);
    args.tiling.seqNum = static_cast<uint32_t>(tilingData.seqNum);
    args.tiling.seqLen = static_cast<uint32_t>(tilingData.seqLen);
    args.tiling.valueHeadNum = static_cast<uint32_t>(tilingData.valueHeadNum);
    args.tiling.totalChunks = static_cast<uint32_t>(tilingData.totalChunks);
    args.tiling.usedCoreNum = static_cast<uint32_t>(tilingData.usedCoreNum);
    args.tiling.headsPerPartition = static_cast<uint32_t>(tilingData.headsPerPartition);
    args.tiling.isVarLen = tilingData.isVarLen;
    args.tiling.outputSequenceMajor = tilingData.outputSequenceMajor;
    args.tiling.stateVFirst = tilingData.stateVFirst;

    constexpr bool kUseAivInputMover =
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        USE_AIV_INPUT_MOVER;
#else
        false;
#endif
    KdaFinalize::DispatchFinalize<kUseAivInputMover>(args);
}
