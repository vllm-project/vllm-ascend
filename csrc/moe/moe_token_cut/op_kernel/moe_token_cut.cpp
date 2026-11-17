/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include "moe_token_cut.h"

using namespace AscendC;
using namespace MoeTokenCutImpl;

extern "C" __global__ __aicore__ void moe_token_cut(
    GM_ADDR topkIds,
    GM_ADDR topkWeights,
    GM_ADDR hidden,
    GM_ADDR cutIdx,
    GM_ADDR newTopk,
    GM_ADDR newTopkWeight,
    GM_ADDR newHidden,
    GM_ADDR newResiduals,
    GM_ADDR actualBs,
    GM_ADDR prefixIndex,
    GM_ADDR noCutRankNum,
    GM_ADDR activeMask,
    GM_ADDR localCutCount,
    GM_ADDR batchSize,
    GM_ADDR enableCut,
    GM_ADDR residuals,
    GM_ADDR discardLabel,
    GM_ADDR discardTokens,
    GM_ADDR discardMoeReqId,
    GM_ADDR reqIdList,
    GM_ADDR discardResiduals,
    GM_ADDR discardTopk,
    GM_ADDR discardTopkWeight,
    GM_ADDR moeCnt,
    GM_ADDR historyBs,
    GM_ADDR originalBs,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    // 先注册 MC2 tiling，确保进入算子主体前 HCCL window 上下文已经初始化。
    REGISTER_TILING_DEFAULT(MoeTokenCutTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeTokenCutTilingData, tilingData, tiling);

    // flag=2 只表示本轮 batchSize <= cutCountThreshold，不是永久停截。
    // 不能用上一轮写回的 noCutRankNum 提前返回：下一轮 bs 变大后仍可能截断，
    // 且 flag=2 的 rank 仍要发送 counts 给其他卡做全局统计。
    TPipe pipe;
    // 跳过运行时保留的 workspace 前缀，本算子只使用 Init 中定义的用户区。
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    MoeTokenCut<DTYPE_TOPK_WEIGHTS> op;
    // printf("0000");
    op.Init(topkIds, topkWeights, hidden, cutIdx, newTopk, newTopkWeight,
        newHidden, newResiduals, actualBs, prefixIndex, noCutRankNum,
        activeMask, localCutCount, batchSize,
        enableCut,
        residuals, discardLabel, discardTokens, discardMoeReqId, reqIdList,
        discardResiduals, discardTopk, discardTopkWeight, moeCnt, historyBs,
        originalBs, userWorkspace, &pipe, &tilingData);
    op.Process();
    // printf("after process");

}
