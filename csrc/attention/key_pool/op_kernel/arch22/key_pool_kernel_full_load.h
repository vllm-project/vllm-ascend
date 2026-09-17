/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file key_pool_kernel_full_load.h
 * \brief
 */

#ifndef KEY_POOL_KERNEL_FULL_LOAD_H
#define KEY_POOL_KERNEL_FULL_LOAD_H

#include "key_pool_comm_arch22.h"
#include "key_pool_template_tiling_key.h"
#include "key_pool_tiling_data.h"
#include "key_pool_tools.h"
#include "key_pool_block_cube_full_load.h"
#include "key_pool_block_vec_full_load.h"

using namespace AscendC;

namespace KeyPool {
template <typename COMP>
class KeyPoolKernelFullLoad {
public:
    __aicore__ inline KeyPoolKernelFullLoad(TPipe *pipe, const optiling::KeyPoolTilingData *__restrict tilingData)
        : pipe_(pipe),
          tilingData_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk, __gm__ uint8_t *gateWeight,
                                __gm__ uint8_t *normWeight, __gm__ uint8_t *normBias, __gm__ uint8_t *stateCache,
                                __gm__ uint8_t *ape, __gm__ uint8_t *cacheBlockTable, __gm__ uint8_t *seqLens,
                                __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *pooledKeyOut,
                                __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    // ================================Init functions==================================
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    // ================================Process functions================================
    __aicore__ inline void InitTilingData();
    // 计算分核基本信息
    __aicore__ inline void CalcSplitCoreInfo();

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeVec();

    __aicore__ inline void CalcCubeParams(RunInfo &info);
    __aicore__ inline void CalcV1Params(Vec1RunInfo &vec1Info);

    using HIDDEN_STATES_T =
        typename AscendC::Conditional<COMP::hiddenStatesDtype == HIDDEN_STATES_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;

    // 常量
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_FLAG = 3;
    static constexpr uint32_t SYNC_V1_FLAG = 4;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;

    // ==============================TilingData&TPipe==============================
    TPipe *pipe_;
    const optiling::KeyPoolTilingData *__restrict tilingData_;
    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> mm1KvResGm;
    GlobalTensor<MM1_OUT_T> mm1ScoreResGm;
    GlobalTensor<MM1_OUT_T> vec1KvCacheGm;
    GlobalTensor<MM1_OUT_T> vec1ScoreCacheGm;
    GlobalTensor<MM1_OUT_T> Vec1InputKvGm;
    GlobalTensor<MM1_OUT_T> Vec1InputScoreGm;
    // ================================Task Info====================================
    KeyPoolTools<COMP> tools_;
    ConstInfo constInfo{};
    uint32_t aiCoreIdx = 0;

    // ==============================Service Define==============================
    KeyPoolBlockCubeFullLoad<COMP> blockCube_;
    KeyPoolBlockVectorFullLoad<COMP> blockVec_;

    uint32_t loopTimes = 0;
    uint32_t cubeLoop = 0;
    uint32_t vec1Loop = 0;
};

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::Init(__gm__ uint8_t *hidden_states, __gm__ uint8_t *wk,
                                                         __gm__ uint8_t *gateWeight, __gm__ uint8_t *normWeight,
                                                         __gm__ uint8_t *normBias, __gm__ uint8_t *stateCache,
                                                         __gm__ uint8_t *ape, __gm__ uint8_t *cacheBlockTable,
                                                         __gm__ uint8_t *seqLens, __gm__ uint8_t *seqUsed,
                                                         __gm__ uint8_t *startPos, __gm__ uint8_t *pooledKeyOut,
                                                         __gm__ uint8_t *workspace)
{
    if ASCEND_IS_AIV {
        constInfo.aiCoreIdx = GetBlockIdx() / 2;
    } else {
        constInfo.aiCoreIdx = GetBlockIdx();
    }

    InitTilingData();
    // init tools
    tools_.toolParams_.seqSize = tilingData_->baseParams.seqSize;
    tools_.toolParams_.cmpRatio = tilingData_->baseParams.cmpRatio;
    tools_.Init(startPos, seqUsed, seqLens);

    // 剔除尾部的无效batch
    for (; constInfo.batchSize > 0; --constInfo.batchSize) {
        uint32_t bSeqUsed = tools_.GetSeqLength(constInfo.batchSize - 1);
        uint64_t bStartPos = tools_.GetStartPos(constInfo.batchSize - 1);
        if (bSeqUsed > 0 || bStartPos > 0) {
            break;
        }
    }

    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }

    // 0. 计算最后一个Tc块的起始位置
    constInfo.bIdxOfLastTc = constInfo.batchSize - 1;
    // 1. 计算head_dim的切分大小, 构建ConstInfo的其他信息
    CalcSplitCoreInfo();
    // 2. 计算循环次数
    loopTimes = 1;
    // 3. 初始化workspace
    InitWorkspace(workspace);
    // 4. 初始化block层
    if ASCEND_IS_AIC {
        blockCube_.InitParams(constInfo, tools_);
        blockCube_.Init(hidden_states, wk, gateWeight, stateCache, ape, cacheBlockTable, seqLens, seqUsed, startPos,
                        pooledKeyOut);
        blockCube_.InitBuffers(pipe_);
        blockCube_.InitGlobalBuffers(mm1KvResGm, mm1ScoreResGm);
    } else {
        blockVec_.InitParams(constInfo, tools_);
        blockVec_.Init(hidden_states, wk, gateWeight, normWeight, normBias, stateCache, ape, cacheBlockTable, seqLens,
                       seqUsed, startPos, pooledKeyOut);
        blockVec_.InitBuffers(pipe_);
        blockVec_.InitVec1GlobalTensor(Vec1InputKvGm, Vec1InputScoreGm, vec1KvCacheGm, vec1ScoreCacheGm);
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::InitTilingData()
{
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.dBaseSize = tilingData_->innerSplitParams.dBaseSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.normEps = tilingData_->baseParams.normEps;
    constInfo.stateCacheStrideDim0 = tilingData_->baseParams.stateCacheStrideDim0;
    constInfo.hSize = tilingData_->baseParams.hiddenSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;

    constInfo.blockNum = tilingData_->pageAttentionParams.blockNum;
    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    constInfo.nSize = tilingData_->baseParams.nSize;
    constInfo.vec1TailCacheSize = tilingData_->workspaceParams.vec1TailCacheSize;
    constInfo.dbWorkspaceRatio = tilingData_->workspaceParams.dbWorkspaceRatio;

    constInfo.mStart = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].mStart;
    constInfo.mEnd = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].mEnd;
    constInfo.nStart = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].nStart;
    constInfo.nEnd = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].nEnd;
    constInfo.kStart = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].kStart;
    constInfo.kEnd = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].kEnd;
    constInfo.mLoopNum = tilingData_->baseParams.mLoopNum;
    constInfo.kBaseNum = tilingData_->baseParams.kBaseNum;
    constInfo.kBaseSize = tilingData_->baseParams.kBaseSize;
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::CalcSplitCoreInfo()
{
    // D方向的基本块数量
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;
    // 核的组数
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;
    // 当前组id
    constInfo.curGroupIdx = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;
    constInfo.mGroupNum = constInfo.coreGroupNum;
    constInfo.mCurGroupIdx = constInfo.curGroupIdx;

    constInfo.mm1KvResSize = constInfo.mBaseSize * constInfo.headDim;
    constInfo.mm1ScoreResSize = constInfo.mBaseSize * constInfo.headDim;
    constInfo.vec1ResSize = constInfo.mBaseSize * constInfo.headDim * constInfo.nSize;

    constInfo.dbSize = constInfo.coreGroupNum * constInfo.mm1KvResSize;
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::InitWorkspace(__gm__ uint8_t *workspace)
{
    uint64_t offset = 0;
    uint64_t mm1KvResStartOffset = offset;
    // mm1KvResGm
    mm1KvResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset + constInfo.curGroupIdx * constInfo.mm1KvResSize * sizeof(MM1_OUT_T)));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1KvResSize * sizeof(MM1_OUT_T);

    uint64_t mm1ScoreResStartOffset = offset;
    // mm1ScoreResGm
    mm1ScoreResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.curGroupIdx * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T)));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T);

    Vec1InputKvGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + mm1KvResStartOffset));

    Vec1InputScoreGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + mm1ScoreResStartOffset));

    vec1KvCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);

    vec1ScoreCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::ComputeMm1(const RunInfo &info)
{
    blockCube_.ComputeMm1(info);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::ComputeVec()
{
    blockVec_.PrepareForComputeVec();
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_C1_V1_FLAG);
    blockVec_.ComputeVec1();
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::AllocEventID()
{
    if ASCEND_IS_AIC {
        blockCube_.AllocEventID(pipe_);
    } else {
        blockVec_.AllocEventID();
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::FreeEventID()
{
    if ASCEND_IS_AIC {
        blockCube_.FreeEventID(pipe_);
    } else {
        blockVec_.FreeEventID();
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::CalcCubeParams(RunInfo &info)
{
    info.cubeDbIdx = (cubeLoop++ & (constInfo.dbWorkspaceRatio - 1));
    info.dealSeqCnt = constInfo.mEnd - constInfo.mStart;
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::CalcV1Params(Vec1RunInfo &vec1Info)
{
    vec1Info.bStart = 0;
    vec1Info.sStart = 0;
    vec1Info.c1v1DbIdx = (vec1Loop++ & (constInfo.dbWorkspaceRatio - 1));
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        uint64_t bSeqCnt = tools_.GetSeqLength(bIdx);
        if (bSeqCnt == 0) {
            continue;
        }
        uint64_t bStartPos = tools_.GetStartPos(bIdx);
        vec1Info.dealTcNum +=
            (bStartPos + bSeqCnt + constInfo.cmpRatio - 1) / constInfo.cmpRatio - bStartPos / constInfo.cmpRatio;
    }
}

template <typename COMP>
__aicore__ inline void KeyPoolKernelFullLoad<COMP>::Process()
{
    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }
    AllocEventID();

    RunInfo extraInfo[1];
    Vec1RunInfo vec1Info{};
    for (uint32_t i = 0; i < loopTimes; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        if ASCEND_IS_AIV {
            CalcV1Params(vec1Info);
        } else {
            CalcCubeParams(extraInfo0);
        }
        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0);
        } else {
            ComputeVec();
        }
    }
    FreeEventID();
}

} // namespace KeyPool

#endif // KEY_POOL_KERNEL_FULL_LOAD_H
