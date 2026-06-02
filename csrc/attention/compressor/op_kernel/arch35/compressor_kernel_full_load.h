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
 * \file compressor_kernel_full_load.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_FULL_LOAD_H
#define COMPRESSOR_KERNEL_FULL_LOAD_H

#include "compressor_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"
#include "compressor_tools.h"
#include "compressor_block_cube_full_load.h"
#include "compressor_block_vec_full_load.h"

using namespace AscendC;

namespace Compressor {
template <typename COMP>
class CompressorKernelFullLoad {
public:
    __aicore__ inline CompressorKernelFullLoad(TPipe* pipe, const optiling::CompressorTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *stateBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    // ================================Init functions==================================
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    // ================================Process functions================================
    __aicore__ inline void InitTilingData();
    // 获取基本块数量
    __aicore__ inline void SkipInvalidBatch(BatchInfo &batchInfo);
    // 计算分核基本信息
    __aicore__ inline void CalcSplitCoreInfo();

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);
    __aicore__ inline void ComputeVec2(const Vec2RunInfo &info);

    __aicore__ inline bool IsNeedExcuteC1(RunInfo info);
    __aicore__ inline bool IsNeedSyncAll(uint32_t curBasicBlockIdx);
    __aicore__ inline void UpdateVec2Info(Vec2RunInfo &vec2Info, uint32_t curBasicBlockIdx, const Vec1RunInfo &info);
    __aicore__ inline bool IsNeedExcuteV2(Vec2RunInfo &vec2Info);
    __aicore__ inline void CalcCubeParams(RunInfo &info);
    __aicore__ inline void CalcV1Params(Vec1RunInfo &vec1Info);

    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;

    // 常量
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_FLAG = 3;
    static constexpr uint32_t SYNC_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_FLAG2 = 5;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 9;

    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::CompressorTilingData* __restrict tilingData_;
    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> mm1KvResGm;
    GlobalTensor<MM1_OUT_T> mm1ScoreResGm;
    GlobalTensor<MM1_OUT_T> vec1KvCacheGm;
    GlobalTensor<MM1_OUT_T> vec1ScoreCacheGm;
    GlobalTensor<MM1_OUT_T> Vec1InputKvGm;
    GlobalTensor<MM1_OUT_T> Vec1InputScoreGm;
    GlobalTensor<VEC1_OUT_T> vec1ResGm;
    GlobalTensor<VEC1_OUT_T> vec2InputGm;
    // ================================Task Info====================================
    CompressorTools<COMP> tools_;
    ConstInfo constInfo{};
    uint32_t aiCoreIdx = 0;

    // ==============================Service Define==============================
    CompressorBlockCubeFullLoad<COMP> blockCube_;
    CompressorBlockVectorFullLoad<COMP> blockVec_;

    uint32_t allCompressedTcNum_ = 0;
    uint32_t curCompressedTcNum_ = 0;
    uint32_t accDealSize = 0;
    uint32_t loopTimes = 0;
    uint32_t cubeLoop = 0;
    uint32_t vec1Loop = 0;
    uint32_t vec2Loop = 0;
    uint32_t kStartIdx_ = 0;
    uint32_t dealKSize_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *stateBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
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
    tools_.Init(startPos, seqUsed, cuSeqlens);

    // 剔除尾部的无效batch
    for (; constInfo.batchSize > 0; --constInfo.batchSize) {
        uint32_t bSeqUsed = tools_.GetSeqLength(constInfo.batchSize - 1);
        if (bSeqUsed > 0) {
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
        blockCube_.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos,
            stateBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockCube_.InitBuffers(pipe_);
        blockCube_.InitGlobalBuffers(mm1KvResGm, mm1ScoreResGm);
    } else {
        blockVec_.InitParams(constInfo, tools_);
        blockVec_.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,
                        cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockVec_.InitBuffers(pipe_);
        blockVec_.InitVec1GlobalTensor(Vec1InputKvGm, Vec1InputScoreGm, vec1KvCacheGm, vec1ScoreCacheGm, vec1ResGm, vec2InputGm);
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::InitTilingData() {
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.dBaseSize = tilingData_->innerSplitParams.dBaseSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.hSize = tilingData_->baseParams.hiddenSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.ropeHeadDim = tilingData_->baseParams.ropeHeadDim;
    constInfo.normEps = tilingData_->baseParams.normEps;
    constInfo.stateCacheStrideDim0 = tilingData_->baseParams.stateCacheStrideDim0;
    constInfo.reciprocalD = tilingData_->baseParams.reciprocalD;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;

    constInfo.blockNum = tilingData_->pageAttentionParams.blockNum;
    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    constInfo.nSize =  tilingData_->baseParams.nSize;
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
__aicore__ inline void CompressorKernelFullLoad<COMP>::SkipInvalidBatch(BatchInfo &batchInfo)
{
    for (; batchInfo.bIdx < constInfo.batchSize; ++batchInfo.bIdx) {
        batchInfo.seqCnt = tools_.GetSeqLength(batchInfo.bIdx);
        if (batchInfo.seqCnt > 0) {
            break;
        }
    }
    batchInfo.remSeqCnt = batchInfo.seqCnt;
    if (tools_.isExistSeqUsed_) {
        batchInfo.seqUsedCnt = tools_.GetSeqUsed(batchInfo.bIdx);
    } else {
        batchInfo.seqUsedCnt = batchInfo.seqCnt;
    }
    if (batchInfo.bIdx < constInfo.batchSize) {
        batchInfo.bStartPos = tools_.GetStartPos(batchInfo.bIdx);
        batchInfo.sIdx = 0;
        batchInfo.headHolderSeq = batchInfo.bStartPos & (constInfo.cmpRatio - 1);
        batchInfo.tcNum = (batchInfo.bStartPos + batchInfo.seqCnt + constInfo.cmpRatio - 1) / constInfo.cmpRatio - batchInfo.bStartPos /  constInfo.cmpRatio;
        batchInfo.compressedTcNum = (batchInfo.bStartPos + batchInfo.seqUsedCnt) / constInfo.cmpRatio - batchInfo.bStartPos /  constInfo.cmpRatio;
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::CalcSplitCoreInfo()
{
    // D方向的基本块数量
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;
    // 核的组数
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;
    // 当前组id
    constInfo.curGroupIdx = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;
    constInfo.mGroupNum = constInfo.coreGroupNum;
    constInfo.mCurGroupIdx = constInfo.curGroupIdx;

    uint32_t coff = (uint32_t)COMP::coff;
    constInfo.mm1KvResSize = constInfo.mBaseSize * constInfo.headDim * coff;
    constInfo.mm1ScoreResSize = constInfo.mBaseSize * constInfo.headDim * coff;
    constInfo.vec1ResSize = constInfo.mBaseSize * constInfo.headDim * constInfo.nSize;

    constInfo.dbSize = constInfo.coreGroupNum * constInfo.mm1KvResSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::InitWorkspace(__gm__ uint8_t *workspace) {
    uint64_t offset = 0;
    uint64_t mm1KvResStartOffset = offset;
    // mm1KvResGm
    mm1KvResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.curGroupIdx * constInfo.mm1KvResSize * sizeof(MM1_OUT_T)));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1KvResSize * sizeof(MM1_OUT_T);

    uint64_t mm1ScoreResStartOffset = offset;
    // mm1ScoreResGm
    mm1ScoreResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.curGroupIdx * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T)));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T);

    Vec1InputKvGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + mm1KvResStartOffset));

    Vec1InputScoreGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + mm1ScoreResStartOffset));

    vec1KvCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);

    vec1ScoreCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);

    uint64_t beforeVecOffset = offset;

    // vec1Res
    vec1ResGm.SetGlobalBuffer(
        (__gm__ VEC1_OUT_T *)(workspace + offset));
    offset +=  constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.vec1ResSize * sizeof(VEC1_OUT_T);
    // vec2Input
    vec2InputGm.SetGlobalBuffer(
        (__gm__ VEC1_OUT_T *)(workspace + beforeVecOffset));
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::ComputeMm1(const RunInfo &info) {
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG + info.cubeDbIdx);
    blockCube_.ComputeMm1(info);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG + info.cubeDbIdx);
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::ComputeVec1(const Vec1RunInfo &info) {
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_C1_V1_FLAG + info.c1v1DbIdx);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG2 + info.c1v1DbIdx);
    blockVec_.ComputeVec1();
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_V1_C1_FLAG + info.c1v1DbIdx);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_V1_FLAG2 + (info.c1v1DbIdx + 1) % constInfo.dbWorkspaceRatio);
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::ComputeVec2(const Vec2RunInfo &info) {
    blockVec_.ComputeVec2();
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::AllocEventID()
{
    if ASCEND_IS_AIC {
        blockCube_.AllocEventID(pipe_);
    } else {
        blockVec_.AllocEventID();
        for (int i = 0; i < constInfo.dbWorkspaceRatio; ++i) {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_V1_C1_FLAG + i);
        }
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_V1_FLAG2);
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::FreeEventID()
{
    if ASCEND_IS_AIC {
        for (int i = 0; i < constInfo.dbWorkspaceRatio; ++i) {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG + i);
        }
        blockCube_.FreeEventID(pipe_);
    } else {
        CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG2 + loopTimes % constInfo.dbWorkspaceRatio);
        blockVec_.FreeEventID();
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernelFullLoad<COMP>::IsNeedExcuteC1(RunInfo info)
{
    // B超出范围则cube不执行
    return info.bStart < constInfo.batchSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::CalcCubeParams(RunInfo &info)
{
    info.cubeDbIdx = (cubeLoop++ & (constInfo.dbWorkspaceRatio - 1));
    info.dealSeqCnt = constInfo.mEnd - constInfo.mStart;
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::CalcV1Params(Vec1RunInfo &vec1Info)
{
    vec1Info.bStart = 0;
    vec1Info.sStart = 0;
    vec1Info.resetResFlag = false;
    vec1Info.c1v1DbIdx = (vec1Loop++ & (constInfo.dbWorkspaceRatio - 1));
    vec1Info.v1v2DbIdx = (vec2Loop & (constInfo.dbWorkspaceRatio - 1));
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        uint64_t bSeqCnt = tools_.GetSeqLength(bIdx);
        if (bSeqCnt == 0) {
            continue;
        }
        uint64_t bStartPos = tools_.GetStartPos(bIdx);
        vec1Info.dealTcNum += (bStartPos + bSeqCnt + constInfo.cmpRatio - 1) / constInfo.cmpRatio - bStartPos / constInfo.cmpRatio;
        vec1Info.dealScSize += (bStartPos + bSeqCnt) / constInfo.cmpRatio - bStartPos / constInfo.cmpRatio;
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernelFullLoad<COMP>::IsNeedExcuteV2(Vec2RunInfo &vec2Info)
{
    return (vec2Info.dealScSize > 0);
}

template <typename COMP>
__aicore__ inline bool CompressorKernelFullLoad<COMP>::IsNeedSyncAll(uint32_t curBasicBlockIdx)
{
    if (allCompressedTcNum_ == 0) {
        return false;
    }

    uint32_t cnt = curBasicBlockIdx + 1;
    if ((cnt == loopTimes) || (cnt % constInfo.nSize == 0)) {
        return true;
    }
    return false;
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::UpdateVec2Info(
    Vec2RunInfo &vec2Info, uint32_t curBasicBlockIdx, const Vec1RunInfo &info)
{
    // nSize轮起始先重置v2Info信息
    if (curBasicBlockIdx % constInfo.nSize == 0) {
        vec2Info.v2DbIdx = (vec2Loop & (constInfo.dbWorkspaceRatio - 1));
        vec2Info.bStart = info.bStart;
        vec2Info.sStart = info.sStart;
        // 将sStart转成bCompressedId
        uint32_t startPos = tools_.GetStartPos(info.bStart);
        if (tools_.isExistSeqUsed_) {
            uint32_t seqUsed = tools_.GetSeqUsed(info.bStart);
            if (vec2Info.sStart >= seqUsed) {
                vec2Info.bStart++;
                vec2Info.sStart = 0;
            }
        }
        vec2Info.bCompressedId = (startPos + vec2Info.sStart) / constInfo.cmpRatio - startPos / constInfo.cmpRatio;

        vec2Info.dealScSize = 0;
    } else if ((curBasicBlockIdx + 1) % constInfo.nSize == 0) {
        vec2Loop++;
    }
    vec2Info.dealScSize += info.dealScSize;
    vec2Info.compressedId += info.dealScSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelFullLoad<COMP>::Process()
{
    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }
    AllocEventID();

    RunInfo extraInfo[1];
    Vec1RunInfo vec1Info{};
    Vec2RunInfo vec2Info{};
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
            ComputeVec1(vec1Info);
            UpdateVec2Info(vec2Info, i, vec1Info);
            SyncAll();
            if (IsNeedExcuteV2(vec2Info)) {
                ComputeVec2(vec2Info);
            }
        }
    }
    FreeEventID();
}

} // namespace Compressor

#endif // COMPRESSOR_KERNEL_FULL_LOAD_H
