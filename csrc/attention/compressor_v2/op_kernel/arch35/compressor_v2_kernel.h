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
 * \file compressor_kernel.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_H
#define COMPRESSOR_KERNEL_H

#include "compressor_v2_comm_arch35.h"
#include "compressor_v2_template_tiling_key.h"
#include "compressor_v2_tiling_data.h"
#include "compressor_v2_tools.h"
#include "compressor_v2_block_cube.h"
#include "compressor_v2_block_vec_normal.h"
#include "compressor_v2_block_vec_full_load.h"

using namespace AscendC;

namespace CompressorV2 {

template <typename COMP, bool IS_FULL_LOAD = false>
class CompressorV2Kernel {
public:
    __aicore__ inline CompressorV2Kernel(TPipe *pipe, const optiling::CompressorV2TilingData *__restrict tilingData)
        : pipe_(pipe),
          tilingData_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                __gm__ uint8_t *stateCache, __gm__ uint8_t *stateBlockTable, __gm__ uint8_t *cuSeqlens,
                                __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut,
                                __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    // ================================Init functions==================================
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    // ================================Process functions================================
    __aicore__ inline void InitTilingData();
    // 运行时K轴切分（NORMAL独有）
    __aicore__ inline void SplitK();
    // 获取基本块数量（NORMAL独有）
    __aicore__ inline uint32_t GetLoopTimes();
    __aicore__ inline void SkipInvalidBatch(BatchInfo &batchInfo);
    // 更新当前组信息（NORMAL独有）
    __aicore__ inline void UpdateCurGroup(BasicBlockInfo &basicBlockInfo, BatchInfo batchInfo, uint32_t &curGroupQuota,
                                          uint32_t curDealSeq);
    // 单次循环的batch切分（NORMAL独有）
    __aicore__ inline BasicBlockInfo SkipOneLoop(BatchInfo &batchInfo);
    // 计算分核基本信息
    __aicore__ inline void CalcSplitCoreInfo();

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info, bool isNeedExcute = true);
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);

    __aicore__ inline bool IsNeedExcuteC1(RunInfo info);
    // 计算Cube/Vec1参数（NORMAL独有）
    __aicore__ inline void CalcC1V1Params(RunInfo &info, Vec1RunInfo &vec1Info, BatchInfo &batchInfo, uint32_t loopIdx);
    // 从constInfo直接读取mStart/mEnd（FULL_LOAD独有）
    __aicore__ inline void CalcCubeParams(RunInfo &info);
    // 计算vec1所需的压缩TC数量（FULL_LOAD独有）

    // NORMAL/FULL_LOAD主循环体分离
    __aicore__ inline void ProcessNormalLoop();
    __aicore__ inline void ProcessFullLoadLoop();

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
    TPipe *pipe_;
    const optiling::CompressorV2TilingData *__restrict tilingData_;
    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> mm1KvResGm;
    GlobalTensor<MM1_OUT_T> mm1ScoreResGm;
    GlobalTensor<MM1_OUT_T> vec1KvCacheGm;
    GlobalTensor<MM1_OUT_T> vec1ScoreCacheGm;
    GlobalTensor<MM1_OUT_T> Vec1InputKvGm;
    GlobalTensor<MM1_OUT_T> Vec1InputScoreGm;
    GlobalTensor<VEC1_OUT_T> vec2InputGm;
    // ================================Task Info====================================
    CompressorV2Tools<COMP> tools_;
    ConstInfo constInfo{};

    // ==============================Service Define==============================
    CompressorV2BlockCube<COMP, IS_FULL_LOAD> blockCube_;
    using VecType = typename AscendC::Conditional<IS_FULL_LOAD, CompressorV2BlockVectorFullLoad<COMP>,
                                                  CompressorV2BlockVectorNormal<COMP>>::type;
    VecType blockVec_;

    uint32_t loopTimes = 0;
    uint32_t cubeLoop = 0;
    uint32_t vec1Loop = 0;
    uint32_t kStartIdx_ = 0;
    uint32_t dealKSize_ = 0;
    uint32_t hStart_ = 0;
    bool isFirstUpdateCurGroup = true;
};

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv,
                                                                    __gm__ uint8_t *wGate, __gm__ uint8_t *stateCache,
                                                                    __gm__ uint8_t *stateBlockTable,
                                                                    __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed,
                                                                    __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut,
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
    if constexpr (!IS_FULL_LOAD) {
        // 2. 运行时K轴切分
        SplitK();
        // 3. 计算循环次数
        loopTimes = GetLoopTimes();
    } else {
        // 2. FULL_LOAD整块加载, 主循环只跑一次
        loopTimes = 1;
    }
    // 4. 初始化workspace
    InitWorkspace(workspace);
    // 5. 初始化block层
    if ASCEND_IS_AIC {
        blockCube_.InitParams(constInfo, tools_);
        blockCube_.Init(x, wKv, wGate, stateCache, stateBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockCube_.InitBuffers(pipe_);
        blockCube_.InitGlobalBuffers(mm1KvResGm, mm1ScoreResGm);
    } else {
        blockVec_.InitParams(constInfo, tools_);
        blockVec_.Init(x, wKv, wGate, stateCache, stateBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockVec_.InitBuffers(pipe_);
        blockVec_.InitVec1GlobalTensor(Vec1InputKvGm, Vec1InputScoreGm, vec1KvCacheGm, vec1ScoreCacheGm);
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::InitTilingData()
{
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.dBaseSize = tilingData_->innerSplitParams.dBaseSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.hSize = tilingData_->baseParams.hiddenSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.stateCacheStrideDim0 = tilingData_->baseParams.stateCacheStrideDim0;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;

    constInfo.blockNum = tilingData_->pageAttentionParams.blockNum;
    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    constInfo.nSize = tilingData_->baseParams.nSize;
    constInfo.vec1TailCacheSize = tilingData_->workspaceParams.vec1TailCacheSize;
    constInfo.dbWorkspaceRatio = tilingData_->workspaceParams.dbWorkspaceRatio;
    constInfo.batchConsistency = tilingData_->baseParams.batchConsistency;

    if constexpr (IS_FULL_LOAD) {
        constInfo.mStart = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].mStart;
        constInfo.mEnd = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].mEnd;
        constInfo.nStart = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].nStart;
        constInfo.nEnd = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].nEnd;
        constInfo.kStart = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].kStart;
        constInfo.kEnd = tilingData_->baseParams.splitCoreParam[constInfo.aiCoreIdx].kEnd;
        constInfo.mLoopNum = tilingData_->baseParams.mLoopNum;
        constInfo.kBaseNum = tilingData_->baseParams.kBaseNum;
        constInfo.kBaseSize = tilingData_->baseParams.kBaseSize;
    } else {
        constInfo.kBaseSize = tilingData_->baseParams.hiddenSize;
        constInfo.kBaseNum = 1;
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::SplitK()
{
    uint32_t mSize = 0;
    for (uint32_t i = 0; i < constInfo.batchSize; i++) {
        uint32_t bSeqUsed = tools_.GetSeqLength(i);
        // 获取m大小
        mSize += bSeqUsed;
    }
    uint32_t mBaseNum = CeilDivT(mSize, constInfo.mBaseSize);
    if (constInfo.dBasicBlockNum * mBaseNum < constInfo.usedCoreNum &&
        constInfo.batchConsistency != BATCH_CONSISTENCY) {
        constInfo.kBaseNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;
        uint32_t kAlignSize = CeilDivT(
            Align(constInfo.hSize, static_cast<uint32_t>(BUFFER_SIZE_BYTE_32B / sizeof(X_T))), constInfo.kBaseNum);
        constInfo.kBaseSize = Trunc(kAlignSize, static_cast<uint32_t>(BUFFER_SIZE_BYTE_32B / sizeof(X_T)));
        // 当切m轴无法满足开满核时，不切m轴(切m处理有点复杂)
        constInfo.mGroupNum = 1;    // 在m轴处理上所有核当一个组
        constInfo.mCurGroupIdx = 0; // 只有一个组
    }
    // 每轮固定不变，预计算后主循环直接复用
    if (constInfo.kBaseNum > 1) {
        kStartIdx_ = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;
        if (constInfo.curGroupIdx + 1 < constInfo.coreGroupNum) {
            dealKSize_ = constInfo.kBaseSize;
            hStart_ = kStartIdx_ * dealKSize_;
        } else {
            dealKSize_ = kStartIdx_ < constInfo.coreGroupNum ? constInfo.hSize - kStartIdx_ * constInfo.kBaseSize : 0;
            hStart_ = kStartIdx_ * constInfo.kBaseSize;
        }
    } else {
        kStartIdx_ = 0;
        dealKSize_ = constInfo.hSize;
        hStart_ = kStartIdx_ * dealKSize_;
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::SkipInvalidBatch(BatchInfo &batchInfo)
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
        batchInfo.headHolderSeq = (uint32_t)(batchInfo.bStartPos % constInfo.cmpRatio);
        batchInfo.tcNum =
            (uint32_t)((batchInfo.bStartPos + batchInfo.seqCnt + constInfo.cmpRatio - 1) / constInfo.cmpRatio -
                       batchInfo.bStartPos / constInfo.cmpRatio);
        batchInfo.compressedTcNum = (uint32_t)((batchInfo.bStartPos + batchInfo.seqUsedCnt) / constInfo.cmpRatio -
                                               batchInfo.bStartPos / constInfo.cmpRatio);
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::UpdateCurGroup(BasicBlockInfo &basicBlockInfo,
                                                                              BatchInfo batchInfo,
                                                                              uint32_t &curGroupQuota,
                                                                              uint32_t curDealSeq)
{
    // 更新当前组的信息
    if (curGroupQuota == 0 && !isFirstUpdateCurGroup) {
        return;
    }
    isFirstUpdateCurGroup = false;
    basicBlockInfo.bIdx = batchInfo.bIdx;
    uint32_t curGroupDealSeq = curGroupQuota < curDealSeq ? curGroupQuota : curDealSeq;
    basicBlockInfo.sIdx = batchInfo.sIdx + curGroupDealSeq;
    basicBlockInfo.dealSeqCnt += curGroupDealSeq;
    curGroupQuota -= curGroupDealSeq;
    // 结尾需要跳batch，需要考虑在当前组起始为末尾，或者当前组起始大于整个M轴
    if ((curGroupQuota == 0 || basicBlockInfo.bIdx == constInfo.batchSize - 1) &&
        basicBlockInfo.sIdx == batchInfo.seqCnt) {
        basicBlockInfo.sIdx = 0;
        for (basicBlockInfo.bIdx++; basicBlockInfo.bIdx < constInfo.batchSize; ++basicBlockInfo.bIdx) {
            uint32_t seqCnt = tools_.GetSeqLength(basicBlockInfo.bIdx);
            if (seqCnt > 0) {
                break;
            }
        }
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline BasicBlockInfo CompressorV2Kernel<COMP, IS_FULL_LOAD>::SkipOneLoop(BatchInfo &batchInfo)
{
    BasicBlockInfo basicBlockInfo{};
    isFirstUpdateCurGroup = true;
    uint32_t curGroupQuota = constInfo.mBaseSize * constInfo.mCurGroupIdx;
    uint32_t quota = constInfo.mGroupNum * constInfo.mBaseSize;

    for (; batchInfo.bIdx < constInfo.batchSize;) {
        uint32_t curDealSeq = 0;
        uint32_t curDealTcNum = 0;
        uint32_t curDealCompressedTcNum = 0;
        // 无法处理完当前整个batch
        if (quota < batchInfo.remSeqCnt) {
            // 向下对齐r，
            uint32_t alignSeq = constInfo.cmpRatio;
            if (batchInfo.bIdx == 0) {
                alignSeq = constInfo.cmpRatio - batchInfo.headHolderSeq;
            }
            if (quota > alignSeq) {
                uint64_t delta = (batchInfo.bStartPos + batchInfo.sIdx + quota) % constInfo.cmpRatio; // 超出对齐的部分
                curDealSeq = quota - delta;
                quota -= curDealSeq;
                curDealTcNum = (curDealSeq + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
                curDealCompressedTcNum = min(curDealTcNum, batchInfo.compressedTcNum);
                // 更新当前组所需信息
                UpdateCurGroup(basicBlockInfo, batchInfo, curGroupQuota, curDealSeq);
                // 更新batch信息
                batchInfo.remSeqCnt = batchInfo.remSeqCnt - curDealSeq;
                batchInfo.sIdx = batchInfo.sIdx + curDealSeq;
                batchInfo.compressedTcNum -= curDealCompressedTcNum;
                batchInfo.tcNum -= curDealTcNum;
                // 更新loop信息
                basicBlockInfo.dealTcNum += curDealTcNum;
                basicBlockInfo.compressedTcNum += curDealCompressedTcNum;
            }
            break;
        } else {
            // 处理整个batch
            quota -= batchInfo.remSeqCnt;
            curDealSeq = batchInfo.remSeqCnt;
            curDealTcNum = batchInfo.tcNum;
            // 更新当前组所需信息
            UpdateCurGroup(basicBlockInfo, batchInfo, curGroupQuota, curDealSeq);
            // 更新batch和loop信息
            batchInfo.remSeqCnt = 0;
            basicBlockInfo.dealTcNum += batchInfo.tcNum;
            basicBlockInfo.compressedTcNum += batchInfo.compressedTcNum;
            batchInfo.bIdx++;
            SkipInvalidBatch(batchInfo);
        }
    }
    uint32_t totalDataSize = constInfo.mGroupNum * constInfo.mBaseSize - quota;
    // 2. 当前组的起始偏移
    uint32_t currentGroupStart = constInfo.mCurGroupIdx * constInfo.mBaseSize;

    // 3. 安全判断
    if (currentGroupStart >= totalDataSize) {
        // 超出尾块
        basicBlockInfo.dealSeqCnt = 0;
    } else {
        // 还在有效范围内，计算剩余量
        uint32_t remaining = totalDataSize - currentGroupStart;
        basicBlockInfo.dealSeqCnt = (remaining < constInfo.mBaseSize) ? remaining : constInfo.mBaseSize;
    }
    return basicBlockInfo;
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline uint32_t CompressorV2Kernel<COMP, IS_FULL_LOAD>::GetLoopTimes()
{
    // 计算主循环次数
    uint32_t loopTimes = 0;
    BatchInfo batchInfo{};
    SkipInvalidBatch(batchInfo);
    for (; batchInfo.bIdx < constInfo.batchSize; ++loopTimes) {
        SkipOneLoop(batchInfo);
    }
    return loopTimes;
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::CalcSplitCoreInfo()
{
    // D方向的基本块数量
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;
    // 核的组数
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;
    // 每个核处理的d方向的索引
    constInfo.dIdx = constInfo.aiCoreIdx % constInfo.dBasicBlockNum;
    // 当前组id
    constInfo.curGroupIdx = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;
    constInfo.mGroupNum = constInfo.coreGroupNum;
    constInfo.mCurGroupIdx = constInfo.curGroupIdx;

    constInfo.mm1ResSize = (uint64_t)constInfo.mBaseSize * constInfo.headDim * constInfo.coreGroupNum;

    uint32_t coff = (uint32_t)COMP::coff;
    constInfo.mm1KvResSize = (uint64_t)constInfo.mBaseSize * constInfo.headDim * coff;
    constInfo.mm1ScoreResSize = (uint64_t)constInfo.mBaseSize * constInfo.headDim * coff;
    constInfo.vec1ResSize = (uint64_t)constInfo.mBaseSize * constInfo.headDim * constInfo.nSize;

    constInfo.dbSize = (uint64_t)constInfo.coreGroupNum * constInfo.mm1KvResSize;
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::InitWorkspace(__gm__ uint8_t *workspace)
{
    uint64_t offset = 0;
    uint64_t mm1KvResStartOffset = offset;
    // mm1KvResGm
    mm1KvResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             (uint64_t)constInfo.curGroupIdx * constInfo.mm1KvResSize * sizeof(MM1_OUT_T)));
    offset +=
        (uint64_t)constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1KvResSize * sizeof(MM1_OUT_T);

    uint64_t mm1ScoreResStartOffset = offset;
    // mm1ScoreResGm
    mm1ScoreResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             (uint64_t)constInfo.curGroupIdx * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T)));
    offset +=
        (uint64_t)constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T);

    Vec1InputKvGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + mm1KvResStartOffset));

    Vec1InputScoreGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + mm1ScoreResStartOffset));

    vec1KvCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);

    vec1ScoreCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::ComputeMm1(const RunInfo &info, bool isNeedExcute)
{
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG + info.cubeDbIdx);
    if (isNeedExcute) {
        blockCube_.ComputeMm1(info);
    }
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG + info.cubeDbIdx);
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::ComputeVec1(const Vec1RunInfo &info)
{
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_C1_V1_FLAG + info.c1v1DbIdx);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG2 + info.c1v1DbIdx);
    if constexpr (IS_FULL_LOAD) {
        blockVec_.ComputeVec1();
    } else {
        blockVec_.ComputeVec1(info);
    }
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_V1_C1_FLAG + info.c1v1DbIdx);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_V1_FLAG2 + (info.c1v1DbIdx + 1) % constInfo.dbWorkspaceRatio);
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::AllocEventID()
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

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::FreeEventID()
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

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline bool CompressorV2Kernel<COMP, IS_FULL_LOAD>::IsNeedExcuteC1(RunInfo info)
{
    // B超出范围则cube不执行
    return info.bStart < constInfo.batchSize;
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::CalcC1V1Params(RunInfo &info, Vec1RunInfo &vec1Info,
                                                                              BatchInfo &batchInfo, uint32_t loopIdx)
{
    vec1Info.bStart = batchInfo.bIdx;
    vec1Info.sStart = batchInfo.sIdx;
    vec1Info.c1v1DbIdx = (vec1Loop++ & (constInfo.dbWorkspaceRatio - 1));
    BasicBlockInfo basicBlockInfo = SkipOneLoop(batchInfo);
    info.cubeDbIdx = (cubeLoop++ & (constInfo.dbWorkspaceRatio - 1));
    info.dealSeqCnt = basicBlockInfo.dealSeqCnt;
    info.dealTcNum = basicBlockInfo.dealTcNum;
    info.bStart = basicBlockInfo.bIdx;
    info.sStart = basicBlockInfo.sIdx;
    info.kStartIdx = kStartIdx_;
    info.dealKSize = dealKSize_;
    info.hStart = hStart_;
    vec1Info.dealTcNum = basicBlockInfo.dealTcNum;
    vec1Info.dealScSize = basicBlockInfo.compressedTcNum;
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::CalcCubeParams(RunInfo &info)
{
    info.cubeDbIdx = (cubeLoop++ & (constInfo.dbWorkspaceRatio - 1));
    info.dealSeqCnt = constInfo.mEnd - constInfo.mStart;
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::ProcessNormalLoop()
{
    BatchInfo batchInfo{};

    RunInfo extraInfo[1];
    Vec1RunInfo vec1Info{};
    SkipInvalidBatch(batchInfo);
    for (uint32_t i = 0; i < loopTimes; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        CalcC1V1Params(extraInfo0, vec1Info, batchInfo, i);
        bool isNeedExcuteC1 = IsNeedExcuteC1(extraInfo0);

        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0, isNeedExcuteC1);
        } else {
            ComputeVec1(vec1Info);
        }
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::ProcessFullLoadLoop()
{
    RunInfo extraInfo[1];
    for (uint32_t i = 0; i < loopTimes; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        if ASCEND_IS_AIC {
            CalcCubeParams(extraInfo0);
            ComputeMm1(extraInfo0);
        } else {
            ComputeVec1(Vec1RunInfo{});
        }
    }
}

template <typename COMP, bool IS_FULL_LOAD>
__aicore__ inline void CompressorV2Kernel<COMP, IS_FULL_LOAD>::Process()
{
    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }
    AllocEventID();
    if constexpr (IS_FULL_LOAD) {
        ProcessFullLoadLoop();
    } else {
        ProcessNormalLoop();
    }
    FreeEventID();
}

} // namespace CompressorV2

#endif // COMPRESSOR_KERNEL_H
