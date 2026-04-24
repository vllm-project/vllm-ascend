/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_kernel_perf.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_PERF_H
#define COMPRESSOR_KERNEL_PERF_H

#include "compressor_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"
#include "compressor_comm.h"
#include "compressor_tools.h"
#if (__CCE_AICORE__ == 220)
#include "arch32/compressor_block_cube_perf.h"
#include "arch32/compressor_block_vec_perf.h"
#else
#include "arch35/compressor_block_cube.h"
#include "arch35/compressor_block_vec.h"
#endif

using namespace AscendC;

namespace Compressor {

struct CmpBlockInfo {
    __aicore__ inline CmpBlockInfo() {};
    __aicore__ inline CmpBlockInfo(uint32_t bIdx, uint32_t sIdx, bool needReset = false) : bIdx(bIdx), sIdx(sIdx), needReset(needReset) {};

    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint32_t bStartPos = 0U;
    bool needReset = false;
    bool isFirst = true;

    uint32_t headSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailSeqCnt = 0U;
    bool isCompress = 0U;
};

struct BasicBlockInfo {
    uint32_t dealTcNum = 0;
    uint32_t compressedTcNum = 0;
    uint32_t dealSeqCnt = 0;
    uint32_t preDealSeqCnt = 0;
};

template <typename COMP>
class CompressorKernelPerf {
public:
    __aicore__ inline CompressorKernelPerf(TPipe* pipe, const optiling::CompressorTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *kvBlockTable,
        __gm__ uint8_t *scoreBlockTable,
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
    __aicore__ inline uint32_t CalcSIdxOfLastTc();
    // 获取基本块数量
    __aicore__ inline void CalcCmpBlockInfo(CmpBlockInfo &cmpBlockInfo);
    __aicore__ inline void SkipInvalidBatch(CmpBlockInfo &cmpBlockInfo, uint32_t maxBatchSize);
    __aicore__ inline uint32_t GetNextCmpSeqCnt(CmpBlockInfo &cmpBlockInfo);
    __aicore__ inline void AcceptUpdate(CmpBlockInfo &cmpBlockInfo);
    __aicore__ inline void UpdateBasicBlockInfo(BasicBlockInfo &basicBlockInfo, CmpBlockInfo &cmpBlockInfo, bool isLeft);
    __aicore__ inline BasicBlockInfo SkipOneBasicBlock(CmpBlockInfo &rightCmpBlockInfo, CmpBlockInfo &leftCmpBlockInfo);
    __aicore__ inline uint32_t GetBasicBlockNum();
    // 计算当前核起始位置
    __aicore__ inline void CalcCurCoreStartIdx(CmpBlockInfo &rightCmpBlockInfo, CmpBlockInfo &leftCmpBlockInfo);
    // 计算分核基本信息
    __aicore__ inline void CalcSplitCoreInfo();

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const Vec2RunInfo &info);

    __aicore__ inline bool IsNeedExcuteC1V1(uint32_t curBasicBlockIdx);
    __aicore__ inline bool IsNeedSyncAll(uint32_t curBasicBlockIdx);
    __aicore__ inline void CalcC1V1Params(RunInfo &info, uint32_t curBasicBlockIdx, CmpBlockInfo &rightCmpBlockInfo, CmpBlockInfo &leftCmpBlockInfo);
    __aicore__ inline void UpdateVec2Info(Vec2RunInfo &vec2Info, uint32_t curBasicBlockIdx, const RunInfo &info);
    __aicore__ inline bool IsNeedExcuteV2(Vec2RunInfo &vec2Info);

    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;

    // 常量
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 6;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 7;
    static constexpr uint32_t dbWorkspaceRatio = 1;

    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::CompressorTilingData* __restrict tilingData_;
    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> preMm1ResGm;
    GlobalTensor<MM1_OUT_T> curMm1ResGm;
    GlobalTensor<VEC1_OUT_T> vec1ResGm;
    GlobalTensor<VEC1_OUT_T> vec2InputGm;
    // ================================Task Info====================================
    CompressorTools<COMP> tools_;
    ConstInfo constInfo{};
    uint32_t aiCoreIdx = 0;

    // ==============================Service Define==============================
    CompressorBlockCubePerf<COMP> blockCube_;
    CompressorBlockVectorPerf<COMP> blockVec_;

    uint32_t allCompressedTcNum_ = 0;
    uint32_t curCompressedTcNum_ = 0;
    uint32_t accDealSize = 0;
};

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *kvBlockTable,
        __gm__ uint8_t *scoreBlockTable,
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
        uint32_t bSeqUsed = tools_.GetSeqUsed(constInfo.batchSize - 1);
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
    constInfo.sIdxOfLastTc = CalcSIdxOfLastTc();
    // 1. 计算基本块数量
    constInfo.tcBasicBlockNum = GetBasicBlockNum();
    // 2. 计算head_dim的切分大小, 构建ConstInfo的其他信息
    CalcSplitCoreInfo();
    // 3. 初始化workspace
    InitWorkspace(workspace);
    // 4. 初始化block层
    if ASCEND_IS_AIC {
#if __CCE_AICORE__ == 310
        blockCube_.InitParams(constInfo);
#else
        blockCube_.InitParams(constInfo, tools_);
#endif
        blockCube_.Init(x, wKv, wGate, kvState, scoreState, ape, normWeight, ropeSin, ropeCos, 
            kvBlockTable, scoreBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockCube_.InitBuffers(pipe_);
#if __CCE_AICORE__ == 310
#else
        blockCube_.InitGlobalBuffers(preMm1ResGm, curMm1ResGm);
#endif
    } else {
        blockVec_.InitParams(constInfo, tools_);
        blockVec_.Init(x, wKv, wGate, kvState, scoreState, ape, normWeight, ropeSin, ropeCos, kvBlockTable, scoreBlockTable, 
                        cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockVec_.InitBuffers(pipe_);
#if __CCE_AICORE__ == 310
        blockVec_.InitVec1GlobalTensor(preMm1ResGm, curMm1ResGm, vec1ResGm, vec2InputGm);
#else 
        blockVec_.InitVec1GlobalTensor(preMm1ResGm, curMm1ResGm, vec1ResGm, vec2InputGm);
#endif
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::InitTilingData() {
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.hSize = tilingData_->baseParams.hiddenSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.ropeHeadDim = tilingData_->baseParams.ropeHeadDim;
    constInfo.normEps = tilingData_->baseParams.normEps;
    constInfo.reciprocalD = tilingData_->baseParams.reciprocalD;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;
    
    constInfo.blockNum = tilingData_->pageAttentionParams.blockNum;
    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    constInfo.preMm1ResSize = tilingData_->workspaceParams.preMm1ResSize;
    constInfo.curMm1ResSize = tilingData_->workspaceParams.curMm1ResSize;
    constInfo.nSize =  tilingData_->baseParams.nSize;
    constInfo.vec1ResSize = tilingData_->workspaceParams.vec1ResSize;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernelPerf<COMP>::CalcSIdxOfLastTc()
{
    // 在Init中已确保倒数第一个batch的bSeqUsed不会是0
    uint32_t cmpRatio = constInfo.cmpRatio;
    uint32_t bSeqUsed = tools_.GetSeqUsed(constInfo.batchSize - 1);
    uint32_t bStartPos = tools_.GetStartPos(constInfo.batchSize - 1);

    // 1.先假设start_pos不再尾块、并且尾块未填满
    uint32_t lastSeqCnt = (bStartPos + bSeqUsed) % cmpRatio;
    // 2.处理结束点正好填满cmp block的情况
    if (lastSeqCnt == 0) {
        lastSeqCnt = cmpRatio;
    }
    // 3.处理start_pos在最后一个压缩块的情况
    if (bSeqUsed < lastSeqCnt) {
        lastSeqCnt = bSeqUsed;
    }

    return (bSeqUsed - lastSeqCnt);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::CalcCmpBlockInfo(CmpBlockInfo &cmpBlockInfo)
{
    uint32_t cmpRatio = constInfo.cmpRatio;
    // 计算压缩块信息
    cmpBlockInfo.headSeqCnt = (cmpBlockInfo.bStartPos + cmpBlockInfo.sIdx) % cmpRatio;
    cmpBlockInfo.validSeqCnt = cmpBlockInfo.bSeqUsed - cmpBlockInfo.sIdx;
    if (cmpBlockInfo.headSeqCnt + cmpBlockInfo.validSeqCnt > cmpRatio) {
        cmpBlockInfo.validSeqCnt = cmpRatio - cmpBlockInfo.headSeqCnt;
        cmpBlockInfo.tailSeqCnt = 0;
    } else {
        cmpBlockInfo.tailSeqCnt = cmpRatio - (cmpBlockInfo.headSeqCnt + cmpBlockInfo.validSeqCnt);
    }
    cmpBlockInfo.isCompress = (cmpBlockInfo.tailSeqCnt == 0);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::SkipInvalidBatch(CmpBlockInfo &cmpBlockInfo, uint32_t maxBatchSize)
{
    for (; cmpBlockInfo.bIdx < maxBatchSize; ++cmpBlockInfo.bIdx) {
        cmpBlockInfo.bSeqUsed = tools_.GetSeqUsed(cmpBlockInfo.bIdx);
        if (cmpBlockInfo.bSeqUsed > 0) {
            break;
        }
    }
    if (cmpBlockInfo.bIdx < maxBatchSize) {
        cmpBlockInfo.bStartPos = tools_.GetStartPos(cmpBlockInfo.bIdx);
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernelPerf<COMP>::GetNextCmpSeqCnt(CmpBlockInfo &cmpBlockInfo)
{
    if (cmpBlockInfo.isFirst) {
        cmpBlockInfo.isFirst = false;
        SkipInvalidBatch(cmpBlockInfo, constInfo.batchSize);
    }

    if (cmpBlockInfo.bIdx >= constInfo.batchSize) {
        return 0;
    }

    CalcCmpBlockInfo(cmpBlockInfo);
    return cmpBlockInfo.validSeqCnt;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::AcceptUpdate(CmpBlockInfo &cmpBlockInfo)
{
    // 更新sIdx和bIdx、以及与bIdx相关的bStartPos和bSeqUsed
    cmpBlockInfo.sIdx += cmpBlockInfo.validSeqCnt;
    if (cmpBlockInfo.sIdx == cmpBlockInfo.bSeqUsed) {
        cmpBlockInfo.sIdx = 0;
        cmpBlockInfo.bIdx++;
        if (cmpBlockInfo.needReset && cmpBlockInfo.bIdx == constInfo.batchSize) {
            cmpBlockInfo.bIdx = 0;
            cmpBlockInfo.needReset = false;
        }
        SkipInvalidBatch(cmpBlockInfo, constInfo.batchSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::UpdateBasicBlockInfo(BasicBlockInfo &basicBlockInfo,
    CmpBlockInfo &cmpBlockInfo, bool isLeft)
{
    if (isLeft) {
        basicBlockInfo.preDealSeqCnt += cmpBlockInfo.validSeqCnt;
    } else {
        basicBlockInfo.dealSeqCnt += cmpBlockInfo.validSeqCnt;
        basicBlockInfo.dealTcNum++;
        if (cmpBlockInfo.tailSeqCnt == 0) {
            basicBlockInfo.compressedTcNum++;
        }
    }
}

template <typename COMP>
__aicore__ inline BasicBlockInfo CompressorKernelPerf<COMP>::SkipOneBasicBlock(
    CmpBlockInfo &rightCmpBlockInfo, CmpBlockInfo &leftCmpBlockInfo)
{
    BasicBlockInfo basicBlockInfo{};
    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t leftFinishSeqCnt = 0;
        uint32_t rightFinishSeqCnt = 0;
        for (;;) {
            uint32_t rightSeqCnt = GetNextCmpSeqCnt(rightCmpBlockInfo);
            if (rightSeqCnt == 0 || (rightFinishSeqCnt + rightSeqCnt > constInfo.mBaseSize)) {
                break;
            }
            uint32_t leftSeqCnt = GetNextCmpSeqCnt(leftCmpBlockInfo);
            if ((leftSeqCnt == 0) || (leftFinishSeqCnt + leftSeqCnt > constInfo.mBaseSize)) {
                break;
            }
            // PRINTF("rightSeqCnt:%d leftSeqCnt:%d\n", rightSeqCnt, leftSeqCnt);
            rightFinishSeqCnt += rightSeqCnt;
            leftFinishSeqCnt += leftSeqCnt;

            // 收集基本块信息
            UpdateBasicBlockInfo(basicBlockInfo, leftCmpBlockInfo, true);
            UpdateBasicBlockInfo(basicBlockInfo, rightCmpBlockInfo, false);

            AcceptUpdate(rightCmpBlockInfo);
            // PRINTF("Right cmpBlockInfo.bIdx:%d cmpBlockInfo.sIdx:%d cmpBlockInfo.bSeqUsed:%d cmpBlockInfo.bStartPos:%d\n",
            //     rightCmpBlockInfo.bIdx, rightCmpBlockInfo.sIdx, rightCmpBlockInfo.bSeqUsed, rightCmpBlockInfo.bStartPos);
            AcceptUpdate(leftCmpBlockInfo);
            // PRINTF("Left cmpBlockInfo.bIdx:%d cmpBlockInfo.sIdx:%d cmpBlockInfo.bSeqUsed:%d cmpBlockInfo.bStartPos:%d\n",
            //     leftCmpBlockInfo.bIdx, leftCmpBlockInfo.sIdx, leftCmpBlockInfo.bSeqUsed, leftCmpBlockInfo.bStartPos);
        }
    } else {
        uint32_t rightFinishSeqCnt = 0;
        for (;;) {
            uint32_t rightSeqCnt = GetNextCmpSeqCnt(rightCmpBlockInfo);
            if (rightSeqCnt == 0 || (rightFinishSeqCnt + rightSeqCnt > constInfo.mBaseSize)) {
                break;
            }
            rightFinishSeqCnt += rightSeqCnt;

            // 收集基本块信息
            UpdateBasicBlockInfo(basicBlockInfo, rightCmpBlockInfo, false);

            AcceptUpdate(rightCmpBlockInfo);
        }
    }

    return basicBlockInfo;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernelPerf<COMP>::GetBasicBlockNum()
{
    // 计算基本块数量
    uint32_t basicBlockNum = 0;
    CmpBlockInfo leftCmpBlockInfo(constInfo.bIdxOfLastTc, constInfo.sIdxOfLastTc, true);
    CmpBlockInfo rightCmpBlockInfo(0, 0, false);

    for (;rightCmpBlockInfo.bIdx < constInfo.batchSize; ++basicBlockNum) {
        BasicBlockInfo basicBlockInfo = SkipOneBasicBlock(rightCmpBlockInfo, leftCmpBlockInfo);
        allCompressedTcNum_ += basicBlockInfo.compressedTcNum;
    }
    return basicBlockNum;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::CalcCurCoreStartIdx(
    CmpBlockInfo &rightCmpBlockInfo, CmpBlockInfo &leftCmpBlockInfo)
{
    uint32_t basicBlockNum = constInfo.curGroupIdx * constInfo.singleCoreDealTcBasicNum;
    for (; basicBlockNum > 0; --basicBlockNum) {
        BasicBlockInfo basicBlockInfo = SkipOneBasicBlock(rightCmpBlockInfo, leftCmpBlockInfo);
        curCompressedTcNum_ += basicBlockInfo.compressedTcNum;
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::CalcSplitCoreInfo()
{
    // 计算D的切分大小
    constInfo.dBaseSize = 64; // 默认按照64切分
    uint32_t maxEnableCoreNum = constInfo.tcBasicBlockNum * (constInfo.headDim / constInfo.dBaseSize);
    uint32_t minEnableCoreNum = 16;
    if (maxEnableCoreNum < minEnableCoreNum) {
        // headDim=128时, dBaseSize=8; headDim=512时, dBaseSize=32
        constInfo.dBaseSize = constInfo.headDim / minEnableCoreNum;
    }

    // D方向的基本块数量
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;
    // 核的组数
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;
    // 每个核处理的d方向的索引
    constInfo.dIdx = (constInfo.aiCoreIdx % constInfo.dBasicBlockNum) * constInfo.dBaseSize;
    // 当前组id
    constInfo.curGroupIdx = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;

    // 单组核在T方向处理的最大基本块数量
    constInfo.singleCoreDealTcBasicNum = (constInfo.tcBasicBlockNum + constInfo.coreGroupNum - 1) / constInfo.coreGroupNum;
    // 尾组所在id
    constInfo.tailGroupIdx = (constInfo.tcBasicBlockNum - 1) / constInfo.singleCoreDealTcBasicNum;
    // 尾组处理基本块数量
    constInfo.tailBasicBlockNum = constInfo.tcBasicBlockNum - constInfo.tailGroupIdx * constInfo.singleCoreDealTcBasicNum;

    // 计算当前核需要处理的基本块个数
    if (constInfo.curGroupIdx < constInfo.tailGroupIdx) {
        constInfo.realDealBasicBlockNum = constInfo.singleCoreDealTcBasicNum;
    } else if (constInfo.curGroupIdx > constInfo.tailGroupIdx) {
        constInfo.realDealBasicBlockNum = 0;
    } else {
        constInfo.realDealBasicBlockNum = constInfo.tailBasicBlockNum;
    }
    // PRINTF("constInfo.dBaseSize:%d dBasicBlockNum:%d coreGroupNum:%d dIdx:%d curGroupIdx:%d singleCoreDealTcBasicNum:%d tailGroupIdx:%d tailBasicBlockNum:%d realDealBasicBlockNum:%d\n",
    //     constInfo.dBaseSize, constInfo.dBasicBlockNum, constInfo.coreGroupNum, constInfo.dIdx, constInfo.curGroupIdx, constInfo.singleCoreDealTcBasicNum, constInfo.tailGroupIdx, constInfo.tailBasicBlockNum, constInfo.realDealBasicBlockNum);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::InitWorkspace(__gm__ uint8_t *workspace) {
    uint64_t offset = 0;
    // preMm1ResGm
    preMm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.aiCoreIdx * dbWorkspaceRatio * constInfo.preMm1ResSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.preMm1ResSize * sizeof(MM1_OUT_T);

    // curMm1ResGm
    curMm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.aiCoreIdx * dbWorkspaceRatio * constInfo.curMm1ResSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.curMm1ResSize * sizeof(MM1_OUT_T);

    uint64_t beforeVecOffset = offset;

    // vec1Res 
    vec1ResGm.SetGlobalBuffer(
        (__gm__ VEC1_OUT_T *)(workspace + offset + (constInfo.dIdx + (constInfo.aiCoreIdx / constInfo.dBasicBlockNum) * dbWorkspaceRatio * constInfo.vec1ResSize * constInfo.dBasicBlockNum) * sizeof(VEC1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.vec1ResSize * sizeof(VEC1_OUT_T);
    // vec2Input
    vec2InputGm.SetGlobalBuffer(
        (__gm__ VEC1_OUT_T *)(workspace + beforeVecOffset +  (constInfo.aiCoreIdx / constInfo.dBasicBlockNum) * dbWorkspaceRatio * constInfo.vec1ResSize * constInfo.dBasicBlockNum * sizeof(VEC1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.vec1ResSize * sizeof(VEC1_OUT_T);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::ComputeMm1(const RunInfo &info) {
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG);
    blockCube_.ComputeMm1(info);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::ComputeVec1(const RunInfo &info) {
#if (__CCE_AICORE__ == 220)
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_C1_V1_FLAG);
#else
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_V>(SYNC_C1_V1_FLAG);
#endif
    blockVec_.ComputeVec1(info);
#if (__CCE_AICORE__ == 220)
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C1_FLAG);
#else
    CrossCoreSetFlag<SYNC_MODE2, PIPE_V>(SYNC_V1_C1_FLAG);
#endif
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::ComputeVec2(const Vec2RunInfo &info) {
    blockVec_.ComputeVec2(info);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::AllocEventID()
{
    if ASCEND_IS_AIC {
        blockCube_.AllocEventID(pipe_);
    } else {
        blockVec_.AllocEventID();
#if (__CCE_AICORE__ == 220)
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C1_FLAG);
#else
        CrossCoreSetFlag<SYNC_MODE2, PIPE_V>(SYNC_V1_C1_FLAG);
#endif
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::FreeEventID()
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG);
        blockCube_.FreeEventID(pipe_);
    } else {
        blockVec_.FreeEventID();
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernelPerf<COMP>::IsNeedExcuteC1V1(uint32_t curBasicBlockIdx)
{
    return (curBasicBlockIdx < constInfo.realDealBasicBlockNum);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::CalcC1V1Params(RunInfo &info,
    uint32_t curBasicBlockIdx, CmpBlockInfo &rightCmpBlockInfo, CmpBlockInfo &leftCmpBlockInfo)
{
    // 计算v1写出偏移
    if (curBasicBlockIdx % constInfo.nSize == 0) {
        info.vec1ResOffset = 0;
        accDealSize = 0;
    } else {
        info.vec1ResOffset = accDealSize * constInfo.headDim;
    }

    info.preFirstSeqCnt = GetNextCmpSeqCnt(leftCmpBlockInfo);
    info.preBStart = leftCmpBlockInfo.bIdx;
    info.preSStart = leftCmpBlockInfo.sIdx;
    info.bStart = rightCmpBlockInfo.bIdx;
    info.sStart = rightCmpBlockInfo.sIdx;
    BasicBlockInfo basicBlockInfo = SkipOneBasicBlock(rightCmpBlockInfo, leftCmpBlockInfo);
    info.dealTcNum = basicBlockInfo.dealTcNum;
    info.dealScSize = basicBlockInfo.compressedTcNum;
    info.dealSeqCnt = basicBlockInfo.dealSeqCnt;
    info.preDealSeqCnt = basicBlockInfo.preDealSeqCnt;
    accDealSize += info.dealScSize;
}

template <typename COMP>
__aicore__ inline bool CompressorKernelPerf<COMP>::IsNeedExcuteV2(Vec2RunInfo &vec2Info)
{
    return (vec2Info.dealScSize > 0);
}

template <typename COMP>
__aicore__ inline bool CompressorKernelPerf<COMP>::IsNeedSyncAll(uint32_t curBasicBlockIdx)
{
    if (allCompressedTcNum_ == 0) {
        return false;
    }

    uint32_t cnt = curBasicBlockIdx + 1;
    if ((cnt == constInfo.singleCoreDealTcBasicNum) || (cnt % constInfo.nSize == 0)) {
        return true;
    }
    return false;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::UpdateVec2Info(
    Vec2RunInfo &vec2Info, uint32_t curBasicBlockIdx, const RunInfo &info)
{
    // nSize轮起始先重置v2Info信息
    if (curBasicBlockIdx % constInfo.nSize == 0) {
        vec2Info.bStart = info.bStart;
        vec2Info.sStart = info.sStart;
        // 将sStart转成bCompressedId
        uint32_t startPos = tools_.GetStartPos(info.bStart);
        vec2Info.bCompressedId = (startPos + info.sStart) / constInfo.cmpRatio - startPos / constInfo.cmpRatio;

        vec2Info.dealScSize = 0;
    }
    vec2Info.dealScSize += info.dealScSize;
    // TODO 应该是加上上一轮的
    vec2Info.compressedId += info.dealScSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::Process()
{
    // 所有batch的有效序列都为0时, 直接退出
    if (constInfo.batchSize == 0) {
        return;
    }
    AllocEventID();

    CmpBlockInfo leftCmpBlockInfo(constInfo.bIdxOfLastTc, constInfo.sIdxOfLastTc, true);
    CmpBlockInfo rightCmpBlockInfo(0, 0, false);
    CalcCurCoreStartIdx(rightCmpBlockInfo, leftCmpBlockInfo);
    // PRINTF("allCompressedTcNum_:%d curCompressedTcNum_:%d\n", allCompressedTcNum_, curCompressedTcNum_);

    RunInfo extraInfo[1];
    Vec2RunInfo vec2Info{};
    for (uint32_t i = 0; i < constInfo.singleCoreDealTcBasicNum; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        bool isNeedExcuteC1V1 = IsNeedExcuteC1V1(i);
        if (isNeedExcuteC1V1) {
            CalcC1V1Params(extraInfo0, i, rightCmpBlockInfo, leftCmpBlockInfo);
        }

        if ASCEND_IS_AIC {
            if (isNeedExcuteC1V1) {
                ComputeMm1(extraInfo0);
            }
        } else {
            if (isNeedExcuteC1V1) {
                ComputeVec1(extraInfo0);
                UpdateVec2Info(vec2Info, i, extraInfo0);
            }

            if (IsNeedSyncAll(i)) {
                SyncAll();
                if (IsNeedExcuteV2(vec2Info)) {
                    ComputeVec2(vec2Info);
                }
                SyncAll();
            }
        }

    }
    FreeEventID();
}

} // namespace Compressor

#endif // COMPRESSOR_KERNEL_PERF_H