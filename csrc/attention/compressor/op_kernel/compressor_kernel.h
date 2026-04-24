/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "compressor_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"
#if (__CCE_AICORE__ == 220)
#include "arch32/compressor_block_cube.h"
#include "arch32/compressor_block_vec.h"
#else
#include "arch35/compressor_block_cube.h"
#include "arch35/compressor_block_vec.h"
#endif

using namespace AscendC;

namespace Compressor {

template <typename COMP>
class CompressorKernel {
public:
    __aicore__ inline CompressorKernel(TPipe* pipe, const optiling::CompressorTilingData* __restrict tilingData)
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
    __aicore__ inline uint32_t CalcTcSize();
    // ================================Process functions================================
    __aicore__ inline void GetSeqLength(uint32_t index);
    __aicore__ inline void GetStartPos(uint32_t index);
    __aicore__ inline void GetCurCoreStartIdx();
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline uint32_t GetRemainTcNum(uint32_t sStart, uint32_t headSize, uint32_t cmpRatio, uint32_t seqLength);
    __aicore__ inline uint32_t GetTcHeadSize(uint32_t curStartPos, uint32_t cmpRatio, uint32_t seqLength);
    __aicore__ inline void CalcParams(RunInfo &info);
    __aicore__ inline void InitTilingData();
    __aicore__ inline bool IsNeedExcute(uint32_t curBasicBlockIdx);
    __aicore__ inline uint32_t GetStartIdx();
    __aicore__ inline uint32_t GetEndIdx();
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeVec1(const RunInfo &info);
    __aicore__ inline void ComputeVec2(const RunInfo &info);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::CompressorTilingData* __restrict tilingData_;
    // ================================Task Info====================================
    ConstInfo constInfo{};

    uint32_t tcStart = 0;

    uint32_t accSeqLength = 0;
    uint32_t curActSeqLength = 0;
    uint32_t lastActSeqLength = 0;
    uint32_t curStartPos = 0;
    uint32_t preActSeqIdx = 0;
    uint32_t preStartPosIdx = 0;

    uint32_t curBStart = 0;
    uint32_t curBEnd = 0;
    uint32_t curSStart = 0;
    uint32_t curSEnd = 0;
    uint32_t curScEnd = 0;

    uint32_t aiCoreIdx = 0;

    bool isExistSeqUsed = false;
    bool isExistStartPos = false;

    // 常量
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 6;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 7;

    // ==============================Service Define==============================
    CompressorBlockCube<COMP> blockCube_;
    CompressorBlockVector<COMP> blockVec_;
    static constexpr uint32_t dbWorkspaceRatio = 1;

    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;

    // GM
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> wkvGm_;
    GlobalTensor<X_T> wgateGm_;
    GlobalTensor<int32_t> kvStateGm_;
    GlobalTensor<int32_t> scoreStateGm_;
    GlobalTensor<int32_t> apeGm_;
    GlobalTensor<X_T> ropeSinGm_;
    GlobalTensor<X_T> ropeCosGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<int32_t> kvBlockTableGm_;
    GlobalTensor<int32_t> scoreBlockTableGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;

    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> preMm1ResGm;
    GlobalTensor<MM1_OUT_T> curMm1ResGm;
    GlobalTensor<VEC1_OUT_T> vec1ResGm;
    GlobalTensor<VEC1_OUT_T> vec2InputGm;
};

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Init(
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
        __gm__ uint8_t *workspace) {
    if ASCEND_IS_AIV {
        constInfo.aiCoreIdx = GetBlockIdx() / 2;
    } else {
        constInfo.aiCoreIdx = GetBlockIdx();
    }

    // GM Init
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    wkvGm_.SetGlobalBuffer((__gm__ X_T *)wKv);
    wgateGm_.SetGlobalBuffer((__gm__ X_T *)wGate);
    kvStateGm_.SetGlobalBuffer((__gm__ int32_t *)kvState);
    scoreStateGm_.SetGlobalBuffer((__gm__ int32_t *)scoreState);
    apeGm_.SetGlobalBuffer((__gm__ int32_t *)ape);
    ropeSinGm_.SetGlobalBuffer((__gm__ X_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ X_T *)ropeCos);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    kvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)kvBlockTable);
    scoreBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)scoreBlockTable);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }

    InitTilingData();

    // 初始化 curActSeqLength、start_pos
    if (isExistSeqUsed) {
        curActSeqLength = sequsedGm_.GetValue(0);
        lastActSeqLength = sequsedGm_.GetValue(constInfo.batchSize - 1);
    } else if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        curActSeqLength = cuSeqlensGm_.GetValue(1);
        accSeqLength = curActSeqLength;
        lastActSeqLength = cuSeqlensGm_.GetValue(constInfo.batchSize) - cuSeqlensGm_.GetValue(constInfo.batchSize - 1);
    } else {
        curActSeqLength = constInfo.sSize;
        lastActSeqLength = constInfo.sSize;
    }

    if (isExistStartPos) {
        curStartPos = startPosGm_.GetValue(0);
    }

    // 计算分核基本信息
    constInfo.tcSize = CalcTcSize();
    constInfo.tcBaseSize = constInfo.mBaseSize / constInfo.cmpRatio;
    constInfo.tcBasicBlockNum = (constInfo.tcSize + constInfo.tcBaseSize - 1) / constInfo.tcBaseSize;                       // TC方向的基本块
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;                                                     // D方向的基本块
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;                                              // 核分为多少组
    constInfo.singleCoreDealTcBasicNum = (constInfo.tcBasicBlockNum + constInfo.coreGroupNum - 1) / constInfo.coreGroupNum; // 处理的最大基本块数量
    constInfo.dIdx = (constInfo.aiCoreIdx % constInfo.dBasicBlockNum) * constInfo.dBaseSize;                    // 每个核处理的d方向的索引

    // 计算当前核需要处理多少基本块
    constInfo.curGroupIdx = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;                                                                               // 当前组id
    constInfo.tailGroupIdx = (constInfo.tcBasicBlockNum + constInfo.singleCoreDealTcBasicNum - 1) / constInfo.singleCoreDealTcBasicNum - 1;                                                            // 尾组所在id
    constInfo.tailBasicBlockNum = constInfo.tcBasicBlockNum % constInfo.singleCoreDealTcBasicNum == 0 ?
                                constInfo.singleCoreDealTcBasicNum : constInfo.tcBasicBlockNum % constInfo.singleCoreDealTcBasicNum;                     // 尾组处理基本块数量
    constInfo.realDealBasicBlockNum = constInfo.curGroupIdx < constInfo.tailGroupIdx ? constInfo.singleCoreDealTcBasicNum : constInfo.tailBasicBlockNum; // 当前组实际处理的基本块
    if (constInfo.curGroupIdx > constInfo.tailGroupIdx) {
        constInfo.realDealBasicBlockNum = 0;
    }
    InitWorkspace(workspace);
    if ASCEND_IS_AIC {
        blockCube_.InitParams(constInfo);
        blockCube_.Init(x, wKv, wGate, kvState, scoreState, ape, normWeight, ropeSin, ropeCos, 
            kvBlockTable, scoreBlockTable, cuSeqlens, seqUsed, startPos, cmpKvOut);
        blockCube_.InitBuffers(pipe_);
#if __CCE_AICORE__ == 310
#else
        blockCube_.InitGlobalBuffers(preMm1ResGm, curMm1ResGm);
#endif
    } else {
        blockVec_.InitParams(constInfo);
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
__aicore__ inline void CompressorKernel<COMP>::InitTilingData() {
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.dBaseSize = tilingData_->innerSplitParams.dBaseSize;
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
__aicore__ inline void CompressorKernel<COMP>::InitWorkspace(__gm__ uint8_t *workspace) {
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
__aicore__ inline uint32_t CompressorKernel<COMP>::CalcTcSize() {
    uint32_t totalBasicNum = 0;

    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        GetStartPos(bIdx);
        GetSeqLength(bIdx);
        totalBasicNum += GetBasicNum();
    }
    
    return totalBasicNum;
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::GetSeqLength(uint32_t bIdx) {
    if (preActSeqIdx != bIdx) {
        preActSeqIdx = bIdx;
        if (isExistSeqUsed) {
            curActSeqLength = sequsedGm_.GetValue(bIdx);
        } else {
            if constexpr (COMP::xLayout == X_LAYOUT::TH) {
                if (bIdx == 0) {
                    accSeqLength = cuSeqlensGm_.GetValue(bIdx + 1);
                    curActSeqLength = accSeqLength;
                } else {
                    uint32_t tmpSeqLength = accSeqLength;
                    accSeqLength = cuSeqlensGm_.GetValue(bIdx + 1);
                    curActSeqLength = accSeqLength - tmpSeqLength;
                }
            } else {
                curActSeqLength = constInfo.sSize;
            }
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::GetStartPos(uint32_t bIdx) {
    if (!isExistStartPos) {
        curStartPos = 0;
    } else if (preStartPosIdx != bIdx) {
        curStartPos = startPosGm_.GetValue(bIdx);
        preStartPosIdx = bIdx;
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::GetCurCoreStartIdx() {
    // 获取当前核开始索引

    uint32_t totalBasicNum = 0;
    // 在当前batch的seq开始索引位置
    uint32_t startIdx = 0;
    // Tc的开始位置
    tcStart = (constInfo.aiCoreIdx / constInfo.dBasicBlockNum) * constInfo.tcBaseSize * constInfo.singleCoreDealTcBasicNum;
    if (tcStart >= constInfo.tcSize) {
        tcStart = constInfo.tcSize;
        curBEnd = constInfo.batchSize;
        return;
    }
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; ++bIdx) {
        if (totalBasicNum == tcStart) {
            curBEnd = bIdx;
            curSEnd = 0;
            curScEnd = 0;
            return;
        }
        GetStartPos(bIdx);
        GetSeqLength(bIdx);

        // 加上头块，若有
        uint32_t curBasicNum = 0;
        uint32_t headSize = GetTcHeadSize(curStartPos, constInfo.cmpRatio, curActSeqLength);
        curBasicNum = headSize > 0 ? curBasicNum + 1 : curBasicNum;
        // 加上中间整块及尾块
        curBasicNum += (curActSeqLength - headSize + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
        if (totalBasicNum + curBasicNum > tcStart) {
            uint32_t curBasicNumStart = tcStart - totalBasicNum;
            if (curBasicNumStart > 0 && headSize > 0) {
                startIdx = headSize + (curBasicNumStart - 1) * constInfo.cmpRatio;
            } else {
                startIdx = curBasicNumStart * constInfo.cmpRatio;
            }
            curBEnd = bIdx;
            curSEnd = startIdx;
            curScEnd = curBasicNumStart;
            return;
        }
        totalBasicNum += curBasicNum;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetBasicNum() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos % constInfo.cmpRatio != 0) {
        headSize = constInfo.cmpRatio - curStartPos % constInfo.cmpRatio;
        headSize = headSize > curActSeqLength ? 0 : headSize;
        curBasicNum = headSize > 0 ? curBasicNum + 1 : curBasicNum;
    }
    // 加上中间整块及尾块
    curBasicNum += (curActSeqLength - headSize + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetTcHeadSize(uint32_t curStartPos, uint32_t cmpRatio, uint32_t seqLength) {
    // 计算头块大小
    uint32_t headSize = 0;
    if (curStartPos % cmpRatio != 0) {
        headSize = (cmpRatio - curStartPos % cmpRatio);
        headSize = headSize > seqLength ? 0 : headSize;     // 处理seq不足head大小的情况
    }

    return headSize;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernel<COMP>::GetRemainTcNum(uint32_t sStart, uint32_t headSize, uint32_t cmpRatio, uint32_t seqLength) {
    // 计算当前batch剩余的Tc块数量
    uint32_t curRemainTcNum = 0;
    if (sStart == 0) {
        curRemainTcNum = (seqLength - headSize + cmpRatio - 1) / cmpRatio;
        curRemainTcNum = headSize == 0 ? curRemainTcNum : curRemainTcNum + 1;
    } else {
        curRemainTcNum = (seqLength - sStart + cmpRatio - 1) / cmpRatio;
    }

    return curRemainTcNum;
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::CalcParams(RunInfo &info) {
    if (curBEnd == constInfo.batchSize - 1 && curSEnd == lastActSeqLength) {
        return;
    }

    curBStart = curBEnd;
    curSStart = curSEnd;
    info.scStart = curScEnd;

    // sEnd到了seq末尾，切换到下一个batch
    GetSeqLength(curBStart);
    // 跳batch处理，batch=0且seq=0时不跳
    if (curSStart == curActSeqLength && (curBStart != 0 || curActSeqLength != 0)) {
        curBStart++;
        curSStart = 0;
        info.scStart = 0;
    }

    info.bStart = curBStart;
    info.sStart = curSStart;
    
    // 1. 计算本次需要处理的Tc块数量
    uint32_t dealTcNum = constInfo.tcBaseSize + tcStart <= constInfo.tcSize ? constInfo.tcBaseSize : constInfo.tcSize - tcStart;
    info.dealTcNum = dealTcNum;
    tcStart += dealTcNum;           // 更新全局tc块起始偏移，下次使用
    uint32_t accBasicNum = 0;
    info.dealScSize = 0;

    uint32_t tcNumCount = dealTcNum;
    
    for (uint32_t bIdx = curBStart; bIdx < constInfo.batchSize; ++bIdx) {
        curBEnd = bIdx;
        info.bEnd = curBEnd;

        GetSeqLength(bIdx);
        GetStartPos(bIdx);
        uint32_t headSize = GetTcHeadSize(curStartPos, constInfo.cmpRatio, curActSeqLength);
        uint32_t curBatchTcNum = GetRemainTcNum(curSStart, headSize, constInfo.cmpRatio, curActSeqLength);  // 当前batch需要处理的Tc块
        uint32_t curDealTcNum = (tcNumCount >= curBatchTcNum) ? curBatchTcNum : tcNumCount;         // 本次能处理多少个Tc块

        tcNumCount -= curDealTcNum;                                                                         // 更新需要处理Tc块的计数
        bool isSeqFinish = (curDealTcNum == curBatchTcNum);                                                 // 处理到当前batch的末尾
        bool hasTail = ((curActSeqLength - headSize) % constInfo.cmpRatio) != 0;                            // 是否存在尾块

        uint32_t curValidSc = curDealTcNum;                                                                    // 更新sc
        if (isSeqFinish && hasTail && curValidSc > 0) {
            curValidSc --;
        }
        info.dealScSize += curValidSc;
        // 2. 当前batch全部处理完
        if (isSeqFinish) {
            curSEnd = curActSeqLength;
            uint32_t curBatchTotalTcNum = GetRemainTcNum(0, headSize, constInfo.cmpRatio, curActSeqLength);  // 当前batch总Tc块
            uint32_t tcNumBefore = curBatchTotalTcNum - curDealTcNum;
            info.scEnd = tcNumBefore + curDealTcNum;

            if (hasTail && info.scEnd > 0) {
                info.scEnd--;
            }
            curScEnd = info.scEnd;
        } else {
            // 3. 处理到当前batch刚好攒满需要处理的Tc块
            uint32_t remainSLen = 0;
            if (curSStart == 0) {
                // 加上头块如果有
                uint32_t midTc = (headSize > 0) ? (curDealTcNum - 1) : curDealTcNum;
                remainSLen = headSize + midTc * constInfo.cmpRatio;
                info.scEnd = curValidSc;
            } else {
                remainSLen = curDealTcNum * constInfo.cmpRatio;
                info.scEnd = info.scStart + curValidSc;
            }

            curSEnd = curSStart + remainSLen;
            // 尾块处理
            if (curSEnd > curActSeqLength) {
                curSEnd = curActSeqLength;
            }
        }
        info.sEnd = curSEnd;
        curScEnd = info.scEnd;

        if (tcNumCount > 0) {
            curSStart = 0;
        } else {
            break;
        }
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernel<COMP>::IsNeedExcute(uint32_t curBasicBlockIdx) {
    // 处理v2的非完整轮，修正后的需要循环的次数
    uint32_t fixBasicBlockNum = (constInfo.realDealBasicBlockNum + constInfo.nSize - 1) / constInfo.nSize * constInfo.nSize;
    if (constInfo.curGroupIdx > constInfo.tailGroupIdx || curBasicBlockIdx >= fixBasicBlockNum) {
        return false;
    }
    return true;
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeMm1(const RunInfo &info) {
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG);
    blockCube_.ComputeMm1(info);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::ComputeVec1(const RunInfo &info) {
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
__aicore__ inline void CompressorKernel<COMP>::ComputeVec2(const RunInfo &info) {
    blockVec_.ComputeVec2(info);
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::AllocEventID() {
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
__aicore__ inline void CompressorKernel<COMP>::FreeEventID() {
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG);
        blockCube_.FreeEventID(pipe_);
    } else {
        blockVec_.FreeEventID();
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernel<COMP>::Process() {
    AllocEventID();

    RunInfo extraInfo[1];
    GetCurCoreStartIdx();

    RunInfo vec2Info{};
    for (uint32_t i = 0; i < constInfo.singleCoreDealTcBasicNum; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        // 获取各切分轴的起始核结束索引
        CalcParams(extraInfo0);
        if ((i % constInfo.nSize) == 0) {
            vec2Info.bStart = extraInfo0.bStart;
            vec2Info.sStart = extraInfo0.sStart;
            vec2Info.scStart = extraInfo0.scStart;
            vec2Info.dealTcNum = 0;
            vec2Info.dealScSize = 0;
        }

        extraInfo0.vec1ResOffset = vec2Info.dealScSize * constInfo.headDim;
        bool isNeedExcute = IsNeedExcute(i);
        if ASCEND_IS_AIC {
            if (isNeedExcute && i < constInfo.realDealBasicBlockNum) {
                ComputeMm1(extraInfo0);
            }
        } else {
            if (isNeedExcute && i < constInfo.realDealBasicBlockNum) {
                ComputeVec1(extraInfo0);
                vec2Info.dealTcNum += extraInfo0.dealTcNum;
                vec2Info.dealScSize += extraInfo0.dealScSize;
            }
            
            // 累积N个基本块/最后一次循环
            if ((i + 1) % constInfo.nSize == 0 || (i + 1) == constInfo.singleCoreDealTcBasicNum) {
                SyncAll();
                if (isNeedExcute) {
                    vec2Info.bEnd = extraInfo0.bEnd;
                    vec2Info.sEnd = extraInfo0.sEnd;
                    vec2Info.scEnd = extraInfo0.scEnd;
                    // 累计已压缩的块为0时, 不需要执行后续计算
                    if (vec2Info.dealScSize) {
                        ComputeVec2(vec2Info);
                    }
                }
                SyncAll();
            }
        }
    }
    FreeEventID();
}

} // namespace Compressor

#endif // COMPRESSOR_KERNEL_H