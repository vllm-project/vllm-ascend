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
 * \file sparse_flash_attention_kernel_mla.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
#define SPARSE_FLASH_ATTENTION_KERNEL_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_service_cube_mla.h"
#include "sparse_flash_attention_service_vector_mla.h"
#include "sparse_flash_attention_common_arch35.h"
#include "sparse_flash_attention_kvcache.h"

#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using matmul::MatmulType;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType> class SparseFlashAttentionKernelMla {
public:
    ARGS_TRAITS;

    __aicore__ inline SparseFlashAttentionKernelMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
                                const SparseFlashAttentionTilingDataMla *__restrict tiling,
                                __gm__ uint8_t *gmTiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *workspace,
        const SparseFlashAttentionTilingDataMla *__restrict tiling,
        TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);

    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev, uint32_t s2EndPrev);

    TPipe *pipe;

    const SparseFlashAttentionTilingDataMla *__restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1RightBuffers;
    CVSharedParams sharedParams;
    /* GM信息 */
    __gm__ int32_t *actualSeqKvlenAddr = nullptr;
    __gm__ int32_t *actualSeqQlenAddr = nullptr;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    uint32_t usedCoreNum = 0U;

    GlobalTensor<int32_t> oriTopkLengthGm;
    bool hasOriTopkLength = false;
    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 切G时最大s2Loop */
    int64_t maxS2LoopCnt;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    uint32_t crossCoreSyncBufId = 0;
};

template <typename CubeBlockType, typename VecBlockType>
 __aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *actualSeqLengthsQ,
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, const SparseFlashAttentionTilingDataMla *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }

    constInfo.s1BaseSize = 64;
    constInfo.s2BaseSize = 128;

    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams,  \
        this->aicIdx, constInfo.subBlockIdx, actualSeqLengthsQ, actualSeqLengths);
    if ASCEND_IS_AIV {
        constInfo.bSize = this->sharedParams.bSize;
        constInfo.gSize = this->sharedParams.gSize;
        constInfo.s1Size = this->sharedParams.s1Size;
        constInfo.dSizeV = 512;
        constInfo.needInit = this->sharedParams.needInit;
        constInfo.returnSoftmaxLse = this->sharedParams.returnSoftmaxLse;
    }
    vecBlock.CleanOutput(attentionOut, softmaxMax, softmaxSum, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, l1BufferManager, query, queryRope);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, queryRope, keyRope, sparseIndices, \
        blockTable, actualSeqLengthsQ, actualSeqLengths, softmaxMax, softmaxSum, \
        workspace, tiling, tPipe); // gm设置
    this->InitCalcParamsEach();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType> __aicore__ inline
void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitCalcParamsEach()
{
    // 计算总的基本块
    maxS2LoopCnt = 0; // 所有核中最大累计s2Loop
    uint32_t totalBaseNum = 0;
    uint32_t s1GBaseSize = constInfo.gSize;
    uint32_t actBatchS2 = 1;
    uint32_t coreNum = GetBlockNum(); // G128时相邻两个cube核处理一个s1，coreNum减半
    uint32_t currCoreIdx = aicIdx;
    if constexpr (IS_SPLIT_G) {
        coreNum = coreNum >> 1;
        currCoreIdx = currCoreIdx >> 1;
    }
    uint32_t actBatchS1 = 1;
    for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx); // 不切S2，只关注S1
        if (actBatchS1 < constInfo.s1Size) {
            constInfo.needInit = true;
        }
        totalBaseNum += actBatchS1 * actBatchS2;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = (totalBaseNum + avgBaseNum - 1) / avgBaseNum << 1;
        }
    } else {
        if constexpr (IS_SPLIT_G) {
            usedCoreNum = totalBaseNum << 1;
        } else {
            usedCoreNum = totalBaseNum;
        }
    }

    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = avgBaseNum * (constInfo.sparseBlockCount + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }

    if (aicIdx >= usedCoreNum) {
        return;
    }
	// 计算当前核的基本块
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t lastValidBIdx = 0;
    uint32_t lastValidactBatchS1 = 0;
    bool setStart = false;
    targetBaseNum = (currCoreIdx + 1) * avgBaseNum; // 计算当前的目标权重
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.bSize * constInfo.n2Size; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.n2Size;
        actBatchS1 = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
        for (uint32_t s1GIdx = 0; s1GIdx < actBatchS1; s1GIdx++) {
            accumBaseNum += 1;
            if (!setStart && accumBaseNum >= targetStartBaseNum) {
                constInfo.bN2Start = bN2Idx;
                constInfo.gS1Start = s1GIdx;
                setStart = true;
            }
            if (accumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                constInfo.bN2End = bN2Idx;
                constInfo.gS1End = s1GIdx;
                constInfo.s2End = 0;
                if (currCoreIdx != 0) {
                    GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
                }
                return;
            }
        }
        if ((actBatchS1 > 0) && (actBatchS2 > 0)) {
            lastValidBIdx = bIdx;
            lastValidactBatchS1 = actBatchS1;
        }
    }
    if (!setStart) {
        constInfo.bN2Start = lastValidBIdx;
        constInfo.gS1Start = lastValidactBatchS1 - 1;
    }
    if (accumBaseNum < targetBaseNum) {
		// 更新最后一个核的End分核信息
        constInfo.bN2End = lastValidBIdx;
        constInfo.gS1End = lastValidactBatchS1 - 1;
        constInfo.s2End = 0;
        if (currCoreIdx != 0) {
            GetAxisStartIdx(constInfo.bN2Start, constInfo.gS1Start, 0);
        }
        return;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline uint64_t
SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::GetBalanceActualSeqLengths(
    GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqQlenAddr[bIdx] - actualSeqQlenAddr[bIdx - 1];
        } else if (bIdx == 0) {
            return actualSeqQlenAddr[0];
        } else {
            return 0;
        }
    } else {
        if (constInfo.isActualLenDimsNull == 1) {
            return constInfo.s1Size;
        } else {
            return actualSeqQlenAddr[bIdx];
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                uint32_t s1GEndPrev,
                                                                                uint32_t s2EndPrev)
{
    uint32_t bEndPrev = bN2EndPrev / constInfo.n2Size;
    uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
    uint32_t s1GPrevBaseNum = actualSeqQPrev;
    constInfo.bN2Start = bN2EndPrev;
    constInfo.gS1Start = s1GEndPrev;
    
    constInfo.s2Start = 0;
    if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
        constInfo.gS1Start = 0;
        constInfo.bN2Start++;
    } else {
        constInfo.gS1Start++;
    }
}

template <typename CubeBlockType, typename VecBlockType> __aicore__ inline
void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, const SparseFlashAttentionTilingDataMla *__restrict tiling, TPipe *tPipe)
{
    if (actualSeqLengthsQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
    }
    if (actualSeqLengths != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t *)actualSeqLengths;
    }

    vecBlock.InitGlobalBuffer(key, value, keyRope, sparseIndices, blockTable, softmaxMax, softmaxSum);
    cubeBlock.InitCubeInput(key, keyRope, sparseIndices, blockTable, actualSeqLengthsQ, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t *workspace)
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 576 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1RightBuffers.Init(l1BufferManager, mm1RightSize);
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIC {
        l1RightBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
    }
    ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    bmm2Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }

    uint32_t v0ResSize = constInfo.s2BaseSize * 576U * sizeof(Q_T);
    int64_t totalOffset;
    if constexpr (IS_SPLIT_G) {
        totalOffset = v0ResSize * 3 * (aicIdx >> 1U);
    } else {
        totalOffset = v0ResSize * 3 * aicIdx;
    }
    gmBufferManager.Init(workspace + totalOffset);
    v0ResGmBuffers.Init(gmBufferManager, v0ResSize);
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    usedCoreNum = sharedParams.usedCoreNum;

    if ASCEND_IS_AIC {
        constInfo.bSize = this->sharedParams.bSize;
        constInfo.gSize = this->sharedParams.gSize;
        constInfo.s1Size = this->sharedParams.s1Size;
        constInfo.dSizeV = 512;
        constInfo.needInit = this->sharedParams.needInit;
    }
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeVInput = sharedParams.dSizeVInput;
    constInfo.dSizeRope = 64;
    constInfo.dSizeNope = 512;
    constInfo.tileSize = sharedParams.tileSize;
    constInfo.sparseBlockCount = sharedParams.sparseBlockCount;
    constInfo.sparseBlockSize = 1;
    constInfo.cmpRatio = sharedParams.cmpRatio;
    constInfo.oriWinLeft = sharedParams.oriWinLeft;
    constInfo.oriWinRight = sharedParams.oriWinRight;
    constInfo.sparseMode = sharedParams.oriMaskMode;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;
    constInfo.gD = constInfo.gSize * constInfo.dSize;
    constInfo.n2GD = constInfo.n2Size * constInfo.gD;
    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.isActualLenDimsNull = sharedParams.isActualSeqLengthsNull;
    constInfo.isActualLenDimsKVNull = sharedParams.isActualSeqLengthsKVNull;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    constInfo.layoutType = sharedParams.layoutType;

    constInfo.isActualLenDimsNull = sharedParams.isActualSeqLengthsNull;
    constInfo.isActualLenDimsKVNull = sharedParams.isActualSeqLengthsKVNull;

    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
    if ASCEND_IS_AIV {
        constInfo.softmaxScale = sharedParams.softmaxScale;
        constInfo.oriBlockSize = sharedParams.oriBlockSize;
        constInfo.oriMaxBlockNumPerBatch = sharedParams.oriMaxBlockNumPerBatch;
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    // bsize + 1-> bsize
    this->constInfo.actualSeqLenSize = this->sharedParams.bSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.bSize;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }

    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    bool hasLoad = aicIdx < usedCoreNum;
    if (!hasLoad) {
        if ASCEND_IS_AIV {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<SFA_SYNC_MODE0, PIPE_MTE3>(15);
                    CrossCoreWaitFlag<SFA_SYNC_MODE0, PIPE_MTE3>(15);
                }
            }
        }
        return;
    }

    // 适配分核左闭右开
    uint32_t bIdx = constInfo.bN2End / constInfo.n2Size;
    uint32_t actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
    uint32_t gS1max = actS1Size;
    if (constInfo.gS1End + 1 < gS1max) {
        /* constInfo.gS1End != gS1max时，gS1End需要往后加一格, bN2End不变 */
        constInfo.gS1End = constInfo.gS1End + 1;
    } else {
        /* constInfo.gS1End == gS1max，bN2End需要往后加一格，bN2End变为0，以代表末尾 */
        constInfo.bN2End = constInfo.bN2End + 1;
        constInfo.gS1End = 0;
    }

    // 分核信息
    uint32_t bN2StartIdx = constInfo.bN2Start;
    uint32_t bN2EndIdx = constInfo.bN2End;
    uint32_t gS1StartIdx = constInfo.gS1Start;
    uint32_t nextGs1Idx = constInfo.gS1End;
    uint32_t s2StartIdx = 0;
    uint32_t s2EndIdx = 0;
    uint32_t s2LoopLimit = 0;

    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    int64_t taskId = 0;
    bool notLast = true;
    RunInfo runInfo[3];
    RunParamStr runParam;
    int64_t multiCoreInnerIdx = 1;

    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        runParam.boIdx = bnIdx;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam, this->constInfo,
            this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx);

        int64_t gS1LoopEnd = lastBN ? (runParam.gs1LoopEndIdx + PRELOAD_NUM) : runParam.gs1LoopEndIdx;
        for (int64_t gS1Index = runParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.gs1LoopEndIdx;
                switch (extraGS1) {
                    case 0:
                        notLastTwoLoop = false;
                        break;
                    case 1:
                        notLast = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastTwoLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
                if constexpr (IS_SPLIT_G) {
                    maxS2LoopCnt -= (s2LoopLimit + 1);
                }
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastTwoLoop) {
                    RunInfo &runInfo1 = runInfo[taskId % 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(),
                            this->l1RightBuffers.Get(), v0ResGmBuffers.Get(), runInfo1, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), v0ResGmBuffers.Get(),
                            runInfo1, this->constInfo, 0);
                    }
                } else {
                    if ASCEND_IS_AIV {
                        if constexpr (IS_SPLIT_G) {
                            if (maxS2LoopCnt > 0) {
                                maxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
                            }
                        }
                    }
                }
                if (taskId > 0 && notLast) {
                    auto &runInfo2 = runInfo[(taskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec1(this->l1RightBuffers.GetReused(),
                            this->bmm1Buffers.Get(), runInfo2, this->constInfo);
                    } else {
                        RunInfo &runInfo2 = runInfo[(taskId + 2) % 3];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1RightBuffers,
                            this->l1RightBuffers.GetReused(), runInfo2, this->constInfo);
                    }
                }
                if (taskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 3];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    if ASCEND_IS_AIV {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        runParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64; // N1=128场景，相邻cube核处理一个s1，第一个cube核承担0-63行g，第二个cube核承担后64行g
    } else {
        runParam.goIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
    int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.kvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        runInfo.s2EndIdx = runParam.s2LineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashAttentionKernelMla<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &runInfo, RunParamStr &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;
    runInfo.vec2MBaseSize = runInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = constInfo.s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}
}
#endif // SPARSE_FLASH_ATTENTION_KERNEL_MLA_H
