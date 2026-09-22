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
 * \file flash_mla_c8_kernel.h
 * \brief MLA 全量化重构模板 Kernel
 */
#ifndef FIA_KERNEL_FULLQUANT_MLA_H_
#define FIA_KERNEL_FULLQUANT_MLA_H_

#include "flash_mla_c8_public.h"
#include "flash_mla_c8_cube.h"
#include "flash_mla_c8_vec.h"
#include "flash_mla_c8_flashdecode.h"
#include "flash_mla_c8_memory.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_mla_with_kvcache_tiling_data.h"
#include <limits>

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionFullQuantMlaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr bool HAS_ROPE = CubeBlockType::HAS_ROPE;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
    static constexpr FlashMlaC8Layout LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr FlashMlaC8Layout LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // CV buffers
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    typename std::conditional<BMM2_TOUB, BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                              BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>::type bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> fiaMetaDataGm;
    uint32_t sectionIdx_ = 0;
    uint32_t sectionNum_ = 0;
    uint32_t aicCoreNum_ = 0;
    uint32_t aivCoreNum_ = 0;
    bool enableFlashDecode_ = false;

    GlobalTensor<float> softmaxLseGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo{};
    const FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tilingData;
    TPipe *pipe = nullptr;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    uint32_t createdTaskCount = 0U;
    uint32_t executedTaskCount = 0U;

    // schduler params
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    ActualSeqLensParser<Q_MODE, uint32_t, true> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE, uint32_t> kvActSeqLensParser;

    __aicore__ inline FlashAttentionFullQuantMlaKernel()
        : cubeBlock(constInfo),
          vecFaBlock(constInfo),
          vecFdBlock(constInfo){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                __gm__ uint8_t *fiaMetaData, const FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe,
                                __gm__ uint8_t *sequsedQ, uint32_t sectionIdx)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        fiaMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)fiaMetaData);
        sectionIdx_ = sectionIdx;
        sectionNum_ = fiaMetaDataGm.GetValue(0);
        enableFlashDecode_ = fiaMetaDataGm.GetValue(1) != 0;
        aicCoreNum_ = fiaMetaDataGm.GetValue(FLASH_MLA_METADATA_AIC_NUM_INDEX);
        aivCoreNum_ = fiaMetaDataGm.GetValue(FLASH_MLA_METADATA_AIV_NUM_INDEX);
        InitConstInfo();

        keyPtr = key;
        valuePtr = value;

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(actualSeqLengths, constInfo.actualSeqLenSize, sequsedQ, sequsedQ ? constInfo.bSize : 0);

        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsKv, constInfo.actualSeqLenKVSize);
        kvActSeqLensParser.Init(actualSeqLengthsGm, constInfo.actualSeqLenKVSize, constInfo.s2Size);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            vecFaBlock.InitVecBlock(tPipe, actualSeqLengths, actualSeqLengthsKv, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue, attenMask, softmaxLse, attentionOut, workspace);
            vecFaBlock.SetQueryLengths(actualSeqLengths, sequsedQ);
            if (sectionIdx_ == 0) {
                vecFaBlock.ClearOutput();
            }
        }

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blockTable, queryRope, keyRope,
                                    actualSeqLengths, actualSeqLengthsKv, dequantScaleQuery, dequantScaleKey,
                                    dequantScaleValue);
            cubeBlock.SetQueryLengths(actualSeqLengths, sequsedQ);
        }
        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm,
                                            this->vecFaBlock.attentionWireGm,
                                            this->actualSeqLengthsGmQ, this->actualSeqLengthsGm, keyPtr);
                vecFdBlock.SetQueryLengths(actualSeqLengths, sequsedQ);
                if (constInfo.isSoftmaxLseEnable) {
                    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
                }
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        constexpr uint32_t mm1OutDtype = sizeof(T);
        constexpr uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);

        if constexpr (BMM2_TOUB) {
            // dualDstCtl=1 writes only M/2 FP32 rows to each AIV.
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);

        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSize = mBaseSize * constInfo.dBasicBlock;
            int64_t prevCoretotalOffset = constInfo.aicIdx * 3 * mm2ResultSize;
            gmBufferManager.Init(workspace + prevCoretotalOffset * sizeof(T));
            bmm2ResGmBuffers.Init(gmBufferManager, mm2ResultSize * sizeof(T));
            workspace = workspace + constInfo.coreNum * 3 * mm2ResultSize * sizeof(T);
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        auto fiaBaseParams = this->tilingData->flashMlaWithKvcacheBaseParams;
        auto fiaAttenMaskParams = this->tilingData->flashMlaWithKvcacheAttenMaskParams;
        auto fiaPageAttentionParams = this->tilingData->flashMlaWithKvcachePageAttentionParams;
        auto fiaWorkspaceParams = this->tilingData->flashMlaWithKvcacheWorkspaceParams;
        auto fiaEmptyTensorParams = this->tilingData->flashMlaWithKvcacheEmptyTensorParams;

        constInfo.needInit = fiaEmptyTensorParams.needInit != 0;
        constInfo.bSize = fiaBaseParams.bSize;
        constInfo.t1Size = fiaBaseParams.t1Size;
        constInfo.t2Size = fiaBaseParams.t2Size;
        constInfo.n2Size = fiaBaseParams.n2Size;
        constInfo.gSize = fiaBaseParams.gSize;
        constInfo.s1Size = fiaBaseParams.s1Size;
        constInfo.s2Size = fiaBaseParams.s2Size;
        constInfo.dSize = fiaBaseParams.dSize;
        constInfo.dSizeV = fiaBaseParams.dSizeV;
        // MLA 不合轴
        constInfo.realN2Size = constInfo.n2Size;
        constInfo.realGSize = constInfo.gSize;
        if constexpr (HAS_ROPE) {
            constInfo.dSizeRope = fiaBaseParams.dSizeRope;
            constInfo.kRopeStrides.bnStride = fiaBaseParams.kRopeStrides.bnStride;
            constInfo.kRopeStrides.n2Stride = fiaBaseParams.kRopeStrides.n2Stride;
        }
        constInfo.actualSeqLenSize = fiaBaseParams.actualSeqLengthsQSize;
        constInfo.actualSeqLenKVSize = fiaBaseParams.actualSeqLengthsKVSize;
        constInfo.scaleValue = static_cast<float>(fiaBaseParams.scaleValue);
        constInfo.isKvContinuous = true;
        constInfo.coreNum = fiaBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FIA_LAYOUT>(fiaBaseParams.outputLayout);
        constInfo.keyStrides.bnStride = fiaBaseParams.keyStrides.bnStride;
        constInfo.keyStrides.n2Stride = fiaBaseParams.keyStrides.n2Stride;
        constInfo.valueStrides.bnStride = fiaBaseParams.valueStrides.bnStride;
        constInfo.valueStrides.n2Stride = fiaBaseParams.valueStrides.n2Stride;

        constInfo.sparseMode = fiaAttenMaskParams.sparseMode;
        constInfo.preTokens = fiaAttenMaskParams.preTokens;
        constInfo.nextTokens = fiaAttenMaskParams.nextTokens;
        constInfo.attenMaskBatch = fiaAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;
        constInfo.isRowInvalidOpen = fiaAttenMaskParams.isRowInvalidOpen;
        constInfo.isExistRowInvalid = fiaAttenMaskParams.isExistRowInvalid;

        constInfo.accumOutSize = fiaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = fiaWorkspaceParams.logSumExpSize;

        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = fiaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = fiaPageAttentionParams.blockSize;
            constInfo.paLayoutType = fiaPageAttentionParams.paLayoutType;
        }
        constInfo.isSoftmaxLseEnable = fiaBaseParams.isSoftMaxLseEnable;

        // 任务起始位置
        constInfo.bN2Start = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX));
        constInfo.gS1OStart = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX));
        constInfo.s2OStart = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX));
        constInfo.bN2End = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX));
        constInfo.gS1OEnd = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX));
        constInfo.s2OEnd = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX));
        constInfo.coreFirstTmpOutWsPos =
            fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return 16U + (sectionIdx_ * aicCoreNum_ + coreIdx) * 16U + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return 16U + sectionNum_ * aicCoreNum_ * 16U +
               (sectionIdx_ * aivCoreNum_ + coreIdx) * 16U + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if constexpr (BMM2_TOUB) {
            if ASCEND_IS_AIV {
                bmm2Buffers.Get().SetCrossCore();
            }
        }
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
        }
        if constexpr (BMM2_TOUB) {
            if ASCEND_IS_AIC {
                bmm2Buffers.Get().WaitCrossCore();
            }
        }
    }

    __aicore__ inline void FlashAttention()
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE];
        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = constInfo.gS1OStart;
        uint32_t s2Cur = constInfo.s2OStart;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        while (shouldDispatchTask || shouldExecuteTask) {
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
            if (shouldExecuteTask) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (bN2Cur != constInfo.bN2End) {
            return bN2Cur < constInfo.bN2End;
        }
        if (gS1Cur != constInfo.gS1OEnd) {
            return gS1Cur < constInfo.gS1OEnd;
        }
        return s2Cur < constInfo.s2OEnd;
    }

    __aicore__ inline bool ShouldExecuteTask(RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        for (uint32_t i = 0; i < PRELOAD_TASK_CACHE_SIZE; i++) {
            if (taskRunInfo[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart) && (s2Cur == constInfo.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo.realN2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
            actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (s2Cur < curS2Start && curS2Start < curS2End) {
            return TASK_DEAL_MODE::NOT_START;
        }
        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start) {
            mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
            headS2Split = constInfo.s2OStart != 0U;
            curS2Start = constInfo.s2OStart;
        }
        if ((bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
            tailS2Split = constInfo.s2OEnd != 0U;
            curS2End = constInfo.s2OEnd;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.realGSize)) -
            1;

        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.realGSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.realGSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }

        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == constInfo.bN2Start && gS1Cur == constInfo.gS1OStart) {
            headS2Split = constInfo.s2OStart > s2StartWithSparse ? true : false;
            curS2Start = AttentionCommon::Max(s2StartWithSparse, constInfo.s2OStart);
        }
        if (bN2Cur == constInfo.bN2End && gS1Cur == constInfo.gS1OEnd) {
            tailS2Split = constInfo.s2OEnd > 0U ? true : false;
            curS2End =
                constInfo.s2OEnd > 0U ? AttentionCommon::Min(s2EndWithSparse, constInfo.s2OEnd) : s2EndWithSparse;
        }
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo.preTokens;
        nextTokenLeftUp = constInfo.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo.sparseMode);

        if (constInfo.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }
        if (constInfo.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL || constInfo.sparseMode == fa_base_vector::TREE) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
                ComputeVec1(runInfo0);
            }
        }

        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                if ASCEND_IS_AIC {
                    ComputeMm2(runInfoNegN);
                } else {
                    ComputeVec2(runInfoNegN);
                }
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoX &runInfo)
    {
        cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoX &runInfo)
    {
        if constexpr (BMM2_TOUB) {
            cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo);
        } else {
            cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo);
        }
    }

    __aicore__ inline void ComputeVec1(RunInfoX &runInfo)
    {
        vecFaBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoX &runInfo)
    {
        if constexpr (BMM2_TOUB) {
            this->vecFaBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo);
        } else {
            this->vecFaBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo);
        }
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / (constInfo.realN2Size);
        info.n2Idx = (bN2Cur / (constInfo.realN2Size / constInfo.n2Size)) % constInfo.n2Size;
        info.realN2Idx = bN2Cur % constInfo.realN2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == FlashMlaC8Layout::LAYOUT_BSH || LAYOUT_Q == FlashMlaC8Layout::LAYOUT_SBH ||
                      LAYOUT_Q == FlashMlaC8Layout::LAYOUT_TND) {
            info.s1Idx = info.gS1Idx / constInfo.realGSize;
        } else {
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.s2LocalIdx = (s2Cur - curS2Start) * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.realGSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        info.isChangeBatch = false;

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
        info.actVecMSize = (info.actMSize + 1) >> 1;
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((constInfo.bN2Start == constInfo.bN2End && constInfo.gS1OStart == constInfo.gS1OEnd)) {
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.realGSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }

        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode()
    {
        if (!enableFlashDecode_) { return; }
        vecFdBlock.InitBuffers(this->pipe);
        AscendC::ICachePreLoad(2);
        uint32_t fdCoreEnable = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_NUM_INDEX)) != 0;
        uint32_t fdBN2Idx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_BN2_IDX_INDEX));
        uint32_t fdMIdx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_IDX_INDEX));
        uint32_t fdS2SplitNum = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_NUM_INDEX));
        uint32_t mStart = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_START_INDEX));
        uint32_t mLen = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_NUM_INDEX));
        uint32_t fdWorkspaceIdx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_IDX_INDEX));

        FDparamsX fdParams = {fdCoreEnable, fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, fdWorkspaceIdx};
        vecFdBlock.AllocEventID();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams);
        vecFdBlock.FreeEventID();
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx < constInfo.coreNum) {
            CrossCoreBufferInit();
            if ASCEND_IS_AIV {
                vecFaBlock.InitBuffers();
                vecFaBlock.AllocEventID();
            } else {
                cubeBlock.InitBuffers();
                cubeBlock.AllocEventID();
            }
            FlashAttention();

            if ASCEND_IS_AIV {
                vecFaBlock.FreeEventID();
            } else {
                cubeBlock.FreeEventID();
            }
            CrossCoreBufferUnInit();
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                FlashDecode();
            }
        }
    }
}; // FlashAttentionFullQuantMlaKernel

} // namespace BaseApi

#endif // FIA_KERNEL_FULLQUANT_MLA_H_
