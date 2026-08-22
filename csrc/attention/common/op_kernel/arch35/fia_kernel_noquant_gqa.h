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
 * \file fia_kernel_noquant_gqa.h
 * \brief
 */

#ifndef FIA_KERNEL_NOQUANT_GQA_H_
#define FIA_KERNEL_NOQUANT_GQA_H_

#include "fia_public_define_arch35.h"
#include "fia_block_cube_noquant_gqa.h"
#include "fia_block_vec_noquant_gqa.h"
#include "../memory_copy_arch35.h"
#include "fia_block_vec_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "fia_tiling_data_noquant_gqa.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernel {
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
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
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
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGmKv;
    GlobalTensor<uint32_t> fiaMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoX constInfo;

    const NoQuantTilingArch35 *__restrict tilingData;
    TPipe *pipe = nullptr;
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;

    uint32_t coreGS1Loops = 0U;
    uint32_t accGS1Loops = 0U;
    uint32_t varlenCalcTimes = 0U;
    int64_t preTaskIdx = 0;
    int64_t curTaskIdx = 0;

    // schduler params
    int64_t validTaskNum = 0;
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

    ActualSeqLensParser<Q_MODE> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE> kvActSeqLensParser;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantGqaKernel()
        : cubeBlock(constInfo), vecFaBlock(constInfo), vecFdBlock(constInfo){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                                __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                                __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope,
                                __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                const NoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        fiaMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)fiaMetaData,
                                      NPU_AIC_CORE_NUM * FA_METADATA_SIZE + NPU_AIV_CORE_NUM * FD_METADATA_SIZE);

        InitConstInfo(queryPaddingSize, kvPaddingSize, actualSharedPrefixLen);


        keyPtr = key;
        valuePtr = value;

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);

        actualSeqLengthsGmKv.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsKv, constInfo.actualSeqLenKVSize);
        kvActSeqLensParser.Init(actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize, constInfo.s2Size);

        InitMMResBuf(workspace);

        if ASCEND_IS_AIV {
            vecFaBlock.InitVecBlock(tPipe, pse, actualSeqLengths, actualSeqLengthsKv, postQuantScale, postQuantOffset,
                attenMask, learnableSink, softmaxLse, attentionOut, workspace);
            vecFaBlock.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, blockTable, queryRope, keyRope,
                actualSeqLengths, actualSeqLengthsKv, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen);
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                vecFdBlock.InitParams();
                vecFdBlock.InitGlobalTensor(this->vecFaBlock.softmaxFDMaxGm, this->vecFaBlock.softmaxFDSumGm,
                                            this->vecFaBlock.accumOutGm, this->vecFaBlock.attentionOutGm,
                                            this->actualSeqLengthsGmQ, this->actualSeqLengthsGmKv, keyPtr,
                                            postQuantScale, postQuantOffset);
                if (constInfo.isSoftmaxLseEnable) {
                    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
                }
            }
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);

        // GM Buffer
        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSize =
                mBaseSize * constInfo.dBasicBlock; // 使用Cube计算的总大小，Gm上的数据按照实际的dSize存储
            int64_t prevCoreTotalOffset = constInfo.aicIdx * (PRELOAD_N + 1) * mm2ResultSize; // 3为preload次数
            // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
            gmBufferManager.Init(workspace + prevCoreTotalOffset * sizeof(T));
            bmm2ResGmBuffers.Init(gmBufferManager, mm2ResultSize * sizeof(T));
            workspace = workspace + constInfo.coreNum * (PRELOAD_N + 1) * mm2ResultSize * sizeof(T);
        }
    }

    __aicore__ inline void InitConstInfo(__gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                                         __gm__ uint8_t *actualSharedPrefixLen)
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = constInfo.aivIdx / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        const auto& fiaBaseParams = this->tilingData->fiaBaseParams;
        const auto& fiaAttenMaskParams = this->tilingData->fiaAttenMaskParams;
        const auto& fiaPageAttentionParams = this->tilingData->fiaPageAttentionParams;
        const auto& fiaWorkspaceParams = this->tilingData->fiaWorkspaceParams;
        const auto& fiaS1OuterSplitCoreParams = this->tilingData->fiaS1OuterSplitCoreParams;
        const auto& fiaEmptyTensorParams = this->tilingData->fiaEmptyTensorParams;

        constInfo.bSize = fiaBaseParams.bSize;
        constInfo.t1Size = fiaBaseParams.t1Size;
        constInfo.t2Size = fiaBaseParams.t2Size;
        constInfo.n2Size = fiaBaseParams.n2Size;
        constInfo.gSize = fiaBaseParams.gSize;
        constInfo.s1Size = fiaBaseParams.s1Size;
        constInfo.s2Size = fiaBaseParams.s2Size;
        constInfo.dSize = fiaBaseParams.dSize;
        constInfo.dSizeV = fiaBaseParams.dSizeV;
        if constexpr (HAS_ROPE) {
            constInfo.dSizeRope = fiaBaseParams.dSizeRope;
        } else {
            constInfo.dSizeRope = 0;
        }
        constInfo.actualSeqLenSize = fiaBaseParams.actualSeqLengthsQSize;
        constInfo.actualSeqLenKVSize = fiaBaseParams.actualSeqLengthsKVSize;
        constInfo.scaleValue = static_cast<float>(fiaBaseParams.scaleValue);
        constInfo.coreNum = fiaBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FIA_LAYOUT>(fiaBaseParams.outputLayout);

        constInfo.keyStrides.bnStride = fiaBaseParams.keyStrides.bnStride;
        constInfo.keyStrides.n2Stride = fiaBaseParams.keyStrides.n2Stride;
        constInfo.valueStrides.bnStride = fiaBaseParams.valueStrides.bnStride;
        constInfo.valueStrides.n2Stride = fiaBaseParams.valueStrides.n2Stride;

        constInfo.sparseMode = fiaAttenMaskParams.sparseMode;
        constInfo.preTokens = fiaAttenMaskParams.preTokens;
        constInfo.nextTokens = fiaAttenMaskParams.nextTokens;
        if constexpr (HAS_MASK) {
            constInfo.attenMaskBatch = fiaAttenMaskParams.attenMaskBatch;
            constInfo.attenMaskS1Size = fiaAttenMaskParams.attenMaskS1Size;
            constInfo.attenMaskS2Size = fiaAttenMaskParams.attenMaskS2Size;
        }
        constInfo.isRowInvalidOpen = fiaAttenMaskParams.isRowInvalidOpen;
        constInfo.isExistRowInvalid = fiaAttenMaskParams.isExistRowInvalid;
        constInfo.accumOutSize = fiaWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = fiaWorkspaceParams.logSumExpSize;
        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo.maxBlockNumPerBatch = fiaPageAttentionParams.maxBlockNumPerBatch;
            constInfo.blockSize = fiaPageAttentionParams.blockSize;
            constInfo.paLayoutType = fiaPageAttentionParams.paLayoutType;
        }
        // LSE
        constInfo.isSoftmaxLseEnable = fiaBaseParams.isSoftMaxLseEnable;

        if (fiaS1OuterSplitCoreParams.enableS1OutSplit) {
            constInfo.enableS1OutSplit = true;
            constInfo.bN2Start = 0;
            constInfo.gS1OStart = 0;
            constInfo.s2OStart = 0;
            constInfo.bN2End = 0;
            constInfo.gS1OEnd = 0;
            constInfo.s2OEnd = 0;
            constInfo.coreFirstTmpOutWsPos = 0;
            constInfo.totalSize = fiaS1OuterSplitCoreParams.totalSize;
        } else {
            // 任务起始位置
            constInfo.enableS1OutSplit = false;
            constInfo.bN2Start = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_BN2_START_INDEX));
            constInfo.gS1OStart = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_M_START_INDEX));
            constInfo.s2OStart = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_S2_START_INDEX));
            constInfo.bN2End = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_BN2_END_INDEX));
            constInfo.gS1OEnd = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_M_END_INDEX));
            constInfo.s2OEnd = fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_S2_END_INDEX));
            constInfo.coreFirstTmpOutWsPos =
                fiaMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        }
        constInfo.dBasicBlock = AttentionCommon::Align(constInfo.dSizeV, 64U);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx)
    {
        return FA_METADATA_SIZE * NPU_AIC_CORE_NUM + FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CalcS1OutSplitLoopTimes()
    {
        int64_t varlenCycleCoreNums = constInfo.coreNum * 2;
        int64_t varlenCalcLoops = constInfo.totalSize / varlenCycleCoreNums; // 需要进行计算的循环次数(正序+倒序为一次循环)
        int64_t varlenCalcLoopsRemain = constInfo.totalSize % varlenCycleCoreNums; // 一次循环正序+倒序为两倍核数
        varlenCalcTimes = varlenCalcLoops * 2;
        if (varlenCalcLoopsRemain >= constInfo.aicIdx + 1) {
            varlenCalcTimes++;
            if (varlenCalcLoopsRemain > constInfo.coreNum &&
                (constInfo.aicIdx + 1) > varlenCycleCoreNums - varlenCalcLoopsRemain) {
                varlenCalcTimes++;
            }
        }
    }

    __aicore__ inline int64_t CalcS1OutTaskIdx()
    {
        if (coreGS1Loops % 2 == 0) {
            return coreGS1Loops * constInfo.coreNum + constInfo.aicIdx;
        }
        return (coreGS1Loops + 1U) * constInfo.coreNum - constInfo.aicIdx - 1U;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
            if constexpr (BMM2_TOUB) {
                bmm2Buffers.Get().SetCrossCore();
                bmm2Buffers.Get().SetCrossCore();
            }
        }
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            if constexpr (BMM2_TOUB) {
                bmm2Buffers.Get().WaitCrossCore();
                bmm2Buffers.Get().WaitCrossCore();
            }
        }
    }

    __aicore__ inline void FlashAttention()
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }

        RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        uint32_t bN2Cur = constInfo.bN2Start;
        uint32_t gS1Cur = constInfo.gS1OStart;
        uint32_t s2Cur = constInfo.s2OStart;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        if (constInfo.enableS1OutSplit) {
            CalcS1OutSplitLoopTimes();
        }

        bool shouldDispatchTask = true;
        bool shouldExecuteTask = false;
        uint32_t createdTaskCount = 0U;
        uint32_t executedTaskCount = 0U;
        while (shouldDispatchTask || shouldExecuteTask) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    if ASCEND_IS_AIV {
                        vecFaBlock.DealZeroActSeqLen(bN2Cur);
                    }
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
            if (shouldExecuteTask) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (constInfo.enableS1OutSplit) {
            return coreGS1Loops < varlenCalcTimes;
        }
        return ((bN2Cur != constInfo.bN2End) || (gS1Cur != constInfo.gS1OEnd) || (s2Cur != constInfo.s2OEnd));
    }

    __aicore__ inline bool ShouldExecuteTask(RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        return validTaskNum > 0;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart) && (s2Cur == constInfo.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
            actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
        }

        if (constInfo.enableS1OutSplit && accGS1Loops != CalcS1OutTaskIdx()) {
            return TASK_DEAL_MODE::SKIP_S1OUT;
        }

        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }

        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (curS2Start >= curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur < curS2Start) {
            return TASK_DEAL_MODE::NOT_START;
        }

        if (s2Cur >= curS2End) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur == curS2Start) {
            mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
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

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
            headS2Split = constInfo.s2OStart != 0U;
            curS2Start = constInfo.s2OStart;
        }

        if (!constInfo.enableS1OutSplit && (bN2Cur == constInfo.bN2End) && (gS1Cur == constInfo.gS1OEnd)) {
            tailS2Split = constInfo.s2OEnd != 0U;
            curS2End = constInfo.s2OEnd;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        
        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfo.gSize)) -
                               1;

        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
                s1FirstToken = 0;
            }
        }
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);
        // 3. trans index of token to index of block
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv)
            || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        // 4. Calc curS2Start, curS2End
        curS2Start = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        curS2End = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        if (bN2Cur == constInfo.bN2Start && gS1Cur == constInfo.gS1OStart) { // first line
            headS2Split = constInfo.s2OStart > curS2Start ? true : false;
            curS2Start = AttentionCommon::Max(curS2Start, constInfo.s2OStart);
        }
        if (bN2Cur == constInfo.bN2End && gS1Cur == constInfo.gS1OEnd) { // last line
            tailS2Split = constInfo.s2OEnd > 0U ? true : false;
            curS2End = constInfo.s2OEnd > 0U ? AttentionCommon::Min(curS2End, constInfo.s2OEnd) : curS2End;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];                  // 本轮任务
        RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE]; // 上PRELOAD_N轮任务
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
                DisableTask(runInfoNegN);
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

    __aicore__ inline void EnableTask(RunInfoX &runInfo)
    {
        runInfo.isValid = true;
        validTaskNum++;
    }

    __aicore__ inline void DisableTask(RunInfoX &runInfo)
    {
        runInfo.isValid = false;
        validTaskNum--;
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoX &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        EnableTask(runInfo);
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / constInfo.n2Size;
        info.n2Idx = bN2Cur % constInfo.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH ||
                      LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;
        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo.gSize;
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

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        if constexpr (USE_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actVecMSize = (info.actMSize + 1) >> 1;
        }
        info.vecMbaseIdx = 0;
        if (constInfo.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if (constInfo.enableS1OutSplit) {
            info.isS2SplitCore = false;
            return;
        }

        if ((constInfo.bN2Start == constInfo.bN2End && constInfo.gS1OStart == constInfo.gS1OEnd)) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == constInfo.bN2Start) && (gS1Cur == constInfo.gS1OStart)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == constInfo.bN2End) &&
                       (gS1Cur == constInfo.gS1OEnd)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void S1OutSplitUpdateAxis(uint32_t &bN2Cur, uint32_t &gS1Cur)
    {
        if (coreGS1Loops >= varlenCalcTimes || bN2Cur >= constInfo.n2Size * constInfo.bSize) {
            return;
        }
        curTaskIdx = CalcS1OutTaskIdx();

        // 本次需要跳过的GS1
        int64_t gS1Skip = curTaskIdx - preTaskIdx;
        accGS1Loops += gS1Skip;
        preTaskIdx = curTaskIdx;
        if (gS1Skip == 0) {
            return;
        }

        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = actSeqLensQ == 0 ? constInfo.gSize : (gS1Size + mBaseSize - 1) / mBaseSize;
        uint64_t gS1RemainCurBN2 = gS1LoopTimes - gS1Cur;

        // gS1Skip 不超过当前BN2剩余的S1G，则只需要更新gS1Cur
        if (gS1Skip < gS1RemainCurBN2) {
            gS1Cur += gS1Skip;
            return;
        }

        // 先跳过当前BN2剩余的S1G，再跳N2
        gS1Skip -= gS1RemainCurBN2;
        gS1Cur = 0;

        uint32_t n2Size = constInfo.n2Size;
        uint32_t n2Idx = bN2Cur % n2Size;
        uint32_t bIdx = bN2Cur / n2Size;
        uint64_t gS1RemainCurBatch = 0;
        // 先跳过当前Batch剩余的N2
        if (n2Idx + 1 < n2Size) {
            bN2Cur++;
            n2Idx++;
            gS1RemainCurBatch = gS1LoopTimes * (n2Size - n2Idx);
            // 这个batch剩余的N2足够 gS1Skip
            if (gS1Skip < gS1RemainCurBatch) {
                bN2Cur += gS1Skip / gS1LoopTimes;
                gS1Cur = gS1Skip % gS1LoopTimes;
                return;
            }
            gS1Skip -= gS1RemainCurBatch;
        }
        n2Idx = 0;
        bIdx++;

        // 跳batch
        while (bIdx < constInfo.bSize) {
            actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
            actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
            gS1Size = actSeqLensQ * constInfo.gSize;
            gS1LoopTimes = actSeqLensQ == 0 ? constInfo.gSize : (gS1Size + mBaseSize - 1) / mBaseSize;
            if (gS1Skip >= n2Size * gS1LoopTimes) {
                bIdx++;
                gS1Skip -= n2Size * gS1LoopTimes;
                continue;
            }

            gS1Cur = gS1Skip % gS1LoopTimes;
            n2Idx = (gS1Skip / gS1LoopTimes) % n2Size;
            bN2Cur = bIdx * n2Size + n2Idx;
            return;
        }

        // 所有batch已经执行完
        bN2Cur = constInfo.bSize * n2Size;
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start;
            return;
        } else if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
            s2Cur++;
            return;
        }

        // 当前BN2未处理完
        s2Cur = 0;
        if (constInfo.enableS1OutSplit) {
            if (taskDealMode != TASK_DEAL_MODE::SKIP_S1OUT) {
                coreGS1Loops++;
            }
            S1OutSplitUpdateAxis(bN2Cur, gS1Cur);
            return;
        }

        uint64_t gS1Size = actSeqLensQ * constInfo.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode()
    {
        vecFdBlock.InitBuffers(this->pipe);
        AscendC::ICachePreLoad(2);
        uint32_t fdCoreEnable = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_CORE_ENABLE_INDEX));
        uint32_t fdBN2Idx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_BN2_IDX_INDEX));
        uint32_t fdMIdx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_M_IDX_INDEX));
        uint32_t fdS2SplitNum = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_S2_SPLIT_NUM_INDEX));
        uint32_t mStart = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_M_START_INDEX));
        uint32_t mLen = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_M_NUM_INDEX));
        uint32_t fdWorkspaceIdx = fiaMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FD_WORKSPACE_IDX_INDEX));

        FDparamsX fdParams = {fdCoreEnable, fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, fdWorkspaceIdx};
        vecFdBlock.AllocEventID();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams);
        vecFdBlock.FreeEventID();
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
}; // FlashAttentionNoQuantGqaKernel

} // namespace BaseApi

#endif // FIA_KERNEL_NOQUANT_GQA_H_