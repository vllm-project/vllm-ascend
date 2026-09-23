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
 * \file flash_attn_kernel_nd.h
 * \brief FlashAttentionNoQuantGqaKernelNd —— Nd 路径专用 kernel 模板（独立类）。
 */

#ifndef FLASH_ATTN_KERNEL_ND_H_
#define FLASH_ATTN_KERNEL_ND_H_

#include "../utils/flash_attn_common_def.h"

#include "flash_attn_block_cube_nd.h"
#include "flash_attn_block_vec_nd.h"
#include "memory_copy_arch35.h"
#include "flash_attn_block_vec_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attn_tiling_data.h"
#include "../../../a5_mla_common/op_kernel/arch_info.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;
template <typename FA_T, typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernelNd {
public:
    using T = float;
    using SEQLEN_T = uint32_t;
    using INPUT_T = typename FA_T::inputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    // task ring buffer 至少需要 PRELOAD_N+1(=3) 个slot；为了用位掩码(loop & MASK)代替取模(loop % SIZE)，
    // 将容量向上取整到最近的2的幂(=4)。代价仅为多1个RunInfo slot，换取消除内层循环的取模scalar bound。
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1, "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");

    ConstInfo_t constInfo_;

    SeqLensTool<LAYOUT_T, SEQLEN_T> qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> kvSeqLensTool_;

    __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tilingData_;

    // block define
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    // 分核信息
    GlobalTensor<uint32_t> faMetaDataGm_;
    GlobalTensor<uint32_t> fdMetaDataGm_;
    // metadata
    uint32_t sectionNum_;
    uint32_t metadata_aic_num_;
    uint32_t metadata_aiv_num_;
    // fa metadata
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;
    // fd metadata
    FDparams fdParams_;

    // schduler params
    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    uint32_t curS2Start_;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_;
    uint32_t prevBN2Idx_;
    uint32_t prevGS1Idx_;
    uint32_t mloop_ = 0;
    bool headS2Split_ = false;
    bool tailS2Split_ = false;
    uint32_t s2FirstToken_ = 0;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantGqaKernelNd()
        : cubeBlock_(constInfo_, qSeqLensTool_, kvSeqLensTool_),
          vecFaBlock_(constInfo_, qSeqLensTool_, kvSeqLensTool_),
          vecFdBlock_(constInfo_, qSeqLensTool_, kvSeqLensTool_){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *fiaMetaData, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tiling)
    {
        tilingData_ = tiling;

        InitConstInfo();

        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadata_aic_num_ = ((__gm__ uint32_t *)fiaMetaData)[METADATA_HEADER_AIC_NUM_INDEX];
        metadata_aiv_num_ = ((__gm__ uint32_t *)fiaMetaData)[METADATA_HEADER_AIV_NUM_INDEX];
        constInfo_.enableFlashDecode =
            static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[METADATA_HEADER_HAS_FD_INDEX]);
        constInfo_.learnableSinkFlag = (learnableSink != nullptr);

        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + METADATA_HEADER_OFFSET),
                                      sectionNum_ * metadata_aic_num_ * METADATA_STRIDE);

        qSeqLensTool_.Init(cuSeqLensQ, constInfo_.cuSeqLensQSize, seqUsedQ, constInfo_.seqUsedQSize, constInfo_.s1Size);
        kvSeqLensTool_.Init(cuSeqLensKv, constInfo_.cuSeqLensKVSize, seqUsedKv, constInfo_.seqUsedKvSize,
                            constInfo_.s2Size);

        if ASCEND_IS_AIC {
            cubeBlock_.InitBlock(query, key, value, blockTable);
        }
        if ASCEND_IS_AIV {
            vecFaBlock_.InitBlock(attenMask, learnableSink, softmaxLse, attentionOut, workspace);
            vecFaBlock_.ClearOutput();
            if (constInfo_.enableFlashDecode) {
                fdMetaDataGm_.SetGlobalBuffer(
                    (__gm__ uint32_t *)(fiaMetaData + METADATA_HEADER_OFFSET +
                                        sectionNum_ * metadata_aic_num_ * METADATA_STRIDE * sizeof(uint32_t)),
                    sectionNum_ * metadata_aiv_num_ * METADATA_STRIDE);
                vecFdBlock_.InitBlock(learnableSink, softmaxLse, attentionOut);
                vecFdBlock_.InitGlobalTensor(vecFaBlock_.softmaxFDMaxGm_, vecFaBlock_.softmaxFDSumGm_,
                                             vecFaBlock_.accumOutGm_);
            }
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx();
        } else {
            constInfo_.aivIdx = GetBlockIdx();
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }
        constInfo_.bSize = tilingData_->flashAttnBaseParams.bSize;
        constInfo_.t1Size = tilingData_->flashAttnBaseParams.t1Size;
        constInfo_.t2Size = tilingData_->flashAttnBaseParams.t2Size;
        constInfo_.n2Size = tilingData_->flashAttnBaseParams.n2Size;
        constInfo_.gSize = tilingData_->flashAttnBaseParams.gSize;
        constInfo_.s1Size = tilingData_->flashAttnBaseParams.s1Size;
        constInfo_.s2Size = tilingData_->flashAttnBaseParams.s2Size;
        constInfo_.dSize = tilingData_->flashAttnBaseParams.dSize;
        constInfo_.dSizeV = tilingData_->flashAttnBaseParams.dSizeV;
        constInfo_.cuSeqLensQSize = tilingData_->flashAttnBaseParams.cuSeqLensQSize;
        constInfo_.cuSeqLensKVSize = tilingData_->flashAttnBaseParams.cuSeqLensKVSize;
        constInfo_.seqUsedQSize = tilingData_->flashAttnBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = tilingData_->flashAttnBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = static_cast<float>(tilingData_->flashAttnBaseParams.scaleValue);
        constInfo_.coreNum = tilingData_->flashAttnBaseParams.coreNum;
        constInfo_.outputLayout = static_cast<FA_LAYOUT>(tilingData_->flashAttnBaseParams.outputLayout);
        constInfo_.needInitOutput = tilingData_->flashAttnBaseParams.needInitOutput;

        constInfo_.sparseMode = tilingData_->flashAttnAttenMaskParams.sparseMode;
        constInfo_.preTokens = tilingData_->flashAttnAttenMaskParams.winLefts;
        constInfo_.nextTokens = tilingData_->flashAttnAttenMaskParams.winRights;
        constInfo_.attenMaskBatch = tilingData_->flashAttnAttenMaskParams.attenMaskBatch;
        constInfo_.attenMaskS1Size = tilingData_->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfo_.attenMaskS2Size = tilingData_->flashAttnAttenMaskParams.attenMaskS2Size;
        constInfo_.isExistRowInvalid = tilingData_->flashAttnAttenMaskParams.isExistRowInvalid;

        constInfo_.accumOutSize = tilingData_->flashAttnWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = tilingData_->flashAttnWorkspaceParams.logSumExpSize;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo_.maxBlockNumPerBatch = tilingData_->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfo_.blockSize = tilingData_->flashAttnPageAttentionParams.blockSize;
            constInfo_.paLayoutType = tilingData_->flashAttnPageAttentionParams.paLayoutType;
            // kvcache非连续场景的strides, 透传给offsetCalculator
            constInfo_.keyBnStride = tilingData_->flashAttnBaseParams.keyBnStride;
            constInfo_.keyN2Stride = tilingData_->flashAttnBaseParams.keyN2Stride;
            constInfo_.valueBnStride = tilingData_->flashAttnBaseParams.valueBnStride;
            constInfo_.valueN2Stride = tilingData_->flashAttnBaseParams.valueN2Stride;
        }
        // LSE
        constInfo_.isSoftmaxLseEnable = tilingData_->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo_.dBasicBlock = BaseApi::Align64Func((uint16_t)constInfo_.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return METADATA_STRIDE * metadata_aic_num_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }
    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * metadata_aiv_num_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
        uint32_t createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        mloop_ = 0;
        headS2Split_ = false;
        tailS2Split_ = false;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx_ = bN2Cur;
        prevGS1Idx_ = gS1Cur;

        bool shouldDispatchTask = true;
        uint32_t validTaskCount = 0; // 未执行(有效)的任务数
        while (shouldDispatchTask || validTaskCount) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            if (validTaskCount) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
                if (executedTaskCount > PRELOAD_N) {
                    validTaskCount--;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        if (bN2Cur != bN2End_) {
            return bN2Cur < bN2End_;
        }
        if (gS1Cur != gS1OEnd_) {
            return gS1Cur < gS1OEnd_;
        }
        return s2Cur < s2OEnd_;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfo_.n2Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            actSeqLensQ_ = qSeqLensTool_.GetActualSeqLength(bIdx);
            actSeqLensKv_ = kvSeqLensTool_.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx_ || gS1Cur != prevGS1Idx_) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx_ = bN2Cur;
            prevGS1Idx_ = gS1Cur;
        }
        // S2有效块区间为[curS2Start_, curS2End_), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start继续计算, 不能跳行 (BAND等sparse模式curS2Start常>0)
        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start_>=curS2End_)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        s2FirstToken_ = 0;
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;

        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split_ = s2OStart_ != 0U;
            curS2Start_ = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split_ = s2OEnd_ != 0U;
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        s2FirstToken_ = 0;
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ_) * static_cast<int64_t>(constInfo_.gSize)) -
            1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT_T>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfo_.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfo_.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ_) ==
                s1GLastToken / static_cast<int64_t>(actSeqLensQ_)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ_);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ_);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
            }
        }

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            s2FirstToken_ = s2FirstToken;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2FirstToken_ = s2FirstToken;

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start_, curS2End_
        curS2Start_ = s2StartWithSparse;
        curS2End_ = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            headS2Split_ = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start_ = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            tailS2Split_ = s2OEnd_ > 0U ? true : false;
            curS2End_ = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfo &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务

        if (runInfo0.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm1(runInfo0);
            } else {
                ComputeVec1(runInfo0);
            }
        }
        if (loop >= PRELOAD_N) {
            RunInfo &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK]; // 上PRELOAD_N轮任务
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

    __aicore__ inline void ComputeMm1(RunInfo &runInfo)
    {
        cubeBlock_.IterateBmm1(runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfo &runInfo)
    {
        cubeBlock_.IterateBmm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(RunInfo &runInfo)
    {
        vecFaBlock_.ProcessVec1(runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfo &runInfo)
    {
        vecFaBlock_.ProcessVec2(runInfo);
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfo &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfo &info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.isFirstFdBlock = (s2Cur == (s2FirstToken_ / s2BaseSize));
        info.bIdx = bN2Cur / constInfo_.n2Size;
        info.n2Idx = bN2Cur % constInfo_.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo_.gSize;
            info.gIdx = info.gS1Idx % constInfo_.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ_;
            info.gIdx = info.gS1Idx / actSeqLensQ_;
        }
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ_;
        info.actS2Size = actSeqLensKv_;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfo_.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign = AttentionCommon::Align(
            (uint32_t)info.actSingleLoopS2Size, (uint32_t)(AttentionCommon::BYTE_BLOCK / sizeof(INPUT_T)));

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);
        info.actVecMSize = (info.actMSize + CV_RATIO - 1) / CV_RATIO;
        info.vecMbaseIdx = 0;
        if (constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if (bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ_ * constInfo_.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

        // 尚未到达有效区间, 快进s2Cur到curS2Start, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前S2未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }
        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }
        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!constInfo_.enableFlashDecode) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock_.template InitBuffers<VecFaBlockType::GetFdBaseOffset()>();
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        vecFdBlock_.FlashDecode(fdParams_);
        AscendC::SyncAll();
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_M_START_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            vecFaBlock_.InitBuffers();
            vecFaBlock_.InitCrossCoreSync();
            vecFaBlock_.AllocEventID();
        } else {
            cubeBlock_.InitCrossCoreSync();
            cubeBlock_.InitBuffers();
            cubeBlock_.AllocEventID();
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                FlashAttention(sectionIdx);
            }
            if ASCEND_IS_AIV {
                FlashDecode(sectionIdx);
            }
        }
        if ASCEND_IS_AIV {
            vecFaBlock_.FreeEventID();
            vecFaBlock_.UnInitCrossCoreSync();
        } else {
            cubeBlock_.FreeEventID();
            cubeBlock_.UnInitCrossCoreSync();
        }
    }
}; // FlashAttentionNoQuantGqaKernelNd

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_KERNEL_ND_H_
