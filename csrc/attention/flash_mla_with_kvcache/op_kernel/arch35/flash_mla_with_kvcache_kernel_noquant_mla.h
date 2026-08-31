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
 * \file flash_mla_with_kvcache_kernel_noquant_mla.h
 * \brief arch35 flash_mla_with_kvcache 非量化 MLA kernel（由 fia_kernel_noquant_mla.h 复制改名；
 *        metadata 消费改为 flash_attn AICPU 多 section 布局，与 flash_attn 同构）
 */

#ifndef FLASH_MLA_WITH_KVCACHE_KERNEL_NOQUANT_MLA_H_
#define FLASH_MLA_WITH_KVCACHE_KERNEL_NOQUANT_MLA_H_

#include "flash_mla_with_kvcache_public_define_arch35.h"
#include "flash_mla_with_kvcache_block_cube_noquant_mla.h"
#include "flash_mla_with_kvcache_block_vec_noquant_mla.h"
#include "memory_copy_arch35.h"
#include "flash_mla_with_kvcache_block_vec_flashdecode_mla.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_mla_with_kvcache_tiling_data.h"

namespace FlashAttnKernel {
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantMlaKernel {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2 C2
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N << 1; // 4 (power-of-2 for bitmask)

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;
    static constexpr bool FLASH_DECODE = VecFaBlockType::FLASH_DECODE;
    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using OUT_T = typename VecFaBlockType::OUT_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    // ---- CV buffers：已静态化至各 block（cube/vec InitBuffers 内表 1 布局），kernel 不再持有 buffer manager ----

    // 多 section AICPU metadata（flash_attn 布局）：
    // 字节布局 = [16-word header][sectionNum*36*16 (FA)][sectionNum*72*16 (FD)]，FA/FD 区分别绑定 GM tensor。
    AscendC::GlobalTensor<uint32_t> faMetaDataGm_;
    AscendC::GlobalTensor<uint32_t> fdMetaDataGm_;
    uint32_t sectionNum_ = 0U;
    FlashMlaWithKvcacheFdParamsX fdParams_ = {};
    // 与 producer（flash_mla_metadata，flash_attn_metadata 字节兼容克隆）的核 strides 一致：（producer
    // 旧名：flash_mla_metadata，历史原因保留） AICPU 侧 FA_METADATA_STRIDE=16 / FD_METADATA_STRIDE=16 /
    // HEAD_METADATA_STRIDE=16， producer 侧 static_assert 已就位于
    // flash_mla_metadata.h（本注释不再复制断言，避免漂移）。（producer 旧名：flash_mla_metadata，历史原因保留）
    static_assert(
        FLASH_ATTN_METADATA_SIZE == 16 && FA_FD_METADATA_SIZE == 16,
        "flash_mla_with_kvcache: metadata per-core stride must be 16 to match flash_mla_metadata AICPU layout");

    ConstInfoX constInfo_;
    FlashMlaSeqLensTool<Q_MODE, KV_MODE> seqLensTool_;

    const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tilingData_;
    // 静态 tensor 模型：block 内自持静态 buffer + Mutex/cross-core 显式同步
    // （与 flash_attn Init 15 参风格一致，入口不进 buffer/pipe 细节）
    CubeBlockType cubeBlock_;
    VecFaBlockType vecFaBlock_;
    VecFdBlockType vecFdBlock_;

    // schduler params
    int64_t validTaskNum_ = 0;
    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    uint64_t cachedS2LoopTimes_ = 0;
    uint64_t cachedG1S1LoopTimes_ = 0;
    uint32_t curS2Start_ = 0;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_ = 0;
    uint32_t prevBN2Idx_ = 0;
    uint32_t prevGS1Idx_ = 0;
    uint32_t mloop_ = 0;
    bool headS2Split_ = false;
    bool tailS2Split_ = false;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantMlaKernel()
        : cubeBlock_(constInfo_, seqLensTool_),
          vecFaBlock_(constInfo_, seqLensTool_),
          vecFdBlock_(constInfo_, seqLensTool_){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *kCache,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                __gm__ uint8_t *cacheSeqlens, __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *metadata,
                                const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tiling)
    {
        this->tilingData_ = tiling;

        // metadata（GM 输入）：section 数在 16-word header 第 0 个 uint32
        sectionNum_ = ((__gm__ uint32_t *)metadata)[0];
        // FA 区从 header 后开始（与 flash_attn 同构，参见 flash_attn_kernel_nd.h metadata 消费段）
        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t *)(metadata + FA_METADATA_HEADER_OFFSET),
                                      FA_AIC_CORE_NUM * FLASH_ATTN_METADATA_SIZE * sectionNum_);
        fdMetaDataGm_.SetGlobalBuffer(
            (__gm__ uint32_t *)(metadata + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * FA_FD_METADATA_SIZE * sectionNum_);

        InitConstInfo();

        // seq-lens 全部 INT32（ACTLEN_T=uint32_t），无 int64 GM 物化、无预核 pass；
        // parser 所有权收敛到 kernel 层（FlashMlaSeqLensTool），block 只读引用
        seqLensTool_.InitQ(cuSeqlensQ, sequsedQ, constInfo_.actualSeqLenSize, constInfo_.s1Size);
        seqLensTool_.InitKv(cacheSeqlens, constInfo_.actualSeqLenKVSize, constInfo_.s2Size);

        if ASCEND_IS_AIV {
            vecFaBlock_.InitVecBlock(attenMask, softmaxLse, attentionOut, workspace);
            vecFaBlock_.ClearOutput();
        }

        if ASCEND_IS_AIC {
            cubeBlock_.InitCubeBlock(query, kCache, blockTable);
        }

        if constexpr (FLASH_DECODE) {
            if ASCEND_IS_AIV {
                // FA fd（Mutex 版）Init 契约：InitBlock(learnableSink=nullptr, softmaxLse, attentionOut)
                // 建 outGmTensor_/softmaxLseGm_/dSizeV_Align_；InitGlobalTensor 仅接 FD 读的 accum/LSE GM。
                vecFdBlock_.InitBlock(nullptr, softmaxLse, attentionOut);
                vecFdBlock_.InitGlobalTensor(this->vecFaBlock_.softmaxFDMaxGm, this->vecFaBlock_.softmaxFDSumGm,
                                             this->vecFaBlock_.accumOutGm);
            }
        }
    }

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = AscendC::GetBlockIdx();
        } else {
            constInfo_.aivIdx = AscendC::GetBlockIdx();
            constInfo_.aicIdx = constInfo_.aivIdx / AscendC::GetSubBlockNum();
            constInfo_.subBlockIdx = AscendC::GetSubBlockIdx();
        }

        const auto &flashMlaWithKvcacheBaseParams = this->tilingData_->flashMlaWithKvcacheBaseParams;
        const auto &flashMlaWithKvcacheAttenMaskParams = this->tilingData_->flashMlaWithKvcacheAttenMaskParams;
        const auto &flashMlaWithKvcachePageAttentionParams = this->tilingData_->flashMlaWithKvcachePageAttentionParams;
        const auto &flashMlaWithKvcacheWorkspaceParams = this->tilingData_->flashMlaWithKvcacheWorkspaceParams;
        const auto &flashMlaWithKvcacheEmptyTensorParams = this->tilingData_->flashMlaWithKvcacheEmptyTensorParams;

        constInfo_.bSize = flashMlaWithKvcacheBaseParams.bSize;
        constInfo_.t1Size = flashMlaWithKvcacheBaseParams.t1Size;
        constInfo_.t2Size = flashMlaWithKvcacheBaseParams.t2Size;
        constInfo_.n2Size = flashMlaWithKvcacheBaseParams.n2Size;
        constInfo_.gSize = flashMlaWithKvcacheBaseParams.gSize;
        constInfo_.s1Size = flashMlaWithKvcacheBaseParams.s1Size;
        constInfo_.s2Size = flashMlaWithKvcacheBaseParams.s2Size;
        constInfo_.dSize = flashMlaWithKvcacheBaseParams.dSize;
        constInfo_.dSizeV = flashMlaWithKvcacheBaseParams.dSizeV;
        // rope 段宽度由 host 从 q last dim - headDimV 推导后经 tiling data 传入
        // （当前 MLA D512 仅支持 rope==64，见 tiling_info_parser GetQkHeadDim）
        constInfo_.dSizeRope = flashMlaWithKvcacheBaseParams.dSizeRope;
        constInfo_.actualSeqLenSize = flashMlaWithKvcacheBaseParams.actualSeqLengthsQSize;
        constInfo_.actualSeqLenKVSize = flashMlaWithKvcacheBaseParams.actualSeqLengthsKVSize;
        constInfo_.scaleValue = flashMlaWithKvcacheBaseParams.scaleValue;
        constInfo_.coreNum = flashMlaWithKvcacheBaseParams.coreNum;
        constInfo_.outputLayout =
            static_cast<FLASH_MLA_WITH_KVCACHE_LAYOUT>(flashMlaWithKvcacheBaseParams.outputLayout);

        constInfo_.keyStrides.bnStride = flashMlaWithKvcacheBaseParams.keyStrides.bnStride;
        constInfo_.keyStrides.n2Stride = flashMlaWithKvcacheBaseParams.keyStrides.n2Stride;
        constInfo_.valueStrides.bnStride = flashMlaWithKvcacheBaseParams.valueStrides.bnStride;
        constInfo_.valueStrides.n2Stride = flashMlaWithKvcacheBaseParams.valueStrides.n2Stride;

        constInfo_.sparseMode = flashMlaWithKvcacheAttenMaskParams.sparseMode;
        constInfo_.preTokens = flashMlaWithKvcacheAttenMaskParams.preTokens;
        constInfo_.nextTokens = flashMlaWithKvcacheAttenMaskParams.nextTokens;
        if constexpr (HAS_MASK) {
            constInfo_.attenMaskBatch = flashMlaWithKvcacheAttenMaskParams.attenMaskBatch;
            constInfo_.attenMaskS1Size = flashMlaWithKvcacheAttenMaskParams.attenMaskS1Size;
            constInfo_.attenMaskS2Size = flashMlaWithKvcacheAttenMaskParams.attenMaskS2Size;
        }
        constInfo_.isRowInvalidOpen = flashMlaWithKvcacheAttenMaskParams.isRowInvalidOpen;
        constInfo_.isExistRowInvalid = flashMlaWithKvcacheAttenMaskParams.isExistRowInvalid;
        constInfo_.accumOutSize = flashMlaWithKvcacheWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = flashMlaWithKvcacheWorkspaceParams.logSumExpSize;
        // pageAttention（仅 PA 实例化：SEL 不含 NO_PA，host IsCapableFeatureCheckMla 亦强制 PA 路由）
        constInfo_.maxBlockNumPerBatch = flashMlaWithKvcachePageAttentionParams.maxBlockNumPerBatch;
        constInfo_.blockSize = flashMlaWithKvcachePageAttentionParams.blockSize;
        constInfo_.paLayoutType = flashMlaWithKvcachePageAttentionParams.paLayoutType;
        // LSE
        constInfo_.isSoftmaxLseEnable = flashMlaWithKvcacheBaseParams.isSoftMaxLseEnable;

        // 每 section 的 FA 任务区间由 FlashAttention(sectionIdx) 内 GetFASectionInfo
        // 从 AICPU metadata 读取（多 section 0 基布局，无 S1 外切）
        constInfo_.dBasicBlock = AttentionCommon::Align(constInfo_.dSizeV, 64U);
        constInfo_.kRopeStrides.bnStride = flashMlaWithKvcacheBaseParams.kRopeStrides.bnStride;
        constInfo_.kRopeStrides.n2Stride = flashMlaWithKvcacheBaseParams.kRopeStrides.n2Stride;
    }

    // 0 基、16 字段/核、无 CORE_ENABLE（flash_attn_kernel_nd.h:206-214 镜像）
    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * sectionIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    // 单 section FA 任务区间读取（重映射：FIA 8 字段 CORE_ENABLE 布局 -> AICPU 16 字段 0 基布局，
    // FA 各字段索引 -1；FA_CORE_ENABLE 删除，核活跃由区间隐含）
    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        constInfo_.bN2Start =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        constInfo_.gS1OStart =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        constInfo_.s2OStart =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        constInfo_.bN2End =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        constInfo_.gS1OEnd =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        constInfo_.s2OEnd =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        constInfo_.coreFirstTmpOutWsPos = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    // 单 section FD 参数读取；FD 核活跃 = mLen > 0（无 CORE_ENABLE 字段，flash_attn_kernel_nd.h:610-611）
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

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfo_.aicIdx >= constInfo_.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);

        FlashMlaWithKvcacheRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        uint32_t bN2Cur = constInfo_.bN2Start;
        uint32_t gS1Cur = constInfo_.gS1OStart;
        uint32_t s2Cur = constInfo_.s2OStart;
        prevBN2Idx_ = bN2Cur;
        prevGS1Idx_ = gS1Cur;

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
                } else {
                    // DEAL_ZERO / SKIP_ZERO / SKIP 等非建任务模式：统一推进轴信息
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
        if (bN2Cur != constInfo_.bN2End) {
            return bN2Cur < constInfo_.bN2End;
        }
        if (gS1Cur != constInfo_.gS1OEnd) {
            return gS1Cur < constInfo_.gS1OEnd;
        }
        return s2Cur < constInfo_.s2OEnd;
    }

    __aicore__ inline bool ShouldExecuteTask(FlashMlaWithKvcacheRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        return validTaskNum_ > 0;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask =
            (bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart) && (s2Cur == constInfo_.s2OStart);
        uint32_t bIdx = bN2Cur / constInfo_.n2Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            actSeqLensKv_ = seqLensTool_.kvActSeqLensParser.GetActualSeqLength(bIdx);
            actSeqLensQ_ = seqLensTool_.qActSeqLensParser.GetActualSeqLength(bIdx);
            cachedS2LoopTimes_ = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
            uint64_t gS1Size = actSeqLensQ_ * constInfo_.gSize;
            cachedG1S1LoopTimes_ = (gS1Size + mBaseSize - 1) / mBaseSize;
        }

        if (cachedS2LoopTimes_ == 0 || cachedG1S1LoopTimes_ == 0) {
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

        if (curS2Start_ >= curS2End_) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur < curS2Start_) {
            return TASK_DEAL_MODE::NOT_START;
        }

        if (s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ_, int64_t actSeqLensKv_, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp += static_cast<int64_t>(actSeqLensQ_) - static_cast<int64_t>(actSeqLensKv_);
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp += static_cast<int64_t>(actSeqLensKv_) - static_cast<int64_t>(actSeqLensQ_);
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart)) {
            headS2Split_ = constInfo_.s2OStart != 0U;
            curS2Start_ = constInfo_.s2OStart;
        }

        if ((bN2Cur == constInfo_.bN2End) && (gS1Cur == constInfo_.gS1OEnd)) {
            tailS2Split_ = constInfo_.s2OEnd != 0U;
            curS2End_ = constInfo_.s2OEnd;
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
        int64_t s1GLastToken =
            AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                 static_cast<int64_t>(actSeqLensQ_) * static_cast<int64_t>(constInfo_.gSize)) -
            1;

        if constexpr (GetOutUbFormat<LAYOUT_Q>() == UbFormat::S1G) {
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
                s1LastToken = static_cast<int64_t>(actSeqLensQ_);
                s1FirstToken = 0;
            }
        }
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);
        // 3. trans index of token to index of block
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));

        // 4. Calc curS2Start_, curS2End_
        curS2Start_ = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        curS2End_ = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        if (bN2Cur == constInfo_.bN2Start && gS1Cur == constInfo_.gS1OStart) { // first line
            headS2Split_ = constInfo_.s2OStart > curS2Start_ ? true : false;
            curS2Start_ = AttentionCommon::Max(curS2Start_, constInfo_.s2OStart);
        }
        if (bN2Cur == constInfo_.bN2End && gS1Cur == constInfo_.gS1OEnd) { // last line
            tailS2Split_ = constInfo_.s2OEnd > 0U ? true : false;
            curS2End_ = constInfo_.s2OEnd > 0U ? AttentionCommon::Min(curS2End_, constInfo_.s2OEnd) : curS2End_;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, FlashMlaWithKvcacheRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        FlashMlaWithKvcacheRunInfoX &runInfo0 = taskRunInfo[loop & (PRELOAD_TASK_CACHE_SIZE - 1)]; // 本轮任务
        FlashMlaWithKvcacheRunInfoX &runInfoNegN =
            taskRunInfo[(loop - PRELOAD_N) & (PRELOAD_TASK_CACHE_SIZE - 1)]; // 上PRELOAD_N轮任务
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

    __aicore__ inline void ComputeMm1(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm1(runInfo);
    }

    __aicore__ inline void ComputeMm2(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        cubeBlock_.IterateBmm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        vecFaBlock_.ProcessVec1(runInfo);
    }

    __aicore__ inline void ComputeVec2(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        this->vecFaBlock_.ProcessVec2(runInfo);
    }

    __aicore__ inline void EnableTask(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        runInfo.isValid = true;
        validTaskNum_++;
    }

    __aicore__ inline void DisableTask(FlashMlaWithKvcacheRunInfoX &runInfo)
    {
        runInfo.isValid = false;
        validTaskNum_--;
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      FlashMlaWithKvcacheRunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        FlashMlaWithKvcacheRunInfoX &runInfo = taskRunInfo[loop & (PRELOAD_TASK_CACHE_SIZE - 1)]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        EnableTask(runInfo);
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      FlashMlaWithKvcacheRunInfoX &info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.bIdx = bN2Cur / constInfo_.n2Size;
        info.n2Idx = bN2Cur % constInfo_.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_BSH || LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            info.s1Idx = info.gS1Idx / constInfo_.gSize;
        } else {
            // GS1 layout
            info.s1Idx = info.gS1Idx % actSeqLensQ_;
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
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(BaseApi::FA_BYTE_BLOCK / sizeof(INPUT_T)));
        info.isChangeBatch = false;

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bN2Start, gS1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = constInfo_.coreFirstTmpOutWsPos;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);
        info.actVecMSize = (info.actMSize + 1) >> 1;
        info.vecMbaseIdx = 0;
        if (constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if ((constInfo_.bN2Start == constInfo_.bN2End && constInfo_.gS1OStart == constInfo_.gS1OEnd)) {
            // 所有任务属于同一个S1G
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bN2Cur == constInfo_.bN2Start) && (gS1Cur == constInfo_.gS1OStart)) {
                // 当前任务属于第一个S1G, 并且第一个S1G的S2被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bN2Cur == constInfo_.bN2End) && (gS1Cur == constInfo_.gS1OEnd)) {
                // 当前任务属于最后一个S1G, 并且最后一个S1G的S2被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        } else if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
            s2Cur++;
            return;
        }

        // 当前BN2未处理完
        s2Cur = 0;

        uint64_t gS1LoopTimes = cachedG1S1LoopTimes_;
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
        GetFDSectionInfo(sectionIdx);
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        vecFdBlock_.InitBuffers();
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        vecFdBlock_.FlashDecode(fdParams_);
        AscendC::SyncAll();
    }

    __aicore__ inline void Process()
    {
        // 编排镜像 flash_attn_kernel_nd.h:626-652：AIV 先 InitBuffers→InitCrossCoreSync→AllocEventID；
        // AIC 先 InitCrossCoreSync→InitBuffers→AllocEventID
        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.InitBuffers();
                vecFaBlock_.InitCrossCoreSync();
                vecFaBlock_.AllocEventID();
            } else {
                cubeBlock_.InitCrossCoreSync();
                cubeBlock_.InitBuffers();
                cubeBlock_.AllocEventID();
            }
        }

        // 多 section 流水：逐 section 读 AICPU metadata 并执行 FA/FD（镜像 flash_attn_kernel_nd.h:637-644）
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                FlashAttention(sectionIdx);
            }

            if constexpr (FLASH_DECODE) {
                if ASCEND_IS_AIV {
                    FlashDecode(sectionIdx);
                }
            }
        }

        if (constInfo_.aicIdx < constInfo_.coreNum) {
            if ASCEND_IS_AIV {
                vecFaBlock_.FreeEventID();
                vecFaBlock_.UnInitCrossCoreSync();
            } else {
                cubeBlock_.FreeEventID();
                cubeBlock_.UnInitCrossCoreSync();
            }
        }
    }
}; // FlashAttentionNoQuantMlaKernel

} // namespace FlashAttnKernel

#endif // FLASH_MLA_WITH_KVCACHE_KERNEL_NOQUANT_MLA_H_
