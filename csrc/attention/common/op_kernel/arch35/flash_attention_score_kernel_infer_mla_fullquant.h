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
 * \file flash_attention_score_kernel_infer_mla_fullquant.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_INFER_MLA_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_KERNEL_INFER_MLA_FULLQUANT_H_
#include "flash_attention_score_kernel_base_fullquant.h"
#include "vf/vf_flash_decode.h"
#include "infer_flash_attention_comm.h"
#include "infer_flash_attention_kvcache.h"
#include "infer_flash_attention_sparse.h"

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelInferMlaFullquant : public FlashAttentionScoreKernelBaseFullquant<FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using BaseClass = FlashAttentionScoreKernelBaseFullquant<FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;

    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, 
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMainLoop();

private:
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDiv(this->constInfo.s2Size, this->constInfo.splitKVNum);
    }
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);
    
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
    }

    this->constInfo.transposeLayout = this->sharedParams.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND)) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    }

    this->constInfo.n2GD = this->constInfo.n2G * this->constInfo.dSize;
    this->constInfo.gDR = this->constInfo.gSize * this->constInfo.dSizeRope;
    this->constInfo.n2GDR = this->constInfo.n2Size * this->constInfo.gDR;
    this->constInfo.mm1RopeKa = this->constInfo.dSizeRope;
    this->constInfo.mm1Ka = this->constInfo.dSize;
    this->constInfo.attentionOutStride = 0;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if constexpr (isFd) {
        actualCoreNums = this->sharedParams.bSize * this->constInfo.n2Size *
                         this->constInfo.splitKVNum; // b * n2 * splitkv
    }

    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内 切分点
    int64_t gS1StartIdx;
    uint32_t bnEndIdx;
    uint32_t bnStartIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->sharedParams.multiCoreInnerLimit;
    if constexpr (!isFd) {
        gS1StartIdx = this->sharedParams.multiCoreInnerOffset;
        bnStartIdx = this->sharedParams.bnStartIdx;
        if (likely((this->sharedParams.coreNum - 1) > this->aicIdx)) {
            bnEndIdx = this->sharedParams.bnEndIdx;
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->sharedParams.bSize * this->constInfo.headNumRatio * this->constInfo.n2Size;
        }
    } else {  // flash decode
        gS1StartIdx = 0;
        bnStartIdx = 0;
        s2LoopLimit = 0;
        bnEndIdx = 1;
    }
    int64_t taskIdx = 0;    // 任务Id
    bool notLast = true;
    bool isLastBmm1 = false;
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;
    if constexpr (isFd) {
        runParam.n2oIdx = (this->aicIdx / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        runParam.boIdx = this->aicIdx / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }

    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParam.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam,
            this->constInfo, this->attenMaskInfo, this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam,
            this->constInfo, lastBN, nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum) > runParam.actualS2Size)
            {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize = runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize : tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1;
        }
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastTwoLoops = true;
            bool notLastThreeLoops = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoops = false;
                        break;
                    case 1:
                        notLastThreeLoops = false;
                        notLastTwoLoops = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoops = false;
                        notLastTwoLoops = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoops) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS,
                                                   BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS,
                                                      BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoops) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskIdx & 3];
                    this->SetRunInfo(runInfo1, runParam, taskIdx, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskIdx > 0 && notLastTwoLoops) {
                    if ASCEND_IS_AIV {
                        auto &runInfo3 = runInfo[(taskIdx + 3) & 3];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                            this->constInfo);
                    }
                }
                if (taskIdx > 1 && notLast) {
                    if ASCEND_IS_AIC {
                        RunInfo<isInfer> &runInfo2 = runInfo[(taskIdx + 2) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        }
                    }
                }
                if (taskIdx > 2) {
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskIdx + 1) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                        } else {
                            this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(),
                                                       runInfo3, this->constInfo);
                        }
                    }
                }
                ++taskIdx;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::Process()
{
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();  // 主循环
    if constexpr (isFd) {
        if ASCEND_IS_AIV {
            SyncAll();
            this->vecBlock.InitFDBuffers(this->constInfo);
            this->vecBlock.FlashDecodeCompute(this->constInfo, this->keyGm,
                this->actualSeqKvlenAddr);
        }
    }
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif