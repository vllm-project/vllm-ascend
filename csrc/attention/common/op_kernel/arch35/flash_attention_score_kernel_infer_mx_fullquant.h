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
 * \file flash_attention_score_kernel_infer_mx_fullquant.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_INFER_MX_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_KERNEL_INFER_MX_FULLQUANT_H_
#include "./flash_attention_score_kernel_base_fullquant.h"
#include "./vf/vf_flash_decode.h"
#include "./infer_flash_attention_comm.h"
#include "./infer_flash_attention_kvcache.h"
#include "./infer_flash_attention_sparse.h"

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelInferMxFullquant : public FlashAttentionScoreKernelBaseFullquant<FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using BaseClass = FlashAttentionScoreKernelBaseFullquant<FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;

    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMainLoop();

private:
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam);
    __aicore__ inline void UnInit();
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDiv(this->constInfo.s2Size, this->constInfo.splitKVNum);
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);

    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
    }
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
    }
    this->constInfo.transposeLayout = this->sharedParams.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND)) {
        this->constInfo.attentionOutStride = (this->constInfo.n2GDv - this->constInfo.dSizeV) *
                                             sizeof(OUTPUT_T);
    } else if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_BNSD) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_BNSD)) {
        this->constInfo.attentionOutStride = 0;
    }
    // prefix
    if constexpr (enableKVPrefix) {
        this->constInfo.prefixLoopCount = (this->constInfo.actualKVPrefixSize + this->constInfo.s2BaseSize - 1) /
                                          this->constInfo.s2BaseSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam,
        this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if constexpr (isFd) {
        // b * n2 * splitkv
        actualCoreNums = this->sharedParams.bSize * this->constInfo.n2Size * this->constInfo.splitKVNum;
    }
    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    int64_t s2LoopLimit;
    uint32_t bnEndIdx;
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
            bnEndIdx = this->constInfo.headNumRatio * this->sharedParams.bSize * this->constInfo.n2Size;
        }
    } else {    // FD
        gS1StartIdx = 0;
        bnStartIdx = 0;
        s2LoopLimit = 0;
        bnEndIdx = 1;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    LocalTensor<INPUT_T> scmTensor[2];
    RunInfo<isInfer> runInfo[3];
    RunParamStr<isInfer> runParams;

    if constexpr (isFd) {
        runParams.boIdx = this->aicIdx / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParams.n2oIdx = (this->aicIdx / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParams.boIdx * this->constInfo.n2Size + runParams.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }

    int64_t coreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParams.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParams.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParams,
            this->constInfo, this->attenMaskInfo, this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParams,
            this->constInfo, lastBN, nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum) >
                runParams.actualS2Size) {
                runParams.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParams.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParams.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            runParams.s1LoopTimes = 1;
        }
        int64_t tempGS1End = lastBN ? (runParams.s1LoopTimes + 2) : runParams.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastTwoLoop = true;
            bool notLast = true;

            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParams.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
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
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParams);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParams, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS,
                                                      BaseClass::useDn,
                                                      BaseClass::enableKVPrefix>(
                    runParams, this->constInfo);
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParams.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastTwoLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParams, taskId, s2LoopCount, s2LoopLimit, coreInnerIdx);
                    int32_t c1v1Loop = CeilDiv(runInfo1.s2RealSize, 256);
                    for (int32_t subLoopIdx = 0; subLoopIdx < c1v1Loop; ++subLoopIdx) {
                        if ASCEND_IS_AIC {
                            this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo, subLoopIdx);
                        }
                        if ASCEND_IS_AIV {
                            if (subLoopIdx % 2 == 0) {
                                this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo1,
                                    this->constInfo, subLoopIdx);
                            } else {
                                this->vecBlock.ProcessVec1(this->l1PBuffers.GetPre(), this->bmm1Buffers.Get(), runInfo1,
                                    this->constInfo, subLoopIdx);
                            }
                        }
                    }
                }
                if (taskId > 1) {
                    RunInfo<isInfer> &runInfoTmp = runInfo[(taskId - 2) & 3];
                    if ASCEND_IS_AIC {
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfoTmp,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfoTmp,
                                this->constInfo);
                        }
                    }
                    if ASCEND_IS_AIV {
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfoTmp, this->constInfo);
                        } else {
                            this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfoTmp, this->constInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++coreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>::Process()
{
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();  // 主循环
    if constexpr (isFd) {
        if ASCEND_IS_AIV {
            SyncAll();
            this->vecBlock.InitFDBuffers(this->constInfo);
            this->vecBlock.FlashDecodeCompute(this->constInfo,
                                              this->keyGm, this->actualSeqKvlenAddr);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>::UnInit()
{
    if ASCEND_IS_AIC {
        this->cubeBlock.UnInit();
    }
    this->l1PBuffers.Uninit(this->l1BufferManager);
    if ASCEND_IS_AIV {
        this->vecBlock.UnInit();
    }
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferMxFullquant<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParams)
{
    constexpr uint64_t fp8QBlockSize = 128U; // 128 is SOuterSize
    constexpr uint64_t fp8KvBlockSize = 256U; // 256 is SInnerSize
    if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        if (runParams.boIdx == 0 || runParams.boIdx == 1) {
            this->s1ScaleNumAcc = runParams.boIdx == 0 ? 0 : CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = runParams.boIdx == 0 ? 0 : CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            this->s1SizeAcc = runParams.boIdx == 0 ? 0 : this->actualSeqQlenAddr[0];
            this->s2SizeAcc = runParams.boIdx == 0 ? 0 : this->actualSeqKvlenAddr[0];
        } else {
            this->s1ScaleNumAcc = CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            for (uint32_t boIdx = 1; boIdx < runParams.boIdx; boIdx++) {
                this->s1ScaleNumAcc += CeilDiv(this->actualSeqQlenAddr[boIdx] - this->actualSeqQlenAddr[boIdx - 1], fp8QBlockSize);
                this->s2ScaleNumAcc += CeilDiv(this->actualSeqKvlenAddr[boIdx] - this->actualSeqKvlenAddr[boIdx - 1], fp8KvBlockSize);
            }
            this->s1SizeAcc = this->actualSeqQlenAddr[runParams.boIdx - 1];
            this->s2SizeAcc = this->actualSeqKvlenAddr[runParams.boIdx - 1];
        }
    }
    // GS1合轴时，g轴信息包含在gS1中；GS1不合轴时，g轴信息包含在bn2g中；
    if (this->constInfo.isGqa) {
        runParams.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParams.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParams.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif