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
 * \file flash_attention_noquant_kernel_infer.h
 * \brief
 */

#ifndef FLASH_ATTENTION_NOQUANT_KERNEL_INFER_H_
#define FLASH_ATTENTION_NOQUANT_KERNEL_INFER_H_
#include "./flash_attention_noquant_kernel_base.h"
#include "./vf/vf_flash_decode.h"
#include "./infer_flash_attention_comm.h"
#include "./infer_flash_attention_kvcache.h"
#include "./infer_flash_attention_sparse.h"

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionNoQuantKernelInfer : public FlashAttentionNoQuantKernelBase<FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value && !IsSameType<OUTPUT_T, float>::value;
    using BaseClass = FlashAttentionNoQuantKernelBase<FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;
    /* =====================UB变量==================== */
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, 
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessS2Loop(RunParamStr<isInfer> &runParam, RunInfo<isInfer> *runInfo, int64_t s2LoopLimit,
        int64_t multiCoreInnerIdx, int64_t &taskId, bool notLastThreeLoop, bool notLastTwoLoop, bool notLast);
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void ProcessS1OutSplitLoop();
    __aicore__ inline int64_t CalcRealCoreIdxVarlen(int64_t calcLoops, int64_t calcLoopsRemain, int64_t cycleCoreNums);
    __aicore__ inline void CalS1OuterSize(const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam);
    __aicore__ inline int64_t CalS1RealSize (const int64_t &actualS1Len, const int64_t &actualS2Len);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam);

private:
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                    RunParamStr<isInfer> &runParam);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDiv(this->constInfo.s2Size, this->constInfo.splitKVNum);
    }
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = this->sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = this->sharedParams.isPostQuantBF16;
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);
    // pageAttention
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
        this->constInfo.keyBnStride = this->sharedParams.keyBnStride;
        this->constInfo.keyN2Stride = this->sharedParams.keyN2Stride;
        this->constInfo.valueBnStride = this->sharedParams.valueBnStride;
        this->constInfo.valueN2Stride = this->sharedParams.valueN2Stride;
        this->constInfo.keyRopeBnStride = this->sharedParams.keyRopeBnStride;
        this->constInfo.keyRopeN2Stride = this->sharedParams.keyRopeN2Stride;
    }

    this->constInfo.transposeLayout = this->sharedParams.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND)) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    } else if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_BNSD) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_BNSD)) {
        this->constInfo.attentionOutStride = 0;
    }

    // prefix
    if constexpr (enableKVPrefix) {
        this->constInfo.prefixLoopCount = (this->constInfo.actualKVPrefixSize + this->constInfo.s2BaseSize - 1) / this->constInfo.s2BaseSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::ProcessS2Loop(
    RunParamStr<isInfer> &runParam, RunInfo<isInfer> *runInfo, int64_t s2LoopLimit, int64_t multiCoreInnerIdx,
    int64_t &taskId, bool notLastThreeLoop, bool notLastTwoLoop, bool notLast)
{
    for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
        if (notLastThreeLoop) {
            RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
            this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
            if ASCEND_IS_AIC {
                this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
            }
        }
        if (taskId > 0 && notLastTwoLoop) {
            if ASCEND_IS_AIV {
                auto &runInfo3 = runInfo[(taskId + 3) & 3];
                this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                    this->constInfo);
            }
        }
        if (taskId > 1 && notLast) {
            if ASCEND_IS_AIC {
                RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                if constexpr (BaseClass::bmm2Write2Ub) {
                    this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                        this->constInfo);
                } else {
                    this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                        this->constInfo);
                }
            }
        }
        if (taskId > 2) {
            if ASCEND_IS_AIV {
                RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                if constexpr (BaseClass::bmm2Write2Ub) {
                    this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                } else {
                    this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                }
            }
        }
        ++taskId;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if constexpr (isFd) {
        actualCoreNums = this->sharedParams.bSize * this->constInfo.n2Size *
                         this->constInfo.splitKVNum; // b * n2 * splitkv
    }
    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->sharedParams.multiCoreInnerLimit;
    if constexpr (!isFd) {
        bnStartIdx = this->sharedParams.bnStartIdx;
        gS1StartIdx = this->sharedParams.multiCoreInnerOffset;
        if (likely((this->sharedParams.coreNum - 1) > this->aicIdx)) {
            bnEndIdx = this->sharedParams.bnEndIdx;
            if (nextGs1Idx != 0) {
                bnEndIdx++;
            }
        } else {
            bnEndIdx = this->sharedParams.bSize * this->constInfo.n2Size *
                this->constInfo.headNumRatio;
        }
    } else {
        gS1StartIdx = 0;
        bnStartIdx = 0;
        bnEndIdx = 1;
        s2LoopLimit = 0;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    LocalTensor<INPUT_T> scmTensor[2];
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;

    if constexpr (isFd) {
        runParam.boIdx = this->aicIdx / (this->constInfo.n2Size * this->constInfo.splitKVNum);
        runParam.n2oIdx = (this->aicIdx / this->constInfo.splitKVNum) % this->constInfo.n2Size;
        bnStartIdx = runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx;
        bnEndIdx = bnStartIdx + 1;
    }
    // 注意这里不等于0是因为，推理的在SetRunInfo中第一次也需要赋值runInfo.s1oIdx，boIdx，n2oIdx，goIdx
    // 训练这些值在multiCoreInnerIdx = 0的时候都是0，两边不统一
    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        if constexpr (!isFd) {
            runParam.boIdx = bnIdx / (this->constInfo.n2Size * this->constInfo.headNumRatio);
            runParam.n2oIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.n2Size;
        }
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, lastBN,
                                                                      nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum) >
                runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize =
                    runParam.actualS2Size -
                    this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum);
                runParam.s2LineEndIdx = tailSInnerLoopSize > this->constInfo.sInnerLoopSize ?
                                        this->constInfo.sInnerLoopSize :
                                        tailSInnerLoopSize;
            }
            runParam.s1LoopTimes = 1;
        }
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            ProcessS2Loop(runParam, runInfo, s2LoopLimit, multiCoreInnerIdx, taskId, notLastThreeLoop, notLastTwoLoop, notLast);
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::CalcRealCoreIdxVarlen(
    int64_t calcLoops, int64_t calcLoopsRemain, int64_t cycleCoreNums)
{
    int64_t realCoreIdx = 0;
    if (calcLoopsRemain == 0) {
        realCoreIdx = calcLoops * cycleCoreNums + this->aicIdx;
    } else {
        realCoreIdx = calcLoops * cycleCoreNums + (cycleCoreNums - this->aicIdx - 1);
    }

    return realCoreIdx;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::CalS1RealSize (
    const int64_t &actualS1Len, const int64_t &actualS2Len) {
    if constexpr (hasAtten) {
        if (this->attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            return CeilDiv(Min(actualS1Len, actualS2Len), this->s1BaseSize);
        }
    }

    return CeilDiv(actualS1Len, this->s1BaseSize);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::CalS1OuterSize(
    const int64_t &multiCoreInnerOffset, RunParamStr<isInfer> &runParam)
{
    int64_t actualS1Outersize = 0;
    this->s1OuterSizeAcc = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;
    runParam.boIdx = 0;
    runParam.b1SSOffset = 0;
    runParam.b1SSOffsetAlign16 = 0;

    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->sharedParams.bSize; ++i) {
        this->GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        int64_t tmpS1Outersize = CalS1RealSize(actualS1Len, actualS2Len);
        actualS1Outersize += (tmpS1Outersize * this->constInfo.n2G);
        if (multiCoreInnerOffset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            runParam.b1SSOffset += actualS1Len * actualS2Len;
            runParam.b1SSOffsetAlign16 += actualS1Len * Align(actualS2Len);
            runParam.boIdx++;
            continue;
        }
        break;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam)
{
    this->GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
    int64_t tmpS1Outersize = CalS1RealSize(runParam.actualS1Size, runParam.actualS2Size);
    int64_t actualS1Outersize = this->s1OuterSizeAcc + tmpS1Outersize * this->constInfo.n2G;

    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += runParam.actualS1Size;
        this->s2SizeAcc += runParam.actualS2Size;
        runParam.b1SSOffset += runParam.actualS1Size * runParam.actualS2Size;
        runParam.boIdx++;
        if (runParam.boIdx >= this->sharedParams.bSize) {
            break;
        }
        this->GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
        tmpS1Outersize = CalS1RealSize(runParam.actualS1Size, runParam.actualS2Size);
        actualS1Outersize = this->s1OuterSizeAcc + tmpS1Outersize * this->constInfo.n2G;
    }

    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    runParam.n2oIdx = (actualS1Outersize / tmpS1Outersize) / this->sharedParams.gSize;
    runParam.goIdx = (actualS1Outersize / tmpS1Outersize) % this->sharedParams.gSize;
    runParam.s1oIdx = actualS1Outersize % tmpS1Outersize;

    runParam.s1RealSize = CalS1RealSize(runParam.actualS1Size, runParam.actualS2Size);
    runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (this->constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::ProcessS1OutSplitLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if (this->aicIdx >= actualCoreNums) {
        return;
    }

    int64_t multiCoreInnerLimit = 0;
    int64_t varlenCalcLoops = 0;                    // 需要进行计算的循环次数(正序+倒序为一次循环)
    int64_t varlenCalcLoopsRemain = 0;
    int64_t varlenCalcTimes = 0;                    // 需要计算的S1方向上基本块总数
    int64_t varlenCycleCoreNums = this->sharedParams.coreNum * 2; // 一次循环正序+倒序为两倍核数
 
    varlenCalcLoops = this->sharedParams.totalSize / varlenCycleCoreNums;
    varlenCalcLoopsRemain = this->sharedParams.totalSize % varlenCycleCoreNums;
    varlenCalcTimes = varlenCalcLoops * 2;
    if (varlenCalcLoopsRemain >= this->aicIdx + 1) {
        varlenCalcTimes++;
        if (varlenCalcLoopsRemain > this->sharedParams.coreNum &&
            (this->aicIdx + 1) > varlenCycleCoreNums - varlenCalcLoopsRemain) {
            varlenCalcTimes++;
        }
    }
    multiCoreInnerLimit = varlenCalcTimes;

    // 初始化AxisIdx
    RunParamStr<isInfer> runParam;
    CalS1OuterSize(this->aicIdx, runParam);

    RunInfo<isInfer> runInfo[4];
    int64_t taskId = 0;
    bool notThirdLast = true;
    bool notSecondLast = true;
    bool notLast = true;
    int64_t thirdLast = multiCoreInnerLimit;
    int64_t secondLast = multiCoreInnerLimit + 1;
    int64_t last = multiCoreInnerLimit + 2;
    multiCoreInnerLimit += 3;
    int64_t realCoreInnerIdx = 0;
    for (int64_t multiCoreInnerIdx = 0; multiCoreInnerIdx < multiCoreInnerLimit; multiCoreInnerIdx++) {
        if (multiCoreInnerIdx == secondLast) {
            notSecondLast = false;
        } else if (multiCoreInnerIdx == last) {
            notLast = false;
        } else if (multiCoreInnerIdx == thirdLast) {
            notThirdLast = false;
        } else {
            // 非最后三次伪循环，需要将当前核处理的次数转化为S1方向上基本块的索引值
            int64_t curCalcLoops = multiCoreInnerIdx >> 1;
            int64_t curCalcLoopsRemain = multiCoreInnerIdx & 1;
            realCoreInnerIdx = CalcRealCoreIdxVarlen(curCalcLoops, curCalcLoopsRemain, varlenCycleCoreNums);
        }

        int64_t s2LoopLimit = 0;
        bool notLastThreeLoop = notThirdLast && notSecondLast && notLast;
        bool notLastTwoLoop = notSecondLast && notLast;
        if (notLastThreeLoop) {
            this->ComputeAxisIdx(realCoreInnerIdx, runParam);
            ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
                this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
            int64_t gS1Index = 0;
            if (this->constInfo.isGqa) {
                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND ||
                              layout == LayOutTypeEnum::LAYOUT_BSH ||
                              layout == LayOutTypeEnum::LAYOUT_SBH) {
                    gS1Index = runParam.s1oIdx * this->constInfo.gSize + runParam.goIdx;
                } else {
                    gS1Index = runParam.goIdx * runParam.actualS1Size + runParam.s1oIdx;
                }
                runParam.gS1Idx = gS1Index;
            } else {
                gS1Index = runParam.s1oIdx;
            }
            bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);

            bool s2NoNeedCalc = ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo);
            if (s1NoNeedCalc || s2NoNeedCalc) {
                continue;
            }

            s2LoopLimit = CeilDiv(runParam.s2LineEndIdx - runParam.s2LineStartIdx, this->s2BaseSize) - 1;
        }
        ProcessS2Loop(runParam, runInfo, s2LoopLimit, multiCoreInnerIdx, taskId, notLastThreeLoop, notLastTwoLoop, notLast);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    if constexpr (enableS1OutSplit) {
        ProcessS1OutSplitLoop();
    } else {
        ProcessMainLoop();
    }
    if constexpr (isFd) {
        if ASCEND_IS_AIV {
            SyncAll();
            this->vecBlock.InitFDBuffers(this->constInfo);
            this->vecBlock.FlashDecodeCompute(this->constInfo, this->keyGm, this->actualSeqQlenAddr,
                                              this->actualSeqKvlenAddr);
        }
    }
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionNoQuantKernelInfer<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    constexpr uint64_t fp8QBlockSize = 128U; // 128 is SOuterSize
    constexpr uint64_t fp8KvBlockSize = 256U; // 256 is SInnerSize
    if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        if (runParam.boIdx == 0 || runParam.boIdx == 1) {
            this->s1ScaleNumAcc = runParam.boIdx == 0 ? 0 : CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = runParam.boIdx == 0 ? 0 : CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            this->s1SizeAcc = runParam.boIdx == 0 ? 0 : this->actualSeqQlenAddr[0];
            this->s2SizeAcc = runParam.boIdx == 0 ? 0 : this->actualSeqKvlenAddr[0];
        } else {
            this->s1ScaleNumAcc = CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            for (uint32_t boIdx = 1; boIdx < runParam.boIdx; boIdx++) {
                this->s1ScaleNumAcc += CeilDiv(this->actualSeqQlenAddr[boIdx] - this->actualSeqQlenAddr[boIdx - 1], fp8QBlockSize);
                this->s2ScaleNumAcc += CeilDiv(this->actualSeqKvlenAddr[boIdx] - this->actualSeqKvlenAddr[boIdx - 1], fp8KvBlockSize);
            }
            this->s1SizeAcc = this->actualSeqQlenAddr[runParam.boIdx - 1];
            this->s2SizeAcc = this->actualSeqKvlenAddr[runParam.boIdx - 1];
        }
    }
    // GS1合轴时，g轴信息包含在gS1中；GS1不合轴时，g轴信息包含在bn2g中；
    if (this->constInfo.isGqa) {
        runParam.gS1Idx = gS1Index * this->s1BaseSize;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND ||
                      layout == LayOutTypeEnum::LAYOUT_BSH ||
                      layout == LayOutTypeEnum::LAYOUT_SBH) {
            runParam.goIdx = runParam.gS1Idx % this->constInfo.gSize;
            runParam.s1oIdx = runParam.gS1Idx / this->constInfo.gSize;
        } else {
            runParam.goIdx = runParam.gS1Idx / runParam.actualS1Size;
            runParam.s1oIdx = runParam.gS1Idx % runParam.actualS1Size;
        }
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
        runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
    }
}
}
#endif