/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file flash_attention_score_kernel_infer_regbase_v2.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_INFER_REGBASE_V2_H_
#define FLASH_ATTENTION_SCORE_KERNEL_INFER_REGBASE_V2_H_
namespace optiling {};
#include "flash_attention_score_kernel_base.h"
#include "vf/vf_flash_decode.h"
#include "infer_flash_attention_comm.h"
#include "./infer_flash_attention_kvcache.h"
#include "./infer_flash_attention_sparse.h"

using namespace AscendC;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelInferRegbaseV2
    : public FlashAttentionScoreKernelBase<FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>,
    CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass =
        FlashAttentionScoreKernelBase<FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>,
        CubeBlockType, VecBlockType>;
    /* =====================UB变量==================== */
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMainLoop();

private:
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDivision(this->constInfo.s2Size, this->constInfo.splitKVNum);
        this->constInfo.actualCombineLoopSize = CeilDivision(this->constInfo.s2Size, this->constInfo.sInnerLoopSize);
    }
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
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
    }

    this->constInfo.isBSNDOut = this->sharedParams.isBSNDOut;
    if (this->constInfo.isBSNDOut == 1) {
        this->constInfo.attentionOutStride = (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo,
        runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if constexpr (isFd) {
        actualCoreNums =
            this->sharedParams.bSize * this->constInfo.n2Size * this->constInfo.splitKVNum; // b * n2 * splitkv
    }

    if (GetBlockIdx() >= actualCoreNums) {
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
            bnEndIdx = this->sharedParams.bSize * this->constInfo.n2Size * this->constInfo.headNumRatio;
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
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo, lastBN, nextGs1Idx);
        if constexpr (isFd) {
            if (this->constInfo.sInnerLoopSize * (this->aicIdx % this->constInfo.splitKVNum) > runParam.actualS2Size) {
                runParam.s2LineEndIdx = 0;
            } else {
                int64_t tailSInnerLoopSize = runParam.actualS2Size -
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
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam,
                    this->constInfo, gS1Index, this->actualSeqQlenAddr, this->pseInfo);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                bool oneLoop = (bnEndIdx - bnStartIdx == 1) &&
                               (runParam.s1LoopTimes - gS1StartIdx == 1) && (runParam.s2LoopEndIdx == 1);
                if (notLastThreeLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if (taskId >= 2) {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.GetPre(), runInfo1, this->constInfo);
                    } else {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                    SetFlag<HardEvent::FIX_V>(BaseClass::SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]);
                }
                if (taskId > 0 && notLastTwoLoop) {
                    auto &runInfo3 = runInfo[(taskId + 3) & 3];
                    WaitFlag<HardEvent::FIX_V>(BaseClass::SYNC_C1_V1_FLAG[runInfo3.taskIdMod2]);
                    if (oneLoop) {
                        this->vecBlock.ProcessVec1(this->l1PBuffers.GetVec(), this->bmm1Buffers.GetPre(), runInfo3,
                            this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec1(this->l1PBuffers.GetVec(), this->bmm1Buffers.Get(), runInfo3,
                            this->constInfo);
                    }
                }
                if (taskId > 1 && notLast) {
                    RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                    WaitFlag<HardEvent::MTE3_MTE1>(BaseClass::SYNC_V1_C2_FLAG[runInfo2.taskIdMod3]);
                    if (taskId >= 4) {
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.GetPre(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    } else {
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                            this->constInfo);
                    }
                    SetFlag<HardEvent::FIX_V>(BaseClass::SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]);
                }
                if (taskId > 2) {
                    RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                    WaitFlag<HardEvent::FIX_V>(BaseClass::SYNC_C2_V2_FLAG[runInfo3.taskIdMod2]);
                    if (oneLoop) {
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.GetPre(), runInfo3, this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
    if constexpr (isFd) {
        SyncAll();
        this->vecBlock.InitFDBuffers(this->constInfo);
        this->vecBlock.FlashDecodeCompute(this->constInfo, this->keyGm, this->actualSeqKvlenAddr);
    }
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelInferRegbaseV2<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    // GS1合轴时，g轴信息包含在gS1中；GS1不合轴时，g轴信息包含在bn2g中；
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif