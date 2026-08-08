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
 * \file flash_attention_noquant_block_vec_train.h
 * \brief
 */
#ifndef FLASH_ATTENTION_NOQUANT_BLOCK_VEC_TRAIN_H_
#define FLASH_ATTENTION_NOQUANT_BLOCK_VEC_TRAIN_H_
#include "flash_attention_noquant_block_vec_base.h"
#include "util_regbase.h"
#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
#include "dropmask.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
TEMPLATES_DEF
class FANoQuantBlockVecTrain
    : public FANoQuantBlockVecBase<FANoQuantBlockVecTrain<TEMPLATE_ARGS>, TEMPLATE_ARGS> {
public:
    using BaseClass = FANoQuantBlockVecBase<FANoQuantBlockVecTrain<TEMPLATE_ARGS>, TEMPLATE_ARGS>;
public:
    __aicore__ inline void InitDropOut(__gm__ uint8_t *dropMask, __gm__ uint8_t *workspace);
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
        __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *prefix,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset,
        uint32_t aicIdx, ConstInfo<isInfer, hasRope> &constInfo);
    /* =====================GM变量==================== */
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    /* =====================UB变量==================== */
    TBuf<> dropMaskBuf;
    TBuf<> dropMaskIndexBuf;
    TQue<QuePosition::VECIN, 1> dropMaskInQue;
    DropMaskInfo dropMaskInfo;
    /* =================编译期常量的基本块信息================= */

    __aicore__ inline FANoQuantBlockVecTrain() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, ConstInfo<isInfer, hasRope> &constInfo) {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
    }
    __aicore__ inline void InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo);
    /* ================共享参数v侧优先初始化，通过ssbuf拷贝到c侧=================*/
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    /* 多核切分根据cube核数做切分，因此这里需要传入aicIdx */
    __aicore__ inline void GetS1LoopRange(CVSharedParams<isInfer, isPa> &sharedParams, const int64_t &aicIdx);
    __aicore__ inline void GenerateDropoutMask(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<uint8_t> &dropMaskUb);
    __aicore__ inline void SoftmaxDataCopyOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> &sumUb,
                                              LocalTensor<float> &maxUb);
    __aicore__ inline void SoftmaxDataCopyOutFp8(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
        LocalTensor<half> &sumUb, LocalTensor<half> &maxUb);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                               int64_t vec2S1Idx, int64_t vec2CalcSize);
private:
};


TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::InitDropOut(__gm__ uint8_t *dropMask,
    __gm__ uint8_t *workspace)
{
    if constexpr (hasDrop) {
        dropMaskInfo.boolMode = this->tilingData->inputParamsRegbase.needDropMaskOp == 1;
        if (dropMaskInfo.boolMode) {
            dropMaskGm.SetGlobalBuffer(workspace + this->tilingData->dropmaskParamsRegbase.dropMaskAddrOffset);
        } else {
            dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *pse, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *&workspace, uint64_t singleCoreOffset, uint32_t aicIdx,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    BaseClass::InitCommonGlobalBuffer(pse, deqScaleQ, deqScaleK, deqScaleV, pScale, postQuantScale, prefix,
        attenMask, learnableSink, workspace, constInfo);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::InitUniqueLocalBuffer(ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (hasDrop) {
        if constexpr (!IsSameType<INPUT_T, float>::value || !BaseClass::containAllOptionalInput) {
            this->tPipe->InitBuffer(dropMaskInQue, 1, 8192);
        }
        this->tPipe->InitBuffer(dropMaskIndexBuf, 2048);
        this->tPipe->InitBuffer(dropMaskBuf, 1024);
        dropMaskInfo.seed = this->tilingData->inputParamsRegbase.seed;
        dropMaskInfo.offset = this->tilingData->inputParamsRegbase.offset;
        constInfo.keepProb = this->tilingData->inputParamsRegbase.keepProb;
        dropMaskInfo.keepProbUint8 = static_cast<uint8_t>(this->tilingData->inputParamsRegbase.keepProbUint8);
        dropMaskInfo.dropMaskOuter = this->tilingData->inputParamsRegbase.dropMaskOuter;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams<isInfer, isPa> &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
    sharedParams.bSize = inputParamsRegbase.bSize;
    sharedParams.n2Size = inputParamsRegbase.n2Size;
    sharedParams.gSize = inputParamsRegbase.gSize;
    sharedParams.s1Size = inputParamsRegbase.s1Size;
    sharedParams.s2Size = inputParamsRegbase.s2Size;
    sharedParams.dSize = inputParamsRegbase.dSize;
    sharedParams.dSizeV = inputParamsRegbase.dSizeV;
    sharedParams.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    if constexpr (hasRope) {
        sharedParams.dSizeRope = inputParamsRegbase.dSizeRope;
    } else {
        sharedParams.dSizeRope = 0;
    }
    sharedParams.preTokens = inputParamsRegbase.preTokens;
    sharedParams.nextTokens = inputParamsRegbase.nextTokens;
    sharedParams.s1SparseValidSize = inputParamsRegbase.s1SparseValidSize;
    sharedParams.s2SparseValidSize = inputParamsRegbase.s2SparseValidSize;
    sharedParams.bandIndex = inputParamsRegbase.bandIndex;
    sharedParams.implMode = inputParamsRegbase.implMode;
    sharedParams.layoutType = inputParamsRegbase.layoutType;
    sharedParams.sparseType = inputParamsRegbase.sparseType;

    auto &multiCoreParamsRegbase = this->tilingData->multiCoreParamsRegbase;
    sharedParams.s1OuterSize = multiCoreParamsRegbase.s1OuterSize;
    /* 多核切分偏移计算 */
    sharedParams.multiCoreInnerOffset = aicIdx * multiCoreParamsRegbase.splitFactorSize;
    sharedParams.multiCoreInnerLimit = sharedParams.multiCoreInnerOffset + multiCoreParamsRegbase.splitFactorSize;
    if (multiCoreParamsRegbase.totalSize < sharedParams.multiCoreInnerLimit) {
        sharedParams.multiCoreInnerLimit = multiCoreParamsRegbase.totalSize;
    }
    sharedParams.splitCoreMode = multiCoreParamsRegbase.splitCoreMode;
    sharedParams.firstFullLoadS1OuterIdx = multiCoreParamsRegbase.firstFullLoadS1OuterIdx;
    sharedParams.totalSize = multiCoreParamsRegbase.totalSize;
    sharedParams.coreNum = multiCoreParamsRegbase.coreNum;
    // 计算sparse场景下s1的循环范围
    GetS1LoopRange(sharedParams, aicIdx);
    /* ssbuf send message */
    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::GetS1LoopRange(CVSharedParams<isInfer, isPa> &sharedParams,
    const int64_t &aicIdx)
{
    if constexpr (layout != LayOutTypeEnum::LAYOUT_TND) {
        if constexpr (!hasAtten) {
            return;
        }
        if (sharedParams.sparseType <= 0) {
            return;
        }
    }
 
    // TND/Sparse 场景下负载均衡后每个核获取的结果
    sharedParams.multiCoreInnerOffset = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx];
    if (likely(this->tilingData->multiCoreParamsRegbase.coreNum - 1 > aicIdx)) {
        sharedParams.multiCoreInnerLimit = this->tilingData->multiCoreParamsRegbase.sparseStartIdx[aicIdx + 1];
    } else {
        sharedParams.multiCoreInnerLimit = this->tilingData->multiCoreParamsRegbase.totalSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::GenerateDropoutMask(
    RunInfo<isInfer> & runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<uint8_t> & dropMaskUb)
{
    if constexpr (hasDrop == true) {
        if (dropMaskInfo.dropMaskOuter == 1) {
            if constexpr (!IsSameType<INPUT_T, float>::value || !BaseClass::containAllOptionalInput) {
                CopyInDropOuter<hasDrop>(dropMaskBuf, dropMaskInQue, dropMaskGm, runInfo, constInfo,
                                         dropMaskInfo);
            } else {
                CopyInDropOuter<hasDrop>(dropMaskBuf, this->attenMaskInQue[1 - runInfo.taskIdMod2], dropMaskGm,
                                         runInfo, constInfo, dropMaskInfo);
            }
        } else {
            GenDropMask<hasDrop>(dropMaskBuf, dropMaskIndexBuf, runInfo, constInfo, dropMaskInfo);
        }
        dropMaskUb = dropMaskBuf.template Get<uint8_t>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::SoftmaxDataCopyOut(
    RunInfo<isInfer> & runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<float> & sumUb, LocalTensor<float> & maxUb)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    int64_t s1Offset;
    if (constInfo.learnableSinkFlag) {
        SinkSubExpAddVF<float>(sumUb, maxUb, constInfo.sinkValue, runInfo.halfS1RealSize);
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // tndSoftmaxOut=1时，softmaxMax、softMaxSum按(T, N, 8)格式存储
        // tndSoftmaxOut=0时，softmaxMax、softMaxSum按(B, N, Sq, 8)格式存储
        if (this->tilingData->inputParamsRegbase.tndSoftmaxOut == 1) {
            bOffset = constInfo.n2G * runInfo.s1SizeAcc;
            s1Offset = 
            (runInfo.s1oIdx * this->s1BaseSize + constInfo.subBlockIdx * runInfo.firstHalfS1RealSize) * constInfo.n2G;
            n2Offset = runInfo.n2oIdx * constInfo.gSize;
            gOffset = runInfo.goIdx;
        } else {
            bOffset = constInfo.n2G * runInfo.s1SizeAcc;
            s1Offset = 
            runInfo.s1oIdx * this->s1BaseSize + constInfo.subBlockIdx * runInfo.firstHalfS1RealSize;
            n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
            gOffset = runInfo.goIdx * runInfo.actualS1Size;
        }
    } else {
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
        s1Offset =
        (runInfo.s1oIdx * this->s1BaseSize + constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    }
    int64_t gmOffset = (bOffset + n2Offset + gOffset + s1Offset) * fp32BaseSize;
    int64_t calculateSize = runInfo.halfS1RealSize * fp32BaseSize;

    this->BroadCastAndCopyOut(runInfo, constInfo, softmaxSumGm, softmaxMaxGm, gmOffset, calculateSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::SoftmaxDataCopyOutFp8(
    RunInfo<isInfer> & runInfo, ConstInfo<isInfer, hasRope> &constInfo,
    LocalTensor<half> & sumUb, LocalTensor<half> & maxUb)
{
    if (unlikely(runInfo.halfS1RealSize == 0)) {
        return;
    }
    int64_t bOffset;
    int64_t n2Offset;
    int64_t gOffset;
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        bOffset = constInfo.n2G * runInfo.s1SizeAcc;
        n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size;
        gOffset = runInfo.goIdx * runInfo.actualS1Size;
    } else {
        bOffset = runInfo.boIdx * constInfo.n2Size * constInfo.gS1;
        n2Offset = runInfo.n2oIdx * constInfo.gS1;
        gOffset = runInfo.goIdx * constInfo.s1Size;
    }
    int64_t s1Offset =
        (runInfo.s1oIdx * this->s1BaseSize + constInfo.subBlockIdx * runInfo.firstHalfS1RealSize);
    int64_t gmOffset = (bOffset + n2Offset + gOffset + s1Offset);
    int64_t calculateSize = runInfo.halfS1RealSize;
    this->BroadCastAndCopyOut(runInfo, constInfo, softmaxSumGm, softmaxMaxGm, gmOffset, calculateSize);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void FANoQuantBlockVecTrain<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo<isInfer> & runInfo, ConstInfo<isInfer, hasRope> &constInfo, LocalTensor<VEC2_RES_T> & vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}
}
#endif // FLASH_ATTENTION_NOQUANT_BLOCK_VEC_TRAIN_H_