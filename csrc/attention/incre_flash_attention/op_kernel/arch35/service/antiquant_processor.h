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
 * \file antiquant_processor.h
 * \brief
 */
#ifndef ANTIQUANT_PROCESSOR_H
#define ANTIQUANT_PROCESSOR_H

#if __has_include("../../../../common/op_kernel/arch35/vf/vf_antiquant_w4.h")
#include "../../../../common/op_kernel/arch35/vf/vf_antiquant_w4.h"
#include "../../../../common/op_kernel/arch35/vf/vf_antiquant_w8.h"
#else
#include "../../../common/arch35/vf/vf_antiquant_w4.h"
#include "../../../common/arch35/vf/vf_antiquant_w8.h"
#endif

struct AntiquantTaskParam {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t kvHeadNum;
    uint32_t headDim;
    uint32_t headDimAlignBlock;
    uint64_t kvStep;

    uint64_t kvGmOffset;
    uint32_t copySplitS;
    uint32_t copyTotalS;

    uint32_t s2BatchOffset;
    bool isLoadAntiqParam;
    bool isFreeAntiqParam;
    bool isExistOffset;
    uint64_t antiqParamOffset;

    uint64_t bIdx;
    uint64_t n2Idx;
    uint64_t s2Idx;
    uint32_t singleSInnerSize;
    uint32_t flashDecodeS2Idx;
    uint32_t sInnerLoopSize;
    uint32_t antiqSeqSize;
    bool isPertensor;
    bool isPerHead;
    uint32_t kvCacheBlockSize;
    uint32_t maxBlockNumPerSeq;
    uint32_t paKvShapeType;
    uint64_t kvPaddingBeginOffset;
};
template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
class AntiquantProcessor
{
public:
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool KV_CONTINUOUS = IFAT::kvContinuous;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr INPUTKVTYPE KVTYPE_T = IFAT::inputKVType;
    static constexpr IFAProfile PROFILE = IFAT::profile;
    static constexpr bool ATTEN_MASK = IFAT::attenMask;
    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    // 目前只有FP4类型是pertoken伪量化，antiquant参数类型是fp16
    static constexpr bool KVFP4 = (IsSameType<KV_T, fp4x2_e1m2_t>::value || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool PAGE_ATTENTION_ANTIQUANT = (IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_PAGE_ATTENTION ||
        IFAT::antiquantMode == ANTIQUANTMODE::PER_TOKEN_HEAD_PAGE_ATTENTION);
    using ANTIQ_PARAMS_T = typename std::conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;

    __aicore__ inline AntiquantProcessor(){};

    __aicore__ inline void Process(TSCM<TPosition::VECIN, 1>& kvScmQueue, GlobalTensor<KV_T> kvGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm, GlobalTensor<int32_t>& blockTableGm,
                                   TQue<QuePosition::VECIN, 1>& kvInputQue, TQue<QuePosition::VECOUT, 1>& kvOutputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue, TBuf<> kvAntiqMxScaleRes,
                                   const AntiquantTaskParam& taskParam);

    __aicore__ inline void CopyAntiqScaleE8M0(LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset,
                                              uint32_t rowCnt, uint32_t grpNum);
    __aicore__ inline void LoadAntiquantParamsPerTokenGroup(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                                            TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                            TBuf<> kvAntiqMxScaleRes,
                                                            const AntiquantTaskParam& taskParam);
    __aicore__ inline void LoadAntiquantParamsPerToken(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                                       GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
                                                       GlobalTensor<int32_t>& blockTableGm,
                                                       TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                       TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                                       const AntiquantTaskParam& taskParam);

    __aicore__ inline void CopyAntiqParam(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                          uint32_t rowCnt, const AntiquantTaskParam& taskParam);
    __aicore__ inline void LoadAntiquantParamsPerChannel(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
                                   TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                    const AntiquantTaskParam& taskParam);
    __aicore__ inline void FreeAntiquantParams(TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                    const AntiquantTaskParam& taskParam);
    __aicore__ inline void Antiquant(uint32_t copyLoopIdx, TQue<QuePosition::VECIN, 1>& kvInputQue,
                                     TQue<QuePosition::VECOUT, 1>& kvOutputQue, LocalTensor<Q_T> outScm,
                                     GlobalTensor<KV_T> kvGm, GlobalTensor<int32_t>& blockTableGm,
                                     uint32_t dealRowCount, const AntiquantTaskParam& taskParam);
    __aicore__ inline void CopyKV(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T>& srcGm, uint64_t offset,
                                  uint32_t rowCnt, uint32_t headDim, uint32_t kvHeadNum,
                                  uint32_t paKvShapeType);
    __aicore__ inline void CopyKVPageAttention(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> srcGm,
                                               GlobalTensor<int32_t>& blockTableGm, const AntiquantTaskParam& taskParam,
                                               uint32_t curSequence, uint32_t dealRowCount);
    __aicore__ inline void CopyAntiquantParamsPageAttention(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                                            GlobalTensor<int32_t>& blockTableGm, const AntiquantTaskParam& taskParam,
                                                            DataCopyExtParams copyInParams,
                                                            DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams);
    __aicore__ inline void AntiquantVec(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb, uint32_t copyLoopIdx,
                                        uint32_t dealRowCount, const AntiquantTaskParam& taskParam);
    __aicore__ inline void CopyAntiquantResToL1(LocalTensor<Q_T>& antiqResScm, LocalTensor<Q_T>& antiqResUb,
                                                uint32_t copyLoopIdx, uint32_t dealRowCount,
                                                const AntiquantTaskParam& taskParam);

private:
    LocalTensor<ANTIQ_PARAMS_T> antiqScale;
    LocalTensor<ANTIQ_PARAMS_T> antiqOffset;
};

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::Process(
    TSCM<TPosition::VECIN, 1>& kvScmQueue, GlobalTensor<KV_T> kvGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
    GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm, GlobalTensor<int32_t>& blockTableGm,
    TQue<QuePosition::VECIN, 1>& kvInputQue, TQue<QuePosition::VECOUT, 1>& kvOutputQue,
    TQue<QuePosition::VECIN, 1>& antiqScaleInputQue, TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
    TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParam& taskParam)
{
    if (taskParam.isLoadAntiqParam) {
        if constexpr (KVFP4) {
            LoadAntiquantParamsPerTokenGroup(antiqScaleGm, antiqScaleInputQue, kvAntiqMxScaleRes, taskParam);
        } else if constexpr (ANTIQUANT_PER_TOKEN) {
            LoadAntiquantParamsPerToken(antiqScaleGm, antiqOffsetGm, blockTableGm, antiqScaleInputQue,
                                        antiqOffsetInputQue, taskParam);
        } else {
            LoadAntiquantParamsPerChannel(antiqScaleGm, antiqOffsetGm, antiqScaleInputQue, antiqOffsetInputQue,
                                          taskParam);
        }
    }

    uint32_t loopCnt = (taskParam.copyTotalS + taskParam.copySplitS - 1) / taskParam.copySplitS;
    uint32_t tailCopyS = taskParam.copyTotalS - (loopCnt - 1) * taskParam.copySplitS;
    LocalTensor<Q_T> scmTensor = kvScmQueue.template AllocTensor<Q_T>();
    for (uint32_t i = 0, actCopyS = taskParam.copySplitS; i < loopCnt; i++) {
        if (i + 1 == loopCnt) {
            actCopyS = tailCopyS;
        }
        Antiquant(i, kvInputQue, kvOutputQue, scmTensor, kvGm, blockTableGm, actCopyS, taskParam);
    }
    kvScmQueue.template EnQue(scmTensor);
    if (taskParam.isFreeAntiqParam) {
        if constexpr (KVFP4) {
        } else {
            FreeAntiquantParams(antiqScaleInputQue, antiqOffsetInputQue, taskParam);
        }
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::CopyAntiqScaleE8M0(LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm,
                                                                    uint64_t offset, uint32_t rowCnt, uint32_t grpNum)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams{};
    copyInParams.blockCount = 1;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = grpNum * rowCnt * sizeof(int8_t);  // FP8和int8大小一样。单位是字节
    copyInParams.srcStride = 0;
    DataCopyPad(dstLocal, srcGm[offset], copyInParams, copyInPadParams);
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerTokenGroup(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
    TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParam& taskParam)
{
    uint32_t grpSize = 32;
    uint32_t grpNum = taskParam.headDim / grpSize;
    uint64_t scaleOffset = 0;

    scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize * grpNum +
                  taskParam.n2Idx * taskParam.seqSize * grpNum + taskParam.s2Idx * taskParam.singleSInnerSize * grpNum +
                  taskParam.kvPaddingBeginOffset * grpNum;
    if constexpr (FLASH_DECODE) {
        scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize * grpNum;  // sInnerLoopSize 用于FD
    }

    // copy fp8_e8m0 scale
    LocalTensor<Q_T> antiqScaleE8M0Ub = antiqScaleInputQue.template AllocTensor<Q_T>();
    CopyAntiqScaleE8M0(antiqScaleE8M0Ub, antiqScaleGm, scaleOffset / 2, taskParam.copyTotalS,
                       grpNum);  // GM上是fp16，实际上FP8E8M0，除以2
    antiqScaleInputQue.template EnQue(antiqScaleE8M0Ub);
    antiqScaleE8M0Ub = antiqScaleInputQue.DeQue<Q_T>();

    antiqScale = kvAntiqMxScaleRes.Get<ANTIQ_PARAMS_T>();
    FaVectorApi::AntiqScaleByVF<Q_T, ANTIQ_PARAMS_T>(antiqScaleE8M0Ub, antiqScale, taskParam.copyTotalS, grpNum);
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerToken(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
    GlobalTensor<int32_t>& blockTableGm, TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
    TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue, const AntiquantTaskParam& taskParam)
{
    uint64_t scaleOffset = 0;
    if (taskParam.isPerHead) {
        scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize + taskParam.n2Idx * taskParam.seqSize +
                      taskParam.s2Idx * taskParam.singleSInnerSize + taskParam.kvPaddingBeginOffset;
    } else {
        scaleOffset = taskParam.bIdx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                      taskParam.kvPaddingBeginOffset;  // s有尾块时，偏移是基于前面对齐的S的计算的
    }
    if constexpr (FLASH_DECODE) {
        scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;  // sInnerLoopSize 用于FD
    }

    // copy
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams{};
    copyInParams.blockCount = 1;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = taskParam.copyTotalS * sizeof(ANTIQ_PARAMS_T);
    copyInParams.srcStride = 0;
    LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqScaleInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
    if constexpr (PAGE_ATTENTION_ANTIQUANT) {
        CopyAntiquantParamsPageAttention(tmpUb, antiqScaleGm, blockTableGm, taskParam, copyInParams, copyInPadParams);
    } else {
        DataCopyPad(tmpUb, antiqScaleGm[scaleOffset], copyInParams, copyInPadParams);
    }
    antiqScaleInputQue.template EnQue(tmpUb);
    antiqScale = antiqScaleInputQue.DeQue<ANTIQ_PARAMS_T>();

    if (taskParam.isExistOffset) {
        LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqOffsetInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
        if constexpr (PAGE_ATTENTION_ANTIQUANT) {
            CopyAntiquantParamsPageAttention(tmpUb, antiqOffsetGm, blockTableGm, taskParam, copyInParams, copyInPadParams);
        } else {
            DataCopyPad(tmpUb, antiqOffsetGm[scaleOffset], copyInParams, copyInPadParams);
        }
        antiqOffsetInputQue.template EnQue(tmpUb);
        antiqOffset = antiqOffsetInputQue.DeQue<ANTIQ_PARAMS_T>();
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::CopyAntiqParam(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                          uint32_t rowCnt, const AntiquantTaskParam& taskParam)
{
    if (taskParam.isPertensor) {
        Duplicate(dstLocal, srcGm.GetValue(0), rowCnt * PROFILE.D);
    } else if (taskParam.isPerHead) {
        Duplicate(dstLocal, srcGm.GetValue(taskParam.n2Idx), rowCnt * PROFILE.D);
    } else {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = taskParam.headDim * sizeof(ANTIQ_PARAMS_T);
        copyInParams.srcStride = static_cast<int64_t>(copyInParams.blockLen) * (-1); // 原来位置复制一份
        copyInParams.dstStride = (PROFILE.D - taskParam.headDim) * sizeof(ANTIQ_PARAMS_T) / BYTE_BLOCK;

        copyInPadParams.isPad = false;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = 0;
        copyInPadParams.paddingValue = 0;
    
        DataCopyPad(dstLocal, srcGm[taskParam.antiqParamOffset], copyInParams, copyInPadParams);
    }
}
template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerChannel(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
    TQue<QuePosition::VECIN, 1>& antiqScaleInputQue, TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
    const AntiquantTaskParam& taskParam)
{
    uint32_t qElementCntPerReg = 256 / sizeof(ANTIQ_PARAMS_T);         // 单个寄存器256B
    uint32_t loopCount = qElementCntPerReg / taskParam.headDimAlignBlock;  // D=64时合并伪量化计算，需要复制2份伪量化参数
    if( loopCount == 0){
        loopCount =1;// D=256、512时至少要拷一次
    }
    LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqScaleInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
    CopyAntiqParam(tmpUb, antiqScaleGm, loopCount, taskParam);

    antiqScaleInputQue.template EnQue(tmpUb);
    antiqScale = antiqScaleInputQue.DeQue<ANTIQ_PARAMS_T>();
    if (taskParam.isExistOffset) {
        LocalTensor<ANTIQ_PARAMS_T> tmpUb = antiqOffsetInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
        CopyAntiqParam(tmpUb, antiqOffsetGm, loopCount, taskParam);
        antiqOffsetInputQue.template EnQue(tmpUb);
        antiqOffset = antiqOffsetInputQue.DeQue<ANTIQ_PARAMS_T>();
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::FreeAntiquantParams(TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                                     TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                                                     const AntiquantTaskParam& taskParam)
{
    antiqScaleInputQue.FreeTensor(antiqScale);
    if (taskParam.isExistOffset) {
        antiqOffsetInputQue.FreeTensor(antiqOffset);
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::Antiquant(uint32_t copyLoopIdx,
                                                           TQue<QuePosition::VECIN, 1>& kvInputQue,
                                                           TQue<QuePosition::VECOUT, 1>& kvOutputQue,
                                                           LocalTensor<Q_T> outScm, GlobalTensor<KV_T> kvGm,
                                                           GlobalTensor<int32_t>& blockTableGm, uint32_t dealRowCount,
                                                           const AntiquantTaskParam& taskParam)
{
    LocalTensor<KV_T> tmpUb = kvInputQue.template AllocTensor<KV_T>();
    if constexpr (!PAGE_ATTENTION) {
        uint64_t kvOffset = taskParam.kvGmOffset + copyLoopIdx * taskParam.copySplitS * taskParam.kvStep;
        CopyKV(tmpUb, kvGm, kvOffset, dealRowCount, taskParam.headDim, taskParam.kvHeadNum,
               taskParam.paKvShapeType);
    } else {
        uint64_t curSeqence = taskParam.s2BatchOffset + copyLoopIdx * taskParam.copySplitS;
        CopyKVPageAttention(tmpUb, kvGm, blockTableGm, taskParam, curSeqence, dealRowCount);
    }
    kvInputQue.template EnQue(tmpUb);
    kvInputQue.DeQue<KV_T>();

    LocalTensor<Q_T> outUb = kvOutputQue.template AllocTensor<Q_T>();
    AntiquantVec(outUb, tmpUb, copyLoopIdx, dealRowCount, taskParam);
    kvInputQue.FreeTensor(tmpUb);

    kvOutputQue.template EnQue(outUb);
    kvOutputQue.DeQue<Q_T>();
    CopyAntiquantResToL1(outScm, outUb, copyLoopIdx, dealRowCount, taskParam);
    kvOutputQue.FreeTensor(outUb);
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::CopyAntiquantParamsPageAttention(LocalTensor<ANTIQ_PARAMS_T> dstLocal,
                                                                      GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                                                      GlobalTensor<int32_t>& blockTableGm,
                                                                      const AntiquantTaskParam& taskParam,
                                                                      DataCopyExtParams copyInParams,
                                                                      DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams)
{
    uint32_t useKvHeadNum = 1;
    uint32_t useN2Idx = 0;
    if (taskParam.isPerHead) {
        useKvHeadNum = taskParam.kvHeadNum;
        useN2Idx = taskParam.n2Idx;
    }
    uint64_t blockTableBaseOffset = taskParam.bIdx * taskParam.maxBlockNumPerSeq;
    uint64_t dstOffset = 0;
    uint32_t copyFinishElmenCnt = 0;
    uint32_t curSequence = taskParam.s2BatchOffset; // 当前S2长度
    while (copyFinishElmenCnt < taskParam.copyTotalS) {
        uint64_t blockIdOffset = curSequence / taskParam.kvCacheBlockSize;
        uint64_t remainElmenCnt = curSequence % taskParam.kvCacheBlockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyElmenCnt = taskParam.kvCacheBlockSize - remainElmenCnt; // 可拷贝数量
        if (copyElmenCnt + copyFinishElmenCnt > taskParam.copyTotalS) {
            copyElmenCnt = taskParam.copyTotalS - copyFinishElmenCnt;
        }
        uint64_t srcOffset = idInBlockTable * taskParam.kvCacheBlockSize * useKvHeadNum +
                             taskParam.kvCacheBlockSize * useN2Idx + remainElmenCnt;
        copyInParams.blockLen = copyElmenCnt * sizeof(ANTIQ_PARAMS_T);
        DataCopyPad(dstLocal[dstOffset], srcGm[srcOffset], copyInParams, copyInPadParams);
        // 更新处理的位置
        dstOffset += copyElmenCnt;
        copyFinishElmenCnt += copyElmenCnt;
        curSequence += copyElmenCnt;
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::CopyKV(LocalTensor<KV_T> dstLocal,
                                                                             GlobalTensor<KV_T> &srcGm, uint64_t offset,
                                                                             uint32_t rowCnt, uint32_t headDim,
                                                                             uint32_t kvHeadNum, uint32_t paKvShapeType)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(KV_T);

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<KV_T> copyInPadParams;

    if constexpr (KVINT4 || KVFP4) {
        typeElementSize = ONE_BLK_SIZE * 2;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = headDim * sizeof(KV_T) / 2; 
    } else {
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = headDim * sizeof(KV_T);
    }

    copyInParams.dstStride = (PROFILE.D - headDim) / typeElementSize;

    if constexpr (PAGE_ATTENTION) {
        if (paKvShapeType == 0) {  // BBH
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        } else {  // BNBD
            copyInParams.srcStride = 0;
        }
    } else {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            copyInParams.srcStride = 0;
        } else {
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        }
    }

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    DataCopyPad(dstLocal, srcGm[offset], copyInParams, copyInPadParams);
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::CopyKVPageAttention(LocalTensor<KV_T> dstLocal,
                                                                     GlobalTensor<KV_T> srcGm,
                                                                     GlobalTensor<int32_t>& blockTableGm,
                                                                     const AntiquantTaskParam& taskParam,
                                                                     uint32_t curSequence, uint32_t dealRowCount)
{
    uint64_t blockTableBaseOffset = taskParam.bIdx * taskParam.maxBlockNumPerSeq;
    uint32_t copyFinishRowCnt = 0;
    while (copyFinishRowCnt < dealRowCount) {
        uint64_t blockIdOffset = curSequence / taskParam.kvCacheBlockSize;  // 获取blcok table上的索引
        uint64_t reaminRowCnt = curSequence % taskParam.kvCacheBlockSize;   // 获取在单个块上超出的行数
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);  // 从block table上的获取编号
        // 计算可以拷贝行数
        uint32_t copyRowCnt = taskParam.kvCacheBlockSize - reaminRowCnt;
        if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
            copyRowCnt = dealRowCount - copyFinishRowCnt;
        }
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == 0) {  // BBH  [blockNums, blockSize, headNum*headDim]
            curOffset =
                (idInBlockTable * taskParam.kvCacheBlockSize + reaminRowCnt) * taskParam.kvHeadNum * taskParam.headDim +
                (uint64_t)(taskParam.n2Idx * taskParam.headDim);
        } else {  // BNBD [blockNums, headNum, blockSize, headDim]
            curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.kvCacheBlockSize * taskParam.headDim +
                        (uint64_t)(taskParam.n2Idx * taskParam.kvCacheBlockSize * taskParam.headDim) +
                        reaminRowCnt * taskParam.headDim;
        }

        CopyKV(dstLocal[copyFinishRowCnt * PROFILE.D], srcGm, curOffset, copyRowCnt, taskParam.headDim,
               taskParam.kvHeadNum, taskParam.paKvShapeType);

        // 更新循环变量
        copyFinishRowCnt += copyRowCnt;
        curSequence += copyRowCnt;
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::AntiquantVec(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb,
                                                              uint32_t copyLoopIdx, uint32_t dealRowCount,
                                                              const AntiquantTaskParam& taskParam)
{
    if constexpr (KVFP4) {
        uint32_t grpNum = taskParam.headDim / 32;
        uint32_t perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS * grpNum * 2;
        LocalTensor<ANTIQ_PARAMS_T> antiqScaleWithOffset = antiqScale[perTokenScaleOffset];
        FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, PROFILE.D, false>(antiqInUb, antiqResUb, antiqOffset,
                                                                 antiqScaleWithOffset, dealRowCount, taskParam.headDim);
    } else if constexpr (ANTIQUANT_PER_TOKEN) {
        uint32_t perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS;
        LocalTensor<ANTIQ_PARAMS_T> antiqScaleWithOffset = antiqScale[perTokenScaleOffset];
        if (taskParam.isExistOffset) {
            LocalTensor<ANTIQ_PARAMS_T> antiqOffsetWithOffset = antiqOffset[perTokenScaleOffset];
            FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, PROFILE.D, true, true>(antiqInUb, antiqResUb, antiqOffsetWithOffset,
                                                                          antiqScaleWithOffset, dealRowCount, taskParam.headDim);
        } else {
            FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, PROFILE.D, false, true>(antiqInUb, antiqResUb, antiqOffset,
                                                                           antiqScaleWithOffset, dealRowCount, taskParam.headDim);
        }
    } else {
        if (taskParam.isExistOffset) {
            FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, PROFILE.D, true>(antiqInUb, antiqResUb, antiqOffset, antiqScale,
                                                                    dealRowCount, taskParam.headDim);
        } else {
            FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, PROFILE.D, false>(antiqInUb, antiqResUb, antiqOffset, antiqScale,
                                                                     dealRowCount, taskParam.headDim);
        }
    }
}

template <typename IFAT, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessor<IFAT, ANTIQUANT_PER_TOKEN>::CopyAntiquantResToL1(LocalTensor<Q_T>& antiqResScm,
                                                                      LocalTensor<Q_T>& antiqResUb, uint32_t copyLoopIdx,
                                                                      uint32_t dealRowCount,
                                                                      const AntiquantTaskParam& taskParam)
{
    uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint16_t dstStep = ALIGN((uint16_t)taskParam.copyTotalS, (uint16_t)16);
    uint64_t outOffset = copyLoopIdx * taskParam.copySplitS * 16;

    struct DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = taskParam.headDimAlignBlock / elementTypeSize;
    dataCopyParams.blockLen = dealRowCount;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = dstStep - dealRowCount;

    DataCopy(antiqResScm[outOffset], antiqResUb, dataCopyParams);
}
#endif
