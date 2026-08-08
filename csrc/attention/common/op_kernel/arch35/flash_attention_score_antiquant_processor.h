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
 * \file flash_attention_score_antiquant_processor.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_ANTIQUANT_PROCESSOR_H
#define FLASH_ATTENTION_SCORE_ANTIQUANT_PROCESSOR_H

#include "vf/vf_antiquant_w4.h"
#include "vf/vf_antiquant_w8.h"

using namespace AscendC;
using namespace fa_base_matmul;

enum class AntiquantTypeEnum : uint8_t {
    PER_CHANNEL = 0, // enable per-channel antiquant mode，include per-tensor
    K_PER_CHANNEL_V_PER_TOKEN = 2, // enable split antiquant mode, k per-channel and v per-token
    PER_TOKEN_PAGE_ATTENTION = 4, // enable per-token antiquant mode, and enable PA for memory management
    PER_TOKEN_HEAD_PAGE_ATTENTION = 5, // enable both per-token and per-head antiquant mode, and enable PA for memory management
    K_PER_TOKEN = 6, // 表示其他伪量化类型
};

enum class KvCacheLayout : uint32_t {
    KV_CACHE_BSH = 0,
    KV_CACHE_BNSD = 1,
    KV_CACHE_NZ = 2,
};

namespace BaseApi {

__aicore__ constexpr uint16_t GetRealDealSize(uint16_t realSize) {
    uint16_t dealSize = ((realSize >> 1) + 31) >> 5 << 5;      // 31 & 5 is Alighup 32
    return (dealSize > realSize) ? realSize : dealSize;
}

__aicore__ constexpr uint64_t AlignUp32(uint64_t size) {
    return (size + 31) >> 5 << 5;      // 31 & 5 is Alignup 32
}

struct AntiquantTaskParamBaseAPI {
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
    bool isPageAttentionAntiquant;
    bool isPertensor;
    bool isPerHead;
    bool isKvCacheNz;
    uint32_t kvCacheBlockSize;
    uint32_t maxBlockNumPerSeq;
    uint32_t paKvShapeType;
    uint64_t kvPaddingBeginOffset;

    bool isPrefixLoop = 0;
};


template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
class AntiquantProcessorBaseAPI
{
public:
    static constexpr bool PAGE_ATTENTION = isPa;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool KVFP4 = (IsSameType<KV_T, fp4x2_e1m2_t>::value || IsSameType<KV_T, fp4x2_e2m1_t>::value);
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    using ANTIQ_PARAMS_T = typename std::conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;

    __aicore__ inline AntiquantProcessorBaseAPI(){};

    __aicore__ inline void ProcessBaseAPI(Buffer<BufferType::L1> &outBufAntiRes, GlobalTensor<KV_T> &kvGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm, GlobalTensor<int32_t>& blockTableGm,
                                   TQue<QuePosition::VECIN, 1>& kvInputQue, TQue<QuePosition::VECOUT, 1>& kvOutputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue, TBuf<> kvAntiqMxScaleRes,
                                   const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize);

    __aicore__ inline void CopyAntiqScaleE8M0(LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset,
                                              uint32_t rowCnt, uint32_t grpNum);
    __aicore__ inline void CopyAntiqScaleE8M0Nz(LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset,
                                              uint32_t rowCnt, uint32_t grpNum, uint32_t s2RealSize);
    __aicore__ inline void LoadAntiquantParamsPerTokenGroup(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                                            TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                            TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParamBaseAPI& taskParam,
                                                            bool isBeforeHalf, int32_t s2RealSize);
    __aicore__ inline void LoadAntiquantParamsPerToken(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                                       GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
                                                       GlobalTensor<int32_t>& blockTableGm,
                                                       TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                       TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                                       const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf, int32_t s2RealSize);

    __aicore__ inline void CopyAntiqParam(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                          uint32_t rowCnt, const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void LoadAntiquantParamsPerChannel(GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
                                   GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
                                   TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                    const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void FreeAntiquantParams(TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                   TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                    const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void AntiquantBaseAPI(uint32_t copyLoopIdx, TQue<QuePosition::VECIN, 1>& kvInputQue,
                                     TQue<QuePosition::VECOUT, 1>& kvOutputQue, LocalTensor<Q_T> outScm,
                                     GlobalTensor<KV_T> &kvGm, GlobalTensor<int32_t>& blockTableGm,
                                     uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam,
                                     int32_t taskId, bool isBeforeHalf, int32_t s2RealSize);
    __aicore__ inline void CopyKV(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T>& srcGm, uint64_t offset,
                                  uint32_t rowCnt, uint32_t headDim, uint32_t kvHeadNum,
                                  uint32_t paKvShapeType);
    __aicore__ inline void CopyKVPaNz(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T>& srcGm, uint64_t offset,
                                  uint32_t rowCnt, uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void CopyKVPageAttention(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T> &srcGm,
                                               GlobalTensor<int32_t>& blockTableGm, const AntiquantTaskParamBaseAPI& taskParam,
                                               uint32_t curSequence, uint32_t dealRowCount);
    __aicore__ inline void CopyAntiquantParamsPageAttention(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                                            GlobalTensor<int32_t>& blockTableGm, const AntiquantTaskParamBaseAPI& taskParam,
                                                            DataCopyExtParams copyInParams,
                                                            DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams);
    template <bool HAS_OFFSET, bool IS_PER_TOKEN>
    __aicore__ inline void CallAntiquantVF(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb,
        LocalTensor<ANTIQ_PARAMS_T>& offsetTensor, LocalTensor<ANTIQ_PARAMS_T>& scaleTensor,
        uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam);
    __aicore__ inline void AntiquantVec(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb, uint32_t copyLoopIdx,
                                        uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf);
    __aicore__ inline void CopyAntiquantResToL1BaseAPI(LocalTensor<Q_T>& antiqResScm, LocalTensor<Q_T>& antiqResUb,
                                                uint32_t copyLoopIdx, uint32_t dealRowCount,
                                                const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize);

private:
    LocalTensor<ANTIQ_PARAMS_T> antiqScale;
    LocalTensor<ANTIQ_PARAMS_T> antiqOffset;
};

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::ProcessBaseAPI(
    Buffer<BufferType::L1> &outBufAntiRes, GlobalTensor<KV_T> &kvGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm,
    GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm, GlobalTensor<int32_t>& blockTableGm,
    TQue<QuePosition::VECIN, 1>& kvInputQue, TQue<QuePosition::VECOUT, 1>& kvOutputQue,
    TQue<QuePosition::VECIN, 1>& antiqScaleInputQue, TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
    TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize)
{
    if (taskParam.isLoadAntiqParam) {
        if constexpr (KVFP4) {
            if (likely(taskParam.copyTotalS > 0)) {
                LoadAntiquantParamsPerTokenGroup(antiqScaleGm, antiqScaleInputQue, kvAntiqMxScaleRes,
                    taskParam, isBeforeHalf, s2RealSize);
            }
        } else if constexpr (ANTIQUANT_PER_TOKEN) {
            if (likely(taskParam.copyTotalS > 0)) {
                LoadAntiquantParamsPerToken(antiqScaleGm, antiqOffsetGm, blockTableGm, antiqScaleInputQue,
                                            antiqOffsetInputQue, taskParam, isBeforeHalf, s2RealSize);
            }
        } else {
            LoadAntiquantParamsPerChannel(antiqScaleGm, antiqOffsetGm, antiqScaleInputQue, antiqOffsetInputQue,
                                          taskParam);
        }
    }
    if (unlikely(taskParam.copyTotalS <= 0)) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(CV_L1_EVENT[taskId % 2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(VC_L1_EVENT[taskId % 2]);
        if (taskParam.isFreeAntiqParam) {
            if constexpr (!KVFP4 && !ANTIQUANT_PER_TOKEN) {
                FreeAntiquantParams(antiqScaleInputQue, antiqOffsetInputQue, taskParam);
            }
        }
        return;
    }
    uint32_t loopCnt = (taskParam.copyTotalS + taskParam.copySplitS - 1) / taskParam.copySplitS;
    uint32_t tailCopyS = taskParam.copyTotalS - (loopCnt - 1) * taskParam.copySplitS;
    LocalTensor<Q_T> scmTensor = outBufAntiRes.GetTensor<Q_T>();
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(CV_L1_EVENT[taskId % 2]);
    for (uint32_t i = 0, actCopyS = taskParam.copySplitS; i < loopCnt; i++) {
        if (i + 1 == loopCnt) {
            actCopyS = tailCopyS;
        }
        AntiquantBaseAPI(i, kvInputQue, kvOutputQue, scmTensor, kvGm, blockTableGm, actCopyS, taskParam, taskId, isBeforeHalf, s2RealSize);
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(VC_L1_EVENT[taskId % 2]);
    if (taskParam.isFreeAntiqParam) {
        if constexpr (KVFP4) {
        } else {
            FreeAntiquantParams(antiqScaleInputQue, antiqOffsetInputQue, taskParam);
        }
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqScaleE8M0(
    LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset, uint32_t rowCnt, uint32_t grpNum)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams{};
    copyInParams.blockCount = 1;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = grpNum * rowCnt * sizeof(int8_t);
    copyInParams.srcStride = 0;
    DataCopyPad(dstLocal, srcGm[offset], copyInParams, copyInPadParams);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqScaleE8M0Nz(
    LocalTensor<Q_T> dstLocal, GlobalTensor<Q_T>& srcGm, uint64_t offset, uint32_t rowCnt, uint32_t grpNum, uint32_t s2RealSize)
{
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams{};
    copyInParams.blockCount = grpNum;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = rowCnt * sizeof(int8_t);
    copyInParams.srcStride = s2RealSize - rowCnt;
    DataCopyPad(dstLocal, srcGm[offset], copyInParams, copyInPadParams);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerTokenGroup(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
    TBuf<> kvAntiqMxScaleRes, const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf, int32_t s2RealSize)
{
    int32_t subBlockIdx;
    if (isBeforeHalf) {
        subBlockIdx = 0;
    } else {
        subBlockIdx = 1;
    }
    uint32_t grpSize = 32;
    uint32_t grpNum = taskParam.headDim / grpSize;
    uint64_t scaleOffset = 0;
    if (taskParam.isKvCacheNz) { 
        scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize * grpNum +
                    taskParam.n2Idx * taskParam.seqSize * grpNum + taskParam.s2Idx * taskParam.singleSInnerSize * 1 +
                    taskParam.kvPaddingBeginOffset * 1 + subBlockIdx * GetRealDealSize(s2RealSize) * 1;
        if constexpr (FLASH_DECODE) {
            scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize * 1;
        }

        LocalTensor<Q_T> antiqScaleE8M0Ub = antiqScaleInputQue.template AllocTensor<Q_T>();
        CopyAntiqScaleE8M0Nz(antiqScaleE8M0Ub, antiqScaleGm, scaleOffset / 2, AlignUp32(taskParam.copyTotalS),
                        grpNum, taskParam.seqSize);
        antiqScaleInputQue.template EnQue(antiqScaleE8M0Ub);
        antiqScaleE8M0Ub = antiqScaleInputQue.DeQue<Q_T>();

        antiqScale = kvAntiqMxScaleRes.Get<ANTIQ_PARAMS_T>();
        FaVectorApi::AntiqScalePerTokenGroupByVF<Q_T, ANTIQ_PARAMS_T>(antiqScaleE8M0Ub, antiqScale, taskParam.copyTotalS, grpNum);
        antiqScaleInputQue.FreeTensor(antiqScaleE8M0Ub);
    } else {
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize * grpNum +
                        taskParam.n2Idx * taskParam.antiqSeqSize * grpNum +
                        taskParam.s2Idx * taskParam.singleSInnerSize * grpNum +
                        taskParam.kvPaddingBeginOffset * grpNum +
                        subBlockIdx * GetRealDealSize(s2RealSize) * grpNum;
            } else {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize * grpNum +
                        taskParam.n2Idx * taskParam.antiqSeqSize * grpNum +
                        (taskParam.s2Idx * taskParam.singleSInnerSize + (taskParam.antiqSeqSize - taskParam.seqSize)) * grpNum +
                        taskParam.kvPaddingBeginOffset * grpNum +
                        subBlockIdx * GetRealDealSize(s2RealSize) * grpNum;
            }
        } else {
            scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize * grpNum +
                    taskParam.n2Idx * taskParam.seqSize * grpNum + taskParam.s2Idx * taskParam.singleSInnerSize * grpNum +
                    taskParam.kvPaddingBeginOffset * grpNum + subBlockIdx * GetRealDealSize(s2RealSize) * grpNum;
        }
        if constexpr (FLASH_DECODE) {
            scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize * grpNum;
        }

        LocalTensor<Q_T> antiqScaleE8M0Ub = antiqScaleInputQue.template AllocTensor<Q_T>();
        CopyAntiqScaleE8M0(antiqScaleE8M0Ub, antiqScaleGm, scaleOffset / 2, taskParam.copyTotalS, grpNum);
        antiqScaleInputQue.template EnQue(antiqScaleE8M0Ub);
        antiqScaleE8M0Ub = antiqScaleInputQue.DeQue<Q_T>();

        antiqScale = kvAntiqMxScaleRes.Get<ANTIQ_PARAMS_T>();
        FaVectorApi::AntiqScaleByVF<Q_T, ANTIQ_PARAMS_T>(antiqScaleE8M0Ub, antiqScale, taskParam.copyTotalS, grpNum);
        antiqScaleInputQue.FreeTensor(antiqScaleE8M0Ub);
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerToken(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
    GlobalTensor<int32_t>& blockTableGm, TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
    TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue, const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf, int32_t s2RealSize)
{
    int32_t subBlockIdx;
    if (isBeforeHalf) {
        subBlockIdx = 0;
    } else {
        subBlockIdx = 1;
    }
    uint64_t scaleOffset = 0;
    if (taskParam.isPerHead) {
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize + taskParam.n2Idx * taskParam.antiqSeqSize +
                      taskParam.s2Idx * taskParam.singleSInnerSize + taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            } else {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize + taskParam.n2Idx * taskParam.antiqSeqSize +
                      taskParam.s2Idx * taskParam.singleSInnerSize + (taskParam.antiqSeqSize - taskParam.seqSize) + taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            }
        } else {
            scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize + taskParam.n2Idx * taskParam.seqSize +
                      taskParam.s2Idx * taskParam.singleSInnerSize + taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
    } else {
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                      taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            } else {
                scaleOffset = taskParam.bIdx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize + (taskParam.antiqSeqSize - taskParam.seqSize) +
                      taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            }
        } else {
            scaleOffset = taskParam.bIdx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                      taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
    }
    if constexpr (FLASH_DECODE) {
        scaleOffset += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams{};
    copyInParams.blockCount = 1;
    copyInParams.dstStride = 0;
    copyInParams.blockLen = taskParam.copyTotalS * sizeof(ANTIQ_PARAMS_T);
    copyInParams.srcStride = 0;
    LocalTensor<ANTIQ_PARAMS_T> antiqScaleTmpUb = antiqScaleInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
    if (taskParam.isPageAttentionAntiquant) {
        CopyAntiquantParamsPageAttention(
            antiqScaleTmpUb, antiqScaleGm, blockTableGm, taskParam, copyInParams, copyInPadParams);
    } else {
        DataCopyPad(antiqScaleTmpUb, antiqScaleGm[scaleOffset], copyInParams, copyInPadParams);
    }
    antiqScaleInputQue.template EnQue(antiqScaleTmpUb);
    
    if (taskParam.isKvCacheNz) {
        antiqScaleTmpUb = antiqScaleInputQue.DeQue<ANTIQ_PARAMS_T>();
        FaVectorApi::AntiqScaleByPertokenVF<Q_T, ANTIQ_PARAMS_T>(antiqScaleTmpUb, taskParam.copyTotalS);
        antiqScaleInputQue.template EnQue(antiqScaleTmpUb);
    } 
    antiqScale = antiqScaleInputQue.DeQue<ANTIQ_PARAMS_T>();

    if (taskParam.isExistOffset) {
        LocalTensor<ANTIQ_PARAMS_T> antiqOffsetTmpUb = antiqOffsetInputQue.template AllocTensor<ANTIQ_PARAMS_T>();
        if (taskParam.isPageAttentionAntiquant) {
            CopyAntiquantParamsPageAttention(
                antiqOffsetTmpUb, antiqOffsetGm, blockTableGm, taskParam, copyInParams, copyInPadParams);
        } else {
            DataCopyPad(antiqOffsetTmpUb, antiqOffsetGm[scaleOffset], copyInParams, copyInPadParams);
        }
        antiqOffsetInputQue.template EnQue(antiqOffsetTmpUb);
        if (taskParam.isKvCacheNz) {
            antiqOffsetTmpUb = antiqOffsetInputQue.DeQue<ANTIQ_PARAMS_T>();
            FaVectorApi::AntiqScaleByPertokenVF<Q_T, ANTIQ_PARAMS_T>(antiqOffsetTmpUb, taskParam.copyTotalS);
            antiqOffsetInputQue.template EnQue(antiqOffsetTmpUb);
        }
        antiqOffset = antiqOffsetInputQue.DeQue<ANTIQ_PARAMS_T>();
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiqParam(LocalTensor<ANTIQ_PARAMS_T> dstLocal, GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                          uint32_t rowCnt, const AntiquantTaskParamBaseAPI& taskParam)
{
    if (taskParam.isPertensor) {
        Duplicate(dstLocal, srcGm.GetValue(0), rowCnt * dBaseSize);
    } else if (taskParam.isPerHead) {
        Duplicate(dstLocal, srcGm.GetValue(taskParam.n2Idx), rowCnt * dBaseSize);
    } else {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<ANTIQ_PARAMS_T> copyInPadParams;
        copyInParams.blockCount = rowCnt;
        copyInParams.blockLen = taskParam.headDim * sizeof(ANTIQ_PARAMS_T);
        copyInParams.srcStride = static_cast<int64_t>(copyInParams.blockLen) * (-1);
        copyInParams.dstStride = (dBaseSize - taskParam.headDim) * sizeof(ANTIQ_PARAMS_T) / BYTE_BLOCK;

        copyInPadParams.isPad = false;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = 0;
        copyInPadParams.paddingValue = 0;
    
        DataCopyPad(dstLocal, srcGm[taskParam.antiqParamOffset], copyInParams, copyInPadParams);
    }
}
template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::LoadAntiquantParamsPerChannel(
    GlobalTensor<ANTIQ_PARAMS_T>& antiqScaleGm, GlobalTensor<ANTIQ_PARAMS_T>& antiqOffsetGm,
    TQue<QuePosition::VECIN, 1>& antiqScaleInputQue, TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
    const AntiquantTaskParamBaseAPI& taskParam)
{
    uint32_t qElementCntPerReg = 256 / sizeof(ANTIQ_PARAMS_T);
    uint32_t loopCount = qElementCntPerReg / taskParam.headDimAlignBlock;
    if( loopCount == 0){
        loopCount =1;
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

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::FreeAntiquantParams(TQue<QuePosition::VECIN, 1>& antiqScaleInputQue,
                                                                     TQue<QuePosition::VECIN, 1>& antiqOffsetInputQue,
                                                                     const AntiquantTaskParamBaseAPI& taskParam)
{
    antiqScaleInputQue.FreeTensor(antiqScale);
    if (taskParam.isExistOffset) {
        antiqOffsetInputQue.FreeTensor(antiqOffset);
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::AntiquantBaseAPI(uint32_t copyLoopIdx,
                                                           TQue<QuePosition::VECIN, 1>& kvInputQue,
                                                           TQue<QuePosition::VECOUT, 1>& kvOutputQue,
                                                           LocalTensor<Q_T> outScm, GlobalTensor<KV_T> &kvGm,
                                                           GlobalTensor<int32_t>& blockTableGm, uint32_t dealRowCount,
                                                           const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize)
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
    AntiquantVec(outUb, tmpUb, copyLoopIdx, dealRowCount, taskParam, isBeforeHalf);
    kvInputQue.FreeTensor(tmpUb);

    kvOutputQue.template EnQue(outUb);
    kvOutputQue.DeQue<Q_T>();
    CopyAntiquantResToL1BaseAPI(outScm, outUb, copyLoopIdx, dealRowCount, taskParam, taskId, isBeforeHalf, s2RealSize);
    kvOutputQue.FreeTensor(outUb);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiquantParamsPageAttention(LocalTensor<ANTIQ_PARAMS_T> dstLocal,
                                                                      GlobalTensor<ANTIQ_PARAMS_T>& srcGm,
                                                                      GlobalTensor<int32_t>& blockTableGm,
                                                                      const AntiquantTaskParamBaseAPI& taskParam,
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
    uint32_t curSequence = taskParam.s2BatchOffset;
    while (copyFinishElmenCnt < taskParam.copyTotalS) {
        uint64_t blockIdOffset = curSequence / taskParam.kvCacheBlockSize;
        uint64_t remainElmenCnt = curSequence % taskParam.kvCacheBlockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyElmenCnt = taskParam.kvCacheBlockSize - remainElmenCnt;
        if (copyElmenCnt + copyFinishElmenCnt > taskParam.copyTotalS) {
            copyElmenCnt = taskParam.copyTotalS - copyFinishElmenCnt;
        }
        uint64_t srcOffset = idInBlockTable * taskParam.kvCacheBlockSize * useKvHeadNum +
                             taskParam.kvCacheBlockSize * useN2Idx + remainElmenCnt;
        copyInParams.blockLen = copyElmenCnt * sizeof(ANTIQ_PARAMS_T);
        DataCopyPad(dstLocal[dstOffset], srcGm[srcOffset], copyInParams, copyInPadParams);
        dstOffset += copyElmenCnt;
        copyFinishElmenCnt += copyElmenCnt;
        curSequence += copyElmenCnt;
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKV(LocalTensor<KV_T> dstLocal,
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

    copyInParams.dstStride = (dBaseSize - headDim) / typeElementSize;

    if constexpr (PAGE_ATTENTION) {
        if (paKvShapeType == 0) {
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        } else {
            copyInParams.srcStride = 0;
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
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

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKVPaNz(LocalTensor<KV_T> dstLocal, 
                                                                             GlobalTensor<KV_T> &srcGm, uint64_t offset, uint32_t rowCnt,
                                                                             uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint32_t blockElemNum;
    if constexpr (KVINT4 || KVFP4) {
        blockElemNum = ONE_BLK_SIZE * 2;
    } else {
        blockElemNum = ONE_BLK_SIZE / sizeof(KV_T);
    }
    uint32_t actDataLen = (rowCnt * typeElementSize + blockElemNum - 1) / blockElemNum;
    uint32_t dstRowStride = (dealRowCount * typeElementSize + blockElemNum - 1) / blockElemNum;
    uint32_t srcRowStride = taskParam.kvCacheBlockSize * typeElementSize / blockElemNum;
    DataCopyParams intriParams;
    intriParams.blockCount = taskParam.headDim / typeElementSize;
    intriParams.blockLen = actDataLen;
    intriParams.dstStride = dstRowStride - actDataLen;
    intriParams.srcStride = srcRowStride - actDataLen;
    DataCopy(dstLocal, srcGm[offset], intriParams);
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyKVPageAttention(LocalTensor<KV_T> dstLocal,
                                                                     GlobalTensor<KV_T> &srcGm,
                                                                     GlobalTensor<int32_t>& blockTableGm,
                                                                     const AntiquantTaskParamBaseAPI& taskParam,
                                                                     uint32_t curSequence, uint32_t dealRowCount)
{
    uint32_t typeElementSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint64_t blockTableBaseOffset = taskParam.bIdx * taskParam.maxBlockNumPerSeq;
    uint32_t copyFinishRowCnt = 0;
    while (copyFinishRowCnt < dealRowCount) {
        uint64_t blockIdOffset = curSequence / taskParam.kvCacheBlockSize;
        uint64_t reaminRowCnt = curSequence % taskParam.kvCacheBlockSize;
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = taskParam.kvCacheBlockSize - reaminRowCnt;
        if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
            copyRowCnt = dealRowCount - copyFinishRowCnt;
        }
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_NZ)) { // NZ
            curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.headDim * taskParam.kvCacheBlockSize +
                (uint64_t)(taskParam.n2Idx * taskParam.headDim * taskParam.kvCacheBlockSize) +
                reaminRowCnt * typeElementSize;
            CopyKVPaNz(dstLocal[copyFinishRowCnt * typeElementSize], srcGm, curOffset, copyRowCnt, dealRowCount, taskParam);
        } else {
            if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_BSH)) { // BBH
                curOffset =
                    (idInBlockTable * taskParam.kvCacheBlockSize + reaminRowCnt) * taskParam.kvHeadNum * taskParam.headDim +
                    (uint64_t)(taskParam.n2Idx * taskParam.headDim);
            } else { // BNBD
                curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.kvCacheBlockSize * taskParam.headDim +
                            (uint64_t)(taskParam.n2Idx * taskParam.kvCacheBlockSize * taskParam.headDim) +
                            reaminRowCnt * taskParam.headDim;
            }
            CopyKV(dstLocal[copyFinishRowCnt * dBaseSize], srcGm, curOffset, copyRowCnt, taskParam.headDim,
                taskParam.kvHeadNum, taskParam.paKvShapeType);
        }

        copyFinishRowCnt += copyRowCnt;
        curSequence += copyRowCnt;
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
template <bool HAS_OFFSET, bool IS_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CallAntiquantVF(
    LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb,
    LocalTensor<ANTIQ_PARAMS_T>& offsetTensor, LocalTensor<ANTIQ_PARAMS_T>& scaleTensor,
    uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam)
{
    if (taskParam.isKvCacheNz) {
        FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, HAS_OFFSET, IS_PER_TOKEN, true>(
            antiqInUb, antiqResUb, offsetTensor, scaleTensor, dealRowCount, taskParam.headDim, taskParam.copyTotalS);
    } else {
        FaVectorApi::AntiquantVF<Q_T, KV_T, ANTIQ_PARAMS_T, dBaseSize, HAS_OFFSET, IS_PER_TOKEN, false>(
            antiqInUb, antiqResUb, offsetTensor, scaleTensor, dealRowCount, taskParam.headDim, taskParam.copyTotalS);
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::AntiquantVec(LocalTensor<Q_T>& antiqResUb, LocalTensor<KV_T>& antiqInUb,
                                                              uint32_t copyLoopIdx, uint32_t dealRowCount,
                                                              const AntiquantTaskParamBaseAPI& taskParam, bool isBeforeHalf)
{
    if constexpr (KVFP4) {
        uint32_t grpNum = taskParam.headDim / 32;
        uint32_t perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS * grpNum * 2;
        if (taskParam.isKvCacheNz) {
            perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS;
        }
        LocalTensor<ANTIQ_PARAMS_T> antiqScaleWithOffset = antiqScale[perTokenScaleOffset];
        CallAntiquantVF<false, false>(antiqResUb, antiqInUb, antiqOffset,
            antiqScaleWithOffset, dealRowCount, taskParam);
    } else if constexpr (ANTIQUANT_PER_TOKEN) {
        uint32_t perTokenScaleOffset = copyLoopIdx * taskParam.copySplitS;
        if (taskParam.isKvCacheNz) {
            perTokenScaleOffset = perTokenScaleOffset >> 1U;
        }
        LocalTensor<ANTIQ_PARAMS_T> antiqScaleWithOffset = antiqScale[perTokenScaleOffset];
        if (taskParam.isExistOffset) {
            LocalTensor<ANTIQ_PARAMS_T> antiqOffsetWithOffset = antiqOffset[perTokenScaleOffset];
            CallAntiquantVF<true, true>(antiqResUb, antiqInUb, antiqOffsetWithOffset,
                antiqScaleWithOffset, dealRowCount, taskParam);
        } else {
            CallAntiquantVF<false, true>(antiqResUb, antiqInUb, antiqOffset,
                antiqScaleWithOffset, dealRowCount, taskParam);
        }
    } else {
        if (taskParam.isExistOffset) {
            CallAntiquantVF<true, false>(antiqResUb, antiqInUb, antiqOffset, antiqScale, dealRowCount, taskParam);
        } else {
            CallAntiquantVF<false, false>(antiqResUb, antiqInUb, antiqOffset, antiqScale, dealRowCount, taskParam);
        }
    }
}

template <ANTIQUANT_PROCESSOR_TEMPLATE_DEF, const bool ANTIQUANT_PER_TOKEN>
__aicore__ inline void AntiquantProcessorBaseAPI<ANTIQUANT_TEMPLATE_ARGS, ANTIQUANT_PER_TOKEN>::CopyAntiquantResToL1BaseAPI(
        LocalTensor<Q_T>& antiqResScm, LocalTensor<Q_T>& antiqResUb, uint32_t copyLoopIdx,
        uint32_t dealRowCount, const AntiquantTaskParamBaseAPI& taskParam, int32_t taskId, bool isBeforeHalf, int32_t s2RealSize)
{
    uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint16_t dstStep = ALIGN((uint16_t)s2RealSize, (uint16_t)16);
    int32_t subBlockIdx;
    if (isBeforeHalf) {
        subBlockIdx = 0;
    } else {
        subBlockIdx = 1;
    }
    uint64_t outOffset = subBlockIdx * GetRealDealSize(s2RealSize) * 16 + copyLoopIdx * taskParam.copySplitS * 16;

    struct DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = taskParam.headDimAlignBlock / elementTypeSize;
    dataCopyParams.blockLen = dealRowCount;
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = dstStep - dealRowCount;
    if constexpr (PAGE_ATTENTION) {
        dataCopyParams.srcStride = taskParam.isKvCacheNz ? 0 : 1;
    }

    DataCopy(antiqResScm[outOffset], antiqResUb, dataCopyParams);
}
}
#endif
