/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_generalized.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_E_K_GENERALIZED_H
#define MOE_GATING_TOP_K_E_K_GENERALIZED_H
#include "kernel_operator.h"
#include "common.h"
#include "kernel_utils.h"
namespace MoeGatingTopK {
using namespace AscendC;

template <typename T>
class MoeGatingTopKGenerlized {
public:
    __aicore__ inline MoeGatingTopKGenerlized(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBiasAndInitExpertId();
    __aicore__ inline void CopyInX(int64_t progress);
    __aicore__ inline void ComputeX();
    __aicore__ inline void CopuOutXNorm(int64_t row);
    __aicore__ inline void SortInGroup();
    __aicore__ inline void SelectTopKGroupIndex();
    __aicore__ inline void SelectTopKExpertIdx();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CumputeActualTopKExpertId();
    __aicore__ inline void CopyOut(int64_t row);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TBuf<TPosition::VECCALC> biasBuf_;          // Store input bias
    TBuf<TPosition::VECCALC> expertIdBuf_;      // Expert ID
    TBuf<TPosition::VECCALC> xNormWithBiasBuf_; // Store value after adding bias
    TBuf<TPosition::VECCALC> xNormBuf_;         // Store value after computing sigmoid or softmax
    TBuf<TPosition::VECCALC> sortedInGroupBuf_; // Store sorted results within groups
    TBuf<TPosition::VECCALC> topKExpertIdBuf_;
    TBuf<TPosition::VECCALC> sortedGroupIndexBuf_;
    TBuf<TPosition::VECCALC> calcTmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<float> outGm_;

    int64_t blockIdx_ = 0;
    int64_t perCoreRowCount_ = 0;
    int64_t curCoreRowCount_ = 0;
    int64_t expertCount_ = 0;
    bool addBias_ = false;
    int64_t k_ = 0;
    int64_t kGroup_ = 0;
    int64_t groupCount_ = 0;
    int64_t groupCountAlign_ = 0;
    int64_t perGroupExpertCount_ = 0;
    int64_t perGroupExpertCountAlign_ = 0;
    int64_t groupSelectMode_ = 0;
    int64_t renorm_ = 0;
    int64_t normType_ = 0;
    int64_t outFlag_ = 0;

    int64_t expertCountAlign_ = 0;
    int64_t kAlign_ = 0;
    bool isAlign_ = false;

    const MoeGatingTopKTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::CopyInBiasAndInitExpertId()
{
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();
    LocalTensor<int32_t> expertIdTensor = expertIdBuf_.Get<int32_t>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = groupCount_;
    dataCopyParams.blockLen = perGroupExpertCount_ * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES;

    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if (addBias_) {
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        } else {
            DataCopyPad(biasTensor[expertCountAlign_].ReinterpretCast<T>(), biasGm_, dataCopyParams, dataCopyPadParams);
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(biasTensor, biasTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
                 expertCountAlign_);
            PipeBarrier<PIPE_V>();
        }

        if (!isAlign_) {
            int64_t duplicateNum = perGroupExpertCount_ % ONE_REPEAT_SORT_NUM;
            int duplicateIndex = perGroupExpertCount_ - duplicateNum;
            if (duplicateNum > 0) {
                uint64_t mask0 = UINT64_MAX;
                mask0 = mask0 << duplicateNum;
                mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
                uint64_t mask[2] = {mask0, 0};
                Duplicate(biasTensor.ReinterpretCast<int32_t>()[duplicateIndex], FLOAT32_NEG_INF, mask, groupCount_, 1,
                          perGroupExpertCountAlign_ * sizeof(float) / BLOCK_BYTES);
            }
        }
    }
    ArithProgression(expertIdTensor, static_cast<int32_t>(0), static_cast<int32_t>(1), expertCountAlign_);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::CopyInX(int64_t row)
{
    LocalTensor<float> xInLocalTensor = xInQueue_.AllocTensor<float>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = groupCount_;
    dataCopyParams.blockLen = perGroupExpertCount_ * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES;

    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
    } else {
        DataCopyPad(xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), xGm_[row * expertCount_], dataCopyParams,
                    dataCopyPadParams);
    }
    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::ComputeX()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> xInLocalTensor = xInQueue_.DeQue<float>();
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<float> biasTensor = biasBuf_.Get<float>();

    if constexpr (!IsSameType<T, float>::value) {
        Cast(xInLocalTensor, xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
             expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }

    int64_t duplicateNum = perGroupExpertCount_ % ONE_REPEAT_SORT_NUM;
    int duplicateIndex = perGroupExpertCount_ - duplicateNum;
    if (!isAlign_ && duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(xInLocalTensor.ReinterpretCast<int32_t>()[duplicateIndex], FLOAT32_NEG_INF, mask, groupCount_, 1,
                  (perGroupExpertCountAlign_ * sizeof(float)) / BLOCK_BYTES);
        PipeBarrier<PIPE_V>();
    }
       if (normType_ == 1) { // sigmoid
            LocalTensor<uint8_t> calcNormTmpTensor = calcTmpBuf_.Get<uint8_t>();
            Sigmoid(xNormTensor, xInLocalTensor, calcNormTmpTensor, expertCountAlign_);
            PipeBarrier<PIPE_V>();
    }
        else if (normType_ == 0) { // softmax
        LocalTensor<float> reduceValueTensor = calcTmpBuf_.Get<float>();
        LocalTensor<float> calcTmp = calcTmpBuf_.Get<float>()[BLOCK_BYTES];
        ReduceMax(reduceValueTensor, xInLocalTensor, calcTmp, expertCountAlign_);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float maxValue = reduceValueTensor.GetValue(0);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Adds(xNormTensor, xInLocalTensor, -maxValue, expertCountAlign_);
        PipeBarrier<PIPE_V>();
        Exp(xNormTensor, xNormTensor, expertCountAlign_);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceValueTensor, xNormTensor, calcTmp, expertCountAlign_);
        eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float sumValue = reduceValueTensor.GetValue(0);
        eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Muls(xNormTensor, xNormTensor, 1.0f / sumValue, expertCountAlign_);
        PipeBarrier<PIPE_V>();
    }
    if (addBias_) {
        Add(xNormWithBiasTensor, xNormTensor, biasTensor, expertCountAlign_);
    } else {
        DataCopy(xNormWithBiasTensor, xNormTensor, expertCountAlign_);
    }

    if (!isAlign_ && duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        PipeBarrier<PIPE_V>();
        Duplicate(xNormWithBiasTensor.ReinterpretCast<int32_t>()[duplicateIndex],
                  FLOAT32_NEG_INF, // MIN_FP32,
                  mask, groupCount_, 1, perGroupExpertCountAlign_ * sizeof(float) / BLOCK_BYTES);
    }
    xInQueue_.FreeTensor(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::CopuOutXNorm(int64_t row)
{
    LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    DataCopy(outOutTensor, xNormTensor, expertCountAlign_);
    outOutQueue_.EnQue<float>(outOutTensor);
    outOutTensor = outOutQueue_.DeQue<float>();
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(groupCount_), static_cast<uint32_t>(perGroupExpertCount_ * sizeof(float)),
        static_cast<uint32_t>((perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(float) / BLOCK_BYTES), 0, 0};
    DataCopyPad(outGm_[row * expertCount_], outOutTensor, dataCopyParams);
    outOutQueue_.FreeTensor(outOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::SortInGroup()
{
    LocalTensor<float> xNormWithBiasTensor = xNormWithBiasBuf_.Get<float>();
    LocalTensor<uint32_t> expertIdTensor = expertIdBuf_.Get<uint32_t>();
    LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<float> tmpLocal = calcTmpBuf_.Get<float>();
    if (perGroupExpertCountAlign_ == ONE_REPEAT_SORT_NUM) {
        PipeBarrier<PIPE_V>();
        Sort32(sortedInGroupTensor, xNormWithBiasTensor, expertIdTensor, groupCount_);
    } else {
        for (int64_t group = 0; group < groupCount_; group++) {
            PipeBarrier<PIPE_V>();
            Sort<float, true>(sortedInGroupTensor[group * perGroupExpertCountAlign_ * CONSTANT_TWO],
                              xNormWithBiasTensor[group * perGroupExpertCountAlign_],
                              expertIdTensor[group * perGroupExpertCountAlign_], tmpLocal,
                              perGroupExpertCountAlign_ / ONE_REPEAT_SORT_NUM);
        }
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::SelectTopKGroupIndex()
{
    LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<float> valueSelectedFromGroupTensor = calcTmpBuf_.GetWithOffset<float>(groupCountAlign_ * 2, 0);
    LocalTensor<uint32_t> maskTensor =
        calcTmpBuf_.GetWithOffset<uint32_t>(groupCountAlign_, groupCountAlign_ * 2 * sizeof(float));
    LocalTensor<float> topValueInGroupTensor =
        calcTmpBuf_.GetWithOffset<float>(groupCountAlign_, groupCountAlign_ * 3 * sizeof(float));
    LocalTensor<uint32_t> groupIndex =
        calcTmpBuf_.GetWithOffset<uint32_t>(groupCountAlign_, groupCountAlign_ * 4 * sizeof(float));
    LocalTensor<float> sortedTopValue =
        calcTmpBuf_.GetWithOffset<float>(groupCountAlign_ * 2, groupCountAlign_ * 5 * sizeof(float));
    LocalTensor<float> sortTmp =
        calcTmpBuf_.GetWithOffset<float>(groupCountAlign_ * 2, groupCountAlign_ * 7 * sizeof(float));

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    uint64_t rsvdCnt = 0; // Used to store the number of elements retained after filtering
    PipeBarrier<PIPE_V>();
    if (groupSelectMode_ == 1) {              // top2 sum
                                                          // Extract the first two elements of each group
        maskTensor.SetValue(0, static_cast<uint32_t>(5)); // b0101
        maskTensor.SetValue(1, static_cast<uint32_t>(0));
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);

        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = groupCount_;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride =
            Ceil(perGroupExpertCountAlign_ * (sizeof(float) + sizeof(uint32_t)), BLOCK_BYTES);
        gatherMaskParams.src1RepeatStride = 0;
        GatherMask(valueSelectedFromGroupTensor, sortedInGroupTensor, maskTensor, true,
                   static_cast<uint32_t>(ONE_REPEAT_SORT_NUM * CONSTANT_TWO), gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        // Calculate the sum of the first two numbers in each group
        PairReduceSum(topValueInGroupTensor, valueSelectedFromGroupTensor,
                      Ceil(groupCount_ * sizeof(float) * 2, REPEAT_BYTES), REPEAT_BYTES / sizeof(float), 1, 1,
                      CONSTANT_EIGHT); // Calculate the sum of the two largest numbers in each group
    } else {
        maskTensor.SetValue(0, static_cast<uint32_t>(1)); // b0101
        maskTensor.SetValue(1, static_cast<uint32_t>(0));

        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        uint64_t rsvdCnt = 0; // Used to store the number of elements retained after filtering
        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = groupCount_;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride = Ceil(perGroupExpertCountAlign_ * (sizeof(float) + sizeof(uint32_t)), 32);
        gatherMaskParams.src1RepeatStride = 0;
        GatherMask(topValueInGroupTensor, sortedInGroupTensor, maskTensor, true,
                   static_cast<uint32_t>(ONE_REPEAT_SORT_NUM * CONSTANT_TWO), gatherMaskParams, rsvdCnt);
    }

    PipeBarrier<PIPE_V>();
    // Generate group indices
    ArithProgression(groupIndex.ReinterpretCast<int32_t>(), static_cast<int32_t>(0), static_cast<int32_t>(1),
                     groupCount_); // Generate group indices
    PipeBarrier<PIPE_V>();

    int64_t duplicateNum = groupCount_ % ONE_REPEAT_SORT_NUM;
    int duplicateIndex = groupCount_ - duplicateNum;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(topValueInGroupTensor.ReinterpretCast<int32_t>()[duplicateIndex], FLOAT32_NEG_INF, mask, 1, 1,
                  REPEAT_BLOCKS);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();

    // Sort
    Sort<float, true>(sortedTopValue, topValueInGroupTensor, groupIndex, sortTmp, Ceil(groupCount_, 32));
    PipeBarrier<PIPE_V>();

    // Extract group indices
    uint8_t src1Pattern = 2; // Built-in fixed pattern
    GatherMask(groupIndex, sortedTopValue.template ReinterpretCast<uint32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0),
               {1, static_cast<uint8_t>(Ceil(kGroup_ * sizeof(float) * CONSTANT_TWO, 256)), REPEAT_BLOCKS, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    duplicateNum = kGroup_ % ONE_REPEAT_SORT_NUM;
    if (duplicateNum > 0) {
        duplicateIndex = kGroup_ - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
        uint64_t mask[2] = {mask0, 0};
        PipeBarrier<PIPE_V>();
        Duplicate(groupIndex.ReinterpretCast<int32_t>()[duplicateIndex], FLOAT32_NEG_INF, mask, 1, 1, REPEAT_BLOCKS);
    }

    // Sort the selected group indices in descending order
    LocalTensor<float> sortedGroupIndex = sortedGroupIndexBuf_.Get<float>();
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedGroupIndex, groupIndex.ReinterpretCast<float>(), groupIndex, sortTmp, Ceil(kGroup_, 32));
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::SelectTopKExpertIdx()
{
    LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
    LocalTensor<int32_t> sortedGroupIndex = sortedGroupIndexBuf_.Get<int32_t>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<float> mrgSort0Tensor = calcTmpBuf_.Get<float>();

    uint32_t offset[CONSTANT_FOUR] = {0, 0, 0, 0};
    uint16_t lenArr[CONSTANT_FOUR] = {
        static_cast<uint16_t>(perGroupExpertCount_), static_cast<uint16_t>(perGroupExpertCount_),
        static_cast<uint16_t>(perGroupExpertCount_), static_cast<uint16_t>(perGroupExpertCount_)};
    MrgSort4Info params{lenArr, false, 0b1111, 1};
    MrgSortSrcList<float> srcList;

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    for (int32_t i = kGroup_ - 1; i >= 0; i -= CONSTANT_FOUR) {
        int64_t mrgLen = Min(i + 1, CONSTANT_FOUR);
        if (mrgLen > 1) {
            if (mrgLen == MERGE_LIST_FOUR) {
                offset[0] = sortedGroupIndex.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                offset[1] = sortedGroupIndex.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
                offset[2] = sortedGroupIndex.GetValue((i - 2) * 2) * perGroupExpertCountAlign_ * 2;
                offset[3] = sortedGroupIndex.GetValue((i - 3) * 2) * perGroupExpertCountAlign_ * 2;
            } else if (mrgLen == MERGE_LIST_THREE) {
                offset[0] = sortedGroupIndex.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                offset[1] = sortedGroupIndex.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
                offset[2] = sortedGroupIndex.GetValue((i - 2) * 2) * perGroupExpertCountAlign_ * 2;
                offset[3] = 0;
                params.elementLengths[3] = 0;
                params.validBit = 0b111;
            } else {
                offset[0] = sortedGroupIndex.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                offset[1] = sortedGroupIndex.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
                offset[2] = 0;
                offset[3] = 0;
                params.elementLengths[2] = 0;
                params.elementLengths[3] = 0;
                params.validBit = 0b11;
            }

            srcList.src1 = sortedInGroupTensor[offset[0]];
            srcList.src2 = sortedInGroupTensor[offset[1]];
            srcList.src3 = sortedInGroupTensor[offset[2]];
            srcList.src4 = sortedInGroupTensor[offset[3]];

            PipeBarrier<PIPE_V>();
            MrgSort(mrgSort0Tensor[(kGroup_ - 1 - i) * perGroupExpertCountAlign_ * 2], srcList, params);
        } else {
            offset[0] = sortedGroupIndex.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
            PipeBarrier<PIPE_V>();
            DataCopy(mrgSort0Tensor[(kGroup_ - 1 - i) * perGroupExpertCountAlign_ * 2], sortedInGroupTensor[offset[0]],
                     perGroupExpertCountAlign_ * 2);
        }
    }
    int32_t baseLoop = 4;
    LocalTensor<float> srcTensor = mrgSort0Tensor;
    LocalTensor<float> dstTensor = mrgSort0Tensor;
    for (int i = 0; i < tilingData_->vmsCount; i++) {
        if (i % 2 == 0) {
            srcTensor = mrgSort0Tensor;
            dstTensor = sortedInGroupTensor;
        } else {
            srcTensor = sortedInGroupTensor;
            dstTensor = mrgSort0Tensor;
        }

        int32_t nextBaseRow = baseLoop * MERGE_LIST_FOUR;
        int32_t quotient = kGroup_ / nextBaseRow;
        int32_t remainder = kGroup_ - quotient * nextBaseRow;
        if (quotient > 0) {
            MrgSort4Info params;
            MrgSortSrcList<float> srcList;
            params.ifExhaustedSuspension = false;
            params.elementLengths[0] = perGroupExpertCount_ * baseLoop;
            params.elementLengths[1] = perGroupExpertCount_ * baseLoop;
            params.elementLengths[2] = perGroupExpertCount_ * baseLoop;
            params.elementLengths[3] = perGroupExpertCount_ * baseLoop;
            params.validBit = 0b1111;
            params.repeatTimes = 1;
            for (int j = 0; j < quotient; j++) {
                srcList.src1 = srcTensor[perGroupExpertCountAlign_ * baseLoop * 8 * j];
                srcList.src2 = srcTensor[perGroupExpertCountAlign_ * baseLoop * (8 * j + 2)];
                srcList.src3 = srcTensor[perGroupExpertCountAlign_ * baseLoop * (8 * j + 4)];
                srcList.src4 = srcTensor[perGroupExpertCountAlign_ * baseLoop * (8 * j + 6)];
                PipeBarrier<PIPE_V>();
                MrgSort(dstTensor[perGroupExpertCountAlign_ * baseLoop * 8 * j], srcList, params);
            }
        }

        if (remainder > 0) {
            int32_t baseOffset = quotient * nextBaseRow * perGroupExpertCountAlign_ * 2;
            int32_t mrgLen = CeilDiv(remainder, baseLoop);
            int32_t tailRow = remainder - (mrgLen - 1) * baseLoop;
            if (mrgLen > 1) {
                MrgSort4Info params;
                MrgSortSrcList<float> srcList;
                params.repeatTimes = 1;
                params.ifExhaustedSuspension = false;
                params.elementLengths[0] = perGroupExpertCount_ * baseLoop;
                params.elementLengths[1] = perGroupExpertCount_ * baseLoop;
                params.elementLengths[2] = perGroupExpertCount_ * baseLoop;
                params.elementLengths[3] = perGroupExpertCount_ * baseLoop;
                srcList.src1 = srcTensor[baseOffset];
                srcList.src2 = srcTensor[baseOffset + perGroupExpertCountAlign_ * baseLoop * 2];
                if (mrgLen == MERGE_LIST_FOUR) {
                    srcList.src3 = srcTensor[baseOffset + perGroupExpertCountAlign_ * baseLoop * 2 * 2];
                    srcList.src4 = srcTensor[baseOffset + perGroupExpertCountAlign_ * baseLoop * 2 * 3];
                    params.elementLengths[3] = perGroupExpertCount_ * tailRow;
                    params.validBit = 0b1111;
                } else if (mrgLen == MERGE_LIST_THREE) {
                    srcList.src3 = srcTensor[baseOffset + perGroupExpertCountAlign_ * baseLoop * 2 * 2];
                    params.elementLengths[2] = perGroupExpertCount_ * tailRow;
                    params.elementLengths[3] = 0;
                    params.validBit = 0b111;
                } else {
                    params.elementLengths[1] = perGroupExpertCount_ * tailRow;
                    params.elementLengths[2] = 0;
                    params.elementLengths[3] = 0;
                    params.validBit = 0b11;
                }
                PipeBarrier<PIPE_V>();
                MrgSort(dstTensor[baseOffset], srcList, params);
            } else {
                PipeBarrier<PIPE_V>();
                DataCopy(dstTensor[baseOffset], srcTensor[baseOffset], tailRow * perGroupExpertCountAlign_ * 2);
            }
        }
        baseLoop = nextBaseRow;
    }

    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(k_ * sizeof(float) * 2, REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = REPEAT_BLOCKS;
    gatherMaskParams.src1RepeatStride = 0;

    uint64_t rsvdCnt = 0;    // Used to store the number of elements retained after filtering
    uint8_t src1Pattern = 2; // Built-in fixed pattern
    PipeBarrier<PIPE_V>();
    GatherMask(topKExpertId, dstTensor.template ReinterpretCast<int32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::SelectTopKExpertScore()
{
    LocalTensor<float> xNormTensor = xNormBuf_.Get<float>();
    LocalTensor<float> yOutTensor = yOutQueue_.AllocTensor<float>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<int32_t> topKExpertIdWithByte = calcTmpBuf_.Get<int32_t>();
    PipeBarrier<PIPE_V>();
    Muls(topKExpertIdWithByte, topKExpertId, static_cast<int32_t>(sizeof(float)), k_);
    PipeBarrier<PIPE_V>();
    Gather(yOutTensor, xNormTensor, topKExpertIdWithByte.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
           k_);
    bool needRenorm = (normType_ == 1 ) ||  // Case 1: sigmoid + renorm
                      (normType_ == 0 && renorm_ == 1);   // Case 3: softmax + renorm
    if (needRenorm) {       
        LocalTensor<float> maxValueTensor = calcTmpBuf_.Get<float>();
        LocalTensor<float> tmpTensor = calcTmpBuf_.Get<float>()[32];
        PipeBarrier<PIPE_V>();
        ReduceSum(maxValueTensor, yOutTensor, tmpTensor, k_);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        float sumValue = maxValueTensor.GetValue(0) + tilingData_->eps;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Duplicate(tmpTensor, sumValue, k_);
        PipeBarrier<PIPE_V>();
        Div(yOutTensor, yOutTensor, tmpTensor, k_);
    }
    PipeBarrier<PIPE_V>();
    Muls(yOutTensor, yOutTensor, tilingData_->routedScalingFactor, k_);

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(yOutTensor.ReinterpretCast<T>(), yOutTensor, RoundMode::CAST_RINT, k_);
    }

    yOutQueue_.EnQue<float>(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::CumputeActualTopKExpertId()
{
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> topKExpertId = topKExpertIdBuf_.Get<int32_t>();
    LocalTensor<float> topKExpertIdFp32 = calcTmpBuf_.Get<float>();

    PipeBarrier<PIPE_V>();
    Cast(topKExpertIdFp32, topKExpertId, RoundMode::CAST_ROUND, k_);
    PipeBarrier<PIPE_V>();
    Muls(topKExpertIdFp32, topKExpertIdFp32, 1.0f / (float)perGroupExpertCountAlign_, k_);
    PipeBarrier<PIPE_V>();
    Cast(expertIdxOut, topKExpertIdFp32, RoundMode::CAST_TRUNC, k_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdxOut, expertIdxOut, static_cast<int32_t>(perGroupExpertCountAlign_ - perGroupExpertCount_), k_);
    PipeBarrier<PIPE_V>();
    Sub(expertIdxOut, topKExpertId, expertIdxOut, k_);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::CopyOut(int64_t row)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxOut = expertIdxOutQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);
    dataCopyParams.blockLen = k_ * sizeof(int32_t);
    DataCopyPad(expertIdxGm_[row * k_], expertIdxOut, dataCopyParams);
    yOutQueue_.FreeTensor(yOutTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxOut);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
                                                        GM_ADDR out, GM_ADDR workspace,
                                                        const MoeGatingTopKTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    perCoreRowCount_ = tilingData_->perCoreRowCount;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreRowCount_ = tilingData_->lastCoreRowCount;
    } else {
        curCoreRowCount_ = tilingData_->perCoreRowCount;
    }
    expertCount_ = tilingData_->expertCount;
    addBias_ = tilingData_->addBias == 1;
    k_ = tilingData_->k;
    kGroup_ = tilingData_->kGroup;
    groupCount_ = tilingData_->groupCount;
    groupCountAlign_ = Ceil(groupCount_, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    perGroupExpertCount_ = tilingData_->perGroupExpertCount;
    perGroupExpertCountAlign_ = tilingData_->perGroupExpertCountAlign;
    renorm_ = tilingData_->renorm;
    normType_ = tilingData_->normType;
    groupSelectMode_ = tilingData_->groupSelectMode;
    
    expertCountAlign_ = Align(perGroupExpertCountAlign_ * groupCount_, sizeof(float));
    kAlign_ = Align(k_, sizeof(float));

    isAlign_ = perGroupExpertCount_ == perGroupExpertCountAlign_;

    // init input gm buf
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);

    // init output gm buf
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ float *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    // init que
    pipe_->InitBuffer(xInQueue_, 1, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(yOutQueue_, 1, kAlign_ * sizeof(float));
    pipe_->InitBuffer(expertIdxOutQueue_, 1, kAlign_ * sizeof(int32_t));
    pipe_->InitBuffer(outOutQueue_, 1, expertCountAlign_ * sizeof(float));

    pipe_->InitBuffer(biasBuf_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(expertIdBuf_, expertCountAlign_ * sizeof(int32_t));

    pipe_->InitBuffer(xNormBuf_, expertCountAlign_ * sizeof(float));

    pipe_->InitBuffer(xNormWithBiasBuf_, expertCountAlign_ * sizeof(float));
    pipe_->InitBuffer(sortedInGroupBuf_, expertCountAlign_ * (sizeof(float) + sizeof(uint32_t)));

    pipe_->InitBuffer(sortedGroupIndexBuf_, groupCountAlign_ * sizeof(float) * CONSTANT_TWO);
    pipe_->InitBuffer(topKExpertIdBuf_, kAlign_ * sizeof(int32_t));
    pipe_->InitBuffer(calcTmpBuf_, expertCountAlign_ * sizeof(float) * 10);
}

template <typename T>
__aicore__ inline void MoeGatingTopKGenerlized<T>::Process()
{
    CopyInBiasAndInitExpertId();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        if (tilingData_->outFlag) {
            CopuOutXNorm(row);
        }
        SortInGroup();
        SelectTopKGroupIndex();
        SelectTopKExpertIdx();
        SelectTopKExpertScore();
        CumputeActualTopKExpertId();
        CopyOut(row);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_GENERALIZED_H
