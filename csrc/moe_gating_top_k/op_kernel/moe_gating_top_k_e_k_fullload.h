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
 * \file moe_gating_top_k_e_k_fullload.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_E_K_FULLLOAD_H
#define MOE_GATING_TOP_K_E_K_FULLLOAD_H
#include "kernel_operator.h"
#include "common.h"
namespace MoeGatingTopK {
using namespace AscendC;

template <typename T>
class MoeGatingTopKEKFullload {
public:
    __aicore__ inline MoeGatingTopKEKFullload(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                const MoeGatingTopKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBias();
    __aicore__ inline void CopyInX(int64_t progress);
    __aicore__ inline void ComputeX();
    __aicore__ inline void SortInGroup();
    __aicore__ inline void SelectTopKGroupIndex();
    __aicore__ inline void SelectTopKExpertIdx();
    __aicore__ inline void SelectTopKExpertScore();
    __aicore__ inline void CopyOut(int64_t progress);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TBuf<TPosition::VECCALC> biasInQueue_;
    TQue<QuePosition::VECOUT, 1> yOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
    TQue<QuePosition::VECOUT, 1> outOutQueue_;

    TQue<QuePosition::VECOUT, 1> xBiasQueue_;
    TQue<QuePosition::VECOUT, 1> xSigmoidQueue_;
    TQue<QuePosition::VECIN, 1> sigmoidTmpQueue_;
    TQue<QuePosition::VECIN, 1> sortedInGroupQueue_;
    TQue<QuePosition::VECIN, 1> sortedGroupQueue_;
    TBuf<TPosition::VECCALC> calcTmpBuffer_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> biasGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<T> outGm_;

    int64_t blockIdx_;
    int64_t perCoreRowCount_;
    int64_t curCoreRowCount_;
    int64_t expertCount_;
    bool addBias_;
    int64_t k_;
    int64_t kGroup_;
    int64_t groupCount_;
    int64_t groupSelectMode_;
    int64_t renorm_;
    int64_t normType_;
    int64_t outFlag_;
    float routedScalingFactor_;
    float eps_;

    int64_t expertCountAlign_;
    int64_t kAlign_;
    int64_t perGroupExpertCount_;

    const MoeGatingTopKTilingData *tilingData_;
};

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyInBias()
{
    LocalTensor<float> biasTensor = biasInQueue_.Get<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
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
        Cast(biasTensor, biasTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE, expertCount_);
    }
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyInX(int64_t row)
{
    LocalTensor<float> xInLocalTensor = xInQueue_.AllocTensor<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(expertCount_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
    } else {
        DataCopyPad(xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), xGm_[row * expertCount_], dataCopyParams,
                    dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(xInLocalTensor, xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), RoundMode::CAST_NONE,
             expertCount_);
    }

    xInQueue_.EnQue(xInLocalTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::ComputeX()
{
    LocalTensor<float> xSigmoidTensor = xSigmoidQueue_.AllocTensor<float>();
    LocalTensor<float> xInLocalTensor = xInQueue_.DeQue<float>();
    LocalTensor<float> xBiasTensor = xBiasQueue_.AllocTensor<float>();
    LocalTensor<float> biasTensor = biasInQueue_.Get<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = sigmoidTmpQueue_.AllocTensor<uint8_t>(); // 临时空间可以复用
    Sigmoid(xSigmoidTensor, xInLocalTensor, sharedTmpBuffer, expertCount_);
    PipeBarrier<PIPE_V>();
    if (addBias_) {
        Add(xBiasTensor, xSigmoidTensor, biasTensor, expertCount_);
    } else {
        Adds(xBiasTensor, xSigmoidTensor, static_cast<float>(0), expertCount_);
    }

    xSigmoidQueue_.EnQue<float>(xSigmoidTensor);
    xBiasQueue_.EnQue<float>(xBiasTensor);
    xInQueue_.FreeTensor(xInLocalTensor);
    sigmoidTmpQueue_.FreeTensor(sharedTmpBuffer);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SortInGroup()
{
    LocalTensor<float> xBiasTensor = xBiasQueue_.DeQue<float>();
    LocalTensor<float> sortedInGroupTensor = sortedInGroupQueue_.AllocTensor<float>(); // 组内排序的结果, 后续归并需要
    LocalTensor<uint32_t> indexTensor = calcTmpBuffer_.Get<uint32_t>();                // 用于存储排序时的索引
    ArithProgression(indexTensor.ReinterpretCast<int32_t>(), 0, 1, expertCount_); // 生成组索引0 1 2 ......
    PipeBarrier<PIPE_V>();
    Sort32(sortedInGroupTensor, xBiasTensor, indexTensor, expertCount_ / ONE_REPEAT_SORT_NUM); // 组内排序
    sortedInGroupQueue_.EnQue<float>(sortedInGroupTensor);
    xBiasQueue_.FreeTensor(xBiasTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKGroupIndex()
{
    LocalTensor<float> sortedInGroupTensor = sortedInGroupQueue_.DeQue<float>();
    LocalTensor<uint32_t> indexTensor = calcTmpBuffer_.Get<uint32_t>();
    LocalTensor<float> top2ValueInGroupTensor = sigmoidTmpQueue_.AllocTensor<float>(); // 这个临时空间可以复用
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    indexTensor.SetValue(0, static_cast<uint32_t>(5)); // b0101
    indexTensor.SetValue(1, static_cast<uint32_t>(0));
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 8;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8;
    gatherMaskParams.src1RepeatStride = 0;
    GatherMask(top2ValueInGroupTensor, sortedInGroupTensor, indexTensor, true, static_cast<uint32_t>(64),
               gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> groupTop2SumTensor = top2ValueInGroupTensor;
    PairReduceSum(groupTop2SumTensor, top2ValueInGroupTensor, 1, groupCount_ * 2, 1, 1,
                  1); // 计算每个组内最大的两个数之和
    PipeBarrier<PIPE_V>();

    LocalTensor<uint32_t> groupIndexTensor = indexTensor;
    ArithProgression(groupIndexTensor.ReinterpretCast<int32_t>(), 0, 1, groupCount_); // 生成组索引
    PipeBarrier<PIPE_V>();
    // 用最小值补到32个数
    int64_t duplicateNum = ONE_REPEAT_SORT_NUM - groupCount_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << groupCount_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(groupTop2SumTensor, MIN_FP32, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    // 排序，将kgroup选出来
    LocalTensor<float> sortedGroupTensor = sortedGroupQueue_.AllocTensor<float>();
    Sort32(sortedGroupTensor, groupTop2SumTensor, groupIndexTensor, 1);

    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> sortedGroupIndexTensor = indexTensor.ReinterpretCast<int32_t>();
    // 提取组序号
    uint8_t src1Pattern = 2; // 内置固定模式
    GatherMask(sortedGroupIndexTensor, sortedGroupTensor.template ReinterpretCast<int32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0), {1, 1, 0, 0}, rsvdCnt);

    // 需要将组排序(这里是降序，所以下mrgsor的时候反着取，3、2、1、0)
    Cast(sortedGroupTensor, sortedGroupIndexTensor, RoundMode::CAST_ROUND, kGroup_);
    PipeBarrier<PIPE_V>();
    duplicateNum = ONE_REPEAT_SORT_NUM - kGroup_;
    if (duplicateNum > 0) {
        uint64_t mask0 = UINT64_MAX << kGroup_;
        uint64_t mask[2] = {mask0, 0};
        Duplicate(sortedGroupTensor, MIN_FP32, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    Sort32(top2ValueInGroupTensor, sortedGroupTensor, sortedGroupIndexTensor.template ReinterpretCast<uint32_t>(), 1);
    PipeBarrier<PIPE_V>();
    src1Pattern = 1;
    GatherMask(sortedGroupTensor, top2ValueInGroupTensor, src1Pattern, false, static_cast<uint32_t>(0), {1, 1, 0, 0},
               rsvdCnt);
    PipeBarrier<PIPE_V>();
    Cast(sortedGroupIndexTensor, sortedGroupTensor, RoundMode::CAST_ROUND, kGroup_);

    sortedGroupQueue_.FreeTensor(sortedGroupTensor);
    sortedInGroupQueue_.EnQue<float>(sortedInGroupTensor);
    sigmoidTmpQueue_.FreeTensor(top2ValueInGroupTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKExpertIdx()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> topKGroupIndexTensor = calcTmpBuffer_.Get<int32_t>();
    LocalTensor<float> sortedInGroupTensor = sortedInGroupQueue_.DeQue<float>();
    LocalTensor<float> sortedExpertTensor = xInQueue_.AllocTensor<float>();
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = k_;
    params.elementLengths[1] = k_;
    params.elementLengths[2] = k_;
    params.elementLengths[3] = k_;
    params.ifExhaustedSuspension = true;
    params.validBit = 0b1111;
    params.repeatTimes = 1;
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    int64_t listOffset1 = topKGroupIndexTensor.GetValue(3) * perGroupExpertCount_ * 2;
    int64_t listOffset2 = topKGroupIndexTensor.GetValue(2) * perGroupExpertCount_ * 2;
    int64_t listOffset3 = topKGroupIndexTensor.GetValue(1) * perGroupExpertCount_ * 2;
    int64_t listOffset4 = topKGroupIndexTensor.GetValue(0) * perGroupExpertCount_ * 2;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = sortedInGroupTensor[listOffset1];
    srcList.src2 = sortedInGroupTensor[listOffset2];
    srcList.src3 = sortedInGroupTensor[listOffset3];
    srcList.src4 = sortedInGroupTensor[listOffset4];
    MrgSort<float>(sortedExpertTensor, srcList, params);
    PipeBarrier<PIPE_V>();
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 内置固定模式
    GatherMask(expertIdxTensor, sortedExpertTensor.template ReinterpretCast<int32_t>(), src1Pattern, false,
               static_cast<uint32_t>(0), {1, 1, 0, 0}, rsvdCnt);
    xInQueue_.FreeTensor(sortedExpertTensor);
    expertIdxOutQueue_.EnQue(expertIdxTensor);
    sortedInGroupQueue_.FreeTensor(sortedInGroupTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::SelectTopKExpertScore()
{
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertByteIdxTensor = calcTmpBuffer_.Get<int32_t>();
    LocalTensor<float> xSigmoidTensor = xSigmoidQueue_.DeQue<float>();
    LocalTensor<T> yTensor = yOutQueue_.AllocTensor<T>();
    LocalTensor<float> yOutTensor;
    if constexpr (!IsSameType<T, float>::value) {
        yOutTensor = yTensor.template ReinterpretCast<float>()[kAlign_];
    } else {
        yOutTensor = yTensor;
    }
    Muls(expertByteIdxTensor, expertIdxTensor, static_cast<int32_t>(sizeof(float)), k_);
    PipeBarrier<PIPE_V>();
    Gather(yOutTensor, xSigmoidTensor, expertByteIdxTensor.template ReinterpretCast<uint32_t>(),
           static_cast<uint32_t>(0), k_);

    LocalTensor<float> calTensor = calcTmpBuffer_.Get<float>();
    PipeBarrier<PIPE_V>();
    ReduceSum(calTensor, yOutTensor, xSigmoidTensor, k_);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float sumValue = calTensor.GetValue(0) + eps_;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Duplicate(calTensor, sumValue, k_);
    PipeBarrier<PIPE_V>();
    Div(yOutTensor, yOutTensor, calTensor, k_);
    PipeBarrier<PIPE_V>();
    Muls(yOutTensor, yOutTensor, routedScalingFactor_, k_);

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(yTensor, yOutTensor, RoundMode::CAST_RINT, k_);
    }

    xSigmoidQueue_.EnQue<float>(xSigmoidTensor);
    expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    yOutQueue_.EnQue(yTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::CopyOut(int64_t row)
{
    LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
    LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
    LocalTensor<float> xSigmoidTensor = xSigmoidQueue_.DeQue<float>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);
    dataCopyParams.blockLen = k_ * sizeof(int32_t);
    DataCopyPad(expertIdxGm_[row * k_], expertIdxTensor, dataCopyParams);
    xSigmoidQueue_.FreeTensor(xSigmoidTensor);
    expertIdxOutQueue_.FreeTensor(expertIdxTensor);
    yOutQueue_.FreeTensor(yOutTensor);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR expertIdx,
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
    perGroupExpertCount_ = tilingData_->perGroupExpertCount;
    routedScalingFactor_ = tilingData_->routedScalingFactor;
    eps_ = tilingData_->eps;

    expertCountAlign_ = Align(expertCount_, sizeof(float));
    kAlign_ = Align(expertCount_, sizeof(float));

    // init input gm buf
    xGm_.SetGlobalBuffer((__gm__ T *)x + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);
    biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);

    // init output gm buf
    yGm_.SetGlobalBuffer((__gm__ T *)y + perCoreRowCount_ * k_ * blockIdx_, k_);
    expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + perCoreRowCount_ * k_ * blockIdx_, k_);
    outGm_.SetGlobalBuffer((__gm__ T *)out + perCoreRowCount_ * expertCount_ * blockIdx_, expertCount_);

    // init que
    pipe_->InitBuffer(xInQueue_, 2, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(biasInQueue_, expertCountAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));

    pipe_->InitBuffer(xSigmoidQueue_, 1, AlignBytes(expertCount_, sizeof(float)));
    pipe_->InitBuffer(xBiasQueue_, 2, AlignBytes(expertCount_, sizeof(float)));

    pipe_->InitBuffer(yOutQueue_, 2, kAlign_ * sizeof(float) * (sizeof(float) / sizeof(T)));
    pipe_->InitBuffer(expertIdxOutQueue_, 2, AlignBytes(k_, sizeof(int32_t)));
    pipe_->InitBuffer(outOutQueue_, 2, AlignBytes(expertCount_, sizeof(float)));

    pipe_->InitBuffer(sigmoidTmpQueue_, 2, AlignBytes(expertCount_, sizeof(float)));
    pipe_->InitBuffer(sortedInGroupQueue_, 2, AlignBytes(expertCount_, sizeof(float)) * 2);
    pipe_->InitBuffer(sortedGroupQueue_, 2,
                      (groupCount_ + ONE_REPEAT_SORT_NUM - 1) / ONE_REPEAT_SORT_NUM * ONE_REPEAT_SORT_NUM *
                          sizeof(float) * 2);

    pipe_->InitBuffer(calcTmpBuffer_, tilingData_->calTmpBufUbSize);
}

template <typename T>
__aicore__ inline void MoeGatingTopKEKFullload<T>::Process()
{
    CopyInBias();
    for (int64_t row = 0; row < curCoreRowCount_; row++) {
        CopyInX(row);
        ComputeX();
        SortInGroup();
        SelectTopKGroupIndex();
        SelectTopKExpertIdx();
        SelectTopKExpertScore();
        CopyOut(row);
    }
}
} // namespace MoeGatingTopK
#endif // MOE_GATING_TOP_K_E_K_FULLLOAD_H