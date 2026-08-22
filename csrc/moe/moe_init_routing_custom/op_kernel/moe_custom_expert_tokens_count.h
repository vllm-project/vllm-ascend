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
 * \file moe_custom_expert_tokens_count.h
 * \brief
 */
#ifndef MOE_CUSTOM_EXPERT_TOKENS_COUNT_H
#define MOE_CUSTOM_EXPERT_TOKENS_COUNT_H

#include "moe_custom_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingCustom {
using namespace AscendC;

constexpr int64_t EXPERT_ID_VALUE_NUM = 2;
constexpr int64_t CUMSUM_MODE = 0;
constexpr int64_t COUNT_MODE = 1;
constexpr int64_t KEY_VALUE_MODE = 2;
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2;
constexpr int64_t GATHER_SORT_CORE_NUM = 16;
constexpr int64_t DROP_LESS = 0;
constexpr int64_t DROP_PAD = 1;

template <const int HISTOGRAMTYPE>
class ExpertTokensCount {
public:
    __aicore__ inline ExpertTokensCount(){};
    template <bool CALC_ACTUAL_EXPERT_NUM>
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t loop, int64_t curLoopElements);
    __aicore__ inline void Compute(int64_t curLoopElements);
    __aicore__ inline void CopyOut();
    __aicore__ inline void CopyOutExpertTotalCount();

    __aicore__ inline void expertCountCopyIn();
    __aicore__ inline void expertCountCompute();
    __aicore__ inline void expertCountCopyOut();

private:
    GlobalTensor<int32_t> sortedexpertIdxGm_;
    GlobalTensor<int32_t> expertCountTempGm_;
    GlobalTensor<int64_t> expertTokensCountGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<int32_t> expertIdxValueGm_;
    TPipe *pipe_;

    TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
    TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
    TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
    TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

    const MoeCustomExpertTokensCountTilingData *expertTokensCountTilingData_;
    int64_t coreNum_;
    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t perCoreElements_;
    int64_t curCoreElements_ = 0;
    int64_t expertStart_ = 0;
    int64_t expertEnd_ = 0;
    int64_t actualExpertNum_ = 0;
    int64_t coreLoopsNum_ = 0;
    int64_t perCorePerLoopElements_ = 0;
    int64_t perCoreLastLoopElements_ = 0;
    int64_t actualExpertTotalNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t expertCountElements_ = 0;
    bool expertTokensNumFlag_ = false;
    int64_t dropPadMode_ = 0;
    int32_t finalExpertId = -1;
    int32_t expertTokenValue = 0;
    int64_t ep_ = 0;
    int64_t rowIdxType_ = 0;
};

template <const int HISTOGRAMTYPE>
template <bool CALC_ACTUAL_EXPERT_NUM>
__aicore__ inline void
ExpertTokensCount<HISTOGRAMTYPE>::Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                       const MoeInitRoutingCustomTilingData *tilingData, TPipe *tPipe)
{
    coreNum_ = tilingData->coreNum;
    pipe_ = tPipe;
    expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
    blockIdx_ = GetBlockIdx();
    needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
    perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    expertStart_ = tilingData->expertStart;
    expertEnd_ = tilingData->expertEnd;
    actualExpertNum_ = tilingData->actualExpertNum;
    expertNum_ = tilingData->expertNum;
    expertTokensNumFlag_ = tilingData->expertTokensNumFlag;
    dropPadMode_ = tilingData->dropPadMode;
    ep_ = tilingData->ep;
    rowIdxType_ = tilingData->rowIdxType;

    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
    } else {
        curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
        coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
        perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
        perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
    }

    if (CALC_ACTUAL_EXPERT_NUM) {
        // key and value
        int64_t kvFactor = 2;
        GlobalTensor<int32_t> sortedNumGm;
        sortedNumGm.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                    Align(tilingData->n * tilingData->k, sizeof(int32_t)) * kvFactor * kvFactor);
        int32_t totalSortedNum = 0;
        for (int32_t i = 0; i < 16; i++) {
            totalSortedNum += sortedNumGm.GetValue(i);
        }
        perCoreElements_ = Ceil(totalSortedNum, GetBlockNum());
        needCoreNum_ = Ceil(totalSortedNum, perCoreElements_);
        int64_t lastCoreElements = totalSortedNum - (needCoreNum_ - 1) * perCoreElements_;
        if (blockIdx_ == needCoreNum_ - 1) {
            curCoreElements_ = lastCoreElements;
        } else {
            curCoreElements_ = perCoreElements_;
        }
        coreLoopsNum_ = Ceil(curCoreElements_, expertTokensCountTilingData_->perCorePerLoopElements);
        perCorePerLoopElements_ = Ceil(curCoreElements_, coreLoopsNum_);
        perCoreLastLoopElements_ = curCoreElements_ - (coreLoopsNum_ - 1) * perCorePerLoopElements_;
    }

    if constexpr (HISTOGRAMTYPE == KEY_VALUE_MODE) {
        expertCountElements_ = ((actualExpertNum_ + 1) < expertNum_) ? (actualExpertNum_ + 1) * KEY_VALUE_MODE_DIM_NUM :
                                                                       expertNum_ * KEY_VALUE_MODE_DIM_NUM;
    } else {
        expertCountElements_ = actualExpertNum_;
    }
    sortedexpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
    expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t *)expertTokensCount, expertCountElements_);
    expertCountTempGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, actualExpertNum_);
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                            Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        actualExpertNum_);
    expertIdxValueGm_.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
            Align((actualExpertNum_), sizeof(int32_t)) + Align((actualExpertNum_), sizeof(int32_t)),
        coreNum_ * 2);
    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * perCoreElements_,
                                      curCoreElements_);

    if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
        InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);

    pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
    pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));

    pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(expertCountElements_, sizeof(int64_t)));
    pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));

    if (blockIdx_ == 0) {
        InitGlobalMemory(expertTotalCountGm_, 1, 0);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    SyncAll();
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        for (int64_t i = 0; i < coreLoopsNum_; i++) {
            int64_t perLoopElements = (i == (coreLoopsNum_ - 1)) ? perCoreLastLoopElements_ : perCorePerLoopElements_;
            CopyIn(i, perLoopElements);
            Compute(perLoopElements);
            CopyOut();
        }
        if (ep_ == 1) {
            CopyOutExpertTotalCount();
        }
    }
    if (ep_ == 1 || expertTokensNumFlag_ || dropPadMode_ == 1) {
        SyncAll();
    }
    /* copy expert tokens count result from worksapce to output GM. */
    if (blockIdx_ == 0 && expertTokensNumFlag_) {
        expertCountCopyIn();
        expertCountCompute();
        expertCountCopyOut();
    }
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::CopyIn(int64_t loop, int64_t curLoopElements)
{
    LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopElements * sizeof(int32_t)),
                                     0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    int64_t sortedexpertIdxOffset = loop * perCorePerLoopElements_;
    DataCopyPad(sortedExpertIdxInLocal, sortedexpertIdxGm_[sortedexpertIdxOffset], dataCopyParams, dataCopyPadParams);
    sortedExpertIdxInQueue_.EnQue(sortedExpertIdxInLocal);
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::Compute(int64_t curLoopElements)
{
    LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue_.DeQue<int32_t>();
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
    Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
              static_cast<int32_t>(actualExpertNum_));
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    int64_t i = 0;
    int32_t lastExpertId = sortedExpertIdxInLocal.GetValue(0);
    int32_t lastIndex = 0;
    int64_t loopTokenCount = 0;
    int32_t lastlastExpertId = lastExpertId;
    for (i = 1; i < curLoopElements; i++) {
        if ((lastExpertId >= expertEnd_) || (lastExpertId < expertStart_)) {
            break;
        }
        int32_t curExpertId = sortedExpertIdxInLocal.GetValue(i);
        if (curExpertId != lastExpertId || curExpertId >= expertEnd_) {
            if constexpr (HISTOGRAMTYPE == COUNT_MODE || HISTOGRAMTYPE == KEY_VALUE_MODE) {
                expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
                loopTokenCount += i - lastIndex;
            } else {
                for (int64_t j = lastlastExpertId; j < lastExpertId; j++) {
                    expertCountOutLocal.SetValue(j - expertStart_, loopTokenCount);
                }
                loopTokenCount += i - lastIndex;
                expertCountOutLocal.SetValue(lastExpertId - expertStart_, loopTokenCount);
            }
            lastIndex = i;
            lastlastExpertId = lastExpertId;
            lastExpertId = curExpertId;
        }
    }
    if ((i == curLoopElements) && ((lastExpertId >= expertStart_) && (lastExpertId < expertEnd_))) {
        if constexpr (HISTOGRAMTYPE == COUNT_MODE || HISTOGRAMTYPE == KEY_VALUE_MODE) {
            expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
            loopTokenCount += i - lastIndex;
        } else {
            for (int64_t j = lastlastExpertId; j < lastExpertId; j++) {
                expertCountOutLocal.SetValue(j - expertStart_, loopTokenCount);
            }
            loopTokenCount += i - lastIndex;
            expertCountOutLocal.SetValue(lastExpertId - expertStart_, loopTokenCount);
            for (int64_t j = lastExpertId; j < expertEnd_; j++) {
                expertCountOutLocal.SetValue(j - expertStart_, loopTokenCount);
            }
        }
    } else {
        if constexpr (HISTOGRAMTYPE == EXERPT_TOKENS_CUMSUM) {
            for (int64_t j = lastlastExpertId; j < expertEnd_; j++) {
                expertCountOutLocal.SetValue(j - expertStart_, loopTokenCount);
            }
        }
    }
    actualExpertTotalNum_ += loopTokenCount;
    finalExpertId = lastExpertId;
    expertTokenValue = (i - lastIndex);

    expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
    sortedExpertIdxInQueue_.FreeTensor(sortedExpertIdxInLocal);
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::CopyOutExpertTotalCount()
{
    LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
    DataCopyExtParams copyTotalCountParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyTotalCountParams);
    SetAtomicNone();
    expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::CopyOut()
{
    LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)),
                                 0, 0, 0};
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    SetAtomicAdd<int32_t>();
    DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
    SetAtomicNone();

    if (dropPadMode_ == DROP_PAD) {
        expertCountOutLocal.SetValue(0, finalExpertId);
        expertCountOutLocal.SetValue(1, expertTokenValue);
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(EXPERT_ID_VALUE_NUM * sizeof(int32_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expertIdxValueGm_[blockIdx_ * EXPERT_ID_VALUE_NUM], expertCountOutLocal, copyParams);
    }
    expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::expertCountCopyIn()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
    expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::expertCountCompute()
{
    LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
    if constexpr (HISTOGRAMTYPE == KEY_VALUE_MODE) {
        int64_t expertOffset = 0;
        Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
                  static_cast<int32_t>(expertCountElements_ * KEY_VALUE_MODE));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (int64_t i = 0; i < actualExpertNum_; i++) {
            int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
            if (expertCount != 0) {
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM, i + expertStart_);
                expertCountOutLocal.SetValue(expertOffset * KEY_VALUE_MODE_DIM_NUM + 1, expertCount);
                expertOffset++;
            }
        }
    } else {
        Cast(expertCountOutLocal, expertCountTempInLocal, RoundMode::CAST_NONE, actualExpertNum_);
    }

    expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
    expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}

template <const int HISTOGRAMTYPE>
__aicore__ inline void ExpertTokensCount<HISTOGRAMTYPE>::expertCountCopyOut()
{
    LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(expertCountElements_ * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
    copyParams.blockLen = sizeof(int32_t);
    expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
}

} // namespace MoeInitRoutingCustom
#endif // MOE_CUSTOM_EXPERT_TOKENS_COUNT_H