/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_v2_tools.h
 * \brief 放算子都需要、与算子联系紧密、但是又不方便单独独立出来的公共工具
 */

#ifndef COMPRESSOR_V2_TOOLS_H
#define COMPRESSOR_V2_TOOLS_H

#include "compressor_v2_comm_arch22.h"

using namespace AscendC;

namespace CompressorV2 {

struct ToolsParams {
    uint32_t seqSize = 0U;
    uint32_t cmpRatio = 0U;
};

template <typename COMP>
class CompressorV2Tools {
public:
    __aicore__ inline CompressorV2Tools() {}

    __aicore__ inline void Init(__gm__ uint8_t *startPos, __gm__ uint8_t *seqUsed, __gm__ uint8_t *cuSeqlens);

    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint64_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint64_t GetTIdxByBatch(uint32_t bIdx);

public:
    ToolsParams toolParams_{};
    bool isExistSeqUsed_ = false;

private:
    bool isExistStartPos_ = false;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
};

template <typename COMP>
__aicore__ inline void CompressorV2Tools<COMP>::Init(__gm__ uint8_t *startPos, __gm__ uint8_t *seqUsed,
                                                     __gm__ uint8_t *cuSeqlens)
{
    isExistStartPos_ = (startPos != nullptr);
    if (isExistStartPos_) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }

    isExistSeqUsed_ = (seqUsed != nullptr);
    if (isExistSeqUsed_) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }

    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorV2Tools<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed_) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return toolParams_.seqSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint64_t CompressorV2Tools<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return (uint64_t)startPosGm_.GetValue(bIdx);
    } else {
        return 0;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorV2Tools<COMP>::GetSeqLength(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return toolParams_.seqSize;
    }
}

template <typename COMP>
__aicore__ inline uint64_t CompressorV2Tools<COMP>::GetTIdxByBatch(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return (uint64_t)(cuSeqlensGm_.GetValue(bIdx));
    } else {
        return (uint64_t)toolParams_.seqSize * bIdx;
    }
}

// iterator
struct SliceInfo {
    __aicore__ inline SliceInfo(){};
    __aicore__ inline SliceInfo(uint32_t bIdx, uint32_t sIdx)
        : bIdx(bIdx),
          sIdx(sIdx){};

    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint64_t bStartPos = 0U;

    uint32_t headHolderSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailHolderSeqCnt = 0U;

    uint32_t dealSeqCnt = 0;
    uint32_t dealTcSize = 0U;
    uint32_t compressTcSize = 0U;
};

template <typename COMP>
class CompressorV2SliceIterator {
public:
    __aicore__ inline CompressorV2SliceIterator(CompressorV2Tools<COMP> &tools)
        : tools_(tools)
    {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetMaxDealSeqCnt(uint32_t maxDealSeqCnt);
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline SliceInfo &GetSlice();
    __aicore__ inline SliceInfo &GetSliceByCmp();

    bool isFirst_ = true;
    SliceInfo sliceInfo_{};

private:
    CompressorV2Tools<COMP> &tools_;

    // iterator
    uint32_t maxDealSeqCnt_ = 0;
    uint32_t batch_size_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorV2SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    isFirst_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorV2SliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
}

template <typename COMP>
__aicore__ inline void CompressorV2SliceIterator<COMP>::SetMaxDealSeqCnt(uint32_t maxDealSeqCnt)
{
    this->maxDealSeqCnt_ = maxDealSeqCnt;
}

template <typename COMP>
__aicore__ inline bool CompressorV2SliceIterator<COMP>::IsEnd()
{
    return (sliceInfo_.bIdx >= batch_size_) || (maxDealSeqCnt_ == 0);
}

template <typename COMP>
__aicore__ inline void CompressorV2SliceIterator<COMP>::IteratorSlice()
{
    bool isUpdateBatchInfo = false;
    if (!isFirst_) {
        // 更新剩余未处理的行数
        maxDealSeqCnt_ -= sliceInfo_.dealSeqCnt;
        // 更新sIdx和bIdx、以及与bIdx相关的bStartPos和bSeqUsed
        sliceInfo_.sIdx += sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == sliceInfo_.bSeqUsed) {
            sliceInfo_.sIdx = 0;
            sliceInfo_.bIdx++;
            isUpdateBatchInfo = true;
        }
    } else {
        isUpdateBatchInfo = true;
        isFirst_ = false;
    }

    // 更新与bIdx相关的bStartPos和bSeqUsed
    if (isUpdateBatchInfo) {
        // SkipInvalidBatch
        while (sliceInfo_.bIdx < batch_size_) {
            sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
            if (sliceInfo_.bSeqUsed > 0) {
                break;
            }
            sliceInfo_.bIdx++;
        }
        if (sliceInfo_.bIdx < batch_size_) {
            sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline SliceInfo &CompressorV2SliceIterator<COMP>::GetSliceByCmp()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (isFirst_) {
        sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
        isFirst_ = false;
    }
    // 计算头部占位行数、有效数据行数、尾部占位行数
    sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;

    sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
    if (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt > maxDealSeqCnt_) {
        sliceInfo_.validSeqCnt = maxDealSeqCnt_ - sliceInfo_.headHolderSeqCnt;
    }
    sliceInfo_.tailHolderSeqCnt =
        cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    // 头和尾处理，否则需要处理的seq等于cmpRatio
    if (sliceInfo_.validSeqCnt < cmpRatio) {
        sliceInfo_.dealSeqCnt = sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == 0) {
            sliceInfo_.dealSeqCnt = cmpRatio - sliceInfo_.headHolderSeqCnt;
        }
    } else {
        sliceInfo_.dealSeqCnt = cmpRatio;
    }
    sliceInfo_.validSeqCnt = sliceInfo_.dealSeqCnt;

    // 计算本次可以处理的Tc个数
    sliceInfo_.dealTcSize = (sliceInfo_.dealSeqCnt + cmpRatio - 1) / cmpRatio;

    // 因为是一个batch的数据, 只有最后一个压缩块才可能不需要压缩, 此时sliceInfo_.tailHolderSeqCnt > 0
    sliceInfo_.compressTcSize = sliceInfo_.dealTcSize;
    if (sliceInfo_.tailHolderSeqCnt > 0) {
        sliceInfo_.compressTcSize = sliceInfo_.dealTcSize - 1; // 最后一个压缩块不满时，其不需要压缩
    }

    return sliceInfo_;
}

template <typename COMP>
__aicore__ inline SliceInfo &CompressorV2SliceIterator<COMP>::GetSlice()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (isFirst_) {
        sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
        isFirst_ = false;
    }
    // 计算头部占位行数、有效数据行数、尾部占位行数
    sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;
    sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
    if (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt > maxDealSeqCnt_) {
        sliceInfo_.validSeqCnt = maxDealSeqCnt_ - sliceInfo_.headHolderSeqCnt;
    }
    sliceInfo_.tailHolderSeqCnt =
        cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    sliceInfo_.dealSeqCnt = sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt + sliceInfo_.tailHolderSeqCnt;
    // 计算本次可以处理的Tc个数
    sliceInfo_.dealTcSize = sliceInfo_.dealSeqCnt / cmpRatio;

    // 因为是一个batch的数据, 只有最后一个压缩块才可能不需要压缩, 此时sliceInfo_.tailHolderSeqCnt > 0
    sliceInfo_.compressTcSize = sliceInfo_.dealTcSize;
    if (sliceInfo_.tailHolderSeqCnt > 0) {
        sliceInfo_.compressTcSize = sliceInfo_.dealTcSize - 1; // 最后一个压缩块不满时，其不需要压缩
    }

    return sliceInfo_;
}

struct SplitCoreSliceInfo : public SliceInfo {
    __aicore__ inline SplitCoreSliceInfo(){};
    __aicore__ inline SplitCoreSliceInfo(uint32_t bIdx, uint32_t sIdx)
        : SliceInfo(bIdx, sIdx){};

    uint32_t preFirstSeqCnt = 0U; // 左边每次迭代基本块的第一个seqCnt大小
};

template <typename COMP>
class CompressorV2SplitCoreSliceIterator {
public:
    __aicore__ inline CompressorV2SplitCoreSliceIterator(CompressorV2Tools<COMP> &tools)
        : tools_(tools)
    {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetMaxDealSeqCnt(uint32_t maxDealSeqCnt);
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline SplitCoreSliceInfo &GetSlice();
    __aicore__ inline SplitCoreSliceInfo &GetSliceByCmp();
    __aicore__ inline uint32_t GetBIdx();
    __aicore__ inline SplitCoreSliceInfo &GetLeftNextCmpSeqCnt();
    __aicore__ inline SplitCoreSliceInfo &GetRightNextCmpSeqCnt();

    bool isFirst_ = true;
    bool isLeftFirstBath = false;
    bool isMaxDealSeqCntFirst = false;

    SplitCoreSliceInfo sliceInfo_{};

private:
    CompressorV2Tools<COMP> &tools_;

    // iterator
    uint32_t maxDealSeqCnt_ = 0;
    uint32_t batch_size_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorV2SplitCoreSliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    isFirst_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorV2SplitCoreSliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
    isMaxDealSeqCntFirst = true;
}

template <typename COMP>
__aicore__ inline void CompressorV2SplitCoreSliceIterator<COMP>::SetMaxDealSeqCnt(uint32_t maxDealSeqCnt)
{
    this->maxDealSeqCnt_ = maxDealSeqCnt;
}

template <typename COMP>
__aicore__ inline bool CompressorV2SplitCoreSliceIterator<COMP>::IsEnd()
{
    return (sliceInfo_.bIdx >= batch_size_) || (maxDealSeqCnt_ == 0);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorV2SplitCoreSliceIterator<COMP>::GetBIdx()
{
    return sliceInfo_.bIdx;
}

template <typename COMP>
__aicore__ inline void CompressorV2SplitCoreSliceIterator<COMP>::IteratorSlice()
{
    bool isUpdateBatchInfo = false;
    if (isMaxDealSeqCntFirst) {
        isMaxDealSeqCntFirst = false;
    }
    if (!isFirst_) {
        // 更新剩余未处理的行数
        maxDealSeqCnt_ -= sliceInfo_.dealSeqCnt;
        // 更新sIdx和bIdx、以及与bIdx相关的bStartPos和bSeqUsed
        sliceInfo_.sIdx += sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == sliceInfo_.bSeqUsed) {
            sliceInfo_.sIdx = 0;
            // 左边最后一块跳到b=0 s=0处理
            if (isLeftFirstBath) {
                isLeftFirstBath = false;
            } else {
                sliceInfo_.bIdx++;
            }
            isUpdateBatchInfo = true;
        }
    } else {
        isUpdateBatchInfo = true;
        isFirst_ = false;
    }

    // 更新与bIdx相关的bStartPos和bSeqUsed
    if (isUpdateBatchInfo) {
        // SkipInvalidBatch
        while (sliceInfo_.bIdx < batch_size_) {
            sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
            if (sliceInfo_.bSeqUsed > 0) {
                break;
            }
            sliceInfo_.bIdx++;
        }
        if (sliceInfo_.bIdx < batch_size_) {
            sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline SplitCoreSliceInfo &CompressorV2SplitCoreSliceIterator<COMP>::GetLeftNextCmpSeqCnt()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (isFirst_) {
        // 左边 T轴首次减去T轴最后一块
        sliceInfo_.bSeqUsed = tools_.GetSeqUsed(batch_size_ - 1);
        sliceInfo_.bStartPos = tools_.GetStartPos(batch_size_ - 1);
        // 处理最后一块是中间整块或者尾块的情况
        uint32_t lastSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.bSeqUsed) % cmpRatio == 0 ?
                                  cmpRatio :
                                  (sliceInfo_.bStartPos + sliceInfo_.bSeqUsed) % cmpRatio;
        // 处理最后一块是头块的情况
        if (sliceInfo_.bSeqUsed < cmpRatio) {
            lastSeqCnt = sliceInfo_.bSeqUsed;
        }

        sliceInfo_.sIdx = sliceInfo_.bSeqUsed - lastSeqCnt;
        isLeftFirstBath = true;
        isFirst_ = false;
    }
    // 计算头部占位行数、有效数据行数、尾部占位行数
    sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;

    sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
    if (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt > maxDealSeqCnt_) {
        sliceInfo_.validSeqCnt = maxDealSeqCnt_ - sliceInfo_.headHolderSeqCnt;
    }
    sliceInfo_.tailHolderSeqCnt =
        cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    // 头和尾处理，否则需要处理的seq等于cmpRatio
    if (sliceInfo_.validSeqCnt < cmpRatio) {
        sliceInfo_.dealSeqCnt = sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == 0) {
            sliceInfo_.dealSeqCnt = cmpRatio - sliceInfo_.headHolderSeqCnt;
        }
    } else {
        sliceInfo_.dealSeqCnt = cmpRatio;
    }
    sliceInfo_.validSeqCnt = sliceInfo_.dealSeqCnt;

    // 计算本次可以处理的Tc个数
    sliceInfo_.dealTcSize = (sliceInfo_.dealSeqCnt + cmpRatio - 1) / cmpRatio;

    // 因为是一个batch的数据, 只有最后一个压缩块才可能不需要压缩, 此时sliceInfo_.tailHolderSeqCnt > 0
    sliceInfo_.compressTcSize = sliceInfo_.dealTcSize;
    if (sliceInfo_.tailHolderSeqCnt > 0) {
        sliceInfo_.compressTcSize = sliceInfo_.dealTcSize - 1; // 最后一个压缩块不满时，其不需要压缩
    }

    // 记录左边第一个块
    if (isMaxDealSeqCntFirst) {
        sliceInfo_.preFirstSeqCnt = sliceInfo_.dealSeqCnt;
    }

    return sliceInfo_;
}

template <typename COMP>
__aicore__ inline SplitCoreSliceInfo &CompressorV2SplitCoreSliceIterator<COMP>::GetRightNextCmpSeqCnt()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (isFirst_) {
        sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
        isFirst_ = false;
    }
    // 计算头部占位行数、有效数据行数、尾部占位行数
    sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;

    sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
    if (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt > maxDealSeqCnt_) {
        sliceInfo_.validSeqCnt = maxDealSeqCnt_ - sliceInfo_.headHolderSeqCnt;
    }
    sliceInfo_.tailHolderSeqCnt =
        cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    // 头和尾处理，否则需要处理的seq等于cmpRatio
    if (sliceInfo_.validSeqCnt < cmpRatio) {
        sliceInfo_.dealSeqCnt = sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == 0) {
            sliceInfo_.dealSeqCnt = cmpRatio - sliceInfo_.headHolderSeqCnt;
        }
    } else {
        sliceInfo_.dealSeqCnt = cmpRatio;
    }
    sliceInfo_.validSeqCnt = sliceInfo_.dealSeqCnt;

    // 计算本次可以处理的Tc个数
    sliceInfo_.dealTcSize = (sliceInfo_.dealSeqCnt + cmpRatio - 1) / cmpRatio;

    // 因为是一个batch的数据, 只有最后一个压缩块才可能不需要压缩, 此时sliceInfo_.tailHolderSeqCnt > 0
    sliceInfo_.compressTcSize = sliceInfo_.dealTcSize;
    if (sliceInfo_.tailHolderSeqCnt > 0) {
        sliceInfo_.compressTcSize = sliceInfo_.dealTcSize - 1; // 最后一个压缩块不满时，其不需要压缩
    }

    return sliceInfo_;
}

struct Vec1SliceInfo : public SliceInfo {
    __aicore__ inline Vec1SliceInfo(){};
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx)
        : SliceInfo(bIdx, sIdx){};
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt)
        : SliceInfo(bIdx, sIdx),
          dealedSeqCnt(dealedSeqCnt){};

    uint32_t dealedSeqCnt = 0U;
    uint32_t dealedTcCnt = 0U;
    uint32_t bSeqLength = 0U;
    uint32_t compressoredScCnt = 0U;
    bool isFirst = false;
    bool isLast = false;
};

struct StatisticInfo {
    __aicore__ inline StatisticInfo(){};
    __aicore__ inline StatisticInfo(uint32_t actualTcCnt, uint32_t dealSeqCnt, uint32_t compressorScCnt)
        : actualTcCnt(actualTcCnt),
          dealSeqCnt(dealSeqCnt),
          compressorScCnt(compressorScCnt){};

    uint32_t actualTcCnt = 0U;
    uint32_t dealSeqCnt = 0U;
    uint32_t compressorScCnt = 0U;
};

template <typename COMP>
class CompressorV2Vec1SliceIterator {
public:
    __aicore__ inline CompressorV2Vec1SliceIterator(CompressorV2Tools<COMP> &tools)
        : tools_(tools)
    {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt, uint32_t compressoredScCnt);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetDealedSeqCnt(uint32_t dealedSeqCnt);
    __aicore__ inline void SetDealedTcCnt(uint32_t dealedTcCnt);
    __aicore__ inline void SetCompressorV2edScCnt(uint32_t compressoredScCnt);
    __aicore__ inline void SetNeedDealTcSize(uint32_t needDealTcSize);
    __aicore__ inline void SetNeedDealTcSize(uint32_t needDealTcSize, uint32_t canDealTcSize);
    __aicore__ inline uint32_t GetNeedDealTcSize();
    __aicore__ inline bool IsEnd();
    template <bool IS_STATISTIC = false>
    __aicore__ inline void IteratorSlice();
    __aicore__ inline Vec1SliceInfo &GetSlice();
    template <bool IS_STATISTIC = false>
    __aicore__ inline StatisticInfo &FullIteratorSlice();

private:
    CompressorV2Tools<COMP> &tools_;

    bool isFirst_ = true;
    Vec1SliceInfo sliceInfo_{};
    StatisticInfo statisticInfo_{};
    uint32_t needDealTcSize_ = 0U;
    uint32_t batch_size_ = 0U;
};

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    while (tools_.GetSeqLength(sliceInfo_.bIdx) == 0) {
        sliceInfo_.bIdx++;
        if (sliceInfo_.bIdx == batch_size_) {
            sliceInfo_.bIdx = 0;
        }
    }
    sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
    sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
    isFirst_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt,
                                                                  uint32_t compressoredScCnt)
{
    Reset(bIdx, sIdx);
    SetDealedSeqCnt(dealedSeqCnt);
    SetCompressorV2edScCnt(compressoredScCnt);
}

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
}

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::SetDealedSeqCnt(uint32_t dealedSeqCnt)
{
    this->sliceInfo_.dealedSeqCnt = dealedSeqCnt;
}

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::SetCompressorV2edScCnt(uint32_t compressoredScCnt)
{
    this->sliceInfo_.compressoredScCnt = compressoredScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::SetDealedTcCnt(uint32_t dealedTcCnt)
{
    this->sliceInfo_.dealedTcCnt = dealedTcCnt;
}

template <typename COMP>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::SetNeedDealTcSize(uint32_t needDealTcSize)
{
    this->needDealTcSize_ = needDealTcSize;
}

template <typename COMP>
template <bool IS_STATISTIC>
__aicore__ inline void CompressorV2Vec1SliceIterator<COMP>::IteratorSlice()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if constexpr (IS_STATISTIC) {
        statisticInfo_.actualTcCnt += sliceInfo_.dealTcSize;
        statisticInfo_.compressorScCnt += sliceInfo_.compressTcSize;
    }
    needDealTcSize_ -= sliceInfo_.dealTcSize;
    sliceInfo_.dealedSeqCnt += sliceInfo_.validSeqCnt;
    sliceInfo_.compressoredScCnt += sliceInfo_.compressTcSize;
    sliceInfo_.sIdx += sliceInfo_.validSeqCnt;
    if (sliceInfo_.sIdx >= sliceInfo_.bSeqUsed) {
        do {
            const uint32_t seqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
            if (sliceInfo_.bSeqUsed < seqLength) {
                // ── (A) tailH 行推进（不消耗任务量）──
                // slice 处理到 su 为止，su 到下一个压缩块分界点（全局 cr 对齐的"切分点"）
                // 之间的行是 slice 尾部 padding（tailH）。这些行在任务量上已由 slice 的
                // dealTcSize 消耗，但 workspace 行号（dealedSeqCnt）尚未推进——必须在此
                // 推进，否则核起点落在非切分点，后续 slice 从错位位置读数据。
                //
                // ★特殊1：仅 sIdx > 0（slice 确实处理过）时才推进 tailH。
                //   sIdx == 0 表示 seqused == 0 的无效 batch（无 slice），其行全部属于
                //   空洞（块任务量未消耗）；此时按 tailH 推进会"行超前于任务量"
                //   （b=53: 全局对齐量 62 行，但 x 行仅 1 行）→ workspace 偏移虚增越界。
                //
                // ★特殊2：对齐量 clamp 到空洞内。对齐量（< cmpRatio）可能超过本 batch
                //   的 x 行剩余（seqLength - sIdx，如无效 batch 的 x 行很短），
                //   超出的行属于下一个 batch 或窗口外，不得推进。
                if (sliceInfo_.sIdx > 0) {
                    uint64_t nextAlignSIdx =
                        Align(sliceInfo_.bStartPos + sliceInfo_.sIdx, static_cast<uint64_t>(cmpRatio)) -
                        sliceInfo_.bStartPos;
                    uint32_t align =
                        min(static_cast<uint32_t>(nextAlignSIdx - sliceInfo_.sIdx), seqLength - sliceInfo_.sIdx);
                    sliceInfo_.dealedSeqCnt += align;
                    sliceInfo_.sIdx += align;
                }

                // ── 空洞（tailH 之后）对应的压缩块数 tcGap（需消耗的任务量）──
                // 空洞 = [sIdx, seqLength) 的 x 行，全局位置 [bStartPos+sIdx, bStartPos+seqLength)。
                // 块数 = 空洞覆盖的全局 cr 块数。
                //
                // ★特殊3：sIdx == 0（seqused == 0）时起点块（含头部 padding 的块）
                //   未被 slice 消耗，块数从 floor(bStartPos/cr) 起算（= tcNum 公式，起点块计入）；
                //   sIdx > 0（tailH 已推进到切分点）时从 ceil((bStartPos+sIdx)/cr) 起算
                //   （起点块已由 slice 的 dealTcSize 消耗）。
                const uint32_t gapRows = seqLength - sliceInfo_.sIdx;
                uint32_t tcGap;
                if (sliceInfo_.sIdx == 0) {
                    tcGap = static_cast<uint32_t>(
                        CeilDivT(sliceInfo_.bStartPos + seqLength, static_cast<uint64_t>(cmpRatio)) -
                        sliceInfo_.bStartPos / static_cast<uint64_t>(cmpRatio));
                } else {
                    tcGap = static_cast<uint32_t>(
                        CeilDivT(sliceInfo_.bStartPos + seqLength, static_cast<uint64_t>(cmpRatio)) -
                        CeilDivT(sliceInfo_.bStartPos + sliceInfo_.sIdx, static_cast<uint64_t>(cmpRatio)));
                }

                if (needDealTcSize_ < tcGap) {
                    // ── (B) 部分跳过：任务量不足以跳过整个空洞 ──
                    // 只推进任务量对应的行（needTc 个块 = needTc*cmpRatio 行），clamp 到空洞内。
                    //
                    // ★特殊4：needTc == 0 时不推进任何行——行推进必须与任务量消耗严格
                    //   对应；否则核起点"行已推进、块未消耗"，后续核分到无行的块
                    //   （读窗口外数据 / 输出丢失）。
                    //
                    // ★特殊5：起点可能位于全局压缩块中间，不能直接用
                    //   needTc*cmpRatio 计算行数。需要按全局块边界计算跳过量：
                    //   跳过 needTc 个块后，行位置落在
                    //   (floor(globalStart/cr) + needTc) * cr。
                    //   结果再 clamp 到当前空洞，避免越过 batch 末尾。
                    uint32_t skip = 0;
                    if (needDealTcSize_ > 0) {
                        uint64_t globalStartSeqIdx = sliceInfo_.bStartPos + sliceInfo_.sIdx;
                        uint64_t skipEndSeqIdx =
                            (globalStartSeqIdx / static_cast<uint64_t>(cmpRatio) + needDealTcSize_) *
                            static_cast<uint64_t>(cmpRatio);
                        skip = min(static_cast<uint32_t>(skipEndSeqIdx - globalStartSeqIdx), gapRows);
                    }
                    sliceInfo_.dealedSeqCnt += skip;
                    sliceInfo_.sIdx += skip;
                    needDealTcSize_ = 0;
                    break; // 任务量耗尽：迭代终止
                }
                // ── (C) 完整跳过：推进整个空洞，消耗 tcGap 个 Tc ──
                // 空洞全部行在本 batch 内，推进后 sIdx 到达 seqLength（batch 末尾），
                // 随后换到下一个 batch。
                sliceInfo_.dealedSeqCnt += gapRows;
                sliceInfo_.sIdx += gapRows;
                needDealTcSize_ -= tcGap;
            }
            sliceInfo_.bIdx++;
            if (sliceInfo_.bIdx == batch_size_) {
                // 终止而非回绕（防死循环；正常遍历不会触发）
                sliceInfo_.bIdx = batch_size_ - 1;
                sliceInfo_.sIdx = 0;
                sliceInfo_.bSeqUsed = 0;
                sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
                sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
                needDealTcSize_ = 0;
                break;
            }
            sliceInfo_.sIdx = 0;
            sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
            sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
            sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
        } while (sliceInfo_.bSeqUsed == 0);
    }
    if (isFirst_) {
        isFirst_ = false;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorV2Vec1SliceIterator<COMP>::GetNeedDealTcSize()
{
    return needDealTcSize_;
}

template <typename COMP>
__aicore__ inline bool CompressorV2Vec1SliceIterator<COMP>::IsEnd()
{
    return (needDealTcSize_ == 0);
}

template <typename COMP>
__aicore__ inline Vec1SliceInfo &CompressorV2Vec1SliceIterator<COMP>::GetSlice()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (sliceInfo_.bSeqUsed <= sliceInfo_.sIdx) {
        // sIdx == bSeqUsed 时同样视为无效 slice（含 bSeqUsed==0 的空洞 batch 起点）：
        // 否则 GetSlice 会生成 headHolder>0/valid=0 的伪 slice（dealTcSize=1）
        // 每轮白吃 1 个 tc 配额且不推进 → 迭代器卡死在空洞 batch，其后块全部丢失
        sliceInfo_.headHolderSeqCnt = 0;
        sliceInfo_.validSeqCnt = 0;
        sliceInfo_.tailHolderSeqCnt = 0;
        sliceInfo_.dealTcSize = 0;
        sliceInfo_.compressTcSize = 0;
    } else {
        // 计算头部占位行数、有效数据行数、尾部占位行数
        sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;
        sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
        if (CeilDivT(sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt, cmpRatio) > needDealTcSize_) {
            sliceInfo_.validSeqCnt = needDealTcSize_ * cmpRatio - sliceInfo_.headHolderSeqCnt;
        }
        uint64_t globalTotalSeqCnt = sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt;
        sliceInfo_.tailHolderSeqCnt = Align(globalTotalSeqCnt, (uint64_t)cmpRatio) - globalTotalSeqCnt;

        // 计算本次可以处理的Tc个数
        sliceInfo_.dealTcSize =
            (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt + sliceInfo_.tailHolderSeqCnt) / cmpRatio;

        sliceInfo_.compressTcSize =
            (sliceInfo_.headHolderSeqCnt + min(sliceInfo_.validSeqCnt, sliceInfo_.bSeqUsed - sliceInfo_.sIdx)) /
            cmpRatio;
    }

    sliceInfo_.isFirst = isFirst_;
    sliceInfo_.isLast =
        sliceInfo_.bSeqUsed > sliceInfo_.sIdx &&
        CeilDivT(sliceInfo_.headHolderSeqCnt + sliceInfo_.bSeqUsed - sliceInfo_.sIdx, cmpRatio) >= needDealTcSize_;

    return sliceInfo_;
}

template <typename COMP>
template <bool IS_STATISTIC>
__aicore__ inline StatisticInfo &CompressorV2Vec1SliceIterator<COMP>::FullIteratorSlice()
{
    if constexpr (IS_STATISTIC) {
        statisticInfo_ = {0U, 0U, 0U};
        Vec1SliceInfo tempSliceInfo = GetSlice();
        while (!IsEnd()) {
            GetSlice();
            IteratorSlice<IS_STATISTIC>();
        }
        Vec1SliceInfo sliceInfo = GetSlice();
        statisticInfo_.dealSeqCnt = sliceInfo.dealedSeqCnt - tempSliceInfo.dealedSeqCnt;
    } else {
        while (!IsEnd()) {
            GetSlice();
            IteratorSlice<IS_STATISTIC>();
        }
    }
    return statisticInfo_;
}
} // namespace CompressorV2

#endif
