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
 * \file compressor_tools.h
 * \brief 放算子都需要、与算子联系紧密、但是又不方便单独独立出来的公共工具
 */

#ifndef COMPRESSOR_TOOLS_H
#define COMPRESSOR_TOOLS_H

#include "compressor_comm.h"

using namespace AscendC;

namespace Compressor {

struct ToolsParams {
    uint32_t seqSize = 0U;
    uint32_t cmpRatio = 0U;
};

template <typename COMP>
class CompressorTools {
public:
    __aicore__ inline CompressorTools() {}

    __aicore__ inline void Init(__gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos);

    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetTIdxByBatch(uint32_t bIdx);

public:
    ToolsParams toolParams_ {};
    bool isExistSeqUsed_ = false;

private:
    bool isExistStartPos_ = false;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
};

template <typename COMP>
__aicore__ inline void CompressorTools<COMP>::Init(__gm__ uint8_t *startPos, __gm__ uint8_t *seqUsed,
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
__aicore__ inline uint32_t CompressorTools<COMP>::GetSeqUsed(uint32_t bIdx)
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
__aicore__ inline uint32_t CompressorTools<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos_) {
        return (uint32_t)startPosGm_.GetValue(bIdx);
    } else {
        return 0;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorTools<COMP>::GetSeqLength(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return toolParams_.seqSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorTools<COMP>::GetTIdxByBatch(uint32_t bIdx)
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        return (uint32_t)(cuSeqlensGm_.GetValue(bIdx));
    } else {
        return toolParams_.seqSize * bIdx;
    }
}

// iterator
struct SliceInfo {
    __aicore__ inline SliceInfo(){};
    __aicore__ inline SliceInfo(uint32_t bIdx, uint32_t sIdx) : bIdx(bIdx), sIdx(sIdx) {};

    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint32_t bStartPos = 0U;

    uint32_t headHolderSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailHolderSeqCnt = 0U;

    uint32_t dealSeqCnt = 0;
    uint32_t dealTcSize = 0U;
    uint32_t compressTcSize = 0U;
};

template <typename COMP>
class CompressorSliceIterator {
public:
    __aicore__ inline CompressorSliceIterator(CompressorTools<COMP> &tools) : tools_(tools) {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetMaxDealSeqCnt(uint32_t maxDealSeqCnt);
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline SliceInfo& GetSlice();
    __aicore__ inline SliceInfo& GetSliceByCmp();

    bool isFirst_ = true;
    SliceInfo sliceInfo_{};

private:
    CompressorTools<COMP> &tools_;

    // iterator
    uint32_t maxDealSeqCnt_ = 0;
    uint32_t batch_size_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorSliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    isFirst_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorSliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
}

template <typename COMP>
__aicore__ inline void CompressorSliceIterator<COMP>::SetMaxDealSeqCnt(uint32_t maxDealSeqCnt)
{
    this->maxDealSeqCnt_ = maxDealSeqCnt;
}

template <typename COMP>
__aicore__ inline bool CompressorSliceIterator<COMP>::IsEnd()
{
    return (sliceInfo_.bIdx >= batch_size_) || (maxDealSeqCnt_ == 0);
}

template <typename COMP>
__aicore__ inline void CompressorSliceIterator<COMP>::IteratorSlice()
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
__aicore__ inline SliceInfo& CompressorSliceIterator<COMP>::GetSliceByCmp()
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
__aicore__ inline SliceInfo& CompressorSliceIterator<COMP>::GetSlice()
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
    __aicore__ inline SplitCoreSliceInfo() {};
    __aicore__ inline SplitCoreSliceInfo(uint32_t bIdx, uint32_t sIdx) : SliceInfo(bIdx, sIdx) {};

    uint32_t preFirstSeqCnt = 0U; // 左边每次迭代基本块的第一个seqCnt大小
};

template <typename COMP>
class CompressorSplitCoreSliceIterator {
public:
    __aicore__ inline CompressorSplitCoreSliceIterator(CompressorTools<COMP> &tools) : tools_(tools) {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetMaxDealSeqCnt(uint32_t maxDealSeqCnt);
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline SplitCoreSliceInfo& GetSlice();
    __aicore__ inline SplitCoreSliceInfo& GetSliceByCmp();
    __aicore__ inline uint32_t GetBIdx();
    __aicore__ inline SplitCoreSliceInfo& GetLeftNextCmpSeqCnt();
    __aicore__ inline SplitCoreSliceInfo& GetRightNextCmpSeqCnt();

    bool isFirst_ = true;
    bool isLeftFirstBath = false;
    bool isMaxDealSeqCntFirst = false;

    SplitCoreSliceInfo sliceInfo_{};

private:
    CompressorTools<COMP> &tools_;

    // iterator
    uint32_t maxDealSeqCnt_ = 0;
    uint32_t batch_size_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorSplitCoreSliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    isFirst_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorSplitCoreSliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
    isMaxDealSeqCntFirst = true;
}

template <typename COMP>
__aicore__ inline void CompressorSplitCoreSliceIterator<COMP>::SetMaxDealSeqCnt(uint32_t maxDealSeqCnt)
{
    this->maxDealSeqCnt_ = maxDealSeqCnt;
}

template <typename COMP>
__aicore__ inline bool CompressorSplitCoreSliceIterator<COMP>::IsEnd()
{
    return (sliceInfo_.bIdx >= batch_size_) || (maxDealSeqCnt_ == 0);
}

template <typename COMP>
__aicore__ inline uint32_t CompressorSplitCoreSliceIterator<COMP>::GetBIdx()
{
    return sliceInfo_.bIdx;
}

template <typename COMP>
__aicore__ inline void CompressorSplitCoreSliceIterator<COMP>::IteratorSlice()
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
__aicore__ inline SplitCoreSliceInfo& CompressorSplitCoreSliceIterator<COMP>::GetLeftNextCmpSeqCnt()
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
__aicore__ inline SplitCoreSliceInfo& CompressorSplitCoreSliceIterator<COMP>::GetRightNextCmpSeqCnt()
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
    __aicore__ inline Vec1SliceInfo() {};
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx) : SliceInfo(bIdx, sIdx) {};
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt)
        : SliceInfo(bIdx, sIdx), dealedSeqCnt(dealedSeqCnt) {};

    uint32_t dealedSeqCnt = 0U;
    uint32_t dealedTcCnt = 0U;
    uint32_t bSeqLength = 0U;
    uint32_t compressoredScCnt = 0U;
    bool isFirst = false;
    bool isLast = false;
};

struct StatisticInfo {
    __aicore__ inline StatisticInfo() {};
    __aicore__ inline StatisticInfo(uint32_t actualTcCnt, uint32_t dealSeqCnt, uint32_t compressorScCnt)
        : actualTcCnt(actualTcCnt), dealSeqCnt(dealSeqCnt), compressorScCnt(compressorScCnt) {};

    uint32_t actualTcCnt = 0U;
    uint32_t dealSeqCnt = 0U;
    uint32_t compressorScCnt = 0U;
};

template <typename COMP>
class CompressorVec1SliceIterator {
public:
    __aicore__ inline CompressorVec1SliceIterator(CompressorTools<COMP> &tools) : tools_(tools) {}

    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt, uint32_t compressoredScCnt);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetDealedSeqCnt(uint32_t dealedSeqCnt);
    __aicore__ inline void SetDealedTcCnt(uint32_t dealedTcCnt);
    __aicore__ inline void SetCompressoredScCnt(uint32_t compressoredScCnt);
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
    CompressorTools<COMP> &tools_;

    bool isFirst_ = true;
    Vec1SliceInfo sliceInfo_{};
    StatisticInfo statisticInfo_{};
    uint32_t needDealTcSize_ = 0U;
    uint32_t batch_size_ = 0U;
};

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
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
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt,
                                                                uint32_t compressoredScCnt)
{
    Reset(bIdx, sIdx);
    SetDealedSeqCnt(dealedSeqCnt);
    SetCompressoredScCnt(compressoredScCnt);
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetDealedSeqCnt(uint32_t dealedSeqCnt)
{
    this->sliceInfo_.dealedSeqCnt = dealedSeqCnt;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetCompressoredScCnt(uint32_t compressoredScCnt)
{
    this->sliceInfo_.compressoredScCnt = compressoredScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetDealedTcCnt(uint32_t dealedTcCnt)
{
    this->sliceInfo_.dealedTcCnt = dealedTcCnt;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::SetNeedDealTcSize(uint32_t needDealTcSize)
{
    this->needDealTcSize_ = needDealTcSize;
}

template <typename COMP>
template <bool IS_STATISTIC>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::IteratorSlice()
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
            uint32_t seqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
            if (sliceInfo_.bSeqUsed < seqLength) {
                uint32_t nextAlignSIdx = Align(sliceInfo_.bStartPos + sliceInfo_.sIdx, cmpRatio) - sliceInfo_.bStartPos;
                sliceInfo_.dealedSeqCnt += nextAlignSIdx - sliceInfo_.sIdx;
                uint32_t tcGap = CeilDivT(static_cast<int32_t>(seqLength - nextAlignSIdx),
                                    static_cast<int32_t>(cmpRatio));
                if (sliceInfo_.bSeqUsed == 0 && nextAlignSIdx > sliceInfo_.sIdx) {
                    // 此时bseqused所在压缩块未被纳入计算
                    tcGap++;
                }
                sliceInfo_.sIdx = nextAlignSIdx;
                if (needDealTcSize_ < tcGap) {
                    sliceInfo_.dealedSeqCnt += needDealTcSize_ * cmpRatio;
                    sliceInfo_.sIdx += needDealTcSize_ * cmpRatio;
                    needDealTcSize_ = 0;
                    break;
                }
                sliceInfo_.dealedSeqCnt += seqLength - sliceInfo_.sIdx;
                needDealTcSize_ -= tcGap;
            }
            sliceInfo_.bIdx++;
            if (sliceInfo_.bIdx == batch_size_) {
                sliceInfo_.bIdx = 0;
            }
            sliceInfo_.sIdx = 0;
            sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        } while (sliceInfo_.bSeqUsed == 0);
            sliceInfo_.bSeqLength = tools_.GetSeqLength(sliceInfo_.bIdx);
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    }
    if (isFirst_) {
        isFirst_ = false;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorVec1SliceIterator<COMP>::GetNeedDealTcSize()
{
    return needDealTcSize_;
}


template <typename COMP>
__aicore__ inline bool CompressorVec1SliceIterator<COMP>::IsEnd()
{
    return (needDealTcSize_ == 0);
}

template <typename COMP>
__aicore__ inline Vec1SliceInfo& CompressorVec1SliceIterator<COMP>::GetSlice()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (sliceInfo_.bSeqUsed < sliceInfo_.sIdx) {
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
    uint32_t globalTotalSeqCnt = sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt;
    sliceInfo_.tailHolderSeqCnt = Align(globalTotalSeqCnt, cmpRatio) - globalTotalSeqCnt;

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
__aicore__ inline StatisticInfo& CompressorVec1SliceIterator<COMP>::FullIteratorSlice()
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

struct Vec2SliceInfo{
    __aicore__ inline Vec2SliceInfo(){};
    __aicore__ inline Vec2SliceInfo(uint32_t bIdx, uint32_t scIdx) : bIdx(bIdx), scIdx(scIdx)
    {
    }

    uint32_t bIdx = 0U;
    uint32_t scIdx = 0U;
    uint32_t scNum = 0U;
    uint32_t remainScCnt = 0U;     // 当前batch剩余sc数量
    uint32_t bStartPos = 0U;
    uint32_t bSeqUsed = 0U;
    uint32_t bSeqLength = 0U;
    uint32_t dealedScCnt = 0U;     // 全局的dealedScCnt（Reset刷新）
    uint32_t curDealScNum = 0U;    // 当前循环处理的sc数量（IteratorSlice刷新）
    uint32_t bOutputScLen = 0U;    // BSH场景每个batch填充后的输出长度
    uint32_t padScIdx = 0U;        // 当前sc输出位置，TH场景为全局的dealedScCnt，BSH场景则为填充后全局的索引（Reset刷新）
    uint32_t loopDealedScCnt = 0U; // 当前迭代已处理的sc数量（Reset刷新）
};


template <typename COMP>
class CompressorVec2SliceIterator {
public:
    __aicore__ inline CompressorVec2SliceIterator(CompressorTools<COMP> &tools) : tools_(tools)
    {
    }
    __aicore__ inline void Reset(uint32_t bIdx, uint32_t scIdx, uint32_t dealedScCnt);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetNeedDealScSize(uint32_t needDealScSize);
    __aicore__ inline void ResetLoopDealedScCnt();
    __aicore__ inline uint32_t GetNeedDealScSize();
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline Vec2SliceInfo &GetSlice();
private:
    CompressorTools<COMP> &tools_;

    Vec2SliceInfo sliceInfo_{};
    uint32_t needDealScSize_ = 0U;
    uint32_t batch_size_ = 0U;
};


template <typename COMP>
__aicore__ inline void CompressorVec2SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t scIdx, uint32_t dealedScCnt)
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.scIdx = scIdx;
    sliceInfo_.dealedScCnt = dealedScCnt;
    if constexpr (COMP::xLayout == X_LAYOUT::BSH) {
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
        sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        sliceInfo_.scNum = (sliceInfo_.bStartPos + sliceInfo_.bSeqUsed) / cmpRatio - sliceInfo_.bStartPos / cmpRatio;
        sliceInfo_.remainScCnt = sliceInfo_.scNum - sliceInfo_.scIdx;
        sliceInfo_.bOutputScLen = CeilDivT(tools_.GetSeqLength(sliceInfo_.bIdx), cmpRatio);
        sliceInfo_.padScIdx = sliceInfo_.bIdx * sliceInfo_.bOutputScLen + sliceInfo_.scIdx;
    } else {
        sliceInfo_.padScIdx = sliceInfo_.dealedScCnt;
    }
    sliceInfo_.loopDealedScCnt = 0U;
}

template <typename COMP>
__aicore__ inline void CompressorVec2SliceIterator<COMP>::ResetLoopDealedScCnt()
{
    sliceInfo_.loopDealedScCnt = 0U;
}



template <typename COMP>
__aicore__ inline void CompressorVec2SliceIterator<COMP>::SetMaxBatchSize(uint32_t batch_size)
{
    this->batch_size_ = batch_size;
}


template <typename COMP>
__aicore__ inline void CompressorVec2SliceIterator<COMP>::SetNeedDealScSize(uint32_t needDealScSize)
{
    this->needDealScSize_ = needDealScSize;
}

template <typename COMP>
__aicore__ inline void CompressorVec2SliceIterator<COMP>::IteratorSlice()
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        sliceInfo_.padScIdx += sliceInfo_.curDealScNum;
    } else {
        if (needDealScSize_ <= sliceInfo_.remainScCnt) {
            sliceInfo_.scIdx += sliceInfo_.curDealScNum;
            sliceInfo_.padScIdx += sliceInfo_.curDealScNum;
        } else {
            uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
            sliceInfo_.padScIdx += sliceInfo_.bOutputScLen - sliceInfo_.scIdx;
            sliceInfo_.bIdx++;
            sliceInfo_.scIdx = 0;
            sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
            sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
            sliceInfo_.scNum = (sliceInfo_.bStartPos + sliceInfo_.bSeqUsed) / cmpRatio - sliceInfo_.bStartPos / cmpRatio;
        }
        sliceInfo_.remainScCnt = sliceInfo_.scNum - sliceInfo_.scIdx;
    }
    sliceInfo_.dealedScCnt += sliceInfo_.curDealScNum;
    needDealScSize_ -= sliceInfo_.curDealScNum;
    sliceInfo_.loopDealedScCnt += sliceInfo_.curDealScNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorVec2SliceIterator<COMP>::GetNeedDealScSize()
{
    return needDealScSize_;
}


template <typename COMP>
__aicore__ inline bool CompressorVec2SliceIterator<COMP>::IsEnd()
{
    return (needDealScSize_ == 0);
}

template <typename COMP>
__aicore__ inline Vec2SliceInfo &CompressorVec2SliceIterator<COMP>::GetSlice()
{
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        sliceInfo_.curDealScNum = needDealScSize_;
    } else {
        sliceInfo_.curDealScNum = min(sliceInfo_.remainScCnt, needDealScSize_);
    }
    return sliceInfo_;
}



} // namespace Compressor

#endif
