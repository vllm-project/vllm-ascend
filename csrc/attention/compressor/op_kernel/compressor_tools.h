/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_slice_iterator.h
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

private:
    bool isExistStartPos_ = false;
    bool isExistSeqUsed_ = false;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> startPosGm_;
};

template <typename COMP>
__aicore__ inline void CompressorTools<COMP>::Init(
    __gm__ uint8_t *startPos, __gm__ uint8_t *seqUsed, __gm__ uint8_t *cuSeqlens)
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
    __aicore__ inline SliceInfo() {};
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
    sliceInfo_.tailHolderSeqCnt = cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    // 头和尾处理，否则需要处理的seq等于cmpRatio
    if (sliceInfo_.validSeqCnt < cmpRatio) {
        sliceInfo_.dealSeqCnt = sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == 0) {
            sliceInfo_.dealSeqCnt =  cmpRatio - sliceInfo_.headHolderSeqCnt;
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
    sliceInfo_.tailHolderSeqCnt = cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
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

    uint32_t preFirstSeqCnt = 0U;   // 左边每次迭代基本块的第一个seqCnt大小
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
        uint32_t lastSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.bSeqUsed) % cmpRatio == 0 ? cmpRatio : (sliceInfo_.bStartPos + sliceInfo_.bSeqUsed) % cmpRatio;
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
    sliceInfo_.tailHolderSeqCnt = cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    // 头和尾处理，否则需要处理的seq等于cmpRatio
    if (sliceInfo_.validSeqCnt < cmpRatio) {
        sliceInfo_.dealSeqCnt = sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == 0) {
            sliceInfo_.dealSeqCnt =  cmpRatio - sliceInfo_.headHolderSeqCnt;
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
    sliceInfo_.tailHolderSeqCnt = cmpRatio - (sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt) % cmpRatio;
    if (sliceInfo_.tailHolderSeqCnt == cmpRatio) {
        sliceInfo_.tailHolderSeqCnt = 0;
    }

    // 头和尾处理，否则需要处理的seq等于cmpRatio
    if (sliceInfo_.validSeqCnt < cmpRatio) {
        sliceInfo_.dealSeqCnt = sliceInfo_.validSeqCnt;
        if (sliceInfo_.sIdx == 0) {
            sliceInfo_.dealSeqCnt =  cmpRatio - sliceInfo_.headHolderSeqCnt;
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
    __aicore__ inline Vec1SliceInfo(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt) : SliceInfo(bIdx, sIdx), dealedSeqCnt(dealedSeqCnt) {};

    uint32_t preBIdx = 0U;
    uint32_t preSIdx = 0U;
    uint32_t preBSeqUsed = 0U;
    uint32_t preBStartPos = 0U;
    uint32_t preHeadHolderSeqCnt = 0U;
    uint32_t preValidSeqCnt = 0U;
    uint32_t preTailHolderSeqCnt = 0U;
    uint32_t lastTcSeqCnt = 0U;
    uint32_t dealedSeqCnt = 0U;
    uint32_t preDealedSeqCnt = 0U;
    uint32_t dealedTcCnt = 0U;
};

struct SeqCntInfo {
    __aicore__ inline SeqCntInfo() {};

    uint32_t dealedSeqCnt = 0U;
    uint32_t preDealedSeqCnt = 0U;
};

template <typename COMP>
class CompressorVec1SliceIterator {
public:
    __aicore__ inline CompressorVec1SliceIterator(CompressorTools<COMP> &tools) : tools_(tools) {}

    template <bool RESET_FIRST = false>
    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx);
    template <bool RESET_FIRST = false>
    __aicore__ inline void Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt, uint32_t dealedTcCnt);
    __aicore__ inline void SetMaxBatchSize(uint32_t batch_size);
    __aicore__ inline void SetDealedSeqCnt(uint32_t dealedSeqCnt);
    __aicore__ inline void SetDealedTcCnt(uint32_t dealedTcCnt);
    __aicore__ inline void SetNeedDealTcSize(uint32_t needDealTcSize);
    __aicore__ inline uint32_t GetNeedDealTcSize();
    __aicore__ inline bool IsEnd();
    __aicore__ inline void IteratorSlice();
    __aicore__ inline Vec1SliceInfo& GetSlice();
    __aicore__ inline void GetPreTc();
    __aicore__ inline SeqCntInfo FullIteratorSlice();
    __aicore__ inline void Clear();
    __aicore__ inline void Load();
    __aicore__ inline void Save();

private:
    CompressorTools<COMP> &tools_;

    // iterator
    bool isFirst_ = true;
    bool isSaved_ = false;
    Vec1SliceInfo sliceInfo_{};
    Vec1SliceInfo tempSliceInfo_ {};
    uint32_t tempNeedDealTcSize_ = 0;
    uint32_t preFirstSeqCnt_ = 0U;
    uint32_t needDealTcSize_ = 0U;
    uint32_t batch_size_ = 0U;
};

template <typename COMP>
template <bool RESET_FIRST>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx)
{
    sliceInfo_.bIdx = bIdx;
    sliceInfo_.sIdx = sIdx;
    sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
    sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    if constexpr (COMP::coff == COFF::OVERLAP) {
        GetPreTc();
    }
    if constexpr (RESET_FIRST) {
        isFirst_ = true;
    }
}

template <typename COMP>
template <bool RESET_FIRST>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Reset(uint32_t bIdx, uint32_t sIdx, uint32_t dealedSeqCnt, uint32_t dealedTcCnt)
{
    Reset<RESET_FIRST>(bIdx, sIdx);
    SetDealedSeqCnt(dealedSeqCnt);
    SetDealedTcCnt(dealedTcCnt);
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
__aicore__ inline void CompressorVec1SliceIterator<COMP>::GetPreTc()
{
    uint32_t cmpRatio = tools_.toolParams_.cmpRatio;
    if (sliceInfo_.sIdx == 0) {
        // seqlen == 0 ?
        sliceInfo_.preBIdx = sliceInfo_.bIdx;
        do {
            if (sliceInfo_.preBIdx == 0) {
                sliceInfo_.preBIdx = batch_size_ - 1;
            } else {
                sliceInfo_.preBIdx--;
            }
        } while (tools_.GetSeqUsed(sliceInfo_.preBIdx) == 0);
        sliceInfo_.preBSeqUsed = tools_.GetSeqUsed(sliceInfo_.preBIdx);
        sliceInfo_.preBStartPos = tools_.GetStartPos(sliceInfo_.preBIdx);
        sliceInfo_.preSIdx = max(Trunc(sliceInfo_.preBStartPos + sliceInfo_.preBSeqUsed - 1, cmpRatio), sliceInfo_.preBStartPos) - sliceInfo_.preBStartPos;
    } else {
        sliceInfo_.preBIdx = sliceInfo_.bIdx;
        if (sliceInfo_.sIdx < cmpRatio) {
            sliceInfo_.preSIdx = 0;
        } else {
            sliceInfo_.preSIdx = sliceInfo_.sIdx - cmpRatio;
        }
        sliceInfo_.preBSeqUsed = sliceInfo_.bSeqUsed;
        sliceInfo_.preBStartPos = sliceInfo_.bStartPos;
    }
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::IteratorSlice()
{
    // printf("needDealTcSize_: %d -> ", needDealTcSize_);
    needDealTcSize_ -= (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt + sliceInfo_.tailHolderSeqCnt) / tools_.toolParams_.cmpRatio;
    // printf("%d\n", needDealTcSize_);
    sliceInfo_.dealedSeqCnt += sliceInfo_.validSeqCnt;
    sliceInfo_.dealedTcCnt += sliceInfo_.dealTcSize;
    sliceInfo_.sIdx += sliceInfo_.validSeqCnt;
    if (sliceInfo_.sIdx == sliceInfo_.bSeqUsed) {
        do {
            sliceInfo_.bIdx++;
            if (sliceInfo_.bIdx == batch_size_) {
                sliceInfo_.bIdx = 0;
            }
        } while (tools_.GetSeqUsed(sliceInfo_.bIdx) == 0);
        sliceInfo_.sIdx = 0;
        sliceInfo_.bSeqUsed = tools_.GetSeqUsed(sliceInfo_.bIdx);
        sliceInfo_.bStartPos = tools_.GetStartPos(sliceInfo_.bIdx);
    }
    if constexpr (COMP::coff == COFF::OVERLAP) {
        GetPreTc();
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
    if constexpr (COMP::coff == COFF::OVERLAP) {
        sliceInfo_.preHeadHolderSeqCnt = (sliceInfo_.preBStartPos + sliceInfo_.preSIdx) % cmpRatio;
        sliceInfo_.preValidSeqCnt = min(sliceInfo_.preBSeqUsed - sliceInfo_.preSIdx, cmpRatio - sliceInfo_.preHeadHolderSeqCnt);
        sliceInfo_.preTailHolderSeqCnt = cmpRatio - (sliceInfo_.preBStartPos + sliceInfo_.preSIdx + sliceInfo_.preValidSeqCnt) % cmpRatio;
        if (sliceInfo_.preTailHolderSeqCnt == cmpRatio) {
            sliceInfo_.preTailHolderSeqCnt = 0;
        }
        if (isFirst_) {
            preFirstSeqCnt_ = sliceInfo_.preValidSeqCnt;
        }
        sliceInfo_.preDealedSeqCnt = preFirstSeqCnt_ + sliceInfo_.dealedSeqCnt - sliceInfo_.preValidSeqCnt;
    }
    // 计算头部占位行数、有效数据行数、尾部占位行数
    sliceInfo_.headHolderSeqCnt = (sliceInfo_.bStartPos + sliceInfo_.sIdx) % cmpRatio;
    sliceInfo_.validSeqCnt = sliceInfo_.bSeqUsed - sliceInfo_.sIdx;
    if (CeilDivT(sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt, cmpRatio) > needDealTcSize_) {
        sliceInfo_.validSeqCnt = needDealTcSize_ * cmpRatio - sliceInfo_.headHolderSeqCnt;
    }
    uint32_t globalTotalSeqCnt = sliceInfo_.bStartPos + sliceInfo_.sIdx + sliceInfo_.validSeqCnt;
    sliceInfo_.tailHolderSeqCnt = Align(globalTotalSeqCnt, cmpRatio) - globalTotalSeqCnt;
    sliceInfo_.lastTcSeqCnt = globalTotalSeqCnt - max(Trunc(globalTotalSeqCnt - 1, cmpRatio), sliceInfo_.bStartPos + sliceInfo_.sIdx);

    // 计算本次可以处理的Tc个数
    sliceInfo_.dealTcSize = (sliceInfo_.headHolderSeqCnt + sliceInfo_.validSeqCnt + sliceInfo_.tailHolderSeqCnt) / cmpRatio;

    // 因为是一个batch的数据, 只有最后一个压缩块才可能不需要压缩, 此时blockInfo.tailHolderSeqCnt > 0
    sliceInfo_.compressTcSize = sliceInfo_.dealTcSize;
    if (sliceInfo_.tailHolderSeqCnt > 0) {
        sliceInfo_.compressTcSize = sliceInfo_.dealTcSize - 1; // 最后一个压缩块不满时，其不需要压缩
    }

    return sliceInfo_;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Save()
{
    tempSliceInfo_ = GetSlice();
    tempNeedDealTcSize_ = needDealTcSize_;
    isSaved_ = true;
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Load()
{
    if (isSaved_) {
        sliceInfo_ = tempSliceInfo_;
        needDealTcSize_ = tempNeedDealTcSize_;
    }
}

template <typename COMP>
__aicore__ inline void CompressorVec1SliceIterator<COMP>::Clear()
{
    isSaved_ = false;
}

template <typename COMP>
__aicore__ inline SeqCntInfo CompressorVec1SliceIterator<COMP>::FullIteratorSlice()
{
    tempSliceInfo_ = GetSlice();
    tempNeedDealTcSize_ = needDealTcSize_;
    while (!IsEnd()) {
        GetSlice();
        IteratorSlice();
    }
    Vec1SliceInfo sliceInfo = GetSlice();
    SeqCntInfo seqCntInfo {};
    seqCntInfo.dealedSeqCnt = sliceInfo.dealedSeqCnt - tempSliceInfo_.dealedSeqCnt;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        seqCntInfo.preDealedSeqCnt = sliceInfo.preDealedSeqCnt - tempSliceInfo_.preDealedSeqCnt;
    }
    sliceInfo_ = tempSliceInfo_;
    needDealTcSize_ = tempNeedDealTcSize_;
    return seqCntInfo;
}

}

#endif