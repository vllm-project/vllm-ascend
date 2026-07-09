/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_hp.h
 * \brief HighPerformance Kernel: split by indices, no SyncAll, non-deterministic on duplicate indices
 */

#ifndef SCATTER_ND_UPDATE_HP_H
#define SCATTER_ND_UPDATE_HP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdate {
constexpr uint64_t LINEAR_INDEX_COEFF_OFFSET = 3;
constexpr uint64_t INT64_TO_INT_SIZE_RATIO = 2;

template <typename T, typename IndicesT = int, bool isViewStride0 = false>
class ScatterNdUpdateHpKernel {
public:
    __aicore__ inline ScatterNdUpdateHpKernel() = delete;
    __aicore__ inline ScatterNdUpdateHpKernel(GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
                                              const ScatterNdUpdateArch32TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(indices, updates, output);
    }

    __aicore__ inline void InitParams(const ScatterNdUpdateArch32TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        indexDim_ = tiling.linearIndexTiling.indexDim;
        indicesMask_ = tiling.linearIndexTiling.indicesMask;
        scatterLength_ = tiling.scatterTiling.scatterLength;
        varStride0Elements_ = tiling.viewTiling.varStride0Elements;
        firstDimStrideRows_ = tiling.viewTiling.firstDimStrideRows;

        frontIndexNum_ = tiling.hpTiling.hpFrontIndexNum;
        tailIndexNum_ = tiling.hpTiling.hpTailIndexNum;
        frontCoreNum_ = tiling.hpTiling.hpFrontCoreNum;
        tailCoreNum_ = tiling.hpTiling.hpTailCoreNum;
        indexTileLength_ = tiling.hpTiling.hpIndexTileLength;
        scatterTileLength_ = tiling.hpTiling.hpScatterTileLength;
        scatterTileNum_ = tiling.hpTiling.hpScatterTileNum;
        scatterTileTail_ = tiling.hpTiling.hpScatterTileTail;
        rowBytesAligned_ = tiling.hpTiling.hpRowBytesAligned;
        rowsPerBatch_ = tiling.hpTiling.hpRowsPerBatch;

        slotElements_ = rowBytesAligned_ / sizeof(T);
        rowBytes_ = static_cast<uint32_t>(scatterLength_ * sizeof(T));
        isRowAligned_ = (rowBytesAligned_ == rowBytes_);

        if (blockIdx_ < frontCoreNum_) {
            ownRows_ = frontIndexNum_;
            ownStart_ = blockIdx_ * frontIndexNum_;
        } else if (blockIdx_ < frontCoreNum_ + tailCoreNum_) {
            ownRows_ = tailIndexNum_;
            ownStart_ = frontCoreNum_ * frontIndexNum_ + (blockIdx_ - frontCoreNum_) * tailIndexNum_;
        } else {
            ownRows_ = 0;
            ownStart_ = 0;
        }

        if (indexTileLength_ == 0 || ownRows_ == 0) {
            tileNum_ = 0;
            tileTail_ = 0;
        } else {
            tileNum_ = (ownRows_ + indexTileLength_ - 1) / indexTileLength_;
            tileTail_ = ownRows_ - (tileNum_ - 1) * indexTileLength_;
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        uint64_t coeff = std::is_same_v<IndicesT, int64_t> ? (2 * indexDim_ + LINEAR_INDEX_COEFF_OFFSET) :
                                                             (indexDim_ + LINEAR_INDEX_COEFF_OFFSET);
        uint64_t slotBytes = rowsPerBatch_ * rowBytesAligned_;
        pipe.InitBuffer(allUbBuf_, coeff * indexTileLength_ * sizeof(int));
        pipe.InitBuffer(slotBufA_, slotBytes);
        pipe.InitBuffer(slotBufB_, slotBytes);

        allUbLocal_ = allUbBuf_.Get<int>();
        uint64_t off = 0;
        indicesLocal_ = allUbLocal_[off];
        off += indexTileLength_;
        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            indicesInt64Local_ = allUbLocal_[off].ReinterpretCast<int64_t>();
            indicesOriginLocal_ = allUbLocal_[off];
            off += indexTileLength_ * indexDim_ * INT64_TO_INT_SIZE_RATIO;
        } else {
            indicesOriginLocal_ = allUbLocal_[off];
            off += indexTileLength_ * indexDim_;
        }
        addTmpLocal_ = allUbLocal_[off];
        off += indexTileLength_;
        rangeLocal_ = allUbLocal_[off];
    }

    __aicore__ inline void SetGmAddr(GM_ADDR indices, GM_ADDR updates, GM_ADDR output)
    {
        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            indicesGmInt64_.SetGlobalBuffer((__gm__ int64_t*)indices + ownStart_ * indexDim_);
        } else {
            indicesGm_.SetGlobalBuffer((__gm__ int*)indices + ownStart_ * indexDim_);
        }
        updatesGm_.SetGlobalBuffer((__gm__ T*)updates + ownStart_ * scatterLength_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        pingPongFlag_ = 0;

        for (uint64_t t = 0; t < tileNum_; ++t) {
            bool isTail = (t == tileNum_ - 1);
            uint64_t rows = isTail ? tileTail_ : indexTileLength_;
            CopyIndicesIn(t, rows);
            if constexpr (std::is_same_v<IndicesT, int64_t>) {
                CastToInt32(rows);
            }
            ComputeLinearIndexFromIndices(indicesLocal_, indicesOriginLocal_, addTmpLocal_, rangeLocal_, indicesMask_,
                                          indexDim_, rows);
            PipeVToS();
            ScatterTile(t, rows);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void CopyIndicesIn(uint64_t process, uint64_t rows)
    {
        uint64_t gmOffset = process * indexTileLength_ * indexDim_;
        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(rows * indexDim_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
            DataCopyPad(indicesInt64Local_, indicesGmInt64_[gmOffset], copyParams, padParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(rows * indexDim_ * sizeof(int)), 0, 0, 0};
            DataCopyPadExtParams<int> padParams{true, 0, 0, 0};
            DataCopyPad(indicesOriginLocal_, indicesGm_[gmOffset], copyParams, padParams);
        }
        PipeMte2ToS();
    }

    __aicore__ inline void CastToInt32(uint64_t rows)
    {
        Cast(indicesOriginLocal_, indicesInt64Local_, RoundMode::CAST_NONE, rows * indexDim_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScatterTile(uint64_t process, uint64_t rows)
    {
        if (scatterTileNum_ > 1) {
            ScatterTileSlice(process, rows);
        } else if (isRowAligned_) {
            ScatterTileBatchImpl<true>(process, rows);
        } else {
            ScatterTileBatchImpl<false>(process, rows);
        }
    }

    template <bool IsAligned>
    __aicore__ inline void ScatterTileBatchImpl(uint64_t process, uint64_t rows)
    {
        if (rows == 0) {
            return;
        }
        const uint64_t srcBase = process * indexTileLength_;
        const uint64_t batchNum = (rows + rowsPerBatch_ - 1) / rowsPerBatch_;
        const uint64_t gmInStep = rowsPerBatch_ * scatterLength_;
        const DataCopyPadExtParams<T> pad{true, 0, 0, 0};
        const DataCopyExtParams pout{1, rowBytes_, 0, 0, 0};

        uint64_t batchBegin = 0;
        uint64_t gmIn = srcBase * scatterLength_;
        for (uint64_t b = 0; b + 1 < batchNum; ++b) {
            DoBatch<IsAligned>(batchBegin, rowsPerBatch_, gmIn, pad, pout);
            batchBegin += rowsPerBatch_;
            gmIn += gmInStep;
        }
        DoBatch<IsAligned>(batchBegin, rows - batchBegin, gmIn, pad, pout);
    }

    template <bool IsAligned>
    __aicore__ inline void DoBatch(uint64_t batchBegin, uint64_t batchRows, uint64_t gmIn,
                                   const DataCopyPadExtParams<T>& pad, const DataCopyExtParams& pout)
    {
        event_t evt = pingPongFlag_ ? EVENT_ID1 : EVENT_ID0;
        LocalTensor<T> buf = pingPongFlag_ ? slotBufB_.template Get<T>() : slotBufA_.template Get<T>();
        pingPongFlag_ = 1 - pingPongFlag_;

        WaitFlag<HardEvent::MTE3_MTE2>(evt);

        if constexpr (IsAligned) {
            DataCopyExtParams pin{1, static_cast<uint32_t>(batchRows * rowBytes_), 0, 0, 0};
            DataCopyPad(buf, updatesGm_[gmIn], pin, pad);
        } else {
            DataCopyExtParams pin{static_cast<uint16_t>(batchRows), rowBytes_, 0, 0, 0};
            DataCopyPad(buf, updatesGm_[gmIn], pin, pad);
        }

        // MTE2 → MTE3 on the same slot.
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);

        uint64_t ubOffset = 0;
        uint64_t idx = batchBegin;
        for (uint64_t k = 0; k < batchRows; ++k) {
            uint64_t linearIndex = static_cast<uint64_t>(indicesLocal_.GetValue(idx));
            uint64_t gmOut = ResolveOutOffset<isViewStride0>(linearIndex, scatterLength_, firstDimStrideRows_,
                                                             varStride0Elements_, 0);
            DataCopyPad(outputGm_[gmOut], buf[ubOffset], pout);
            ubOffset += slotElements_;
            ++idx;
        }
        SetFlag<HardEvent::MTE3_MTE2>(evt);
    }

    __aicore__ inline void ScatterTileSlice(uint64_t process, uint64_t rows)
    {
        uint64_t srcBase = process * indexTileLength_;
        for (uint64_t i = 0; i < rows; ++i) {
            int64_t linearIndex = static_cast<int64_t>(indicesLocal_.GetValue(i));
            uint64_t srcRow = srcBase + i;
            DataCopyPadExtParams<T> pad{true, 0, 0, 0};
            for (uint64_t s = 0; s < scatterTileNum_; ++s) {
                uint64_t len = (s == scatterTileNum_ - 1) ? scatterTileTail_ : scatterTileLength_;
                uint64_t gmIn = srcRow * scatterLength_ + s * scatterTileLength_;
                uint64_t gmOut = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(linearIndex), scatterLength_,
                                                                 firstDimStrideRows_, varStride0Elements_,
                                                                 s * scatterTileLength_);
                event_t evt = pingPongFlag_ ? EVENT_ID1 : EVENT_ID0;
                LocalTensor<T> buf = pingPongFlag_ ? slotBufB_.template Get<T>() : slotBufA_.template Get<T>();
                pingPongFlag_ = 1 - pingPongFlag_;

                WaitFlag<HardEvent::MTE3_MTE2>(evt);
                DataCopyExtParams pin{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
                DataCopyPad(buf, updatesGm_[gmIn], pin, pad);
                SetFlag<HardEvent::MTE2_MTE3>(evt);
                WaitFlag<HardEvent::MTE2_MTE3>(evt);

                DataCopyExtParams pout{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
                DataCopyPad(outputGm_[gmOut], buf, pout);
                SetFlag<HardEvent::MTE3_MTE2>(evt);
            }
        }
    }

private:
    GlobalTensor<int> indicesGm_;
    GlobalTensor<int64_t> indicesGmInt64_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> allUbBuf_;
    TBuf<TPosition::VECCALC> slotBufA_;
    TBuf<TPosition::VECCALC> slotBufB_;

    LocalTensor<int> allUbLocal_;
    LocalTensor<int> indicesLocal_;
    LocalTensor<int> indicesOriginLocal_;
    LocalTensor<int64_t> indicesInt64Local_;
    LocalTensor<int> addTmpLocal_;
    LocalTensor<int> rangeLocal_;

    uint64_t blockIdx_ = 0;
    uint64_t indexDim_ = 0;
    const uint64_t* indicesMask_ = nullptr;
    uint64_t scatterLength_ = 0;
    uint64_t varStride0Elements_ = 0;
    uint64_t firstDimStrideRows_ = 0;

    // HP tiling 字段（与 tiling.hpTiling 1:1 对应）
    uint64_t frontIndexNum_ = 0;
    uint64_t tailIndexNum_ = 0;
    uint64_t frontCoreNum_ = 0;
    uint64_t tailCoreNum_ = 0;
    uint64_t indexTileLength_ = 0;
    uint64_t scatterTileLength_ = 0;
    uint64_t scatterTileNum_ = 1;
    uint64_t scatterTileTail_ = 0;
    uint64_t rowBytesAligned_ = 0;
    uint64_t rowsPerBatch_ = 1;

    uint64_t slotElements_ = 0;
    uint32_t rowBytes_ = 0;
    bool isRowAligned_ = false;
    uint32_t pingPongFlag_ = 0;

    uint64_t ownStart_ = 0;
    uint64_t ownRows_ = 0;
    uint64_t tileNum_ = 0;
    uint64_t tileTail_ = 0;
};

} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_HP_H
