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
 * \file scatter_nd_update_no_sort.h
 * \brief Scatter Kernel (NoSort)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {

constexpr uint64_t ALIGNED_SIZE_INDEX = 8;

template<typename T>
class ScatterNdUpdateV2KernelNoSort {
public:
    __aicore__ inline ScatterNdUpdateV2KernelNoSort() = delete;
    __aicore__ inline ScatterNdUpdateV2KernelNoSort(
        GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParam(tiling);
        InitBuffers(pipe);
        SetGmAddr(updates, output, workSpace, tiling);
    }

    __aicore__ inline void InitParam(const ScatterNdUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        CalcBlockDistribution(blockIdx_, tiling.scatterTiling.frontNum, tiling.scatterTiling.frontRow,
                              tiling.scatterTiling.tailRow, computeRow_, start_);
        end_ = start_ + computeRow_;
        totalIndexRow_ = tiling.linearIndexTiling.blockNum * tiling.linearIndexTiling.blockLength
                         + tiling.linearIndexTiling.blockRemainLength;

        scatterLength_ = tiling.scatterTiling.scatterLength;
        scatterAlignLength_ = tiling.scatterTiling.scatterAlignLength;
        ubLengthForUpdates_ = tiling.scatterTiling.ubLengthForUpdates;
        scatterTileNum_ = tiling.scatterTiling.scatterTileNum;
        scatterTileLength_ = tiling.scatterTiling.scatterTileLength;
        scatterTileTail_ = tiling.scatterTiling.scatterTileTail;
        scatterTileAlignLength_ = tiling.scatterTiling.scatterTileAlignLength;

        CalcIndexTileParams();
    }

    __aicore__ inline void CalcIndexTileParams()
    {
        uint64_t ubSizeBytes = ubLengthForUpdates_ * sizeof(T);
        uint64_t updateSizeBytes = scatterTileLength_ * sizeof(T);
        uint64_t remainBytes = ubSizeBytes - updateSizeBytes;
        indexTileLength_ = (remainBytes / sizeof(int) / ALIGNED_SIZE_INDEX) * ALIGNED_SIZE_INDEX;
        if (indexTileLength_ == 0) {
            indexTileLength_ = ALIGNED_SIZE_INDEX;
        }
        indexTileNum_ = (totalIndexRow_ + indexTileLength_ - 1) / indexTileLength_;
        indexTileTail_ = totalIndexRow_ - (indexTileNum_ - 1) * indexTileLength_;
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(indexQue_, DOUBLE_BUFFER, indexTileLength_ * sizeof(int));
        pipe.InitBuffer(updateQue_, DOUBLE_BUFFER, scatterTileLength_ * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling)
    {
        linearIndicesGm_.SetGlobalBuffer((__gm__ int*)workSpace);
        updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t tileIdx = 0; tileIdx < indexTileNum_; ++tileIdx) {
            uint64_t curTileLen = (tileIdx == indexTileNum_ - 1) ? indexTileTail_ : indexTileLength_;
            uint64_t gmOffset = tileIdx * indexTileLength_;

            LocalTensor<int> indexLocal = indexQue_.AllocTensor<int>();
            DataCopyExtParams indexCopyParams{1, static_cast<uint32_t>(curTileLen * sizeof(int)), 0, 0, 0};
            DataCopyPadExtParams<int> padParams{true, 0, 0, 0};
            DataCopyPad(indexLocal, linearIndicesGm_[gmOffset], indexCopyParams, padParams);
            indexQue_.EnQue(indexLocal);
            PipeMte2ToS();

            LocalTensor<int> indexData = indexQue_.DeQue<int>();
            for (uint64_t i = 0; i < curTileLen; ++i) {
                int64_t linearIndex = static_cast<int64_t>(indexData.GetValue(i));
                if (linearIndex >= (int64_t)start_ && linearIndex < (int64_t)end_) {
                    uint64_t idx = gmOffset + i;
                    ProcessOneIndex(idx, linearIndex);
                }
            }
            indexQue_.FreeTensor<int>(indexLocal);
        }
    }

    __aicore__ inline void ProcessOneIndex(uint64_t idx, int64_t linearIndex)
    {
        for (uint64_t tileIdx = 0; tileIdx < scatterTileNum_; ++tileIdx) {
            uint64_t tileLength = (tileIdx == scatterTileNum_ - 1) ? scatterTileTail_ : scatterTileLength_;

            LocalTensor<T> updateLocal = updateQue_.AllocTensor<T>();
            uint64_t gmOffset = idx * scatterLength_ + tileIdx * scatterTileLength_;
            DataCopyExtParams updateCopyParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(updateLocal, updatesGm_[gmOffset], updateCopyParams, padParams);
            PipeMte2ToS();

            uint64_t outOffset = linearIndex + tileIdx * scatterTileLength_;
            DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outOffset], updateLocal, outParams);
            PipeMte3ToS();

            updateQue_.FreeTensor<T>(updateLocal);
        }
    }

private:
    GlobalTensor<int> linearIndicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER> indexQue_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER> updateQue_;

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t start_;
    uint64_t end_;
    uint64_t totalIndexRow_;
    uint64_t scatterLength_;
    uint64_t scatterAlignLength_;
    uint64_t ubLengthForUpdates_;
    uint64_t scatterTileNum_;
    uint64_t scatterTileLength_;
    uint64_t scatterTileTail_;
    uint64_t scatterTileAlignLength_;
    uint64_t indexTileLength_;
    uint64_t indexTileNum_;
    uint64_t indexTileTail_;
};

} // namespace ScatterNdUpdateV2
