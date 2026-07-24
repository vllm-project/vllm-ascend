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
 * \file scatter_nd_update_large_index.h
 * \brief LargeIndex Kernel (index > 2^31-1)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {

template<typename T, typename IndicesT = int64_t>
class LargeIndexKernel {
public:
    __aicore__ inline LargeIndexKernel() = delete;
    __aicore__ inline LargeIndexKernel(
        GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
        const ScatterNdUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(indices, updates, output, tiling);
    }

    __aicore__ inline void InitParams(const ScatterNdUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();

        CalcBlockDistribution(blockIdx_, tiling.scatterTiling.frontNum, tiling.scatterTiling.frontRow,
                              tiling.scatterTiling.tailRow, computeRow_, start_);
        end_ = start_ + computeRow_;

        indexDim_ = tiling.linearIndexTiling.indexDim;
        blockLength_ = tiling.linearIndexTiling.blockLength;
        blockNum_ = tiling.linearIndexTiling.blockNum;
        blockRemainLength_ = tiling.linearIndexTiling.blockRemainLength;

        scatterLength_ = tiling.scatterTiling.scatterLength;
        ubLengthForUpdates_ = tiling.scatterTiling.ubLengthForUpdates;
        scatterTileNum_ = tiling.scatterTiling.scatterTileNum;
        scatterTileLength_ = tiling.scatterTiling.scatterTileLength;
        scatterTileTail_ = tiling.scatterTiling.scatterTileTail;

        for (uint64_t i = 0; i < indexDim_; ++i) {
            indicesMask_[i] = tiling.linearIndexTiling.indicesMask[i];
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        uint64_t indicesBytes = (blockLength_ * indexDim_ * sizeof(IndicesT) + 31) & ~31ULL;
        uint64_t updateBufBytes = (ubLengthForUpdates_ * sizeof(T) + 31) & ~31ULL;

        pipe.InitBuffer(indicesBuf, indicesBytes);
        pipe.InitBuffer(updateBuf, updateBufBytes);

        indicesLocal = indicesBuf.Get<IndicesT>();
        updateLocal = updateBuf.Get<T>();
    }

    __aicore__ inline void SetGmAddr(GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
                                      const ScatterNdUpdateV2TilingData& tiling)
    {
        indicesGm_.SetGlobalBuffer((__gm__ IndicesT*)indices);
        updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t blockIdx = 0; blockIdx < blockNum_; ++blockIdx) {
            ProcessOneBlock(blockIdx, false);
        }
        if (blockRemainLength_ != 0) {
            ProcessOneBlock(blockNum_, true);
        }
    }

    __aicore__ inline void ProcessOneBlock(uint64_t blockIdx, bool isTail)
    {
        uint64_t copyRow = isTail ? blockRemainLength_ : blockLength_;
        CopyInIndices(blockIdx, isTail);

        for (uint64_t i = 0; i < copyRow; ++i) {
            uint64_t linearIndex = 0;
            if (ComputeLinearIndex(i, linearIndex) && linearIndex >= start_ && linearIndex < end_) {
                ScatterUpdate(i, linearIndex);
            }
        }
    }

    __aicore__ inline void CopyInIndices(uint64_t blockIdx, bool isTail)
    {
        uint64_t indicesOffset = blockIdx * blockLength_ * indexDim_;
        uint64_t copyRow = isTail ? blockRemainLength_ : blockLength_;

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyRow * indexDim_ * sizeof(IndicesT)), 0, 0, 0};
        DataCopyPadExtParams<IndicesT> padParams{true, 0, 0, 0};
        DataCopyPad(indicesLocal, indicesGm_[indicesOffset], copyParams, padParams);
        PipeMte2ToS();
    }

    __aicore__ inline bool ComputeLinearIndex(uint64_t rowIdx, uint64_t& linearIndex)
    {
        linearIndex = 0;
        for (uint64_t dim = 0; dim < indexDim_; ++dim) {
            int64_t idxValue = static_cast<int64_t>(indicesLocal.GetValue(rowIdx * indexDim_ + dim));
            if (idxValue < 0) {
                return false;
            }
            linearIndex += static_cast<uint64_t>(idxValue) * indicesMask_[dim];
        }
        return true;
    }

    __aicore__ inline void ScatterUpdate(uint64_t rowIdx, uint64_t linearIndex)
    {
        for (uint64_t tileIdx = 0; tileIdx < scatterTileNum_; ++tileIdx) {
            uint64_t tileLength = (tileIdx == scatterTileNum_ - 1) ? scatterTileTail_ : scatterTileLength_;
            uint64_t gmOffset = rowIdx * scatterLength_ + tileIdx * scatterTileLength_;
            DataCopyExtParams updateCopyParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(updateLocal, updatesGm_[gmOffset], updateCopyParams, padParams);
            PipeMte2ToS();

            uint64_t outOffset = static_cast<uint64_t>(linearIndex) + tileIdx * scatterTileLength_;
            DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outOffset], updateLocal, outParams);
            PipeMte3ToS();
        }
    }

private:
    GlobalTensor<IndicesT> indicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;

    TBuf<TPosition::VECCALC> indicesBuf;
    TBuf<TPosition::VECCALC> updateBuf;

    LocalTensor<IndicesT> indicesLocal;
    LocalTensor<T> updateLocal;

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t start_;
    uint64_t end_;
    uint64_t indexDim_;
    uint64_t blockLength_;
    uint64_t blockNum_;
    uint64_t blockRemainLength_;
    uint64_t indicesMask_[8];

    uint64_t scatterLength_;
    uint64_t ubLengthForUpdates_;
    uint64_t scatterTileNum_;
    uint64_t scatterTileLength_;
    uint64_t scatterTileTail_;
};

} // namespace ScatterNdUpdateV2
