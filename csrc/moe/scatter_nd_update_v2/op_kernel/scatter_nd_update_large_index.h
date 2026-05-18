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

template<typename T>
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

        startInt64_ = static_cast<int64_t>(start_);
        endInt64_ = static_cast<int64_t>(end_);

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
        uint64_t indicesInt64Size = ((blockLength_ * indexDim_ * 2) + ALIGN_NUM - 1) & ~(ALIGN_NUM - 1);
        uint64_t updateBufBytes = (ubLengthForUpdates_ * sizeof(T) + 31) & ~31ULL;

        pipe.InitBuffer(indicesBuf, indicesInt64Size * sizeof(int));
        pipe.InitBuffer(updateBuf, updateBufBytes);

        indicesInt64Local = indicesBuf.Get<int>().ReinterpretCast<int64_t>();
        updateLocal = updateBuf.Get<T>();
    }

    __aicore__ inline void SetGmAddr(GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
                                      const ScatterNdUpdateV2TilingData& tiling)
    {
        indicesGmInt64_.SetGlobalBuffer((__gm__ int64_t*)indices);
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
        CopyInInt64(blockIdx, isTail);

        for (uint64_t i = 0; i < copyRow; ++i) {
            int64_t linearIndex = ComputeLinearIndex(i);
            if (linearIndex >= startInt64_ && linearIndex < endInt64_) {
                ScatterUpdate(i, linearIndex);
            }
        }
    }

    __aicore__ inline void CopyInInt64(uint64_t blockIdx, bool isTail)
    {
        uint64_t indicesOffset = blockIdx * blockLength_ * indexDim_;
        uint64_t copyRow = isTail ? blockRemainLength_ : blockLength_;

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyRow * indexDim_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
        DataCopyPad(indicesInt64Local, indicesGmInt64_[indicesOffset], copyParams, padParams);
        PipeMte2ToS();
    }

    __aicore__ inline int64_t ComputeLinearIndex(uint64_t rowIdx)
    {
        int64_t linearIndex = 0;
        for (uint64_t dim = 0; dim < indexDim_; ++dim) {
            int64_t idxValue = indicesInt64Local.GetValue(rowIdx * indexDim_ + dim);
            int64_t stride = static_cast<int64_t>(indicesMask_[dim]);
            linearIndex += idxValue * stride;
        }
        return linearIndex;
    }

    __aicore__ inline void ScatterUpdate(uint64_t rowIdx, int64_t linearIndex)
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
    GlobalTensor<int64_t> indicesGmInt64_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;

    TBuf<TPosition::VECCALC> indicesBuf;
    TBuf<TPosition::VECCALC> updateBuf;

    LocalTensor<int64_t> indicesInt64Local;
    LocalTensor<T> updateLocal;

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t start_;
    uint64_t end_;
    int64_t startInt64_;
    int64_t endInt64_;

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
