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
 * \file scatter_nd_update_v2.h
 * \brief Scatter Kernel (Sort)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {
template<typename T>
class ScatterNdUpdateV2Kernel {
public:
    __aicore__ inline ScatterNdUpdateV2Kernel() = delete;
    __aicore__ inline ScatterNdUpdateV2Kernel(
        GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(updates, output, workSpace, tiling);
    }

    __aicore__ inline void InitParams(const ScatterNdUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        CalcBlockDistribution(blockIdx_, tiling.scatterTiling.frontNum, tiling.scatterTiling.frontRow,
                              tiling.scatterTiling.tailRow, computeRow_, start_);
        end_ = start_ + computeRow_;
        blockNum_ = tiling.linearIndexTiling.blockNum;
        blockLength_ = tiling.linearIndexTiling.blockLength;
        blockRemainLength_ = tiling.linearIndexTiling.blockRemainLength;
        coreNum_ = tiling.linearIndexTiling.coreNum;

        scatterLength_ = tiling.scatterTiling.scatterLength;
        scatterAlignLength_ = tiling.scatterTiling.scatterAlignLength;
        ubLengthForUpdates_ = tiling.scatterTiling.ubLengthForUpdates;
        formDim_ = tiling.scatterTiling.formDim;
        copyRow_ = tiling.scatterTiling.copyRow;
        scatterTileNum_ = tiling.scatterTiling.scatterTileNum;
        scatterTileLength_ = tiling.scatterTiling.scatterTileLength;
        scatterTileTail_ = tiling.scatterTiling.scatterTileTail;
        scatterTileAlignLength_ = tiling.scatterTiling.scatterTileAlignLength;
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(indiceQue_, DOUBLE_BUFFER, blockLength_  * sizeof(int));
        pipe.InitBuffer(posIdxQue_, DOUBLE_BUFFER, blockLength_  * sizeof(int));
        pipe.InitBuffer(updateQue_, DOUBLE_BUFFER, ubLengthForUpdates_ * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling)
    {
        sortedIndicesGm_.SetGlobalBuffer((__gm__ int*)workSpace);
        posIndicesGm_.SetGlobalBuffer((__gm__ int*)workSpace + tiling.linearIndexTiling.sortWorkspace);
        updatesGm_.SetGlobalBuffer((__gm__ T*)updates);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < blockNum_; ++i) {
            CopyIndicesIn(i, false);
            Compute(i, false);
            PipeMte3ToS();
        }
        if (blockRemainLength_ != 0) {
            CopyIndicesIn(blockNum_, true);
            Compute(blockNum_, true);
            PipeMte3ToS();
        }
    }


    __aicore__ inline void CopyIndicesIn(uint64_t process, bool isTail)
    {
        uint64_t copyNum = isTail ? blockRemainLength_ : blockLength_;
        LocalTensor<int> indiceLocal = indiceQue_.AllocTensor<int>();
        LocalTensor<int> posIdxLocal = posIdxQue_.AllocTensor<int>();
        uint64_t indicesOffset = isTail ? (blockNum_ * blockLength_) : (process * blockLength_);
        DataCopyParams indiceCopyParams{1, static_cast<uint16_t>(copyNum * sizeof(int)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(indiceLocal, sortedIndicesGm_[indicesOffset], indiceCopyParams, padParams);
        DataCopyPad(posIdxLocal, posIndicesGm_[indicesOffset], indiceCopyParams, padParams);
        PipeMte2ToS();
        PipeBarrier<PIPE_V>();
        UpdateSearchParam(indiceLocal, isTail);
        indiceQue_.EnQue<int>(indiceLocal);
        posIdxQue_.EnQue<int>(posIdxLocal);
    }

    __aicore__ inline void CopyUpdateIn(LocalTensor<T> &updateLocal, uint64_t gmIdx, uint64_t ubIdx, uint64_t tileIdx, uint64_t tileLength)
    {
        uint64_t gmOffset = gmIdx * scatterLength_ + tileIdx * scatterTileLength_;
        uint64_t ubOffset = ubIdx * scatterTileAlignLength_;
        DataCopyExtParams updateCopyParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(updateLocal[ubOffset], updatesGm_[gmOffset], updateCopyParams, padParams);
        PipeMte2ToS();
    }

    // 降序数组：二分查找边界
    __aicore__ inline int64_t findFirstLt(LocalTensor<int> &indiceLocal, int64_t target, bool isTail)
    {
        int64_t left = 0;
        int64_t right = (isTail ? blockRemainLength_ : blockLength_) - 1;
        int64_t res = isTail ? blockRemainLength_ : blockLength_;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            int64_t value = indiceLocal.GetValue(mid);
            if (value < target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    __aicore__ inline int64_t findLastGe(LocalTensor<int> &indiceLocal, int64_t target, bool isTail)
    {
        int64_t left = 0;
        int64_t right = (isTail ? blockRemainLength_ : blockLength_) - 1;
        int64_t res = -1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            int64_t value = indiceLocal.GetValue(mid);
            if (value >= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    __aicore__ inline void UpdateSearchParam(LocalTensor<int> &indiceLocal, bool isTail)
    {
        int64_t searchNum = isTail ? blockRemainLength_ : blockLength_;
        leftBound_ = findFirstLt(indiceLocal, end_, isTail);
        rightBound_ = findLastGe(indiceLocal, start_, isTail);
        isValidBound_ = (leftBound_ < searchNum && rightBound_ != -1 && leftBound_ <= rightBound_);
    }

    __aicore__ inline void Compute(uint64_t process, bool isTail)
    {
        LocalTensor<int> indiceLocal = indiceQue_.DeQue<int>();
        LocalTensor<int> posIdxLocal = posIdxQue_.DeQue<int>();

        if (!isValidBound_) {
            indiceQue_.FreeTensor<int>(indiceLocal);
            posIdxQue_.FreeTensor<int>(posIdxLocal);
            return;
        }

        for (uint64_t tileIdx = 0; tileIdx < scatterTileNum_; ++tileIdx) {
            uint64_t tileLength = (tileIdx == scatterTileNum_ - 1) ? scatterTileTail_ : scatterTileLength_;

            uint64_t inUbNum = 0;
            LocalTensor<T> updateLocal;

            lastProcessedIdx_ = -1;

            for (int64_t i = rightBound_; i >= leftBound_; --i) {
                if (inUbNum == 0) {
                    updateLocal = updateQue_.AllocTensor<T>();
                }
                int64_t posIdx = posIdxLocal.GetValue(i);
                CopyUpdateIn(updateLocal, posIdx, inUbNum, tileIdx, tileLength);
                inUbNum++;

                if (inUbNum == copyRow_) {
                    updateQue_.EnQue<T>(updateLocal);
                    CopyOut(inUbNum, i, indiceLocal, posIdxLocal, tileIdx, tileLength);
                    inUbNum = 0;
                }
                if (i == leftBound_ && inUbNum != 0) {
                    updateQue_.EnQue<T>(updateLocal);
                    CopyOut(inUbNum, i, indiceLocal, posIdxLocal, tileIdx, tileLength);
                }
            }
        }

        indiceQue_.FreeTensor<int>(indiceLocal);
        posIdxQue_.FreeTensor<int>(posIdxLocal);
    }

    __aicore__ inline void CopyOut(uint64_t inUbNum, int64_t curIdx, LocalTensor<int> &indiceLocal,
                                    LocalTensor<int> &posIdxLocal, uint64_t tileIdx, uint64_t tileLength)
    {
        LocalTensor<T> updateLocal = updateQue_.DeQue<T>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
        for (int64_t i = curIdx + inUbNum - 1; i >= curIdx; --i) {
            int64_t curIdxValue = indiceLocal.GetValue(i);
            if (curIdxValue == lastProcessedIdx_) continue;
            lastProcessedIdx_ = curIdxValue;
            uint64_t outOffset = curIdxValue + tileIdx * scatterTileLength_;
            uint64_t updateOffset = (curIdx + inUbNum - 1 - i) * scatterTileAlignLength_;
            DataCopyPad(outputGm_[outOffset], updateLocal[updateOffset], outParams);
        }
        PipeMte3ToS();
        updateQue_.FreeTensor<T>(updateLocal);
    }

private:

    GlobalTensor<int> sortedIndicesGm_;
    GlobalTensor<int> posIndicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> indiceQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> posIdxQue_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER> updateQue_;

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t start_;
    uint64_t end_;
    uint64_t blockNum_;
    uint64_t blockLength_;
    uint64_t blockRemainLength_;
    uint64_t scatterLength_;
    uint64_t scatterAlignLength_;
    uint64_t ubLengthForUpdates_;
    uint64_t formDim_;
    uint64_t copyRow_;
    uint64_t coreNum_;
    uint64_t scatterTileNum_;
    uint64_t scatterTileLength_;
    uint64_t scatterTileTail_;
    uint64_t scatterTileAlignLength_;
    int64_t leftBound_;
    int64_t rightBound_;
    bool isValidBound_;
    int64_t lastProcessedIdx_;
};
} // namespace ScatterNdUpdateV2
