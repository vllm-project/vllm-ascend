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
 * \file scatter_nd_update_linear_index.h
 * \brief LinearIndex Kernel
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdateV2 {

template<bool isSort, typename IndicesT = int>
class LinearIndexKernel {
public:
    __aicore__ inline LinearIndexKernel() = delete;
    __aicore__ inline LinearIndexKernel(
        GM_ADDR indices, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(indices, workSpace, tiling);
    }

    __aicore__ inline void InitParams(const ScatterNdUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        frontCoreNum_ = tiling.linearIndexTiling.frontCoreNum;
        tailCoreNum_ = tiling.linearIndexTiling.tailCoreNum;
        frontBlockNum_ = tiling.linearIndexTiling.frontBlockNum;
        tailBlockNum_ = tiling.linearIndexTiling.tailBlockNum;
        if (blockIdx_ >= frontCoreNum_) {
            computeNum_ = tailBlockNum_;
        } else {
            computeNum_ = frontBlockNum_;
        }
        ubSize_ = tiling.linearIndexTiling.ubSize;
        coreNum_ = tiling.linearIndexTiling.coreNum;
        blockNum_ = tiling.linearIndexTiling.blockNum;
        blockLength_ = tiling.linearIndexTiling.blockLength;
        blockRemainLength_ = tiling.linearIndexTiling.blockRemainLength;
        indexDim_ = tiling.linearIndexTiling.indexDim;
        indicesMask_ = tiling.linearIndexTiling.indicesMask;
    }

    template<bool isInt64, bool needSort>
    __aicore__ inline void InitBuffersUnified()
    {
        uint64_t offset = 0;
        indicesLocal = allUbLocal[offset];
        offset += blockLength_;
        uint64_t indicesOffset = offset;
        if constexpr (isInt64) {
            indicesInt64Local = allUbLocal[offset].ReinterpretCast<int64_t>();
            indicesOriginLocal = allUbLocal[offset];
            offset += blockLength_ * indexDim_ * 2;
        } else {
            indicesOriginLocal = allUbLocal[offset];
            offset += blockLength_ * indexDim_;
        }
        addTmpLocal = allUbLocal[offset];
        offset += blockLength_;
        rangeLocal = allUbLocal[offset];
        offset += blockLength_;
        if constexpr (isSort) {
            resLocal = allUbLocal[indicesOffset].ReinterpretCast<float>();
            indicesOffset += blockLength_ * 2;
            posIdxLocal = allUbLocal[indicesOffset];
            indicesOffset += blockLength_;
            sortTmpLocal = allUbLocal[indicesOffset].ReinterpretCast<float>();
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(allUbBuf, ubSize_);
        allUbLocal = allUbBuf.Get<int>();
        if constexpr (isSort) {
            if constexpr (std::is_same_v<IndicesT, int64_t>) {
                InitBuffersUnified<true, true>();
            } else {
                InitBuffersUnified<false, true>();
            }
        } else {
            if constexpr (std::is_same_v<IndicesT, int64_t>) {
                InitBuffersUnified<true, false>();
            } else {
                InitBuffersUnified<false, false>();
            }
        }
    }

    __aicore__ inline void SetGmAddr(GM_ADDR indices, GM_ADDR workSpace, const ScatterNdUpdateV2TilingData& tiling)
    {
        indiceAddrOffset_ =
            blockIdx_ < tiling.linearIndexTiling.frontCoreNum ?
                tiling.linearIndexTiling.frontBlockNum * blockLength_ * blockIdx_ :
                tiling.linearIndexTiling.frontCoreNum * tiling.linearIndexTiling.frontBlockNum * blockLength_ +
                    (blockIdx_ - tiling.linearIndexTiling.frontCoreNum) * tiling.linearIndexTiling.tailBlockNum *
                        blockLength_;

        sortedIndicesGm_.SetGlobalBuffer((__gm__ int*)workSpace + indiceAddrOffset_);
        if constexpr (isSort) {
            posIndicesGm_.SetGlobalBuffer((__gm__ int*)workSpace + tiling.linearIndexTiling.sortWorkspace + indiceAddrOffset_);
        }
        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            indicesGmInt64_.SetGlobalBuffer((__gm__ int64_t*)indices + indiceAddrOffset_ * indexDim_);
        } else {
            indicesGm_.SetGlobalBuffer((__gm__ int*)indices + indiceAddrOffset_ * indexDim_);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (isSort) {
            for (uint64_t i = 0; i < computeNum_; i++) {
                ProcessOneWithSort(i, false);
            }
            uint64_t lastActiveCore = (blockNum_ == 0) ? 0 :
                (tailCoreNum_ == 0 ? frontCoreNum_ - 1 : frontCoreNum_ + tailCoreNum_ - 1);
            if (blockIdx_ == lastActiveCore && blockRemainLength_ != 0) {
                ProcessOneWithSort(computeNum_, true);
            }
        } else {
            for (uint64_t i = 0; i < computeNum_; i++) {
                ProcessOne(i, false);
            }
            uint64_t lastActiveCore = (blockNum_ == 0) ? 0 :
                (tailCoreNum_ == 0 ? frontCoreNum_ - 1 : frontCoreNum_ + tailCoreNum_ - 1);
            if (blockIdx_ == lastActiveCore && blockRemainLength_ != 0) {
                ProcessOne(computeNum_, true);
            }
        }
    }

    __aicore__ inline void ProcessOne(uint64_t idx, bool isTail)
    {
        CopyIn(idx, isTail);
        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            CastToInt32(idx, isTail);
        }
        Compute4LinearIndex(idx, isTail);
        CopyOut(idx, isTail);
    }

    __aicore__ inline void ProcessOneWithSort(uint64_t idx, bool isTail)
    {
        CopyIn(idx, isTail);
        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            CastToInt32(idx, isTail);
        }
        Compute4LinearIndex(idx, isTail);
        ComputeForSort(idx, isTail);
        CopyOut(idx, isTail);
    }

    __aicore__ inline void CopyIn(uint64_t process, bool isTail)
    {
        uint64_t indicesOffset = process * blockLength_ * indexDim_;
        uint64_t copyRow = isTail ? blockRemainLength_ : blockLength_;

        if constexpr (std::is_same_v<IndicesT, int64_t>) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyRow * indexDim_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> padParams{true, 0, 0, 0};
            DataCopyPad(indicesInt64Local, indicesGmInt64_[indicesOffset], copyParams, padParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyRow * indexDim_ * sizeof(int)), 0, 0, 0};
            DataCopyPadExtParams<int> padParams{true, 0, 0, 0};
            DataCopyPad(indicesOriginLocal, indicesGm_[indicesOffset], copyParams, padParams);
        }
        PipeMte2ToS();
    }

    __aicore__ inline void CastToInt32(uint64_t process, bool isTail)
    {
        uint64_t computeRow = isTail ? blockRemainLength_ : blockLength_;
        uint64_t totalElements = computeRow * indexDim_;
        Cast(indicesOriginLocal, indicesInt64Local, RoundMode::CAST_NONE, totalElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute4LinearIndex(uint64_t process, bool isTail)
    {
        uint64_t computeRow = isTail ? blockRemainLength_ : blockLength_;
        int32_t malValue = indexDim_ * sizeof(int);
        Duplicate<int>(indicesLocal, 0, computeRow);
        CreateVecIndex(rangeLocal, (int)0, computeRow);
        PipeBarrier<PIPE_V>();
        Muls(rangeLocal, rangeLocal, malValue, computeRow);
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < indexDim_; ++i) {
            if (i != 0) {
                Adds(rangeLocal, rangeLocal, (int)(sizeof(int)), computeRow);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<uint32_t> rangeLocalCasted = rangeLocal.ReinterpretCast<uint32_t>();
            Gather(addTmpLocal, indicesOriginLocal, rangeLocalCasted, (uint32_t)0, (uint32_t)computeRow);
            PipeBarrier<PIPE_V>();
            Muls(addTmpLocal, addTmpLocal, (int)indicesMask_[i], computeRow);
            PipeBarrier<PIPE_V>();
            Add(indicesLocal, indicesLocal, addTmpLocal, computeRow);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (!isSort) {
            PipeVToMte3();
        }
    }

    __aicore__ inline void ComputeForSort(uint64_t process, bool isTail)
    {
        LocalTensor<float> indicesLocalFp32 = indicesLocal.ReinterpretCast<float>();
        uint64_t computeRow = isTail ? blockRemainLength_ : blockLength_;
        uint64_t computeRowAligned = (computeRow + ALIGNED_BLOCK_NUM - 1) & ~(ALIGNED_BLOCK_NUM - 1);
        uint64_t repeatTimes = computeRowAligned / ALIGNED_BLOCK_NUM;

        uint64_t repeatId = computeRow / ALIGNED_BLOCK_NUM;
        uint64_t repeatRemain = computeRow % ALIGNED_BLOCK_NUM;
        int addValue = indiceAddrOffset_ + process * blockLength_;

        Cast(indicesLocalFp32, indicesLocal, RoundMode::CAST_ROUND, computeRowAligned);
        if (repeatRemain != 0) {
            // 对齐处理：不足32的部分设为-1
            Duplicate<int>(rangeLocal, -1, (uint32_t)ALIGNED_BLOCK_NUM);
            PipeBarrier<PIPE_V>();
            Cast(rangeLocal, indicesLocalFp32[ALIGNED_BLOCK_NUM * repeatId], RoundMode::CAST_ROUND, (uint32_t)repeatRemain);
            PipeBarrier<PIPE_V>();
            Cast(indicesLocalFp32[ALIGNED_BLOCK_NUM * repeatId], rangeLocal, RoundMode::CAST_ROUND, (uint32_t)ALIGNED_BLOCK_NUM);
            PipeBarrier<PIPE_V>();
        }
        Duplicate<int>(posIdxLocal, -1, computeRowAligned);
        PipeBarrier<PIPE_V>();
        CreateVecIndex<int>(posIdxLocal, 0U, computeRow);
        LocalTensor<uint32_t> posIdxULocal = posIdxLocal.ReinterpretCast<uint32_t>();
        PipeBarrier<PIPE_V>();
        Sort<float, true>(resLocal, indicesLocalFp32, posIdxULocal, sortTmpLocal, repeatTimes);
        PipeBarrier<PIPE_V>();
        Extract(indicesLocalFp32, posIdxULocal, resLocal, repeatTimes);
        PipeBarrier<PIPE_V>();
        Cast(indicesLocal, indicesLocalFp32, RoundMode::CAST_ROUND, computeRowAligned);
        PipeBarrier<PIPE_V>();
        Adds(posIdxLocal, posIdxLocal, addValue, computeRow);
        PipeBarrier<PIPE_V>();
        PipeVToMte3();
    }

    __aicore__ inline void CopyOut(uint64_t process, bool isTail)
    {
        uint64_t outOffset = process * blockLength_;
        uint64_t copyRow = isTail ? blockRemainLength_ : blockLength_;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyRow * sizeof(int)), 0, 0, 0};
        DataCopyPad(sortedIndicesGm_[outOffset], indicesLocal, copyParams);
        if constexpr (isSort) {
            DataCopyPad(posIndicesGm_[outOffset], posIdxLocal, copyParams);
        }
        PipeMte3ToS();
    }

private:
    GlobalTensor<int> indicesGm_;
    GlobalTensor<int64_t> indicesGmInt64_;
    GlobalTensor<int> sortedIndicesGm_;
    GlobalTensor<int> posIndicesGm_;
    TBuf<TPosition::VECCALC> allUbBuf;

    LocalTensor<int> allUbLocal;
    LocalTensor<int> indicesLocal;
    LocalTensor<int> indicesOriginLocal;
    LocalTensor<int64_t> indicesInt64Local;
    LocalTensor<int> addTmpLocal;
    LocalTensor<int> rangeLocal;
    LocalTensor<int> posIdxLocal;
    LocalTensor<float> sortTmpLocal;
    LocalTensor<float> resLocal;

    uint64_t ubSize_;
    uint64_t coreNum_;
    uint64_t blockIdx_;
    uint64_t indexDim_;
    uint64_t computeNum_;
    uint64_t blockNum_;
    uint64_t blockLength_;
    uint64_t blockRemainLength_;
    uint64_t frontCoreNum_;
    uint64_t tailCoreNum_;
    uint64_t frontBlockNum_;
    uint64_t tailBlockNum_;
    const uint64_t* indicesMask_;
    uint64_t indiceAddrOffset_;
};

} // namespace ScatterNdUpdateV2
