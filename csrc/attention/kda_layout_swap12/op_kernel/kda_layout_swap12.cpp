/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t SWAP12_MTE2_MTE3_EVENT_ID = 0;
constexpr uint32_t SWAP12_MTE3_MTE2_EVENT_ID = 1;
constexpr uint32_t SWAP12_UB_ELEMENTS = 8192;
constexpr uint32_t SWAP12_MAX_GROUP_ROWS = 64;

template <typename T>
class KdaLayoutSwap12Kernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const KdaLayoutSwap12TilingData &tiling, TPipe *pipe)
    {
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        pipe_ = pipe;
        batch_ = static_cast<uint64_t>(tiling.batch);
        firstDim_ = static_cast<uint64_t>(tiling.firstDim);
        secondDim_ = static_cast<uint64_t>(tiling.secondDim);
        tailDim_ = static_cast<uint64_t>(tiling.tailDim);
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        pipe_->InitBuffer(copyBuf_, SWAP12_UB_ELEMENTS * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (CanUseGroupedCopy()) {
            ProcessGroupedRows();
            return;
        }
        ProcessSingleRows();
    }

private:
    __aicore__ inline bool CanUseGroupedCopy() const
    {
        uint64_t rowBytes = tailDim_ * static_cast<uint64_t>(sizeof(T));
        uint64_t blockLen = rowBytes / 32;
        uint64_t srcStride = secondDim_ > 0 ? (secondDim_ - 1) * blockLen : 0;
        return rowBytes >= 32 && rowBytes % 32 == 0 && blockLen > 0 && blockLen <= 65535 &&
               srcStride <= 65535 && tailDim_ <= SWAP12_UB_ELEMENTS;
    }

    __aicore__ inline uint64_t GroupRows() const
    {
        uint64_t rows = SWAP12_UB_ELEMENTS / tailDim_;
        if (rows == 0) {
            rows = 1;
        }
        if (rows > SWAP12_MAX_GROUP_ROWS) {
            rows = SWAP12_MAX_GROUP_ROWS;
        }
        return rows;
    }

    __aicore__ inline void ProcessSingleRows()
    {
        uint64_t rowCount = batch_ * firstDim_ * secondDim_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t row = coreIdx; row < rowCount; row += usedCoreNum_) {
            CopySwappedRow(row);
        }
    }

    __aicore__ inline void ProcessGroupedRows()
    {
        uint64_t rowsPerTile = GroupRows();
        uint64_t tileCount = (firstDim_ + rowsPerTile - 1) / rowsPerTile;
        uint64_t taskCount = batch_ * secondDim_ * tileCount;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t task = coreIdx; task < taskCount; task += usedCoreNum_) {
            uint64_t tileIdx = task % tileCount;
            uint64_t rem = task / tileCount;
            uint64_t j = rem % secondDim_;
            uint64_t b = rem / secondDim_;
            uint64_t firstStart = tileIdx * rowsPerTile;
            uint64_t rows = firstDim_ - firstStart;
            if (rows > rowsPerTile) {
                rows = rowsPerTile;
            }
            CopySwappedRows(b, firstStart, j, rows);
        }
    }

    __aicore__ inline void CopySwappedRows(uint64_t b, uint64_t firstStart, uint64_t j, uint64_t rows)
    {
        LocalTensor<T> local = copyBuf_.Get<T>();
        uint64_t rowBytes = tailDim_ * static_cast<uint64_t>(sizeof(T));
        uint64_t blockLen = rowBytes / 32;
        uint64_t srcBase = ((b * firstDim_ + firstStart) * secondDim_ + j) * tailDim_;
        uint64_t dstBase = ((b * secondDim_ + j) * firstDim_ + firstStart) * tailDim_;

        DataCopyParams copyInParams;
        copyInParams.blockCount = static_cast<uint16_t>(rows);
        copyInParams.blockLen = static_cast<uint16_t>(blockLen);
        copyInParams.srcStride = static_cast<uint16_t>((secondDim_ - 1) * blockLen);
        copyInParams.dstStride = 0;
        DataCopy(local, x_[srcBase], copyInParams);
        SetFlag<HardEvent::MTE2_MTE3>(SWAP12_MTE2_MTE3_EVENT_ID);
        WaitFlag<HardEvent::MTE2_MTE3>(SWAP12_MTE2_MTE3_EVENT_ID);

        DataCopy(y_[dstBase], local, static_cast<uint32_t>(rows * tailDim_));
        SetFlag<HardEvent::MTE3_MTE2>(SWAP12_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(SWAP12_MTE3_MTE2_EVENT_ID);
    }

    __aicore__ inline void CopyTile(uint64_t srcOffset, uint64_t dstOffset, uint64_t elems)
    {
        LocalTensor<T> local = copyBuf_.Get<T>();
        uint64_t rowBytes = elems * static_cast<uint64_t>(sizeof(T));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(local, x_[srcOffset], static_cast<uint32_t>(elems));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(local, x_[srcOffset], params, padParams);
        }
        SetFlag<HardEvent::MTE2_MTE3>(SWAP12_MTE2_MTE3_EVENT_ID);
        WaitFlag<HardEvent::MTE2_MTE3>(SWAP12_MTE2_MTE3_EVENT_ID);
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(y_[dstOffset], local, static_cast<uint32_t>(elems));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPad(y_[dstOffset], local, params);
        }
        SetFlag<HardEvent::MTE3_MTE2>(SWAP12_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(SWAP12_MTE3_MTE2_EVENT_ID);
    }

    __aicore__ inline void CopySwappedRow(uint64_t row)
    {
        uint64_t perBatchRows = firstDim_ * secondDim_;
        uint64_t b = row / perBatchRows;
        uint64_t rem = row - b * perBatchRows;
        uint64_t i = rem / secondDim_;
        uint64_t j = rem - i * secondDim_;

        uint64_t srcBase = ((b * firstDim_ + i) * secondDim_ + j) * tailDim_;
        uint64_t dstBase = ((b * secondDim_ + j) * firstDim_ + i) * tailDim_;
        for (uint64_t off = 0; off < tailDim_; off += SWAP12_UB_ELEMENTS) {
            uint64_t elems = tailDim_ - off;
            if (elems > SWAP12_UB_ELEMENTS) {
                elems = SWAP12_UB_ELEMENTS;
            }
            CopyTile(srcBase + off, dstBase + off, elems);
        }
    }

    GlobalTensor<T> x_;
    GlobalTensor<T> y_;
    TBuf<TPosition::VECCALC> copyBuf_;
    TPipe *pipe_ = nullptr;
    uint64_t batch_ = 0;
    uint64_t firstDim_ = 0;
    uint64_t secondDim_ = 0;
    uint64_t tailDim_ = 0;
    uint64_t usedCoreNum_ = 1;
};
} // namespace

extern "C" __global__ __aicore__ void kda_layout_swap12(GM_ADDR x, GM_ADDR dependency, GM_ADDR y,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    (void)dependency;
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        KdaLayoutSwap12Kernel<float> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        KdaLayoutSwap12Kernel<bfloat16_t> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        KdaLayoutSwap12Kernel<half> op;
        op.Init(x, y, tilingData, &pipe);
        op.Process();
    }
}
