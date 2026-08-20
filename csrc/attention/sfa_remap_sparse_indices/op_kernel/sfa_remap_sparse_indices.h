/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef SFA_REMAP_SPARSE_INDICES_H
#define SFA_REMAP_SPARSE_INDICES_H

#include "kernel_operator.h"

namespace SfaRemapSparseIndices {
using namespace AscendC;

struct SfaRemapSparseIndicesTilingData {
    uint32_t rows;
    uint32_t topK;
    uint32_t dcpSize;
    uint32_t dcpRank;
    uint32_t interleaveSize;
    uint32_t interleaveShift;
    uint32_t dcpInterleaveShift;
    uint32_t usePowerOfTwo;
    uint32_t rowsPerCore;
    uint32_t bufferBytes;
};

class SfaRemapSparseIndicesKernel {
public:
    __aicore__ inline explicit SfaRemapSparseIndicesKernel(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const SfaRemapSparseIndicesTilingData* tilingData)
    {
        rows_ = tilingData->rows;
        topK_ = tilingData->topK;
        dcpSize_ = tilingData->dcpSize;
        dcpRank_ = tilingData->dcpRank;
        interleaveSize_ = tilingData->interleaveSize;
        interleaveShift_ = tilingData->interleaveShift;
        dcpInterleaveShift_ = tilingData->dcpInterleaveShift;
        usePowerOfTwo_ = tilingData->usePowerOfTwo;
        rowsPerCore_ = tilingData->rowsPerCore;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input), rows_ * topK_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output), rows_ * topK_);
        pipe_->InitBuffer(inputQueue_, 1, tilingData->bufferBytes);
        pipe_->InitBuffer(outputQueue_, 1, tilingData->bufferBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = GetBlockIdx() * rowsPerCore_;
        uint32_t end = start + rowsPerCore_;
        if (end > rows_) {
            end = rows_;
        }
        for (uint32_t row = start; row < end; ++row) {
            CopyIn(row);
            Compute();
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row)
    {
        LocalTensor<int32_t> input = inputQueue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(input, inputGm_[row * topK_], copyParams, padParams);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void ComputeScalar(const LocalTensor<int32_t>& input,
                                         const LocalTensor<int32_t>& output)
    {
        uint32_t writePos = 0;
        Duplicate(output, static_cast<int32_t>(-1), topK_);
        event_t vectorToScalar = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);

        for (uint32_t column = 0; column < topK_; ++column) {
            int32_t value = input.GetValue(column);
            if (value < 0) {
                continue;
            }
            uint32_t unsignedValue = static_cast<uint32_t>(value);
            uint32_t owner = 0;
            int32_t remapped = -1;
            if (usePowerOfTwo_ != 0) {
                owner = (unsignedValue >> interleaveShift_) & (dcpSize_ - 1);
                remapped = static_cast<int32_t>(
                    ((unsignedValue >> dcpInterleaveShift_) << interleaveShift_) |
                    (unsignedValue & (interleaveSize_ - 1)));
            } else {
                uint32_t block = unsignedValue / interleaveSize_;
                owner = block % dcpSize_;
                remapped = static_cast<int32_t>(
                    (block / dcpSize_) * interleaveSize_ + unsignedValue % interleaveSize_);
            }
            if (owner == dcpRank_) {
                output.SetValue(writePos, remapped);
                ++writePos;
            }
        }

        event_t scalarToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarToMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> input = inputQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> output = outputQueue_.AllocTensor<int32_t>();
        ComputeScalar(input, output);
        inputQueue_.FreeTensor(input);
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        LocalTensor<int32_t> output = outputQueue_.DeQue<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(outputGm_[row * topK_], output, copyParams);
        outputQueue_.FreeTensor(output);
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    uint32_t rows_;
    uint32_t topK_;
    uint32_t dcpSize_;
    uint32_t dcpRank_;
    uint32_t interleaveSize_;
    uint32_t interleaveShift_;
    uint32_t dcpInterleaveShift_;
    uint32_t usePowerOfTwo_;
    uint32_t rowsPerCore_;
};
}  // namespace SfaRemapSparseIndices

#endif
