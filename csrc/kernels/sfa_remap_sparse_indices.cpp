/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"

namespace {

// GLM-5.2 uses 2048. Keeping a larger static UB capacity also covers other
// sparse-attention configurations without changing the kernel ABI.
constexpr uint32_t kMaxTopK = 8192;
constexpr uint32_t kBufferBytes = kMaxTopK * sizeof(int32_t);

class SfaRemapSparseIndices {
public:
    __aicore__ inline explicit SfaRemapSparseIndices(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ int32_t* input,
        __gm__ int32_t* output,
        uint32_t rows,
        uint32_t topK,
        uint32_t dcpSize,
        uint32_t dcpRank,
        uint32_t interleaveSize,
        uint32_t interleaveShift,
        uint32_t dcpInterleaveShift,
        uint32_t usePowerOfTwo,
        uint32_t rowsPerCore)
    {
        inputGm_.SetGlobalBuffer(input, rows * topK);
        outputGm_.SetGlobalBuffer(output, rows * topK);
        rows_ = rows;
        topK_ = topK;
        dcpSize_ = dcpSize;
        dcpRank_ = dcpRank;
        interleaveSize_ = interleaveSize;
        interleaveShift_ = interleaveShift;
        dcpInterleaveShift_ = dcpInterleaveShift;
        usePowerOfTwo_ = usePowerOfTwo;
        rowsPerCore_ = rowsPerCore;
        pipe_->InitBuffer(inputQueue_, 1, kBufferBytes);
        pipe_->InitBuffer(outputQueue_, 1, kBufferBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t start = core * rowsPerCore_;
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
        AscendC::LocalTensor<int32_t> input = inputQueue_.AllocTensor<int32_t>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(input, inputGm_[row * topK_], copyParams, padParams);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<int32_t> input = inputQueue_.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> output = outputQueue_.AllocTensor<int32_t>();
        uint32_t writePos = 0;

        // Stable compaction has a scalar prefix dependency. Vector initializes
        // the invalid tail, then Scalar only writes locally-owned values.
        AscendC::Duplicate(output, static_cast<int32_t>(-1), topK_);
        event_t vectorToScalar = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);

        for (uint32_t column = 0; column < topK_; ++column) {
            const int32_t value = input.GetValue(column);
            if (value < 0) {
                continue;
            }
            const uint32_t unsignedValue = static_cast<uint32_t>(value);
            uint32_t owner = 0;
            int32_t remapped = -1;
            if (usePowerOfTwo_ != 0) {
                owner = (unsignedValue >> interleaveShift_) & (dcpSize_ - 1);
                remapped = static_cast<int32_t>(
                    ((unsignedValue >> dcpInterleaveShift_) << interleaveShift_)
                    | (unsignedValue & (interleaveSize_ - 1)));
            } else {
                const uint32_t block = unsignedValue / interleaveSize_;
                owner = block % dcpSize_;
                remapped = static_cast<int32_t>(
                    (block / dcpSize_) * interleaveSize_ + unsignedValue % interleaveSize_);
            }
            if (owner == dcpRank_) {
                output.SetValue(writePos, remapped);
                ++writePos;
            }
        }

        event_t scalarToMte3 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3);

        inputQueue_.FreeTensor(input);
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        AscendC::LocalTensor<int32_t> output = outputQueue_.DeQue<int32_t>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm_[row * topK_], output, copyParams);
        outputQueue_.FreeTensor(output);
    }

    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::GlobalTensor<int32_t> inputGm_;
    AscendC::GlobalTensor<int32_t> outputGm_;
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

}  // namespace

extern "C" __global__ __aicore__ void sfa_remap_sparse_indices_kernel(
    __gm__ int32_t* input,
    __gm__ int32_t* output,
    uint32_t rows,
    uint32_t topK,
    uint32_t dcpSize,
    uint32_t dcpRank,
    uint32_t interleaveSize,
    uint32_t interleaveShift,
    uint32_t dcpInterleaveShift,
    uint32_t usePowerOfTwo,
    uint32_t rowsPerCore)
{
    AscendC::TPipe pipe;
    SfaRemapSparseIndices op(&pipe);
    op.Init(input, output, rows, topK, dcpSize, dcpRank, interleaveSize,
            interleaveShift, dcpInterleaveShift, usePowerOfTwo, rowsPerCore);
    op.Process();
}

namespace vllm_ascend {

extern void sfa_remap_sparse_indices_impl(
    void* stream,
    void* input,
    void* output,
    uint32_t rows,
    uint32_t topK,
    uint32_t dcpSize,
    uint32_t dcpRank,
    uint32_t interleaveSize,
    uint32_t interleaveShift,
    uint32_t dcpInterleaveShift,
    uint32_t usePowerOfTwo,
    uint32_t vectorCoreCount)
{
    const uint32_t blockDim = rows < vectorCoreCount ? rows : vectorCoreCount;
    const uint32_t rowsPerCore = (rows + blockDim - 1) / blockDim;
    sfa_remap_sparse_indices_kernel<<<blockDim, nullptr, stream>>>(
        static_cast<int32_t*>(input), static_cast<int32_t*>(output), rows, topK,
        dcpSize, dcpRank, interleaveSize, interleaveShift,
        dcpInterleaveShift, usePowerOfTwo, rowsPerCore);
}

}  // namespace vllm_ascend
