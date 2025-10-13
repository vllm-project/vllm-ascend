/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "types.h"

template <typename scalar_t>
class SGMVExpand {
public:
    using X_T = float;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr uint64_t LORA_RANK_8 = 8;
    static constexpr uint64_t LORA_RANK_16 = 16;
    static constexpr uint64_t LORA_RANK_32 = 32;
    static constexpr uint64_t LORA_RANK_64 = 64;
    static constexpr uint64_t SUPPORTED_RANKS[] = {LORA_RANK_8, LORA_RANK_16, LORA_RANK_32, LORA_RANK_64};
    static constexpr int32_t BUFFER_NUM = 2;

    // The vector unit reads 8 blocks (32 bytes each and 256 bytes in total) of contiguous data each time.
    static constexpr int32_t NUM_BYTES_PER_REPEAT = 256;
    static constexpr int32_t NUM_BLOCKS_PER_REPEAT = 8;
    // The maximum number of elements in a single iteration is 256 / sizeof(intermediate data type).
    static constexpr int32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);
    // Mask is used to control the elements that participate in computation in each iteration.
    static constexpr int32_t MASK_COUNT = NUM_BYTES_PER_REPEAT / sizeof(float);
    // Refer to numOutputElementsPerInputTile_ initialization for the constraints on the following constants.
    static constexpr int32_t W_IN_TILE_NUM_ELEMENTS = 8192;
    static constexpr int32_t Y_OUT_TILE_NUM_ELEMENTS = 4096;
    static constexpr int32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE_NUM_ELEMENTS / NUM_ELEMENTS_PER_REPEAT;
    // BlockReduceSum would generate(BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT)floats. 
    // So need to read them all and apply PairReduceSum
    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_16 = 
        (BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
    // The second PairReduceSum for rank=32, needs half of the repetition that happened for rank=16.
    // Same for rank=64, we do not support ranks greater than 64.
    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_32 = (PAIR_REDUCE_NUM_REPEATS_16 + 1) / 2;

public:
    __aicore__ inline SGMVExpand(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* weight, __gm__ void* loraIndices, uint32_t loraIndicesSize,
                                __gm__ void* seqLen, uint32_t seqLenSize, __gm__ void* yIn, __gm__ void* yOut,
                                uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank,
                                uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        maxLoRARank_ = maxLoRARank;
        outputHiddenDim_ = outputHiddenDim;
        sliceOffset_ = sliceOffset;
        outputFullDim_ = outputFullDim;
        singleLoRAWeightLen_ = maxLoRARank_ * outputHiddenDim_;

        xGm_.SetGlobalBuffer((__gm__ X_T *)x);
        wGm_.SetGlobalBuffer((__gm__ W_T *)weight);
        yInGm_.SetGlobalBuffer((__gm__ Y_T *)yIn);
        yOutGm_.SetGlobalBuffer((__gm__ Y_T *)yOut);
        loraIndicesGm_.SetGlobalBuffer((__gm__ int64_t *)loraIndices, loraIndicesSize);
        seqLenGm_.SetGlobalBuffer((__gm__ int64_t *)seqLen, seqLenSize);

        pipe_->InitBuffer(inQueueX_, 1, NUM_ELEMENTS_PER_REPEAT * sizeof(X_T));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, W_IN_TILE_NUM_ELEMENTS * sizeof(W_T));
        pipe_->InitBuffer(inQueueY_, BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));

        pipe_->InitBuffer(dupBufferX_, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(tmpBufferW_, W_IN_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(inBufferY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBufferY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));

        // Each compute iteration would generate not one, but several output elements.
        // Therefore, the following variable would determine how many output elements are calculated in each iteration.
        numOutputElementsPerInputTile_ = BLOCK_REDUCE_NUM_REPEATS * (NUM_ELEMENTS_PER_REPEAT / maxLoRARank_);
        numStreamInPerOutputTile_ = Y_OUT_TILE_NUM_ELEMENTS / numOutputElementsPerInputTile_;

    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIdx = blockIdx * numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        if (endIdx > batchSize_) {
            endIdx = batchSize_;
        }
        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            yOffset_ = outputFullDim_ * idx + sliceOffset_;

            // Set up LoRA index
            CopyInIndex(idx);
            if (reqLoRAIndex_ < 0) {
                continue;
            }
            reqLoRAWeightOffset_ = reqLoRAIndex_ * singleLoRAWeightLen_;

            CopyInX(idx);
            int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
            for (int32_t i = 0; i < numStreamOut; i++) {
                CopyInY(i);
                for (int32_t j = 0; j < numStreamInPerOutputTile_; j++) {
                    CopyInW(i * numStreamInPerOutputTile_ + j);
                    Compute(j * numOutputElementsPerInputTile_);
                }
                ScaleOutput();
                CopyOut(i);
            }
            ComputeLastIteration();
        }
    }

private:
    __aicore__ inline void CopyInIndex(const int64_t idx)
    {
        // Look up the LoRA index
        int64_t weightIdx = idx;
        uint64_t i = 0;
        for (; i < seqLenGm_.GetSize(); i++) {
            int64_t repeatValue = seqLenGm_.GetValue(i);
            if (weightIdx >= repeatValue) {
                weightIdx -= repeatValue;
                continue;
            }
            break;
        }
        reqLoRAIndex_ = (i < seqLenGm_.GetSize()) ? loraIndicesGm_.GetValue(i) : -1;
    }

    __aicore__ inline void ComputeLastIteration()
    {
        int32_t remainingY = outputHiddenDim_ % Y_OUT_TILE_NUM_ELEMENTS;
        if (remainingY == 0) {
            return;
        }
        int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
        int32_t remainingW = remainingY * maxLoRARank_;
        int32_t numCompleteWTileInForLastIteration = remainingW / W_IN_TILE_NUM_ELEMENTS;
        int32_t remainingWForLastRepeat = remainingW % W_IN_TILE_NUM_ELEMENTS;

        CopyInY(numStreamOut, remainingY);

        int32_t outputIdx = 0;
        for (outputIdx = 0; outputIdx < numCompleteWTileInForLastIteration; outputIdx++) {
            CopyInW(numStreamOut * numStreamInPerOutputTile_ + outputIdx);
            Compute(outputIdx * numOutputElementsPerInputTile_);
        }

        if (remainingWForLastRepeat != 0) {
            CopyInW(numStreamOut * numStreamInPerOutputTile_ + numCompleteWTileInForLastIteration,
                    remainingWForLastRepeat);
            int32_t lastRepeatCount = remainingWForLastRepeat / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduceRepeat16 = 
                (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduceRepeat32 = (pairReduceRepeat16 + 1) / 2;
            int32_t lastComputeOutputElement = outputIdx * numOutputElementsPerInputTile_;
            Compute(lastComputeOutputElement, lastRepeatCount, pairReduceRepeat16, pairReduceRepeat32);
        }

        ScaleOutput(remainingY);
        CopyOut(numStreamOut, remainingY);
    }

    __aicore__ inline void CopyInX(const int64_t idx)
    {
        AscendC::LocalTensor<X_T> xLocal = inQueueX_.AllocTensor<X_T>();
        if constexpr (std::is_same_v<X_T, float>) {
            DataCopy(xLocal, xGm_[maxLoRARank_ * idx], maxLoRARank_);
        } else {
            uint16_t blockLen = static_cast<uint16_t>(maxLoRARank_ * sizeof(X_T));
            DataCopyPad(xLocal, xGm_[maxLoRARank_ * idx], {1, blockLen, 0, 0}, {});
        }
        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.DeQue<X_T>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();

        // As we are generating multiple output elements with one API invocation,
        // we need to duplicate the X vector multiple times to fill one NUM_BYTES_PER_REPEAT
        if constexpr (std::is_same_v<X_T, float>) {
            for (int32_t i = 0; i < NUM_ELEMENTS_PER_REPEAT; i += maxLoRARank_) {
                for (int32_t j = 0; j < maxLoRARank_; j++) {
                    float entry = xLocal.GetValue(j);
                    xDup.SetValue(i + j, entry);
                }
            }
        } else {
            Cast(xDup, xLocal, AscendC::RoundMode::CAST_NONE, maxLoRARank_);
            pipe_barrier(PIPE_V);

            for (int32_t i = maxLoRARank_; i < NUM_ELEMENTS_PER_REPEAT; i += maxLoRARank_) {
                for (int32_t j = 0; j < maxLoRARank_; j++) {
                    float entry = xDup.GetValue(j);
                    xDup.SetValue(i + j, entry);
                }
            }
        }
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInY(int32_t progress, int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.AllocTensor<Y_T>();
        DataCopy(yInLocal, yInGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], numElements);
        inQueueY_.EnQue(yInLocal);
    }

    __aicore__ inline void CopyInW(int32_t progress, int32_t numElements = W_IN_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        DataCopy(wLocal, wGm_[reqLoRAWeightOffset_ + progress * W_IN_TILE_NUM_ELEMENTS], numElements);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void ScaleOutput(int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.DeQue<Y_T>();
        AscendC::LocalTensor<float> yInLocalFP32 = inBufferY_.Get<float>();
        Cast(yInLocalFP32, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
        pipe_barrier(PIPE_V);
        inQueueY_.FreeTensor(yInLocal);

        Add(yLocal, yLocal, yInLocalFP32, numElements);
        pipe_barrier(PIPE_V);

        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        pipe_barrier(PIPE_V);

        outQueueY_.EnQue<Y_T>(yOutLocal);
    }

    __aicore__ inline void Compute(int32_t progress,
                                   int32_t blockReduceRepeatCount=BLOCK_REDUCE_NUM_REPEATS,
                                   int32_t pairReduceRepeat16=PAIR_REDUCE_NUM_REPEATS_16,
                                   int32_t pairReduceRepeat32=PAIR_REDUCE_NUM_REPEATS_32)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferW_.Get<float>();

        Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, MASK_COUNT, blockReduceRepeatCount, castParams_);
        pipe_barrier(PIPE_V);
        inQueueW_.FreeTensor(wLocal);

        Mul(wTmpTensor, xDup, wTmpTensor, MASK_COUNT, blockReduceRepeatCount, dotProductParams_);
        pipe_barrier(PIPE_V);

        if (maxLoRARank_ == LORA_RANK_8) {
            BlockReduceSum(yLocal[progress], wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
        } else if (maxLoRARank_ == LORA_RANK_16) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
            PairReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat16, MASK_COUNT,
                          reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
        } else if (maxLoRARank_ == LORA_RANK_32) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
            PairReduceSum(wTmpTensor, wTmpTensor, pairReduceRepeat16, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
            PairReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat32, MASK_COUNT,
                          reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
        } else if (maxLoRARank_ == LORA_RANK_64) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
            BlockReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat16, MASK_COUNT,
                          reduceSumParams_.dstRepStride, reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.DeQue<Y_T>();
        DataCopy(yOutGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], yOutLocal, numElements);
        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY_, inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_, dupBufferX_, inBufferY_, tmpBufferY_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<Y_T> yInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    AscendC::GlobalTensor<int64_t> loraIndicesGm_;
    AscendC::GlobalTensor<int64_t> seqLenGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t maxLoRARank_;
    uint32_t outputHiddenDim_;
    uint32_t sliceOffset_;
    uint32_t outputFullDim_;
    uint32_t singleLoRAWeightLen_;
    int64_t reqLoRAIndex_;
    uint64_t reqLoRAWeightOffset_;
    uint32_t numOutputElementsPerInputTile_;
    uint32_t numStreamInPerOutputTile_;
    uint64_t yOffset_;

    // The block stride is set to 1, and 8 blocks in the same repeat are processed continuously.
    // The repeat stride is 8, so the vector unit reads 8 consecutive blocks in the first repeat,
    // reads next 8 consecutive blocks in the second repeat.
    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};

    // For each repeat in BlockReduceSum and PairReduceSum we should move forward only one block,
    // so we set dstRepStride = 1
    AscendC::UnaryRepeatParams reduceSumParams_ = {1, 1, 1, 8};

    // When the repeat stride is 0, the vector unit repeatedly reads and computes the first 8 consecutive blocks.
    // For xDup we repeatedly use it, so we set src0RepStride = 0
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};

};

#define SGMV_EXPAND_TYPE_DECLARE(TYPE)                                                                                 \
    extern "C" __global__ __aicore__ void sgmv_expand_##TYPE(__gm__ void* x, __gm__ void* weight,                      \
                                                             __gm__ void* loraIndices, uint32_t loraIndicesSize,       \
                                                             __gm__ void* seqLen, uint32_t seqLenSize,                 \
                                                             __gm__ void* yIn,  __gm__ void* yOut,                     \
                                                             uint32_t batchSize, uint32_t numTokensPerCore,            \
                                                             uint32_t maxLoRARank, uint32_t outputHiddenDim,           \
                                                             uint32_t sliceOffset, uint32_t outputFullDim)             \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SGMVExpand<TYPE> op(&pipe);                                                                                    \
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize,                                           \
                yIn, yOut, batchSize, numTokensPerCore, maxLoRARank,                                                   \
                outputHiddenDim, sliceOffset, outputFullDim);                                                          \
        op.Process();                                                                                                  \
    }

// declare all dtype kernel
SGMV_EXPAND_TYPE_DECLARE(half)
#if (__CCE_AICORE__ >= 220)
    SGMV_EXPAND_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void sgmv_expand_impl(AscendType type, void* stream, void* x, void* weight, 
                             void* loraIndices, uint32_t loraIndicesSize,
                             void* seqLen, uint32_t seqLenSize,
                             void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank,
                             uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == AscendType::FP16) {
        sgmv_expand_half<<<blockDim, nullptr, stream>>>(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, 
                                                        yIn, yOut, batchSize,
                                                        numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, 
                                                        outputFullDim);
    } else if (type == AscendType::BF16) {
        #if (__CCE_AICORE__ >= 220)
            sgmv_expand_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weight, loraIndices, loraIndicesSize, 
                                                                  seqLen, seqLenSize, yIn, yOut, batchSize,
                                                                  numTokensPerCore, maxLoRARank, outputHiddenDim,
                                                                  sliceOffset, outputFullDim);
        #endif
    } else {
        return;
    }
}

} // namespace vllm_ascend