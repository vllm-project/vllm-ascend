/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Fused LoRA: y[:, off:off+H] += scale * (x @ A) @ B.
 * The rank intermediate stays in UB (no GM round-trip, no separate expand).
 */

#include "kernel_operator.h"
#include "types.h"

template <typename scalar_t, bool USE_SEQ_LEN>
class SGMVLora {
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr uint64_t LORA_RANK_8 = 8;
    static constexpr uint64_t LORA_RANK_16 = 16;
    static constexpr uint64_t LORA_RANK_32 = 32;
    static constexpr uint64_t LORA_RANK_64 = 64;
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t SHRINK_TILE = 8192;  // H=5120 one ReduceSum; 4096 vs shrink 11776 was 1 ulp
    static constexpr int32_t NUM_BYTES_PER_REPEAT = 256;
    static constexpr int32_t NUM_BLOCKS_PER_REPEAT = 8;
    static constexpr int32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);
    static constexpr int32_t MASK_COUNT = NUM_ELEMENTS_PER_REPEAT;
    static constexpr int32_t W_IN_TILE_NUM_ELEMENTS = 8192;  // match sgmv_expand.cpp
    static constexpr int32_t Y_OUT_TILE_NUM_ELEMENTS = 4096;
    static constexpr int32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE_NUM_ELEMENTS / NUM_ELEMENTS_PER_REPEAT;
    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_16 =
        (BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_32 = (PAIR_REDUCE_NUM_REPEATS_16 + 1) / 2;

public:
    __aicore__ inline SGMVLora(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void *x, __gm__ void *weightA, __gm__ void *weightB, __gm__ void *loraIndices,
                                uint32_t loraIndicesSize, __gm__ void *seqLen, uint32_t seqLenSize, __gm__ void *y,
                                uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim,
                                uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset,
                                uint32_t outputFullDim, float scale)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        outputHiddenDim_ = outputHiddenDim;
        sliceOffset_ = sliceOffset;
        outputFullDim_ = outputFullDim;
        scale_ = scale;
        incremental_ = inputHiddenDim_ > SHRINK_TILE;
        singleALen_ = inputHiddenDim_ * maxLoRARank_;
        singleBLen_ = maxLoRARank_ * outputHiddenDim_;

        xGm_.SetGlobalBuffer((__gm__ X_T *)x);
        wAGm_.SetGlobalBuffer((__gm__ W_T *)weightA);
        wBGm_.SetGlobalBuffer((__gm__ W_T *)weightB);
        yInGm_.SetGlobalBuffer((__gm__ Y_T *)y);
        yOutGm_.SetGlobalBuffer((__gm__ Y_T *)y);
        loraIndicesGm_.SetGlobalBuffer((__gm__ int64_t *)loraIndices, loraIndicesSize);
        if constexpr (USE_SEQ_LEN) {
            seqLenGm_.SetGlobalBuffer((__gm__ int64_t *)seqLen, seqLenSize);
        }

        pipe_->InitBuffer(inQueueX_, 1, SHRINK_TILE * sizeof(X_T));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, W_IN_TILE_NUM_ELEMENTS * sizeof(W_T));
        pipe_->InitBuffer(inQueueY_, BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferX_, SHRINK_TILE * sizeof(float));
        pipe_->InitBuffer(tmpBufferW_, W_IN_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(dupBufferX_, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(inBufferY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBufferY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tBuf_, LORA_RANK_64 * sizeof(float));

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
            CopyInIndex(idx);
            if (reqLoRAIndex_ < 0) {
                continue;
            }
            reqAOffset_ = reqLoRAIndex_ * singleALen_;
            reqBOffset_ = reqLoRAIndex_ * singleBLen_;
            yOffset_ = outputFullDim_ * idx + sliceOffset_;

            if (incremental_) {
                ShrinkImpl<true>(idx);
            } else {
                ShrinkImpl<false>(idx);
            }
            ScaleT();
            DuplicateT();
            ExpandImpl();
        }
    }

private:
    __aicore__ inline void CopyInIndex(const int64_t idx)
    {
        if constexpr (USE_SEQ_LEN) {
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
        } else {
            reqLoRAIndex_ = loraIndicesGm_.GetValue(idx);
        }
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ShrinkImpl(const int64_t idx)
    {
        AscendC::LocalTensor<float> tLocal = tBuf_.Get<float>();
        if constexpr (!INCREMENTAL_MODE) {
            CopyInX(idx, 0, inputHiddenDim_);
            AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueX_.FreeTensor(xLocal);
        }
        for (int i = 0; i < maxLoRARank_; i++) {
            float acc(0);
            for (int32_t j = 0; j < inputHiddenDim_ / SHRINK_TILE; j++) {
                if constexpr (INCREMENTAL_MODE) {
                    CopyInX(idx, j);
                }
                CopyInWA(i, j);
                ShrinkDot<INCREMENTAL_MODE>(acc);
            }
            ShrinkLast<INCREMENTAL_MODE>(idx, i, acc);
            tLocal.SetValue(i, acc);
        }
    }

    __aicore__ inline void CopyInX(const int64_t idx, int32_t colIdx, int32_t numElements = SHRINK_TILE)
    {
        AscendC::LocalTensor<X_T> xLocal = inQueueX_.AllocTensor<X_T>();
        DataCopy(xLocal, xGm_[inputHiddenDim_ * idx + colIdx * SHRINK_TILE], numElements);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInWA(int32_t rowIdx, int32_t colIdx, int32_t numElements = SHRINK_TILE)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        DataCopy(wLocal, wAGm_[reqAOffset_ + rowIdx * inputHiddenDim_ + colIdx * SHRINK_TILE], numElements);
        inQueueW_.EnQue(wLocal);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ShrinkDot(float &acc, int32_t numElements = SHRINK_TILE)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferW_.Get<float>();
        if constexpr (INCREMENTAL_MODE) {
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, numElements);
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueX_.FreeTensor(xLocal);
            inQueueW_.FreeTensor(wLocal);
        } else {
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueW_.FreeTensor(wLocal);
        }
        Mul(wTmpTensor, xTmpTensor, wTmpTensor, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum<float>(wTmpTensor, wTmpTensor, wTmpTensor, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        acc += wTmpTensor.GetValue(0);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ShrinkLast(const int64_t idx, int32_t rowIdx, float &acc)
    {
        int32_t colIdx = inputHiddenDim_ / SHRINK_TILE;
        int32_t remaining = inputHiddenDim_ % SHRINK_TILE;
        if (remaining == 0) {
            return;
        }
        if constexpr (INCREMENTAL_MODE) {
            CopyInX(idx, colIdx, remaining);
        }
        CopyInWA(rowIdx, colIdx, remaining);
        ShrinkDot<INCREMENTAL_MODE>(acc, remaining);
    }

    __aicore__ inline void ScaleT()
    {
        AscendC::LocalTensor<float> tLocal = tBuf_.Get<float>();
        Muls(tLocal, tLocal, scale_, maxLoRARank_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DuplicateT()
    {
        AscendC::LocalTensor<float> tLocal = tBuf_.Get<float>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();
        for (int32_t i = 0; i < NUM_ELEMENTS_PER_REPEAT; i += maxLoRARank_) {
            for (int32_t j = 0; j < maxLoRARank_; j++) {
                xDup.SetValue(i + j, tLocal.GetValue(j));
            }
        }
    }

    __aicore__ inline void ExpandImpl()
    {
        int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
        for (int32_t i = 0; i < numStreamOut; i++) {
            AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
            Duplicate(yLocal, static_cast<float>(0), Y_OUT_TILE_NUM_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            CopyInY(i);
            for (int32_t j = 0; j < numStreamInPerOutputTile_; j++) {
                CopyInWB(i * numStreamInPerOutputTile_ + j);
                ExpandCompute(j * numOutputElementsPerInputTile_);
            }
            ScaleOutput();
            CopyOutY(i);
        }
        ExpandLast();
    }

    __aicore__ inline void ExpandLast()
    {
        int32_t remainingY = outputHiddenDim_ % Y_OUT_TILE_NUM_ELEMENTS;
        if (remainingY == 0) {
            return;
        }
        int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
        int32_t remainingW = remainingY * maxLoRARank_;
        int32_t numCompleteWTile = remainingW / W_IN_TILE_NUM_ELEMENTS;
        int32_t remainingWLast = remainingW % W_IN_TILE_NUM_ELEMENTS;

        CopyInY(numStreamOut, remainingY);
        {
            AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
            Duplicate(yLocal, static_cast<float>(0), Y_OUT_TILE_NUM_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
        }
        int32_t outputIdx = 0;
        for (outputIdx = 0; outputIdx < numCompleteWTile; outputIdx++) {
            CopyInWB(numStreamOut * numStreamInPerOutputTile_ + outputIdx);
            ExpandCompute(outputIdx * numOutputElementsPerInputTile_);
        }
        if (remainingWLast != 0) {
            CopyInWB(numStreamOut * numStreamInPerOutputTile_ + numCompleteWTile, remainingWLast);
            int32_t lastRepeatCount = remainingWLast / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduceRepeat16 =
                (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduceRepeat32 = (pairReduceRepeat16 + 1) / 2;
            ExpandCompute(outputIdx * numOutputElementsPerInputTile_, lastRepeatCount, pairReduceRepeat16,
                          pairReduceRepeat32);
        }
        ScaleOutput(remainingY);
        CopyOutY(numStreamOut, remainingY);
    }

    __aicore__ inline void CopyInY(int32_t progress, int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.AllocTensor<Y_T>();
        DataCopy(yInLocal, yInGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], numElements);
        inQueueY_.EnQue(yInLocal);
    }

    __aicore__ inline void CopyInWB(int32_t progress, int32_t numElements = W_IN_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        DataCopy(wLocal, wBGm_[reqBOffset_ + progress * W_IN_TILE_NUM_ELEMENTS], numElements);
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void ExpandCompute(int32_t progress, int32_t blockReduceRepeatCount = BLOCK_REDUCE_NUM_REPEATS,
                                         int32_t pairReduceRepeat16 = PAIR_REDUCE_NUM_REPEATS_16,
                                         int32_t pairReduceRepeat32 = PAIR_REDUCE_NUM_REPEATS_32)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferW_.Get<float>();

        Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, MASK_COUNT, blockReduceRepeatCount, castParams_);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueW_.FreeTensor(wLocal);

        Mul(wTmpTensor, xDup, wTmpTensor, MASK_COUNT, blockReduceRepeatCount, dotProductParams_);
        AscendC::PipeBarrier<PIPE_V>();

        if (maxLoRARank_ == LORA_RANK_8) {
            BlockReduceSum(yLocal[progress], wTmpTensor, blockReduceRepeatCount, MASK_COUNT, reduceSumParams_.dstRepStride,
                           reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (maxLoRARank_ == LORA_RANK_16) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT, reduceSumParams_.dstRepStride,
                           reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat16, MASK_COUNT, reduceSumParams_.dstRepStride,
                          reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (maxLoRARank_ == LORA_RANK_32) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT, reduceSumParams_.dstRepStride,
                           reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(wTmpTensor, wTmpTensor, pairReduceRepeat16, MASK_COUNT, reduceSumParams_.dstRepStride,
                          reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            PairReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat32, MASK_COUNT, reduceSumParams_.dstRepStride,
                          reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        } else if (maxLoRARank_ == LORA_RANK_64) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT, reduceSumParams_.dstRepStride,
                           reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
            BlockReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat16, MASK_COUNT, reduceSumParams_.dstRepStride,
                           reduceSumParams_.srcBlkStride, reduceSumParams_.srcRepStride);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ScaleOutput(int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.DeQue<Y_T>();
        AscendC::LocalTensor<float> yInLocalFP32 = inBufferY_.Get<float>();
        Cast(yInLocalFP32, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueY_.FreeTensor(yInLocal);

        Add(yLocal, yLocal, yInLocalFP32, numElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueY_.EnQue<Y_T>(yOutLocal);
    }

    __aicore__ inline void CopyOutY(int32_t progress, int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.DeQue<Y_T>();
        DataCopy(yOutGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], yOutLocal, numElements);
        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueW_, inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferX_, tmpBufferW_, dupBufferX_, inBufferY_, tmpBufferY_, tBuf_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wAGm_, wBGm_;
    AscendC::GlobalTensor<Y_T> yInGm_, yOutGm_;
    AscendC::GlobalTensor<int64_t> loraIndicesGm_;
    AscendC::GlobalTensor<int64_t> seqLenGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t outputHiddenDim_;
    uint32_t sliceOffset_;
    uint32_t outputFullDim_;
    float scale_;
    uint32_t singleALen_;
    uint32_t singleBLen_;
    int64_t reqLoRAIndex_;
    uint64_t reqAOffset_;
    uint64_t reqBOffset_;
    uint32_t numOutputElementsPerInputTile_;
    uint32_t numStreamInPerOutputTile_;
    uint64_t yOffset_;
    bool incremental_;

    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::UnaryRepeatParams reduceSumParams_ = {1, 1, 1, 8};
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};
};

#define SGMV_LORA_DECLARE(TYPE)                                                                                        \
    extern "C" __global__ __aicore__ void sgmv_lora_##TYPE(                                                            \
        __gm__ void* x, __gm__ void* weightA, __gm__ void* weightB, __gm__ void* loraIndices,                          \
        uint32_t loraIndicesSize, __gm__ void* seqLen, uint32_t seqLenSize, __gm__ void* y, uint32_t batchSize,        \
        uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, uint32_t outputHiddenDim,            \
        uint32_t sliceOffset, uint32_t outputFullDim, float scale)                                                     \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SGMVLora<TYPE, true> op(&pipe);                                                                                \
        op.Init(x, weightA, weightB, loraIndices, loraIndicesSize, seqLen, seqLenSize, y, batchSize, numTokensPerCore, \
                inputHiddenDim, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim, scale);                      \
        op.Process();                                                                                                  \
    }

#define BGMV_LORA_DECLARE(TYPE)                                                                                        \
    extern "C" __global__ __aicore__ void bgmv_lora_##TYPE(                                                            \
        __gm__ void* x, __gm__ void* weightA, __gm__ void* weightB, __gm__ void* indices, uint32_t indicesSize,        \
        __gm__ void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank,  \
        uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim, float scale)                           \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SGMVLora<TYPE, false> op(&pipe);                                                                               \
        op.Init(x, weightA, weightB, indices, indicesSize, nullptr, 0, y, batchSize, numTokensPerCore, inputHiddenDim, \
                maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim, scale);                                      \
        op.Process();                                                                                                  \
    }

SGMV_LORA_DECLARE(half)
BGMV_LORA_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
SGMV_LORA_DECLARE(bfloat16_t)
BGMV_LORA_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void sgmv_lora_impl(AscendType type, void *stream, void *x, void *weightA, void *weightB, void *loraIndices,
                           uint32_t loraIndicesSize, void *seqLen, uint32_t seqLenSize, void *y, uint32_t batchSize,
                           uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank,
                           uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim, float scale)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == AscendType::FP16) {
        sgmv_lora_half<<<blockDim, nullptr, stream>>>(x, weightA, weightB, loraIndices, loraIndicesSize, seqLen,
                                                      seqLenSize, y, batchSize, numTokensPerCore, inputHiddenDim,
                                                      maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim, scale);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        sgmv_lora_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weightA, weightB, loraIndices, loraIndicesSize, seqLen,
                                                            seqLenSize, y, batchSize, numTokensPerCore, inputHiddenDim,
                                                            maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim,
                                                            scale);
#endif
    }
}

extern void bgmv_lora_impl(AscendType type, void *stream, void *x, void *weightA, void *weightB, void *indices,
                           uint32_t indicesSize, void *y, uint32_t batchSize, uint32_t numTokensPerCore,
                           uint32_t inputHiddenDim, uint32_t maxLoRARank, uint32_t outputHiddenDim,
                           uint32_t sliceOffset, uint32_t outputFullDim, float scale)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == AscendType::FP16) {
        bgmv_lora_half<<<blockDim, nullptr, stream>>>(x, weightA, weightB, indices, indicesSize, y, batchSize,
                                                      numTokensPerCore, inputHiddenDim, maxLoRARank, outputHiddenDim,
                                                      sliceOffset, outputFullDim, scale);
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        bgmv_lora_bfloat16_t<<<blockDim, nullptr, stream>>>(x, weightA, weightB, indices, indicesSize, y, batchSize,
                                                            numTokensPerCore, inputHiddenDim, maxLoRARank,
                                                            outputHiddenDim, sliceOffset, outputFullDim, scale);
#endif
    }
}
} // namespace vllm_ascend
