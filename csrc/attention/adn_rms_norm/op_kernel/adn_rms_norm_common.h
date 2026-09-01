#ifndef ADN_RMS_NORM_COMMON_H
#define ADN_RMS_NORM_COMMON_H

#include "kernel_operator.h"

namespace AdnRmsNorm {
using namespace AscendC;

constexpr uint32_t FP32_VECTOR_SIZE = 64;
constexpr uint32_t FP32_BLOCK_SIZE = 8;
constexpr uint32_t MAX_REPEAT = 255;

__aicore__ inline void ReduceRowsSmallH(
    const LocalTensor<float>& rowSums,
    const LocalTensor<float>& square,
    const LocalTensor<float>& reduceTmp,
    uint32_t rows,
    uint32_t hiddenSize)
{
    Duplicate(reduceTmp, 0.0f, rows * FP32_VECTOR_SIZE);
    PipeBarrier<PIPE_V>();

    const uint8_t sourceRowStride = static_cast<uint8_t>(hiddenSize / FP32_BLOCK_SIZE);
    for (uint32_t col = 0; col < hiddenSize; col += FP32_VECTOR_SIZE) {
        Add(
            reduceTmp,
            square[col],
            reduceTmp,
            FP32_VECTOR_SIZE,
            static_cast<uint8_t>(rows),
            {1, 1, 1, FP32_BLOCK_SIZE, sourceRowStride, FP32_BLOCK_SIZE});
        PipeBarrier<PIPE_V>();
    }

    AscendCUtils::SetMask<float>(FP32_VECTOR_SIZE);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, false>(
            rowSums,
            reduceTmp,
            FP32_VECTOR_SIZE,
            static_cast<uint8_t>(rows),
            1,
            1,
            FP32_BLOCK_SIZE);
    }
#else
    WholeReduceSum<float, false>(
        rowSums,
        reduceTmp,
        FP32_VECTOR_SIZE,
        static_cast<uint8_t>(rows),
        1,
        1,
        FP32_BLOCK_SIZE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceOneRowLargeH(
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    uint32_t count)
{
    uint32_t bodyCount = 1;
    while ((bodyCount << 1) <= count) {
        bodyCount <<= 1;
    }
    const uint32_t tailCount = count - bodyCount;
    if (tailCount > 0) {
        Add(src, src, src[bodyCount], tailCount);
        PipeBarrier<PIPE_V>();
    }
    while (bodyCount > FP32_VECTOR_SIZE) {
        bodyCount >>= 1;
        Add(src, src, src[bodyCount], bodyCount);
        PipeBarrier<PIPE_V>();
    }

    AscendCUtils::SetMask<float>(FP32_VECTOR_SIZE);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, false>(dst, src, FP32_VECTOR_SIZE, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(dst, src, FP32_VECTOR_SIZE, 1, 1, 1, FP32_BLOCK_SIZE);
#endif
    PipeBarrier<PIPE_V>();
}

template <uint32_t H>
__aicore__ inline void ReduceRows(
    const LocalTensor<float>& rowSums,
    const LocalTensor<float>& square,
    const LocalTensor<float>& reduceTmp,
    uint32_t rows)
{
    if constexpr (H <= 256) {
        ReduceRowsSmallH(rowSums, square, reduceTmp, rows, H);
    } else {
        for (uint32_t row = 0; row < rows; ++row) {
            ReduceOneRowLargeH(rowSums[row], square[row * H], H);
        }
    }
}

template <uint32_t H>
__aicore__ inline void ApplyRowScales(
    const LocalTensor<float>& dst,
    const LocalTensor<float>& src,
    const LocalTensor<float>& scales,
    const LocalTensor<float>& broadcastScratch,
    uint32_t rows)
{
    const uint32_t scaleBlocks = (rows + FP32_BLOCK_SIZE - 1) / FP32_BLOCK_SIZE;
    Brcb(broadcastScratch, scales, scaleBlocks, {1, FP32_BLOCK_SIZE});
    PipeBarrier<PIPE_V>();

    if constexpr (H <= 256) {
        const BinaryRepeatParams params{
            1,
            1,
            0,
            static_cast<uint8_t>(H / FP32_BLOCK_SIZE),
            static_cast<uint8_t>(H / FP32_BLOCK_SIZE),
            1};
        for (uint32_t col = 0; col < H; col += FP32_VECTOR_SIZE) {
            Mul(
                dst[col],
                src[col],
                broadcastScratch,
                FP32_VECTOR_SIZE,
                static_cast<uint8_t>(rows),
                params);
        }
    } else {
        const BinaryRepeatParams params{
            1,
            1,
            0,
            FP32_BLOCK_SIZE,
            FP32_BLOCK_SIZE,
            0};
        for (uint32_t row = 0; row < rows; ++row) {
            Mul(
                dst[row * H],
                src[row * H],
                broadcastScratch[row * FP32_BLOCK_SIZE],
                FP32_VECTOR_SIZE,
                static_cast<uint8_t>(H / FP32_VECTOR_SIZE),
                params);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t H>
__aicore__ inline void ApplyGamma(
    const LocalTensor<float>& data,
    const LocalTensor<float>& gamma,
    uint32_t rows)
{
    if constexpr (H <= 256) {
        const BinaryRepeatParams params{
            1,
            1,
            1,
            static_cast<uint8_t>(H / FP32_BLOCK_SIZE),
            static_cast<uint8_t>(H / FP32_BLOCK_SIZE),
            0};
        for (uint32_t col = 0; col < H; col += FP32_VECTOR_SIZE) {
            Mul(
                data[col],
                data[col],
                gamma[col],
                FP32_VECTOR_SIZE,
                static_cast<uint8_t>(rows),
                params);
        }
    } else {
        for (uint32_t row = 0; row < rows; ++row) {
            Mul(data[row * H], data[row * H], gamma, H);
        }
    }
    PipeBarrier<PIPE_V>();
}

}  // namespace AdnRmsNorm

#endif  // ADN_RMS_NORM_COMMON_H
