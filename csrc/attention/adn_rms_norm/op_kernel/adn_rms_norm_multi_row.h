#ifndef ADN_RMS_NORM_MULTI_ROW_H
#define ADN_RMS_NORM_MULTI_ROW_H

#include "adn_rms_norm_common.h"

namespace AdnRmsNorm {
using namespace AscendC;

template <uint32_t H, uint32_t MAX_ROWS_PER_TILE>
class MultiRowKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR gamma,
        GM_ADDR y,
        const AdnRmsNormTilingData* tiling)
    {
        numRows_ = tiling->numRows;
        rowsPerTile_ = tiling->rowsPerTile;
        epsilon_ = tiling->epsilon;
        invHiddenSize_ = tiling->invHiddenSize;

        const uint32_t blockIdx = GetBlockIdx();
        const uint64_t baseRows = tiling->baseRowsPerCore;
        const uint32_t extraCores = tiling->extraRowCoreCount;
        localRows_ = baseRows + (blockIdx < extraCores ? 1 : 0);
        startRow_ = blockIdx * baseRows + (blockIdx < extraCores ? blockIdx : extraCores);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), numRows_ * H);
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gamma), H);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), numRows_ * H);

        pipe_.InitBuffer(gammaQueue_, 1, H * sizeof(half));
        pipe_.InitBuffer(inputQueue_, 2, MAX_ROWS_PER_TILE * H * sizeof(half));
        pipe_.InitBuffer(outputQueue_, 2, MAX_ROWS_PER_TILE * H * sizeof(half));
        pipe_.InitBuffer(gammaFloatBuf_, H * sizeof(float));
        pipe_.InitBuffer(xFloatBuf_, MAX_ROWS_PER_TILE * H * sizeof(float));
        pipe_.InitBuffer(squareBuf_, MAX_ROWS_PER_TILE * H * sizeof(float));
        pipe_.InitBuffer(reduceTmpBuf_, MAX_ROWS_PER_TILE * FP32_VECTOR_SIZE * sizeof(float));
        pipe_.InitBuffer(
            rowSumsBuf_,
            ((MAX_ROWS_PER_TILE + FP32_BLOCK_SIZE - 1) / FP32_BLOCK_SIZE) *
                FP32_BLOCK_SIZE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (localRows_ == 0) {
            return;
        }
        CopyGamma();

        const uint64_t tileCount = (localRows_ + rowsPerTile_ - 1) / rowsPerTile_;
        CopyIn(0);
        for (uint64_t tile = 0; tile < tileCount; ++tile) {
            if (tile + 1 < tileCount) {
                CopyIn(tile + 1);
            }
            Compute(tile);
            CopyOut(tile);
        }
    }

private:
    __aicore__ inline uint32_t TileRows(uint64_t tile) const
    {
        const uint64_t consumed = tile * rowsPerTile_;
        const uint64_t remaining = localRows_ - consumed;
        return static_cast<uint32_t>(remaining < rowsPerTile_ ? remaining : rowsPerTile_);
    }

    __aicore__ inline uint64_t TileOffset(uint64_t tile) const
    {
        return (startRow_ + tile * rowsPerTile_) * H;
    }

    __aicore__ inline void CopyGamma()
    {
        LocalTensor<half> gammaHalf = gammaQueue_.AllocTensor<half>();
        DataCopy(gammaHalf, gammaGm_, H);
        gammaQueue_.EnQue(gammaHalf);
        gammaHalf = gammaQueue_.DeQue<half>();
        LocalTensor<float> gammaFloat = gammaFloatBuf_.Get<float>();
        Cast(gammaFloat, gammaHalf, RoundMode::CAST_NONE, H);
        gammaQueue_.FreeTensor(gammaHalf);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyIn(uint64_t tile)
    {
        const uint32_t rows = TileRows(tile);
        LocalTensor<half> input = inputQueue_.AllocTensor<half>();
        DataCopy(input, xGm_[TileOffset(tile)], rows * H);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(uint64_t tile)
    {
        const uint32_t rows = TileRows(tile);
        const uint32_t elements = rows * H;
        LocalTensor<half> input = inputQueue_.DeQue<half>();
        LocalTensor<half> output = outputQueue_.AllocTensor<half>();
        LocalTensor<float> xFloat = xFloatBuf_.Get<float>();
        LocalTensor<float> square = squareBuf_.Get<float>();
        LocalTensor<float> reduceTmp = reduceTmpBuf_.Get<float>();
        LocalTensor<float> rowSums = rowSumsBuf_.Get<float>();
        LocalTensor<float> gammaFloat = gammaFloatBuf_.Get<float>();

        Cast(xFloat, input, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        Mul(square, xFloat, xFloat, elements);
        PipeBarrier<PIPE_V>();

        Duplicate(
            rowSums,
            0.0f,
            ((rows + FP32_BLOCK_SIZE - 1) / FP32_BLOCK_SIZE) * FP32_BLOCK_SIZE);
        PipeBarrier<PIPE_V>();
        ReduceRows<H>(rowSums, square, reduceTmp, rows);
        Muls(rowSums, rowSums, invHiddenSize_, rows);
        PipeBarrier<PIPE_V>();
        Adds(rowSums, rowSums, epsilon_, rows);
        PipeBarrier<PIPE_V>();
        Sqrt(rowSums, rowSums, rows);
        PipeBarrier<PIPE_V>();
        Duplicate(reduceTmp, 1.0f, rows);
        PipeBarrier<PIPE_V>();
        Div(rowSums, reduceTmp, rowSums, rows);
        PipeBarrier<PIPE_V>();

        ApplyRowScales<H>(xFloat, xFloat, rowSums, reduceTmp, rows);
        ApplyGamma<H>(xFloat, gammaFloat, rows);
        Cast(output, xFloat, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(uint64_t tile)
    {
        const uint32_t rows = TileRows(tile);
        LocalTensor<half> output = outputQueue_.DeQue<half>();
        DataCopy(yGm_[TileOffset(tile)], output, rows * H);
        outputQueue_.FreeTensor(output);
    }

    GlobalTensor<half> xGm_;
    GlobalTensor<half> gammaGm_;
    GlobalTensor<half> yGm_;
    uint64_t numRows_ = 0;
    uint64_t startRow_ = 0;
    uint64_t localRows_ = 0;
    uint32_t rowsPerTile_ = 1;
    float epsilon_ = 0.0f;
    float invHiddenSize_ = 0.0f;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> gammaFloatBuf_;
    TBuf<TPosition::VECCALC> xFloatBuf_;
    TBuf<TPosition::VECCALC> squareBuf_;
    TBuf<TPosition::VECCALC> reduceTmpBuf_;
    TBuf<TPosition::VECCALC> rowSumsBuf_;
};

}  // namespace AdnRmsNorm

#endif  // ADN_RMS_NORM_MULTI_ROW_H
