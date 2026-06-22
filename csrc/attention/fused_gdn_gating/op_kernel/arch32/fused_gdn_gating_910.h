/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
 
#ifndef FUSED_GDN_GATING_910_H
#define FUSED_GDN_GATING_910_H

#include <type_traits>

#include "kernel_operator.h"
#include "../fused_gdn_gating_tiling_data.h"

namespace FusedGdnGating {

using namespace AscendC;

constexpr uint32_t BYTES_PER_BLOCK = 32;
constexpr uint32_t BF16_PER_BLOCK  = BYTES_PER_BLOCK / sizeof(int16_t);
constexpr uint32_t FP32_PER_BLOCK  = BYTES_PER_BLOCK / sizeof(float);
constexpr uint32_t MASK_ALIGN_ELEMS = 64;

constexpr uint32_t DMA_ALIGN_ELEMS = BYTES_PER_BLOCK / sizeof(int16_t);

template <typename T>
__aicore__ inline T CeilDiv(T a, T b) { return (a + b - 1) / b; }

template <typename T>
__aicore__ inline T AlignUp(T a, T b) { return CeilDiv(a, b) * b; }

template <typename InDtype, typename ParamDtype>
class KernelFusedGdnGating {
public:
    __aicore__ inline KernelFusedGdnGating() {}

    __aicore__ inline void Init(GM_ADDR aLogGm, GM_ADDR aGm, GM_ADDR bGm, GM_ADDR dtBiasGm,
                                GM_ADDR gGm, GM_ADDR betaOutputGm,
                                const FusedGdnGatingTilingData *tiling, TPipe *pipe)
    {
        pipe_ = pipe;
        numHeads_ = tiling->numHeads;
        numBatches_ = tiling->numBatches;
        rowsPerIter_ = tiling->rowsPerIter;
        useBulkDma_ = (tiling->useBulkDma != 0);
        beta_ = tiling->beta;
        threshold_ = tiling->threshold;

        alignedHeadsHalf_  = AlignUp<uint32_t>(numHeads_, MASK_ALIGN_ELEMS);
        alignedHeadsFloat_ = AlignUp<uint32_t>(numHeads_, MASK_ALIGN_ELEMS);
        alignedHeadsMask_ = alignedHeadsFloat_;
        constexpr uint32_t paramAlignElems = BYTES_PER_BLOCK / sizeof(ParamDtype);
        alignedHeadsParam_ = AlignUp<uint32_t>(numHeads_, paramAlignElems);

        aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ParamDtype *>(aLogGm), numHeads_);
        dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ParamDtype *>(dtBiasGm), numHeads_);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm),
                             static_cast<uint64_t>(numBatches_) * numHeads_);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(bGm),
                             static_cast<uint64_t>(numBatches_) * numHeads_);
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gGm),
                             static_cast<uint64_t>(numBatches_) * numHeads_);
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(betaOutputGm),
                                static_cast<uint64_t>(numBatches_) * numHeads_);

        pipe_->InitBuffer(aInQue_, 1, rowsPerIter_ * alignedHeadsHalf_ * sizeof(InDtype));
        pipe_->InitBuffer(bInQue_, 1, rowsPerIter_ * alignedHeadsHalf_ * sizeof(InDtype));
        pipe_->InitBuffer(gOutQue_, 1, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
        pipe_->InitBuffer(betaOutQue_, 1, rowsPerIter_ * alignedHeadsHalf_ * sizeof(InDtype));

        pipe_->InitBuffer(aLogInQue_, 1, 1 * alignedHeadsParam_ * sizeof(ParamDtype));
        pipe_->InitBuffer(dtBiasInQue_, 1, 1 * alignedHeadsParam_ * sizeof(ParamDtype));
        pipe_->InitBuffer(negExpInQue_, 1, 1 * alignedHeadsFloat_ * sizeof(float));
        pipe_->InitBuffer(dtBiasPreloadQue_, 1, 1 * alignedHeadsFloat_ * sizeof(float));

        if (rowsPerIter_ > 1) {
            pipe_->InitBuffer(dtBiasMultiBuf_, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
            pipe_->InitBuffer(negExpMultiBuf_, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
        }

        pipe_->InitBuffer(xBuf_, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
        pipe_->InitBuffer(betaXBuf_, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
        pipe_->InitBuffer(softplusTmpBuf_, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
        pipe_->InitBuffer(thresholdMaskBuf_, rowsPerIter_ * alignedHeadsMask_ * sizeof(uint8_t));
        pipe_->InitBuffer(betaFp32Buf_, rowsPerIter_ * alignedHeadsFloat_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        PreloadConstants();

        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        if (blockNum == 0) { blockNum = 1; }

        uint32_t totalChunks = CeilDiv<uint32_t>(numBatches_, rowsPerIter_);
        uint32_t chunksPerBlock = CeilDiv<uint32_t>(totalChunks, blockNum);
        uint32_t chunkStart = blockIdx * chunksPerBlock;
        uint32_t chunkEnd = chunkStart + chunksPerBlock;
        if (chunkEnd > totalChunks) { chunkEnd = totalChunks; }

        for (uint32_t chunk = chunkStart; chunk < chunkEnd; ++chunk) {
            ProcessOneChunk(chunk);
        }
    }

private:
    __aicore__ inline void PreloadConstants()
    {
        LocalTensor<ParamDtype> tmpALog = aLogInQue_.template AllocTensor<ParamDtype>();
        dtBiasTensor_ = negExpInQue_.template AllocTensor<float>();

        DataCopyExtParams paramCopyParams{1, static_cast<uint32_t>(numHeads_ * sizeof(ParamDtype)),
                                         0, 0, 0};
        DataCopyPadExtParams<ParamDtype> paramPadParams{false, 0, 0, static_cast<ParamDtype>(0)};

        DataCopyPad(tmpALog, aLogGm_, paramCopyParams, paramPadParams);
        aLogInQue_.template EnQue<ParamDtype>(tmpALog);
        tmpALog = aLogInQue_.template DeQue<ParamDtype>();

        if constexpr (std::is_same<ParamDtype, float>()) {
            Adds(dtBiasTensor_, tmpALog, 0.0f, numHeads_);
        } else {
            Cast(dtBiasTensor_, tmpALog, RoundMode::CAST_NONE, numHeads_);
        }
        PipeBarrier<PIPE_V>();

        Exp(dtBiasTensor_, dtBiasTensor_, numHeads_);
        PipeBarrier<PIPE_V>();
        Muls(dtBiasTensor_, dtBiasTensor_, -1.0f, numHeads_);
        PipeBarrier<PIPE_V>();

        aLogInQue_.FreeTensor(tmpALog);

        negExpInQue_.template EnQue<float>(dtBiasTensor_);
        dtBiasTensor_ = negExpInQue_.template DeQue<float>();

        LocalTensor<ParamDtype> tmpDtBias = dtBiasInQue_.template AllocTensor<ParamDtype>();
        dtBiasPreloaded_ = dtBiasPreloadQue_.template AllocTensor<float>();
        DataCopyPad(tmpDtBias, dtBiasGm_, paramCopyParams, paramPadParams);
        dtBiasInQue_.template EnQue<ParamDtype>(tmpDtBias);
        tmpDtBias = dtBiasInQue_.template DeQue<ParamDtype>();
        if constexpr (std::is_same<ParamDtype, float>()) {
            Adds(dtBiasPreloaded_, tmpDtBias, 0.0f, numHeads_);
        } else {
            Cast(dtBiasPreloaded_, tmpDtBias, RoundMode::CAST_NONE, numHeads_);
        }
        PipeBarrier<PIPE_V>();
        dtBiasInQue_.FreeTensor(tmpDtBias);
        dtBiasPreloadQue_.template EnQue<float>(dtBiasPreloaded_);
        dtBiasPreloaded_ = dtBiasPreloadQue_.template DeQue<float>();

        if (rowsPerIter_ > 1) {
            LocalTensor<float> dtBiasMulti = dtBiasMultiBuf_.Get<float>();
            LocalTensor<float> negExpMulti = negExpMultiBuf_.Get<float>();
            for (uint32_t r = 0; r < rowsPerIter_; ++r) {
                const uint32_t off = r * alignedHeadsFloat_;
                Adds(dtBiasMulti[off], dtBiasPreloaded_, 0.0f, numHeads_);
                Adds(negExpMulti[off], dtBiasTensor_,    0.0f, numHeads_);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessOneChunk(uint32_t chunkIdx)
    {
        const uint32_t baseRow = chunkIdx * rowsPerIter_;
        if (baseRow >= numBatches_) {
            return;
        }
        const uint32_t remaining = (numBatches_ > baseRow) ? (numBatches_ - baseRow) : 0;
        const uint32_t validRows = (remaining >= rowsPerIter_) ? rowsPerIter_ : remaining;
        const bool isFullChunk = (validRows == rowsPerIter_);

        LocalTensor<InDtype> aLocal = aInQue_.template AllocTensor<InDtype>();
        LocalTensor<InDtype> bLocal = bInQue_.template AllocTensor<InDtype>();
        LocalTensor<float>   gLocal = gOutQue_.template AllocTensor<float>();
        LocalTensor<InDtype> betaLocal = betaOutQue_.template AllocTensor<InDtype>();

        if (useBulkDma_ && isFullChunk) {
            const uint64_t rowOffset = static_cast<uint64_t>(baseRow) * numHeads_;
            const uint32_t rowBytesHalf = numHeads_ * static_cast<uint32_t>(sizeof(InDtype));
            const uint32_t inputDstGap =
                (alignedHeadsHalf_ - numHeads_) * static_cast<uint32_t>(sizeof(InDtype)) / BYTES_PER_BLOCK;
            DataCopyExtParams bulkCopyParams{static_cast<uint16_t>(rowsPerIter_),
                                             rowBytesHalf, 0, inputDstGap, 0};
            DataCopyPadExtParams<InDtype> bulkPadParams{false, 0, 0, static_cast<InDtype>(0)};
            DataCopyPad(aLocal, aGm_[rowOffset], bulkCopyParams, bulkPadParams);
            DataCopyPad(bLocal, bGm_[rowOffset], bulkCopyParams, bulkPadParams);
        } else {
            for (uint32_t r = 0; r < validRows; ++r) {
                const uint64_t rowOffset = static_cast<uint64_t>(baseRow + r) * numHeads_;
                DataCopyExtParams rowCopyParams{1, static_cast<uint32_t>(numHeads_ * sizeof(InDtype)), 0, 0, 0};
                DataCopyPadExtParams<InDtype> rowPadParams{false, 0, 0, static_cast<InDtype>(0)};
                DataCopyPad(aLocal[r * alignedHeadsHalf_], aGm_[rowOffset], rowCopyParams, rowPadParams);
                DataCopyPad(bLocal[r * alignedHeadsHalf_], bGm_[rowOffset], rowCopyParams, rowPadParams);
            }
        }

        aInQue_.template EnQue<InDtype>(aLocal);
        bInQue_.template EnQue<InDtype>(bLocal);
        aLocal = aInQue_.template DeQue<InDtype>();
        bLocal = bInQue_.template DeQue<InDtype>();

        LocalTensor<float> x           = xBuf_.Get<float>();
        LocalTensor<float> betaX       = betaXBuf_.Get<float>();
        LocalTensor<float> softplusTmp = softplusTmpBuf_.Get<float>();
        LocalTensor<uint8_t> thresholdMask = thresholdMaskBuf_.Get<uint8_t>();
        LocalTensor<float> betaFp32    = betaFp32Buf_.Get<float>();

        const uint32_t multiCount = validRows * alignedHeadsFloat_;
        const uint32_t maskCount = validRows * alignedHeadsMask_;

        Cast(x,        aLocal, RoundMode::CAST_NONE, multiCount);
        Cast(betaFp32, bLocal, RoundMode::CAST_NONE, multiCount);
        PipeBarrier<PIPE_V>();

        if (rowsPerIter_ > 1) {
            LocalTensor<float> dtBiasMulti = dtBiasMultiBuf_.Get<float>();
            LocalTensor<float> negExpMulti = negExpMultiBuf_.Get<float>();
            Add(x, x, dtBiasMulti, multiCount);
            PipeBarrier<PIPE_V>();
            Muls(betaX, x, beta_, multiCount);
            PipeBarrier<PIPE_V>();
            Mins(softplusTmp, betaX, threshold_, multiCount);
            PipeBarrier<PIPE_V>();
            Exp(softplusTmp, softplusTmp, multiCount);
            PipeBarrier<PIPE_V>();
            Adds(softplusTmp, softplusTmp, 1.0f, multiCount);
            PipeBarrier<PIPE_V>();
            Ln(softplusTmp, softplusTmp, multiCount);
            PipeBarrier<PIPE_V>();
            Muls(softplusTmp, softplusTmp, 1.0f / beta_, multiCount);
            PipeBarrier<PIPE_V>();
            CompareScalar(thresholdMask, betaX, threshold_, CMPMODE::LE, maskCount);
            PipeBarrier<PIPE_V>();
            Select(gLocal, thresholdMask, softplusTmp, x, SELMODE::VSEL_TENSOR_TENSOR_MODE, multiCount);
            PipeBarrier<PIPE_V>();
            Mul(gLocal, gLocal, negExpMulti, multiCount);
            PipeBarrier<PIPE_V>();
        } else {
            Add(x, x, dtBiasPreloaded_, numHeads_);
            PipeBarrier<PIPE_V>();
            Muls(betaX, x, beta_, multiCount);
            PipeBarrier<PIPE_V>();
            Mins(softplusTmp, betaX, threshold_, multiCount);
            PipeBarrier<PIPE_V>();
            Exp(softplusTmp, softplusTmp, multiCount);
            PipeBarrier<PIPE_V>();
            Adds(softplusTmp, softplusTmp, 1.0f, multiCount);
            PipeBarrier<PIPE_V>();
            Ln(softplusTmp, softplusTmp, multiCount);
            PipeBarrier<PIPE_V>();
            Muls(softplusTmp, softplusTmp, 1.0f / beta_, multiCount);
            PipeBarrier<PIPE_V>();
            CompareScalar(thresholdMask, betaX, threshold_, CMPMODE::LE, maskCount);
            PipeBarrier<PIPE_V>();
            Select(gLocal, thresholdMask, softplusTmp, x, SELMODE::VSEL_TENSOR_TENSOR_MODE, multiCount);
            PipeBarrier<PIPE_V>();
            Mul(gLocal, gLocal, dtBiasTensor_, multiCount);
            PipeBarrier<PIPE_V>();
        }

        Muls(betaFp32, betaFp32, -1.0f, multiCount);
        PipeBarrier<PIPE_V>();
        Exp(betaFp32, betaFp32, multiCount);
        PipeBarrier<PIPE_V>();
        Duplicate(x, 1.0f, multiCount);
        PipeBarrier<PIPE_V>();
        Add(betaFp32, betaFp32, x, multiCount);
        PipeBarrier<PIPE_V>();
        Div(x, x, betaFp32, multiCount);
        PipeBarrier<PIPE_V>();
        Cast(betaLocal, x, RoundMode::CAST_RINT, multiCount);
        PipeBarrier<PIPE_V>();

        aInQue_.FreeTensor(aLocal);
        bInQue_.FreeTensor(bLocal);

        gOutQue_.template EnQue<float>(gLocal);
        betaOutQue_.template EnQue<InDtype>(betaLocal);

        gLocal    = gOutQue_.template DeQue<float>();
        betaLocal = betaOutQue_.template DeQue<InDtype>();

        if (useBulkDma_ && isFullChunk) {
            const uint64_t rowOffset = static_cast<uint64_t>(baseRow) * numHeads_;
            const uint32_t gSrcGap =
                (alignedHeadsFloat_ - numHeads_) * static_cast<uint32_t>(sizeof(float)) / BYTES_PER_BLOCK;
            const uint32_t bSrcGap =
                (alignedHeadsHalf_ - numHeads_) * static_cast<uint32_t>(sizeof(InDtype)) / BYTES_PER_BLOCK;
            DataCopyExtParams gOutParams{static_cast<uint16_t>(rowsPerIter_),
                                         numHeads_ * static_cast<uint32_t>(sizeof(float)),
                                         gSrcGap, 0, 0};
            DataCopyExtParams bOutParams{static_cast<uint16_t>(rowsPerIter_),
                                         numHeads_ * static_cast<uint32_t>(sizeof(InDtype)),
                                         bSrcGap, 0, 0};
            DataCopyPad(gGm_[rowOffset], gLocal, gOutParams);
            DataCopyPad(betaGm_[rowOffset], betaLocal, bOutParams);
        } else {
            for (uint32_t r = 0; r < validRows; ++r) {
                const uint64_t rowOffset = static_cast<uint64_t>(baseRow + r) * numHeads_;
                DataCopyParams gOutParams{1, static_cast<uint16_t>(numHeads_ * sizeof(float)), 0, 0};
                DataCopyParams bOutParams{1, static_cast<uint16_t>(numHeads_ * sizeof(InDtype)), 0, 0};
                DataCopyPad(gGm_[rowOffset], gLocal[r * alignedHeadsFloat_], gOutParams);
                DataCopyPad(betaGm_[rowOffset], betaLocal[r * alignedHeadsHalf_], bOutParams);
            }
        }

        gOutQue_.FreeTensor(gLocal);
        betaOutQue_.FreeTensor(betaLocal);
    }

private:
    TPipe *pipe_{nullptr};

    GlobalTensor<ParamDtype> aLogGm_;
    GlobalTensor<ParamDtype> dtBiasGm_;
    GlobalTensor<InDtype> aGm_;
    GlobalTensor<InDtype> bGm_;
    GlobalTensor<float>   gGm_;
    GlobalTensor<InDtype> betaGm_;

    TQue<QuePosition::VECIN,  1> aInQue_;
    TQue<QuePosition::VECIN,  1> bInQue_;
    TQue<QuePosition::VECIN,  1> aLogInQue_;
    TQue<QuePosition::VECIN,  1> dtBiasInQue_;
    TQue<QuePosition::VECIN,  1> negExpInQue_;
    TQue<QuePosition::VECIN,  1> dtBiasPreloadQue_;
    TQue<QuePosition::VECOUT, 1> gOutQue_;
    TQue<QuePosition::VECOUT, 1> betaOutQue_;

    TBuf<TPosition::VECCALC> dtBiasMultiBuf_;
    TBuf<TPosition::VECCALC> negExpMultiBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> betaXBuf_;
    TBuf<TPosition::VECCALC> softplusTmpBuf_;
    TBuf<TPosition::VECCALC> thresholdMaskBuf_;
    TBuf<TPosition::VECCALC> betaFp32Buf_;

    LocalTensor<float> dtBiasTensor_;     // neg_exp(A_log), 1 row
    LocalTensor<float> dtBiasPreloaded_;  // dt_bias, 1 row

    uint32_t numHeads_{0};
    uint32_t numBatches_{0};
    uint32_t rowsPerIter_{1};
    bool     useBulkDma_{false};
    uint32_t alignedHeadsHalf_{0};
    uint32_t alignedHeadsFloat_{0};
    uint32_t alignedHeadsMask_{0};
    uint32_t alignedHeadsParam_{0};
    float    beta_{1.0f};
    float    threshold_{20.0f};
};

} // namespace FusedGdnGating

#endif // FUSED_GDN_GATING_910_H
