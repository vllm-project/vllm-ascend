/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_compress_latent.h
 * \brief TurboQuant 4-bit compression of the MLA KV latent.
 *
 * Input  latent    [numTokens, headDim] fp32, already rotated by the signed Hadamard and NOT normalized.
 * Input  centroids [16]                 fp32, the Lloyd-Max codebook, sorted ascending.
 * Output slot      [numTokens, slotSize] uint8. Mode 0 uses alignUp(headDim / 2 + 2, 64); mode 1 uses
 *                  headDim / 2 + 2.
 *
 * Per token: norm = ||z|| ; u = z / norm ; nibble[d] = #{midpoint boundaries <= u[d]}, i.e. the index of
 * the nearest centroid ; the nibbles are packed two per byte in dim order into slot[0, headDim/2) and the
 * fp16 scale is stored at slot[headDim/2, headDim/2 + 2). Mode 0 stores the latent norm and zeroes the
 * remaining pad bytes. Mode 1 stores norm(latent) / norm(selectedCentroids) without output padding.
 *
 * The nearest-centroid search is a chain of 15 dependent compare/select/add rounds whose per-instruction
 * issue cost dominates the actual arithmetic, so tokensPerBatch tokens are folded into every vector
 * instruction. Only the steps that are inherently per-token stay narrow: the L2 reduction, the per-token
 * rescale and the strided nibble store.
 */
#ifndef TURBO_QUANT_COMPRESS_LATENT_H
#define TURBO_QUANT_COMPRESS_LATENT_H

#include "kernel_operator.h"

namespace TurboQuantCompressLatent {
using namespace AscendC;

constexpr uint32_t N_CENT = 16;
constexpr uint32_t ALIGN_BYTES = 64;
constexpr uint32_t SCALE_BYTES = 2;
// Keep in sync with TQ_COMPRESS_MAX_TOKENS_PER_BATCH in the tiling header.
constexpr uint32_t MAX_TOKENS_PER_BATCH = 12;
// Each token's L2 norm reduces into its own 64B-aligned slot so one V->S sync covers the whole batch.
constexpr uint32_t NORM_SLOT_FLOATS = 16;
constexpr float NORM_EPS = 1e-16f;
constexpr float NIBBLE_SIGN_THRESHOLD = 8.0f;
constexpr float NIBBLE_SIGN_OFFSET = -16.0f;
constexpr uint32_t OUTPUT_MODE_COMPACT_CORRECTED = 1;

__aicore__ inline uint32_t AlignUpTo(uint32_t value, uint32_t align) { return (value + align - 1) / align * align; }

class KernelTurboQuantCompressLatent {
public:
    __aicore__ inline KernelTurboQuantCompressLatent() {}

    __aicore__ inline void Init(GM_ADDR latent, GM_ADDR centroids, GM_ADDR slotOut, uint32_t numTokens,
                                uint32_t tokensPerCore, uint32_t headDim, uint32_t outputSlotSize,
                                uint32_t tokensPerBatch, uint32_t outputMode)
    {
        numTokens_ = numTokens;
        headDim_ = headDim;
        packedBytes_ = headDim / 2;
        slotSize_ = AlignUpTo(packedBytes_ + SCALE_BYTES, ALIGN_BYTES);
        outputSlotSize_ = outputSlotSize;
        compactCorrected_ = outputMode == OUTPUT_MODE_COMPACT_CORRECTED;
        batch_ = tokensPerBatch < 1 ? 1 : tokensPerBatch;
        if (batch_ > MAX_TOKENS_PER_BATCH) {
            batch_ = MAX_TOKENS_PER_BATCH;
        }

        uint32_t coreIdx = GetBlockIdx();
        tokStart_ = coreIdx * tokensPerCore;
        tokEnd_ = tokStart_ + tokensPerCore;
        if (tokEnd_ > numTokens_) {
            tokEnd_ = numTokens_;
        }

        latentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(latent));
        centGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slotOut));

        uint32_t batchElems = batch_ * headDim_;
        uint32_t fp32Bytes = AlignUpTo(batchElems * sizeof(float), ALIGN_BYTES);
        uint32_t halfBytes = AlignUpTo(batchElems * sizeof(half), ALIGN_BYTES);
        uint32_t maskBytes = AlignUpTo(batchElems / 8, ALIGN_BYTES);
        uint32_t slotBytes = AlignUpTo(batch_ * slotSize_, ALIGN_BYTES);
        uint32_t workBytes = AlignUpTo(headDim_ * sizeof(float), ALIGN_BYTES);
        uint32_t normBytes = AlignUpTo(batch_ * NORM_SLOT_FLOATS * sizeof(float), ALIGN_BYTES);

        pipe_.InitBuffer(inQ_, 1, fp32Bytes);
        pipe_.InitBuffer(outQ_, 1, slotBytes);
        pipe_.InitBuffer(uBuf_, fp32Bytes);
        pipe_.InitBuffer(nibBuf_, fp32Bytes);
        pipe_.InitBuffer(tmpBuf_, fp32Bytes);
        pipe_.InitBuffer(selBuf_, fp32Bytes);
        pipe_.InitBuffer(oneBuf_, fp32Bytes);
        pipe_.InitBuffer(packHalfBuf_, halfBytes);
        pipe_.InitBuffer(maskBuf_, maskBytes);
        pipe_.InitBuffer(workBuf_, workBytes);
        pipe_.InitBuffer(normBuf_, normBytes);
        pipe_.InitBuffer(centBuf_, AlignUpTo(N_CENT * sizeof(float), ALIGN_BYTES));
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> cent = centBuf_.Get<float>();
        DataCopy(cent, centGm_, N_CENT);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i + 1 < N_CENT; ++i) {
            // 15 midpoint boundaries; counting how many a value exceeds yields the nearest-centroid index
            bnd_[i] = (cent.GetValue(i) + cent.GetValue(i + 1)) * 0.5f;
        }
        LocalTensor<float> one = oneBuf_.Get<float>();
        Duplicate(one, 1.0f, batchElems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = tokStart_; t < tokEnd_; t += batch_) {
            uint32_t count = tokEnd_ - t;
            if (count > batch_) {
                count = batch_;
            }
            CopyIn(t, count);
            Compute(count);
            CopyOut(t, count);
        }
    }

private:
    __aicore__ inline void WaitVectorToScalar()
    {
        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
    }

    __aicore__ inline void CopyIn(uint32_t t, uint32_t count)
    {
        LocalTensor<float> in = inQ_.AllocTensor<float>();
        DataCopy(in, latentGm_[static_cast<uint64_t>(t) * headDim_], count * headDim_);
        inQ_.EnQue(in);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        const uint32_t elems = count * headDim_;
        LocalTensor<float> in = inQ_.DeQue<float>();
        LocalTensor<uint8_t> slot = outQ_.AllocTensor<uint8_t>();
        LocalTensor<float> u = uBuf_.Get<float>();
        LocalTensor<float> nib = nibBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> sel = selBuf_.Get<float>();
        LocalTensor<float> one = oneBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> norm = normBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        // Zero the whole slot region up front: the nibble area is overwritten below, so this leaves the
        // vecNorm hole and the trailing pad zeroed with a single vector instruction.
        LocalTensor<half> slotHalf = slot.ReinterpretCast<half>();
        Duplicate(slotHalf, static_cast<half>(0), count * slotSize_ / sizeof(half));
        PipeBarrier<PIPE_V>();

        Mul(tmp, in, in, elems);
        PipeBarrier<PIPE_V>();
        // The L2 reduction is per token; each result lands in its own slot so one V->S sync covers them all.
        for (uint32_t i = 0; i < count; ++i) {
            ReduceSum(norm[i * NORM_SLOT_FLOATS], tmp[i * headDim_], work, headDim_);
        }
        PipeBarrier<PIPE_V>();
        WaitVectorToScalar();
        for (uint32_t i = 0; i < count; ++i) {
            normScalar_[i] = sqrt(norm.GetValue(i * NORM_SLOT_FLOATS) + NORM_EPS);
            Muls(u[i * headDim_], in[i * headDim_], 1.0f / normScalar_[i], headDim_); // u = z / norm
        }
        PipeBarrier<PIPE_V>();

        Duplicate(nib, 0.0f, elems);
        PipeBarrier<PIPE_V>();
        for (uint32_t b = 0; b + 1 < N_CENT; ++b) {
            CompareScalar(mask, u, bnd_[b], CMPMODE::GE, elems); // mask = u >= bnd[b]
            PipeBarrier<PIPE_V>();
            Select(sel, mask, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, elems);
            PipeBarrier<PIPE_V>();
            Add(nib, nib, sel, elems);
            PipeBarrier<PIPE_V>();
        }

        if (compactCorrected_) {
            // The read side reconstructs centroid[nibble] * scale. Correct the original latent norm by
            // the selected codebook vector norm so the reconstructed row retains the intended magnitude.
            LocalTensor<float> cent = centBuf_.Get<float>();
            // Select the codebook value through vector masks so the index remains a vector value.
            Duplicate(tmp, cent.GetValue(0), elems);
            // Keep scalar thresholds as compile-time constants for aicore compilation.
#define TQ_SELECT_CENTROID(C)                                             \
    CompareScalar(mask, nib, static_cast<float>(C), CMPMODE::GE, elems);  \
    PipeBarrier<PIPE_V>();                                                \
    Duplicate(sel, cent.GetValue(C), elems);                              \
    PipeBarrier<PIPE_V>();                                                \
    Select(tmp, mask, sel, tmp, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems); \
    PipeBarrier<PIPE_V>();
            TQ_SELECT_CENTROID(1)
            TQ_SELECT_CENTROID(2)
            TQ_SELECT_CENTROID(3)
            TQ_SELECT_CENTROID(4)
            TQ_SELECT_CENTROID(5)
            TQ_SELECT_CENTROID(6)
            TQ_SELECT_CENTROID(7)
            TQ_SELECT_CENTROID(8)
            TQ_SELECT_CENTROID(9)
            TQ_SELECT_CENTROID(10)
            TQ_SELECT_CENTROID(11)
            TQ_SELECT_CENTROID(12)
            TQ_SELECT_CENTROID(13)
            TQ_SELECT_CENTROID(14)
            TQ_SELECT_CENTROID(15)
#undef TQ_SELECT_CENTROID
            PipeBarrier<PIPE_V>();
            Mul(tmp, tmp, tmp, elems);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < count; ++i) {
                ReduceSum(norm[i * NORM_SLOT_FLOATS], tmp[i * headDim_], work, headDim_);
            }
            PipeBarrier<PIPE_V>();
            WaitVectorToScalar();
            for (uint32_t i = 0; i < count; ++i) {
                normScalar_[i] /= sqrt(norm.GetValue(i * NORM_SLOT_FLOATS) + NORM_EPS);
            }
        }

        // int4b_t HW pack: nib(0..15, dim order) -> signed s(-8..7) -> half -> int4b_t (low nibble first).
        // s = (nib < 8) ? nib : nib - 16, i.e. the same 4 bits reinterpreted as two's complement.
        LocalTensor<half> packHalf = packHalfBuf_.Get<half>();
        CompareScalar(mask, nib, NIBBLE_SIGN_THRESHOLD, CMPMODE::LT, elems);
        PipeBarrier<PIPE_V>();
        Adds(sel, nib, NIBBLE_SIGN_OFFSET, elems);
        PipeBarrier<PIPE_V>();
        Select(tmp, mask, nib, sel, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        PipeBarrier<PIPE_V>();
        Cast(packHalf, tmp, RoundMode::CAST_RINT, elems);
        PipeBarrier<PIPE_V>();
        // Slots are slotSize apart while the nibbles are only headDim/2 wide, so the pack stays per token.
        for (uint32_t i = 0; i < count; ++i) {
            LocalTensor<int4b_t> packed = slot[i * slotSize_].ReinterpretCast<int4b_t>();
            Cast(packed, packHalf[i * headDim_], RoundMode::CAST_RINT, headDim_);
        }
        PipeBarrier<PIPE_V>();

        WaitVectorToScalar();
        for (uint32_t i = 0; i < count; ++i) {
            half normHalf = static_cast<half>(normScalar_[i]);
            uint16_t normBits = *reinterpret_cast<uint16_t*>(&normHalf);
            slot.SetValue(i * slotSize_ + packedBytes_, static_cast<uint8_t>(normBits & 0xff));
            slot.SetValue(i * slotSize_ + packedBytes_ + 1, static_cast<uint8_t>((normBits >> 8) & 0xff));
        }

        inQ_.FreeTensor(in);
        outQ_.EnQue(slot);
    }

    __aicore__ inline void CopyOut(uint32_t t, uint32_t count)
    {
        LocalTensor<uint8_t> slot = outQ_.DeQue<uint8_t>();
        if (compactCorrected_) {
            for (uint32_t i = 0; i < count; ++i) {
                DataCopyParams copyParams{1, static_cast<uint16_t>(outputSlotSize_), 0, 0};
                DataCopyPad(slotGm_[static_cast<uint64_t>(t + i) * outputSlotSize_], slot[i * slotSize_], copyParams);
            }
        } else {
            DataCopy(slotGm_[static_cast<uint64_t>(t) * outputSlotSize_], slot, count * slotSize_);
        }
        outQ_.FreeTensor(slot);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQ_;
    TQue<QuePosition::VECOUT, 1> outQ_;
    TBuf<TPosition::VECCALC> uBuf_, nibBuf_, tmpBuf_, selBuf_, oneBuf_, maskBuf_;
    TBuf<TPosition::VECCALC> packHalfBuf_, workBuf_, normBuf_, centBuf_;
    GlobalTensor<float> latentGm_;
    GlobalTensor<float> centGm_;
    GlobalTensor<uint8_t> slotGm_;
    uint32_t numTokens_ = 0;
    uint32_t headDim_ = 0;
    uint32_t slotSize_ = 0;
    uint32_t outputSlotSize_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t batch_ = 1;
    uint32_t tokStart_ = 0;
    uint32_t tokEnd_ = 0;
    bool compactCorrected_ = false;
    float bnd_[N_CENT] = {};
    float normScalar_[MAX_TOKENS_PER_BATCH] = {};
};

} // namespace TurboQuantCompressLatent
#endif // TURBO_QUANT_COMPRESS_LATENT_H
