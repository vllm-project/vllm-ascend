/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_situ_quant.h
 * \brief DequantSituQuant kernel: Dequant -> Situ -> Quant
 */

#ifndef DEQUANT_SITU_QUANT_H
#define DEQUANT_SITU_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include <type_traits>

#define TEMPLATE_DSQ_DECLARE template <bool hasDequantBias>
#define TEMPLATE_DSQ_ARGS hasDequantBias

namespace DequantSituQuantOps {
using namespace AscendC;

constexpr static int64_t DB_BUFFER = 1;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t MASK_NUM_T32 = 256 / sizeof(float);
constexpr static int64_t MASK_BLK_STRIDE = 8;
constexpr static int64_t ELEM_PER_REP_FP32 = 64;
constexpr static int64_t MAX_REPEAT = 255;
constexpr static int64_t SWI_FACTOR = 2;
constexpr static float DYNAMIC_QUANT_FACTOR = 1.0 / 127.0;

TEMPLATE_DSQ_DECLARE
class DequantSituQuantKernel {
public:
    __aicore__ inline DequantSituQuantKernel(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dequantScale, GM_ADDR dequantBias, GM_ADDR quantScale,
                                GM_ADDR quantOffset, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
                                const DequantSituQuantTilingData* tilingData)
    {
        tl_ = tilingData;
        blockIdx_ = GetBlockIdx();

        rowLen_ = tl_->rowLen;
        colLen_ = tl_->colLen;
        inDimy_ = colLen_ * SWI_FACTOR;
        outDimy_ = colLen_;
        baseRowLen_ = tl_->baseRowLen;
        baseColLen_ = tl_->baseColLen < colLen_ ? tl_->baseColLen : colLen_;
        usedCoreNum_ = tl_->usedCoreNum;
        activateLeft_ = tl_->activateLeft;
        quantMode_ = tl_->quantMode;
        quantIsOne_ = tl_->quantIsOne;
        quantScaleIsEmpty_ = tl_->quantScaleIsEmpty;
        quantOffsetIsEmpty_ = tl_->quantOffsetIsEmpty;
        beta_ = tl_->beta;
        linearBeta_ = tl_->linearBeta;

        if (rowLen_ < usedCoreNum_) {
            usedCoreNum_ = rowLen_;
        }
        int64_t perRoundCnt = usedCoreNum_ == 0 ? 0 : rowLen_ / usedCoreNum_;
        int64_t remainCnt = rowLen_ - usedCoreNum_ * perRoundCnt;
        curCoreRowNum_ = perRoundCnt;
        if (blockIdx_ < remainCnt) {
            curCoreRowNum_ = perRoundCnt + 1;
            inputCopyOffset_ = blockIdx_ * curCoreRowNum_;
        } else {
            inputCopyOffset_ = remainCnt * (perRoundCnt + 1) + (blockIdx_ - remainCnt) * perRoundCnt;
        }

        xGm_.SetGlobalBuffer((__gm__ int8_t*)x + inputCopyOffset_ * inDimy_, curCoreRowNum_ * inDimy_);
        dequantScaleGm_.SetGlobalBuffer((__gm__ float*)dequantScale);
        if constexpr (hasDequantBias) {
            dequantBiasGm_.SetGlobalBuffer((__gm__ float*)dequantBias);
        }
        if (quantScaleIsEmpty_ == 0) {
            quantScaleGm_.SetGlobalBuffer((__gm__ float*)quantScale);
        }
        if (quantOffsetIsEmpty_ == 0) {
            quantOffsetGm_.SetGlobalBuffer((__gm__ float*)quantOffset);
        }
        yGm_.SetGlobalBuffer((__gm__ int8_t*)y + inputCopyOffset_ * outDimy_, curCoreRowNum_ * outDimy_);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale + inputCopyOffset_, curCoreRowNum_);

        if (quantScaleIsEmpty_ == 0 && quantIsOne_) {
            quantScaleVal_ = quantScaleGm_.GetValue(0);
            if (quantScaleVal_ == 0.0f) {
                quantScaleVal_ = 1.0f;
            } else {
                quantScaleVal_ = 1.0f / quantScaleVal_;
            }
        }
        if (quantOffsetIsEmpty_ == 0 && quantIsOne_) {
            quantOffsetVal_ = quantOffsetGm_.GetValue(0);
        }

        // Check if dequant_scale is scalar (shape [1])
        dequantScaleIsOne_ = (dequantScaleGm_.GetSize() == 1);
        if (dequantScaleIsOne_) {
            dequantScaleVal_ = dequantScaleGm_.GetValue(0);
        }
        if constexpr (hasDequantBias) {
            dequantBiasIsOne_ = (dequantBiasGm_.GetSize() == 1);
            if (dequantBiasIsOne_) {
                dequantBiasVal_ = dequantBiasGm_.GetValue(0);
            }
        }

        curColNum_ = baseColLen_;
        InitUbBuffer();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        processCompute();
    }

protected:
    __aicore__ inline void InitUbBuffer()
    {
        int64_t alignColNum = curColNum_ == Align(curColNum_, sizeof(int8_t)) ?
                                  curColNum_ :
                                  Align(curColNum_, sizeof(int8_t));
        int64_t alignInDimy = alignColNum * SWI_FACTOR;

        pipe_->InitBuffer(inQueueX_, DB_BUFFER, alignInDimy * sizeof(int8_t));
        pipe_->InitBuffer(dequantScaleBuf_, alignInDimy * sizeof(float));
        if constexpr (hasDequantBias) {
            pipe_->InitBuffer(dequantBiasBuf_, alignInDimy * sizeof(float));
        }
        if (quantScaleIsEmpty_ == 0 && !quantIsOne_) {
            int64_t quantBufElems = (quantMode_ == 1) ? alignColNum : alignInDimy;
            pipe_->InitBuffer(quantBuf_, quantBufElems * sizeof(float));
        }
        // outQueue_ must hold max(int8 output, float situ output) + scale + padding
        // Situ computation uses outQueue as float buffer [H] floats = H*4 bytes
        // Final output is [H] int8 = H bytes + scale [1] float = 4 bytes
        pipe_->InitBuffer(outQueue_, 1, alignColNum * sizeof(float) + sizeof(float) + BLOCK_SIZE);

        // temp buffers for compute: dequantOut[2H] + situTemp[2H] = 4H floats
        pipe_->InitBuffer(tmpBuf_, alignInDimy * SWI_FACTOR * sizeof(float));
        // cast buffer for int8<->float conversion intermediates
        pipe_->InitBuffer(castBuf_, alignInDimy * SWI_FACTOR * sizeof(float));
    }

    __aicore__ inline void CopyInDequantParams(int64_t colOffset)
    {
        // Sync V→MTE2: ensure previous tile's V operations finish before
        // overwriting dequantScaleBuf_ (TBuf has no automatic pipeline sync)
        event_t eventV2MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventV2MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventV2MTE2);

        if (!dequantScaleIsOne_) {
            // x layout: [up(0:H), gate(H:2H)] — load up and gate separately
            DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            LocalTensor<float> dequantScaleLocal = dequantScaleBuf_.template Get<float>();
            // Load dequant_scale for up: ds[colOffset : colOffset + curColNum]
            DataCopyPad(dequantScaleLocal, dequantScaleGm_[colOffset], params, padParams);
            // Load dequant_scale for gate: ds[outDimy_ + colOffset : outDimy_ + colOffset + curColNum]
            DataCopyPad(dequantScaleLocal[curColNum_], dequantScaleGm_[outDimy_ + colOffset], params, padParams);
            dequantScaleLocal_ = dequantScaleLocal;
        }

        if constexpr (hasDequantBias) {
            if (!dequantBiasIsOne_) {
                DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                LocalTensor<float> dequantBiasLocal = dequantBiasBuf_.template Get<float>();
                DataCopyPad(dequantBiasLocal, dequantBiasGm_[colOffset], params, padParams);
                DataCopyPad(dequantBiasLocal[curColNum_], dequantBiasGm_[outDimy_ + colOffset], params, padParams);
                dequantBiasLocal_ = dequantBiasLocal;
            }
        }

        // Sync MTE2→V: ensure TBuf DataCopyPad completes before vector compute reads it
        event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    }

    __aicore__ inline void CopyInQuantParams(int64_t colOffset)
    {
        if (quantScaleIsEmpty_ == 0 && !quantIsOne_) {
            DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            LocalTensor<float> quantLocal = quantBuf_.template Get<float>();
            DataCopyPad(quantLocal, quantScaleGm_[colOffset], params, padParams);
            if (quantOffsetIsEmpty_ == 0) {
                DataCopyPad(quantLocal[curColNum_], quantOffsetGm_[colOffset], params, padParams);
            }
            quantLocal_ = quantLocal;

            // Sync MTE2→V: ensure TBuf DataCopyPad completes before vector compute reads it
            event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        }
    }

    __aicore__ inline void CopyIn(int64_t rowIdx, int64_t colOffset)
    {
        // x layout: [up(0:H), gate(H:2H)] — load up and gate separately
        DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(int8_t)), 0, 0, 0};
        DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};

        LocalTensor<int8_t> xLocal = inQueueX_.template AllocTensor<int8_t>();
        // Load up: x[rowIdx * inDimy + colOffset : ... + curColNum]
        DataCopyPad(xLocal, xGm_[rowIdx * inDimy_ + colOffset], params, padParams);
        // Load gate: x[rowIdx * inDimy + outDimy_ + colOffset : ... + curColNum]
        DataCopyPad(xLocal[curColNum_], xGm_[rowIdx * inDimy_ + outDimy_ + colOffset], params, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeDequant(int64_t rowIdx)
    {
        LocalTensor<int8_t> xLocalI8 = inQueueX_.template DeQue<int8_t>();

        LocalTensor<float> tmpF32 = tmpBuf_.template Get<float>();
        int64_t tileLen = curColNum_ * SWI_FACTOR;
        LocalTensor<float> dequantOut = tmpF32;
        LocalTensor<float> situTemp = tmpF32[tileLen];

        // Step 1: Cast int8 -> half -> fp32
        LocalTensor<half> tmpHalf = castBuf_.template Get<half>();
        Cast(tmpHalf, xLocalI8, RoundMode::CAST_NONE, tileLen);
        PipeBarrier<PIPE_V>();
        Cast(dequantOut, tmpHalf, RoundMode::CAST_NONE, tileLen);
        PipeBarrier<PIPE_V>();
        inQueueX_.FreeTensor(xLocalI8);

        // Step 2: Mul dequant_scale
        if (dequantScaleIsOne_) {
            Muls(dequantOut, dequantOut, dequantScaleVal_, tileLen);
        } else {
            Mul(dequantOut, dequantOut, dequantScaleLocal_, tileLen);
        }
        PipeBarrier<PIPE_V>();

        // Step 3: Add dequant_bias (if exists)
        if constexpr (hasDequantBias) {
            if (dequantBiasIsOne_) {
                Adds(dequantOut, dequantOut, dequantBiasVal_, tileLen);
                PipeBarrier<PIPE_V>();
            } else {
                Add(dequantOut, dequantOut, dequantBiasLocal_, tileLen);
                PipeBarrier<PIPE_V>();
            }
        }

        // Store dequantOut for Situ computation
        dequantOut_ = dequantOut;
        situTemp_ = situTemp;
    }

    __aicore__ inline void ComputeSitu()
    {
        int64_t H = curColNum_;
        LocalTensor<float> dequantOut = dequantOut_;
        LocalTensor<float> tmp = situTemp_;

        // gate and up: activateLeft=0 means gate=right half, up=left half
        // activateLeft=1 means gate=left half, up=right half
        int64_t gateOffset = (activateLeft_ == 1) ? 0 : H;
        int64_t upOffset = (activateLeft_ == 1) ? H : 0;

        LocalTensor<float> gate = dequantOut[gateOffset];
        LocalTensor<float> up = dequantOut[upOffset];

        // tmpBuf_ layout: [0:2H] = dequantOut (no longer needed), [2H:4H] = situTemp
        // Reuse situTemp for Situ computation:
        // tmp[0:H] = tanh result (beta * tanh(gate/beta))
        // tmp[H:2H] = sigmoid result + sigmoid denom
        LocalTensor<float> tanhResult = tmp;
        LocalTensor<float> sigmoidResult = tmp[H];

        // Step 1: tanh(gate / beta) * beta
        float invBeta = 1.0f / beta_;
        Muls(tanhResult, gate, invBeta, H);
        PipeBarrier<PIPE_V>();
        Tanh(tanhResult, tanhResult, H);
        PipeBarrier<PIPE_V>();

        Muls(tanhResult, tanhResult, beta_, H);
        PipeBarrier<PIPE_V>();

        // Step 2: sigmoid(gate) = 1 / (1 + exp(-gate))
        // Numerically stable: avoids positive-input exp overflow.
        LocalTensor<float> denomTmp = dequantOut[gateOffset];
        Muls(sigmoidResult, gate, -1.0f, H);
        PipeBarrier<PIPE_V>();
        Exp(sigmoidResult, sigmoidResult, H);
        PipeBarrier<PIPE_V>();

        Adds(denomTmp, sigmoidResult, 1.0f, H); // 1 + exp(-gate)
        PipeBarrier<PIPE_V>();

        // sigmoid = 1 / (1 + exp(-gate))
        // Use Level 0 Div instead of Reciprocal for better precision.
        // src0 is a single datablock of 1.0f, reused across all repeats via
        // src0BlkStride=0 and src0RepStride=0.
        LocalTensor<float> onesBlock = castBuf_.template Get<float>();
        Duplicate<float>(onesBlock, 1.0f, 8);
        PipeBarrier<PIPE_V>();

        constexpr uint64_t maskFp32 = static_cast<uint64_t>(ELEM_PER_REP_FP32);
        uint32_t fullReps = static_cast<uint32_t>(H / maskFp32);
        uint32_t remainder = static_cast<uint32_t>(H % maskFp32);
        BinaryRepeatParams divParams(1, 0, 1, 8, 0, 8);

        if (fullReps > 0) {
            Div(sigmoidResult, onesBlock, denomTmp, maskFp32,
                static_cast<uint8_t>(fullReps), divParams);
            PipeBarrier<PIPE_V>();
        }
        if (remainder > 0) {
            Div(sigmoidResult[fullReps * maskFp32], onesBlock,
                denomTmp[fullReps * maskFp32], remainder, 1, divParams);
            PipeBarrier<PIPE_V>();
        }

        // Step 3: situ_a = tanhResult * sigmoidResult
        Mul(tanhResult, tanhResult, sigmoidResult, H);
        PipeBarrier<PIPE_V>();

        // Step 4: if linear_beta > 0: up = linear_beta * tanh(up / linear_beta)
        if (linearBeta_ > 0.0f) {
            float invLinearBeta = 1.0f / linearBeta_;
            Muls(up, up, invLinearBeta, H);
            PipeBarrier<PIPE_V>();
            Tanh(up, up, H);
            PipeBarrier<PIPE_V>();
            Muls(up, up, linearBeta_, H);
            PipeBarrier<PIPE_V>();
        }

        // Step 5: output = situ_a * up = tanhResult * up
        // Write to gate buffer (no longer needed, avoids aliasing with up)
        LocalTensor<float> situOut = dequantOut[gateOffset];
        Mul(situOut, tanhResult, up, H);
        PipeBarrier<PIPE_V>();

        // situOut now holds the Situ output [H] in fp32, stored in gate buffer region
        situOut_ = situOut;
    }

    __aicore__ inline void ComputeQuant()
    {
        int64_t H = curColNum_;
        LocalTensor<float> situOut = situOut_;

        if (quantMode_ == 1) {
            // Dynamic quant
            DynamicQuant(situOut);
        } else {
            // Static quant
            StaticQuant(situOut);
        }
    }

    __aicore__ inline void StaticQuant(LocalTensor<float>& situOut)
    {
        int64_t H = curColNum_;

        if (quantScaleIsEmpty_ == 0) {
            if (quantIsOne_) {
                Muls(situOut, situOut, quantScaleVal_, H);
                PipeBarrier<PIPE_V>();
                if (quantOffsetIsEmpty_ == 0) {
                    Adds(situOut, situOut, quantOffsetVal_, H);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                Div(situOut, situOut, quantLocal_, H);
                PipeBarrier<PIPE_V>();
                if (quantOffsetIsEmpty_ == 0) {
                    Add(situOut, situOut, quantLocal_[H], H);
                    PipeBarrier<PIPE_V>();
                }
            }
        }

        // Allocate outQueue and cast fp32 -> int8
        LocalTensor<float> outLocal = outQueue_.template AllocTensor<float>();
        LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
        CastFloatToInt8(situOut, yOut, H);
        outQueue_.EnQue<float>(outLocal);
    }

    __aicore__ inline void DynamicQuant(LocalTensor<float>& situOut)
    {
        int64_t H = curColNum_;

        if (quantScaleIsEmpty_ == 0) {
            Mul(situOut, situOut, quantLocal_, H);
            PipeBarrier<PIPE_V>();
        }

        // Compute per-row abs max using situTemp_ (no longer needed after Situ)
        LocalTensor<float> absBuf = situTemp_;
        Abs(absBuf, situOut, H);
        PipeBarrier<PIPE_V>();

        // Allocate outQueue for final output: [H int8][1 float scale]
        LocalTensor<float> outLocal = outQueue_.template AllocTensor<float>();
        LocalTensor<float> scaleOut = outLocal[H];
        LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();

        ComputeReduceMax(absBuf, H);
        PipeBarrier<PIPE_V>();

        Muls(scaleOut, absBuf, DYNAMIC_QUANT_FACTOR, 1);
        PipeBarrier<PIPE_V>();

        event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventV2S);
        WaitFlag<HardEvent::V_S>(eventV2S);
        float scaleVal = scaleOut.GetValue(0);
        if (scaleVal == 0.0f) {
            scaleVal = 1.0f;
        }
        float invScale = 1.0f / scaleVal;
        Muls(situOut, situOut, invScale, H);
        PipeBarrier<PIPE_V>();

        CastFloatToInt8(situOut, yOut, H);
        outQueue_.EnQue<float>(outLocal);
    }

    __aicore__ inline void CastFloatToInt8(const LocalTensor<float>& src, LocalTensor<int8_t>& dst, int64_t count)
    {
        // FP32 -> INT32 (rint)
        LocalTensor<int32_t> tmpI32 = castBuf_.template Get<int32_t>();
        Cast(tmpI32, src, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);

        // INT32 -> FP16 (round)
        LocalTensor<float> tmpF32 = castBuf_.template Get<float>();
        LocalTensor<half> tmpF16 = tmpF32.ReinterpretCast<half>();
        Cast(tmpF16, tmpI32, RoundMode::CAST_ROUND, count);
        PipeBarrier<PIPE_V>();

        // FP16 -> INT8 (trunc)
        Cast(dst, tmpF16, RoundMode::CAST_TRUNC, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calCount)
    {
        uint32_t repsFp32 = static_cast<uint32_t>(calCount >> 6);
        uint32_t offsetsFp32 = repsFp32 << 6;
        uint32_t remsFp32 = static_cast<uint32_t>(calCount & 0x3f);

        if (likely(repsFp32 > 1)) {
            if (repsFp32 - 1 > MAX_REPEAT) {
                Max(tempRes, tempRes[ELEM_PER_REP_FP32], tempRes, ELEM_PER_REP_FP32, MAX_REPEAT,
                    {1, 1, 1, 0, 8, 0});
                PipeBarrier<PIPE_V>();
                Max(tempRes, tempRes[ELEM_PER_REP_FP32 * MAX_REPEAT], tempRes, ELEM_PER_REP_FP32,
                    repsFp32 - MAX_REPEAT - 1, {1, 1, 1, 0, 8, 0});
            } else {
                Max(tempRes, tempRes[ELEM_PER_REP_FP32], tempRes, ELEM_PER_REP_FP32, repsFp32 - 1,
                    {1, 1, 1, 0, 8, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
            Max(tempRes, tempRes[offsetsFp32], tempRes, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = repsFp32 > 0 ? ELEM_PER_REP_FP32 : calCount;
        WholeReduceMax(tempRes, tempRes, mask, 1, 8, 1, 8);
    }

    __aicore__ inline void CopyOut(int64_t rowIdx, int64_t colOffset)
    {
        LocalTensor<float> outLocal = outQueue_.template DeQue<float>();
        LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();

        DataCopyExtParams dataCopyYParams{1, static_cast<uint32_t>(curColNum_ * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(yGm_[rowIdx * outDimy_ + colOffset], yOut, dataCopyYParams);

        if (quantMode_ == 1) {
            LocalTensor<float> scaleOut = outLocal[curColNum_];
            DataCopyExtParams dataCopyScaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(scaleGm_[rowIdx], scaleOut, dataCopyScaleParams);
        }

        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void processCompute()
    {
        int64_t lastColNum = baseColLen_;
        int64_t colLoops = 1;
        if (baseColLen_ < colLen_) {
            colLoops = (colLen_ + baseColLen_ - 1) / baseColLen_;
            lastColNum = colLen_ - (colLoops - 1) * baseColLen_;
        }

        if (quantMode_ == 1 && colLoops > 1) {
            // Dynamic mode with column tiling: two-pass (recompute) approach
            // Pass 1: compute per-row absmax across all column tiles
            // Pass 2: re-compute Situ, quantize with global scale, output
            for (int64_t i = 0; i < curCoreRowNum_; i++) {
                float scaleVal = DynamicComputeRowScale(i, colLoops, lastColNum);
                // Sync MTE3→V between Pass 1 and Pass 2
                event_t eventMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV);
                DynamicQuantizeAndOutput(i, scaleVal, colLoops, lastColNum);
            }
        } else {
            // Single-pass approach (no column tiling or static mode)
            // Row-major order: process all tiles for each row before moving to next row
            for (int64_t i = 0; i < curCoreRowNum_; i++) {
                for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
                    curColNum_ = (colLoop == colLoops - 1) ? lastColNum : baseColLen_;
                    curColNum_ = (curColNum_ == 0) ? baseColLen_ : curColNum_;

                    CopyInDequantParams(colLoop * baseColLen_);
                    if (quantScaleIsEmpty_ == 0 && !quantIsOne_) {
                        CopyInQuantParams(colLoop * baseColLen_);
                    }
                    CopyIn(i, colLoop * baseColLen_);
                    ComputeDequant(i);
                    ComputeSitu();
                    ComputeQuant();
                    CopyOut(i, colLoop * baseColLen_);
                }
            }
        }
    }

    __aicore__ inline void ApplySmoothScale(LocalTensor<float>& situOut)
    {
        if (quantScaleIsEmpty_ == 0) {
            if (quantIsOne_) {
                Muls(situOut, situOut, quantScaleVal_, curColNum_);
            } else {
                Mul(situOut, situOut, quantLocal_, curColNum_);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float DynamicComputeRowScale(int64_t rowIdx, int64_t colLoops, int64_t lastColNum)
    {
        float rowAbsMax = 0.0f;
        for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
            curColNum_ = (colLoop == colLoops - 1) ? lastColNum : baseColLen_;
            curColNum_ = (curColNum_ == 0) ? baseColLen_ : curColNum_;

            CopyInDequantParams(colLoop * baseColLen_);
            if (quantScaleIsEmpty_ == 0 && !quantIsOne_) {
                CopyInQuantParams(colLoop * baseColLen_);
            }
            CopyIn(rowIdx, colLoop * baseColLen_);
            ComputeDequant(rowIdx);
            ComputeSitu();
            ApplySmoothScale(situOut_);

            // Compute absmax for this tile
            LocalTensor<float> absBuf = situTemp_;
            Abs(absBuf, situOut_, curColNum_);
            PipeBarrier<PIPE_V>();
            ComputeReduceMax(absBuf, curColNum_);
            PipeBarrier<PIPE_V>();

            event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventV2S);
            WaitFlag<HardEvent::V_S>(eventV2S);
            float tileMax = absBuf.GetValue(0);
            if (tileMax > rowAbsMax) {
                rowAbsMax = tileMax;
            }
        }

        float scaleVal = rowAbsMax * DYNAMIC_QUANT_FACTOR;
        if (scaleVal == 0.0f) {
            scaleVal = 1.0f;
        }
        return scaleVal;
    }

    __aicore__ inline void DynamicQuantizeAndOutput(int64_t rowIdx, float scaleVal, int64_t colLoops, int64_t lastColNum)
    {
        float invScale = 1.0f / scaleVal;

        for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
            curColNum_ = (colLoop == colLoops - 1) ? lastColNum : baseColLen_;
            curColNum_ = (curColNum_ == 0) ? baseColLen_ : curColNum_;

            CopyInDequantParams(colLoop * baseColLen_);
            if (quantScaleIsEmpty_ == 0 && !quantIsOne_) {
                CopyInQuantParams(colLoop * baseColLen_);
            }
            CopyIn(rowIdx, colLoop * baseColLen_);
            ComputeDequant(rowIdx);
            ComputeSitu();
            ApplySmoothScale(situOut_);

            // Quantize with global scale
            Muls(situOut_, situOut_, invScale, curColNum_);
            PipeBarrier<PIPE_V>();

            // Cast to int8 and output
            LocalTensor<float> outLocal = outQueue_.template AllocTensor<float>();
            LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
            CastFloatToInt8(situOut_, yOut, curColNum_);

            // Write scale for first tile only
            if (colLoop == 0) {
                LocalTensor<float> scaleOut = outLocal[curColNum_];
                Duplicate<float>(scaleOut, scaleVal, 1);
                PipeBarrier<PIPE_V>();
            }

            outQueue_.EnQue<float>(outLocal);

            // CopyOut y (always) and scale (first tile only)
            LocalTensor<float> outLocalDeq = outQueue_.template DeQue<float>();
            LocalTensor<int8_t> yOutDeq = outLocalDeq.template ReinterpretCast<int8_t>();
            DataCopyExtParams dataCopyYParams{1, static_cast<uint32_t>(curColNum_ * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(yGm_[rowIdx * outDimy_ + colLoop * baseColLen_], yOutDeq, dataCopyYParams);

            if (colLoop == 0) {
                LocalTensor<float> scaleOutDeq = outLocalDeq[curColNum_];
                DataCopyExtParams dataCopyScaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(scaleGm_[rowIdx], scaleOutDeq, dataCopyScaleParams);
            }

            outQueue_.FreeTensor(outLocalDeq);

            // Sync MTE3→MTE2 between tiles: ensure copy-out completes before
            // next tile's CopyInDequantParams overwrites TBuf buffers
            if (colLoop + 1 < colLoops) {
                event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
            }
        }
    }

    __aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
    {
        constexpr int64_t BLOCK_BYTES = 32;
        if (bytes == 0) {
            return 0;
        }
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
    }

protected:
    TPipe* pipe_ = nullptr;
    const DequantSituQuantTilingData* tl_ = nullptr;

    int64_t blockIdx_ = 0;
    int64_t rowLen_ = 0;
    int64_t colLen_ = 0;
    int64_t inDimy_ = 0;
    int64_t outDimy_ = 0;
    int64_t baseRowLen_ = 0;
    int64_t baseColLen_ = 0;
    int64_t curColNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t curCoreRowNum_ = 0;
    int64_t inputCopyOffset_ = 0;
    int64_t activateLeft_ = 0;
    int64_t quantMode_ = 0;
    bool quantIsOne_ = false;
    int64_t quantScaleIsEmpty_ = 1;
    int64_t quantOffsetIsEmpty_ = 1;
    float beta_ = 1.0f;
    float linearBeta_ = 0.0f;
    float quantScaleVal_ = 1.0f;
    float quantOffsetVal_ = 0.0f;
    bool dequantScaleIsOne_ = false;
    float dequantScaleVal_ = 1.0f;
    bool dequantBiasIsOne_ = false;
    float dequantBiasVal_ = 0.0f;

    GlobalTensor<int8_t> xGm_;
    GlobalTensor<float> dequantScaleGm_;
    GlobalTensor<float> dequantBiasGm_;
    GlobalTensor<float> quantScaleGm_;
    GlobalTensor<float> quantOffsetGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TBuf<TPosition::VECCALC> dequantScaleBuf_;
    TBuf<TPosition::VECCALC> dequantBiasBuf_;
    TBuf<TPosition::VECCALC> quantBuf_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> castBuf_;

    // Intermediate results passed between compute stages
    LocalTensor<float> dequantScaleLocal_;
    LocalTensor<float> dequantBiasLocal_;
    LocalTensor<float> quantLocal_;
    LocalTensor<float> dequantOut_;
    LocalTensor<float> situTemp_;
    LocalTensor<float> situOut_;
};

// ---------------------------------------------------------------------------
// K3 Kernel: INT32/BF16 path with MoE routing and per-row dynamic quant
// ---------------------------------------------------------------------------

constexpr int64_t K3_MASK_FP32 = 256 / sizeof(float);
constexpr int64_t K3_MASK_BLK_STRIDE = 8;
constexpr float K3_DYNAMIC_QUANT_FACTOR = 1.0f / 127.0f;

template <typename XType>
class DequantSituQuantK3Kernel {
public:
    __aicore__ inline explicit DequantSituQuantK3Kernel(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR groupIndex,
        GM_ADDR y, GM_ADDR scale, const DequantSituQuantTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        rowLen_ = static_cast<int64_t>(tilingData_->rowLen);
        inputWidth_ = static_cast<int64_t>(tilingData_->inputWidth);
        outputWidth_ = static_cast<int64_t>(tilingData_->outputWidth);
        expertNum_ = static_cast<int64_t>(tilingData_->expertNum);
        usedCoreNum_ = static_cast<int64_t>(tilingData_->usedCoreNum);
        hasBias_ = tilingData_->dequantBiasIsEmpty == 0;
        hasGroupIndex_ = tilingData_->hasGroupIndex != 0;
        activateLeft_ = tilingData_->activateLeft;
        beta_ = tilingData_->beta;
        linearBeta_ = tilingData_->linearBeta;

        xGm_.SetGlobalBuffer((__gm__ XType*)x, rowLen_ * inputWidth_);
        if constexpr (std::is_same_v<XType, int32_t>) {
            weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale, expertNum_ * inputWidth_);
            activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale, rowLen_);
            if (hasBias_) {
                biasGm_.SetGlobalBuffer((__gm__ float*)bias, expertNum_ * inputWidth_);
            }
        }
        if (hasGroupIndex_) {
            groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex, expertNum_);
        }
        yGm_.SetGlobalBuffer((__gm__ int8_t*)y, rowLen_ * outputWidth_);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale, rowLen_);

        const int64_t inputBytes = inputWidth_ * static_cast<int64_t>(sizeof(XType));
        const int64_t paramBytes = inputWidth_ * static_cast<int64_t>(sizeof(float));
        const int64_t outputBytes = outputWidth_ * static_cast<int64_t>(sizeof(int8_t)) + 32;
        pipe_->InitBuffer(xQueue_, 1, inputBytes);
        if constexpr (std::is_same_v<XType, int32_t>) {
            pipe_->InitBuffer(weightScaleQueue_, 1, paramBytes);
            if (hasBias_) {
                pipe_->InitBuffer(biasQueue_, 1, paramBytes);
            }
        } else {
            pipe_->InitBuffer(dequantBuf_, inputWidth_ * static_cast<int64_t>(sizeof(float)));
        }
        pipe_->InitBuffer(outQueue_, 1, outputBytes);
        pipe_->InitBuffer(tmpBuf_, inputWidth_ * static_cast<int64_t>(sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        if (usedCoreNum_ <= 0 || blockIdx_ >= usedCoreNum_) {
            return;
        }

        if (!hasGroupIndex_) {
            ProcessGroup(0, rowLen_, 0);
            return;
        }

        if constexpr (!std::is_same_v<XType, int32_t>) {
            // BF16 GMM output may retain the dispatch buffer capacity. Only
            // the leading sum(group_index) rows contain routed tokens.
            int64_t validRows = 0;
            for (int64_t expertIdx = 0; expertIdx < expertNum_ && validRows < rowLen_; ++expertIdx) {
                const int64_t requestedRows = groupIndexGm_.GetValue(expertIdx);
                const int64_t remainingRows = rowLen_ - validRows;
                if (requestedRows > 0) {
                    validRows += requestedRows > remainingRows ? remainingRows : requestedRows;
                }
            }
            if (validRows > 0) {
                ProcessGroup(0, validRows, 0);
            }
            return;
        }

        int64_t groupOffset = 0;
        for (int64_t expertIdx = 0; expertIdx < expertNum_ && groupOffset < rowLen_; ++expertIdx) {
            const int64_t requestedRows = groupIndexGm_.GetValue(expertIdx);
            const int64_t remainingRows = rowLen_ - groupOffset;
            const int64_t groupRows = requestedRows <= 0 ? 0 :
                (requestedRows > remainingRows ? remainingRows : requestedRows);
            if (groupRows > 0) {
                ProcessGroup(expertIdx, groupRows, groupOffset);
                groupOffset += groupRows;
            }
        }
    }

private:
    __aicore__ inline void ProcessGroup(int64_t expertIdx, int64_t groupRows, int64_t groupOffset)
    {
        const int64_t rowsPerCore = (groupRows + usedCoreNum_ - 1) / usedCoreNum_;
        const int64_t localGroupOffset = blockIdx_ * rowsPerCore;
        if (localGroupOffset >= groupRows) {
            return;
        }
        const int64_t localRows =
            groupRows - localGroupOffset < rowsPerCore ? groupRows - localGroupOffset : rowsPerCore;
        const int64_t firstRow = groupOffset + localGroupOffset;

        if constexpr (std::is_same_v<XType, int32_t>) {
            CopyInExpertParams(expertIdx);
            weightScaleLocal_ = weightScaleQueue_.DeQue<float>();
            if (hasBias_) {
                biasLocal_ = biasQueue_.DeQue<float>();
            }
        }

        for (int64_t localRow = 0; localRow < localRows; ++localRow) {
            const int64_t rowIdx = firstRow + localRow;
            CopyInRow(rowIdx);
            ComputeRow(rowIdx);
            CopyOutRow(rowIdx);
        }

        if constexpr (std::is_same_v<XType, int32_t>) {
            weightScaleQueue_.FreeTensor(weightScaleLocal_);
            if (hasBias_) {
                biasQueue_.FreeTensor(biasLocal_);
            }
        }
    }

    __aicore__ inline void CopyInExpertParams(int64_t expertIdx)
    {
        const uint32_t paramBytes = static_cast<uint32_t>(inputWidth_ * sizeof(float));
        DataCopyExtParams params{1, paramBytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        const int64_t paramOffset = expertIdx * inputWidth_;

        LocalTensor<float> weightScaleLocal = weightScaleQueue_.AllocTensor<float>();
        DataCopyPad(weightScaleLocal, weightScaleGm_[paramOffset], params, padParams);
        weightScaleQueue_.EnQue(weightScaleLocal);

        if (hasBias_) {
            LocalTensor<float> biasLocal = biasQueue_.AllocTensor<float>();
            DataCopyPad(biasLocal, biasGm_[paramOffset], params, padParams);
            biasQueue_.EnQue(biasLocal);
        }
    }

    __aicore__ inline void CopyInRow(int64_t rowIdx)
    {
        const uint32_t inputBytes = static_cast<uint32_t>(inputWidth_ * sizeof(XType));
        DataCopyExtParams params{1, inputBytes, 0, 0, 0};
        DataCopyPadExtParams<XType> padParams{false, 0, 0, 0};
        LocalTensor<XType> xLocal = xQueue_.AllocTensor<XType>();
        DataCopyPad(xLocal, xGm_[rowIdx * inputWidth_], params, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeRow(int64_t rowIdx)
    {
        LocalTensor<XType> xLocal = xQueue_.DeQue<XType>();
        LocalTensor<float> xLocalF32;
        if constexpr (std::is_same_v<XType, int32_t>) {
            xLocalF32 = xLocal.template ReinterpretCast<float>();
        } else {
            xLocalF32 = dequantBuf_.Get<float>();
        }
        Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, inputWidth_);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<XType, int32_t>) {
            const float activationScale = activationScaleGm_.GetValue(rowIdx);
            Mul(xLocalF32, xLocalF32, weightScaleLocal_, inputWidth_);
            PipeBarrier<PIPE_V>();
            Muls(xLocalF32, xLocalF32, activationScale, inputWidth_);
            PipeBarrier<PIPE_V>();
            if (hasBias_) {
                Add(xLocalF32, xLocalF32, biasLocal_, inputWidth_);
                PipeBarrier<PIPE_V>();
            }
        }

        LocalTensor<float> temp = tmpBuf_.Get<float>();
        int64_t gateOffset = (activateLeft_ == 1) ? 0 : outputWidth_;
        int64_t upOffset = (activateLeft_ == 1) ? outputWidth_ : 0;

        LocalTensor<float> gate = xLocalF32[gateOffset];
        LocalTensor<float> up = xLocalF32[upOffset];
        LocalTensor<float> sigmoid = temp;
        LocalTensor<float> ones = temp[outputWidth_];

        Adds(sigmoid, gate, 0.0f, outputWidth_);
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, 1.0f / beta_, outputWidth_);
        PipeBarrier<PIPE_V>();
        Tanh(gate, gate, outputWidth_);
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, beta_, outputWidth_);
        PipeBarrier<PIPE_V>();

        Muls(sigmoid, sigmoid, -1.0f, outputWidth_);
        PipeBarrier<PIPE_V>();
        Exp(sigmoid, sigmoid, outputWidth_);
        PipeBarrier<PIPE_V>();
        Adds(sigmoid, sigmoid, 1.0f, outputWidth_);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(ones, 1.0f, outputWidth_);
        PipeBarrier<PIPE_V>();
        Div(sigmoid, ones, sigmoid, outputWidth_);
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, sigmoid, outputWidth_);
        PipeBarrier<PIPE_V>();

        if (linearBeta_ > 0.0f) {
            Muls(up, up, 1.0f / linearBeta_, outputWidth_);
            PipeBarrier<PIPE_V>();
            Tanh(up, up, outputWidth_);
            PipeBarrier<PIPE_V>();
            Muls(up, up, linearBeta_, outputWidth_);
            PipeBarrier<PIPE_V>();
        }
        Mul(gate, gate, up, outputWidth_);
        PipeBarrier<PIPE_V>();

        DynamicQuant(gate, temp);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void DynamicQuant(LocalTensor<float>& situ, LocalTensor<float>& temp)
    {
        Abs(temp, situ, outputWidth_);
        PipeBarrier<PIPE_V>();
        ComputeReduceMax(temp);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(temp, temp, K3_MASK_FP32, 1, K3_MASK_BLK_STRIDE, 1, K3_MASK_BLK_STRIDE,
                       ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();

        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        float scaleValue = temp.GetValue(0) * K3_DYNAMIC_QUANT_FACTOR;
        if (scaleValue <= 0.0f) {
            scaleValue = 1.0f;
        }
        Muls(situ, situ, 1.0f / scaleValue, outputWidth_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        LocalTensor<int8_t> yLocal = outLocal.ReinterpretCast<int8_t>();
        // Scale is packed after int8 data, aligned to float boundary
        int64_t scaleIdx = (outputWidth_ + static_cast<int64_t>(sizeof(float)) - 1) / sizeof(float);
        LocalTensor<float> scaleLocal = outLocal[scaleIdx];
        Duplicate<float>(scaleLocal, scaleValue, 1);
        PipeBarrier<PIPE_V>();
        CastFloatToInt8(situ, temp, yLocal);
        outQueue_.EnQue<float>(outLocal);
    }

    __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& temp)
    {
        const uint32_t vectorCycles = static_cast<uint32_t>(outputWidth_ / K3_MASK_FP32);
        const uint32_t remainder = static_cast<uint32_t>(outputWidth_ % K3_MASK_FP32);

        if (vectorCycles > 1) {
            BinaryRepeatParams repeatParams;
            repeatParams.dstBlkStride = 1;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = 0;
            repeatParams.src0RepStride = K3_MASK_BLK_STRIDE;
            repeatParams.src1RepStride = 0;
            Max(temp, temp[K3_MASK_FP32], temp, K3_MASK_FP32, static_cast<uint8_t>(vectorCycles - 1), repeatParams);
            PipeBarrier<PIPE_V>();
        }
        if (remainder > 0 && vectorCycles > 0) {
            Max(temp, temp[vectorCycles * K3_MASK_FP32], temp, remainder, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = vectorCycles > 0 ? K3_MASK_FP32 : outputWidth_;
        WholeReduceMax(temp, temp, mask, 1, K3_MASK_BLK_STRIDE, 1, K3_MASK_BLK_STRIDE,
                       ReduceOrder::ORDER_ONLY_VALUE);
    }

    __aicore__ inline void CastFloatToInt8(
        const LocalTensor<float>& src, LocalTensor<float>& temp, LocalTensor<int8_t>& dst)
    {
        LocalTensor<int32_t> tempInt32 = temp[outputWidth_].ReinterpretCast<int32_t>();
        Cast(tempInt32, src, RoundMode::CAST_RINT, outputWidth_);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.0f);

        LocalTensor<half> tempHalf = temp.ReinterpretCast<half>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, outputWidth_);
        PipeBarrier<PIPE_V>();
        Cast(dst, tempHalf, RoundMode::CAST_TRUNC, outputWidth_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutRow(int64_t rowIdx)
    {
        LocalTensor<float> outLocal = outQueue_.DeQue<float>();
        LocalTensor<int8_t> yLocal = outLocal.ReinterpretCast<int8_t>();
        int64_t scaleIdx = (outputWidth_ + static_cast<int64_t>(sizeof(float)) - 1) / sizeof(float);
        LocalTensor<float> scaleLocal = outLocal[scaleIdx];

        DataCopyExtParams yParams{1, static_cast<uint32_t>(outputWidth_ * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(yGm_[rowIdx * outputWidth_], yLocal, yParams);
        DataCopyExtParams scaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(scaleGm_[rowIdx], scaleLocal, scaleParams);
        outQueue_.FreeTensor(outLocal);
    }

    TPipe* pipe_ = nullptr;
    const DequantSituQuantTilingData* tilingData_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t rowLen_ = 0;
    int64_t inputWidth_ = 0;
    int64_t outputWidth_ = 0;
    int64_t expertNum_ = 0;
    int64_t usedCoreNum_ = 0;
    uint32_t activateLeft_ = 1;
    bool hasBias_ = false;
    bool hasGroupIndex_ = false;
    float beta_ = 4.0f;
    float linearBeta_ = 25.0f;

    GlobalTensor<XType> xGm_;
    GlobalTensor<float> weightScaleGm_;
    GlobalTensor<float> activationScaleGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> weightScaleQueue_;
    TQue<QuePosition::VECIN, 1> biasQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> dequantBuf_;
    LocalTensor<float> weightScaleLocal_;
    LocalTensor<float> biasLocal_;
};

} // namespace DequantSituQuantOps
#endif // DEQUANT_SITU_QUANT_H
