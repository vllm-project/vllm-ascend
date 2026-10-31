/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#pragma once

#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif
#endif

namespace KdaGateCumsum {

using namespace AscendC;

constexpr float RCP_LN2 = 1.4426950408889634f;
constexpr uint32_t GATE_ROW_ELEMENTS = 256;
constexpr uint32_t GATE_PIPELINE_DEPTH = 2;
constexpr uint32_t GATE_BULK_ROWS = 64;
constexpr uint32_t GATE_BULK_COLS = 128;
constexpr uint32_t GATE_BULK_ELEMENTS = GATE_BULK_ROWS * GATE_BULK_COLS;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
static __simd_vf__ inline void AccumulateGateRowRegbase(__ubuf__ float *input, __ubuf__ float *acc,
                                                        __ubuf__ float *output, uint16_t count)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ELEMENTS_PER_PAIR = 2 * FLOAT_ELEMENTS_PER_REG;

    for (uint16_t offset = 0; offset < count; offset += ELEMENTS_PER_PAIR) {
        RegTensor<float> inputZeroReg;
        RegTensor<float> inputOneReg;
        RegTensor<float> accZeroReg;
        RegTensor<float> accOneReg;
        MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();

        LoadAlign<float, LoadDist::DIST_NORM>(inputZeroReg, input + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(inputOneReg, input + offset + FLOAT_ELEMENTS_PER_REG);
        LoadAlign<float, LoadDist::DIST_NORM>(accZeroReg, acc + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(accOneReg, acc + offset + FLOAT_ELEMENTS_PER_REG);

        Muls(inputZeroReg, inputZeroReg, RCP_LN2, floatMask);
        Muls(inputOneReg, inputOneReg, RCP_LN2, floatMask);
        Add(accZeroReg, accZeroReg, inputZeroReg, floatMask);
        Add(accOneReg, accOneReg, inputOneReg, floatMask);

        StoreAlign(acc + offset, accZeroReg, floatMask);
        StoreAlign(acc + offset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
        StoreAlign(output + offset, accZeroReg, floatMask);
        StoreAlign(output + offset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
    }
}

static __simd_vf__ inline void AccumulateGateChunk128Regbase(__ubuf__ float *input,
                                                              __ubuf__ float *output,
                                                              uint16_t rows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ROW_ELEMENTS = 2 * FLOAT_ELEMENTS_PER_REG;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> accZeroReg;
    RegTensor<float> accOneReg;
    Duplicate(accZeroReg, 0.0f, floatMask);
    Duplicate(accOneReg, 0.0f, floatMask);
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        RegTensor<float> inputZeroReg;
        RegTensor<float> inputOneReg;
        LoadAlign<float, LoadDist::DIST_NORM>(inputZeroReg, input + rowOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            inputOneReg, input + rowOffset + FLOAT_ELEMENTS_PER_REG);
        Muls(inputZeroReg, inputZeroReg, RCP_LN2, floatMask);
        Muls(inputOneReg, inputOneReg, RCP_LN2, floatMask);
        Add(accZeroReg, accZeroReg, inputZeroReg, floatMask);
        Add(accOneReg, accOneReg, inputOneReg, floatMask);
        StoreAlign(output + rowOffset, accZeroReg, floatMask);
        StoreAlign(output + rowOffset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
    }
}

template <bool HAS_BIAS>
static __simd_vf__ inline void AccumulateSafeGateChunk128Regbase(
    __ubuf__ float *input, __ubuf__ float *bias, __ubuf__ float *output,
    uint16_t rows, float expA, float lowerBound)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ROW_ELEMENTS = 2 * FLOAT_ELEMENTS_PER_REG;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> accZeroReg;
    RegTensor<float> accOneReg;
    RegTensor<float> oneZeroReg;
    RegTensor<float> oneOneReg;
    RegTensor<float> biasZeroReg;
    RegTensor<float> biasOneReg;
    Duplicate(accZeroReg, 0.0f, floatMask);
    Duplicate(accOneReg, 0.0f, floatMask);
    Duplicate(oneZeroReg, 1.0f, floatMask);
    Duplicate(oneOneReg, 1.0f, floatMask);
    if constexpr (HAS_BIAS) {
        LoadAlign<float, LoadDist::DIST_NORM>(biasZeroReg, bias);
        LoadAlign<float, LoadDist::DIST_NORM>(biasOneReg, bias + FLOAT_ELEMENTS_PER_REG);
    }

    const float gateScale = lowerBound * RCP_LN2;
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        RegTensor<float> gateZeroReg;
        RegTensor<float> gateOneReg;
        RegTensor<float> sigmoidZeroReg;
        RegTensor<float> sigmoidOneReg;
        LoadAlign<float, LoadDist::DIST_NORM>(gateZeroReg, input + rowOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(
            gateOneReg, input + rowOffset + FLOAT_ELEMENTS_PER_REG);
        if constexpr (HAS_BIAS) {
            Add(gateZeroReg, gateZeroReg, biasZeroReg, floatMask);
            Add(gateOneReg, gateOneReg, biasOneReg, floatMask);
        }
        Muls(gateZeroReg, gateZeroReg, -expA, floatMask);
        Muls(gateOneReg, gateOneReg, -expA, floatMask);
        Exp(gateZeroReg, gateZeroReg, floatMask);
        Exp(gateOneReg, gateOneReg, floatMask);
        Adds(gateZeroReg, gateZeroReg, 1.0f, floatMask);
        Adds(gateOneReg, gateOneReg, 1.0f, floatMask);
        Div(sigmoidZeroReg, oneZeroReg, gateZeroReg, floatMask);
        Div(sigmoidOneReg, oneOneReg, gateOneReg, floatMask);
        Muls(sigmoidZeroReg, sigmoidZeroReg, gateScale, floatMask);
        Muls(sigmoidOneReg, sigmoidOneReg, gateScale, floatMask);
        Add(accZeroReg, accZeroReg, sigmoidZeroReg, floatMask);
        Add(accOneReg, accOneReg, sigmoidOneReg, floatMask);
        StoreAlign(output + rowOffset, accZeroReg, floatMask);
        StoreAlign(output + rowOffset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
    }
}

#endif

template <typename T, bool USE_GATE_IN_KERNEL, bool SAFE_GATE>
class KdaGateCumsumKernel {
public:
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR gk,
                                const TilingData &tiling, TPipe *pipe)
    {
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(g));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        dtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        cuSeqlens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        pipe_ = pipe;
        batch_ = static_cast<uint64_t>(tiling.batch);
        t_ = static_cast<uint64_t>(tiling.t);
        hv_ = static_cast<uint64_t>(tiling.hv);
        k_ = static_cast<uint64_t>(tiling.k);
        rank_ = static_cast<uint64_t>(tiling.rank);
        chunkSize_ = static_cast<uint64_t>(tiling.chunkSize);
        seqNum_ = static_cast<uint64_t>(tiling.seqNum);
        hasCuSeqlens_ = tiling.hasCuSeqlens != 0;
        hasALog_ = tiling.hasALog != 0;
        hasDtBias_ = tiling.hasDtBias != 0;
        inputSequenceMajor_ = tiling.inputSequenceMajor != 0;
        lowerBound_ = tiling.lowerBound;
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        maxChunks_ = (t_ + chunkSize_ - 1) / chunkSize_;
        pipe_->InitBuffer(rowBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(accBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(outBuf_, GATE_PIPELINE_DEPTH * GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(oneBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(biasBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(inBuf_, GATE_PIPELINE_DEPTH * GATE_ROW_ELEMENTS * sizeof(T));
        pipe_->InitBuffer(chunkBuf_, 2 * GATE_BULK_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 32);
        pipe_->InitBuffer(scalarI64Buf_, 32);
        AllocEvents();
    }

    __aicore__ inline void Process()
    {
        uint64_t taskCount = hasCuSeqlens_ ? seqNum_ * hv_ : batch_ * hv_ * maxChunks_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t task = coreIdx; task < taskCount; task += usedCoreNum_) {
            ProcessTask(task);
        }
        ReleaseEvents();
    }

private:
    __aicore__ inline void AllocEvents()
    {
        for (uint32_t slot = 0; slot < GATE_PIPELINE_DEPTH; ++slot) {
            inputMte2ToVEvent_[slot] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            inputVToMte2Event_[slot] = pipe_->AllocEventID<HardEvent::V_MTE2>();
            outputVToMte3Event_[slot] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            outputMte3ToVEvent_[slot] = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
        auxMte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        scalarVToSEvent_ = pipe_->AllocEventID<HardEvent::V_S>();
        bulkMte3ToMte2Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void ReleaseEvents()
    {
        for (uint32_t slot = 0; slot < GATE_PIPELINE_DEPTH; ++slot) {
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(inputVToMte2Event_[slot]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(outputVToMte3Event_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputMte3ToVEvent_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(auxMte2ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::V_S>(scalarVToSEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
    }

    __aicore__ inline uint64_t InputOffset(uint64_t b, uint64_t t, uint64_t hv, uint64_t k) const
    {
        if (inputSequenceMajor_) {
            if (rank_ == 4) {
                return ((b * t_ + t) * hv_ + hv) * k_ + k;
            }
            return (t * hv_ + hv) * k_ + k;
        }
        if (rank_ == 4) {
            return ((b * hv_ + hv) * t_ + t) * k_ + k;
        }
        return (hv * t_ + t) * k_ + k;
    }

    __aicore__ inline uint64_t OutputOffset(uint64_t b, uint64_t t, uint64_t hv, uint64_t k) const
    {
        if (rank_ == 4) {
            return ((b * hv_ + hv) * t_ + t) * k_ + k;
        }
        return (hv * t_ + t) * k_ + k;
    }

    __aicore__ inline void CopyVectorIn(LocalTensor<T> &dst, GlobalTensor<T> &src, uint64_t offset, uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(T));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(dst, src[offset], params, padParams);
        }
    }

    __aicore__ inline void CopyFloatVectorIn(LocalTensor<float> &dst, GlobalTensor<float> &src, uint64_t offset,
                                             uint64_t count)
    {
        uint64_t rowBytes = count * sizeof(float);
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopyPad(dst, src[offset], params, padParams);
        }
    }

    __aicore__ inline void CopyFloatVectorOut(GlobalTensor<float> &dst, uint64_t offset, LocalTensor<float> &src,
                                              uint64_t count)
    {
        uint64_t rowBytes = count * sizeof(float);
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst[offset], src, static_cast<uint32_t>(count));
        } else {
            DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
            DataCopyPad(dst[offset], src, params);
        }
    }

    __aicore__ inline void CopyGateRowsIn(LocalTensor<T> &dst, uint64_t b, uint64_t start,
                                          uint64_t hv, uint64_t rows)
    {
        if (!inputSequenceMajor_) {
            CopyVectorIn(dst, g_, InputOffset(b, start, hv, 0), rows * k_);
            return;
        }
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(k_ * sizeof(T)),
            static_cast<uint32_t>((hv_ - 1) * k_ * sizeof(T)),
            0,
            0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dst, g_[InputOffset(b, start, hv, 0)], params, padParams);
    }

    __aicore__ inline void PrefetchGateRow(uint64_t offset, uint32_t slot)
    {
        LocalTensor<T> input = inBuf_.Get<T>()[slot * GATE_ROW_ELEMENTS];
        CopyVectorIn(input, g_, offset, k_);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot]);
    }

    __aicore__ inline void MaterializeGateRow(uint32_t slot, LocalTensor<float> &row)
    {
        LocalTensor<T> input = inBuf_.Get<T>()[slot * GATE_ROW_ELEMENTS];
        if constexpr (IsSameType<T, float>::value) {
            Adds(row, input, 0.0f, static_cast<uint32_t>(k_));
        } else {
            Cast(row, input, RoundMode::CAST_NONE, static_cast<uint32_t>(k_));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ReadFloat(GlobalTensor<float> &tensor, uint64_t offset)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
        Adds(scalar, scalar, 0.0f, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(scalarVToSEvent_);
        WaitFlag<HardEvent::V_S>(scalarVToSEvent_);
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline int64_t ReadInt64(GlobalTensor<int64_t> &tensor, uint64_t offset)
    {
        LocalTensor<int64_t> scalar = scalarI64Buf_.Get<int64_t>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(int64_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
        SetFlag<HardEvent::V_S>(scalarVToSEvent_);
        WaitFlag<HardEvent::V_S>(scalarVToSEvent_);
        __ubuf__ int64_t *ptr = (__ubuf__ int64_t *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline float ExpScalar(float x)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        Duplicate(scalar, x, 1);
        PipeBarrier<PIPE_V>();
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(scalarVToSEvent_);
        WaitFlag<HardEvent::V_S>(scalarVToSEvent_);
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline void PrepareGate(uint64_t hv)
    {
        if constexpr (USE_GATE_IN_KERNEL) {
            expA_ = ExpScalar(ReadFloat(aLog_, hv));
            if (hasDtBias_) {
                LocalTensor<float> bias = biasBuf_.Get<float>();
                CopyFloatVectorIn(bias, dtBias_, hv * k_, k_);
                SetFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
            }
        }
    }

    __aicore__ inline void ApplyGate(LocalTensor<float> &row)
    {
        if constexpr (USE_GATE_IN_KERNEL) {
            if (hasDtBias_) {
                LocalTensor<float> bias = biasBuf_.Get<float>();
                Add(row, row, bias, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
            }
            if constexpr (SAFE_GATE) {
                Muls(row, row, expA_, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();

                LocalTensor<float> tmp = tmpBuf_.Get<float>();
                Muls(tmp, row, -1.0f, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Exp(tmp, tmp, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Adds(tmp, tmp, 1.0f, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();

                LocalTensor<float> one = oneBuf_.Get<float>();
                Duplicate(one, 1.0f, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Div(row, one, tmp, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Muls(row, row, lowerBound_, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> positive = oneBuf_.Get<float>();
                LocalTensor<float> tmp = tmpBuf_.Get<float>();
                Maxs(positive, row, 0.0f, static_cast<uint32_t>(k_));
                Abs(tmp, row, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Muls(tmp, tmp, -1.0f, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Exp(tmp, tmp, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Adds(tmp, tmp, 1.0f, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Ln(tmp, tmp, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Add(row, positive, tmp, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Muls(row, row, -expA_, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        uint64_t hv = hasCuSeqlens_ ? task % hv_ : (task / maxChunks_) % hv_;
        PrepareGate(hv);
        if (!hasCuSeqlens_) {
            uint64_t chunk = task % maxChunks_;
            uint64_t b = task / (maxChunks_ * hv_);
            uint64_t start = chunk * chunkSize_;
            uint64_t end = start + chunkSize_;
            if (end > t_) {
                end = t_;
            }
            ProcessChunk(b, hv, start, end);
            return;
        }
        uint64_t seq = task / hv_;
        uint64_t seqStart = static_cast<uint64_t>(ReadInt64(cuSeqlens_, seq));
        uint64_t seqEnd = static_cast<uint64_t>(ReadInt64(cuSeqlens_, seq + 1));
        for (uint64_t start = seqStart; start < seqEnd; start += chunkSize_) {
            uint64_t end = start + chunkSize_;
            if (end > seqEnd) {
                end = seqEnd;
            }
            ProcessChunk(0, hv, start, end);
        }
    }

    __aicore__ inline void ProcessChunk(uint64_t b, uint64_t hv, uint64_t start, uint64_t end)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        if constexpr (USE_GATE_IN_KERNEL && SAFE_GATE && IsSameType<T, float>::value) {
            if (k_ == GATE_BULK_COLS && chunkSize_ == GATE_BULK_ROWS && end - start == GATE_BULK_ROWS) {
                ProcessChunkBulkFp32SafeVec(b, hv, start);
                return;
            }
        }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (USE_GATE_IN_KERNEL && SAFE_GATE && IsSameType<T, float>::value) {
            if (k_ == GATE_BULK_COLS && chunkSize_ == GATE_BULK_ROWS) {
                ProcessChunkBulkFp32Safe(b, hv, start, end);
                return;
            }
        }
        if constexpr (!USE_GATE_IN_KERNEL && IsSameType<T, float>::value) {
            if (k_ == GATE_BULK_COLS && chunkSize_ == GATE_BULK_ROWS) {
                ProcessChunkBulkFp32(b, hv, start, end);
                return;
            }
        }
#endif
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> row = rowBuf_.Get<float>();
        Duplicate(acc, 0.0f, static_cast<uint32_t>(k_));
        PipeBarrier<PIPE_V>();
        uint64_t rows = end - start;
        if (rows == 0) {
            return;
        }

        PrefetchGateRow(InputOffset(b, start, hv, 0), 0);

        for (uint64_t rowIdx = 0; rowIdx < rows; ++rowIdx) {
            uint64_t token = start + rowIdx;
            uint32_t slot = static_cast<uint32_t>(rowIdx & 1);
            WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot]);

            if (rowIdx + 1 < rows) {
                uint32_t nextSlot = slot ^ 1;
                if (rowIdx >= 1) {
                    WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_[nextSlot]);
                }
                PrefetchGateRow(InputOffset(b, token + 1, hv, 0), nextSlot);
            }

            uint32_t outputSlot = slot;
            if (rowIdx >= GATE_PIPELINE_DEPTH) {
                WaitFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[outputSlot]);
            }
            LocalTensor<float> output =
                outBuf_.Get<float>()[outputSlot * GATE_ROW_ELEMENTS];

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!USE_GATE_IN_KERNEL && IsSameType<T, float>::value) {
                if ((k_ % 128) == 0) {
                    LocalTensor<float> input = inBuf_.Get<float>()[slot * GATE_ROW_ELEMENTS];
                    AccumulateGateRowRegbase(
                        (__ubuf__ float *)input.GetPhyAddr(), (__ubuf__ float *)acc.GetPhyAddr(),
                        (__ubuf__ float *)output.GetPhyAddr(), static_cast<uint16_t>(k_));
                    PipeBarrier<PIPE_V>();
                } else {
                    MaterializeGateRow(slot, row);
                    Muls(row, row, RCP_LN2, static_cast<uint32_t>(k_));
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, row, static_cast<uint32_t>(k_));
                    PipeBarrier<PIPE_V>();
                    Adds(output, acc, 0.0f, static_cast<uint32_t>(k_));
                }
            } else {
                MaterializeGateRow(slot, row);
                ApplyGate(row);
                Muls(row, row, RCP_LN2, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Add(acc, acc, row, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
                Adds(output, acc, 0.0f, static_cast<uint32_t>(k_));
            }
#else
            MaterializeGateRow(slot, row);
            ApplyGate(row);
            Muls(row, row, RCP_LN2, static_cast<uint32_t>(k_));
            PipeBarrier<PIPE_V>();
            Add(acc, acc, row, static_cast<uint32_t>(k_));
            PipeBarrier<PIPE_V>();
            Adds(output, acc, 0.0f, static_cast<uint32_t>(k_));
#endif

            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_[slot]);
            SetFlag<HardEvent::V_MTE3>(outputVToMte3Event_[outputSlot]);
            WaitFlag<HardEvent::V_MTE3>(outputVToMte3Event_[outputSlot]);
            CopyFloatVectorOut(gk_, OutputOffset(b, token, hv, 0), output, k_);
            SetFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[outputSlot]);
        }

        uint64_t drainStart = rows > GATE_PIPELINE_DEPTH ? rows - GATE_PIPELINE_DEPTH : 0;
        for (uint64_t rowIdx = drainStart; rowIdx < rows; ++rowIdx) {
            uint32_t outputSlot = static_cast<uint32_t>(rowIdx & 1);
            WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_[outputSlot]);
            WaitFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[outputSlot]);
        }
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    __aicore__ inline void ScanGateChunk64x128(LocalTensor<float> input, LocalTensor<float> output)
    {
        constexpr uint32_t rowElements = GATE_BULK_COLS;
        Adds(input, output, 0.0f, rowElements);
        Add(input[rowElements], output[rowElements], output, (GATE_BULK_ROWS - 1) * rowElements);
        PipeBarrier<PIPE_V>();

        Adds(output, input, 0.0f, 2 * rowElements);
        Add(output[2 * rowElements], input[2 * rowElements], input, (GATE_BULK_ROWS - 2) * rowElements);
        PipeBarrier<PIPE_V>();

        Adds(input, output, 0.0f, 4 * rowElements);
        Add(input[4 * rowElements], output[4 * rowElements], output, (GATE_BULK_ROWS - 4) * rowElements);
        PipeBarrier<PIPE_V>();

        Adds(output, input, 0.0f, 8 * rowElements);
        Add(output[8 * rowElements], input[8 * rowElements], input, (GATE_BULK_ROWS - 8) * rowElements);
        PipeBarrier<PIPE_V>();

        Adds(input, output, 0.0f, 16 * rowElements);
        Add(input[16 * rowElements], output[16 * rowElements], output, (GATE_BULK_ROWS - 16) * rowElements);
        PipeBarrier<PIPE_V>();

        Adds(output, input, 0.0f, 32 * rowElements);
        Add(output[32 * rowElements], input[32 * rowElements], input, (GATE_BULK_ROWS - 32) * rowElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessChunkBulkFp32SafeVec(uint64_t b, uint64_t hv, uint64_t start)
    {
        constexpr uint32_t elems = GATE_BULK_ELEMENTS;
        LocalTensor<float> input = chunkBuf_.Get<float>();
        LocalTensor<float> output = chunkBuf_.Get<float>()[GATE_BULK_ELEMENTS];
        CopyGateRowsIn(input, b, start, hv, GATE_BULK_ROWS);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[0]);
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[0]);

        Duplicate(output, 1.0f, elems);
        if (hasDtBias_) {
            LocalTensor<float> bias = biasBuf_.Get<float>();
            for (uint32_t row = 0; row < GATE_BULK_ROWS; ++row) {
                Add(input[row * GATE_BULK_COLS], input[row * GATE_BULK_COLS], bias, GATE_BULK_COLS);
            }
            PipeBarrier<PIPE_V>();
        }
        Muls(input, input, -expA_, elems);
        PipeBarrier<PIPE_V>();
        Exp(input, input, elems);
        PipeBarrier<PIPE_V>();
        Adds(input, input, 1.0f, elems);
        PipeBarrier<PIPE_V>();
        Div(output, output, input, elems);
        PipeBarrier<PIPE_V>();
        Muls(output, output, lowerBound_ * RCP_LN2, elems);
        PipeBarrier<PIPE_V>();
        ScanGateChunk64x128(input, output);

        SetFlag<HardEvent::V_MTE3>(outputVToMte3Event_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputVToMte3Event_[0]);
        CopyFloatVectorOut(gk_, OutputOffset(b, start, hv, 0), output, elems);
        SetFlag<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[0]);
        WaitFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[0]);
    }
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessChunkBulkFp32Safe(uint64_t b, uint64_t hv, uint64_t start, uint64_t end)
    {
        uint64_t rows = end - start;
        if (rows == 0) {
            return;
        }
        uint32_t elems = static_cast<uint32_t>(rows * k_);
        LocalTensor<float> input = chunkBuf_.Get<float>();
        LocalTensor<float> output = chunkBuf_.Get<float>()[GATE_BULK_ELEMENTS];
        CopyGateRowsIn(input, b, start, hv, rows);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[0]);
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[0]);
        LocalTensor<float> bias = biasBuf_.Get<float>();
        if (hasDtBias_) {
            AccumulateSafeGateChunk128Regbase<true>(
                (__ubuf__ float *)input.GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                (__ubuf__ float *)output.GetPhyAddr(), static_cast<uint16_t>(rows), expA_, lowerBound_);
        } else {
            AccumulateSafeGateChunk128Regbase<false>(
                (__ubuf__ float *)input.GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                (__ubuf__ float *)output.GetPhyAddr(), static_cast<uint16_t>(rows), expA_, lowerBound_);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(outputVToMte3Event_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputVToMte3Event_[0]);
        CopyFloatVectorOut(gk_, OutputOffset(b, start, hv, 0), output, elems);
        SetFlag<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[0]);
        WaitFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[0]);
    }

    __aicore__ inline void ProcessChunkBulkFp32(uint64_t b, uint64_t hv, uint64_t start, uint64_t end)
    {
        uint64_t rows = end - start;
        if (rows == 0) {
            return;
        }
        uint32_t elems = static_cast<uint32_t>(rows * k_);
        LocalTensor<float> buffer0 = chunkBuf_.Get<float>();
        CopyGateRowsIn(buffer0, b, start, hv, rows);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[0]);
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[0]);
        AccumulateGateChunk128Regbase(
            (__ubuf__ float *)buffer0.GetPhyAddr(), (__ubuf__ float *)buffer0.GetPhyAddr(),
            static_cast<uint16_t>(rows));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(outputVToMte3Event_[0]);
        WaitFlag<HardEvent::V_MTE3>(outputVToMte3Event_[0]);
        CopyFloatVectorOut(gk_, OutputOffset(b, start, hv, 0), buffer0, elems);
        SetFlag<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(bulkMte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[0]);
        WaitFlag<HardEvent::MTE3_V>(outputMte3ToVEvent_[0]);
    }
#endif

    GlobalTensor<T> g_;
    GlobalTensor<float> aLog_;
    GlobalTensor<float> dtBias_;
    GlobalTensor<int64_t> cuSeqlens_;
    GlobalTensor<float> gk_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rowBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> oneBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> scalarI64Buf_;
    TEventID inputMte2ToVEvent_[GATE_PIPELINE_DEPTH];
    TEventID inputVToMte2Event_[GATE_PIPELINE_DEPTH];
    TEventID outputVToMte3Event_[GATE_PIPELINE_DEPTH];
    TEventID outputMte3ToVEvent_[GATE_PIPELINE_DEPTH];
    TEventID auxMte2ToVEvent_;
    TEventID scalarVToSEvent_;
    TEventID bulkMte3ToMte2Event_;
    uint64_t batch_ = 0;
    uint64_t t_ = 0;
    uint64_t hv_ = 0;
    uint64_t k_ = 0;
    uint64_t rank_ = 0;
    uint64_t chunkSize_ = 0;
    uint64_t seqNum_ = 0;
    uint64_t maxChunks_ = 0;
    bool hasCuSeqlens_ = false;
    bool hasALog_ = false;
    bool hasDtBias_ = false;
    bool inputSequenceMajor_ = false;
    float expA_ = 1.0f;
    float lowerBound_ = -5.0f;
    uint64_t usedCoreNum_ = 1;
};

template <typename T, bool USE_GATE_IN_KERNEL, bool SAFE_GATE, typename TilingData>
__aicore__ inline void RunKdaGateCumsum(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR gk,
                                        const TilingData &tilingData, TPipe *pipe)
{
    KdaGateCumsumKernel<T, USE_GATE_IN_KERNEL, SAFE_GATE> op;
    op.Init(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
    op.Process();
}

template <typename T, typename TilingData>
__aicore__ inline void DispatchKdaGateCumsum(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens,
                                             GM_ADDR gk, const TilingData &tilingData, TPipe *pipe)
{
    if (tilingData.useGateInKernel != 0) {
        if (tilingData.safeGate != 0) {
            RunKdaGateCumsum<T, true, true>(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
        } else {
            RunKdaGateCumsum<T, true, false>(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
        }
    } else {
        RunKdaGateCumsum<T, false, false>(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
    }
}
} // namespace KdaGateCumsum
