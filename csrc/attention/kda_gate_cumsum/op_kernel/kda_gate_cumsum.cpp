/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float RCP_LN2 = 1.4426950408889634f;
constexpr uint32_t GATE_MTE2_V_EVENT_ID = 0;
constexpr uint32_t GATE_V_MTE3_EVENT_ID = 1;
constexpr uint32_t GATE_MTE3_MTE2_EVENT_ID = 2;
constexpr uint32_t GATE_SCALAR_MTE2_V_EVENT_ID = 3;
constexpr uint32_t GATE_SCALAR_V_S_EVENT_ID = 4;
constexpr uint32_t GATE_MTE3_V_EVENT_ID = 5;
constexpr uint32_t GATE_ROW_ELEMENTS = 256;

template <typename T, bool SAFE_GATE>
class KdaGateCumsumKernel {
public:
    __aicore__ inline void Init(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR gk,
                                const KdaGateCumsumTilingData &tiling, TPipe *pipe)
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
        layout_ = static_cast<uint64_t>(tiling.layout);
        chunkSize_ = static_cast<uint64_t>(tiling.chunkSize);
        seqNum_ = static_cast<uint64_t>(tiling.seqNum);
        hasCuSeqlens_ = tiling.hasCuSeqlens != 0;
        hasALog_ = tiling.hasALog != 0;
        hasDtBias_ = tiling.hasDtBias != 0;
        lowerBound_ = tiling.lowerBound;
        usedCoreNum_ = static_cast<uint64_t>(tiling.usedCoreNum);
        maxChunks_ = (t_ + chunkSize_ - 1) / chunkSize_;
        pipe_->InitBuffer(rowBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(accBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(oneBuf_, GATE_ROW_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(inBuf_, GATE_ROW_ELEMENTS * sizeof(T));
        pipe_->InitBuffer(scalarBuf_, 32);
        pipe_->InitBuffer(scalarI64Buf_, 32);
    }

    __aicore__ inline void Process()
    {
        uint64_t taskCount = hasCuSeqlens_ ? seqNum_ * hv_ : batch_ * hv_ * maxChunks_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t task = coreIdx; task < taskCount; task += usedCoreNum_) {
            ProcessTask(task);
        }
    }

private:
    __aicore__ inline uint64_t Offset(uint64_t b, uint64_t t, uint64_t hv, uint64_t k) const
    {
        if (layout_ == 1) {
            return ((b * hv_ + hv) * t_ + t) * k_ + k;
        }
        if (layout_ == 3) {
            return (hv * t_ + t) * k_ + k;
        }
        if (rank_ == 4) {
            return ((b * t_ + t) * hv_ + hv) * k_ + k;
        }
        return (t * hv_ + hv) * k_ + k;
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

    __aicore__ inline void LoadGateRow(uint64_t offset, LocalTensor<float> &row)
    {
        if constexpr (IsSameType<T, float>::value) {
            CopyVectorIn(row, g_, offset, k_);
        } else {
            LocalTensor<T> inLocal = inBuf_.Get<T>();
            CopyVectorIn(inLocal, g_, offset, k_);
            SetFlag<HardEvent::MTE2_V>(GATE_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(GATE_MTE2_V_EVENT_ID);
            Cast(row, inLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(k_));
            PipeBarrier<PIPE_V>();
            return;
        }
        SetFlag<HardEvent::MTE2_V>(GATE_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(GATE_MTE2_V_EVENT_ID);
        Adds(row, row, 0.0f, static_cast<uint32_t>(k_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ReadFloat(GlobalTensor<float> &tensor, uint64_t offset)
    {
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(GATE_SCALAR_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(GATE_SCALAR_MTE2_V_EVENT_ID);
        Adds(scalar, scalar, 0.0f, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(GATE_SCALAR_V_S_EVENT_ID);
        WaitFlag<HardEvent::V_S>(GATE_SCALAR_V_S_EVENT_ID);
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline int64_t ReadInt64(GlobalTensor<int64_t> &tensor, uint64_t offset)
    {
        LocalTensor<int64_t> scalar = scalarI64Buf_.Get<int64_t>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(int64_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalar, tensor[offset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(GATE_SCALAR_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(GATE_SCALAR_MTE2_V_EVENT_ID);
        SetFlag<HardEvent::V_S>(GATE_SCALAR_V_S_EVENT_ID);
        WaitFlag<HardEvent::V_S>(GATE_SCALAR_V_S_EVENT_ID);
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
        SetFlag<HardEvent::V_S>(GATE_SCALAR_V_S_EVENT_ID);
        WaitFlag<HardEvent::V_S>(GATE_SCALAR_V_S_EVENT_ID);
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline void ApplyGate(uint64_t hv, LocalTensor<float> &row)
    {
        if constexpr (SAFE_GATE) {
            if (hasDtBias_) {
                LocalTensor<float> tmp = tmpBuf_.Get<float>();
                CopyFloatVectorIn(tmp, dtBias_, hv * k_, k_);
                SetFlag<HardEvent::MTE2_V>(GATE_MTE2_V_EVENT_ID);
                WaitFlag<HardEvent::MTE2_V>(GATE_MTE2_V_EVENT_ID);
                Add(row, row, tmp, static_cast<uint32_t>(k_));
                PipeBarrier<PIPE_V>();
            }

            float expA = hasALog_ ? ExpScalar(ReadFloat(aLog_, hv)) : 1.0f;
            Muls(row, row, expA, static_cast<uint32_t>(k_));
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
        }
    }

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        if (!hasCuSeqlens_) {
            uint64_t chunk = task % maxChunks_;
            uint64_t hv = (task / maxChunks_) % hv_;
            uint64_t b = task / (maxChunks_ * hv_);
            uint64_t start = chunk * chunkSize_;
            uint64_t end = start + chunkSize_;
            if (end > t_) {
                end = t_;
            }
            ProcessChunk(b, hv, start, end);
            return;
        }
        uint64_t hv = task % hv_;
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
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> row = rowBuf_.Get<float>();
        Duplicate(acc, 0.0f, static_cast<uint32_t>(k_));
        PipeBarrier<PIPE_V>();
        for (uint64_t t = start; t < end; ++t) {
            LoadGateRow(Offset(b, t, hv, 0), row);
            ApplyGate(hv, row);
            Muls(row, row, RCP_LN2, static_cast<uint32_t>(k_));
            PipeBarrier<PIPE_V>();
            Add(acc, acc, row, static_cast<uint32_t>(k_));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(GATE_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(GATE_V_MTE3_EVENT_ID);
            CopyFloatVectorOut(gk_, Offset(b, t, hv, 0), acc, k_);
            SetFlag<HardEvent::MTE3_MTE2>(GATE_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::MTE3_MTE2>(GATE_MTE3_MTE2_EVENT_ID);
            SetFlag<HardEvent::MTE3_V>(GATE_MTE3_V_EVENT_ID);
            WaitFlag<HardEvent::MTE3_V>(GATE_MTE3_V_EVENT_ID);
        }
    }

    GlobalTensor<T> g_;
    GlobalTensor<float> aLog_;
    GlobalTensor<float> dtBias_;
    GlobalTensor<int64_t> cuSeqlens_;
    GlobalTensor<float> gk_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rowBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> oneBuf_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> scalarI64Buf_;
    uint64_t batch_ = 0;
    uint64_t t_ = 0;
    uint64_t hv_ = 0;
    uint64_t k_ = 0;
    uint64_t rank_ = 0;
    uint64_t layout_ = 0;
    uint64_t chunkSize_ = 0;
    uint64_t seqNum_ = 0;
    uint64_t maxChunks_ = 0;
    bool hasCuSeqlens_ = false;
    bool hasALog_ = false;
    bool hasDtBias_ = false;
    float lowerBound_ = -5.0f;
    uint64_t usedCoreNum_ = 1;
};

template <typename T, bool SAFE_GATE>
__aicore__ inline void RunKdaGateCumsum(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR gk,
                                        const KdaGateCumsumTilingData &tilingData, TPipe *pipe)
{
    KdaGateCumsumKernel<T, SAFE_GATE> op;
    op.Init(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
    op.Process();
}

template <typename T>
__aicore__ inline void DispatchKdaGateCumsumBySafeGate(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens,
                                                       GM_ADDR gk, const KdaGateCumsumTilingData &tilingData,
                                                       TPipe *pipe)
{
    if (tilingData.safeGate != 0) {
        RunKdaGateCumsum<T, true>(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
    } else {
        RunKdaGateCumsum<T, false>(g, aLog, dtBias, cuSeqlens, gk, tilingData, pipe);
    }
}
} // namespace

extern "C" __global__ __aicore__ void kda_gate_cumsum(GM_ADDR g, GM_ADDR aLog, GM_ADDR dtBias,
                                                       GM_ADDR cuSeqlens, GM_ADDR gk, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (tilingData.dataType == 2) {
        DispatchKdaGateCumsumBySafeGate<float>(g, aLog, dtBias, cuSeqlens, gk, tilingData, &pipe);
    } else if (tilingData.dataType == 1) {
        DispatchKdaGateCumsumBySafeGate<bfloat16_t>(g, aLog, dtBias, cuSeqlens, gk, tilingData, &pipe);
    } else {
        DispatchKdaGateCumsumBySafeGate<half>(g, aLog, dtBias, cuSeqlens, gk, tilingData, &pipe);
    }
}
