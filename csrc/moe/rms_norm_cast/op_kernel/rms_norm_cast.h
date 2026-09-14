/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef VLLM_ASCEND_RMS_NORM_CAST_KERNEL_H
#define VLLM_ASCEND_RMS_NORM_CAST_KERNEL_H

#include "rms_norm_cast_base.h"

using namespace AscendC;
using namespace RmsNormCast;

template <typename T>
class KernelRmsNormCast {
public:
    __aicore__ inline explicit KernelRmsNormCast(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                GM_ADDR y_fp32,
                                const RmsNormCastTilingData* tiling)
    {
        num_row_ = tiling->num_row;
        num_col_ = tiling->num_col;
        num_col_aligned_ = tiling->num_col_aligned;
        rows_per_core_ = tiling->rows_per_core;
        inv_num_col_ = tiling->inv_num_col;
        epsilon_ = tiling->epsilon;
        block_idx_ = GetBlockIdx();
        const uint32_t row_begin = block_idx_ * rows_per_core_;
        row_end_ = Min(num_row_, row_begin + rows_per_core_);
        row_begin_ = row_begin;

        x_gm_.SetGlobalBuffer((__gm__ T*)x, num_row_ * num_col_);
        gamma_gm_.SetGlobalBuffer((__gm__ T*)gamma, num_col_);
        y_gm_.SetGlobalBuffer((__gm__ T*)y, num_row_ * num_col_);
        y_fp32_gm_.SetGlobalBuffer((__gm__ float*)y_fp32,
                                   num_row_ * num_col_);

        pipe_->InitBuffer(x_buf_, num_col_aligned_ * sizeof(T));
        pipe_->InitBuffer(gamma_buf_, num_col_aligned_ * sizeof(T));
        pipe_->InitBuffer(fp32_buf_, num_col_aligned_ * sizeof(float));
        pipe_->InitBuffer(work_buf_, num_col_aligned_ * sizeof(float));
        pipe_->InitBuffer(reduce_buf_, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (row_begin_ >= num_row_) {
            return;
        }
        LocalTensor<T> gamma_local = gamma_buf_.Get<T>();
        DataCopyCustom<T>(gamma_local, gamma_gm_, num_col_);
        event_t gamma_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(gamma_ready);
        WaitFlag<HardEvent::MTE2_V>(gamma_ready);

        for (uint32_t row = row_begin_; row < row_end_; ++row) {
            ProcessRow(row, gamma_local);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row,
                                      LocalTensor<T>& gamma_local)
    {
        LocalTensor<T> x_local = x_buf_.Get<T>();
        LocalTensor<float> x_fp32 = fp32_buf_.Get<float>();
        LocalTensor<float> work = work_buf_.Get<float>();
        LocalTensor<float> reduce = reduce_buf_.Get<float>();
        const uint32_t offset = row * num_col_;

        DataCopyCustom<T>(x_local, x_gm_[offset], num_col_);
        event_t x_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(x_ready);
        WaitFlag<HardEvent::MTE2_V>(x_ready);
        Cast(x_fp32, x_local, RoundMode::CAST_NONE, num_col_);
        PipeBarrier<PIPE_V>();
        Mul(work, x_fp32, x_fp32, num_col_);
        PipeBarrier<PIPE_V>();
        Muls(work, work, inv_num_col_, num_col_);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(work, work, reduce, num_col_);
        PipeBarrier<PIPE_V>();
        Adds(work, work, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(work, work, 1);
        Duplicate(reduce, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(work, reduce, work, 1);
        PipeBarrier<PIPE_V>();

        event_t vector_to_scalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vector_to_scalar);
        WaitFlag<HardEvent::V_S>(vector_to_scalar);
        const float rstd = work.GetValue(0);
        event_t scalar_to_vector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalar_to_vector);
        WaitFlag<HardEvent::S_V>(scalar_to_vector);

        Muls(x_fp32, x_fp32, rstd, num_col_);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSame<T, half>::value) {
            Cast(x_local, x_fp32, RoundMode::CAST_NONE, num_col_);
            PipeBarrier<PIPE_V>();
            Mul(x_local, x_local, gamma_local, num_col_);
        } else {
            Cast(work, gamma_local, RoundMode::CAST_NONE, num_col_);
            PipeBarrier<PIPE_V>();
            Mul(x_fp32, x_fp32, work, num_col_);
            PipeBarrier<PIPE_V>();
            Cast(x_local, x_fp32, RoundMode::CAST_RINT, num_col_);
        }
        PipeBarrier<PIPE_V>();

        // The FP32 result deliberately widens the rounded low-precision
        // result. HashTopK therefore sees exactly the same values as the
        // original RMSNorm followed by Tensor.float().
        Cast(x_fp32, x_local, RoundMode::CAST_NONE, num_col_);
        PipeBarrier<PIPE_V>();
        event_t output_ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(output_ready);
        WaitFlag<HardEvent::V_MTE3>(output_ready);
        DataCopyCustom<T>(y_gm_[offset], x_local, num_col_);
        DataCopyCustom<float>(y_fp32_gm_[offset], x_fp32, num_col_);
        event_t output_done = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(output_done);
        WaitFlag<HardEvent::MTE3_MTE2>(output_done);
    }

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> x_buf_;
    TBuf<TPosition::VECCALC> gamma_buf_;
    TBuf<TPosition::VECCALC> fp32_buf_;
    TBuf<TPosition::VECCALC> work_buf_;
    TBuf<TPosition::VECCALC> reduce_buf_;
    GlobalTensor<T> x_gm_;
    GlobalTensor<T> gamma_gm_;
    GlobalTensor<T> y_gm_;
    GlobalTensor<float> y_fp32_gm_;
    uint32_t num_row_;
    uint32_t num_col_;
    uint32_t num_col_aligned_;
    uint32_t rows_per_core_;
    uint32_t row_begin_;
    uint32_t row_end_;
    uint32_t block_idx_;
    float inv_num_col_;
    float epsilon_;
};
#endif
