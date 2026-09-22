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

        // Slots are indexed by the parity of the LOCAL row number so the
        // prologue and the row loop always agree on which buffer holds
        // which row (absolute-parity indexing was the NaN bug of the first
        // pipelining attempt).
        pipe_->InitBuffer(x_buf_[0], num_col_aligned_ * sizeof(T));
        pipe_->InitBuffer(x_buf_[1], num_col_aligned_ * sizeof(T));
        pipe_->InitBuffer(gamma_buf_, num_col_aligned_ * sizeof(T));
        // The fp16 path multiplies in the b16 domain and never widens gamma.
        if constexpr (!IsSame<T, half>::value) {
            pipe_->InitBuffer(gamma_fp32_buf_, num_col_aligned_ * sizeof(float));
        }
        pipe_->InitBuffer(fp32_buf_[0], num_col_aligned_ * sizeof(float));
        pipe_->InitBuffer(fp32_buf_[1], num_col_aligned_ * sizeof(float));
        pipe_->InitBuffer(work_buf_, num_col_aligned_ * sizeof(float));
        pipe_->InitBuffer(reduce_buf_, NUM_PER_REP_FP32 * sizeof(float));
        pipe_->InitBuffer(rstd_buf_, sizeof(float) * 8);
        pipe_->InitBuffer(gather_off_buf_, sizeof(uint32_t) * 8);
    }

    __aicore__ inline void Process()
    {
        if (row_begin_ >= num_row_) {
            return;
        }

        LocalTensor<T> gamma_local = gamma_buf_.Get<T>();
        DataCopyCustom<T>(gamma_local, gamma_gm_, num_col_);
        auto gamma_evt = pipe_->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(gamma_evt);
        WaitFlag<HardEvent::MTE2_V>(gamma_evt);

        // The bf16 path multiplies in fp32 domain, so widen gamma once per
        // core instead of re-casting it on every row.
        if constexpr (!IsSame<T, half>::value) {
            LocalTensor<float> gamma_fp32_local = gamma_fp32_buf_.Get<float>();
            Cast(gamma_fp32_local, gamma_local, RoundMode::CAST_NONE,
                 num_col_);
            PipeBarrier<PIPE_V>();
        }

        // Gather offsets for the per-row rstd broadcast: eight zeros so
        // Gather replicates work[0] into a full 32B block.
        LocalTensor<uint32_t> gather_off = gather_off_buf_.Get<uint32_t>();
        Duplicate(gather_off, 0u, 8);
        PipeBarrier<PIPE_V>();

        // Prologue: prefetch the first row into slot 0. Its completion flag
        // is consumed at the top of the first loop iteration. The gamma id
        // stays occupied until after this allocation so the two MTE2_V
        // flags can never collapse into one (a collapsed pair would
        // deadlock the single-row tail case).
        const uint32_t local_rows = row_end_ - row_begin_;
        load_evt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        DataCopyCustom<T>(x_buf_[0].Get<T>(), x_gm_[row_begin_ * num_col_],
                          num_col_);
        SetFlag<HardEvent::MTE2_V>(load_evt_);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(gamma_evt);

        for (uint32_t i = 0; i < local_rows; ++i) {
            ProcessRow(i, local_rows, gamma_local, gather_off);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t i, uint32_t local_rows,
                                      LocalTensor<T>& gamma_local,
                                      LocalTensor<uint32_t>& gather_off)
    {
        const uint32_t slot = i & 1U;
        LocalTensor<T> x_local = x_buf_[slot].Get<T>();
        LocalTensor<float> x_fp32 = fp32_buf_[slot].Get<float>();
        LocalTensor<float> work = work_buf_.Get<float>();
        LocalTensor<float> reduce = reduce_buf_.Get<float>();
        LocalTensor<float> rstd = rstd_buf_.Get<float>();
        const uint32_t offset = (row_begin_ + i) * num_col_;

        // Wait for this row's input, then prefetch the next row into the
        // other slot. The waits below run on the MTE2 pipe: slot (i+1)&1 was
        // last written by row i-1 (V produced y there, MTE3 stored it).
        WaitFlag<HardEvent::MTE2_V>(load_evt_);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(load_evt_);
        if (i + 1 < local_rows) {
            if (i >= 1) {
                WaitFlag<HardEvent::V_MTE2>(slot_free_evt_);
                pipe_->ReleaseEventID<HardEvent::V_MTE2>(slot_free_evt_);
                WaitFlag<HardEvent::MTE3_MTE2>(y_stored_evt_);
                pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(y_stored_evt_);
            }
            load_evt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
            DataCopyCustom<T>(x_buf_[(i + 1) & 1U].Get<T>(),
                              x_gm_[(row_begin_ + i + 1) * num_col_],
                              num_col_);
            SetFlag<HardEvent::MTE2_V>(load_evt_);
        }

        // x_fp32[slot] was last read by the MTE3 store of row i-2, which
        // shares this slot's parity; keep two event ids in flight.
        if (i >= 2) {
            WaitFlag<HardEvent::MTE3_V>(fp32_stored_evt_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(fp32_stored_evt_[slot]);
        }

        Cast(x_fp32, x_local, RoundMode::CAST_NONE, num_col_);
        PipeBarrier<PIPE_V>();
        Mul(work, x_fp32, x_fp32, num_col_);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(work, work, reduce, num_col_);
        PipeBarrier<PIPE_V>();
        // Scale the single-element mean instead of the whole vector before
        // the reduction: sum(x^2)/N == (sum(x^2)) * (1/N).
        Muls(work, work, inv_num_col_, 1);
        PipeBarrier<PIPE_V>();
        Adds(work, work, epsilon_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(work, work, 1);
        Duplicate(reduce, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(work, reduce, work, 1);
        PipeBarrier<PIPE_V>();

        // Broadcast rstd within the vector pipe: Gather replicates work[0]
        // into a 32B block, then Mul reads that block with src1 strides
        // pinned to zero. This keeps rstd in UB and avoids the V_S / S_V
        // scalar round trip (the only wait that would stall the issuing
        // thread). Bit-identical to Muls with the same fp32 value.
        Gather(rstd, work, gather_off, 0, 8);
        PipeBarrier<PIPE_V>();
        const uint32_t repeats = num_col_ / NUM_PER_REP_FP32;
        const uint32_t tail = num_col_ % NUM_PER_REP_FP32;
        BinaryRepeatParams bcast_params(1, 1, 0, 8, 8, 0);
        if (repeats > 0) {
            Mul(x_fp32, x_fp32, rstd, NUM_PER_REP_FP32, repeats,
                bcast_params);
        }
        if (tail > 0) {
            Mul(x_fp32[repeats * NUM_PER_REP_FP32],
                x_fp32[repeats * NUM_PER_REP_FP32], rstd, tail, 1,
                bcast_params);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (IsSame<T, half>::value) {
            Cast(x_local, x_fp32, RoundMode::CAST_NONE, num_col_);
            PipeBarrier<PIPE_V>();
            Mul(x_local, x_local, gamma_local, num_col_);
        } else {
            LocalTensor<float> gamma_fp32_local = gamma_fp32_buf_.Get<float>();
            Mul(x_fp32, x_fp32, gamma_fp32_local, num_col_);
            PipeBarrier<PIPE_V>();
            Cast(x_local, x_fp32, RoundMode::CAST_RINT, num_col_);
        }
        PipeBarrier<PIPE_V>();

        // The FP32 result deliberately widens the rounded low-precision
        // result. HashTopK therefore sees exactly the same values as the
        // original RMSNorm followed by Tensor.float().
        Cast(x_fp32, x_local, RoundMode::CAST_NONE, num_col_);
        PipeBarrier<PIPE_V>();

        // x[slot] holds y from the RINT cast above and is dead for V after
        // the widening cast, so it can host the load of row i+2. Tail rows
        // whose buffers are never reused skip the flags: every SetFlag must
        // be paired with a WaitFlag before the kernel exits, otherwise the
        // stale flags poison the event state of the next kernel launched on
        // this core (observed as a hang of the following op).
        if (i + 2 < local_rows) {
            slot_free_evt_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(slot_free_evt_);
        }

        // Hand both outputs to MTE3; the stores of this row overlap the
        // vector work of the next row.
        auto output_evt = pipe_->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(output_evt);
        WaitFlag<HardEvent::V_MTE3>(output_evt);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(output_evt);
        DataCopyCustom<T>(y_gm_[offset], x_local, num_col_);
        if (i + 2 < local_rows) {
            y_stored_evt_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(y_stored_evt_);
        }
        DataCopyCustom<float>(y_fp32_gm_[offset], x_fp32, num_col_);
        if (i + 2 < local_rows) {
            fp32_stored_evt_[slot] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            SetFlag<HardEvent::MTE3_V>(fp32_stored_evt_[slot]);
        }
    }

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> x_buf_[2];
    TBuf<TPosition::VECCALC> gamma_buf_;
    TBuf<TPosition::VECCALC> gamma_fp32_buf_;
    TBuf<TPosition::VECCALC> fp32_buf_[2];
    TBuf<TPosition::VECCALC> work_buf_;
    TBuf<TPosition::VECCALC> reduce_buf_;
    TBuf<TPosition::VECCALC> rstd_buf_;
    TBuf<TPosition::VECCALC> gather_off_buf_;
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
    // Event ids are allocated (not fetched) so several flags can be in
    // flight per direction, mirroring what TQue does for depth-2 queues.
    TEventID load_evt_ = 0;
    TEventID slot_free_evt_ = 0;
    TEventID y_stored_evt_ = 0;
    TEventID fp32_stored_evt_[2] = {0, 0};
};
#endif
