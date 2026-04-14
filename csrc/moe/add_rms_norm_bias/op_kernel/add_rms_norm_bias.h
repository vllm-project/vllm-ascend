/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias.h
 * \brief add rms norm bias file
 */
#ifndef ADD_RMS_NORM_H_
#define ADD_RMS_NORM_H_
#include "./rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;

template <typename T>
class KernelAddRmsNormBias {
public:
    __aicore__ inline KernelAddRmsNormBias(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, const AddRMSNormBiasTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tiling->num_row;
        this->numCol = tiling->num_col;
        this->blockFactor = tiling->block_factor;
        this->rowFactor = tiling->row_factor;
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (float)1.0 / numCol : 0;
        this->nullptrBeta = tiling->nullptr_beta;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        }
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if (!this->nullptrBeta) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta, numCol);
        }
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_ * blockFactor, blockFactor);
        xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        if (!this->nullptrBeta) {
            Ppipe->InitBuffer(inQueueBeta, BUFFER_NUM, ubFactor * sizeof(T));
        }
        Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGammaBeta();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if (!this->nullptrBeta) {
            betaLocal = inQueueBeta.DeQue<T>();
        }
        uint32_t i_o_max = RmsNorm::CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal, betaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal, betaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        if (!this->nullptrBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal, LocalTensor<T>& betaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            Compute(i_i, gammaLocal, betaLocal, rstdLocal);
            CopyOutY(gm_bias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x2Local = sqxBuf.Get<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            x2Local = x2Local[ubFactor];
        }

        DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
        DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
        inQueueX.EnQue(x1Local_in);
        auto x1Local = inQueueX.DeQue<T>();

        if constexpr (is_same<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            Add(xLocal, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else if constexpr (is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1_fp32, x1_fp32, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0, numCol);
        }
        inQueueX.FreeTensor(x1Local);

        // CopyOut x1 + x2
        outQueueY.EnQue(xLocal);
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(xGm[gm_bias], x_out, numCol);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void CopyInGammaBeta()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
        if (!this->nullptrBeta) {
            LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
            DataCopyCustom<T>(betaLocal, betaGm, numCol);
            inQueueBeta.EnQue(betaLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<float> gammaLocal, LocalTensor<float> betaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        Mul(sqx, xLocal, xLocal, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Muls(yLocal, xLocal, rstdValue, numCol);
        inQueueX.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, numCol);
        if (!this->nullptrBeta) {
            PipeBarrier<PIPE_V>();
            Add(yLocal, betaLocal, yLocal, numCol);
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void Compute(
        uint32_t inner_progress, LocalTensor<bfloat16_t> gammaLocal, LocalTensor<bfloat16_t> betaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, x_fp32, x_fp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol); // gamma_fp32 reuse sqx
        PipeBarrier<PIPE_V>();
        Mul(x_fp32, x_fp32, sqx, numCol);
        if (!this->nullptrBeta) {
            PipeBarrier<PIPE_V>();
            Cast(sqx, betaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x_fp32, x_fp32, sqx, numCol);
        }
        PipeBarrier<PIPE_V>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        outQueueY.EnQue<bfloat16_t>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<half> gammaLocal, LocalTensor<half> betaLocal, LocalTensor<float> rstdLocal)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, x_fp32, x_fp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, numCol);
        if (!this->nullptrBeta) {
            PipeBarrier<PIPE_V>();
            Add(yLocal, betaLocal, yLocal, numCol);
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe* Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
    uint32_t nullptrBeta = 0;
};
#endif // ADD_RMS_NORM_BIAS_H_