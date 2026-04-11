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
 * \file add_rms_norm_bias_split_d.h
 * \brief add rms norm bias split d file
 */
#ifndef ADD_RMS_NORM_BIAS_SPLIT_D_H_
#define ADD_RMS_NORM_BIAS_SPLIT_D_H_
#include "./rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;

template <typename T>
class KernelAddRmsNormBiasSplitD {
public:
    __aicore__ inline KernelAddRmsNormBiasSplitD(TPipe* pipe)
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
        } else {
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

        // pipe alloc memory to queue, the unit is Bytes.
        // We need 2 buffers here for both x1 and x2.
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, 2 * ubFactor * sizeof(T));
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
        Ppipe->InitBuffer(sumBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t i_o_max = RmsNorm::CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
        uint32_t j_max = RmsNorm::CeilDiv(numCol, ubFactor);
        uint32_t col_tail = numCol - (j_max - 1) * ubFactor;
        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, j_max, col_tail);
        }
        SubProcess(i_o_max - 1, row_tail, j_max, col_tail);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, uint32_t j_max, uint32_t col_tail)
    {
        LocalTensor<float> sumLocal = sumBuf.Get<float>();

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        Duplicate(rstdLocal, (float)0.0, calc_row_num);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < j_max - 1; j++) {
            ComputeFormer(i_o, calc_row_num, j, rstdLocal, sumLocal, ubFactor);
        }
        // do tail
        ComputeFormer(i_o, calc_row_num, j_max - 1, rstdLocal, sumLocal, col_tail);
        ComputeRstd(rstdLocal, calc_row_num);

        for (uint32_t j = 0; j < j_max - 1; j++) {
            ComputeLatter(i_o, calc_row_num, j, rstdLocal, ubFactor);
        }
        ComputeLatter(i_o, calc_row_num, j_max - 1, rstdLocal, col_tail);
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }

private:
    __aicore__ inline void CopyInAndAdd(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> x1x2_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x1_in = x1x2_in[0];
        LocalTensor<T> x2_in = x1x2_in[ubFactor];
        DataCopyCustom<T>(x1_in, x1Gm[i_idx * numCol + j_idx * ubFactor], num);
        DataCopyCustom<T>(x2_in, x2Gm[i_idx * numCol + j_idx * ubFactor], num);
        inQueueX.EnQue(x1x2_in);
        LocalTensor<T> x1x2Local = inQueueX.DeQue<T>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[ubFactor];

        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

        if constexpr (is_same<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();

            Add(xLocal, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1_fp32
        } else if constexpr (is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = x1x2Local.template ReinterpretCast<float>();

            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();

            Add(x1_fp32, x1_fp32, x2_fp32, num);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1_fp32
        } else {
            Add(x1Local, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0.0, num);
            // x1+x2 saved in inQueueX
        }
        inQueueX.FreeTensor(x1x2Local);

        // copy out to workspace && x_out
        outQueueY.EnQue(xLocal);
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(xGm[i_idx * numCol + j_idx * ubFactor], x_out, num);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void ComputeFormer(
        uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx, LocalTensor<float>& rstdLocal,
        LocalTensor<float>& sumLocal, uint32_t num)
    {
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
            ComputeSum(i_i, sumLocal, num);
        }
        BlockReduceSumFP32(sumLocal, sumLocal, calc_row_num * NUM_PER_BLK_FP32);
        Add(rstdLocal, rstdLocal, sumLocal, calc_row_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSum(uint32_t i_i_idx, LocalTensor<float>& sumLocal, uint32_t num)
    {
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            PipeBarrier<PIPE_V>();
            Mul(sqx, x_fp32, x_fp32, num);
        } else {
            LocalTensor<T> xLocal = inQueueX.AllocTensor<float>();
            PipeBarrier<PIPE_V>();
            Mul(sqx, xLocal, xLocal, num);
            inQueueX.FreeTensor(xLocal);
        }
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, num);
        PipeBarrier<PIPE_V>();
        // 8 means 8 fp32 pre block
        ReduceSumFP32ToBlock(sumLocal[i_i_idx * 8], sqx, reduce_buf_local, num);
    }

    __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num)
    {
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        Adds(rstdLocal, rstdLocal, epsilon, num);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, num);
        Duplicate(reduce_buf_local, ONE, num);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, reduce_buf_local, rstdLocal, num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLatter(
        uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        CopyInGammaBeta(j_idx, num);
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if (!this->nullptrBeta) {
            betaLocal = inQueueBeta.DeQue<T>();
        }
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            CopyInX(i_o_idx * rowFactor + i_i, j_idx, num);
            ComputeY(i_i, gammaLocal, betaLocal, rstdLocal, num);
            CopyOutY(i_o_idx * rowFactor + i_i, j_idx, num);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if (!this->nullptrBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void CopyInGammaBeta(uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm[j_idx * ubFactor], num);
        inQueueGamma.EnQue(gammaLocal);
        if (!this->nullptrBeta) {
            LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
            DataCopyCustom<T>(betaLocal, betaGm[j_idx * ubFactor], num);
            inQueueBeta.EnQue(betaLocal);
        }
    }

    __aicore__ inline void CopyInX(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyCustom<T>(xLocal, xGm[i_idx * numCol + j_idx * ubFactor], num);
        inQueueX.EnQue<T>(xLocal);
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            Cast(x_fp32, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            inQueueX.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void ComputeY(
        uint32_t i_i_idx, LocalTensor<half>& gammaLocal, LocalTensor<half>& betaLocal, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = rstdLocal.GetValue(i_i_idx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstdValue, num);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, num);
        PipeBarrier<PIPE_V>();
        if (!this->nullptrBeta) {
            Add(yLocal, betaLocal, yLocal, num);
            PipeBarrier<PIPE_V>();
        }
        outQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void ComputeY(
        uint32_t i_i_idx, LocalTensor<float>& gammaLocal, LocalTensor<float>& betaLocal, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = rstdLocal.GetValue(i_i_idx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Muls(yLocal, xLocal, rstdValue, num);
        inQueueX.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, num);
        PipeBarrier<PIPE_V>();
        if (!this->nullptrBeta) {
            Add(yLocal, betaLocal, yLocal, num);
            PipeBarrier<PIPE_V>();
        }
        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void ComputeY(
        uint32_t i_i_idx, LocalTensor<bfloat16_t>& gammaLocal, LocalTensor<bfloat16_t>& betaLocal, LocalTensor<float>& rstdLocal, uint32_t num)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = rstdLocal.GetValue(i_i_idx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstdValue, num);
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
        PipeBarrier<PIPE_V>();
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        Mul(x_fp32, x_fp32, sqx, num);
        PipeBarrier<PIPE_V>();
        if (!this->nullptrBeta) {
            Cast(sqx, betaLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Add(x_fp32, x_fp32, sqx, num);
            PipeBarrier<PIPE_V>();
        }
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<bfloat16_t>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t i_idx, uint32_t j_idx, uint32_t num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(yGm[i_idx * numCol + j_idx * ubFactor], yLocal, num);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t i_o_idx, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        DataCopyCustom<float>(rstdGm[i_o_idx * rowFactor], rstdLocal, num);
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
    TBuf<TPosition::VECCALC> sumBuf;
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

    int tempbufNum;
};
#endif // _ADD_RMS_NORM_BIAS_SPLIT_D_H_