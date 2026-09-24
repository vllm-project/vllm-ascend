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

        // pipe alloc memory to queue/buffers, the unit is Bytes.
        // bf16 takes the row-pipelined path (manual TBuf slots + events);
        // fp16/fp32 keep the queue-based serial path.
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
        // 8 zero byte-offsets for the rstd Gather plus the 8-lane broadcast
        // block the Gather produces (rstd stays in the vector domain, the
        // per-row Muls with a scalar operand becomes a stride-0 block Mul).
        Ppipe->InitBuffer(zeroOffBuf, NUM_PER_BLK_FP32 * sizeof(uint32_t));
        Ppipe->InitBuffer(rstdBcastBuf, NUM_PER_BLK_FP32 * sizeof(float));
        if constexpr (is_same<T, bfloat16_t>::value) {
            // bf16 math must stay in the fp32 domain (c220 has no element-wise
            // bf16 ops), so gamma/beta are widened once per core instead of
            // once per row.
            Ppipe->InitBuffer(gammaFp32Buf, ubFactor * sizeof(float));
            if (!this->nullptrBeta) {
                Ppipe->InitBuffer(betaFp32Buf, ubFactor * sizeof(float));
            }
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
            // Row-pipelined path: double-buffered x slots (the load of row
            // i+1 overlaps the vector chain of row i), single fp32 workspaces
            // (only the V pipe touches them, and V is in-order), and a small
            // rstd accumulator so rstd never enters the scalar pipe.
            Ppipe->InitBuffer(x1PipeBuf[0], ubFactor * sizeof(T));
            Ppipe->InitBuffer(x1PipeBuf[1], ubFactor * sizeof(T));
            Ppipe->InitBuffer(x2PipeBuf[0], ubFactor * sizeof(T));
            Ppipe->InitBuffer(x2PipeBuf[1], ubFactor * sizeof(T));
            Ppipe->InitBuffer(oneBuf, NUM_PER_BLK_FP32 * sizeof(float));
            Ppipe->InitBuffer(rstdAccBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
            Ppipe->InitBuffer(iotaBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(uint32_t));
            Ppipe->InitBuffer(compBuf, rowFactor * sizeof(float));
        } else {
            Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
            Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
            if (!this->nullptrBeta) {
                Ppipe->InitBuffer(inQueueBeta, BUFFER_NUM, ubFactor * sizeof(T));
            }
            Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
            Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));
            if constexpr (is_same<T, half>::value) {
                Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (is_same<T, bfloat16_t>::value) {
            ProcessPipelined();
            return;
        }
        CopyInGammaBeta();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if (!this->nullptrBeta) {
            betaLocal = inQueueBeta.DeQue<T>();
        }
        // Per-core preamble: zero Gather offsets, and for bf16 widen
        // gamma/beta once (the row loop would otherwise repeat the casts).
        LocalTensor<uint32_t> zeroOffsets = zeroOffBuf.Get<uint32_t>();
        Duplicate(zeroOffsets, ZERO_UINT, NUM_PER_BLK_FP32);
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
    // ------------------------------------------------------------------
    // bf16 row-pipelined path (the DeepSeek production configuration).
    //
    // Row i uses slot (i & 1) of x1PipeBuf/x2PipeBuf (local row parity, so
    // prologue and loop always agree). x1[slot] carries the loaded x1 and
    // later the x output (x1+x2 rounded); x2[slot] carries the loaded x2 and
    // later the y output. Loads of row i+1 are issued while the vector chain
    // of row i runs; the two stores of row i overlap row i+1, so neither
    // copy pipe is ever on the vector critical path. The single fp32
    // workspaces are safe because only the V pipe touches them and V is
    // in-order.
    //
    // Event discipline (AllocEventID/SetFlag/WaitFlag/ReleaseEventID, the
    // TQue pattern): one in-flight flag per direction plus one V_MTE3 pair
    // per row. Waits sit on the consumer pipe (MTE2_V on V, V_MTE2/MTE3_MTE2
    // on MTE2, V_MTE3 on MTE3), so the issuing scalar thread never blocks.
    // A flag is only raised when its consumer exists (i + 2 < rowWork), so
    // the kernel exits with zero pending flags - a dangling SetFlag would
    // poison the event state of the next kernel on this core.
    //
    // rstd stays in the vector domain end to end: reduced into sqx[0],
    // accumulated into an 8-lane-per-row block (32B-aligned vector writes),
    // compacted once per rowFactor rows with a Gather and stored with one
    // contiguous DataCopyPad. No V_S/S_V round trip remains anywhere.
    // ------------------------------------------------------------------
    __aicore__ inline void ProcessPipelined()
    {
        LocalTensor<uint32_t> zeroOff = zeroOffBuf.Get<uint32_t>();
        LocalTensor<float> one8 = oneBuf.Get<float>();
        LocalTensor<uint32_t> iota = iotaBuf.Get<uint32_t>();
        Duplicate(zeroOff, ZERO_UINT, NUM_PER_BLK_FP32);
        Duplicate(one8, ONE, NUM_PER_BLK_FP32);
        for (uint32_t b = 0; b < rowFactor; b++) {
            Duplicate(iota[b * NUM_PER_BLK_FP32], b * ONE_BLK_SIZE, NUM_PER_BLK_FP32);
        }
        PipeBarrier<PIPE_V>();

        // Stage gamma/beta through the row-0 x slots, widen once per core,
        // then free the slots for row 0's loads behind one V_MTE2 flag.
        LocalTensor<T> stage1 = x1PipeBuf[0].Get<T>();
        DataCopyCustom<T>(stage1, gammaGm, numCol);
        if (!this->nullptrBeta) {
            DataCopyCustom<T>(x2PipeBuf[0].Get<T>(), betaGm, numCol);
        }
        TEventID evtStage = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(evtStage);
        WaitFlag<HardEvent::MTE2_V>(evtStage); // runs on the V pipe
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(evtStage);
        LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
        Cast(gammaFp32, stage1, RoundMode::CAST_NONE, numCol);
        if (!this->nullptrBeta) {
            Cast(betaFp32Buf.Get<float>(), x2PipeBuf[0].Get<T>(), RoundMode::CAST_NONE, numCol);
        }
        PipeBarrier<PIPE_V>();
        TEventID evtStaged = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(evtStaged); // V pipe: casts done, slots free

        // Prologue: prefetch row 0; its completion flag is consumed at the
        // top of the first loop iteration.
        loadEvt = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        WaitFlag<HardEvent::V_MTE2>(evtStaged); // runs on the MTE2 pipe
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(evtStaged);
        DataCopyCustom<T>(x1PipeBuf[0].Get<T>(), x1Gm, numCol);
        DataCopyCustom<T>(x2PipeBuf[0].Get<T>(), x2Gm, numCol);
        SetFlag<HardEvent::MTE2_V>(loadEvt);

        for (uint32_t i = 0; i < rowWork; i++) {
            ProcessPipelinedRow(i);
        }
    }

    __aicore__ inline void ProcessPipelinedRow(uint32_t i)
    {
        const uint32_t slot = i & 1U;
        const uint32_t gmBias = i * numCol;
        LocalTensor<T> x1Local = x1PipeBuf[slot].Get<T>();
        LocalTensor<T> x2Local = x2PipeBuf[slot].Get<T>();
        LocalTensor<float> xf = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce = reduceFp32Buf.Get<float>();
        LocalTensor<float> rstd8 = rstdBcastBuf.Get<float>();
        LocalTensor<uint32_t> zeroOff = zeroOffBuf.Get<uint32_t>();

        // Wait for this row's loads (wait runs on the V pipe), then prefetch
        // row i+1 into the other slot: its last writer was row i-1, and the
        // guards below are the flags row i-1 raised (waits run on MTE2).
        WaitFlag<HardEvent::MTE2_V>(loadEvt);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(loadEvt);
        if (i + 1 < rowWork) {
            if (i >= 1) {
                WaitFlag<HardEvent::V_MTE2>(slotFreeEvt);
                GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(slotFreeEvt);
                WaitFlag<HardEvent::MTE3_MTE2>(storedEvt);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(storedEvt);
            }
            loadEvt = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
            DataCopyCustom<T>(x1PipeBuf[(i + 1) & 1U].Get<T>(), x1Gm[(i + 1) * numCol], numCol);
            DataCopyCustom<T>(x2PipeBuf[(i + 1) & 1U].Get<T>(), x2Gm[(i + 1) * numCol], numCol);
            SetFlag<HardEvent::MTE2_V>(loadEvt);
        }

        Cast(xf, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Cast(sqx, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Add(xf, xf, sqx, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xf, RoundMode::CAST_RINT, numCol); // x output
        PipeBarrier<PIPE_V>();
        Mul(sqx, xf, xf, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqx, sqx, reduce, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, 1);
        PipeBarrier<PIPE_V>();
        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqx, sqx, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, oneBuf.Get<float>(), sqx, 1);
        PipeBarrier<PIPE_V>();
        const uint32_t b = i % rowFactor;
        Adds(rstdAccBuf.Get<float>()[b * NUM_PER_BLK_FP32], sqx, (float)0.0, 1);
        PipeBarrier<PIPE_V>();
        Gather(rstd8, sqx, zeroOff, ZERO_UINT, NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();
        int32_t repeatTimes = numCol / NUM_PER_REP_FP32;
        int32_t tailCount = numCol % NUM_PER_REP_FP32;
        if (likely(repeatTimes > 0)) {
            Mul(xf, xf, rstd8, NUM_PER_REP_FP32, repeatTimes,
                {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
        }
        if (unlikely(tailCount != 0)) {
            Mul(xf[repeatTimes * NUM_PER_REP_FP32], xf[repeatTimes * NUM_PER_REP_FP32], rstd8, tailCount, 1,
                {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
        }
        PipeBarrier<PIPE_V>();
        Cast(x2Local, xf, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        Cast(xf, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(xf, xf, gammaFp32Buf.Get<float>(), numCol);
        PipeBarrier<PIPE_V>();
        if (!this->nullptrBeta) {
            Add(xf, xf, betaFp32Buf.Get<float>(), numCol);
            PipeBarrier<PIPE_V>();
        }
        Cast(x2Local, xf, RoundMode::CAST_RINT, numCol); // y output
        PipeBarrier<PIPE_V>();
        if (i + 2 < rowWork) {
            slotFreeEvt = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(slotFreeEvt); // V done with both slots
        }

        // Compact this row block's rstds, then hand everything to MTE3; the
        // stores of row i overlap the vector chain of row i+1.
        bool blockEnd = (b == (uint32_t)(rowFactor - 1)) || (i + 1 == rowWork);
        uint32_t n = b + 1; // rows accumulated in this rstd block
        if (blockEnd) {
            Gather(compBuf.Get<float>(), rstdAccBuf.Get<float>(), iotaBuf.Get<uint32_t>(), ZERO_UINT, n);
            PipeBarrier<PIPE_V>();
        }
        TEventID evtOut = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(evtOut); // V pipe: outputs ready
        WaitFlag<HardEvent::V_MTE3>(evtOut); // runs on the MTE3 pipe
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evtOut);
        DataCopyCustom<T>(xGm[gmBias], x1Local, numCol);
        DataCopyCustom<T>(yGm[gmBias], x2Local, numCol);
        if (blockEnd) {
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
            DataCopyCustom<float>(rstdGm[i + 1 - n], compBuf.Get<float>(), n);
#endif
        }
        if (i + 2 < rowWork) {
            storedEvt = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(storedEvt); // stores done, slots reusable
        }
    }

    // Broadcasts sqx[0] (the freshly computed rstd, whose block start is
    // 32B-aligned) into an 8-lane block via Gather, then multiplies it into
    // dst over count floats with a stride-0 src1 (antiquant/batchnorm-style
    // block broadcast). Numerically identical to
    // Muls(dst, src, rstdScalar, count): the same fp32 rstd takes part in the
    // same fp32 multiplies, but rstd never leaves the vector pipe, so the
    // per-row V_S/S_V scalar sync for the Muls operand disappears.
    __aicore__ inline void MulByRstd(const LocalTensor<float>& dst, const LocalTensor<float>& src,
        const LocalTensor<float>& sqx, uint32_t count)
    {
        LocalTensor<float> rstd8 = rstdBcastBuf.Get<float>();
        LocalTensor<uint32_t> zeroOffsets = zeroOffBuf.Get<uint32_t>();
        Gather(rstd8, sqx, zeroOffsets, ZERO_UINT, NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();
        int32_t repeatTimes = count / NUM_PER_REP_FP32;
        int32_t tailCount = count % NUM_PER_REP_FP32;
        if (likely(repeatTimes > 0)) {
            Mul(dst, src, rstd8, NUM_PER_REP_FP32, repeatTimes,
                {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
        }
        if (unlikely(tailCount != 0)) {
            Mul(dst[repeatTimes * NUM_PER_REP_FP32], src[repeatTimes * NUM_PER_REP_FP32], rstd8, tailCount, 1,
                {1, 1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, 0});
        }
        PipeBarrier<PIPE_V>();
    }

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

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, 1);
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
        MulByRstd(yLocal, xLocal, sqx, numCol);
        inQueueX.FreeTensor(xLocal);
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

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, 1);
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
        MulByRstd(x_fp32, x_fp32, sqx, numCol);
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> gammaFp32 = gammaFp32Buf.Get<float>();
        Mul(x_fp32, x_fp32, gammaFp32, numCol);
        if (!this->nullptrBeta) {
            PipeBarrier<PIPE_V>();
            LocalTensor<float> betaFp32 = betaFp32Buf.Get<float>();
            Add(x_fp32, x_fp32, betaFp32, numCol);
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

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, 1);
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
        MulByRstd(x_fp32, x_fp32, sqx, numCol);
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
    TBuf<TPosition::VECCALC> zeroOffBuf;
    TBuf<TPosition::VECCALC> rstdBcastBuf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
    TBuf<TPosition::VECCALC> betaFp32Buf;
    // bf16 pipelined path buffers (see ProcessPipelined).
    TBuf<TPosition::VECCALC> x1PipeBuf[2];
    TBuf<TPosition::VECCALC> x2PipeBuf[2];
    TBuf<TPosition::VECCALC> oneBuf;
    TBuf<TPosition::VECCALC> rstdAccBuf;
    TBuf<TPosition::VECCALC> iotaBuf;
    TBuf<TPosition::VECCALC> compBuf;
    TEventID loadEvt = 0;
    TEventID slotFreeEvt = 0;
    TEventID storedEvt = 0;
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