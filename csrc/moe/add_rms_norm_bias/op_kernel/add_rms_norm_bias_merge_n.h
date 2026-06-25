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
 * \file add_rms_norm_bias_merge_n.h
 * \brief add rms norm bias merge n file
 */
#ifndef ADD_RMS_NORM_BIAS_MERGE_N_H_
#define ADD_RMS_NORM_BIAS_MERGE_N_H_
#include "./rms_norm_base.h"

using namespace AscendC;
using namespace RmsNorm;

template <typename T>
class KernelAddRmsNormBiasMergeN {
public:
    __aicore__ inline KernelAddRmsNormBiasMergeN(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, const AddRMSNormBiasTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tiling->num_row;
        this->numCol = tiling->num_col;
        this->numColAlign = tiling->num_col_align;
        this->blockFactor = tiling->block_factor;
        this->rowFactor = tiling->row_factor;
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = tiling->avg_factor;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
            this->rowLoop = tiling->row_loop;
            this->rowTail = tiling->row_tail;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = tiling->last_block_factor;
            this->rowLoop = tiling->last_block_row_loop;
            this->rowTail = tiling->last_block_row_tail;
        }
        this->mulLoopFp32 = tiling->mul_loop_fp32;
        this->mulTailFp32 = tiling->mul_tail_fp32;
        this->dstRepStrideFp32 = tiling->dst_rep_stride_fp32;
        this->mulLoopFp16 = tiling->mul_loop_fp16;
        this->mulTailFp16 = tiling->mul_tail_fp16;
        this->dstRepStrideFp16 = tiling->dst_rep_stride_fp16;
        this->isPerformance = tiling->is_performance;
        this->nullptrBeta = tiling->nullptr_beta;
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
        Ppipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        if (!this->nullptrBeta) {
            Ppipe->InitBuffer(inQueueBeta, BUFFER_NUM, ubFactor * sizeof(T));
        }
        Ppipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));
#else
        Ppipe->InitBuffer(rstdBuf, rowFactor * sizeof(float));
#endif
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(tmpBuf, rowFactor * NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGammaBeta();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if (!this->nullptrBeta) {
            betaLocal = inQueueBeta.DeQue<T>();
        }
        for (uint32_t i_o = 0; i_o < rowLoop - 1; i_o++) {
            MainCompute(i_o, rowFactor, gammaLocal, betaLocal);
        }
        MainCompute(rowLoop - 1, rowTail, gammaLocal, betaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        if (!this->nullptrBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void MainCompute(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T>& gammaLocal, LocalTensor<T>& betaLocal)
    {
        uint32_t gm_bias = i_o * rowFactor * numCol;
        uint32_t elementNum = calc_row_num * numColAlign;
        CopyInX(gm_bias, calc_row_num);
        LocalTensor<T> xLocal = ComputeX(elementNum);
        CopyOutX(gm_bias, calc_row_num);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        ComputeRstd(xLocal, rstdLocal, calc_row_num, elementNum);
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
#else
        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        ComputeRstd(xLocal, rstdLocal, calc_row_num, elementNum);
#endif
        ComputeY(xLocal, gammaLocal, betaLocal, rstdLocal, calc_row_num, elementNum);
        CopyOutY(gm_bias, calc_row_num);
    }

private:
    __aicore__ inline void CopyInX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        LocalTensor<T> x1Local = inQueueX.AllocTensor<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(x1Local, x1Gm[gm_bias], calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(x1Local, x1Gm[gm_bias], calc_row_num, numCol);
        }
        inQueueX.EnQue(x1Local);
        LocalTensor<T> x2Local = inQueueX.AllocTensor<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(x2Local, x2Gm[gm_bias], calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(x2Local, x2Gm[gm_bias], calc_row_num, numCol);
        }
        inQueueX.EnQue(x2Local);
    }

    __aicore__ inline LocalTensor<T> ComputeX(uint32_t elementNum)
    {
        LocalTensor<T> x1Local = inQueueX.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX.DeQue<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();
        if constexpr (!is_same<T, bfloat16_t>::value) {
            Add(xLocal, x1Local, x2Local, elementNum);
        } else {
            LocalTensor<float> x1Fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2Fp32 = sqxBuf.Get<float>();
            Cast(x1Fp32, x1Local, RoundMode::CAST_NONE, elementNum);
            Cast(x2Fp32, x2Local, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32, x1Fp32, x2Fp32, elementNum);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1Fp32, RoundMode::CAST_RINT, elementNum);
        }
        inQueueX.FreeTensor(x1Local);
        inQueueX.FreeTensor(x2Local);
        outQueueY.EnQue(xLocal);
        PipeBarrier<PIPE_V>();
        return xLocal;
    }

    __aicore__ inline void CopyOutX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        // CopyOut x1 + x2
        auto xOut = outQueueY.DeQue<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(xGm[gm_bias], xOut, calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(xGm[gm_bias], xOut, calc_row_num, numCol);
        }
        outQueueY.FreeTensor(xOut);
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

    __aicore__ inline void ComputeRstd(LocalTensor<T> xLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num, uint32_t elementNum)
    {
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        if constexpr (!is_same<T, float>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            Cast(x_fp32, xLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(sqx, x_fp32, x_fp32, elementNum);
        } else {
            Mul(sqx, xLocal, xLocal, elementNum);
        }
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, elementNum);
        PipeBarrier<PIPE_V>();

        ReduceSumMultiN(rstdLocal, sqx, tmpLocal, calc_row_num, numCol, numColAlign);
        PipeBarrier<PIPE_V>();
        Adds(rstdLocal, rstdLocal, epsilon, calc_row_num);
        PipeBarrier<PIPE_V>();

        Sqrt(rstdLocal, rstdLocal, calc_row_num);
        Duplicate(tmpLocal, ONE, calc_row_num);
        PipeBarrier<PIPE_V>();

        Div(rstdLocal, tmpLocal, rstdLocal, calc_row_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeY(
        LocalTensor<T> xLocal, LocalTensor<T> gammaLocal, LocalTensor<T> betaLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num, uint32_t elementNum)
    {
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        uint32_t splidRow = 240;
        uint32_t rowRepeatLoop1 = calc_row_num / splidRow;
        uint32_t rowRepeatTail1 = calc_row_num - rowRepeatLoop1 * splidRow;
        for(uint32_t r_i = 0; r_i < rowRepeatLoop1; r_i ++) {
          Brcb(tmpLocal[r_i * splidRow * MOV_8], rstdLocal[r_i * splidRow], splidRow, {1, 8});
        }
        PipeBarrier<PIPE_V>();
        
        if(rowRepeatTail1 > 0) {
          Brcb(tmpLocal[rowRepeatLoop1 * splidRow * MOV_8], rstdLocal[rowRepeatLoop1 * splidRow], rowRepeatTail1, {1, 8});
          PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if constexpr (!is_same<T, float>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            repeatByRow<float>(x_fp32, x_fp32, tmpLocal, calc_row_num, ONE_UINT);
            if constexpr (is_same<T, half>::value) {
                Cast(yLocal, x_fp32, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, x_fp32, RoundMode::CAST_RINT, elementNum);
            }
        } else {
            repeatByRow<float>(yLocal, xLocal, tmpLocal, calc_row_num, ONE_UINT);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (is_same<T, half>::value) {
            repeatByRow<half>(yLocal, yLocal, gammaLocal, calc_row_num, TWO_UINT);
            if (!this->nullptrBeta) {
                addRepeatByRow<half>(yLocal, yLocal, betaLocal, calc_row_num, TWO_UINT);
            }
        } else if constexpr (is_same<T, bfloat16_t>::value) {
            LocalTensor<float> sqx = sqxBuf.Get<float>();
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            Cast(x_fp32, yLocal, RoundMode::CAST_NONE, elementNum);
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            repeatByRow<float>(x_fp32, x_fp32, sqx, calc_row_num, THREE_UINT);
            if (!this->nullptrBeta) {
                Cast(sqx, betaLocal, RoundMode::CAST_NONE, elementNum);
                PipeBarrier<PIPE_V>();
                addRepeatByRow<float>(x_fp32, x_fp32, sqx, calc_row_num, THREE_UINT);
            }
            Cast(yLocal, x_fp32, RoundMode::CAST_RINT, elementNum);
        } else {
            repeatByRow<float>(yLocal, yLocal, gammaLocal, calc_row_num, THREE_UINT);
            if (!this->nullptrBeta) {
                addRepeatByRow<float>(yLocal, yLocal, betaLocal, calc_row_num, THREE_UINT);
            }
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num, numCol);
        }
        outQueueY.FreeTensor(yLocal);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
       LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
        outQueueRstd.FreeTensor(rstdLocal);
    }
#endif
    
    template <typename U>
    __aicore__ inline void repeatByRow(const LocalTensor<U>& dstLocal, const LocalTensor<U>& src1Local, const LocalTensor<U>& src2Local, uint32_t calc_row_num, uint32_t type)
    {   
        // TWO_UINT=gammaFp16 ONE_UINT=rstd
        uint32_t strideParams[6] = {mulLoopFp32, mulTailFp32, 64, 1, dstRepStrideFp32, 0};
        if (type == TWO_UINT) {
            strideParams[0] = mulLoopFp16;
            strideParams[1] = mulTailFp16;
            strideParams[2] = 128;
            strideParams[4] = dstRepStrideFp16;
        } else if (type == ONE_UINT) {
            strideParams[3] = 0;
            strideParams[5] = 1;
        }
        uint32_t singlT = 255;
        uint32_t rowRepeatLoop = calc_row_num / singlT;
        uint32_t rowRepeatTail = calc_row_num - rowRepeatLoop * singlT;
        uint32_t offset2 = 0;
        for(uint32_t r_i = 0; r_i < rowRepeatLoop; r_i ++) {
            offset2 = type == 1 ? (r_i * singlT * MOV_8) : 0;
            mulRepeat<U>(dstLocal[r_i * singlT * numColAlign], src1Local[r_i * singlT * numColAlign], src2Local[offset2], singlT, strideParams);
        }
        if(rowRepeatTail > 0) {
            offset2 = type == 1 ? (rowRepeatLoop * singlT * MOV_8) : 0;
            uint32_t offset1 = rowRepeatLoop * singlT * numColAlign;
            mulRepeat<U>(dstLocal[offset1], src1Local[offset1], src2Local[offset2], rowRepeatTail, strideParams);
        }
    }

    template <typename U>
    __aicore__ inline void mulRepeat(const LocalTensor<U>& dstLocal, const LocalTensor<U>& src1Local, const LocalTensor<U>& src2Local, uint32_t calcRowNum, uint32_t strideParams[6])
    {
        uint32_t mulLoop = strideParams[0];
        uint32_t mulTail = strideParams[1];
        uint32_t strideNum = strideParams[2];
        uint8_t src1BlkStride = static_cast<uint8_t>(strideParams[3]);
        uint8_t dstRepStride = static_cast<uint8_t>(strideParams[4]);
        uint8_t src1RepStride = static_cast<uint8_t>(strideParams[5]);
        if(src1BlkStride == 0) {
          for (uint32_t m_i = 0; m_i < mulLoop; m_i++) {
            Mul(dstLocal[m_i * strideNum], src1Local[m_i * strideNum], src2Local, strideNum, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
          if(mulTail > 0) {
            Mul(dstLocal[mulLoop * strideNum], src1Local[mulLoop * strideNum], src2Local, mulTail, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
        } else {
          for (uint32_t m_i = 0; m_i < mulLoop; m_i++) {
              Mul(dstLocal[m_i * strideNum], src1Local[m_i * strideNum], src2Local[m_i * strideNum], strideNum, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
          if(mulTail > 0) {
              Mul(dstLocal[mulLoop * strideNum], src1Local[mulLoop * strideNum], src2Local[mulLoop * strideNum], mulTail, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
        }
    }

    template <typename U>
    __aicore__ inline void addRepeatByRow(const LocalTensor<U>& dstLocal, const LocalTensor<U>& src1Local, const LocalTensor<U>& src2Local, uint32_t calc_row_num, uint32_t type)
    {   
        // TWO_UINT=gammaFp16 ONE_UINT=rstd
        uint32_t strideParams[6] = {mulLoopFp32, mulTailFp32, 64, 1, dstRepStrideFp32, 0};
        if (type == TWO_UINT) {
            strideParams[0] = mulLoopFp16;
            strideParams[1] = mulTailFp16;
            strideParams[2] = 128;
            strideParams[4] = dstRepStrideFp16;
        } else if (type == ONE_UINT) {
            strideParams[3] = 0;
            strideParams[5] = 1;
        }
        uint32_t singlT = 255;
        uint32_t rowRepeatLoop = calc_row_num / singlT;
        uint32_t rowRepeatTail = calc_row_num - rowRepeatLoop * singlT;
        uint32_t offset2 = 0;
        for(uint32_t r_i = 0; r_i < rowRepeatLoop; r_i ++) {
            offset2 = type == 1 ? (r_i * singlT * MOV_8) : 0;
            addRepeat<U>(dstLocal[r_i * singlT * numColAlign], src1Local[r_i * singlT * numColAlign], src2Local[offset2], singlT, strideParams);
        }
        if(rowRepeatTail > 0) {
            offset2 = type == 1 ? (rowRepeatLoop * singlT * MOV_8) : 0;
            uint32_t offset1 = rowRepeatLoop * singlT * numColAlign;
            addRepeat<U>(dstLocal[offset1], src1Local[offset1], src2Local[offset2], rowRepeatTail, strideParams);
        }
    }

    template <typename U>
    __aicore__ inline void addRepeat(const LocalTensor<U>& dstLocal, const LocalTensor<U>& src1Local, const LocalTensor<U>& src2Local, uint32_t calcRowNum, uint32_t strideParams[6])
    {
        uint32_t addLoop = strideParams[0];
        uint32_t addTail = strideParams[1];
        uint32_t strideNum = strideParams[2];
        uint8_t src1BlkStride = static_cast<uint8_t>(strideParams[3]);
        uint8_t dstRepStride = static_cast<uint8_t>(strideParams[4]);
        uint8_t src1RepStride = static_cast<uint8_t>(strideParams[5]);
        if(src1BlkStride == 0) {
          for (uint32_t m_i = 0; m_i < addLoop; m_i++) {
            Add(dstLocal[m_i * strideNum], src1Local[m_i * strideNum], src2Local, strideNum, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
          if(addTail > 0) {
            Add(dstLocal[addLoop * strideNum], src1Local[addLoop * strideNum], src2Local, addTail, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
        } else {
          for (uint32_t m_i = 0; m_i < addLoop; m_i++) {
              Add(dstLocal[m_i * strideNum], src1Local[m_i * strideNum], src2Local[m_i * strideNum], strideNum, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
          if(addTail > 0) {
              Add(dstLocal[addLoop * strideNum], src1Local[addLoop * strideNum], src2Local[addLoop * strideNum], addTail, calcRowNum, {1, 1, src1BlkStride, dstRepStride, dstRepStride, src1RepStride});
          }
          PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe* Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    // create queues for output, in this case depth is equal to buffer num
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
#else
    TBuf<TPosition::VECCALC> rstdBuf;
#endif
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t numColAlign;
    uint32_t blockFactor; // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    bool isNumColAlign = true;
#else
    bool isNumColAlign = false;
#endif
    uint8_t isPerformance = 0;
    uint32_t rowLoop = 1;
    uint32_t rowTail = 0;
    uint32_t mulLoopFp32;
    uint32_t mulTailFp32;
    uint8_t dstRepStrideFp32;
    uint32_t mulLoopFp16;
    uint32_t mulTailFp16;
    uint8_t dstRepStrideFp16;
    uint32_t nullptrBeta = 0;
};
#endif // _ADD_RMS_NORM_BIAS_MERGE_N_H_