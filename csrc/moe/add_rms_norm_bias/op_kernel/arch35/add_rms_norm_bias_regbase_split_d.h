/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Derived from cann/ops-nn v9.2.0-beta.2, commit
// 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6.
// Source: norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_split_d.h


/* !
 * \file add_rms_norm_bias_regbase_split_d.h
 * \brief
 */
#ifndef ADD_RMS_NORM_BIAS_REGBASE_SPLIT_D_H
#define ADD_RMS_NORM_BIAS_REGBASE_SPLIT_D_H

#include "add_rms_norm_bias_regbase_common.h"
#include "../rms_norm_base.h"
namespace AddRmsNormBiasA5 {
using namespace AscendC;
using AddRmsNormBiasA5::ALIGN_32_FACTOR;
using AddRmsNormBiasA5::ComputeLatterY;
using AddRmsNormBiasA5::CONST_FACTOR_2;
using AddRmsNormBiasA5::Min;
using NormCommon::ComputeMultiLevelRstd;
using RmsNorm::ComputeMultiLevelReduce;
using RmsNorm::ComputeRstd;
using RmsNorm::ComputeSum;
using RmsNorm::DataCopyImpl;
constexpr uint64_t ALIGN_512_FACTOR = 512;
constexpr uint32_t SUM_COUNT = 2;
template <typename T>
class KernelAddRmsNormBiasRegBaseSplitD {
public:
    __aicore__ inline KernelAddRmsNormBiasRegBaseSplitD(TPipe* pipe) { pPipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                                const AddRMSNormBiasRegbaseTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        blockFactor = tiling->blockFactor;
        ubFactor = tiling->ubFactor;
        ubLoop = tiling->ubLoop;
        rowFactor = tiling->rowFactor;
        epsilon = tiling->epsilon;
        colBufferLength = tiling->colBuferLength;
        avgFactor = tiling->avgFactor;
        nullptrBeta = tiling->nullptr_beta;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        const uint64_t blockRowOffset = static_cast<uint64_t>(GetBlockIdx()) * blockFactor;
        const uint64_t blockOffset = blockRowOffset * numCol;
        const uint64_t blockLength = static_cast<uint64_t>(rowWork) * numCol;
        xGm1.SetGlobalBuffer((__gm__ T*)x1 + blockOffset, blockLength);
        xGm2.SetGlobalBuffer((__gm__ T*)x2 + blockOffset, blockLength);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if (!nullptrBeta) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta, numCol);
            pPipe->InitBuffer(inQueueBeta, BUFFER_NUM, colBufferLength * sizeof(T));
        }
        yGm.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockRowOffset, rowWork);
        xOutGm.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength);
        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T));
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T));
        pPipe->InitBuffer(inQueueGamma, nullptrBeta ? DOUBLE_BUFFER_NUM : BUFFER_NUM, colBufferLength * sizeof(T));
        pPipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T));
        pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T));
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rowFactor * sizeof(float));
        if constexpr (!is_same<T, float>::value) {
            pPipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
            pPipe->InitBuffer(xFp32Buf1, ubFactor * sizeof(float));
        }
        pPipe->InitBuffer(workLocalBuf, ONCE_VECTOR_SIZE * sizeof(float));
        pPipe->InitBuffer(level1Buf, ONCE_VECTOR_SIZE * sizeof(float));
        pPipe->InitBuffer(level2Buf, ONCE_VECTOR_SIZE * sizeof(float));
        pPipe->InitBuffer(level3Buf, ONCE_VECTOR_SIZE * sizeof(float));
        pPipe->InitBuffer(tempBuf, V_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t repeatTimes = rowWork / rowFactor + (rowWork % rowFactor != 0);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint32_t remain = rowWork - repeat * rowFactor;
            uint32_t calRowNum = Min(remain, rowFactor);
            SubProcess(repeat, calRowNum);
        }
    }

    __aicore__ inline void SubProcess(uint32_t rowRepeat, uint32_t calRowNum)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        Duplicate(rstdLocal, (float)0.0, rowFactor);
        uint32_t colRepeats = numCol / ubFactor + (numCol % ubFactor != 0);

        for (uint32_t row = 0; row < calRowNum; row++) {
            uint32_t split = ubLoop * ubFactor;
            uint32_t colTail = numCol - split;
            uint64_t offsets = (static_cast<uint64_t>(rowRepeat) * rowFactor + row) * numCol;
            uint32_t tail = colTail % ubFactor;
            uint32_t tailLoop = colTail / ubFactor;
            uint32_t masterLoop = tail != 0 ? 1 : 0;
            masterLoop = ubLoop - tailLoop - masterLoop;
            ComputeFormer(offsets, rstdLocal, row, masterLoop, tailLoop, tail);
        }

        ComputeRstd(rstdLocal, epsilon, avgFactor, calRowNum);

        for (uint32_t repeat = 0; repeat < colRepeats; repeat++) {
            uint32_t remain = numCol - repeat * ubFactor;
            uint32_t calColNum = Min(remain, ubFactor);
            ComputeLatter(rowRepeat, calRowNum, repeat, rstdLocal, calColNum);
        }

        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(rowRepeat, calRowNum);
    }

private:
    __aicore__ inline void CopyInX(uint64_t offset, uint32_t count, uint32_t left = 0, uint32_t right = 0)
    {
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{
            true,                        // isPad
            static_cast<uint8_t>(left),  // leftPadding
            static_cast<uint8_t>(right), // rightPadding
            static_cast<T>(0.0)          // paddingValue
        };
        DataCopyImpl<T>(xLocal1, xGm1[offset], 1, count, 0, 0, padParams);
        inQueueX1.EnQue(xLocal1);

        DataCopyImpl<T>(xLocal2, xGm2[offset], 1, count, 0, 0, padParams);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGamma(uint32_t colRepeat, uint32_t calColNum)
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyImpl<T>(gammaLocal, gammaGm[static_cast<uint64_t>(colRepeat) * ubFactor], 1, calColNum, 0, 0);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeFormerHandle(LocalTensor<float>& dstLocal, uint64_t srcOffset, uint64_t dstOffset,
                                               uint32_t count, uint32_t power)
    {
        uint32_t calCount = CeilAlign((uint64_t)(count * sizeof(T)), ALIGN_32_FACTOR) / sizeof(T);
        CopyInX(srcOffset, count, 0, calCount - count);
        LocalTensor<T> xLocal1 = inQueueX1.DeQue<T>();
        LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
        LocalTensor<float> workLocal = workLocalBuf.Get<float>();
        uint32_t calNum = CeilAlign((uint64_t)(count * sizeof(T)), ALIGN_512_FACTOR) / sizeof(T);
        if (calNum - calCount > 0) {
            Duplicate(xLocal1[calCount], (T)0.0, calNum - calCount);
            Duplicate(xLocal2[calCount], (T)0.0, calNum - calCount);
        }
        ComputeFormerImplV2(dstLocal, xLocal1, xLocal2, workLocal, dstOffset, calNum, power);
        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
    }

    __aicore__ inline void CopyInBeta(uint32_t colRepeat, uint32_t calColNum)
    {
        LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
        DataCopyImpl<T>(betaLocal, betaGm[static_cast<uint64_t>(colRepeat) * ubFactor], 1, calColNum, 0, 0);
        inQueueBeta.EnQue(betaLocal);
    }

    __aicore__ inline void ComputeFormer(uint64_t curRow, LocalTensor<float> dstLocal, uint32_t position,
                                         uint32_t masterLoop, uint32_t tailLoop, uint32_t tail)
    {
        uint64_t offset{curRow};
        uint32_t level1{0};
        uint32_t level2{0};
        uint32_t level3{0};
        LocalTensor<float> level1Local = level1Buf.Get<float>();
        LocalTensor<float> level2Local = level2Buf.Get<float>();
        LocalTensor<float> level3Local = level3Buf.Get<float>();
        LocalTensor<float> tempLocal = tempBuf.Get<float>();
        Duplicate(level1Local, (float)0.0, ONCE_VECTOR_SIZE);
        Duplicate(level2Local, (float)0.0, ONCE_VECTOR_SIZE);
        Duplicate(level3Local, (float)0.0, ONCE_VECTOR_SIZE);
        // stage1: 处理整尾块逻辑
        for (uint32_t repeat = 0; repeat < tailLoop; repeat++) {
            ComputeFormerHandle(tempLocal, offset, 0, ubFactor, ubFactor);
            offset += ubFactor;

            ComputeFormerHandle(tempLocal, offset, 1, ubFactor, ubFactor);
            offset += ubFactor;

            ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        // stage2: 处理非整尾块逻辑
        if (tail > 0 && tail <= ubFactor / CONST_FACTOR_2) {
            ComputeFormerHandle(tempLocal, offset, 0, ubFactor / CONST_FACTOR_2 + tail, ubFactor / CONST_FACTOR_2);
            offset += ubFactor / CONST_FACTOR_2 + tail;

            ComputeFormerHandle(tempLocal, offset, 1, ubFactor / CONST_FACTOR_2, ubFactor / CONST_FACTOR_2);
            offset += ubFactor / CONST_FACTOR_2;

            ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        } else if (tail > ubFactor / CONST_FACTOR_2) {
            ComputeFormerHandle(tempLocal, offset, 0, ubFactor, ubFactor);
            offset += ubFactor;

            ComputeFormerHandle(tempLocal, offset, 1, tail, ubFactor / CONST_FACTOR_2);
            offset += tail;

            ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        // stage3: 处理主块逻辑
        for (uint32_t repeat = 0; repeat < masterLoop; repeat++) {
            ComputeFormerHandle(level1Local, offset, level1, ubFactor, ubFactor);
            offset += ubFactor;
            level1 += 1;
            ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        ComputeMultiLevelRstd<false>(dstLocal, position, level1Local, level2Local, level3Local, level1, level2, level3);
    }

    __aicore__ inline void ComputeLatter(uint32_t rowRepeat, uint32_t calRowNum, uint32_t colRepeat,
                                         LocalTensor<float>& rstdLocal, uint32_t calColNum)
    {
        CopyInGamma(colRepeat, calColNum);
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if (!nullptrBeta) {
            CopyInBeta(colRepeat, calColNum);
            betaLocal = inQueueBeta.DeQue<T>();
        }
        for (uint32_t row = 0; row < calRowNum; row++) {
            uint64_t curRow = static_cast<uint64_t>(rowRepeat) * rowFactor + row;
            uint64_t offset = curRow * numCol + static_cast<uint64_t>(colRepeat) * ubFactor;
            CopyInX(offset, calColNum);
            LocalTensor<T> xLocal1 = inQueueX1.DeQue<T>();
            LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
            LocalTensor<float> xFp32;
            LocalTensor<float> xFp32Other;
            if constexpr (!is_same<T, float>::value) {
                xFp32 = xFp32Buf.Get<float>();
                xFp32Other = xFp32Buf1.Get<float>();
                Cast(xFp32, xLocal1, AscendC::RoundMode::CAST_NONE, calColNum);
                Cast(xFp32Other, xLocal2, AscendC::RoundMode::CAST_NONE, calColNum);
                PipeBarrier<PIPE_V>();
                AscendC::Add(xFp32, xFp32, xFp32Other, calColNum);
                PipeBarrier<PIPE_V>();
            } else {
                AscendC::Add(xLocal1, xLocal1, xLocal2, calColNum);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            LocalTensor<T> xOutLocal = outQueueX.AllocTensor<T>();
            uint32_t calCount = CeilAlign((uint64_t)(calColNum * sizeof(T)), ALIGN_512_FACTOR) / sizeof(T);
            if constexpr (!is_same<T, float>::value) {
                if (nullptrBeta) {
                    ComputeLatterY<T, false>(xFp32, gammaLocal, betaLocal, yLocal, rstdLocal, row, calCount, xOutLocal);
                } else {
                    ComputeLatterY<T, true>(xFp32, gammaLocal, betaLocal, yLocal, rstdLocal, row, calCount, xOutLocal);
                }
            } else {
                if (nullptrBeta) {
                    ComputeLatterY<T, false>(xLocal1, gammaLocal, betaLocal, yLocal, rstdLocal, row, calCount, xOutLocal);
                } else {
                    ComputeLatterY<T, true>(xLocal1, gammaLocal, betaLocal, yLocal, rstdLocal, row, calCount, xOutLocal);
                }
            }
            inQueueX1.FreeTensor(xLocal1);
            inQueueX2.FreeTensor(xLocal2);
            outQueueY.EnQue<T>(yLocal);
            outQueueX.EnQue<T>(xOutLocal);
            CopyOutY(curRow, colRepeat, calColNum);
            CopyOutX(curRow, colRepeat, calColNum);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if (!nullptrBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void CopyOutY(uint64_t curRow, uint32_t curCol, uint32_t calColNum)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyImpl<T>(yGm[curRow * numCol + static_cast<uint64_t>(curCol) * ubFactor], yLocal, 1, calColNum, 0, 0);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t rowRepeat, uint32_t calRowNum)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyImpl<float>(rstdGm[static_cast<uint64_t>(rowRepeat) * rowFactor], rstdLocal, 1, calRowNum, 0, 0);
        outQueueRstd.FreeTensor(rstdLocal);
    }

    __aicore__ inline void CopyOutX(uint64_t curRow, uint32_t curCol, uint32_t calColNum)
    {
        LocalTensor<T> xOutLocal = outQueueX.DeQue<T>();
        DataCopyImpl<T>(xOutGm[curRow * numCol + static_cast<uint64_t>(curCol) * ubFactor], xOutLocal, 1, calColNum, 0, 0);
        outQueueX.FreeTensor(xOutLocal);
    }

private:
    TPipe* pPipe = nullptr;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueX;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> xFp32Buf1;
    TBuf<TPosition::VECCALC> workLocalBuf;
    TBuf<TPosition::VECCALC> level1Buf;
    TBuf<TPosition::VECCALC> level2Buf;
    TBuf<TPosition::VECCALC> level3Buf;
    TBuf<TPosition::VECCALC> tempBuf;
    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xOutGm;
    GlobalTensor<float> rstdGm;
    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;
    uint32_t ubFactor;
    uint32_t colBufferLength;
    uint32_t ubLoop;
    uint32_t rowFactor;
    float epsilon;
    float avgFactor;
    uint32_t nullptrBeta{1};
    uint32_t rowWork{1};
};
} // namespace AddRmsNormBiasA5
#endif // _ADD_RMS_NORM_BIAS_REGBASE_SPLIT_D_H
