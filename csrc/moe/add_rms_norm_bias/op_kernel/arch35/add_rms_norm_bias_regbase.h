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
// Source: norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h


/* !
 * \file add_rms_norm_bias_regbase.h
 * \brief
 */
#ifndef ADD_RMS_NORM_BIAS_REGBASE_H
#define ADD_RMS_NORM_BIAS_REGBASE_H
#include "add_rms_norm_bias_regbase_common.h"
#include "../rms_norm_base.h"
#include "kernel_operator.h"
#include "deps/reduce_common_regbase.h"

namespace AddRmsNormBiasA5 {
using namespace AscendC;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr int32_t NDDMA_DIM = 5;
constexpr int32_t UNROLL_NUM = 2;

constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;

using RmsNorm::DataCopyCustom;
using RmsNorm::DataCopyImpl;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

constexpr static uint32_t BLOCK_SIZE = RmsNorm::GetUbBlockSize();
constexpr static uint32_t VL_FP32 = RmsNorm::GetVRegSize() / sizeof(float);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}
template <typename T>
class KernelAddRmsNormBiasRegBase {
public:
    __aicore__ inline KernelAddRmsNormBiasRegBase(TPipe* pipe) { pPipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                                const AddRMSNormBiasRegbaseRFullLoadTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        blockFactor = tiling->blockFactor;
        binAddQuotient = tiling->binAddQuotient;
        rowFactor = tiling->rowFactor;
        epsilon = tiling->epsilon;
        numColAlign = tiling->numColAlign;
        avgFactor = tiling->avgFactor;
        nullptrBeta = tiling->nullptr_beta;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        uint64_t rstdUbSizeAlignSize = CeilAlign(rowFactor, static_cast<uint64_t>(VL_FP32)) * sizeof(float);
        uint16_t binaryAddQuotientLoop = (binAddQuotient + VL_FP32 - 1) / VL_FP32;
        uint32_t binaryAddBufLen = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32 * sizeof(float) *
                                   rowFactor;

        xGm1.SetGlobalBuffer((__gm__ T*)x1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        xGm2.SetGlobalBuffer((__gm__ T*)x2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if (!nullptrBeta) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta, numCol);
            pPipe->InitBuffer(inQueueBeta, BUFFER_NUM, numColAlign * sizeof(T));
        }
        yGm.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);
        xOutGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);

        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(inQueueGamma, BUFFER_NUM, numColAlign * sizeof(T));
        pPipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueX, nullptrBeta ? DOUBLE_BUFFER_NUM : BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xReduceBuff, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xFp32Buff, numColAlign * sizeof(float) * rowFactor);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufLen);
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if (!nullptrBeta) {
            CopyInBeta();
            betaLocal = inQueueBeta.DeQue<T>();
        }
        uint32_t rowLoopCount = CeilDiv(rowWork, rowFactor);
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopCount; rowLoopIdx++) {
            uint64_t rowLoopOffset = rowLoopIdx * rowFactor * numCol;
            uint32_t curRows = Min(rowWork - rowLoopIdx * rowFactor, rowFactor);
            Compute(rowLoopIdx, gammaLocal, betaLocal, curRows, rowLoopOffset);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if (!nullptrBeta) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t rowLoopIdx, LocalTensor<T> gammaLocal, LocalTensor<T>& betaLocal,
                                   uint32_t curRows, uint64_t rowLoopOffset)
    {
        CopyInXMutiMoveAlign(rowLoopOffset, numColAlign, curRows);
        LocalTensor<T> xLocal1 = inQueueX1.DeQue<T>();
        LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
        LocalTensor<T> xOutLocal = outQueueX.AllocTensor<T>();
        LocalTensor<float> xFp32Local = xFp32Buff.Get<float>();

        CalculateXAdd(xLocal1, xLocal2, xOutLocal, xFp32Local, curRows, numColAlign);
        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
        outQueueX.EnQue<T>(xOutLocal);
        CopyOutX(rowLoopOffset, curRows, numColAlign);

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff.Get<float>();
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
            xFp32Local, xReduceLocal, binaryAddBuf, static_cast<uint16_t>(curRows), numColAlign, numCol,
            static_cast<uint32_t>(binAddQuotient), static_cast<uint32_t>(BLK_B32));
        NormCommon::ComputeRstdNewtonRaphson<true, true>(xReduceLocal, rstdLocal, curRows, epsilon, avgFactor, VL_FP32);
        outQueueRstd.EnQue<float>(rstdLocal);

        rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyExtParams rstdCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curRows * sizeof(float)),
                                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad(rstdGm[rowLoopIdx * rowFactor], rstdLocal, rstdCopyParams);

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if (nullptrBeta) {
            CalculateY<false>(xFp32Local, gammaLocal, betaLocal, yLocal, rstdLocal, curRows, numColAlign, numCol);
        } else {
            CalculateY<true>(xFp32Local, gammaLocal, betaLocal, yLocal, rstdLocal, curRows, numColAlign, numCol);
        }
        outQueueRstd.FreeTensor(rstdLocal);
        outQueueY.EnQue<T>(yLocal);
        CopyOutY(rowLoopOffset, curRows, numColAlign);
    }

    __aicore__ inline void CalculateXAdd(LocalTensor<T>& xLocal1, LocalTensor<T>& xLocal2, LocalTensor<T>& xOutLocal,
                                         LocalTensor<float>& xFp32Local, uint32_t curRows, uint32_t numColAlign)
    {
        __ubuf__ T* x1InUb = (__ubuf__ T*)xLocal1.GetPhyAddr();
        __ubuf__ T* x2InUb = (__ubuf__ T*)xLocal2.GetPhyAddr();
        __ubuf__ T* xOutInUb = (__ubuf__ T*)xOutLocal.GetPhyAddr();
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();

        uint32_t sreg = curRows * numColAlign;
        uint16_t loopCount = (sreg + VL_FP32 - 1) / VL_FP32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(sreg);
                LoadRegForDtype<T>(x1InUb, x1, pregLoop, offset);
                LoadRegForDtype<T>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreRegForDtype<T>(xOutInUb, xSum, pregLoop, offset);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xFp32Tmp + offset, xSum, pregLoop);
            }
        }
    }

    template <bool HAS_BETA>
    __aicore__ inline void CalculateY(LocalTensor<float>& xFp32Local, LocalTensor<T>& gammaLocal,
                                      LocalTensor<T>& betaLocal,
                                      LocalTensor<T>& yLocal, LocalTensor<float>& rstdLocal, uint32_t curRows,
                                      uint32_t numColAlign, uint32_t reduceNum)
    {
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __ubuf__ T* gammaInUb = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ T* yInUb = (__ubuf__ T*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdInUb = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ T* betaInUb = nullptr;
        if constexpr (HAS_BETA) {
            betaInUb = (__ubuf__ T*)betaLocal.GetPhyAddr();
        }

        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((reduceNum + VL_FP32 - 1) / VL_FP32);
        uint16_t loopRowsFold = loopRows / 2;
        uint16_t loopRowsHasLast = loopRows % 2;

        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;
            RegTensor<float> rstd1Reg;
            RegTensor<float> rstd2Reg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul1UnrollReg;
            RegTensor<float> mul2Reg;
            RegTensor<float> mul2UnrollReg;

            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = reduceNum;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (2 * i + 1));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (2 * i) * numColAlign + r * VL_FP32;
                    uint32_t offset2 = (2 * i + 1) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
                    LoadRegForDtype<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
                    if constexpr (HAS_BETA) {
                        LoadRegForDtype<T>(betaInUb, betaReg, regCurLoop, r * VL_FP32);
                        Add(mul2Reg, mul2Reg, betaReg, regCurLoop);
                        Add(mul2UnrollReg, mul2UnrollReg, betaReg, regCurLoop);
                    }
                    StoreRegForDtype<T>(yInUb, mul2Reg, regCurLoop, offset1);
                    StoreRegForDtype<T>(yInUb, mul2UnrollReg, regCurLoop, offset2);
                }
            }
            for (uint16_t i = 0; i < loopRowsHasLast; ++i) {
                uint32_t sregCount = reduceNum;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (2 * loopRowsFold) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    if constexpr (HAS_BETA) {
                        LoadRegForDtype<T>(betaInUb, betaReg, regCurLoop, r * VL_FP32);
                        Add(mul2Reg, mul2Reg, betaReg, regCurLoop);
                    }
                    StoreRegForDtype<T>(yInUb, mul2Reg, regCurLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t offset, uint32_t curCols, uint32_t curRows = 0)
    {
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(curRows),                                               // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)),                                    // blockLen
            static_cast<uint32_t>(0),                                                     // srcStride
            static_cast<uint32_t>((numColAlign - curCols) * sizeof(T) / ALIGN_32_FACTOR), // dstStride
            0                                                                             // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(xLocal1, xGm1[offset], extParams, padParams);
        DataCopyPad(xLocal2, xGm2[offset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                  // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),                  // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(gammaLocal, gammaGm, copyParams, padParams);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInBeta()
    {
        LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0.0)};
        DataCopyPad(betaLocal, betaGm, copyParams, padParams);
        inQueueBeta.EnQue(betaLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(srcStride),          // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> xLocal = outQueueX.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(srcStride),          // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(xOutGm[offset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

private:
    TPipe* pPipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueX;
    TBuf<TPosition::VECCALC> xReduceBuff;
    TBuf<TPosition::VECCALC> xFp32Buff;
    TBuf<TPosition::VECCALC> binaryAddBuf;
    NdDmaParams<T, NDDMA_DIM> dmaParam_;
    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xOutGm;
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t rowFactor;
    uint64_t binAddQuotient;
    float epsilon;
    float avgFactor;
    uint32_t nullptrBeta{1};
    uint64_t rowWork{1};
};
} // namespace AddRmsNormBiasA5
#endif // ADD_RMS_NORM_BIAS_REGBASE_H
