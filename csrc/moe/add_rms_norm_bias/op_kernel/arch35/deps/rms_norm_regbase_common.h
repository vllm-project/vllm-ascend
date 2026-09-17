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
// Source: norm/rms_norm/op_kernel/arch35/rms_norm_regbase_common.h


/*!
 * \file rms_norm_regbase_common.h
 * \brief RmsNorm regbase common
 */
#ifndef ADD_RMS_NORM_BIAS_A5_OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_COMMON_H
#define ADD_RMS_NORM_BIAS_A5_OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_COMMON_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../rms_norm_base.h"
#include "reduce_common_regbase.h"

namespace RmsNorm {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

constexpr AscendC::MicroAPI::CastTrait castTraitFp322Fp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitFp322Hifp8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND,
};

template <typename T_Y>
__aicore__ inline uint64_t GetOverflowMode()
{
#if (__NPU_ARCH__ == 3510)
    if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value ||
                  IsSameType<T_Y, hifloat8_t>::value) {
        return AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    }
#endif
    return 0;
}

template <typename T_Y>
__aicore__ inline void SetOverflowMode(uint64_t mode)
{
#if (__NPU_ARCH__ == 3510)
    if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value ||
                  IsSameType<T_Y, hifloat8_t>::value) {
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(mode);
    }
#endif
}

template <typename T, typename U, typename R>
__aicore__ inline void YCopyOutImpl(const U& dstTensor, const R& srcTensor, uint32_t blockCount, uint32_t blockLen,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams extParams{
        static_cast<uint16_t>(blockCount),           // blockCount
        static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
        srcStride,                                   // srcStride
        dstStride,                                   // dstStride
        0                                            // rsv
    };
    DataCopyPad(dstTensor, srcTensor, extParams);
}

/*!
 * DataCopy custom implement
 * @tparam T DataCopy data type
 * @tparam U Destination Operand type
 * @tparam R Source Operand type
 * @param dstTensor Destination Tensor
 * @param srcTensor Source Tensor
 * @param blockCount burst
 * @param blockLen burst length
 * @param padParams pad params
 * @return void
 */
template <typename T, typename U, typename R>
__aicore__ inline void DataCopyImpl(const U& dstTensor, const R& srcTensor, uint32_t blockCount, uint32_t blockLen,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0,
                                    const DataCopyPadExtParams<T> padParams = {false, 0, 0, 0})
{
    DataCopyExtParams extParams{
        static_cast<uint16_t>(blockCount),           // blockCount
        static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
        srcStride,                                   // srcStride
        dstStride,                                   // dstStride
        0                                            // rsv
    };
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
        DataCopyPad(dstTensor, srcTensor, extParams, padParams);
    } else {
        DataCopyPad(dstTensor, srcTensor, extParams);
    }
}

template <typename T_X>
__aicore__ inline void CopyInX(TQue<QuePosition::VECIN, 1>& inQueueX, GlobalTensor<T_X>& srcGm, uint64_t gmOffset,
                               uint32_t blockLen, uint32_t left = 0, uint32_t right = 0)
{
    LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();
    DataCopyPadExtParams<T_X> padParams{
        true,                        // isPad
        static_cast<uint8_t>(left),  // leftPadding
        static_cast<uint8_t>(right), // rightPadding
        static_cast<T_X>(0.0)        // paddingValue
    };
    DataCopyImpl<T_X>(xLocal, srcGm[gmOffset], 1, blockLen, 0, 0, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T_X>
__aicore__ inline void CopyOutX(GlobalTensor<T_X>& xGm, TQue<QuePosition::VECOUT, 1>& outQueueX, uint64_t gmOffset,
                                uint32_t blockLen)
{
    LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();
    DataCopyImpl<T_X>(xGm[gmOffset], xLocal, 1, blockLen);
    outQueueX.FreeTensor(xLocal);
}

template <typename T_Y>
__aicore__ inline void CopyOutY(GlobalTensor<T_Y>& yGm, TQue<QuePosition::VECOUT, 1>& outQueueY, uint64_t gmOffset,
                                uint32_t blockLen)
{
    LocalTensor<T_Y> yLocal = outQueueY.DeQue<T_Y>();
    YCopyOutImpl<T_Y>(yGm[gmOffset], yLocal, 1, blockLen);
    outQueueY.FreeTensor(yLocal);
}

/*!
 * rstd = 1 / sqrt(mean / n + epsilon)
 * @param rstdLocal store mean value Tensor
 * @param epsilon RmsNorm's attr
 * @param count The num of mean
 * @return void
 */
__aicore__ inline void ComputeRstd(LocalTensor<float>& rstdLocal, float epsilon, float avgFactor, uint32_t count)
{
    __ubuf__ float* rstdLocalAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    uint32_t calCount = count;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);
    __VEC_SCOPE__
    {
        RegTensor<float> vReg, srcReg, dstReg;
        MaskReg maskReg;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            maskReg = UpdateMask<float>(sreg);
            LoadAlign(srcReg, rstdLocalAddr + i * V_LENGTH);
            Muls(srcReg, srcReg, avgFactor, maskReg);
            Adds(dstReg, srcReg, epsilon, maskReg);
            Sqrt(vReg, dstReg, maskReg);
            Duplicate(srcReg, float(1.0), maskReg);
            Div(dstReg, srcReg, vReg, maskReg);
            StoreAlign(rstdLocalAddr + i * V_LENGTH, dstReg, maskReg);
        }
    }
}

template <typename DG, bool IS_GEMMA>
__aicore__ inline void GemmaWithOutFloat(RegTensor<DG> gammaReg1, RegTensor<DG> gammaReg2,
                                         RegTensor<float>& gammaFp32Reg1, RegTensor<float>& gammaFp32Reg2,
                                         MaskReg pregMask)
{
    if constexpr (IS_GEMMA) {
        RegTensor<float> gammaTmp1, gammaTmp2;
        Cast<float, DG, castTraitB162B32>(gammaTmp1, gammaReg1, pregMask);
        Cast<float, DG, castTraitB162B32>(gammaTmp2, gammaReg2, pregMask);
        Adds(gammaFp32Reg1, gammaTmp1, 1.0f, pregMask);
        Adds(gammaFp32Reg2, gammaTmp2, 1.0f, pregMask);
    } else {
        Cast<float, DG, castTraitB162B32>(gammaFp32Reg1, gammaReg1, pregMask);
        Cast<float, DG, castTraitB162B32>(gammaFp32Reg2, gammaReg2, pregMask);
    }
}

template <typename DG, bool IS_GEMMA>
__aicore__ inline void GemmaWithFloat(__ubuf__ DG* gammaAddr1, __ubuf__ DG* gammaAddr2, RegTensor<DG>& gammaReg1,
                                      RegTensor<DG>& gammaReg2, MaskReg maskReg, uint16_t i)
{
    if constexpr (IS_GEMMA) {
        RegTensor<float> gammaTmp1, gammaTmp2;
        LoadAlign(gammaTmp1, gammaAddr1 + i * V_LENGTH);
        LoadAlign(gammaTmp2, gammaAddr2 + i * V_LENGTH);
        Adds(gammaReg1, gammaTmp1, 1.0f, maskReg);
        Adds(gammaReg2, gammaTmp2, 1.0f, maskReg);
    } else {
        LoadAlign(gammaReg1, gammaAddr1 + i * V_LENGTH);
        LoadAlign(gammaReg2, gammaAddr2 + i * V_LENGTH);
    }
}

/*!
 * compute multi N yLocal = xLocal * rstd * gammaLocal
 *
 * @param xLocal input xLocal
 * @param gammaLocal input gammaLocal
 * @param yLocal output yLocal
 * @param rstd input rstd
 * @param count vector compute length
 * @return void
 */
template <typename DX, typename DG, bool IS_GEMMA>
__aicore__ inline void ComputeYMultiN(LocalTensor<float>& xLocal, LocalTensor<DG>& gammaLocal, LocalTensor<DX>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t offset, uint32_t count, uint32_t curRows)
{
    uint32_t calCount = count / 2;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __ubuf__ float* xAddr1 = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* xAddr2 = (__ubuf__ float*)xLocal.GetPhyAddr() + calCount;
    __ubuf__ DG* gammaAddr1 = (__ubuf__ DG*)gammaLocal.GetPhyAddr();
    __ubuf__ DG* gammaAddr2 = (__ubuf__ DG*)gammaLocal.GetPhyAddr() + calCount;
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __ubuf__ DX* yAddr1 = (__ubuf__ DX*)yLocal.GetPhyAddr();
    __ubuf__ DX* yAddr2 = (__ubuf__ DX*)yLocal.GetPhyAddr() + calCount;

    if constexpr (!IsSameType<DX, float>::value && !IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(curRows); row++) {
                uint32_t sreg = (uint32_t)calCount;
                RegTensor<DX> yB16Reg1, yB16Reg2;
                RegTensor<DG> gammaReg1, gammaReg2;
                RegTensor<float> rstdReg;
                RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
                RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
                MaskReg pregMask;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + offset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    pregMask = UpdateMask<float>(sreg);
                    LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                    LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                    LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg2, gammaAddr2 + i * V_LENGTH);
                    GemmaWithOutFloat<DG, IS_GEMMA>(gammaReg1, gammaReg2, gammaFp32Reg1, gammaFp32Reg2, pregMask);
                    Mul(dst1Reg, xReg1, rstdReg, pregMask);
                    Mul(dst2Reg, xReg2, rstdReg, pregMask);
                    Mul(yReg1, dst1Reg, gammaFp32Reg1, pregMask);
                    Mul(yReg2, dst2Reg, gammaFp32Reg2, pregMask);
                    Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, pregMask);
                    Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, pregMask);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, pregMask);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, pregMask);
                }
                offset++;
                xAddr1 += count;
                xAddr2 += count;
                yAddr1 += count;
                yAddr2 += count;
            }
        }
    } else if constexpr (!IsSameType<DX, float>::value and IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(curRows); row++) {
                uint32_t sreg = (uint32_t)calCount;
                RegTensor<DX> yB16Reg1, yB16Reg2;
                RegTensor<DG> gammaReg1, gammaReg2;
                RegTensor<float> rstdReg;
                RegTensor<float> xReg1, dst1Reg, yReg1;
                RegTensor<float> xReg2, dst2Reg, yReg2;
                MaskReg maskReg;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + offset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                    LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                    GemmaWithFloat<DG, IS_GEMMA>(gammaAddr1, gammaAddr2, gammaReg1, gammaReg2, maskReg, i);
                    Mul(dst1Reg, xReg1, rstdReg, maskReg);
                    Mul(dst2Reg, xReg2, rstdReg, maskReg);
                    Mul(yReg1, dst1Reg, gammaReg1, maskReg);
                    Mul(yReg2, dst2Reg, gammaReg2, maskReg);
                    Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                    Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
                }
                offset++;
                xAddr1 += count;
                xAddr2 += count;
                yAddr1 += count;
                yAddr2 += count;
            }
        }
    } else {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(curRows); row++) {
                uint32_t sreg = (uint32_t)calCount;
                RegTensor<float> rstdReg;
                RegTensor<float> xReg1, gammaReg1, yReg1, vRegTmp1;
                RegTensor<float> xReg2, gammaReg2, yReg2, vRegTmp2;
                MaskReg maskReg;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + offset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                    LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                    GemmaWithFloat<DG, IS_GEMMA>(gammaAddr1, gammaAddr2, gammaReg1, gammaReg2, maskReg, i);
                    Mul(vRegTmp1, xReg1, rstdReg, maskReg);
                    Mul(vRegTmp2, xReg2, rstdReg, maskReg);
                    Mul(yReg1, vRegTmp1, gammaReg1, maskReg);
                    Mul(yReg2, vRegTmp2, gammaReg2, maskReg);
                    StoreAlign(yAddr1 + i * V_LENGTH, yReg1, maskReg);
                    StoreAlign(yAddr2 + i * V_LENGTH, yReg2, maskReg);
                }
                offset++;
                xAddr1 += count;
                xAddr2 += count;
                yAddr1 += count;
                yAddr2 += count;
            }
        }
    }
}

/*!
 * compute yLocal = xLocal * rstd * gammaLocal
 *
 * @param xLocal input xLocal
 * @param gammaLocal input gammaLocal
 * @param yLocal output yLocal
 * @param rstd input rstd
 * @param count vector compute length
 * @return void
 */
template <typename DX, typename DG, bool IS_GEMMA>
__aicore__ inline void ComputeLatterY(LocalTensor<DX>& xLocal, LocalTensor<DG>& gammaLocal, LocalTensor<DX>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t offset, uint32_t count)
{
    uint32_t calCount = count / 2;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __ubuf__ DX* xAddr1 = (__ubuf__ DX*)xLocal.GetPhyAddr();
    __ubuf__ DX* xAddr2 = (__ubuf__ DX*)xLocal.GetPhyAddr() + calCount;
    __ubuf__ DG* gammaAddr1 = (__ubuf__ DG*)gammaLocal.GetPhyAddr();
    __ubuf__ DG* gammaAddr2 = (__ubuf__ DG*)gammaLocal.GetPhyAddr() + calCount;
    __ubuf__ float* srcAddr2 = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __ubuf__ DX* yAddr1 = (__ubuf__ DX*)yLocal.GetPhyAddr();
    __ubuf__ DX* yAddr2 = (__ubuf__ DX*)yLocal.GetPhyAddr() + calCount;

    if constexpr (!IsSameType<DX, float>::value and !IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            RegTensor<DX> xB16Reg1, yB16Reg1, xB16Reg2, yB16Reg2;
            RegTensor<DG> gammaReg1, gammaReg2;
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
            RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg1, xAddr1 + i * V_LENGTH);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg2, xAddr2 + i * V_LENGTH);
                LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg1, gammaAddr1 + i * V_LENGTH);
                LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg2, gammaAddr2 + i * V_LENGTH);
                if constexpr (IS_GEMMA) {
                    RegTensor<float> gammaTmp1, gammaTmp2;
                    Cast<float, DG, castTraitB162B32>(gammaTmp1, gammaReg1, maskReg);
                    Cast<float, DG, castTraitB162B32>(gammaTmp2, gammaReg2, maskReg);
                    Adds(gammaFp32Reg1, gammaTmp1, 1.0f, maskReg);
                    Adds(gammaFp32Reg2, gammaTmp2, 1.0f, maskReg);
                } else {
                    Cast<float, DG, castTraitB162B32>(gammaFp32Reg1, gammaReg1, maskReg);
                    Cast<float, DG, castTraitB162B32>(gammaFp32Reg2, gammaReg2, maskReg);
                }
                Cast<float, DX, castTraitB162B32>(xReg1, xB16Reg1, maskReg);
                Cast<float, DX, castTraitB162B32>(xReg2, xB16Reg2, maskReg);
                Mul(dst1Reg, xReg1, rstdReg, maskReg);
                Mul(dst2Reg, xReg2, rstdReg, maskReg);
                Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
                Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
            }
        }
    } else if constexpr (!IsSameType<DX, float>::value and IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            RegTensor<DX> xB16Reg1, yB16Reg1, xB16Reg2, yB16Reg2;
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
            RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg1, xAddr1 + i * V_LENGTH);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg2, xAddr2 + i * V_LENGTH);
                if constexpr (IS_GEMMA) {
                    RegTensor<float> gammaTmp1, gammaTmp2;
                    LoadAlign(gammaTmp1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaTmp2, gammaAddr2 + i * V_LENGTH);
                    Adds(gammaFp32Reg1, gammaTmp1, 1.0f, maskReg);
                    Adds(gammaFp32Reg2, gammaTmp2, 1.0f, maskReg);
                } else {
                    LoadAlign(gammaFp32Reg1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaFp32Reg2, gammaAddr2 + i * V_LENGTH);
                }
                Cast<float, DX, castTraitB162B32>(xReg1, xB16Reg1, maskReg);
                Cast<float, DX, castTraitB162B32>(xReg2, xB16Reg2, maskReg);
                Mul(dst1Reg, xReg1, rstdReg, maskReg);
                Mul(dst2Reg, xReg2, rstdReg, maskReg);
                Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
                Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, gammaReg1, yReg1, vRegTmp1;
            RegTensor<float> xReg2, gammaReg2, yReg2, vRegTmp2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                if constexpr (IS_GEMMA) {
                    RegTensor<float> gammaTmp1, gammaTmp2;
                    LoadAlign(gammaTmp1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaTmp2, gammaAddr2 + i * V_LENGTH);
                    Adds(gammaReg1, gammaTmp1, 1.0f, maskReg);
                    Adds(gammaReg2, gammaTmp2, 1.0f, maskReg);
                } else {
                    LoadAlign(gammaReg1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaReg2, gammaAddr2 + i * V_LENGTH);
                }
                Mul(vRegTmp1, xReg1, rstdReg, maskReg);
                Mul(vRegTmp2, xReg2, rstdReg, maskReg);
                Mul(yReg1, vRegTmp1, gammaReg1, maskReg);
                Mul(yReg2, vRegTmp2, gammaReg2, maskReg);
                StoreAlign(yAddr1 + i * V_LENGTH, yReg1, maskReg);
                StoreAlign(yAddr2 + i * V_LENGTH, yReg2, maskReg);
            }
        }
    }
}

/*!
 * The num of each level elements is 256, ReduceSum these elements and store to the next level.
 * @param level1Local level1Tensor
 * @param level2Local level2Tensor
 * @param level3Local level3Tensor
 * @param level1 level1 elements
 * @param level2 level2 elements
 * @param level3 level3 elements
 * @return void
 */
__aicore__ inline void ComputeMultiLevelReduce(LocalTensor<float>& level1Local, LocalTensor<float>& level2Local,
                                               LocalTensor<float>& level3Local, uint32_t& level1, uint32_t& level2,
                                               uint32_t& level3)
{
    if (level1 == NormCommon::ONCE_VECTOR_SIZE) {
        LevelMergeRstd<false>(level2Local, level1Local, level2, NormCommon::ONCE_VECTOR_SIZE);
        level1 = 0;
        level2 += 1;
    }
    if (level2 == NormCommon::ONCE_VECTOR_SIZE) {
        LevelMergeRstd<false>(level3Local, level2Local, level3, NormCommon::ONCE_VECTOR_SIZE);
        level2 = 0;
        level3 += 1;
    }
}

__aicore__ inline void ComputeSum(LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, uint32_t offset,
                                  uint32_t count)
{
    uint32_t meanTile = count;
    uint32_t meanSreg = meanTile;

    __ubuf__ float* srcAddr = (__ubuf__ float*)srcLocal.GetPhyAddr();
    __ubuf__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vReg, vMean;
        MaskReg pregLoop;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        {
            pregLoop = UpdateMask<float>(meanSreg);
            LoadAlign(vReg, srcAddr + 0);
            Reduce<ReduceType::SUM>(vMean, vReg, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vMean, pregMerge);
        }
    }
}

template <typename T, bool SAVE_FP32>
__aicore__ inline void LoadSquareRemainTile(__ubuf__ T* mainAddr, __ubuf__ T* tailAddr, uint16_t offset1,
                                            uint16_t offset2, RegTensor<float>& mainA, RegTensor<float>& mainB,
                                            RegTensor<float>& tailA, RegTensor<float>& tailB, MaskReg& pregLoop,
                                            __ubuf__ float* xFp32MainAddr = nullptr,
                                            __ubuf__ float* xFp32TailAddr = nullptr)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16MainA, xFp16MainB, xFp16TailA, xFp16TailB;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA, mainAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB, mainAddr + offset2);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailA, tailAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16TailB, tailAddr + offset2);
        Cast<float, half, castTraitB162B32>(mainA, xFp16MainA, pregLoop);
        Cast<float, half, castTraitB162B32>(mainB, xFp16MainB, pregLoop);
        Cast<float, half, castTraitB162B32>(tailA, xFp16TailA, pregLoop);
        Cast<float, half, castTraitB162B32>(tailB, xFp16TailB, pregLoop);
        if constexpr (SAVE_FP32) {
            StoreAlign(xFp32MainAddr + offset1, mainA, pregLoop);
            StoreAlign(xFp32MainAddr + offset2, mainB, pregLoop);
            StoreAlign(xFp32TailAddr + offset1, tailA, pregLoop);
            StoreAlign(xFp32TailAddr + offset2, tailB, pregLoop);
        }
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBFp16MainA, xBFp16MainB, xBFp16TailA, xBFp16TailB;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA, mainAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB, mainAddr + offset2);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailA, tailAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16TailB, tailAddr + offset2);
        Cast<float, bfloat16_t, castTraitB162B32>(mainA, xBFp16MainA, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB, xBFp16MainB, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(tailA, xBFp16TailA, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(tailB, xBFp16TailB, pregLoop);
        if constexpr (SAVE_FP32) {
            StoreAlign(xFp32MainAddr + offset1, mainA, pregLoop);
            StoreAlign(xFp32MainAddr + offset2, mainB, pregLoop);
            StoreAlign(xFp32TailAddr + offset1, tailA, pregLoop);
            StoreAlign(xFp32TailAddr + offset2, tailB, pregLoop);
        }
    } else {
        LoadAlign(mainA, mainAddr + offset1);
        LoadAlign(mainB, mainAddr + offset2);
        LoadAlign(tailA, tailAddr + offset1);
        LoadAlign(tailB, tailAddr + offset2);
    }
    Mul(mainA, mainA, mainA, pregLoop);
    Mul(mainB, mainB, mainB, pregLoop);
    Mul(tailA, tailA, tailA, pregLoop);
    Mul(tailB, tailB, tailB, pregLoop);
}

template <typename T, bool SAVE_FP32>
__aicore__ inline void LoadSquareMasterTile(__ubuf__ T* masterAddr, uint16_t offset1, uint16_t offset2,
                                            RegTensor<float>& mainA, RegTensor<float>& mainB, MaskReg& pregLoop,
                                            __ubuf__ float* xFp32MasterAddr = nullptr)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16MainA, xFp16MainB;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainA, masterAddr + offset1);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16MainB, masterAddr + offset2);
        Cast<float, half, castTraitB162B32>(mainA, xFp16MainA, pregLoop);
        Cast<float, half, castTraitB162B32>(mainB, xFp16MainB, pregLoop);
        if constexpr (SAVE_FP32) {
            StoreAlign(xFp32MasterAddr + offset1, mainA, pregLoop);
            StoreAlign(xFp32MasterAddr + offset2, mainB, pregLoop);
        }
        Mul(mainA, mainA, mainA, pregLoop);
        Mul(mainB, mainB, mainB, pregLoop);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBFp16MainA, xBFp16MainB;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainA, masterAddr + offset1);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBFp16MainB, masterAddr + offset2);
        Cast<float, bfloat16_t, castTraitB162B32>(mainA, xBFp16MainA, pregLoop);
        Cast<float, bfloat16_t, castTraitB162B32>(mainB, xBFp16MainB, pregLoop);
        if constexpr (SAVE_FP32) {
            StoreAlign(xFp32MasterAddr + offset1, mainA, pregLoop);
            StoreAlign(xFp32MasterAddr + offset2, mainB, pregLoop);
        }
        Mul(mainA, mainA, mainA, pregLoop);
        Mul(mainB, mainB, mainB, pregLoop);
    } else {
        LoadAlign(mainA, masterAddr + offset1);
        LoadAlign(mainB, masterAddr + offset2);
        Mul(mainA, mainA, mainA, pregLoop);
        Mul(mainB, mainB, mainB, pregLoop);
    }
}

template <typename T>
__aicore__ inline void ComputeFormerImplV1MultiN(LocalTensor<T>& xLocal, LocalTensor<float>& xFp32,
                                                 LocalTensor<float>& workLocal, LocalTensor<float>& rstdLocal,
                                                 float avgFactor, float epsilon, uint32_t offset, uint32_t count,
                                                 uint32_t powerSplit, uint32_t curRows)
{
    uint32_t remainTile = count - powerSplit;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;

    __ubuf__ T* mainAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* tailAddr = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(powerSplit);
    __ubuf__ T* masterAddr = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(remainTile);
    __ubuf__ float *xFp32MainAddr, *xFp32TailAddr, *xFp32MasterAddr;
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
        xFp32MainAddr = (__ubuf__ float*)xFp32.GetPhyAddr();
        xFp32TailAddr = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(powerSplit);
        xFp32MasterAddr = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(remainTile);
    }
    uint32_t curRowsAlign = CeilDiv((int32_t)curRows, 2);
    int64_t unrollOffset = (curRows / 2) * count;
    bool isWithTail = curRowsAlign - (curRows / 2);
    uint32_t tailOffset = offset + curRows / 2;

    __ubuf__ T* mainAddr1 = (__ubuf__ T*)xLocal.GetPhyAddr() + unrollOffset;
    __ubuf__ T* tailAddr1 = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(powerSplit) + unrollOffset;
    __ubuf__ T* masterAddr1 = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(remainTile) + unrollOffset;
    __ubuf__ float *xFp32MainAddr1, *xFp32TailAddr1, *xFp32MasterAddr1;
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
        xFp32MainAddr1 = (__ubuf__ float*)xFp32.GetPhyAddr() + unrollOffset;
        xFp32TailAddr1 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(powerSplit) + unrollOffset;
        xFp32MasterAddr1 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(remainTile) + unrollOffset;
    }

    __ubuf__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    __ubuf__ float* workAddr1 = (__ubuf__ float*)workLocal.GetPhyAddr() + NormCommon::ONCE_VECTOR_SIZE;
    __ubuf__ float* rstdAddr1 = (__ubuf__ float*)rstdLocal.GetPhyAddr() + curRows / 2;

    __VEC_SCOPE__
    {
        for (uint16_t row = 0; row < static_cast<uint16_t>(curRows / 2); row++) {
            uint32_t remainSreg = remainTile;
            uint32_t masterSreg = masterTile;
            uint32_t mergeSreg = mergeTile;
            uint32_t meanSreg = meanTile;
            RegTensor<float> mainA, mainB, tailA, tailB, vMean, vDupReg, rstdReg;
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            uint32_t remainSreg1 = remainTile;
            uint32_t masterSreg1 = masterTile;
            uint32_t mergeSreg1 = mergeTile;
            uint32_t meanSreg1 = meanTile;
            RegTensor<float> mainA1, mainB1, tailA1, tailB1, vMean1, vDupReg1, rstdReg1;
            MaskReg pregMain1 = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge1 = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop1;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregLoop = UpdateMask<float>(remainSreg);
                LoadSquareRemainTile<T, true>(mainAddr, tailAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA,
                                              mainB, tailA, tailB, pregLoop, xFp32MainAddr, xFp32TailAddr);
                Add(mainA, mainA, tailA, pregLoop);
                Add(mainB, mainB, tailB, pregLoop);
                Add(mainA, mainA, mainB, pregLoop);
                Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregLoop = UpdateMask<float>(masterSreg);
                LoadSquareMasterTile<T, true>(masterAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA, mainB,
                                              pregLoop, xFp32MasterAddr);
                Add(mainA, mainA, mainB, pregLoop);
                Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vMean, pregMerge);
            }
            // unroll part
            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregLoop1 = UpdateMask<float>(remainSreg1);
                LoadSquareRemainTile<T, true>(mainAddr1, tailAddr1, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH,
                                              mainA1, mainB1, tailA1, tailB1, pregLoop1, xFp32MainAddr1,
                                              xFp32TailAddr1);
                Add(mainA1, mainA1, tailA1, pregLoop1);
                Add(mainB1, mainB1, tailB1, pregLoop1);
                Add(mainA1, mainA1, mainB1, pregLoop1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean1, pregMerge1);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregLoop1 = UpdateMask<float>(masterSreg1);
                LoadSquareMasterTile<T, true>(masterAddr1, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA1,
                                              mainB1, pregLoop1, xFp32MasterAddr1);
                Add(mainA1, mainA1, mainB1, pregLoop1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + remainRepeats + i, vMean1, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregLoop = UpdateMask<float>(mergeSreg);
                LoadAlign(mainA, workAddr + (i * 2 + 0) * V_LENGTH);
                LoadAlign(mainB, workAddr + (i * 2 + 1) * V_LENGTH);
                Add(mainA, mainA, mainB, pregLoop);
                Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
            }
            // unroll part
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregLoop1 = UpdateMask<float>(mergeSreg1);
                LoadAlign(mainA1, workAddr1 + (i * 2 + 0) * V_LENGTH);
                LoadAlign(mainB1, workAddr1 + (i * 2 + 1) * V_LENGTH);
                Add(mainA1, mainA1, mainB1, pregLoop1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean1, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            {
                pregLoop = UpdateMask<float>(meanSreg);
                LoadAlign(mainA, workAddr + 0);
                Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
                Muls(vMean, vMean, avgFactor, pregMerge);
                Adds(vMean, vMean, epsilon, pregMerge);
                Sqrt(vMean, vMean, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(rstdReg, vDupReg, vMean, pregMerge);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + offset, rstdReg, pregMerge);
            }
            // unroll part
            {
                pregLoop1 = UpdateMask<float>(meanSreg1);
                LoadAlign(mainA1, workAddr1 + 0);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                Muls(vMean1, vMean1, avgFactor, pregMerge1);
                Adds(vMean1, vMean1, epsilon, pregMerge1);
                Sqrt(vMean1, vMean1, pregMerge1);
                Duplicate(vDupReg1, float(1.0), pregMerge1);
                Div(rstdReg1, vDupReg1, vMean1, pregMerge1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr1 + offset, rstdReg1, pregMerge1);
            }
            offset += 1;
            mainAddr += int64_t(count);
            tailAddr += int64_t(count);
            masterAddr += int64_t(count);
            xFp32MainAddr += int64_t(count);
            xFp32TailAddr += int64_t(count);
            xFp32MasterAddr += int64_t(count);

            mainAddr1 += int64_t(count);
            tailAddr1 += int64_t(count);
            masterAddr1 += int64_t(count);
            xFp32MainAddr1 += int64_t(count);
            xFp32TailAddr1 += int64_t(count);
            xFp32MasterAddr1 += int64_t(count);
        }
    }
    uint32_t tailDataOffset = unrollOffset + (curRows / 2) * count;
    __ubuf__ T* mainAddr2 = (__ubuf__ T*)xLocal.GetPhyAddr() + tailDataOffset;
    __ubuf__ T* tailAddr2 = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(powerSplit) + tailDataOffset;
    __ubuf__ T* masterAddr2 = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(remainTile) + tailDataOffset;
    __ubuf__ float *xFp32MainAddr2, *xFp32TailAddr2, *xFp32MasterAddr2;
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
        xFp32MainAddr2 = (__ubuf__ float*)xFp32.GetPhyAddr() + tailDataOffset;
        xFp32TailAddr2 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(powerSplit) + tailDataOffset;
        xFp32MasterAddr2 = (__ubuf__ float*)xFp32.GetPhyAddr() + int64_t(remainTile) + tailDataOffset;
    }
    if (isWithTail) {
        __VEC_SCOPE__
        {
            uint32_t remainSreg1 = remainTile;
            uint32_t masterSreg1 = masterTile;
            uint32_t mergeSreg1 = mergeTile;
            uint32_t meanSreg1 = meanTile;
            RegTensor<float> mainA1, mainB1, tailA1, tailB1, vMean1, vDupReg1, rstdReg1;
            MaskReg pregMain1 = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge1 = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop1;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregLoop1 = UpdateMask<float>(remainSreg1);
                LoadSquareRemainTile<T, true>(mainAddr2, tailAddr2, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH,
                                              mainA1, mainB1, tailA1, tailB1, pregLoop1, xFp32MainAddr2,
                                              xFp32TailAddr2);
                Add(mainA1, mainA1, tailA1, pregLoop1);
                Add(mainB1, mainB1, tailB1, pregLoop1);
                Add(mainA1, mainA1, mainB1, pregLoop1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean1, pregMerge1);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregLoop1 = UpdateMask<float>(masterSreg1);
                LoadSquareMasterTile<T, true>(masterAddr2, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA1,
                                              mainB1, pregLoop1, xFp32MasterAddr2);
                Add(mainA1, mainA1, mainB1, pregLoop1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + remainRepeats + i, vMean1, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregLoop1 = UpdateMask<float>(mergeSreg1);
                LoadAlign(mainA1, workAddr1 + (i * 2 + 0) * V_LENGTH);
                LoadAlign(mainB1, workAddr1 + (i * 2 + 1) * V_LENGTH);
                Add(mainA1, mainA1, mainB1, pregLoop1);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr1 + i, vMean1, pregMerge1);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            {
                pregLoop1 = UpdateMask<float>(meanSreg1);
                LoadAlign(mainA1, workAddr1 + 0);
                Reduce<ReduceType::SUM>(vMean1, mainA1, pregLoop1);
                Muls(vMean1, vMean1, avgFactor, pregMerge1);
                Adds(vMean1, vMean1, epsilon, pregMerge1);
                Sqrt(vMean1, vMean1, pregMerge1);
                Duplicate(vDupReg1, float(1.0), pregMerge1);
                Div(rstdReg1, vDupReg1, vMean1, pregMerge1);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr1 + tailOffset, rstdReg1, pregMerge1);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ComputeFormerImplV2(LocalTensor<float>& dstLocal, LocalTensor<T>& xLocal,
                                           LocalTensor<float>& workLocal, uint32_t offset, uint32_t count,
                                           uint32_t powerSplit)
{
    uint32_t remainTile = count - powerSplit;
    uint32_t remainSreg = remainTile;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint32_t masterSreg = masterTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint32_t mergeSreg = mergeTile;
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;
    uint32_t meanSreg = meanTile;

    __ubuf__ T* mainAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* tailAddr = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(powerSplit);
    __ubuf__ T* masterAddr = (__ubuf__ T*)xLocal.GetPhyAddr() + int64_t(remainTile);

    __ubuf__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __ubuf__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> mainA, mainB, tailA, tailB, vMean, vDupReg, rstdReg;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
            pregLoop = UpdateMask<float>(remainSreg);
            LoadSquareRemainTile<T, false>(mainAddr, tailAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA,
                                           mainB, tailA, tailB, pregLoop);
            Add(mainA, mainA, tailA, pregLoop);
            Add(mainB, mainB, tailB, pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
        }
        for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
            pregLoop = UpdateMask<float>(masterSreg);
            LoadSquareMasterTile<T, false>(masterAddr, (i * 2 + 0) * V_LENGTH, (i * 2 + 1) * V_LENGTH, mainA, mainB,
                                           pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vMean, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
            pregLoop = UpdateMask<float>(mergeSreg);
            LoadAlign(mainA, workAddr + (i * 2 + 0) * V_LENGTH);
            LoadAlign(mainB, workAddr + (i * 2 + 1) * V_LENGTH);
            Add(mainA, mainA, mainB, pregLoop);
            Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vMean, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        {
            pregLoop = UpdateMask<float>(meanSreg);
            LoadAlign(mainA, workAddr + 0);
            Reduce<ReduceType::SUM>(vMean, mainA, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vMean, pregMerge);
        }
    }
}

} // namespace RmsNorm
#endif // ADD_RMS_NORM_BIAS_A5_OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_COMMON_H
