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
// Source: norm/norm_common/op_kernel/reduce_common_regbase.h

/*!
 * \file reduce_common_regbase.h
 * \brief reduce common regbase file
 */
#ifndef ADD_RMS_NORM_BIAS_A5_REDUCE_COMMON_REGBASE_H_RMS_NORM
#define ADD_RMS_NORM_BIAS_A5_REDUCE_COMMON_REGBASE_H_RMS_NORM
#include "kernel_operator.h"

namespace NormCommon {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

namespace NormCommonRegbase {
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline T Aligned(T value, T alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace NormCommonRegbase

constexpr int32_t VL_SIZE = NormCommonRegbase::GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / static_cast<int32_t>(sizeof(float)));
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr uint16_t DICHOTOMY_ADD_COEFF = 2;

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline void DichotomyAdd(RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop,
                                    uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> tmpReg1;
    RegTensor<float> tmpReg2;
    RegTensor<float> tmpReg3;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop = innerLoop / DICHOTOMY_ADD_COEFF;
        for (uint16_t i = 0; i < innerLoop; i++) {
            DataCopy(tmpReg1, src + i * V_LENGTH);
            DataCopy(tmpReg2, src + (i + innerLoop) * V_LENGTH);
            Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
            DataCopy(src + i * V_LENGTH, tmpReg3, pregMain);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    uint32_t sreg0 = lastNum;
    MaskReg pregLoop = UpdateMask<float>(sreg0);
    DataCopy(tmpReg3, src);
    ReduceSum(dstReg, tmpReg3, pregLoop);
}

template <typename U>
__aicore__ inline void LoadTwoCloseRegVF(RegTensor<U>& dstA, RegTensor<U>& dstB, __local_mem__ U* srcAddr,
                                         uint16_t offset)
{
    if constexpr (IsSameType<U, float>::value) {
        DataCopy(dstA, srcAddr + offset);
        DataCopy(dstB, srcAddr + offset + V_LENGTH);
    } else {
        DataCopy<U, LoadDist::DIST_UNPACK_B16>(dstA, srcAddr + offset);
        DataCopy<U, LoadDist::DIST_UNPACK_B16>(dstB, srcAddr + offset + V_LENGTH);
    }
}

template <typename U>
__aicore__ inline void CastAddVF(RegTensor<float>& dstReg, RegTensor<U>& src1Reg, RegTensor<U>& src2Reg,
                                 MaskReg& pregLoop)
{
    if constexpr (IsSameType<U, float>::value) {
        Add(dstReg, src1Reg, src2Reg, pregLoop);
    } else {
        RegTensor<float> src1RegFp32, src2RegFp32;
        Cast<float, U, castTraitB162B32>(src1RegFp32, src1Reg, pregLoop);
        Cast<float, U, castTraitB162B32>(src2RegFp32, src2Reg, pregLoop);
        Add(dstReg, src1RegFp32, src2RegFp32, pregLoop);
    }
}

/**
 * @brief Load and cast to fp32 reg.
 * @param offset idx of VF loop.
 */
template <typename T>
__aicore__ inline void LoadCastRegVF(RegTensor<float>& dstTensor, __local_mem__ T* srcAddr, uint16_t offset,
                                     MaskReg& pregLoop)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstTensor, srcAddr + offset * V_LENGTH);
    } else {
        RegTensor<T> loadTmp;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(loadTmp, srcAddr + offset * V_LENGTH);
        Cast<float, T, castTraitB162B32>(dstTensor, loadTmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void CastStoreTwoCloseRegVF(__local_mem__ T* dstAddr, RegTensor<float>& srcA, RegTensor<float>& srcB,
                                              uint16_t offset, MaskReg& pregLoop)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dstAddr + offset, srcA, pregLoop);
        DataCopy(dstAddr + offset + V_LENGTH, srcB, pregLoop);
    } else {
        RegTensor<T> srcATmp, srcBTmp;
        Cast<T, float, castTraitB322B16>(srcATmp, srcA, pregLoop);
        Cast<T, float, castTraitB322B16>(srcBTmp, srcB, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + offset, srcATmp, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr + offset + V_LENGTH, srcBTmp, pregLoop);
    }
}

/**
 * @brief Use VF to Compute reduceSum.
 *        dstLocal = reduceSum((x1+x2)^2)
 *        If HAS_XOUT is true, return xOut = (x1.to(float) + x2.to(float)).to(dtype).
 *        If HAS_XOUT_FP32 is true, return xOutFp32 = x1.to(float) + x2.to(float).
 *        If IS_RSTD is true, dstLocal = 1.0 / sqrt(avgFactor * reduceSum((x1+x2)^2) + epsilon)
 *        Use float32 VL_LENGTH
 */
template <typename U, bool HAS_XOUT = false, bool HAS_XOUT_FP32 = false, bool IS_RSTD = false>
__aicore__ inline void ReduceSumRstd(LocalTensor<float>& dstLocal, LocalTensor<U>& xOutLocal,
                                     LocalTensor<float>& xOutFp32Local, LocalTensor<U>& x1Local,
                                     LocalTensor<U>& x2Local, LocalTensor<float>& workLocal, uint32_t dstOffset,
                                     uint32_t count, uint32_t powerSplit, float avgFactor = 1.0f, float epsilon = 0.0f)
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

    __local_mem__ U* x1MainAddr = (__ubuf__ U*)x1Local.GetPhyAddr();
    __local_mem__ U* x1TailAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x1MasterAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* x2MainAddr = (__ubuf__ U*)x2Local.GetPhyAddr();
    __local_mem__ U* x2TailAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x2MasterAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* xOutMainAddr;
    __local_mem__ U* xOutTailAddr;
    __local_mem__ U* xOutMasterAddr;
    if constexpr (HAS_XOUT) {
        xOutMainAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr();
        xOutTailAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(powerSplit);
        xOutMasterAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* xOutFp32MainAddr;
    __local_mem__ float* xOutFp32TailAddr;
    __local_mem__ float* xOutFp32MasterAddr;
    if constexpr (HAS_XOUT_FP32) {
        xOutFp32MainAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr();
        xOutFp32TailAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(powerSplit);
        xOutFp32MasterAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> mainA, mainB, tailA, tailB, vSum, vDupReg;
        RegTensor<U> x1MainA, x1MainB, x1TailA, x1TailB;
        RegTensor<U> x2MainA, x2MainB, x2TailA, x2TailB;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
            pregLoop = UpdateMask<float>(remainSreg);
            uint16_t offset = i * 2 * V_LENGTH;
            // 1. Copy in reg
            LoadTwoCloseRegVF(x1MainA, x1MainB, x1MainAddr, offset);
            LoadTwoCloseRegVF(x1TailA, x1TailB, x1TailAddr, offset);
            LoadTwoCloseRegVF(x2MainA, x2MainB, x2MainAddr, offset);
            LoadTwoCloseRegVF(x2TailA, x2TailB, x2TailAddr, offset);
            // 2. Cast add
            CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
            CastAddVF(tailA, x1TailA, x2TailA, pregLoop);
            CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
            CastAddVF(tailB, x1TailB, x2TailB, pregLoop);
            if constexpr (HAS_XOUT) {
                CastStoreTwoCloseRegVF(xOutMainAddr, mainA, mainB, offset, pregLoop);
                CastStoreTwoCloseRegVF(xOutTailAddr, tailA, tailB, offset, pregLoop);
            }
            if constexpr (HAS_XOUT_FP32) {
                CastStoreTwoCloseRegVF(xOutFp32MainAddr, mainA, mainB, offset, pregLoop);
                CastStoreTwoCloseRegVF(xOutFp32TailAddr, tailA, tailB, offset, pregLoop);
            }
            // 3. Cal x^2
            Mul(mainA, mainA, mainA, pregLoop);
            Mul(tailA, tailA, tailA, pregLoop);
            Mul(mainB, mainB, mainB, pregLoop);
            Mul(tailB, tailB, tailB, pregLoop);
            Add(mainA, mainA, tailA, pregLoop);
            Add(mainB, mainB, tailB, pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vSum, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
        }
        for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
            uint16_t offset = i * 2 * V_LENGTH;
            pregLoop = UpdateMask<float>(masterSreg);
            // 1. Copy in reg
            LoadTwoCloseRegVF(x1MainA, x1MainB, x1MasterAddr, offset);
            LoadTwoCloseRegVF(x2MainA, x2MainB, x2MasterAddr, offset);
            // 2. Cast add
            CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
            CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
            if constexpr (HAS_XOUT) {
                CastStoreTwoCloseRegVF(xOutMasterAddr, mainA, mainB, offset, pregLoop);
            }
            if constexpr (HAS_XOUT_FP32) {
                CastStoreTwoCloseRegVF(xOutFp32MasterAddr, mainA, mainB, offset, pregLoop);
            }
            // 3. Cal x^2
            Mul(mainA, mainA, mainA, pregLoop);
            Mul(mainB, mainB, mainB, pregLoop);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vSum, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vSum, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
            pregLoop = UpdateMask<float>(mergeSreg);
            uint16_t offset = i * 2 * V_LENGTH;
            LoadTwoCloseRegVF(mainA, mainB, workAddr, offset);
            Add(mainA, mainA, mainB, pregLoop);
            ReduceSum(vSum, mainA, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        pregLoop = UpdateMask<float>(meanSreg);
        DataCopy(mainA, workAddr);
        ReduceSum(vSum, mainA, pregLoop);
        if constexpr (IS_RSTD) {
            Muls(vSum, vSum, avgFactor, pregMerge);
            Adds(vSum, vSum, epsilon, pregMerge);
            Sqrt(vSum, vSum, pregMerge);
            Duplicate(vDupReg, float(1.0), pregMerge);
            Div(vSum, vDupReg, vSum, pregMerge);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + dstOffset, vSum, pregMerge);
    }
}

/**
 * @brief Use VF to Compute reduceSum(multi line).
 *        dstLocal = reduceSum((x1+x2)^2)
 *        If HAS_XOUT_FP32 is true, return xOutFp32 = x1.to(float) + x2.to(float).
 *        If IS_RSTD is true, dstLocal = 1.0 / sqrt(avgFactor * reduceSum((x1+x2)^2) + epsilon)
 *        Use float32 VL_LENGTH
 */
template <typename U, bool HAS_XOUT = false, bool HAS_XOUT_FP32 = false, bool IS_RSTD = false>
__aicore__ inline void ReduceSumRstdMulti(LocalTensor<float>& rstdLocal, LocalTensor<U>& xOutLocal,
                                          LocalTensor<float>& xOutFp32Local, LocalTensor<U>& x1Local,
                                          LocalTensor<U>& x2Local, LocalTensor<float>& workLocal,
                                          uint32_t rstdOffsetStart, uint32_t count, uint32_t powerSplit,
                                          uint32_t repeatTimes, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    uint32_t rstdOffset = rstdOffsetStart;
    uint32_t remainTile = count - powerSplit;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;

    __local_mem__ U* x1MainAddr = (__ubuf__ U*)x1Local.GetPhyAddr();
    __local_mem__ U* x1TailAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x1MasterAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* x2MainAddr = (__ubuf__ U*)x2Local.GetPhyAddr();
    __local_mem__ U* x2TailAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x2MasterAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U *xOutMainAddr, *xOutTailAddr, *xOutMasterAddr;
    if constexpr (HAS_XOUT) {
        xOutMainAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr();
        xOutTailAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(powerSplit);
        xOutMasterAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float *xOutFp32MainAddr, *xOutFp32TailAddr, *xOutFp32MasterAddr;
    if constexpr (HAS_XOUT_FP32) {
        xOutFp32MainAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr();
        xOutFp32TailAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(powerSplit);
        xOutFp32MasterAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();

        for (uint16_t row = 0; row < (uint16_t)repeatTimes; row++) {
            uint32_t remainSreg = remainTile;
            uint32_t masterSreg = masterTile;
            uint32_t mergeSreg = mergeTile;
            uint32_t meanSreg = meanTile;
            RegTensor<U> x1MainA, x1MainB, x1TailA, x1TailB;
            RegTensor<U> x2MainA, x2MainB, x2TailA, x2TailB;
            RegTensor<float> mainA, mainB, tailA, tailB, vSum, vDupReg, rstdReg;
            MaskReg pregLoop;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregLoop = UpdateMask<float>(remainSreg);
                uint16_t offset = i * 2 * V_LENGTH;
                // 1. Copy in reg
                LoadTwoCloseRegVF(x1MainA, x1MainB, x1MainAddr, offset);
                LoadTwoCloseRegVF(x1TailA, x1TailB, x1TailAddr, offset);
                LoadTwoCloseRegVF(x2MainA, x2MainB, x2MainAddr, offset);
                LoadTwoCloseRegVF(x2TailA, x2TailB, x2TailAddr, offset);
                // 2. Cast add
                CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
                CastAddVF(tailA, x1TailA, x2TailA, pregLoop);
                CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
                CastAddVF(tailB, x1TailB, x2TailB, pregLoop);
                if constexpr (HAS_XOUT) {
                    CastStoreTwoCloseRegVF(xOutMainAddr, mainA, mainB, offset, pregLoop);
                    CastStoreTwoCloseRegVF(xOutTailAddr, tailA, tailB, offset, pregLoop);
                }
                if constexpr (HAS_XOUT_FP32) {
                    CastStoreTwoCloseRegVF(xOutFp32MainAddr, mainA, mainB, offset, pregLoop);
                    CastStoreTwoCloseRegVF(xOutFp32TailAddr, tailA, tailB, offset, pregLoop);
                }
                // 3. Cal x^2
                Mul(mainA, mainA, mainA, pregLoop);
                Mul(tailA, tailA, tailA, pregLoop);
                Mul(mainB, mainB, mainB, pregLoop);
                Mul(tailB, tailB, tailB, pregLoop);
                Add(mainA, mainA, tailA, pregLoop);
                Add(mainB, mainB, tailB, pregLoop);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vSum, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregLoop = UpdateMask<float>(masterSreg);
                uint16_t offset = i * 2 * V_LENGTH;
                // 1. Copy in reg
                LoadTwoCloseRegVF(x1MainA, x1MainB, x1MasterAddr, offset);
                LoadTwoCloseRegVF(x2MainA, x2MainB, x2MasterAddr, offset);
                // 2. Cast add
                CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
                CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
                if constexpr (HAS_XOUT) {
                    CastStoreTwoCloseRegVF(xOutMasterAddr, mainA, mainB, offset, pregLoop);
                }
                if constexpr (HAS_XOUT_FP32) {
                    CastStoreTwoCloseRegVF(xOutFp32MasterAddr, mainA, mainB, offset, pregLoop);
                }
                // 3. Cal x^2
                Mul(mainA, mainA, mainA, pregLoop);
                Mul(mainB, mainB, mainB, pregLoop);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vSum, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vSum, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregLoop = UpdateMask<float>(mergeSreg);
                uint16_t offset = i * 2 * V_LENGTH;
                LoadTwoCloseRegVF(mainA, mainB, workAddr, offset);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vSum, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            pregLoop = UpdateMask<float>(meanSreg);
            DataCopy(mainA, workAddr);
            ReduceSum(vSum, mainA, pregLoop);
            if constexpr (IS_RSTD) {
                Muls(vSum, vSum, avgFactor, pregMerge);
                Adds(vSum, vSum, epsilon, pregMerge);
                Sqrt(vSum, vSum, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(rstdReg, vDupReg, vSum, pregMerge);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + rstdOffset, rstdReg, pregMerge);

            rstdOffset++;
            x1MainAddr += int64_t(count);
            x1TailAddr += int64_t(count);
            x1MasterAddr += int64_t(count);
            x2MainAddr += int64_t(count);
            x2TailAddr += int64_t(count);
            x2MasterAddr += int64_t(count);
            if constexpr (HAS_XOUT) {
                xOutMainAddr += int64_t(count);
                xOutTailAddr += int64_t(count);
                xOutMasterAddr += int64_t(count);
            }
            if constexpr (HAS_XOUT_FP32) {
                xOutFp32MainAddr += int64_t(count);
                xOutFp32TailAddr += int64_t(count);
                xOutFp32MasterAddr += int64_t(count);
            }
        }
    }
}

template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstdNewtonRaphsonReg(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& preg,
                                                   float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    RegTensor<float> r;
    RegTensor<float> y;
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> t1;
    RegTensor<float> t3;
    RegTensor<float> t4;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    MaskReg cmpRegZero;
    MaskReg cmpRegInf;

    Duplicate(scalarInf, POS_INF, preg);
    Duplicate(scalarZero, float(0.0), preg);
    Duplicate(one, float(1.0), preg);
    Duplicate(scalar1, SCALAR3, preg);
    Duplicate(t1, SCALAR2, preg);
    Duplicate(s, float(1.0), preg);

    Adds(var, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(var, var, SCALAR0, preg);
    }
    Div(r, one, var, preg);
    Sqrt(y, r, preg);
    Muls(t, var, SCALAR1, preg);
    Mul(t, t, y, preg);
    Mula(t1, t, y, preg);
    Mul(rstd, y, t1, preg);
    Muls(t3, var, float(-1.0), preg);
    Mula(s, t3, r, preg);
    Muls(t4, rstd, float(-1.0), preg);
    Mula(r, t4, rstd, preg);
    Mula(s, var, r, preg);
    Mul(s, s, rstd, preg);
    Mula(rstd, s, scalar1, preg);
    CompareScalar(cmpRegZero, var, POS_INF, preg);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    CompareScalar(cmpRegInf, var, float(0.0), preg);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

template <typename T>
__aicore__ inline void LoadTensorUnAlignForDtypeT(__local_mem__ T*& src, RegTensor<float>& dst,
                                                  AscendC::MicroAPI::UnalignReg& uSrc, MaskReg& preg,
                                                  uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::DataCopyUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, uSrc, src,
                                                                                                    postUpdateStride);
    } else {
        RegTensor<T> xB16;
        RegTensor<T> xB16Unpack;
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(xB16, uSrc, src,
                                                                                                postUpdateStride);
        UnPack((RegTensor<uint32_t>&)xB16Unpack, (RegTensor<uint16_t>&)xB16);
        Cast<float, T, castTraitB162B32>(dst, xB16Unpack, preg);
    }
}

template <typename T>
__aicore__ inline void StoreTensorUnAlignForDtypeT(__local_mem__ T*& dst, RegTensor<float>& src,
                                                   AscendC::MicroAPI::UnalignReg& uDst, MaskReg& preg,
                                                   uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::DataCopyUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, src, uDst,
                                                                                                    postUpdateStride);
    } else {
        RegTensor<T> xB16;
        RegTensor<T> xB16Pack;
        Cast<T, float, castTraitB322B16>(xB16, src, preg);
        Pack((RegTensor<uint16_t>&)xB16Pack, (RegTensor<uint32_t>&)xB16);
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, xB16Pack, uDst,
                                                                                                postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void LoadTensorUnAlignForDtypeT(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
                                                  uint32_t postUpdateStride)
{
    AscendC::MicroAPI::UnalignReg uSrc;
    __local_mem__ T* srcTmp = src;
    AscendC::MicroAPI::DataCopyUnAlignPre(uSrc, srcTmp);
    LoadTensorUnAlignForDtypeT(srcTmp, dst, uSrc, preg, postUpdateStride);
}

template <typename T>
__aicore__ inline void StoreTensorUnAlignForDtypeT(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg,
                                                   uint32_t postUpdateStride)
{
    AscendC::MicroAPI::UnalignReg uDst;
    __local_mem__ T* dstTmp = dst;
    StoreTensorUnAlignForDtypeT(dstTmp, src, uDst, preg, postUpdateStride);
    AscendC::MicroAPI::DataCopyUnAlignPost(dstTmp, uDst, 0);
}

// NOTE: x is overwritten in place (x = (x - mean) * scale * rstd); only y is the
// downstream-usable result. Callers must not rely on the original x after this call.
__aicore__ inline void NormalizeWithScaleBiasReg(RegTensor<float>& x, RegTensor<float>& scale, RegTensor<float>& bias,
                                                 RegTensor<float>& mean, RegTensor<float>& rstd, RegTensor<float>& y,
                                                 MaskReg& preg)
{
    Sub(x, x, mean, preg);
    Mul(x, x, scale, preg);
    Mul(x, x, rstd, preg);
    Add(y, x, bias, preg);
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(__local_mem__ float* src, __local_mem__ float* dst, uint32_t rowCount,
                                                float epsilon, float avgFactor = 1.0f, uint32_t vectorLen = V_LENGTH)
{
    uint16_t loopRows = static_cast<uint16_t>((rowCount + vectorLen - 1) / vectorLen);
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;

        uint32_t sreg = rowCount;
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(var, src + i * vectorLen);
            if constexpr (NEED_AVG_FACTOR) {
                Muls(var, var, avgFactor, pregLoop);
            }
            ComputeRstdNewtonRaphsonReg<NEED_MAX>(var, rstd, pregLoop, epsilon);
            DataCopy(dst + i * vectorLen, rstd, pregLoop);
        }
    }
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(LocalTensor<float> srcLocal, LocalTensor<float> dstLocal,
                                                uint32_t rowCount, float epsilon, float avgFactor = 1.0f,
                                                uint32_t vectorLen = V_LENGTH)
{
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    ComputeRstdNewtonRaphson<NEED_MAX, NEED_AVG_FACTOR>(src, dst, rowCount, epsilon, avgFactor, vectorLen);
}

/*!
 * @brief Compute ReduceSum mean
 *        IS_RSTD: if True, will cal rstd otherwise sum.
 * @param dstLocal dst levelTensor
 * @param srcLocal src LevelTensor
 * @param offset dst offset
 * @param count src level size, must be ONCE_VECTOR_SIZE
 * @param avgFactor avgFactor for cal rstd
 * @param epsilon epsilon for cal rstd
 * @return
 */
template <bool IS_RSTD>
__aicore__ inline void LevelMergeRstd(LocalTensor<float>& dstLocal, LocalTensor<float> srcLocal, uint64_t offset,
                                      uint32_t count, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    uint64_t calCount = count / 4; // Div 4 for VF parallel execution.
    uint32_t sreg = (uint32_t)(calCount);
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);
    uint32_t meanTile = repeatTimes;

    __local_mem__ float* src1Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 0 * calCount;
    __local_mem__ float* src2Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 1 * calCount;
    __local_mem__ float* src3Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 2 * calCount;
    __local_mem__ float* src4Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 3 * calCount;
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vRegA, vRegB, vRegC, vRegD, dstReg, vSum, vDupReg;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(vRegA, src1Addr + i * V_LENGTH);
            DataCopy(vRegB, src2Addr + i * V_LENGTH);
            DataCopy(vRegC, src3Addr + i * V_LENGTH);
            DataCopy(vRegD, src4Addr + i * V_LENGTH);
            Add(vRegA, vRegA, vRegB, pregLoop);
            Add(vRegC, vRegC, vRegD, pregLoop);
            Add(dstReg, vRegA, vRegC, pregLoop);
            ReduceSum(vSum, dstReg, pregLoop);
            if constexpr (IS_RSTD) {
                Muls(vSum, vSum, avgFactor, pregMerge);
                Adds(vSum, vSum, epsilon, pregMerge);
                Sqrt(vSum, vSum, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(vSum, vDupReg, vSum, pregMerge);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vSum, pregMerge);
        }
    }
}

/*!
 * @brief compute final ReduceSum result
 *        IS_RSTD: if True, will cal rstd otherwise sum.
 * @param dstLocal dst Tensor
 * @param offset dst offset
 * @param level1Local level1 Tensor
 * @param level2Local level2 Tensor
 * @param level3Local level3 Tensor
 * @param level1 level1 elements
 * @param level2 level2 elements
 * @param level3 level3 elements
 * @param avgFactor avgFactor for cal rstd
 * @param epsilon epsilon for cal rstd
 * @return
 */
template <bool IS_RSTD>
__aicore__ inline void ComputeMultiLevelRstd(LocalTensor<float>& dstLocal, uint32_t offset,
                                             LocalTensor<float>& level1Local, LocalTensor<float>& level2Local,
                                             LocalTensor<float>& level3Local, uint32_t& level1, uint32_t& level2,
                                             float avgFactor = 1.0f, float epsilon = 0.0f)
{
    if (level1 > 0 && level1 < ONCE_VECTOR_SIZE) {
        LevelMergeRstd<IS_RSTD>(dstLocal, level1Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    } else if (level2 > 0 && level2 < ONCE_VECTOR_SIZE) {
        LevelMergeRstd<IS_RSTD>(dstLocal, level2Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    } else {
        LevelMergeRstd<IS_RSTD>(dstLocal, level3Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    }
}

namespace NormCommonRegbase {

template <typename T, LoadDist FLOAT_LOAD_DIST = LoadDist::DIST_NORM,
          LoadDist NON_FLOAT_LOAD_DIST = LoadDist::DIST_UNPACK_B16>
__aicore__ inline void LoadRegForDtype(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, FLOAT_LOAD_DIST>(dst, src + offset);
    } else {
        RegTensor<T> srcReg;
        DataCopy<T, NON_FLOAT_LOAD_DIST>(srcReg, src + offset);
        Cast<float, T, castTraitB162B32>(dst, srcReg, preg);
    }
}

template <typename T, StoreDist FLOAT_STORE_DIST = StoreDist::DIST_NORM,
          StoreDist NON_FLOAT_STORE_DIST = StoreDist::DIST_PACK_B32>
__aicore__ inline void StoreRegForDtype(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, FLOAT_STORE_DIST>(dst + offset, src, preg);
    } else {
        RegTensor<T> dstReg;
        Cast<T, float, castTraitB322B16>(dstReg, src, preg);
        DataCopy<T, NON_FLOAT_STORE_DIST>(dst + offset, dstReg, preg);
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanVL(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                          uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop = UpdateMask<float>(reduceNum);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t i = 0; i < rows; ++i) {
            LoadRegForDtype<T>(xPtr, xReg, pregLoop, static_cast<uint32_t>(i) * rowStride);
            Mul(xReg, xReg, xReg, pregLoop);
            ReduceSum(sumReg, xReg, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, sumReg, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                             uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - V_LENGTH;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregTail = UpdateMask<float>(tailLen);
        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            LoadRegForDtype<T>(xPtr, xReg, pregFull, baseOffset);
            LoadRegForDtype<T>(xPtr + V_LENGTH, xFoldReg, pregTail, baseOffset);
            Mul(xReg, xReg, xReg, pregFull);
            Mul(xFoldReg, xFoldReg, xFoldReg, pregTail);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregTail);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
        }
    }
}

template <typename T, int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateSquareReduceSumCommon(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                      __local_mem__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                      uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride)
{
    uint16_t foldLoops = static_cast<uint16_t>((foldPoint + V_LENGTH - 1) / V_LENGTH);
    uint32_t lastNum = foldPoint / V_LENGTH;
    uint32_t tail = (reduceNum > foldPoint) ? reduceNum - foldPoint : 0;
    uint16_t tailCeilLoops = static_cast<uint16_t>((tail + V_LENGTH - 1) / V_LENGTH);
    uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>(foldLoops - tailCeilLoops);

    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
            uint32_t sregTail = tail;
            for (uint16_t j = 0; j < tailCeilLoops; ++j) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = static_cast<uint32_t>(j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                LoadRegForDtype<T>(xPtr + foldPoint, xFoldReg, pregFull, offset);
                Mul(xFoldReg, xFoldReg, xFoldReg, pregLoop);
                Add(sumReg, xReg, xFoldReg, pregFull);
                ReduceSum(reduceReg, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + j, reduceReg, pregOne);
            }
            for (uint16_t j = 0; j < firstFlodWithOutAddLoops; ++j) {
                uint32_t offset = static_cast<uint32_t>(tailCeilLoops + j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                ReduceSum(reduceReg, xReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + tailCeilLoops + j, reduceReg,
                                                                   pregOne);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                DataCopy(xReg, tmpPtr + static_cast<uint32_t>(i) * tmpStride);
                ReduceSum(reduceReg, xReg, pregLast);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        } else if constexpr (LAST_LOOP_NUMS == DICHOTOMY_ADD_COEFF) {
            lastNum -= V_LENGTH;
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
                DataCopy(xReg, tmpPtr + tmpOffset);
                DataCopy(xFoldReg, tmpPtr + tmpOffset + V_LENGTH);
                ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                           pregLast);
                Add(sumReg, xReg, xFoldReg, pregFull);
                ReduceSum(reduceReg, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        }
    }
}

template <typename T>
// Squares input values inside this function, then reduces each row.
__aicore__ inline void CalculateSquareReduceSum(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                __local_mem__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride,
                                                uint32_t branchNum = 0)
{
    uint32_t reduceBranchNum = branchNum == 0 ? reduceNum : branchNum;
    if (reduceBranchNum <= V_LENGTH) {
        CalculateSquareReduceSumLessThanVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH + V_LENGTH) {
        CalculateSquareReduceSumLessThanTwoVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH * V_LENGTH * DICHOTOMY_ADD_COEFF) {
        CalculateSquareReduceSumCommon<T, 1>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride);
    } else {
        CalculateSquareReduceSumCommon<T, DICHOTOMY_ADD_COEFF>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum,
                                                               foldPoint, tmpStride);
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(LocalTensor<T>& xLocal, LocalTensor<float>& dstLocal,
                                                LocalTensor<float>& tmpLocal, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t blockAlign,
                                                uint32_t branchNum = 0)
{
    __local_mem__ T* xPtr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ float* dstPtr = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)tmpLocal.GetPhyAddr();
    uint32_t foldLoops = (foldPoint + V_LENGTH - 1) / V_LENGTH;
    uint32_t tmpStride = (foldLoops + blockAlign - 1) / blockAlign * blockAlign;
    CalculateSquareReduceSum<T>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride, branchNum);
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(LocalTensor<T>& xLocal, LocalTensor<float>& dstLocal,
                                                TBuf<TPosition::VECCALC>& tmpBuf, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t blockAlign,
                                                uint32_t branchNum = 0)
{
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    CalculateSquareReduceSum<T>(xLocal, dstLocal, tmpLocal, rows, rowStride, reduceNum, foldPoint, blockAlign,
                                branchNum);
}

__aicore__ inline void CalculateReduceSumLessThanVL(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                                    uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop = UpdateMask<float>(reduceNum);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr);
        ReduceSum(sumReg, xReg, pregLoop);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, sumReg, pregOne);
    }
}

__aicore__ inline void CalculateReduceSumLessThanTwoVL(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                                       uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - V_LENGTH;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailLen);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr);
        DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, xPtr + V_LENGTH);
        ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0), pregTail);
        Add(sumReg, xReg, xFoldReg, pregFull);
        ReduceSum(reduceReg, sumReg, pregFull);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, reduceReg, pregOne);
    }
}

template <int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateReduceSumCommon(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                                __local_mem__ float* tmpPtr, uint32_t reduceNum, uint32_t foldPoint)
{
    uint16_t foldLoops = static_cast<uint16_t>((foldPoint + V_LENGTH - 1) / V_LENGTH);
    uint32_t lastNum = foldPoint / V_LENGTH;
    uint32_t tail = reduceNum - foldPoint;
    uint16_t tailCeilLoops = static_cast<uint16_t>((tail + V_LENGTH - 1) / V_LENGTH);
    uint16_t tailFullLoops = static_cast<uint16_t>(tail / V_LENGTH);

    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t r = 0; r < tailFullLoops; ++r) {
            uint32_t offset = static_cast<uint32_t>(r) * V_LENGTH;
            DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, xPtr + foldPoint + offset);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + r, reduceReg, pregOne);
        }
        uint32_t tailRemain = tail - static_cast<uint32_t>(tailFullLoops) * V_LENGTH;
        if (tailRemain != 0) {
            pregLoop = UpdateMask<float>(tailRemain);
            uint32_t offset = static_cast<uint32_t>(tailFullLoops) * V_LENGTH;
            DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr + offset);
            DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, xPtr + foldPoint + offset);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregLoop);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tailFullLoops, reduceReg, pregOne);
        }
        // Fix the original local implementations' fixed-offset bug in the remaining reduce blocks.
        for (uint16_t r = 0; r < static_cast<uint16_t>(foldLoops - tailCeilLoops); ++r) {
            uint32_t offset = static_cast<uint32_t>(tailCeilLoops + r);
            DataCopy<float, LoadDist::DIST_NORM>(xReg, xPtr + offset * V_LENGTH);
            ReduceSum(reduceReg, xReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + offset, reduceReg, pregOne);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastNum);
            DataCopy<float, LoadDist::DIST_NORM>(xReg, tmpPtr);
            ReduceSum(reduceReg, xReg, pregLast);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, reduceReg, pregOne);
        } else if constexpr (LAST_LOOP_NUMS == DICHOTOMY_ADD_COEFF) {
            lastNum -= V_LENGTH;
            MaskReg pregLast = UpdateMask<float>(lastNum);
            DataCopy<float, LoadDist::DIST_NORM>(xReg, tmpPtr);
            DataCopy<float, LoadDist::DIST_NORM>(xFoldReg, tmpPtr + V_LENGTH);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregLast);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, reduceReg, pregOne);
        }
    }
}

// Reduces an fp32 buffer whose values have already been squared by the caller.
__aicore__ inline void CalculateReduceSum(__local_mem__ float* xPtr, __local_mem__ float* dstPtr,
                                          __local_mem__ float* tmpPtr, uint32_t reduceNum, uint32_t foldPoint)
{
    if (reduceNum <= V_LENGTH) {
        CalculateReduceSumLessThanVL(xPtr, dstPtr, reduceNum);
    } else if (reduceNum <= V_LENGTH + V_LENGTH) {
        CalculateReduceSumLessThanTwoVL(xPtr, dstPtr, reduceNum);
    } else if (reduceNum <= V_LENGTH * V_LENGTH * DICHOTOMY_ADD_COEFF) {
        CalculateReduceSumCommon<1>(xPtr, dstPtr, tmpPtr, reduceNum, foldPoint);
    } else {
        CalculateReduceSumCommon<DICHOTOMY_ADD_COEFF>(xPtr, dstPtr, tmpPtr, reduceNum, foldPoint);
    }
}

__aicore__ inline void CalculateReduceSum(LocalTensor<float>& xLocal, LocalTensor<float>& dstLocal,
                                          LocalTensor<float>& tmpLocal, uint32_t reduceNum, uint32_t foldPoint)
{
    __local_mem__ float* xPtr = (__local_mem__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* dstPtr = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)tmpLocal.GetPhyAddr();
    CalculateReduceSum(xPtr, dstPtr, tmpPtr, reduceNum, foldPoint);
}

__aicore__ inline void CalculateReduceSum(LocalTensor<float>& xLocal, LocalTensor<float>& dstLocal,
                                          TBuf<TPosition::VECCALC>& tmpBuf, uint32_t reduceNum, uint32_t foldPoint)
{
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    CalculateReduceSum(xLocal, dstLocal, tmpLocal, reduceNum, foldPoint);
}
} // namespace NormCommonRegbase
} // namespace NormCommon

#endif // ADD_RMS_NORM_BIAS_A5_REDUCE_COMMON_REGBASE_H_RMS_NORM
