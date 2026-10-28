/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// FP4-g32 (MXFP4) 量化核心（TilingKey 2001/2101）：非主场景、可能日落，本文件自成闭环可整体删除。
// 现役走 VFProcessMxFp4RowBatchPerf；文件尾部为仅退役壳 kernel.h 引用的基线死代码。

#ifndef KV_COMPRESS_EPILOG_V2_QUANT_FP4_G32_H
#define KV_COMPRESS_EPILOG_V2_QUANT_FP4_G32_H

#include "kernel_operator.h"
#include "kv_compress_epilog_v2_common.h"

namespace KvCompressEpilogV2Ops {

constexpr int64_t KCEV2_FP4_OUT_ELEMS_PER_BLOCK = 64;
constexpr int64_t KCEV2_FP4_TWO = 2;

// 行批流水：阶段 1 统计 + scale 位链（maxExp 驻留寄存器，保留 Inf/NaN/0 位级修正），
// 阶段 2 数据段逐半字节精确存储；每批仅 1 次 LocalMemBar，scale 内联写融合输出行。
template <typename T>
__simd_vf__ inline void VFProcessMxFp4RowBatchPerfVF(
    __ubuf__ uint8_t *outputBase, __ubuf__ bfloat16_t *inputAddr, __ubuf__ uint16_t *scratchBase,
    uint16_t rowCount, uint32_t d, uint32_t dataCol, uint32_t concatCol,
    uint32_t inputStride, uint32_t outputStride, uint32_t scratchStride, uint32_t xLoops)
{
    using namespace AscendC::Reg;
    {
        constexpr uint32_t vregBytes = KCEV2_VL_FP32 * sizeof(float);
        constexpr uint32_t ubBlockBytes = 32U;
        const uint32_t vlForB16 = vregBytes / sizeof(bfloat16_t);
        const uint32_t blocksPerVreg = vregBytes / ubBlockBytes;
        const uint32_t groupCount = d / KCEV2_GROUP_ELEMS;
        const uint32_t padCount = (outputStride - concatCol) > 0U ? outputStride - concatCol : 0U;

        static constexpr AscendC::Reg::CastTrait fp4CastTrait = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        RegTensor<T> even;
        RegTensor<T> odd;
        RegTensor<T> evenMul;
        RegTensor<T> oddMul;
        RegTensor<uint16_t> evenExp;
        RegTensor<uint16_t> oddExp;
        RegTensor<uint16_t> maxExp;
        RegTensor<uint16_t> expMaskReg;
        RegTensor<uint16_t> fp4MaxExp;
        RegTensor<uint16_t> sharedExp;
        RegTensor<uint16_t> scaleValue;
        RegTensor<uint16_t> invBias;
        RegTensor<uint16_t> halfScale;
        RegTensor<uint16_t> zero;
        RegTensor<uint16_t> nan;
        RegTensor<uint16_t> specialInv;
        RegTensor<fp4x2_e2m1_t> evenFp4;
        RegTensor<fp4x2_e2m1_t> oddFp4;
        RegTensor<uint16_t> halfScaleBcast;
        RegTensor<uint8_t> zeroBytes;
        MaskReg finiteMask;
        MaskReg nonZeroMask;
        MaskReg clampMask;
        MaskReg specialMask;
        MaskReg dataMask;
        UnalignRegForStore scaleOutStoreReg;
        UnalignRegForStore halfSlotStoreReg;
        UnalignRegForStore padStoreReg;
        Duplicate(expMaskReg, KCEV2_BF16_EXP_MASK);
        Duplicate(fp4MaxExp, KCEV2_FP4_E2M1_MAX_EXP);
        Duplicate(invBias, KCEV2_BF16_INV_BIAS);
        Duplicate(zero, static_cast<uint16_t>(0));
        Duplicate(nan, KCEV2_BF16_NAN);
        Duplicate(specialInv, KCEV2_FP4_SPECIAL_INV);
        Duplicate(zeroBytes, static_cast<uint8_t>(0));

        // 阶段 1：逐行统计 + scale 链，maxExp 保持在寄存器中。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ T *src = inputAddr + row * inputStride;
            __ubuf__ uint8_t *outRow = outputBase + row * outputStride;
            __ubuf__ uint16_t *scaleOut = reinterpret_cast<__ubuf__ uint16_t *>(outRow + dataCol);
            __ubuf__ uint16_t *halfSlot = scratchBase + row * scratchStride;
            for (uint32_t loop = 0; loop < xLoops; ++loop) {
                const uint32_t valid = d - loop * KCEV2_PERF_STAT_ELEMS > KCEV2_PERF_STAT_ELEMS ?
                                           KCEV2_PERF_STAT_ELEMS : d - loop * KCEV2_PERF_STAT_ELEMS;
                uint32_t evenCount = valid / 2U;
                uint32_t oddCount = valid / 2U;
                MaskReg evenMask = UpdateMask<T>(evenCount);
                MaskReg oddMask = UpdateMask<T>(oddCount);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                    even, odd, src, vlForB16 * KCEV2_FP4_TWO);
                And(evenExp, reinterpret_cast<RegTensor<uint16_t> &>(even), expMaskReg, evenMask);
                And(oddExp, reinterpret_cast<RegTensor<uint16_t> &>(odd), expMaskReg, oddMask);
                Max(maxExp, evenExp, oddExp, evenMask);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX, uint16_t>(maxExp, maxExp, evenMask);
                const uint32_t groupsInBeat =
                    groupCount - loop * KCEV2_PERF_MAX_EXP_PER_BEAT > KCEV2_PERF_MAX_EXP_PER_BEAT ?
                    KCEV2_PERF_MAX_EXP_PER_BEAT : groupCount - loop * KCEV2_PERF_MAX_EXP_PER_BEAT;
                uint32_t chainCount = groupsInBeat;
                MaskReg chainMask = UpdateMask<uint16_t>(chainCount);
                Compare<uint16_t, CMPMODE::NE>(finiteMask, maxExp, expMaskReg, chainMask);
                Compare<uint16_t, CMPMODE::NE>(nonZeroMask, maxExp, zero, chainMask);
                Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, fp4MaxExp, chainMask);
                Select<uint16_t>(maxExp, fp4MaxExp, maxExp, clampMask);
                Sub(sharedExp, maxExp, fp4MaxExp, chainMask);
                Select<uint16_t>(scaleValue, sharedExp, nan, finiteMask);
                Select<uint16_t>(scaleValue, scaleValue, zero, nonZeroMask);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    scaleOut, scaleValue, scaleOutStoreReg, groupsInBeat);
                StoreUnAlignPost(scaleOut, scaleOutStoreReg, 0);

                Sub(halfScale, invBias, sharedExp, chainMask);
                Select<uint16_t>(halfScale, halfScale, nan, finiteMask);
                Select<uint16_t>(halfScale, halfScale, zero, nonZeroMask);
                Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, invBias, chainMask);
                Select<uint16_t>(halfScale, specialInv, halfScale, specialMask);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    halfSlot, halfScale, halfSlotStoreReg, groupsInBeat);
                StoreUnAlignPost(halfSlot, halfSlotStoreReg, 0);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        // 阶段 2：逐行数据（基线 ComputeData 循环 + 融合输出行布局）。
        // 逐半字节精确存储：尾拍只写 ceil(valid/2) 个 nibble 对，不越界到 scale/pad 区。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ T *src = inputAddr + row * inputStride;
            __ubuf__ uint8_t *outRow = outputBase + row * outputStride;
            __ubuf__ uint16_t *halfScaleAddr = scratchBase + row * scratchStride;
            for (uint32_t loop = 0; loop < xLoops; ++loop) {
                const uint32_t valid = d - loop * KCEV2_PERF_STAT_ELEMS > KCEV2_PERF_STAT_ELEMS ?
                                           KCEV2_PERF_STAT_ELEMS : d - loop * KCEV2_PERF_STAT_ELEMS;
                uint32_t halfCount = valid / 2U;
                dataMask = UpdateMask<T>(halfCount);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                    even, odd, src, vlForB16 * KCEV2_FP4_TWO);
                LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                    halfScaleBcast, halfScaleAddr, blocksPerVreg);
                Mul(evenMul, even, reinterpret_cast<RegTensor<T> &>(halfScaleBcast), dataMask);
                Mul(oddMul, odd, reinterpret_cast<RegTensor<T> &>(halfScaleBcast), dataMask);
                Interleave(evenMul, oddMul, evenMul, oddMul);
                __ubuf__ int8_t *outPtr =
                    reinterpret_cast<__ubuf__ int8_t *>(outRow + loop * KCEV2_PERF_STAT_ELEMS / 2U);
                const uint32_t evenElems = valid > vlForB16 ? vlForB16 : valid;
                const uint32_t oddElems = valid > vlForB16 ? valid - vlForB16 : 0U;
                uint32_t evenCastCount = evenElems;
                MaskReg evenCastMask = UpdateMask<T>(evenCastCount);
                AscendC::Reg::Cast<fp4x2_e2m1_t, T, fp4CastTrait>(evenFp4, evenMul, evenCastMask);
                StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                    outPtr, reinterpret_cast<RegTensor<int8_t> &>(evenFp4), evenCastMask);
                if (oddElems > 0U) {
                    uint32_t oddCastCount = oddElems;
                    MaskReg oddCastMask = UpdateMask<T>(oddCastCount);
                    AscendC::Reg::Cast<fp4x2_e2m1_t, T, fp4CastTrait>(oddFp4, oddMul, oddCastMask);
                    StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                        outPtr + vlForB16 / 2U, reinterpret_cast<RegTensor<int8_t> &>(oddFp4),
                        oddCastMask);
                }
            }
            if (padCount > 0U) {
                __ubuf__ uint8_t *padPtr = outRow + concatCol;
                StoreUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(padPtr, zeroBytes, padStoreReg, padCount);
                StoreUnAlignPost(padPtr, padStoreReg, 0);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessMxFp4RowBatchPerf(
    const LocalTensor<uint8_t> &output, const LocalTensor<bfloat16_t> &input,
    const LocalTensor<uint16_t> &scratch, uint16_t rowCount, uint32_t d,
    uint32_t dataCol, uint32_t concatCol, uint32_t kvCacheCol)
{
    constexpr uint32_t vregBytes = KCEV2_VL_FP32 * sizeof(float);
    const uint32_t vlForB16 = vregBytes / sizeof(bfloat16_t);
    const uint32_t groupCount = d / KCEV2_GROUP_ELEMS;
    const uint32_t inputStride = RoundUp<bfloat16_t>(d);
    const uint32_t outputStride = RoundUp<uint8_t>(kvCacheCol);
    const uint32_t scratchStride = RoundUp<uint16_t>(groupCount);
    const uint32_t xLoops =
        static_cast<uint32_t>((d + vlForB16 * KCEV2_FP4_TWO - 1) / (vlForB16 * KCEV2_FP4_TWO));
    auto *inputAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outputBase = reinterpret_cast<__ubuf__ uint8_t *>(output.GetPhyAddr());
    auto *scratchBase = reinterpret_cast<__ubuf__ uint16_t *>(scratch.GetPhyAddr());
    VFProcessMxFp4RowBatchPerfVF<T>(outputBase, inputAddr, scratchBase, rowCount, d, dataCol, concatCol,
                                  inputStride, outputStride, scratchStride, xLoops);
}

// ==== 基线参考实现（死代码）：仅退役壳 kernel.h 引用，现役通路走上方 RowBatch 版 ====
// 删除本段需同步删除 kernel.h；其内部逐算子语义已由 RowBatch 版本保持。 =====================

// 全仓零引用，仅随基线参考保留。
constexpr AscendC::MicroAPI::CastTrait KCEV2_B16_TO_FP4 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__simd_vf__ inline void VFComputeMaxExpMxFp4(
    __ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCount,
    uint16_t loopNum, uint32_t vlForB16, uint32_t blocksPerVreg)
{
    using namespace AscendC::Reg;
    {
        RegTensor<T> even;
        RegTensor<T> odd;
        RegTensor<uint16_t> evenExp;
        RegTensor<uint16_t> oddExp;
        RegTensor<uint16_t> expMask;
        RegTensor<uint16_t> maxExp;
        MaskReg evenMask;
        MaskReg oddMask;
        UnalignRegForStore storeReg;
        Duplicate(expMask, KCEV2_BF16_EXP_MASK);
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            evenMask = UpdateMask<T>(totalCount);
            oddMask = UpdateMask<T>(totalCount);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                even, odd, srcAddr, vlForB16 * KCEV2_FP4_TWO);
            And(evenExp, reinterpret_cast<RegTensor<uint16_t> &>(even), expMask, evenMask);
            And(oddExp, reinterpret_cast<RegTensor<uint16_t> &>(odd), expMask, evenMask);
            Max(maxExp, evenExp, oddExp, evenMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(maxExp, maxExp, evenMask);
            StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, maxExp, storeReg, blocksPerVreg);
        }
        StoreUnAlignPost(maxExpAddr, storeReg, 0);
    }
}

__simd_vf__ inline void VFComputeScaleMxFp4(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *scaleAddr,
    __ubuf__ uint16_t *halfScaleAddr, uint32_t scaleCount,
    uint16_t loopNum, uint32_t vlForB16)
{
    using namespace AscendC::Reg;
    {
        RegTensor<uint16_t> expMask;
        RegTensor<uint16_t> maxExp;
        RegTensor<uint16_t> fp4MaxExp;
        RegTensor<uint16_t> sharedExp;
        RegTensor<uint16_t> scaleValue;
        RegTensor<uint16_t> invBias;
        RegTensor<uint16_t> halfScale;
        RegTensor<uint16_t> zero;
        RegTensor<uint16_t> nan;
        RegTensor<uint16_t> specialInv;
        MaskReg finiteMask;
        MaskReg nonZeroMask;
        MaskReg clampMask;
        MaskReg specialMask;
        MaskReg scaleMask;
        Duplicate(expMask, KCEV2_BF16_EXP_MASK);
        Duplicate(fp4MaxExp, KCEV2_FP4_E2M1_MAX_EXP);
        Duplicate(invBias, KCEV2_BF16_INV_BIAS);
        Duplicate(zero, static_cast<uint16_t>(0));
        Duplicate(nan, KCEV2_BF16_NAN);
        Duplicate(specialInv, KCEV2_FP4_SPECIAL_INV);
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            scaleMask = UpdateMask<uint16_t>(scaleCount);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpAddr, vlForB16);
            Compare<uint16_t, CMPMODE::NE>(finiteMask, maxExp, expMask, scaleMask);
            Compare<uint16_t, CMPMODE::NE>(nonZeroMask, maxExp, zero, scaleMask);
            Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, fp4MaxExp, scaleMask);
            Select<uint16_t>(maxExp, fp4MaxExp, maxExp, clampMask);
            Sub(sharedExp, maxExp, fp4MaxExp, scaleMask);
            Select<uint16_t>(scaleValue, sharedExp, nan, finiteMask);
            Select<uint16_t>(scaleValue, scaleValue, zero, nonZeroMask);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(scaleAddr, scaleValue, vlForB16, scaleMask);

            Sub(halfScale, invBias, sharedExp, scaleMask);
            Select<uint16_t>(halfScale, halfScale, nan, finiteMask);
            Select<uint16_t>(halfScale, halfScale, zero, nonZeroMask);
            Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, invBias, scaleMask);
            Select<uint16_t>(halfScale, specialInv, halfScale, specialMask);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(halfScaleAddr, halfScale, vlForB16, scaleMask);
        }
    }
}

template <typename T>
__simd_vf__ inline void VFComputeDataMxFp4(
    __ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleAddr,
    __ubuf__ int8_t *outputAddr, uint32_t totalCount,
    uint16_t loopNum, uint32_t vlForB16, uint32_t blocksPerVreg)
{
    using namespace AscendC::Reg;
    {
        RegTensor<uint16_t> halfScale;
        RegTensor<T> even;
        RegTensor<T> odd;
        RegTensor<fp4x2_e2m1_t> evenFp4;
        RegTensor<fp4x2_e2m1_t> oddFp4;
        MaskReg dataMask;
        static constexpr AscendC::Reg::CastTrait fp4CastTrait = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            dataMask = UpdateMask<T>(totalCount);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                even, odd, srcAddr, vlForB16 * KCEV2_FP4_TWO);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                halfScale, halfScaleAddr, blocksPerVreg);
            Mul(even, even, reinterpret_cast<RegTensor<T> &>(halfScale), dataMask);
            Mul(odd, odd, reinterpret_cast<RegTensor<T> &>(halfScale), dataMask);
            Interleave(even, odd, even, odd);
            AscendC::Reg::Cast<fp4x2_e2m1_t, T, fp4CastTrait>(evenFp4, even, dataMask);
            AscendC::Reg::Cast<fp4x2_e2m1_t, T, fp4CastTrait>(oddFp4, odd, dataMask);
            StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outputAddr, reinterpret_cast<RegTensor<int8_t> &>(evenFp4),
                KCEV2_FP4_OUT_ELEMS_PER_BLOCK, dataMask);
            StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outputAddr, reinterpret_cast<RegTensor<int8_t> &>(oddFp4),
                KCEV2_FP4_OUT_ELEMS_PER_BLOCK, dataMask);
        }
    }
}

__aicore__ inline uint32_t ScaleRowBytesMxFp4(uint32_t scaleCol)
{
    return ((scaleCol * sizeof(bfloat16_t) + 63U) / 64U) * 64U;
}

__aicore__ static void VFProcessMxFp4Verified(
    const LocalTensor<int8_t> &output, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<bfloat16_t> &input, const LocalTensor<uint16_t> &maxExp,
    const LocalTensor<uint16_t> &halfScale, uint16_t rowCount, uint32_t d)
{
    constexpr uint32_t vregBytes = KCEV2_VL_FP32 * sizeof(float);
    constexpr uint32_t ubBlockBytes = 32U;
    const uint32_t vlForB16 = vregBytes / sizeof(bfloat16_t);
    const uint32_t blocksPerVreg = vregBytes / ubBlockBytes;
    const uint32_t scaleCol = d / KCEV2_GROUP_ELEMS;
    const uint32_t xStride = RoundUp<bfloat16_t>(d);
    const uint32_t outputStride = RoundUp<int8_t>(d / 2);
    const uint32_t scaleStrideBytes = ScaleRowBytesMxFp4(scaleCol);
    const uint16_t xLoops = static_cast<uint16_t>(
        (d + vlForB16 * KCEV2_FP4_TWO - 1) / (vlForB16 * KCEV2_FP4_TWO));
    const uint16_t scaleLoops = static_cast<uint16_t>((scaleCol + vlForB16 - 1) / vlForB16);
    auto *inputAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outputAddr = reinterpret_cast<__ubuf__ int8_t *>(output.GetPhyAddr());
    auto *scaleByteAddr = reinterpret_cast<__ubuf__ uint8_t *>(scale.GetPhyAddr());
    auto *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExp.GetPhyAddr());
    auto *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScale.GetPhyAddr());
    for (uint16_t row = 0; row < rowCount; ++row) {
        VFComputeMaxExpMxFp4(inputAddr + row * xStride, maxExpAddr, d, xLoops, vlForB16, blocksPerVreg);
        VFComputeScaleMxFp4(maxExpAddr,
                            reinterpret_cast<__ubuf__ uint16_t *>(scaleByteAddr + row * scaleStrideBytes),
                            halfScaleAddr, scaleCol, scaleLoops, vlForB16);
        VFComputeDataMxFp4(inputAddr + row * xStride, halfScaleAddr,
                           outputAddr + row * outputStride, d, xLoops, vlForB16, blocksPerVreg);
    }
}

// ==== 基线参考实现结束 =======================================================

}  // namespace KvCompressEpilogV2Ops

#endif
