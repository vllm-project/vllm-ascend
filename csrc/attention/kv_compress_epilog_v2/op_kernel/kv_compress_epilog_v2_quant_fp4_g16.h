/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// FP4-g16 量化核心（TilingKey 2002/2102），纯 UB 进出。
// scale 位链整行组一次向量化 + E2B_B16 半字广播全通道 Mul/Cast，每批仅 2 次 LocalMemBar。

#ifndef KV_COMPRESS_EPILOG_V2_QUANT_FP4_G16_H
#define KV_COMPRESS_EPILOG_V2_QUANT_FP4_G16_H

#include "kv_compress_epilog_v2_common.h"

namespace KvCompressEpilogV2Ops {

constexpr uint32_t KCEV2_G16_GROUP_ELEMS = 16U;
constexpr uint16_t KCEV2_G16_BF16_ABS_MASK = 0x7FFFU;
constexpr int32_t KCEV2_PERF_G16_STAT_ELEMS = 128;
constexpr int32_t KCEV2_PERF_G16_MAX_EXP_PER_BEAT = 8;
constexpr int32_t KCEV2_PERF_G16_CHAIN_GROUPS = 128;

constexpr AscendC::Reg::CastTrait KCEV2_G16_B16_TO_FP4 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__simd_vf__ inline void VFProcessMxFp4Group16RowBatchPerfVF(
    __ubuf__ uint8_t *outputBase, __ubuf__ bfloat16_t *inputAddr, __ubuf__ uint16_t *scratchBase,
    uint16_t rowCount, uint32_t d, uint32_t dataCol, uint32_t concatCol,
    uint32_t inputStride, uint32_t outputStride, uint32_t scratchStride,
    uint32_t statBeats, uint32_t chainChunks)
{
    using namespace AscendC::Reg;
    {
        const uint32_t groupCount = d / KCEV2_G16_GROUP_ELEMS;
        const uint32_t padCount = (outputStride - concatCol) > 0U ? outputStride - concatCol : 0U;

        RegTensor<bfloat16_t> xChunk;
        RegTensor<uint16_t> absBits;
        RegTensor<uint16_t> absMaskReg;
        RegTensor<uint16_t> amaxBits;
        RegTensor<uint16_t> amaxU16;
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
        RegTensor<uint16_t> halfScaleBcast;
        RegTensor<bfloat16_t> xQuant;
        RegTensor<fp4x2_e2m1_t> fp4Output;
        RegTensor<uint8_t> zeroBytes;
        MaskReg finiteMask;
        MaskReg nonZeroMask;
        MaskReg clampMask;
        MaskReg specialMask;
        MaskReg dataMask;
        MaskReg allU16 = CreateMask<uint16_t, MaskPattern::ALL>();
        UnalignRegForStore amaxStoreReg;
        UnalignRegForStore scaleOutStoreReg;
        UnalignRegForStore halfSlotStoreReg;
        UnalignRegForStore padStoreReg;
        Duplicate(absMaskReg, KCEV2_G16_BF16_ABS_MASK, allU16);
        Duplicate(expMaskReg, KCEV2_BF16_EXP_MASK);
        Duplicate(fp4MaxExp, KCEV2_FP4_E2M1_MAX_EXP);
        Duplicate(invBias, KCEV2_BF16_INV_BIAS);
        Duplicate(zero, static_cast<uint16_t>(0));
        Duplicate(nan, KCEV2_BF16_NAN);
        Duplicate(specialInv, KCEV2_FP4_SPECIAL_INV);
        Duplicate(zeroBytes, static_cast<uint8_t>(0));

        // 阶段 1：逐行分组 amax 位（B16 幅值位），每 128 元素拍 8 组。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ bfloat16_t *src = inputAddr + row * inputStride;
            __ubuf__ uint16_t *amaxSlot = scratchBase + row * scratchStride;
            for (uint32_t beat = 0; beat < statBeats; ++beat) {
                const uint32_t valid = d - beat * KCEV2_PERF_G16_STAT_ELEMS > KCEV2_PERF_G16_STAT_ELEMS ?
                                           KCEV2_PERF_G16_STAT_ELEMS : d - beat * KCEV2_PERF_G16_STAT_ELEMS;
                uint32_t dataCount = valid;
                MaskReg statMask = UpdateMask<bfloat16_t>(dataCount);
                LoadAlign(xChunk, src + beat * KCEV2_PERF_G16_STAT_ELEMS);
                And(absBits, reinterpret_cast<RegTensor<uint16_t> &>(xChunk), absMaskReg, statMask);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX, uint16_t>(amaxBits, absBits, statMask);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    amaxSlot, amaxBits, amaxStoreReg, KCEV2_PERF_G16_MAX_EXP_PER_BEAT);
            }
            StoreUnAlignPost(amaxSlot, amaxStoreReg, 0);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        // 阶段 2：整行 scale 位链，scale 内联写输出行，halfScale 写 scratch。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ uint16_t *amaxPtr = scratchBase + row * scratchStride;
            __ubuf__ uint8_t *outRow = outputBase + row * outputStride;
            __ubuf__ uint16_t *scaleOut = reinterpret_cast<__ubuf__ uint16_t *>(outRow + dataCol);
            __ubuf__ uint16_t *halfSlot = scratchBase + row * scratchStride;
            for (uint32_t chunk = 0; chunk < chainChunks; ++chunk) {
                const uint32_t validGroups =
                    groupCount - chunk * KCEV2_PERF_G16_CHAIN_GROUPS > KCEV2_PERF_G16_CHAIN_GROUPS ?
                    KCEV2_PERF_G16_CHAIN_GROUPS : groupCount - chunk * KCEV2_PERF_G16_CHAIN_GROUPS;
                uint32_t chainCount = validGroups;
                MaskReg chainMask = UpdateMask<uint16_t>(chainCount);
                LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(amaxU16, amaxPtr, KCEV2_PERF_G16_CHAIN_GROUPS);
                And(maxExp, amaxU16, expMaskReg, chainMask);
                Compare<uint16_t, CMPMODE::NE>(finiteMask, maxExp, expMaskReg, chainMask);
                Compare<uint16_t, CMPMODE::NE>(nonZeroMask, maxExp, zero, chainMask);
                Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, fp4MaxExp, chainMask);
                Select<uint16_t>(maxExp, fp4MaxExp, maxExp, clampMask);
                Sub(sharedExp, maxExp, fp4MaxExp, chainMask);
                Select<uint16_t>(scaleValue, sharedExp, nan, finiteMask);
                Select<uint16_t>(scaleValue, scaleValue, zero, nonZeroMask);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    scaleOut, scaleValue, scaleOutStoreReg, validGroups);
                StoreUnAlignPost(scaleOut, scaleOutStoreReg, 0);

                Sub(halfScale, invBias, sharedExp, chainMask);
                Select<uint16_t>(halfScale, halfScale, nan, finiteMask);
                Select<uint16_t>(halfScale, halfScale, zero, nonZeroMask);
                Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, invBias, chainMask);
                Select<uint16_t>(halfScale, specialInv, halfScale, specialMask);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    halfSlot, halfScale, halfSlotStoreReg, validGroups);
                StoreUnAlignPost(halfSlot, halfSlotStoreReg, 0);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        // 阶段 3：逐行数据，E2B_B16 halfScale 广播 + 全通道 Mul/Cast(fp4x2)/Pack。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ bfloat16_t *src = inputAddr + row * inputStride;
            __ubuf__ uint8_t *outRow = outputBase + row * outputStride;
            __ubuf__ uint16_t *halfSlot = scratchBase + row * scratchStride;
            for (uint32_t beat = 0; beat < statBeats; ++beat) {
                const uint32_t valid = d - beat * KCEV2_PERF_G16_STAT_ELEMS > KCEV2_PERF_G16_STAT_ELEMS ?
                                           KCEV2_PERF_G16_STAT_ELEMS : d - beat * KCEV2_PERF_G16_STAT_ELEMS;
                uint32_t dataCount = valid;
                dataMask = UpdateMask<bfloat16_t>(dataCount);
                LoadAlign(xChunk, src + beat * KCEV2_PERF_G16_STAT_ELEMS);
                LoadAlign<uint16_t, LoadDist::DIST_E2B_B16>(halfScaleBcast, halfSlot + beat * 8U);
                Mul(xQuant, xChunk, reinterpret_cast<RegTensor<bfloat16_t> &>(halfScaleBcast), dataMask);
                Cast<fp4x2_e2m1_t, bfloat16_t, KCEV2_G16_B16_TO_FP4>(fp4Output, xQuant, dataMask);
                StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                    reinterpret_cast<__ubuf__ int8_t *>(outRow + beat * (KCEV2_PERF_G16_STAT_ELEMS / 2U)),
                    reinterpret_cast<RegTensor<int8_t> &>(fp4Output), dataMask);
            }
            if (padCount > 0U) {
                __ubuf__ uint8_t *padPtr = outRow + concatCol;
                StoreUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(padPtr, zeroBytes, padStoreReg, padCount);
                StoreUnAlignPost(padPtr, padStoreReg, 0);
            }
        }
    }
}

__aicore__ inline void VFProcessMxFp4Group16RowBatchPerf(
    const LocalTensor<uint8_t> &output, const LocalTensor<bfloat16_t> &input,
    const LocalTensor<uint16_t> &scratch, uint16_t rowCount, uint32_t d,
    uint32_t dataCol, uint32_t concatCol, uint32_t kvCacheCol)
{
    const uint32_t groupCount = d / KCEV2_G16_GROUP_ELEMS;
    const uint32_t inputStride = RoundUp<bfloat16_t>(d);
    const uint32_t outputStride = RoundUp<uint8_t>(kvCacheCol);
    const uint32_t scratchStride = RoundUp<uint16_t>(groupCount);
    const uint32_t statBeats =
        static_cast<uint32_t>(CeilDiv(static_cast<int32_t>(d), KCEV2_PERF_G16_STAT_ELEMS));
    const uint32_t chainChunks =
        static_cast<uint32_t>(CeilDiv(static_cast<int32_t>(groupCount), KCEV2_PERF_G16_CHAIN_GROUPS));
    auto *inputAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outputBase = reinterpret_cast<__ubuf__ uint8_t *>(output.GetPhyAddr());
    auto *scratchBase = reinterpret_cast<__ubuf__ uint16_t *>(scratch.GetPhyAddr());
    VFProcessMxFp4Group16RowBatchPerfVF(outputBase, inputAddr, scratchBase, rowCount, d, dataCol,
                                        concatCol, inputStride, outputStride, scratchStride, statBeats,
                                        chainChunks);
}

}  // namespace KvCompressEpilogV2Ops

#endif
