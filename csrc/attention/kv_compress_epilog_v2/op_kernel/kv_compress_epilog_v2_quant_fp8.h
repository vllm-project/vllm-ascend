/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// FP8-g32 (MXFP8) 量化核心（TilingKey 2000/2100），纯 UB 进出，无 tiling/pipe 依赖。
// rs=0 走 VFProcessMxFp8<TCache,false> 数值基线；rs=1 走 VFProcessMxFp8FullRowPerf 全行三段流水。

#ifndef KV_COMPRESS_EPILOG_V2_QUANT_FP8_H
#define KV_COMPRESS_EPILOG_V2_QUANT_FP8_H

#include "kernel_operator.h"
#include "kv_compress_epilog_v2_common.h"

namespace KvCompressEpilogV2Ops {

constexpr float KCEV2_FP8_E5M2_MAX = 57344.0f;
constexpr float KCEV2_FP8_E4M3_MAX = 448.0f;
constexpr uint32_t KCEV2_FAST_LOG_SHIFT = 23U;
constexpr uint32_t KCEV2_EXP_MASK = 0xFFU;
constexpr uint32_t KCEV2_MANTISSA_MASK = (1U << 23U) - 1U;

constexpr AscendC::MicroAPI::CastTrait KCEV2_B16_TO_F32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait KCEV2_F32_TO_B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait KCEV2_F32_TO_FP8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait KCEV2_PERF_B16_TO_F32_LOW = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait KCEV2_PERF_F32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait KCEV2_PERF_F32_TO_FP8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadB16AsFloat(RegTensor<float> &dst, __local_mem__ T *src, MaskReg mask)
{
    RegTensor<T> tmp;
    DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src);
    Cast<float, T, KCEV2_B16_TO_F32>(dst, tmp, mask);
}

template <typename T>
__aicore__ inline void StoreFloatAsFp8(__local_mem__ T *dst, RegTensor<float> &src, MaskReg mask)
{
    RegTensor<T> tmp;
    Cast<T, float, KCEV2_F32_TO_FP8>(tmp, src, mask);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst, tmp, mask);
}

// <TCache,true> 实例化仅剩退役壳 kernel.h（回滚参考）引用，现役 rs=1 走 FullRow 版本。
template <typename TCache, bool roundScale>
__aicore__ inline void VFProcessMxFp8(
    const LocalTensor<uint8_t> &output, const LocalTensor<bfloat16_t> &input, uint16_t rowCount,
    uint32_t d, uint32_t dataCol, uint32_t concatCol, uint32_t cacheCol)
{
    __local_mem__ TCache *outputData = reinterpret_cast<__local_mem__ TCache *>(output.GetPhyAddr());
    __local_mem__ bfloat16_t *outputScale = reinterpret_cast<__local_mem__ bfloat16_t *>(output.GetPhyAddr());
    __local_mem__ bfloat16_t *inputData = reinterpret_cast<__local_mem__ bfloat16_t *>(input.GetPhyAddr());
    const uint32_t inputStride = RoundUp<bfloat16_t>(d);
    const uint16_t groupCount = static_cast<uint16_t>(d / KCEV2_GROUP_ELEMS);
    const float fp8Max = IsSameType<TCache, fp8_e5m2_t>::value ? KCEV2_FP8_E5M2_MAX : KCEV2_FP8_E4M3_MAX;
    const float fp8Min = -fp8Max;
    const float coeff = 1.0f / fp8Max;

    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xAbs;
        RegTensor<float> amax;
        RegTensor<float> scale;
        RegTensor<float> scaleDup;
        RegTensor<bfloat16_t> scaleBf16;
        RegTensor<uint32_t> exp;
        RegTensor<uint32_t> mantissa;
        RegTensor<uint32_t> expMask;
        RegTensor<uint32_t> mantissaMask;
        RegTensor<uint32_t> roundUp;
        RegTensor<uint32_t> zero;
        RegTensor<uint32_t> one;
        RegTensor<int32_t> unbiasedExp;
        RegTensor<uint8_t> zeroBytes;
        MaskReg groupMask;
        MaskReg hasMantissa;
        MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
        MaskReg byteMask = CreateMask<uint8_t, MaskPattern::ALL>();
        UnalignReg paddingReg;
        Duplicate(zero, static_cast<uint32_t>(0), scalarMask);
        Duplicate(one, static_cast<uint32_t>(1), scalarMask);
        Duplicate(expMask, KCEV2_EXP_MASK, scalarMask);
        Duplicate(mantissaMask, KCEV2_MANTISSA_MASK, scalarMask);
        Duplicate(zeroBytes, static_cast<uint8_t>(0), byteMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            for (uint16_t group = 0; group < groupCount; ++group) {
                uint32_t valid = KCEV2_GROUP_ELEMS;
                groupMask = UpdateMask<float>(valid);
                LoadB16AsFloat(x, inputData + row * inputStride + group * KCEV2_GROUP_ELEMS, groupMask);
                Abs(xAbs, x, groupMask);
                ReduceMax(amax, xAbs, groupMask);
                Maxs(scale, amax, 1.0e-4f, scalarMask);
                Muls(scale, scale, coeff, scalarMask);
                if constexpr (roundScale) {
                    ShiftRights(exp, reinterpret_cast<RegTensor<uint32_t> &>(scale),
                                static_cast<int16_t>(KCEV2_FAST_LOG_SHIFT), scalarMask);
                    And(exp, exp, expMask, scalarMask);
                    And(mantissa, reinterpret_cast<RegTensor<uint32_t> &>(scale), mantissaMask, scalarMask);
                    Compare<uint32_t, CMPMODE::NE>(hasMantissa, mantissa, zero, scalarMask);
                    Select(roundUp, one, zero, hasMantissa);
                    Adds(unbiasedExp, reinterpret_cast<RegTensor<int32_t> &>(exp), -127, scalarMask);
                    Add(unbiasedExp, unbiasedExp, reinterpret_cast<RegTensor<int32_t> &>(roundUp), scalarMask);
                    Adds(unbiasedExp, unbiasedExp, 127, scalarMask);
                    ShiftLefts(reinterpret_cast<RegTensor<int32_t> &>(scale), unbiasedExp,
                               static_cast<int16_t>(KCEV2_FAST_LOG_SHIFT), scalarMask);
                }
                Duplicate(scaleDup, scale, groupMask);
                Div(x, x, scaleDup, groupMask);
                Maxs(x, x, fp8Min, groupMask);
                Mins(x, x, fp8Max, groupMask);
                StoreFloatAsFp8(outputData + row * cacheCol + group * KCEV2_GROUP_ELEMS, x, groupMask);
                Cast<bfloat16_t, float, KCEV2_F32_TO_B16>(scaleBf16, scale, scalarMask);
                DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
                    outputScale + (row * cacheCol + dataCol) / sizeof(bfloat16_t) + group, scaleBf16, scalarMask);
            }
            const uint32_t padCount = cacheCol - concatCol;
            if (padCount > 0) {
                __local_mem__ uint8_t *pad = reinterpret_cast<__local_mem__ uint8_t *>(output.GetPhyAddr()) +
                    row * cacheCol + concatCol;
                DataCopyUnAlign(pad, zeroBytes, paddingReg, padCount);
                DataCopyUnAlignPost(pad, paddingReg, 0);
            }
        }
    }
}

// rs=1 全行三段寄存器流水：数值链与基线 VFProcessMxFp8<TCache,true> 逐步等价。
// amax 用位级 abs（And 0x7FFF），有限输入下与浮点 Abs/ReduceMax 逐位一致。
template <typename TCache>
__simd_vf__ inline void VFProcessMxFp8FullRowPerfVF(
    __ubuf__ uint8_t *outputBase, __ubuf__ bfloat16_t *inputAddr, __ubuf__ float *scratchBase,
    uint16_t rowCount, uint32_t d, uint32_t dataCol, uint32_t concatCol,
    uint32_t inputStride, uint32_t outputStride, uint32_t scratchStride,
    uint32_t statBeats, uint32_t scaleChunks, uint32_t dataChunks)
{
    using namespace AscendC::Reg;
    {
        const uint32_t groupCount = d / KCEV2_GROUP_ELEMS;
        const float fp8Max = IsSameType<TCache, fp8_e5m2_t>::value ? KCEV2_FP8_E5M2_MAX : KCEV2_FP8_E4M3_MAX;
        const float fp8Min = -fp8Max;
        const float coeff = 1.0f / fp8Max;
        const uint32_t padCount = (outputStride - concatCol) > 0U ? outputStride - concatCol : 0U;

        RegTensor<bfloat16_t> even;
        RegTensor<bfloat16_t> odd;
        RegTensor<bfloat16_t> xUnp;
        RegTensor<uint16_t> evenAbs;
        RegTensor<uint16_t> oddAbs;
        RegTensor<uint16_t> absMax;
        RegTensor<uint16_t> amaxBits;
        RegTensor<uint16_t> amaxU16;
        RegTensor<uint16_t> absMaskReg;
        RegTensor<uint16_t> scaleB16;
        RegTensor<float> amaxF;
        RegTensor<float> scaleF;
        RegTensor<float> xF;
        RegTensor<float> bcastF;
        RegTensor<int32_t> arReg;
        RegTensor<uint32_t> idxReg;
        RegTensor<uint32_t> gsizeReg;
        RegTensor<TCache> fp8Reg;
        RegTensor<uint32_t> expMaskReg;
        RegTensor<uint32_t> mantissaMaskReg;
        RegTensor<uint32_t> zeroU32;
        RegTensor<uint32_t> oneU32;
        RegTensor<uint32_t> expField;
        RegTensor<uint32_t> mantissaField;
        RegTensor<uint32_t> roundUp;
        RegTensor<int32_t> unbiasedExp;
        RegTensor<uint8_t> zeroBytes;
        MaskReg hasMantissa;
        MaskReg allU16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg allF32 = CreateMask<float, MaskPattern::ALL>();
        MaskReg byteMask = CreateMask<uint8_t, MaskPattern::ALL>();
        UnalignRegForStore amaxStoreReg;
        UnalignRegForStore f32SlotStoreReg;
        UnalignRegForStore padStoreReg;
        Duplicate(absMaskReg, static_cast<uint16_t>(0x7FFFU), allU16);
        Duplicate(expMaskReg, KCEV2_EXP_MASK, allF32);
        Duplicate(mantissaMaskReg, KCEV2_MANTISSA_MASK, allF32);
        Duplicate(zeroU32, static_cast<uint32_t>(0), allF32);
        Duplicate(oneU32, static_cast<uint32_t>(1), allF32);
        Duplicate(zeroBytes, static_cast<uint8_t>(0), byteMask);
        // 保留原设置（消费方已替换为等价的 ShiftRights by 5，见阶段 3 注释）。
        Duplicate(gsizeReg, static_cast<uint32_t>(KCEV2_GROUP_ELEMS));

        // 阶段 1：逐行分组 amax 位（B16 幅值位）写入 scratch。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ bfloat16_t *src = inputAddr + row * inputStride;
            __ubuf__ uint16_t *amaxSlot =
                reinterpret_cast<__ubuf__ uint16_t *>(scratchBase + row * scratchStride);
            for (uint32_t beat = 0; beat < statBeats; ++beat) {
                const uint32_t valid = d - beat * KCEV2_PERF_STAT_ELEMS > KCEV2_PERF_STAT_ELEMS ?
                                           KCEV2_PERF_STAT_ELEMS : d - beat * KCEV2_PERF_STAT_ELEMS;
                uint32_t evenCount = valid / 2U;
                uint32_t oddCount = valid / 2U;
                MaskReg evenMask = UpdateMask<bfloat16_t>(evenCount);
                MaskReg oddMask = UpdateMask<bfloat16_t>(oddCount);
                LoadAlign<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                    even, odd, src, KCEV2_PERF_STAT_ELEMS);
                And(evenAbs, reinterpret_cast<RegTensor<uint16_t> &>(even), absMaskReg, evenMask);
                And(oddAbs, reinterpret_cast<RegTensor<uint16_t> &>(odd), absMaskReg, oddMask);
                Max(absMax, evenAbs, oddAbs, evenMask);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX, uint16_t>(amaxBits, absMax, evenMask);
                StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                    amaxSlot, amaxBits, amaxStoreReg, KCEV2_PERF_MAX_EXP_PER_BEAT);
            }
            StoreUnAlignPost(amaxSlot, amaxStoreReg, 0);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        // 阶段 2：整行 scale 链单寄存器完成；chunk 逆序处理，F32 scale 写不覆盖待读 amax。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ float *rowSlotF = scratchBase + row * scratchStride;
            __ubuf__ uint16_t *amaxBase = reinterpret_cast<__ubuf__ uint16_t *>(rowSlotF);
            __ubuf__ uint8_t *outRow = outputBase + row * outputStride;
            __ubuf__ bfloat16_t *scaleOutBf16 = reinterpret_cast<__ubuf__ bfloat16_t *>(outRow + dataCol);
            for (int32_t chunk = static_cast<int32_t>(scaleChunks) - 1; chunk >= 0; --chunk) {
                const uint32_t chunkBase = static_cast<uint32_t>(chunk) * KCEV2_VL_FP32;
                const uint32_t validGroups = groupCount - chunkBase > KCEV2_VL_FP32 ?
                                                 KCEV2_VL_FP32 : groupCount - chunkBase;
                uint32_t chainCount = validGroups;
                MaskReg chainMask = UpdateMask<float>(chainCount);
                LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(amaxU16, amaxBase + chunkBase);
                Cast<float, bfloat16_t, KCEV2_PERF_B16_TO_F32_LOW>(
                    amaxF, reinterpret_cast<RegTensor<bfloat16_t> &>(amaxU16), chainMask);
                Maxs(scaleF, amaxF, 1.0e-4f, chainMask);
                Muls(scaleF, scaleF, coeff, chainMask);
                ShiftRights(expField, reinterpret_cast<RegTensor<uint32_t> &>(scaleF),
                            static_cast<int16_t>(KCEV2_FAST_LOG_SHIFT), chainMask);
                And(expField, expField, expMaskReg, chainMask);
                And(mantissaField, reinterpret_cast<RegTensor<uint32_t> &>(scaleF), mantissaMaskReg, chainMask);
                Compare<uint32_t, CMPMODE::NE>(hasMantissa, mantissaField, zeroU32, chainMask);
                Select(roundUp, oneU32, zeroU32, hasMantissa);
                Adds(unbiasedExp, reinterpret_cast<RegTensor<int32_t> &>(expField), -127, chainMask);
                Add(unbiasedExp, unbiasedExp, reinterpret_cast<RegTensor<int32_t> &>(roundUp), chainMask);
                Adds(unbiasedExp, unbiasedExp, 127, chainMask);
                ShiftLefts(reinterpret_cast<RegTensor<int32_t> &>(scaleF), unbiasedExp,
                           static_cast<int16_t>(KCEV2_FAST_LOG_SHIFT), chainMask);
                __ubuf__ float *f32SlotPtr = rowSlotF + chunkBase;
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(
                    f32SlotPtr, scaleF, f32SlotStoreReg, validGroups);
                StoreUnAlignPost(f32SlotPtr, f32SlotStoreReg, 0);
                // B16 scale 内联写输出行：Cast 后经 DIST_PACK_B32 紧凑存储（RMSNorm 生产形态）。
                Cast<bfloat16_t, float, KCEV2_PERF_F32_TO_B16>(
                    reinterpret_cast<RegTensor<bfloat16_t> &>(scaleB16), scaleF, chainMask);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
                    scaleOutBf16 + chunkBase, reinterpret_cast<RegTensor<bfloat16_t> &>(scaleB16), chainMask);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        // 阶段 3：逐行数据 64 元素分块，Gather 广播分组 scale 后 Div/clamp/Cast 为 FP8。
        for (uint16_t row = 0; row < rowCount; ++row) {
            __ubuf__ bfloat16_t *src = inputAddr + row * inputStride;
            __ubuf__ uint8_t *outRow = outputBase + row * outputStride;
            __ubuf__ float *scaleSlotF = scratchBase + row * scratchStride;
            __ubuf__ TCache *outData = reinterpret_cast<__ubuf__ TCache *>(outRow);
            for (uint32_t chunk = 0; chunk < dataChunks; ++chunk) {
                const uint32_t elemBase = chunk * KCEV2_VL_FP32;
                const uint32_t valid = d - elemBase > KCEV2_VL_FP32 ? KCEV2_VL_FP32 : d - elemBase;
                uint32_t pregCount = valid;
                MaskReg preg = UpdateMask<uint32_t>(pregCount);
                LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xUnp, src + elemBase);
                Cast<float, bfloat16_t, KCEV2_PERF_B16_TO_F32_LOW>(
                    xF, reinterpret_cast<RegTensor<bfloat16_t> &>(xUnp), preg);
                Arange(arReg, static_cast<int32_t>(elemBase));
                // 分组索引 = lanePos >> 5：u32 非负，与除以 32 的结果逐位一致。
                ShiftRights(reinterpret_cast<RegTensor<uint32_t> &>(idxReg),
                            reinterpret_cast<RegTensor<uint32_t> &>(arReg),
                            static_cast<int16_t>(5), preg);
                Gather(bcastF, scaleSlotF, reinterpret_cast<RegTensor<uint32_t> &>(idxReg), preg);
                Div(xF, xF, bcastF, preg);
                Maxs(xF, xF, fp8Min, preg);
                Mins(xF, xF, fp8Max, preg);
                Cast<TCache, float, KCEV2_PERF_F32_TO_FP8>(fp8Reg, xF, preg);
                StoreAlign<TCache, StoreDist::DIST_PACK4_B32>(outData + elemBase, fp8Reg, preg);
            }
            if (padCount > 0U) {
                __ubuf__ uint8_t *padPtr = outRow + concatCol;
                StoreUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(padPtr, zeroBytes, padStoreReg, padCount);
                StoreUnAlignPost(padPtr, padStoreReg, 0);
            }
        }
    }
}

template <typename TCache>
__aicore__ inline void VFProcessMxFp8FullRowPerf(
    const LocalTensor<uint8_t> &output, const LocalTensor<bfloat16_t> &input,
    const LocalTensor<float> &scratch, uint16_t rowCount, uint32_t d,
    uint32_t dataCol, uint32_t concatCol, uint32_t kvCacheCol)
{
    const uint32_t groupCount = d / KCEV2_GROUP_ELEMS;
    const uint32_t inputStride = RoundUp<bfloat16_t>(d);
    const uint32_t outputStride = RoundUp<uint8_t>(kvCacheCol);
    // 逐行 scratch：u16 amax 位与 F32 组 scale 别名复用（4*G 字节，32B 对齐）。
    const uint32_t scratchStride = RoundUp<float>(groupCount);
    const uint32_t statBeats = static_cast<uint32_t>(CeilDiv(static_cast<int32_t>(d), KCEV2_PERF_STAT_ELEMS));
    const uint32_t scaleChunks =
        static_cast<uint32_t>(CeilDiv(static_cast<int32_t>(groupCount), static_cast<int32_t>(KCEV2_VL_FP32)));
    const uint32_t dataChunks =
        static_cast<uint32_t>(CeilDiv(static_cast<int32_t>(d), static_cast<int32_t>(KCEV2_VL_FP32)));
    auto *inputAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outputBase = reinterpret_cast<__ubuf__ uint8_t *>(output.GetPhyAddr());
    auto *scratchBase = reinterpret_cast<__ubuf__ float *>(scratch.GetPhyAddr());
    VFProcessMxFp8FullRowPerfVF<TCache>(outputBase, inputAddr, scratchBase, rowCount, d, dataCol, concatCol,
                                        inputStride, outputStride, scratchStride, statBeats, scaleChunks,
                                        dataChunks);
}

}  // namespace KvCompressEpilogV2Ops

#endif
