// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_SELECT_H
#define INDEXER_REGION_SCORE_SELECT_H
#include "indexer_region_score_common.h"
#include "lib/sort/sort.h"

namespace LICommon {
template <typename T>
__aicore__ inline T Align(T value, T unit)
{
    return unit == 0 ? 0 : (value + unit - 1) / unit * unit;
}
}

namespace topkb32gather {
using namespace AscendC;
// Keep the histogram's interleaved layout, convert the counts, then write out.
__simd_callee__ inline void StoreHistogram(__ubuf__ uint32_t *histogramsBuf,
    Reg::RegTensor<uint16_t> &cout0, Reg::RegTensor<uint16_t> &cout1,
    Reg::RegTensor<uint32_t> &cout0U32Even, Reg::RegTensor<uint32_t> &cout0U32Odd,
    Reg::RegTensor<uint32_t> &cout1U32Even, Reg::RegTensor<uint32_t> &cout1U32Odd,
    Reg::MaskReg &pregB16Reg, Reg::MaskReg &pregB32Reg)
{
    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg);
}

template <bool LOAD_K, bool LAST = false>
__simd_vf__ void FindTargetBinVFImpl(__ubuf__ uint32_t *idxBuf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *kValue, uint32_t bottomK,
    __ubuf__ uint32_t *idx0Buf = nullptr, __ubuf__ uint32_t *idx1Buf = nullptr, __ubuf__ uint32_t *idx2Buf = nullptr)
{
    Reg::MaskReg pregB32Reg = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> btmK;
    if constexpr (LOAD_K) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK, kValue);
    } else {
        Reg::Duplicate(btmK, bottomK);
    }
    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx;
        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32Reg);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxBuf, sqzIdx, alignIdx);
    }
    Reg::StoreUnAlignPost(idxBuf, alignIdx);
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    if constexpr (LAST) {
        Reg::RegTensor<uint32_t> idx0, idx1, idx2, idx3;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx0, idx0Buf);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx1, idx1Buf);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx2, idx2Buf);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx3, idxBuf);
        Reg::ShiftLefts(idx0, idx0, (int16_t)24, pregB32Reg);
        Reg::ShiftLefts(idx1, idx1, (int16_t)16, pregB32Reg);
        Reg::ShiftLefts(idx2, idx2, (int16_t)8, pregB32Reg);
        Reg::Add(idx0, idx0, idx1, pregB32Reg);
        Reg::Add(idx0, idx0, idx2, pregB32Reg);
        Reg::Add(idx0, idx0, idx3, pregB32Reg);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(kValue, idx0, pregB32Reg);
    } else {
        Reg::RegTensor<uint32_t> idx;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx, idxBuf);
        Reg::RegTensor<uint8_t> idxAll1;
        Reg::RegTensor<uint32_t> idxPrev;
        Reg::RegTensor<uint32_t> prevBinValue;
        Reg::Duplicate(idxAll1, 1);
        Reg::RegTensor<uint32_t> zeroAll;
        Reg::Duplicate(zeroAll, 0);
        Reg::MaskReg pregZero = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint32_t, CMPMODE::EQ>(pregZero, idx, zeroAll, pregB32Reg);
        Reg::Sub(idxPrev, idx, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32Reg);
        Reg::ShiftRights(idxPrev, idxPrev, (int16_t)24, pregB32Reg);
        Reg::Gather(prevBinValue, histogramsBuf, idxPrev, pregB32Reg);
        Reg::Select(prevBinValue, zeroAll, prevBinValue, pregZero);
        Reg::RegTensor<uint32_t> nextK;
        Reg::Sub(nextK, btmK, prevBinValue, pregB32Reg);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32Reg);
    }
}

// Compile-time byte stages share the load, prefix mask and histogram layout.
template <uint32_t BYTE>
__simd_vf__ void HistogramsVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
    __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop)
{
    auto all32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    auto all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    auto all8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> cout0, cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);
    Reg::RegTensor<uint32_t> cout0Even, cout0Odd, cout1Even, cout1Odd;
    Reg::RegTensor<uint32_t> idx0, idx1, idx2;
    if constexpr (BYTE >= 1U) Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    if constexpr (BYTE >= 2U) Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    if constexpr (BYTE >= 3U) Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);
    Reg::RegTensor<uint32_t> lo0, hi0, lo1, hi1;
    Reg::RegTensor<uint8_t> byte0, byte1, byte2, byte3;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(lo0, hi0, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(lo1, hi1, inputBuf + i * 256 + 128);
        Reg::DeInterleave(byte1, byte0, (Reg::RegTensor<uint8_t> &)hi0, (Reg::RegTensor<uint8_t> &)hi1);
        if constexpr (BYTE >= 2U) {
            Reg::DeInterleave(byte3, byte2, (Reg::RegTensor<uint8_t> &)lo0, (Reg::RegTensor<uint8_t> &)lo1);
        }
        auto keep = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        if constexpr (BYTE >= 1U) {
            Reg::Compare<uint8_t, CMPMODE::EQ>(keep, byte0, (Reg::RegTensor<uint8_t> &)idx0, all8);
        }
        if constexpr (BYTE >= 2U) {
            Reg::MaskReg match;
            Reg::Compare<uint8_t, CMPMODE::EQ>(match, byte1, (Reg::RegTensor<uint8_t> &)idx1, all8);
            Reg::And(keep, keep, match, all8);
        }
        if constexpr (BYTE >= 3U) {
            Reg::MaskReg match;
            Reg::Compare<uint8_t, CMPMODE::EQ>(match, byte2, (Reg::RegTensor<uint8_t> &)idx2, all8);
            Reg::And(keep, keep, match, all8);
        }
        if constexpr (BYTE == 0U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte0, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte0, keep);
        }
        if constexpr (BYTE == 1U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte1, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte1, keep);
        }
        if constexpr (BYTE == 2U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte2, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte2, keep);
        }
        if constexpr (BYTE == 3U) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, byte3, keep);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, byte3, keep);
        }
    }
    StoreHistogram(histogramsBuf, cout0, cout1, cout0Even, cout0Odd, cout1Even, cout1Odd, all16, all32);
}
template <bool EQUAL>
__simd_vf__ void FindIdxOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf,
    __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    // The equality pass appends to the greater-than pass's squeeze count.
    if constexpr (!EQUAL) Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIdx;
    Reg::RegTensor<uint32_t> kthValue, input, packed;
    Reg::RegTensor<int32_t> indices;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::MaskReg selected;
        Reg::Arange(indices, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(input, inputBuf + i * 64);
        Reg::Compare<uint32_t, EQUAL ? CMPMODE::EQ : CMPMODE::GT>(selected, input, kthValue, all);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(packed, (Reg::RegTensor<uint32_t> &)indices, selected);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, packed, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

// The input is 256-aligned; four rounds of byte histograms filter out the TopK IDs, and scores are gathered on demand.
__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &tmpIdxLocal,
                                const LocalTensor<uint32_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t validLen)
{
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputValueBuf = (__ubuf__ uint32_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();
    uint32_t bottomK = validLen - topK + 1;
    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;
    uint16_t histogramsLoopNum = (validLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (validLen + repeatSize32 - 1) / repeatSize32;
    HistogramsVFImpl<0U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<false>(idx0Buf, nkValueBuf, histogramsBuf, nkValueBuf, bottomK);
    HistogramsVFImpl<1U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<true>(idx1Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0);
    HistogramsVFImpl<2U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<true>(idx2Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0);
    HistogramsVFImpl<3U>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum);
    FindTargetBinVFImpl<true, true>(idx3Buf, nkValueBuf, histogramsBuf, nkValueBuf, 0U, idx0Buf, idx1Buf, idx2Buf);
    int32_t count = LICommon::Align(topK, (uint32_t)64) - topK / 64 * 64;
    AscendC::Duplicate(tmpIdxLocal[topK / 64 * 64], (uint32_t)(0), count);
    // Emit the IDs above the threshold first.
    FindIdxOutputVFImpl<false>(tmpIdxBuf, inputValueBuf, nkValueBuf, inputLoopNum);
    // Then fill up with the IDs equal to the threshold, in input order.
    FindIdxOutputVFImpl<true>(tmpIdxBuf, inputValueBuf, nkValueBuf, inputLoopNum);
}

} // namespace topkb32gather

namespace IndexerRegionScorePaged {
constexpr uint32_t SELECT_CHUNK = 16384U;
constexpr uint32_t SELECT_MAX_K = 512U;
constexpr uint32_t SELECT_INPUT = SELECT_CHUNK + SELECT_MAX_K;
constexpr uint32_t SELECT_HISTOGRAM_WORDS = 6U * 256U;
constexpr uint32_t SELECT_SORT_BYTES = 16U * 1024U;

__simd_vf__ void PrepareRadixKeys(__ubuf__ uint32_t *buffer, uint32_t validCount,
                                  uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zeroBits;
    Reg::RegTensor<uint32_t> signBit;
    Reg::RegTensor<uint32_t> allBits;
    Reg::RegTensor<int32_t> validLimit;
    Reg::RegTensor<float> zeroFloat;
    Reg::RegTensor<float> validThreshold;
    Reg::Duplicate(zeroBits, 0U);
    Reg::Duplicate(signBit, 0x80000000U);
    Reg::Duplicate(allBits, 0xffffffffU);
    Reg::Duplicate(validLimit, static_cast<int32_t>(validCount));
    Reg::Duplicate(zeroFloat, 0.0f);
    Reg::Duplicate(validThreshold, -__builtin_inff());
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> bits;
        Reg::RegTensor<int32_t> positions;
        Reg::RegTensor<uint32_t> sign;
        Reg::RegTensor<uint32_t> xorMask;
        Reg::RegTensor<uint32_t> key;
        Reg::MaskReg isZero;
        Reg::MaskReg isNegative;
        Reg::MaskReg inRange;
        Reg::MaskReg scoreValid;
        Reg::MaskReg keep;
        Reg::LoadAlign<uint32_t>(bits, buffer + offset);
        Reg::Arange(positions, static_cast<int32_t>(offset));
        Reg::Compare<float, CMPMODE::EQ>(isZero,
            reinterpret_cast<Reg::RegTensor<float> &>(bits), zeroFloat, all);
        Reg::Select(bits, zeroBits, bits, isZero);
        Reg::And(sign, bits, signBit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(isNegative, sign, zeroBits, all);
        Reg::Select(xorMask, allBits, signBit, isNegative);
        Reg::Xor(key, bits, xorMask, all);
        Reg::Compare<int32_t, CMPMODE::LT>(inRange, positions, validLimit, all);
        Reg::Compare<float, CMPMODE::GT>(scoreValid,
            reinterpret_cast<Reg::RegTensor<float> &>(bits), validThreshold, all);
        Reg::And(keep, inRange, scoreValid, all);
        Reg::Select(key, key, zeroBits, keep);
        Reg::StoreAlign<uint32_t>(buffer + offset, key, all);
    }
}

// Retained candidates precede the new chunk and stay ordered by logical ID.
// Zero-key padding cannot beat any valid ordered FP32 key.
__aicore__ inline void CopyCarryKeys(const LocalTensor<uint32_t> &dst,
    const LocalTensor<uint32_t> &src, uint32_t count, uint32_t paddedCount)
{
    auto input = reinterpret_cast<__ubuf__ uint32_t *>(src.GetPhyAddr());
    auto output = reinterpret_cast<__ubuf__ uint32_t *>(dst.GetPhyAddr());
    __VEC_SCOPE__ {
        auto all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint32_t> keys, zero;
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Duplicate(zero, 0U);
        for (uint16_t part = 0U; part < static_cast<uint16_t>(paddedCount / 64U); ++part) {
            MicroAPI::LoadAlign(keys, input + part * 64U);
            MicroAPI::Arange(positions, static_cast<int32_t>(part * 64U));
            MicroAPI::MaskReg valid;
            MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(valid, positions, static_cast<int32_t>(count), all);
            MicroAPI::Select(keys, keys, zero, valid);
            MicroAPI::StoreAlign(output + part * 64U, keys, all);
        }
    }
}

template <bool KEEP_KEYS>
__aicore__ inline void GatherSelection(const LocalTensor<uint32_t> &outIds,
    const LocalTensor<uint32_t> &outKeys, const LocalTensor<uint32_t> &positions,
    const LocalTensor<uint32_t> &keys, const LocalTensor<uint32_t> &carryIds,
    uint32_t carryCount, uint32_t firstId, uint32_t topK)
{
    auto positionPtr = reinterpret_cast<__ubuf__ uint32_t *>(positions.GetPhyAddr());
    auto keyPtr = reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr());
    auto carryPtr = reinterpret_cast<__ubuf__ uint32_t *>(carryIds.GetPhyAddr());
    auto outIdPtr = reinterpret_cast<__ubuf__ uint32_t *>(outIds.GetPhyAddr());
    auto outKeyPtr = reinterpret_cast<__ubuf__ uint32_t *>(outKeys.GetPhyAddr());
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint32_t> selected, selectedKeys, safeCarry, oldIds, newIds, ids, zero;
        MicroAPI::Duplicate(zero, 0U);
        uint32_t left = topK;
        for (uint16_t part = 0U; part < static_cast<uint16_t>((topK + 63U) / 64U); ++part) {
            auto mask = MicroAPI::UpdateMask<uint32_t>(left);
            MicroAPI::MaskReg isCarry;
            MicroAPI::LoadAlign(selected, positionPtr + part * 64U);
            if constexpr (KEEP_KEYS) {
                MicroAPI::Gather(selectedKeys, keyPtr, selected, mask);
            }
            MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(isCarry, selected, carryCount, mask);
            MicroAPI::Select(safeCarry, selected, zero, isCarry);
            MicroAPI::Gather(oldIds, carryPtr, safeCarry, mask);
            MicroAPI::Adds(newIds, selected, firstId - carryCount, mask);
            MicroAPI::Select(ids, oldIds, newIds, isCarry);
            MicroAPI::StoreAlign(outIdPtr + part * 64U, ids, mask);
            if constexpr (KEEP_KEYS) {
                MicroAPI::StoreAlign(outKeyPtr + part * 64U, selectedKeys, mask);
            }
        }
    }
}

// Host validation bounds region IDs below 2^24, so FP32 preserves them exactly.
// Descending negative IDs provide an ascending integer-ID order.
constexpr MicroAPI::CastTrait SELECT_ID_TO_FLOAT{
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

__aicore__ inline void PrepareFinalIdSort(const LocalTensor<float> &values,
    const LocalTensor<uint32_t> &ids, uint32_t count, uint32_t alignedCount)
{
    auto valuePtr = reinterpret_cast<__ubuf__ float *>(values.GetPhyAddr());
    auto idPtr = reinterpret_cast<__ubuf__ uint32_t *>(ids.GetPhyAddr());
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<uint32_t> id, invalid;
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::RegTensor<float> converted, negative, negativeInfinity;
        MicroAPI::Duplicate(invalid, 0xffffffffU);
        MicroAPI::Duplicate(negativeInfinity, -__builtin_inff());
        uint32_t left = alignedCount;
        for (uint16_t part = 0U; part < static_cast<uint16_t>((alignedCount + 63U) / 64U); ++part) {
            auto mask = MicroAPI::UpdateMask<uint32_t>(left);
            MicroAPI::MaskReg valid;
            MicroAPI::LoadAlign(id, idPtr + part * 64U);
            MicroAPI::Arange(positions, static_cast<int32_t>(part * 64U));
            MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(valid, positions, static_cast<int32_t>(count), mask);
            MicroAPI::Cast<float, int32_t, SELECT_ID_TO_FLOAT>(converted,
                reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(id), mask);
            MicroAPI::Muls(negative, converted, -1.0F, mask);
            MicroAPI::Select(negative, negative, negativeInfinity, valid);
            MicroAPI::Select(id, id, invalid, valid);
            MicroAPI::StoreAlign(valuePtr + part * 64U, negative, mask);
            MicroAPI::StoreAlign(idPtr + part * 64U, id, mask);
        }
    }
}

__aicore__ inline void FillUnderKIds(const LocalTensor<int32_t> &dst,
    uint32_t topK, uint32_t history)
{
    auto ids = reinterpret_cast<__ubuf__ int32_t *>(dst.GetPhyAddr());
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<int32_t> sequence, invalid;
        MicroAPI::Duplicate(invalid, -1);
        uint32_t left = topK;
        for (uint16_t part = 0U; part < static_cast<uint16_t>((topK + 63U) / 64U); ++part) {
            auto mask = MicroAPI::UpdateMask<int32_t>(left);
            MicroAPI::MaskReg valid;
            MicroAPI::Arange(sequence, static_cast<int32_t>(part * 64U));
            MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(valid, sequence,
                static_cast<int32_t>(history), mask);
            MicroAPI::Select(sequence, sequence, invalid, valid);
            MicroAPI::StoreAlign(ids + part * 64U, sequence, mask);
        }
    }
}

class RegionSelector {
public:
    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR actualKvLengths,
        GM_ADDR cuSeqLens, GM_ADDR ids, const PagedTilingData &data, TPipe *pipe)
    {
        data_ = data;
        topK_ = data.selectionConfig & 0xffffU;
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));
        ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ids));
        actual_ = reinterpret_cast<__gm__ int32_t *>(actualKvLengths);
        cu_ = reinterpret_cast<__gm__ int32_t *>(cuSeqLens);
        pipe->InitBuffer(inputBuffer_, SELECT_INPUT * sizeof(uint32_t));
        pipe->InitBuffer(indexBuffer_, SELECT_INPUT * sizeof(uint32_t));
        pipe->InitBuffer(carryIdBuffer_, SELECT_MAX_K * sizeof(uint32_t));
        pipe->InitBuffer(carryKeyBuffer_, SELECT_MAX_K * sizeof(uint32_t));
        pipe->InitBuffer(pickedIdBuffer_, SELECT_MAX_K * sizeof(uint32_t));
        pipe->InitBuffer(pickedKeyBuffer_, SELECT_MAX_K * sizeof(uint32_t));
        pipe->InitBuffer(histogramBuffer_, SELECT_HISTOGRAM_WORDS * sizeof(uint32_t));
        pipe->InitBuffer(sortBuffer_, SELECT_SORT_BYTES);
        keys_ = inputBuffer_.Get<uint32_t>();
        positions_ = indexBuffer_.Get<uint32_t>();
        carryIds_ = carryIdBuffer_.Get<uint32_t>();
        carryKeys_ = carryKeyBuffer_.Get<uint32_t>();
        pickedIds_ = pickedIdBuffer_.Get<uint32_t>();
        pickedKeys_ = pickedKeyBuffer_.Get<uint32_t>();
        histogram_ = histogramBuffer_.Get<uint32_t>();
        sortScratch_ = sortBuffer_.Get<uint8_t>();
        sorted_ = carryIds_.ReinterpretCast<int32_t>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t workers = data_.usedCubeCores * GetTaskRation();
        // A single worker owns rows sharing a 32-byte destination transaction.
        uint32_t rowsPerUnit = 1U;
        while ((rowsPerUnit * topK_) % 8U != 0U) {
            ++rowsPerUnit;
        }
        if (data_.batchTokens != 0U) {
            for (uint32_t group = 0U; group < data_.groups; ++group) {
                const uint64_t begin = static_cast<uint64_t>(group) * data_.totalTokens + data_.batchBegin;
                const uint64_t end = begin + data_.batchCount;
                const uint64_t firstUnit = begin / rowsPerUnit;
                // Ownership follows the global output transaction, even where
                // a batch or group boundary splits an unaligned TopK row block.
                for (uint64_t unit = firstUnit + (GetBlockIdx() + workers - firstUnit % workers) % workers;
                     unit * rowsPerUnit < end; unit += workers) {
                    const uint64_t first = unit * rowsPerUnit;
                    for (uint64_t row = first > begin ? first : begin;
                         row < end && row < first + rowsPerUnit; ++row) {
                        ProcessRow(static_cast<uint32_t>(row));
                    }
                }
            }
            return;
        }
        const uint32_t rows = data_.groups * data_.totalTokens;
        const uint32_t units = rows / rowsPerUnit + (rows % rowsPerUnit != 0U);
        for (uint64_t unit = GetBlockIdx(); unit < units; unit += workers) {
            const uint32_t begin = static_cast<uint32_t>(unit * rowsPerUnit);
            const uint32_t end = begin + PagedMin(rows - begin, rowsPerUnit);
            for (uint32_t token = begin; token < end; ++token) {
                ProcessRow(token);
            }
        }
    }

private:
    __aicore__ inline uint32_t History(uint32_t token) const
    {
        uint32_t lo = 0U;
        uint32_t hi = data_.numRequests;
        while (lo + 1U < hi) {
            const uint32_t mid = (lo + hi) / 2U;
            if (cu_[mid] <= static_cast<int32_t>(token)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const int64_t position = static_cast<int64_t>(actual_[lo]) -
            (static_cast<int64_t>(cu_[lo + 1U]) - cu_[lo]) + token - cu_[lo];
        if (position <= 0) {
            return 0U;
        }
        return PagedMin(data_.seqK, static_cast<uint32_t>(position / data_.regionSize));
    }

    __aicore__ inline void ProcessRow(uint32_t token)
    {
        const uint32_t history = History(token % data_.totalTokens);
        if (history <= topK_) {
            FillUnderKIds(sorted_, topK_, history);
        } else {
            SelectRow(token, history);
        }
        const auto ready = static_cast<event_t>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(ids_[static_cast<uint64_t>(token) * topK_], sorted_, copy);
        SetFlag<HardEvent::MTE3_V>(ready);
        WaitFlag<HardEvent::MTE3_V>(ready);
    }

    __aicore__ inline void SelectRow(uint32_t token, uint32_t history)
    {
        uint32_t carryCount = 0U;
        const uint32_t paddedK = PagedAlignUp(topK_, 64U);
        const uint64_t scoreOffset = static_cast<uint64_t>(token / data_.totalTokens) * data_.outputStrideGroup +
            static_cast<uint64_t>(token % data_.totalTokens - data_.batchBegin) * data_.outputStrideToken;
        for (uint32_t first = 0U; first < history; first += SELECT_CHUNK) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            const uint32_t count = PagedMin(SELECT_CHUNK, history - first);
            const uint32_t aligned = PagedAlignUp(carryCount + count, 256U);
            const DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
            const DataCopyPadExtParams<float> padding{false, 0U, 0U, 0.0F};
            auto incoming = keys_[carryCount].ReinterpretCast<float>();
            DataCopyPad(incoming, scores_[scoreOffset + first],
                copy, padding);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (carryCount != 0U) {
                CopyCarryKeys(keys_, carryKeys_, topK_, carryCount);
            }
            PrepareRadixKeys(reinterpret_cast<__ubuf__ uint32_t *>(incoming.GetPhyAddr()),
                count, aligned - carryCount);
            PipeBarrier<PIPE_V>();
            // The source radix primitive gathers all ties; the index buffer must
            // hold the whole tile, even though only its first K entries are used.
            topkb32gather::LiTopKVF(positions_, keys_, histogram_,
                histogram_[256U], histogram_[512U], histogram_[768U], histogram_[1024U],
                histogram_[1280U], topK_, aligned);
            PipeBarrier<PIPE_V>();
            if (first + count == history) {
                GatherSelection<false>(pickedIds_, pickedKeys_, positions_, keys_, carryIds_,
                    carryCount, first, topK_);
                PipeBarrier<PIPE_V>();
                const uint32_t alignedK = PagedAlignUp(topK_, 32U);
                auto values = pickedKeys_.ReinterpretCast<float>();
                auto packed = positions_.ReinterpretCast<float>();
                auto scratch = sortScratch_.ReinterpretCast<float>();
                PrepareFinalIdSort(values, pickedIds_, topK_, alignedK);
                PipeBarrier<PIPE_V>();
                Sort<float, true>(packed, values, pickedIds_, scratch, alignedK / 32U);
                PipeBarrier<PIPE_V>();
                Extract(values, carryIds_, packed, alignedK / 32U);
            } else {
                GatherSelection<true>(pickedIds_, pickedKeys_, positions_, keys_, carryIds_,
                    carryCount, first, topK_);
                PipeBarrier<PIPE_V>();
                Sort<uint32_t, uint32_t>(carryIds_, carryKeys_, pickedIds_, pickedKeys_, sortScratch_, topK_);
            }
            PipeBarrier<PIPE_V>();
            carryCount = paddedK;
        }
    }

    PagedTilingData data_{};
    uint32_t topK_ = 0U;
    GlobalTensor<float> scores_;
    GlobalTensor<int32_t> ids_;
    __gm__ int32_t *actual_ = nullptr;
    __gm__ int32_t *cu_ = nullptr;
    TBuf<TPosition::VECCALC> inputBuffer_, indexBuffer_, carryIdBuffer_, carryKeyBuffer_;
    TBuf<TPosition::VECCALC> pickedIdBuffer_, pickedKeyBuffer_, histogramBuffer_, sortBuffer_;
    LocalTensor<uint32_t> keys_, positions_, carryIds_, carryKeys_, pickedIds_, pickedKeys_, histogram_;
    LocalTensor<int32_t> sorted_;
    LocalTensor<uint8_t> sortScratch_;
};
} // namespace IndexerRegionScorePaged
#endif
