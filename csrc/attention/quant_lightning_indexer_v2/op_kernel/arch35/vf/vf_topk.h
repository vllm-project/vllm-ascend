/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file vf_topk.h
 * \brief
 */

#ifndef VF_TOP_K_H
#define VF_TOP_K_H

namespace topkb32 {
__simd_callee__ inline void StoreHistogramResult(__ubuf__ uint32_t *histogramsBuf, Reg::RegTensor<uint16_t> &cout0,
                                                 Reg::RegTensor<uint16_t> &cout1, Reg::MaskReg &pregB16,
                                                 Reg::MaskReg &pregB32)
{
    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_callee__ inline void FindTargetBinAndUpdateNextK(__ubuf__ uint32_t *idxBuf, __ubuf__ uint32_t *nkValueBuf,
                                                        __ubuf__ uint32_t *histogramsBuf,
                                                        Reg::RegTensor<uint32_t> &btmK, Reg::MaskReg &pregB32)
{
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxBuf, sqzIdx, alignIdx);
    }
    Reg::StoreUnAlignPost(idxBuf, alignIdx);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx, idxBuf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg pregZero = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(pregZero, idx, zeroAll, pregB32);
    Reg::Sub(idxPrev, idx, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32);
    Reg::ShiftRights(idxPrev, idxPrev, (int16_t)24, pregB32);

    Reg::Gather(prevBinValue, histogramsBuf, idxPrev, pregB32);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, pregZero);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK, prevBinValue, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    // 32bit 高16bit
    Reg::RegTensor<uint32_t> vreg0U16;
    // 32bit 低16bit
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg0,
                                                                                                          pregB8);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg0,
                                                                                                          pregB8);
    }

    StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, uint32_t bottomK)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> btmK;
    Reg::Duplicate(btmK, bottomK);

    FindTargetBinAndUpdateNextK(idx0Buf, nkValueBuf, histogramsBuf, btmK, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                        __ubuf__ uint32_t *idx0Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> idx0;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);

    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg1,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg1,
                                                                                                          pregEQ);
    }

    StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

// kValue新的bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *nkValueBuf,
                                           __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> btmK1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK1, kValue);

    FindTargetBinAndUpdateNextK(idx1Buf, nkValueBuf, histogramsBuf, btmK1, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                       __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t> &)vreg1U16, (Reg::RegTensor<uint8_t> &)vreg3U16);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t> &)idx1, pregB8);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ, pregEQ0, pregEQ1, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg2,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg2,
                                                                                                          pregEQ);
    }

    StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> btmK2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK2, kValue);

    FindTargetBinAndUpdateNextK(idx2Buf, nkValueBuf, histogramsBuf, btmK2, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                      __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                      __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::RegTensor<uint32_t> idx2;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> vreg0U16;
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t> &)vreg1U16, (Reg::RegTensor<uint8_t> &)vreg3U16);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t> &)idx1, pregB8);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (Reg::RegTensor<uint8_t> &)idx2, pregB8);

        Reg::MaskReg pregEQ0And1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8);
        Reg::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg3,
                                                                                                          pregEQAll);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg3,
                                                                                                          pregEQAll);
    }

    StoreHistogramResult(histogramsBuf, cout0, cout1, pregB16, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *idx0Buf,
                               __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *idx3Buf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx3;

    Reg::RegTensor<uint32_t> btmK3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK3, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx3;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx3, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idx3Buf, sqzIdx3, alignIdx3);
    }
    Reg::StoreUnAlignPost(idx3Buf, alignIdx3);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    Reg::RegTensor<uint32_t> idx2;
    Reg::RegTensor<uint32_t> idx3;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx1, idx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx2, idx2Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(idx3, idx3Buf);

    Reg::ShiftLefts(idx0, idx0, (int16_t)24, pregB32);
    Reg::ShiftLefts(idx1, idx1, (int16_t)16, pregB32);
    Reg::ShiftLefts(idx2, idx2, (int16_t)8, pregB32);

    // ADD
    Reg::Add(idx0, idx0, idx1, pregB32);
    Reg::Add(idx0, idx0, idx2, pregB32);
    Reg::Add(idx0, idx0, idx3, pregB32);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(kValue, idx0, pregB32);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::Arange(idxC, beginIdx + i * 64);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        Reg::MaskReg poutGT = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint32_t> sqzIdxOut;
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    Reg::RegTensor<int32_t> idxC;
    Reg::Arange(idxC, beginIdx);

    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf);

    Reg::MaskReg poutEQ = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> sqzIdxOut;
    Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutEQ);
    Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindValueGTOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputBuf,
                                         __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignValue;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        Reg::MaskReg poutGT = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint32_t> sqzValueOut;
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    }
    Reg::StoreUnAlignPost(outputValueBuf, alignValue);
}

__simd_vf__ void FindValueEQOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputBuf,
                                         __ubuf__ uint32_t *kValue)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::UnalignRegForStore alignValue;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;

    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf);

    Reg::MaskReg poutEQ = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> sqzValueOut;
    Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutEQ);
    Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    Reg::StoreUnAlignPost(outputValueBuf, alignValue);
}

__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                const LocalTensor<uint32_t> &outputValueLocal, const LocalTensor<uint32_t> &inputLocal,
                                const LocalTensor<uint32_t> &tmpIdxLocal, const LocalTensor<uint32_t> &tmpValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t s2SeqLen)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputBuf = (__ubuf__ uint32_t *)inputLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpValueBuf = (__ubuf__ uint32_t *)tmpValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint32_t bottomK = s2SeqLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (s2SeqLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (s2SeqLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = (topK + 64 - 1) / 64;

    // find kth-value
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputBuf, histogramsLoopNum, flag);
    FindFirstTargetBinVFImpl(idx0Buf, nkValueBuf, histogramsBuf, bottomK);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, histogramsLoopNum, flag);
    FindSecondTargetBinVFImpl(idx1Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, histogramsLoopNum, flag);
    FindThirdTargetBinVFImpl(idx2Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);

    // filter
    // 输出大于k-value的值value
    FindValueGTOutputVFImpl(outputValueBuf, inputBuf, nkValueBuf, inputLoopNum);
    // value-当前偏移大于k-value的值在AR特殊寄存器中的有效字节数
    int64_t arValueNum = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    // value-剩余需要输出等于k-value的数量
    int64_t remainValueNum = topK - (arValueNum / sizeof(uint32_t));
    for (uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arValueNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arValueNumPerLoop - arValueNum) / sizeof(uint32_t)) < remainValueNum) {
            // 调用一次查找等于k-value情况的过程；64: 单次循环处理的元素块大小
            FindValueEQOutputVFImpl(outputValueBuf, inputBuf + i * 64, nkValueBuf);
        } else {
            break;
        }
    }

    // 输出大于k-value的值idx
    FindIdxGTOutputVFImpl(outputIdxBuf, inputBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // idx-当前偏移大于k-value的值在AR特殊寄存器中的有效字节数
    int64_t arIdxNum = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
    int64_t remainIdxNum = topK - (arIdxNum / sizeof(uint32_t));
    for (uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arIdxNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arIdxNumPerLoop - arIdxNum) / sizeof(uint32_t)) < remainIdxNum) {
            // 调用一次查找等于k-value情况的过程
            beginIdx = i * 64;                                                            // 64: 块起始偏移量
            FindIdxEQOutputVFImpl(outputIdxBuf, inputBuf + i * 64, beginIdx, nkValueBuf); // 64: 块起始偏移量
        } else {
            break;
        }
    }
}
} // namespace topkb32
#endif
