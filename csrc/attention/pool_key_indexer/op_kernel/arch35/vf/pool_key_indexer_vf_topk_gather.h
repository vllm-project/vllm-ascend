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
 * \file pool_key_indexer_vf_topk_gather.h
 * \brief
 */

#ifndef POOL_KEY_INDEXER_VF_TOP_K_GATHER_H
#define POOL_KEY_INDEXER_VF_TOP_K_GATHER_H

namespace topkb32gather {
template <typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32Reg0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg0 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    // 32bit 高16bit
    Reg::RegTensor<uint32_t> vreg0U16;
    // 32bit 低16bit
    Reg::RegTensor<uint32_t> vreg1U16;
    Reg::RegTensor<uint32_t> vreg2U16;
    Reg::RegTensor<uint32_t> vreg3U16;

    Reg::RegTensor<uint8_t> vreg0;
    Reg::RegTensor<uint8_t> vreg1;
    Reg::RegTensor<uint8_t> vreg2;
    Reg::RegTensor<uint8_t> vreg3;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg0,
                                                                                                          pregB8Reg0);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg0,
                                                                                                          pregB8Reg0);
    }

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg0);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg0);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg0);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg0);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg0);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg0);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *histogramsBuf, uint32_t bottomK)
{
    Reg::MaskReg pregB32Reg1 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx0;

    Reg::RegTensor<uint32_t> btmK;
    Reg::Duplicate(btmK, bottomK);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx0;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32Reg1);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx0, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idx0Buf, sqzIdx0, alignIdx0);
    }
    Reg::StoreUnAlignPost(idx0Buf, alignIdx0);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx0;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev0;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg0, idx0, zeroAll, pregB32Reg1);
    Reg::Sub(idxPrev0, idx0, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32Reg1);
    Reg::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32Reg1);

    Reg::Gather(prevBinValue, histogramsBuf, idxPrev0, pregB32Reg1);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK, prevBinValue, pregB32Reg1);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32Reg1);
}

template <typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                        __ubuf__ uint32_t *idx0Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32Reg2 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg1 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

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

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8Reg1);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg1,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg1,
                                                                                                          pregEQ);
    }

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg1);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg1);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg1);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg1);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg2);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg2);
}

// kValue新的bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *nkValueBuf,
                                           __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    Reg::MaskReg pregB32Reg3 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx1;

    Reg::RegTensor<uint32_t> btmK1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK1, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx1;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK1, pregB32Reg3);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx1, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idx1Buf, sqzIdx1, alignIdx1);
    }
    Reg::StoreUnAlignPost(idx1Buf, alignIdx1);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx1;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev1;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg1 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg1, idx1, zeroAll, pregB32Reg3);
    Reg::Sub(idxPrev1, idx1, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32Reg3);
    Reg::ShiftRights(idxPrev1, idxPrev1, (int16_t)24, pregB32Reg3);

    Reg::Gather(prevBinValue, histogramsBuf, idxPrev1, pregB32Reg3);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg1);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK1, prevBinValue, pregB32Reg3);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32Reg3);
}

template <typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                       __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf, uint16_t vfLoop,
                                       bool init)
{
    Reg::MaskReg pregB32Reg4 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg2 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0, cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd;

    Reg::RegTensor<uint32_t> idx0;
    Reg::RegTensor<uint32_t> idx1;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    Reg::RegTensor<uint32_t> vreg0U16, vreg1U16, vreg2U16, vreg3U16;

    Reg::RegTensor<uint8_t> vreg0, vreg1, vreg2, vreg3;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t> &)vreg1U16, (Reg::RegTensor<uint8_t> &)vreg3U16);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8Reg2);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t> &)idx1, pregB8Reg2);

        Reg::MaskReg pregEQ = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ, pregEQ0, pregEQ1, pregB8Reg2);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg2,
                                                                                                          pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg2,
                                                                                                          pregEQ);
    }

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg2);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg2);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg2);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg2);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg4);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg4);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *nkValueBuf,
                                          __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf)
{
    Reg::MaskReg pregB32Reg5 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx2;

    Reg::RegTensor<uint32_t> btmK2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK2, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::RegTensor<int32_t> idxC;
        Reg::RegTensor<uint32_t> cout;
        Reg::RegTensor<uint32_t> sqzIdx2;

        Reg::MaskReg pregGE = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

        Reg::Arange(idxC, i * 64);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK2, pregB32Reg5);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdx2, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idx2Buf, sqzIdx2, alignIdx2);
    }
    Reg::StoreUnAlignPost(idx2Buf, alignIdx2);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idx2;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev2;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg2 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg2, idx2, zeroAll, pregB32Reg5);
    Reg::Sub(idxPrev2, idx2, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32Reg5);
    Reg::ShiftRights(idxPrev2, idxPrev2, (int16_t)24, pregB32Reg5);

    Reg::Gather(prevBinValue, histogramsBuf, idxPrev2, pregB32Reg5);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg2);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK2, prevBinValue, pregB32Reg5);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32Reg5);
}

template <typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *inputBuf,
                                      __ubuf__ uint32_t *idx0Buf, __ubuf__ uint32_t *idx1Buf,
                                      __ubuf__ uint32_t *idx2Buf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32Reg6 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16Reg3 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8Reg3 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0, cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> cout0U32Even, cout0U32Odd, cout1U32Even, cout1U32Odd;

    Reg::RegTensor<uint32_t> idx0, idx1, idx2;
    // 0x000000fc -> 0xfcfcfcfc
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);

    Reg::RegTensor<uint32_t> vreg0U16, vreg1U16, vreg2U16, vreg3U16;

    Reg::RegTensor<uint8_t> vreg0, vreg1, vreg2, vreg3;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        Reg::DeInterleave(vreg1, vreg0, (Reg::RegTensor<uint8_t> &)vreg0U16, (Reg::RegTensor<uint8_t> &)vreg2U16);
        Reg::DeInterleave(vreg3, vreg2, (Reg::RegTensor<uint8_t> &)vreg1U16, (Reg::RegTensor<uint8_t> &)vreg3U16);

        Reg::MaskReg pregEQ0 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQ2 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (Reg::RegTensor<uint8_t> &)idx0, pregB8Reg3);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (Reg::RegTensor<uint8_t> &)idx1, pregB8Reg3);
        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (Reg::RegTensor<uint8_t> &)idx2, pregB8Reg3);

        Reg::MaskReg pregEQ0And1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregEQAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8Reg3);
        Reg::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8Reg3);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cout0, vreg3,
                                                                                                          pregEQAll);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cout1, vreg3,
                                                                                                          pregEQAll);
    }

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16Reg3);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16Reg3);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16Reg3);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16Reg3);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32Reg6);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd,
                                                              pregB32Reg6);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf, __ubuf__ uint32_t *idx0Buf,
                               __ubuf__ uint32_t *idx1Buf, __ubuf__ uint32_t *idx2Buf, __ubuf__ uint32_t *idx3Buf)
{
    Reg::MaskReg pregB32Reg7 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

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
        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32Reg7);
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

    Reg::ShiftLefts(idx0, idx0, (int16_t)24, pregB32Reg7);
    Reg::ShiftLefts(idx1, idx1, (int16_t)16, pregB32Reg7);
    Reg::ShiftLefts(idx2, idx2, (int16_t)8, pregB32Reg7);

    // ADD
    Reg::Add(idx0, idx0, idx1, pregB32Reg7);
    Reg::Add(idx0, idx0, idx2, pregB32Reg7);
    Reg::Add(idx0, idx0, idx3, pregB32Reg7);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(kValue, idx0, pregB32Reg7);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg8 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

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
        Reg::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32Reg8);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutGT);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *inputBuf, uint32_t beginIdx,
                                       __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg9 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg poutEQ;

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(kthValue, kValue);

    Reg::RegTensor<uint32_t> vregInput;
    Reg::RegTensor<int32_t> idxC;
    Reg::RegTensor<uint32_t> sqzIdxOut;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::Arange(idxC, beginIdx + i * 64);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        Reg::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32Reg9);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint32_t> &)idxC, poutEQ);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出最终的Value
 */
__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint32_t *outputValueBuf, __ubuf__ uint32_t *inputValueBuf,
                                       __ubuf__ uint32_t *tmpIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg10 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Gather(outputValue, inputValueBuf, tmpIdx, pregB32Reg10);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputValueBuf + i * 64, outputValue, pregB32Reg10);
    }
}

/**
    输出最终的Idx
 */
__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint32_t *tmpIdxBuf,
                                     __ubuf__ uint32_t *hisIdxBuf, uint32_t topK, uint32_t loopIndex, uint16_t vfLoop)
{
    Reg::MaskReg pregB32Reg11 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregNow;
    Reg::MaskReg pregHis;

    Reg::RegTensor<uint32_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputGatherIdx;
    Reg::RegTensor<uint32_t> outputAddsIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Compares<uint32_t, CMPMODE::GT>(pregNow, tmpIdx, topK - 1, pregB32Reg11);
        Reg::Xor(pregHis, pregNow, pregB32Reg11, pregB32Reg11);

        Reg::Gather(outputGatherIdx, hisIdxBuf, tmpIdx, pregHis);
        Reg::Adds(outputAddsIdx, tmpIdx, loopIndex, pregNow);

        Reg::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32Reg11);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputGatherIdx, pregB32Reg11);
    }
}

__simd_vf__ void IndicesAddOffsetVF(__ubuf__ uint32_t *indicesOutBuf, uint32_t outputIdxOffset, uint32_t vfLoop)
{
    Reg::MaskReg pregB32Reg12 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> outIndices;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(outIndices, indicesOutBuf + i * 64);
        Reg::Adds(outIndices, outIndices, outputIdxOffset, pregB32Reg12);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(indicesOutBuf + i * 64, outIndices, pregB32Reg12);
    }
}

/**
 * @brief LiTopKVF 对一个validLen的输入进行topk算法，输出idx_tmp
 * @param tmpIdxLocal Temp阶段输出的TopKIndex;如果s2SeqLen < 8K作为最终输出 validLen * 4B
 * @param outputValueLocal 如果s2SeqLen > 8K并且是首轮输出Value topK * 4B
 * @param inputValueLocal 输入Value validLen * 4B
 * @param histogramsLocal 直方图 256 * 4B
 * @param idx0Local 目标桶第一个八位 256 * 4B
 * @param idx1Local 目标桶第二个八位 256 * 4B
 * @param idx2Local 目标桶第三个八位 256 * 4B
 * @param idx3Local 目标桶第四个八位 256 * 4B
 * @param nkValueLocal 存储next_k的值 64 * 4B
 * @param topK topK元素
 * @param validLen 有效元素个数:QPkiCommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256)
 */
template <bool ISOUTVALUE>
__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t> &tmpIdxLocal, const LocalTensor<uint32_t> &outputValueLocal,
                                const LocalTensor<uint32_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idx0Local,
                                const LocalTensor<uint32_t> &idx1Local, const LocalTensor<uint32_t> &idx2Local,
                                const LocalTensor<uint32_t> &idx3Local, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t validLen)
{
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputValueBuf = (__ubuf__ uint32_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idx0Buf = (__ubuf__ uint32_t *)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t *idx1Buf = (__ubuf__ uint32_t *)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t *idx2Buf = (__ubuf__ uint32_t *)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t *idx3Buf = (__ubuf__ uint32_t *)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint32_t bottomK = validLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (validLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (validLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = (topK + 64 - 1) / 64;

    // find kth-value
    HistogramsFirstVFImpl<uint32_t>(histogramsBuf, inputValueBuf, histogramsLoopNum, flag);
    FindFirstTargetBinVFImpl(idx0Buf, nkValueBuf, histogramsBuf, bottomK);
    HistogramsSecondVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, histogramsLoopNum, flag);
    FindSecondTargetBinVFImpl(idx1Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsThirdVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, histogramsLoopNum, flag);
    FindThirdTargetBinVFImpl(idx2Buf, nkValueBuf, nkValueBuf, histogramsBuf);
    HistogramsLastVFImpl<uint32_t>(histogramsBuf, inputValueBuf, idx0Buf, idx1Buf, idx2Buf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idx0Buf, idx1Buf, idx2Buf, idx3Buf);

    // filter
    int32_t count = PkiCommon::Align(topK, (uint32_t)64) - topK / 64 * 64;
    AscendC::Duplicate(tmpIdxLocal[topK / 64 * 64], (uint32_t)(0), count);
    // 输出大于k-value的值idx
    FindIdxGTOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // 输出等于k-value的值idx
    FindIdxEQOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);

    if constexpr (ISOUTVALUE) {
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, topkLoopNum);
    }
}

/**
 * @brief 通过idx_tmp gather出实际的TopKIndex，s2SeqLen > 8K才会执行
 * @param outputIdxLocal 输出Idx 有效:topK * 4B
 * @param outputValueLocal 输出Value topK * 4B(以后需要输出实际value使用)
 * @param inputValueLocal 输入Value validLen * 4B
 * @param tmpIdxLocal 本轮tmpIdx输入 validLen * 4B (0 ~ validLen - 1)
 * @param hisIdxLocal 上一轮实际Idx输入 有效:topK * 4B
 * @param topK topK元素个数
 * @param loopBasicIdx 当前循环需要加上得基准Index
 * @param validLen 有效元素个数
 */
__aicore__ inline void LiTopKGatherVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                      const LocalTensor<uint32_t> &outputValueLocal,
                                      const LocalTensor<uint32_t> &inputValueLocal,
                                      const LocalTensor<uint32_t> &tmpIdxLocal,
                                      const LocalTensor<uint32_t> &hisIdxLocal, uint32_t topK, uint32_t loopBasicIdx,
                                      uint32_t validLen)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *outputValueBuf = (__ubuf__ uint32_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *inputValueBuf = (__ubuf__ uint32_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *tmpIdxBuf = (__ubuf__ uint32_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;

    FindRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topK, loopBasicIdx, topkLoopNum32);
}

__aicore__ inline void IndicesAddOffset(const LocalTensor<uint32_t> &indicesOutLocal, uint32_t outputIdxOffset,
                                        uint32_t topK)
{
    __ubuf__ uint32_t *indicesOutBuf = (__ubuf__ uint32_t *)indicesOutLocal.GetPhyAddr();
    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;
    IndicesAddOffsetVF(indicesOutBuf, outputIdxOffset, topkLoopNum32);
}
} // namespace topkb32gather
#endif
