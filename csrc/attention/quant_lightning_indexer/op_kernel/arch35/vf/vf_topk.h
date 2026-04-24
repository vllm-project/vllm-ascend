/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file vf_top_k.h
* \brief
*/

#ifndef VF_TOP_K_H
#define VF_TOP_K_H

namespace topkb32 {
template<typename T>
__simd_vf__ void HistogramsFirstVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图cout0 0-127 cout1 128-255 
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    // 32bit 高16bit
    MicroAPI::RegTensor<uint32_t> vreg0U16;
    // 32bit 低16bit
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg0, pregB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg0, pregB8);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindFirstTargetBinVFImpl(__ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* nkValueBuf, __ubuf__ uint32_t* histogramsBuf, uint32_t bottomK)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx0;

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Duplicate(btmK, bottomK);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx0;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx0, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx0Buf, sqzIdx0, alignIdx0);
    }
    MicroAPI::StoreUnAlignPost(idx0Buf, alignIdx0);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev0;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg0, idx0, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev0, idx0, (MicroAPI::RegTensor<uint32_t>&)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev0, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template<typename T>
__simd_vf__ void HistogramsSecondVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* idx0Buf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图0-127 128-255 
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    // 0x000000fc -> 0xfcfcfcfc
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ, vreg0, (MicroAPI::RegTensor<uint8_t>&)idx0, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg1, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg1, pregEQ);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

// kValue新的bottomK
__simd_vf__ void FindSecondTargetBinVFImpl(__ubuf__ uint32_t* idx1Buf, __ubuf__ uint32_t* nkValueBuf,  __ubuf__ uint32_t* kValue, __ubuf__ uint32_t* histogramsBuf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx1;

    MicroAPI::RegTensor<uint32_t> btmK1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK1, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx1;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK1, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx1, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx1Buf, sqzIdx1, alignIdx1);
    } 
    MicroAPI::StoreUnAlignPost(idx1Buf, alignIdx1);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev1;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg1, idx1, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev1, idx1, (MicroAPI::RegTensor<uint32_t>&)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev1, idxPrev1, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev1, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg1);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK1, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template<typename T>
__simd_vf__ void HistogramsThirdVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* idx1Buf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图0-127 128-255 
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    // 0x000000fc -> 0xfcfcfcfc
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);
        MicroAPI::DeInterleave(vreg3, vreg2, (MicroAPI::RegTensor<uint8_t>&)vreg1U16, (MicroAPI::RegTensor<uint8_t>&)vreg3U16);

        MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t>&)idx0, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t>&)idx1, pregB8);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::And(pregEQ, pregEQ0, pregEQ1, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg2, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg2, pregEQ);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);


    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindThirdTargetBinVFImpl(__ubuf__ uint32_t* idx2Buf, __ubuf__ uint32_t* nkValueBuf, __ubuf__ uint32_t* kValue, __ubuf__ uint32_t* histogramsBuf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx2;

    MicroAPI::RegTensor<uint32_t> btmK2;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK2, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx2;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK2, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx2, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx2Buf, sqzIdx2, alignIdx2);
    }
    MicroAPI::StoreUnAlignPost(idx2Buf, alignIdx2);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx2;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev2;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg2 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg2, idx2, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev2, idx2, (MicroAPI::RegTensor<uint32_t>&)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev2, idxPrev2, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev2, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg2);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK2, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template<typename T>
__simd_vf__ void HistogramsLastVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* idx1Buf, __ubuf__ uint32_t* idx2Buf, uint16_t vfLoop, bool init)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB8 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    // 计算直方图0-127 128-255 
    MicroAPI::RegTensor<uint16_t> cout0;
    MicroAPI::RegTensor<uint16_t> cout1;
    MicroAPI::Duplicate(cout0, 0);
    MicroAPI::Duplicate(cout1, 0);

    MicroAPI::RegTensor<uint32_t> cout0U32Even;
    MicroAPI::RegTensor<uint32_t> cout0U32Odd;
    MicroAPI::RegTensor<uint32_t> cout1U32Even;
    MicroAPI::RegTensor<uint32_t> cout1U32Odd;

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::RegTensor<uint32_t> idx2;
    // 0x000000fc -> 0xfcfcfcfc
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx0, idx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx1, idx1Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idx2, idx2Buf);

    MicroAPI::RegTensor<uint32_t> vreg0U16;
    MicroAPI::RegTensor<uint32_t> vreg1U16;
    MicroAPI::RegTensor<uint32_t> vreg2U16;
    MicroAPI::RegTensor<uint32_t> vreg3U16;

    MicroAPI::RegTensor<uint8_t> vreg0;
    MicroAPI::RegTensor<uint8_t> vreg1;
    MicroAPI::RegTensor<uint8_t> vreg2;
    MicroAPI::RegTensor<uint8_t> vreg3;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg1U16, vreg0U16, inputBuf + i * 256);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B16>(vreg3U16, vreg2U16, inputBuf + (i * 256) + 128);

        MicroAPI::DeInterleave(vreg1, vreg0, (MicroAPI::RegTensor<uint8_t>&)vreg0U16, (MicroAPI::RegTensor<uint8_t>&)vreg2U16);
        MicroAPI::DeInterleave(vreg3, vreg2, (MicroAPI::RegTensor<uint8_t>&)vreg1U16, (MicroAPI::RegTensor<uint8_t>&)vreg3U16);

        MicroAPI::MaskReg pregEQ0 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQ2 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ0, vreg0, (MicroAPI::RegTensor<uint8_t>&)idx0, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ1, vreg1, (MicroAPI::RegTensor<uint8_t>&)idx1, pregB8);
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ2, vreg2, (MicroAPI::RegTensor<uint8_t>&)idx2, pregB8);

        MicroAPI::MaskReg pregEQ0And1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregEQAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::And(pregEQ0And1, pregEQ0, pregEQ1, pregB8);
        MicroAPI::And(pregEQAll, pregEQ0And1, pregEQ2, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, vreg3, pregEQAll);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, vreg3, pregEQAll);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t* kValue, __ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* idx0Buf, __ubuf__ uint32_t* idx1Buf, __ubuf__ uint32_t* idx2Buf, __ubuf__ uint32_t* idx3Buf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx3;

    MicroAPI::RegTensor<uint32_t> btmK3;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK3, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdx3;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK3, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdx3, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idx3Buf, sqzIdx3, alignIdx3);
    }
    MicroAPI::StoreUnAlignPost(idx3Buf, alignIdx3);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idx0;
    MicroAPI::RegTensor<uint32_t> idx1;
    MicroAPI::RegTensor<uint32_t> idx2;
    MicroAPI::RegTensor<uint32_t> idx3;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx0, idx0Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx1, idx1Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx2, idx2Buf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(idx3, idx3Buf);

    MicroAPI::ShiftLefts(idx0, idx0, (int16_t)24, pregB32);
    MicroAPI::ShiftLefts(idx1, idx1, (int16_t)16, pregB32);
    MicroAPI::ShiftLefts(idx2, idx2, (int16_t)8, pregB32);

    // ADD
    MicroAPI::Add(idx0, idx0, idx1, pregB32);
    MicroAPI::Add(idx0, idx0, idx2, pregB32);
    MicroAPI::Add(idx0, idx0, idx3, pregB32);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(kValue, idx0, pregB32);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint32_t* inputBuf, uint32_t beginIdx, __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::Arange(idxC, beginIdx + i * 64);
        
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        MicroAPI::MaskReg poutGT = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<uint32_t> sqzIdxOut;
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint32_t>&)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint32_t* inputBuf, uint32_t beginIdx, __ubuf__ uint32_t* kValue)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    MicroAPI::RegTensor<int32_t> idxC;
    MicroAPI::Arange(idxC, beginIdx);

    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf);

    MicroAPI::MaskReg poutEQ = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> sqzIdxOut;
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint32_t>&)idxC, poutEQ);
    MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindValueGTOutputVFImpl(__ubuf__ uint32_t* outputValueBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignValue;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf + i * 64);

        MicroAPI::MaskReg poutGT = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<uint32_t> sqzValueOut;
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(poutGT, vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutGT);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    }
    MicroAPI::StoreUnAlignPost(outputValueBuf, alignValue);
}

__simd_vf__ void FindValueEQOutputVFImpl(__ubuf__ uint32_t* outputValueBuf, __ubuf__ uint32_t* inputBuf, __ubuf__ uint32_t* kValue)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::UnalignRegForStore alignValue;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(kthValue, kValue);

    MicroAPI::RegTensor<uint32_t> vregInput;

    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputBuf);

    MicroAPI::MaskReg poutEQ = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> sqzValueOut;
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(poutEQ, vregInput, kthValue, pregB32);

    MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzValueOut, vregInput, poutEQ);
    MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputValueBuf, sqzValueOut, alignValue);
    MicroAPI::StoreUnAlignPost(outputValueBuf, alignValue);
}

__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t>& outputIdxLocal,
                                const LocalTensor<uint32_t>& outputValueLocal,
                                const LocalTensor<uint32_t>& inputLocal,
                                const LocalTensor<uint32_t>& tmpIdxLocal,
                                const LocalTensor<uint32_t>& tmpValueLocal,
                                const LocalTensor<uint32_t>& histogramsLocal, 
                                const LocalTensor<uint32_t>& idx0Local, 
                                const LocalTensor<uint32_t>& idx1Local, 
                                const LocalTensor<uint32_t>& idx2Local, 
                                const LocalTensor<uint32_t>& idx3Local,
                                const LocalTensor<uint32_t>& nkValueLocal,
                                uint32_t topK,
                                uint32_t s2SeqLen)
{
    __ubuf__ uint32_t* outputIdxBuf = (__ubuf__ uint32_t*)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* outputValueBuf = (__ubuf__ uint32_t*)outputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t* inputBuf = (__ubuf__ uint32_t*)inputLocal.GetPhyAddr();
    __ubuf__ uint32_t* tmpIdxBuf = (__ubuf__ uint32_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* tmpValueBuf = (__ubuf__ uint32_t*)tmpValueLocal.GetPhyAddr();
    __ubuf__ uint32_t* histogramsBuf = (__ubuf__ uint32_t*)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t* idx0Buf = (__ubuf__ uint32_t*)idx0Local.GetPhyAddr();
    __ubuf__ uint32_t* idx1Buf = (__ubuf__ uint32_t*)idx1Local.GetPhyAddr();
    __ubuf__ uint32_t* idx2Buf = (__ubuf__ uint32_t*)idx2Local.GetPhyAddr();
    __ubuf__ uint32_t* idx3Buf = (__ubuf__ uint32_t*)idx3Local.GetPhyAddr();
    __ubuf__ uint32_t* nkValueBuf = (__ubuf__ uint32_t*)nkValueLocal.GetPhyAddr();

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
    for(uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arValueNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arValueNumPerLoop - arValueNum) / sizeof(uint32_t)) < remainValueNum) {
            // 调用一次查找等于k-value情况的过程
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
    for(uint16_t i = 0; i < inputLoopNum; ++i) {
        int64_t arIdxNumPerLoop = AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>();
        if (((arIdxNumPerLoop - arIdxNum) / sizeof(uint32_t)) < remainIdxNum) {
            // 调用一次查找等于k-value情况的过程
            beginIdx = i * 64;
            FindIdxEQOutputVFImpl(outputIdxBuf, inputBuf + i * 64, beginIdx, nkValueBuf);
        } else {
            break;
        }
    }
}

}
#endif