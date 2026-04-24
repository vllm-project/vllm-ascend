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
* \file vf_top_k_16.h
* \brief
*/

#ifndef VF_TOP_K_16_H
#define VF_TOP_K_16_H

#define DIV(x, y) (((x + y - 1) / y))

namespace topkb16 {
template<typename T>
__simd_vf__ void HistogramsHighVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint16_t* inputBuf, uint16_t vfLoop, bool init)
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

    MicroAPI::RegTensor<uint16_t> vregHigh;
    MicroAPI::RegTensor<uint16_t> vregLow;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh, inputBuf + i * 256);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, (MicroAPI::RegTensor<uint8_t>&)vregHigh, pregB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, (MicroAPI::RegTensor<uint8_t>&)vregHigh, pregB8);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindHighTargetBinVFImpl(__ubuf__ uint32_t* idxHighBuf, __ubuf__ uint32_t* nkValueBuf, __ubuf__ uint32_t* histogramsBuf, uint32_t bottomK)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdxHigh;

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Duplicate(btmK, bottomK);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdxHigh;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxHigh, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idxHighBuf, sqzIdxHigh, alignIdxHigh);
    }
    MicroAPI::StoreUnAlignPost(idxHighBuf, alignIdxHigh);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idxHigh;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idxHigh, idxHighBuf);

    MicroAPI::RegTensor<uint8_t> idxAll1;
    MicroAPI::RegTensor<uint32_t> idxPrev0;
    MicroAPI::RegTensor<uint32_t> prevBinValue;
    MicroAPI::Duplicate(idxAll1, 1);

    MicroAPI::RegTensor<uint32_t> zeroAll;
    MicroAPI::Duplicate(zeroAll, 0);

    MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(preg0, idxHigh, zeroAll, pregB32);
    MicroAPI::Sub(idxPrev0, idxHigh, (MicroAPI::RegTensor<uint32_t>&)idxAll1, pregB32);
    MicroAPI::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    MicroAPI::Gather(prevBinValue, histogramsBuf, idxPrev0, pregB32);
    MicroAPI::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    MicroAPI::RegTensor<uint32_t> nextK;
    MicroAPI::Sub(nextK, btmK, prevBinValue, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}


template<typename T>
__simd_vf__ void HistogramsLowVFImpl(__ubuf__ uint32_t* histogramsBuf, __ubuf__ uint16_t* inputBuf, __ubuf__ uint32_t* idxHighBuf, uint16_t vfLoop, bool init)
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

    MicroAPI::RegTensor<uint32_t> idxHigh;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idxHigh, idxHighBuf);

    MicroAPI::RegTensor<uint16_t> vregHigh;
    MicroAPI::RegTensor<uint16_t> vregLow;

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {MicroAPI::RegLayout::ZERO,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr MicroAPI::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {MicroAPI::RegLayout::ONE,
                MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh, inputBuf + i * 256);

        MicroAPI::MaskReg pregEQ = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ, (MicroAPI::RegTensor<uint8_t>&)vregHigh, (MicroAPI::RegTensor<uint8_t>&)idxHigh, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0, (MicroAPI::RegTensor<uint8_t>&)vregLow, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1, (MicroAPI::RegTensor<uint8_t>&)vregLow, pregEQ);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t* kValue, __ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* idxHighBuf, __ubuf__ uint32_t* idxLowBuf)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdxLow;

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK, kValue);

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::RegTensor<uint32_t> cout;
        MicroAPI::RegTensor<uint32_t> sqzIdxLow;

        MicroAPI::MaskReg pregGE = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange(idxC, i * 64);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxLow, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(idxLowBuf, sqzIdxLow, alignIdxLow);
    }
    MicroAPI::StoreUnAlignPost(idxLowBuf, alignIdxLow);

    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();

    MicroAPI::RegTensor<uint32_t> idxHigh;
    MicroAPI::RegTensor<uint32_t> idxLow;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B8>(idxHigh, idxHighBuf);
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(idxLow, idxLowBuf);

    MicroAPI::RegTensor<uint16_t> idxTmp;
    MicroAPI::Duplicate(idxTmp, 0xff00);

    MicroAPI::And(idxHigh, idxHigh, (MicroAPI::RegTensor<uint32_t>&)idxTmp, pregB32);

    MicroAPI::RegTensor<uint32_t> idxK;
    MicroAPI::Add(idxK, idxHigh, idxLow, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM_B16>(kValue, idxK, pregB32);
}

__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint16_t* inputBuf, uint32_t beginIdx, __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(kthValue, kValue);
    MicroAPI::ShiftRights(kthValue, kthValue, (int16_t)16, pregB32);

    MicroAPI::RegTensor<uint16_t> vregInput;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::RegTensor<int32_t> idxC;
        MicroAPI::Arange(idxC, beginIdx + i * 64);
        
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInput, inputBuf + i * 64);

        MicroAPI::MaskReg poutGT = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::RegTensor<uint32_t> sqzIdxOut;
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(poutGT, (MicroAPI::RegTensor<uint32_t>&)vregInput, kthValue, pregB32);

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint32_t>&)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint16_t* inputBuf, uint32_t beginIdx, __ubuf__ uint32_t* kValue)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(kthValue, kValue);
    MicroAPI::ShiftRights(kthValue, kthValue, (int16_t)16, pregB32);

    MicroAPI::RegTensor<uint16_t> vregInput;

    MicroAPI::RegTensor<int32_t> idxC;
    MicroAPI::Arange(idxC, beginIdx);

    MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInput, inputBuf);

    MicroAPI::MaskReg poutEQ = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint32_t> sqzIdxOut;
    MicroAPI::Compare<uint32_t, CMPMODE::EQ>(poutEQ, (MicroAPI::RegTensor<uint32_t>&)vregInput, kthValue, pregB32);

    MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint32_t>&)idxC, poutEQ);
    MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

__aicore__ inline void LiTopKVF(const LocalTensor<uint32_t>& outputIdxLocal,
                                const LocalTensor<uint32_t>& outputValueLocal,
                                const LocalTensor<uint16_t>& inputLocal,
                                const LocalTensor<uint32_t>& tmpIdxLocal,
                                const LocalTensor<uint32_t>& tmpValueLocal,
                                const LocalTensor<uint32_t>& histogramsLocal, 
                                const LocalTensor<uint32_t>& idxHighLocal, 
                                const LocalTensor<uint32_t>& idxLowLocal, 
                                const LocalTensor<uint32_t>& nkValueLocal,
                                uint32_t topK,
                                uint32_t s2SeqLen)
{
    __ubuf__ uint32_t* outputIdxBuf = (__ubuf__ uint32_t*)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* outputValueBuf = (__ubuf__ uint32_t*)outputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t* inputBuf = (__ubuf__ uint16_t*)inputLocal.GetPhyAddr();
    __ubuf__ uint32_t* tmpIdxBuf = (__ubuf__ uint32_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* tmpValueBuf = (__ubuf__ uint32_t*)tmpValueLocal.GetPhyAddr();
    __ubuf__ uint32_t* histogramsBuf = (__ubuf__ uint32_t*)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t* idxHighBuf = (__ubuf__ uint32_t*)idxHighLocal.GetPhyAddr();
    __ubuf__ uint32_t* idxLowBuf = (__ubuf__ uint32_t*)idxLowLocal.GetPhyAddr();
    __ubuf__ uint32_t* nkValueBuf = (__ubuf__ uint32_t*)nkValueLocal.GetPhyAddr();

    uint32_t bottomK = s2SeqLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (s2SeqLen + repeatSize8 - 1) / repeatSize8; 
    uint16_t inputLoopNum = (s2SeqLen + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum = DIV(topK, 64);

    // find kth-value
    HistogramsHighVFImpl<uint16_t>(histogramsBuf, inputBuf, histogramsLoopNum, flag);
    FindHighTargetBinVFImpl(idxHighBuf, nkValueBuf, histogramsBuf, bottomK);

    HistogramsLowVFImpl<uint16_t>(histogramsBuf, inputBuf, idxHighBuf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idxHighBuf, idxLowBuf);

    // filter
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