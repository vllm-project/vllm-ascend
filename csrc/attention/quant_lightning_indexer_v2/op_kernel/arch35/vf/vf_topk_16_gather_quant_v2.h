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
 * \file vf_topk_16_gather_quant_v2.h
 * \brief
 */

#ifndef VF_TOPK_16_GATHER_QUANT_V2_H
#define VF_TOPK_16_GATHER_QUANT_V2_H

namespace topkb16gather {

template <typename T>
__simd_vf__ void HistogramsHighVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint16_t *inputBuf, uint16_t vfLoop,
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

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    Reg::RegTensor<uint16_t> vregHigh;
    Reg::RegTensor<uint16_t> vregLow;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh, inputBuf + i * 256);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            cout0, (Reg::RegTensor<uint8_t> &)vregHigh, pregB8);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            cout1, (Reg::RegTensor<uint8_t> &)vregHigh, pregB8);
    }

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindHighTargetBinVFImpl(__ubuf__ uint32_t *idxHighBuf, __ubuf__ uint32_t *nkValueBuf,
                                         __ubuf__ uint32_t *histogramsBuf, uint32_t bottomK)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregGE;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdxHigh;

    Reg::RegTensor<uint32_t> btmK;
    Reg::Duplicate(btmK, bottomK);

    Reg::RegTensor<int32_t> idxC;
    Reg::RegTensor<uint32_t> cout;
    Reg::RegTensor<uint32_t> sqzIdxHigh;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::Arange(idxC, i * 64);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);

        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxHigh, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxHighBuf, sqzIdxHigh, alignIdxHigh);
    }
    Reg::StoreUnAlignPost(idxHighBuf, alignIdxHigh);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idxHigh;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idxHigh, idxHighBuf);

    Reg::RegTensor<uint8_t> idxAll1;
    Reg::RegTensor<uint32_t> idxPrev0;
    Reg::RegTensor<uint32_t> prevBinValue;
    Reg::Duplicate(idxAll1, 1);

    Reg::RegTensor<uint32_t> zeroAll;
    Reg::Duplicate(zeroAll, 0);

    Reg::MaskReg preg0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Compare<uint32_t, CMPMODE::EQ>(preg0, idxHigh, zeroAll, pregB32);
    Reg::Sub(idxPrev0, idxHigh, (Reg::RegTensor<uint32_t> &)idxAll1, pregB32);
    Reg::ShiftRights(idxPrev0, idxPrev0, (int16_t)24, pregB32);

    Reg::Gather(prevBinValue, histogramsBuf, idxPrev0, pregB32);
    Reg::Select(prevBinValue, zeroAll, prevBinValue, preg0);

    Reg::RegTensor<uint32_t> nextK;
    Reg::Sub(nextK, btmK, prevBinValue, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(nkValueBuf, nextK, pregB32);
}

template <typename T>
__simd_vf__ void HistogramsLowVFImpl(__ubuf__ uint32_t *histogramsBuf, __ubuf__ uint16_t *inputBuf,
                                     __ubuf__ uint32_t *idxHighBuf, uint16_t vfLoop, bool init)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregEQ;

    // 计算直方图0-127 128-255
    Reg::RegTensor<uint16_t> cout0;
    Reg::RegTensor<uint16_t> cout1;
    Reg::Duplicate(cout0, 0);
    Reg::Duplicate(cout1, 0);

    Reg::RegTensor<uint32_t> cout0U32Even;
    Reg::RegTensor<uint32_t> cout0U32Odd;
    Reg::RegTensor<uint32_t> cout1U32Even;
    Reg::RegTensor<uint32_t> cout1U32Odd;

    Reg::RegTensor<uint32_t> idxHigh;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idxHigh, idxHighBuf);

    Reg::RegTensor<uint16_t> vregHigh;
    Reg::RegTensor<uint16_t> vregLow;

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_EVEN = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                       Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait CAST_TRAIT_UINT16_TOUINT32_ODD = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_DINTLV_B8>(vregLow, vregHigh, inputBuf + i * 256);

        Reg::Compare<uint8_t, CMPMODE::EQ>(pregEQ, (Reg::RegTensor<uint8_t> &)vregHigh,
                                           (Reg::RegTensor<uint8_t> &)idxHigh, pregB8);

        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
            cout0, (Reg::RegTensor<uint8_t> &)vregLow, pregEQ);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
            cout1, (Reg::RegTensor<uint8_t> &)vregLow, pregEQ);
    }

    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    Reg::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf, cout0U32Even, cout0U32Odd, pregB32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128, cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t *kValue, __ubuf__ uint32_t *histogramsBuf,
                               __ubuf__ uint32_t *idxHighBuf, __ubuf__ uint32_t *idxLowBuf)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregGE;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdxLow;

    Reg::RegTensor<uint32_t> btmK;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(btmK, kValue);

    Reg::RegTensor<int32_t> idxC;
    Reg::RegTensor<uint32_t> cout;
    Reg::RegTensor<uint32_t> sqzIdxLow;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
        Reg::Arange(idxC, i * 64);

        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(cout, histogramsBuf + i * 64);

        Reg::Compare<uint32_t, CMPMODE::GE>(pregGE, cout, btmK, pregB32);

        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxLow, (Reg::RegTensor<uint32_t> &)idxC, pregGE);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxLowBuf, sqzIdxLow, alignIdxLow);
    }
    Reg::StoreUnAlignPost(idxLowBuf, alignIdxLow);

    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::RegTensor<uint32_t> idxHigh;
    Reg::RegTensor<uint32_t> idxLow;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B8>(idxHigh, idxHighBuf);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B16>(idxLow, idxLowBuf);

    Reg::RegTensor<uint16_t> idxTmp;
    Reg::Duplicate(idxTmp, 0xff00);

    Reg::And(idxHigh, idxHigh, (Reg::RegTensor<uint32_t> &)idxTmp, pregB32);

    Reg::RegTensor<uint32_t> idxK;
    Reg::Add(idxK, idxHigh, idxLow, pregB16);

    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM_B16>(kValue, idxK, pregB32);
}

/**
    输出所有大于的kth-value的Index
 */
__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint16_t *outputIdxBuf, __ubuf__ uint16_t *inputValueBuf,
                                       uint16_t beginIdx, __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg poutGT;

    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B16>(kthValue, kValue);

    Reg::RegTensor<uint16_t> vregInput;
    Reg::RegTensor<int16_t> idxC;
    Reg::RegTensor<uint16_t> sqzIdxOut;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::Arange(idxC, beginIdx + i * 128);

        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(vregInput, inputValueBuf + i * 128);

        Reg::Compare<uint16_t, CMPMODE::GT>(poutGT, vregInput, (Reg::RegTensor<uint16_t> &)kthValue, pregB16);

        Reg::Squeeze<uint16_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint16_t> &)idxC, poutGT);
        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出所有等于的kth-value的Index
 */
__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint16_t *outputIdxBuf, __ubuf__ uint16_t *inputValueBuf,
                                       uint16_t beginIdx, __ubuf__ uint32_t *kValue, uint16_t vfLoop)
{
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg poutEQ;

    Reg::UnalignRegForStore alignIdx;

    Reg::RegTensor<uint32_t> kthValue;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B16>(kthValue, kValue);

    Reg::RegTensor<uint16_t> vregInput;
    Reg::RegTensor<int16_t> idxC;
    Reg::RegTensor<uint16_t> sqzIdxOut;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::Arange(idxC, beginIdx + i * 128);

        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(vregInput, inputValueBuf + i * 128);

        Reg::Compare<uint16_t, CMPMODE::EQ>(poutEQ, vregInput, (Reg::RegTensor<uint16_t> &)kthValue, pregB16);

        Reg::Squeeze<uint16_t, Reg::GatherMaskMode::STORE_REG>(sqzIdxOut, (Reg::RegTensor<uint16_t> &)idxC, poutEQ);
        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出最终的Value
 */
__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint16_t *outputValueBuf, __ubuf__ uint16_t *inputValueBuf,
                                       __ubuf__ uint16_t *tmpIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint16_t> tmpIdx;
    Reg::RegTensor<uint16_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 128);

        Reg::Gather(outputValue, inputValueBuf, tmpIdx, pregB16);

        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(outputValueBuf + i * 128, outputValue, pregB16);
    }
}

/**
    输出最终的Idx
 */
__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint16_t *tmpIdxBuf,
                                     __ubuf__ uint32_t *hisIdxBuf, uint32_t topK, uint32_t loopIndex, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::MaskReg pregNow;
    Reg::MaskReg pregHis;

    Reg::RegTensor<uint16_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputGatherIdx;
    Reg::RegTensor<uint32_t> outputAddsIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Compares<uint32_t, CMPMODE::GT>(pregNow, (Reg::RegTensor<uint32_t> &)tmpIdx, topK - 1, pregB32);
        Reg::Xor(pregHis, pregNow, pregB32, pregB32);

        Reg::Gather(outputGatherIdx, hisIdxBuf, (Reg::RegTensor<uint32_t> &)tmpIdx, pregHis);
        Reg::Adds(outputAddsIdx, (Reg::RegTensor<uint32_t> &)tmpIdx, loopIndex, pregNow);

        Reg::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputGatherIdx, pregB32);
    }
}

__simd_vf__ void IndicesAddOffsetVF(__ubuf__ uint32_t *indicesOutBuf, uint32_t outputIdxOffset, uint32_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint32_t> outIndices;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(outIndices, indicesOutBuf + i * 64);
        Reg::Adds(outIndices, outIndices, outputIdxOffset, pregB32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(indicesOutBuf + i * 64, outIndices, pregB32);
    }
}

__aicore__ inline void IndicesAddOffset(const LocalTensor<uint32_t> &indicesOutLocal, uint32_t outputIdxOffset,
                                        uint32_t topK)
{
    __ubuf__ uint32_t *indicesOutBuf = (__ubuf__ uint32_t *)indicesOutLocal.GetPhyAddr();
    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;
    IndicesAddOffsetVF(indicesOutBuf, outputIdxOffset, topkLoopNum32);
}

/**
 * @brief LiTopKVF 对一个validLen的输入进行topk算法，输出idx_tmp
 * @param tmpIdxLocal Temp阶段输出的TopKIndex;如果s2SeqLen < 16K作为最终输出 validLen * 2B
 * @param outputValueLocal 如果s2SeqLen > 16K并且是首轮输出Value topK * 2B
 * @param inputValueLocal 输入Value validLen * 2B
 * @param histogramsLocal 直方图 256 * 4B
 * @param idxHighLocal 目标桶高八位 256 * 4B
 * @param idxLowLocal 目标桶低八位 256 * 4B
 * @param nkValueLocal 存储next_k的值 64 * 4B
 * @param topK topK元素
 * @param validLen 有效元素个数:QLIV2Common::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256)
 */
template <bool ISOUTVALUE> // 是否输出VALUE
__aicore__ inline void LiTopKVF(const LocalTensor<uint16_t> &tmpIdxLocal, const LocalTensor<uint16_t> &outputValueLocal,
                                const LocalTensor<uint16_t> &inputValueLocal,
                                const LocalTensor<uint32_t> &histogramsLocal, const LocalTensor<uint32_t> &idxHighLocal,
                                const LocalTensor<uint32_t> &idxLowLocal, const LocalTensor<uint32_t> &nkValueLocal,
                                uint32_t topK, uint32_t validLen)
{
    __ubuf__ uint16_t *tmpIdxBuf = (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t *outputValueBuf = (__ubuf__ uint16_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t *inputValueBuf = (__ubuf__ uint16_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t *histogramsBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t *idxHighBuf = (__ubuf__ uint32_t *)idxHighLocal.GetPhyAddr();
    __ubuf__ uint32_t *idxLowBuf = (__ubuf__ uint32_t *)idxLowLocal.GetPhyAddr();
    __ubuf__ uint32_t *nkValueBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();

    uint32_t bottomK = validLen - topK + 1;
    uint32_t beginIdx = 0;
    bool flag = true;

    const uint16_t repeatSize8 = 256;
    const uint16_t repeatSize16 = 128;
    const uint16_t repeatSize32 = 64;

    uint16_t histogramsLoopNum = (validLen + repeatSize8 - 1) / repeatSize8;
    uint16_t inputLoopNum = (validLen + repeatSize16 - 1) / repeatSize16;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;
    uint16_t topkLoopNum16 = (topK + repeatSize16 - 1) / repeatSize16;

    // find kth-value
    HistogramsHighVFImpl<uint16_t>(histogramsBuf, inputValueBuf, histogramsLoopNum, flag);
    FindHighTargetBinVFImpl(idxHighBuf, nkValueBuf, histogramsBuf, bottomK);

    HistogramsLowVFImpl<uint16_t>(histogramsBuf, inputValueBuf, idxHighBuf, histogramsLoopNum, flag);
    FindKthVFImpl(nkValueBuf, histogramsBuf, idxHighBuf, idxLowBuf);

    // filter
    AscendC::Duplicate(tmpIdxLocal, (uint16_t)(0), QLIV2Common::Align(topK, (uint32_t)128));
    // 输出大于k-value的值idx
    FindIdxGTOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);
    // 输出等于k-value的值idx
    FindIdxEQOutputVFImpl(tmpIdxBuf, inputValueBuf, (uint32_t)(0), nkValueBuf, inputLoopNum);

    // 是否输出Value
    if constexpr (ISOUTVALUE) {
        FindValueOutputVFImpl(outputValueBuf, inputValueBuf, tmpIdxBuf, topkLoopNum16);
    }
}

/**
    LD:输出最终的Idx
*/
__simd_vf__ void FindLDRealIndexVFImpl(__ubuf__ uint32_t *outputIdxBuf, __ubuf__ uint16_t *tmpIdxBuf,
                                       __ubuf__ uint32_t *hisIdxBuf, uint16_t vfLoop)
{
    Reg::MaskReg pregB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint16_t> tmpIdx;
    Reg::RegTensor<uint32_t> outputIdx;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * 64);

        Reg::Gather(outputIdx, hisIdxBuf, (Reg::RegTensor<uint32_t> &)tmpIdx, pregB32);

        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputIdx, pregB32);
    }
}

/**
 * @brief 通过idx_tmp gather出实际的TopKIndex，s2SeqLen > 16K才会执行
 * @param outputIdxLocal 输出Idx 有效:topK * 2B
 * @param outputValueLocal 输出Value topK * 2B(以后需要输出实际value使用)
 * @param inputValueLocal 输入Value validLen * 2B
 * @param tmpIdxLocal 本轮tmpIdx输入 validLen * 2B (0 ~ validLen - 1)
 * @param hisIdxLocal 上一轮实际Idx输入 有效:topK * 4B
 * @param topK topK元素个数
 * @param loopBasicIdx 当前循环需要加上得基准Index
 * @param validLen 有效元素个数
 */
__aicore__ inline void LiTopKGatherVF(const LocalTensor<uint32_t> &outputIdxLocal,
                                      const LocalTensor<uint16_t> &outputValueLocal,
                                      const LocalTensor<uint16_t> &inputValueLocal,
                                      const LocalTensor<uint16_t> &tmpIdxLocal,
                                      const LocalTensor<uint32_t> &hisIdxLocal, uint32_t topK, uint32_t loopBasicIdx,
                                      uint32_t validLen)
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t *outputValueBuf = (__ubuf__ uint16_t *)outputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t *inputValueBuf = (__ubuf__ uint16_t *)inputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t *tmpIdxBuf = (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    const uint16_t repeatSize32 = 64;
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum16 = (topK + repeatSize16 - 1) / repeatSize16;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;

    FindRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topK, loopBasicIdx, topkLoopNum32);
}

/**
    LD:gather最终的Idx
*/
__aicore__ inline void LiTopKLDGatherVF(const LocalTensor<uint32_t> &outputIdxLocal, // 输出Idx topK * 2B
                                        const LocalTensor<uint16_t> &tmpIdxLocal,    // 本轮tmpIdx输入 validLen * 2B
                                        const LocalTensor<uint32_t> &hisIdxLocal,    // 上一轮Idx输入 topK * 4B
                                        uint32_t topK)                               // topK元素个数
{
    __ubuf__ uint32_t *outputIdxBuf = (__ubuf__ uint32_t *)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t *tmpIdxBuf = (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisIdxBuf = (__ubuf__ uint32_t *)hisIdxLocal.GetPhyAddr();

    const uint16_t repeatSize32 = 64;
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum16 = (topK + repeatSize16 - 1) / repeatSize16;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;

    FindLDRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topkLoopNum32);
}
} // namespace topkb16gather
#endif // VF_TOPK_16_GATHER_QUANT_V2_H
