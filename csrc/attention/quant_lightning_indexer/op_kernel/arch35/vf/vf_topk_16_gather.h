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
* \file vf_top_k_16_gather.h
* \brief
*/

#ifndef VF_TOP_K_16_GATHER_H
#define VF_TOP_K_16_GATHER_H

namespace topkb16gather {
    
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

    MicroAPI::MaskReg pregGE;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdxHigh;

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::Duplicate(btmK, bottomK);

    MicroAPI::RegTensor<int32_t> idxC;
    MicroAPI::RegTensor<uint32_t> cout;
    MicroAPI::RegTensor<uint32_t> sqzIdxHigh;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
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

    MicroAPI::MaskReg pregEQ;

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

    MicroAPI::MaskReg pregGE;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdxLow;

    MicroAPI::RegTensor<uint32_t> btmK;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(btmK, kValue);

    MicroAPI::RegTensor<int32_t> idxC;
    MicroAPI::RegTensor<uint32_t> cout;
    MicroAPI::RegTensor<uint32_t> sqzIdxLow;

    for (uint16_t i = 0; i < (uint16_t)(4); ++i) {
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

/**
    输出所有大于的kth-value的Index
 */
__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint16_t* outputIdxBuf, __ubuf__ uint16_t* inputValueBuf, uint16_t beginIdx, __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg poutGT;

    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(kthValue, kValue);

    MicroAPI::RegTensor<uint16_t> vregInput;
    MicroAPI::RegTensor<int16_t> idxC;
    MicroAPI::RegTensor<uint16_t> sqzIdxOut;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::Arange(idxC, beginIdx + i * 128);
        
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputValueBuf + i * 128);

        MicroAPI::Compare<uint16_t, CMPMODE::GT>(poutGT, vregInput, (MicroAPI::RegTensor<uint16_t>&)kthValue, pregB16);

        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint16_t>&)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出所有等于的kth-value的Index
 */
__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint16_t* outputIdxBuf, __ubuf__ uint16_t* inputValueBuf, uint16_t beginIdx, __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg poutEQ;

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(kthValue, kValue);

    MicroAPI::RegTensor<uint16_t> vregInput;
    MicroAPI::RegTensor<int16_t> idxC;
    MicroAPI::RegTensor<uint16_t> sqzIdxOut;

    for(uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::Arange(idxC, beginIdx + i * 128);

        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputValueBuf + i * 128);

        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(poutEQ, vregInput, (MicroAPI::RegTensor<uint16_t>&)kthValue, pregB16);

        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut, (MicroAPI::RegTensor<uint16_t>&)idxC, poutEQ);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出最终的Value
 */
__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint16_t* outputValueBuf, __ubuf__ uint16_t* inputValueBuf, __ubuf__ uint16_t* tmpIdxBuf, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint16_t> outputValue;

    for(uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 128);

        MicroAPI::Gather(outputValue, inputValueBuf, tmpIdx, pregB16);

        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(outputValueBuf + i * 128, outputValue, pregB16);
    }
}

/**
    输出最终的Idx
 */
__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint16_t* tmpIdxBuf, __ubuf__ uint32_t* hisIdxBuf, uint32_t topK, uint32_t loopIndex, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg pregNow;
    MicroAPI::MaskReg pregHis;

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputGatherIdx;
    MicroAPI::RegTensor<uint32_t> outputAddsIdx;

    for(uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * 64);

        MicroAPI::Compares<uint32_t, CMPMODE::GT>(pregNow, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, topK - 1, pregB32);
        MicroAPI::Xor(pregHis, pregNow, pregB32, pregB32);
    
        MicroAPI::Gather(outputGatherIdx, hisIdxBuf, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, pregHis);
        MicroAPI::Adds(outputAddsIdx, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, loopIndex, pregNow);

        MicroAPI::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputGatherIdx, pregB32);
    }
}

/**
    输出Idx_Temp
 */
template<bool ISOUTVALUE> // 是否输出VALUE
__aicore__ inline void LiTopKVF(const LocalTensor<uint16_t>& tmpIdxLocal, // Temp阶段输出的Index;如果s2SeqLen < 16K作为最终输出 validLen * 2B
                                const LocalTensor<uint16_t>& outputValueLocal, // 如果s2SeqLen > 16K并且是首轮输出Value topK * 2B
                                const LocalTensor<uint16_t>& inputValueLocal, // 输入Value validLen * 2B
                                const LocalTensor<uint32_t>& histogramsLocal, // 直方图 256 * 4B 
                                const LocalTensor<uint32_t>& idxHighLocal, // 目标桶高八位 256 * 4B 
                                const LocalTensor<uint32_t>& idxLowLocal, // 目标桶低八位 256 * 4B 
                                const LocalTensor<uint32_t>& nkValueLocal, // 存储next_k的值 64 * 4B
                                uint32_t topK, // topK元素
                                uint32_t validLen) // 有效元素个数
{
    __ubuf__ uint16_t* tmpIdxBuf = (__ubuf__ uint16_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t* outputValueBuf = (__ubuf__ uint16_t*)outputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t* inputValueBuf = (__ubuf__ uint16_t*)inputValueLocal.GetPhyAddr();
    __ubuf__ uint32_t* histogramsBuf = (__ubuf__ uint32_t*)histogramsLocal.GetPhyAddr();
    __ubuf__ uint32_t* idxHighBuf = (__ubuf__ uint32_t*)idxHighLocal.GetPhyAddr();
    __ubuf__ uint32_t* idxLowBuf = (__ubuf__ uint32_t*)idxLowLocal.GetPhyAddr();
    __ubuf__ uint32_t* nkValueBuf = (__ubuf__ uint32_t*)nkValueLocal.GetPhyAddr();

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
    gather最终的Idx
 */
__aicore__ inline void LiTopKGatherVF(const LocalTensor<uint32_t>& outputIdxLocal, // 输出Idx topK * 2B
                                      const LocalTensor<uint16_t>& outputValueLocal, // 输出Value topK * 2B
                                      const LocalTensor<uint16_t>& inputValueLocal, // 输入Value validLen * 2B
                                      const LocalTensor<uint16_t>& tmpIdxLocal, // 本轮tmpIdx输入 validLen * 2B
                                      const LocalTensor<uint32_t>& hisIdxLocal, // 上一轮Idx输入 topK * 4B
                                      uint32_t topK, // topK元素个数
                                      uint32_t loopBasicIdx, // 当前循环需要加上得基准Index
                                      uint32_t validLen) // 有效元素个数
{
    __ubuf__ uint32_t* outputIdxBuf = (__ubuf__ uint32_t*)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t* outputValueBuf = (__ubuf__ uint16_t*)outputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t* inputValueBuf = (__ubuf__ uint16_t*)inputValueLocal.GetPhyAddr();
    __ubuf__ uint16_t* tmpIdxBuf = (__ubuf__ uint16_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* hisIdxBuf = (__ubuf__ uint32_t*)hisIdxLocal.GetPhyAddr();

    const uint16_t repeatSize32 = 64;
    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum16 = (topK + repeatSize16 - 1) / repeatSize16;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;

    FindRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topK, loopBasicIdx, topkLoopNum32);
}

}
#endif