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
* \file vf_top_k_16_gather.h
* \brief
*/

#ifndef VLLM_FAST_LIGHTNING_INDEXER_A5_PAYLOAD_VF_TOP_K_16_GATHER_H
#define VLLM_FAST_LIGHTNING_INDEXER_A5_PAYLOAD_VF_TOP_K_16_GATHER_H

#include "../../lim_payload_layout.h"

namespace VllmFastTopkb16gather {

template<typename T>
__simd_vf__ void HistogramsHighVFImpl(__ubuf__ uint32_t* histogramsBuf,
                                      __ubuf__ uint16_t* inputBuf,
                                      uint16_t vfLoop, bool init)
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
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0,
                                                                    (MicroAPI::RegTensor<uint8_t>&)vregHigh,
                                                                    pregB8);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1,
                                                                    (MicroAPI::RegTensor<uint8_t>&)vregHigh,
                                                                    pregB8);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf,
                                                            cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128,
                                                            cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindHighTargetBinVFImpl(__ubuf__ uint32_t* idxHighBuf,
                                         __ubuf__ uint32_t* nkValueBuf,
                                         __ubuf__ uint32_t* histogramsBuf,
                                         uint32_t bottomK)
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

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
                                                    sqzIdxHigh, (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
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
__simd_vf__ void HistogramsLowVFImpl(__ubuf__ uint32_t* histogramsBuf,
                                     __ubuf__ uint16_t* inputBuf, __ubuf__ uint32_t* idxHighBuf,
                                     uint16_t vfLoop, bool init)
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

        MicroAPI::Compare<uint8_t, CMPMODE::EQ>(pregEQ,
                                                (MicroAPI::RegTensor<uint8_t>&)vregHigh,
                                                (MicroAPI::RegTensor<uint8_t>&)idxHigh, pregB8);

        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout0,
                            (MicroAPI::RegTensor<uint8_t>&)vregLow, pregEQ);
        MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                             MicroAPI::HistogramsType::ACCUMULATE>(cout1,
                            (MicroAPI::RegTensor<uint8_t>&)vregLow, pregEQ);
    }

    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout0U32Even, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout0U32Odd, cout0, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_EVEN>(cout1U32Even, cout1, pregB16);
    MicroAPI::Cast<uint32_t, uint16_t, CAST_TRAIT_UINT16_TOUINT32_ODD>(cout1U32Odd, cout1, pregB16);

    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf,
                                                                        cout0U32Even, cout0U32Odd, pregB32);
    MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(histogramsBuf + 128,
                                                                        cout1U32Even, cout1U32Odd, pregB32);
}

__simd_vf__ void FindKthVFImpl(__ubuf__ uint32_t* kValue,
                               __ubuf__ uint32_t* histogramsBuf, __ubuf__ uint32_t* idxHighBuf,
                               __ubuf__ uint32_t* idxLowBuf)
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

        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxLow,
                                                                         (MicroAPI::RegTensor<uint32_t>&)idxC, pregGE);
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
__simd_vf__ void FindIdxGTOutputVFImpl(__ubuf__ uint16_t* outputIdxBuf,
                                         __ubuf__ uint16_t* inputValueBuf, uint16_t beginIdx,
                                         __ubuf__ uint32_t* kValue, uint16_t vfLoop)
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

        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint16_t>&)idxC, poutGT);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出所有等于的kth-value的Index
 */
__simd_vf__ void FindIdxEQOutputVFImpl(__ubuf__ uint16_t* outputIdxBuf,
                                         __ubuf__ uint16_t* inputValueBuf, uint16_t beginIdx,
                                         __ubuf__ uint32_t* kValue, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg poutEQ;

    MicroAPI::UnalignRegForStore alignIdx;

    MicroAPI::RegTensor<uint32_t> kthValue;
    MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_BRC_B16>(kthValue, kValue);

    MicroAPI::RegTensor<uint16_t> vregInput;
    MicroAPI::RegTensor<int16_t> idxC;
    MicroAPI::RegTensor<uint16_t> sqzIdxOut;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::Arange(idxC, beginIdx + i * 128);

        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputValueBuf + i * 128);

        MicroAPI::Compare<uint16_t, CMPMODE::EQ>(poutEQ, vregInput, (MicroAPI::RegTensor<uint16_t>&)kthValue, pregB16);

        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(sqzIdxOut,
                                                                         (MicroAPI::RegTensor<uint16_t>&)idxC, poutEQ);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, sqzIdxOut, alignIdx);
    }
    MicroAPI::StoreUnAlignPost(outputIdxBuf, alignIdx);
}

/**
    输出最终的Value
 */
__simd_vf__ void FindValueOutputVFImpl(__ubuf__ uint16_t* outputValueBuf,
                                         __ubuf__ uint16_t* inputValueBuf,
                                         __ubuf__ uint16_t* tmpIdxBuf, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint16_t> outputValue;

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 128);

        MicroAPI::Gather(outputValue, inputValueBuf, tmpIdx, pregB16);

        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(outputValueBuf + i * 128, outputValue, pregB16);
    }
}

/**
    输出最终的Idx
 */
__simd_vf__ void FindRealIndexVFImpl(__ubuf__ uint32_t* outputIdxBuf,
                                     __ubuf__ uint16_t* tmpIdxBuf, __ubuf__ uint32_t* hisIdxBuf,
                                     uint32_t topK, uint32_t loopIndex, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    MicroAPI::MaskReg pregNow;
    MicroAPI::MaskReg pregHis;

    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputGatherIdx;
    MicroAPI::RegTensor<uint32_t> outputAddsIdx;
    MicroAPI::RegTensor<uint32_t> loopIndexLowVec;
    MicroAPI::RegTensor<uint32_t> loopIndexHighVec;
    MicroAPI::RegTensor<uint32_t> loopIndexVec;
    MicroAPI::Duplicate(loopIndexLowVec, loopIndex & 0xFFFFu);
    MicroAPI::Duplicate(loopIndexHighVec, loopIndex >> 16);
    MicroAPI::ShiftLefts(loopIndexHighVec, loopIndexHighVec, (int16_t)16, pregB32);
    MicroAPI::Add(loopIndexVec, loopIndexLowVec, loopIndexHighVec, pregB32);

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * 64);

        MicroAPI::Compares<uint32_t, CMPMODE::GT>(pregNow, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, topK - 1, pregB32);
        MicroAPI::Xor(pregHis, pregNow, pregB32, pregB32);

        MicroAPI::Gather(outputGatherIdx, hisIdxBuf, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, pregHis);
        MicroAPI::Add(outputAddsIdx, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, loopIndexVec, pregNow);

        MicroAPI::Add(outputGatherIdx, outputGatherIdx, outputAddsIdx, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputGatherIdx, pregB32);
    }
}

/**
    输出最终的Payload。

    previous survivor 从 hisPayloadBuf 传播完整 payload；current candidate
    从连续的 currentSlotsBuf gather slot，并编码 slot/miss code + token_id。
 */
__simd_vf__ void FindRealPayloadVFImpl(__ubuf__ uint32_t* outputPayloadBuf,
                                       __ubuf__ uint16_t* tmpIdxBuf,
                                       __ubuf__ uint32_t* hisPayloadBuf,
                                       __ubuf__ uint16_t* currentSlotsBuf,
                                       uint32_t previousLen, uint32_t tokenBase,
                                       uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    static constexpr MicroAPI::CastTrait CAST_EVEN = {MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait CAST_ODD = {MicroAPI::RegLayout::ONE,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    MicroAPI::RegTensor<uint16_t> previousLenVec;
    MicroAPI::Duplicate(previousLenVec, static_cast<uint16_t>(previousLen));
    MicroAPI::RegTensor<uint32_t> tokenBaseLowVec;
    MicroAPI::RegTensor<uint32_t> tokenBaseHighVec;
    MicroAPI::RegTensor<uint32_t> tokenBaseVec;
    MicroAPI::RegTensor<uint32_t> sourceMaskVec;
    MicroAPI::Duplicate(tokenBaseLowVec, tokenBase & 0xFFFFu);
    MicroAPI::Duplicate(tokenBaseHighVec, tokenBase >> 16);
    MicroAPI::ShiftLefts(tokenBaseHighVec, tokenBaseHighVec, (int16_t)16, pregB32);
    MicroAPI::Add(tokenBaseVec, tokenBaseLowVec, tokenBaseHighVec, pregB32);
    MicroAPI::Duplicate(sourceMaskVec, lim_payload_layout::SOURCE_MASK);

    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::RegTensor<uint16_t> tmpIdx;
        MicroAPI::RegTensor<uint16_t> currentIdx;
        MicroAPI::RegTensor<uint16_t> currentSlot;
        MicroAPI::MaskReg pregNow16;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(tmpIdx, tmpIdxBuf + i * 128);
        MicroAPI::Compares<uint16_t, CMPMODE::GE>(pregNow16, tmpIdx,
                                                  static_cast<uint16_t>(previousLen), pregB16);
        MicroAPI::Sub(currentIdx, tmpIdx, previousLenVec, pregNow16);
        MicroAPI::Gather(currentSlot, currentSlotsBuf, currentIdx, pregNow16);

        MicroAPI::RegTensor<uint32_t> idxEven;
        MicroAPI::RegTensor<uint32_t> idxOdd;
        MicroAPI::RegTensor<uint32_t> currentIdxEven;
        MicroAPI::RegTensor<uint32_t> currentIdxOdd;
        MicroAPI::RegTensor<uint32_t> slotEven;
        MicroAPI::RegTensor<uint32_t> slotOdd;
        MicroAPI::Cast<uint32_t, uint16_t, CAST_EVEN>(idxEven, tmpIdx, pregB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_ODD>(idxOdd, tmpIdx, pregB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_EVEN>(currentIdxEven, currentIdx, pregB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_ODD>(currentIdxOdd, currentIdx, pregB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_EVEN>(slotEven, currentSlot, pregB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_ODD>(slotOdd, currentSlot, pregB16);

        MicroAPI::MaskReg pregNowEven;
        MicroAPI::MaskReg pregNowOdd;
        MicroAPI::MaskReg pregHisEven;
        MicroAPI::MaskReg pregHisOdd;
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(pregNowEven, idxEven, previousLen, pregB32);
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(pregNowOdd, idxOdd, previousLen, pregB32);
        MicroAPI::Xor(pregHisEven, pregNowEven, pregB32, pregB32);
        MicroAPI::Xor(pregHisOdd, pregNowOdd, pregB32, pregB32);

        MicroAPI::RegTensor<uint32_t> payloadEven;
        MicroAPI::RegTensor<uint32_t> payloadOdd;
        MicroAPI::RegTensor<uint32_t> currentEven;
        MicroAPI::RegTensor<uint32_t> currentOdd;
        MicroAPI::RegTensor<uint32_t> encodedSlotEven;
        MicroAPI::RegTensor<uint32_t> encodedSlotOdd;
        MicroAPI::Gather(payloadEven, hisPayloadBuf, idxEven, pregHisEven);
        MicroAPI::Gather(payloadOdd, hisPayloadBuf, idxOdd, pregHisOdd);
        MicroAPI::Add(currentEven, currentIdxEven, tokenBaseVec, pregNowEven);
        MicroAPI::Add(currentOdd, currentIdxOdd, tokenBaseVec, pregNowOdd);
        MicroAPI::And(currentEven, currentEven, sourceMaskVec, pregNowEven);
        MicroAPI::And(currentOdd, currentOdd, sourceMaskVec, pregNowOdd);
        // currentSlot is int16 bits: valid slots are below MISS_SLOT, while
        // miss -1 == 0xFFFF encodes as the reserved 15-bit MISS_SLOT.
        MicroAPI::ShiftLefts(encodedSlotEven, slotEven,
                             static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), pregNowEven);
        MicroAPI::ShiftLefts(encodedSlotOdd, slotOdd,
                             static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), pregNowOdd);
        MicroAPI::Add(currentEven, currentEven, encodedSlotEven, pregB32);
        MicroAPI::Add(currentOdd, currentOdd, encodedSlotOdd, pregB32);
        MicroAPI::Add(payloadEven, payloadEven, currentEven, pregB32);
        MicroAPI::Add(payloadOdd, payloadOdd, currentOdd, pregB32);

        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            outputPayloadBuf + i * 128, payloadEven, payloadOdd, pregB32);
    }
}

// Gather survivor payload when the current candidates have already been
// compacted into a packed uint32 payload stream. Processing 64 consecutive
// indices preserves score/payload correspondence.
__simd_vf__ void FindExistingPayloadVFImpl(__ubuf__ uint32_t *outputPayloadBuf,
                                           __ubuf__ uint16_t *tmpIdxBuf,
                                           __ubuf__ uint32_t *hisPayloadBuf,
                                           __ubuf__ uint32_t *currentPayloadBuf,
                                           uint32_t previousLen,
                                           uint16_t vfLoop)
{
    MicroAPI::MaskReg allB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg allB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait CAST_EVEN = {MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait CAST_ODD = {MicroAPI::RegLayout::ONE,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    MicroAPI::RegTensor<uint16_t> previousLenB16;
    MicroAPI::Duplicate(previousLenB16, static_cast<uint16_t>(previousLen));

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::RegTensor<uint16_t> idxB16;
        MicroAPI::RegTensor<uint16_t> currentIdxB16;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(
            idxB16, tmpIdxBuf + i * 128);
        MicroAPI::Sub(currentIdxB16, idxB16, previousLenB16, allB16);

        MicroAPI::RegTensor<uint32_t> idxEven;
        MicroAPI::RegTensor<uint32_t> idxOdd;
        MicroAPI::RegTensor<uint32_t> currentIdxEven;
        MicroAPI::RegTensor<uint32_t> currentIdxOdd;
        MicroAPI::Cast<uint32_t, uint16_t, CAST_EVEN>(idxEven, idxB16, allB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_ODD>(idxOdd, idxB16, allB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_EVEN>(currentIdxEven, currentIdxB16, allB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_ODD>(currentIdxOdd, currentIdxB16, allB16);

        MicroAPI::MaskReg isCurrentEven;
        MicroAPI::MaskReg isCurrentOdd;
        MicroAPI::MaskReg isPreviousEven;
        MicroAPI::MaskReg isPreviousOdd;
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(isCurrentEven, idxEven, previousLen, allB32);
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(isCurrentOdd, idxOdd, previousLen, allB32);
        MicroAPI::Xor(isPreviousEven, isCurrentEven, allB32, allB32);
        MicroAPI::Xor(isPreviousOdd, isCurrentOdd, allB32, allB32);

        MicroAPI::RegTensor<uint32_t> zero;
        MicroAPI::RegTensor<uint32_t> previousIdxEven;
        MicroAPI::RegTensor<uint32_t> previousIdxOdd;
        MicroAPI::RegTensor<uint32_t> safeCurrentIdxEven;
        MicroAPI::RegTensor<uint32_t> safeCurrentIdxOdd;
        MicroAPI::Duplicate(zero, static_cast<uint32_t>(0));
        MicroAPI::Select(previousIdxEven, idxEven, zero, isPreviousEven);
        MicroAPI::Select(previousIdxOdd, idxOdd, zero, isPreviousOdd);
        MicroAPI::Select(safeCurrentIdxEven, currentIdxEven, zero, isCurrentEven);
        MicroAPI::Select(safeCurrentIdxOdd, currentIdxOdd, zero, isCurrentOdd);

        MicroAPI::RegTensor<uint32_t> previousEven;
        MicroAPI::RegTensor<uint32_t> previousOdd;
        MicroAPI::RegTensor<uint32_t> currentEven;
        MicroAPI::RegTensor<uint32_t> currentOdd;
        MicroAPI::RegTensor<uint32_t> outputEven;
        MicroAPI::RegTensor<uint32_t> outputOdd;
        MicroAPI::Gather(previousEven, hisPayloadBuf, previousIdxEven, allB32);
        MicroAPI::Gather(previousOdd, hisPayloadBuf, previousIdxOdd, allB32);
        MicroAPI::Gather(currentEven, currentPayloadBuf, safeCurrentIdxEven, allB32);
        MicroAPI::Gather(currentOdd, currentPayloadBuf, safeCurrentIdxOdd, allB32);
        MicroAPI::Select(outputEven, currentEven, previousEven, isCurrentEven);
        MicroAPI::Select(outputOdd, currentOdd, previousOdd, isCurrentOdd);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            outputPayloadBuf + i * 128, outputEven, outputOdd, allB32);
    }
}

// The first histogram stage has no previous survivor. Avoid issuing any
// gather from the uninitialized ping-pong buffer: on A5 that unused source can
// still contaminate lanes before Select. Load 128 B16 indices and gather the
// current payload directly while preserving even/odd lane order.
__simd_vf__ void FindCurrentPayloadVFImpl(__ubuf__ uint32_t *outputPayloadBuf,
                                          __ubuf__ uint16_t *tmpIdxBuf,
                                          __ubuf__ uint32_t *currentPayloadBuf,
                                          uint16_t vfLoop)
{
    MicroAPI::MaskReg allB16 = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg allB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait CAST_EVEN = {MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait CAST_ODD = {MicroAPI::RegLayout::ONE,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::RegTensor<uint16_t> idxB16;
        MicroAPI::RegTensor<uint32_t> idxEven;
        MicroAPI::RegTensor<uint32_t> idxOdd;
        MicroAPI::RegTensor<uint32_t> payloadEven;
        MicroAPI::RegTensor<uint32_t> payloadOdd;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(
            idxB16, tmpIdxBuf + i * 128);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_EVEN>(idxEven, idxB16, allB16);
        MicroAPI::Cast<uint32_t, uint16_t, CAST_ODD>(idxOdd, idxB16, allB16);
        MicroAPI::Gather(payloadEven, currentPayloadBuf, idxEven, allB32);
        MicroAPI::Gather(payloadOdd, currentPayloadBuf, idxOdd, allB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            outputPayloadBuf + i * 128, payloadEven, payloadOdd, allB32);
    }
}

// Initialize first-stage survivor payload with a miss slot placeholder. The
// real slot bits are filled after each contiguous cache_slots DMA.
__simd_vf__ void InitFirstStreamingPayloadVFImpl(
    __ubuf__ uint32_t *outputPayloadBuf,
    __ubuf__ uint16_t *tmpIdxBuf,
    uint32_t tokenBase,
    uint16_t vfLoop)
{
    MicroAPI::MaskReg allB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> tokenBaseVec;
    MicroAPI::RegTensor<uint32_t> missBitsVec;
    MicroAPI::RegTensor<uint32_t> sourceMaskVec;
    MicroAPI::Duplicate(tokenBaseVec, tokenBase);
    MicroAPI::Duplicate(sourceMaskVec, lim_payload_layout::SOURCE_MASK);
    MicroAPI::Duplicate(missBitsVec,
                        lim_payload_layout::MISS_SLOT << lim_payload_layout::SOURCE_BITS);

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::RegTensor<uint16_t> tmpIdx;
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            tmpIdx, tmpIdxBuf + i * 64);
        MicroAPI::Add(payload,
                      (MicroAPI::RegTensor<uint32_t>&)tmpIdx,
                      tokenBaseVec, allB32);
        MicroAPI::And(payload, payload, sourceMaskVec, allB32);
        MicroAPI::Add(payload, payload, missBitsVec, allB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            outputPayloadBuf + i * 64, payload, allB32);
    }
}

// Initialize a later stage. Previous survivors gather their existing payload;
// current-trunk survivors receive {miss_slot, token_id} until their slot chunk
// is streamed into UB.
__simd_vf__ void InitMergedStreamingPayloadVFImpl(
    __ubuf__ uint32_t *outputPayloadBuf,
    __ubuf__ uint16_t *tmpIdxBuf,
    __ubuf__ uint32_t *previousPayloadBuf,
    uint32_t previousLen,
    uint32_t tokenBase,
    uint16_t vfLoop)
{
    MicroAPI::MaskReg allB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> previousLenVec;
    MicroAPI::RegTensor<uint32_t> tokenBaseVec;
    MicroAPI::RegTensor<uint32_t> missBitsVec;
    MicroAPI::RegTensor<uint32_t> sourceMaskVec;
    MicroAPI::RegTensor<uint32_t> zeroVec;
    MicroAPI::Duplicate(previousLenVec, previousLen);
    MicroAPI::Duplicate(tokenBaseVec, tokenBase);
    MicroAPI::Duplicate(sourceMaskVec, lim_payload_layout::SOURCE_MASK);
    MicroAPI::Duplicate(missBitsVec,
                        lim_payload_layout::MISS_SLOT << lim_payload_layout::SOURCE_BITS);
    MicroAPI::Duplicate(zeroVec, static_cast<uint32_t>(0));

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::RegTensor<uint16_t> tmpIdx;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            tmpIdx, tmpIdxBuf + i * 64);
        MicroAPI::MaskReg isCurrent;
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(
            isCurrent, (MicroAPI::RegTensor<uint32_t>&)tmpIdx,
            previousLen, allB32);
        MicroAPI::RegTensor<uint32_t> safePrevious;
        MicroAPI::Select(safePrevious, zeroVec,
                         (MicroAPI::RegTensor<uint32_t>&)tmpIdx, isCurrent);

        MicroAPI::RegTensor<uint32_t> previous;
        MicroAPI::Gather(previous, previousPayloadBuf, safePrevious, allB32);

        MicroAPI::RegTensor<uint32_t> current;
        MicroAPI::Sub(current, (MicroAPI::RegTensor<uint32_t>&)tmpIdx,
                      previousLenVec, allB32);
        MicroAPI::Add(current, current, tokenBaseVec, allB32);
        MicroAPI::And(current, current, sourceMaskVec, allB32);
        MicroAPI::Add(current, current, missBitsVec, allB32);

        MicroAPI::RegTensor<uint32_t> output;
        MicroAPI::Select(output, current, previous, isCurrent);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            outputPayloadBuf + i * 64, output, allB32);
    }
}

// Fill slot bits for survivors whose current-trunk relative index belongs to
// the slot chunk currently resident in UB.
__simd_vf__ void ApplyStreamingSlotChunkVFImpl(
    __ubuf__ uint32_t *outputPayloadBuf,
    __ubuf__ uint16_t *tmpIdxBuf,
    __ubuf__ uint32_t *slotChunkBuf,
    uint32_t previousLen,
    uint32_t chunkBase,
    uint32_t chunkLen,
    uint16_t vfLoop)
{
    MicroAPI::MaskReg allB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> previousLenVec;
    MicroAPI::RegTensor<uint32_t> chunkBaseVec;
    MicroAPI::RegTensor<uint32_t> tokenMaskVec;
    MicroAPI::RegTensor<uint32_t> zeroVec;
    MicroAPI::Duplicate(previousLenVec, previousLen);
    MicroAPI::Duplicate(chunkBaseVec, chunkBase);
    MicroAPI::Duplicate(tokenMaskVec, lim_payload_layout::SOURCE_MASK);
    MicroAPI::Duplicate(zeroVec, static_cast<uint32_t>(0));
    uint32_t chunkEnd = chunkBase + chunkLen;

    for (uint16_t i = 0; i < vfLoop; ++i) {
        MicroAPI::RegTensor<uint16_t> tmpIdx;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            tmpIdx, tmpIdxBuf + i * 64);

        MicroAPI::RegTensor<uint32_t> currentIdx;
        MicroAPI::Sub(currentIdx,
                      (MicroAPI::RegTensor<uint32_t>&)tmpIdx,
                      previousLenVec, allB32);
        MicroAPI::MaskReg geBase;
        MicroAPI::MaskReg ltEnd;
        MicroAPI::MaskReg inChunk;
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(
            geBase, currentIdx, chunkBase, allB32);
        MicroAPI::Compares<uint32_t, CMPMODE::LT>(
            ltEnd, currentIdx, chunkEnd, allB32);
        MicroAPI::And(inChunk, geBase, ltEnd, allB32);

        MicroAPI::RegTensor<uint32_t> localIdx;
        MicroAPI::Sub(localIdx, currentIdx, chunkBaseVec, allB32);
        MicroAPI::Select(localIdx, localIdx, zeroVec, inChunk);

        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::Gather(slot, slotChunkBuf, localIdx, allB32);
        MicroAPI::ShiftLefts(slot, slot,
            static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), allB32);

        MicroAPI::RegTensor<uint32_t> oldPayload;
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::RegTensor<uint32_t> updated;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            oldPayload, outputPayloadBuf + i * 64);
        MicroAPI::And(token, oldPayload, tokenMaskVec, allB32);
        MicroAPI::Add(updated, token, slot, allB32);
        MicroAPI::Select(updated, updated, oldPayload, inChunk);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            outputPayloadBuf + i * 64, updated, allB32);
    }
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
 * @param validLen 有效元素个数:VllmFastHistTopkIndexUpdateA5Common::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256)
 */
template<bool ISOUTVALUE> // 是否输出VALUE
__aicore__ inline void LiTopKVF(const LocalTensor<uint16_t>& tmpIdxLocal,
                                const LocalTensor<uint16_t>& outputValueLocal,
                                const LocalTensor<uint16_t>& inputValueLocal,
                                const LocalTensor<uint32_t>& histogramsLocal,
                                const LocalTensor<uint32_t>& idxHighLocal,
                                const LocalTensor<uint32_t>& idxLowLocal,
                                const LocalTensor<uint32_t>& nkValueLocal,
                                uint32_t topK,
                                uint32_t validLen)
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
    int32_t count = VllmFastHistTopkIndexUpdateA5Common::Align(topK, (uint32_t)128) - topK / 128 * 128;
    AscendC::Duplicate(tmpIdxLocal[topK / 128 * 128], (uint16_t)(0), count);
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
__aicore__ inline void LiTopKGatherVF(const LocalTensor<uint32_t>& outputIdxLocal,
                                      const LocalTensor<uint16_t>& outputValueLocal,
                                      const LocalTensor<uint16_t>& inputValueLocal,
                                      const LocalTensor<uint16_t>& tmpIdxLocal,
                                      const LocalTensor<uint32_t>& hisIdxLocal,
                                      uint32_t topK,
                                      uint32_t loopBasicIdx,
                                      uint32_t validLen)
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

/**
 * @brief 通过 idx_tmp gather 出 survivor payload，s2SeqLen > 16K 执行。
 *
 * previous survivor 保留历史 payload，current candidate 从当前 chunk 的
 * compact slot 表中批量 gather 并生成完整 payload。
 */
__aicore__ inline void LiTopKGatherPayloadVF(const LocalTensor<uint32_t>& outputPayloadLocal,
                                            const LocalTensor<uint16_t>& outputValueLocal,
                                            const LocalTensor<uint16_t>& inputValueLocal,
                                            const LocalTensor<uint16_t>& tmpIdxLocal,
                                            const LocalTensor<uint32_t>& hisPayloadLocal,
                                            const LocalTensor<uint16_t>& currentSlotsLocal,
                                            uint32_t topK,
                                            uint32_t previousLen,
                                            uint32_t tokenBase,
                                            uint32_t validLen)
{
    __ubuf__ uint32_t* outputPayloadBuf = (__ubuf__ uint32_t*)outputPayloadLocal.GetPhyAddr();
    __ubuf__ uint16_t* tmpIdxBuf = (__ubuf__ uint16_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* hisPayloadBuf = (__ubuf__ uint32_t*)hisPayloadLocal.GetPhyAddr();
    __ubuf__ uint16_t* currentSlotsBuf = (__ubuf__ uint16_t*)currentSlotsLocal.GetPhyAddr();

    const uint16_t repeatSize16 = 128;
    uint16_t topkLoopNum16 = (topK + repeatSize16 - 1) / repeatSize16;
    (void)outputValueLocal;
    (void)inputValueLocal;
    (void)validLen;

    FindRealPayloadVFImpl(outputPayloadBuf, tmpIdxBuf, hisPayloadBuf, currentSlotsBuf,
                          previousLen, tokenBase, topkLoopNum16);
}

__aicore__ inline void LiTopKGatherExistingPayloadVF(
    const LocalTensor<uint32_t>& outputPayloadLocal,
    const LocalTensor<uint16_t>& tmpIdxLocal,
    const LocalTensor<uint32_t>& hisPayloadLocal,
    const LocalTensor<uint32_t>& currentPayloadLocal,
    uint32_t topK,
    uint32_t previousLen)
{
    __ubuf__ uint32_t *outputPayloadBuf =
        (__ubuf__ uint32_t *)outputPayloadLocal.GetPhyAddr();
    __ubuf__ uint16_t *tmpIdxBuf =
        (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *hisPayloadBuf =
        (__ubuf__ uint32_t *)hisPayloadLocal.GetPhyAddr();
    __ubuf__ uint32_t *currentPayloadBuf =
        (__ubuf__ uint32_t *)currentPayloadLocal.GetPhyAddr();
    uint16_t loopNum = static_cast<uint16_t>((topK + 127) / 128);
    FindExistingPayloadVFImpl(outputPayloadBuf, tmpIdxBuf, hisPayloadBuf,
                              currentPayloadBuf, previousLen, loopNum);
}

__aicore__ inline void LiTopKGatherCurrentPayloadVF(
    const LocalTensor<uint32_t>& outputPayloadLocal,
    const LocalTensor<uint16_t>& tmpIdxLocal,
    const LocalTensor<uint32_t>& currentPayloadLocal,
    uint32_t topK)
{
    __ubuf__ uint32_t *outputPayloadBuf =
        (__ubuf__ uint32_t *)outputPayloadLocal.GetPhyAddr();
    __ubuf__ uint16_t *tmpIdxBuf =
        (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t *currentPayloadBuf =
        (__ubuf__ uint32_t *)currentPayloadLocal.GetPhyAddr();
    uint16_t loopNum = static_cast<uint16_t>((topK + 127) / 128);
    FindCurrentPayloadVFImpl(outputPayloadBuf, tmpIdxBuf, currentPayloadBuf, loopNum);
}

// The long-source high four bits follow the same histogram survivor indices
// as the packed payload. A current score chunk derives its tag from its
// absolute source position; global reduction gathers staged candidate tags.
__simd_vf__ void GatherSourceTagsVFImpl(
    __ubuf__ uint16_t *outputTags, __ubuf__ uint16_t *tmpIndices,
    __ubuf__ uint16_t *previousTags, __ubuf__ uint16_t *currentTags,
    uint16_t previousLen, uint32_t tokenBase, bool stagedCurrent,
    uint16_t loops)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint16_t> previousLenVec;
    MicroAPI::RegTensor<uint16_t> baseTagVec;
    MicroAPI::RegTensor<uint16_t> nextTagVec;
    MicroAPI::RegTensor<uint16_t> zeroVec;
    MicroAPI::Duplicate(previousLenVec, previousLen);
    MicroAPI::Duplicate(baseTagVec, static_cast<uint16_t>(tokenBase >>
        lim_payload_layout::SOURCE_BITS));
    MicroAPI::Duplicate(nextTagVec, static_cast<uint16_t>(
        (tokenBase >> lim_payload_layout::SOURCE_BITS) + 1U));
    MicroAPI::Duplicate(zeroVec, static_cast<uint16_t>(0));
    const uint32_t boundary = lim_payload_layout::PACKED_SOURCE_CAPACITY -
        (tokenBase & lim_payload_layout::SOURCE_MASK);
    for (uint16_t loop = 0; loop < loops; ++loop) {
        MicroAPI::RegTensor<uint16_t> indices;
        MicroAPI::RegTensor<uint16_t> currentIndices;
        MicroAPI::RegTensor<uint16_t> tags;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(
            indices, tmpIndices + loop * 128U);
        MicroAPI::Sub(currentIndices, indices, previousLenVec, all);
        if (stagedCurrent) {
            MicroAPI::RegTensor<uint16_t> safeCurrent;
            if (previousLen != 0U) {
                MicroAPI::MaskReg isCurrent;
                MicroAPI::Compares<uint16_t, CMPMODE::GE>(
                    isCurrent, indices, previousLen, all);
                MicroAPI::Select(safeCurrent, currentIndices, zeroVec, isCurrent);
            } else {
                MicroAPI::Add(safeCurrent, currentIndices, zeroVec, all);
            }
            MicroAPI::Gather(tags, currentTags, safeCurrent, all);
        } else if (boundary < 65536U) {
            MicroAPI::MaskReg nextPage;
            MicroAPI::Compares<uint16_t, CMPMODE::GE>(
                nextPage, currentIndices, static_cast<uint16_t>(boundary), all);
            MicroAPI::Select(tags, nextTagVec, baseTagVec, nextPage);
        } else {
            MicroAPI::Duplicate(tags, static_cast<uint16_t>(tokenBase >>
                lim_payload_layout::SOURCE_BITS));
        }
        if (previousLen != 0U) {
            MicroAPI::MaskReg isPrevious;
            MicroAPI::RegTensor<uint16_t> safePrevious;
            MicroAPI::RegTensor<uint16_t> oldTags;
            MicroAPI::Compares<uint16_t, CMPMODE::LT>(
                isPrevious, indices, previousLen, all);
            MicroAPI::Select(safePrevious, indices, zeroVec, isPrevious);
            MicroAPI::Gather(oldTags, previousTags, safePrevious, all);
            MicroAPI::Select(tags, oldTags, tags, isPrevious);
        }
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(
            outputTags + loop * 128U, tags, all);
    }
}

__aicore__ inline void LiTopKGatherSourceTagsVF(
    const LocalTensor<uint16_t>& outputTags,
    const LocalTensor<uint16_t>& tmpIndices,
    const LocalTensor<uint16_t>& previousTags,
    const LocalTensor<uint16_t>& currentTags,
    uint32_t topK, uint32_t previousLen,
    uint32_t tokenBase, bool stagedCurrent)
{
    GatherSourceTagsVFImpl(
        (__ubuf__ uint16_t *)outputTags.GetPhyAddr(),
        (__ubuf__ uint16_t *)tmpIndices.GetPhyAddr(),
        (__ubuf__ uint16_t *)previousTags.GetPhyAddr(),
        (__ubuf__ uint16_t *)currentTags.GetPhyAddr(),
        static_cast<uint16_t>(previousLen), tokenBase, stagedCurrent,
        static_cast<uint16_t>((topK + 127U) / 128U));
}

__aicore__ inline void InitFirstStreamingPayloadVF(
    const LocalTensor<uint32_t>& outputPayloadLocal,
    const LocalTensor<uint16_t>& tmpIdxLocal,
    uint32_t topK,
    uint32_t tokenBase)
{
    uint16_t loopNum = static_cast<uint16_t>((topK + 63) / 64);
    InitFirstStreamingPayloadVFImpl(
        (__ubuf__ uint32_t *)outputPayloadLocal.GetPhyAddr(),
        (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr(),
        tokenBase, loopNum);
}

__aicore__ inline void InitMergedStreamingPayloadVF(
    const LocalTensor<uint32_t>& outputPayloadLocal,
    const LocalTensor<uint16_t>& tmpIdxLocal,
    const LocalTensor<uint32_t>& previousPayloadLocal,
    uint32_t topK,
    uint32_t previousLen,
    uint32_t tokenBase)
{
    uint16_t loopNum = static_cast<uint16_t>((topK + 63) / 64);
    InitMergedStreamingPayloadVFImpl(
        (__ubuf__ uint32_t *)outputPayloadLocal.GetPhyAddr(),
        (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)previousPayloadLocal.GetPhyAddr(),
        previousLen, tokenBase, loopNum);
}

__aicore__ inline void ApplyStreamingSlotChunkVF(
    const LocalTensor<uint32_t>& outputPayloadLocal,
    const LocalTensor<uint16_t>& tmpIdxLocal,
    const LocalTensor<int32_t>& slotChunkLocal,
    uint32_t topK,
    uint32_t previousLen,
    uint32_t chunkBase,
    uint32_t chunkLen)
{
    uint16_t loopNum = static_cast<uint16_t>((topK + 63) / 64);
    ApplyStreamingSlotChunkVFImpl(
        (__ubuf__ uint32_t *)outputPayloadLocal.GetPhyAddr(),
        (__ubuf__ uint16_t *)tmpIdxLocal.GetPhyAddr(),
        (__ubuf__ uint32_t *)slotChunkLocal.GetPhyAddr(),
        previousLen, chunkBase, chunkLen, loopNum);
}

}
#endif // VLLM_FAST_LIGHTNING_INDEXER_A5_PAYLOAD_VF_TOP_K_16_GATHER_H
