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
 * \file attenmask.h
 * \brief
 */

#ifndef ATTENMASK_H
#define ATTENMASK_H

#include "util_regbase.h"
#include "flash_attention_score_common_regbase.h"
#include "../const_def.h"

using namespace AscendC;
using namespace AscendC::MicroAPI;
namespace regbaseutil {
enum class AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE = 1,
    RIGHT_DOWN_CAUSAL_MODE = 2,
    BAND_MODE = 3,
    PREFIX_MODE = 4,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE = 6
};

enum class AttenMaskComputeMode {
    NORMAL_MODE = 0,
    CAUSAL_OR_NEXT_ONLY_MODE,
    PRE_ONLY_MODE,
    PRE_AND_NEXT_MODE,
    NO_NEED_COMPUTE_MODE,
    PREFIX_COMPUTE_MODE,
    PREFIX_N_COMPUTE_MODE
};

struct AttenMaskInfo {
    int64_t preTokens;
    int64_t nextTokens;
    uint8_t compressMode;
    int64_t attenMaskShapeType;
    int64_t attenMaskS1Size;
    int64_t attenMaskS2Size;
    int64_t bandIndex;
    GM_ADDR prefixNAddr;
    int64_t attenMaskOffsetPre;
    AttenMaskComputeMode computeMode = AttenMaskComputeMode::NORMAL_MODE;
    int64_t attenMaskS1Offset;
};

template <bool isInfer = false, bool hasRope = false>
__aicore__ inline void BoolCopyInRegbase(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t s2BaseSize, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = CeilDiv(s2Size, blockBytes);
    dataCopyParams.dstStride = CeilDiv(s2BaseSize, blockBytes) - dataCopyParams.blockLen;
    if (totalS2Size % blockBytes == 0 && s2Size % blockBytes == 0) {
        dataCopyParams.srcStride = (totalS2Size - s2Size) / blockBytes;
        if constexpr (isInfer == true) {
            if (constInfo.isGqa) {
                dataCopyParams.blockCount = 1;
                // IFA GS1合轴后, s1Size = gSize * s1 (1) , 但mask实际只有s1 (1) 行，因此需要循环拷贝
                for (uint32_t i = 0; i < s1Size; ++i) {
                    // 需要用 s2BaseSize, 兼容S2方向有尾块情况
                    DataCopy(dstTensor[i * s2BaseSize], srcTensor[srcOffset], dataCopyParams);
                }
                return;
            }
        }
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
    } else {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = s1Size;
        dataCopyExtParams.dstStride = CeilDiv(s2BaseSize, blockBytes) - CeilDiv(s2Size, blockBytes);
        dataCopyExtParams.blockLen = s2Size;
        dataCopyExtParams.srcStride = totalS2Size - s2Size;
        DataCopyPadExtParams<uint8_t> dataCopyPadParams;
        if constexpr (isInfer == true) {
            if (constInfo.isGqa) {
                dataCopyExtParams.blockCount = 1;
                // IFA GS1合轴后, 1Size = gSize * s1 (1) , 但mask实际只有s1 (1) 行，因此需要循环拷贝
                for (uint32_t i = 0; i < s1Size; ++i) {
                    // 需要用 s2BaseSize, 兼容S2方向有尾块情况
                    DataCopyPad(dstTensor[i * s2BaseSize], srcTensor[srcOffset], dataCopyExtParams, dataCopyPadParams);
                }
                return;
            }
        }
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyExtParams, dataCopyPadParams);
    }
}

template <bool hasAtten, bool isInfer = false, bool hasRope = false, bool isMxfp8FullQuant = false>
__aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre,
                                               int64_t s1Offset, const RunInfo<isInfer> &runInfo, 
                                               ConstInfo<isInfer, hasRope> &constInfo,
                                               AttenMaskInfo &attenMaskInfo)
{
    if constexpr (hasAtten == true) {
        uint32_t s2BaseSize = constInfo.s2BaseSize;
        if constexpr (isMxfp8FullQuant) {
            s2BaseSize /= 2;
        }
        int64_t causalOrNextFactor = deltaCausalOrNext - s2BaseSize;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
            attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            if (causalOrNextFactor >= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            }
            return;
        }
        if (((attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) ||
            (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE) &&
                runInfo.boIdx == attenMaskInfo.bandIndex) ||
            (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE) &&
                runInfo.boIdx == attenMaskInfo.bandIndex))) {
            int64_t preFactor = deltaPre + 1 + constInfo.s1BaseSize;
            if (causalOrNextFactor >= 0 && preFactor <= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else if (causalOrNextFactor < 0 && preFactor <= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            } else if (causalOrNextFactor >= 0 && preFactor > 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
            } else {
                attenMaskInfo.computeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
            }
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            int64_t preFactor = deltaPre - constInfo.s2BaseSize;
            // Triangular part and rectangular part have one is not counted, then the whole is not counted,
            // otherwise it needs to be calculated
            if (causalOrNextFactor >= 0 || preFactor >= 0 || deltaPre > runInfo.actualS2Size) {
                // attenmask value is all 0, no need to compute
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                int64_t intersectionX = runInfo.actualS1Size - runInfo.actualS2Size +
                    ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx];
                if (s1Offset >= intersectionX) {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
                } else if (s1Offset + constInfo.s1BaseSize <= intersectionX) {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE;
                } else {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::PREFIX_COMPUTE_MODE;
                }
            }
        }
        return;
    }
}

template <bool hasAtten, bool enableKVPrefix, DTemplateType dTemplateType = DTemplateType::Aligned128,
    bool isInfer = false, bool hasRope = false, bool isFd = false>
__aicore__ inline int64_t ComputeOffsetForNoCompress(const RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo, AttenMaskInfo &attenMaskInfo)
{
    if constexpr (hasAtten == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t gOffset = 0;

        if (attenMaskInfo.attenMaskShapeType == attenMaskBN2GS1S2) {
            bOffset = runInfo.b1SSAttenMaskOffset * constInfo.n2G;
            n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size * runInfo.actualS2Size;
            gOffset = runInfo.goIdx * runInfo.actualS1Size * runInfo.actualS2Size;
        } else if (attenMaskInfo.attenMaskShapeType == attenMaskBS1S2) {
            bOffset = runInfo.b1SSAttenMaskOffset;
        }
        int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset;
        if constexpr(isInfer) { 
            if (constInfo.isGqa) { 
                s1Offset = s1Offset % constInfo.s1Size; 
            } else if (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
                if (constInfo.layoutType == (uint32_t)LayOutTypeEnum::LAYOUT_BNSD) {
                    s1Offset = attenMaskInfo.attenMaskS1Offset;
                } else {
                    s1Offset = s1Offset / constInfo.gSize;
                }
            }
        }
        int64_t s2Offset = 0;
        if constexpr (enableKVPrefix) {
            if ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) < constInfo.prefixLoopCount) {
                s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
            } else {
                s2Offset = runInfo.s2StartIdx + (runInfo.s2LoopCount - constInfo.prefixLoopCount) * constInfo.s2BaseSize + constInfo.actualKVPrefixSize;
            }
        } else {
            s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
        }
        if constexpr (isFd) {
            s2Offset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize;
        }
        if constexpr (isInfer) {
            if (hasRope && (dTemplateType == DTemplateType::Aligned576) && isInfer) {
                if (constInfo.layoutType != (uint32_t)LayOutTypeEnum::LAYOUT_BNSD) {
                    s1Offset += (runInfo.nextTokensOfMlaPerBatch < 0) ? -runInfo.nextTokensOfMlaPerBatch : 0;
                }
            } else {
                s1Offset += (runInfo.nextTokensPerBatch < 0) ? -runInfo.nextTokensPerBatch : 0;
            }
            s1Offset += runInfo.queryLeftPaddingSize;
            s2Offset += runInfo.kvLeftPaddingSize;
        }
        s1Offset *= attenMaskInfo.attenMaskS2Size;
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    }
}

__aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta, const uint32_t &s1BaseSize,
                                                 const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size,
                                                 const int64_t &vecCoreOffset, const bool useDn = false)
{
    if (useDn) {
        if (delta >= 0) {
            return AttentionCommon::Min(delta, s2BaseSize) + vecCoreOffset + 1;
        }
        return (AttentionCommon::Min(-1 * delta, s1BaseSize)) * attenMaskS2Size + vecCoreOffset + 1;
    }
    if (delta <= 0) {
        return AttentionCommon::Min(-1 * delta, s1BaseSize) + vecCoreOffset * attenMaskS2Size;
    }
    return (AttentionCommon::Min(delta, s2BaseSize) + vecCoreOffset) * attenMaskS2Size;
}

__aicore__ inline int64_t ComputeOffsetForPrefixRectangle(const int64_t &delta, const uint32_t &s2BaseSize,
                                                          const uint32_t &attenMaskS2Size)
{
    // attenMask S1 is same to S2
    if (delta <= 0) {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2; // 2048 * 2048 + 1024
    } else if (delta >= s2BaseSize) {
        return attenMaskS2Size * attenMaskS2Size; // 2048 * 2048 + 0
    } else {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2 - delta; // 2048 * 2048 + (1024 - delta)
    }
}

#ifndef __CCE_KT_TEST__
__simd_vf__ inline void MergeBandVF(const uint64_t maskPreUb, const uint64_t maskNextUb,
                                    const uint16_t loopCount)
{
    RegTensor<uint32_t> vreg_pre;
    RegTensor<uint32_t> vreg_next;
    RegTensor<uint32_t> vreg_xor;
    RegTensor<uint32_t> vreg_not;
    RegTensor<uint32_t> vreg_or;
    MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();
    Duplicate(vreg_xor, 0x1010101);

    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb + i * 64);
        LoadAlign(vreg_next, (__ubuf__ uint32_t*&)maskNextUb + i * 64);
        Xor(vreg_not, vreg_pre, vreg_xor, preg_all);
        Or(vreg_or, vreg_not, vreg_next, preg_all);
        StoreAlign((__ubuf__ uint32_t*&)maskNextUb + i * 64, vreg_or, preg_all);
    }
}

template <bool hasAtten>
__aicore__ inline void MergeBandModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                         int32_t &halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop;
    uint64_t rowNumTimesEachLoop;
    if (s2BaseSize > regBytes) {
        rowNumEachLoop = 1;
        rowNumTimesEachLoop = static_cast<uint16_t>(s2BaseSize) / regBytes;
    } else {
        rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
        rowNumTimesEachLoop = 1;
    }
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = (halfS1RealSizeLoop / rowNumEachLoop) * rowNumTimesEachLoop;

    MergeBandVF(maskPreUb, maskNextUb, loopCount);
}

__simd_vf__ inline void MergePrefixVF(const uint64_t maskPreUb, const uint64_t maskNextUb,
                                      const uint16_t loopCount)
{
    RegTensor<uint32_t> vreg_pre;
    RegTensor<uint32_t> vreg_next;
    RegTensor<uint32_t> vreg_and;
    MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb + i * 64);
        LoadAlign(vreg_next, (__ubuf__ uint32_t*&)maskNextUb + i * 64);
        And(vreg_and, vreg_pre, vreg_next, preg_all);
        StoreAlign((__ubuf__ uint32_t*&)maskNextUb + i * 64, vreg_and, preg_all);
    }
}

template <bool hasAtten>
__aicore__ inline void MergePrefixModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                         int32_t &halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop;
    uint64_t rowNumTimesEachLoop;
    if (s2BaseSize > regBytes) {
        rowNumEachLoop = 1;
        rowNumTimesEachLoop = static_cast<uint16_t>(s2BaseSize) / regBytes;
    } else {
        rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
        rowNumTimesEachLoop = 1;
    }
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = (halfS1RealSizeLoop / rowNumEachLoop) * rowNumTimesEachLoop;

    MergePrefixVF(maskPreUb, maskNextUb, loopCount);
}
#else
template <bool hasAtten>
__aicore__ inline void MergeBandModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                         int32_t &halfS1RealSize, int64_t s2BaseSize)
{
}

template <bool hasAtten>
__aicore__ inline void MergePrefixModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                           int32_t &halfS1RealSize, int64_t s2BaseSize)
{
}
#endif
                                      
template <bool hasAtten, bool enableKVPrefix = false, bool isFd = false, bool hasRope = false, bool isInfer = false,
    DTemplateType dTemplateType = DTemplateType::Aligned128, bool isMxfp8FullQuant = false>
__aicore__ inline int64_t ComputeAttenMaskOffset(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
    AttenMaskInfo &attenMaskInfo, const bool useDn = false, const int32_t subLoop = 0)
{
    if constexpr (hasAtten == true) {
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
            return ComputeOffsetForNoCompress<hasAtten, enableKVPrefix, dTemplateType, isInfer, hasRope, isFd>(runInfo, constInfo, attenMaskInfo);
        }
        if (constInfo.layoutType == (uint32_t)LayOutTypeEnum::LAYOUT_TND && !isInfer) {
            // compress mode
            int64_t delta = 0;
            int64_t deltaPre = 0;
            int64_t deltaN = runInfo.actualS1Size - runInfo.actualS2Size;
            int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize;
            if constexpr (isInfer) {
                if (constInfo.isGqa) {
                    s1Offset = s1Offset % constInfo.s1Size;
                }
            }
            int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
            if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset;
            } else if (attenMaskInfo.compressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
            } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
                int64_t tmpPre = attenMaskInfo.preTokens;
                int64_t tmpNext = attenMaskInfo.nextTokens;
                int64_t transPreTokens = runInfo.actualS1Size - AttentionCommon::Max(runInfo.actualS2Size - tmpPre, 0);
                int64_t transNextTokens = runInfo.actualS2Size - AttentionCommon::Max(runInfo.actualS1Size - tmpNext, 0);
                deltaPre = s1Offset - s2Offset - transPreTokens - 1;
                int64_t maskOffsetPre = ComputeOffsetForCausal(deltaPre, constInfo.s1BaseSize, constInfo.s2BaseSize,
                                                               attenMaskInfo.attenMaskS2Size, runInfo.vecCoreOffset);
                attenMaskInfo.attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask operation.
                delta = s1Offset - s2Offset + transNextTokens;
            } else if (attenMaskInfo.compressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE)) {
                if (runInfo.boIdx == attenMaskInfo.bandIndex) {
                    int64_t tmpPre = attenMaskInfo.preTokens;
                    int64_t tmpNext = attenMaskInfo.nextTokens;
                    int64_t transPreTokens = runInfo.actualS1Size - AttentionCommon::Max(runInfo.actualS2Size - tmpPre, 0);
                    int64_t transNextTokens = runInfo.actualS2Size - AttentionCommon::Max(runInfo.actualS1Size - tmpNext, 0);
                    deltaPre = s1Offset - s2Offset - transPreTokens - 1;
                    int64_t maskOffsetPre = ComputeOffsetForCausal(deltaPre, constInfo.s1BaseSize, constInfo.s2BaseSize,
                                                                attenMaskInfo.attenMaskS2Size, runInfo.vecCoreOffset);
                    attenMaskInfo.attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask
                    delta = s1Offset - s2Offset + transNextTokens;
                } else {
                    delta = s1Offset - s2Offset - deltaN;
                }
            } else if (attenMaskInfo.compressMode ==
                       static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE)) {
                if (runInfo.boIdx == attenMaskInfo.bandIndex) {
                    int64_t tmpPre = attenMaskInfo.preTokens;
                    int64_t tmpNext = attenMaskInfo.nextTokens;
                    int64_t transPreTokens = runInfo.actualS1Size - AttentionCommon::Max(runInfo.actualS2Size - tmpPre, 0);
                    int64_t transNextTokens = runInfo.actualS2Size - AttentionCommon::Max(runInfo.actualS1Size - tmpNext, 0);
                    deltaPre = s1Offset - s2Offset - transPreTokens - 1;
                    int64_t maskOffsetPre = ComputeOffsetForCausal(deltaPre, constInfo.s1BaseSize, constInfo.s2BaseSize,
                                                                attenMaskInfo.attenMaskS2Size, runInfo.vecCoreOffset);
                    attenMaskInfo.attenMaskOffsetPre = maskOffsetPre; // save offset value for the 2nd mask operation.
                    delta = s1Offset - s2Offset + transNextTokens;
                } else {
                    delta = s1Offset - s2Offset;
                }
            } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
                delta = s1Offset - s2Offset - deltaN;
                if ((runInfo.actualS1Size + ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx]) >
                    runInfo.actualS2Size) {
                    // prefix reuse attenMaskOffsetPre
                    deltaPre = ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx] - runInfo.s2StartIdx -
                               runInfo.s2LoopCount * constInfo.s2BaseSize;
                    attenMaskInfo.attenMaskOffsetPre = ComputeOffsetForPrefixRectangle(
                        deltaPre, constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size);
                }
            } else {
                return 0;
            }
            GetAttenMaskComputeMode<hasAtten>(delta, deltaPre, s1Offset, runInfo, constInfo, attenMaskInfo);
            return ComputeOffsetForCausal(delta, constInfo.s1BaseSize, constInfo.s2BaseSize,
                                          attenMaskInfo.attenMaskS2Size, runInfo.vecCoreOffset, useDn);
        }
        // compress mode, 推理场景的TND和TND的offset计算相同，因为mask被padding到了最大的s2Size
        int64_t deltaCausalOrNext = 0;
        int64_t deltaPre = 0;
        int64_t deltaN = runInfo.actualS1Size - runInfo.actualS2Size;
        if constexpr (enableKVPrefix) {
            deltaN -= constInfo.actualKVPrefixSize;
        }
        int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize;
        if constexpr (isInfer) {
            if (constInfo.isGqa) {
                s1Offset = s1Offset % constInfo.s1Size;
            }
        }
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576) && isInfer) {
            if (constInfo.layoutType == (uint32_t)LayOutTypeEnum::LAYOUT_BNSD) {
                s1Offset = attenMaskInfo.attenMaskS1Offset;
            } else {
                if (runInfo.nextTokensPerBatch < 0) {
                    s1Offset -= runInfo.nextTokensPerBatch;
                }
                s1Offset = (s1Offset + runInfo.vecCoreOffset) / constInfo.gSize;
            }
        }
        int64_t s2Offset = 0;
        if constexpr (enableKVPrefix) {
            if ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) < constInfo.prefixLoopCount) {
                s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
            } else {
                s2Offset = runInfo.s2StartIdx + (runInfo.s2LoopCount - constInfo.prefixLoopCount) * constInfo.s2BaseSize + constInfo.actualKVPrefixSize;
            }
        } else {
            s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
            if constexpr (isMxfp8FullQuant) {
                s2Offset += subLoop * (constInfo.s2BaseSize / 2);
            }
        }
        if constexpr (isFd) {
            s2Offset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize;
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset;
        } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
            if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576) && isInfer) {
                deltaCausalOrNext = runInfo.nextTokensOfMlaPerBatch + s1Offset - s2Offset;
            }
        } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576) && isInfer) {
                deltaPre = -runInfo.preTokensOfMlaPerBatch + s1Offset - s2Offset - 1;
                deltaCausalOrNext = runInfo.nextTokensOfMlaPerBatch + s1Offset - s2Offset;
                attenMaskInfo.attenMaskOffsetPre = ComputeOffsetForCausal(deltaPre, constInfo.s1BaseSize,
                    constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size, 0);
            } else {
                if constexpr (isInfer) {
                    /* 推理的S1循环会跳过无效行，训练的不会；原因是推理在最开始存在无效行场景下会对
                    整个输出进行初始化为0，所以可以跳过无效行，训练由于还需要对max、sum赋值成特殊值，没有跳过无效行，
                    直接计算。RIGHT_DOWN_CAUSAL_MODE中不需要做这个转换时因为推理的runInfo.actualS1Size已经减掉了无效
                    的行。*/
                    if (runInfo.nextTokensPerBatch < 0) {
                        s1Offset -= runInfo.nextTokensPerBatch;
                    }
                }
                deltaPre = s1Offset - s2Offset - runInfo.preTokensPerBatch - 1;
                deltaCausalOrNext = s1Offset - s2Offset + runInfo.nextTokensPerBatch;
                attenMaskInfo.attenMaskOffsetPre = ComputeOffsetForCausal(deltaPre, constInfo.s1BaseSize,
                    constInfo.s2BaseSize, attenMaskInfo.attenMaskS2Size, runInfo.vecCoreOffset, useDn);
            }
        } else if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
            if ((constInfo.s1Size + ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx]) >
                runInfo.actualS2Size) {
                // prefix reuse attenMaskOffsetPre
                deltaPre = ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx] - runInfo.s2StartIdx -
                           runInfo.s2LoopCount * constInfo.s2BaseSize;
                attenMaskInfo.attenMaskOffsetPre = ComputeOffsetForPrefixRectangle(deltaPre, constInfo.s2BaseSize,
                                                                                   attenMaskInfo.attenMaskS2Size);
            }
        } else {
            return 0;
        }
        GetAttenMaskComputeMode<hasAtten, isInfer, hasRope, isMxfp8FullQuant>(deltaCausalOrNext, deltaPre, s1Offset, runInfo, constInfo, attenMaskInfo);
        int64_t ret = 0;
        if constexpr (hasRope && (dTemplateType == DTemplateType::Aligned576)) {
            ret = ComputeOffsetForCausal(deltaCausalOrNext, constInfo.s1BaseSize, constInfo.s2BaseSize,
                attenMaskInfo.attenMaskS2Size, 0);
        } else {
            uint32_t s2BaseSize = constInfo.s2BaseSize;
            if constexpr (isMxfp8FullQuant) {
                s2BaseSize /= 2;
            }
            ret = ComputeOffsetForCausal(deltaCausalOrNext, constInfo.s1BaseSize, s2BaseSize,
                attenMaskInfo.attenMaskS2Size, runInfo.vecCoreOffset, useDn);
        }
        return ret;
    }
}

template <bool hasAtten, bool isFd = false, bool hasRope = false, bool isInfer = false>
__aicore__ inline void AttenMaskCopyIn(TQue<QuePosition::VECIN, 1> &attenMaskInQue, GlobalTensor<uint8_t> &srcTensor,
                                       RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, AttenMaskInfo &attenMaskInfo,
                                       bool isPre = false)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten>(runInfo, constInfo, attenMaskInfo);
        BoolCopyInRegbase<isInfer>(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo);
        attenMaskInQue.template EnQue(attenMaskUb);
        return;
    }
}

template <bool hasAtten, bool isFd = false, bool enableKVPrefix = false, bool hasRope = false, bool isInfer = false>
__aicore__ inline void AttenMaskCopyIn(TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                       TQue<QuePosition::VECIN, 1> &attenMaskInQuePre, GlobalTensor<uint8_t> &srcTensor,
                                       RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, AttenMaskInfo &attenMaskInfo)
{
    if constexpr (hasAtten == true) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten, enableKVPrefix, isFd>(runInfo, constInfo, attenMaskInfo);
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            if (attenMaskInfo.computeMode == AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
                maskOffset = 4194304; // 2048*2048
            }
            if (attenMaskInfo.computeMode == AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE) {
                maskOffset = attenMaskInfo.attenMaskOffsetPre;
            }
            BoolCopyInRegbase<isInfer>(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                              attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo);
            attenMaskInQue.template EnQue(attenMaskUb);
            if (attenMaskInfo.computeMode == AttenMaskComputeMode::PREFIX_COMPUTE_MODE) {
                LocalTensor<uint8_t> attenMaskUbPre = attenMaskInQuePre.template AllocTensor<uint8_t>();
                BoolCopyInRegbase<isInfer>(attenMaskUbPre, srcTensor, attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize,
                    runInfo.s2RealSize, attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo);
                attenMaskInQuePre.template EnQue(attenMaskUbPre);
                attenMaskInQuePre.template DeQue<uint8_t>();
                attenMaskInQue.template DeQue<uint8_t>();
                MergePrefixModeMask<hasAtten>(attenMaskUbPre, attenMaskUb, runInfo.halfS1RealSize,
                                              constInfo.s2BaseSize);
                attenMaskInQuePre.template FreeTensor(attenMaskUbPre);
                attenMaskInQue.template EnQue(attenMaskUb);
            }
            return;
        }
        BoolCopyInRegbase<isInfer>(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
            attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo);
        attenMaskInQue.template EnQue(attenMaskUb);
        if (((attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) ||
            (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE) &&
                runInfo.boIdx == attenMaskInfo.bandIndex) ||
            (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE) &&
             runInfo.boIdx == attenMaskInfo.bandIndex)) &&
            (attenMaskInfo.computeMode == AttenMaskComputeMode::PRE_ONLY_MODE ||
             attenMaskInfo.computeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE)) {
            LocalTensor<uint8_t> attenMaskUbPre = attenMaskInQuePre.template AllocTensor<uint8_t>();
            BoolCopyInRegbase<isInfer>(attenMaskUbPre, srcTensor, attenMaskInfo.attenMaskOffsetPre, runInfo.halfS1RealSize,
                runInfo.s2RealSize, attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo);
            attenMaskInQuePre.template EnQue(attenMaskUbPre);
            attenMaskInQuePre.template DeQue<uint8_t>();
            attenMaskInQue.template DeQue<uint8_t>();
            MergeBandModeMask<hasAtten>(attenMaskUbPre, attenMaskUb, runInfo.halfS1RealSize, constInfo.s2BaseSize);
            attenMaskInQuePre.template FreeTensor(attenMaskUbPre);
            attenMaskInQue.template EnQue(attenMaskUb);
        }
        return;
    }
}

template <bool hasAtten, bool hasRope = false, bool isInfer = false, bool isMxfp8FullQuant = false,
          bool optionalDn = false>
__aicore__ inline void AttenMaskCopyInDn(TQue<QuePosition::VECIN, 1> &attenMaskInQue,
                                         GlobalTensor<uint8_t> &srcTensor,
                                         RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                         AttenMaskInfo &attenMaskInfo, bool needAtten, const int32_t subLoop = 0)
{
    if constexpr (hasAtten) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        if constexpr (optionalDn) {
            if (needAtten) {
                int64_t maskOffset = ComputeAttenMaskOffset<hasAtten, false, false, false, isInfer, DTemplateType::Aligned128, isMxfp8FullQuant>(runInfo, constInfo, attenMaskInfo, true, subLoop);
                if (attenMaskInfo.computeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
                    BoolCopyInRegbase<isInfer>(attenMaskUb, srcTensor, maskOffset, constInfo.s2BaseSize,
                                               runInfo.s1RealSizeAlign64 >> 1, attenMaskInfo.attenMaskS2Size,
                                               runInfo.s1RealSizeAlign64 >> 1, constInfo);
                }
            }
        } else {
            int64_t maskOffset = ComputeAttenMaskOffset<hasAtten, false, false, false, isInfer, DTemplateType::Aligned128, isMxfp8FullQuant>(runInfo, constInfo, attenMaskInfo, true, subLoop);
            if (attenMaskInfo.computeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
                int32_t s2RealSize = runInfo.s2RealSize;
                if constexpr (isMxfp8FullQuant) {
                    if (runInfo.s2RealSize > 256) {
                        s2RealSize = subLoop == 0 ? 256 : runInfo.s2RealSize - 256;
                    }
                }
                BoolCopyInRegbase<isInfer>(attenMaskUb, srcTensor, maskOffset, s2RealSize,
                                        constInfo.s1BaseSize >> 1, attenMaskInfo.attenMaskS2Size, constInfo.s1BaseSize >> 1, constInfo);
            }
        }
        attenMaskInQue.template EnQue(attenMaskUb);
    }
}

}

#endif // ATTENMASK_H
