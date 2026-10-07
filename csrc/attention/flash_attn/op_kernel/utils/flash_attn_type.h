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
 * \file flash_attn_type.h
 * \brief flash_attn 核心类型定义: FAType / 布局格式映射 / SeqLensTool / ClipSInnerToken
 */
#ifndef FLASH_ATTN_TYPE_H_
#define FLASH_ATTN_TYPE_H_

#include "flash_attn_common_def.h"
#include "../arch35/memory_copy_arch35.h"
#include "../../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;

namespace FlashAttnKernel {

template <FA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    static_assert((LAYOUT_T == FA_LAYOUT::BSND) || (LAYOUT_T == FA_LAYOUT::BNSD) || (LAYOUT_T == FA_LAYOUT::TND),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
        return GmFormat::TNGD;
    }
}

template <FA_LAYOUT LAYOUT_KV, bool PAGE_ATTENTION>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (PAGE_ATTENTION) {
        static_assert((LAYOUT_KV == FA_LAYOUT::BSND) || (LAYOUT_KV == FA_LAYOUT::BNSD) || (LAYOUT_KV == FA_LAYOUT::NZ),
                      "Get Key or Value GmFormat fail, LAYOUT_KV is incorrect when PageAttention");
        if constexpr (LAYOUT_KV == FA_LAYOUT::BSND) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (LAYOUT_KV == FA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        } else if constexpr (LAYOUT_KV == FA_LAYOUT::NZ) {
            return GmFormat::PA_NZ;
        }
    } else {
        static_assert((LAYOUT_KV == FA_LAYOUT::BSND) || (LAYOUT_KV == FA_LAYOUT::BNSD) || (LAYOUT_KV == FA_LAYOUT::TND),
                      "Get Key or Value GmFormat fail, LAYOUT_KV is incorrect when KV Continuous");
        if constexpr (LAYOUT_KV == FA_LAYOUT::BSND) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT_KV == FA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT_KV == FA_LAYOUT::TND) {
            return GmFormat::TND;
        }
    }
}

template <FA_LAYOUT LAYOUT>
__aicore__ inline constexpr bool IS_TND()
{
    return (LAYOUT == FA_LAYOUT::TND);
}

template <FA_LAYOUT LAYOUT_OUT>
__aicore__ inline constexpr GmFormat GetAttentionOutGmFormat()
{
    static_assert((LAYOUT_OUT == FA_LAYOUT::BSND) || (LAYOUT_OUT == FA_LAYOUT::BNSD) || (LAYOUT_OUT == FA_LAYOUT::TND),
                  "Get OUT GmFormat fail, LAYOUT_OUT is incorrect");
    if constexpr (LAYOUT_OUT == FA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_OUT == FA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_OUT == FA_LAYOUT::TND) {
        return GmFormat::TNGD;
    }
}

template <FA_LAYOUT LAYOUT_OUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT_OUT == FA_LAYOUT::BNSD) || (LAYOUT_OUT == FA_LAYOUT::BSND) || (LAYOUT_OUT == FA_LAYOUT::TND),
                  "Get OutAttention UB GmFormat fail, LAYOUT_OUT is incorrect");
    if constexpr (LAYOUT_OUT == FA_LAYOUT::BSND || LAYOUT_OUT == FA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_OUT == FA_LAYOUT::BNSD) {
        return UbFormat::GS1;
    }
}

template <typename INPUT_T, typename OUTPUT_T, bool PAGE_ATTENTION, FA_LAYOUT LAYOUT_T, FA_LAYOUT LAYOUT_KV,
          FA_LAYOUT LAYOUT_OUT, regbaseutil::S1TemplateType s1TemplateType = regbaseutil::S1TemplateType::Aligned128,
          regbaseutil::S2TemplateType s2TemplateType = regbaseutil::S2TemplateType::Aligned128,
          regbaseutil::DTemplateType dTemplateType = regbaseutil::DTemplateType::Aligned128,
          regbaseutil::DTemplateType dVTemplateType = regbaseutil::DTemplateType::Aligned128, bool HAS_MASK = false,
          typename... Args>
struct FAType {
    using inputType = INPUT_T;
    using outputType = OUTPUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr FA_LAYOUT qLayout = LAYOUT_T;
    static constexpr FA_LAYOUT kvLayout = LAYOUT_KV;
    static constexpr FA_LAYOUT attnOutLayout = LAYOUT_OUT;
    static constexpr regbaseutil::S1TemplateType mBaseSize = s1TemplateType;
    static constexpr regbaseutil::S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr regbaseutil::DTemplateType dBaseSize = dTemplateType;
    static constexpr regbaseutil::DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
};

template <FA_LAYOUT LAYOUT_T, typename SEQLEN_T>
class SeqLensTool {
public:
    ActualSeqLensParser<ActualSeqLensMode::ACCUM, SEQLEN_T, true> cuSeqLensParser;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, SEQLEN_T, false> seqUsedParser;

    __aicore__ inline void Init(__gm__ uint8_t *cuSeqLensGmAddr, uint32_t cuSeqLensDims, __gm__ uint8_t *seqUsedGmAddr,
                                uint32_t seqUsedDims, uint64_t defaultSeqUsedVal)
    {
        cuSeqLensParser.Init(cuSeqLensGmAddr, cuSeqLensDims, seqUsedGmAddr, seqUsedDims);
        seqUsedParser.Init(seqUsedGmAddr, seqUsedDims, defaultSeqUsedVal);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx)
    {
        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            return cuSeqLensParser.GetActualSeqLength(bIdx);
        } else {
            return seqUsedParser.GetActualSeqLength(bIdx);
        }
    }
};

__aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

} // namespace FlashAttnKernel

#endif // FLASH_ATTN_TYPE_H_
