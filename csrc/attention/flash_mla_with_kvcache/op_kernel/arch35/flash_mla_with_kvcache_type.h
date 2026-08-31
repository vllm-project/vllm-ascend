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
 * \file flash_mla_with_kvcache_type.h
 * \brief arch35 flash_mla_with_kvcache 编译期模板参数聚合类型（镜像 flash_attn 的
 *        utils/flash_attn_type.h FAType 组织模式，语义命名保留 MLA 语境）。
 *        cube/vec/fd 三类 block 参数集不一致（vec 多 OUTPUT_T/outLayout/hasMask，
 *        cube 无 mask/out 概念），故分别为每类建聚合类型；无 isFd——FD block 与 vec
 *        块的 FD 能力均恒实例化（FD 运行时由 metadata mLen>0 决定），FATypeFd 不含
 *        flashDecode 成员，FATypeVec::flashDecode 恒 true。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TYPE_H_
#define FLASH_MLA_WITH_KVCACHE_TYPE_H_

#if __has_include("../../../common/op_kernel/arch35/util_regbase.h")
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/gm_layout.h"
#include "../../../common/op_kernel/memcopy/fa_ub_tensor.h"
#else
#include "../../common/arch35/util_regbase.h"
#include "../../common/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/memcopy/parser.h"
#include "../../common/memcopy/gm_layout.h"
#include "../../common/memcopy/fa_ub_tensor.h"
#endif

namespace FlashAttnKernel {

// ---------------------------------------------------------------------------
// 布局/格式推导 helper（源自 fork common/op_kernel/memcopy/{gm_format,mem_format}.h，
// 上游已删除上述共享头并改为"每算子族本地携带"惯例——参考 flash_attn_type.h:33/49/101、
// fused_infer_attention_score arch35 memory_copy_arch35_fused_infer.h:28/38/134）
// ---------------------------------------------------------------------------
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return UbFormat::GS1;
    }
}

// Cube block 参数聚合（8 参：INPUT_T、T、layout、s1、s2、d、dv、KvLayoutType）。
// PAGE_ATTENTION 派生规则与 block 内 (KvLayoutType > 0) 一致（四件套 KV_FORMAT 同源）。
template <typename INPUT_T, typename T, LayOutTypeEnum LAYOUT = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
struct FATypeCube {
    using inputType = INPUT_T;
    using mmType = T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
};

// Vec block 参数聚合（11 参：INPUT_T、T、OUTPUT_T、layout、outLayout、s1、s2、d、dv、
// hasMask、KvLayoutType）。flashDecode 成员恒 true（FD 能力恒实例化，对齐 flash_attn 的
// FAType——FA 亦无 FLASH_DECODE 参数；FD 是否执行由运行时 metadata mLen>0 决定）。
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum LAYOUT = LayOutTypeEnum::None,
          LayOutTypeEnum OUT_LAYOUT = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool HAS_MASK = false, uint8_t KvLayoutType = 0>
struct FATypeVec {
    using inputType = INPUT_T;
    using mmType = T;
    using outputType = OUTPUT_T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT;
    static constexpr LayOutTypeEnum attnOutLayout = OUT_LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
    static constexpr bool flashDecode = true; // FD 恒实例化，运行时由 metadata 决定
};

// FD (flashdecode) block 参数聚合（11 参，无 flashDecode：FD 由 metadata mLen>0 运行时使能，
// FD 能力恒实例化、无编译期开关）。
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum LAYOUT = LayOutTypeEnum::None,
          LayOutTypeEnum OUT_LAYOUT = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool HAS_MASK = false, uint8_t KvLayoutType = 0>
struct FATypeFd {
    using inputType = INPUT_T;
    using mmType = T;
    using outputType = OUTPUT_T;
    static constexpr LayOutTypeEnum qLayout = LAYOUT;
    static constexpr LayOutTypeEnum attnOutLayout = OUT_LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
};

// kernel 持有的 seq-lens 解析工具（镜像 flash_attn utils/flash_attn_type.h:134-155 SeqLensTool
// 的"kernel 持有工具、block 构造收引用"组织方式）。与 FA 的关键差异（Oracle 约束）：
// 1) 保留 q 侧 seqused_q==nullptr 双分支——FA 的 SeqLensTool 无 null-seqused 概念，直接移植
//    会错误合并空指针分支并改变 seqused=NULL/0 槽位边角语义；
// 2) q 侧 ACCUM 带首零头（WITH_ZERO_HEAD=true，flash_attn cu_seqlens 约定）；
// 3) kv 侧 BY_BATCH（cache_seqlens 非累加）。
// seq-lens 为 INT32 接口，ACTLEN_T=uint32_t；cu_seqlens_q [b+1] / seqused_q [b] / cache_seqlens [b]。
template <ActualSeqLensMode Q_MODE, ActualSeqLensMode KV_MODE>
class FlashMlaSeqLensTool {
public:
    using SEQLEN_T = uint32_t;

    // cu_seqlens_q / cache_seqlens 的 GM 张量（offsetCalculator 直接消费；kernel Init 内绑定
    // 后各 block 只读引用，替代 block 各自持副本）
    GlobalTensor<SEQLEN_T> actualSeqLengthsGmQ;
    GlobalTensor<SEQLEN_T> actualSeqLengthsGm;

    // q 侧 parser 的 WITH_ZERO_HEAD 仅对 ACCUM（cu_seqlens，TND）语义成立；BY_BATCH（BSND/BNSD
    // 逐 batch 定长）无首零头——common/op_kernel/memcopy/parser.h 仅实例化 BY_BATCH,false。
    static constexpr bool Q_WITH_ZERO_HEAD = (Q_MODE == ActualSeqLensMode::ACCUM);
    ActualSeqLensParser<Q_MODE, SEQLEN_T, Q_WITH_ZERO_HEAD> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE, SEQLEN_T, false> kvActSeqLensParser;

    __aicore__ inline void InitQ(__gm__ uint8_t *cuSeqlensQAddr, __gm__ uint8_t *sequsedQAddr,
                                 uint32_t actualSeqLenSize, uint64_t s1Size)
    {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ SEQLEN_T *)cuSeqlensQAddr, actualSeqLenSize);
        if (sequsedQAddr == nullptr) {
            // 双分支 1（保留，不压平）：无 seqused_q → cu_seqlens_q 差分语义（含首零头累加）
            qActSeqLensParser.Init(actualSeqLengthsGmQ, actualSeqLenSize, s1Size);
        } else {
            // 双分支 2：有 seqused_q → 按逐 batch 实际长度取值（seqused 长度 = b = actualSeqLenSize - 1）；
            // BY_BATCH（BSND/BNSD）无 cu_seqlens、无 ACCUM 四参 Init，按 seqused 逐 batch 定长解析
            if constexpr (Q_WITH_ZERO_HEAD) {
                qActSeqLensParser.Init((__gm__ uint8_t *)cuSeqlensQAddr, actualSeqLenSize,
                                       (__gm__ uint8_t *)sequsedQAddr, actualSeqLenSize - 1);
            } else {
                qActSeqLensParser.Init((__gm__ uint8_t *)sequsedQAddr, actualSeqLenSize - 1, s1Size);
            }
        }
    }

    __aicore__ inline void InitKv(__gm__ uint8_t *cacheSeqlensAddr, uint32_t actualSeqLenKVSize, uint64_t s2Size)
    {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ SEQLEN_T *)cacheSeqlensAddr, actualSeqLenKVSize);
        kvActSeqLensParser.Init(actualSeqLengthsGm, actualSeqLenKVSize, s2Size);
    }
};

} // namespace FlashAttnKernel

#endif // FLASH_MLA_WITH_KVCACHE_TYPE_H_
