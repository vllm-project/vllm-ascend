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
 * \file flash_mla_with_kvcache.cpp
 * \brief flash_mla_with_kvcache Kernel 唯一入口；按 tiling key 编译期推导模板参数，
 *        分发 FlashAttentionNoQuantMlaKernel。单层 __global__，四模板参数
 *        (InOutLayoutType, KvLayoutType, HasAttenMask, Config) 直接解析 q 布局与
 *        模板块大小，无中间分发层；无 isFd（FD 能力恒实例化，运行时由 metadata
 *        各 section mLen>0 驱动）。q 支持 BSND/BNSD/TND 三布局，TND→NTD 输出转置
 *        （InOutLayoutType_TND_NTD）；rope 已并入 q/k_cache 576 宽单张量，无独立 rope 输入。
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"

#include "arch35/flash_mla_with_kvcache_template_tiling_key.h"
#include "../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "adv_api/activation/softmax.h"
#if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN)
#include "../../a5_mla_common/op_kernel/arch35/c8_pipeline/flash_mla_c8_kernel.h"
#else
#include "utils/flash_mla_with_kvcache_type.h"
using namespace optiling;
#include "arch35/flash_mla_with_kvcache_kernel.h"
#endif

using namespace AscendC;

#if (ORIG_DTYPE_Q != DT_FLOAT8_E4M3FN)
// ============ 入口 layout 推导（if-constexpr 直接映射，无表查找）============
// InOutLayoutType → q 布局（TND_NTD 输入仍为 TND q 布局）
template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT GetQueryLayoutMla()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_TND) || (inOutLayoutType == InOutLayoutType_TND_NTD),
                  "GetQueryLayoutMla fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND) {
        return FLASH_MLA_WITH_KVCACHE_LAYOUT::BSH;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD) {
        return FLASH_MLA_WITH_KVCACHE_LAYOUT::BNSD;
    } else { // InOutLayoutType_TND / InOutLayoutType_TND_NTD
        return FLASH_MLA_WITH_KVCACHE_LAYOUT::TND;
    }
}

// InOutLayoutType → out 布局（TND_NTD 转置输出为 NTD，其余 out 恒等于 q）
template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT GetOutLayoutMla()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_TND) || (inOutLayoutType == InOutLayoutType_TND_NTD),
                  "GetOutLayoutMla fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_TND_NTD) {
        return FLASH_MLA_WITH_KVCACHE_LAYOUT::NTD;
    } else {
        return GetQueryLayoutMla<inOutLayoutType>();
    }
}

// KvLayoutType → 是否 PA 分页（NO_PA=0 不实例化）
template <uint8_t KvLayoutType>
__aicore__ inline constexpr bool IsPageAttentionMla()
{
    static_assert(KvLayoutType <= KvLayoutType_PA_NZ, "IsPageAttentionMla fail, KvLayoutType is incorrect");
    return (KvLayoutType != KvLayoutType_NO_PA);
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType>
__aicore__ inline constexpr GmFormat GetKvLayoutMla()
{
    return FlashAttnKernel::GetKVGmFormat<GetQueryLayoutMla<inOutLayoutType>(), KvLayoutType,
                                          IsPageAttentionMla<KvLayoutType>()>();
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
// 形参序必须吃齐 def 输入序 cache_seqlens(3)/cu_seqlens_q(4)/seqused_q(5)（运行时按 def 序绑槽位；
// 曾用 FIA 内部序致 cacheSeqlens 收到 seqused_q=NULL 槽 → kv 长度解析解引用 GM 0 越界）。
__global__ __aicore__ void flash_mla_with_kvcache(__gm__ uint8_t *query, __gm__ uint8_t *kCache,
                                                  __gm__ uint8_t *blockTable, __gm__ uint8_t *cacheSeqlens,
                                                  __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                  __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
                                                  __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                                  __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                                  __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse,
                                                  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashMlaWithKvcacheTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // DT_BF16/DT_FLOAT16 为 device 头树 basic_api/kernel_type.h 定义的预处理宏
    // （kernel_operator.h 传递 include）；编译宏名
    // ORIG_DTYPE_{Q,K_CACHE,ATTN_OUT} 由 asc_opc 按输入名派生（fia 内部名
    // ORIG_DTYPE_QUERY/KEY/ATTENTION_OUT 未被定义 → 恒走 bf16 分支，勿用）。
#if (ORIG_DTYPE_Q == DT_BF16)
    using INPUT_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    using INPUT_T = half;
    using OUT_T = half;
#endif

    // —— 编译期解析 tiling key（ConfigValue[config] 查模板块大小）——
    constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT qLayout = GetQueryLayoutMla<inOutLayoutType>();
    constexpr FLASH_MLA_WITH_KVCACHE_LAYOUT outLayout = GetOutLayoutMla<inOutLayoutType>();
    constexpr GmFormat kvGmFormat = GetKvLayoutMla<inOutLayoutType, KvLayoutType>();
    constexpr bool pageAttention = IsPageAttentionMla<KvLayoutType>();
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(96);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(112);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    // 根因（静态图）：固定 shape 下 tiling 是编译期常量字节数组而非 __gm__ buffer，
    // 不能直接 reinterpret_cast 成结构体指针访问，必须先拷贝到栈局部结构体对象再取指针。
    GET_TILING_DATA_MEMBER(optiling::FlashMlaWithKvcacheTilingData, baseTiling, baseTilingIn, tiling);
    const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;

    // —— 模板参数聚合（cube/vec/fd 参数集不一致 → 各建聚合类型，见 utils/flash_mla_with_kvcache_type.h）——
    using FA_T_Cube = FlashAttnKernel::FATypeCube<INPUT_T, float, qLayout, s1TemplateType, s2TemplateType,
                                                  dTemplateType, dVTemplateType, KvLayoutType>;
    using FA_T_Vec =
        FlashAttnKernel::FATypeVec<INPUT_T, float, OUT_T, qLayout, outLayout, s1TemplateType, s2TemplateType,
                                   dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;
    // FD block 参数聚合（无 flashDecode：FD 恒实例化，运行时由 metadata mLen>0 驱动）
    using FA_T_Fd = FlashAttnKernel::FATypeFd<INPUT_T, float, OUT_T, qLayout, outLayout, s1TemplateType, s2TemplateType,
                                              dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;

    // 编译期 core-type 选型：#ifdef __DAV_C310_CUBE__ 由 950
    // 编译在 AIC TU 定义、AIV TU 不定义（mla 编译产物实测确认），语义与
    // g_coreType==AscendC::AIC/AIV 等价 —— AIC 核取 cube 实块 + vec 哑块（FD 槽复用哑块），
    // AIV 核反之取 vec 实块 + cube 哑块。
#ifdef __DAV_C310_CUBE__
    using CubeBlock = FlashAttnKernel::FlashMlaWithKvcacheNoQuantMlaBlockCube<FA_T_Cube>;
    using VecFaBlock = FlashAttnKernel::FlashMlaWithKvcacheNoQuantMlaBlockVecDummy<FA_T_Vec>;
    using VecFdBlock = FlashAttnKernel::FlashMlaWithKvcacheNoQuantMlaBlockVecDummy<FA_T_Vec>;
#else
    using CubeBlock = FlashAttnKernel::FlashMlaWithKvcacheNoQuantMlaBlockCubeDummy<FA_T_Cube>;
    using VecFaBlock = FlashAttnKernel::FlashMlaWithKvcacheNoQuantMlaBlockVec<FA_T_Vec>;
    using VecFdBlock = FlashAttnKernel::FlashMlaWithKvcacheBlockVecFlashDecodeMla<FA_T_Fd>;
#endif
    using Kernel = FlashAttnKernel::FlashAttentionNoQuantMlaKernel<CubeBlock, VecFaBlock, VecFdBlock>;

    Kernel op;
    // 无独立 queryRope/keyRope；q/k_cache 为 576 宽单张量（nope512+rope64 合并）。
    // value 与 key 同一 GM 指针（k_cache 的 nope 段即 V，k==v 语义）。
    // metadata（GM 输入，AICPU 多 section 布局）直接透传，section 循环在 Process 内。
    // 静态 tensor 模型：block 内自管 buffer，入口仅接地址
    op.Init(query, kCache, attnMask, cuSeqlensQ, sequsedQ, cacheSeqlens, blockTable, softmaxLse, attnOut, user,
            metadata, tilingData);
    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
}

#else
template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config, FlashMlaC8Layout outputLayout>
__aicore__ inline void RunFlashMlaC8(
    __gm__ uint8_t *query, __gm__ uint8_t *kCache, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *cacheSeqlens, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *tilingData)
{
    using namespace BaseApi;
    constexpr auto inputLayout = FlashMlaC8Layout::LAYOUT_TND;
    constexpr auto m = S1TemplateType::Aligned64;
    constexpr auto s = S2TemplateType::Aligned128;
    constexpr auto d = DTemplateType::Aligned576;
    constexpr auto dv = DTemplateType::Aligned512;
    constexpr auto pse = PseTypeEnum::PSE_NONE_TYPE;
    using CubeNormal = FAFullQuantMlaBlockCube<fp8_e4m3fn_t, float, inputLayout, m, s, d, dv,
                                              true, KvLayoutType, false, false, true, false>;
    using CubeDummy = FAFullQuantMlaBlockCubeDummy<fp8_e4m3fn_t, float, inputLayout, m, s, d, dv,
                                                  true, KvLayoutType, false, false, true, false>;
    using VecNormal = FAFullQuantMlaBlockVec<fp8_e4m3fn_t, float, bfloat16_t, inputLayout, outputLayout,
                                            m, s, d, dv, pse, hasAttenMask, false, true, KvLayoutType,
                                            true, false, false, true, false>;
    using VecDummy = FAFullQuantMlaBlockVecDummy<fp8_e4m3fn_t, float, bfloat16_t, inputLayout, outputLayout,
                                                m, s, d, dv, pse, hasAttenMask, false, true, KvLayoutType,
                                                true, false, false, true, false>;
    using FdNormal = FiaBlockVecFlashDecodeFullQuant<fp8_e4m3fn_t, float, bfloat16_t, inputLayout, outputLayout,
                                                    m, s, d, dv, pse, hasAttenMask, false, true, KvLayoutType,
                                                    false, false, true, false>;
    using FdDummy = FiaBlockVecFlashDecodeFullQuantDummy<fp8_e4m3fn_t, float, bfloat16_t, inputLayout, outputLayout,
                                                       m, s, d, dv, pse, hasAttenMask, false, true, KvLayoutType,
                                                       false, false, true, false>;
#ifdef __DAV_C310_CUBE__
    using Kernel = FlashAttentionFullQuantMlaKernel<CubeNormal, VecDummy, FdDummy>;
#else
    using Kernel = FlashAttentionFullQuantMlaKernel<CubeDummy, VecNormal, FdNormal>;
#endif
    const uint32_t sectionCount = ((__gm__ uint32_t *)metadata)[0];
    if constexpr (outputLayout == FlashMlaC8Layout::LAYOUT_NTD_DCP) {
        if (sectionCount == 0) {
            if ASCEND_IS_AIV {
                const auto &base = tilingData->flashMlaWithKvcacheBaseParams;
                AttentionCommon::ConstInfo_t<AttentionCommon::FiaKernelType::FULL_QUANT> info{};
                info.t1Size = base.t1Size;
                info.realN2Size = base.n2Size;
                info.realGSize = base.gSize;
                info.coreNum = base.coreNum;
                info.aivIdx = GetBlockIdx();
                GlobalTensor<bfloat16_t> output;
                GlobalTensor<float> wire, lse;
                output.SetGlobalBuffer((__gm__ bfloat16_t *)attnOut);
                wire.SetGlobalBuffer((__gm__ float *)attnOut);
                lse.SetGlobalBuffer((__gm__ float *)softmaxLse);
                InitDcpOutput<bfloat16_t, 0>(output, wire, lse, info);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    // A section owns its buffer-ring state. Drain it before FD reuses the UB,
    // then recreate the buffer managers for the following metadata section.
    for (uint32_t section = 0; section < sectionCount; ++section) {
        fa_base_matmul::ResetIdCounter();
        TPipe pipe;
        Kernel op;
        op.Init(query, kCache, kCache, attnMask, cuSeqlensQ, cacheSeqlens, blockTable,
                dequantScaleQuery, dequantScaleKey, dequantScaleKey, queryRope, keyRope,
                softmaxLse, attnOut, GetUserWorkspace(workspace), metadata, tilingData,
                &pipe, sequsedQ, section);
        op.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
__global__ __aicore__ void flash_mla_with_kvcache(
    __gm__ uint8_t *query, __gm__ uint8_t *kCache, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *cacheSeqlens, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
    __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashMlaWithKvcacheTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_MEMBER(optiling::FlashMlaWithKvcacheTilingData, baseTiling, baseTilingIn, tiling);
    // Retain the public TND_NTD tiling key; only C8's final GM writer differs.
    if constexpr (inOutLayoutType == InOutLayoutType_TND_NTD) {
        if (baseTilingIn.flashMlaWithKvcacheBaseParams.outputLayout ==
            static_cast<uint32_t>(FlashMlaC8Layout::LAYOUT_NTD_DCP)) {
            RunFlashMlaC8<inOutLayoutType, KvLayoutType, hasAttenMask, config,
                          FlashMlaC8Layout::LAYOUT_NTD_DCP>(query, kCache, blockTable, cacheSeqlens, cuSeqlensQ, sequsedQ, attnMask, metadata,
            queryRope, keyRope, dequantScaleQuery, dequantScaleKey, attnOut, softmaxLse, workspace, &baseTilingIn);
            return;
        }
    }
    constexpr auto outputLayout = inOutLayoutType == InOutLayoutType_TND_NTD
        ? FlashMlaC8Layout::LAYOUT_NTD : FlashMlaC8Layout::LAYOUT_TND;
    RunFlashMlaC8<inOutLayoutType, KvLayoutType, hasAttenMask, config, outputLayout>(query, kCache, blockTable, cacheSeqlens, cuSeqlensQ, sequsedQ, attnMask, metadata,
            queryRope, keyRope, dequantScaleQuery, dequantScaleKey, attnOut, softmaxLse, workspace, &baseTilingIn);
}
#endif
