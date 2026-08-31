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
 *        分发 FlashAttentionNoQuantMlaKernel（由 fia_kernel_noquant_mla.h 复制改名）。
 *        结构完全对齐 flash_attn.cpp：单层 __global__，四模板参数
 *        (InOutLayoutType, KvLayoutType, HasAttenMask, Config) 直接解析 q 布局与
 *        模板块大小，无中间分发层；无 isFd（FD 能力恒实例化，运行时由 metadata
 *        各 section mLen>0 驱动）。q 仅支持 BSND/BNSD/TND 三布局，不支持输出转置
 *        （out 布局恒等于 q）；rope 已并入 q/k_cache 576 宽单张量，无独立 rope 输入。
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"

#include "arch35/flash_mla_with_kvcache_template_tiling_key.h"
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#else
#include "../common/arch35/flash_attention_score_common_regbase_arch35.h"
#endif
#include "adv_api/activation/softmax.h"
#if __has_include("../../common/op_kernel/arch35/flash_attention_score_tiling_regbase_arch35.h")
#include "../../common/op_kernel/arch35/flash_attention_score_tiling_regbase_arch35.h"
#else
#include "../common/arch35/flash_attention_score_tiling_regbase_arch35.h"
#endif
#include "arch35/flash_mla_with_kvcache_type.h"
using namespace optiling; // tiling_regbase 提供；kernel 头 static_assert 需在解析期可见 optiling 域（与 FA 入口同款）
#include "arch35/flash_mla_with_kvcache_kernel_noquant_mla.h"

using namespace AscendC;

// ============ 入口 layout 推导（flash_attn.cpp 同款：if-constexpr 直接映射，无表查找）============
// InOutLayoutType → q 布局（MLA 不支持输出转置，out 布局恒等于 q，无需 GetOutLayout）
template <uint8_t inOutLayoutType>
__aicore__ inline constexpr LayOutTypeEnum GetQueryLayoutMla()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_TND),
                  "GetQueryLayoutMla fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND) {
        return LayOutTypeEnum::LAYOUT_BSH;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD) {
        return LayOutTypeEnum::LAYOUT_BNSD;
    } else { // InOutLayoutType_TND
        return LayOutTypeEnum::LAYOUT_TND;
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
                                                  __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse,
                                                  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashMlaWithKvcacheTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // DT_BF16/DT_FLOAT16 为 device 头树 basic_api/kernel_type.h 定义的预处理宏
    // （kernel_operator.h 传递 include），直接宏比较对齐 flash_attn.cpp；编译宏名
    // ORIG_DTYPE_{Q,K_CACHE,ATTN_OUT} 由 asc_opc 按输入名派生（fia 内部名
    // ORIG_DTYPE_QUERY/KEY/ATTENTION_OUT 未被定义 → 恒走 bf16 分支，勿用）。
#if (ORIG_DTYPE_Q == DT_BF16)
    using INPUT_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    using INPUT_T = half;
    using OUT_T = half;
#endif

    fa_base_matmul::idCounterNum = 0;
    // —— 编译期解析 tiling key（flash_attn.cpp 同款：ConfigValue[config] 查模板块大小）——
    constexpr LayOutTypeEnum qLayout = GetQueryLayoutMla<inOutLayoutType>();
    constexpr LayOutTypeEnum outLayout = qLayout; // MLA 不支持输出转置
    constexpr GmFormat kvGmFormat = GetKvLayoutMla<inOutLayoutType, KvLayoutType>();
    constexpr bool pageAttention = IsPageAttentionMla<KvLayoutType>();
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    // 根因（静态图）：固定 shape 下 tiling 是编译期常量字节数组而非 __gm__ buffer，
    // 不能直接 reinterpret_cast 成结构体指针访问，必须先拷贝到栈局部结构体对象再取指针。
    GET_TILING_DATA_MEMBER(optiling::FlashMlaWithKvcacheTilingData, baseTiling, baseTilingIn, tiling);
    const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;

    // —— 模板参数聚合（FAType 式，镜像 flash_attn utils/flash_attn_type.h；
    // cube/vec/fd 参数集不一致 → 各建聚合类型，见 arch35/flash_mla_with_kvcache_type.h）——
    using FA_T_Cube = FlashAttnKernel::FATypeCube<INPUT_T, float, qLayout, s1TemplateType, s2TemplateType, dTemplateType,
                                          dVTemplateType, KvLayoutType>;
    using FA_T_Vec = FlashAttnKernel::FATypeVec<INPUT_T, float, OUT_T, qLayout, outLayout, s1TemplateType, s2TemplateType,
                                        dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;
    // FD block 参数聚合（无 flashDecode：FD 恒实例化，运行时由 metadata mLen>0 驱动）
    using FA_T_Fd = FlashAttnKernel::FATypeFd<INPUT_T, float, OUT_T, qLayout, outLayout, s1TemplateType, s2TemplateType,
                                      dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;

    // 编译期 core-type 选型（与 flash_attn.cpp 的 #ifdef __DAV_C310_CUBE__ 同构）：该宏由 950
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
    // 静态 tensor 模型：block 内自管 buffer（与 flash_attn Init 15 参风格一致，入口仅接地址）
    op.Init(query, kCache, attnMask, cuSeqlensQ, sequsedQ, cacheSeqlens, blockTable, softmaxLse, attnOut, user,
            metadata, tilingData);
    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
}
