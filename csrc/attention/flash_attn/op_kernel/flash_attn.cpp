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
 * \file flash_attn.cpp
 * \brief FlashAttn Kernel 唯一入口；按 tiling key 编译期推导模板参数，分发 Dn / Nd 两套模板。
 *        参考 quant_flash_attn.cpp 单层 __global__ 风格。
 */

#include "arch35/flash_attn_template_tiling_key.h"
#include "utils/flash_attn_common_def.h"
#include "../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "arch35/flash_attn_kernel_dn.h"
#include "arch35/flash_attn_kernel_nd.h"

using namespace AscendC;

template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FA_LAYOUT GetQueryLayout()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_BNSD_BSND) || (inOutLayoutType == InOutLayoutType_TND),
                  "Get Query Layout fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND) {
        return FA_LAYOUT::BSND;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD || inOutLayoutType == InOutLayoutType_BNSD_BSND) {
        return FA_LAYOUT::BNSD;
    } else if constexpr (inOutLayoutType == InOutLayoutType_TND) {
        return FA_LAYOUT::TND;
    }
}

template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FA_LAYOUT GetOutLayout()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_BNSD_BSND) || (inOutLayoutType == InOutLayoutType_TND),
                  "Get AttnOut Layout fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND || inOutLayoutType == InOutLayoutType_BNSD_BSND) {
        return FA_LAYOUT::BSND;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD) {
        return FA_LAYOUT::BNSD;
    } else if constexpr (inOutLayoutType == InOutLayoutType_TND) {
        return FA_LAYOUT::TND;
    }
}

template <uint8_t inOutLayoutType, uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr FA_LAYOUT GetKvLayout()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                      (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_NZ),
                  "Get Key/Value Layout fail, KV_STORAGE_MODE is incorrect");
    if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) {
        return GetQueryLayout<inOutLayoutType>();
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) {
        return FA_LAYOUT::BSND; // block内的格式类似于BSND
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) {
        return FA_LAYOUT::BNSD; // block内的格式类似于BNSD
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_NZ) {
        return FA_LAYOUT::NZ; // block内的格式类似于BNSD
    }
}

template <uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr bool IsPageAttention()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                      (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_NZ),
                  "Get PAGE_ATTENTION flag fail, KV_STORAGE_MODE is incorrect");
    return (KV_STORAGE_MODE != KV_STORAGE_MODE_CONTINUE);
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t templateId, uint8_t config>
__global__ __aicore__ void flash_attn(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                      __gm__ uint8_t *blockTable, __gm__ uint8_t *cuSeqLensQ,
                                      __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
                                      __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata,
                                      __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                      __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttnTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (inOutLayoutType == InOutLayoutType_TND && KvLayoutType == KvLayoutType_NO_PA) {
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(cuSeqLensQ), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(cuSeqLensKv), 0);
    }

#if (ORIG_DTYPE_Q == DT_BF16)
    using INPUT_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    using INPUT_T = half;
    using OUT_T = half;
#endif

#if (ORIG_DTYPE_Q == DT_BF16) || (ORIG_DTYPE_Q == DT_FLOAT16)
    fa_base_matmul::ResetIdCounter();

    constexpr FA_LAYOUT qLayout = GetQueryLayout<inOutLayoutType>();
    constexpr FA_LAYOUT outLayout = GetOutLayout<inOutLayoutType>();
    constexpr FA_LAYOUT kvLayout = GetKvLayout<inOutLayoutType, KvLayoutType>();
    constexpr bool pageAttention = IsPageAttention<KvLayoutType>();

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    // 根因（静态图）：固定 shape 下 tiling 是编译期常量字节数组而非 __gm__ buffer，
    // 不能直接 reinterpret_cast 成结构体指针访问，必须先拷贝到栈局部结构体对象再取指针。
    // 该宏由编译器(generate_tiling_code.py)按动态/静态图自动生成两套展开：
    //   动态图 → (__gm__ FlashAttnTilingData*)tiling 直接强转，零拷贝；
    //   静态图 → convert_from_bytes 把常量数组拷成栈局部对象，dst_ptr 取其地址。
    GET_TILING_DATA_PTR_WITH_STRUCT(FlashAttnTilingData, tilingData, tiling);

    InitSocState();

    using FA_T = FlashAttnKernel::FAType<INPUT_T, OUT_T, pageAttention, qLayout, kvLayout, outLayout, s1TemplateType,
                                         s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask>;

    if constexpr (templateId == FA_Template_DN) {
        using CubeBlock = FlashAttnKernel::FANoQuantGqaBlockCubeDn<FA_T>;
        using VecFaBlock = FlashAttnKernel::FANoQuantGqaBlockVecDn<FA_T>;
        using VecFdBlock = FlashAttnKernel::FiaBlockVecFlashDecode<FA_T>;
        using VecDummy = FlashAttnKernel::FANoQuantGqaBlockVecDummyDn<FA_T>;
        using CubeDummy = FlashAttnKernel::FANoQuantGqaBlockCubeDummyDn<FA_T>;
#ifdef __DAV_CUBE__
        using Kernel = FlashAttnKernel::FlashAttentionNoQuantGqaKernelDn<FA_T, CubeBlock, VecDummy, VecDummy>;
#else
        using Kernel = FlashAttnKernel::FlashAttentionNoQuantGqaKernelDn<FA_T, CubeDummy, VecFaBlock, VecFdBlock>;
#endif
        Kernel op;
        op.Init(query, key, value, blockTable, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata,
                attnOut, softmaxLse, user, &tilingData->baseTiling);
        op.Process();
    } else {
        using CubeBlock = FlashAttnKernel::FANoQuantGqaBlockCubeNd<FA_T>;
        using VecFaBlock = FlashAttnKernel::FANoQuantGqaBlockVecNd<FA_T>;
        using VecFdBlock = FlashAttnKernel::FiaBlockVecFlashDecode<FA_T>;
        using VecDummy = FlashAttnKernel::FANoQuantGqaBlockVecDummyNd<FA_T>;
        using CubeDummy = FlashAttnKernel::FANoQuantGqaBlockCubeDummyNd<FA_T>;
#ifdef __DAV_CUBE__
        using Kernel = FlashAttnKernel::FlashAttentionNoQuantGqaKernelNd<FA_T, CubeBlock, VecDummy, VecDummy>;
#else
        using Kernel = FlashAttnKernel::FlashAttentionNoQuantGqaKernelNd<FA_T, CubeDummy, VecFaBlock, VecFdBlock>;
#endif
        Kernel op;
        op.Init(query, key, value, blockTable, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata,
                attnOut, softmaxLse, user, &tilingData->baseTiling);
        op.Process();
    }
#endif

    AscendC::PipeBarrier<PIPE_ALL>();
}
