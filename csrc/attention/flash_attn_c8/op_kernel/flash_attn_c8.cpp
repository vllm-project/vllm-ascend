/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "adv_api/activation/softmax.h"
#include "flash_attn_c8_tiling_data.h"
#include "flash_attn_c8_blocks.h"

using namespace AscendC;

template <bool hasMask, bool packedP = false>
__aicore__ inline void RunFlashAttnC8(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR dequantScaleQuery, GM_ADDR dequantScaleKey, GM_ADDR dequantScaleValue,
    GM_ADDR cuSeqLensQ, GM_ADDR cuSeqLensKv, GM_ADDR sequsedQ, GM_ADDR attnMask,
    GM_ADDR metadata, GM_ADDR attnOut, GM_ADDR softmaxLse, GM_ADDR workspace,
    const optiling::FlashMlaWithKvcacheNoQuantTilingArch35 *tilingData)
{
    using namespace BaseApi;
    constexpr auto layout = FlashMlaC8Layout::LAYOUT_TND;
    constexpr auto m = S1TemplateType::Aligned128;
    constexpr auto n = S2TemplateType::Aligned128;
    constexpr auto d = static_cast<DTemplateType>(192);
    constexpr auto dv = DTemplateType::Aligned128;
    constexpr auto pse = PseTypeEnum::PSE_NONE_TYPE;
#ifdef __DAV_C310_CUBE__
    using Cube = FlashAttnC8BlockCube;
    using Vec = FAFullQuantMlaBlockVecDummy<fp8_e4m3fn_t, float, bfloat16_t, layout, layout,
        m, n, d, dv, pse, hasMask, false, true, 0, true, false, false, true, false>;
    using Fd = FiaBlockVecFlashDecodeFullQuantDummy<fp8_e4m3fn_t, float, bfloat16_t, layout, layout,
        m, n, d, dv, pse, hasMask, false, true, 0, false, false, true, false>;
#else
    using Cube = FAFullQuantMlaBlockCubeDummy<fp8_e4m3fn_t, float, layout,
        m, n, d, dv, true, 0, false, false, true, false>;
    using Vec = FlashAttnC8BlockVec<hasMask, packedP>;
    using Fd = FiaBlockVecFlashDecodeFullQuant<fp8_e4m3fn_t, float, bfloat16_t, layout, layout,
        m, n, d, dv, pse, hasMask, false, true, 0, false, false, true, false>;
#endif
    using Kernel = FlashAttentionFullQuantMlaKernel<Cube, Vec, Fd>;
    const uint32_t sections = ((__gm__ uint32_t *)metadata)[0];
    // The public variable-length ABI includes the leading zero for Q and KV.
    // Decode's nonpaged KV parser consumes only cumulative sequence ends.
    GM_ADDR kvEnds = cuSeqLensKv + sizeof(uint32_t);
    for (uint32_t section = 0; section < sections; ++section) {
        fa_base_matmul::ResetIdCounter();
        TPipe pipe;
        Kernel op;
        op.Init(query, key, value, attnMask, cuSeqLensQ, kvEnds, nullptr,
                dequantScaleQuery, dequantScaleKey, dequantScaleValue, queryRope, keyRope,
                softmaxLse, attnOut, GetUserWorkspace(workspace), metadata, tilingData,
                &pipe, sequsedQ, section);
        op.Process();
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void flash_attn_c8(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR dequantScaleQuery, GM_ADDR dequantScaleKey, GM_ADDR dequantScaleValue,
    GM_ADDR cuSeqLensQ, GM_ADDR cuSeqLensKv, GM_ADDR sequsedQ, GM_ADDR attnMask,
    GM_ADDR metadata, GM_ADDR attnOut, GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttnC8TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_MEMBER(optiling::FlashAttnC8TilingData, baseTiling, baseTilingIn, tiling);
    if (TILING_KEY_IS(0)) {
        RunFlashAttnC8<false>(query, key, value, queryRope, keyRope,
            dequantScaleQuery, dequantScaleKey, dequantScaleValue,
            cuSeqLensQ, cuSeqLensKv, sequsedQ, attnMask, metadata,
            attnOut, softmaxLse, workspace, &baseTilingIn);
    } else if (TILING_KEY_IS(1)) {
        RunFlashAttnC8<true>(query, key, value, queryRope, keyRope,
            dequantScaleQuery, dequantScaleKey, dequantScaleValue,
            cuSeqLensQ, cuSeqLensKv, sequsedQ, attnMask, metadata,
            attnOut, softmaxLse, workspace, &baseTilingIn);
    } else if (TILING_KEY_IS(2)) {
        RunFlashAttnC8<false, true>(query, key, value, queryRope, keyRope,
            dequantScaleQuery, dequantScaleKey, dequantScaleValue,
            cuSeqLensQ, cuSeqLensKv, sequsedQ, attnMask, metadata,
            attnOut, softmaxLse, workspace, &baseTilingIn);
    } else if (TILING_KEY_IS(3)) {
        RunFlashAttnC8<true, true>(query, key, value, queryRope, keyRope,
            dequantScaleQuery, dequantScaleKey, dequantScaleValue,
            cuSeqLensQ, cuSeqLensKv, sequsedQ, attnMask, metadata,
            attnOut, softmaxLse, workspace, &baseTilingIn);
    }
}
