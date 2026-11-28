/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "hc_post_pre_norm_config.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"

#if __has_include("../../hc_pre/op_kernel/hc_pre_m_k_split_core.h")
  #include "../../hc_pre/op_kernel/hc_pre_m_k_split_core.h"
  #include "../../rms_norm_cast/op_kernel/rms_norm_cast.h"
#else
  #include "../hc_pre/hc_pre_m_k_split_core.h"
  #include "../rms_norm_cast/rms_norm_cast.h"
#endif

#include "hc_post_pre_norm_post_mix.cpp"

using namespace AscendC;

namespace {
// Keep the literal keys in TILING_KEY_IS in sync with host tiling.
static_assert(HcPostPreNorm::kWithFp32TilingKey == 2 && HcPostPreNorm::kWithoutFp32TilingKey == 3);

template <bool WriteFp32>
__aicore__ inline void ProcessHcPostPreNorm(GM_ADDR residual, GM_ADDR ffn_hc_fn, GM_ADDR ffn_hc_scale,
                                            GM_ADDR ffn_hc_base, GM_ADDR attn_pre, GM_ADDR attn_out,
                                            GM_ADDR attn_post, GM_ADDR attn_comb, GM_ADDR ffn_norm_weight,
                                            GM_ADDR y, GM_ADDR ffn_post, GM_ADDR ffn_comb, GM_ADDR ffn_pre,
                                            GM_ADDR new_residual, GM_ADDR y_fp32, GM_ADDR userWorkspace,
                                            const HcPreTilingData& tilingData) {
  GM_ADDR mixedInput = userWorkspace + tilingData.mixedInputOffset;
  const auto numRows = static_cast<uint32_t>(tilingData.bs);
  const auto hidden = static_cast<uint32_t>(tilingData.d);
  const auto vectorCores = static_cast<uint32_t>(tilingData.cubeCoreNum * HcPre::CV_RATIO);

  {
    TPipe pipe;
    if ASCEND_IS_AIV {
      HcPostPreNorm::KernelPostMix postMix;
      postMix.Init(attn_out, residual, attn_post, attn_comb, attn_pre, new_residual, mixedInput, numRows, hidden,
                   vectorCores, &pipe);
      postMix.Process();
      PipeBarrier<PIPE_ALL>();
    }
    // Include idle blocks so projection can safely read every residual tile.
    SyncAll<false>();
    pipe.Destroy();
  }

  {
    TPipe pipe;
    HcPre::HcPreMembaseKSplitCorePart1<bfloat16_t> projection;
    projection.Init(new_residual, ffn_hc_fn, userWorkspace, &tilingData, &pipe);
    projection.Process();
    pipe.Destroy();
  }

  if ASCEND_IS_AIV {
    {
      TPipe pipe;
      HcPre::HcPreMembaseKSplitCorePart2<bfloat16_t, true, false> coefficients;
      coefficients.Init(new_residual, ffn_hc_scale, ffn_hc_base, mixedInput, ffn_post, ffn_comb, attn_pre, ffn_pre,
                        userWorkspace, &tilingData, &pipe);
      coefficients.Process();
      PipeBarrier<PIPE_ALL>();
      pipe.Destroy();
    }

    const uint32_t rowsPerCore = (numRows + vectorCores - 1) / vectorCores;
    if (GetBlockIdx() * rowsPerCore < numRows) {
      TPipe pipe;
      KernelRmsNormCast<bfloat16_t, WriteFp32> norm(&pipe);
      norm.Init(mixedInput, ffn_norm_weight, y, y_fp32, numRows, hidden, hidden, rowsPerCore,
                tilingData.inputNormInvHidden, tilingData.inputNormEps);
      norm.Process();
      pipe.Destroy();
    }
  }
}
}  // namespace

extern "C" __global__ __aicore__ void hc_post_pre_norm(GM_ADDR residual, GM_ADDR ffn_hc_fn, GM_ADDR ffn_hc_scale,
                                                       GM_ADDR ffn_hc_base, GM_ADDR attn_pre, GM_ADDR attn_out,
                                                       GM_ADDR attn_post, GM_ADDR attn_comb, GM_ADDR ffn_norm_weight,
                                                       GM_ADDR y, GM_ADDR ffn_post, GM_ADDR ffn_comb, GM_ADDR ffn_pre,
                                                       GM_ADDR new_residual, GM_ADDR y_fp32, GM_ADDR workspace,
                                                       GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWorkspace = GetUserWorkspace(workspace);
  if (userWorkspace == nullptr) {
    return;
  }

  if (TILING_KEY_IS(2)) {
    GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tilingData, tiling);
    ProcessHcPostPreNorm<true>(residual, ffn_hc_fn, ffn_hc_scale, ffn_hc_base, attn_pre, attn_out, attn_post,
                              attn_comb, ffn_norm_weight, y, ffn_post, ffn_comb, ffn_pre, new_residual,
                              y_fp32, userWorkspace, tilingData);
  } else if (TILING_KEY_IS(3)) {
    GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, tilingData, tiling);
    ProcessHcPostPreNorm<false>(residual, ffn_hc_fn, ffn_hc_scale, ffn_hc_base, attn_pre, attn_out, attn_post,
                               attn_comb, ffn_norm_weight, y, ffn_post, ffn_comb, ffn_pre, new_residual,
                               y_fp32, userWorkspace, tilingData);
  }
}
