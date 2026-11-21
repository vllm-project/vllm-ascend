// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "arch35/grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h"
#include "ascendc/host_api/tiling/template_argument.h"

using namespace AscendC;
using GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData;

// The template compiler emits key 0 for NZ (MXFP8) and key 1 for ZN (MXFP4).
ASCENDC_TPL_ARGS_DECL(GmmSwigluOaiMxQuant,
                      ASCENDC_TPL_UINT_DECL(TRANS_B, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TRANS_A, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0),
                      ASCENDC_TPL_UINT_DECL(BACKEND, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 1));

#if ORIG_DTYPE_X == DT_FLOAT4_E2M1
  #define GMM_SWIGLU_OAI_TRANS_B 1
#else
  #define GMM_SWIGLU_OAI_TRANS_B 0
#endif

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TRANS_B, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_OAI_TRANS_B),
                                     ASCENDC_TPL_UINT_SEL(TRANS_A, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BACKEND, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(GMMSwigluQuantV2TensorApiTilingData)));

template <int8_t TRANS_B, int8_t TRANS_A, int8_t BACKEND>
__global__ __aicore__ void gmm_swiglu_oai_mx_quant(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight,
                                                       GM_ADDR weightScale, GM_ADDR y, GM_ADDR yScale,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  REGISTER_TILING_DEFAULT(GMMSwigluQuantV2TensorApiTilingData);
  constexpr int kFloatOverflowMode = 60;
  const auto overflowMode = GetCtrlSpr<kFloatOverflowMode, kFloatOverflowMode>();
  SetCtrlSpr<kFloatOverflowMode, kFloatOverflowMode>(0);
  if constexpr (TRANS_B == 1) {
    GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::zn_layout_ptn, true>(
        x, weight, weightScale, xScale, nullptr, nullptr, groupList, y, yScale, workspace, tiling);
  } else {
    GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::nz_layout_ptn, true>(
        x, weight, weightScale, xScale, nullptr, nullptr, groupList, y, yScale, workspace, tiling);
  }
  SetCtrlSpr<kFloatOverflowMode, kFloatOverflowMode>(overflowMode);
}
