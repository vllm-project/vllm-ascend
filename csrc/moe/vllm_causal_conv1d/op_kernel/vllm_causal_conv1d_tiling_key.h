/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vllm_causal_conv1d_tiling_key.h
 * \brief vllm_causal_conv1d tiling key declare
 */

#ifndef __VLLM_CAUSAL_CONV1D_TILING_KEY_H__
#define __VLLM_CAUSAL_CONV1D_TILING_KEY_H__

#include "vllm_causal_conv1d_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

#define VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN 0
#define VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE 1
#define VLLM_CAUSAL_CONV1D_TPL_WIDTH_RUNTIME 0
#define VLLM_CAUSAL_CONV1D_TPL_WIDTH_2 1
#define VLLM_CAUSAL_CONV1D_TPL_WIDTH_3 2
#define VLLM_CAUSAL_CONV1D_TPL_WIDTH_4 3
#define VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_INVALID 0
#define VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS 1
#define VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD 2
ASCENDC_TPL_ARGS_DECL(VllmCausalConv1d,
                      ASCENDC_TPL_UINT_DECL(runModeKey, 1, ASCENDC_TPL_UI_LIST, VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN,
                                            VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE),
                      ASCENDC_TPL_UINT_DECL(widthKey, 2, ASCENDC_TPL_UI_LIST, VLLM_CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                                            VLLM_CAUSAL_CONV1D_TPL_WIDTH_2, VLLM_CAUSAL_CONV1D_TPL_WIDTH_3,
                                            VLLM_CAUSAL_CONV1D_TPL_WIDTH_4),
                      ASCENDC_TPL_UINT_DECL(fnPlanKey, 2, ASCENDC_TPL_UI_LIST, VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_INVALID,
                                            VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD));

#define VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(RUN_MODE, WIDTH, FN_PLAN)                                                 \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(runModeKey, ASCENDC_TPL_UI_LIST, RUN_MODE),                    \
                         ASCENDC_TPL_UINT_SEL(widthKey, ASCENDC_TPL_UI_LIST, WIDTH),                          \
                         ASCENDC_TPL_UINT_SEL(fnPlanKey, ASCENDC_TPL_UI_LIST, FN_PLAN),                       \
                         ASCENDC_TPL_TILING_STRUCT_SEL(VllmCausalConv1dTilingData))

// Keep entries in encoded tiling-key order: real-device sub-kernel dispatch is sensitive to declaration order.
ASCENDC_TPL_SEL(
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, VLLM_CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_INVALID),
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN, VLLM_CAUSAL_CONV1D_TPL_WIDTH_2,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN, VLLM_CAUSAL_CONV1D_TPL_WIDTH_3,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN, VLLM_CAUSAL_CONV1D_TPL_WIDTH_4,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN, VLLM_CAUSAL_CONV1D_TPL_WIDTH_2,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD),
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN, VLLM_CAUSAL_CONV1D_TPL_WIDTH_3,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD),
    VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY(VLLM_CAUSAL_CONV1D_TPL_RUN_MODE_FN, VLLM_CAUSAL_CONV1D_TPL_WIDTH_4,
                                VLLM_CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD));

#undef VLLM_CAUSAL_CONV1D_TPL_SEL_ENTRY

#endif // __VLLM_CAUSAL_CONV1D_TILING_KEY_H__
