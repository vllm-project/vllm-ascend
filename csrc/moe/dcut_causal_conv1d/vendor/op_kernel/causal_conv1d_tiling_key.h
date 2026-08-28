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
 * \file causal_conv1d_tiling_key.h
 * \brief causal_conv1d tiling key declare
 */

#ifndef __CAUSAL_CONV1D_TILING_KEY_H__
#define __CAUSAL_CONV1D_TILING_KEY_H__

#include "causal_conv1d_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

#define CAUSAL_CONV1D_TPL_RUN_MODE_FN 0
#define CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE 1
#define CAUSAL_CONV1D_TPL_WIDTH_RUNTIME 0
#define CAUSAL_CONV1D_TPL_WIDTH_2 1
#define CAUSAL_CONV1D_TPL_WIDTH_3 2
#define CAUSAL_CONV1D_TPL_WIDTH_4 3
#define CAUSAL_CONV1D_TPL_FN_PLAN_INVALID 0
#define CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS 1
#define CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD 2
ASCENDC_TPL_ARGS_DECL(CausalConv1d,
                      ASCENDC_TPL_UINT_DECL(runModeKey, 1, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_RUN_MODE_FN,
                                            CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE),
                      ASCENDC_TPL_UINT_DECL(widthKey, 2, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                                            CAUSAL_CONV1D_TPL_WIDTH_2, CAUSAL_CONV1D_TPL_WIDTH_3,
                                            CAUSAL_CONV1D_TPL_WIDTH_4),
                      ASCENDC_TPL_UINT_DECL(fnPlanKey, 2, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_FN_PLAN_INVALID,
                                            CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD));

#define CAUSAL_CONV1D_TPL_SEL_ENTRY(RUN_MODE, WIDTH, FN_PLAN)                                                 \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(runModeKey, ASCENDC_TPL_UI_LIST, RUN_MODE),                    \
                         ASCENDC_TPL_UINT_SEL(widthKey, ASCENDC_TPL_UI_LIST, WIDTH),                          \
                         ASCENDC_TPL_UINT_SEL(fnPlanKey, ASCENDC_TPL_UI_LIST, FN_PLAN),                       \
                         ASCENDC_TPL_TILING_STRUCT_SEL(CausalConv1dTilingData))

// Keep entries in encoded tiling-key order: real-device sub-kernel dispatch is sensitive to declaration order.
ASCENDC_TPL_SEL(
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                                CAUSAL_CONV1D_TPL_FN_PLAN_INVALID),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_2,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_3,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_4,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_2,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_3,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_4,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD));

#undef CAUSAL_CONV1D_TPL_SEL_ENTRY

#endif // __CAUSAL_CONV1D_TILING_KEY_H__
