/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GUMBEL_SAMPLE_TILING_KEY_H
#define GUMBEL_SAMPLE_TILING_KEY_H
#include "tiling/template_argument.h"

// 单维 TilingKey：applyTemp ∈ {0, 1}
//   0 = 跳过 z/temp 缩放（apply_temperature=false）
//   1 = 应用 z/temp 缩放（apply_temperature=true，默认）
ASCENDC_TPL_ARGS_DECL(GumbelSample,
    ASCENDC_TPL_UINT_DECL(applyTemp, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(applyTemp, ASCENDC_TPL_UI_LIST, 0, 1)));

#endif  // GUMBEL_SAMPLE_TILING_KEY_H
