/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file COMPRESSOR_template_tiling_key.h
 * \brief
 */

#ifndef COMPRESSOR_TEMPLATE_TILING_KEY_H
#define COMPRESSOR_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define ASCENDC_TPL_1_BW 1 // 每个参数占用1个bit位
#define ASCENDC_TPL_2_BW 2 // 每个参数占用2个bit位
#define ASCENDC_TPL_4_BW 4 // 每个参数占用4个bit位

// 可表示的tilingkey范围为64bit，注意不可超过限制
ASCENDC_TPL_ARGS_DECL(compressor, // 算子唯一标识，与opType保持一致
    // 可能需要切分之后的headdim
    // bit:0 LAYOUT 0:BSH 1:TH 
    ASCENDC_TPL_UINT_DECL(X_LAYOUT, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit:1-4 x的dtype  0:BF16 1:FP16
    ASCENDC_TPL_UINT_DECL(X_DTYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    // bit:5-6  coff 1:无需overlap 2:需要overlap
    ASCENDC_TPL_UINT_DECL(COFF, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 1, 2),
    // bit:7-8  rotary_mode 1:half 2:interleave
    ASCENDC_TPL_UINT_DECL(ROTARY_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 1, 2),
    // bit:9-10  template_id 0:empty_tensor 1:normal 2:perf
    ASCENDC_TPL_UINT_DECL(TEMPLATE_ID, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
);

ASCENDC_TPL_SEL(

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(X_LAYOUT, ASCENDC_TPL_UI_LIST, 0, 1),
                        ASCENDC_TPL_UINT_SEL(X_DTYPE, ASCENDC_TPL_UI_LIST, 0, 1),
                        ASCENDC_TPL_UINT_SEL(COFF, ASCENDC_TPL_UI_LIST, 1, 2),
                        ASCENDC_TPL_UINT_SEL(ROTARY_MODE, ASCENDC_TPL_UI_LIST, 1, 2),
                        ASCENDC_TPL_UINT_SEL(TEMPLATE_ID, ASCENDC_TPL_UI_LIST, 0, 1, 2),
                        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::CompressorTilingData)),
);

#endif // COMPRESSOR_TEMPLATE_TILING_KEY_H