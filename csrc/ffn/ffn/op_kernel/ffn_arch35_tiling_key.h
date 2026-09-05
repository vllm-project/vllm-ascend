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
 * \file ffn_arch35_tiling_key.h
 * \brief arch35 fused 路径的模板参数（TilingKey）声明：
 *        数据类型（bf16/fp16）× 激活（gelu/silu/swiglu）× 模式（basic/streamK）。
 */

#ifndef FFN_ARCH35_TILING_KEY_H
#define FFN_ARCH35_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define FFN_TPL_DTYPE_BF16 0
#define FFN_TPL_DTYPE_FP16 1

#define FFN_TPL_ACT_GELU 0
#define FFN_TPL_ACT_SILU 1
#define FFN_TPL_ACT_SWIGLU 2

#define FFN_TPL_MODE_BASIC 0
#define FFN_TPL_MODE_STREAMK 1

// 模板参数
ASCENDC_TPL_ARGS_DECL(FFN,
                      ASCENDC_TPL_UINT_DECL(DTYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FFN_TPL_DTYPE_BF16,
                                            FFN_TPL_DTYPE_FP16),
                      ASCENDC_TPL_UINT_DECL(ACT, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FFN_TPL_ACT_GELU,
                                            FFN_TPL_ACT_SILU, FFN_TPL_ACT_SWIGLU),
                      ASCENDC_TPL_UINT_DECL(MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FFN_TPL_MODE_BASIC,
                                            FFN_TPL_MODE_STREAMK));

// 合法组合：
//  - bf16 gelu/silu：basic + streamK（down split-K）
//  - bf16 swiglu：basic only（down 无 streamK）
//  - fp16 gelu/silu：basic + streamK
//  - fp16 swiglu：basic only
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, FFN_TPL_DTYPE_BF16),
        ASCENDC_TPL_UINT_SEL(ACT, ASCENDC_TPL_UI_LIST, FFN_TPL_ACT_GELU, FFN_TPL_ACT_SILU),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, FFN_TPL_MODE_BASIC, FFN_TPL_MODE_STREAMK)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, FFN_TPL_DTYPE_BF16),
        ASCENDC_TPL_UINT_SEL(ACT, ASCENDC_TPL_UI_LIST, FFN_TPL_ACT_SWIGLU),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, FFN_TPL_MODE_BASIC)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, FFN_TPL_DTYPE_FP16),
        ASCENDC_TPL_UINT_SEL(ACT, ASCENDC_TPL_UI_LIST, FFN_TPL_ACT_GELU, FFN_TPL_ACT_SILU),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, FFN_TPL_MODE_BASIC, FFN_TPL_MODE_STREAMK)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(DTYPE, ASCENDC_TPL_UI_LIST, FFN_TPL_DTYPE_FP16),
        ASCENDC_TPL_UINT_SEL(ACT, ASCENDC_TPL_UI_LIST, FFN_TPL_ACT_SWIGLU),
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, FFN_TPL_MODE_BASIC)), );

#endif // FFN_ARCH35_TILING_KEY_H
