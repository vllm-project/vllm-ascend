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
 * \file grouped_matmul_situ_quant_tiling_key.h
 * \brief GroupedMatmulSituQuant 模板参数声明。
 *
 * 与 grouped_matmul_swiglu_quant_v2 的 MxA8W4 路径保持同一组模板维度：
 *   QUANT_B_TRANS / QUANT_A_TRANS : weight / x 是否转置（本算子只支持 NZ 且 weight 转置，故 B 恒为 TRANS）
 *   IS_SINGLE_MULTI_SINGLE        : weight 是以单张量堆叠(E,...)给出，还是 TensorList 逐专家给出
 */
#ifndef GROUPED_MATMUL_SITU_QUANT_TILING_KEY_H
#define GROUPED_MATMUL_SITU_QUANT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"

#define GMSQ_NO_TRANS 0
#define GMSQ_TRANS 1
#define GMSQ_NOT_SINGLE_MULTI_SINGLE 0
#define GMSQ_SINGLE_MULTI_SINGLE 1

// 模板参数
ASCENDC_TPL_ARGS_DECL(GroupedMatmulSituQuant, // 算子OpType
                      ASCENDC_TPL_UINT_DECL(QUANT_B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, GMSQ_NO_TRANS,
                                            GMSQ_TRANS),
                      ASCENDC_TPL_UINT_DECL(QUANT_A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, GMSQ_NO_TRANS,
                                            GMSQ_TRANS),
                      ASCENDC_TPL_BOOL_DECL(IS_SINGLE_MULTI_SINGLE, GMSQ_SINGLE_MULTI_SINGLE,
                                            GMSQ_NOT_SINGLE_MULTI_SINGLE));

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMSQ_TRANS),
                         ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMSQ_NO_TRANS),
                         ASCENDC_TPL_BOOL_SEL(IS_SINGLE_MULTI_SINGLE, GMSQ_NOT_SINGLE_MULTI_SINGLE,
                                              GMSQ_SINGLE_MULTI_SINGLE)),
    ASCENDC_TPL_TILING_STRUCT_SEL(GMMSQArch35Tiling::GMMSQWeightQuantTilingData), );
#endif // GROUPED_MATMUL_SITU_QUANT_TILING_KEY_H
