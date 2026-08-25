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
 * \file turbo_quant_sparse_flash_attention_template_tiling_key.h
 * \brief
 */

#ifndef TURBOQUANT_SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H
#define TURBOQUANT_SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define QSFA_LAYOUT_BSND 0
#define QSFA_LAYOUT_TND 1
#define QSFA_LAYOUT_PA_BSND 2

#define ASCENDC_TPL_4_BW 4

#define C_TEMPLATE 0
#define V_TEMPLATE 1

// 注意：DECL 中的取值列表决定 TilingKey 的编码（按列表下标而非字面值），
// 收窄列表会改变已生成的 TilingKey 与 kernel 二进制（实测 1158 -> 2），
// 且实测无论是否收窄都只生成 1 个 kernel 二进制，故此处保留完整取值域，
// 仅在下方 ASCENDC_TPL_SEL 中删除不可达的参数组合。
ASCENDC_TPL_ARGS_DECL(TurboQuantSparseFlashAttention, ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_BSND,
                                            QSFA_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_BSND,
                                            QSFA_LAYOUT_TND, QSFA_LAYOUT_PA_BSND),
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, C_TEMPLATE,
                                            V_TEMPLATE),
                      ASCENDC_TPL_BOOL_DECL(IS_SPLIT_G, 0, 1), );

// 支持的模板参数组合。用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
// query 仅支持 TND、KV 仅支持 PA_BSND，故只有下面一组模板参数可达；
// 其余组合（PAGE_ATTENTION=0、BSND query、TND KV）已随支持范围收窄一并移除。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0), ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_TND),
                                     ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, QSFA_LAYOUT_PA_BSND),
                                     ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, V_TEMPLATE),
                                     ASCENDC_TPL_BOOL_SEL(IS_SPLIT_G, 0, 1), ), );

#endif // TEMPLATE_TILING_KEY
