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
 * \file sparse_attn_sharedkv_template_tiling_key.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHARED_TEMPLATE_TILING_KEY_H
#define SPARSE_ATTN_SHARED_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SAS_LAYOUT_BSND 0
#define SAS_LAYOUT_TND 1
#define SAS_LAYOUT_PA_ND 2

#define ASCENDC_TPL_4_BW 4

#define SWA_TEMPLATE 0
#define CFA_TEMPLATE 1
#define SCFA_TEMPLATE 2
// 模板参数支持的范围定义
ASCENDC_TPL_ARGS_DECL(SparseAttnSharedkv, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_BSND,
                                            SAS_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_PA_ND, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SWA_TEMPLATE,
                                            CFA_TEMPLATE, SCFA_TEMPLATE), );

// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_PA_ND, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, SWA_TEMPLATE), ),

    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0),
        ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
        ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_PA_ND, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
        ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, CFA_TEMPLATE),
    ),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLASH_DECODE, 0),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(KV_LAYOUT_T, ASCENDC_TPL_UI_LIST, SAS_LAYOUT_PA_ND, SAS_LAYOUT_BSND, SAS_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, SCFA_TEMPLATE), ), );

#endif // TEMPLATE_TILING_KEY