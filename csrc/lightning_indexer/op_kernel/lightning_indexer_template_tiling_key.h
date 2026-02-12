/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_LI_H_
#define TEMPLATE_TILING_KEY_LI_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define LI_TPL_FP16 1
#define LI_TPL_INT32 3
#define LI_TPL_BF16 27

#define LI_LAYOUT_BSND 0
#define LI_LAYOUT_TND 1
#define LI_LAYOUT_PA_BSND 2

#define ASCENDC_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(LightningIndexer,
                      ASCENDC_TPL_DTYPE_DECL(DT_Q, LI_TPL_FP16, LI_TPL_BF16),
                      ASCENDC_TPL_DTYPE_DECL(DT_K, LI_TPL_FP16, LI_TPL_BF16),
                      ASCENDC_TPL_DTYPE_DECL(DT_OUT, LI_TPL_INT32), ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, LI_LAYOUT_BSND,
                                             LI_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(K_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            LI_LAYOUT_PA_BSND, LI_LAYOUT_BSND, LI_LAYOUT_TND), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, LI_TPL_FP16), ASCENDC_TPL_DTYPE_SEL(DT_K, LI_TPL_FP16),
                         ASCENDC_TPL_DTYPE_SEL(DT_OUT, LI_TPL_INT32),
                         ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_BSND, LI_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_PA_BSND), ),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, LI_TPL_BF16), ASCENDC_TPL_DTYPE_SEL(DT_K, LI_TPL_BF16),
                         ASCENDC_TPL_DTYPE_SEL(DT_OUT, LI_TPL_INT32),
                         ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_BSND, LI_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_PA_BSND), ),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, LI_TPL_FP16), ASCENDC_TPL_DTYPE_SEL(DT_K, LI_TPL_FP16),
                         ASCENDC_TPL_DTYPE_SEL(DT_OUT, LI_TPL_INT32),
                         ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 0),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_BSND, LI_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST,
                          LI_LAYOUT_BSND, LI_LAYOUT_TND), ),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, LI_TPL_BF16), ASCENDC_TPL_DTYPE_SEL(DT_K, LI_TPL_BF16),
                         ASCENDC_TPL_DTYPE_SEL(DT_OUT, LI_TPL_INT32),
                         ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 0),
                         ASCENDC_TPL_UINT_SEL(LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_BSND, LI_LAYOUT_TND),
                         ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST, LI_LAYOUT_BSND, LI_LAYOUT_TND), ), );

#endif