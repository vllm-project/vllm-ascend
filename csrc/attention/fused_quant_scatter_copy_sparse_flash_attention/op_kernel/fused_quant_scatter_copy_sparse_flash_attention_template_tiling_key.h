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
 * \file fused_quant_scatter_copy_sparse_flash_attention_template_tiling_key.h
 * \brief
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(FusedQuantScatterCopySparseFlashAttention,
    ASCENDC_TPL_BOOL_DECL(SPLIT_HEADS, 0, 1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(SPLIT_HEADS, 0, 1)));

#endif
