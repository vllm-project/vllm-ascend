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
 * \file paged_select_attention_tiling_register.cpp
 * \brief
 */

#include "paged_select_attention_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingPrepareForPagedSelectAttention(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PagedSelectAttention)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .Tiling(DoOpTilingPagedSelectAttention)
    .TilingParse<PagedSelectAttentionCompileInfo>(TilingPrepareForPagedSelectAttention);
} // namespace optiling
