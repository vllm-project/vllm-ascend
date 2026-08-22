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
 * \file incre_flash_attention_tiling_register.cpp
 * \brief
 */

#include "incre_flash_attention_tiling_impl.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingPrepareForIncreFlashAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(IncreFlashAttention)
    .Tiling(TilingIncreFlashAttention)
    .TilingParse<IncreFlashAttentionCompileInfo>(TilingPrepareForIncreFlashAttention)
    .TilingInputsDataDependency({5}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU});
} // namespace optiling