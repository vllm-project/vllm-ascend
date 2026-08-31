/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_tiling_register.cpp
 * \brief
 */

#include "fused_infer_attention_score_v2_sink_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingPrepareForFusedInferAttentionScoreV2Sink(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}
// ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX,
IMPL_OP_OPTILING(FusedInferAttentionScoreV2Sink)
    .Tiling(DoOpTilingFusedInferAttentionScoreV2Sink)
    .TilingParse<FusedInferAttentionScoreV2SinkCompileInfo>(
        TilingPrepareForFusedInferAttentionScoreV2Sink); // Register entrance functions to the framework

} // namespace optiling