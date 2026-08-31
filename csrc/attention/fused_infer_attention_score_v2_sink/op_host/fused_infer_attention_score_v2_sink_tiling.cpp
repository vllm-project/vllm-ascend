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
 * \file fused_infer_attention_score_v2_sink_tiling.cpp
 * \brief
 */
#include "fused_infer_attention_score_v2_sink_tiling.h"
#include "error/ops_error.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "fused_infer_attention_score_v2_sink_tiling_v3.h"
#include "register/device_op_impl_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingFusedInferAttentionScoreV2Sink(gert::TilingContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("FusedInferAttentionScoreV2Sink", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (RouteToFia(context)) {
        return TilingFusedInferAttentionScoreV2SinkV3(context);
    }
    return ge::GRAPH_FAILED;
}

FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScoreV2Sink(gert::TilingContext *context)
{
    OPS_CHECK(context == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR("FusedInferAttentionScoreV2Sink", "Tiling context is null."),
              return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_CHECK(platformInfoPtr == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
              return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto socShortName = ascendcPlatform.GetSocVersion();
    if (socShortName == platform_ascendc::SocVersion::ASCEND910_55) {
        return ge::GRAPH_FAILED;
    } else {
        return TilingFusedInferAttentionScoreV2Sink(context);
    }
    return ge::GRAPH_SUCCESS;
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingFusedInferAttentionScoreV2Sink
    (gert::TilingContext *context)
{
    return DoOpTilingFusedInferAttentionScoreV2Sink(context);
}
}
DEVICE_IMPL_OP_OPTILING(FusedInferAttentionScoreV2Sink)
    .Tiling(optiling::DeviceDoOpTilingFusedInferAttentionScoreV2Sink);
} // namespace optiling