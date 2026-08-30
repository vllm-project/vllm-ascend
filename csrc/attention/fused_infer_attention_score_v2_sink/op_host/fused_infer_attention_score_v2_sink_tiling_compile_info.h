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
 * \file fused_infer_attention_score_v2_sink_tiling_compile_info.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_COMPILE_INFO_H
#define FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_COMPILE_INFO_H

#include "register/tilingdata_base.h"
#include <tiling/platform/platform_ascendc.h>

namespace optiling {
struct FusedInferAttentionScoreV2SinkCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_COMPILE_INFO_H