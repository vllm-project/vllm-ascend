/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file custom_fused_infer_attention_v310_tiling.cc
 * \brief
 */

#include "custom_fused_infer_attention_v310_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
namespace optiling {
ge::graphStatus TilingPrepareForIncreFlashAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(CustomFusedInferAttentionV310)
    .Tiling(TilingIncreFlashAttention)
    .TilingParse<IncreFlashAttentionCompileInfo>(TilingPrepareForIncreFlashAttention)
    .TilingInputsDataDependency({5}, {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU});
} // namespace optiling
