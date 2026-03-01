/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "../op_kernel/matmul_gelu_tiling.h"
#include "register/op_def_registry.h"

#include "graph/utils/type_utils.h"
#include "tiling/tiling_api.h"

#include "log/ops_log.h"
#include "error/ops_error.h"

constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_BIAS = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulGeluTilingData *tiling = context->GetTilingData<MatmulGeluTilingData>();

    const ge::DataType xDataType = context->GetInputDesc(INPUT_INDEX_X)->GetDataType();
    const gert::Shape xShape = context->GetInputShape(INPUT_INDEX_X)->GetStorageShape();
    const gert::Shape weightShape = context->GetInputShape(INPUT_INDEX_WEIGHT)->GetStorageShape();
    const gert::Shape BiasShape = context->GetInputShape(INPUT_INDEX_BIAS)->GetStorageShape();
    size_t  m = xShape.GetDim(0);
    size_t  k = xShape.GetDim(1);
    size_t  n = BiasShape.GetDim(0);
    tiling->m = m;
    tiling->n = n;
    tiling->k = k;

    tiling->transB = true;

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t totalCoreNum = ascendcPlatform.GetCoreNumAic();

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    const size_t userWorkspaceSize = m * n * sizeof(float);
    currentWorkspace[0] = systemWorkspacesSize + userWorkspaceSize;

    context->SetBlockDim(totalCoreNum);

    return ge::GRAPH_SUCCESS;
}

struct MatmulGeluCompileInfo {};
ge::graphStatus TilingParseForMatmulGelu(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulGelu)
    .Tiling(TilingFunc)
    .TilingParse<MatmulGeluCompileInfo>(TilingParseForMatmulGelu);
