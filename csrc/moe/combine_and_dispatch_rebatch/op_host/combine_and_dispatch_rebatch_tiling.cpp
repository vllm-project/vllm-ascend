// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "../op_kernel/combine_and_dispatch_rebatch_tiling.h"
#include <cstdio>

namespace optiling {

static ge::graphStatus CombineAndDispatchRebatchTiling(gert::TilingContext *context)
{
    //printf("start tiling\n");
    auto attrs = context->GetAttrs();
    int64_t is_combine_rebatch = *attrs->GetAttrPointer<int64_t>(0);
    int64_t max_bs = *attrs->GetAttrPointer<int64_t>(1);
    int64_t k = *attrs->GetAttrPointer<int64_t>(2);

    // Get input shape to determine hidden_size
    auto batch_before_dispatch_shape = context->GetInputShape(0);
    int64_t hidden_size = 1;
    if (batch_before_dispatch_shape != nullptr && batch_before_dispatch_shape->GetStorageShape().GetDimNum() >= 2) {
        hidden_size = batch_before_dispatch_shape->GetStorageShape().GetDim(1);
    }

    // Get AIV core count
    // auto platform_info = context->GetPlatformInfo();
     uint32_t aiv_num = 48; // default for ascend910_93
    // if (platform_info != nullptr) {
    //     aiv_num = platform_info->GetCoreNumAiv();
    // }

    constexpr uint64_t UB_SIZE = 192 * 1024; // 192KB UB

    CombineAndDispatchRebatchTilingData tiling_data;
    tiling_data.combineAndDispatchRebatchInfo.isCombineRebatch = static_cast<uint32_t>(is_combine_rebatch);
    tiling_data.combineAndDispatchRebatchInfo.hiddenSize = static_cast<uint32_t>(hidden_size);
    tiling_data.combineAndDispatchRebatchInfo.maxBs = static_cast<uint32_t>(max_bs);
    tiling_data.combineAndDispatchRebatchInfo.k = static_cast<uint32_t>(k);
    // tiling_data.combineAndDispatchRebatchInfo.aivNum = aiv_num;
    tiling_data.combineAndDispatchRebatchInfo.totalUbSize = UB_SIZE;

    context->SetBlockDim(aiv_num);

    auto tiling_size = context->GetRawTilingData()->GetDataSize();
    context->GetRawTilingData()->Append(
        reinterpret_cast<const uint8_t *>(&tiling_data),
        sizeof(CombineAndDispatchRebatchTilingData));

    // // Set workspace size
    // context->SetTilingKey(1);
    size_t workspace_size = 1;
    context->GetWorkspaceSizes(workspace_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CombineAndDispatchRebatchTilingPrepare(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(CombineAndDispatchRebatch)
    .Tiling(CombineAndDispatchRebatchTiling)
    .TilingParse<CombineAndDispatchRebatchTilingData>(CombineAndDispatchRebatchTilingPrepare);

}  // namespace optiling
