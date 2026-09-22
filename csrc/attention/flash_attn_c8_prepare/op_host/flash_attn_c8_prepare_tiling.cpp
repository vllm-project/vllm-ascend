// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/flash_attn_c8_prepare_tiling_data.h"
namespace optiling {
static ge::graphStatus TilingFlashAttnC8Prepare(gert::TilingContext *ctx)
{
    const auto *attrs = ctx->GetAttrs();
    if (!attrs || !ctx->GetPlatformInfo()) return ge::GRAPH_FAILED;
    auto *data = ctx->GetTilingData<FlashAttnC8PrepareTilingData>();
    data->queryTokens = *attrs->GetAttrPointer<int64_t>(0);
    data->keyTokens = *attrs->GetAttrPointer<int64_t>(1);
    data->heads = *attrs->GetAttrPointer<int64_t>(2);
    data->queryTokenStride = *attrs->GetAttrPointer<int64_t>(3);
    data->queryHeadStride = *attrs->GetAttrPointer<int64_t>(4);
    data->keyTokenStride = *attrs->GetAttrPointer<int64_t>(5);
    data->keyHeadStride = *attrs->GetAttrPointer<int64_t>(6);
    data->valueTokenStride = *attrs->GetAttrPointer<int64_t>(7);
    data->valueHeadStride = *attrs->GetAttrPointer<int64_t>(8);
    data->ropeTokenStride = *attrs->GetAttrPointer<int64_t>(9);
    data->ropeHeadStride = *attrs->GetAttrPointer<int64_t>(10);
    if (data->queryTokens <= 0 || data->keyTokens <= 0 || data->heads <= 0 || data->heads > 256 ||
        data->queryTokenStride < 192 || data->queryHeadStride < 192 ||
        data->keyTokenStride < 128 || data->keyHeadStride < 128 ||
        data->valueTokenStride < 128 || data->valueHeadStride < 128 ||
        data->ropeTokenStride < 64 || data->ropeHeadStride < 0) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const int64_t tasks = (std::max(data->queryTokens, data->keyTokens) * data->heads +
        FlashAttnC8Config::PREPARE_ROWS - 1) / FlashAttnC8Config::PREPARE_ROWS;
    ctx->SetBlockDim(std::min<int64_t>(tasks, platform.GetCoreNumAiv()));
    ctx->SetTilingKey(*attrs->GetAttrPointer<bool>(11) ? 1 : 0);
    ctx->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(FlashAttnC8Prepare).Tiling(TilingFlashAttnC8Prepare);
}
