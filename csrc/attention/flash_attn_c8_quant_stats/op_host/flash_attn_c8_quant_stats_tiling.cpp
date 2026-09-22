// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/flash_attn_c8_quant_stats_tiling_data.h"
namespace optiling {
static ge::graphStatus TilingFlashAttnC8QuantStats(gert::TilingContext *ctx)
{
    const auto *attrs = ctx->GetAttrs();
    if (!attrs || !ctx->GetPlatformInfo()) return ge::GRAPH_FAILED;
    auto *data = ctx->GetTilingData<FlashAttnC8QuantStatsTilingData>();
    data->tokens = *attrs->GetAttrPointer<int64_t>(0);
    data->heads = *attrs->GetAttrPointer<int64_t>(1);
    data->keyTokenStride = *attrs->GetAttrPointer<int64_t>(2);
    data->keyHeadStride = *attrs->GetAttrPointer<int64_t>(3);
    data->valueTokenStride = *attrs->GetAttrPointer<int64_t>(4);
    data->valueHeadStride = *attrs->GetAttrPointer<int64_t>(5);
    if (data->tokens <= 0 || data->heads <= 0 || data->heads > 256 ||
        data->keyTokenStride < 128 || data->keyHeadStride < 128 ||
        data->valueTokenStride < 128 || data->valueHeadStride < 128) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const int64_t tasks = ((data->tokens + FlashAttnC8Config::STATS_TOKENS - 1) /
        FlashAttnC8Config::STATS_TOKENS) * data->heads;
    ctx->SetBlockDim(std::min<int64_t>(tasks, platform.GetCoreNumAiv()));
    ctx->SetTilingKey(0);
    ctx->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(FlashAttnC8QuantStats).Tiling(TilingFlashAttnC8QuantStats);
}
