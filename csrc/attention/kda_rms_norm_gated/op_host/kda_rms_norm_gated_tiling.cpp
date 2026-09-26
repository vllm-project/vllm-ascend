// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <limits>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/kda_rms_norm_gated_tiling_data.h"
namespace optiling {
static ge::graphStatus TilingKdaRmsNormGated(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    if (!attrs || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    auto *data = context->GetTilingData<KdaRmsNormGatedTilingData>();
    data->tokens = *attrs->GetAttrPointer<int64_t>(0);
    data->heads = *attrs->GetAttrPointer<int64_t>(1);
    data->xTokenStride = *attrs->GetAttrPointer<int64_t>(2);
    data->gateTokenStride = *attrs->GetAttrPointer<int64_t>(3);
    data->epsilon = *attrs->GetAttrPointer<float>(4);
    data->reserved = 0;
    if (data->tokens <= 0 || data->heads <= 0 || data->heads > 128 ||
        data->xTokenStride < data->heads * 128 || data->gateTokenStride < data->heads * 128 ||
        data->xTokenStride - data->heads * 128 > std::numeric_limits<uint32_t>::max() / 2 ||
        data->gateTokenStride - data->heads * 128 > std::numeric_limits<uint32_t>::max() / 2 ||
        !std::isfinite(data->epsilon) || data->epsilon <= 0) return ge::GRAPH_FAILED;
    data->tileTokens = 128 / data->heads;
    const int64_t tiles = (data->tokens + data->tileTokens - 1) / data->tileTokens;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t usedBytes = data->tileTokens * data->heads * 128 * 2 * 6 + 512;
    if (usedBytes > ubBytes) return ge::GRAPH_FAILED;
    context->SetBlockDim(std::min<int64_t>(tiles, platform.GetCoreNumAiv()));
    const bool sigmoidOnly = *attrs->GetAttrPointer<bool>(5);
    context->SetTilingKey((context->GetInputDesc(2)->GetDataType() == ge::DT_FLOAT ? 1 : 0) +
        (sigmoidOnly ? 2 : 0));
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(KdaRmsNormGated).Tiling(TilingKdaRmsNormGated);
}
