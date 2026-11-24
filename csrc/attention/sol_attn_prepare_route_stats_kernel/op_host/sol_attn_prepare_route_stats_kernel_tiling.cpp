#include "sol_attn_prepare_route_stats_kernel_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    SolAttnPrepareRouteStatsKernelTilingData tiling;
    auto attrs = context->GetAttrs();
    coreNum = std::min(coreNum, ascendcPlatform.GetCoreNumAiv() / 2);
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    tiling.set_B(B);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    tiling.set_H(H);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    tiling.set_NT(NT);
    const int32_t NT_Q = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    tiling.set_NT_Q(NT_Q);
    const int32_t NG = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    tiling.set_NG(NG);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    tiling.set_NPAD(NPAD);
    const int32_t NT_Q_PAD8 = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
    tiling.set_NT_Q_PAD8(NT_Q_PAD8);
    const int32_t query_blocks_per_tile = (int32_t)*attrs->GetAttrPointer<int64_t>(7);
    tiling.set_query_blocks_per_tile(query_blocks_per_tile);
    const int32_t sink_start = (int32_t)*attrs->GetAttrPointer<int64_t>(8);
    tiling.set_sink_start(sink_start);
    const int32_t sink_end = (int32_t)*attrs->GetAttrPointer<int64_t>(9);
    tiling.set_sink_end(sink_end);
    const float scale_log2 = *attrs->GetAttrPointer<float>(10);
    tiling.set_scale_log2(scale_log2);
    const float scale_sq = *attrs->GetAttrPointer<float>(11);
    tiling.set_scale_sq(scale_sq);
    const float tau = *attrs->GetAttrPointer<float>(12);
    tiling.set_tau(tau);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetScheduleMode(1);
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = std::max<size_t>(std::max<size_t>(((size_t)(0) + (size_t)(((((B * H) * NPAD) * 128))) * 4), ((size_t)(((((((B * H) * NPAD) * 128) * 4) + (((B * H) * NPAD) * 4)))) + (size_t)(393216) * 4)), ((size_t)((((((B * H) * NPAD) * 128) * 4))) + (size_t)((((B * H) * NPAD))) * 4));
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SolAttnPrepareRouteStatsKernel).Tiling(TilingFunc);
}  // namespace optiling
