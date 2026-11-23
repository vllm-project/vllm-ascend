#include "sol_attn_prepare_route_pack_kernel_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    SolAttnPrepareRoutePackKernelTilingData tiling;
    auto attrs = context->GetAttrs();
    const int32_t HC = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    tiling.set_HC(HC);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    tiling.set_NT(NT);
    const int32_t NT_Q = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    tiling.set_NT_Q(NT_Q);
    const int32_t NG = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    tiling.set_NG(NG);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    tiling.set_NPAD(NPAD);
    const int32_t NT_Q_PAD8 = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    tiling.set_NT_Q_PAD8(NT_Q_PAD8);
    const int32_t TILE_M = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
    tiling.set_TILE_M(TILE_M);
    const int32_t sink_s0 = (int32_t)*attrs->GetAttrPointer<int64_t>(7);
    tiling.set_sink_s0(sink_s0);
    const int32_t sink_s1 = (int32_t)*attrs->GetAttrPointer<int64_t>(8);
    tiling.set_sink_s1(sink_s1);
    const float scale_log2 = *attrs->GetAttrPointer<float>(9);
    tiling.set_scale_log2(scale_log2);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SolAttnPrepareRoutePackKernel).Tiling(TilingFunc);
}  // namespace optiling
