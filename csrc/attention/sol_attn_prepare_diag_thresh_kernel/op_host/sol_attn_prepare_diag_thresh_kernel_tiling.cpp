#include "sol_attn_prepare_diag_thresh_kernel_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    SolAttnPrepareDiagThreshKernelTilingData tiling;
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    tiling.set_B(B);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    tiling.set_H(H);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    tiling.set_NT(NT);
    const int32_t NG = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    tiling.set_NG(NG);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    tiling.set_NPAD(NPAD);
    const float scale_log2 = *attrs->GetAttrPointer<float>(5);
    tiling.set_scale_log2(scale_log2);
    const float tau = *attrs->GetAttrPointer<float>(6);
    tiling.set_tau(tau);
    const float inv_nt = *attrs->GetAttrPointer<float>(7);
    tiling.set_inv_nt(inv_nt);
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

IMPL_OP_OPTILING(SolAttnPrepareDiagThreshKernel).Tiling(TilingFunc);
}  // namespace optiling
