#include "sol_attn_prepare_qkv_kernel_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    SolAttnPrepareQkvKernelTilingData tiling;
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    tiling.set_B(B);
    const int32_t T = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    tiling.set_T(T);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    tiling.set_H(H);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    tiling.set_NT(NT);
    const int32_t NG = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    tiling.set_NG(NG);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    tiling.set_NPAD(NPAD);
    const int32_t T_PAD_KV = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
    tiling.set_T_PAD_KV(T_PAD_KV);
    const float inv_tail_len = *attrs->GetAttrPointer<float>(7);
    tiling.set_inv_tail_len(inv_tail_len);
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

IMPL_OP_OPTILING(SolAttnPrepareQkvKernel).Tiling(TilingFunc);
}  // namespace optiling
