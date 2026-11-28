/**
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lim_copy_descriptors_linear_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// The aclnn single-op path hands this operator flattened storage shapes for
// custom-vendor builds, so the tiling derives nothing from shapes: every
// geometry dimension travels inside the params int64 tensor and is read by
// the kernel. The grid is one block per AIV core; the kernel grid-strides
// the flattened miss space (A3 launched block ids are neither contiguous
// nor stable, so the stride loop tolerates any scheduled subset).
ge::graphStatus Tiling4LimCopyDescriptorsLinear(gert::TilingContext *context)
{
    LimCopyDescriptorsLinearTilingData tiling;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    const int64_t used = coreNum > 0 ? coreNum : 1;
    context->SetBlockDim(static_cast<uint32_t>(used));

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    tiling.set_usedCoreNum(used);
    context->SetTilingKey(0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4LimCopyDescriptorsLinear(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LimCopyDescriptorsLinear)
    .Tiling(Tiling4LimCopyDescriptorsLinear)
    .TilingParse<LimCopyDescriptorsLinearCompileInfo>(TilingPrepare4LimCopyDescriptorsLinear);

} // namespace optiling
