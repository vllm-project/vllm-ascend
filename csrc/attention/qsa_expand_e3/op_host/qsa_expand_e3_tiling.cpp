#include "qsa_expand_e3_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
static ge::graphStatus QsaExpandE3TilingFunc(gert::TilingContext* context)
{
    const auto* shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    const int64_t rows = shape->GetStorageShape().GetDim(0);
    if (rows <= 0 || rows > UINT32_MAX) {
        OP_LOGE(context->GetNodeName(), "groups rows must be in (0, UINT32_MAX].");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) {
        cores = platform.GetCoreNum();
    }
    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }
    QsaExpandE3TilingData tiling;
    tiling.set_rows(static_cast<uint32_t>(rows));
    context->SetBlockDim(std::min(static_cast<uint32_t>(rows), cores));
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    auto raw = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QsaExpandE3).Tiling(QsaExpandE3TilingFunc);
}  // namespace optiling
