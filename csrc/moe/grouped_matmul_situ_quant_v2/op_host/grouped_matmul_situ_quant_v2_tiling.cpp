#include <string>
#include <vector>
#include <register/op_impl_registry.h>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include "grouped_matmul_situ_quant_v2_a8w4_tiling.h"
#include "grouped_matmul_situ_quant_v2_a8w4_base_tiling.h"

namespace optiling {
namespace {

constexpr size_t INPUT_X_INDEX = 0;

// A3 的 A8W4 MSD 走 GroupedMatmulSituQuantV2BaseTiling（模板注册，见下方 REGISTER_OPS_TILING_TEMPLATE）。
constexpr int32_t GMMSITU_BASE_TILING_TEMPLATE = 1;

} // namespace

// REGISTER_OPS_TILING_TEMPLATE 会用 ## 把 class_name 拼进符号名，因此只能传裸标识符，
// 这里用别名把带命名空间的完整类名引进来。
using GMMSITU_BASE_TILING_CLASS = GroupedMatmulSituQuantV2Tiling::GroupedMatmulSituQuantV2BaseTiling;
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSituQuantV2, GMMSITU_BASE_TILING_CLASS, GMMSITU_BASE_TILING_TEMPLATE);

ge::graphStatus Tiling4GroupedMatmulSituQuantV2(gert::TilingContext *context)
{
    std::vector<int32_t> registerList = {GMMSITU_BASE_TILING_TEMPLATE};
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

ge::graphStatus TilingPrepare4GroupedMatmulSituQuantV2(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto *platformInfoPtr = context->GetPlatformInfo();
    auto *compileInfoPtr = context->GetCompiledInfo<GMMSituV2CompileInfo>();
    if (platformInfoPtr == nullptr || compileInfoPtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "[gmsq_tiling] FAIL: platformInfo or compileInfo is null");
        return ge::GRAPH_FAILED;
    }

    // A3 的 A8W4 MSD tiling 依赖这几项平台信息；Ascend950 的 arch35 实现不读取 compileInfo。
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize_);
    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSituQuantV2)
    .Tiling(Tiling4GroupedMatmulSituQuantV2)
    .TilingParse<GMMSituV2CompileInfo>(TilingPrepare4GroupedMatmulSituQuantV2);

} // namespace optiling
