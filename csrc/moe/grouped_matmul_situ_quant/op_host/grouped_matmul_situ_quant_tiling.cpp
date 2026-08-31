#include <algorithm>
#include <cstring>
#include <register/op_impl_registry.h>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/grouped_matmul_situ_quant_tiling.h"

namespace optiling {
namespace {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_WEIGHT_INDEX = 2;
constexpr size_t ATTR_GROUP_LIST_TYPE_INDEX = 0;
constexpr size_t ATTR_BETA_INDEX = 1;
constexpr size_t ATTR_LINEAR_BETA_INDEX = 2;
constexpr uint32_t BASE_M = 128;
constexpr uint32_t MAIN_BLOCK_N2 = 64;

struct GroupedMatmulSituQuantCompileInfo {};

} // namespace

ge::graphStatus Tiling4GroupedMatmulSituQuant(gert::TilingContext *context)
{
    const auto *xShape = context->GetInputShape(INPUT_X_INDEX);
    const auto *weightShape = context->GetInputShape(INPUT_WEIGHT_INDEX);
    if (xShape == nullptr || weightShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &xOriginShape = xShape->GetOriginShape();
    const auto &weightOriginShape = weightShape->GetOriginShape();
    if (xOriginShape.GetDimNum() != 2 || weightOriginShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t k = xOriginShape.GetDim(1);
    const int64_t expertCount = weightOriginShape.GetDim(0);
    const int64_t packedWeightElements = weightOriginShape.GetShapeSize();
    if (k <= 0 || expertCount <= 0 || k % 64 != 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t packedElementsPerColumn = expertCount * (k / 2);
    if (packedElementsPerColumn <= 0 || packedWeightElements % packedElementsPerColumn != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t n = packedWeightElements / packedElementsPerColumn;
    const int64_t n2 = n / 2;
    if (n <= 0 || n % 2 != 0 || n2 % MAIN_BLOCK_N2 != 0) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *groupListType = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_LIST_TYPE_INDEX);
    const auto *beta = attrs->GetAttrPointer<float>(ATTR_BETA_INDEX);
    const auto *linearBeta = attrs->GetAttrPointer<float>(ATTR_LINEAR_BETA_INDEX);
    if (groupListType == nullptr || beta == nullptr || linearBeta == nullptr ||
        (*groupListType != 0 && *groupListType != 1) || *beta == 0.0f || *linearBeta == 0.0f) {
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = std::max<uint32_t>(1, platform.GetCoreNumAic());

    gmm_situ::SituTilingHeader tiling = {};
    tiling.coreNum = coreNum;
    tiling.activeCount = static_cast<uint32_t>(expertCount);
    tiling.kSize = static_cast<uint32_t>(k);
    tiling.nSize = static_cast<uint32_t>(n);
    tiling.baseM = BASE_M;
    tiling.mainBlockSize = MAIN_BLOCK_N2;
    tiling.firstTailBlockSize = 0;
    tiling.reserved = static_cast<uint32_t>(*groupListType);
    tiling.mainBlockCount = static_cast<uint64_t>(n2 / MAIN_BLOCK_N2);
    tiling.firstTailBlockCount = 0;
    tiling.beta = *beta;
    tiling.invBeta = 1.0f / *beta;
    tiling.linearBeta = *linearBeta;
    tiling.invLinearBeta = 1.0f / *linearBeta;

    auto *rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < sizeof(tiling)) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(rawTiling->GetData(), &tiling, sizeof(tiling));
    rawTiling->SetDataSize(sizeof(tiling));

    context->SetTilingKey(0);
    context->SetBlockDim(coreNum);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GroupedMatmulSituQuant(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSituQuant)
    .Tiling(Tiling4GroupedMatmulSituQuant)
    .TilingParse<GroupedMatmulSituQuantCompileInfo>(TilingPrepare4GroupedMatmulSituQuant);

} // namespace optiling
