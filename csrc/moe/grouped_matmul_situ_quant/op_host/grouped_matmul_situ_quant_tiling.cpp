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
constexpr size_t INPUT_GROUP_LIST_INDEX = 4;
constexpr size_t ATTR_GROUP_LIST_TYPE_INDEX = 0;
constexpr size_t ATTR_BETA_INDEX = 1;
constexpr size_t ATTR_LINEAR_BETA_INDEX = 2;
constexpr uint32_t BASE_M = 128;
constexpr uint32_t MAIN_BLOCK_N2 = 64;
constexpr uint32_t TENSOR_LIST_FLAG = 1U << 1;

struct GroupedMatmulSituQuantCompileInfo {};

} // namespace

ge::graphStatus Tiling4GroupedMatmulSituQuant(gert::TilingContext *context)
{
    const auto *xShape = context->GetDynamicInputShape(INPUT_X_INDEX, 0);
    const auto *weightShape = context->GetDynamicInputShape(INPUT_WEIGHT_INDEX, 0);
    const auto *groupListShape = context->GetDynamicInputShape(INPUT_GROUP_LIST_INDEX, 0);
    if (xShape == nullptr || weightShape == nullptr || groupListShape == nullptr) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: null shape ptr (x=%p weight=%p groupList=%p)",
                static_cast<const void *>(xShape), static_cast<const void *>(weightShape),
                static_cast<const void *>(groupListShape));
        return ge::GRAPH_FAILED;
    }

    const auto &xOriginShape = xShape->GetOriginShape();
    const auto &weightOriginShape = weightShape->GetOriginShape();
    const auto &groupListOriginShape = groupListShape->GetOriginShape();
    const bool isTensorList = context->GetDynamicInputShape(INPUT_WEIGHT_INDEX, 1) != nullptr;
    OP_LOGI(context->GetNodeName(),
            "[gmsq_tiling] enter: xDim=%zu xDims=[%ld,%ld] wDim=%zu wDim0=%ld wShapeSize=%ld list=%d",
            xOriginShape.GetDimNum(),
            xOriginShape.GetDimNum() > 0 ? xOriginShape.GetDim(0) : -1,
            xOriginShape.GetDimNum() > 1 ? xOriginShape.GetDim(1) : -1,
            weightOriginShape.GetDimNum(),
            weightOriginShape.GetDimNum() > 0 ? weightOriginShape.GetDim(0) : -1,
            weightOriginShape.GetShapeSize(), isTensorList);
    if (xOriginShape.GetDimNum() != 2 || weightOriginShape.GetDimNum() < 1 ||
        groupListOriginShape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: xDim=%zu (want 2), wDim=%zu (want >=1), groupListDim=%zu (want 1)",
                xOriginShape.GetDimNum(), weightOriginShape.GetDimNum(), groupListOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t k = xOriginShape.GetDim(1);
    const int64_t expertCount = isTensorList ? groupListOriginShape.GetDim(0) : weightOriginShape.GetDim(0);
    const int64_t packedWeightElements = weightOriginShape.GetShapeSize();
    if (k <= 0 || expertCount <= 0 || k % 64 != 0) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: k=%ld E=%ld (k%%64=%ld)", k, expertCount, k % 64);
        return ge::GRAPH_FAILED;
    }

    const int64_t packedElementsPerColumn = (isTensorList ? 1 : expertCount) * (k / 2);
    if (packedElementsPerColumn <= 0 || packedWeightElements % packedElementsPerColumn != 0) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: weight packed=%ld perCol=%ld rem=%ld",
                packedWeightElements, packedElementsPerColumn,
                packedWeightElements % packedElementsPerColumn);
        return ge::GRAPH_FAILED;
    }
    const int64_t n = packedWeightElements / packedElementsPerColumn;
    const int64_t n2 = n / 2;
    if (n <= 0 || n % 2 != 0 || n2 % MAIN_BLOCK_N2 != 0) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: n=%ld n%%2=%ld n2=%ld n2%%%u=%ld", n, n % 64, n2,
                MAIN_BLOCK_N2, n2 % MAIN_BLOCK_N2);
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "[gmsq_tiling] FAIL: attrs is null");
        return ge::GRAPH_FAILED;
    }
    const auto *groupListType = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_LIST_TYPE_INDEX);
    const auto *beta = attrs->GetAttrPointer<float>(ATTR_BETA_INDEX);
    const auto *linearBeta = attrs->GetAttrPointer<float>(ATTR_LINEAR_BETA_INDEX);
    if (groupListType == nullptr || beta == nullptr || linearBeta == nullptr ||
        (*groupListType != 0 && *groupListType != 1) || *beta == 0.0f || *linearBeta == 0.0f) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: attrs glt=%ld beta=%f linearBeta=%f (ptr nulls: %d/%d/%d)",
                groupListType ? *groupListType : -999, beta ? *beta : -999.0f,
                linearBeta ? *linearBeta : -999.0f, groupListType == nullptr, beta == nullptr,
                linearBeta == nullptr);
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = std::max<uint32_t>(1, platform.GetCoreNumAic());

    // The executor's tiling buffer is a growable container (initial capacity is
    // a small default); Append() expands it and tracks the data size, whereas
    // writing through GetTilingData<T>() fails while capacity is still 8 bytes.
    gmm_situ::SituTilingHeader tiling = {};
    tiling.coreNum = coreNum;
    tiling.activeCount = static_cast<uint32_t>(expertCount);
    tiling.kSize = static_cast<uint32_t>(k);
    tiling.nSize = static_cast<uint32_t>(n);
    tiling.baseM = BASE_M;
    tiling.mainBlockSize = MAIN_BLOCK_N2;
    tiling.firstTailBlockSize = 0;
    tiling.reserved = static_cast<uint32_t>(*groupListType) | (isTensorList ? TENSOR_LIST_FLAG : 0U);
    tiling.mainBlockCount = static_cast<uint64_t>(n2 / MAIN_BLOCK_N2);
    tiling.firstTailBlockCount = 0;
    tiling.beta = *beta;
    tiling.invBeta = 1.0f / *beta;
    tiling.linearBeta = *linearBeta;
    tiling.invLinearBeta = 1.0f / *linearBeta;

    auto *rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->Append(tiling) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(),
                "[gmsq_tiling] FAIL: Append header failed (rawTiling=%p cap=%zu need=%zu)",
                static_cast<void *>(rawTiling),
                rawTiling ? rawTiling->GetCapacity() : 0, sizeof(tiling));
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context->GetNodeName(),
            "[gmsq_tiling] OK: k=%ld E=%ld n=%ld n2=%ld glt=%ld beta=%f lBeta=%f cores=%u",
            k, expertCount, n, n2, *groupListType, *beta, *linearBeta, coreNum);
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
