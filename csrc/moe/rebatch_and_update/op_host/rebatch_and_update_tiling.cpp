#include <cstdint>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/rebatch_and_update_tiling.h"

namespace {
constexpr uint32_t INPUT_BATCH_IDX = 0;
constexpr uint32_t INPUT_PREFIX_INDEX_IDX = 3;
constexpr uint32_t INPUT_BLOCK_TABLE_IDX = 20;
constexpr uint32_t INPUT_DISCARD_POSITIONS_IDX = 22;
constexpr uint32_t ATTR_IS_COMBINE_IDX = 0;
constexpr uint32_t ATTR_MAX_BS_IDX = 1;
constexpr uint32_t ATTR_K_IDX = 2;
constexpr uint32_t ATTR_LAYER_IDX = 3;
constexpr uint32_t AIV_NUM = 48;
constexpr uint64_t WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
}  // namespace

namespace optiling {

static ge::graphStatus RebatchAndUpdateTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OPS_LOG_E(nodeName, "attrs is nullptr."),
              return ge::GRAPH_FAILED);

    auto isCombine = attrs->GetAttrPointer<int64_t>(ATTR_IS_COMBINE_IDX);
    auto maxBs = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_IDX);
    auto k = attrs->GetAttrPointer<int64_t>(ATTR_K_IDX);
    auto layerIdx = attrs->GetAttrPointer<int64_t>(ATTR_LAYER_IDX);
    OPS_CHECK(isCombine == nullptr || maxBs == nullptr || k == nullptr ||
                  layerIdx == nullptr,
              OPS_LOG_E(nodeName, "required attr is null."),
              return ge::GRAPH_FAILED);

    auto *tiling =
        context->GetTilingData<RebatchAndUpdateTilingData>();
    OPS_CHECK(tiling == nullptr,
              OPS_LOG_E(nodeName, "tiling data is nullptr."),
              return ge::GRAPH_FAILED);

    auto batchShape = context->GetInputShape(INPUT_BATCH_IDX);
    auto prefixIndexShape = context->GetInputShape(INPUT_PREFIX_INDEX_IDX);
    auto blockTableShape = context->GetInputShape(INPUT_BLOCK_TABLE_IDX);
    auto discardPositionsShape =
        context->GetInputShape(INPUT_DISCARD_POSITIONS_IDX);
    OPS_CHECK(batchShape == nullptr || prefixIndexShape == nullptr ||
                  blockTableShape == nullptr ||
                  discardPositionsShape == nullptr,
              OPS_LOG_E(nodeName, "required input shape is null."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(batchShape->GetStorageShape().GetDimNum() < 2 ||
                  prefixIndexShape->GetStorageShape().GetDimNum() != 1 ||
                  blockTableShape->GetStorageShape().GetDimNum() < 2 ||
                  discardPositionsShape->GetStorageShape().GetDimNum() < 2,
              OPS_LOG_E(nodeName, "required input rank is invalid."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(prefixIndexShape->GetStorageShape().GetDim(0) < *maxBs,
              OPS_LOG_E(nodeName, "prefixIndex capacity is smaller than max_bs."),
              return ge::GRAPH_FAILED);

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto &rebatch = tiling->rebatch.combineAndDispatchRebatchInfo;
    rebatch.isCombineRebatch = static_cast<uint32_t>(*isCombine);
    rebatch.hiddenSize = static_cast<uint32_t>(
        batchShape->GetStorageShape().GetDim(1));
    rebatch.maxBs = static_cast<uint32_t>(*maxBs);
    rebatch.k = static_cast<uint32_t>(*k);
    rebatch.totalUbSize = ubSize;

    auto &update = tiling->update.info;
    update.maxBs = static_cast<uint32_t>(*maxBs);
    update.originalBs = 0;
    update.tableSize = static_cast<uint32_t>(
        blockTableShape->GetStorageShape().GetDim(1));
    update.layerIdx = static_cast<uint32_t>(*layerIdx);
    update.maxDiscardBs = static_cast<uint32_t>(
        discardPositionsShape->GetStorageShape().GetDim(1));
    update.totalUbSize = ubSize;

    context->SetBlockDim(AIV_NUM);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaceSizes == nullptr,
              OPS_LOG_E(nodeName, "workspace sizes is nullptr."),
              return ge::GRAPH_FAILED);
    workspaceSizes[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

struct RebatchAndUpdateCompileInfo {};

static ge::graphStatus RebatchAndUpdateTilingParse(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RebatchAndUpdate)
    .Tiling(RebatchAndUpdateTiling)
    .TilingParse<RebatchAndUpdateCompileInfo>(
        RebatchAndUpdateTilingParse);

}  // namespace optiling