#include "first_fill_scatter_copy_tiling.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "error/ops_error.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint32_t HBM_K_ROPE = 0;
constexpr uint32_t HBM_KV_CACHE = 1;
constexpr uint32_t DRAM_K_ROPE = 2;
constexpr uint32_t DRAM_KV_CACHE = 3;
constexpr uint32_t HBM_BLOCK_TABLE = 4;
constexpr uint32_t DRAM_BLOCK_TABLE = 5;
constexpr uint32_t MISS_SOURCE_IDS = 6;
constexpr uint32_t MISS_DST_SLOTS = 7;
constexpr uint32_t MISS_COUNTS = 8;
constexpr uint32_t CACHE_TOKENS = 9;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t KPE_DIM = 64;
constexpr int64_t CKV_DIM = 512;
// Fixed public request-level miss metadata capacity for MTP0..MTP15.
constexpr int64_t MISS_CAPACITY = 32768;
constexpr size_t WORKSPACE_SIZE = 32;

bool IsShape(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dim : dims) {
        if (dim >= 0 && shape.GetDim(index) != dim) {
            return false;
        }
        ++index;
    }
    return true;
}

ge::graphStatus TilingFirstFillScatterCopy(
    gert::TilingContext *context)
{
    const auto hbmRopeInput = context->GetInputShape(HBM_K_ROPE);
    const auto hbmKvInput = context->GetInputShape(HBM_KV_CACHE);
    const auto dramRopeInput = context->GetInputShape(DRAM_K_ROPE);
    const auto dramKvInput = context->GetInputShape(DRAM_KV_CACHE);
    const auto hbmTableInput = context->GetInputShape(HBM_BLOCK_TABLE);
    const auto dramTableInput = context->GetInputShape(DRAM_BLOCK_TABLE);
    const auto sourceInput = context->GetInputShape(MISS_SOURCE_IDS);
    const auto destinationInput = context->GetInputShape(MISS_DST_SLOTS);
    const auto countInput = context->GetInputShape(MISS_COUNTS);
    const auto cacheInput = context->GetInputShape(CACHE_TOKENS);
    OPS_ERR_IF(hbmRopeInput == nullptr || hbmKvInput == nullptr ||
                   dramRopeInput == nullptr || dramKvInput == nullptr ||
                   hbmTableInput == nullptr || dramTableInput == nullptr ||
                   sourceInput == nullptr || destinationInput == nullptr ||
                   countInput == nullptr || cacheInput == nullptr,
               OPS_LOG_E(context->GetNodeName(),
                         "A required first-fill copy input shape is missing."),
               return ge::GRAPH_FAILED);

    const gert::Shape hbmRope = hbmRopeInput->GetStorageShape();
    const gert::Shape hbmKv = hbmKvInput->GetStorageShape();
    const gert::Shape dramRope = dramRopeInput->GetStorageShape();
    const gert::Shape dramKv = dramKvInput->GetStorageShape();
    const gert::Shape hbmTable = hbmTableInput->GetStorageShape();
    const gert::Shape dramTable = dramTableInput->GetStorageShape();
    const gert::Shape source = sourceInput->GetStorageShape();
    const gert::Shape destination = destinationInput->GetStorageShape();
    const gert::Shape counts = countInput->GetStorageShape();
    const gert::Shape cacheTokens = cacheInput->GetStorageShape();

    OPS_ERR_IF(!IsShape(counts, {-1}) || counts.GetDim(0) <= 0,
               OPS_LOG_E(context->GetNodeName(),
                         "miss_counts must be [B], B > 0."),
               return ge::GRAPH_FAILED);
    const int64_t batchSize = counts.GetDim(0);
    OPS_ERR_IF(!IsShape(hbmRope, {-1, BLOCK_SIZE, 1, KPE_DIM}) ||
                   !IsShape(hbmKv, {hbmRope.GetDim(0), BLOCK_SIZE, 1, CKV_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "HBM KPE/CKV must be [blocks,128,1,64/512]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(dramRope, {-1, BLOCK_SIZE, KPE_DIM}) ||
                   !IsShape(dramKv, {dramRope.GetDim(0), BLOCK_SIZE, CKV_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "DRAM KPE/CKV must be [blocks,128,64/512]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(hbmTable, {batchSize, -1}) || hbmTable.GetDim(1) <= 0 ||
                   !IsShape(dramTable, {batchSize, -1}) || dramTable.GetDim(1) <= 0 ||
                   !IsShape(source, {batchSize, MISS_CAPACITY}) ||
                   !IsShape(destination, {batchSize, MISS_CAPACITY}) ||
                   !IsShape(cacheTokens, {batchSize}),
               OPS_LOG_E(context->GetNodeName(),
                         "First-fill tables/metadata have inconsistent shapes."),
               return ge::GRAPH_FAILED);

    const auto hbmRopeDesc = context->GetInputDesc(HBM_K_ROPE);
    OPS_ERR_IF(hbmRopeDesc == nullptr,
               OPS_LOG_E(context->GetNodeName(),
                         "HBM KPE descriptor is missing."),
               return ge::GRAPH_FAILED);
    const ge::DataType floatingType = hbmRopeDesc->GetDataType();
    OPS_ERR_IF(floatingType != ge::DT_BF16 && floatingType != ge::DT_FLOAT16,
               OPS_LOG_E(context->GetNodeName(),
                         "First-fill KV tensors must be bf16/fp16."),
               return ge::GRAPH_FAILED);
    for (uint32_t index : {HBM_KV_CACHE, DRAM_K_ROPE, DRAM_KV_CACHE}) {
        const auto desc = context->GetInputDesc(index);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != floatingType,
                   OPS_LOG_E(context->GetNodeName(),
                             "All first-fill KV tensors must share one dtype."),
                   return ge::GRAPH_FAILED);
    }
    for (uint32_t index : {HBM_BLOCK_TABLE, DRAM_BLOCK_TABLE,
                           MISS_SOURCE_IDS, MISS_DST_SLOTS,
                           MISS_COUNTS, CACHE_TOKENS}) {
        const auto desc = context->GetInputDesc(index);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(context->GetNodeName(),
                             "First-fill metadata must be int32."),
                   return ge::GRAPH_FAILED);
    }

    const auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Platform info is missing."),
               return ge::GRAPH_FAILED);
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t availableCores = platform.GetCoreNumAiv();
    OPS_ERR_IF(availableCores <= 0,
               OPS_LOG_E(context->GetNodeName(),
                         "First-fill copy requires at least one AIV core."),
               return ge::GRAPH_FAILED);

    FirstFillScatterCopyTilingData tiling;
    const int64_t totalPairSlots = batchSize * MISS_CAPACITY;
    const int64_t usedCoreNum =
        totalPairSlots < availableCores ? totalPairSlots : availableCores;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_totalPairSlots(totalPairSlots);
    tiling.set_batchSize(batchSize);
    tiling.set_copyCap(MISS_CAPACITY);
    tiling.set_hbmMaxBlockNum(hbmTable.GetDim(1));
    tiling.set_dramMaxBlockNum(dramTable.GetDim(1));

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1U);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Workspace array is missing."),
               return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;
    const auto raw = context->GetRawTilingData();
    OPS_ERR_IF(raw == nullptr || raw->GetCapacity() < tiling.GetDataSize(),
               OPS_LOG_E(context->GetNodeName(),
                         "First-fill tiling buffer is unavailable or too small."),
               return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareFirstFillScatterCopy(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(FirstFillScatterCopy)
    .Tiling(TilingFirstFillScatterCopy)
    .TilingParse<FirstFillScatterCopyCompileInfo>(
        TilingPrepareFirstFillScatterCopy);

}  // namespace optiling
