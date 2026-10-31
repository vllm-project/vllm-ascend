/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "fused_quant_lightning_indexer_manage_tiling.h"
#include <algorithm>
#include "../op_kernel/fused_quant_lightning_indexer_manage_template_tiling_key.h"
// s2 chunk 粒度与 kernel 侧单一来源(host 兼容 shim 见该头)。
#include "../op_kernel/arch22/fused_quant_lightning_indexer_manage_workspace.h"
#include <limits>
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_tiling.h"
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_workspace.h"
#include "../op_kernel/arch35/lim_payload_layout.h"
#include "../op_kernel/arch35/lim_simt_config.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

namespace arch35 {
namespace {
enum InputIndex : size_t {
    INDEX_WEIGHTS = 0, QUERY_DEQUANT_SCALE, QUERY,
    INDEX_KEY_DEQUANT_SCALE, INDEX_KEY_CACHE, INDEX_BLOCK_TABLE,
    ACTUAL_SEQ_LENGTHS_QUERY, ACTUAL_SEQ_LENGTHS_KEY,
    OFFLOAD_SEQ_LENGTHS_KEY, NUM_CACHE_TOKENS, REQUEST_STATE,
    REQ_POOL_ENTRIES, CACHE_SLOTS_POOL,
};
constexpr int64_t BLOCK_SIZE = lim_request_limits::BLOCK_SIZE;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t MAX_ROUTES = lim_request_limits::MAX_ROUTES;
constexpr int64_t MAX_SOURCE_CAPACITY = lim_payload_layout::MAX_SOURCE_CAPACITY;
constexpr size_t ATTR_KEY_STRIDE = 0;
constexpr size_t ATTR_SCALE_STRIDE = 1;
constexpr size_t ATTR_WEIGHT_STRIDE = 2;

bool Rank(const gert::Shape &shape, size_t rank) { return shape.GetDimNum() == rank; }
}

static ge::graphStatus TilingFusedQuantLightningIndexerManage(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) return ge::GRAPH_FAILED;
    for (size_t i = 0; i <= CACHE_SLOTS_POOL; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr)
            return ge::GRAPH_FAILED;
    }
    const auto dtype = [context](size_t i) { return context->GetInputDesc(i)->GetDataType(); };
    if (dtype(INDEX_WEIGHTS) != ge::DT_BF16 ||
        dtype(QUERY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(QUERY) != ge::DT_FLOAT8_E4M3FN ||
        dtype(INDEX_KEY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(INDEX_KEY_CACHE) != ge::DT_FLOAT8_E4M3FN) return ge::GRAPH_FAILED;
    for (size_t i = INDEX_BLOCK_TABLE; i <= CACHE_SLOTS_POOL; ++i)
        if (dtype(i) != ge::DT_INT32) return ge::GRAPH_FAILED;

    const auto weights = context->GetInputShape(INDEX_WEIGHTS)->GetStorageShape();
    const auto queryScale = context->GetInputShape(QUERY_DEQUANT_SCALE)->GetStorageShape();
    const auto query = context->GetInputShape(QUERY)->GetStorageShape();
    const auto keyScale = context->GetInputShape(INDEX_KEY_DEQUANT_SCALE)->GetStorageShape();
    const auto key = context->GetInputShape(INDEX_KEY_CACHE)->GetStorageShape();
    const auto blockTable = context->GetInputShape(INDEX_BLOCK_TABLE)->GetStorageShape();
    const auto queryEnds = context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const auto pool = context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();
    if (!Rank(weights, 2) || !Rank(queryScale, 2) || !Rank(query, 3) ||
        !Rank(keyScale, 3) || !Rank(key, 4) || !Rank(blockTable, 2) ||
        !Rank(queryEnds, 1) || !Rank(pool, 2)) return ge::GRAPH_FAILED;
    const int64_t totalQueries = query.GetDim(0);
    const int64_t heads = query.GetDim(1);
    const int64_t batch = queryEnds.GetDim(0);
    const int64_t sourceCapacity = blockTable.GetDim(1) * BLOCK_SIZE;
    if (batch <= 0 || totalQueries < batch || totalQueries > batch * MAX_ROUTES ||
        (heads != 32 && heads != 64) || query.GetDim(2) != HEAD_DIM ||
        weights.GetDim(0) != totalQueries || weights.GetDim(1) != heads ||
        queryScale.GetDim(0) != totalQueries || queryScale.GetDim(1) != heads ||
        key.GetDim(0) <= 0 || key.GetDim(1) != BLOCK_SIZE ||
        key.GetDim(2) != 1 || key.GetDim(3) != HEAD_DIM ||
        keyScale.GetDim(0) != key.GetDim(0) || keyScale.GetDim(1) != BLOCK_SIZE ||
        keyScale.GetDim(2) != 1 || blockTable.GetDim(0) != batch ||
        sourceCapacity <= 0 || sourceCapacity > MAX_SOURCE_CAPACITY ||
        pool.GetDim(0) <= 0 || pool.GetDim(1) != sourceCapacity)
        return ge::GRAPH_FAILED;
    for (size_t i = ACTUAL_SEQ_LENGTHS_KEY; i <= REQ_POOL_ENTRIES; ++i) {
        const auto shape = context->GetInputShape(i)->GetStorageShape();
        if (!Rank(shape, 1) || shape.GetDim(0) != batch) return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const int64_t *keyStride = attrs->GetAttrPointer<int64_t>(ATTR_KEY_STRIDE);
    const int64_t *scaleStride = attrs->GetAttrPointer<int64_t>(ATTR_SCALE_STRIDE);
    const int64_t *weightStride = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_STRIDE);
    if (keyStride == nullptr || scaleStride == nullptr || weightStride == nullptr ||
        *keyStride < BLOCK_SIZE * HEAD_DIM || *scaleStride < BLOCK_SIZE ||
        *weightStride < heads ||
        static_cast<uint64_t>(*keyStride) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*scaleStride) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*weightStride) > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t mixGroups = std::min<uint32_t>(
        platform.GetCoreNumAic(), platform.GetCoreNumAiv() / 2U);
    if (mixGroups == 0U) return ge::GRAPH_FAILED;
    // A5 switches 32-head launches to one four-route tile above 64 query
    // rows. Also switch when two reserved tiles per request would exceed
    // the available mix groups: Q=1/2 leaves every second tile empty, so
    // modulo scheduling would serialize requests on only half the cores.
    const uint32_t queryTileSize =
        heads == 32 &&
            (totalQueries > 64 || static_cast<uint32_t>(batch) * 2U > mixGroups)
        ? 4U : 2U;
    // The LI phase balances the flattened S2 block pool across mix groups;
    // a group may process more than one query tile. Reserve the full launch.
    const uint32_t usedCores = mixGroups;
    // Shape alone does not reveal the largest per-request Q. Reserve enough
    // rows for either the original B x 4 layout or the compact T-row layout.
    const uint64_t routeStorageRows = std::max<uint64_t>(
        4U * static_cast<uint64_t>(batch),
        static_cast<uint64_t>(totalQueries));
    const uint64_t scoreStride =
        4U * static_cast<uint64_t>(sourceCapacity) * sizeof(uint16_t);
    const uint64_t candidateSegments =
        fused_quant_li_manage_workspace::CandidateSegmentCapacity(
            static_cast<uint64_t>(sourceCapacity), usedCores);
    const uint64_t qliWorkspace =
        fused_quant_li_manage_workspace::TotalBytes(
            static_cast<uint64_t>(sourceCapacity), routeStorageRows,
            candidateSegments,
            sourceCapacity > lim_payload_layout::PACKED_SOURCE_CAPACITY);
    if (routeStorageRows > std::numeric_limits<uint32_t>::max() ||
        scoreStride > std::numeric_limits<uint32_t>::max() ||
        qliWorkspace > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;
    auto *tiling = context->GetTilingData<::FusedQuantLightningIndexerManageTilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr) return ge::GRAPH_FAILED;
    tiling->usedCoreNum = usedCores;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->totalQueryRows = static_cast<uint32_t>(totalQueries);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(sourceCapacity);
    tiling->indexHeads = static_cast<uint32_t>(heads);
    tiling->maxBlockNumPerBatch = static_cast<uint32_t>(blockTable.GetDim(1));
    tiling->weightStride = static_cast<uint32_t>(*weightStride);
    tiling->keyStride = static_cast<uint32_t>(*keyStride);
    tiling->scaleStride = static_cast<uint32_t>(*scaleStride);
    tiling->managementWorkspaceOffset = static_cast<uint32_t>(qliWorkspace);
    tiling->scoreWorkspaceStride = static_cast<uint32_t>(scoreStride);
    tiling->scoreRowStride = static_cast<uint32_t>(sourceCapacity);
    tiling->queryTileSize = queryTileSize;
    tiling->routeStorageRows = static_cast<uint32_t>(routeStorageRows);
    tiling->compactRouteRows = 0U;
    const uint64_t managementWorkspace =
        static_cast<uint64_t>(batch + totalQueries) * 32U;
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        qliWorkspace + managementWorkspace;
    context->SetBlockDim(platform.CalcTschBlockDim(
        usedCores * 2U, usedCores, usedCores * 2U));
    context->SetScheduleMode(1);
#if LIM_SIMT_POSTPROCESS
    // Match the A5 reference: reserve 32 KiB for SIMT DCache and 8 KiB
    // for runtime/stack so TPipe UB buffers cannot overlap those regions.
    if (context->SetLocalMemorySize(LIM_SIMT_USER_UB_BYTES) !=
        ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
#endif
    context->GetRawTilingData()->SetDataSize(sizeof(*tiling));
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN)));
    return ge::GRAPH_SUCCESS;
}
} // namespace arch35


ge::graphStatus FusedQuantLightningIndexerManageTiling::GetNpuInfo(FusedQuantLightningIndexerManageTilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("FusedQuantLightningIndexerManage", "opName got from TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = context_->GetNodeName();
    tilingInfo.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(tilingInfo.platformInfo == nullptr, OPS_LOG_E(tilingInfo.opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(tilingInfo.opName, "num of core obtained is 0."),
               return ge::GRAPH_FAILED);

    tilingInfo.socVersion = ascendcPlatform.GetSocVersion();
    OPS_ERR_IF((tilingInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) &&
                   (tilingInfo.socVersion != platform_ascendc::SocVersion::ASCEND910_93),
               OPS_LOG_E(tilingInfo.opName, "SOC Version[%d] is not supported.",
                         static_cast<int32_t>(tilingInfo.socVersion)),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(tilingInfo.opName, "workspace size buffer is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(tilingInfo.opName, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedQuantLightningIndexerManageTiling::GetTensorInfo(FusedQuantLightningIndexerManageTilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(KEY_INDEX);
    op.key.shape = context_->GetInputShape(KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    op.reqPoolEntries.desc = context_->GetInputDesc(REQ_POOL_ENTRIES_INDEX);
    op.reqPoolEntries.shape = context_->GetInputShape(REQ_POOL_ENTRIES_INDEX);
    op.cacheSlots.desc = context_->GetInputDesc(CACHE_SLOTS_INDEX);
    op.cacheSlots.shape = context_->GetInputShape(CACHE_SLOTS_INDEX);
    op.cacheTokens.desc = context_->GetInputDesc(CACHE_TOKENS_INDEX);
    op.cacheTokens.shape = context_->GetInputShape(CACHE_TOKENS_INDEX);
    op.actualSeqLengths.desc = context_->GetInputDesc(OFFLOAD_SEQ_K_INDEX);
    op.actualSeqLengths.shape = context_->GetInputShape(OFFLOAD_SEQ_K_INDEX);
    op.blockTable.desc = context_->GetInputDesc(BLOCK_TABLE_INDEX);
    op.blockTable.shape = context_->GetInputShape(BLOCK_TABLE_INDEX);
    op.topkIndexOut.desc = context_->GetOutputDesc(TOPK_INDEX);
    op.topkIndexOut.shape = context_->GetOutputShape(TOPK_INDEX);
    op.topkSlotsOut.desc = context_->GetOutputDesc(TOPK_SLOTS_INDEX);
    op.topkSlotsOut.shape = context_->GetOutputShape(TOPK_SLOTS_INDEX);
    uint32_t missCountIndex = mtp_ ? 5U : MISS_COUNT_INDEX;
    uint32_t cacheSlotsOutIndex = mtp_ ? 6U : CACHE_SLOTS_OUT_INDEX;
    if (mtp_) {
        op.topkMissCountOut.desc = context_->GetOutputDesc(2U);
        op.topkMissCountOut.shape = context_->GetOutputShape(2U);
        op.missSrcOut.desc = context_->GetOutputDesc(3U);
        op.missSrcOut.shape = context_->GetOutputShape(3U);
        op.missSlotsOut.desc = context_->GetOutputDesc(4U);
        op.missSlotsOut.shape = context_->GetOutputShape(4U);
    }
    op.missCountOut.desc = context_->GetOutputDesc(missCountIndex);
    op.missCountOut.shape = context_->GetOutputShape(missCountIndex);
    op.cacheSlotsOut.desc = context_->GetOutputDesc(cacheSlotsOutIndex);
    op.cacheSlotsOut.shape = context_->GetOutputShape(cacheSlotsOutIndex);

    OPS_ERR_IF(op.query.desc == nullptr || op.query.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.key.desc == nullptr || op.key.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc == nullptr || op.weights.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "weights desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.reqPoolEntries.desc == nullptr || op.reqPoolEntries.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc == nullptr || op.cacheSlots.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheTokens.desc == nullptr || op.cacheTokens.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc == nullptr || op.actualSeqLengths.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "offload_seq_lengths_key desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc == nullptr || op.topkIndexOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_index desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkSlotsOut.desc == nullptr || op.topkSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missCountOut.desc == nullptr || op.missCountOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_count desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlotsOut.desc == nullptr || op.cacheSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots output desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && (op.topkMissCountOut.desc == nullptr ||
                        op.topkMissCountOut.shape == nullptr ||
                        op.missSrcOut.desc == nullptr || op.missSrcOut.shape == nullptr ||
                        op.missSlotsOut.desc == nullptr || op.missSlotsOut.shape == nullptr),
               OPS_LOG_E(tilingInfo.opName, "MTP union miss outputs are nullptr."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedQuantLightningIndexerManageTiling::CheckDtype(const FusedQuantLightningIndexerManageTilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    ge::DataType qType = op.query.desc->GetDataType();
    ge::DataType kType = op.key.desc->GetDataType();
    ge::DataType wType = op.weights.desc->GetDataType();
    // C8: query/key int8, index weights stay bf16 (head-merge operand of mma2)
    OPS_ERR_IF(qType != kType || qType != ge::DT_INT8,
               OPS_LOG_E(tilingInfo.opName, "C8 query/key dtype must be int8 and match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wType != ge::DT_BF16,
               OPS_LOG_E(tilingInfo.opName, "C8 weights dtype must be bf16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX) == nullptr ||
                   context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX) == nullptr ||
                   context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT16 ||
                   context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX)->GetDataType() != ge::DT_FLOAT16,
               OPS_LOG_E(tilingInfo.opName, "C8 dequant scales must be fp16."),
               return ge::GRAPH_FAILED);
    for (uint32_t input = BLOCK_TABLE_INDEX; input <= CACHE_SLOTS_INDEX; ++input) {
        OPS_ERR_IF(context_->GetInputDesc(input) == nullptr ||
                       context_->GetInputDesc(input)->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(tilingInfo.opName, "metadata inputs must be int32."),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(op.cacheSlots.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "cache_slots dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.reqPoolEntries.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheTokens.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "offload_seq_lengths_key dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "block_table dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.topkSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   (mtp_ && op.topkMissCountOut.desc->GetDataType() != ge::DT_INT32) ||
                   op.missCountOut.desc->GetDataType() != ge::DT_INT32 ||
                    op.cacheSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                    (mtp_ && (op.missSrcOut.desc->GetDataType() != ge::DT_INT32 ||
                              op.missSlotsOut.desc->GetDataType() != ge::DT_INT32)),
               OPS_LOG_E(tilingInfo.opName,
                         "topk_index/topk_slots/miss_count/cache_slots output must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedQuantLightningIndexerManageTiling::CheckShape(FusedQuantLightningIndexerManageTilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &reqPoolShape = op.reqPoolEntries.shape->GetStorageShape();
    const auto &cacheShape = op.cacheSlots.shape->GetStorageShape();
    const auto &cacheTokensShape = op.cacheTokens.shape->GetStorageShape();
    const auto &seqShape = op.actualSeqLengths.shape->GetStorageShape();
    const auto &blockShape = op.blockTable.shape->GetStorageShape();
    const auto &indexOutShape = op.topkIndexOut.shape->GetStorageShape();
    const auto &slotsOutShape = op.topkSlotsOut.shape->GetStorageShape();
    const auto *topkMissCountOutShape =
        mtp_ ? &op.topkMissCountOut.shape->GetStorageShape() : nullptr;
    const auto &missCountOutShape = op.missCountOut.shape->GetStorageShape();
    const auto &cacheSlotsOutShape = op.cacheSlotsOut.shape->GetStorageShape();
    const auto *missSrcShape = mtp_ ? &op.missSrcOut.shape->GetStorageShape() : nullptr;
    const auto *missSlotsShape = mtp_ ? &op.missSlotsOut.shape->GetStorageShape() : nullptr;
    const gert::StorageShape *queryScaleStorage = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    const gert::StorageShape *keyScaleStorage = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);
    OPS_ERR_IF(queryScaleStorage == nullptr || keyScaleStorage == nullptr,
               OPS_LOG_E(tilingInfo.opName, "dequant scale shapes are nullptr."),
               return ge::GRAPH_FAILED);
    const auto &queryScaleShape = queryScaleStorage->GetStorageShape();
    const auto &keyScaleShape = keyScaleStorage->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be TND [B, N1, 128], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights must be [B, N1], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(queryScaleShape.GetDimNum() != DIM_NUM_TWO ||
                   queryScaleShape.GetDim(0) != qShape.GetDim(0) ||
                   queryScaleShape.GetDim(1) != qShape.GetDim(1),
               OPS_LOG_E(tilingInfo.opName, "query_dequant_scale must be [T,N]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyScaleShape.GetDimNum() != DIM_NUM_THREE ||
                   keyScaleShape.GetDim(0) != kShape.GetDim(0) ||
                   keyScaleShape.GetDim(1) != kShape.GetDim(1) ||
                   keyScaleShape.GetDim(2) != kShape.GetDim(2),
               OPS_LOG_E(tilingInfo.opName, "index_key_dequant_scale must be [blocks,128,1]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots must be [pool_size, source_capacity]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(reqPoolShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries must be rank 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheTokensShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens must be rank 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "offload_seq_lengths_key must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDimNum() != DIM_NUM_THREE || slotsOutShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must be [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "miss_count must be [B]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots output must be rank 2."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && topkMissCountOutShape->GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "topk_miss_counts must be [T]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && (missSrcShape->GetDimNum() != DIM_NUM_TWO ||
                        missSlotsShape->GetDimNum() != DIM_NUM_TWO),
               OPS_LOG_E(tilingInfo.opName, "MTP miss outputs must be [B, 32768]."),
               return ge::GRAPH_FAILED);

    tilingInfo.tSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.bSize = static_cast<uint32_t>(blockShape.GetDim(0));
    tilingInfo.n1Size = static_cast<uint32_t>(qShape.GetDim(1));
    tilingInfo.n2Size = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_TWO));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(DIM_IDX_ONE));
    tilingInfo.s2Size = 0;
    tilingInfo.poolSize = static_cast<uint32_t>(cacheShape.GetDim(0));
    tilingInfo.cacheSlotsSize = static_cast<uint32_t>(cacheShape.GetDim(1));

    OPS_ERR_IF(tilingInfo.bSize == 0 || tilingInfo.tSize < tilingInfo.bSize ||
                   tilingInfo.tSize > tilingInfo.bSize * 14U,
               OPS_LOG_E(tilingInfo.opName, "requires B <= T <= 14B."),
               return ge::GRAPH_FAILED);
    uint32_t metadataBatch = tilingInfo.bSize;
    for (uint32_t input = ACTUAL_SEQ_Q_INDEX; input <= REQ_POOL_ENTRIES_INDEX; ++input) {
        const gert::StorageShape *storage = context_->GetInputShape(input);
        OPS_ERR_IF(storage == nullptr || storage->GetStorageShape().GetDimNum() != DIM_NUM_ONE ||
                       storage->GetStorageShape().GetDim(0) != metadataBatch,
                   OPS_LOG_E(tilingInfo.opName, "request metadata tensors must be [B]."),
                   return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(reqPoolShape.GetShapeSize() != metadataBatch ||
                    cacheTokensShape.GetShapeSize() != metadataBatch ||
                    seqShape.GetShapeSize() != metadataBatch || blockShape.GetDim(0) != metadataBatch,
               OPS_LOG_E(tilingInfo.opName,
                         "query batch, req_pool_entries, cache_tokens, sequence lengths, and block_table batch must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.poolSize == 0 || tilingInfo.cacheSlotsSize == 0,
               OPS_LOG_E(tilingInfo.opName, "cache_slots dimensions must be positive."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.maxBlockNumPerBatch == 0,
               OPS_LOG_E(tilingInfo.opName, "block_table must contain at least one block per request."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.maxBlockNumPerBatch > (1U << 14),
               OPS_LOG_E(tilingInfo.opName, "block_table capacity must be <= 16384 blocks for the 21-bit source format."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize != 128,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be 128."),
               return ge::GRAPH_FAILED);
    const uint64_t sourceCapacity =
        static_cast<uint64_t>(tilingInfo.blockSize) *
        tilingInfo.maxBlockNumPerBatch;
    OPS_ERR_IF(sourceCapacity > (1U << 21),
               OPS_LOG_E(tilingInfo.opName,
                          "cache_slots capacity must be <= 2^21 tokens."),
               return ge::GRAPH_FAILED);
    tilingInfo.s2Size = static_cast<uint32_t>(sourceCapacity);
    OPS_ERR_IF(tilingInfo.s2Size != tilingInfo.cacheSlotsSize,
               OPS_LOG_E(tilingInfo.opName,
                          "cache_slots capacity must equal table capacity."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n2Size != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n1Size != 32 && tilingInfo.n1Size != 64,
               OPS_LOG_E(tilingInfo.opName, "decode query N1 must be 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(DIM_IDX_TWO) != DECODE_HEAD_DIM || kShape.GetDim(DIM_IDX_THREE) != DECODE_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.tSize || wShape.GetDim(1) != tilingInfo.n1Size,
               OPS_LOG_E(tilingInfo.opName, "weights must match query [T, N1]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDim(0) != tilingInfo.tSize || indexOutShape.GetDim(1) != DECODE_N2 ||
                    indexOutShape.GetDim(2) != DECODE_OUTPUT_CAPACITY ||
                    slotsOutShape.GetDim(0) != tilingInfo.tSize || slotsOutShape.GetDim(1) != DECODE_N2 ||
                    slotsOutShape.GetDim(2) != DECODE_OUTPUT_CAPACITY,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must have shape [T, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDim(0) != metadataBatch,
               OPS_LOG_E(tilingInfo.opName, "miss_count must have shape [B]."),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && topkMissCountOutShape->GetDim(0) != tilingInfo.tSize,
               OPS_LOG_E(tilingInfo.opName,
                         "topk_miss_counts must have shape [T]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(mtp_ && (missSrcShape->GetDim(0) != metadataBatch || missSrcShape->GetDim(1) != 32768 ||
                        missSlotsShape->GetDim(0) != metadataBatch || missSlotsShape->GetDim(1) != 32768),
               OPS_LOG_E(tilingInfo.opName, "MTP miss outputs must have shape [B, 32768]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDim(0) != cacheShape.GetDim(0) ||
                   cacheSlotsOutShape.GetDim(1) != cacheShape.GetDim(1),
               OPS_LOG_E(tilingInfo.opName,
                         "cache_slots output must match the request-state pool shape."),
               return ge::GRAPH_FAILED);

    tilingInfo.inputQType = op.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedQuantLightningIndexerManageTiling::ParseAndCheck(FusedQuantLightningIndexerManageTilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedQuantLightningIndexerManageTiling::DoTiling(FusedQuantLightningIndexerManageTilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    constexpr uint32_t SCHEDULE_BALANCED = 1;
    constexpr uint32_t SCHEDULE_AUTO = 2;
    uint32_t selectedScheduleMode =
        (tilingInfo->bSize % aicNum) == 0U ? SCHEDULE_AUTO : SCHEDULE_BALANCED;
    tilingInfo->usedCoreNum =
        selectedScheduleMode == SCHEDULE_BALANCED ? aicNum : std::min(tilingInfo->bSize, aicNum);
    uint32_t requestedAivNum = std::min(aivNum, tilingInfo->usedCoreNum * 2U);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(requestedAivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t DOUBLE_BUFFER = 2;
    constexpr uint32_t M_BASE_SIZE = 64;
    constexpr uint64_t S2_BASE_SIZE = MtpC8Workspace::S2_BASE_SIZE;
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (mtp_) {
        // C8 两段mma后 mm1Res 已是每核 [s1BaseSize=8, s2BaseSize] 的 fp32 最终
        // 分数行 (双缓冲), 另加 vec0 折叠的 w*q_scale workspace (mBaseSize * 16
        // elements, 每核单缓冲)。与 kernel.h Init / MtpC8Workspace 保持同序。
        constexpr uint32_t MTP_S1_BASE_SIZE = 8U;
        constexpr uint32_t MTP_BLOCK_CUBE = 16U;
        const uint32_t MTP_M_BASE_SIZE = 8U * tilingInfo->n1Size;
        workspaceSize += static_cast<uint64_t>(MTP_S1_BASE_SIZE) * S2_BASE_SIZE *
                         MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * blockDim;
        workspaceSize += static_cast<uint64_t>(MTP_M_BASE_SIZE) * MTP_BLOCK_CUBE * sizeof(uint16_t) * blockDim;
    } else {
        workspaceSize += M_BASE_SIZE * S2_BASE_SIZE * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * blockDim;
    }
    uint64_t scoreStride = ((static_cast<uint64_t>(tilingInfo->s2Size) + S2_BASE_SIZE - 1) / S2_BASE_SIZE) *
                           S2_BASE_SIZE;
    if (!mtp_) {
        workspaceSize += static_cast<uint64_t>(tilingInfo->bSize) * scoreStride * sizeof(float);
    }
    if (mtp_) {
        constexpr uint32_t S1_BASE_SIZE = 8;
        constexpr uint32_t LD_HEAD_TAIL = 2;
        constexpr uint32_t VALUE_AND_INDEX = 2;
        constexpr uint32_t LD_PARAM_NUM = 16;
        workspaceSize += static_cast<uint64_t>(blockDim) * LD_HEAD_TAIL * S1_BASE_SIZE *
                         VALUE_AND_INDEX * DECODE_SPARSE_COUNT * sizeof(float);
        workspaceSize += static_cast<uint64_t>(blockDim) * LD_HEAD_TAIL * S1_BASE_SIZE *
                         LD_PARAM_NUM * sizeof(int64_t);
        constexpr uint32_t MTP_PAIR_CAPACITY = 4U * DECODE_SPARSE_COUNT;
        constexpr uint32_t MTP_THRESHOLD_STRIDE = 8U;
        constexpr uint32_t MTP_ROUTE_COUNT_STRIDE = 8U;
        const uint64_t metadataBatch = tilingInfo->bSize;
        // Keep this order synchronized with MtpC8Workspace::{Pair0Offset,
        // Pair1Offset, ScoreOffset, ThresholdOffset, RouteCountOffset} in the kernel.
        workspaceSize += metadataBatch * MTP_PAIR_CAPACITY * sizeof(float) * 2U;
        workspaceSize += static_cast<uint64_t>(tilingInfo->tSize) * scoreStride * sizeof(float);
        workspaceSize += static_cast<uint64_t>(tilingInfo->tSize) * MTP_THRESHOLD_STRIDE * sizeof(float);
        workspaceSize += static_cast<uint64_t>(tilingInfo->tSize) * MTP_ROUTE_COUNT_STRIDE * sizeof(int32_t);
    }
    constexpr uint32_t PARTIAL_SLOTS_PER_CORE = 2;
    constexpr uint32_t PARTIAL_META_INTS_PER_CORE = 8;
    constexpr uint32_t TOPK_PAIR_ELEMS = DECODE_SPARSE_COUNT * 2;
    if (!mtp_) {
        workspaceSize +=
            static_cast<uint64_t>(blockDim) * PARTIAL_SLOTS_PER_CORE * TOPK_PAIR_ELEMS * sizeof(float);
        workspaceSize += static_cast<uint64_t>(blockDim) * PARTIAL_META_INTS_PER_CORE * sizeof(int32_t);
    }
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_tSize(tilingInfo->tSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_poolSize(tilingInfo->poolSize);
    tilingData_.set_n1Size(tilingInfo->n1Size);
    tilingData_.set_cacheSlotsSize(tilingInfo->cacheSlotsSize);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_scheduleMode(selectedScheduleMode);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(tilingInfo->inputQType));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForFusedQuantLightningIndexerManage(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForFusedQuantLightningIndexerManage(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("FusedQuantLightningIndexerManage", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    auto *platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
               OPS_LOG_E("FusedQuantLightningIndexerManage", "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        return arch35::TilingFusedQuantLightningIndexerManage(context);
    }
    // arch22 retains the existing A2/A3 validation and tiling implementation.
    FusedQuantLightningIndexerManageTilingInfo liInfo;
    FusedQuantLightningIndexerManageTiling liTiling(context, true);
    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(FusedQuantLightningIndexerManage)
    .Tiling(TilingForFusedQuantLightningIndexerManage)
    .TilingParse<FusedQuantLightningIndexerManageCompileInfo>(TilingPrepareForFusedQuantLightningIndexerManage);

} // namespace optiling
