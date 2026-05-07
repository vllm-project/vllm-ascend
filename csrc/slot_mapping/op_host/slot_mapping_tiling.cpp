/**
 * @file slot_mapping_tiling.cpp
 * @brief SlotMapping TilingFunc 实现
 *
 * 接口变更：原先 numTokens / maxNumTokens / blockSize / blockTableStride 四个标量
 * 通过 int32 tensor 走 GM 传入（其中 num_tokens_t = full((max_num_tokens,), val)
 * 浪费 16 KB），此版本全部通过 Attr + shape 推导，减少 kernel 侧 GM load。
 *
 * Tensor 输入：queryStartLoc / positions / blockTable
 * Attr：numTokens, maxNumTokens, blockSize, totalCpWorldSize, totalCpRank,
 *       cpKvCacheInterleaveSize, padId
 *
 * 分核策略：per-position，每核 tilePerCore 个连续 position（对齐 8 × int64 = 64 B
 * cache line，避免 false sharing），`numBlocks = ceil(maxNumTokens/tilePerCore)`
 * 且不超过 coreNum。
 */

#include "slot_mapping_tiling.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

namespace optiling {

// Per-core ranges are aligned to one cache line (16 × int32 = 64 B) so that
// adjacent cores write to disjoint cache lines (no false sharing).
constexpr int32_t kSlotAlignment = 16;

// Input 索引
constexpr int IDX_QUERY_START_LOC = 0;
constexpr int IDX_POSITIONS = 1;
constexpr int IDX_BLOCK_TABLE = 2;

// Attr 索引（与 slot_mapping_def.cpp 顺序一致）
constexpr int ATTR_NUM_TOKENS = 0;
constexpr int ATTR_MAX_NUM_TOKENS = 1;
constexpr int ATTR_BLOCK_SIZE = 2;
constexpr int ATTR_TOTAL_CP_WORLD_SIZE = 3;
constexpr int ATTR_TOTAL_CP_RANK = 4;
constexpr int ATTR_CP_KV_CACHE_INTERLEAVE_SIZE = 5;
constexpr int ATTR_PAD_ID = 6;

static void GetCompileParameters(gert::TilingContext* context, uint32_t& coreNum)
{
    auto ptrCompileInfo = reinterpret_cast<const SlotMappingCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = ascendcPlatform.GetCoreNum();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_I(context, "Enter TilingFunc for SlotMapping");

    uint32_t coreNum = 0;
    GetCompileParameters(context, coreNum);
    if (coreNum == 0) {
        coreNum = 1;
    }

    // -------------------- Attrs --------------------
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(context, "Attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto getAttr = [&](int idx, int64_t defaultVal) -> int64_t {
        const int64_t* p = attrs->GetAttrPointer<int64_t>(idx);
        return p ? *p : defaultVal;
    };
    int64_t numTokens = getAttr(ATTR_NUM_TOKENS, 0);
    int64_t maxNumTokens = getAttr(ATTR_MAX_NUM_TOKENS, 0);
    int64_t blockSize = getAttr(ATTR_BLOCK_SIZE, 1);
    int64_t totalCpWorldSize = getAttr(ATTR_TOTAL_CP_WORLD_SIZE, 1);
    int64_t totalCpRank = getAttr(ATTR_TOTAL_CP_RANK, 0);
    int64_t cpKvCacheInterleaveSize = getAttr(ATTR_CP_KV_CACHE_INTERLEAVE_SIZE, 1);
    int64_t padId = getAttr(ATTR_PAD_ID, -1);

    if (maxNumTokens < 1) {
        maxNumTokens = 1;
    }
    if (numTokens < 0) {
        numTokens = 0;
    }
    if (numTokens > maxNumTokens) {
        numTokens = maxNumTokens;
    }
    if (blockSize < 1) {
        blockSize = 1;
    }
    if (totalCpWorldSize < 1) {
        totalCpWorldSize = 1;
    }
    if (cpKvCacheInterleaveSize < 1) {
        cpKvCacheInterleaveSize = 1;
    }

    // -------------------- Derived from tensor shapes --------------------
    const gert::StorageShape* queryStartLocShape = context->GetInputShape(IDX_QUERY_START_LOC);
    const gert::StorageShape* blockTableShape = context->GetInputShape(IDX_BLOCK_TABLE);

    int32_t numReqs = 1;
    if (queryStartLocShape != nullptr) {
        const gert::Shape& s = queryStartLocShape->GetStorageShape();
        if (s.GetDimNum() == 1 && s.GetDim(0) > 1) {
            numReqs = static_cast<int32_t>(s.GetDim(0)) - 1;
        }
    }
    if (numReqs < 1) {
        numReqs = 1;
    }

    int32_t blockTableStride = 1;
    if (blockTableShape != nullptr) {
        const gert::Shape& s = blockTableShape->GetStorageShape();
        if (s.GetDimNum() >= 2) {
            blockTableStride = static_cast<int32_t>(s.GetDim(1));
        }
    }

    // -------------------- 分核策略 --------------------
    int32_t tilePerCore = static_cast<int32_t>(
        (maxNumTokens + static_cast<int64_t>(coreNum) - 1) / static_cast<int64_t>(coreNum));
    if (tilePerCore < kSlotAlignment) {
        tilePerCore = kSlotAlignment;
    } else {
        tilePerCore = ((tilePerCore + kSlotAlignment - 1) / kSlotAlignment) * kSlotAlignment;
    }
    int32_t numBlocks = static_cast<int32_t>((maxNumTokens + tilePerCore - 1) / tilePerCore);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    if (numBlocks > static_cast<int32_t>(coreNum)) {
        numBlocks = static_cast<int32_t>(coreNum);
    }

    context->SetBlockDim(static_cast<uint32_t>(numBlocks));
    context->SetTilingKey(0);

    SlotMappingTilingData tiling;
    tiling.set_numReqs(numReqs);
    tiling.set_blockTableStride(blockTableStride);
    tiling.set_totalCpWorldSize(static_cast<int32_t>(totalCpWorldSize));
    tiling.set_totalCpRank(static_cast<int32_t>(totalCpRank));
    tiling.set_cpKvCacheInterleaveSize(static_cast<int32_t>(cpKvCacheInterleaveSize));
    tiling.set_padId(static_cast<int32_t>(padId));
    tiling.set_blockSize(static_cast<int32_t>(blockSize));
    tiling.set_maxNumTokens(static_cast<int32_t>(maxNumTokens));
    tiling.set_numTokens(static_cast<int32_t>(numTokens));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    OPS_LOG_I(context, "BlockDim=%d numReqs=%d tilePerCore=%d maxTokens=%ld numTokens=%ld bs=%ld",
              numBlocks, numReqs, tilePerCore, maxNumTokens, numTokens, blockSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SlotMapping(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<SlotMappingCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SlotMapping)
    .Tiling(TilingFunc)
    .TilingParse<SlotMappingCompileInfo>(TilingPrepare4SlotMapping);

}  // namespace optiling
