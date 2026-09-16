#include "fused_copy_sfa_mtp_tiling.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "error/ops_error.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {

constexpr uint32_t QUERY = 0;
constexpr uint32_t KEY = 1;
constexpr uint32_t VALUE = 2;
constexpr uint32_t SPARSE_INDICES = 3;
constexpr uint32_t CACHE_TOKENS = 4;
constexpr uint32_t HBM_BLOCK_TABLE = 5;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_QUERY = 6;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_KV = 7;
constexpr uint32_t QUERY_ROPE = 8;
constexpr uint32_t HBM_KEY_ROPE = 9;
constexpr uint32_t DRAM_KEY_ROPE = 10;
constexpr uint32_t DRAM_KV_CACHE = 11;
constexpr uint32_t DRAM_BLOCK_TABLE = 12;
constexpr uint32_t TOPK_SOURCE_IDS = 13;
constexpr uint32_t TOPK_MISS_COUNTS = 14;
constexpr uint32_t MISS_SOURCE_IDS = 15;
constexpr uint32_t MISS_DST_SLOTS = 16;
constexpr uint32_t MISS_COUNTS = 17;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t CKV_DIM = 512;
constexpr int64_t KPE_DIM = 64;
constexpr int64_t MAX_QUERY_COUNT = 16;
constexpr int64_t SPARSE_COUNT = 2048;
// MTP15 has sixteen TopK=2048 query rows per request.
constexpr int64_t MISS_CAPACITY = 32768;

bool IsShape(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : dims) {
        if (dim >= 0 && shape.GetDim(idx) != dim) {
            return false;
        }
        ++idx;
    }
    return true;
}

ge::graphStatus CheckFusedInputs(
    gert::TilingContext *context,
    uint32_t &copyCap,
    uint32_t &missCap,
    uint32_t &hbmMaxBlocks,
    uint32_t &dramMaxBlocks)
{
    const auto query = context->GetInputShape(QUERY);
    const auto key = context->GetInputShape(KEY);
    const auto value = context->GetInputShape(VALUE);
    const auto sparse = context->GetInputShape(SPARSE_INDICES);
    const auto cacheTokens = context->GetInputShape(CACHE_TOKENS);
    const auto hbmTable = context->GetInputShape(HBM_BLOCK_TABLE);
    const auto actualQ = context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY);
    const auto actualKv = context->GetInputShape(ACTUAL_SEQ_LENGTHS_KV);
    const auto queryRope = context->GetInputShape(QUERY_ROPE);
    const auto hbmRope = context->GetInputShape(HBM_KEY_ROPE);
    const auto dramRope = context->GetInputShape(DRAM_KEY_ROPE);
    const auto dramKv = context->GetInputShape(DRAM_KV_CACHE);
    const auto dramTable = context->GetInputShape(DRAM_BLOCK_TABLE);
    const auto sourceIds = context->GetInputShape(TOPK_SOURCE_IDS);
    const auto topkMissCounts = context->GetInputShape(TOPK_MISS_COUNTS);
    const auto missSourceIds = context->GetInputShape(MISS_SOURCE_IDS);
    const auto missDstSlots = context->GetInputShape(MISS_DST_SLOTS);
    const auto missCounts = context->GetInputShape(MISS_COUNTS);
    OPS_ERR_IF(query == nullptr || key == nullptr || value == nullptr ||
                   sparse == nullptr || cacheTokens == nullptr ||
                   hbmTable == nullptr || actualQ == nullptr ||
                   actualKv == nullptr || queryRope == nullptr ||
                   hbmRope == nullptr || dramRope == nullptr ||
                   dramKv == nullptr || dramTable == nullptr ||
                   sourceIds == nullptr || topkMissCounts == nullptr ||
                   missSourceIds == nullptr || missDstSlots == nullptr ||
                   missCounts == nullptr,
               OPS_LOG_E(context->GetNodeName(),
                         "A required fused MTP input shape is missing."),
               return ge::GRAPH_FAILED);

    const gert::Shape q = query->GetStorageShape();
    const gert::Shape hbmKv = key->GetStorageShape();
    const gert::Shape hbmValue = value->GetStorageShape();
    const gert::Shape slots = sparse->GetStorageShape();
    const gert::Shape cache = cacheTokens->GetStorageShape();
    const gert::Shape hbmBt = hbmTable->GetStorageShape();
    const gert::Shape qLens = actualQ->GetStorageShape();
    const gert::Shape kvLens = actualKv->GetStorageShape();
    const gert::Shape qRope = queryRope->GetStorageShape();
    const gert::Shape hbmKpe = hbmRope->GetStorageShape();
    const gert::Shape dramKpe = dramRope->GetStorageShape();
    const gert::Shape dramCkv = dramKv->GetStorageShape();
    const gert::Shape dramBt = dramTable->GetStorageShape();
    const gert::Shape sources = sourceIds->GetStorageShape();
    const gert::Shape queryMissCounts = topkMissCounts->GetStorageShape();
    const gert::Shape requestMissSourceIds = missSourceIds->GetStorageShape();
    const gert::Shape requestMissDstSlots = missDstSlots->GetStorageShape();
    const gert::Shape requestMissCounts = missCounts->GetStorageShape();

    OPS_ERR_IF(!IsShape(cache, {-1}) || cache.GetDim(0) <= 0,
               OPS_LOG_E(context->GetNodeName(),
                         "cache_tokens must have shape [B], B > 0."),
               return ge::GRAPH_FAILED);
    const int64_t batchSize = cache.GetDim(0);
    OPS_ERR_IF(q.GetDimNum() != 3,
               OPS_LOG_E(context->GetNodeName(),
                         "MTP query must have rank 3."),
               return ge::GRAPH_FAILED);
    const int64_t totalQueryTokens = q.GetDim(0);
    const int64_t numQueryHeads = q.GetDim(1);
    OPS_ERR_IF(!IsShape(q, {-1, -1, CKV_DIM}) ||
                   totalQueryTokens < batchSize ||
                   totalQueryTokens > batchSize * MAX_QUERY_COUNT ||
                   numQueryHeads < 8 || numQueryHeads > 128 ||
                   (numQueryHeads & (numQueryHeads - 1)) != 0,
               OPS_LOG_E(context->GetNodeName(),
                         "MTP query must be [T,N,512], B <= T <= 16B, "
                         "N in {8,16,32,64,128}."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(qRope, {totalQueryTokens, q.GetDim(1), KPE_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "MTP query_rope must be [T,N,64]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(hbmKv, {-1, BLOCK_SIZE, 1, CKV_DIM}) ||
                   !IsShape(hbmValue, {hbmKv.GetDim(0), BLOCK_SIZE, 1, CKV_DIM}) ||
                   !IsShape(hbmKpe, {hbmKv.GetDim(0), BLOCK_SIZE, 1, KPE_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "HBM CKV/KPE must be [blocks,128,1,512/64]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(dramCkv, {-1, BLOCK_SIZE, CKV_DIM}) ||
                   !IsShape(dramKpe, {dramCkv.GetDim(0), BLOCK_SIZE, KPE_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "DRAM CKV/KPE must be [blocks,128,512/64]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(slots, {totalQueryTokens, 1, SPARSE_COUNT}) ||
                   !IsShape(hbmBt, {batchSize, -1}) || hbmBt.GetDim(1) <= 0 ||
                   !IsShape(dramBt, {batchSize, -1}) || dramBt.GetDim(1) <= 0 ||
                   !IsShape(qLens, {batchSize}) ||
                   !IsShape(kvLens, {batchSize}) ||
                   !IsShape(sources, {totalQueryTokens, 1, SPARSE_COUNT}) ||
                   !IsShape(queryMissCounts, {totalQueryTokens}) ||
                   !IsShape(requestMissSourceIds, {batchSize, MISS_CAPACITY}) ||
                   !IsShape(requestMissDstSlots, {batchSize, MISS_CAPACITY}) ||
                   !IsShape(requestMissCounts, {batchSize}),
               OPS_LOG_E(context->GetNodeName(),
                         "MTP slots, lengths, tables, and miss metadata have inconsistent shapes."),
               return ge::GRAPH_FAILED);

    copyCap = static_cast<uint32_t>(sources.GetDim(2));
    missCap = static_cast<uint32_t>(requestMissSourceIds.GetDim(1));
    hbmMaxBlocks = static_cast<uint32_t>(hbmBt.GetDim(1));
    dramMaxBlocks = static_cast<uint32_t>(dramBt.GetDim(1));
    OPS_ERR_IF(copyCap != SPARSE_COUNT,
               OPS_LOG_E(context->GetNodeName(),
                         "MTP aligned source width must be 2048."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCap != MISS_CAPACITY,
               OPS_LOG_E(context->GetNodeName(),
                         "Request-level miss metadata width must be 32768."),
               return ge::GRAPH_FAILED);

    const ge::DataType floatingType = context->GetInputDesc(QUERY)->GetDataType();
    OPS_ERR_IF(floatingType != ge::DT_BF16 && floatingType != ge::DT_FLOAT16,
               OPS_LOG_E(context->GetNodeName(),
                         "Floating inputs must be bf16/fp16."),
               return ge::GRAPH_FAILED);
    for (uint32_t idx : {KEY, VALUE, QUERY_ROPE, HBM_KEY_ROPE,
                         DRAM_KEY_ROPE, DRAM_KV_CACHE}) {
        const auto desc = context->GetInputDesc(idx);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != floatingType,
                   OPS_LOG_E(context->GetNodeName(),
                             "All floating inputs must share one dtype."),
                   return ge::GRAPH_FAILED);
    }
    for (uint32_t idx : {SPARSE_INDICES, CACHE_TOKENS, HBM_BLOCK_TABLE,
                         ACTUAL_SEQ_LENGTHS_QUERY, ACTUAL_SEQ_LENGTHS_KV,
                         DRAM_BLOCK_TABLE, TOPK_SOURCE_IDS,
                         TOPK_MISS_COUNTS, MISS_SOURCE_IDS,
                         MISS_DST_SLOTS, MISS_COUNTS}) {
        const auto desc = context->GetInputDesc(idx);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(context->GetNodeName(),
                             "All fused MTP metadata must be int32."),
                   return ge::GRAPH_FAILED);
    }

    const auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Platform info is missing."),
               return ge::GRAPH_FAILED);
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OPS_ERR_IF(aicNum == 0 || aivNum < aicNum * 2U,
               OPS_LOG_E(context->GetNodeName(),
                         "MTP fused copy+Attention requires two AIV cores per AIC."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus TilingFusedCopySfaMtp(
    gert::TilingContext *context)
{
    CopySfaMtpSFATilingInfo sfaInfo;
    CopySfaMtpSFAInfoParser parser(context);
    if (parser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t copyCap = 0;
    uint32_t missCap = 0;
    uint32_t hbmMaxBlocks = 0;
    uint32_t dramMaxBlocks = 0;
    if (CheckFusedInputs(context, copyCap, missCap, hbmMaxBlocks,
                         dramMaxBlocks) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // CheckFusedInputs owns this operator's variable-width MTP contract. The
    // shared checker intentionally remains strict for standalone B/4B SFA.
    CopySfaMtpSFAMlaTiling sfaTiling(context);
    if (sfaTiling.DoOpTiling(&sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // The shared SFA tiler serializes the payload we need, but its generic
    // template key (578 for the GLM MTP3 shape) belongs to the standalone
    // SFA kernel.  The fused MTP operator has one fixed specialization,
    // matching sparse_tail_attention_mtp, and is compiled under key 1.
    context->SetTilingKey(1U);
    // Match the MTP sparse-tail scheduler for one AIC plus two AIVs.
    context->SetScheduleMode(1);

    auto raw = context->GetRawTilingData();
    OPS_ERR_IF(raw == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Raw tiling data is missing."),
               return ge::GRAPH_FAILED);
    FusedCopySfaMtpTilingData fusedTiling;
    const size_t baseSize = raw->GetDataSize();
    const size_t fusedSize = fusedTiling.GetDataSize();
    constexpr size_t fusedSuffixSize = sizeof(uint32_t) * 4U;
    OPS_ERR_IF(fusedSize != baseSize + fusedSuffixSize ||
                   raw->GetCapacity() < fusedSize,
               OPS_LOG_E(context->GetNodeName(),
                         "Unexpected SFA/fused-MTP tiling layout: base=%zu, fused=%zu, capacity=%zu.",
                         baseSize, fusedSize, raw->GetCapacity()),
               return ge::GRAPH_FAILED);
    auto *payload = static_cast<uint8_t *>(raw->GetData());
    std::memcpy(payload + baseSize, &copyCap, sizeof(copyCap));
    std::memcpy(payload + baseSize + sizeof(uint32_t),
                &missCap, sizeof(missCap));
    std::memcpy(payload + baseSize + sizeof(uint32_t) * 2U,
                &hbmMaxBlocks, sizeof(hbmMaxBlocks));
    std::memcpy(payload + baseSize + sizeof(uint32_t) * 3U,
                &dramMaxBlocks, sizeof(dramMaxBlocks));
    raw->SetDataSize(fusedSize);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Workspace array is missing."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareFusedCopySfaMtp(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedCopySfaMtp)
    .Tiling(TilingFusedCopySfaMtp)
    .TilingParse<FusedCopySfaMtpCompileInfo>(
        TilingPrepareFusedCopySfaMtp);

}  // namespace optiling
