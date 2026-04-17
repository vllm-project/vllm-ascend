#include <cstdint>
#include <algorithm>
#include "log/ops_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/ngram_spec_decode_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

using namespace ge;
namespace {
constexpr uint32_t INPUT_TOKEN_IDS_INDEX = 0;
constexpr uint32_t INPUT_NUM_TOKENS_INDEX = 1;
constexpr uint32_t INPUT_SAMPLED_INDEX = 2;
constexpr uint32_t INPUT_DISCARD_INDEX = 3;

constexpr uint32_t ATTR_VOCAB_SIZE_INDEX = 0;
constexpr uint32_t ATTR_MIN_N_INDEX = 1;
constexpr uint32_t ATTR_MAX_N_INDEX = 2;
constexpr uint32_t ATTR_K_INDEX = 3;

constexpr int64_t ELEM_SIZE = 4;  // int32
}  // namespace

namespace optiling {

static ge::graphStatus NgramSpecDecodeTilingFunc(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    NgramSpecDecodeTilingData *tilingData = context->GetTilingData<NgramSpecDecodeTilingData>();
    OPS_CHECK(tilingData == nullptr,
        OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
        OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto vocabSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_VOCAB_SIZE_INDEX));
    auto minNPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_MIN_N_INDEX));
    auto maxNPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_MAX_N_INDEX));
    auto kPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_K_INDEX));

    OPS_CHECK(vocabSizePtr == nullptr, OPS_LOG_E(nodeName, "vocabSizePtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(minNPtr == nullptr, OPS_LOG_E(nodeName, "minNPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(maxNPtr == nullptr, OPS_LOG_E(nodeName, "maxNPtr is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(kPtr == nullptr, OPS_LOG_E(nodeName, "kPtr is null."), return ge::GRAPH_FAILED);

    int64_t vocab_size = *vocabSizePtr;
    int64_t min_n = *minNPtr;
    int64_t max_n = *maxNPtr;
    int64_t k = *kPtr;

    const gert::StorageShape *tokenIdsShape = context->GetInputShape(INPUT_TOKEN_IDS_INDEX);
    const gert::StorageShape *sampledShape = context->GetInputShape(INPUT_SAMPLED_INDEX);
    OPS_CHECK(tokenIdsShape == nullptr, OPS_LOG_E(nodeName, "tokenIdsShape is null."), return ge::GRAPH_FAILED);
    OPS_CHECK(sampledShape == nullptr, OPS_LOG_E(nodeName, "sampledShape is null."), return ge::GRAPH_FAILED);

    int64_t batch_size = tokenIdsShape->GetStorageShape().GetDim(0);
    int64_t max_seq_len = tokenIdsShape->GetStorageShape().GetDim(1);
    int64_t max_new_tokens = sampledShape->GetStorageShape().GetDim(1);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ub_size_limit = static_cast<int64_t>(ubSize);

    int64_t align_elems = 32 / ELEM_SIZE;
    int64_t max_seq_len_align = ((max_seq_len + align_elems - 1) / align_elems) * align_elems;
    int64_t max_new_tokens_align = ((max_new_tokens + align_elems - 1) / align_elems) * align_elems;
    int64_t k_align = ((k + align_elems - 1) / align_elems) * align_elems;

    int64_t ub_per_row = (max_seq_len_align + max_new_tokens_align + k_align) * ELEM_SIZE;
    int64_t ub_overhead = 4 * 32 + static_cast<int64_t>(max_n) * ELEM_SIZE
        + ((max_seq_len_align + 7) / 8);  // maskBuf
    int64_t ub_available = ub_size_limit - ub_overhead;
    int64_t max_block_rows = (ub_available > 0) ? (ub_available / ub_per_row) : 1;
    max_block_rows = std::max(max_block_rows, static_cast<int64_t>(1));

    int64_t block_dim = std::min(batch_size, static_cast<int64_t>(aivNum));
    int64_t rows_per_core = (block_dim > 0) ? (batch_size / block_dim) : 0;
    int64_t former_num = (block_dim > 0) ? (block_dim - 1) : 0;
    int64_t tail_rows = batch_size - former_num * rows_per_core;
    int64_t block_rows = std::min(rows_per_core, max_block_rows);

    tilingData->ngramInfo.batchSize = static_cast<uint32_t>(batch_size);
    tilingData->ngramInfo.maxSeqLen = static_cast<uint32_t>(max_seq_len);
    tilingData->ngramInfo.maxNewTokens = static_cast<uint32_t>(max_new_tokens);
    tilingData->ngramInfo.vocabSize = static_cast<uint32_t>(vocab_size);
    tilingData->ngramInfo.minN = static_cast<uint32_t>(min_n);
    tilingData->ngramInfo.maxN = static_cast<uint32_t>(max_n);
    tilingData->ngramInfo.k = static_cast<uint32_t>(k);
    tilingData->ngramInfo.formerNum = static_cast<uint32_t>(former_num);
    tilingData->ngramInfo.rowsPerCore = static_cast<uint32_t>(rows_per_core);
    tilingData->ngramInfo.tailRows = static_cast<uint32_t>(tail_rows);
    tilingData->ngramInfo.blockRows = static_cast<uint32_t>(block_rows);

    context->SetBlockDim(static_cast<uint32_t>(block_dim));

    OPS_LOG_D(nodeName, "batchSize=%lu, maxSeqLen=%lu, maxNewTokens=%lu, k=%lu, blockDim=%lu, blockRows=%lu",
        batch_size, max_seq_len, max_new_tokens, k, block_dim, block_rows);

    return ge::GRAPH_SUCCESS;
}

struct NgramSpecDecodeCompileInfo {};

ge::graphStatus TilingParseForNgramSpecDecode(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NgramSpecDecode)
    .Tiling(NgramSpecDecodeTilingFunc)
    .TilingParse<NgramSpecDecodeCompileInfo>(TilingParseForNgramSpecDecode);

}  // namespace optiling
