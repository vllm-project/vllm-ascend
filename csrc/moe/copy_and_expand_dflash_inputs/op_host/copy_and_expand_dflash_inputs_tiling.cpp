/**
 * @file copy_and_expand_dflash_inputs_tiling.cpp
 * @brief CopyAndExpandDflashInputs TilingFunc implementation
 */

#include "copy_and_expand_dflash_inputs_tiling.h"
#include "tiling_base/error_log.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

#include <algorithm>

namespace optiling {

static constexpr int IDX_TARGET_POSITIONS = 1;
static constexpr int IDX_QUERY_START_LOC = 3;
static constexpr int IDX_BLOCK_TABLE = 5;

static constexpr int ATTR_PARALLEL_DRAFTING_TOKEN_ID = 0;
static constexpr int ATTR_BLOCK_SIZE = 1;
static constexpr int ATTR_NUM_QUERY_PER_REQ = 2;
static constexpr int ATTR_NUM_SPECULATIVE_TOKENS = 3;
static constexpr int ATTR_SAMPLE_FROM_ANCHOR = 4;

static void GetCompileParameters(gert::TilingContext* context, uint32_t& coreNum)
{
    auto ptrCompileInfo = reinterpret_cast<const CopyAndExpandDflashInputsCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = ascendcPlatform.GetCoreNum();
    } else {
        coreNum = ptrCompileInfo->totalCoreNum;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_I(context, "Enter TilingFunc for CopyAndExpandDflashInputs");

    uint32_t coreNum;
    GetCompileParameters(context, coreNum);

    // num_reqs from query_start_loc shape [num_reqs + 1]
    auto queryStartLocShape = context->GetInputShape(IDX_QUERY_START_LOC);
    uint32_t numReqs = 0;
    if (queryStartLocShape != nullptr &&
        queryStartLocShape->GetStorageShape().GetDimNum() > 0) {
        int64_t dim0 = queryStartLocShape->GetStorageShape().GetDim(0);
        numReqs = (dim0 > 1) ? static_cast<uint32_t>(dim0 - 1) : 0;
    }

    // num_context from target_positions shape [num_context]
    auto targetPositionsShape = context->GetInputShape(IDX_TARGET_POSITIONS);
    uint32_t numContext = 0;
    if (targetPositionsShape != nullptr &&
        targetPositionsShape->GetStorageShape().GetDimNum() > 0) {
        numContext = static_cast<uint32_t>(targetPositionsShape->GetStorageShape().GetDim(0));
    }

    // block_table stride from shape [max_reqs, max_blocks]
    auto blockTableShape = context->GetInputShape(IDX_BLOCK_TABLE);
    uint32_t blockTableStride = 0;
    if (blockTableShape != nullptr &&
        blockTableShape->GetStorageShape().GetDimNum() > 1) {
        blockTableStride = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(1));
    }

    auto attrs = context->GetAttrs();
    int32_t parallelDraftingTokenId = *(attrs->GetAttrPointer<int32_t>(ATTR_PARALLEL_DRAFTING_TOKEN_ID));
    int32_t blockSize = *(attrs->GetAttrPointer<int32_t>(ATTR_BLOCK_SIZE));
    int32_t numQueryPerReq = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM_QUERY_PER_REQ));
    int32_t numSpeculativeTokens = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM_SPECULATIVE_TOKENS));
    bool sampleFromAnchor = *(attrs->GetAttrPointer<bool>(ATTR_SAMPLE_FROM_ANCHOR));

    // Per-request output segments are not block-aligned; a single core keeps
    // neighboring writes from overlapping at block tails.
    uint32_t usedCoreNum = 1;

    context->SetTilingKey(1);

    CopyAndExpandDflashInputsTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numReqs(numReqs);
    tiling.set_numContext(numContext);
    tiling.set_blockTableStride(blockTableStride);
    tiling.set_parallelDraftingTokenId(parallelDraftingTokenId);
    tiling.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.set_numQueryPerReq(static_cast<uint32_t>(numQueryPerReq));
    tiling.set_numSpeculativeTokens(static_cast<uint32_t>(numSpeculativeTokens));
    tiling.set_sampleFromAnchor(sampleFromAnchor ? 1u : 0u);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(usedCoreNum);

    OPS_LOG_I(context,
        "numReqs: %u, numContext: %u, blockTableStride: %u, numQueryPerReq: %d, sampleFromAnchor: %d",
        numReqs, numContext, blockTableStride, numQueryPerReq, sampleFromAnchor ? 1 : 0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CopyAndExpandDflashInputs(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4CopyAndExpandDflashInputs running.");
    auto compileInfo = context->GetCompiledInfo<CopyAndExpandDflashInputsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CopyAndExpandDflashInputs)
    .Tiling(TilingFunc)
    .TilingParse<CopyAndExpandDflashInputsCompileInfo>(TilingPrepare4CopyAndExpandDflashInputs);

}  // namespace optiling
