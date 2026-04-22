/**
 * @file copy_and_expand_dflash_inputs_tiling.cpp
 * @brief CopyAndExpandDflashInputs TilingFunc implementation
 */

#include "copy_and_expand_dflash_inputs_tiling.h"
#include "register/op_def_registry.h"
#include "log/ops_log.h"

#include <algorithm>

namespace optiling {

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& coreNum)
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
    OPS_LOG_D(context, "TilingFunc running.");

    // ========== 1. Get hardware core count ==========
    uint32_t coreNum;
    GetCompileParameters(context, coreNum);

    // ========== 2. Derive num_reqs from query_start_loc shape ==========
    // query_start_loc is the 3rd input (index 2), shape [num_reqs + 1]
    auto queryStartLocShape = context->GetInputShape(2);
    if (queryStartLocShape == nullptr) {
        OPS_LOG_E(context, "Failed to get query_start_loc shape (input index 2)");
        return ge::GRAPH_PARAM_INVALID;
    }
    if (queryStartLocShape->GetStorageShape().GetDimNum() == 0) {
        OPS_LOG_E(context, "query_start_loc has no dimensions");
        return ge::GRAPH_PARAM_INVALID;
    }
    int64_t dim0 = queryStartLocShape->GetStorageShape().GetDim(0);
    if (dim0 <= 1) {
        OPS_LOG_E(context, "query_start_loc first dimension must be > 1, got %ld", dim0);
        return ge::GRAPH_PARAM_INVALID;
    }
    uint32_t numReqs = static_cast<uint32_t>(dim0 - 1);
    OPS_LOG_I(context, "Derived numReqs: %u from query_start_loc shape [%ld]", numReqs, dim0);

    // ========== 3. Get block_table stride from input shape ==========
    // block_table is the 5th input (index 4), shape [num_reqs, max_blocks]
    // The stride is the second dimension (max_blocks)
    auto blockTableShape = context->GetInputShape(4);
    if (blockTableShape == nullptr) {
        OPS_LOG_E(context, "Failed to get block_table shape (input index 4)");
        return ge::GRAPH_PARAM_INVALID;
    }
    if (blockTableShape->GetStorageShape().GetDimNum() < 2) {
        OPS_LOG_E(context, "block_table must have at least 2 dimensions, got %d",
                  blockTableShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_PARAM_INVALID;
    }
    int64_t blockTableDim0 = blockTableShape->GetStorageShape().GetDim(0);
    int64_t blockTableDim1 = blockTableShape->GetStorageShape().GetDim(1);
    if (blockTableDim0 != static_cast<int64_t>(numReqs)) {
        OPS_LOG_E(context, "block_table first dimension (%ld) must equal numReqs (%u)",
                  blockTableDim0, numReqs);
        return ge::GRAPH_PARAM_INVALID;
    }
    if (blockTableDim1 <= 0) {
        OPS_LOG_E(context, "block_table second dimension (max_blocks) must be > 0, got %ld",
                  blockTableDim1);
        return ge::GRAPH_PARAM_INVALID;
    }
    uint32_t blockTableStride = static_cast<uint32_t>(blockTableDim1);
    OPS_LOG_I(context, "blockTable shape dims: num_reqs=%ld, max_blocks=%ld",
              blockTableDim0, blockTableDim1);
    OPS_LOG_I(context, "Final blockTableStride: %u", blockTableStride);

    // ========== 4. Get operator attributes ==========
    auto attrs = context->GetAttrs();

    int64_t parallelDraftingTokenId = *(attrs->GetAttrPointer<int64_t>(0));
    int64_t numQueryPerReq = *(attrs->GetAttrPointer<int64_t>(1));
    int64_t numSpeculativeTokens = *(attrs->GetAttrPointer<int64_t>(2));
    int64_t blockSize = *(attrs->GetAttrPointer<int64_t>(3));
    int64_t totalInputTokens = *(attrs->GetAttrPointer<int64_t>(4));
    bool hasNumRejected = *(attrs->GetAttrPointer<bool>(5));

    // ========== 5. Compute core distribution ==========
    uint32_t usedCoreNum = std::min(coreNum, numReqs);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    uint32_t reqsPerCore   = numReqs / usedCoreNum;
    uint32_t remainderReqs = numReqs % usedCoreNum;

    // ========== 6. Set tiling_key ==========
    context->SetTilingKey(1);

    // ========== 7. Get output shape ==========
    uint32_t totalQueryTokens = numReqs * numQueryPerReq;

    // ========== 8. Fill TilingData ==========
    CopyAndExpandDflashInputsTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numReqs(numReqs);
    tiling.set_reqsPerCore(reqsPerCore);
    tiling.set_remainderReqs(remainderReqs);
    tiling.set_parallelDraftingTokenId(parallelDraftingTokenId);
    tiling.set_numQueryPerReq(static_cast<uint32_t>(numQueryPerReq));
    tiling.set_numSpeculativeTokens(static_cast<uint32_t>(numSpeculativeTokens));
    tiling.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.set_totalInputTokens(static_cast<uint32_t>(totalInputTokens));
    tiling.set_hasNumRejected(hasNumRejected ? 1 : 0);
    tiling.set_blockTableStride(blockTableStride);
    tiling.set_totalQueryTokens(totalQueryTokens);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ========== 9. Set block_dim ==========
    context->SetBlockDim(usedCoreNum);

    OPS_LOG_I(context, "Block Dim: %u", usedCoreNum);
    OPS_LOG_I(context,
        "numReqs: %u, numQueryPerReq: %ld, numSpeculativeTokens: %ld, totalInputTokens: %ld, totalQueryTokens: %u",
        numReqs, numQueryPerReq, numSpeculativeTokens, totalInputTokens, totalQueryTokens);
    OPS_LOG_I(context,
        "blockTableStride: %u, blockSize: %ld, hasNumRejected: %d",
        blockTableStride, blockSize, hasNumRejected ? 1 : 0);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4CopyAndExpandDflashInputs(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4CopyAndExpandDflashInputs running.");
    OPS_LOG_I(context, "TilingPrepare4CopyAndExpandDflashInputs running.");
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