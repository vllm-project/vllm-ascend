
#include "reshape_and_cache_bnsd_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ReshapeAndCacheBNSDTilingData tiling;
    auto keyShape = context->GetInputShape(0)->GetStorageShape();
    auto keyCacheShape = context->GetInputShape(1)->GetStorageShape();
    auto slotMappingShape = context->GetInputShape(2)->GetStorageShape();
    auto seqLenShape = context->GetInputShape(3)->GetStorageShape();
    int64_t numRow = 1;

    for (size_t i = 0; i < keyCacheShape.GetDimNum() - 1; ++i) {
        numRow *= keyCacheShape.GetDim(i);
    }

    uint32_t numTokens = static_cast<uint32_t>(keyShape.GetDim(0));
    uint32_t headDim = static_cast<uint32_t>(keyShape.GetDim(1));
    uint32_t numBlocks = static_cast<uint32_t>(keyCacheShape.GetDim(0));
    uint32_t numHeads = static_cast<uint32_t>(keyCacheShape.GetDim(1));
    uint32_t blockSize = static_cast<uint32_t>(keyCacheShape.GetDim(2));
    uint32_t batchSeqLen = static_cast<uint32_t>(slotMappingShape.GetDim(0));
    uint32_t batch = static_cast<uint32_t>(seqLenShape.GetDim(0));
    uint32_t numCore = ascendcPlatform.GetCoreNumAiv();

    tiling.set_numTokens(numTokens);
    tiling.set_headDim(headDim);
    tiling.set_numBlocks(numBlocks);
    tiling.set_numHeads(numHeads);
    tiling.set_blockSize(blockSize);
    tiling.set_batchSeqLen(batchSeqLen);
    tiling.set_batch(batch);
    tiling.set_numCore(numCore);
    
    context->SetTilingKey(0);
    context->SetBlockDim(numCore);

    size_t *workspaces = context->GetWorkspaceSizes(1); // get second variable
    workspaces[0] = 16 * 1024 * 1024;  

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForReshapeAndCacheBnsd(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(ReshapeAndCacheBnsd)
    .Tiling(TilingFunc)
    .TilingParse<reshapeAndCacheBnsdCompileInfo>(TilingPrepareForReshapeAndCacheBnsd);
}