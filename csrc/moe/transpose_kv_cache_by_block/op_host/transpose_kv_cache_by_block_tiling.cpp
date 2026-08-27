#include "transpose_kv_cache_by_block_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/ops_log.h"
#include <algorithm>

namespace optiling {

constexpr uint64_t DATA_SIZE = 2;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t DB_ON = 2;

constexpr uint32_t FULL_LOAD = 0;
constexpr uint32_t SPLIT_BLOCK_SIZE_ALIGNED_AND_DB = 1;
constexpr uint32_t SPLIT_BLOCK_SIZE_UNALIGNED_AND_DB = 3;
constexpr uint32_t SPLIT_BLOCK_SIZE_ALIGNED_AND_NOT_DB = 2;
constexpr uint32_t SPLIT_BLOCK_SIZE_UNALIGNED_AND_NOT_DB = 4;

void findFactorsOptimized(std::vector<int64_t> &factors, int64_t n) {

    for (int64_t i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            factors.push_back(i);

            if (i != n / i) {
                factors.push_back(n / i);
            }
        }
    }

    sort(factors.begin(), factors.end());
}

ge::graphStatus CalTiling(gert::TilingContext* context, TransposeKvCacheByBlockTilingData &tiling)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfoPtr, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t useCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto attr = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attr, return ge::GRAPH_FAILED);
    const int64_t* blockSizePtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* headNumPtr = attr->GetAttrPointer<int64_t>(1);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(2);
    const int64_t* splitNumPtr = attr->GetAttrPointer<int64_t>(3);
    const int64_t* layerNumPtr = attr->GetAttrPointer<int64_t>(4);
    OPS_CHECK(blockSizePtr == nullptr || headNumPtr == nullptr || headDimPtr == nullptr ||
                splitNumPtr == nullptr || layerNumPtr == nullptr,
                OPS_LOG_E(context->GetNodeName(), "Get attr failed."),
                return ge::GRAPH_FAILED);

    auto blockIDsTensor = context->GetDynamicInputTensor(2, 0);
    OPS_LOG_E_IF_NULL(context, blockIDsTensor, return ge::GRAPH_FAILED);

    gert::Shape blockIDsTensorShape = blockIDsTensor->GetStorageShape();
    int64_t calBlockNum = static_cast<int64_t>(blockIDsTensorShape.GetDim(0));

    tiling.set_calBlockNum(static_cast<uint32_t>(calBlockNum));

    int64_t blockSize = *blockSizePtr;
    int64_t headNum = *headNumPtr;
    int64_t headDim = *headDimPtr;
    int64_t splitNum = *splitNumPtr;
    int64_t layerNum = *layerNumPtr;
    uint32_t tilingKey = FULL_LOAD;

    if (headDim * DATA_SIZE % BLOCK_SIZE != 0) {
        OPS_LOG_E(context, "headDim * DATA_SIZE must be a multiple of 32 bytes.");
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> factors;
    findFactorsOptimized(factors, useCoreNum);

    uint32_t factorIndex = 0;
    bool findSplitNum = true;
    int64_t blockSizeSplitNum = factors[factorIndex];
    uint64_t dataSizeloadOnce = blockSize * headNum * headDim * DATA_SIZE;
    // if can full load, not split blockSize and db
    if (dataSizeloadOnce > ubSize) {
        tilingKey = SPLIT_BLOCK_SIZE_ALIGNED_AND_DB;
        // split blockSize and db
        while (dataSizeloadOnce > (ubSize / DB_ON)) {
            factorIndex += 1;
            if (factorIndex == factors.size()) {
                tilingKey = FULL_LOAD;
                findSplitNum = false;
                break;
            }
            blockSizeSplitNum = factors[factorIndex];
            dataSizeloadOnce = ((blockSize + blockSizeSplitNum - 1) / blockSizeSplitNum) * headNum * headDim * DATA_SIZE;
        }
        if (tilingKey == SPLIT_BLOCK_SIZE_ALIGNED_AND_DB && (blockSize % blockSizeSplitNum != 0)) {
            tilingKey = SPLIT_BLOCK_SIZE_UNALIGNED_AND_DB;
        }
    }

    if (!findSplitNum) {
        tilingKey = SPLIT_BLOCK_SIZE_ALIGNED_AND_NOT_DB;
        // split blockSize but not db
        findSplitNum = true;
        factorIndex = 0;
        blockSizeSplitNum = factors[factorIndex];
        dataSizeloadOnce = blockSize * headNum * headDim * DATA_SIZE;
        while (dataSizeloadOnce > ubSize) {
            factorIndex += 1;
            if (factorIndex == factors.size()) {
                tilingKey = FULL_LOAD;
                findSplitNum = false;
                break;
            }
            blockSizeSplitNum = factors[factorIndex];
            dataSizeloadOnce = ((blockSize + blockSizeSplitNum - 1) / blockSizeSplitNum) * headNum * headDim * DATA_SIZE;
        }
        if (tilingKey == SPLIT_BLOCK_SIZE_ALIGNED_AND_NOT_DB && (blockSize % blockSizeSplitNum != 0)) {
            tilingKey = SPLIT_BLOCK_SIZE_UNALIGNED_AND_NOT_DB;
        }
    }

    // headNum * headDim too large
    if (!findSplitNum) {
        OPS_LOG_E(context, "headNum * headDim * sizeof(half) > ubSize "
                  "or blockSize * headNum * headDim * sizeof(half) > ubSize * vectorCoreNum. "
                  "Currently, splitting headNum or headDim is not supported.");
        return ge::GRAPH_FAILED;
    }
    tiling.set_blockSizePerTime(static_cast<uint32_t>((blockSize + blockSizeSplitNum - 1) / blockSizeSplitNum));
    tiling.set_blockSizePerTimeTail(static_cast<uint32_t>(blockSize % blockSizeSplitNum));
    tiling.set_blockSizeSplitNum(static_cast<uint32_t>(blockSizeSplitNum));

    tiling.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.set_headNum(static_cast<uint32_t>(headNum));
    tiling.set_headDim(static_cast<uint32_t>(headDim));
    tiling.set_splitNum(static_cast<uint32_t>(splitNum));
    tiling.set_layerNum(static_cast<uint32_t>(layerNum));

    int64_t totalRound = layerNum * calBlockNum;

    if ((totalRound * blockSizeSplitNum) < useCoreNum) {
        useCoreNum = totalRound * blockSizeSplitNum;
    }
    int64_t blockPerCore = totalRound / (useCoreNum / blockSizeSplitNum);
    int64_t tailCoreNum = totalRound % (useCoreNum / blockSizeSplitNum);

    tiling.set_useCoreNum(static_cast<uint32_t>(useCoreNum));
    tiling.set_blockPerCore(static_cast<uint32_t>(blockPerCore));
    tiling.set_tailCoreNum(static_cast<uint32_t>(tailCoreNum));
    context->SetBlockDim(useCoreNum);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TransposeKvCacheByBlockTilingFunc(gert::TilingContext* context)
{

  TransposeKvCacheByBlockTilingData tiling;
  auto status = CalTiling(context, tiling);
  OP_CHECK(status != ge::GRAPH_SUCCESS, OPS_LOG_E(context->GetNodeName(), "Cal tiling failed."),
           return ge::GRAPH_FAILED);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

struct TransposeKvCacheByBlockCompileInfo {};
ge::graphStatus TilingParseForTransposeKvCacheByBlock(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransposeKvCacheByBlock)
    .Tiling(TransposeKvCacheByBlockTilingFunc)
    .TilingParse<TransposeKvCacheByBlockCompileInfo>(TilingParseForTransposeKvCacheByBlock);
}