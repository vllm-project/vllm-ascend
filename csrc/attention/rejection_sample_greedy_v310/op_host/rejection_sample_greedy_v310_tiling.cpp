/**
 * @file rejection_sample_greedy_v310_tiling.cpp
 * @brief RejectionSampleGreedyV310 tiling implementation
 */

#include "rejection_sample_greedy_v310_tiling.h"

#include <algorithm>

#include "log/ops_log.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr int INPUT_CUMULATIVE_COUNTS = 0;
static constexpr int ATTR_MAX_SPEC_LEN = 0;
static constexpr int ATTR_ALIGNED_OUTPUT_LEN = 1;
static constexpr uint32_t INT32_ELEMENTS_PER_BLOCK = 8;

static uint32_t GetCoreNum(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const RejectionSampleGreedyV310CompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr) {
        return compileInfo->totalCoreNum;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return ascendcPlatform.GetCoreNum();
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto cumulativeShape = context->GetInputShape(INPUT_CUMULATIVE_COUNTS);
    auto attrs = context->GetAttrs();
    if (cumulativeShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* maxSpecLenAttr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SPEC_LEN);
    const int64_t* alignedOutputLenAttr = attrs->GetAttrPointer<int64_t>(ATTR_ALIGNED_OUTPUT_LEN);
    if (maxSpecLenAttr == nullptr || alignedOutputLenAttr == nullptr || *maxSpecLenAttr < 0 ||
        *alignedOutputLenAttr < *maxSpecLenAttr + 1 ||
        *alignedOutputLenAttr % INT32_ELEMENTS_PER_BLOCK != 0) {
        OPS_LOG_E(context, "Invalid rejection sampling attributes");
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = 0;
    const auto& storageShape = cumulativeShape->GetStorageShape();
    if (storageShape.GetDimNum() > 0 && storageShape.GetDim(0) > 0) {
        batchSize = static_cast<uint32_t>(storageShape.GetDim(0));
    }
    const uint32_t totalCoreNum = GetCoreNum(context);
    const uint32_t usedCoreNum = std::max(1U, std::min(batchSize, totalCoreNum));

    RejectionSampleGreedyV310TilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_batchSize(batchSize);
    tiling.set_maxSpecLen(static_cast<uint32_t>(*maxSpecLenAttr));
    tiling.set_alignedOutputLen(static_cast<uint32_t>(*alignedOutputLenAttr));
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4RejectionSampleGreedyV310(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<RejectionSampleGreedyV310CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RejectionSampleGreedyV310)
    .Tiling(TilingFunc)
    .TilingParse<RejectionSampleGreedyV310CompileInfo>(TilingPrepare4RejectionSampleGreedyV310);

}  // namespace optiling
