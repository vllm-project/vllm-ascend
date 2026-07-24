#include "register/op_def_registry.h"
#include "gumbel_sample_tiling.h"

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// 内部 vocab 维分块：4096 fp32 = 16KB，远低于 UB；4096/64=64 ≤ 255 满足约束 #8
constexpr uint32_t GUMBEL_SAMPLE_BLOCK_SIZE = 4096;

static ge::graphStatus GumbelSampleTilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 优先从 CompileInfo 取核数（编译期缓存），回退到实时查询。
    uint32_t aivCoreNum = 0;
    auto ptrCompileInfo = reinterpret_cast<const GumbelSampleCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo != nullptr && ptrCompileInfo->totalCoreNum > 0) {
        aivCoreNum = ptrCompileInfo->totalCoreNum;
    } else {
        auto platformInfo = context->GetPlatformInfo();
        if (platformInfo == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    }
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto logitsShapeBundle = context->GetInputShape(0);
    if (logitsShapeBundle == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto logitsShape = logitsShapeBundle->GetStorageShape();
    if (logitsShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }

    // idx_mapping 是第 1 个输入（序号 1），其 dim0 = num_tokens
    auto idxMappingShapeBundle = context->GetInputShape(1);
    if (idxMappingShapeBundle == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto idxMappingShape = idxMappingShapeBundle->GetStorageShape();
    if (idxMappingShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    // temperature 是第 2 个输入（序号 2），其 dim0 = num_req_states
    auto tempShapeBundle = context->GetInputShape(2);
    if (tempShapeBundle == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tempShape = tempShapeBundle->GetStorageShape();
    if (tempShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    int64_t numTokensI64    = logitsShape.GetDim(0);
    int64_t vocabSizeI64    = logitsShape.GetDim(1);
    int64_t numTokensIdxI64 = idxMappingShape.GetDim(0);  // must equal num_tokens
    int64_t numReqStatesI64 = tempShape.GetDim(0);        // == num_req_states
    if (numTokensI64 <= 0 || vocabSizeI64 <= 0 || numTokensIdxI64 <= 0 || numReqStatesI64 <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (numTokensI64 != numTokensIdxI64) {
        return ge::GRAPH_FAILED;
    }
    uint32_t numTokens    = static_cast<uint32_t>(numTokensI64);
    uint32_t numReqStates = static_cast<uint32_t>(numReqStatesI64);
    uint32_t vocabSize    = static_cast<uint32_t>(vocabSizeI64);

    uint32_t hasProcessedLogits = 0;
    uint32_t processedLogitsStride = 0;
    uint32_t numSpeculativeSteps = 0;
    auto processedShapePtr = context->GetOutputShape(1);
    if (processedShapePtr != nullptr) {
        auto processedShape = processedShapePtr->GetStorageShape();
        if (processedShape.GetDimNum() == 2) {
            int64_t maxReqsI64 = processedShape.GetDim(0);
            int64_t processedVocabI64 = processedShape.GetDim(1);
            if (maxReqsI64 > 0 && processedVocabI64 == vocabSizeI64) {
                hasProcessedLogits = 1;
                processedLogitsStride = vocabSize;
            }
        } else if (processedShape.GetDimNum() == 3) {
            int64_t maxReqsI64 = processedShape.GetDim(0);
            int64_t stepsI64 = processedShape.GetDim(1);
            int64_t processedVocabI64 = processedShape.GetDim(2);
            if (maxReqsI64 > 0 && stepsI64 > 0 && processedVocabI64 == vocabSizeI64) {
                hasProcessedLogits = 1;
                numSpeculativeSteps = static_cast<uint32_t>(stepsI64);
                processedLogitsStride = static_cast<uint32_t>(stepsI64 * vocabSizeI64);
            }
        }
    }

    uint32_t hasProcessedLogitsCol = 0;
    auto processedColShapeBundle = context->GetInputShape(5);
    if (processedColShapeBundle != nullptr) {
        auto processedColShape = processedColShapeBundle->GetStorageShape();
        if (processedColShape.GetDimNum() == 0 ||
            (processedColShape.GetDimNum() == 1 && processedColShape.GetDim(0) == 1)) {
            hasProcessedLogitsCol = 1;
        }
    }

    uint32_t usedCoreNum = (numTokens < aivCoreNum) ? numTokens : aivCoreNum;
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    uint32_t formerNum  = numTokens % usedCoreNum;
    uint32_t nRowsLarge = CeilDivU32(numTokens, usedCoreNum);
    uint32_t nRowsSmall = numTokens / usedCoreNum;
    uint32_t numTiles = CeilDivU32(vocabSize, GUMBEL_SAMPLE_BLOCK_SIZE);
    uint32_t lastTileLen = vocabSize - (numTiles - 1) * GUMBEL_SAMPLE_BLOCK_SIZE;

    uint32_t applyTemp = 1;  // 默认 true
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* applyTempAttr = attrs->GetAttrPointer<bool>(0);
        if (applyTempAttr != nullptr) {
            applyTemp = (*applyTempAttr) ? 1u : 0u;
        }
    }

    GumbelSampleTilingData tiling;
    tiling.set_numReqs(numTokens);
    tiling.set_numReqStates(numReqStates);
    tiling.set_numTokens(numTokens);
    tiling.set_vocabSize(vocabSize);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_formerNum(formerNum);
    tiling.set_nRowsLarge(nRowsLarge);
    tiling.set_nRowsSmall(nRowsSmall);
    tiling.set_blockSize(GUMBEL_SAMPLE_BLOCK_SIZE);
    tiling.set_numTiles(numTiles);
    tiling.set_lastTileLen(lastTileLen);
    tiling.set_applyTemp(applyTemp);
    tiling.set_hasProcessedLogits(hasProcessedLogits);
    tiling.set_hasProcessedLogitsCol(hasProcessedLogitsCol);
    tiling.set_processedLogitsStride(processedLogitsStride);
    tiling.set_numSpeculativeSteps(numSpeculativeSteps);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(usedCoreNum);
    // TilingKey 与 op_kernel/gumbel_sample_tiling_key.h 中 ASCENDC_TPL_SEL 声明一致：0/1
    context->SetTilingKey(applyTemp);

    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 16 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GumbelSample(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<GumbelSampleCompileInfo>();
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GumbelSample)
    .Tiling(GumbelSampleTilingFunc)
    .TilingParse<GumbelSampleCompileInfo>(TilingPrepare4GumbelSample);

}  // namespace optiling
