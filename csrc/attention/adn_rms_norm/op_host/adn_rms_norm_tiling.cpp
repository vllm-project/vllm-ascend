#include "adn_rms_norm_tiling.h"

#include <algorithm>
#include <cstdint>

#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint32_t H128 = 128;
constexpr uint32_t H256 = 256;
constexpr uint32_t H2048 = 2048;
constexpr uint32_t KEY_H128 = 100;
constexpr uint32_t KEY_H256 = 200;
constexpr uint32_t KEY_H2048 = 300;

struct KernelChoice {
    uint32_t tilingKey;
    uint32_t maxRowsPerTile;
};

bool SelectKernel(int64_t hiddenSize, KernelChoice& choice)
{
    if (hiddenSize == H128) {
        choice = {KEY_H128, 32};
        return true;
    }
    if (hiddenSize == H256) {
        choice = {KEY_H256, 16};
        return true;
    }
    if (hiddenSize == H2048) {
        choice = {KEY_H2048, 2};
        return true;
    }
    return false;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* xShape = context->GetInputShape(0);
    const auto* gammaShape = context->GetInputShape(1);
    const auto* attrs = context->GetAttrs();
    if (xShape == nullptr || gammaShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorageShape = xShape->GetStorageShape();
    const size_t rank = xStorageShape.GetDimNum();
    if (rank == 0) {
        OPS_LOG_E(context, "x must have at least one dimension");
        return ge::GRAPH_FAILED;
    }
    const int64_t hiddenSize64 = xStorageShape.GetDim(rank - 1);
    const int64_t elementCount = xStorageShape.GetShapeSize();
    if (hiddenSize64 <= 0 || elementCount <= 0 || elementCount % hiddenSize64 != 0) {
        OPS_LOG_E(context, "invalid x shape: elements=%ld hidden=%ld", elementCount, hiddenSize64);
        return ge::GRAPH_FAILED;
    }

    KernelChoice choice{};
    if (!SelectKernel(hiddenSize64, choice)) {
        OPS_LOG_E(context, "hidden size must be 128, 256, or 2048; got %ld", hiddenSize64);
        return ge::GRAPH_FAILED;
    }
    const uint32_t hiddenSize = static_cast<uint32_t>(hiddenSize64);
    if (gammaShape->GetStorageShape().GetShapeSize() != hiddenSize64) {
        OPS_LOG_E(context, "gamma element count must equal hidden size %u", hiddenSize);
        return ge::GRAPH_FAILED;
    }

    uint32_t coreNum = 1;
    const auto* compileInfo = reinterpret_cast<const AdnRmsNormCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->totalCoreNum > 0) {
        coreNum = compileInfo->totalCoreNum;
    } else {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        coreNum = std::max<uint32_t>(platform.GetCoreNum(), 1);
    }

    const uint64_t numRows = static_cast<uint64_t>(elementCount / hiddenSize64);
    const uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(coreNum, numRows));
    const uint64_t baseRowsPerCore = numRows / usedCoreNum;
    const uint32_t extraRowCoreCount = static_cast<uint32_t>(numRows % usedCoreNum);
    const uint32_t rowsPerTile = static_cast<uint32_t>(
        std::min<uint64_t>(choice.maxRowsPerTile, baseRowsPerCore + (extraRowCoreCount > 0 ? 1 : 0)));
    const float epsilon = *attrs->GetAttrPointer<float>(0);

    AdnRmsNormTilingData tiling;
    tiling.set_numRows(numRows);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_rowsPerTile(std::max<uint32_t>(rowsPerTile, 1));
    tiling.set_baseRowsPerCore(baseRowsPerCore);
    tiling.set_extraRowCoreCount(extraRowCoreCount);
    tiling.set_reducePartCount(hiddenSize / 64);
    tiling.set_epsilon(epsilon);
    tiling.set_invHiddenSize(1.0f / static_cast<float>(hiddenSize));
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(choice.tilingKey);
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AdnRmsNorm(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<AdnRmsNormCompileInfo>();
    if (compileInfo == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->totalCoreNum = platform.GetCoreNum();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdnRmsNorm)
    .Tiling(TilingFunc)
    .TilingParse<AdnRmsNormCompileInfo>(TilingPrepare4AdnRmsNorm);

}  // namespace optiling
