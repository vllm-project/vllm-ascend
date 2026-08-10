#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/ops_log.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "prepare_next_token_ids_padded_tiling.h"

namespace {

constexpr uint32_t INPUT_SAMPLED_TOKEN_IDS_INDEX = 0;
constexpr uint32_t INPUT_DISCARD_REQUEST_MASK_INDEX = 1;
constexpr uint32_t INPUT_BACKUP_NEXT_TOKEN_IDS_INDEX = 2;
constexpr uint32_t ATTR_VOCAB_SIZE_INDEX = 0;

constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t INT32_BYTES = sizeof(int32_t);
constexpr uint64_t BOOL_BYTES = sizeof(uint8_t);
constexpr uint64_t UB_RESERVED_BYTES = 1024;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

uint64_t GetRequiredUbBytes(uint32_t rows, uint32_t sampledTokensAligned)
{
    // One sampled-token input buffer, one bool discard-mask buffer,
    // one backup-token input buffer, and two output buffers.
    const uint64_t sampledBytes = AlignUp(
        static_cast<uint64_t>(rows) * sampledTokensAligned * INT32_BYTES,
        BLOCK_BYTES);
    const uint64_t discardBytes = AlignUp(
        static_cast<uint64_t>(rows) * BOOL_BYTES,
        BLOCK_BYTES);
    const uint64_t int32VectorBytes = AlignUp(
        static_cast<uint64_t>(rows) * INT32_BYTES,
        BLOCK_BYTES);

    return sampledBytes + discardBytes + 3 * int32VectorBytes;
}

}  // namespace

namespace optiling {

static ge::graphStatus PrepareNextTokenIdsPaddedTilingFunc(
    gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();

    auto* tilingData =
        context->GetTilingData<PrepareNextTokenIdsPaddedTilingInfo>();
    OPS_CHECK(
        tilingData == nullptr,
        OPS_LOG_E(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);

    const auto* sampledShape =
        context->GetInputShape(INPUT_SAMPLED_TOKEN_IDS_INDEX);
    const auto* discardShape =
        context->GetInputShape(INPUT_DISCARD_REQUEST_MASK_INDEX);
    const auto* backupShape =
        context->GetInputShape(INPUT_BACKUP_NEXT_TOKEN_IDS_INDEX);

    OPS_CHECK(
        sampledShape == nullptr || discardShape == nullptr ||
            backupShape == nullptr,
        OPS_LOG_E(nodeName, "One or more input shapes are nullptr."),
        return ge::GRAPH_FAILED);

    const auto& sampledStorageShape = sampledShape->GetStorageShape();
    const auto& discardStorageShape = discardShape->GetStorageShape();
    const auto& backupStorageShape = backupShape->GetStorageShape();

    OPS_CHECK(
        sampledStorageShape.GetDimNum() != 2,
        OPS_LOG_E(nodeName, "sampledTokenIds must be a 2D tensor."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        discardStorageShape.GetDimNum() != 1,
        OPS_LOG_E(nodeName, "discardRequestMask must be a 1D tensor."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        backupStorageShape.GetDimNum() != 1,
        OPS_LOG_E(nodeName, "backupNextTokenIds must be a 1D tensor."),
        return ge::GRAPH_FAILED);

    const int64_t batchSize = sampledStorageShape.GetDim(0);
    const int64_t sampledTokensPerRequest =
        sampledStorageShape.GetDim(1);

    OPS_CHECK(
        batchSize <= 0 || sampledTokensPerRequest <= 0,
        OPS_LOG_E(
            nodeName,
            "sampledTokenIds dimensions must be positive, but got "
            "batchSize=%ld and sampledTokensPerRequest=%ld.",
            batchSize,
            sampledTokensPerRequest),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        discardStorageShape.GetDim(0) != batchSize,
        OPS_LOG_E(
            nodeName,
            "discardRequestMask length must equal batchSize, but got "
            "%ld and %ld.",
            discardStorageShape.GetDim(0),
            batchSize),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        backupStorageShape.GetDim(0) != batchSize,
        OPS_LOG_E(
            nodeName,
            "backupNextTokenIds length must equal batchSize, but got "
            "%ld and %ld.",
            backupStorageShape.GetDim(0),
            batchSize),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        batchSize > static_cast<int64_t>(
                        std::numeric_limits<uint32_t>::max()) ||
            sampledTokensPerRequest > static_cast<int64_t>(
                                          std::numeric_limits<uint32_t>::max()),
        OPS_LOG_E(nodeName, "Input dimensions exceed uint32_t range."),
        return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OPS_CHECK(
        attrs == nullptr,
        OPS_LOG_E(nodeName, "attrs is nullptr."),
        return ge::GRAPH_FAILED);

    const auto* vocabSizePtr = attrs->GetAttrPointer<int64_t>(
        static_cast<int>(ATTR_VOCAB_SIZE_INDEX));
    OPS_CHECK(
        vocabSizePtr == nullptr,
        OPS_LOG_E(nodeName, "vocabSize attribute is nullptr."),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        *vocabSizePtr <= 0 ||
            *vocabSizePtr > std::numeric_limits<int32_t>::max(),
        OPS_LOG_E(
            nodeName,
            "vocabSize must be in range (0, INT32_MAX], but got %ld.",
            *vocabSizePtr),
        return ge::GRAPH_FAILED);

    platform_ascendc::PlatformAscendC ascendcPlatform(
        context->GetPlatformInfo());
    const uint32_t availableCoreNum = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK(
        availableCoreNum == 0,
        OPS_LOG_E(nodeName, "No AI Vector Core is available."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);
    OPS_CHECK(
        ubSize <= UB_RESERVED_BYTES,
        OPS_LOG_E(
            nodeName,
            "UB size is too small: %lu bytes.",
            ubSize),
        return ge::GRAPH_FAILED);

    const uint32_t batchSizeU32 = static_cast<uint32_t>(batchSize);
    const uint32_t sampledTokensU32 =
        static_cast<uint32_t>(sampledTokensPerRequest);
    constexpr uint32_t int32ElementsPerBlock =
        static_cast<uint32_t>(BLOCK_BYTES / INT32_BYTES);
    const uint64_t sampledTokensAlignedU64 = AlignUp(
        sampledTokensU32,
        int32ElementsPerBlock);
    OPS_CHECK(
        sampledTokensAlignedU64 > std::numeric_limits<uint32_t>::max(),
        OPS_LOG_E(nodeName, "Aligned sampled-token width exceeds uint32_t range."),
        return ge::GRAPH_FAILED);
    const uint32_t sampledTokensAligned =
        static_cast<uint32_t>(sampledTokensAlignedU64);

    const uint32_t usedCoreNum =
        std::min(batchSizeU32, availableCoreNum);
    const uint32_t rowsPerCore = batchSizeU32 / usedCoreNum;
    const uint32_t extraRowCoreNum = batchSizeU32 % usedCoreNum;
    const uint32_t maxRowsOnOneCore =
        rowsPerCore + (extraRowCoreNum > 0 ? 1U : 0U);

    const uint64_t usableUbSize = ubSize - UB_RESERVED_BYTES;
    const uint64_t approximateBytesPerRow =
        static_cast<uint64_t>(sampledTokensAligned) * INT32_BYTES +
        BOOL_BYTES + 3 * INT32_BYTES;
    const uint64_t approximateRowsPerTile =
        std::max<uint64_t>(1, usableUbSize / approximateBytesPerRow);
    uint32_t rowsPerTile = static_cast<uint32_t>(
        std::min<uint64_t>(maxRowsOnOneCore, approximateRowsPerTile));
    while (rowsPerTile > 1 &&
           GetRequiredUbBytes(rowsPerTile, sampledTokensAligned) >
               usableUbSize) {
        --rowsPerTile;
    }
    OPS_CHECK(
        GetRequiredUbBytes(rowsPerTile, sampledTokensAligned) > usableUbSize,
        OPS_LOG_E(
            nodeName,
            "One request row does not fit in UB: aligned sampled tokens=%u, "
            "usable UB=%lu bytes.",
            sampledTokensAligned,
            usableUbSize),
        return ge::GRAPH_FAILED);

    tilingData->batchSize = batchSizeU32;
    tilingData->sampledTokensPerRequest = sampledTokensU32;
    tilingData->sampledTokensAligned = sampledTokensAligned;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->rowsPerCore = rowsPerCore;
    tilingData->extraRowCoreNum = extraRowCoreNum;
    tilingData->rowsPerTile = rowsPerTile;
    tilingData->vocabSize = static_cast<int32_t>(*vocabSizePtr);

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);

    OPS_LOG_D(
        nodeName,
        "batchSize=%u, sampledTokensPerRequest=%u, "
        "sampledTokensAligned=%u, usedCoreNum=%u, rowsPerCore=%u, "
        "extraRowCoreNum=%u, rowsPerTile=%u, vocabSize=%d, ubSize=%lu.",
        tilingData->batchSize,
        tilingData->sampledTokensPerRequest,
        tilingData->sampledTokensAligned,
        tilingData->usedCoreNum,
        tilingData->rowsPerCore,
        tilingData->extraRowCoreNum,
        tilingData->rowsPerTile,
        tilingData->vocabSize,
        ubSize);

    return ge::GRAPH_SUCCESS;
}

struct PrepareNextTokenIdsPaddedCompileInfo {};

static ge::graphStatus TilingParseForPrepareNextTokenIdsPadded(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrepareNextTokenIdsPadded)
    .Tiling(PrepareNextTokenIdsPaddedTilingFunc)
    .TilingParse<PrepareNextTokenIdsPaddedCompileInfo>(
        TilingParseForPrepareNextTokenIdsPadded);

}  // namespace optiling
