/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "compressor_metadata_tiling.h"

#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr uint32_t ROPE_COS_INDEX = 0;
constexpr uint32_t ROPE_SIN_INDEX = 1;
constexpr uint32_t CU_SEQLENS_INDEX = 2;
constexpr uint32_t START_POS_INDEX = 3;
constexpr uint32_t KV_BLOCK_TABLE_INDEX = 4;
constexpr uint32_t COMPRESS_COS_INDEX = 0;
constexpr uint32_t COMPRESS_SIN_INDEX = 1;
constexpr uint32_t SLOT_MAPPING_INDEX = 2;
constexpr uint32_t SLOT_MAPPING_FLAT = 1;
constexpr uint32_t SLOT_MAPPING_BLOCK_OFFSET = 2;
constexpr int64_t MAX_UINT32_VALUE = 0xFFFFFFFFLL;
constexpr int64_t MAX_INT32_VALUE = 0x7FFFFFFFLL;

constexpr uint32_t TILING_KEY_FLOAT = 1;
constexpr uint32_t TILING_KEY_FLOAT16 = 2;
constexpr uint32_t TILING_KEY_BF16 = 3;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_TILE_ROWS = 512;
constexpr uint32_t MAX_DATACOPY_BLOCK_COUNT = 4095;
constexpr uint32_t ROWS_PER_CORE_TARGET = 64;
constexpr uint32_t UB_RESERVED_BYTES = 16 * 1024;

uint32_t AlignUp(uint64_t value, uint32_t align)
{
    return static_cast<uint32_t>((value + align - 1) / align * align);
}

uint32_t CeilDiv(uint64_t lhs, uint64_t rhs)
{
    return static_cast<uint32_t>((lhs + rhs - 1) / rhs);
}
}  // namespace

static ge::graphStatus CompressorMetadataTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        aivCoreNum = ascendcPlatform.GetCoreNum();
    }
    if (aivCoreNum == 0) {
        OP_LOGE(context->GetNodeName(), "Failed to get AIV core num.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        OP_LOGE(context->GetNodeName(), "Failed to get UB size.");
        return ge::GRAPH_FAILED;
    }

    auto outputShape = context->GetOutputShape(COMPRESS_COS_INDEX);
    auto compressSinShape = context->GetOutputShape(COMPRESS_SIN_INDEX);
    auto slotMappingShape = context->GetOutputShape(SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, compressSinShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotMappingShape);
    auto outputDimNum = outputShape->GetStorageShape().GetDimNum();
    if (outputDimNum < 2) {
        OP_LOGE(context->GetNodeName(), "compressCos dim num should be at least 2.");
        return ge::GRAPH_FAILED;
    }
    if (compressSinShape->GetStorageShape().GetDimNum() != outputDimNum) {
        OP_LOGE(context->GetNodeName(), "compressCos and compressSin dim num mismatch.");
        return ge::GRAPH_FAILED;
    }
    for (size_t dimIdx = 0; dimIdx < outputDimNum; ++dimIdx) {
        if (compressSinShape->GetStorageShape().GetDim(dimIdx) != outputShape->GetStorageShape().GetDim(dimIdx)) {
            OP_LOGE(context->GetNodeName(), "compressCos and compressSin shape mismatch.");
            return ge::GRAPH_FAILED;
        }
    }
    int64_t numRows = outputShape->GetStorageShape().GetDim(0);
    int64_t ropeDim = outputShape->GetStorageShape().GetDim(outputDimNum - 1);
    if (numRows <= 0 || ropeDim <= 0 || numRows > MAX_UINT32_VALUE || ropeDim > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "compressCos shape is invalid.");
        return ge::GRAPH_FAILED;
    }

    auto ropeCosShape = context->GetInputShape(ROPE_COS_INDEX);
    auto ropeSinShape = context->GetInputShape(ROPE_SIN_INDEX);
    auto cuSeqlensShape = context->GetInputShape(CU_SEQLENS_INDEX);
    auto startPosShape = context->GetInputShape(START_POS_INDEX);
    auto kvBlockTableShape = context->GetInputShape(KV_BLOCK_TABLE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeCosShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeSinShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqlensShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, startPosShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvBlockTableShape);
    if (ropeCosShape->GetStorageShape().GetDimNum() != 2 || ropeSinShape->GetStorageShape().GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "ropeCos and ropeSin should be 2D tensors.");
        return ge::GRAPH_FAILED;
    }
    int64_t ropeRows = ropeCosShape->GetStorageShape().GetDim(0);
    int64_t ropeCosDim = ropeCosShape->GetStorageShape().GetDim(1);
    if (ropeRows <= 0 || ropeCosDim <= 0 || ropeRows > MAX_UINT32_VALUE || ropeCosDim != ropeDim ||
        ropeSinShape->GetStorageShape().GetDim(0) != ropeRows ||
        ropeSinShape->GetStorageShape().GetDim(1) != ropeCosDim) {
        OP_LOGE(context->GetNodeName(), "ropeCos and ropeSin shape mismatch.");
        return ge::GRAPH_FAILED;
    }
    int64_t cuSeqlensDim0 = cuSeqlensShape->GetStorageShape().GetDim(0);
    if (cuSeqlensDim0 < 2 || cuSeqlensDim0 > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "cuSeqlens dim0 should be at least 2.");
        return ge::GRAPH_FAILED;
    }
    if (startPosShape->GetStorageShape().GetDimNum() != 1 ||
        startPosShape->GetStorageShape().GetDim(0) <= 0 ||
        startPosShape->GetStorageShape().GetDim(0) > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "startPos should be a non-empty 1D tensor.");
        return ge::GRAPH_FAILED;
    }
    if (kvBlockTableShape->GetStorageShape().GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "kvBlockTable should be a 2D tensor.");
        return ge::GRAPH_FAILED;
    }
    int64_t kvBlockTableRows = kvBlockTableShape->GetStorageShape().GetDim(0);
    int64_t kvBlockTableStride = kvBlockTableShape->GetStorageShape().GetDim(1);
    if (kvBlockTableRows <= 0 || kvBlockTableStride <= 0 || kvBlockTableRows > MAX_UINT32_VALUE ||
        kvBlockTableStride > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "kvBlockTable shape is invalid.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* kvBlockSizePtr = attrs->GetInt(0);
    const int64_t* slotMappingFormatPtr = attrs->GetInt(1);
    const int64_t* cmpRatioPtr = attrs->GetInt(2);
    const int64_t* actualNumReqsPtr = attrs->GetInt(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvBlockSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotMappingFormatPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpRatioPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualNumReqsPtr);
    if (*kvBlockSizePtr <= 0 || *kvBlockSizePtr > MAX_INT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "kvBlockSize should be in (0, INT32_MAX].");
        return ge::GRAPH_FAILED;
    }
    if (*cmpRatioPtr <= 0 || *cmpRatioPtr > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "cmpRatio should be in (0, UINT32_MAX].");
        return ge::GRAPH_FAILED;
    }
    if (*slotMappingFormatPtr != SLOT_MAPPING_FLAT && *slotMappingFormatPtr != SLOT_MAPPING_BLOCK_OFFSET) {
        OP_LOGE(context->GetNodeName(), "slotMappingFormat should be 1(flat) or 2(block_offset).");
        return ge::GRAPH_FAILED;
    }
    auto slotMappingDimNum = slotMappingShape->GetStorageShape().GetDimNum();
    if ((*slotMappingFormatPtr == SLOT_MAPPING_FLAT &&
         (slotMappingDimNum != 1 || slotMappingShape->GetStorageShape().GetDim(0) != numRows)) ||
        (*slotMappingFormatPtr == SLOT_MAPPING_BLOCK_OFFSET &&
         (slotMappingDimNum != 2 || slotMappingShape->GetStorageShape().GetDim(0) != numRows ||
          slotMappingShape->GetStorageShape().GetDim(1) != 2))) {
        OP_LOGE(context->GetNodeName(), "slotMapping shape does not match slotMappingFormat.");
        return ge::GRAPH_FAILED;
    }
    if (*actualNumReqsPtr <= 0 || *actualNumReqsPtr >= cuSeqlensDim0 ||
        *actualNumReqsPtr > startPosShape->GetStorageShape().GetDim(0) ||
        *actualNumReqsPtr > kvBlockTableRows ||
        *actualNumReqsPtr > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "actualNumReqs is invalid.");
        return ge::GRAPH_FAILED;
    }

    CompressorMetadataTilingData tilingData;
    tilingData.set_numRows(static_cast<uint32_t>(numRows));
    tilingData.set_numReqs(static_cast<uint32_t>(cuSeqlensDim0 - 1));
    tilingData.set_actualNumReqs(static_cast<uint32_t>(*actualNumReqsPtr));
    tilingData.set_ropeRows(static_cast<uint32_t>(ropeRows));
    tilingData.set_ropeDim(static_cast<uint32_t>(ropeDim));
    tilingData.set_kvBlockTableStride(static_cast<uint32_t>(kvBlockTableStride));
    tilingData.set_kvBlockSize(static_cast<uint32_t>(*kvBlockSizePtr));
    tilingData.set_slotMappingFormat(static_cast<uint32_t>(*slotMappingFormatPtr));
    tilingData.set_cmpRatio(static_cast<uint32_t>(*cmpRatioPtr));

    auto ropeDesc = context->GetInputDesc(ROPE_COS_INDEX);
    auto ropeSinDesc = context->GetInputDesc(ROPE_SIN_INDEX);
    auto compressCosDesc = context->GetOutputDesc(COMPRESS_COS_INDEX);
    auto compressSinDesc = context->GetOutputDesc(COMPRESS_SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeSinDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, compressCosDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, compressSinDesc);
    auto ropeDtype = ropeDesc->GetDataType();
    if (ropeSinDesc->GetDataType() != ropeDtype ||
        compressCosDesc->GetDataType() != ropeDtype ||
        compressSinDesc->GetDataType() != ropeDtype) {
        OP_LOGE(context->GetNodeName(), "rope and compress output dtypes should match.");
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = 0;
    uint32_t dtypeSize = 0;
    if (ropeDtype == ge::DataType::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
        dtypeSize = sizeof(float);
    } else if (ropeDtype == ge::DataType::DT_FLOAT16) {
        tilingKey = TILING_KEY_FLOAT16;
        dtypeSize = sizeof(uint16_t);
    } else if (ropeDtype == ge::DataType::DT_BF16) {
        tilingKey = TILING_KEY_BF16;
        dtypeSize = sizeof(uint16_t);
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported rope dtype.");
        return ge::GRAPH_FAILED;
    }

    uint32_t actualNumReqs = static_cast<uint32_t>(*actualNumReqsPtr);
    uint32_t cmpRatio = static_cast<uint32_t>(*cmpRatioPtr);
    if (static_cast<uint64_t>(ropeDim) * dtypeSize > MAX_UINT32_VALUE ||
        (static_cast<uint64_t>(actualNumReqs) + 1) * sizeof(int32_t) > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "tiling byte size exceeds UINT32_MAX.");
        return ge::GRAPH_FAILED;
    }
    uint32_t ropeRowBytes = static_cast<uint32_t>(ropeDim) * dtypeSize;
    if (static_cast<uint64_t>(cmpRatio - 1) * ropeRowBytes > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "rope stride exceeds UINT32_MAX.");
        return ge::GRAPH_FAILED;
    }
    uint32_t ropeRowBytesAligned = AlignUp(ropeRowBytes, ALIGN_BYTES);
    uint32_t slotCols = (*slotMappingFormatPtr == SLOT_MAPPING_FLAT) ? 1U : 2U;
    uint32_t reqTableBytes = AlignUp((static_cast<uint64_t>(actualNumReqs) + 1) * sizeof(int32_t), ALIGN_BYTES);
    uint64_t fixedUbBytes = static_cast<uint64_t>(reqTableBytes) * 3 + ALIGN_BYTES + UB_RESERVED_BYTES;
    uint64_t rowUbBytes =
        static_cast<uint64_t>(BUFFER_NUM) * ropeRowBytesAligned * 2 + slotCols * sizeof(int32_t) + sizeof(int32_t);
    if (rowUbBytes > MAX_UINT32_VALUE) {
        OP_LOGE(context->GetNodeName(), "row UB footprint exceeds UINT32_MAX.");
        return ge::GRAPH_FAILED;
    }
    uint64_t minUbBytes = static_cast<uint64_t>(reqTableBytes) * 3 + ALIGN_BYTES + rowUbBytes;
    if (ubSize <= minUbBytes) {
        OP_LOGE(context->GetNodeName(), "UB size is insufficient for compressor metadata.");
        return ge::GRAPH_FAILED;
    }
    uint32_t tileRows = 1;
    if (ubSize > fixedUbBytes && rowUbBytes > 0) {
        tileRows = static_cast<uint32_t>((ubSize - fixedUbBytes) / rowUbBytes);
        tileRows = std::max(tileRows, 1U);
    }
    tileRows = std::min(tileRows, MAX_TILE_ROWS);
    tileRows = std::min(tileRows, MAX_DATACOPY_BLOCK_COUNT);

    uint32_t usedCoreNum =
        std::min(aivCoreNum, std::max(1U, CeilDiv(static_cast<uint64_t>(numRows), ROWS_PER_CORE_TARGET)));
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_tileRows(tileRows);
    tilingData.set_ropeRowBytes(ropeRowBytes);
    tilingData.set_ropeRowBytesAligned(ropeRowBytesAligned);
    tilingData.set_slotCols(slotCols);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
    *workspaceSize = 0;
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCompressorMetadata(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorMetadata)
    .Tiling(CompressorMetadataTilingFunc)
    .TilingParse<CompressorMetadataCompileInfo>(TilingParseForCompressorMetadata);

}  // namespace optiling
