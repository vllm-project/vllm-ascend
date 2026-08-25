/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "sfa_remap_sparse_indices_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {
constexpr uint32_t INPUT_INDEX = 0;
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t DCP_SIZE_ATTR_INDEX = 0;
constexpr uint32_t DCP_RANK_ATTR_INDEX = 1;
constexpr uint32_t INTERLEAVE_SIZE_ATTR_INDEX = 2;
constexpr uint32_t MAX_TOP_K = 8192;
constexpr uint32_t MAX_CHUNK_ELEMENTS = 2048;
constexpr uint32_t MIN_CHUNK_ELEMENTS = 256;
constexpr uint32_t VECTOR_ALIGN_ELEMENTS = 256;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
constexpr uint32_t POWER_OF_TWO_BUFFER_COUNT = 4;
constexpr uint32_t VECTOR_MAGIC_BUFFER_COUNT = 8;
constexpr uint32_t MASK_BUFFER_COUNT = 3;
constexpr uint32_t LIBDIVIDE_ADD_MARKER = 0x40;

uint32_t AlignUp(uint64_t value, uint32_t alignment)
{
    return static_cast<uint32_t>((value + alignment - 1) / alignment * alignment);
}

uint32_t Log2Floor(uint32_t value)
{
    uint32_t shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

struct U32MagicDivider {
    uint32_t magic;
    uint32_t more;
};

// Host-side equivalent of libdivide_u32_gen. The A3 kernel synthesizes the
// corresponding unsigned mulhi exactly from 16-bit partial products.
U32MagicDivider MakeU32MagicDivider(uint32_t divisor)
{
    uint32_t floorLog2 = Log2Floor(divisor);
    if (IsPowerOfTwo(divisor)) {
        return {0, floorLog2};
    }

    uint64_t dividend = uint64_t{1} << (32 + floorLog2);
    uint32_t proposedMagic = static_cast<uint32_t>(dividend / divisor);
    uint32_t remainder = static_cast<uint32_t>(dividend % divisor);
    uint32_t error = divisor - remainder;
    uint32_t more = floorLog2;
    if (error >= (uint32_t{1} << floorLog2)) {
        proposedMagic += proposedMagic;
        uint32_t twiceRemainder = remainder + remainder;
        if (twiceRemainder >= divisor || twiceRemainder < remainder) {
            ++proposedMagic;
        }
        more |= LIBDIVIDE_ADD_MARKER;
    }
    return {proposedMagic + 1, more};
}

}  // namespace

static ge::graphStatus SfaRemapSparseIndicesTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

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

    auto inputShape = context->GetInputShape(INPUT_INDEX);
    auto outputShape = context->GetOutputShape(OUTPUT_INDEX);
    auto inputDesc = context->GetInputDesc(INPUT_INDEX);
    auto outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    const auto& inputStorageShape = inputShape->GetStorageShape();
    const auto& outputStorageShape = outputShape->GetStorageShape();
    size_t dimNum = inputStorageShape.GetDimNum();
    if (dimNum == 0 || outputStorageShape.GetDimNum() != dimNum) {
        OP_LOGE(context->GetNodeName(), "Input and output must have the same non-zero rank.");
        return ge::GRAPH_FAILED;
    }
    if (inputDesc->GetDataType() != ge::DT_INT32 || outputDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "Input and output dtype must be int32.");
        return ge::GRAPH_FAILED;
    }

    uint64_t elementCount = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        int64_t inputDim = inputStorageShape.GetDim(i);
        int64_t outputDim = outputStorageShape.GetDim(i);
        if (inputDim <= 0 || inputDim != outputDim) {
            OP_LOGE(context->GetNodeName(), "Input and output shapes must be equal and non-empty.");
            return ge::GRAPH_FAILED;
        }
        if (elementCount > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(inputDim)) {
            OP_LOGE(context->GetNodeName(), "Input element count exceeds UINT32_MAX.");
            return ge::GRAPH_FAILED;
        }
        elementCount *= static_cast<uint64_t>(inputDim);
    }

    int64_t topKValue = inputStorageShape.GetDim(dimNum - 1);
    if (topKValue <= 0 || topKValue > MAX_TOP_K) {
        OP_LOGE(context->GetNodeName(), "The last dimension must be in [1, 8192].");
        return ge::GRAPH_FAILED;
    }
    uint32_t topK = static_cast<uint32_t>(topKValue);
    uint64_t rowsValue = elementCount / topK;
    if (rowsValue == 0 || rowsValue > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "Row count is outside the uint32 range.");
        return ge::GRAPH_FAILED;
    }
    uint32_t rows = static_cast<uint32_t>(rowsValue);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dcpSizePtr = attrs->GetInt(DCP_SIZE_ATTR_INDEX);
    const int64_t* dcpRankPtr = attrs->GetInt(DCP_RANK_ATTR_INDEX);
    const int64_t* interleaveSizePtr = attrs->GetInt(INTERLEAVE_SIZE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dcpSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, dcpRankPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, interleaveSizePtr);
    if (*dcpSizePtr <= 1 || *dcpSizePtr > std::numeric_limits<int32_t>::max() ||
        *dcpRankPtr < 0 || *dcpRankPtr >= *dcpSizePtr ||
        *interleaveSizePtr <= 0 || *interleaveSizePtr > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "Invalid dcpSize, dcpRank, or interleaveSize.");
        return ge::GRAPH_FAILED;
    }

    uint32_t dcpSize = static_cast<uint32_t>(*dcpSizePtr);
    uint32_t dcpRank = static_cast<uint32_t>(*dcpRankPtr);
    uint32_t interleaveSize = static_cast<uint32_t>(*interleaveSizePtr);
    uint32_t interleaveShift = Log2Floor(interleaveSize);
    uint32_t dcpInterleaveShift = interleaveShift + Log2Floor(dcpSize);
    bool usePowerOfTwo = IsPowerOfTwo(dcpSize) && IsPowerOfTwo(interleaveSize);
    U32MagicDivider interleaveDivider = MakeU32MagicDivider(interleaveSize);
    U32MagicDivider dcpDivider = MakeU32MagicDivider(dcpSize);

    // Process wide rows in stable chunks. This keeps the exact Vector path for
    // every supported topK instead of falling back to a Scalar compaction when
    // the full row does not fit in UB.
    uint32_t fullBufferCount = usePowerOfTwo ? POWER_OF_TWO_BUFFER_COUNT : VECTOR_MAGIC_BUFFER_COUNT;
    uint32_t chunkElements = std::min(topK, MAX_CHUNK_ELEMENTS);
    uint32_t alignedElements = 0;
    uint32_t bufferBytes = 0;
    uint32_t maskBytes = 0;
    uint64_t requiredUbBytes = 0;
    while (true) {
        alignedElements = AlignUp(chunkElements, VECTOR_ALIGN_ELEMENTS);
        bufferBytes = AlignUp(static_cast<uint64_t>(alignedElements) * sizeof(int32_t), ALIGN_BYTES);
        maskBytes = AlignUp((alignedElements + 7) / 8, ALIGN_BYTES);
        requiredUbBytes = static_cast<uint64_t>(bufferBytes) * fullBufferCount +
                          static_cast<uint64_t>(maskBytes) * MASK_BUFFER_COUNT + UB_RESERVED_BYTES;
        if (requiredUbBytes <= ubSize) {
            break;
        }
        if (chunkElements <= MIN_CHUNK_ELEMENTS) {
            break;
        }
        chunkElements = std::max(MIN_CHUNK_ELEMENTS, chunkElements / 2);
    }
    if (requiredUbBytes > ubSize) {
        OP_LOGE(context->GetNodeName(),
                "UB size is insufficient for a 256-element Vector chunk: required=%lu, available=%lu.",
                requiredUbBytes, ubSize);
        return ge::GRAPH_FAILED;
    }
    bool useVectorMagicDivision = !usePowerOfTwo;

    uint32_t usedCoreNum = std::min(rows, aivCoreNum);
    uint32_t rowsPerCore = (rows + usedCoreNum - 1) / usedCoreNum;

    SfaRemapSparseIndicesTilingData tilingData;
    tilingData.set_rows(rows);
    tilingData.set_topK(topK);
    tilingData.set_dcpSize(dcpSize);
    tilingData.set_dcpRank(dcpRank);
    tilingData.set_interleaveSize(interleaveSize);
    tilingData.set_interleaveShift(interleaveShift);
    tilingData.set_dcpInterleaveShift(dcpInterleaveShift);
    tilingData.set_usePowerOfTwo(static_cast<uint32_t>(usePowerOfTwo));
    tilingData.set_useVectorMagicDivision(static_cast<uint32_t>(useVectorMagicDivision));
    tilingData.set_interleaveMagic(interleaveDivider.magic);
    tilingData.set_interleaveMore(interleaveDivider.more);
    tilingData.set_dcpMagic(dcpDivider.magic);
    tilingData.set_dcpMore(dcpDivider.more);
    tilingData.set_rowsPerCore(rowsPerCore);
    tilingData.set_chunkElements(chunkElements);
    tilingData.set_bufferBytes(bufferBytes);
    tilingData.set_maskBytes(maskBytes);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSize);
    *workspaceSize = 0;
    context->SetBlockDim(usedCoreNum);
    // The generated kernel has one entry, so every shape dispatches through
    // tiling key 0.
    context->SetTilingKey(0);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSfaRemapSparseIndices(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SfaRemapSparseIndices)
    .Tiling(SfaRemapSparseIndicesTilingFunc)
    .TilingParse<SfaRemapSparseIndicesCompileInfo>(TilingParseForSfaRemapSparseIndices);
}  // namespace optiling
