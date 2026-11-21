#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../op_kernel/kvtc_restore_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr size_t RECONSTRUCTED_IR_INDEX = 0;
constexpr size_t DST_BLOCK_IDS_IR_INDEX = 1;
constexpr size_t CACHES_IR_INDEX = 2;

constexpr uint32_t COPY_BLOCK_BYTES = 32U;
constexpr uint64_t UB_SAFETY_BYTES = 16ULL * 1024ULL;
constexpr uint32_t BUFFER_NUM = 2U;

bool ToU32(int64_t value, uint32_t& out)
{
    if (value <= 0 || static_cast<uint64_t>(value) >
            std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

bool MulU32(uint32_t a, uint32_t b, uint32_t& out)
{
    const uint64_t value = static_cast<uint64_t>(a) * b;
    if (value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

uint32_t ChooseRowsPerTile(
    uint32_t selectedBlocks,
    uint32_t numLayers,
    uint32_t blockSize,
    uint32_t layerSize,
    uint32_t availableAiv,
    uint64_t ubBytes)
{
    if (selectedBlocks == 0U || numLayers == 0U || blockSize == 0U ||
        layerSize == 0U || availableAiv == 0U ||
        ubBytes <= UB_SAFETY_BYTES) {
        return 0U;
    }

    constexpr uint32_t candidates[] = {
        512U, 256U, 128U, 64U, 32U, 16U, 8U, 4U, 2U, 1U};

    for (uint32_t rows : candidates) {
        if (rows > blockSize || blockSize % rows != 0U) {
            continue;
        }
        const uint64_t tileBytes =
            static_cast<uint64_t>(rows) * layerSize * sizeof(uint16_t);
        if (BUFFER_NUM * tileBytes + UB_SAFETY_BYTES > ubBytes) {
            continue;
        }
        const uint64_t tasks =
            static_cast<uint64_t>(selectedBlocks) * numLayers *
            (blockSize / rows);
        if (tasks >= availableAiv) {
            return rows;
        }
    }

    for (uint32_t rows : candidates) {
        if (rows > blockSize || blockSize % rows != 0U) {
            continue;
        }
        const uint64_t tileBytes =
            static_cast<uint64_t>(rows) * layerSize * sizeof(uint16_t);
        if (BUFFER_NUM * tileBytes + UB_SAFETY_BYTES <= ubBytes) {
            return rows;
        }
    }
    return 0U;
}

ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* reconstructedStorage =
        context->GetRequiredInputShape(RECONSTRUCTED_IR_INDEX);
    const gert::StorageShape* dstIdsStorage =
        context->GetRequiredInputShape(DST_BLOCK_IDS_IR_INDEX);
    const gert::StorageShape* firstCacheStorage =
        context->GetDynamicInputShape(CACHES_IR_INDEX, 0);
    if (reconstructedStorage == nullptr || dstIdsStorage == nullptr ||
        firstCacheStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& reconstructed =
        reconstructedStorage->GetStorageShape();
    const gert::Shape& dstIds = dstIdsStorage->GetStorageShape();
    const gert::Shape& firstCache = firstCacheStorage->GetStorageShape();
    if (reconstructed.GetDimNum() != 2 || dstIds.GetDimNum() != 1 ||
        firstCache.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    uint32_t selectedBlocks = 0U;
    uint32_t numGpuBlocks = 0U;
    uint32_t blockSize = 0U;
    uint32_t numKvHeads = 0U;
    uint32_t headSize = 0U;
    if (!ToU32(dstIds.GetDim(0), selectedBlocks) ||
        !ToU32(firstCache.GetDim(0), numGpuBlocks) ||
        !ToU32(firstCache.GetDim(1), blockSize) ||
        !ToU32(firstCache.GetDim(2), numKvHeads) ||
        !ToU32(firstCache.GetDim(3), headSize)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t numLayers = 0U;
    while (true) {
        const gert::StorageShape* cacheStorage =
            context->GetDynamicInputShape(
                CACHES_IR_INDEX, static_cast<size_t>(numLayers));
        if (cacheStorage == nullptr) {
            break;
        }
        const gert::Shape& cache = cacheStorage->GetStorageShape();
        if (cache.GetDimNum() != 4 ||
            cache.GetDim(0) != static_cast<int64_t>(numGpuBlocks) ||
            cache.GetDim(1) != static_cast<int64_t>(blockSize) ||
            cache.GetDim(2) != static_cast<int64_t>(numKvHeads) ||
            cache.GetDim(3) != static_cast<int64_t>(headSize) ||
            numLayers == std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        ++numLayers;
    }

    uint32_t layerSize = 0U;
    uint32_t k = 0U;
    uint32_t m = 0U;
    if (numLayers == 0U ||
        !MulU32(numKvHeads, headSize, layerSize) ||
        !MulU32(numLayers, layerSize, k) ||
        !MulU32(selectedBlocks, blockSize, m) ||
        reconstructed.GetDim(0) != static_cast<int64_t>(m) ||
        reconstructed.GetDim(1) != static_cast<int64_t>(k) ||
        static_cast<uint64_t>(layerSize) * sizeof(uint16_t) %
                COPY_BLOCK_BYTES != 0U ||
        static_cast<uint64_t>(k) * sizeof(uint16_t) %
                COPY_BLOCK_BYTES != 0U) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableAiv = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (availableAiv == 0U || ubBytes == 0U) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rowsPerTile = ChooseRowsPerTile(
        selectedBlocks, numLayers, blockSize, layerSize,
        availableAiv, ubBytes);
    if (rowsPerTile == 0U) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rowTilesPerBlock = blockSize / rowsPerTile;
    uint32_t tasksPerSelected = 0U;
    uint32_t taskCount = 0U;
    uint32_t tileElements = 0U;
    if (!MulU32(rowTilesPerBlock, numLayers, tasksPerSelected) ||
        !MulU32(selectedBlocks, tasksPerSelected, taskCount) ||
        !MulU32(rowsPerTile, layerSize, tileElements)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t tileBytes64 =
        static_cast<uint64_t>(tileElements) * sizeof(uint16_t);
    const uint64_t blockLen32 =
        static_cast<uint64_t>(layerSize) * sizeof(uint16_t) /
        COPY_BLOCK_BYTES;
    const uint64_t inputStride32 =
        static_cast<uint64_t>(k - layerSize) * sizeof(uint16_t) /
        COPY_BLOCK_BYTES;
    if (BUFFER_NUM * tileBytes64 + UB_SAFETY_BYTES > ubBytes ||
        tileBytes64 > std::numeric_limits<uint32_t>::max() ||
        rowsPerTile > std::numeric_limits<uint16_t>::max() ||
        blockLen32 == 0U ||
        blockLen32 > std::numeric_limits<uint16_t>::max() ||
        inputStride32 > std::numeric_limits<uint16_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    KvtcRestoreTilingData* data =
        context->GetTilingData<KvtcRestoreTilingData>();
    if (data == nullptr) {
        return ge::GRAPH_FAILED;
    }

    data->numLayers = numLayers;
    data->numGpuBlocks = numGpuBlocks;
    data->selectedBlocks = selectedBlocks;
    data->blockSize = blockSize;
    data->numKvHeads = numKvHeads;
    data->headSize = headSize;
    data->layerSize = layerSize;
    data->k = k;
    data->rowsPerTile = rowsPerTile;
    data->rowTilesPerBlock = rowTilesPerBlock;
    data->taskCount = taskCount;
    data->usedAiv = std::min(availableAiv, taskCount);
    data->tileElements = tileElements;
    data->tileBytes = static_cast<uint32_t>(tileBytes64);
    data->blockLen32 = static_cast<uint32_t>(blockLen32);
    data->inputStride32 = static_cast<uint32_t>(inputStride32);

    context->SetBlockDim(data->usedAiv);
    context->SetTilingKey(0U);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(KvtcRestore).Tiling(TilingFunc);

} // namespace optiling
