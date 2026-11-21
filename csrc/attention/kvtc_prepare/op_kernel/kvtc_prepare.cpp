#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include "kvtc_prepare_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t BUFFER_NUM = 2U;

// Split [0, totalTasks) into contiguous, balanced ranges. Compared with a
// grid-stride loop this keeps one AIV on consecutive cache layers and normally
// inside one selected block, so block_ids and address state are reused.
__aicore__ inline void GetTaskRange(
    uint32_t totalTasks,
    uint32_t usedAiv,
    uint32_t core,
    uint32_t& begin,
    uint32_t& count)
{
    const uint32_t tasksPerCore = totalTasks / usedAiv;
    const uint32_t extra = totalTasks - tasksPerCore * usedAiv;
    const uint32_t beforeExtra = core < extra ? core : extra;
    begin = core * tasksPerCore + beforeExtra;
    count = tasksPerCore + (core < extra ? 1U : 0U);
}

} // namespace

extern "C" __global__ __aicore__
void kvtc_prepare(
    GM_ADDR caches,
    GM_ADDR blockIds,
    GM_ADDR prepared,
    GM_ADDR workspace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KvtcPrepareTilingData);
    GET_TILING_DATA(tiling, tilingGm);
    (void)workspace;

    const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
    if (core >= tiling.usedAiv || tiling.usedAiv == 0U) {
        return;
    }

    uint32_t taskBegin = 0U;
    uint32_t localTaskCount = 0U;
    GetTaskRange(
        tiling.taskCount,
        tiling.usedAiv,
        core,
        taskBegin,
        localTaskCount);
    if (localTaskCount == 0U) {
        return;
    }

    ListTensorDesc cacheList(reinterpret_cast<__gm__ void*>(caches));
    const __gm__ int64_t* blockIdsPtr =
        reinterpret_cast<const __gm__ int64_t*>(blockIds);

    const uint64_t cacheBlockElements =
        static_cast<uint64_t>(tiling.blockSize) * tiling.layerSize;
    const uint64_t cacheElements =
        static_cast<uint64_t>(tiling.numGpuBlocks) * cacheBlockElements;
    const uint64_t outputBlockElements =
        static_cast<uint64_t>(tiling.blockSize) * tiling.k;

    GlobalTensor<bfloat16_t> outputGm;
    outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(prepared),
        static_cast<uint64_t>(tiling.selectedBlocks) *
            outputBlockElements);

    TPipe pipe;
    // No vector arithmetic is performed yet. TQueBind lets MTE2 and MTE3
    // reuse two UB buffers and preserves the working double-buffer pipeline.
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> transferQueue;
    pipe.InitBuffer(transferQueue, BUFFER_NUM, tiling.tileBytes);

    const DataCopyParams copyOutParams{
        static_cast<uint16_t>(tiling.rowsPerTile),
        static_cast<uint16_t>(tiling.blockLen32),
        0U,
        static_cast<uint16_t>(tiling.outputStride32)};

    // Production geometry enters this path: a task copies one complete
    // [B,H,D] cache layer. The loop contains no integer divide or modulo and
    // reloads block_ids only when it crosses a selected block.
    if (tiling.rowTilesPerBlock == 1U) {
        uint32_t selected = taskBegin / tiling.numLayers;
        uint32_t layer = taskBegin - selected * tiling.numLayers;

        int64_t physicalBlock64 = blockIdsPtr[selected];
        if (physicalBlock64 < 0 ||
            static_cast<uint64_t>(physicalBlock64) >=
                tiling.numGpuBlocks) {
            return;
        }
        uint64_t sourceBlockBase =
            static_cast<uint64_t>(physicalBlock64) * cacheBlockElements;
        uint64_t outputLayerOffset =
            static_cast<uint64_t>(selected) * outputBlockElements +
            static_cast<uint64_t>(layer) * tiling.layerSize;

        for (uint32_t localTask = 0U;
             localTask < localTaskCount;
             ++localTask) {
            __gm__ bfloat16_t* cachePtr =
                cacheList.GetDataPtr<__gm__ bfloat16_t>(layer);
            GlobalTensor<bfloat16_t> cacheGm;
            cacheGm.SetGlobalBuffer(cachePtr, cacheElements);

            LocalTensor<bfloat16_t> local =
                transferQueue.AllocTensor<bfloat16_t>();
            DataCopy(
                local,
                cacheGm[sourceBlockBase],
                tiling.tileElements);
            transferQueue.EnQue(local);

            local = transferQueue.DeQue<bfloat16_t>();
            DataCopy(outputGm[outputLayerOffset], local, copyOutParams);
            transferQueue.FreeTensor(local);

            ++layer;
            outputLayerOffset += tiling.layerSize;
            if (layer == tiling.numLayers &&
                localTask + 1U < localTaskCount) {
                layer = 0U;
                ++selected;
                physicalBlock64 = blockIdsPtr[selected];
                if (physicalBlock64 < 0 ||
                    static_cast<uint64_t>(physicalBlock64) >=
                        tiling.numGpuBlocks) {
                    return;
                }
                sourceBlockBase =
                    static_cast<uint64_t>(physicalBlock64) *
                    cacheBlockElements;
                outputLayerOffset =
                    static_cast<uint64_t>(selected) *
                    outputBlockElements;
            }
        }
        return;
    }

    // General path for small-L tests and shapes where a full cache layer does
    // not expose enough AIV work. State is decoded once at taskBegin and then
    // advanced with additions instead of four dynamic divisions per task.
    uint32_t layer = taskBegin % tiling.numLayers;
    const uint32_t rowTask = taskBegin / tiling.numLayers;
    uint32_t rowTileIndex = rowTask % tiling.rowTilesPerBlock;
    uint32_t selected = rowTask / tiling.rowTilesPerBlock;

    int64_t physicalBlock64 = blockIdsPtr[selected];
    if (physicalBlock64 < 0 ||
        static_cast<uint64_t>(physicalBlock64) >= tiling.numGpuBlocks) {
        return;
    }

    uint64_t sourceRowBase =
        static_cast<uint64_t>(physicalBlock64) * cacheBlockElements +
        static_cast<uint64_t>(rowTileIndex) * tiling.tileElements;
    uint64_t outputRowBase =
        static_cast<uint64_t>(selected) * outputBlockElements +
        static_cast<uint64_t>(rowTileIndex) * tiling.rowsPerTile * tiling.k;
    uint64_t outputLayerOffset =
        outputRowBase + static_cast<uint64_t>(layer) * tiling.layerSize;

    for (uint32_t localTask = 0U;
         localTask < localTaskCount;
         ++localTask) {
        __gm__ bfloat16_t* cachePtr =
            cacheList.GetDataPtr<__gm__ bfloat16_t>(layer);
        GlobalTensor<bfloat16_t> cacheGm;
        cacheGm.SetGlobalBuffer(cachePtr, cacheElements);

        LocalTensor<bfloat16_t> local =
            transferQueue.AllocTensor<bfloat16_t>();
        DataCopy(local, cacheGm[sourceRowBase], tiling.tileElements);
        transferQueue.EnQue(local);

        local = transferQueue.DeQue<bfloat16_t>();
        DataCopy(outputGm[outputLayerOffset], local, copyOutParams);
        transferQueue.FreeTensor(local);

        ++layer;
        outputLayerOffset += tiling.layerSize;
        if (layer != tiling.numLayers ||
            localTask + 1U == localTaskCount) {
            continue;
        }

        layer = 0U;
        ++rowTileIndex;
        if (rowTileIndex == tiling.rowTilesPerBlock) {
            rowTileIndex = 0U;
            ++selected;
            physicalBlock64 = blockIdsPtr[selected];
            if (physicalBlock64 < 0 ||
                static_cast<uint64_t>(physicalBlock64) >=
                    tiling.numGpuBlocks) {
                return;
            }
            sourceRowBase =
                static_cast<uint64_t>(physicalBlock64) *
                cacheBlockElements;
            outputRowBase =
                static_cast<uint64_t>(selected) * outputBlockElements;
        } else {
            sourceRowBase += tiling.tileElements;
            outputRowBase +=
                static_cast<uint64_t>(tiling.rowsPerTile) * tiling.k;
        }
        outputLayerOffset = outputRowBase;
    }
}
