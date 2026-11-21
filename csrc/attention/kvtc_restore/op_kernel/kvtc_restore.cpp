#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include "kvtc_restore_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t BUFFER_NUM = 2U;

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
void kvtc_restore(
    GM_ADDR reconstructed,
    GM_ADDR dstBlockIds,
    GM_ADDR caches,
    GM_ADDR status,
    GM_ADDR workspace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KvtcRestoreTilingData);
    GET_TILING_DATA(tiling, tilingGm);
    (void)workspace;

    const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
    if (tiling.usedAiv == 0U || core >= tiling.usedAiv) {
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
    const __gm__ int64_t* dstIdsPtr =
        reinterpret_cast<const __gm__ int64_t*>(dstBlockIds);

    // Only core 0 writes status. Check every input, including blocks assigned
    // to other cores, so a successful core cannot overwrite another's error.
    // Other cores retain their local bounds checks; invalid input may leave a
    // partially restored cache, and status is consumed only after completion.
    if (core == 0U) {
        int32_t result = 0;
        for (uint32_t selected = 0U; selected < tiling.selectedBlocks; ++selected) {
            const int64_t block = dstIdsPtr[selected];
            if (block < 0 || static_cast<uint64_t>(block) >= tiling.numGpuBlocks) {
                result = -1;
                break;
            }
        }
        for (uint32_t layer = 0U; result == 0 && layer < tiling.numLayers; ++layer) {
            if (cacheList.GetDataPtr<__gm__ bfloat16_t>(layer) == nullptr) {
                result = -1;
            }
        }
        GlobalTensor<int32_t> statusGm;
        statusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status), 1U);
        statusGm.SetValue(0U, result);
        if (result != 0) {
            return;
        }
    }

    const uint64_t cacheBlockElements =
        static_cast<uint64_t>(tiling.blockSize) * tiling.layerSize;
    const uint64_t cacheElements =
        static_cast<uint64_t>(tiling.numGpuBlocks) * cacheBlockElements;
    const uint64_t inputBlockElements =
        static_cast<uint64_t>(tiling.blockSize) * tiling.k;

    GlobalTensor<bfloat16_t> inputGm;
    inputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(reconstructed),
        static_cast<uint64_t>(tiling.selectedBlocks) * inputBlockElements);

    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> transferQueue;
    pipe.InitBuffer(transferQueue, BUFFER_NUM, tiling.tileBytes);

    const DataCopyParams copyInParams{
        static_cast<uint16_t>(tiling.rowsPerTile),
        static_cast<uint16_t>(tiling.blockLen32),
        static_cast<uint16_t>(tiling.inputStride32),
        0U};

    // Production B=128 uses one row tile. Keep its task loop division-free.
    if (tiling.rowTilesPerBlock == 1U) {
        uint32_t selected = taskBegin / tiling.numLayers;
        uint32_t layer = taskBegin - selected * tiling.numLayers;

        int64_t physicalBlock64 = dstIdsPtr[selected];
        if (physicalBlock64 < 0 ||
            static_cast<uint64_t>(physicalBlock64) >= tiling.numGpuBlocks) {
            return;
        }

        uint64_t inputLayerOffset =
            static_cast<uint64_t>(selected) * inputBlockElements +
            static_cast<uint64_t>(layer) * tiling.layerSize;
        uint64_t destinationBlockBase =
            static_cast<uint64_t>(physicalBlock64) * cacheBlockElements;

        for (uint32_t localTask = 0U;
             localTask < localTaskCount;
             ++localTask) {
            __gm__ bfloat16_t* cachePtr =
                cacheList.GetDataPtr<__gm__ bfloat16_t>(layer);
            if (cachePtr == nullptr) {
                return;
            }
            GlobalTensor<bfloat16_t> cacheGm;
            cacheGm.SetGlobalBuffer(cachePtr, cacheElements);

            LocalTensor<bfloat16_t> local =
                transferQueue.AllocTensor<bfloat16_t>();
            DataCopy(local, inputGm[inputLayerOffset], copyInParams);
            transferQueue.EnQue(local);

            local = transferQueue.DeQue<bfloat16_t>();
            DataCopy(cacheGm[destinationBlockBase], local, tiling.tileElements);
            transferQueue.FreeTensor(local);

            ++layer;
            inputLayerOffset += tiling.layerSize;
            if (layer == tiling.numLayers &&
                localTask + 1U < localTaskCount) {
                layer = 0U;
                ++selected;
                physicalBlock64 = dstIdsPtr[selected];
                if (physicalBlock64 < 0 ||
                    static_cast<uint64_t>(physicalBlock64) >=
                        tiling.numGpuBlocks) {
                    return;
                }
                inputLayerOffset =
                    static_cast<uint64_t>(selected) * inputBlockElements;
                destinationBlockBase =
                    static_cast<uint64_t>(physicalBlock64) *
                    cacheBlockElements;
            }
        }
        return;
    }

    // General small-shape path.
    uint32_t layer = taskBegin % tiling.numLayers;
    const uint32_t rowTask = taskBegin / tiling.numLayers;
    uint32_t rowTile = rowTask % tiling.rowTilesPerBlock;
    uint32_t selected = rowTask / tiling.rowTilesPerBlock;

    int64_t physicalBlock64 = dstIdsPtr[selected];
    if (physicalBlock64 < 0 ||
        static_cast<uint64_t>(physicalBlock64) >= tiling.numGpuBlocks) {
        return;
    }

    uint64_t inputRowBase =
        static_cast<uint64_t>(selected) * inputBlockElements +
        static_cast<uint64_t>(rowTile) * tiling.rowsPerTile * tiling.k;
    uint64_t inputLayerOffset =
        inputRowBase + static_cast<uint64_t>(layer) * tiling.layerSize;
    uint64_t destinationRowBase =
        static_cast<uint64_t>(physicalBlock64) * cacheBlockElements +
        static_cast<uint64_t>(rowTile) * tiling.tileElements;

    for (uint32_t localTask = 0U;
         localTask < localTaskCount;
         ++localTask) {
        __gm__ bfloat16_t* cachePtr =
            cacheList.GetDataPtr<__gm__ bfloat16_t>(layer);
        if (cachePtr == nullptr) {
            return;
        }
        GlobalTensor<bfloat16_t> cacheGm;
        cacheGm.SetGlobalBuffer(cachePtr, cacheElements);

        LocalTensor<bfloat16_t> local =
            transferQueue.AllocTensor<bfloat16_t>();
        DataCopy(local, inputGm[inputLayerOffset], copyInParams);
        transferQueue.EnQue(local);

        local = transferQueue.DeQue<bfloat16_t>();
        DataCopy(cacheGm[destinationRowBase], local, tiling.tileElements);
        transferQueue.FreeTensor(local);

        ++layer;
        inputLayerOffset += tiling.layerSize;
        if (layer != tiling.numLayers ||
            localTask + 1U == localTaskCount) {
            continue;
        }

        layer = 0U;
        ++rowTile;
        if (rowTile == tiling.rowTilesPerBlock) {
            rowTile = 0U;
            ++selected;
            physicalBlock64 = dstIdsPtr[selected];
            if (physicalBlock64 < 0 ||
                static_cast<uint64_t>(physicalBlock64) >=
                    tiling.numGpuBlocks) {
                return;
            }
            inputRowBase =
                static_cast<uint64_t>(selected) * inputBlockElements;
            destinationRowBase =
                static_cast<uint64_t>(physicalBlock64) * cacheBlockElements;
        } else {
            inputRowBase +=
                static_cast<uint64_t>(tiling.rowsPerTile) * tiling.k;
            destinationRowBase += tiling.tileElements;
        }
        inputLayerOffset = inputRowBase;
    }
}
