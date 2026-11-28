#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include "kvtc_restore_rope_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t DATA_BUFFER_NUM = 2U;

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

__aicore__ inline void GetLayerRange(
    uint32_t numLayers,
    uint32_t layerGroups,
    uint32_t group,
    uint32_t& begin,
    uint32_t& count)
{
    const uint32_t layersPerGroup = numLayers / layerGroups;
    const uint32_t extra = numLayers - layersPerGroup * layerGroups;
    const uint32_t beforeExtra = group < extra ? group : extra;
    begin = group * layersPerGroup + beforeExtra;
    count = layersPerGroup + (group < extra ? 1U : 0U);
}

} // namespace

extern "C" __global__ __aicore__
void kvtc_restore_rope(
    GM_ADDR reconstructed,
    GM_ADDR dstBlockIds,
    GM_ADDR tokenStarts,
    GM_ADDR cosSinCache,
    GM_ADDR caches,
    GM_ADDR status,
    GM_ADDR workspace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KvtcRestoreRopeTilingData);
    GET_TILING_DATA(tiling, tilingGm);
    (void)workspace;

    const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
    if (tiling.usedAiv == 0U || core >= tiling.usedAiv ||
        tiling.layerGroups == 0U || tiling.rowsPerTile == 0U) {
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
    const __gm__ int64_t* tokenStartsPtr =
        reinterpret_cast<const __gm__ int64_t*>(tokenStarts);

    // Only core 0 writes status. Check every input, including blocks assigned
    // to other cores, so a successful core cannot overwrite another's error.
    // Other cores retain their local bounds checks; invalid input may leave a
    // partially restored cache, and status is consumed only after completion.
    if (core == 0U) {
        int32_t result = 0;
        for (uint32_t selected = 0U; selected < tiling.selectedBlocks; ++selected) {
            const int64_t block = dstIdsPtr[selected];
            const int64_t start = tokenStartsPtr[selected];
            if (block < 0 || static_cast<uint64_t>(block) >= tiling.numGpuBlocks ||
                start < 0 || static_cast<uint64_t>(start) + tiling.blockSize > tiling.ropeCacheRows) {
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
    const uint32_t halfRotary = tiling.headSize / 2U;

    GlobalTensor<bfloat16_t> inputGm;
    inputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(reconstructed),
        static_cast<uint64_t>(tiling.selectedBlocks) * inputBlockElements);

    GlobalTensor<float> ropeGm;
    ropeGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(cosSinCache),
        static_cast<uint64_t>(tiling.ropeCacheRows) * tiling.headSize);

    TPipe pipe;
    TQue<TPosition::VECIN, DATA_BUFFER_NUM> inputQueue;
    TQue<TPosition::VECOUT, DATA_BUFFER_NUM> outputQueue;
    TQue<TPosition::VECIN, 1> ropeQueue;
    pipe.InitBuffer(inputQueue, DATA_BUFFER_NUM, tiling.tileBytes);
    pipe.InitBuffer(outputQueue, DATA_BUFFER_NUM, tiling.tileBytes);
    pipe.InitBuffer(ropeQueue, 1, tiling.ropeTileBytes);

    TBuf<TPosition::VECCALC> inputFloatBuf;
    TBuf<TPosition::VECCALC> outputFloatBuf;
    TBuf<TPosition::VECCALC> ropeTmpBuf;
    pipe.InitBuffer(inputFloatBuf, tiling.layerSize * sizeof(float));
    pipe.InitBuffer(outputFloatBuf, tiling.layerSize * sizeof(float));
    pipe.InitBuffer(
        ropeTmpBuf,
        tiling.numKvHeads * halfRotary * sizeof(float));

    LocalTensor<float> inputFloat =
        inputFloatBuf.Get<float>(tiling.layerSize);
    LocalTensor<float> outputFloat =
        outputFloatBuf.Get<float>(tiling.layerSize);
    LocalTensor<float> ropeTmp =
        ropeTmpBuf.Get<float>(tiling.numKvHeads * halfRotary);

    const uint8_t headRepeat =
        static_cast<uint8_t>(tiling.numKvHeads);
    const uint8_t fullHeadStride =
        static_cast<uint8_t>(tiling.headSize * sizeof(float) / 32U);
    const uint8_t halfHeadStride =
        static_cast<uint8_t>(halfRotary * sizeof(float) / 32U);
    const BinaryRepeatParams mulFullParams{
        1U, 1U, 1U,
        fullHeadStride, fullHeadStride, 0U};
    const BinaryRepeatParams mulHalfParams{
        1U, 1U, 1U,
        halfHeadStride, fullHeadStride, 0U};
    const BinaryRepeatParams combineParams{
        1U, 1U, 1U,
        fullHeadStride, fullHeadStride, halfHeadStride};

    const DataCopyParams copyInParams{
        static_cast<uint16_t>(tiling.rowsPerTile),
        static_cast<uint16_t>(tiling.blockLen32),
        static_cast<uint16_t>(tiling.inputStride32),
        0U};

    for (uint32_t localTask = 0U;
         localTask < localTaskCount;
         ++localTask) {
        const uint32_t task = taskBegin + localTask;
        const uint32_t layerGroup = task % tiling.layerGroups;
        const uint32_t rowTask = task / tiling.layerGroups;
        const uint32_t rowTile = rowTask % tiling.rowTilesPerBlock;
        const uint32_t selected = rowTask / tiling.rowTilesPerBlock;

        const int64_t physicalBlock64 = dstIdsPtr[selected];
        const int64_t tokenStart64 = tokenStartsPtr[selected];
        if (physicalBlock64 < 0 || tokenStart64 < 0 ||
            static_cast<uint64_t>(physicalBlock64) >=
                tiling.numGpuBlocks) {
            return;
        }

        const uint32_t rowStart = rowTile * tiling.rowsPerTile;
        const uint64_t positionStart =
            static_cast<uint64_t>(tokenStart64) + rowStart;
        if (positionStart + tiling.rowsPerTile > tiling.ropeCacheRows) {
            return;
        }

        uint32_t layerBegin = 0U;
        uint32_t layerCount = 0U;
        GetLayerRange(
            tiling.numLayers,
            tiling.layerGroups,
            layerGroup,
            layerBegin,
            layerCount);
        if (layerCount == 0U) {
            continue;
        }

        LocalTensor<float> ropeLocal = ropeQueue.AllocTensor<float>();
        DataCopy(
            ropeLocal,
            ropeGm[positionStart * tiling.headSize],
            tiling.ropeTileElements);
        ropeQueue.EnQue(ropeLocal);
        ropeLocal = ropeQueue.DeQue<float>();

        const uint64_t inputRowBase =
            static_cast<uint64_t>(selected) * inputBlockElements +
            static_cast<uint64_t>(rowStart) * tiling.k;
        const uint64_t destinationRowBase =
            static_cast<uint64_t>(physicalBlock64) * cacheBlockElements +
            static_cast<uint64_t>(rowStart) * tiling.layerSize;

        LocalTensor<bfloat16_t> firstInput =
            inputQueue.AllocTensor<bfloat16_t>();
        DataCopy(
            firstInput,
            inputGm[inputRowBase +
                static_cast<uint64_t>(layerBegin) * tiling.layerSize],
            copyInParams);
        inputQueue.EnQue(firstInput);

        for (uint32_t layerOffset = 0U;
             layerOffset < layerCount;
             ++layerOffset) {
            const uint32_t layer = layerBegin + layerOffset;
            LocalTensor<bfloat16_t> inputLocal =
                inputQueue.DeQue<bfloat16_t>();

            if (layerOffset + 1U < layerCount) {
                const uint32_t nextLayer = layer + 1U;
                LocalTensor<bfloat16_t> nextInput =
                    inputQueue.AllocTensor<bfloat16_t>();
                DataCopy(
                    nextInput,
                    inputGm[inputRowBase +
                        static_cast<uint64_t>(nextLayer) *
                            tiling.layerSize],
                    copyInParams);
                inputQueue.EnQue(nextInput);
            }

            LocalTensor<bfloat16_t> outputLocal =
                outputQueue.AllocTensor<bfloat16_t>();

            for (uint32_t row = 0U;
                 row < tiling.rowsPerTile;
                 ++row) {
                const uint32_t ropeOffset = row * tiling.headSize;

                for (uint32_t head = 0U;
                     head < tiling.numKvHeads;
                     ++head) {
                    const uint32_t dataOffset =
                        row * tiling.layerSize + head * tiling.headSize;
                    Cast(
                        inputFloat[head * tiling.headSize],
                        inputLocal[dataOffset],
                        RoundMode::CAST_NONE,
                        tiling.headSize);
                }
                PipeBarrier<PIPE_V>();

                // Forward NeoX RoPE. Coefficient row is [cos | +sin].
                // [a|b] -> [a*cos - b*sin | b*cos + a*sin].
                Mul(
                    outputFloat,
                    inputFloat,
                    ropeLocal[ropeOffset],
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    mulFullParams);
                Mul(
                    ropeTmp,
                    inputFloat[halfRotary],
                    ropeLocal[ropeOffset + halfRotary],
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    mulHalfParams);
                PipeBarrier<PIPE_V>();
                Sub(
                    outputFloat,
                    outputFloat,
                    ropeTmp,
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    combineParams);
                PipeBarrier<PIPE_V>();

                Mul(
                    outputFloat[halfRotary],
                    inputFloat[halfRotary],
                    ropeLocal[ropeOffset],
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    mulFullParams);
                Mul(
                    ropeTmp,
                    inputFloat,
                    ropeLocal[ropeOffset + halfRotary],
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    mulHalfParams);
                PipeBarrier<PIPE_V>();
                Add(
                    outputFloat[halfRotary],
                    outputFloat[halfRotary],
                    ropeTmp,
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    combineParams);
                PipeBarrier<PIPE_V>();

                for (uint32_t head = 0U;
                     head < tiling.numKvHeads;
                     ++head) {
                    const uint32_t dataOffset =
                        row * tiling.layerSize + head * tiling.headSize;
                    Cast(
                        outputLocal[dataOffset],
                        outputFloat[head * tiling.headSize],
                        RoundMode::CAST_RINT,
                        tiling.headSize);
                }
                PipeBarrier<PIPE_V>();
            }

            inputQueue.FreeTensor(inputLocal);
            outputQueue.EnQue(outputLocal);
            outputLocal = outputQueue.DeQue<bfloat16_t>();

            __gm__ bfloat16_t* cachePtr =
                cacheList.GetDataPtr<__gm__ bfloat16_t>(layer);
            if (cachePtr == nullptr) {
                return;
            }
            GlobalTensor<bfloat16_t> cacheGm;
            cacheGm.SetGlobalBuffer(cachePtr, cacheElements);
            DataCopy(
                cacheGm[destinationRowBase],
                outputLocal,
                tiling.tileElements);
            outputQueue.FreeTensor(outputLocal);
        }

        ropeQueue.FreeTensor(ropeLocal);
    }
}
