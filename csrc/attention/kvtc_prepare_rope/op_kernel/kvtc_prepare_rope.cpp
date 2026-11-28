#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

#include "kvtc_prepare_rope_tiling.h"

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
void kvtc_prepare_rope(
    GM_ADDR caches,
    GM_ADDR blockIds,
    GM_ADDR tokenStarts,
    GM_ADDR invCosSinCache,
    GM_ADDR prepared,
    GM_ADDR workspace,
    GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(KvtcPrepareRopeTilingData);
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
    const __gm__ int64_t* blockIdsPtr =
        reinterpret_cast<const __gm__ int64_t*>(blockIds);
    const __gm__ int64_t* tokenStartsPtr =
        reinterpret_cast<const __gm__ int64_t*>(tokenStarts);

    const uint64_t cacheBlockElements =
        static_cast<uint64_t>(tiling.blockSize) * tiling.layerSize;
    const uint64_t cacheElements =
        static_cast<uint64_t>(tiling.numGpuBlocks) * cacheBlockElements;
    const uint64_t outputBlockElements =
        static_cast<uint64_t>(tiling.blockSize) * tiling.k;
    const uint32_t halfRotary = tiling.headSize / 2U;

    GlobalTensor<float> invRopeGm;
    invRopeGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(invCosSinCache),
        static_cast<uint64_t>(tiling.ropeCacheRows) * tiling.headSize);

    GlobalTensor<bfloat16_t> outputGm;
    outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(prepared),
        static_cast<uint64_t>(tiling.selectedBlocks) *
            outputBlockElements);

    TPipe pipe;
    TQue<TPosition::VECIN, DATA_BUFFER_NUM> inputQueue;
    TQue<TPosition::VECOUT, DATA_BUFFER_NUM> outputQueue;
    TQue<TPosition::VECIN, 1> ropeQueue;
    pipe.InitBuffer(
        inputQueue, DATA_BUFFER_NUM, tiling.tileBytes);
    pipe.InitBuffer(
        outputQueue, DATA_BUFFER_NUM, tiling.tileBytes);
    pipe.InitBuffer(
        ropeQueue, 1, tiling.ropeTileBytes);

    TBuf<TPosition::VECCALC> cacheFloatBuf;
    TBuf<TPosition::VECCALC> outputFloatBuf;
    TBuf<TPosition::VECCALC> ropeTmpBuf;
    pipe.InitBuffer(
        cacheFloatBuf, tiling.layerSize * sizeof(float));
    pipe.InitBuffer(
        outputFloatBuf, tiling.layerSize * sizeof(float));
    pipe.InitBuffer(
        ropeTmpBuf,
        tiling.numKvHeads * halfRotary * sizeof(float));

    LocalTensor<float> cacheFloat =
        cacheFloatBuf.Get<float>(tiling.layerSize);
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

    const DataCopyParams copyOutParams{
        static_cast<uint16_t>(tiling.rowsPerTile),
        static_cast<uint16_t>(tiling.blockLen32),
        0U,
        static_cast<uint16_t>(tiling.outputStride32)};

    for (uint32_t localTask = 0U;
         localTask < localTaskCount;
         ++localTask) {
        const uint32_t task = taskBegin + localTask;
        const uint32_t layerGroup = task % tiling.layerGroups;
        const uint32_t rowTask = task / tiling.layerGroups;
        const uint32_t rowTile = rowTask % tiling.rowTilesPerBlock;
        const uint32_t selected = rowTask / tiling.rowTilesPerBlock;

        const int64_t physicalBlock64 = blockIdsPtr[selected];
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

        // One contiguous RoPE load is reused by every layer in this task.
        LocalTensor<float> ropeLocal = ropeQueue.AllocTensor<float>();
        DataCopy(
            ropeLocal,
            invRopeGm[positionStart * tiling.headSize],
            tiling.ropeTileElements);
        ropeQueue.EnQue(ropeLocal);
        ropeLocal = ropeQueue.DeQue<float>();

        const uint64_t sourceRowBase =
            static_cast<uint64_t>(physicalBlock64) * cacheBlockElements +
            static_cast<uint64_t>(rowStart) * tiling.layerSize;
        const uint64_t outputRowBase =
            static_cast<uint64_t>(selected) * outputBlockElements +
            static_cast<uint64_t>(rowStart) * tiling.k;

        // Prime MTE2. In every following iteration the next cache layer is
        // issued before vector compute starts on the current layer, allowing
        // the two engines to overlap through the two VECIN buffers.
        __gm__ bfloat16_t* firstCachePtr =
            cacheList.GetDataPtr<__gm__ bfloat16_t>(layerBegin);
        if (firstCachePtr == nullptr) {
            ropeQueue.FreeTensor(ropeLocal);
            return;
        }
        GlobalTensor<bfloat16_t> firstCacheGm;
        firstCacheGm.SetGlobalBuffer(firstCachePtr, cacheElements);
        LocalTensor<bfloat16_t> firstInputLocal =
            inputQueue.AllocTensor<bfloat16_t>();
        DataCopy(
            firstInputLocal,
            firstCacheGm[sourceRowBase],
            tiling.tileElements);
        inputQueue.EnQue(firstInputLocal);

        for (uint32_t layerOffset = 0U;
             layerOffset < layerCount;
             ++layerOffset) {
            const uint32_t layer = layerBegin + layerOffset;
            LocalTensor<bfloat16_t> inputLocal =
                inputQueue.DeQue<bfloat16_t>();

            if (layerOffset + 1U < layerCount) {
                const uint32_t nextLayer = layer + 1U;
                __gm__ bfloat16_t* nextCachePtr =
                    cacheList.GetDataPtr<__gm__ bfloat16_t>(nextLayer);
                if (nextCachePtr == nullptr) {
                    return;
                }
                GlobalTensor<bfloat16_t> nextCacheGm;
                nextCacheGm.SetGlobalBuffer(nextCachePtr, cacheElements);
                LocalTensor<bfloat16_t> nextInputLocal =
                    inputQueue.AllocTensor<bfloat16_t>();
                DataCopy(
                    nextInputLocal,
                    nextCacheGm[sourceRowBase],
                    tiling.tileElements);
                inputQueue.EnQue(nextInputLocal);
            }

            LocalTensor<bfloat16_t> outputLocal =
                outputQueue.AllocTensor<bfloat16_t>();

            for (uint32_t row = 0U;
                 row < tiling.rowsPerTile;
                 ++row) {
                const uint32_t ropeOffset = row * tiling.headSize;

                // Convert all heads in this row, then process them together
                // with vector repeat strides. src1RepStride=0 broadcasts the
                // same RoPE coefficients to every KV head.
                for (uint32_t head = 0U;
                     head < tiling.numKvHeads;
                     ++head) {
                    const uint32_t dataOffset =
                        row * tiling.layerSize + head * tiling.headSize;

                    Cast(
                        cacheFloat[head * tiling.headSize],
                        inputLocal[dataOffset],
                        RoundMode::CAST_NONE,
                        tiling.headSize);
                }
                PipeBarrier<PIPE_V>();

                // Inverse NeoX RoPE. Input is [a|b], coefficient row is
                // [cos|-sin]. This preserves the formula used by the old
                // working fused compression kernel.
                Mul(
                    outputFloat,
                    cacheFloat,
                    ropeLocal[ropeOffset],
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    mulFullParams);
                Mul(
                    ropeTmp,
                    cacheFloat[halfRotary],
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
                    cacheFloat[halfRotary],
                    ropeLocal[ropeOffset],
                    static_cast<uint64_t>(halfRotary),
                    headRepeat,
                    mulFullParams);
                Mul(
                    ropeTmp,
                    cacheFloat,
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

            const uint64_t outputLayerOffset =
                outputRowBase + static_cast<uint64_t>(layer) *
                    tiling.layerSize;
            DataCopy(
                outputGm[outputLayerOffset],
                outputLocal,
                copyOutParams);
            outputQueue.FreeTensor(outputLocal);
        }

        ropeQueue.FreeTensor(ropeLocal);
    }
}
