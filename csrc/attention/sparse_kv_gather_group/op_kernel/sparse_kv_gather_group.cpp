/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"
#include "sparse_kv_gather_kernel.h"
using namespace AscendC;
using namespace BaseApi;

template <typename TopkT, typename CurPosT>
__aicore__ inline void FindCurrentTopkSlot(
    __gm__ uint8_t *topkIndices,
    __gm__ uint8_t *curPos,
    __gm__ uint8_t *currentTopkSlots,
    const uint32_t numActual,
    const uint32_t topkN,
    TPipe *pipe)
{
    GlobalTensor<TopkT> topkGm;
    GlobalTensor<CurPosT> curPosGm;
    GlobalTensor<int32_t> currentTopkSlotsGm;
    topkGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkT *>(topkIndices));
    curPosGm.SetGlobalBuffer(reinterpret_cast<__gm__ CurPosT *>(curPos));
    currentTopkSlotsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(currentTopkSlots));
    TBuf<> currentSlotBuf;
    constexpr uint32_t slotsPerCacheLine = 8U;
    pipe->InitBuffer(
        currentSlotBuf,
        slotsPerCacheLine * sizeof(int32_t));
    LocalTensor<int32_t> currentSlotUb =
        currentSlotBuf.Get<int32_t>();

    for (uint32_t queryIdx = GetBlockIdx(); queryIdx < numActual;
         queryIdx += GetBlockNum()) {
        const int64_t currentPos =
            static_cast<int64_t>(curPosGm.GetValue(queryIdx));
        int32_t currentSlot = -1;
        const uint64_t rowOffset =
            static_cast<uint64_t>(queryIdx) * topkN;
        for (uint32_t slot = 0; slot < topkN; ++slot) {
            if (static_cast<int64_t>(
                    topkGm.GetValue(rowOffset + slot)) == currentPos) {
                currentSlot = static_cast<int32_t>(slot);
                break;
            }
        }
        currentSlotUb.SetValue(0, currentSlot);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(
            currentTopkSlotsGm[
                static_cast<uint64_t>(queryIdx) * slotsPerCacheLine],
            currentSlotUb,
            slotsPerCacheLine);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}

extern "C" __global__ __aicore__ void sparse_kv_gather_group(
    __gm__ uint8_t *pagedCtkv0, __gm__ uint8_t *pagedKpe0, __gm__ uint8_t *pagedCtkv1, __gm__ uint8_t *pagedKpe1, __gm__ uint8_t *pagedCtkv2, __gm__ uint8_t *pagedKpe2,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndices, __gm__ uint8_t *curPos,
    __gm__ uint8_t *outCtkv0, __gm__ uint8_t *outKpe0, __gm__ uint8_t *outCtkv1, __gm__ uint8_t *outKpe1, __gm__ uint8_t *outCtkv2, __gm__ uint8_t *outKpe2,
    __gm__ uint8_t *currentTopkSlots,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if ASCEND_IS_AIC { return; }
    (void)workspace;
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.topkIndicesType == SKG_INDEX_TYPE_INT64) {
        if (tilingData.curPosType == SKG_INDEX_TYPE_INT64) {
            FindCurrentTopkSlot<int64_t, int64_t>(
                topkIndices, curPos, currentTopkSlots,
                tilingData.numActual, tilingData.topkN, &pipe);
        } else {
            FindCurrentTopkSlot<int64_t, int32_t>(
                topkIndices, curPos, currentTopkSlots,
                tilingData.numActual, tilingData.topkN, &pipe);
        }
    } else if (tilingData.curPosType == SKG_INDEX_TYPE_INT64) {
        FindCurrentTopkSlot<int32_t, int64_t>(
            topkIndices, curPos, currentTopkSlots,
            tilingData.numActual, tilingData.topkN, &pipe);
    } else {
        FindCurrentTopkSlot<int32_t, int32_t>(
            topkIndices, curPos, currentTopkSlots,
            tilingData.numActual, tilingData.topkN, &pipe);
    }

#define RUN_SKG_LAYER(ID) \
    SparseKvGatherKernel op##ID; \
    op##ID.Init(pagedCtkv##ID, pagedKpe##ID, blockTable, topkIndices, curPos, outCtkv##ID, outKpe##ID, tilingData.numBlocks, tilingData.maxBlocks, tilingData.topkN, tilingData.totalSlots, tilingData.slotsPerCore, tilingData.usedCoreNum, tilingData.blockTableType, tilingData.topkIndicesType, tilingData.curPosType, &pipe); \
    op##ID.Process()
    RUN_SKG_LAYER(0);
    if (tilingData.numCacheLayers > 1U) { RUN_SKG_LAYER(1); }
    if (tilingData.numCacheLayers > 2U) { RUN_SKG_LAYER(2); }
#undef RUN_SKG_LAYER
}
