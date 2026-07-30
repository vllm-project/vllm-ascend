/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t SKP_CTKV_DIM = 512;
constexpr uint32_t SKP_KPE_DIM = 64;
constexpr uint32_t SKP_COMBINED_DIM = SKP_CTKV_DIM + SKP_KPE_DIM;
constexpr uint32_t SKP_INDEX_TYPE_INT32 = 0;
constexpr uint32_t SKP_INDEX_TYPE_INT64 = 1;
}  // namespace

extern "C" __global__ __aicore__ void sparse_kv_patch(
    __gm__ uint8_t *pagedCtkv,
    __gm__ uint8_t *pagedKpe,
    __gm__ uint8_t *slotMapping,
    __gm__ uint8_t *currentTopkSlots,
    __gm__ uint8_t *prefetchedCtkv,
    __gm__ uint8_t *prefetchedKpe,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    (void)workspace;

    GET_TILING_DATA(tilingData, tiling);

    GlobalTensor<int32_t> currentTopkSlotsGm;
    currentTopkSlotsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(currentTopkSlots));

    GlobalTensor<uint16_t> pagedCtkvGm;
    GlobalTensor<uint16_t> pagedKpeGm;
    GlobalTensor<uint16_t> prefetchedCtkvGm;
    GlobalTensor<uint16_t> prefetchedKpeGm;
    pagedCtkvGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint16_t *>(pagedCtkv));
    pagedKpeGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint16_t *>(pagedKpe));
    prefetchedCtkvGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint16_t *>(prefetchedCtkv));
    prefetchedKpeGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint16_t *>(prefetchedKpe));

    TPipe pipe;
    TBuf<> stage;
    pipe.InitBuffer(stage, SKP_COMBINED_DIM * sizeof(uint16_t));
    LocalTensor<uint16_t> stageUb = stage.Get<uint16_t>();
    LocalTensor<uint16_t> ctkvUb = stageUb;
    LocalTensor<uint16_t> kpeUb = stageUb[SKP_CTKV_DIM];
    TEventID loadDone = pipe.AllocEventID<HardEvent::MTE2_MTE3>();
    TEventID storeDone = pipe.AllocEventID<HardEvent::MTE3_MTE2>();

    for (uint32_t queryIdx = GetBlockIdx();
        queryIdx < tilingData.numActual; queryIdx += GetBlockNum()) {
        const int32_t topkSlot =
            currentTopkSlotsGm.GetValue(
                static_cast<uint64_t>(queryIdx) * 8U);
        if (topkSlot < 0 ||
            static_cast<uint32_t>(topkSlot) >= tilingData.topkN) {
            continue;
        }

        int64_t physicalSlot = -1;
        if (tilingData.slotMappingType == SKP_INDEX_TYPE_INT64) {
            GlobalTensor<int64_t> slotMappingGm;
            slotMappingGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t *>(slotMapping));
            physicalSlot = slotMappingGm.GetValue(queryIdx);
        } else {
            GlobalTensor<int32_t> slotMappingGm;
            slotMappingGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t *>(slotMapping));
            physicalSlot = static_cast<int64_t>(
                slotMappingGm.GetValue(queryIdx));
        }
        if (physicalSlot < 0 ||
            static_cast<uint64_t>(physicalSlot) >=
                tilingData.numPhysicalSlots) {
            continue;
        }

        DataCopy(ctkvUb, pagedCtkvGm[physicalSlot * SKP_CTKV_DIM],
                 SKP_CTKV_DIM);
        DataCopy(kpeUb, pagedKpeGm[physicalSlot * SKP_KPE_DIM],
                 SKP_KPE_DIM);
        SetFlag<HardEvent::MTE2_MTE3>(loadDone);
        WaitFlag<HardEvent::MTE2_MTE3>(loadDone);

        const uint64_t outputSlot =
            static_cast<uint64_t>(queryIdx) * tilingData.topkN +
            static_cast<uint32_t>(topkSlot);
        DataCopy(prefetchedCtkvGm[outputSlot * SKP_CTKV_DIM],
                 ctkvUb, SKP_CTKV_DIM);
        DataCopy(prefetchedKpeGm[outputSlot * SKP_KPE_DIM],
                 kpeUb, SKP_KPE_DIM);
        SetFlag<HardEvent::MTE3_MTE2>(storeDone);
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
    }
}
