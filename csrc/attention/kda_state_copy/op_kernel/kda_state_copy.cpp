// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "kda_state_copy_tiling_data.h"
namespace KdaStateCopy {
using namespace AscendC;
template <HardEvent Event> __aicore__ inline void Sync()
{
    SetFlag<Event>(0);
    WaitFlag<Event>(0);
}
template <typename Index> class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR indices, GM_ADDR hasInitialState,
        GM_ADDR destination, const optiling::KdaStateCopyTilingData &tiling)
    {
        data = tiling;
        sourceGm.SetGlobalBuffer((__gm__ uint8_t *)source);
        destinationGm.SetGlobalBuffer((__gm__ uint8_t *)destination);
        indicesGm.SetGlobalBuffer((__gm__ Index *)indices);
        if (data.hasInitialState) flagsGm.SetGlobalBuffer((__gm__ uint8_t *)hasInitialState);
        pipe.InitBuffer(buffer, optiling::KDA_STATE_COPY_TILE_BYTES);
    }
    __aicore__ inline void Process()
    {
        constexpr int64_t tileBytes = optiling::KDA_STATE_COPY_TILE_BYTES;
        const int64_t tilesPerState = (data.payloadBytes + tileBytes - 1) / tileBytes;
        auto bytes = buffer.Get<uint8_t>();
        auto words = buffer.Get<uint32_t>();
        for (int64_t task = GetBlockIdx(); task < data.selectedRows * tilesPerState; task += GetBlockNum()) {
            const int64_t row = task / tilesPerState;
            const int64_t offset = (task % tilesPerState) * tileBytes;
            // Widen the index before multiplying by a potentially >4 GiB
            // page stride, and never form/read an invalid cache address.
            const int64_t slot = static_cast<int64_t>(indicesGm.GetValue(row));
            const bool valid = slot >= 0 && slot < data.cacheRows;
            if (data.toCache && !valid) continue;
            const int64_t remaining = data.payloadBytes - offset;
            const uint32_t length = static_cast<uint32_t>(remaining < tileBytes ? remaining : tileBytes);
            bool readSource = valid;
            if (!data.toCache && data.hasInitialState && flagsGm.GetValue(row) == 0) readSource = false;
            if (readSource) {
                const int64_t sourceOffset = (data.toCache ? row * data.payloadBytes : slot * data.cacheStrideBytes) + offset;
                Sync<HardEvent::MTE3_MTE2>();
                DataCopyPad(bytes, sourceGm[sourceOffset], DataCopyExtParams{1, length, 0, 0, 0},
                    DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                Sync<HardEvent::MTE2_MTE3>();
            } else {
                Sync<HardEvent::MTE3_V>();
                Duplicate(words, static_cast<uint32_t>(0), tileBytes / sizeof(uint32_t));
                Sync<HardEvent::V_MTE3>();
            }
            const int64_t destinationOffset =
                (data.toCache ? slot * data.cacheStrideBytes : row * data.payloadBytes) + offset;
            DataCopyPad(destinationGm[destinationOffset], bytes, DataCopyExtParams{1, length, 0, 0, 0});
        }
    }
private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    GlobalTensor<uint8_t> sourceGm, destinationGm, flagsGm;
    GlobalTensor<Index> indicesGm;
    optiling::KdaStateCopyTilingData data;
};
}
extern "C" __global__ __aicore__ void kda_state_copy(GM_ADDR source, GM_ADDR indices, GM_ADDR hasInitialState,
    GM_ADDR destination, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::KdaStateCopyTilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        KdaStateCopy::Kernel<int32_t> kernel;
        kernel.Init(source, indices, hasInitialState, destination, data);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        KdaStateCopy::Kernel<int64_t> kernel;
        kernel.Init(source, indices, hasInitialState, destination, data);
        kernel.Process();
    }
}
