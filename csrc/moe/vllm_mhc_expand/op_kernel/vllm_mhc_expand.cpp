// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

// Copy raw 16-bit values: FP16/BF16 NaNs, subnormals and signed zero stay intact.
class KernelMhcExpand {
    using T = uint16_t;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const VllmMhcExpandTilingData& params)
    {
        p_ = params;
        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);
    }
    __aicore__ inline void Process()
    {
        if (p_.tilesPerRow == 1 && p_.hidden % 64 == 0 &&
            p_.hidden * p_.mhcMult <= 2040 && p_.tileLength * 2 >= (p_.mhcMult + 1) * p_.hidden) {
            Bulk();
        } else {
            Forward();
        }
    }
private:
    __aicore__ inline void Bulk() {
        const uint32_t d = static_cast<uint32_t>(p_.hidden);
        const uint32_t m = static_cast<uint32_t>(p_.mhcMult);
        const uint32_t capacity = p_.tileLength * 2 / ((m + 1) * d);
        const uint32_t batch = capacity < 16 ? capacity : 16;
        LocalTensor<T> in(TPosition::VECIN, 0, batch * m * d);
        const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (uint64_t token = static_cast<uint64_t>(GetBlockIdx()) * batch;
             token < p_.tokens; token += static_cast<uint64_t>(GetBlockNum()) * batch) {
            const uint32_t rows = p_.tokens - token < batch ? static_cast<uint32_t>(p_.tokens - token) : batch;
            const DataCopyExtParams load{1, rows * d * 2, 0, 0, 0};
            DataCopyPad(in, input_[token * d], load, pad);
            Sync<HardEvent::MTE2_MTE3>();
            // DataCopyPad supports UB -> GM; a GM destination stride is in bytes.
            // The local source stride uses 32-byte blocks and is zero here.
            const DataCopyExtParams store{static_cast<uint16_t>(rows), d * 2, 0, (m - 1) * d * 2, 0};
            for (uint32_t stream = 0; stream < m; ++stream)
                DataCopyPad(output_[(token * m + stream) * d], in, store);
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void Forward() {
        if (p_.tilesPerRow == 1 && p_.hidden % 16 == 0 &&
            (p_.mhcMult - 1) * p_.hidden * sizeof(T) <= UINT32_MAX) {
            ForwardRows();
            return;
        }
        LocalTensor<T> in(TPosition::VECIN, 0, p_.tileLength * 2);
        for (uint64_t tile = GetBlockIdx(); tile < p_.totalTiles; tile += GetBlockNum()) {
            const uint64_t token = tile / p_.tilesPerRow;
            const uint64_t column = (tile % p_.tilesPerRow) * p_.tileLength;
            const uint32_t count = p_.hidden - column < p_.tileLength
                ? static_cast<uint32_t>(p_.hidden - column) : p_.tileLength;
            const uint64_t flat = token * p_.hidden + column;
            const DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(in, input_[flat], copy, pad);
            Sync<HardEvent::MTE2_MTE3>();
            // UB -> GM DataCopyPad discards padding instead of overwriting a tail.
            for (uint64_t stream = 0; stream < p_.mhcMult; ++stream) {
                const auto offset = (token * p_.mhcMult + stream) * p_.hidden + column;
                DataCopyPad(output_[offset], in, copy);
            }
            // Every output copy reads the immutable tile; wait before reusing it.
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void ForwardRows() {
        LocalTensor<T> in(TPosition::VECIN, 0, p_.tileLength * 2);
        // Use every launched core for small batches; pair rows only when there
        // are enough tokens to keep the cores busy.
        const uint32_t rowsPerGroup = p_.tokens <= GetBlockNum() ? 1 : 2;
        const uint64_t groups = (p_.tokens + rowsPerGroup - 1) / rowsPerGroup;
        for (uint64_t group = GetBlockIdx(); group < groups; group += GetBlockNum()) {
            const uint64_t token = group * rowsPerGroup;
            const uint32_t rows = p_.tokens - token >= rowsPerGroup
                ? rowsPerGroup : static_cast<uint32_t>(p_.tokens - token);
            const uint32_t rowBytes = static_cast<uint32_t>(p_.hidden * sizeof(T));
            const DataCopyExtParams load{1, rows * rowBytes, 0, 0, 0};
            const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(in, input_[token * p_.hidden], load, pad);
            Sync<HardEvent::MTE2_MTE3>();
            // dstStride is measured in bytes for GM, not in 32-byte UB blocks.
            const DataCopyExtParams store{static_cast<uint16_t>(rows), rowBytes, 0,
                static_cast<uint32_t>((p_.mhcMult - 1) * rowBytes), 0};
            for (uint64_t stream = 0; stream < p_.mhcMult; ++stream) {
                DataCopyPad(output_[(token * p_.mhcMult + stream) * p_.hidden], in, store);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        constexpr event_t id = EVENT_ID0;
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    GlobalTensor<T> input_, output_;
    VllmMhcExpandTilingData p_;
};

extern "C" __global__ __aicore__ void vllm_mhc_expand(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(params, tiling);
    if (TILING_KEY_IS(1)) {
        KernelMhcExpand op;
        op.Init(x, y, params);
        op.Process();
    }
}
