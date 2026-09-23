// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t NOPE = 128;
constexpr uint32_t ROPE = 64;
constexpr uint32_t KEY = NOPE + ROPE;
constexpr uint32_t MAX_STRIDED_DMA_ROWS = 32;

class FlashMlaBf16PrepareKernel {
public:
    __aicore__ inline void Init(GM_ADDR keyNope, GM_ADDR value, GM_ADDR keyRope,
        GM_ADDR key, GM_ADDR packedValue, const FlashMlaBf16PrepareTilingData &data, TPipe *pipe)
    {
        data_ = data;
        keyNope_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(keyNope));
        value_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(value));
        keyRope_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(keyRope));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
        packedValue_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(packedValue));
        const uint32_t rows = data_.tileTokens * data_.heads;
        keyElements_ = rows * KEY;
        valueElements_ = rows * NOPE;
        ropeElements_ = data_.tileTokens * data_.ropeHeads * ROPE;
        // At most 128 rows: double-buffered K, V and per-head RoPE use <=192KiB.
        pipe->InitBuffer(keyBuffer_, 2 * keyElements_ * sizeof(bfloat16_t));
        pipe->InitBuffer(valueBuffer_, 2 * valueElements_ * sizeof(bfloat16_t));
        pipe->InitBuffer(ropeBuffer_, 2 * ropeElements_ * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        const int64_t tiles = (data_.tokens + data_.tileTokens - 1) / data_.tileTokens;
        int64_t tile = GetBlockIdx();
        if (tile >= tiles) return;
        uint32_t iteration = 0;
        CopyIn(tile, 0);
        for (; tile < tiles; tile += data_.usedCores, ++iteration) {
            const uint32_t slot = iteration % 2;
            const int64_t next = tile + data_.usedCores;
            if (next < tiles) {
                const uint32_t nextSlot = 1 - slot;
                if (iteration >= 1) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(nextSlot));
                // Prefetch the next contiguous token tile before packing this
                // one. Two slots overlap MTE2, vector broadcast and MTE3.
                CopyIn(next, nextSlot);
            }
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(slot));
            PackRope(tile, slot);
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(slot));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(slot));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(slot));
            const int64_t token = tile * data_.tileTokens;
            const uint32_t rows = TileTokens(tile) * data_.heads;
            DataCopy(key_[token * data_.heads * KEY], keyBuffer_.Get<bfloat16_t>()[slot * keyElements_], rows * KEY);
            DataCopy(packedValue_[token * data_.heads * NOPE],
                valueBuffer_.Get<bfloat16_t>()[slot * valueElements_], rows * NOPE);
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(slot));
        }
        // Only the final two writes remain outstanding; previous slots were
        // consumed before prefetch reused their storage.
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>((iteration - 1) % 2));
        if (iteration > 1) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(iteration % 2));
    }

private:
    __aicore__ inline uint32_t TileTokens(int64_t tile)
    {
        const int64_t remaining = data_.tokens - tile * data_.tileTokens;
        return static_cast<uint32_t>(remaining < data_.tileTokens ? remaining : data_.tileTokens);
    }

    __aicore__ inline void LoadRows(LocalTensor<bfloat16_t> destination,
        const GlobalTensor<bfloat16_t> &source, int64_t token, uint32_t tokens,
        int64_t heads, uint32_t width, int64_t tokenStride, int64_t headStride, uint32_t ubWidth)
    {
        const uint32_t destinationGap = (ubWidth - width) * sizeof(bfloat16_t) / 32;
        if (heads == 1 || tokenStride == heads * headStride) {
            const int64_t rowStride = heads == 1 ? tokenStride : headStride;
            const uint32_t rows = tokens * heads;
            for (uint32_t first = 0; first < rows; first += MAX_STRIDED_DMA_ROWS) {
                const uint32_t count = rows - first < MAX_STRIDED_DMA_ROWS ? rows - first : MAX_STRIDED_DMA_ROWS;
                DataCopyExtParams params{static_cast<uint16_t>(count),
                    static_cast<uint32_t>(width * sizeof(bfloat16_t)),
                    static_cast<uint32_t>((rowStride - width) * sizeof(bfloat16_t)), destinationGap, 0};
                DataCopyPad(destination[first * ubWidth], source[token * tokenStride + first * rowStride],
                    params, DataCopyPadExtParams<bfloat16_t>{});
            }
        } else {
            // Preserve non-contiguous token views without a producer-side copy.
            for (uint32_t t = 0; t < tokens; ++t) {
                for (uint32_t first = 0; first < heads; first += MAX_STRIDED_DMA_ROWS) {
                    const uint32_t count = heads - first < MAX_STRIDED_DMA_ROWS ?
                        heads - first : MAX_STRIDED_DMA_ROWS;
                    DataCopyExtParams params{static_cast<uint16_t>(count),
                        static_cast<uint32_t>(width * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((headStride - width) * sizeof(bfloat16_t)), destinationGap, 0};
                    DataCopyPad(destination[(t * heads + first) * ubWidth],
                        source[(token + t) * tokenStride + first * headStride],
                        params, DataCopyPadExtParams<bfloat16_t>{});
                }
            }
        }
    }

    __aicore__ inline void CopyIn(int64_t tile, uint32_t slot)
    {
        const int64_t token = tile * data_.tileTokens;
        const uint32_t tokens = TileTokens(tile);
        // NoPE goes directly to its final 192-wide UB row, leaving the RoPE
        // field for the vector broadcast. V is already densely packed in UB.
        LoadRows(keyBuffer_.Get<bfloat16_t>()[slot * keyElements_], keyNope_, token, tokens,
            data_.heads, NOPE, data_.keyTokenStride, data_.keyHeadStride, KEY);
        LoadRows(valueBuffer_.Get<bfloat16_t>()[slot * valueElements_], value_, token, tokens,
            data_.heads, NOPE, data_.valueTokenStride, data_.valueHeadStride, NOPE);
        LoadRows(ropeBuffer_.Get<bfloat16_t>()[slot * ropeElements_], keyRope_, token, tokens,
            data_.ropeHeads, ROPE, data_.ropeTokenStride, data_.ropeHeadStride, ROPE);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(slot));
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(slot));
    }

    __aicore__ inline void PackRope(int64_t tile, uint32_t slot)
    {
        auto key = keyBuffer_.Get<bfloat16_t>()[slot * keyElements_];
        auto rope = ropeBuffer_.Get<bfloat16_t>()[slot * ropeElements_];
        const uint32_t tokens = TileTokens(tile);
        if (data_.ropeHeads == 1) {
            for (uint32_t t = 0; t < tokens; ++t) {
                Copy(key[t * data_.heads * KEY + NOPE], rope[t * ROPE], ROPE,
                    static_cast<uint8_t>(data_.heads), {1, 1, 12, 0});
            }
        } else {
            Copy(key[NOPE], rope, ROPE, static_cast<uint8_t>(tokens * data_.heads), {1, 1, 12, 4});
        }
    }

    FlashMlaBf16PrepareTilingData data_;
    GlobalTensor<bfloat16_t> keyNope_, value_, keyRope_, key_, packedValue_;
    TBuf<TPosition::VECCALC> keyBuffer_, valueBuffer_, ropeBuffer_;
    uint32_t keyElements_, valueElements_, ropeElements_;
};
} // namespace

extern "C" __global__ __aicore__ void flash_mla_bf16_prepare(GM_ADDR keyNope, GM_ADDR value,
    GM_ADDR keyRope, GM_ADDR key, GM_ADDR packedValue, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    FlashMlaBf16PrepareKernel kernel;
    kernel.Init(keyNope, value, keyRope, key, packedValue, data, &pipe);
    kernel.Process();
}
