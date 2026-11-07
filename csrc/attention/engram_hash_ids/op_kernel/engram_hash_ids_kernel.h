// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "kernel_operator.h"
#include "engram_hash_ids_constants.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/engram_hash_ids_math.h"
#else
#include "arch22/engram_hash_ids_math.h"
#endif

namespace EngramHashIds {
using namespace AscendC;

// Integer-only hashing preserves exact int64 multiplication, XOR and remainder.
template <typename InputT, typename MapT, typename PositionT, typename LookbackT>
class EngramHashIdsKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR inputIds, GM_ADDR tokenMap, GM_ADDR deadMask, GM_ADDR positions,
        GM_ADDR blockTable, GM_ADDR queryStartLoc, GM_ADDR reqIds, GM_ADDR multipliers,
        GM_ADDR primes, GM_ADDR offsets, GM_ADDR cache, GM_ADDR lookbackTokenIds,
        GM_ADDR lookbackDeadMask, GM_ADDR output, const EngramHashIdsTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        sources_.Init(inputIds, tokenMap, deadMask, positions, blockTable, queryStartLoc, reqIds,
                      cache, lookbackTokenIds, lookbackDeadMask, tiling, pipe);
        multipliers_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(multipliers));
        primes_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(primes));
        offsets_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(offsets));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
        arithmetic_.Init(pipe);
        pipe->InitBuffer(outputBuffer_, TOKEN_TILE_ELEMENTS * HEAD_TILE_ELEMENTS * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileSize = tiling_.tokenTileSize;
        const int64_t tiles = tiling_.numTokens / tileSize + (tiling_.numTokens % tileSize != 0);
        const int64_t tasks = tiles * tiling_.numLayers;
        for (int64_t task = GetBlockIdx(); task < tasks; task += tiling_.usedCoreNum) {
            const int64_t begin = task / tiling_.numLayers * tileSize;
            const uint32_t count = tiling_.numTokens - begin < tileSize ? tiling_.numTokens - begin : tileSize;
            HashTile(begin, count, task % tiling_.numLayers);
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        SetFlag<event>(COPY_EVENT_ID);
        WaitFlag<event>(COPY_EVENT_ID);
    }

    __aicore__ inline void WriteHeads(int64_t begin, uint32_t count, int64_t layer, int64_t shift)
    {
        const auto output = outputBuffer_.Get<int32_t>();
        const int64_t columns = (tiling_.maxNgram - 1) * tiling_.numHeads;
        const int64_t col = (shift - 1) * tiling_.numHeads;
        const int64_t paramBase = layer * columns + col;
        const int64_t outputBase = (begin * tiling_.numLayers + layer) * columns + col;
        const int64_t outputRowElements = tiling_.numLayers * columns;
        for (int64_t headStart = 0; headStart < tiling_.numHeads; headStart += HEAD_TILE_ELEMENTS) {
            const int64_t remaining = tiling_.numHeads - headStart;
            const uint32_t heads = remaining < HEAD_TILE_ELEMENTS ? remaining : HEAD_TILE_ELEMENTS;
            const uint32_t localRowElements =
                (heads + OUTPUT_BLOCK_ELEMENTS - 1) / OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
            for (uint32_t head = 0; head < heads; ++head) {
                const int64_t index = paramBase + headStart + head;
                // Load parameters once per tile, rather than once per token.
                const int64_t prime = primes_.GetValue(index);
                const uint64_t offset = static_cast<uint64_t>(offsets_.GetValue(index));
#if (__CCE_AICORE__ == 310)
                arithmetic_.WriteHead(output, head, prime, offset, count);
#else
                arithmetic_.PrepareHead(prime, offset, count);
                for (uint32_t lane = 0; lane < count; ++lane) {
                    output.SetValue(lane * localRowElements + head, arithmetic_.Output(lane, prime, offset));
                }
#endif
            }
            Sync<HardEvent::S_MTE3>();
            const uint32_t rowBytes = heads * static_cast<uint32_t>(sizeof(int32_t));
            const uint64_t dstStride = (outputRowElements - heads) * static_cast<uint64_t>(sizeof(int32_t));
            if (dstStride <= MAX_DMA_STRIDE_BYTES) {
                const DataCopyExtParams copy{
                    static_cast<uint16_t>(count), rowBytes, 0, static_cast<uint32_t>(dstStride), 0};
                // UB rows are padded to a data block; GM rows keep the exact head count.
                DataCopyPad(output_[outputBase + headStart], output, copy);
            } else {
                const DataCopyExtParams rowCopy{COPY_BLOCK_COUNT, rowBytes, 0, 0, 0};
                for (uint32_t lane = 0; lane < count; ++lane) {
                    DataCopyPad(output_[outputBase + headStart + lane * outputRowElements],
                                output[lane * localRowElements], rowCopy);
                }
            }
            Sync<HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline void HashTile(int64_t begin, uint32_t count, int64_t layer)
    {
        sources_.Prepare(begin, count, arithmetic_);
        for (int64_t shift = 0; shift < tiling_.maxNgram; ++shift) {
            const uint64_t multiplier = static_cast<uint64_t>(multipliers_.GetValue(layer * tiling_.maxNgram + shift));
            sources_.HashShift(arithmetic_, begin, count, shift, multiplier);
            if (shift > 0) {
                WriteHeads(begin, count, layer, shift);
            }
        }
    }

    EngramHashIdsTilingData tiling_;
    GlobalTensor<int64_t> multipliers_;
    GlobalTensor<int64_t> primes_;
    GlobalTensor<int64_t> offsets_;
    GlobalTensor<int32_t> output_;
    HashSources<InputT, MapT, PositionT, LookbackT> sources_;
    HashArithmetic arithmetic_;
    TBuf<TPosition::VECCALC> outputBuffer_;
};
} // namespace EngramHashIds
