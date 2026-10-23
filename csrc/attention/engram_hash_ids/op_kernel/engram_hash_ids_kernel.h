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
        inputIds_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(inputIds));
        tokenMap_.SetGlobalBuffer(reinterpret_cast<__gm__ MapT *>(tokenMap));
        deadMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(deadMask));
        positions_.SetGlobalBuffer(reinterpret_cast<__gm__ PositionT *>(positions));
        queryStartLoc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
        reqIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(reqIds));
        multipliers_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(multipliers));
        primes_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(primes));
        offsets_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(offsets));
        lookbackTokenIds_.SetGlobalBuffer(reinterpret_cast<__gm__ LookbackT *>(lookbackTokenIds));
        lookbackDeadMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(lookbackDeadMask));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
        if (tiling_.hasCache != 0) {
            blockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
            cache_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cache));
        }
        pipe->InitBuffer(requestBuffer_, TOKEN_TILE_ELEMENTS * sizeof(int32_t));
        pipe->InitBuffer(positionBuffer_, TOKEN_TILE_ELEMENTS * sizeof(PositionT));
        pipe->InitBuffer(chunkStartBuffer_, TOKEN_TILE_ELEMENTS * sizeof(int64_t));
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
            CacheMetadata(begin, count);
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

    __aicore__ inline int64_t WrapAdd(int64_t left, int64_t right) const
    {
        return static_cast<int64_t>(static_cast<uint64_t>(left) + static_cast<uint64_t>(right));
    }

    __aicore__ inline int64_t WrapMul(int64_t left, int64_t right) const
    {
        return static_cast<int64_t>(static_cast<uint64_t>(left) * static_cast<uint64_t>(right));
    }

    __aicore__ inline int64_t WrapSub(int64_t left, int64_t right) const
    {
        return static_cast<int64_t>(static_cast<uint64_t>(left) - static_cast<uint64_t>(right));
    }

    __aicore__ inline int64_t Clamp(int64_t value, int64_t low, int64_t high) const
    {
        return value < low ? low : (value > high ? high : value);
    }

    __aicore__ inline void CacheMetadata(int64_t begin, uint32_t count)
    {
        Sync<HardEvent::S_MTE2>();
        const auto requests = requestBuffer_.Get<int32_t>();
        const auto positions = positionBuffer_.Get<PositionT>();
        const auto starts = chunkStartBuffer_.Get<int64_t>();
        const DataCopyExtParams requestCopy{
            COPY_BLOCK_COUNT, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(requests, reqIds_[begin], requestCopy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        if (tiling_.positionStride == 1) {
            const DataCopyExtParams positionCopy{
                COPY_BLOCK_COUNT, count * static_cast<uint32_t>(sizeof(PositionT)), 0, 0, 0};
            DataCopyPad(positions, positions_[begin], positionCopy, DataCopyPadExtParams<PositionT>{false, 0, 0, 0});
        }
        Sync<HardEvent::MTE2_S>();
        int64_t previousRequest = UNKNOWN_TOKEN_ID;
        int64_t chunkStart = 0;
        for (uint32_t lane = 0; lane < count; ++lane) {
            if (tiling_.positionStride != 1) {
                positions.SetValue(lane, positions_.GetValue(WrapMul(begin + lane, tiling_.positionStride)));
            }
            const int64_t request = requests.GetValue(lane);
            if (request != previousRequest) {
                const int64_t index = queryStartLoc_.GetValue(WrapMul(request, tiling_.queryStride));
                const int64_t chunkIndex = index < tiling_.numTokens ? index : tiling_.numTokens - 1;
                chunkStart = positions_.GetValue(WrapMul(chunkIndex, tiling_.positionStride));
                previousRequest = request;
            }
            starts.SetValue(lane, chunkStart);
        }
    }

    __aicore__ inline int64_t ReadSource(
        int64_t token, int64_t request, int64_t chunkStart, int64_t lookback, int64_t shift)
    {
        if (lookback >= chunkStart) {
            const int64_t batchIndex = token > shift ? token - shift : 0;
            const int64_t tokenId = inputIds_.GetValue(WrapMul(batchIndex, tiling_.inputStride));
            const int64_t source = tokenMap_.GetValue(tokenId);
            return deadMask_.GetValue(WrapMul(batchIndex, tiling_.maskStride)) != 0 ? tiling_.deadId : source;
        }

        // Triton subtracts one in the positions dtype before promotion to int64.
        const PositionT previous = static_cast<PositionT>(static_cast<uint64_t>(chunkStart) - 1);
        const int64_t col = WrapSub(static_cast<int64_t>(previous), lookback);
        if (col >= 0 && col < tiling_.lookbackDepth) {
            const int64_t tokenId = lookbackTokenIds_.GetValue(
                WrapAdd(WrapMul(request, tiling_.lookbackRowStride), WrapMul(col, tiling_.lookbackColStride)));
            if (tokenId > UNKNOWN_TOKEN_ID) {
                const int64_t source = tokenMap_.GetValue(tokenId);
                const bool dead = lookbackDeadMask_.GetValue(
                    WrapAdd(WrapMul(request, tiling_.lookbackMaskRowStride),
                            WrapMul(col, tiling_.lookbackMaskColStride))) != 0;
                return dead ? tiling_.deadId : source;
            }
        }
        if (tiling_.hasCache == 0) {
            // Only the no-cache fallback is int32 in Triton; blocked padding stays int64.
            return static_cast<int64_t>(static_cast<int32_t>(tiling_.padId));
        }

        // Cache entries already contain mapped token IDs (including dead_id).
        const uint64_t extent = static_cast<uint64_t>(tiling_.maxBlocks) * tiling_.cacheBlockSize;
        const int64_t position = Clamp(lookback, 0, static_cast<int64_t>(extent - 1));
        const int64_t blockRow = request < tiling_.numTableRows ? request : tiling_.numTableRows - 1;
        const int64_t block = blockTable_.GetValue(
            WrapAdd(WrapMul(blockRow, tiling_.tableRowStride),
                    WrapMul(position / tiling_.cacheBlockSize, tiling_.tableColStride)));
        const int64_t cacheIndex = WrapAdd(WrapMul(block, tiling_.cacheBlockSize), position % tiling_.cacheBlockSize);
        const int64_t slot = Clamp(cacheIndex, 0, tiling_.numSlots - 1);
        return cache_.GetValue(slot);
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
        const auto requests = requestBuffer_.Get<int32_t>();
        const auto positions = positionBuffer_.Get<PositionT>();
        const auto starts = chunkStartBuffer_.Get<int64_t>();
        for (uint32_t lane = 0; lane < count; ++lane) {
            blocked_[lane] = false;
            arithmetic_.ResetLane(lane);
        }
        for (int64_t shift = 0; shift < tiling_.maxNgram; ++shift) {
            const uint64_t multiplier = static_cast<uint64_t>(multipliers_.GetValue(layer * tiling_.maxNgram + shift));
            for (uint32_t lane = 0; lane < count; ++lane) {
                const int64_t lookback = WrapSub(static_cast<int64_t>(positions.GetValue(lane)), shift);
                const int64_t source = ReadSource(begin + lane, requests.GetValue(lane), starts.GetValue(lane),
                                                  lookback, shift);
                blocked_[lane] = blocked_[lane] || lookback < 0 || source == tiling_.deadId;
                const int64_t value = blocked_[lane] ? tiling_.padId : source;
                arithmetic_.Accumulate(lane, value, multiplier);
            }
            arithmetic_.FinishShift(multiplier, count);
            if (shift > 0) {
                WriteHeads(begin, count, layer, shift);
            }
        }
    }

    EngramHashIdsTilingData tiling_;
    GlobalTensor<InputT> inputIds_;
    GlobalTensor<MapT> tokenMap_;
    GlobalTensor<uint8_t> deadMask_;
    GlobalTensor<PositionT> positions_;
    GlobalTensor<int32_t> blockTable_;
    GlobalTensor<int32_t> queryStartLoc_;
    GlobalTensor<int32_t> reqIds_;
    GlobalTensor<int64_t> multipliers_;
    GlobalTensor<int64_t> primes_;
    GlobalTensor<int64_t> offsets_;
    GlobalTensor<int32_t> cache_;
    GlobalTensor<LookbackT> lookbackTokenIds_;
    GlobalTensor<uint8_t> lookbackDeadMask_;
    GlobalTensor<int32_t> output_;
    TBuf<TPosition::VECCALC> requestBuffer_;
    TBuf<TPosition::VECCALC> positionBuffer_;
    TBuf<TPosition::VECCALC> chunkStartBuffer_;
    HashArithmetic arithmetic_;
    TBuf<TPosition::VECCALC> outputBuffer_;
    bool blocked_[TOKEN_TILE_ELEMENTS];
};
} // namespace EngramHashIds
