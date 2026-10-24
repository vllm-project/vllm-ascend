// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "../engram_hash_ids_constants.h"

namespace EngramHashIds {
using namespace AscendC;

// A5 count-based vector APIs support exact int64/uint64 arithmetic.
class HashArithmetic {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        // Separate rolling halves satisfy BitwiseXor's no-overlap requirement.
        pipe->InitBuffer(rollingBuffer_, ROLLING_BUFFER_COUNT * TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
        pipe->InitBuffer(sourceBuffer_, TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
        pipe->InitBuffer(scratchBuffer_, TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
        pipe->InitBuffer(divisorBuffer_, TOKEN_TILE_ELEMENTS * sizeof(int64_t));
        pipe->InitBuffer(remainderBuffer_, TOKEN_TILE_ELEMENTS * sizeof(int64_t));
        pipe->InitBuffer(outputIndexBuffer_, INDEX_BUFFER_COUNT * TOKEN_TILE_ELEMENTS * sizeof(uint32_t));
        static_assert(HEAD_TILE_ELEMENTS <= OUTPUT_BLOCK_ELEMENTS, "A5 output rows must fit one data block");
        const auto indices = outputIndexBuffer_.Get<uint32_t>();
        for (uint32_t lane = 0; lane < TOKEN_TILE_ELEMENTS; ++lane) {
            indices.SetValue(lane, lane * sizeof(uint64_t));
            indices.SetValue(TOKEN_TILE_ELEMENTS + lane, lane * DATA_BLOCK_BYTES);
        }
    }

    __aicore__ inline void Reset(uint32_t count)
    {
        Sync<HardEvent::S_V>();
        Duplicate(rollingBuffer_.Get<uint64_t>()[rollingOffset_], uint64_t(0), count);
    }

    __aicore__ inline void FinishShift(uint64_t multiplier, uint32_t count, const LocalTensor<uint64_t> &source)
    {
        const auto rolling = rollingBuffer_.Get<uint64_t>();
        const auto product = scratchBuffer_.Get<uint64_t>();
        const uint32_t next = TOKEN_TILE_ELEMENTS - rollingOffset_;
        // SIMT writes history on the Vector pipeline; no scalar UB handoff is needed.
        PipeBarrier<PIPE_V>();
        Muls(product, source, multiplier, count);
        PipeBarrier<PIPE_V>();
        BitwiseXor(rolling[next], rolling[rollingOffset_], product, count);
        Sync<HardEvent::V_S>();
        rollingOffset_ = next;
    }

    __aicore__ inline void PrepareHead(int64_t prime, uint64_t offset, uint32_t count)
    {
        Sync<HardEvent::S_V>();
        if (prime == -1) {
            // Avoid INT64_MIN / -1; every remainder is zero for this divisor.
            Duplicate(sourceBuffer_.Get<uint64_t>(), offset, count);
        } else {
            const auto rolling = rollingBuffer_.Get<int64_t>()[rollingOffset_];
            const auto divisor = divisorBuffer_.Get<int64_t>();
            const auto quotient = sourceBuffer_.Get<int64_t>();
            const auto product = scratchBuffer_.Get<int64_t>();
            const auto remainder = remainderBuffer_.Get<int64_t>();
            Duplicate(divisor, prime, count);
            PipeBarrier<PIPE_V>();
            Div(quotient, rolling, divisor, count);
            PipeBarrier<PIPE_V>();
            Mul(product, quotient, divisor, count);
            PipeBarrier<PIPE_V>();
            Sub(remainder, rolling, product, count);
            PipeBarrier<PIPE_V>();
            // Unsigned addition wraps before the final low-32-bit store.
            Adds(sourceBuffer_.Get<uint64_t>(), remainderBuffer_.Get<uint64_t>(), offset, count);
        }
    }

    __aicore__ inline void WriteHead(const LocalTensor<int32_t>& output, uint32_t head,
                                   int64_t prime, uint64_t offset, uint32_t count)
    {
        PrepareHead(prime, offset, count);
        PipeBarrier<PIPE_V>();
        const auto indices = outputIndexBuffer_.Get<uint32_t>();
        const auto packed = scratchBuffer_.Get<int32_t>();
        // Read the low word as bits: numeric Cast may saturate instead of truncating.
        Gather(packed, sourceBuffer_.Get<int32_t>(), indices, uint32_t(0), count);
        PipeBarrier<PIPE_V>();
        Scatter(output, packed, indices[TOKEN_TILE_ELEMENTS], head * uint32_t(sizeof(int32_t)), count);
        // Keep the existing scalar-to-DMA handoff in the common output path.
        Sync<HardEvent::V_S>();
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        SetFlag<event>(COPY_EVENT_ID);
        WaitFlag<event>(COPY_EVENT_ID);
    }

    static constexpr uint32_t ROLLING_BUFFER_COUNT = 2;
    static constexpr uint32_t INDEX_BUFFER_COUNT = 2;
    uint32_t rollingOffset_ = 0;
    TBuf<TPosition::VECCALC> rollingBuffer_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> scratchBuffer_;
    TBuf<TPosition::VECCALC> divisorBuffer_;
    TBuf<TPosition::VECCALC> remainderBuffer_;
    TBuf<TPosition::VECCALC> outputIndexBuffer_;
};

namespace SourceSimt {
// Bound UB use independently of maxNgram; common n-grams need one VF per task.
constexpr uint32_t HISTORY_SHIFT_TILE_ELEMENTS = 8;
constexpr uint32_t THREAD_COUNT = TOKEN_TILE_ELEMENTS;

template <typename InputT, typename MapT, typename PositionT, typename LookbackT>
struct Inputs {
    __gm__ InputT *inputIds;
    __gm__ MapT *tokenMap;
    __gm__ uint8_t *deadMask;
    __gm__ PositionT *positions;
    __gm__ int32_t *blockTable;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *reqIds;
    __gm__ int32_t *cache;
    __gm__ LookbackT *lookbackTokenIds;
    __gm__ uint8_t *lookbackDeadMask;
};

__simt_callee__ inline int64_t WrapAdd(int64_t left, int64_t right)
{
    return static_cast<int64_t>(static_cast<uint64_t>(left) + static_cast<uint64_t>(right));
}

__simt_callee__ inline int64_t WrapMul(int64_t left, int64_t right)
{
    return static_cast<int64_t>(static_cast<uint64_t>(left) * static_cast<uint64_t>(right));
}

__simt_callee__ inline int64_t WrapSub(int64_t left, int64_t right)
{
    return static_cast<int64_t>(static_cast<uint64_t>(left) - static_cast<uint64_t>(right));
}

__simt_callee__ inline int64_t Clamp(int64_t value, int64_t low, int64_t high)
{
    return value < low ? low : (value > high ? high : value);
}

template <typename InputT, typename MapT, typename PositionT, typename LookbackT>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_COUNT) inline void PrepareHistory(
    Inputs<InputT, MapT, PositionT, LookbackT> inputs, EngramHashIdsTilingData tiling,
    __ubuf__ uint64_t *history, __ubuf__ uint32_t *blockedState,
    int64_t begin, uint32_t count, int64_t firstShift, uint32_t shifts)
{
    const uint32_t lane = Simt::GetThreadIdx();
    if (lane >= count) return;

    const int64_t token = begin + lane;
    const int64_t request = inputs.reqIds[token];
    const int64_t position = inputs.positions[WrapMul(token, tiling.positionStride)];
    const int64_t index = inputs.queryStartLoc[WrapMul(request, tiling.queryStride)];
    const int64_t chunkIndex = index < tiling.numTokens ? index : tiling.numTokens - 1;
    const int64_t chunkStart = inputs.positions[WrapMul(chunkIndex, tiling.positionStride)];
    bool blocked = firstShift != 0 && blockedState[lane] != 0;
    for (uint32_t localShift = 0; localShift < shifts; ++localShift) {
        const int64_t shift = firstShift + localShift;
        const int64_t lookback = WrapSub(position, shift);
        int64_t source;
        if (lookback >= chunkStart) {
            const int64_t batchIndex = token > shift ? token - shift : 0;
            const int64_t tokenId = inputs.inputIds[WrapMul(batchIndex, tiling.inputStride)];
            source = inputs.tokenMap[tokenId];
            if (inputs.deadMask[WrapMul(batchIndex, tiling.maskStride)] != 0) source = tiling.deadId;
        } else {
            // Subtract in the original positions dtype before promoting to int64.
            const PositionT previous = static_cast<PositionT>(static_cast<uint64_t>(chunkStart) - 1);
            const int64_t col = WrapSub(static_cast<int64_t>(previous), lookback);
            bool known = false;
            if (col >= 0 && col < tiling.lookbackDepth) {
                const int64_t tokenId = inputs.lookbackTokenIds[
                    WrapAdd(WrapMul(request, tiling.lookbackRowStride), WrapMul(col, tiling.lookbackColStride))];
                if (tokenId > UNKNOWN_TOKEN_ID) {
                    source = inputs.tokenMap[tokenId];
                    if (inputs.lookbackDeadMask[WrapAdd(WrapMul(request, tiling.lookbackMaskRowStride),
                                                       WrapMul(col, tiling.lookbackMaskColStride))] != 0) {
                        source = tiling.deadId;
                    }
                    known = true;
                }
            }
            if (!known) {
                if (tiling.hasCache == 0) {
                    source = static_cast<int64_t>(static_cast<int32_t>(tiling.padId));
                } else {
                    const uint64_t extent = static_cast<uint64_t>(tiling.maxBlocks) * tiling.cacheBlockSize;
                    const int64_t cachePosition = Clamp(lookback, 0, static_cast<int64_t>(extent - 1));
                    const int64_t row = request < tiling.numTableRows ? request : tiling.numTableRows - 1;
                    const int64_t block = inputs.blockTable[WrapAdd(WrapMul(row, tiling.tableRowStride),
                        WrapMul(cachePosition / tiling.cacheBlockSize, tiling.tableColStride))];
                    const int64_t cacheIndex = WrapAdd(WrapMul(block, tiling.cacheBlockSize),
                                                       cachePosition % tiling.cacheBlockSize);
                    source = inputs.cache[Clamp(cacheIndex, 0, tiling.numSlots - 1)];
                }
            }
        }
        blocked = blocked || lookback < 0 || source == tiling.deadId;
        history[localShift * TOKEN_TILE_ELEMENTS + lane] = static_cast<uint64_t>(blocked ? tiling.padId : source);
    }
    // Preserve each token's state when a large n-gram spans multiple history groups.
    if (firstShift + shifts < tiling.maxNgram) blockedState[lane] = blocked;
}
} // namespace SourceSimt

template <typename InputT, typename MapT, typename PositionT, typename LookbackT>
class HashSources {
public:
    __aicore__ inline void Init(
        GM_ADDR inputIds, GM_ADDR tokenMap, GM_ADDR deadMask, GM_ADDR positions,
        GM_ADDR blockTable, GM_ADDR queryStartLoc, GM_ADDR reqIds, GM_ADDR cache,
        GM_ADDR lookbackTokenIds, GM_ADDR lookbackDeadMask,
        const EngramHashIdsTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        inputs_ = {reinterpret_cast<__gm__ InputT *>(inputIds), reinterpret_cast<__gm__ MapT *>(tokenMap),
                   reinterpret_cast<__gm__ uint8_t *>(deadMask), reinterpret_cast<__gm__ PositionT *>(positions),
                   reinterpret_cast<__gm__ int32_t *>(blockTable), reinterpret_cast<__gm__ int32_t *>(queryStartLoc),
                   reinterpret_cast<__gm__ int32_t *>(reqIds), reinterpret_cast<__gm__ int32_t *>(cache),
                   reinterpret_cast<__gm__ LookbackT *>(lookbackTokenIds),
                   reinterpret_cast<__gm__ uint8_t *>(lookbackDeadMask)};
        pipe->InitBuffer(historyBuffer_, SourceSimt::HISTORY_SHIFT_TILE_ELEMENTS * TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
        pipe->InitBuffer(blockedBuffer_, TOKEN_TILE_ELEMENTS * sizeof(uint32_t));
    }

    __aicore__ inline void Prepare(int64_t, uint32_t count, HashArithmetic &arithmetic)
    {
        arithmetic.Reset(count);
    }

    __aicore__ inline void HashShift(HashArithmetic &arithmetic, int64_t begin, uint32_t count,
                                    int64_t shift, uint64_t multiplier)
    {
        const uint32_t localShift = shift % SourceSimt::HISTORY_SHIFT_TILE_ELEMENTS;
        const auto history = historyBuffer_.Get<uint64_t>();
        if (localShift == 0) {
            const int64_t remaining = tiling_.maxNgram - shift;
            const uint32_t shifts = remaining < SourceSimt::HISTORY_SHIFT_TILE_ELEMENTS ?
                remaining : SourceSimt::HISTORY_SHIFT_TILE_ELEMENTS;
            Simt::VF_CALL<SourceSimt::PrepareHistory<InputT, MapT, PositionT, LookbackT>>(
                Simt::Dim3(tiling_.tokenTileSize), inputs_, tiling_,
                reinterpret_cast<__ubuf__ uint64_t *>(history.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(blockedBuffer_.Get<uint32_t>().GetPhyAddr()),
                begin, count, shift, shifts);
        }
        arithmetic.FinishShift(multiplier, count, history[localShift * TOKEN_TILE_ELEMENTS]);
    }

private:
    EngramHashIdsTilingData tiling_;
    SourceSimt::Inputs<InputT, MapT, PositionT, LookbackT> inputs_;
    TBuf<TPosition::VECCALC> historyBuffer_;
    TBuf<TPosition::VECCALC> blockedBuffer_;
};
} // namespace EngramHashIds
