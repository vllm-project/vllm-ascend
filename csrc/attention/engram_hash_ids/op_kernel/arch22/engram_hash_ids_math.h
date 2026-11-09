// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <climits>

#include "kernel_operator.h"
#include "../engram_hash_ids_constants.h"

namespace EngramHashIds {
using namespace AscendC;

// Exact low-64-bit multiplication and XOR using A3's narrower vector operations.
class HashArithmetic {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(rollingBuffer_, TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
        pipe->InitBuffer(sourceBuffer_, TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
        pipe->InitBuffer(workBuffer_, WORK_PLANES * TOKEN_TILE_ELEMENTS * sizeof(int32_t));
        pipe->InitBuffer(indexBuffer_, (1 + WORDS_PER_VALUE) * TOKEN_TILE_ELEMENTS * sizeof(uint32_t));
        const auto indices = indexBuffer_.Get<uint32_t>();
        for (uint32_t lane = 0; lane < TOKEN_TILE_ELEMENTS; ++lane) {
            indices.SetValue(lane, lane * sizeof(uint64_t));
            for (uint32_t word = 0; word < WORDS_PER_VALUE; ++word) {
                indices.SetValue(TOKEN_TILE_ELEMENTS + lane * WORDS_PER_VALUE + word,
                                 (word * TOKEN_TILE_ELEMENTS + lane) * sizeof(uint32_t));
            }
        }
    }

    __aicore__ inline void ResetLane(uint32_t lane)
    {
        rollingBuffer_.Get<uint64_t>().SetValue(lane, uint64_t(0));
    }

    __aicore__ inline void Accumulate(uint32_t lane, int64_t value, uint64_t)
    {
        sourceBuffer_.Get<uint64_t>().SetValue(lane, static_cast<uint64_t>(value));
    }

    __aicore__ inline void FinishShift(uint64_t multiplier, uint32_t count)
    {
        const auto work = workBuffer_.Get<int32_t>();
        const auto low = work[RESULT_LOW * TOKEN_TILE_ELEMENTS];
        const auto high = work[RESULT_HIGH * TOKEN_TILE_ELEMENTS];
        const auto inputLow = work[INPUT_LOW * TOKEN_TILE_ELEMENTS];
        const auto inputHigh = work[INPUT_HIGH * TOKEN_TILE_ELEMENTS];
        const auto digitLow = work[DIGIT_LOW * TOKEN_TILE_ELEMENTS];
        const auto digitHigh = work[DIGIT_HIGH * TOKEN_TILE_ELEMENTS];
        const auto carry = work[CARRY * TOKEN_TILE_ELEMENTS];
        const auto scratch = work[SCRATCH * TOKEN_TILE_ELEMENTS];
        const auto temporary = work[TEMPORARY * TOKEN_TILE_ELEMENTS];
        const auto indices = indexBuffer_.Get<uint32_t>();
        const int32_t multiplierLow = static_cast<int32_t>(multiplier);
        const int32_t multiplierHigh = static_cast<int32_t>(multiplier >> WORD_BITS);
        const int32_t factorLow = multiplier & UINT16_MAX;
        const int32_t factorHigh = (multiplier >> DIGIT_BITS) & UINT16_MAX;
        Sync<HardEvent::S_V>();
        Gather(inputLow, sourceBuffer_.Get<int32_t>(), indices, uint32_t(0), count);
        Gather(inputHigh, sourceBuffer_.Get<int32_t>(), indices, uint32_t(sizeof(uint32_t)), count);
        PipeBarrier<PIPE_V>();

        // Cross products only contribute their low 32 bits to the result's high word.
        Muls(high, inputHigh, multiplierLow, count);
        Muls(scratch, inputLow, multiplierHigh, count);
        LowDigit(digitLow, inputLow, count);
        ShiftRight(digitHigh.ReinterpretCast<uint32_t>(), inputLow.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        PipeBarrier<PIPE_V>();
        Add(high, high, scratch, count);

        // Compute the full inputLow * multiplierLow from four unsigned 16-bit products.
        Muls(low, digitLow, factorLow, count);
        Muls(carry, digitHigh, factorLow, count);
        PipeBarrier<PIPE_V>();
        ShiftRight(scratch.ReinterpretCast<uint32_t>(), low.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        PipeBarrier<PIPE_V>();
        Add(carry, carry, scratch, count); // t = a1*b0 + high16(a0*b0)
        PipeBarrier<PIPE_V>();
        Muls(scratch, digitLow, factorHigh, count);
        PipeBarrier<PIPE_V>();
        LowDigit(temporary, carry, count);
        Add(scratch, scratch, temporary, count); // u = a0*b1 + low16(t)
        PipeBarrier<PIPE_V>();
        Muls(temporary, digitHigh, factorHigh, count);
        ShiftRight(carry.ReinterpretCast<uint32_t>(), carry.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        PipeBarrier<PIPE_V>();
        Add(high, high, carry, count);
        ShiftRight(carry.ReinterpretCast<uint32_t>(), scratch.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        PipeBarrier<PIPE_V>();
        Add(high, high, carry, count);
        ShiftLeft(scratch.ReinterpretCast<uint32_t>(), scratch.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        LowDigit(low, low, count);
        Add(high, high, temporary, count);
        Add(low, low, scratch, count);
        PipeBarrier<PIPE_V>();

        // Reuse the input storage for the interleaved uint64 product, then stay on PIPE_V for XOR.
        Gather(sourceBuffer_.Get<int32_t>(), work, indices[TOKEN_TILE_ELEMENTS], uint32_t(0),
               count * WORDS_PER_VALUE);
        PipeBarrier<PIPE_V>();
        const auto rolling = rollingBuffer_.Get<uint16_t>();
        const auto product = sourceBuffer_.Get<uint16_t>();
        const auto xorTemporary = inputLow.ReinterpretCast<uint16_t>();
        const uint32_t elements = count * XOR_PARTS_PER_VALUE;
        // a ^ b = (a | b) & ~(a & b), without any numeric conversion.
        And(xorTemporary, rolling, product, elements);
        PipeBarrier<PIPE_V>();
        Or(rolling, rolling, product, elements);
        Not(xorTemporary, xorTemporary, elements);
        PipeBarrier<PIPE_V>();
        And(rolling, rolling, xorTemporary, elements);
        Sync<HardEvent::V_S>();
    }

    __aicore__ inline void PrepareHead(int64_t, uint64_t, uint32_t) {}

    __aicore__ inline int32_t Output(uint32_t lane, int64_t prime, uint64_t offset)
    {
        const int64_t value = static_cast<int64_t>(rollingBuffer_.Get<uint64_t>().GetValue(lane));
        // INT64_MIN % -1 is undefined in C++; the required remainder is zero.
        const int64_t remainder = prime == -1 ? 0 : value % prime;
        return static_cast<int32_t>(static_cast<uint64_t>(remainder) + offset);
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        SetFlag<event>(COPY_EVENT_ID);
        WaitFlag<event>(COPY_EVENT_ID);
    }

    __aicore__ inline void LowDigit(const LocalTensor<int32_t>& dst, const LocalTensor<int32_t>& src,
                                   uint32_t count)
    {
        ShiftLeft(dst.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        PipeBarrier<PIPE_V>();
        ShiftRight(dst.ReinterpretCast<uint32_t>(), dst.ReinterpretCast<uint32_t>(), DIGIT_BITS, count);
        PipeBarrier<PIPE_V>();
    }

    enum WorkPlane {
        RESULT_LOW, RESULT_HIGH, INPUT_LOW, INPUT_HIGH, DIGIT_LOW, DIGIT_HIGH,
        CARRY, SCRATCH, TEMPORARY, WORK_PLANES
    };
    static constexpr uint32_t WORD_BITS = sizeof(uint32_t) * CHAR_BIT;
    static constexpr uint32_t DIGIT_BITS = sizeof(uint16_t) * CHAR_BIT;
    static constexpr uint32_t WORDS_PER_VALUE = sizeof(uint64_t) / sizeof(uint32_t);
    static constexpr uint32_t XOR_PARTS_PER_VALUE = sizeof(uint64_t) / sizeof(uint16_t);
    TBuf<TPosition::VECCALC> rollingBuffer_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> workBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
};

// A3 retains the scalar history selection and metadata reuse.
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
        inputIds_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(inputIds));
        tokenMap_.SetGlobalBuffer(reinterpret_cast<__gm__ MapT *>(tokenMap));
        deadMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(deadMask));
        positions_.SetGlobalBuffer(reinterpret_cast<__gm__ PositionT *>(positions));
        queryStartLoc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
        reqIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(reqIds));
        lookbackTokenIds_.SetGlobalBuffer(reinterpret_cast<__gm__ LookbackT *>(lookbackTokenIds));
        lookbackDeadMask_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(lookbackDeadMask));
        if (tiling_.hasCache != 0) {
            blockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
            cache_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cache));
        }
        pipe->InitBuffer(requestBuffer_, TOKEN_TILE_ELEMENTS * sizeof(int32_t));
        pipe->InitBuffer(positionBuffer_, TOKEN_TILE_ELEMENTS * sizeof(PositionT));
        pipe->InitBuffer(chunkStartBuffer_, TOKEN_TILE_ELEMENTS * sizeof(int64_t));
    }

    __aicore__ inline void Prepare(int64_t begin, uint32_t count, HashArithmetic &arithmetic)
    {
        CacheMetadata(begin, count);
        for (uint32_t lane = 0; lane < count; ++lane) {
            blocked_[lane] = false;
            arithmetic.ResetLane(lane);
        }
    }

    __aicore__ inline void HashShift(HashArithmetic &arithmetic, int64_t begin, uint32_t count,
                                    int64_t shift, uint64_t multiplier)
    {
        const auto requests = requestBuffer_.Get<int32_t>();
        const auto positions = positionBuffer_.Get<PositionT>();
        const auto starts = chunkStartBuffer_.Get<int64_t>();
        for (uint32_t lane = 0; lane < count; ++lane) {
            const int64_t lookback = WrapSub(static_cast<int64_t>(positions.GetValue(lane)), shift);
            const int64_t source = ReadSource(begin + lane, requests.GetValue(lane), starts.GetValue(lane),
                                              lookback, shift);
            blocked_[lane] = blocked_[lane] || lookback < 0 || source == tiling_.deadId;
            const int64_t value = blocked_[lane] ? tiling_.padId : source;
            arithmetic.Accumulate(lane, value, multiplier);
        }
        arithmetic.FinishShift(multiplier, count);
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

    EngramHashIdsTilingData tiling_;
    GlobalTensor<InputT> inputIds_;
    GlobalTensor<MapT> tokenMap_;
    GlobalTensor<uint8_t> deadMask_;
    GlobalTensor<PositionT> positions_;
    GlobalTensor<int32_t> blockTable_;
    GlobalTensor<int32_t> queryStartLoc_;
    GlobalTensor<int32_t> reqIds_;
    GlobalTensor<int32_t> cache_;
    GlobalTensor<LookbackT> lookbackTokenIds_;
    GlobalTensor<uint8_t> lookbackDeadMask_;
    TBuf<TPosition::VECCALC> requestBuffer_;
    TBuf<TPosition::VECCALC> positionBuffer_;
    TBuf<TPosition::VECCALC> chunkStartBuffer_;
    bool blocked_[TOKEN_TILE_ELEMENTS];
};
} // namespace EngramHashIds
