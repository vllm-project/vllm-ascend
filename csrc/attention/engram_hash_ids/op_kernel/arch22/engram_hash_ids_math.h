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
} // namespace EngramHashIds
