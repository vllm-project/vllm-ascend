// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "kernel_operator.h"
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

    __aicore__ inline void ResetLane(uint32_t lane)
    {
        rollingBuffer_.Get<uint64_t>().SetValue(rollingOffset_ + lane, uint64_t(0));
    }

    __aicore__ inline void Accumulate(uint32_t lane, int64_t value, uint64_t)
    {
        sourceBuffer_.Get<uint64_t>().SetValue(lane, static_cast<uint64_t>(value));
    }

    __aicore__ inline void FinishShift(uint64_t multiplier, uint32_t count)
    {
        const auto rolling = rollingBuffer_.Get<uint64_t>();
        const auto product = scratchBuffer_.Get<uint64_t>();
        const uint32_t next = TOKEN_TILE_ELEMENTS - rollingOffset_;
        Sync<HardEvent::S_V>();
        Muls(product, sourceBuffer_.Get<uint64_t>(), multiplier, count);
        PipeBarrier<PIPE_V>();
        BitwiseXor(rolling[next], rolling[rollingOffset_], product, count);
        Sync<HardEvent::V_S>();
        rollingOffset_ = next;
    }

    template <bool scalarOutput = true>
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
        if constexpr (scalarOutput) Sync<HardEvent::V_S>();
    }

    __aicore__ inline void WriteHead(const LocalTensor<int32_t>& output, uint32_t head,
                                   int64_t prime, uint64_t offset, uint32_t count)
    {
        PrepareHead<false>(prime, offset, count);
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

    __aicore__ inline int32_t Output(uint32_t lane, int64_t, uint64_t)
    {
        // A vector Cast to int32 may saturate; narrowing here preserves wrapping.
        return static_cast<int32_t>(sourceBuffer_.Get<uint64_t>().GetValue(lane));
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
} // namespace EngramHashIds
