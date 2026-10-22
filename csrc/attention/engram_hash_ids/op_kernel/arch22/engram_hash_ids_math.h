// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include "kernel_operator.h"
#include "../engram_hash_ids_constants.h"

namespace EngramHashIds {
using namespace AscendC;

// Keep the A3 scalar arithmetic and its per-token update order.
class HashArithmetic {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(rollingBuffer_, TOKEN_TILE_ELEMENTS * sizeof(uint64_t));
    }

    __aicore__ inline void ResetLane(uint32_t lane)
    {
        rollingBuffer_.Get<uint64_t>().SetValue(lane, uint64_t(0));
    }

    __aicore__ inline void Accumulate(uint32_t lane, int64_t value, uint64_t multiplier)
    {
        const auto rolling = rollingBuffer_.Get<uint64_t>();
        // Unsigned multiplication preserves the low 64 bits, including overflow.
        rolling.SetValue(lane, rolling.GetValue(lane) ^ (static_cast<uint64_t>(value) * multiplier));
    }

    __aicore__ inline void FinishShift(uint64_t, uint32_t) {}
    __aicore__ inline void PrepareHead(int64_t, uint64_t, uint32_t) {}

    __aicore__ inline int32_t Output(uint32_t lane, int64_t prime, uint64_t offset)
    {
        const int64_t value = static_cast<int64_t>(rollingBuffer_.Get<uint64_t>().GetValue(lane));
        // INT64_MIN % -1 is undefined in C++; the required remainder is zero.
        const int64_t remainder = prime == -1 ? 0 : value % prime;
        return static_cast<int32_t>(static_cast<uint64_t>(remainder) + offset);
    }

private:
    TBuf<TPosition::VECCALC> rollingBuffer_;
};
} // namespace EngramHashIds
