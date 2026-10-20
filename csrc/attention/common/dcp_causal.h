// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef VLLM_ASCEND_DCP_CAUSAL_H
#define VLLM_ASCEND_DCP_CAUSAL_H

#include <cstdint>

namespace vllm_ascend::dcp {
// int32 context: [world size, rank, raw-token interleave, global seq lengths...].
constexpr uint32_t SIZE_INDEX = 0;
constexpr uint32_t RANK_INDEX = 1;
constexpr uint32_t INTERLEAVE_INDEX = 2;
constexpr uint32_t SEQ_LENS_OFFSET = 3;

#ifdef __CCE_AICORE__
__aicore__
#endif
inline uint32_t LocalCompressedLength(int64_t rawEnd, uint32_t ratio, uint32_t size, uint32_t rank,
                                      uint32_t interleave)
{
    // Empty/padded queries, and queries preceding the first complete group.
    if (rawEnd <= 0) {
        return 0;
    }
    const int64_t aligned = rawEnd / ratio * ratio;
    const int64_t cycle = static_cast<int64_t>(interleave) * size;
    const int64_t base = aligned / cycle * interleave;
    const int64_t remainder = aligned % cycle - static_cast<int64_t>(rank) * interleave;
    const int64_t tail = remainder < 0 ? 0 : (remainder > interleave ? interleave : remainder);
    return static_cast<uint32_t>((base + tail) / ratio);
}
}  // namespace vllm_ascend::dcp
#endif
