// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend::hisparse {
struct alignas(8) ResolverTilingData {
    uint64_t geometry[25];
    uint32_t ubBytes;
};

inline constexpr uint64_t kResolverUbBytes = 175616;
inline constexpr uint64_t kResolverDescriptorBytes = 512;

constexpr uint64_t resolver_scratch_words(uint64_t hot, uint64_t hash_size) {
    const uint64_t tiles = (hot + 31) / 32;
    return (2 * hash_size + hot + 2 * tiles + 64 + 15) / 16 * 16;
}
}  // namespace vllm_ascend::hisparse
