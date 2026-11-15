// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend::hisparse {
// SIMT kernels must reserve DCache space out of the UB capacity reported by
// GetCoreMemSize before declaring local memory via SetLocalMemorySize (the
// same pattern as k2q_csr / scatter_nd_update_sk on this target).
inline constexpr uint64_t kResolverDcacheReserveBytes = 32 * 1024;
// Host-side static upper bound: A5 reports 256 KiB of UB per AIV, minus the
// DCache reservation. Used only where the platform API is unavailable.
inline constexpr uint64_t kResolverMaxSimtUbBytes = 256 * 1024 - kResolverDcacheReserveBytes;
inline constexpr uint64_t kResolverDescriptorBytes = 512;

inline constexpr uint64_t resolver_simt_ub_budget(uint64_t platform_ub_bytes) {
    return platform_ub_bytes > kResolverDcacheReserveBytes ?
        platform_ub_bytes - kResolverDcacheReserveBytes : 0;
}

constexpr uint64_t resolver_scratch_words(uint64_t hot, uint64_t hash_size) {
    const uint64_t tiles = (hot + 31) / 32;
    return (2 * hash_size + hot + 2 * tiles + 64 + 15) / 16 * 16;
}

constexpr uint64_t resolver_scratch_bytes(uint64_t hot, uint64_t hash_size) {
    return resolver_scratch_words(hot, hash_size) * sizeof(int32_t) + kResolverDescriptorBytes;
}

constexpr uint64_t resolver_hash_capacity(uint64_t hot, uint64_t union_rows, uint64_t ub_budget) {
    uint64_t hash_size = 1;
    while (hash_size <= union_rows) hash_size <<= 1;
    const uint64_t tiles = (hot + 31) / 32;
    const uint64_t fixed_words = hot + 2 * tiles + 64;
    const uint64_t available_words = (ub_budget - kResolverDescriptorBytes) / 64 * 16;
    const uint64_t max_hash_size = (available_words - fixed_words) / 2;
    return hash_size < max_hash_size ? hash_size : max_hash_size;
}
}  // namespace vllm_ascend::hisparse
