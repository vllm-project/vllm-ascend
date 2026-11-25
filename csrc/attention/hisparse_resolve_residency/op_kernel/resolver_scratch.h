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

// arch22 kernels use scalar UB storage without the SIMT DCache reservation; the
// runtime reserve is 16 KiB and the complete hot-tag row stays resident across
// the union walk (see the migrated arch22 resolver header).
inline constexpr uint64_t kResolverArch22ReserveBytes = 16 * 1024;
inline constexpr uint64_t kResolverArch22Tile = 256;
inline constexpr uint64_t kResolverArch22MinResidentHot = 3072;
inline constexpr uint64_t kResolverArch22MaxResidentColumns = 1024;

inline constexpr uint64_t resolver_arch22_ub_budget(uint64_t platform_ub_bytes) {
    return platform_ub_bytes > kResolverArch22ReserveBytes ?
        platform_ub_bytes - kResolverArch22ReserveBytes : 0;
}

#if defined(__CCE_AICORE__)
#define HISPARSE_ARCH22_LAYOUT __aicore__ inline
#else
#define HISPARSE_ARCH22_LAYOUT constexpr
#endif

HISPARSE_ARCH22_LAYOUT uint64_t resolver_arch22_align(uint64_t bytes) {
    return (bytes + 31) / 32 * 32;
}

HISPARSE_ARCH22_LAYOUT uint64_t resolver_arch22_values_bytes(uint64_t hot, uint64_t hash_size) {
    const uint64_t columns = hash_size < kResolverArch22MaxResidentColumns ? hash_size : kResolverArch22MaxResidentColumns;
    const uint64_t slots = hash_size * sizeof(int16_t);
    const uint64_t table = hot >= kResolverArch22MinResidentHot ? columns * sizeof(int32_t) : 0;
    return resolver_arch22_align(slots > table ? slots : table);
}

HISPARSE_ARCH22_LAYOUT uint64_t resolver_arch22_order_bytes(uint64_t hot) {
    const uint64_t slots = hot * sizeof(int16_t);
    const uint64_t resident = 10 * kResolverArch22Tile * sizeof(int32_t);
    return resolver_arch22_align(hot >= kResolverArch22MinResidentHot && slots < resident ? resident : slots);
}

HISPARSE_ARCH22_LAYOUT uint64_t resolver_arch22_scratch_bytes(uint64_t hot, uint64_t hash_size) {
    return resolver_arch22_align(hash_size * sizeof(int32_t)) + resolver_arch22_values_bytes(hot, hash_size) +
           resolver_arch22_align(hot * sizeof(int32_t)) + resolver_arch22_order_bytes(hot) +
           2 * kResolverArch22Tile * sizeof(int32_t) + kResolverDescriptorBytes;
}

HISPARSE_ARCH22_LAYOUT uint64_t resolver_arch22_hash_capacity(uint64_t hot, uint64_t union_rows, uint64_t ub_budget) {
    uint64_t lower = union_rows + 1;
    uint64_t upper = 1;
    while (upper <= union_rows) upper <<= 1;
    // The caller already validated that lower fits; keep the largest usable hash.
    while (lower < upper) {
        const uint64_t middle = (lower + upper + 1) / 2;
        if (resolver_arch22_scratch_bytes(hot, middle) <= ub_budget) {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    return lower;
}
#undef HISPARSE_ARCH22_LAYOUT
}  // namespace vllm_ascend::hisparse
