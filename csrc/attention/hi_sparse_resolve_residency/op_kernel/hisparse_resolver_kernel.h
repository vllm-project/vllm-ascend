// SPDX-License-Identifier: Apache-2.0
// Request-union algorithm adapted from vLLM PR #59235 (Apache-2.0).
// SIMT implementation: one AIV per contiguous request group; one launch.
// Descriptor and scratch are invocation-owned. All mutable inputs are re-read.
#pragma once

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include <cstdint>
#include "resolver_scratch.h"

namespace {
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kTotal = kWarps;
constexpr int kError = kWarps + 2;
static_assert(kError < 64, "Prefix/error scratch exceeds reservation");
constexpr int kHit = int(0x80000000U);
constexpr int kMiss = 0x40000000;
constexpr int kSlotMask = 0xFFFF;

__simt_callee__ inline int minimum(int a, int b) { return a < b ? a : b; }
__simt_callee__ inline int maximum(int a, int b) { return a > b ? a : b; }

// Per-warp sums suffice for a deterministic, stable block prefix scan.
// All threads participate, including masked lanes. No cross-AIV barriers.
__simt_callee__ inline int prefix(int value, __ubuf__ int* sums, int tid) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int offset = 1; offset < 32; offset *= 2) {
        int other = asc_shfl_up(value, offset, 32);
        if (lane >= offset) value += other;
    }
    if (lane == 31) sums[warp] = value;
    asc_syncthreads();
    // One warp computes all warp prefixes instead of every thread repeatedly
    // loading preceding sums. Both original block barriers remain in place.
    if (warp == 0) {
        int total = lane < kWarps ? sums[lane] : 0;
        for (int offset = 1; offset < 32; offset *= 2) {
            int other = asc_shfl_up(total, offset, 32);
            if (lane >= offset) total += other;
        }
        if (lane < kWarps) sums[lane] = total;
        if (lane == kWarps - 1) sums[kTotal] = total;
    }
    asc_syncthreads();
    return value + (warp == 0 ? 0 : sums[warp - 1]);
}

// Tile scan when a single warp covers every tile: the exclusive prefix is
// derived in registers from the warp scan, so one barrier publishes both the
// per-tile offsets and the block total. Replaces prefix()'s two barriers and
// the caller's per-batch loop barrier with a single barrier.
__simt_callee__ inline int scan_tiles_small(
    int count, __ubuf__ int* offsets, __ubuf__ int* sums, int tid, int tiles) {
    int inclusive = count;
    for (int offset = 1; offset < 32; offset *= 2) {
        const int other = asc_shfl_up(inclusive, offset, 32);
        if ((tid & 31) >= offset) inclusive += other;
    }
    if (tid == 31) sums[kTotal] = inclusive;
    if (tid < tiles) offsets[tid] = inclusive - count;
    asc_syncthreads();
    return sums[kTotal];
}

__simt_callee__ inline int hash_slot(int key, int size) {
    const uint32_t mixed = uint32_t(key) * 2654435761U;
    return (size & (size - 1)) == 0 ? int(mixed & (size - 1)) :
        int((uint64_t(mixed) * uint32_t(size)) >> 32);
}

__simt_callee__ inline void store_miss(__ubuf__ uint64_t* d, int64_t index, int value) {
    if (!d[15]) return;
    if (d[56]) ((__gm__ uint8_t*)d[15])[index] = uint8_t(value);
    else ((__gm__ int32_t*)d[15])[index] = value;
}

__simt_callee__ inline int lookup(__ubuf__ int* keys, int key, int size) {
    int h = hash_slot(key, size);
    for (int probes = 0; probes < size; ++probes) {
        int old = keys[h];
        if (old == key) return h;
        if (old == -1) return -1;
        h = h + 1 == size ? 0 : h + 1;
    }
    return -1;
}

template<bool kQualified>
__simt_callee__ inline int attention_row(__ubuf__ uint64_t* d, int id) {
    // Widen before multiplying: a later int64 addition cannot fix overflow.
    const int block = kQualified ? 128 : int(d[48]);
    const int64_t mapped = id >= 0 ? int64_t(id / block) * int(d[51]) + id % block : -1;
    return mapped >= 0 && mapped < int64_t(d[55]) ? int(mapped) : -1;
}

// Only the optional Attention output is compacted. Refill still consumes the
// original virtual-page row IDs and the unchanged request-union swap plan.
template<bool kQualified>
__simt_callee__ inline void compact_attention(
    __ubuf__ uint64_t* d, __ubuf__ int* scratch, int row, bool already_mapped = false) {
    const int tid = threadIdx.x;
    const int topk = int(d[33]);
    const int tiles = (topk + 31) / 32;
    auto physical = (__gm__ int32_t*)d[8];
    auto attention = (__gm__ int32_t*)d[9];
    auto counts = (__gm__ int32_t*)d[11];
    const int64_t source_row = int64_t(row) * topk;
    const int64_t target_row = int64_t(row) * d[50];
    __ubuf__ int* values = scratch;
    auto masks = (__ubuf__ uint32_t*)(values + topk);
    __ubuf__ int* offsets = (__ubuf__ int*)(masks + tiles);
    __ubuf__ int* sums = offsets + tiles;
    // The old hash/LRU allocation covers topk + two warp tables + prefix sums.
    // Stage before scattering: neither aliasing nor thread order may lose a row.
    for (int base = 0; base < topk; base += kThreads) {
        const int col = base + tid;
        const int value = col < topk ? (already_mapped ? attention[target_row + col] :
            attention_row<kQualified>(d, physical[source_row + col])) : -1;
        if (col < topk) values[col] = value;
        const uint32_t mask = asc_ballot(int(value >= 0));
        if ((tid & 31) == 0 && col < topk) masks[col >> 5] = mask;
    }
    asc_syncthreads();
    int total;
    if (tiles <= 32) {
        total = scan_tiles_small(tid < tiles ? __popc(masks[tid]) : 0, offsets, sums, tid, tiles);
    } else {
        total = 0;
        for (int base = 0; base < tiles; base += kThreads) {
            const int tile = base + tid;
            const int count = tile < tiles ? __popc(masks[tile]) : 0;
            const int preceding = prefix(count, sums, tid);
            const int tile_total = sums[kTotal];
            if (tile < tiles) offsets[tile] = total + preceding - count;
            asc_syncthreads();
            total += tile_total;
        }
    }
    for (int col = tid; col < topk; col += kThreads) {
        const int value = values[col];
        if (value >= 0) {
            const uint32_t bit = uint32_t(1) << (col & 31);
            const int destination = offsets[col >> 5] + __popc(masks[col >> 5] & (bit - 1));
            attention[target_row + destination] = value;
        }
    }
    for (int col = total + tid; col < topk; col += kThreads) attention[target_row + col] = -1;
    if (tid == 0 && counts) counts[int64_t(row) * d[52]] = total;
    asc_syncthreads();
}

template<bool kQualified>
__simt_vf__ __launch_bounds__(kThreads) inline void resolve_vf(
    __ubuf__ uint64_t* d, __ubuf__ int* scratch, int first) {
    const int tid = threadIdx.x;
    auto logical = (__gm__ int32_t*)d[0];
    auto requests = (__gm__ int32_t*)d[1];
    auto states = (__gm__ int32_t*)d[2];
    auto source = (__gm__ int32_t*)d[3];
    auto resident = (__gm__ int32_t*)d[4];
    auto table = (__gm__ int32_t*)d[5];
    auto dgi = (__gm__ int32_t*)d[6];
    auto lru = (__gm__ int16_t*)d[7];
    auto output = (__gm__ int32_t*)d[8];
    auto attention = (__gm__ int32_t*)d[9];
    auto resolved = (__gm__ int32_t*)d[10];
    auto counts = (__gm__ int32_t*)d[11];
    auto swap_host = (__gm__ int32_t*)d[12];
    auto swap_device = (__gm__ int32_t*)d[13];
    auto swap_counts = (__gm__ int32_t*)d[14];
    auto stats = (__gm__ int64_t*)d[16];
    auto errors = (__gm__ int32_t*)d[17];
    auto num_valid_rows = (__gm__ int32_t*)d[18];
    const int rows = d[32];
    const int topk = kQualified ? 2048 : int(d[33]);
    const int hot = kQualified ? 10240 : int(d[34]);
    const int hash_size = kQualified ? 16384 : int(d[54]);
    const bool compact = d[55] != 0;
    const int valid_rows = num_valid_rows ? minimum(rows, maximum(0, num_valid_rows[0])) : rows;
    if (first >= valid_rows) {
        if (tid < 8 && errors) errors[first * 8 + tid] = 0;
        for (int i = tid; i < topk; i += kThreads) {
            const int64_t index = int64_t(first) * topk + i;
            output[index] = -1;
            if (attention) attention[int64_t(first) * d[50] + i] = -1;
            if (resolved) resolved[index] = -1;
            store_miss(d, index, 0);
        }
        if (tid == 0) {
            if (counts) counts[int64_t(first) * d[52]] = 0;
            if (swap_counts) swap_counts[first] = 0;
        }
        return;
    }
    int request = requests ? requests[first] : first;
    int state = states ? ((request >= 0 && request < int(d[37])) ? states[request] : -1) : request;
    bool state_valid = state >= 0 && state < int(d[38]);
    // Source/Resident tables are immutable for this invocation. A complete
    // device-table proof lets every AIV translate its own row independently;
    // a mixed request still uses the original leader-owned union/LRU path.
    if (kQualified && source && resident && requests && attention && num_valid_rows &&
        int(d[50]) == topk && state_valid && state < int(d[43]) &&
        request >= 0 && request < int(d[39]) && int(d[40]) > 0 && int(d[44]) > 0) {
        const int host0 = source[int64_t(request) * d[41]];
        const int page0 = resident[int64_t(state) * d[45]];
        // The ordinary cold case returns to the old path without a page scan.
        if (host0 <= 0 || (page0 >= 0 && page0 != int(d[47]))) {
            bool missing = false;
            for (int col = tid; col < int(d[40]); col += kThreads) {
                const int host_page = source[int64_t(request) * d[41] + col];
                const int page = col < int(d[44]) ? resident[int64_t(state) * d[45] + col] : int(d[47]);
                missing |= host_page > 0 && (page < 0 || page > 0xFFFFFF || page == int(d[47]));
            }
            const int lane = tid & 31;
            const uint32_t warp_missing = asc_ballot(int(missing));
            if (lane == 0) scratch[tid >> 5] = warp_missing != 0;
            asc_syncthreads();
            // Each warp reads the same flags, so the branch is uniform
            // without publishing a second reduction through shared memory.
            const bool all_resident = asc_ballot(int(lane < kWarps && scratch[lane] != 0)) == 0;
            if (compact && tid == 0) scratch[kError] = 0;
            // Every lane must finish reading the proof before fallback code
            // can repurpose this AIV's scratch for hash keys.
            asc_syncthreads();
            if (all_resident) {
                int valid_count = 0;
                bool attention_invalid = false;
                for (int i = tid; i < topk; i += kThreads) {
                    const int token = logical[int64_t(first) * d[36] + i];
                    const int col = token >= 0 ? token / 128 : -1;
                    const int host_page = token >= 0 && col < int(d[40]) ? source[int64_t(request) * d[41] + col] : 0;
                    int host = token >= 0 && host_page > 0 ? host_page * 128 + token % 128 : -1;
                    const int page = token >= 0 && col < int(d[44]) ? resident[int64_t(state) * d[45] + col] : int(d[47]);
                    const int device = token >= 0 && page >= 0 && page != int(d[47]) ? page * 128 + token % 128 : -1;
                    if (host < 0 || int64_t(host) >= int64_t(d[35])) host = -1;
                    const int64_t index = int64_t(first) * topk + i;
                    output[index] = device;
                    const int mapped = compact ? attention_row<kQualified>(d, device) :
                        (device >= 0 ? device / 128 * int(d[51]) + device % 128 : -1);
                    attention[int64_t(first) * d[50] + i] = mapped;
                    attention_invalid |= mapped < 0;
                    if (resolved) resolved[index] = host;
                    store_miss(d, index, 0);
                    valid_count += device >= 0 || host >= 0;
                }
                if (compact) {
                    if (attention_invalid) asc_atomic_or(scratch + kError, 1);
                    asc_syncthreads();
                    if (scratch[kError]) {
                        asc_syncthreads();
                        compact_attention<kQualified>(d, scratch, first, true);
                    } else if (tid == 0 && counts) {
                        counts[int64_t(first) * d[52]] = topk;
                    }
                } else if (counts) {
                    prefix(valid_count, scratch, tid);
                    if (tid == 0) counts[int64_t(first) * d[52]] = scratch[kTotal];
                }
                if (tid == 0 && swap_counts) swap_counts[first] = 0;
                // Nonleader error rows were untouched by the original path.
                if (errors && (first == 0 || requests[first - 1] != request) && tid < 8) errors[first * 8 + tid] = 0;
                return;
            }
        }
    }
    if (requests && first > 0 && requests[first - 1] == request) return;
    int end = first + 1;
    while (requests && end < valid_rows && requests[end] == request) ++end;
    __ubuf__ int* keys = scratch;
    __ubuf__ int* values = keys + hash_size;
    __ubuf__ int* temp = values + hash_size;
    __ubuf__ uint32_t* masks = (__ubuf__ uint32_t*)(temp + hot);
    __ubuf__ int* offsets = (__ubuf__ int*)(masks + (hot + 31) / 32);
    __ubuf__ int* sums = offsets + (hot + 31) / 32;
    if (tid < kWarps + 8) sums[tid] = 0;
    if (tid < 8 && errors) errors[first * 8 + tid] = 0;
    asc_syncthreads();
    int cold_references = 0;
    for (int row = first; row < end; ++row) {
        int valid_count = 0;
        for (int i = tid; i < topk; i += kThreads) {
            int token = logical[int64_t(row) * d[36] + i];
            int host = token, device = -1;
            if (source) {
                int col = token >= 0 ? token / int(d[42]) : -1;
                bool valid = state_valid && token >= 0 && request >= 0 && request < int(d[39]) && col < int(d[40]);
                int page = valid ? source[int64_t(request) * d[41] + col] : 0;
                host = valid && page > 0 ? page * int(d[42]) + token % int(d[42]) : -1;
                if (resident) {
                    col = token >= 0 ? token / int(d[46]) : -1;
                    valid = state_valid && token >= 0 && state < int(d[43]) && col < int(d[44]);
                    page = valid ? resident[int64_t(state) * d[45] + col] : int(d[47]);
                    device = valid && page >= 0 && page != int(d[47]) ? page * int(d[46]) + token % int(d[46]) : -1;
                }
            }
            if (!state_valid || host < 0 || int64_t(host) >= int64_t(d[35])) host = -1;
            if (!state_valid) device = -1;
            int64_t index = int64_t(row) * topk + i;
            output[index] = device;
            if (resolved) resolved[index] = host;
            store_miss(d, index, 0);
            valid_count += device >= 0 || host >= 0;
            if (device < 0 && host >= 0) {
                ++cold_references;

            }
        }
        if (counts && !compact) {
            prefix(valid_count, sums, tid);
            if (tid == 0) counts[int64_t(row) * d[52]] = sums[kTotal];
            asc_syncthreads();
        }
    }
    prefix(cold_references, sums, tid);
    int total_references = sums[kTotal];
    asc_syncthreads();
    if (!state_valid || total_references == 0 || sums[kError]) {
        const bool failed = sums[kError] != 0;
        if (compact) asc_syncthreads();
        for (int row = first; row < end; ++row) {
            if (tid == 0 && swap_counts) swap_counts[row] = 0;
            if (compact) {
                compact_attention<kQualified>(d, scratch, row);
            } else for (int i = tid; i < topk; i += kThreads) {
                int value = output[int64_t(row) * topk + i];
                if (attention) attention[int64_t(row) * d[50] + i] =
                    value >= 0 ? value / int(d[48]) * int(d[51]) + value % int(d[48]) : -1;
            }
        }
        if (tid == 0 && failed && errors) errors[first * 8] = -1;
        return;
    }
    // Match upstream's Resident-first path: initialize the union only when
    // the request actually needs Host/Hot resolution. Re-read translated rows;
    // resolved is invocation-owned and always bound by the wrapper.
    for (int i = tid; i < hash_size; i += kThreads) {
        keys[i] = -1;
        values[i] = 0x7FFFFFFF;
    }
    asc_syncthreads();
    for (int row = first; row < end; ++row) {
        for (int i = tid; i < topk; i += kThreads) {
            const int64_t index = int64_t(row) * topk + i;
            const int host = resolved[index];
            if (output[index] < 0 && host >= 0) {
                int h = hash_slot(host, hash_size);
                bool inserted = false;
                for (int probe = 0; probe < hash_size; ++probe) {
                    // Keys only transition from empty to an immutable host
                    // ID during insertion. Occupied reads need no CAS. Values
                    // only decrease, so a settled earlier position needs no min.
                    int old = keys[h];
                    if (old == -1) old = asc_atomic_cas(keys + h, -1, host);
                    if (old == -1 || old == host) {
                        int position = (row - first) * topk + i;
                        if (values[h] > position) asc_atomic_min(values + h, position);
                        inserted = true;
                        break;
                    }
                    h = h + 1 == hash_size ? 0 : h + 1;
                }
                if (!inserted) asc_atomic_or(sums + kError, 1);
            }
        }
    }
    asc_syncthreads();
    if (sums[kError]) {
        if (tid == 0 && errors) errors[first * 8] = -1;
        if (compact) asc_syncthreads();
        for (int row = first; row < end; ++row) {
            if (tid == 0 && swap_counts) swap_counts[row] = 0;
            if (compact) {
                compact_attention<kQualified>(d, scratch, row);
            } else for (int i = tid; i < topk; i += kThreads) {
                const int value = output[int64_t(row) * topk + i];
                if (attention) attention[int64_t(row) * d[50] + i] =
                    value >= 0 ? value / int(d[48]) * int(d[51]) + value % int(d[48]) : -1;
            }
        }
        return;
    }
    // Publish all hits before choosing ANY eviction. Duplicate cached host IDs
    // protect every copy; the minimum slot is selected, matching sorted keys.
    for (int slot = tid; slot < hot; slot += kThreads) {
        int host = dgi[int64_t(state) * hot + slot];
        int h = host >= 0 ? lookup(keys, host, hash_size) : -1;
        // Save membership while every Hot slot is visited. Reuse the existing
        // partition scratch only until all LRU membership masks are complete.
        temp[slot] = int(h >= 0);
        if (h >= 0) asc_atomic_min(values + h, kHit | slot);
    }
    asc_syncthreads();
    // Compute all warp hit masks first, then scan the tile counts once.
    // Preserve the exact stable LRU order while avoiding one block scan per
    // 512-slot tile. No cross-AIV state or graph-lifetime changes.
    for (int base = 0; base < hot; base += kThreads) {
        int p = base + tid;
        int slot = p < hot ? int(lru[int64_t(state) * hot + p]) : -1;
        bool member = slot >= 0 && slot < hot && temp[slot] != 0;
        uint32_t mask = asc_ballot(int(p < hot && member));
        if ((tid & 31) == 0 && p < hot) {
            masks[p >> 5] = mask;
            offsets[p >> 5] = __popc(mask);
        }
    }
    asc_syncthreads();
    int tiles = (hot + 31) / 32;
    int hit_count;
    if (tiles <= 32) {
        hit_count = scan_tiles_small(tid < tiles ? offsets[tid] : 0, offsets, sums, tid, tiles);
    } else {
        hit_count = 0;
        for (int base = 0; base < tiles; base += kThreads) {
            int tile = base + tid;
            int count = tile < tiles ? offsets[tile] : 0;
            int preceding = prefix(count, sums, tid);
            int tile_total = sums[kTotal];
            if (tile < tiles) offsets[tile] = hit_count + preceding - count;
            asc_syncthreads();
            hit_count += tile_total;
        }
    }
    for (int p = tid; p < hot; p += kThreads) {
        int slot = int(lru[int64_t(state) * hot + p]);
        uint32_t mask = masks[p >> 5];
        uint32_t bit = uint32_t(1) << (p & 31);
        bool hit = (mask & bit) != 0;
        int rank = offsets[p >> 5] + __popc(mask & (bit - 1)) + int(hit);
        temp[hit ? rank - 1 : hot - 1 - (p - rank)] = slot;
    }
    asc_syncthreads();
    // Count distinct misses first: bounds failure must not partially mutate LRU.
    int own_misses = 0;
    for (int h = tid; h < hash_size; h += kThreads) own_misses += keys[h] >= 0 && values[h] >= 0;
    prefix(own_misses, sums, tid);
    int total_misses = sums[kTotal];
    asc_syncthreads();
    if (hit_count + total_misses > int(d[53]) || total_misses > hot - hit_count) {
        if (tid == 0 && errors) errors[first * 8] = hit_count + total_misses;
        for (int row = first; row < end; ++row) {
            if (tid == 0 && swap_counts) swap_counts[row] = 0;
            if (compact && tid == 0 && counts) counts[int64_t(row) * d[52]] = 0;
            for (int i = tid; i < topk; i += kThreads) {
                output[int64_t(row) * topk + i] = -1;
                if (attention) attention[int64_t(row) * d[50] + i] = -1;
            }
        }
        return;
    }
    // Scan all first-reference miss flags before changing hash values.
    // Stable global-position offsets preserve duplicate ordering and victims.
    int positions = (end - first) * topk;
    if (positions <= minimum(tiles, kThreads) * 32) {
        for (int base = 0; base < positions; base += kThreads) {
            int position = base + tid;
            bool active = position < positions;
            int row = first + position / topk, col = position % topk;
            int host = active && output[int64_t(row) * topk + col] < 0 ? resolved[int64_t(row) * topk + col] : -1;
            int h = host >= 0 ? lookup(keys, host, hash_size) : -1;
            uint32_t mask = asc_ballot(int(h >= 0 && values[h] == position));
            if ((tid & 31) == 0 && active) {
                masks[position >> 5] = mask;
                offsets[position >> 5] = __popc(mask);
            }
        }
        asc_syncthreads();
        int tiles = (positions + 31) / 32;
        if (tiles <= 32) {
            scan_tiles_small(tid < tiles ? offsets[tid] : 0, offsets, sums, tid, tiles);
        } else {
            int preceding = prefix(tid < tiles ? offsets[tid] : 0, sums, tid);
            if (tid < tiles) offsets[tid] = preceding - __popc(masks[tid]);
            asc_syncthreads();
        }
        for (int position = tid; position < positions; position += kThreads) {
            uint32_t mask = masks[position >> 5];
            uint32_t bit = uint32_t(1) << (position & 31);
            if (mask & bit) {
                int rank = offsets[position >> 5] + __popc(mask & (bit - 1));
                int row = first + position / topk, col = position % topk;
                int host = resolved[int64_t(row) * topk + col];
                int h = lookup(keys, host, hash_size);
                int slot = temp[hot - 1 - rank];
                int block = table[int64_t(request) * d[49] + slot / int(d[48])];
                int physical = block * int(d[48]) + slot % int(d[48]);
                values[h] = kMiss | slot;
                dgi[int64_t(state) * hot + slot] = host;
                if (swap_host) swap_host[int64_t(first) * topk + rank] = host;
                if (swap_device) swap_device[int64_t(first) * topk + rank] = physical;
                store_miss(d, int64_t(row) * topk + col, 1);
            }
        }
        asc_syncthreads();
    } else {
    int consumed = 0;
    for (int base = 0; base < (end - first) * topk; base += kThreads) {
        int position = base + tid;
        bool active = position < (end - first) * topk;
        int row = first + position / topk, col = position % topk;
        int host = active && output[int64_t(row) * topk + col] < 0 ? resolved[int64_t(row) * topk + col] : -1;
        int h = host >= 0 ? lookup(keys, host, hash_size) : -1;
        bool miss = h >= 0 && values[h] == position;
        int rank = prefix(miss, sums, tid) - 1 + consumed;
        int tile_misses = sums[kTotal];
        if (miss) {
            int slot = temp[hot - 1 - rank];
            int block = table[int64_t(request) * d[49] + slot / int(d[48])];
            int physical = block * int(d[48]) + slot % int(d[48]);
            values[h] = kMiss | slot;
            dgi[int64_t(state) * hot + slot] = host;
            if (swap_host) swap_host[int64_t(first) * topk + rank] = host;
            if (swap_device) swap_device[int64_t(first) * topk + rank] = physical;
            store_miss(d, int64_t(row) * topk + col, 1);
        }
        // Finish this tile's reads and miss-slot writes before the next tile
        // reuses the block prefix scratch or the output pass reads the slots.
        asc_syncthreads();
        consumed += tile_misses;
    }
    }
    int remaining = hot - hit_count - total_misses;
    for (int p = tid; p < hot; p += kThreads) {
        int source_pos = p < remaining ? hot - 1 - total_misses - p :
            (p < remaining + total_misses ? hot - 1 - (p - remaining) : p - remaining - total_misses);
        lru[int64_t(state) * hot + p] = int16_t(temp[source_pos]);
    }
    // The successful union path leaves kError clear. Its final barrier below
    // also retires all LRU readers before any compaction reuses scratch.
    bool attention_invalid = false;
    for (int row = first; row < end; ++row) {
        if (tid == 0 && swap_counts) swap_counts[row] = minimum(topk, maximum(0, total_misses - (row - first) * topk));
        for (int i = tid; i < topk; i += kThreads) {
            int64_t index = int64_t(row) * topk + i;
            int physical = output[index];
            int host = resolved[index];
            if (physical < 0 && host >= 0) {
                int h = lookup(keys, host, hash_size);
                int slot = values[h] & kSlotMask;
                int block = table[int64_t(request) * d[49] + slot / int(d[48])];
                physical = block * int(d[48]) + slot % int(d[48]);
            }
            output[index] = physical;
            if (attention) {
                const int mapped = compact ? attention_row<kQualified>(d, physical) :
                    (physical >= 0 ? physical / int(d[48]) * int(d[51]) + physical % int(d[48]) : -1);
                attention[int64_t(row) * d[50] + i] = mapped;
                attention_invalid |= mapped < 0;
            }
        }
    }
    if (compact) {
        if (attention_invalid) asc_atomic_or(sums + kError, 1);
        asc_syncthreads();
        if (sums[kError]) {
            asc_syncthreads();
            for (int row = first; row < end; ++row) compact_attention<kQualified>(d, scratch, row, true);
        } else if (tid == 0 && counts) {
            for (int row = first; row < end; ++row) counts[int64_t(row) * d[52]] = topk;
        }
    }
    if (tid == 0 && stats) {
        asc_atomic_add(stats, int64_t(total_references - total_misses));
        asc_atomic_add(stats + 1, int64_t(total_misses));
    }
}
}  // namespace
