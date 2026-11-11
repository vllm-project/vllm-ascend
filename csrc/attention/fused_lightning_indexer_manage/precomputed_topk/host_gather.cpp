// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

struct GatherParticipant {
    uint64_t* control;
    uint8_t* packed;
    int rank;
    int world_size;
    uint64_t epoch;
};

struct SelectiveHostGather {
    const uint64_t* sources;
    const uint64_t* destinations;
    const uint32_t* lengths;
    uint8_t* packed_k;
    uint8_t* packed_v;
    std::size_t entries;
    std::size_t k_bytes;
    std::size_t v_bytes;
    bool to_device;
    GatherParticipant* participant;
};

static void copy_rows(const SelectiveHostGather& plan) noexcept {
    for (std::size_t i = 0; i < plan.entries; ++i) {
        if (plan.lengths[i] != 0) {
            auto* packed = plan.packed_k + i * plan.k_bytes;
            if (plan.to_device) {
                std::memcpy(packed, reinterpret_cast<const void*>(plan.sources[i]), plan.lengths[i]);
            } else {
                std::memcpy(reinterpret_cast<void*>(plan.destinations[i]), packed, plan.lengths[i]);
            }
        }
        const std::size_t v_index = plan.entries + i;
        if (plan.lengths[v_index] != 0) {
            auto* packed = plan.packed_v + i * plan.v_bytes;
            if (plan.to_device) {
                std::memcpy(packed, reinterpret_cast<const void*>(plan.sources[v_index]), plan.lengths[v_index]);
            } else {
                std::memcpy(reinterpret_cast<void*>(plan.destinations[v_index]), packed, plan.lengths[v_index]);
            }
        }
    }
}

static void wait_epoch(const uint64_t* value, uint64_t epoch) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (__atomic_load_n(value, __ATOMIC_ACQUIRE) < epoch) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fputs("Selective Host gather peer synchronization timed out\n", stderr);
            std::abort();
        }
        std::this_thread::yield();
    }
}

static void gather_missing_rows(void* data) noexcept {
    const auto& plan = *static_cast<const SelectiveHostGather*>(data);
    if (!plan.to_device || plan.participant == nullptr) {
        copy_rows(plan);
        return;
    }
    auto& participant = *plan.participant;
    auto* control = participant.control;
    const uint64_t epoch = ++participant.epoch;
    uint64_t fingerprint = 14695981039346656037ULL;
    for (std::size_t i = 0; i < 2 * plan.entries; ++i) {
        if (plan.lengths[i] != 0) {
            fingerprint = (fingerprint ^ plan.sources[i]) * 1099511628211ULL;
            fingerprint = (fingerprint ^ i) * 1099511628211ULL;
            fingerprint = (fingerprint ^ plan.lengths[i]) * 1099511628211ULL;
        }
    }
    auto selected = plan;
    selected.packed_k = participant.packed;
    selected.packed_v = participant.packed + plan.entries * plan.k_bytes;
    if (participant.rank == 0) {
        for (int rank = 1; rank < participant.world_size; ++rank) {
            wait_epoch(control + 6 + rank, epoch - 1);
        }
        copy_rows(selected);
        control[1] = fingerprint;
        control[2] = plan.entries;
        control[3] = plan.k_bytes;
        control[4] = plan.v_bytes;
        __atomic_store_n(control, epoch, __ATOMIC_RELEASE);
    } else {
        wait_epoch(control, epoch);
        if (__atomic_load_n(control, __ATOMIC_ACQUIRE) != epoch || control[1] != fingerprint ||
            control[2] != plan.entries || control[3] != plan.k_bytes || control[4] != plan.v_bytes) {
            std::fputs("Selective Host gather TP plans disagree\n", stderr);
            std::abort();
        }
    }
}

static void gather_dma_complete(void* data) noexcept {
    auto& participant = *static_cast<GatherParticipant*>(data);
    auto* control = participant.control;
    const uint64_t epoch = participant.epoch;
    // Enqueued after H2D, so no rank can overwrite shared rows while a peer's SDMA reads them.
    __atomic_store_n(control + 6 + participant.rank, epoch, __ATOMIC_RELEASE);
}

extern "C" aclError selective_enqueue_host_gather(void* stream, SelectiveHostGather* plan) {
    return aclrtLaunchHostFunc(stream, gather_missing_rows, plan);
}

extern "C" aclError selective_enqueue_host_gather_done(void* stream, GatherParticipant* participant) {
    return aclrtLaunchHostFunc(stream, gather_dma_complete, participant);
}
