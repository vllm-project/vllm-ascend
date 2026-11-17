// SPDX-License-Identifier: Apache-2.0
// arch22 implementation of the Core HiSparse request-union ABI. Like the A5
// resolver, one request leader owns all state mutations for its query group.
// Use UB hash/tag storage and explicit scalar/MTE ordering, not A5 SIMT APIs.
//
// Migrated from the former standalone csrc/hisparse directory: the kernels are
// now compiled as standard aclnn ops (op_host + op_kernel). The entry points
// live in the op_kernel .cpp files; this header only carries the shared
// Resolver implementation and the AccumulateStats kernel.
#pragma once

#include "kernel_operator.h"
#include <cstdint>

struct HiSparseArch22Descriptor {
    uint64_t pointers[19];
    uint64_t geometry[25];
};

namespace {
using namespace AscendC;
constexpr int32_t kHit = int32_t(0x80000000U);
constexpr int32_t kMiss = 0x40000000;
constexpr int32_t kSlotMask = 0xFFFF;
constexpr int32_t kTile = 256;
constexpr int32_t kMaxResidentColumns = 1024;


template <typename T>
__aicore__ inline T Read(uint64_t address, int64_t index = 0)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
    return tensor.GetValue(index);
}

template <HardEvent event>
__aicore__ inline void Sync()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

template <typename T>
__aicore__ inline void Store(uint64_t address, int64_t offset, LocalTensor<T> data, uint32_t count)
{
    if (!address || !count) return;
    GlobalTensor<T> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
    Sync<HardEvent::S_MTE3>();
    DataCopyPad(destination[offset], data, DataCopyExtParams{1, count * uint32_t(sizeof(T)), 0, 0, 0});
    // Scratch is reused immediately, including for sub-32-byte count writes.
    Sync<HardEvent::MTE3_S>();
}

class Resolver {
public:
    __aicore__ inline Resolver(const HiSparseArch22Descriptor &descriptor, TPipe &pipe) : d(descriptor)
    {
        rows = d.geometry[0];
        topk = d.geometry[1];
        hot = d.geometry[2];
        hashSize = d.geometry[22];
        pipe.InitBuffer(keysBuffer, hashSize * sizeof(int32_t));
        pipe.InitBuffer(valuesBuffer, hashSize * sizeof(int32_t));
        pipe.InitBuffer(tagsBuffer, hot * sizeof(int32_t));
        pipe.InitBuffer(orderBuffer, hot * sizeof(int32_t));
        pipe.InitBuffer(ioBuffer, 2 * kTile * sizeof(int32_t));
        keys = keysBuffer.Get<int32_t>();
        values = valuesBuffer.Get<int32_t>();
        tags = tagsBuffer.Get<int32_t>();
        order = orderBuffer.Get<int32_t>();
        io = ioBuffer.Get<int32_t>();
    }

    __aicore__ inline int32_t Request(int32_t row) const
    {
        return d.pointers[1] ? Read<int32_t>(d.pointers[1], row) : row;
    }

    __aicore__ inline void Translate(int32_t token, int32_t &host, int32_t &device) const
    {
        host = token;
        device = -1;
        if (d.pointers[3]) {
            int32_t col = token >= 0 ? token / d.geometry[10] : -1;
            bool valid = token >= 0 && request >= 0 && request < d.geometry[7] && col < d.geometry[8];
            int32_t page = valid ? Read<int32_t>(d.pointers[3], int64_t(request) * d.geometry[9] + col) : 0;
            host = valid && page > 0 ? page * d.geometry[10] + token % d.geometry[10] : -1;
            if (d.pointers[4]) {
                col = token >= 0 ? token / d.geometry[14] : -1;
                valid = token >= 0 && state >= 0 && state < d.geometry[11] && col < d.geometry[12];
                page = valid ? Read<int32_t>(d.pointers[4], int64_t(state) * d.geometry[13] + col) : d.geometry[15];
                device = valid && page >= 0 && page != d.geometry[15]
                    ? page * d.geometry[14] + token % d.geometry[14] : -1;
            }
        }
        if (host < 0 || host >= d.geometry[3]) host = -1;
    }

    __aicore__ inline int32_t Lookup(int32_t host, bool insert = false, int32_t position = 0)
    {
        uint32_t mixed = uint32_t(host) * 2654435761U;
        int32_t h = uint64_t(mixed) * uint32_t(hashSize) >> 32;
        for (int32_t probe = 0; probe < hashSize; ++probe) {
            int32_t key = keys.GetValue(h);
            if (key == host) return h;
            if (key == -1) {
                if (!insert) return -1;
                keys.SetValue(h, host);
                values.SetValue(h, position);
                return h;
            }
            if (++h == hashSize) h = 0;
        }
        return -1;
    }

    __aicore__ inline int32_t Physical(int32_t slot) const
    {
        int32_t block = Read<int32_t>(d.pointers[5], int64_t(request) * d.geometry[17] + slot / d.geometry[16]);
        return block * d.geometry[16] + slot % d.geometry[16];
    }

    __aicore__ inline void Scalar(uint64_t address, int64_t index, int32_t value)
    {
        io.SetValue(0, value);
        Store(address, index, io, 1);
    }

    __aicore__ inline bool PublishResident(bool allowMixed = false)
    {
        // Reuse the union scratch before constructing any hash/LRU state.
        // Restrict geometry so all vector integer/fp32 conversions are exact;
        // the general union path retains the full ABI outside this fast path.
        const int32_t cols = d.geometry[8];
        if (!d.pointers[3] || !d.pointers[4] || request < 0 || request >= d.geometry[7] ||
            state >= d.geometry[11] || cols <= 0 || cols > kMaxResidentColumns || cols != d.geometry[12] ||
            cols > hashSize || hot < 3072 || topk < kTile || topk > 2048 || topk % kTile ||
            d.geometry[10] != 128 || d.geometry[14] != 128 || d.geometry[16] != 128) return false;
        GlobalTensor<int32_t> source, resident, logical;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[3]));
        resident.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[4]));
        logical.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[0]));
        DataCopyPad(keys, source[int64_t(request) * d.geometry[9]],
                    DataCopyExtParams{1, uint32_t(cols * 4), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        DataCopyPad(values, resident[int64_t(state) * d.geometry[13]],
                    DataCopyExtParams{1, uint32_t(cols * 4), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_S>();
        bool hasCold = false;
        auto coldPages = tags[kMaxResidentColumns].ReinterpretCast<float>();
        for (int32_t col = 0; col < cols; ++col) {
            int32_t hostPage = keys.GetValue(col), devicePage = values.GetValue(col);
            int64_t host = int64_t(hostPage) * 128;
            if (hostPage <= 0 || host >= int64_t(d.geometry[3])) host = -128;
            else if (host + 127 >= int64_t(d.geometry[3])) return false;
            int64_t device = devicePage >= 0 && devicePage != int64_t(d.geometry[15])
                ? int64_t(devicePage) * 128 : -128;
            const bool cold = host >= 0 && device < 0;
            if (cold && !allowMixed) return false;
            hasCold |= cold;
            if (allowMixed) coldPages.SetValue(col, cold ? 1.0f : 0.0f);
            int64_t mapped = device < 0 ? -128 : int64_t(devicePage) * d.geometry[19];
            if (device > INT32_MAX - 127 || mapped > INT32_MAX - 127) return false;
            if (d.geometry[23] && mapped >= 0) {
                if (mapped >= int64_t(d.geometry[23])) mapped = -128;
                else if (mapped + 127 >= int64_t(d.geometry[23])) return false;
            }
            keys.SetValue(col, int32_t(host));
            values.SetValue(col, int32_t(device));
            tags.SetValue(col, int32_t(mapped));
        }
        auto token = order;
        auto safe = order[kTile];
        auto column = order[2 * kTile];
        auto offset = order[3 * kTile];
        auto result = order[4 * kTile];
        auto floats = order[5 * kTile].ReinterpretCast<float>();
        auto lower = order[6 * kTile];
        auto upper = order[7 * kTile];
        auto zeros = order[8 * kTile];
        auto mask = order[9 * kTile].ReinterpretCast<uint8_t>();
        auto tokenMask = order[9 * kTile + kTile / 4].ReinterpretCast<uint8_t>();
        Sync<HardEvent::S_V>();
        Duplicate(lower, int32_t(-1), kTile);
        Duplicate(upper, cols * 128 - 1, kTile);
        Duplicate(zeros, int32_t(0), kTile);
        PipeBarrier<PIPE_V>();
        const bool compact = d.geometry[23] != 0;
        if (hasCold) {
            // Only the request leader enters this path. Prove every live row
            // before publishing anything: a later row may require Hot/LRU
            // mutation and the fallback then owns the entire request output.
            for (int32_t row = first; row < end; ++row) {
                for (int32_t base = 0; base < topk; base += kTile) {
                    DataCopyPad(token, logical[int64_t(row) * d.geometry[4] + base],
                                DataCopyExtParams{1, uint32_t(kTile * 4), 0, 0, 0},
                                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                    Sync<HardEvent::MTE2_V>();
                    Max(safe, token, zeros, kTile);
                    PipeBarrier<PIPE_V>();
                    Min(safe, safe, upper, kTile);
                    PipeBarrier<PIPE_V>();
                    ShiftRight(column.ReinterpretCast<uint32_t>(), safe.ReinterpretCast<uint32_t>(), uint32_t(7), kTile);
                    PipeBarrier<PIPE_V>();
                    ShiftLeft(column.ReinterpretCast<uint32_t>(), column.ReinterpretCast<uint32_t>(), uint32_t(2), kTile);
                    PipeBarrier<PIPE_V>();
                    Cast(floats, token, RoundMode::CAST_NONE, kTile);
                    PipeBarrier<PIPE_V>();
                    CompareScalar(mask, floats, 0.0f, CMPMODE::GE, kTile);
                    PipeBarrier<PIPE_V>();
                    CompareScalar(tokenMask, floats, float(cols * 128), CMPMODE::LT, kTile);
                    PipeBarrier<PIPE_V>();
                    And(tokenMask.ReinterpretCast<uint16_t>(), tokenMask.ReinterpretCast<uint16_t>(),
                        mask.ReinterpretCast<uint16_t>(), kTile / 16);
                    PipeBarrier<PIPE_V>();
                    Gather(floats, coldPages, column.ReinterpretCast<uint32_t>(), 0, uint32_t(kTile));
                    PipeBarrier<PIPE_V>();
                    CompareScalar(mask, floats, 0.0f, CMPMODE::GT, kTile);
                    PipeBarrier<PIPE_V>();
                    And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                        tokenMask.ReinterpretCast<uint16_t>(), kTile / 16);
                    PipeBarrier<PIPE_V>();
                    Sync<HardEvent::V_S>();
                    auto words = mask.ReinterpretCast<uint32_t>();
                    for (int32_t word = 0; word < kTile / 32; ++word) {
                        if (words.GetValue(word)) return false;
                    }
                    Sync<HardEvent::V_MTE2>();
                }
            }
        }
        for (int32_t row = first; row < end; ++row) {
            int32_t count = 0, compactCount = 0;
            for (int32_t base = 0; base < topk; base += kTile) {
                const int32_t n = topk - base < kTile ? topk - base : kTile;
                DataCopyPad(token, logical[int64_t(row) * d.geometry[4] + base],
                            DataCopyExtParams{1, uint32_t(n * 4), 0, 0, 0},
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                Sync<HardEvent::MTE2_V>();
                // Clamp the Gather address before accessing UB. Original
                // int32 tokens are used only for validity, never as offsets.
                Max(safe, token, zeros, n);
                PipeBarrier<PIPE_V>();
                Min(safe, safe, upper, n);
                PipeBarrier<PIPE_V>();
                ShiftRight(column.ReinterpretCast<uint32_t>(), safe.ReinterpretCast<uint32_t>(), uint32_t(7), n);
                PipeBarrier<PIPE_V>();
                ShiftLeft(offset.ReinterpretCast<uint32_t>(), column.ReinterpretCast<uint32_t>(), uint32_t(7), n);
                PipeBarrier<PIPE_V>();
                Sub(offset, safe, offset, n);
                PipeBarrier<PIPE_V>();
                ShiftLeft(column.ReinterpretCast<uint32_t>(), column.ReinterpretCast<uint32_t>(), uint32_t(2), n);
                PipeBarrier<PIPE_V>();
                // Casting the original token can round large int32 values,
                // but preserves both comparisons against [0, cols*128).
                Cast(floats, token, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, floats, 0.0f, CMPMODE::GE, n);
                PipeBarrier<PIPE_V>();
                CompareScalar(tokenMask, floats, float(cols * 128), CMPMODE::LT, n);
                PipeBarrier<PIPE_V>();
                And(tokenMask.ReinterpretCast<uint16_t>(), tokenMask.ReinterpretCast<uint16_t>(),
                    mask.ReinterpretCast<uint16_t>(), n / 16);
                PipeBarrier<PIPE_V>();
                for (int32_t kind = 0; kind < 3; ++kind) {
                    if (kind == 2 && !d.pointers[9]) continue;
                    auto table = kind == 0 ? keys : (kind == 1 ? values : tags);
                    Gather(result, table, column.ReinterpretCast<uint32_t>(), 0, uint32_t(n));
                    PipeBarrier<PIPE_V>();
                    Add(result, result, offset, n);
                    PipeBarrier<PIPE_V>();
                    Max(result, result, lower, n);
                    PipeBarrier<PIPE_V>();
                    Select(result.ReinterpretCast<float>(), tokenMask, result.ReinterpretCast<float>(),
                           lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
                    PipeBarrier<PIPE_V>();
                    int32_t kept = n;
                    bool packed = false;
                    if ((kind == 1 && !compact) || (kind == 2 && compact)) {
                        // Only the sign is tested in fp32. GatherMask copies
                        // the original integer payload without rounding IDs.
                        Cast(floats, result, RoundMode::CAST_NONE, n);
                        PipeBarrier<PIPE_V>();
                        CompareScalar(mask, floats, 0.0f, CMPMODE::GE, n);
                        PipeBarrier<PIPE_V>();
                        Sync<HardEvent::V_S>();
                        auto words = mask.ReinterpretCast<uint32_t>();
                        for (int32_t word = 0; word < n / 32; ++word) {
                            packed |= words.GetValue(word) != UINT32_MAX;
                        }
                        if (packed) {
                            GatherMaskParams params;
                            params.repeatTimes = 1;
                            params.src0BlockStride = 1;
                            params.src0RepeatStride = 8;
                            params.src1RepeatStride = 8;
                            uint64_t reserved = 0;
                            Sync<HardEvent::S_V>();
                            GatherMask(io.ReinterpretCast<uint32_t>(), result.ReinterpretCast<uint32_t>(),
                                       words, true, uint32_t(n), params, reserved);
                            PipeBarrier<PIPE_V>();
                            Sync<HardEvent::V_S>();
                            kept = int32_t(reserved);
                        }
                    }
                    Sync<HardEvent::V_S>();
                    if (kind == 0) {
                        Store(d.pointers[10], int64_t(row) * topk + base, result, n);
                    } else if (kind == 1) {
                        if (!compact) count += kept;
                        Store(d.pointers[8], int64_t(row) * topk + base, result, n);
                    } else if (compact) {
                        Store(d.pointers[9], int64_t(row) * d.geometry[18] + compactCount,
                              packed ? io : result, kept);
                        compactCount += kept;
                    } else {
                        Store(d.pointers[9], int64_t(row) * d.geometry[18] + base, result, n);
                    }
                    Sync<HardEvent::S_V>();
                }
                if (d.pointers[15]) {
                    Sync<HardEvent::V_S>();
                    Store(d.pointers[15], (int64_t(row) * topk + base) * (d.geometry[24] ? 1 : 4),
                          zeros.ReinterpretCast<uint8_t>(), n * (d.geometry[24] ? 1 : 4));
                }
                Sync<HardEvent::V_MTE2>();
            }
            Sync<HardEvent::V_S>();
            if (compact) {
                for (int32_t base = compactCount; base < topk; base += kTile) {
                    Store(d.pointers[9], int64_t(row) * d.geometry[18] + base, lower,
                          topk - base < kTile ? topk - base : kTile);
                }
                count = compactCount;
            }
            Scalar(d.pointers[11], int64_t(row) * d.geometry[20], count);
            Scalar(d.pointers[14], row, 0);
            Store(d.pointers[17], int64_t(row) * 8, zeros, 8);
        }
        return true;
    }

    __aicore__ inline void Process()
    {
        first = GetBlockIdx();
        int32_t validRows = rows;
        if (d.pointers[18]) {
            validRows = Read<int32_t>(d.pointers[18]);
            validRows = validRows < 0 ? 0 : (validRows > rows ? rows : validRows);
        }
        request = first < validRows ? Request(first) : -1;
        end = first + 1;
        state = request;
        if (d.pointers[2]) {
            state = request >= 0 && request < d.geometry[5] ? Read<int32_t>(d.pointers[2], request) : -1;
        }
        bool valid = state >= 0 && state < d.geometry[6] && first < validRows;
        // Every row uses the same immutable request-table proof. A successful
        // proof only publishes this row and never mutates Hot/LRU state.
        if (valid && PublishResident()) return;
        // Mixed requests retain a single writer for their union and all rows.
        if (first < validRows && first && d.pointers[1] && Request(first - 1) == request) return;
        while (d.pointers[1] && end < validRows && Request(end) == request) ++end;
        if (!valid) {
            Publish(false);
            return;
        }
        if (PublishResident(true)) return;
        for (int32_t h = 0; h < hashSize; ++h) keys.SetValue(h, -1);
        // Construct the entire cold union before inspecting or evicting Hot.
        for (int32_t row = first; row < end; ++row) {
            for (int32_t col = 0; col < topk; ++col) {
                int32_t host, device;
                Translate(Read<int32_t>(d.pointers[0], int64_t(row) * d.geometry[4] + col), host, device);
                if (device < 0 && host >= 0) {
                    ++references;
                    if (Lookup(host, true, (row - first) * topk + col) < 0) error = -1;
                }
            }
        }
        if (!references || error) {
            Publish(true);
            return;
        }
        GlobalTensor<int32_t> globalTags;
        globalTags.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[6]));
        DataCopyPad(tags, globalTags[int64_t(state) * hot],
                    DataCopyExtParams{1, uint32_t(hot * sizeof(int32_t)), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_S>();
        for (int32_t slot = 0; slot < hot; ++slot) {
            int32_t host = tags.GetValue(slot);
            int32_t h = host >= 0 ? Lookup(host) : -1;
            if (h >= 0) {
                int32_t value = kHit | slot;
                if (values.GetValue(h) > value) values.SetValue(h, value);
            }
        }
        // LRU is oldest-first. Preserve hit order and reverse the evictable
        // partition exactly as the shared A5/Core algorithm does.
        int32_t hits = 0, evictable = 0;
        for (int32_t p = 0; p < hot; ++p) {
            int32_t slot = Read<int16_t>(d.pointers[7], int64_t(state) * hot + p);
            if (slot < 0 || slot >= hot) { error = -1; break; }
            int32_t host = tags.GetValue(slot);
            bool hit = host >= 0 && Lookup(host) >= 0;
            order.SetValue(hit ? hits++ : hot - 1 - evictable++, slot);
        }
        for (int32_t h = 0; h < hashSize; ++h) {
            if (keys.GetValue(h) >= 0 && values.GetValue(h) >= 0) ++misses;
        }
        if (hits + misses > d.geometry[21] || hits + misses > hot) error = hits + misses;
        if (error) { Publish(true); return; }

        int32_t rank = 0, buffered = 0;
        for (int32_t row = first; row < end; ++row) {
            for (int32_t col = 0; col < topk; ++col) {
                int32_t host, device;
                Translate(Read<int32_t>(d.pointers[0], int64_t(row) * d.geometry[4] + col), host, device);
                int32_t h = device < 0 && host >= 0 ? Lookup(host) : -1;
                if (h >= 0 && values.GetValue(h) == (row - first) * topk + col) {
                    int32_t slot = order.GetValue(hot - 1 - rank);
                    values.SetValue(h, kMiss | slot);
                    tags.SetValue(slot, host);
                    io.SetValue(buffered, host);
                    io.SetValue(kTile + buffered, Physical(slot));
                    ++rank;
                    if (++buffered == kTile || rank == misses) {
                        Store(d.pointers[12], int64_t(first) * topk + rank - buffered, io, buffered);
                        Store(d.pointers[13], int64_t(first) * topk + rank - buffered, io[kTile], buffered);
                        buffered = 0;
                    }
                }
            }
        }
        Store(d.pointers[6], int64_t(state) * hot, tags, hot);
        int32_t remaining = hot - hits - misses;
        auto lruOut = io.ReinterpretCast<int16_t>();
        for (int32_t base = 0; base < hot; base += kTile) {
            int32_t count = hot - base < kTile ? hot - base : kTile;
            for (int32_t i = 0; i < count; ++i) {
                int32_t p = base + i;
                int32_t source = p < remaining ? hot - 1 - misses - p :
                    (p < remaining + misses ? hot - 1 - (p - remaining) : p - remaining - misses);
                lruOut.SetValue(i, int16_t(order.GetValue(source)));
            }
            Store(d.pointers[7], int64_t(state) * hot + base, lruOut, count);
        }
        Publish(true);
    }

    __aicore__ inline void Publish(bool valid)
    {
        // Recompute translations; never read another program's output or
        // scalar-cache data that has just been updated by MTE3.
        const bool compact = d.geometry[23] != 0;
        for (int32_t row = first; row < end; ++row) {
            int32_t count = 0, compactCount = 0;
            for (int32_t base = 0; base < topk; base += kTile) {
                int32_t extent = topk - base < kTile ? topk - base : kTile;
                for (int32_t i = 0; i < extent; ++i) {
                    int32_t host = -1, device = -1;
                    if (valid) Translate(Read<int32_t>(d.pointers[0], int64_t(row) * d.geometry[4] + base + i), host, device);
                    io.SetValue(kTile + i, host);
                    if (valid && !error && device < 0 && host >= 0) {
                        int32_t h = Lookup(host);
                        device = h >= 0 ? Physical(values.GetValue(h) & kSlotMask) : -1;
                    }
                    if (error) device = -1;
                    io.SetValue(i, device);
                    count += device >= 0;
                }
                Store(d.pointers[8], int64_t(row) * topk + base, io, extent);
                Store(d.pointers[10], int64_t(row) * topk + base, io[kTile], extent);
                if (d.pointers[9]) {
                    int32_t kept = 0;
                    for (int32_t i = 0; i < extent; ++i) {
                        int32_t device = io.GetValue(i);
                        int64_t mapped = device < 0 ? -1 :
                            int64_t(device / d.geometry[16]) * d.geometry[19] + device % d.geometry[16];
                        if (!compact) io.SetValue(i, mapped);
                        else if (mapped >= 0 && mapped < int64_t(d.geometry[23])) io.SetValue(kept++, mapped);
                    }
                    if (compact) {
                        // Keep order and duplicates; only Attention is compacted.
                        // Virtual indices and the refill plan retain their ABI.
                        Store(d.pointers[9], int64_t(row) * d.geometry[18] + compactCount, io, kept);
                        compactCount += kept;
                    } else {
                        Store(d.pointers[9], int64_t(row) * d.geometry[18] + base, io, extent);
                    }
                }
                // The Core compact plan is the primary interface. Optional
                // masks identify only the first occurrence of each miss.
                if (d.pointers[15]) {
                    for (int32_t i = 0; i < extent; ++i) {
                        int32_t host = io.GetValue(kTile + i);
                        int32_t h = valid && !error && host >= 0 ? Lookup(host) : -1;
                        bool miss = h >= 0 && (values.GetValue(h) & kMiss) && values.GetValue(h) >= 0;
                        // Repeated references must not submit duplicate copies.
                        if (miss) {
                            for (int32_t earlier = first; earlier <= row && miss; ++earlier) {
                                int32_t limit = earlier == row ? base + i : topk;
                                for (int32_t col = 0; col < limit; ++col) {
                                    int32_t oldHost, oldDevice;
                                    Translate(Read<int32_t>(d.pointers[0], int64_t(earlier) * d.geometry[4] + col), oldHost, oldDevice);
                                    if (oldDevice < 0 && oldHost == host) { miss = false; break; }
                                }
                            }
                        }
                        if (d.geometry[24]) io.ReinterpretCast<uint8_t>().SetValue(i, uint8_t(miss));
                        else io.SetValue(i, int32_t(miss));
                    }
                    if (d.geometry[24]) Store(d.pointers[15], int64_t(row) * topk + base, io.ReinterpretCast<uint8_t>(), extent);
                    else Store(d.pointers[15], int64_t(row) * topk + base, io, extent);
                }
            }
            if (compact) {
                for (int32_t i = 0; i < kTile; ++i) io.SetValue(i, -1);
                for (int32_t base = compactCount; base < topk; base += kTile) {
                    int32_t extent = topk - base < kTile ? topk - base : kTile;
                    Store(d.pointers[9], int64_t(row) * d.geometry[18] + base, io, extent);
                }
                count = compactCount;
            }
            Scalar(d.pointers[11], int64_t(row) * d.geometry[20], count);
            int32_t swaps = misses - (row - first) * topk;
            swaps = swaps < 0 ? 0 : (swaps > topk ? topk : swaps);
            Scalar(d.pointers[14], row, error ? 0 : swaps);
            for (int32_t i = 0; i < 8; ++i) io.SetValue(i, 0);
            if (row == first) {
                io.SetValue(0, error);
                io.SetValue(1, error ? 0 : references - misses);
                io.SetValue(2, error ? 0 : misses);
            }
            Store(d.pointers[17], int64_t(row) * 8, io, 8);
        }
    }

private:
    const HiSparseArch22Descriptor &d;
    int32_t rows, topk, hot, hashSize;
    int32_t first = 0, end = 0, request = -1, state = -1;
    int32_t references = 0, misses = 0, error = 0;
    TBuf<TPosition::VECCALC> keysBuffer, valuesBuffer, tagsBuffer, orderBuffer, ioBuffer;
    LocalTensor<int32_t> keys, values, tags, order, io;
};

}  // namespace

__global__ __aicore__ void AccumulateStats(uint64_t errors, uint64_t stats, int32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 32);
    auto result = buffer.Get<int64_t>();
    int64_t hits = Read<int64_t>(stats), misses = Read<int64_t>(stats, 1);
    for (int32_t row = 0; row < rows; ++row) {
        hits += Read<int32_t>(errors, int64_t(row) * 8 + 1);
        misses += Read<int32_t>(errors, int64_t(row) * 8 + 2);
    }
    result.SetValue(0, hits);
    result.SetValue(1, misses);
    Store(stats, 0, result, 2);
}
