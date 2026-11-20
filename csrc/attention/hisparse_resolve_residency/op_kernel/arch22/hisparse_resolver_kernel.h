// SPDX-License-Identifier: Apache-2.0
// arch22 implementation of the Core HiSparse request-union ABI. Like the A5
// resolver, one request leader owns all state mutations for its query group.
// Use UB hash/tag storage and explicit scalar/MTE ordering, not A5 SIMT APIs.
//
// Migrated from the former standalone csrc/hisparse directory: the kernels are
// now compiled as standard aclnn ops (op_host + op_kernel). The entry points
// live in the op_kernel .cpp files; this header only carries the shared
// Resolver implementation. The adapter accumulates per-request counts
// using INT64 tensor operations to preserve carry between counter words.
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

    // Page tables repeat one entry across geometry[10]/[14] consecutive
    // tokens and the block table across geometry[16] slots, so the last
    // valid entry is cached to drop the dependent GM scalar loads that
    // dominate the fallback path. No in-kernel writer mutates these tables,
    // and request/state stay fixed for the whole group, so the cache is
    // stable between resets at construction time.
    __aicore__ inline void Translate(int32_t token, int32_t &host, int32_t &device)
    {
        host = token;
        device = -1;
        if (d.pointers[3]) {
            const uint64_t reqTable = d.pointers[3];
            const uint64_t stateTable = d.pointers[4];
            const int64_t reqStride = d.geometry[9];
            const int64_t stateStride = d.geometry[13];
            const int32_t tokenSpan = d.geometry[10];
            const int32_t residentSpan = d.geometry[14];
            int32_t col = token >= 0 ? token / tokenSpan : -1;
            bool valid = token >= 0 && request >= 0 && request < d.geometry[7] && col < d.geometry[8];
            if (valid) {
                int32_t page;
                if (col == reqCol) {
                    page = reqPage;
                } else {
                    page = Read<int32_t>(reqTable, int64_t(request) * reqStride + col);
                    reqCol = col;
                    reqPage = page;
                }
                host = page > 0 ? page * tokenSpan + token % tokenSpan : -1;
            } else {
                host = -1;
            }
            if (stateTable) {
                col = token >= 0 ? token / residentSpan : -1;
                valid = token >= 0 && state >= 0 && state < d.geometry[11] && col < d.geometry[12];
                if (valid) {
                    int32_t resident;
                    if (col == stCol) {
                        resident = stPage;
                    } else {
                        resident = Read<int32_t>(stateTable, int64_t(state) * stateStride + col);
                        stCol = col;
                        stPage = resident;
                    }
                    device = resident >= 0 && resident != d.geometry[15]
                        ? resident * residentSpan + token % residentSpan : -1;
                }
            }
        }
        if (host < 0 || host >= d.geometry[3]) host = -1;
    }

    __aicore__ inline int32_t Lookup(int32_t host, bool insert = false, int32_t position = 0)
    {
        const int32_t size = hashSize;
        uint32_t mixed = uint32_t(host) * 2654435761U;
        int32_t h = uint64_t(mixed) * uint32_t(size) >> 32;
        for (int32_t probe = 0; probe < size; ++probe) {
            int32_t key = keys.GetValue(h);
            if (key == host) return h;
            if (key == -1) {
                if (!insert) return -1;
                keys.SetValue(h, host);
                values.SetValue(h, position);
                return h;
            }
            if (++h == size) h = 0;
        }
        return -1;
    }

    __aicore__ inline int32_t Physical(int32_t slot)
    {
        const int32_t blockSpan = d.geometry[16];
        const int32_t col = slot / blockSpan;
        int32_t block;
        if (col == blkCol) {
            block = blkPage;
        } else {
            block = Read<int32_t>(d.pointers[5], int64_t(request) * d.geometry[17] + col);
            blkCol = col;
            blkPage = block;
        }
        return block * blockSpan + slot % blockSpan;
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

    // Scratch layout during the union only. order[] is free here: the LRU
    // permutation, the Hot tag row and the emit tiles are all built after this
    // pass, so the whole buffer is available.
    static constexpr int32_t kVecBase = 0;                       // 14 tiles
    static constexpr int32_t kMaskBase = 14 * kTile;             // 3 bit masks
    static constexpr int32_t kPageBase = 16 * kTile;             // 2 x 512 pages
    static constexpr int32_t kVecPageColumns = 512;

    __aicore__ inline void LoadUnionPages()
    {
        const int32_t cols = d.geometry[8];
        GlobalTensor<int32_t> source, resident;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[3]));
        resident.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[4]));
        auto srcPages = order[kPageBase];
        auto resPages = order[kPageBase + kVecPageColumns];
        DataCopyPad(srcPages, source[int64_t(request) * d.geometry[9]],
                    DataCopyExtParams{1, uint32_t(cols * 4), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        DataCopyPad(resPages, resident[int64_t(state) * d.geometry[13]],
                    DataCopyExtParams{1, uint32_t(cols * 4), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>();
    }

    __aicore__ inline void UnionTile(int32_t row, int32_t base, int32_t n)
    {
        const int32_t cols = d.geometry[8];
        auto token = order[kVecBase + 0 * kTile];
        auto safe = order[kVecBase + 1 * kTile];
        auto column = order[kVecBase + 2 * kTile];
        auto offset = order[kVecBase + 3 * kTile];
        auto host = order[kVecBase + 4 * kTile];
        auto device = order[kVecBase + 5 * kTile];
        auto pos = order[kVecBase + 6 * kTile];
        auto stage = order[kVecBase + 7 * kTile];
        auto keptPos = order[kVecBase + 8 * kTile];
        auto keptHost = order[kVecBase + 9 * kTile];
        auto floats = order[kVecBase + 10 * kTile].ReinterpretCast<float>();
        auto lower = order[kVecBase + 11 * kTile];
        auto upper = order[kVecBase + 12 * kTile];
        auto zeros = order[kVecBase + 13 * kTile];
        auto vMask = order[kMaskBase].ReinterpretCast<uint8_t>();
        auto vHostMask = order[kMaskBase + kTile / 4].ReinterpretCast<uint8_t>();
        auto vPageMask = order[kMaskBase + 2 * kTile / 4].ReinterpretCast<uint8_t>();
        auto srcPages = order[kPageBase];
        auto resPages = order[kPageBase + kVecPageColumns];

        GlobalTensor<int32_t> logical;
        logical.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(d.pointers[0]));
        DataCopyPad(token, logical[int64_t(row) * d.geometry[4] + base],
                    DataCopyExtParams{1, uint32_t(n * 4), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>();
        Duplicate(lower, int32_t(-1), n);
        Duplicate(upper, cols * 128 - 1, n);
        Duplicate(zeros, int32_t(0), n);
        PipeBarrier<PIPE_V>();
        // Clamp, then derive column/offset with shifts only: no division.
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
        // token validity: 0 <= token < cols * 128.
        Cast(floats, token, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vMask, floats, 0.0f, CMPMODE::GE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vHostMask, floats, float(cols * 128), CMPMODE::LT, n);
        PipeBarrier<PIPE_V>();
        And(vHostMask.ReinterpretCast<uint16_t>(), vHostMask.ReinterpretCast<uint16_t>(),
            vMask.ReinterpretCast<uint16_t>(), n / 16);
        PipeBarrier<PIPE_V>();
        // host = page * 128 + offset, gated by page > 0, token range and hostRows.
        Gather(host, srcPages, column.ReinterpretCast<uint32_t>(), 0, uint32_t(n));
        PipeBarrier<PIPE_V>();
        Cast(floats, host, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vMask, floats, 0.0f, CMPMODE::GT, n);
        PipeBarrier<PIPE_V>();
        ShiftLeft(host.ReinterpretCast<uint32_t>(), host.ReinterpretCast<uint32_t>(), uint32_t(7), n);
        PipeBarrier<PIPE_V>();
        Add(host, host, offset, n);
        PipeBarrier<PIPE_V>();
        Select(host.ReinterpretCast<float>(), vMask, host.ReinterpretCast<float>(),
               lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        Select(host.ReinterpretCast<float>(), vHostMask, host.ReinterpretCast<float>(),
               lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        Cast(floats, host, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vPageMask, floats, float(int32_t(d.geometry[3])), CMPMODE::LT, n);
        PipeBarrier<PIPE_V>();
        Select(host.ReinterpretCast<float>(), vPageMask, host.ReinterpretCast<float>(),
               lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        // device = resident * 128 + offset, gated by resident >= 0, non-null
        // block and the same token range.
        Gather(device, resPages, column.ReinterpretCast<uint32_t>(), 0, uint32_t(n));
        PipeBarrier<PIPE_V>();
        Cast(floats, device, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vMask, floats, 0.0f, CMPMODE::GE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vPageMask, floats, float(int32_t(d.geometry[15])), CMPMODE::NE, n);
        PipeBarrier<PIPE_V>();
        And(vMask.ReinterpretCast<uint16_t>(), vMask.ReinterpretCast<uint16_t>(),
            vPageMask.ReinterpretCast<uint16_t>(), n / 16);
        PipeBarrier<PIPE_V>();
        ShiftLeft(device.ReinterpretCast<uint32_t>(), device.ReinterpretCast<uint32_t>(), uint32_t(7), n);
        PipeBarrier<PIPE_V>();
        Add(device, device, offset, n);
        PipeBarrier<PIPE_V>();
        Select(device.ReinterpretCast<float>(), vMask, device.ReinterpretCast<float>(),
               lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        Select(device.ReinterpretCast<float>(), vHostMask, device.ReinterpretCast<float>(),
               lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        // cold = device < 0 && host >= 0.
        Cast(floats, device, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vMask, floats, 0.0f, CMPMODE::LT, n);
        PipeBarrier<PIPE_V>();
        Cast(floats, host, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(vPageMask, floats, 0.0f, CMPMODE::GE, n);
        PipeBarrier<PIPE_V>();
        And(vMask.ReinterpretCast<uint16_t>(), vMask.ReinterpretCast<uint16_t>(),
            vPageMask.ReinterpretCast<uint16_t>(), n / 16);
        PipeBarrier<PIPE_V>();
        // Staging reuses the device output buffer: cold host or -1.
        Select(stage.ReinterpretCast<float>(), vMask, host.ReinterpretCast<float>(),
               lower.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
        Store(d.pointers[8], int64_t(row) * topk + base, stage, n);
        // Compact the cold positions/hosts and record every first reference.
        CreateVecIndex(pos, int32_t(int64_t(row - first) * topk + base), uint32_t(n));
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
        GatherMaskParams params;
        params.repeatTimes = 1;
        params.src0BlockStride = 1;
        params.src0RepeatStride = 8;
        params.src1RepeatStride = 8;
        uint64_t posCount = 0;
        uint64_t hostCount = 0;
        GatherMask(keptPos.ReinterpretCast<uint32_t>(), pos.ReinterpretCast<uint32_t>(),
                   vMask.ReinterpretCast<uint32_t>(), true, uint32_t(n), params, posCount);
        PipeBarrier<PIPE_V>();
        GatherMask(keptHost.ReinterpretCast<uint32_t>(), host.ReinterpretCast<uint32_t>(),
                   vMask.ReinterpretCast<uint32_t>(), true, uint32_t(n), params, hostCount);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>();
        if (posCount != hostCount) {
            error = -1;
            return;
        }
        for (int32_t i = 0; i < int32_t(hostCount); ++i) {
            ++references;
            if (Lookup(int32_t(keptHost.GetValue(i)), true, int32_t(keptPos.GetValue(i))) < 0) error = -1;
        }
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
        // One vector Duplicate replaces the scalar clear; the hash must be
        // ready before the first Lookup insert.
        Duplicate(keys, int32_t(-1), hashSize);
        Sync<HardEvent::V_S>();
        // Construct the entire cold union before inspecting or evicting Hot.
        // Descriptor fields are hoisted into locals: the compiler cannot
        // prove the opaque UB tensor ops do not alias the descriptor, so
        // every d.* access would otherwise re-load UB inside the hot loop.
        const uint64_t logicalBase = d.pointers[0];
        const int64_t logicalStride = d.geometry[4];
        // The output buffer is not written until Publish, so it carries the cold
        // host id for the later passes. A resident token stores -1 and never
        // enters the union.
        const uint64_t stagingBase = d.pointers[8];
        // The union scratch (14 tiles, 3 masks, two 512-entry page rows) is
        // carved out of order[], so a Hot row smaller than that keeps the
        // scalar path instead of writing past the buffer.
        // A partial tile would leave mask bits above n from the previous
        // CompareScalar: the packed mask is written one 64-lane repeat at a
        // time and the tail is only cleared for full repeats. SFA top-k is a
        // multiple of 128, so requiring a multiple of 64 keeps the vector path
        // for every production shape and sends the rest to the scalar union.
        const bool vecUnion = d.pointers[3] && d.pointers[4] && d.geometry[8] > 0 &&
            d.geometry[8] <= kVecPageColumns && d.geometry[8] == d.geometry[12] &&
            d.geometry[10] == 128 && d.geometry[14] == 128 &&
            hot >= kPageBase + 2 * kVecPageColumns && topk % 64 == 0;
        if (vecUnion) LoadUnionPages();
        int32_t position = 0;
        for (int32_t row = first; row < end; ++row) {
            for (int32_t base = 0; base < topk; base += kTile) {
                const int32_t n = topk - base < kTile ? topk - base : kTile;
                if (vecUnion) {
                    UnionTile(row, base, n);
                    position += n;
                    continue;
                }
                for (int32_t i = 0; i < n; ++i) {
                    int32_t host, device;
                    Translate(Read<int32_t>(logicalBase, int64_t(row) * logicalStride + base + i), host, device);
                    const bool cold = device < 0 && host >= 0;
                    if (cold) {
                        ++references;
                        if (Lookup(host, true, position) < 0) error = -1;
                    }
                    io.SetValue(i, cold ? host : -1);
                    ++position;
                }
                Store(stagingBase, int64_t(row) * topk + base, io, n);
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
        // LRU is oldest-first and always a permutation of the hot slots (the
        // resolver itself rebuilds it every launch, and Core keeps the same
        // invariant). One pass over the LRU list therefore both marks the
        // smallest hit slot per union entry and builds the reversed
        // evictable/hit partition, halving the hash probes per slot.
        // The permutation only holds slot ids (< hot), so it lives in the low
        // half of order[] as int16 and leaves the upper half for the vector
        // miss scan below.
        auto perm = order.ReinterpretCast<int16_t>();
        int32_t hits = 0, evictable = 0;
        const uint64_t lruBase = d.pointers[7];
        for (int32_t p = 0; p < hot; ++p) {
            int32_t slot = Read<int16_t>(lruBase, int64_t(state) * hot + p);
            if (slot < 0 || slot >= hot) { error = -1; break; }
            int32_t host = tags.GetValue(slot);
            bool hit = false;
            if (host >= 0) {
                int32_t h = Lookup(host);
                if (h >= 0) {
                    hit = true;
                    int32_t value = kHit | slot;
                    if (values.GetValue(h) > value) values.SetValue(h, value);
                }
            }
            perm.SetValue(hit ? hits++ : hot - 1 - evictable++, int16_t(slot));
        }
        // Count distinct misses with vector compares; only the tail below a
        // full kTile repeat stays scalar. The scratch sits above the int16
        // permutation, which is the only live part of order[] here.
        const bool vecScan = hot >= 27 * kTile;
        const int32_t scanned = vecScan ? hashSize / kTile * kTile : 0;
        if (vecScan) {
            auto scanFloats = order[24 * kTile].ReinterpretCast<float>();
            auto scanKeys = order[25 * kTile];
            auto scanMask = order[26 * kTile].ReinterpretCast<uint8_t>();
            auto scanValueMask = order[26 * kTile + kTile / 4].ReinterpretCast<uint8_t>();
            for (int32_t base = 0; base < scanned; base += kTile) {
                Cast(scanFloats, keys[base], RoundMode::CAST_NONE, kTile);
                PipeBarrier<PIPE_V>();
                CompareScalar(scanMask, scanFloats, 0.0f, CMPMODE::GE, kTile);
                PipeBarrier<PIPE_V>();
                Cast(scanFloats, values[base], RoundMode::CAST_NONE, kTile);
                PipeBarrier<PIPE_V>();
                CompareScalar(scanValueMask, scanFloats, 0.0f, CMPMODE::GE, kTile);
                PipeBarrier<PIPE_V>();
                And(scanMask.ReinterpretCast<uint16_t>(), scanMask.ReinterpretCast<uint16_t>(),
                    scanValueMask.ReinterpretCast<uint16_t>(), kTile / 16);
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_S>();
                GatherMaskParams params;
                params.repeatTimes = 1;
                params.src0BlockStride = 1;
                params.src0RepeatStride = 8;
                params.src1RepeatStride = 8;
                uint64_t kept = 0;
                GatherMask(scanKeys, keys[base], scanMask.ReinterpretCast<uint32_t>(), true,
                           uint32_t(kTile), params, kept);
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_S>();
                misses += int32_t(kept);
            }
        }
        for (int32_t h = scanned; h < hashSize; ++h) {
            if (keys.GetValue(h) >= 0 && values.GetValue(h) >= 0) ++misses;
        }
        if (hits + misses > d.geometry[21] || hits + misses > hot) error = hits + misses;
        if (error) { Publish(true); return; }

        // The union pass already resolved every cold token, and a slot is
        // assigned on the first reference, so this pass emits the final
        // output/attention directly instead of leaving it to a third pass.
        // The miss mask keeps the old path: it needs the finished hash.
        const bool compact = d.geometry[23] != 0;
        const uint64_t hostOut = d.pointers[10];
        const uint64_t attentionOut = d.pointers[9];
        const int64_t attentionStride = d.geometry[18];
        const int32_t blockSpan = d.geometry[16];
        const int64_t attentionGap = d.geometry[19];
        const int64_t attentionCapacity = d.geometry[23];
        const int32_t permWords = (hot + 1) / 2;
        const int32_t emitBase = (permWords + 7) / 8 * 8;
        const bool emitFits = emitBase + 2 * kTile + 8 <= hot &&
            (emitBase + 2 * kTile + 8 <= 24 * kTile || emitBase >= 27 * kTile);
        const bool emitInline = !d.pointers[15] && emitFits;
        auto outTile = order[emitBase];
        auto hostTile = order[emitBase + kTile];
        auto rowOut = order[emitBase + 2 * kTile];
        int32_t rank = 0, buffered = 0;
        const int64_t swapBase = int64_t(first) * topk;
        position = 0;
        for (int32_t row = first; row < end; ++row) {
            int32_t count = 0, compactCount = 0;
            for (int32_t base = 0; base < topk; base += kTile) {
                const int32_t extent = topk - base < kTile ? topk - base : kTile;
                for (int32_t i = 0; i < extent; ++i) {
                    // A cached id is always a cold host; -1 means resident or
                    // invalid and needs no slot here.
                    const int32_t cached = Read<int32_t>(stagingBase, int64_t(row) * topk + base + i);
                    int32_t host = -1, device = -1;
                    if (cached >= 0) {
                        host = cached;
                        int32_t h = Lookup(host);
                        if (h >= 0) {
                            const int32_t value = values.GetValue(h);
                            if (value == position) {
                                int32_t slot = perm.GetValue(hot - 1 - rank);
                                const int32_t physical = Physical(slot);
                                values.SetValue(h, kMiss | slot);
                                tags.SetValue(slot, host);
                                io.SetValue(buffered, host);
                                io.SetValue(kTile + buffered, physical);
                                ++rank;
                                if (++buffered == kTile || rank == misses) {
                                    Store(d.pointers[12], swapBase + rank - buffered, io, buffered);
                                    Store(d.pointers[13], swapBase + rank - buffered, io[kTile], buffered);
                                    buffered = 0;
                                }
                                device = physical;
                            } else {
                                device = Physical(value & kSlotMask);
                            }
                        }
                    } else if (emitInline) {
                        // Resident or invalid: only this token still needs the
                        // page tables.
                        Translate(Read<int32_t>(logicalBase, int64_t(row) * logicalStride + base + i), host, device);
                    }
                    if (emitInline) {
                        outTile.SetValue(i, device);
                        hostTile.SetValue(i, host);
                        count += device >= 0;
                    }
                    ++position;
                }
                if (!emitInline) continue;
                Store(d.pointers[8], int64_t(row) * topk + base, outTile, extent);
                Store(hostOut, int64_t(row) * topk + base, hostTile, extent);
                if (attentionOut) {
                    int32_t kept = 0;
                    for (int32_t i = 0; i < extent; ++i) {
                        int32_t device = outTile.GetValue(i);
                        int64_t mapped = device < 0 ? -1 :
                            int64_t(device / blockSpan) * attentionGap + device % blockSpan;
                        if (!compact) outTile.SetValue(i, mapped);
                        else if (mapped >= 0 && mapped < attentionCapacity) outTile.SetValue(kept++, mapped);
                    }
                    if (compact) {
                        Store(attentionOut, int64_t(row) * attentionStride + compactCount, outTile, kept);
                        compactCount += kept;
                    } else {
                        Store(attentionOut, int64_t(row) * attentionStride + base, outTile, extent);
                    }
                }
            }
            if (!emitInline) continue;
            if (compact) {
                for (int32_t i = 0; i < kTile; ++i) outTile.SetValue(i, -1);
                for (int32_t base = compactCount; base < topk; base += kTile) {
                    int32_t extent = topk - base < kTile ? topk - base : kTile;
                    Store(attentionOut, int64_t(row) * attentionStride + base, outTile, extent);
                }
                count = compactCount;
            }
            // io[] may still hold buffered swap rows, so the per-row scalars
            // use their own scratch slot.
            rowOut.SetValue(0, count);
            Store(d.pointers[11], int64_t(row) * d.geometry[20], rowOut, 1);
            int32_t swaps = misses - (row - first) * topk;
            swaps = swaps < 0 ? 0 : (swaps > topk ? topk : swaps);
            rowOut.SetValue(0, swaps);
            Store(d.pointers[14], row, rowOut, 1);
            for (int32_t i = 0; i < 8; ++i) rowOut.SetValue(i, 0);
            if (row == first) {
                rowOut.SetValue(0, error);
                rowOut.SetValue(1, error ? 0 : references - misses);
                rowOut.SetValue(2, error ? 0 : misses);
            }
            Store(d.pointers[17], int64_t(row) * 8, rowOut, 8);
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
                lruOut.SetValue(i, perm.GetValue(source));
            }
            Store(d.pointers[7], int64_t(state) * hot + base, lruOut, count);
        }
        if (!emitInline) Publish(true);
    }

    __aicore__ inline void Publish(bool valid)
    {
        // Recompute translations; never read another program's output or
        // scalar-cache data that has just been updated by MTE3. Descriptor
        // fields are hoisted into locals: the compiler cannot prove the
        // opaque UB tensor ops do not alias the descriptor, so every d.*
        // access would otherwise re-load UB inside the hot loop.
        const bool compact = d.geometry[23] != 0;
        const uint64_t logicalBase = d.pointers[0];
        const int64_t logicalStride = d.geometry[4];
        const uint64_t deviceOut = d.pointers[8];
        const uint64_t stagingBase = d.pointers[8];
        const uint64_t hostOut = d.pointers[10];
        const uint64_t attentionOut = d.pointers[9];
        const int64_t attentionStride = d.geometry[18];
        const int32_t blockSpan = d.geometry[16];
        const int64_t attentionGap = d.geometry[19];
        const int64_t attentionCapacity = d.geometry[23];
        const uint64_t maskOut = d.pointers[15];
        const bool byteMask = d.geometry[24] != 0;
        for (int32_t row = first; row < end; ++row) {
            int32_t count = 0, compactCount = 0;
            for (int32_t base = 0; base < topk; base += kTile) {
                int32_t extent = topk - base < kTile ? topk - base : kTile;
                for (int32_t i = 0; i < extent; ++i) {
                    int32_t host = -1, device = -1;
                    if (valid) {
                        const int32_t cached = Read<int32_t>(stagingBase, int64_t(row) * topk + base + i);
                        if (cached >= 0) host = cached;
                        else {
                            Translate(Read<int32_t>(logicalBase, int64_t(row) * logicalStride + base + i), host,
                                      device);
                        }
                    }
                    io.SetValue(kTile + i, host);
                    if (valid && !error && device < 0 && host >= 0) {
                        int32_t h = Lookup(host);
                        device = h >= 0 ? Physical(values.GetValue(h) & kSlotMask) : -1;
                    }
                    if (error) device = -1;
                    io.SetValue(i, device);
                    count += device >= 0;
                }
                Store(deviceOut, int64_t(row) * topk + base, io, extent);
                Store(hostOut, int64_t(row) * topk + base, io[kTile], extent);
                if (attentionOut) {
                    int32_t kept = 0;
                    for (int32_t i = 0; i < extent; ++i) {
                        int32_t device = io.GetValue(i);
                        int64_t mapped = device < 0 ? -1 :
                            int64_t(device / blockSpan) * attentionGap + device % blockSpan;
                        if (!compact) io.SetValue(i, mapped);
                        else if (mapped >= 0 && mapped < attentionCapacity) io.SetValue(kept++, mapped);
                    }
                    if (compact) {
                        // Keep order and duplicates; only Attention is compacted.
                        // Virtual indices and the refill plan retain their ABI.
                        Store(attentionOut, int64_t(row) * attentionStride + compactCount, io, kept);
                        compactCount += kept;
                    } else {
                        Store(attentionOut, int64_t(row) * attentionStride + base, io, extent);
                    }
                }
                // The Core compact plan is the primary interface. Optional
                // masks identify only the first occurrence of each miss.
                if (maskOut) {
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
                                    Translate(Read<int32_t>(logicalBase, int64_t(earlier) * logicalStride + col), oldHost, oldDevice);
                                    if (oldDevice < 0 && oldHost == host) { miss = false; break; }
                                }
                            }
                        }
                        if (byteMask) io.ReinterpretCast<uint8_t>().SetValue(i, uint8_t(miss));
                        else io.SetValue(i, int32_t(miss));
                    }
                    if (byteMask) Store(maskOut, int64_t(row) * topk + base, io.ReinterpretCast<uint8_t>(), extent);
                    else Store(maskOut, int64_t(row) * topk + base, io, extent);
                }
            }
            if (compact) {
                for (int32_t i = 0; i < kTile; ++i) io.SetValue(i, -1);
                for (int32_t base = compactCount; base < topk; base += kTile) {
                    int32_t extent = topk - base < kTile ? topk - base : kTile;
                    Store(attentionOut, int64_t(row) * attentionStride + base, io, extent);
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
    // Last valid page/block-table entries cached by Translate/Physical.
    int32_t reqCol = -1, reqPage = 0, stCol = -1, stPage = 0, blkCol = -1, blkPage = 0;
    TBuf<TPosition::VECCALC> keysBuffer, valuesBuffer, tagsBuffer, orderBuffer, ioBuffer;
    LocalTensor<int32_t> keys, values, tags, order, io;
};

}  // namespace
