// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "compressor_serial_params.h"

namespace CompressorSerial {
using namespace AscendC;

__simt_callee__ inline int64_t Start(__gm__ const int32_t* start, int64_t b,
                                   __gm__ const CompressorSerialParams* p)
{ return (p->flags & HAS_START) ? start[b] : 0; }

__simt_callee__ inline int64_t Begin(__gm__ const int32_t* cu, int64_t b,
                                   __gm__ const CompressorSerialParams* p)
{ return (p->flags & HAS_CU) ? cu[b] : b * p->sequence; }

__simt_callee__ inline int64_t Used(__gm__ const int32_t* cu, __gm__ const int32_t* used,
                                  int64_t b, __gm__ const CompressorSerialParams* p)
{
    if (p->flags & HAS_USED) return used[b];
    return (p->flags & HAS_CU) ? cu[b + 1] - cu[b] : p->sequence;
}

__simt_callee__ inline int64_t CacheRow(__gm__ const int32_t* table, int64_t b, int64_t pos,
                                      __gm__ const CompressorSerialParams* p)
{
    if (pos < 0) return -1;
    int64_t page, slot;
    if (p->cacheMode == 2) {
        page = (p->flags & HAS_TABLE) ? table[b] : b;
        slot = pos % p->blockSize;
    } else {
        int64_t logical = pos / p->blockSize;
        if (logical >= p->tableWidth) return -1;
        page = (p->flags & HAS_TABLE) ? table[b * p->tableWidth + logical]
                                    : b * p->tableWidth + logical;
        if ((p->flags & HAS_TABLE) && page == 0) return -1;
        slot = pos % p->blockSize;
    }
    if (page < 0 || page >= p->cacheBlocks) return -1;
    return page * p->cacheStride + slot * (2 * p->coff * p->dim);
}

// One thread computes one output channel. Softmax is stable even for large logits.
template<int Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void Pool(
    __gm__ const float* projection, __gm__ const float* state, __gm__ const float* ape,
    __gm__ const int32_t* table, __gm__ const int32_t* cu, __gm__ const int32_t* used,
    __gm__ const int32_t* start, __gm__ float* pooled,
    __gm__ const CompressorSerialParams* p)
{
    int64_t first = Simt::GetBlockIdx() * Threads + Simt::GetThreadIdx();
    int64_t step = Simt::GetBlockNum() * Threads;
    for (int64_t idx = first; idx < p->outputRows * p->dim; idx += step) {
        int64_t outRow = idx / p->dim, channel = idx % p->dim;
        int64_t b = 0, local = outRow, st = 0, count = 0;
        for (; b < p->batch; ++b) {
            st = Start(start, b, p);
            count = (st + Used(cu, used, b, p)) / p->ratio - st / p->ratio;
            int64_t capacity = p->layout == 3 ? (p->sequence + p->ratio - 1) / p->ratio : count;
            if (local < capacity) break;
            local -= capacity;
        }
        if (b >= p->batch || local >= count) { pooled[idx] = 0; continue; }
        int64_t groupEnd = (st / p->ratio + local + 1) * p->ratio;
        int64_t begin = Begin(cu, b, p);
        float maximum = -3.402823466e+38F, denominator = 0, numerator = 0;
        for (int64_t q = 0; q < p->coff * p->ratio; ++q) {
            int64_t pos = groupEnd - p->coff * p->ratio + q;
            if (pos < 0) continue;
            int64_t half = p->coff == 2 && q >= p->ratio ? 1 : 0;
            int64_t col = half * p->dim + channel;
            float kv, score;
            if (pos >= st) {
                int64_t row = begin + pos - st;
                kv = projection[row * p->projectionStride + col];
                score = projection[row * p->projectionStride + p->gateOffset + col]
                        + ape[(pos % p->ratio) * p->coff * p->dim + col];
            } else {
                int64_t row = CacheRow(table, b, pos, p);
                if (row < 0) continue;
                kv = state[row + col];
                score = state[row + p->coff * p->dim + col];
            }
            float next = score > maximum ? score : maximum;
            float oldScale = expf(maximum - next);
            float weight = expf(score - next);
            numerator = numerator * oldScale + weight * kv;
            denominator = denominator * oldScale + weight;
            maximum = next;
        }
        pooled[idx] = denominator == 0 ? 0 : numerator / denominator;
    }
}

template<int Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void Update(
    __gm__ const float* projection, __gm__ float* state, __gm__ const float* ape,
    __gm__ const int32_t* table, __gm__ const int32_t* cu, __gm__ const int32_t* used,
    __gm__ const int32_t* start, __gm__ const CompressorSerialParams* p)
{
    int64_t first = Simt::GetBlockIdx() * Threads + Simt::GetThreadIdx();
    int64_t step = Simt::GetBlockNum() * Threads;
    int64_t width = p->coff * p->dim;
    for (int64_t idx = first; idx < p->batch * width; idx += step) {
        int64_t b = idx / width, col = idx % width;
        int64_t st = Start(start, b, p), n = Used(cu, used, b, p), end = st + n;
        // Persist every valid token, including groups already consumed.
        // This observable cache side effect is checked against the black box.
        int64_t tail = st;
        int64_t begin = Begin(cu, b, p);
        for (int64_t pos = tail; pos < end; ++pos) {
            int64_t dst = CacheRow(table, b, pos, p);
            if (dst < 0) continue;
            int64_t src = (begin + pos - st) * p->projectionStride;
            state[dst + col] = projection[src + col];
            state[dst + width + col] = projection[src + p->gateOffset + col]
                                     + ape[(pos % p->ratio) * width + col];
        }
    }
}
} // namespace CompressorSerial
