// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "compressor_serial_params.h"

namespace CompressorSerial {
using namespace AscendC;

__simt_callee__ inline int32_t Start(__gm__ const int32_t* start, int32_t b,
                                   __gm__ const CompressorSerialParams* p)
{ return (p->flags & HAS_START) ? start[b] : 0; }

__simt_callee__ inline int32_t Begin(__gm__ const int32_t* cu, int32_t b,
                                   __gm__ const CompressorSerialParams* p)
{ return (p->flags & HAS_CU) ? cu[b] : b * p->sequence; }

__simt_callee__ inline int32_t Used(__gm__ const int32_t* cu, __gm__ const int32_t* used,
                                  int32_t b, __gm__ const CompressorSerialParams* p)
{
    if (p->flags & HAS_USED) return used[b];
    return (p->flags & HAS_CU) ? cu[b + 1] - cu[b] : p->sequence;
}

__simt_callee__ inline int32_t CacheRow(__gm__ const int32_t* table, int32_t b, int32_t pos,
                                      __gm__ const CompressorSerialParams* p)
{
    if (pos < 0) return -1;
    int32_t page, slot;
    if (p->cacheMode == 2) {
        page = (p->flags & HAS_TABLE) ? table[b] : b;
        slot = pos % p->blockSize;
    } else {
        int32_t logical = pos / p->blockSize;
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
template<int Threads, int Ratio = 0, int Dim = 0, int Coff = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void Pool(
    __gm__ const float* projection, __gm__ const float* state, __gm__ const float* ape,
    __gm__ const int32_t* table, __gm__ const int32_t* cu, __gm__ const int32_t* used,
    __gm__ const int32_t* start, __gm__ float* pooled,
    __gm__ const CompressorSerialParams* p)
{
    const int32_t ratio = Ratio ? Ratio : p->ratio;
    const int32_t dim = Dim ? Dim : p->dim;
    const int32_t coff = Coff ? Coff : p->coff;
    int32_t first = Simt::GetBlockIdx() * Threads + Simt::GetThreadIdx();
    int32_t step = Simt::GetBlockNum() * Threads;
    for (int32_t idx = first; idx < p->outputRows * dim; idx += step) {
        int32_t outRow = idx / dim, channel = idx % dim;
        int32_t b = 0, local = outRow, st = 0, count = 0;
        for (; b < p->batch; ++b) {
            st = Start(start, b, p);
            count = (st + Used(cu, used, b, p)) / ratio - st / ratio;
            int32_t capacity = p->layout == 3 ? (p->sequence + ratio - 1) / ratio : count;
            if (local < capacity) break;
            local -= capacity;
        }
        if (b >= p->batch || local >= count) { pooled[idx] = 0; continue; }
        int32_t groupEnd = (st / ratio + local + 1) * ratio;
        int32_t begin = Begin(cu, b, p);
        float maximum = -3.402823466e+38F, denominator = 0, numerator = 0;
        #pragma unroll
        for (int32_t q = 0; q < coff * ratio; ++q) {
            int32_t pos = groupEnd - coff * ratio + q;
            if (pos < 0) continue;
            int32_t half = coff == 2 && q >= ratio ? 1 : 0;
            int32_t col = half * dim + channel;
            float kv, score;
            if (pos >= st) {
                int32_t row = begin + pos - st;
                kv = projection[row * p->projectionStride + col];
                score = projection[row * p->projectionStride + p->gateOffset + col]
                        + ape[(pos % ratio) * coff * dim + col];
            } else {
                int32_t row = CacheRow(table, b, pos, p);
                if (row < 0) continue;
                kv = state[row + col];
                score = state[row + coff * dim + col];
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

template<int Threads, int Ratio = 0, int Dim = 0, int Coff = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void PoolWarp(
    __gm__ const float* projection, __gm__ const float* state, __gm__ const float* ape,
    __gm__ const int32_t* table, __gm__ const int32_t* cu, __gm__ const int32_t* used,
    __gm__ const int32_t* start, __gm__ float* pooled,
    __gm__ const CompressorSerialParams* p)
{
    const int32_t ratio = Ratio ? Ratio : p->ratio;
    const int32_t dim = Dim ? Dim : p->dim;
    const int32_t coff = Coff ? Coff : p->coff;
    int32_t lane = Simt::GetThreadIdx() % 32;
    int32_t qLane = lane / 8;
    int32_t first = Simt::GetBlockIdx() * (Threads / 32) * 8 + (Simt::GetThreadIdx() / 32) * 8 + lane % 8;
    int32_t step = Simt::GetBlockNum() * (Threads / 32) * 8;
    for (int32_t idx = first; idx < p->outputRows * dim; idx += step) {
        int32_t outRow = idx / dim, channel = idx % dim;
        int32_t b = 0, local = outRow, st = 0, count = 0;
        for (; b < p->batch; ++b) {
            st = Start(start, b, p);
            count = (st + Used(cu, used, b, p)) / ratio - st / ratio;
            int32_t capacity = p->layout == 3 ? (p->sequence + ratio - 1) / ratio : count;
            if (local < capacity) break;
            local -= capacity;
        }
        if (b >= p->batch || local >= count) { pooled[idx] = 0; continue; }
        int32_t groupEnd = (st / ratio + local + 1) * ratio;
        int32_t begin = Begin(cu, b, p);
        float maximum = -3.402823466e+38F, denominator = 0, numerator = 0;
        #pragma unroll
        for (int32_t q = qLane; q < coff * ratio; q += 4) {
            int32_t pos = groupEnd - coff * ratio + q;
            if (pos < 0) continue;
            int32_t half = coff == 2 && q >= ratio ? 1 : 0;
            int32_t col = half * dim + channel;
            float kv, score;
            if (pos >= st) {
                int32_t row = begin + pos - st;
                kv = projection[row * p->projectionStride + col];
                score = projection[row * p->projectionStride + p->gateOffset + col]
                        + ape[(pos % ratio) * coff * dim + col];
            } else {
                int32_t row = CacheRow(table, b, pos, p);
                if (row < 0) continue;
                kv = state[row + col];
                score = state[row + coff * dim + col];
            }
            float next = score > maximum ? score : maximum;
            float oldScale = expf(maximum - next);
            float weight = expf(score - next);
            numerator = numerator * oldScale + weight * kv;
            denominator = denominator * oldScale + weight;
            maximum = next;
        }
        float other = asc_shfl_xor(maximum, 8);
        float globalMaximum = maximum > other ? maximum : other;
        other = asc_shfl_xor(globalMaximum, 16);
        globalMaximum = globalMaximum > other ? globalMaximum : other;
        float scale = expf(maximum - globalMaximum);
        numerator *= scale;
        denominator *= scale;
        numerator += asc_shfl_xor(numerator, 8);
        denominator += asc_shfl_xor(denominator, 8);
        numerator += asc_shfl_xor(numerator, 16);
        denominator += asc_shfl_xor(denominator, 16);
        if (qLane == 0) pooled[idx] = denominator == 0 ? 0 : numerator / denominator;
    }
}

template<int Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void Update(
    __gm__ const float* projection, __gm__ float* state, __gm__ const float* ape,
    __gm__ const int32_t* table, __gm__ const int32_t* cu, __gm__ const int32_t* used,
    __gm__ const int32_t* start, __gm__ const CompressorSerialParams* p)
{
    int32_t first = Simt::GetBlockIdx() * Threads + Simt::GetThreadIdx();
    int32_t step = Simt::GetBlockNum() * Threads;
    int32_t width = p->coff * p->dim;
    for (int32_t idx = first; idx < p->tokens * width; idx += step) {
        int32_t row = idx / width, col = idx % width, b = 0, begin = 0;
        for (; b < p->batch; ++b) {
            begin = Begin(cu, b, p);
            if (row >= begin && row < begin + Used(cu, used, b, p)) break;
        }
        if (b >= p->batch) continue;
        int32_t pos = Start(start, b, p) + row - begin;
        int32_t dst = CacheRow(table, b, pos, p);
        if (dst < 0) continue;
        int32_t src = row * p->projectionStride;
        state[dst + col] = projection[src + col];
        state[dst + width + col] = projection[src + p->gateOffset + col]
                                 + ape[(pos % p->ratio) * width + col];
    }
}
} // namespace CompressorSerial
