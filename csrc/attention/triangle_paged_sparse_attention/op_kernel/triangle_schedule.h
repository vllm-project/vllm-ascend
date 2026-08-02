/*
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is licensed under CANN Open Software License Agreement
 * Version 2.0. See LICENSE in the repository root.
 *
 * Fixed TriangleMix schedule shared with the earlier 910B compile probe.
 *
 * This helper enumerates only logical KV intervals.  It does not pack K/V and
 * it does not permit the skipped middle region to enter QK or PV.
 */
#ifndef TRIANGLE_PAGED_ATTENTION_SCHEDULE_H
#define TRIANGLE_PAGED_ATTENTION_SCHEDULE_H

#include "kernel_operator.h"

namespace TrianglePaged {

constexpr uint32_t kQueryHeads = 32;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kPageSize = 128;
constexpr uint32_t kSinkTokens = 8;
constexpr uint32_t kWindowTokens = 512;
constexpr uint32_t kDenseTailTokens = 128;
constexpr uint32_t kQueryTile = 32;

struct KvInterval {
    uint32_t begin;
    uint32_t end;  // Exclusive absolute logical token index.
};

struct TileSchedule {
    KvInterval interval[2];
    uint32_t count;
    uint32_t dense;
};

/*
 * A q-tile can straddle either TriangleMix boundary.  Scheduling the whole
 * tile as dense in that case is semantically wrong: sparse rows would submit
 * the skipped middle keys to Cube.  Split the tile into homogeneous spans so
 * each span can run an independent online-softmax reduction.
 */
struct QuerySpan {
    uint32_t begin;  // Inclusive absolute query position.
    uint32_t end;    // Exclusive absolute query position.
    uint32_t sparse;
};

struct QuerySpanSchedule {
    QuerySpan span[3];
    uint32_t count;
};

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t MaxU32(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

__aicore__ inline void AppendQuerySpan(
    QuerySpanSchedule& result,
    uint32_t begin,
    uint32_t end,
    uint32_t sparse)
{
    if (begin >= end || result.count >= 3U) {
        return;
    }
    result.span[result.count] = {begin, end, sparse};
    ++result.count;
}

__aicore__ inline QuerySpanSchedule SplitQueryTile(
    uint32_t qBegin,
    uint32_t qEnd,
    uint32_t sparseBegin,
    uint32_t sparseEnd)
{
    QuerySpanSchedule result{};
    if (qBegin >= qEnd) {
        return result;
    }
    if (sparseBegin >= sparseEnd) {
        AppendQuerySpan(result, qBegin, qEnd, 0U);
        return result;
    }

    const uint32_t densePrefixEnd = MinU32(qEnd, sparseBegin);
    AppendQuerySpan(result, qBegin, densePrefixEnd, 0U);

    const uint32_t sparseSpanBegin = MaxU32(qBegin, sparseBegin);
    const uint32_t sparseSpanEnd = MinU32(qEnd, sparseEnd);
    AppendQuerySpan(result, sparseSpanBegin, sparseSpanEnd, 1U);

    const uint32_t denseTailBegin = MaxU32(qBegin, sparseEnd);
    AppendQuerySpan(result, denseTailBegin, qEnd, 0U);
    return result;
}

__aicore__ inline TileSchedule BuildTileSchedule(
    uint32_t qBegin,
    uint32_t qEnd,
    uint32_t kvLength,
    uint32_t sparseBegin,
    uint32_t sparseEnd,
    uint32_t sinkTokens = kSinkTokens,
    uint32_t windowTokens = kWindowTokens)
{
    TileSchedule result{};
    qEnd = MinU32(qEnd, kvLength);

    const bool fullySparse = qBegin >= sparseBegin && qEnd <= sparseEnd;
    if (!fullySparse) {
        result.interval[0] = {0, qEnd};
        result.count = qEnd == 0 ? 0 : 1;
        result.dense = 1;
        return result;
    }

    const uint32_t sinkEnd = MinU32(sinkTokens, kvLength);
    const uint32_t localBeginRaw =
        qBegin > windowTokens ? qBegin - windowTokens : 0;
    const uint32_t localBegin = MinU32(localBeginRaw, qEnd);

    // Merge overlapping ranges so sink logits are never counted twice.
    if (localBegin <= sinkEnd) {
        result.interval[0] = {0, qEnd};
        result.count = qEnd == 0 ? 0 : 1;
        result.dense = 0;
        return result;
    }

    result.interval[0] = {0, sinkEnd};
    result.interval[1] = {localBegin, qEnd};
    result.count = (sinkEnd == 0 ? 0 : 1) + (localBegin < qEnd ? 1 : 0);
    if (sinkEnd == 0 && localBegin < qEnd) {
        result.interval[0] = result.interval[1];
    }
    result.dense = 0;
    return result;
}

}  // namespace TrianglePaged

#endif  // TRIANGLE_PAGED_ATTENTION_SCHEDULE_H
