/* Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0. */
#ifndef PIVOT_LIGHTNING_INDEXER_PIVOT_REUSE_H
#define PIVOT_LIGHTNING_INDEXER_PIVOT_REUSE_H

#ifndef PIVOT_REUSE_DEBUG
#define PIVOT_REUSE_DEBUG 0
#endif

namespace LIKernel {
constexpr uint32_t PIVOT_READY_EVENT = 7;

template <typename LIT>
struct PivotProxyType : LIType<typename LIT::queryType, typename LIT::keyType,
    typename LIT::outputType, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND, LIT::weightsTypeFlag> {
    static constexpr bool pivotReuse = true;
};

template <typename LIT>
__aicore__ inline void PivotReuseProbe(__gm__ const char *flag, int stage, bool pivot,
    const PivotLITilingData &tiling, uint32_t batch, uint32_t rows, uint32_t keys, uint32_t groups)
{
#if PIVOT_REUSE_DEBUG
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            AscendC::printf("[PIVOT_REUSE] %s stage=%d pivot=%d bSize=%d batch=%d rows=%d keys=%d groups=%d\n",
                flag, stage, static_cast<int>(pivot), static_cast<int>(tiling.bSize), static_cast<int>(batch),
                static_cast<int>(rows), static_cast<int>(keys), static_cast<int>(groups));
            AscendC::printf("[PIVOT_REUSE] %s gSize=%d sparseCount=%d sparseMode=%d s1Size=%d s2Size=%d\n",
                flag, static_cast<int>(tiling.gSize), static_cast<int>(tiling.sparseCount),
                static_cast<int>(tiling.sparseMode), static_cast<int>(tiling.s1Size),
                static_cast<int>(tiling.s2Size));
        }
    }
#else
    (void)flag;
    (void)stage;
    (void)pivot;
    (void)tiling;
    (void)batch;
    (void)rows;
    (void)keys;
    (void)groups;
#endif
}

template <typename T>
__aicore__ inline void PivotMeanRows(__gm__ uint8_t *input, __gm__ uint8_t *output,
    uint32_t width, uint32_t sourceRow, uint32_t outputRow, uint32_t count,
    LocalTensor<T> raw, LocalTensor<float> accum,
    LocalTensor<float> temp)
{
    GlobalTensor<T> src, dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    DataCopy(raw, src[sourceRow * width], count * width);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if constexpr (std::is_same<T, float>::value) {
        Adds(accum, raw, 0.0f, width);
    } else {
        Cast(accum, raw, RoundMode::CAST_NONE, width);
    }
    for (uint32_t row = 1; row < count; ++row) {
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(temp, raw[row * width], 0.0f, width);
        } else {
            Cast(temp, raw[row * width], RoundMode::CAST_NONE, width);
        }
        PipeBarrier<PIPE_V>();
        Add(accum, accum, temp, width);
    }
    PipeBarrier<PIPE_V>();
    Muls(accum, accum, 1.0f / count, width);
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<T, float>::value) {
        Adds(raw, accum, 0.0f, width);
    } else {
        Cast(raw, accum, RoundMode::CAST_RINT, width);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(dst[outputRow * width], raw, width);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    // The next input DMA reuses raw: ordering only the vector pipe is not enough.
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename LIT>
__aicore__ inline bool TryPivotReuse(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *weights, __gm__ uint8_t *queryLengths, __gm__ uint8_t *keyLengths,
    __gm__ uint8_t *table, __gm__ uint8_t *indices, __gm__ uint8_t *values,
    __gm__ uint8_t *workspace, const PivotLITilingData &original, TPipe &pipe)
{
    using Q = typename LIT::queryType;
    using W = typename std::conditional<LIT::weightsTypeFlag, float, Q>::type;
    if constexpr (!LIT::pageAttention || LIT::keyLayout != LI_LAYOUT::PA_BSND) {
        PivotReuseProbe<LIT>("flag1", 1, false, original, 0, 0, 0, 0);
        return false;
    }
    if (original.bSize == 0 || original.gSize != 32 || original.sparseCount != PIVOT_TOPK ||
        original.sparseMode != 3 || original.returnValue || original.usedCoreNum == 0 ||
        queryLengths == nullptr || keyLengths == nullptr ||
        (LIT::layout != LI_LAYOUT::TND && original.bSize != 1)) {
        PivotReuseProbe<LIT>("flag2", 2, false, original, 0, 0, 0, 0);
        return false;
    }

    GlobalTensor<uint32_t> qlen, klen;
    qlen.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(queryLengths));
    klen.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(keyLengths));
    uint32_t rowStart = 0;
    uint32_t totalGroups = 0;
    bool pooled = false;
    for (uint32_t b = 0; b < original.bSize; ++b) {
        uint32_t end = qlen.GetValue(b);
        uint32_t keys = klen.GetValue(b);
        uint32_t rows = end >= rowStart ? end - rowStart : 0;
        if (end < rowStart || (keys != 0 && keys < rows)) {
            PivotReuseProbe<LIT>("flag3", 3, false, original, b, rows, keys, totalGroups);
            return false;
        }
        PivotGeometry geometry(rows, keys);
        totalGroups += geometry.Groups();
        pooled |= keys != 0 && geometry.Groups() < geometry.rows;
        rowStart = end;
    }
    if (!pooled || (LIT::layout == LI_LAYOUT::BSND && rowStart != original.s1Size)) {
        PivotReuseProbe<LIT>("flag4", 4, false, original, original.bSize, rowStart, 0, totalGroups);
        return false;
    }
    PivotReuseProbe<LIT>("flag5", 5, true, original, original.bSize, rowStart, 0, totalGroups);

    PivotWorkspace offsets(totalGroups, original.bSize);
    auto groups = reinterpret_cast<__gm__ PivotGroup *>(workspace + offsets.groups);
    GlobalTensor<uint32_t> groupWords, groupEnds;
    groupWords.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(groups));
    groupEnds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace + offsets.lengths));

    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> rawBuf, accumBuf, tempBuf, metaBuf;
        pipe.InitBuffer(rawBuf, 32768);
        pipe.InitBuffer(accumBuf, 16384);
        pipe.InitBuffer(tempBuf, 16384);
        pipe.InitBuffer(metaBuf, 32);
        auto meta = metaBuf.Get<uint32_t>();
        uint32_t id = GetBlockIdx();
        uint32_t cores = GetBlockNum() * 2;
        uint32_t base = 0;
        rowStart = 0;
        for (uint32_t b = 0; b < original.bSize; ++b) {
            uint32_t end = qlen.GetValue(b);
            uint32_t keys = klen.GetValue(b);
            PivotGeometry geometry(end - rowStart, keys);
            uint32_t count = geometry.Groups();
            if (id == 0) {
                meta.SetValue(0, base + count);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopyPad(groupEnds[b], meta, {1, sizeof(uint32_t), 0, 0});
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
            // Distribute the flattened group space, including across short batches.
            for (uint32_t g = (id + cores - base % cores) % cores; g < count; g += cores) {
                PivotGroup group = geometry.Group(rowStart, keys, g);
                if (keys != 0) {
                    PivotMeanRows<Q>(query, workspace, PIVOT_QUERY_WIDTH, group.sourceRow, base + g, group.copies,
                        rawBuf.Get<Q>(), accumBuf.Get<float>(), tempBuf.Get<float>());
                    PivotMeanRows<W>(weights, workspace + offsets.weights, PIVOT_WEIGHT_WIDTH,
                        group.sourceRow, base + g, group.copies,
                        rawBuf.Get<W>(), accumBuf.Get<float>(), tempBuf.Get<float>());
                }
                meta.SetValue(0, group.sourceRow);
                meta.SetValue(1, group.copies);
                meta.SetValue(2, group.visibleKeys);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopyPad(groupWords[(base + g) * 3], meta, {1, sizeof(PivotGroup), 0, 0});
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
            base += count;
            rowStart = end;
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        CrossCoreSetFlag<2, PIPE_MTE3>(PIVOT_READY_EVENT);
        pipe.Reset();
    } else {
        CrossCoreWaitFlag(PIVOT_READY_EVENT);
    }

    PivotLITilingData proxy = original;
    proxy.s1Size = totalGroups;
    PivotLightningIndexerKernel<PivotProxyType<LIT>> op;
    op.Init(workspace, key, workspace + offsets.weights, workspace + offsets.lengths, keyLengths,
        table, workspace + offsets.indices, values, workspace + offsets.native, &proxy, &pipe, groups);
    op.Process();

    if ASCEND_IS_AIV {
        // LD may finish on another core. Publish every complete top-k before expansion.
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        pipe.Reset();
        TBuf<TPosition::VECCALC> resultBuf;
        pipe.InitBuffer(resultBuf, PIVOT_TOPK * sizeof(int32_t));
        auto result = resultBuf.Get<int32_t>();
        GlobalTensor<int32_t> packed, output;
        packed.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + offsets.indices));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));
        for (uint32_t g = GetBlockIdx(); g < totalGroups; g += GetBlockNum() * 2) {
            PivotGroup group{groups[g].sourceRow, groups[g].copies, groups[g].visibleKeys};
            if (group.visibleKeys == 0) {
                Duplicate(result, int32_t(-1), PIVOT_TOPK);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            } else {
                DataCopy(result, packed[static_cast<uint64_t>(g) * PIVOT_TOPK], PIVOT_TOPK);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            }
            for (uint32_t r = 0; r < group.copies; ++r) {
                if (group.visibleKeys != 0) {
                    for (uint32_t j = 0; j < r; ++j) {
                        result.SetValue(PIVOT_TOPK - r + j, group.visibleKeys + j);
                    }
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(output[static_cast<uint64_t>(group.sourceRow + r) * PIVOT_TOPK], result, PIVOT_TOPK);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
        }
    }
    return true;
}
} // namespace LIKernel
#endif
