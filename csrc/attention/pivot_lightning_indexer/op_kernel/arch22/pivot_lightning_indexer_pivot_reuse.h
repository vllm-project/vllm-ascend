/*
 * Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0.
 * Experimental single-request reuse. The native operator ABI is unchanged.
 */
#ifndef PIVOT_LIGHTNING_INDEXER_PIVOT_REUSE_H
#define PIVOT_LIGHTNING_INDEXER_PIVOT_REUSE_H

namespace LIKernel {
constexpr uint32_t PIVOT_READY_EVENT = 7;

template <typename LIT>
struct PivotProxyType : LIType<typename LIT::queryType, typename LIT::keyType,
    typename LIT::outputType, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, LIT::weightsTypeFlag> {
    static constexpr bool pivotReuse = true;
};

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
        return false;
    }
    // sparseMode 3 only. Mode 0 used to be accepted here too, with the proxy forced
    // causal (see below) so the group bound stayed at the head's causal length. That
    // pairing is not a mode-0 caller's semantics: the window is skipped whenever the
    // caller's mask is off -- perRowCausal is the caller's attenMaskFlag -- so a
    // mode-0 group would be clipped to the head's view and never handed its trailing
    // keys back, dropping exactly the keys mode 0 exists to expose. Refusing is both
    // simpler and exact: a mode-0 caller now gets native's own answer, unchanged.
    // Nothing loses reuse by this -- no pivot_lightning_indexer caller in vllm_ascend
    // passes 0; device_op.py hardcodes 3 in both of its branches.
    if (original.bSize != 1 || original.gSize != 32 || original.sparseCount != 2048 ||
        original.sparseMode != 3 ||
        queryLengths == nullptr || keyLengths == nullptr) {
        return false;
    }
    GlobalTensor<int32_t> qlen, klen;
    qlen.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryLengths));
    klen.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(keyLengths));
    uint32_t rows = qlen.GetValue(0);
    uint32_t length = klen.GetValue(0);
    // Two rows are enough to pool. The ragged group is the reason the cutoff is
    // not four: rows=6 already folds rows 4 and 5 into ONE proxy through the same
    // `copies = min(rows - sourceRow, 4)` branch that rows=2 uses on its only
    // group, so rejecting 2 and 3 while accepting 6 would be arbitrary. Only a
    // lone row stays native -- its proxy would be the identity (mean of one row)
    // over the same causal bound native already scans, i.e. the pooling round trip
    // with nothing pooled.
    //
    // Four and eight are the MTP3/MTP7 decode shapes, which is what this is for:
    // decode dominates the step count. It reaches here as mode 3 like everything
    // else -- device_op.py hardcodes sparse_mode=3 on every pivot_lightning_indexer call
    // -- so a decode group gets the same per-row treatment a prefill group does.
    if (rows < 2 || length < rows ||
        (LIT::layout == LI_LAYOUT::BSND && original.s1Size != rows)) {
        return false;
    }
    uint32_t denseRows = 0;
    // Reviewed paper boundary: first group position (1-based) >= 4096. No longer
    // keyed on sparseMode -- only mode 3 reaches this point, which is the case the
    // boundary was derived for -- and left unconditional so a group head short of
    // 4096 keys is treated as the degenerate case it is.
    if (length - rows + 1 < 4096) {
        denseRows = Min(rows, ((4095 - (length - rows) + 3) / 4) * 4);
    }
    if (denseRows == rows) {
        return false;
    }
    uint32_t groups = denseRows + (rows - denseRows + 3) / 4;
    uint64_t weightOffset = static_cast<uint64_t>(rows) * 8192;
    uint64_t privateWorkspace = (static_cast<uint64_t>(rows) * 8320 + 511) / 512 * 512;

    // AIVs prepare disjoint packed rows. All AIVs then signal their
    // paired AIC, so no cube can consume the proxy before global publication.
    if ASCEND_IS_AIV {
        uint32_t id = GetBlockIdx();
        TBuf<TPosition::VECCALC> rawBuf, accumBuf, tempBuf;
        pipe.InitBuffer(rawBuf, 32768);
        pipe.InitBuffer(accumBuf, 16384);
        pipe.InitBuffer(tempBuf, 16384);
        for (uint32_t row = id; row < groups; row += GetBlockNum() * 2) {
            uint32_t sourceRow = row < denseRows ? row : denseRows + (row - denseRows) * 4;
            uint32_t count = row < denseRows ? 1 : Min(4U, rows - sourceRow);
            PivotMeanRows<Q>(query, workspace, 4096, sourceRow, row, count,
                rawBuf.Get<Q>(), accumBuf.Get<float>(), tempBuf.Get<float>());
            PivotMeanRows<W>(weights, workspace + weightOffset, 32, sourceRow, row, count,
                rawBuf.Get<W>(), accumBuf.Get<float>(), tempBuf.Get<float>());
        }
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SyncAll();
        CrossCoreSetFlag<2, PIPE_MTE3>(PIVOT_READY_EVENT);
        pipe.Reset();
    } else {
        CrossCoreWaitFlag(PIVOT_READY_EVENT);
    }

    PivotLITilingData proxy = original;
    proxy.bSize = 1;
    proxy.s1Size = groups;
    proxy.s2Size = length;
    // The proxy row stands in for its group's FIRST row, so the keys it may score
    // are that row's causal keys -- not the caller's whole key range. sparseMode is
    // the only thing that gates that: kernel.h reads it once, into
    // `attenMaskFlag`, and everything causal hangs off that flag -- the per-row
    // score bound `cuRealAcSeq`, the scanned block count, and the local window.
    //
    // The bound comes out as `s2Size - pivotInputRows + PivotSourceRow(g) + 1`,
    // which is exactly the group head's causal length: rows 0-3 of an 8-row call
    // inherit row 0's view and rows 4-7 inherit row 4's.
    //
    // Only mode-3 callers reach this point (see the gate above), so `original`
    // already carries 3 and this assignment is an identity. Written out anyway: the
    // proxy's correctness rests on its mode being causal, and that must not quietly
    // become a property of a gate seventy lines up.
    proxy.sparseMode = 3;
    using ProxyType = PivotProxyType<LIT>;
    PivotLightningIndexerKernel<ProxyType> op;
    op.Init(workspace, key, workspace + weightOffset,
        nullptr, nullptr,
        table, indices, values, workspace + privateWorkspace, &proxy, &pipe, rows, denseRows);
    op.Process();
    return true;
}
} // namespace LIKernel
#endif
