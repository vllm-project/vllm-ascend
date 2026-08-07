/*
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is licensed under CANN Open Software License Agreement
 * Version 2.0. See LICENSE in the repository root.
 */
#ifndef TRIANGLE_PAGED_ATTENTION_FIA_FAST_PATH_H
#define TRIANGLE_PAGED_ATTENTION_FIA_FAST_PATH_H

#include "triangle_paged_block_mmad.h"
#include "triangle_paged_sparse_attention_tiling.h"
#include "triangle_schedule.h"

namespace TrianglePaged {

using namespace AscendC;
using namespace NpuArch;

constexpr uint32_t kFastImplementation = 2;
constexpr uint32_t kKvTile = 512;
constexpr uint32_t kCubeInnerKvTile = 128;
constexpr uint32_t kGroupSize = kQueryHeads / kKvHeads;
constexpr uint32_t kCubeRows = kQueryTile * kGroupSize;
constexpr uint32_t kWorkspaceAlignment = 512;

enum class IntervalKind : uint32_t {
    kSink = 0,
    kLocal = 1,
    kDense = 2,
};

__aicore__ inline uint32_t IntervalTileCount(KvInterval interval)
{
    const uint32_t length = interval.end - interval.begin;
    return (length + kKvTile - 1U) / kKvTile;
}

__aicore__ inline bool NeedsBoundaryMask(
    IntervalKind kind,
    uint32_t keyBegin,
    uint32_t keyEnd,
    uint32_t qBegin,
    uint32_t qEnd,
    uint32_t windowTokens)
{
    if (kind == IntervalKind::kSink) {
        return false;
    }
    if (keyEnd > qBegin + 1U) {
        return true;
    }
    if (kind == IntervalKind::kLocal) {
        const uint32_t latestLower =
            qEnd - 1U > windowTokens
                ? qEnd - 1U - windowTokens
                : 0U;
        return keyBegin < latestLower;
    }
    return false;
}

class TrianglePagedFiaFastPath {
private:
    using ArchTag = NpuArch::Arch::AtlasA2;
    using Element = bfloat16_t;
    using LayoutQ = NpuArch::layout::RowMajor;
    using LayoutK = NpuArch::layout::ColumnMajor;
    using LayoutV = NpuArch::layout::RowMajor;
    using LayoutS = NpuArch::layout::RowMajor;
    using LayoutP = NpuArch::layout::RowMajor;
    using LayoutO = NpuArch::layout::RowMajor;
    using LayoutOTmp = NpuArch::layout::RowMajor;
    using LayoutUpdate = NpuArch::layout::RowMajor;
    using LayoutLse = NpuArch::layout::RowMajor;
    using LayoutMask = NpuArch::layout::RowMajor;

    using QType = NpuArch::Gemm::GemmType<Element, LayoutQ>;
    using KType = NpuArch::Gemm::GemmType<Element, LayoutK>;
    using VType = NpuArch::Gemm::GemmType<Element, LayoutV>;
    using SType = NpuArch::Gemm::GemmType<float, LayoutS>;
    using PType = NpuArch::Gemm::GemmType<Element, LayoutP>;
    using OType = NpuArch::Gemm::GemmType<Element, LayoutO>;
    using OTmpType = NpuArch::Gemm::GemmType<float, LayoutOTmp>;
    using UpdateType = NpuArch::Gemm::GemmType<float, LayoutUpdate>;
    using LseType = NpuArch::Gemm::GemmType<float, LayoutLse>;
    using MaskType = NpuArch::Gemm::GemmType<int8_t, LayoutMask>;

    using QkL1Shape =
        NpuArch::GemmShape<
            kCubeRows, kCubeInnerKvTile, kHeadDim>;
    using QkL0Shape =
        NpuArch::GemmShape<
            kCubeRows, kCubeInnerKvTile, kHeadDim>;
    using QkPolicy =
        NpuArch::Gemm::MmadAtlasA2SFAIQK<false, false>;
    using QkBase = NpuArch::Gemm::Block::BlockMmad<
        QkPolicy, QkL1Shape, QkL0Shape, QType, KType, SType>;
    using QkMmad = DirectPagedQkMmad<QkBase>;

    // The outer KV tile is 512, but Atlas A2 L0 ping-pong stages can hold
    // only 128 BF16 K/N elements for this M=128 geometry.  Keep Cube tiles
    // at 128: QK naturally emits four score sub-tiles, and PV accumulates four
    // K=128 MMADs into one output tile.
    using PvL1Shape =
        NpuArch::GemmShape<kCubeRows, kHeadDim, kKvTile>;
    using PvL0Shape =
        NpuArch::GemmShape<
            kCubeRows, kHeadDim, kCubeInnerKvTile>;
    using PvPolicy =
        NpuArch::Gemm::MmadAtlasA2SFAIPV<false, false>;
    using PvBase = NpuArch::Gemm::Block::BlockMmad<
        PvPolicy, PvL1Shape, PvL0Shape, PType, VType, OTmpType>;
    using PvMmad = DirectPagedPvMmad<PvBase>;

    static constexpr uint32_t kQkL1Bytes =
        QkBase::L1A_SIZE + QkBase::L1B_SIZE * QkBase::STAGES;
    static constexpr uint32_t kPvL1Bytes =
        PvBase::L1A_SIZE * PvBase::STAGES + PvBase::L1B_SIZE;
    static_assert(
        kQkL1Bytes + kPvL1Bytes <= ArchTag::L1_SIZE,
        "QK and PV L1 allocations exceed Atlas A2 L1");

    static constexpr uint32_t kQkL0ARequired =
        QkL0Shape::M * QkL0Shape::K * sizeof(Element);
    static constexpr uint32_t kQkL0BRequired =
        QkL0Shape::N * QkL0Shape::K * sizeof(Element);
    static constexpr uint32_t kQkL0CRequired =
        QkL0Shape::M * QkL0Shape::N * sizeof(float);
    static constexpr uint32_t kPvL0ARequired =
        PvL0Shape::M * PvL0Shape::K * sizeof(Element);
    static constexpr uint32_t kPvL0BRequired =
        PvL0Shape::N * PvL0Shape::K * sizeof(Element);
    static constexpr uint32_t kPvL0CRequired =
        PvL0Shape::M * PvL0Shape::N * sizeof(float);
    static_assert(
        kQkL0ARequired <= QkBase::L0A_PINGPONG_BUF_SIZE &&
            kQkL0BRequired <= QkBase::L0B_PINGPONG_BUF_SIZE &&
            kQkL0CRequired <= QkBase::L0C_PINGPONG_BUF_SIZE,
        "QK L0 tile exceeds an Atlas A2 ping-pong stage");
    static_assert(
        kPvL0ARequired <= PvBase::L0A_PINGPONG_BUF_SIZE &&
            kPvL0BRequired <= PvBase::L0B_PINGPONG_BUF_SIZE &&
            kPvL0CRequired <= PvBase::L0C_PINGPONG_BUF_SIZE,
        "PV L0 tile exceeds an Atlas A2 ping-pong stage");

    using SoftmaxPolicy =
        NpuArch::Epilogue::EpilogueAtlasA2OnlineSoftmax<
            NpuArch::Epilogue::LseMode::NONE, float>;
    using OnlineSoftmax =
        NpuArch::Epilogue::Block::BlockEpilogue<
            SoftmaxPolicy, PType, SType, MaskType>;
    using RescalePolicy =
        NpuArch::Epilogue::EpilogueAtlasA2RescaleO<
            NpuArch::Epilogue::LseMode::NONE, float>;
    using RescaleOutput =
        NpuArch::Epilogue::Block::BlockEpilogue<
            RescalePolicy, OType, OTmpType, UpdateType, LseType>;

public:
    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR keyCache,
        GM_ADDR valueCache,
        GM_ADDR blockTable,
        GM_ADDR attentionOut,
        GM_ADDR workspace,
        const TrianglePagedSparseAttentionTilingData& tiling)
    {
        tiling_ = tiling;
        const uint64_t queryElements =
            static_cast<uint64_t>(tiling_.queryTokens) *
            kQueryHeads * kHeadDim;
        const uint64_t cacheElements =
            static_cast<uint64_t>(tiling_.physicalPageCount) *
            kPageSize * kKvHeads * kHeadDim;
        query_.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(query), queryElements);
        key_.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(keyCache), cacheElements);
        value_.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(valueCache), cacheElements);
        blockTable_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(blockTable),
            tiling_.blockTablePageCapacity);
        output_.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(attentionOut),
            queryElements);

#ifdef __DAV_C220_VEC__
        coreIndex_ = GetBlockIdx() / GetSubBlockNum();
#else
        coreIndex_ = GetBlockIdx();
#endif
        __gm__ uint8_t* coreWorkspace =
            reinterpret_cast<__gm__ uint8_t*>(workspace) +
            static_cast<uint64_t>(coreIndex_) *
                tiling_.workspacePerCoreBytes;
        score_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                coreWorkspace + tiling_.scoreOffsetBytes));
        probability_.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(
                coreWorkspace + tiling_.probabilityOffsetBytes));
        outputTmp_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                coreWorkspace + tiling_.outputTmpOffsetBytes));
        outputUpdate_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                coreWorkspace + tiling_.outputUpdateOffsetBytes));
        lseScratch_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                coreWorkspace + tiling_.lseScratchOffsetBytes));
    }

    __aicore__ inline void Process()
    {
        if (tiling_.implementationStatus != kFastImplementation ||
            tiling_.abiVersion != 2U ||
            tiling_.queryTile != kQueryTile ||
            tiling_.kvTile != kKvTile ||
            tiling_.groupSize != kGroupSize) {
            return;
        }

#ifdef __DAV_C220_CUBE__
        InitCubeEvents();
        QkMmad qk(resource_);
        PvMmad pv(resource_, kQkL1Bytes);
#endif
#ifdef __DAV_C220_VEC__
        InitVectorEvents();
        OnlineSoftmax onlineSoftmax(resource_, tiling_.scale);
        RescaleOutput rescaleOutput(resource_);
#endif

        NpuArch::Arch::CrossCoreFlag qkReady(1);
        NpuArch::Arch::CrossCoreFlag softmaxReady(2);
        NpuArch::Arch::CrossCoreFlag pvReady(3);

        for (uint32_t task = coreIndex_;
             task < tiling_.taskCount;
             task += tiling_.activeAicCores) {
            const uint32_t queryTileIndex = task / kKvHeads;
            const uint32_t kvHead = task % kKvHeads;
            const uint32_t queryRow =
                queryTileIndex * kQueryTile;
            const uint32_t queryTileTokens =
                MinU32(kQueryTile, tiling_.queryTokens - queryRow);
            const uint32_t queryTileBegin =
                tiling_.queryStart + queryRow;
            const uint32_t queryTileEnd =
                queryTileBegin + queryTileTokens;
            const uint64_t kvHeadOffset =
                static_cast<uint64_t>(kvHead) * kHeadDim;

            const QuerySpanSchedule querySpans = SplitQueryTile(
                queryTileBegin,
                queryTileEnd,
                tiling_.sparseBegin,
                tiling_.sparseEnd);
            for (uint32_t spanIndex = 0;
                 spanIndex < querySpans.count;
                 ++spanIndex) {
                const QuerySpan querySpan =
                    querySpans.span[spanIndex];
                const uint32_t spanQueryRow =
                    queryRow + querySpan.begin - queryTileBegin;
                const uint32_t queryTokens =
                    querySpan.end - querySpan.begin;
                const uint32_t qBegin = querySpan.begin;
                const uint32_t qEnd = querySpan.end;
                const uint32_t rows = queryTokens * kGroupSize;
                const uint64_t queryOffset =
                    (static_cast<uint64_t>(spanQueryRow) *
                         kQueryHeads +
                     kvHead * kGroupSize) *
                    kHeadDim;

            const TileSchedule schedule = BuildTileSchedule(
                qBegin,
                qEnd,
                tiling_.seqLen,
                tiling_.sparseBegin,
                tiling_.sparseEnd,
                tiling_.sinkTokens,
                tiling_.localWindow);
            uint32_t totalTiles = 0;
            for (uint32_t interval = 0;
                 interval < schedule.count;
                 ++interval) {
                totalTiles +=
                    IntervalTileCount(schedule.interval[interval]);
            }
            if (totalTiles == 0U) {
                continue;
            }

#ifdef __DAV_C220_CUBE__
            LayoutQ layoutQ(rows, kHeadDim);
            uint32_t groupHeads = kGroupSize;
            qk.loadQGM(
                query_[queryOffset],
                layoutQ,
                rows,
                groupHeads,
                static_cast<uint64_t>(kQueryHeads * kHeadDim));
#endif

            uint32_t tileOrdinal = 0;
            for (uint32_t intervalIndex = 0;
                 intervalIndex < schedule.count;
                 ++intervalIndex) {
                const KvInterval interval =
                    schedule.interval[intervalIndex];
                const IntervalKind kind =
                    querySpan.sparse == 0U
                        ? IntervalKind::kDense
                        : (interval.begin == 0U &&
                                   interval.end <= tiling_.sinkTokens
                               ? IntervalKind::kSink
                               : IntervalKind::kLocal);
                const uint32_t tiles =
                    IntervalTileCount(interval);
                for (uint32_t tile = 0; tile < tiles; ++tile) {
                    const uint32_t keyBegin =
                        interval.begin + tile * kKvTile;
                    const uint32_t keyCount =
                        MinU32(kKvTile, interval.end - keyBegin);
                    const uint32_t keyEnd = keyBegin + keyCount;
                    const bool first = tileOrdinal == 0U;
                    const bool last =
                        tileOrdinal + 1U == totalTiles;
                    const bool boundaryMask = NeedsBoundaryMask(
                        kind,
                        keyBegin,
                        keyEnd,
                        qBegin,
                        qEnd,
                        tiling_.localWindow);

                    LayoutS layoutS(rows, keyCount, kKvTile);
                    NpuArch::GemmCoord qkShape{
                        rows, keyCount, kHeadDim};
#ifdef __DAV_C220_CUBE__
                    LayoutK layoutK(
                        kKvHeads * kHeadDim, kKvTile);
                    qk(
                        query_[queryOffset],
                        key_[kvHeadOffset],
                        score_,
                        blockTable_,
                        layoutQ,
                        layoutK,
                        layoutS,
                        qkShape,
                        keyBegin,
                        kPageSize,
                        kKvHeads * kHeadDim);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        qkReady);
#endif
#ifdef __DAV_C220_VEC__
                    NpuArch::Arch::CrossCoreWaitFlag(qkReady);
                    const uint32_t proceduralMaskMode =
                        !boundaryMask
                            ? 0U
                            : (kind == IntervalKind::kLocal ? 2U : 1U);
                    const NpuArch::Epilogue::Block::
                        ProceduralBoundaryMaskParams proceduralMask{
                            proceduralMaskMode,
                            qBegin,
                            keyBegin,
                            tiling_.localWindow};
                    LayoutP layoutP(rows, keyCount, kKvTile);
                    onlineSoftmax(
                        probability_,
                        score_,
                        layoutP,
                        layoutS,
                        qkShape,
                        first,
                        0,
                        queryTokens,
                        kGroupSize,
                        0,
                        softmaxReady,
                        proceduralMask);
#endif

                    NpuArch::GemmCoord pvShape{
                        rows, kHeadDim, keyCount};
                    LayoutOTmp layoutOTmp(
                        rows, kHeadDim, kHeadDim);
#ifdef __DAV_C220_CUBE__
                    LayoutP layoutP(rows, keyCount, kKvTile);
                    LayoutV layoutV(kKvTile, kKvHeads * kHeadDim);
                    pv(
                        probability_,
                        value_[kvHeadOffset],
                        outputTmp_,
                        blockTable_,
                        layoutP,
                        layoutV,
                        layoutOTmp,
                        pvShape,
                        keyBegin,
                        kPageSize,
                        kKvHeads * kHeadDim,
                        softmaxReady);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(
                        pvReady);
#endif
#ifdef __DAV_C220_VEC__
                    LayoutO layoutO(
                        tiling_.queryTokens,
                        kQueryHeads * kHeadDim);
                    LayoutUpdate layoutUpdate(
                        rows, kHeadDim, kHeadDim);
                    LayoutLse layoutLse(
                        tiling_.queryTokens, kQueryHeads);
                    NpuArch::Arch::CrossCoreWaitFlag(pvReady);
                    rescaleOutput(
                        output_[queryOffset],
                        outputTmp_,
                        outputUpdate_,
                        lseScratch_,
                        layoutO,
                        layoutOTmp,
                        layoutUpdate,
                        layoutLse,
                        pvShape,
                        queryTokens,
                        kGroupSize,
                        first,
                        last,
                        0);
#endif
                    ++tileOrdinal;
                }
            }
        }
        }
        FinalizeEvents();
    }

private:
    __aicore__ inline void InitCubeEvents()
    {
#ifdef __DAV_C220_CUBE__
        for (uint32_t event = 0; event < 8U; ++event) {
            SetFlag<HardEvent::M_MTE1>(event);
            SetFlag<HardEvent::MTE1_MTE2>(event);
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
#endif
    }

    __aicore__ inline void InitVectorEvents()
    {
#ifdef __DAV_C220_VEC__
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
#endif
    }

    /*
     * Event tokens are resources, not just ordering annotations.  Drain the
     * same tokens initialized above before the MIX kernel returns; in
     * particular MTE3_MTE2(EVENT_ID6) is produced only after the final
     * rescale CopyOToGm has completed.
     */
    __aicore__ inline void FinalizeEvents()
    {
#ifdef __DAV_C220_CUBE__
        for (uint32_t event = 0; event < 8U; ++event) {
            WaitFlag<HardEvent::M_MTE1>(event);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        for (uint32_t event = 0; event < 8U; ++event) {
            WaitFlag<HardEvent::MTE1_MTE2>(event);
        }
#endif
#ifdef __DAV_C220_VEC__
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);

        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
#endif
        PipeBarrier<PIPE_ALL>();
    }

    TrianglePagedSparseAttentionTilingData tiling_{};
    uint32_t coreIndex_{0};
    NpuArch::Arch::Resource<ArchTag> resource_;

    GlobalTensor<Element> query_;
    GlobalTensor<Element> key_;
    GlobalTensor<Element> value_;
    GlobalTensor<int32_t> blockTable_;
    GlobalTensor<Element> output_;
    GlobalTensor<float> score_;
    GlobalTensor<Element> probability_;
    GlobalTensor<float> outputTmp_;
    GlobalTensor<float> outputUpdate_;
    GlobalTensor<float> lseScratch_;
};

}  // namespace TrianglePaged

#endif  // TRIANGLE_PAGED_ATTENTION_FIA_FAST_PATH_H
