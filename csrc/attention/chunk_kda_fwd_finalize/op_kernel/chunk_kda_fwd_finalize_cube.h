/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_CUBE_H
#define CHUNK_KDA_FWD_FINALIZE_CUBE_H

#include <type_traits>
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif
#else
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_kda_fwd_finalize_struct.h"

namespace KdaFinalize {

using namespace AscendC;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using FinalizeArch = Catlass::Arch::Ascend950;
constexpr bool kFinalizeA5 = true;
#else
using FinalizeArch = Catlass::Arch::AtlasA2;
constexpr bool kFinalizeA5 = false;
#endif

template <bool StateVFirst, bool OutputSequenceMajor, bool UseAivInputMover>
class FinalizeCube {
    using Element = bfloat16_t;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using HLayout = std::conditional_t<StateVFirst, LayoutCM, LayoutRM>;
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<
        FinalizeArch, Element, LayoutRM, Element, HLayout, float, LayoutRM>;
    using TileCopyAV = Catlass::Gemm::Tile::PackedTileCopyTla<
        FinalizeArch, Element, LayoutRM, Element, LayoutRM, float, LayoutRM>;

    static constexpr uint32_t kQOffset = 0;
    static constexpr uint32_t kHOffset = kQOffset + 16 * 1024;
    static constexpr uint32_t kAOffset = kHOffset + 32 * 1024;
    static constexpr uint32_t kVOffset = kAOffset + 8 * 1024;
    static constexpr uint32_t kL1HeadBytes = 72 * 1024;
    static constexpr uint32_t kL0AOffset[2] = {0, 16 * 1024};
    static constexpr uint32_t kL0BOffset[2] = {0, 32 * 1024};
    static constexpr uint8_t kL0CMutexBase = 2;

public:
    __aicore__ inline void Init(const FinalizeArgs &args)
    {
        args_ = args;
        core_ = WorkgroupId();
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.qgScaled));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.aqk));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.vNew));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.h));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.attnOut));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<Element>(static_cast<Element>(0));
        }
#endif
    }

    __aicore__ inline void Process()
    {
        if (core_ >= args_.tiling.usedCoreNum || args_.tiling.usedCoreNum == 0) {
            return;
        }
        const uint32_t partitions = CeilDiv(args_.tiling.valueHeadNum,
                                            args_.tiling.headsPerPartition);
        const uint32_t total = TotalWorkItems(args_);
        const uint32_t begin = WorkBegin(total, core_, args_.tiling.usedCoreNum);
        const uint32_t end = WorkEnd(total, core_, args_.tiling.usedCoreNum);
        // 主体只分 chunk；余下不足一轮核数的 chunk 才按 4-head 组均摊。
        const uint32_t cores = args_.tiling.usedCoreNum;
        const bool balanceTail = kFinalizeA5 && !args_.tiling.isVarLen &&
            partitions == 1 && total >= cores && total % cores != 0;
        const uint32_t fullChunksPerCore = total / cores;
        const uint32_t headGroups = CeilDiv(args_.tiling.valueHeadNum, Shape::kHeadsPerGroup);
        const uint32_t tailGroups = (total % cores) * headGroups;
        const uint32_t tailBegin = WorkBegin(tailGroups, core_, cores);
        const uint32_t tailEnd = WorkEnd(tailGroups, core_, cores);
        const uint32_t workCount = balanceTail
            ? fullChunksPerCore + tailEnd - tailBegin : end - begin;
        uint32_t outputSlot = 0;
        bool outputSlotUsed[2] = {false, false};
        for (uint32_t index = 0; index < workCount; ++index) {
            const uint32_t work = begin + index;
            uint32_t chunkId = work / partitions;
            uint32_t headBegin = (work % partitions) * args_.tiling.headsPerPartition;
            uint32_t headEnd = headBegin + args_.tiling.headsPerPartition;
            if (balanceTail) {
                if (index < fullChunksPerCore) {
                    chunkId = core_ * fullChunksPerCore + index;
                    headBegin = 0;
                    headEnd = args_.tiling.valueHeadNum;
                } else {
                    const uint32_t tail = tailBegin + index - fullChunksPerCore;
                    chunkId = fullChunksPerCore * cores + tail / headGroups;
                    headBegin = (tail % headGroups) * Shape::kHeadsPerGroup;
                    headEnd = headBegin + Shape::kHeadsPerGroup;
                }
            }
            FinalizeChunk chunk{};
            if (!ResolveChunk(args_, chunkId, chunk)) {
                continue;
            }
            if (headEnd > args_.tiling.valueHeadNum) {
                headEnd = args_.tiling.valueHeadNum;
            }
            for (uint32_t group = headBegin; group < headEnd; group += Shape::kHeadsPerGroup) {
                uint32_t active = headEnd - group;
                if (active > Shape::kHeadsPerGroup) {
                    active = Shape::kHeadsPerGroup;
                }
                for (uint32_t localHead = 0; localHead < active; ++localHead) {
                    StageC0(chunk, group + localHead, localHead, outputSlot);
                    outputSlotUsed[outputSlot] = true;
                    outputSlot ^= 1;
                }
            }
        }
        if constexpr (kFinalizeA5) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // 最后一轮 Fixpipe 没有后继 MMAD 消费互斥量，在 M pipe 侧等待写回完成。
            for (uint32_t slot = 0; slot < 2; ++slot) {
                if (outputSlotUsed[slot]) {
                    const uint8_t mutexId = static_cast<uint8_t>(kL0CMutexBase + slot);
                    Mutex::Lock<PIPE_M>(mutexId);
                    Mutex::Unlock<PIPE_M>(mutexId);
                }
            }
#endif
        } else {
            for (uint32_t plane = 0; plane < 2; ++plane) {
                if (l0OperandInFlight_[plane]) {
                    WaitFlag<HardEvent::M_MTE1>(plane);
                }
                if (l0OutputInFlight_[plane]) {
                    WaitFlag<HardEvent::FIX_M>(plane);
                }
            }
        }
    }

private:
    template <typename TileCopy, typename LayoutB>
    __aicore__ inline void LoadOperands(GlobalTensor<Element> &sourceA,
                                        GlobalTensor<Element> &sourceB,
                                        uint64_t offsetA, uint64_t offsetB,
                                        uint32_t l1AOffset, uint32_t l1BOffset,
                                        uint32_t m, uint32_t k, uint32_t aStride,
                                        uint32_t bRows, uint32_t n)
    {
        using L1A = typename TileCopy::LayoutTagL1A;
        using L1B = typename TileCopy::LayoutTagL1B;
        auto gmALayout = tla::MakeLayout<Element, LayoutRM>(Shape::kChunkRows, aStride);
        auto gmBLayout = tla::MakeLayout<Element, LayoutB>(bRows, n);
        auto gmATensor = tla::MakeTensor(sourceA[offsetA], gmALayout, Catlass::Arch::PositionGM{});
        auto gmBTensor = tla::MakeTensor(sourceB[offsetB], gmBLayout, Catlass::Arch::PositionGM{});
        auto gmABlock = GetTile(gmATensor, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto gmBBlock = GetTile(gmBTensor, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        using CopyGmA = typename TileCopy::template CopyGmToL1A<decltype(gmABlock)>;
        using CopyGmB = typename TileCopy::template CopyGmToL1B<decltype(gmBBlock)>;
        auto l1A = resource_.l1Buf.template GetBufferByByte<Element>(l1AOffset);
        auto l1B = resource_.l1Buf.template GetBufferByByte<Element>(l1BOffset);
        auto l1ATensor = tla::MakeTensor(l1A, tla::MakeLayout<Element, L1A>(Shape::kChunkRows, aStride),
                                        Catlass::Arch::PositionL1{});
        auto l1BTensor = tla::MakeTensor(l1B, tla::MakeLayout<Element, L1B>(bRows, n),
                                        Catlass::Arch::PositionL1{});
        CopyGmA{}(l1ATensor, gmABlock);
        if constexpr (!kFinalizeA5) {
            if (m < 16) {
                // A2/A3 MMAD 最少读取 16 行；zN 每个 K 分形占 64 个 32B 块。
                Fill(l1A[m * 16], InitConstValueParams<Element>(
                    static_cast<uint16_t>(CeilDiv(k, 16)), static_cast<uint16_t>(16 - m),
                    static_cast<uint16_t>(48 + m), static_cast<Element>(0.0f)));
            }
        }
        CopyGmB{}(l1BTensor, gmBBlock);
    }

    template <typename TileCopy, bool InitC>
    __aicore__ inline void ComputeProduct(uint32_t l1AOffset, uint32_t l1BOffset,
                                          uint32_t m, uint32_t k, uint32_t aStride,
                                          uint32_t bRows, uint32_t n,
                                          uint32_t operandPlane, uint32_t outputSlot)
    {
        using L1A = typename TileCopy::LayoutTagL1A;
        using L1B = typename TileCopy::LayoutTagL1B;
        using L0A = typename TileCopy::LayoutTagL0A;
        using L0B = typename TileCopy::LayoutTagL0B;
        using CopyA = typename TileCopy::CopyL1ToL0A;
        using CopyB = typename TileCopy::CopyL1ToL0B;
        using Mmad = Catlass::Gemm::Tile::TileMmadTla<FinalizeArch, Element, L1A>;

        auto l1A = resource_.l1Buf.template GetBufferByByte<Element>(l1AOffset);
        auto l1B = resource_.l1Buf.template GetBufferByByte<Element>(l1BOffset);
        auto l1ATensor = tla::MakeTensor(l1A, tla::MakeLayout<Element, L1A>(Shape::kChunkRows, aStride),
                                        Catlass::Arch::PositionL1{});
        auto l1BTensor = tla::MakeTensor(l1B, tla::MakeLayout<Element, L1B>(bRows, n),
                                        Catlass::Arch::PositionL1{});
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Lock<PIPE_MTE1>(static_cast<uint8_t>(4 + operandPlane));
#endif
        auto l0A = resource_.l0ABuf.template GetBufferByByte<Element>(kL0AOffset[operandPlane]);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<Element>(kL0BOffset[operandPlane]);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            outputSlot * Shape::kProductBytes);
        // 尾块的 L0 分形必须按真实 M/K 紧凑排列；以 64 行布局裁成 1 行
        // 会让 MMAD 把后续 K 分形误读为尚未写入的 L0 地址。
        auto tensorL0A = tla::MakeTensor(l0A, tla::MakeLayout<Element, L0A>(m, k),
                                        Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, tla::MakeLayout<Element, L0B>(k, n),
                                        Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n),
                                        Catlass::Arch::PositionL0C{});
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        if (l0OperandInFlight_[operandPlane]) {
            WaitFlag<HardEvent::M_MTE1>(operandPlane);
        }
#endif
        auto tileL1A = GetTile(l1ATensor, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(l1BTensor, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        CopyA{}(tileL0A, tileL1A);
        CopyB{}(tileL0B, tileL1B);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_MTE1>(static_cast<uint8_t>(4 + operandPlane));
        Mutex::Lock<PIPE_M>(static_cast<uint8_t>(4 + operandPlane));
#else
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        SetFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(0);
#endif
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        // A2/A3 的 MMAD 至少处理一个 16 行分形，尾块仍只写回有效行。
        const uint32_t madM = kFinalizeA5 ? m : (m < 16 ? 16 : m);
        Mmad{}(tileL0C, tileL0A, tileL0B, madM, n, k, InitC, 0);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_M>(static_cast<uint8_t>(4 + operandPlane));
#else
        SetFlag<HardEvent::M_MTE1>(operandPlane);
        l0OperandInFlight_[operandPlane] = true;
#endif
    }

    __aicore__ inline void StoreOutput(const FinalizeChunk &chunk,
                                       uint32_t head, uint32_t outputSlot)
    {
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            outputSlot * Shape::kProductBytes);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Lock<PIPE_FIX>(static_cast<uint8_t>(kL0CMutexBase + outputSlot));
#else
        WaitFlag<HardEvent::M_FIX>(outputSlot);
#endif
        constexpr uint32_t n = Shape::kHeadDim;
        const uint32_t dstStride = OutputSequenceMajor
                                       ? args_.tiling.valueHeadNum * n
                                       : n;
        auto fix = FixpipeParamsV220(
            n, chunk.validRows, CeilDiv(chunk.validRows, 16) * 16,
            dstStride, false);
        fix.quantPre = QuantMode_t::F322BF16;
        Fixpipe<Element, float, CFG_ROW_MAJOR>(
            out_[OutputOffset(args_, chunk, head, 0)], l0C, fix);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_FIX>(static_cast<uint8_t>(kL0CMutexBase + outputSlot));
#else
        SetFlag<HardEvent::FIX_M>(outputSlot);
        l0OutputInFlight_[outputSlot] = true;
#endif
    }

    __aicore__ inline void StageC0(const FinalizeChunk &chunk,
                                    uint32_t head, uint32_t localHead,
                                    uint32_t outputSlot)
    {
        const uint32_t lane = localHead * kL1HeadBytes;
        if constexpr (!UseAivInputMover) {
            const uint64_t qOffset = InputOffset(args_, chunk, head, Shape::kHeadDim);
            const uint64_t aOffset = InputOffset(args_, chunk, head, Shape::kChunkRows);
            const uint64_t hOffset = StateOffset(args_, chunk, head);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            Mutex::Lock<PIPE_MTE2>(0);
#endif
            LoadOperands<TileCopyQH, HLayout>(
                q_, h_, qOffset, hOffset, lane + kQOffset, lane + kHOffset,
                chunk.validRows, Shape::kHeadDim, Shape::kHeadDim,
                Shape::kHeadDim, Shape::kHeadDim);
            LoadOperands<TileCopyAV, LayoutRM>(
                a_, v_, aOffset, qOffset, lane + kAOffset, lane + kVOffset,
                chunk.validRows, chunk.validRows, Shape::kChunkRows,
                Shape::kChunkRows, Shape::kHeadDim);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            Mutex::Unlock<PIPE_MTE2>(0);
            Mutex::Lock<PIPE_MTE1>(0);
#else
            SetFlag<HardEvent::MTE2_MTE1>(0);
            WaitFlag<HardEvent::MTE2_MTE1>(0);
            if (l0OutputInFlight_[outputSlot]) {
                WaitFlag<HardEvent::FIX_M>(outputSlot);
                l0OutputInFlight_[outputSlot] = false;
            }
#endif
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            // Q/H 先 ready，第一路 MTE1/MMAD 可与 AIV 的 Aqk/V 搬运重叠。
            CrossCoreWaitFlag<A5Sync::kCrossCoreMode, PIPE_MTE1>(
                A5Sync::kAicQhReadyFlagId[localHead]);
            Mutex::Lock<PIPE_MTE1>(0);
#endif
        }
        // 第一项初始化 L0C，第二项直接累加；两个槽仅用于跨 head 的 MMAD/Fixpipe 流水。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Lock<PIPE_M>(static_cast<uint8_t>(kL0CMutexBase + outputSlot));
#endif
        ComputeProduct<TileCopyQH, true>(lane + kQOffset, lane + kHOffset,
                                         chunk.validRows, Shape::kHeadDim,
                                         Shape::kHeadDim, Shape::kHeadDim,
                                         Shape::kHeadDim, 0, outputSlot);
        if constexpr (UseAivInputMover) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            CrossCoreWaitFlag<A5Sync::kCrossCoreMode, PIPE_MTE1>(
                A5Sync::kAicAvReadyFlagId[localHead]);
#endif
        }
        ComputeProduct<TileCopyAV, false>(lane + kAOffset, lane + kVOffset,
                                          chunk.validRows, chunk.validRows,
                                          Shape::kChunkRows, Shape::kChunkRows,
                                          Shape::kHeadDim, 1, outputSlot);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_MTE1>(0);
        if constexpr (UseAivInputMover) {
            // Q/H 与 Aqk/V 均已离开 L1，AIV 可提前复用该 head 槽。
            CrossCoreSetFlag<A5Sync::kCrossCoreMode, PIPE_MTE1>(
                A5Sync::kAicL1ReusableFlagId[localHead]);
        }
        Mutex::Unlock<PIPE_M>(static_cast<uint8_t>(kL0CMutexBase + outputSlot));
#else
        SetFlag<HardEvent::M_FIX>(outputSlot);
#endif
        StoreOutput(chunk, head, outputSlot);
    }

    FinalizeArgs args_{};
    uint32_t core_ = 0;
    bool l0OperandInFlight_[2] = {false, false};
    bool l0OutputInFlight_[2] = {false, false};
    GlobalTensor<Element> q_;
    GlobalTensor<Element> a_;
    GlobalTensor<Element> v_;
    GlobalTensor<Element> h_;
    GlobalTensor<Element> out_;
    Catlass::Arch::Resource<FinalizeArch> resource_;
};

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_CUBE_H
