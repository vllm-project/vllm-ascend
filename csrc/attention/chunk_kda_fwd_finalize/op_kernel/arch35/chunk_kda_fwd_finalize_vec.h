/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH35_CHUNK_KDA_FWD_FINALIZE_VEC_H
#define ARCH35_CHUNK_KDA_FWD_FINALIZE_VEC_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "../chunk_kda_fwd_finalize_struct.h"

namespace KdaFinalize::Arch35 {

using namespace AscendC;

template <bool StateVFirst>
class FinalizeInputMover {
    using Element = bfloat16_t;

    // 每个 UB 槽的行都增加一个 32B datablock：128 列 pitch=9，64 列 pitch=5。
    // ping/pong 基址相隔 128 KiB，使并发 MTE2/MTE3 访问配对 bank。
    static constexpr uint32_t kUbSlotBase[2] = {0, 128 * 1024};
    static constexpr uint8_t kUbQhMutexId[2] = {0, 1};
    static constexpr uint8_t kUbAvMutexId[2] = {2, 3};
    static constexpr uint32_t kUbQOffset = 0;
    static constexpr uint32_t kUbHOffset = kUbQOffset + 18 * 1024;
    static constexpr uint32_t kUbAOffset = kUbHOffset + 36 * 1024;
    static constexpr uint32_t kUbVOffset = kUbAOffset + 10 * 1024;

    static constexpr uint32_t kL1QOffset = 0;
    static constexpr uint32_t kL1HOffset = kL1QOffset + 16 * 1024;
    static constexpr uint32_t kL1AOffset = kL1HOffset + 32 * 1024;
    static constexpr uint32_t kL1VOffset = kL1AOffset + 8 * 1024;
    static constexpr uint32_t kL1HeadBytes = 72 * 1024;

public:
    __aicore__ inline void Init(const FinalizeArgs &args)
    {
        args_ = args;
        workgroup_ = WorkgroupId();
        aiv_ = static_cast<uint32_t>(GetSubBlockIdx());
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.qgScaled));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.aqk));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.vNew));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.h));
        q_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        v_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        h_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        l1Buffer_ = LocalTensor<uint8_t>(TPosition::A1, 0, 512 * 1024);
    }

    __aicore__ inline void Process()
    {
        if (workgroup_ >= args_.tiling.usedCoreNum || args_.tiling.usedCoreNum == 0) {
            return;
        }
        bool l1SlotInFlight[2] = {false, false};
        uint32_t ubSlot = 0;
        const uint32_t partitions = CeilDiv(args_.tiling.valueHeadNum,
                                            args_.tiling.headsPerPartition);
        const uint32_t total = TotalWorkItems(args_);
        const uint32_t begin = WorkBegin(total, workgroup_, args_.tiling.usedCoreNum);
        const uint32_t end = WorkEnd(total, workgroup_, args_.tiling.usedCoreNum);
        // 与 AIC 使用相同的主体 chunk / 尾部 4-head 组映射和消费顺序。
        const uint32_t cores = args_.tiling.usedCoreNum;
        const bool balanceTail = !args_.tiling.isVarLen &&
            partitions == 1 && total >= cores && total % cores != 0;
        const uint32_t fullChunksPerCore = total / cores;
        const uint32_t headGroups = CeilDiv(args_.tiling.valueHeadNum, Shape::kHeadsPerGroup);
        const uint32_t tailGroups = (total % cores) * headGroups;
        const uint32_t tailBegin = WorkBegin(tailGroups, workgroup_, cores);
        const uint32_t tailEnd = WorkEnd(tailGroups, workgroup_, cores);
        const uint32_t workCount = balanceTail
            ? fullChunksPerCore + tailEnd - tailBegin : end - begin;
        for (uint32_t index = 0; index < workCount; ++index) {
            const uint32_t work = begin + index;
            uint32_t chunkId = work / partitions;
            uint32_t headBegin = (work % partitions) * args_.tiling.headsPerPartition;
            uint32_t headEnd = headBegin + args_.tiling.headsPerPartition;
            if (balanceTail) {
                if (index < fullChunksPerCore) {
                    chunkId = workgroup_ * fullChunksPerCore + index;
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
            for (uint32_t group = headBegin; group < headEnd;
                 group += Shape::kHeadsPerGroup) {
                uint32_t active = headEnd - group;
                if (active > Shape::kHeadsPerGroup) {
                    active = Shape::kHeadsPerGroup;
                }
                // AIV0 负责 localHead 0/2，AIV1 负责 1/3；AIC 按 0,1,2,3 消费。
                for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
                    const uint32_t localHead = localSlot * 2 + aiv_;
                    if (localHead >= active) {
                        continue;
                    }
                    const uint32_t head = group + localHead;
                    LoadQhInputsToUb(chunk, head, ubSlot);
                    PublishQhInputsToL1(chunk, localHead, localSlot, ubSlot,
                                        l1SlotInFlight[localSlot]);
                    LoadAvInputsToUb(chunk, head, ubSlot);
                    PublishAvInputsToL1(chunk, localHead, localSlot, ubSlot);
                    l1SlotInFlight[localSlot] = true;
                    ubSlot ^= 1;
                }
            }
        }
        // 收掉最后一次 free，保证每个 ready 都有对应消费，flag 可安全复用。
        for (uint32_t localSlot = 0; localSlot < 2; ++localSlot) {
            if (l1SlotInFlight[localSlot]) {
                CrossCoreWaitFlag<A5Sync::kCrossCoreMode, PIPE_MTE2>(
                    A5Sync::kAivL1ReusableFlagId[localSlot]);
            }
        }
    }

private:
    __aicore__ inline void LoadQhInputsToUb(const FinalizeChunk &chunk,
                                            uint32_t head, uint32_t ubSlot)
    {
        const uint32_t base = kUbSlotBase[ubSlot];
        auto q = resource_.ubBuf.template GetBufferByByte<Element>(base + kUbQOffset);
        auto h = resource_.ubBuf.template GetBufferByByte<Element>(base + kUbHOffset);
        const uint64_t qOffset = InputOffset(args_, chunk, head, Shape::kHeadDim);
        const uint64_t hOffset = StateOffset(args_, chunk, head);
        const DataCopyParams qvParams{
            static_cast<uint16_t>(chunk.validRows), 8, 0, 1};
        const DataCopyParams hParams{128, 8, 0, 1};

        Mutex::Lock<PIPE_MTE2>(kUbQhMutexId[ubSlot]);
        DataCopy(q, q_[qOffset], qvParams);
        DataCopy(h, h_[hOffset], hParams);
        Mutex::Unlock<PIPE_MTE2>(kUbQhMutexId[ubSlot]);
    }

    __aicore__ inline void LoadAvInputsToUb(const FinalizeChunk &chunk,
                                            uint32_t head, uint32_t ubSlot)
    {
        const uint32_t base = kUbSlotBase[ubSlot];
        auto a = resource_.ubBuf.template GetBufferByByte<Element>(base + kUbAOffset);
        auto v = resource_.ubBuf.template GetBufferByByte<Element>(base + kUbVOffset);
        const uint64_t qOffset = InputOffset(args_, chunk, head, Shape::kHeadDim);
        const uint64_t aOffset = InputOffset(args_, chunk, head, Shape::kChunkRows);
        const uint16_t aBlocks = static_cast<uint16_t>(CeilDiv(chunk.validRows, 16));
        const DataCopyParams qvParams{
            static_cast<uint16_t>(chunk.validRows), 8, 0, 1};
        const DataCopyParams aParams{
            static_cast<uint16_t>(chunk.validRows), aBlocks,
            static_cast<uint16_t>(4 - aBlocks),
            static_cast<uint16_t>(5 - aBlocks)};

        Mutex::Lock<PIPE_MTE2>(kUbAvMutexId[ubSlot]);
        DataCopy(a, a_[aOffset], aParams);
        DataCopy(v, v_[qOffset], qvParams);
        Mutex::Unlock<PIPE_MTE2>(kUbAvMutexId[ubSlot]);
    }

    __aicore__ inline void CopyPaddedRowsToL1Zn(
        LocalTensor<Element> dstL1, LocalTensor<Element> srcUb,
        uint32_t rows, uint32_t columns, uint32_t l1PaddedRows,
        uint16_t ubRowBlocks)
    {
        constexpr uint32_t kC0Elements = 16;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        const DataCopyParams params{
            static_cast<uint16_t>(rows), 1,
            static_cast<uint16_t>(ubRowBlocks - 1), 0};
        for (uint32_t column = 0; column < columns; column += kC0Elements) {
            const uint32_t l1Offset =
                (column / kC0Elements) * l1PaddedRows * kC0Elements;
            DataCopy(dstL1[l1Offset], srcUb[column], params, enhanced);
        }
    }

    __aicore__ inline void CopyColumnMajorHToL1Nz(
        LocalTensor<Element> dstL1, LocalTensor<Element> srcUb)
    {
        constexpr uint32_t kC0Elements = 16;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        // UB 的一个物理行对应 H 的一列，144 BF16 pitch 为 9 个 datablock。
        const DataCopyParams params{128, 1, 8, 0};
        #pragma unroll 8
        for (uint32_t row = 0; row < Shape::kHeadDim; row += kC0Elements) {
            const uint32_t l1Offset =
                (row / kC0Elements) * Shape::kHeadDim * kC0Elements;
            DataCopy(dstL1[l1Offset], srcUb[row], params, enhanced);
        }
    }

    __aicore__ inline void PublishQhInputsToL1(
        const FinalizeChunk &chunk, uint32_t localHead, uint32_t localSlot,
        uint32_t ubSlot, bool waitL1Free)
    {
        const uint32_t ubBase = kUbSlotBase[ubSlot];
        auto qUb = resource_.ubBuf.template GetBufferByByte<Element>(ubBase + kUbQOffset);
        auto hUb = resource_.ubBuf.template GetBufferByByte<Element>(ubBase + kUbHOffset);
        const uint32_t l1Base = localHead * kL1HeadBytes;
        auto qL1 = l1Buffer_[l1Base + kL1QOffset].template ReinterpretCast<Element>();
        auto hL1 = l1Buffer_[l1Base + kL1HOffset].template ReinterpretCast<Element>();

        if (waitL1Free) {
            CrossCoreWaitFlag<A5Sync::kCrossCoreMode, PIPE_MTE3>(
                A5Sync::kAivL1ReusableFlagId[localSlot]);
        }
        Mutex::Lock<PIPE_MTE3>(kUbQhMutexId[ubSlot]);
        CopyPaddedRowsToL1Zn(qL1, qUb, chunk.validRows, Shape::kHeadDim,
                             Shape::kChunkRows, 9);
        if constexpr (StateVFirst) {
            CopyColumnMajorHToL1Nz(hL1, hUb);
        } else {
            CopyPaddedRowsToL1Zn(hL1, hUb, Shape::kHeadDim, Shape::kHeadDim,
                                 Shape::kHeadDim, 9);
        }
        Mutex::Unlock<PIPE_MTE3>(kUbQhMutexId[ubSlot]);
        CrossCoreSetFlag<A5Sync::kCrossCoreMode, PIPE_MTE3>(
            A5Sync::kAivQhReadyFlagId[localSlot]);
    }

    __aicore__ inline void PublishAvInputsToL1(
        const FinalizeChunk &chunk, uint32_t localHead, uint32_t localSlot,
        uint32_t ubSlot)
    {
        const uint32_t ubBase = kUbSlotBase[ubSlot];
        auto aUb = resource_.ubBuf.template GetBufferByByte<Element>(ubBase + kUbAOffset);
        auto vUb = resource_.ubBuf.template GetBufferByByte<Element>(ubBase + kUbVOffset);
        const uint32_t l1Base = localHead * kL1HeadBytes;
        auto aL1 = l1Buffer_[l1Base + kL1AOffset].template ReinterpretCast<Element>();
        auto vL1 = l1Buffer_[l1Base + kL1VOffset].template ReinterpretCast<Element>();

        Mutex::Lock<PIPE_MTE3>(kUbAvMutexId[ubSlot]);
        CopyPaddedRowsToL1Zn(
            aL1, aUb, chunk.validRows, CeilDiv(chunk.validRows, 16) * 16,
            Shape::kChunkRows, 5);
        CopyPaddedRowsToL1Zn(vL1, vUb, chunk.validRows, Shape::kHeadDim,
                             Shape::kChunkRows, 9);
        Mutex::Unlock<PIPE_MTE3>(kUbAvMutexId[ubSlot]);
        CrossCoreSetFlag<A5Sync::kCrossCoreMode, PIPE_MTE3>(
            A5Sync::kAivAvReadyFlagId[localSlot]);
    }

    FinalizeArgs args_{};
    uint32_t workgroup_ = 0;
    uint32_t aiv_ = 0;
    GlobalTensor<Element> q_;
    GlobalTensor<Element> a_;
    GlobalTensor<Element> v_;
    GlobalTensor<Element> h_;
    LocalTensor<uint8_t> l1Buffer_;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_;
};

} // namespace KdaFinalize::Arch35

#endif // ARCH35_CHUNK_KDA_FWD_FINALIZE_VEC_H
