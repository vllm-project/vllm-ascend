/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GBSA_SPARSE_PAIR_COPY_HPP
#define GBSA_SPARSE_PAIR_COPY_HPP

namespace GbsaSparsePairCopy {

template <bool IS_QK, class Element, class Layout>
__aicore__ inline void Emit(AscendC::LocalTensor<Element> dst, AscendC::GlobalTensor<Element> src, const Layout &layout,
                            uint32_t dstBegin, uint64_t source, uint32_t rows, uint32_t ndNum, uint32_t sourceStride,
                            uint32_t embed, uint32_t tokenStride)
{
    constexpr uint32_t c0 = 32U / sizeof(Element);
    AscendC::Nd2NzParams params;
    params.ndNum = ndNum;
    params.nValue = rows;
    params.dValue = embed;
    params.srcDValue = tokenStride;
    params.srcNdMatrixStride = sourceStride;
    params.dstNzC0Stride = (IS_QK ? layout.stride(1) : layout.stride(3)) / c0;
    params.dstNzNStride = (IS_QK ? layout.stride(2) : layout.stride(0)) / c0;
    params.dstNzMatrixStride = ndNum == 2U ? rows * c0 : 0U;
    const uint32_t offset = IS_QK ? layout.GetOffset(NpuArch::MatrixCoord{0, dstBegin}) :
                                    layout.GetOffset(NpuArch::MatrixCoord{dstBegin, 0});
    AscendC::DataCopy(dst[offset], src[source], params);
}

constexpr uint32_t KV_COPY_SUB_TILE_TOKENS = 128U;
constexpr uint32_t KV_COPY_MAX_TILE_TOKENS = 512U;
constexpr uint32_t KV_COPY_MIN_BLOCK_Y = 1U;
constexpr uint32_t KV_COPY_INVALID_COUNT = KV_COPY_SUB_TILE_TOKENS / KV_COPY_MIN_BLOCK_Y + 1U;
struct KvCopySegment {
    uint64_t source;
    uint16_t stride;
    uint16_t rows;
    uint16_t dstRow;
    uint16_t ndNum;
};
static_assert(sizeof(KvCopySegment) == 16U, "KV copy segment must remain 16 bytes");
struct KvCopyInfo {
    KvCopySegment *records;
    uint32_t count;
};

__aicore__ inline uint32_t Log2PowerOfTwo(uint32_t value)
{
    return static_cast<uint32_t>((value & 0xAAAAAAAAU) != 0U) |
           (static_cast<uint32_t>((value & 0xCCCCCCCCU) != 0U) << 1U) |
           (static_cast<uint32_t>((value & 0xF0F0F0F0U) != 0U) << 2U) |
           (static_cast<uint32_t>((value & 0xFF00FF00U) != 0U) << 3U) |
           (static_cast<uint32_t>((value & 0xFFFF0000U) != 0U) << 4U);
}

struct KvCopyGeometry {
    uint32_t pageBlockShift;
    uint32_t pageBlockMask;
    uint32_t blockStrideShift;
    uint64_t blockStride;
    bool shiftBlockStride;

    __aicore__ inline uint64_t InnerOffset(uint32_t block) const
    {
        const uint64_t inPage = block & pageBlockMask;
        return shiftBlockStride ? (inPage << blockStrideShift) : inPage * blockStride;
    }
};

__aicore__ inline bool MakeKvCopyGeometry(uint32_t blockY, uint32_t tokenStride, KvCopyGeometry &geometry)
{
    if (blockY == 0U || blockY > 128U || (blockY & (blockY - 1U)) != 0U) {
        return false;
    }
    const uint32_t yShift = Log2PowerOfTwo(blockY);
    geometry.pageBlockShift = 7U - yShift;
    geometry.pageBlockMask = (1U << geometry.pageBlockShift) - 1U;
    geometry.blockStride = static_cast<uint64_t>(blockY) * tokenStride;
    geometry.shiftBlockStride = tokenStride != 0U && (tokenStride & (tokenStride - 1U)) == 0U;
    geometry.blockStrideShift = geometry.shiftBlockStride ? yShift + Log2PowerOfTwo(tokenStride) : 0U;
    return true;
}

template <bool IS_QK, class Element, class Layout>
__aicore__ inline void EmitAndRecord(AscendC::LocalTensor<Element> dst,
    AscendC::GlobalTensor<Element> src, const Layout &layout, uint32_t dstBegin,
    uint64_t source, uint64_t peerSource, uint32_t rows, uint32_t ndNum,
    uint32_t sourceStride, uint32_t peerStride, uint32_t embed, uint32_t tokenStride,
    KvCopyInfo *plan, uint32_t recordBase)
{
    Emit<IS_QK>(dst, src, layout, dstBegin, source, rows, ndNum, sourceStride, embed, tokenStride);
    if constexpr (IS_QK) {
        if (plan != nullptr) {
            auto &r = plan->records[plan->count++];
            r.source = peerSource;
            r.stride = static_cast<uint16_t>(peerStride);
            r.rows = static_cast<uint16_t>(rows);
            r.dstRow = static_cast<uint16_t>(dstBegin - recordBase);
            r.ndNum = static_cast<uint16_t>(ndNum);
        }
    }
}

template <bool IS_QK, class Element, class Layout>
__aicore__ inline bool TryEmitDenseRun(AscendC::LocalTensor<Element> dst, AscendC::GlobalTensor<Element> src,
                                       const Layout &layout, AscendC::GlobalTensor<int32_t> indices,
                                       AscendC::GlobalTensor<int32_t> table, uint32_t firstSlot, uint32_t blocks,
                                       uint32_t dstBegin, uint32_t embed, uint32_t tokenStride, uint64_t pageStride,
                                       uint32_t blockY, uint32_t kvYBlockNum, uint32_t kvSeqlen,
                                       const KvCopyGeometry &geometry)
{
    if (blocks == 0U) {
        return true;
    }
    const uint32_t blocksPerPage = geometry.pageBlockMask + 1U;
    const __gm__ int32_t *idx = indices.GetPhyAddr(firstSlot);
    const __gm__ int32_t *pages = table.GetPhyAddr();
    const uint32_t first = static_cast<uint32_t>(idx[0]);
    const uint32_t last = static_cast<uint32_t>(idx[blocks - 1U]);
    if (last < first || last - first + 1U != blocks || last >= kvYBlockNum ||
        (static_cast<uint64_t>(last) + 1U) * blockY > kvSeqlen) {
        return false;
    }
    for (uint32_t i = 1U; i + 1U < blocks; ++i) {
        if (static_cast<uint32_t>(idx[i]) != first + i) {
            return false;
        }
    }
    uint32_t block = first;
    uint32_t remain = blocks;
    uint32_t dstRow = dstBegin;
    uint32_t cachedLogicalPage = 0xFFFFFFFFU;
    uint32_t cachedPhysicalPage = 0U;
    while (remain != 0U) {
        const uint32_t inPage = block & geometry.pageBlockMask;
        const uint32_t room = blocksPerPage - inPage;
        const uint32_t take = remain < room ? remain : room;
        const uint32_t logicalPage = block >> geometry.pageBlockShift;
        if (logicalPage != cachedLogicalPage) {
            cachedLogicalPage = logicalPage;
            cachedPhysicalPage = static_cast<uint32_t>(pages[logicalPage]);
        }
        const uint32_t page = cachedPhysicalPage;
        const uint64_t inner = geometry.InnerOffset(block);
        Emit<IS_QK>(dst, src, layout, dstRow, static_cast<uint64_t>(page) * pageStride + inner, take * blockY, 1U, 0U,
                    embed, tokenStride);
        dstRow += take * blockY;
        block += take;
        remain -= take;
    }
    return true;
}

template <bool IS_QK, class Element, class Layout>
__aicore__ inline void Load(AscendC::LocalTensor<Element> dst, AscendC::GlobalTensor<Element> src, const Layout &layout,
                            AscendC::GlobalTensor<int32_t> indices, AscendC::GlobalTensor<int32_t> table,
                            uint32_t firstSlot, uint32_t blocks, uint32_t dstBegin, uint32_t embed,
                            uint32_t tokenStride, uint64_t pageStride, uint64_t peerPageStride, uint32_t blockY,
                            uint32_t kvYBlockNum, uint32_t kvSeqlen, KvCopyInfo *plan = nullptr,
                            uint32_t recordBase = 0U)
{
    KvCopyGeometry geometry;
    if (!MakeKvCopyGeometry(blockY, tokenStride, geometry)) {
        return;
    }
    if (TryEmitDenseRun<IS_QK>(dst, src, layout, indices, table, firstSlot, blocks, dstBegin, embed, tokenStride,
                               pageStride, blockY, kvYBlockNum, kvSeqlen, geometry)) {
        return;
    }
    const __gm__ int32_t *idx = indices.GetPhyAddr(firstSlot);
    const __gm__ int32_t *pages = table.GetPhyAddr();
    uint32_t cachedLogicalPage = 0xFFFFFFFFU;
    uint32_t cachedPhysicalPage = 0U;
    if (plan != nullptr) {
        plan->count = 0U;
    }
    const uint32_t validByLength = kvSeqlen == 0U ? 0U : ((kvSeqlen - 1U) >> (7U - geometry.pageBlockShift)) + 1U;
    const uint32_t validBlocks = kvYBlockNum < validByLength ? kvYBlockNum : validByLength;
    uint64_t pending = 0, pendingPeer = 0, pendingSpan = 0;
    uint32_t length = 0, dealt = 0;
    for (uint32_t i = 0; i < blocks; ++i) {
        const uint32_t block = static_cast<uint32_t>(idx[i]);
        if (block >= validBlocks) {
            break;
        }
        const uint32_t logicalPage = block >> geometry.pageBlockShift;
        if (logicalPage != cachedLogicalPage) {
            cachedLogicalPage = logicalPage;
            cachedPhysicalPage = static_cast<uint32_t>(pages[logicalPage]);
        }
        const uint32_t page = cachedPhysicalPage;
        const uint64_t inner = geometry.InnerOffset(block);
        const uint64_t source = static_cast<uint64_t>(page) * pageStride + inner;
        const uint64_t peer = static_cast<uint64_t>(page) * peerPageStride + inner;
        if (length == 0U) {
            pending = source;
            pendingPeer = peer;
            length = blockY;
            pendingSpan = geometry.blockStride;
        } else if (source == pending + pendingSpan && peer == pendingPeer + pendingSpan) {
            length += blockY;
            pendingSpan += geometry.blockStride;
        } else {
            const bool reverse = source < pending;
            const uint64_t gap = reverse ? pending - source : source - pending;
            const uint64_t peerGap = peer < pendingPeer ? pendingPeer - peer : peer - pendingPeer;
            if (length == blockY && reverse == (peer < pendingPeer) && gap != 0U && gap < 65536U && peerGap < 65536U &&
                (gap & 15U) == 0U && (peerGap & 15U) == 0U) {
                EmitAndRecord<IS_QK>(dst, src, layout, dstBegin + dealt, reverse ? source : pending,
                                     reverse ? peer : pendingPeer, blockY, 2U, static_cast<uint32_t>(gap),
                                     static_cast<uint32_t>(peerGap), embed, tokenStride, plan, recordBase);
                dealt += blockY * 2U;
                length = 0U;
            } else {
                EmitAndRecord<IS_QK>(dst, src, layout, dstBegin + dealt, pending, pendingPeer, length, 1U, 0U, 0U,
                                     embed, tokenStride, plan, recordBase);
                dealt += length;
                pending = source;
                pendingPeer = peer;
                length = blockY;
                pendingSpan = geometry.blockStride;
            }
        }
    }
    if (length != 0U) {
        EmitAndRecord<IS_QK>(dst, src, layout, dstBegin + dealt, pending, pendingPeer, length, 1U, 0U, 0U, embed,
                             tokenStride, plan, recordBase);
    }
}

} // namespace GbsaSparsePairCopy
#endif
