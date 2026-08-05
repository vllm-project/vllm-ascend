/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef SPARSE_KV_GATHER_KERNEL_H
#define SPARSE_KV_GATHER_KERNEL_H

#include "kernel_operator.h"

namespace BaseApi {

using namespace AscendC;

constexpr uint32_t SKG_LOCAL_CTKV_DIM = 512;
constexpr uint32_t SKG_LOCAL_KPE_DIM = 64;
constexpr uint32_t SKG_LOCAL_COMBINED_DIM =
    SKG_LOCAL_CTKV_DIM + SKG_LOCAL_KPE_DIM;
constexpr uint32_t SKG_LOCAL_STAGE_ROWS = 16;
constexpr uint32_t SKG_LOCAL_PAIR_WIDTH = 2;
constexpr uint32_t SKG_LOCAL_CTKV_STAGE_DIM =
    SKG_LOCAL_STAGE_ROWS * SKG_LOCAL_CTKV_DIM;
constexpr uint32_t SKG_LOCAL_KPE_STAGE_DIM =
    SKG_LOCAL_STAGE_ROWS * SKG_LOCAL_KPE_DIM;
constexpr uint32_t SKG_LOCAL_STAGE_DIM =
    SKG_LOCAL_CTKV_STAGE_DIM + SKG_LOCAL_KPE_STAGE_DIM;
constexpr uint32_t SKG_LOCAL_BLOCK_SIZE = 128;
constexpr uint32_t SKG_LOCAL_BLOCK_SHIFT = 7;
constexpr uint32_t SKG_LOCAL_BLOCK_MASK = SKG_LOCAL_BLOCK_SIZE - 1;
constexpr uint32_t SKG_STAGE_BUFFER_NUM = 2;
constexpr uint32_t SKG_UINT16_BYTES = sizeof(uint16_t);
constexpr uint32_t SKG_CTKV_ROW_BYTES =
    SKG_LOCAL_CTKV_DIM * SKG_UINT16_BYTES;
constexpr uint32_t SKG_KPE_ROW_BYTES =
    SKG_LOCAL_KPE_DIM * SKG_UINT16_BYTES;
constexpr uint64_t SKG_MAX_DMA_EXT_STRIDE = 0xFFFFFFFFULL;
constexpr bool SKG_ENABLE_PAIR_MOVE = true;

// Index-type encoding, must match Host SKGIndexType.
constexpr uint32_t SKG_INDEX_TYPE_INT32 = 0U;
constexpr uint32_t SKG_INDEX_TYPE_INT64 = 1U;

class SparseKvGatherKernel {
public:
    __aicore__ inline SparseKvGatherKernel() = default;

    __aicore__ inline void Init(
        __gm__ uint8_t *pagedCtkv,
        __gm__ uint8_t *pagedKpe,
        __gm__ uint8_t *blockTable,
        __gm__ uint8_t *topkIndices,
        __gm__ uint8_t *curPos,
        __gm__ uint8_t *outCtkv,
        __gm__ uint8_t *outKpe,
        uint32_t numBlocks,
        uint32_t maxBlocks,
        uint32_t topkN,
        uint64_t totalSlots,
        uint64_t slotsPerCore,
        uint32_t usedCoreNum,
        uint32_t blockTableType,
        uint32_t topkIndicesType,
        uint32_t curPosType,
        TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t ReadIndex(
        const GlobalTensor<int32_t> &tensorI32,
        const GlobalTensor<int64_t> &tensorI64,
        uint32_t type,
        uint64_t offset) const;

    template <typename IndexT>
    __aicore__ inline int64_t ReadIndexTyped(
        const GlobalTensor<IndexT> &tensor,
        uint64_t offset) const;

    __aicore__ inline int64_t ResolvePhysicalToken(
        uint32_t queryIdx,
        int64_t logicalToken) const;

    template <typename IndexT>
    __aicore__ inline int64_t ResolvePhysicalTokenTyped(
        uint32_t queryIdx,
        int64_t logicalToken,
        const GlobalTensor<IndexT> &blockTable) const;

    __aicore__ inline void WriteZero(
        uint64_t flatSlot,
        const LocalTensor<uint16_t> &zeroUb) const;

    __aicore__ inline bool CanGatherPair(
        int64_t physicalToken0,
        int64_t physicalToken1) const;

    __aicore__ inline void LoadOneToStage(
        uint32_t row,
        int64_t physicalToken,
        LocalTensor<uint16_t> &ctkvUb,
        LocalTensor<uint16_t> &kpeUb) const;

    __aicore__ inline void LoadPairToStage(
        uint32_t row,
        int64_t physicalToken0,
        int64_t physicalToken1,
        LocalTensor<uint16_t> &ctkvUb,
        LocalTensor<uint16_t> &kpeUb) const;

    __aicore__ inline void GatherOne(
        uint64_t flatSlot,
        int64_t physicalToken,
        uint32_t pingPong,
        LocalTensor<uint16_t> &stageUb);

    __aicore__ inline void GatherPair(
        uint64_t flatSlot,
        int64_t physicalToken0,
        int64_t physicalToken1,
        uint32_t pingPong,
        LocalTensor<uint16_t> &stageUb);

    __aicore__ inline void GatherFullValidChunk(
        uint64_t flatSlot,
        uint32_t dealRows,
        const int64_t *physicalTokens,
        uint32_t pingPong,
        LocalTensor<uint16_t> &stageUb);

    template <typename IndexT>
    __aicore__ inline void ProcessTyped(
        const GlobalTensor<IndexT> &blockTable,
        const GlobalTensor<IndexT> &topkIndices,
        const GlobalTensor<IndexT> &curPos);

    __aicore__ inline void ProcessGeneric();

    GlobalTensor<uint16_t> pagedCtkvGm_;
    GlobalTensor<uint16_t> pagedKpeGm_;
    GlobalTensor<uint16_t> outCtkvGm_;
    GlobalTensor<uint16_t> outKpeGm_;

    GlobalTensor<int32_t> blockTableI32Gm_;
    GlobalTensor<int64_t> blockTableI64Gm_;
    GlobalTensor<int32_t> topkIndicesI32Gm_;
    GlobalTensor<int64_t> topkIndicesI64Gm_;
    GlobalTensor<int32_t> curPosI32Gm_;
    GlobalTensor<int64_t> curPosI64Gm_;

    uint32_t numBlocks_ = 0;
    uint32_t maxBlocks_ = 0;
    uint32_t topkN_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t blockTableType_ = 0;
    uint32_t topkIndicesType_ = 0;
    uint32_t curPosType_ = 0;

    uint64_t totalSlots_ = 0;
    uint64_t slotsPerCore_ = 0;

    uint32_t coreIdx_ = 0;

    TPipe *pipe_ = nullptr;
    TBuf<> stageBuf_;
    TBuf<> zeroBuf_;

    TEventID mte2ToMte3_[SKG_STAGE_BUFFER_NUM];
    TEventID mte3ToMte2_[SKG_STAGE_BUFFER_NUM];
    TEventID vectorToMte3_;
};

__aicore__ inline void SparseKvGatherKernel::Init(
    __gm__ uint8_t *pagedCtkv,
    __gm__ uint8_t *pagedKpe,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *topkIndices,
    __gm__ uint8_t *curPos,
    __gm__ uint8_t *outCtkv,
    __gm__ uint8_t *outKpe,
    const uint32_t numBlocks,
    const uint32_t maxBlocks,
    const uint32_t topkN,
    const uint64_t totalSlots,
    const uint64_t slotsPerCore,
    const uint32_t usedCoreNum,
    const uint32_t blockTableType,
    const uint32_t topkIndicesType,
    const uint32_t curPosType,
    TPipe *pipe)
{
    pipe_ = pipe;

    numBlocks_ = numBlocks;
    maxBlocks_ = maxBlocks;
    topkN_ = topkN;
    totalSlots_ = totalSlots;
    slotsPerCore_ = slotsPerCore;
    usedCoreNum_ = usedCoreNum;
    blockTableType_ = blockTableType;
    topkIndicesType_ = topkIndicesType;
    curPosType_ = curPosType;

    coreIdx_ = GetBlockIdx();

    pagedCtkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(pagedCtkv));
    pagedKpeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(pagedKpe));
    outCtkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(outCtkv));
    outKpeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(outKpe));

    blockTableI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
    blockTableI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(blockTable));
    topkIndicesI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(topkIndices));
    topkIndicesI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(topkIndices));
    curPosI32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(curPos));
    curPosI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(curPos));

    pipe_->InitBuffer(
        stageBuf_, SKG_STAGE_BUFFER_NUM * SKG_LOCAL_STAGE_DIM * sizeof(uint16_t));
    pipe_->InitBuffer(zeroBuf_, SKG_LOCAL_COMBINED_DIM * sizeof(uint16_t));

    for (uint32_t i = 0; i < SKG_STAGE_BUFFER_NUM; ++i) {
        mte2ToMte3_[i] = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2_[i] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }
    vectorToMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
}

__aicore__ inline int64_t SparseKvGatherKernel::ReadIndex(
    const GlobalTensor<int32_t> &tensorI32,
    const GlobalTensor<int64_t> &tensorI64,
    const uint32_t type,
    const uint64_t offset) const
{
    if (type == SKG_INDEX_TYPE_INT64) {
        return tensorI64.GetValue(offset);
    }
    return static_cast<int64_t>(tensorI32.GetValue(offset));
}

template <typename IndexT>
__aicore__ inline int64_t SparseKvGatherKernel::ReadIndexTyped(
    const GlobalTensor<IndexT> &tensor,
    const uint64_t offset) const
{
    return static_cast<int64_t>(tensor.GetValue(offset));
}

__aicore__ inline int64_t SparseKvGatherKernel::ResolvePhysicalToken(
    const uint32_t queryIdx,
    const int64_t logicalToken) const
{
    if (logicalToken < 0) {
        return -1;
    }

    const uint64_t logicalBlock =
        static_cast<uint64_t>(logicalToken) >> SKG_LOCAL_BLOCK_SHIFT;
    if (logicalBlock >= maxBlocks_) {
        return -1;
    }

    const uint64_t blockTableOffset =
        static_cast<uint64_t>(queryIdx) * maxBlocks_ + logicalBlock;
    const int64_t physicalBlock = ReadIndex(
        blockTableI32Gm_, blockTableI64Gm_, blockTableType_, blockTableOffset);
    if (physicalBlock < 0 || physicalBlock >= static_cast<int64_t>(numBlocks_)) {
        return -1;
    }

    const uint64_t blockOffset =
        static_cast<uint64_t>(logicalToken) & SKG_LOCAL_BLOCK_MASK;
    return physicalBlock * static_cast<int64_t>(SKG_LOCAL_BLOCK_SIZE) +
           static_cast<int64_t>(blockOffset);
}

template <typename IndexT>
__aicore__ inline int64_t SparseKvGatherKernel::ResolvePhysicalTokenTyped(
    const uint32_t queryIdx,
    const int64_t logicalToken,
    const GlobalTensor<IndexT> &blockTable) const
{
    if (logicalToken < 0) {
        return -1;
    }

    const uint64_t logicalBlock =
        static_cast<uint64_t>(logicalToken) >> SKG_LOCAL_BLOCK_SHIFT;
    if (logicalBlock >= maxBlocks_) {
        return -1;
    }

    const uint64_t blockTableOffset =
        static_cast<uint64_t>(queryIdx) * maxBlocks_ + logicalBlock;
    const int64_t physicalBlock = ReadIndexTyped(blockTable, blockTableOffset);
    if (physicalBlock < 0 || physicalBlock >= static_cast<int64_t>(numBlocks_)) {
        return -1;
    }

    const uint64_t blockOffset =
        static_cast<uint64_t>(logicalToken) & SKG_LOCAL_BLOCK_MASK;
    return physicalBlock * static_cast<int64_t>(SKG_LOCAL_BLOCK_SIZE) +
           static_cast<int64_t>(blockOffset);
}

__aicore__ inline void SparseKvGatherKernel::WriteZero(
    const uint64_t flatSlot,
    const LocalTensor<uint16_t> &zeroUb) const
{
    DataCopy(outCtkvGm_[flatSlot * SKG_LOCAL_CTKV_DIM],
             zeroUb, SKG_LOCAL_CTKV_DIM);
    DataCopy(outKpeGm_[flatSlot * SKG_LOCAL_KPE_DIM],
             zeroUb[SKG_LOCAL_CTKV_DIM], SKG_LOCAL_KPE_DIM);
}

__aicore__ inline bool SparseKvGatherKernel::CanGatherPair(
    const int64_t physicalToken0,
    const int64_t physicalToken1) const
{
    if (physicalToken1 <= physicalToken0) {
        return false;
    }

    const uint64_t tokenGap = static_cast<uint64_t>(
        physicalToken1 - physicalToken0);
    const uint64_t ctkvSrcStrideBytes =
        (tokenGap - 1U) * SKG_CTKV_ROW_BYTES;
    return ctkvSrcStrideBytes <= SKG_MAX_DMA_EXT_STRIDE;
}

__aicore__ inline void SparseKvGatherKernel::LoadOneToStage(
    const uint32_t row,
    const int64_t physicalToken,
    LocalTensor<uint16_t> &ctkvUb,
    LocalTensor<uint16_t> &kpeUb) const
{
    DataCopy(
        ctkvUb[static_cast<uint64_t>(row) * SKG_LOCAL_CTKV_DIM],
        pagedCtkvGm_[static_cast<uint64_t>(physicalToken) * SKG_LOCAL_CTKV_DIM],
        SKG_LOCAL_CTKV_DIM);
    DataCopy(
        kpeUb[static_cast<uint64_t>(row) * SKG_LOCAL_KPE_DIM],
        pagedKpeGm_[static_cast<uint64_t>(physicalToken) * SKG_LOCAL_KPE_DIM],
        SKG_LOCAL_KPE_DIM);
}

__aicore__ inline void SparseKvGatherKernel::LoadPairToStage(
    const uint32_t row,
    const int64_t physicalToken0,
    const int64_t physicalToken1,
    LocalTensor<uint16_t> &ctkvUb,
    LocalTensor<uint16_t> &kpeUb) const
{
    const uint64_t tokenGap = static_cast<uint64_t>(
        physicalToken1 - physicalToken0);

    DataCopyExtParams ctkvParams{};
    ctkvParams.blockCount = SKG_LOCAL_PAIR_WIDTH;
    ctkvParams.blockLen = SKG_CTKV_ROW_BYTES;
    ctkvParams.srcStride = static_cast<uint32_t>(
        (tokenGap - 1U) * SKG_CTKV_ROW_BYTES);
    ctkvParams.dstStride = 0;

    DataCopyExtParams kpeParams{};
    kpeParams.blockCount = SKG_LOCAL_PAIR_WIDTH;
    kpeParams.blockLen = SKG_KPE_ROW_BYTES;
    kpeParams.srcStride = static_cast<uint32_t>(
        (tokenGap - 1U) * SKG_KPE_ROW_BYTES);
    kpeParams.dstStride = 0;

    DataCopyPadExtParams<uint16_t> padParams{};
    padParams.isPad = false;

    DataCopyPad(
        ctkvUb[static_cast<uint64_t>(row) * SKG_LOCAL_CTKV_DIM],
        pagedCtkvGm_[static_cast<uint64_t>(physicalToken0) * SKG_LOCAL_CTKV_DIM],
        ctkvParams,
        padParams);
    DataCopyPad(
        kpeUb[static_cast<uint64_t>(row) * SKG_LOCAL_KPE_DIM],
        pagedKpeGm_[static_cast<uint64_t>(physicalToken0) * SKG_LOCAL_KPE_DIM],
        kpeParams,
        padParams);
}

__aicore__ inline void SparseKvGatherKernel::GatherOne(
    const uint64_t flatSlot,
    const int64_t physicalToken,
    const uint32_t pingPong,
    LocalTensor<uint16_t> &stageUb)
{
    LocalTensor<uint16_t> chunkUb =
        stageUb[pingPong * SKG_LOCAL_STAGE_DIM];
    LocalTensor<uint16_t> ctkvUb = chunkUb;
    LocalTensor<uint16_t> kpeUb = chunkUb[SKG_LOCAL_CTKV_STAGE_DIM];

    LoadOneToStage(0, physicalToken, ctkvUb, kpeUb);

    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pingPong]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pingPong]);

    DataCopy(outCtkvGm_[flatSlot * SKG_LOCAL_CTKV_DIM],
             ctkvUb, SKG_LOCAL_CTKV_DIM);
    DataCopy(outKpeGm_[flatSlot * SKG_LOCAL_KPE_DIM],
             kpeUb, SKG_LOCAL_KPE_DIM);

    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
}

__aicore__ inline void SparseKvGatherKernel::GatherPair(
    const uint64_t flatSlot,
    const int64_t physicalToken0,
    const int64_t physicalToken1,
    const uint32_t pingPong,
    LocalTensor<uint16_t> &stageUb)
{
    LocalTensor<uint16_t> chunkUb =
        stageUb[pingPong * SKG_LOCAL_STAGE_DIM];
    LocalTensor<uint16_t> ctkvUb = chunkUb;
    LocalTensor<uint16_t> kpeUb = chunkUb[SKG_LOCAL_CTKV_STAGE_DIM];

    LoadPairToStage(0, physicalToken0, physicalToken1, ctkvUb, kpeUb);

    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pingPong]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pingPong]);

    DataCopy(outCtkvGm_[flatSlot * SKG_LOCAL_CTKV_DIM],
             ctkvUb, SKG_LOCAL_PAIR_WIDTH * SKG_LOCAL_CTKV_DIM);
    DataCopy(outKpeGm_[flatSlot * SKG_LOCAL_KPE_DIM],
             kpeUb, SKG_LOCAL_PAIR_WIDTH * SKG_LOCAL_KPE_DIM);

    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
}

__aicore__ inline void SparseKvGatherKernel::GatherFullValidChunk(
    const uint64_t flatSlot,
    const uint32_t dealRows,
    const int64_t *physicalTokens,
    const uint32_t pingPong,
    LocalTensor<uint16_t> &stageUb)
{
    LocalTensor<uint16_t> chunkUb =
        stageUb[pingPong * SKG_LOCAL_STAGE_DIM];
    LocalTensor<uint16_t> ctkvUb = chunkUb;
    LocalTensor<uint16_t> kpeUb = chunkUb[SKG_LOCAL_CTKV_STAGE_DIM];

    uint32_t row = 0;
    while (row + 1U < dealRows) {
        if (SKG_ENABLE_PAIR_MOVE &&
            CanGatherPair(physicalTokens[row], physicalTokens[row + 1U])) {
            LoadPairToStage(
                row, physicalTokens[row], physicalTokens[row + 1U], ctkvUb, kpeUb);
            row += 2U;
        } else {
            LoadOneToStage(row, physicalTokens[row], ctkvUb, kpeUb);
            ++row;
        }
    }
    if (row < dealRows) {
        LoadOneToStage(row, physicalTokens[row], ctkvUb, kpeUb);
    }

    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pingPong]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pingPong]);

    DataCopy(
        outCtkvGm_[flatSlot * SKG_LOCAL_CTKV_DIM],
        ctkvUb,
        static_cast<uint64_t>(dealRows) * SKG_LOCAL_CTKV_DIM);
    DataCopy(
        outKpeGm_[flatSlot * SKG_LOCAL_KPE_DIM],
        kpeUb,
        static_cast<uint64_t>(dealRows) * SKG_LOCAL_KPE_DIM);

    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
}

template <typename IndexT>
__aicore__ inline void SparseKvGatherKernel::ProcessTyped(
    const GlobalTensor<IndexT> &blockTable,
    const GlobalTensor<IndexT> &topkIndices,
    const GlobalTensor<IndexT> &curPos)
{
    const uint64_t slotStart = static_cast<uint64_t>(coreIdx_) * slotsPerCore_;
    uint64_t slotEnd = slotStart + slotsPerCore_;
    if (slotEnd > totalSlots_) {
        slotEnd = totalSlots_;
    }
    if (slotStart >= slotEnd) {
        return;
    }

    LocalTensor<uint16_t> stageUb = stageBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> zeroUb = zeroBuf_.Get<uint16_t>();

    Duplicate(zeroUb, static_cast<uint16_t>(0), SKG_LOCAL_COMBINED_DIM);
    SetFlag<HardEvent::V_MTE3>(vectorToMte3_);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3_);

    bool stageInFlight[SKG_STAGE_BUFFER_NUM] = {false, false};
    uint32_t pingPong = 0;

    uint32_t queryIdx = static_cast<uint32_t>(slotStart / topkN_);
    uint32_t slotInQuery = static_cast<uint32_t>(
        slotStart - static_cast<uint64_t>(queryIdx) * topkN_);
    int64_t currentPos = ReadIndexTyped(curPos, queryIdx);

    uint64_t flatSlot = slotStart;
    while (flatSlot < slotEnd) {
        const uint64_t rowsLeftCore = slotEnd - flatSlot;
        const uint32_t rowsLeftQuery = topkN_ - slotInQuery;
        uint32_t dealRows = SKG_LOCAL_STAGE_ROWS;
        if (rowsLeftCore < dealRows) {
            dealRows = static_cast<uint32_t>(rowsLeftCore);
        }
        if (rowsLeftQuery < dealRows) {
            dealRows = rowsLeftQuery;
        }

        int64_t physicalTokens[SKG_LOCAL_STAGE_ROWS];
        bool fullValid = true;
        for (uint32_t row = 0; row < dealRows; ++row) {
            const int64_t logicalToken = ReadIndexTyped(topkIndices, flatSlot + row);
            int64_t physicalToken = -1;
            if (logicalToken >= 0 && logicalToken != currentPos) {
                physicalToken = ResolvePhysicalTokenTyped(
                    queryIdx, logicalToken, blockTable);
            }
            physicalTokens[row] = physicalToken;
            fullValid = fullValid && (physicalToken >= 0);
        }

        if (fullValid) {
            if (stageInFlight[pingPong]) {
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
            }
            GatherFullValidChunk(
                flatSlot, dealRows, physicalTokens, pingPong, stageUb);
            stageInFlight[pingPong] = true;
            pingPong ^= 1U;
        } else {
            uint32_t row = 0;
            while (row < dealRows) {
                const bool hasSecond = row + 1U < dealRows;
                if (hasSecond && SKG_ENABLE_PAIR_MOVE &&
                    physicalTokens[row] >= 0 && physicalTokens[row + 1U] >= 0 &&
                    CanGatherPair(physicalTokens[row], physicalTokens[row + 1U])) {
                    if (stageInFlight[pingPong]) {
                        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
                    }
                    GatherPair(
                        flatSlot + row,
                        physicalTokens[row],
                        physicalTokens[row + 1U],
                        pingPong,
                        stageUb);
                    stageInFlight[pingPong] = true;
                    pingPong ^= 1U;
                    row += 2U;
                    continue;
                }

                if (physicalTokens[row] < 0) {
                    WriteZero(flatSlot + row, zeroUb);
                } else {
                    if (stageInFlight[pingPong]) {
                        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
                    }
                    GatherOne(
                        flatSlot + row,
                        physicalTokens[row],
                        pingPong,
                        stageUb);
                    stageInFlight[pingPong] = true;
                    pingPong ^= 1U;
                }
                ++row;
            }
        }

        flatSlot += dealRows;
        slotInQuery += dealRows;
        if (slotInQuery == topkN_ && flatSlot < slotEnd) {
            slotInQuery = 0;
            ++queryIdx;
            currentPos = ReadIndexTyped(curPos, queryIdx);
        }
    }

    for (uint32_t i = 0; i < SKG_STAGE_BUFFER_NUM; ++i) {
        if (stageInFlight[i]) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[i]);
        }
    }
}

__aicore__ inline void SparseKvGatherKernel::ProcessGeneric()
{
    const uint64_t slotStart = static_cast<uint64_t>(coreIdx_) * slotsPerCore_;
    uint64_t slotEnd = slotStart + slotsPerCore_;
    if (slotEnd > totalSlots_) {
        slotEnd = totalSlots_;
    }
    if (slotStart >= slotEnd) {
        return;
    }

    LocalTensor<uint16_t> stageUb = stageBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> zeroUb = zeroBuf_.Get<uint16_t>();

    Duplicate(zeroUb, static_cast<uint16_t>(0), SKG_LOCAL_COMBINED_DIM);
    SetFlag<HardEvent::V_MTE3>(vectorToMte3_);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3_);

    bool stageInFlight[SKG_STAGE_BUFFER_NUM] = {false, false};
    uint32_t pingPong = 0;

    uint32_t queryIdx = static_cast<uint32_t>(slotStart / topkN_);
    uint32_t slotInQuery = static_cast<uint32_t>(
        slotStart - static_cast<uint64_t>(queryIdx) * topkN_);
    int64_t currentPos = ReadIndex(
        curPosI32Gm_, curPosI64Gm_, curPosType_, queryIdx);

    uint64_t flatSlot = slotStart;
    while (flatSlot < slotEnd) {
        const int64_t logicalToken0 = ReadIndex(
            topkIndicesI32Gm_, topkIndicesI64Gm_, topkIndicesType_, flatSlot);

        int64_t physicalToken0 = -1;
        if (logicalToken0 >= 0 && logicalToken0 != currentPos) {
            physicalToken0 = ResolvePhysicalToken(queryIdx, logicalToken0);
        }

        const bool hasSecondSlot =
            flatSlot + 1U < slotEnd && slotInQuery + 1U < topkN_;
        if (hasSecondSlot) {
            const int64_t logicalToken1 = ReadIndex(
                topkIndicesI32Gm_, topkIndicesI64Gm_, topkIndicesType_,
                flatSlot + 1U);

            int64_t physicalToken1 = -1;
            if (logicalToken1 >= 0 && logicalToken1 != currentPos) {
                physicalToken1 = ResolvePhysicalToken(queryIdx, logicalToken1);
            }

            if (SKG_ENABLE_PAIR_MOVE &&
                physicalToken0 >= 0 && physicalToken1 >= 0 &&
                CanGatherPair(physicalToken0, physicalToken1)) {
                if (stageInFlight[pingPong]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
                }
                GatherPair(flatSlot, physicalToken0, physicalToken1,
                           pingPong, stageUb);
                stageInFlight[pingPong] = true;
                pingPong ^= 1U;

                flatSlot += 2U;
                slotInQuery += 2U;
                if (slotInQuery == topkN_ && flatSlot < slotEnd) {
                    slotInQuery = 0;
                    ++queryIdx;
                    currentPos = ReadIndex(
                        curPosI32Gm_, curPosI64Gm_, curPosType_, queryIdx);
                }
                continue;
            }
        }

        if (physicalToken0 < 0) {
            WriteZero(flatSlot, zeroUb);
        } else {
            if (stageInFlight[pingPong]) {
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[pingPong]);
            }
            GatherOne(flatSlot, physicalToken0, pingPong, stageUb);
            stageInFlight[pingPong] = true;
            pingPong ^= 1U;
        }

        ++flatSlot;
        ++slotInQuery;
        if (slotInQuery == topkN_ && flatSlot < slotEnd) {
            slotInQuery = 0;
            ++queryIdx;
            currentPos = ReadIndex(
                curPosI32Gm_, curPosI64Gm_, curPosType_, queryIdx);
        }
    }

    for (uint32_t i = 0; i < SKG_STAGE_BUFFER_NUM; ++i) {
        if (stageInFlight[i]) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[i]);
        }
    }
}

__aicore__ inline void SparseKvGatherKernel::Process()
{
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }

    const bool allInt32 =
        blockTableType_ == SKG_INDEX_TYPE_INT32 &&
        topkIndicesType_ == SKG_INDEX_TYPE_INT32 &&
        curPosType_ == SKG_INDEX_TYPE_INT32;
    if (allInt32) {
        ProcessTyped<int32_t>(
            blockTableI32Gm_, topkIndicesI32Gm_, curPosI32Gm_);
        return;
    }

    const bool allInt64 =
        blockTableType_ == SKG_INDEX_TYPE_INT64 &&
        topkIndicesType_ == SKG_INDEX_TYPE_INT64 &&
        curPosType_ == SKG_INDEX_TYPE_INT64;
    if (allInt64) {
        ProcessTyped<int64_t>(
            blockTableI64Gm_, topkIndicesI64Gm_, curPosI64Gm_);
        return;
    }

    ProcessGeneric();
}

}  // namespace BaseApi

#endif  // SPARSE_KV_GATHER_KERNEL_H
