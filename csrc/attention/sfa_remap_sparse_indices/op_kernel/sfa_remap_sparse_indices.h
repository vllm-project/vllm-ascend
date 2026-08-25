/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef SFA_REMAP_SPARSE_INDICES_H
#define SFA_REMAP_SPARSE_INDICES_H

#include "kernel_operator.h"

namespace SfaRemapSparseIndices {
using namespace AscendC;

struct SfaRemapSparseIndicesTilingData {
    uint32_t rows;
    uint32_t topK;
    uint32_t dcpSize;
    uint32_t dcpRank;
    uint32_t interleaveSize;
    uint32_t interleaveShift;
    uint32_t dcpInterleaveShift;
    uint32_t usePowerOfTwo;
    uint32_t useVectorMagicDivision;
    uint32_t interleaveMagic;
    uint32_t interleaveMore;
    uint32_t dcpMagic;
    uint32_t dcpMore;
    uint32_t rowsPerCore;
    uint32_t chunkElements;
    uint32_t bufferBytes;
    uint32_t maskBytes;
};

class SfaRemapSparseIndicesKernel {
public:
    __aicore__ inline explicit SfaRemapSparseIndicesKernel(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                const SfaRemapSparseIndicesTilingData* tilingData)
    {
        rows_ = tilingData->rows;
        topK_ = tilingData->topK;
        dcpSize_ = tilingData->dcpSize;
        dcpRank_ = tilingData->dcpRank;
        interleaveSize_ = tilingData->interleaveSize;
        interleaveShift_ = tilingData->interleaveShift;
        dcpInterleaveShift_ = tilingData->dcpInterleaveShift;
        usePowerOfTwo_ = tilingData->usePowerOfTwo;
        useVectorMagicDivision_ = tilingData->useVectorMagicDivision;
        interleaveMagic_ = tilingData->interleaveMagic;
        interleaveMore_ = tilingData->interleaveMore;
        dcpMagic_ = tilingData->dcpMagic;
        dcpMore_ = tilingData->dcpMore;
        rowsPerCore_ = tilingData->rowsPerCore;
        chunkElements_ = tilingData->chunkElements;
        maskBytes_ = tilingData->maskBytes;

        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input), rows_ * topK_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output), rows_ * topK_);
        pipe_->InitBuffer(inputQueue_, 1, tilingData->bufferBytes);
        pipe_->InitBuffer(outputQueue_, 1, tilingData->bufferBytes);
        pipe_->InitBuffer(temp0_, tilingData->bufferBytes);
        pipe_->InitBuffer(temp1_, tilingData->bufferBytes);
        if (useVectorMagicDivision_ != 0) {
            pipe_->InitBuffer(temp2_, tilingData->bufferBytes);
            pipe_->InitBuffer(temp3_, tilingData->bufferBytes);
            pipe_->InitBuffer(temp4_, tilingData->bufferBytes);
            pipe_->InitBuffer(temp5_, tilingData->bufferBytes);
        }
        pipe_->InitBuffer(ownerMask_, tilingData->maskBytes);
        pipe_->InitBuffer(validMask_, tilingData->maskBytes);
        pipe_->InitBuffer(combinedMask_, tilingData->maskBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t start = GetBlockIdx() * rowsPerCore_;
        uint32_t end = start + rowsPerCore_;
        if (end > rows_) {
            end = rows_;
        }
        for (uint32_t row = start; row < end; ++row) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        uint32_t writePos = 0;
        for (uint32_t offset = 0; offset < topK_; offset += chunkElements_) {
            activeElements_ = topK_ - offset;
            if (activeElements_ > chunkElements_) {
                activeElements_ = chunkElements_;
            }
            vectorElements_ = (activeElements_ + 255U) & ~255U;
            CopyIn(row, offset);
            uint32_t validCount = ComputeChunk();
            CopyOut(row, writePos, validCount);
            writePos += validCount;
        }
        FillTail(row, writePos);
    }

    __aicore__ inline void CopyIn(uint32_t row, uint32_t offset)
    {
        LocalTensor<int32_t> input = inputQueue_.AllocTensor<int32_t>();
        if (vectorElements_ != activeElements_) {
            Duplicate(input, static_cast<int32_t>(-1), vectorElements_);
            event_t vectorToMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        }
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(activeElements_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(input, inputGm_[row * topK_ + offset], copyParams, padParams);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void MulHighU32(const LocalTensor<int32_t>& input,
                                     const LocalTensor<int32_t>& high,
                                     const LocalTensor<int32_t>& scratch1,
                                     const LocalTensor<int32_t>& scratch2,
                                     const LocalTensor<int32_t>& scratch3,
                                     const LocalTensor<int32_t>& scratch4,
                                     uint32_t magic)
    {
        constexpr uint32_t LIMB_BITS = 16;
        uint32_t lowMagic = magic & 0xffffU;
        uint32_t highMagic = magic >> LIMB_BITS;
        auto inputU32 = input.ReinterpretCast<uint32_t>();
        auto highU32 = high.ReinterpretCast<uint32_t>();
        auto scratch1U32 = scratch1.ReinterpretCast<uint32_t>();
        auto scratch2U32 = scratch2.ReinterpretCast<uint32_t>();
        auto scratch3U32 = scratch3.ReinterpretCast<uint32_t>();
        auto scratch4U32 = scratch4.ReinterpretCast<uint32_t>();

        // Split x into x0/x1, then form four exact 16x16 low-32 products.
        ShiftRight(scratch1U32, inputU32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(highU32, scratch1U32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        Sub(high, input, high, vectorElements_);
        PipeBarrier<PIPE_V>();
        Muls(scratch2, high, static_cast<int32_t>(lowMagic), vectorElements_);
        Muls(scratch3, high, static_cast<int32_t>(highMagic), vectorElements_);
        Muls(scratch4, scratch1, static_cast<int32_t>(lowMagic), vectorElements_);
        PipeBarrier<PIPE_V>();
        Muls(high, scratch1, static_cast<int32_t>(highMagic), vectorElements_);
        PipeBarrier<PIPE_V>();

        // high32 = p3 + p1_hi + p2_hi
        //          + ((p0_hi + p1_lo + p2_lo) >> 16).
        ShiftRight(scratch1U32, scratch3U32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(high, high, scratch1, vectorElements_);
        ShiftRight(scratch1U32, scratch4U32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(high, high, scratch1, vectorElements_);
        ShiftRight(scratch2U32, scratch2U32, LIMB_BITS, vectorElements_);
        ShiftLeft(scratch3U32, scratch3U32, LIMB_BITS, vectorElements_);
        ShiftLeft(scratch4U32, scratch4U32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        ShiftRight(scratch3U32, scratch3U32, LIMB_BITS, vectorElements_);
        ShiftRight(scratch4U32, scratch4U32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(scratch2, scratch2, scratch3, vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(scratch2, scratch2, scratch4, vectorElements_);
        PipeBarrier<PIPE_V>();
        ShiftRight(scratch2U32, scratch2U32, LIMB_BITS, vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(high, high, scratch2, vectorElements_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DivideU32Exact(const LocalTensor<int32_t>& numerator,
                                         const LocalTensor<int32_t>& quotient,
                                         const LocalTensor<int32_t>& scratch1,
                                         const LocalTensor<int32_t>& scratch2,
                                         const LocalTensor<int32_t>& scratch3,
                                         const LocalTensor<int32_t>& scratch4,
                                         uint32_t magic, uint32_t more)
    {
        constexpr uint32_t SHIFT_MASK = 0x1f;
        constexpr uint32_t ADD_MARKER = 0x40;
        auto numeratorU32 = numerator.ReinterpretCast<uint32_t>();
        auto quotientU32 = quotient.ReinterpretCast<uint32_t>();
        auto scratch1U32 = scratch1.ReinterpretCast<uint32_t>();
        if (magic == 0) {
            ShiftRight(quotientU32, numeratorU32, more, vectorElements_);
            PipeBarrier<PIPE_V>();
            return;
        }

        MulHighU32(numerator, quotient, scratch1, scratch2, scratch3, scratch4, magic);
        if ((more & ADD_MARKER) != 0) {
            Sub(scratch1, numerator, quotient, vectorElements_);
            PipeBarrier<PIPE_V>();
            ShiftRight(scratch1U32, scratch1U32, 1U, vectorElements_);
            PipeBarrier<PIPE_V>();
            Add(scratch1, scratch1, quotient, vectorElements_);
            PipeBarrier<PIPE_V>();
            ShiftRight(quotientU32, scratch1U32, more & SHIFT_MASK, vectorElements_);
        } else {
            ShiftRight(quotientU32, quotientU32, more, vectorElements_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t CompactVector(const LocalTensor<int32_t>& input,
                                             const LocalTensor<int32_t>& owner,
                                             const LocalTensor<int32_t>& remapped,
                                             const LocalTensor<int32_t>& output)
    {
        LocalTensor<uint8_t> ownerMask = ownerMask_.Get<uint8_t>();
        LocalTensor<uint8_t> validMask = validMask_.Get<uint8_t>();
        LocalTensor<uint8_t> combinedMask = combinedMask_.Get<uint8_t>();
        LocalTensor<float> ownerFloat = temp0_.Get<float>();
        LocalTensor<float> inputFloat = output.ReinterpretCast<float>();

        Adds(owner, owner, -static_cast<int32_t>(dcpRank_), vectorElements_);
        PipeBarrier<PIPE_V>();
        Cast(ownerFloat, owner, RoundMode::CAST_ROUND, vectorElements_);
        PipeBarrier<PIPE_V>();
        CompareScalar(ownerMask, ownerFloat, 0.0f, CMPMODE::EQ, vectorElements_);
        PipeBarrier<PIPE_V>();
        Cast(inputFloat, input, RoundMode::CAST_ROUND, vectorElements_);
        PipeBarrier<PIPE_V>();
        CompareScalar(validMask, inputFloat, 0.0f, CMPMODE::GE, vectorElements_);
        PipeBarrier<PIPE_V>();
        // Materialize the packed predicates in ordinary UB storage before
        // GatherMask consumes them. This is required by the validated A3 path.
        And(combinedMask.ReinterpretCast<uint16_t>(), ownerMask.ReinterpretCast<uint16_t>(),
            validMask.ReinterpretCast<uint16_t>(), vectorElements_ / 16U);
        PipeBarrier<PIPE_V>();
        GatherMaskParams gatherParams;
        gatherParams.repeatTimes = 1;
        gatherParams.src0BlockStride = 1;
        gatherParams.src0RepeatStride = 8;
        gatherParams.src1RepeatStride = 8;
        uint64_t validCount = 0;
        GatherMask<int32_t, uint32_t>(
            output, remapped, combinedMask.ReinterpretCast<uint32_t>(), true,
            activeElements_, gatherParams, validCount);
        PipeBarrier<PIPE_V>();
        return static_cast<uint32_t>(validCount);
    }

    __aicore__ inline uint32_t ComputeVectorPowerOfTwo(const LocalTensor<int32_t>& input,
                                                       const LocalTensor<int32_t>& output)
    {
        LocalTensor<int32_t> block = temp0_.Get<int32_t>();
        LocalTensor<int32_t> remapped = temp1_.Get<int32_t>();
        LocalTensor<int32_t> owner = output;
        auto inputU32 = input.ReinterpretCast<uint32_t>();
        auto blockU32 = block.ReinterpretCast<uint32_t>();
        auto remappedU32 = remapped.ReinterpretCast<uint32_t>();
        auto ownerU32 = owner.ReinterpretCast<uint32_t>();
        uint32_t dcpShift = dcpInterleaveShift_ - interleaveShift_;

        // block = index / interleaveSize; remapped temporarily holds block / dcpSize.
        ShiftRight(blockU32, inputU32, interleaveShift_, vectorElements_);
        ShiftRight(remappedU32, inputU32, dcpInterleaveShift_, vectorElements_);
        PipeBarrier<PIPE_V>();

        // owner = block - (block / dcpSize) * dcpSize.
        ShiftLeft(ownerU32, remappedU32, dcpShift, vectorElements_);
        PipeBarrier<PIPE_V>();
        Sub(owner, block, owner, vectorElements_);
        PipeBarrier<PIPE_V>();

        // remapped = (block / dcpSize) * interleaveSize + index % interleaveSize.
        ShiftLeft(remappedU32, remappedU32, interleaveShift_, vectorElements_);
        ShiftLeft(blockU32, blockU32, interleaveShift_, vectorElements_);
        PipeBarrier<PIPE_V>();
        Sub(block, input, block, vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(remapped, remapped, block, vectorElements_);
        PipeBarrier<PIPE_V>();

        return CompactVector(input, owner, remapped, output);
    }

    __aicore__ inline uint32_t ComputeVectorMagic(const LocalTensor<int32_t>& input,
                                                  const LocalTensor<int32_t>& output)
    {
        LocalTensor<int32_t> block = temp0_.Get<int32_t>();
        LocalTensor<int32_t> blockGroup = temp1_.Get<int32_t>();
        LocalTensor<int32_t> owner = temp2_.Get<int32_t>();
        LocalTensor<int32_t> remapped = temp3_.Get<int32_t>();
        LocalTensor<int32_t> scratch4 = temp4_.Get<int32_t>();
        LocalTensor<int32_t> scratch5 = temp5_.Get<int32_t>();

        DivideU32Exact(input, block, blockGroup, owner, remapped, scratch4,
                       interleaveMagic_, interleaveMore_);
        DivideU32Exact(block, blockGroup, owner, remapped, scratch4, scratch5,
                       dcpMagic_, dcpMore_);

        Muls(owner, blockGroup, static_cast<int32_t>(dcpSize_), vectorElements_);
        PipeBarrier<PIPE_V>();
        Sub(owner, block, owner, vectorElements_);
        Muls(remapped, block, static_cast<int32_t>(interleaveSize_), vectorElements_);
        PipeBarrier<PIPE_V>();
        Sub(remapped, input, remapped, vectorElements_);
        Muls(scratch4, blockGroup, static_cast<int32_t>(interleaveSize_), vectorElements_);
        PipeBarrier<PIPE_V>();
        Add(remapped, scratch4, remapped, vectorElements_);
        PipeBarrier<PIPE_V>();

        return CompactVector(input, owner, remapped, output);
    }

    __aicore__ inline uint32_t ComputeChunk()
    {
        LocalTensor<int32_t> input = inputQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> output = outputQueue_.AllocTensor<int32_t>();
        uint32_t validCount = usePowerOfTwo_ != 0 ? ComputeVectorPowerOfTwo(input, output)
                                                  : ComputeVectorMagic(input, output);
        inputQueue_.FreeTensor(input);
        outputQueue_.EnQue(output);
        return validCount;
    }

    __aicore__ inline void CopyOut(uint32_t row, uint32_t writePos, uint32_t validCount)
    {
        LocalTensor<int32_t> output = outputQueue_.DeQue<int32_t>();
        if (validCount != 0) {
            event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(validCount * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(outputGm_[row * topK_ + writePos], output, copyParams);
            event_t mte3ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(mte3ToVector);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
        }
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void FillTail(uint32_t row, uint32_t writePos)
    {
        while (writePos < topK_) {
            activeElements_ = topK_ - writePos;
            if (activeElements_ > chunkElements_) {
                activeElements_ = chunkElements_;
            }
            LocalTensor<int32_t> output = outputQueue_.AllocTensor<int32_t>();
            Duplicate(output, static_cast<int32_t>(-1), activeElements_);
            outputQueue_.EnQue(output);
            output = outputQueue_.DeQue<int32_t>();
            event_t vectorToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToMte3);
            WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(activeElements_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(outputGm_[row * topK_ + writePos], output, copyParams);
            event_t mte3ToVector = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(mte3ToVector);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
            outputQueue_.FreeTensor(output);
            writePos += activeElements_;
        }
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> temp0_;
    TBuf<QuePosition::VECCALC> temp1_;
    TBuf<QuePosition::VECCALC> temp2_;
    TBuf<QuePosition::VECCALC> temp3_;
    TBuf<QuePosition::VECCALC> temp4_;
    TBuf<QuePosition::VECCALC> temp5_;
    TBuf<QuePosition::VECCALC> ownerMask_;
    TBuf<QuePosition::VECCALC> validMask_;
    TBuf<QuePosition::VECCALC> combinedMask_;
    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    uint32_t rows_;
    uint32_t topK_;
    uint32_t dcpSize_;
    uint32_t dcpRank_;
    uint32_t interleaveSize_;
    uint32_t interleaveShift_;
    uint32_t dcpInterleaveShift_;
    uint32_t usePowerOfTwo_;
    uint32_t useVectorMagicDivision_;
    uint32_t interleaveMagic_;
    uint32_t interleaveMore_;
    uint32_t dcpMagic_;
    uint32_t dcpMore_;
    uint32_t rowsPerCore_;
    uint32_t chunkElements_;
    uint32_t maskBytes_;
    uint32_t activeElements_;
    uint32_t vectorElements_;
};
}  // namespace SfaRemapSparseIndices

#endif
