/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_decode_update_vector.h
 * \brief
 */
#ifndef lightning_indexer_decode_update_VECTOR_H
#define lightning_indexer_decode_update_VECTOR_H

#include "lightning_indexer_decode_update_common.h"
#include "kernel_operator.h"

namespace LIServiceVec {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr int32_t INVALID_INDEX = -1;
constexpr uint8_t VEC_REPEAT_MAX = 255;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;
constexpr int64_t VALUE_AND_INDEX_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MRG_QUE_0 = 0;
constexpr int64_t MRG_QUE_1 = 1;
constexpr int64_t MRG_QUE_2 = 2;
constexpr int64_t MRG_QUE_3 = 3;
constexpr int64_t MRG_BLOCK_2 = 2;
constexpr int64_t MRG_BLOCK_3 = 3;
constexpr int64_t MRG_BLOCK_4 = 4;
constexpr int64_t SORT32_SEGMENT_SIZE = 32;
constexpr int64_t SORT32_SEGMENT_PAIR_NUM = SORT32_SEGMENT_SIZE * VALUE_AND_INDEX_NUM;
constexpr int64_t SORT32_GROUP_SIZE = SORT32_SEGMENT_SIZE * MRG_BLOCK_4;
constexpr int64_t SORT32_GROUP_PAIR_NUM = SORT32_GROUP_SIZE * VALUE_AND_INDEX_NUM;
constexpr uint32_t INDEX_BITS = 18;
constexpr uint32_t INDEX_MASK = (1u << INDEX_BITS) - 1;
constexpr uint32_t INDEX_MASK_SHIFT = 32 - INDEX_BITS;
constexpr uint32_t INVALID_FLAG_SHIFT = 14;
constexpr int32_t INVALID_SLOT14 = (1 << INVALID_FLAG_SHIFT) - 1;
constexpr int32_t HIT_MISS_KEY_BASE_BITS = static_cast<int32_t>(0x3F800000u);
constexpr int32_t INVALID_SLOT_DELTA = -16384;
constexpr float INVALID_EVICT_KEY = -1.0e20f;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &mMoutGm,
                              GlobalTensor<T> &weightScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    AscendC::DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}


template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}


template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb, LocalTensor<float> &tmpBuff,
                               int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, LICommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}


__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - clz(value);
        return (1 << pow);
    }
}


__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void InitSortOutBuf(const LocalTensor<float> &src, int64_t eleNum)
{
    uint64_t mask1[2] = {0x5555555555555555, 0};
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
    int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
    int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
    int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
    for (int i = 0; i < forLoop; i++) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), INVALID_INDEX, mask0, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
    }
    if (forRemain > 0) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], NEG_INF,
                           mask1, forRemain, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM],
                           INVALID_INDEX, mask0, forRemain, 1, B32_VEC_REPEAT_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void MergeSort32Segments(LocalTensor<float> &sort32Local, const LocalTensor<float> &pairOut,
                                           int64_t sort32Repeats)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[MRG_QUE_0] = SORT32_SEGMENT_SIZE;
    params.elementLengths[MRG_QUE_1] = SORT32_SEGMENT_SIZE;
    params.elementLengths[MRG_QUE_2] = SORT32_SEGMENT_SIZE;
    params.elementLengths[MRG_QUE_3] = SORT32_SEGMENT_SIZE;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b1111;
    params.repeatTimes = 1;

    int64_t groupNum = sort32Repeats / MRG_BLOCK_4;
    for (int64_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
        int64_t srcOffset = groupIdx * SORT32_GROUP_PAIR_NUM;
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = sort32Local[srcOffset];
        srcList.src2 = sort32Local[srcOffset + SORT32_SEGMENT_PAIR_NUM];
        srcList.src3 = sort32Local[srcOffset + SORT32_SEGMENT_PAIR_NUM * MRG_QUE_2];
        srcList.src4 = sort32Local[srcOffset + SORT32_SEGMENT_PAIR_NUM * MRG_QUE_3];
        AscendC::MrgSort<float>(pairOut[srcOffset], srcList, params);
    }
    AscendC::PipeBarrier<PIPE_V>();

    if (groupNum <= 1) {
        return;
    }

    params.elementLengths[MRG_QUE_0] = SORT32_GROUP_SIZE;
    params.elementLengths[MRG_QUE_1] = SORT32_GROUP_SIZE;
    params.elementLengths[MRG_QUE_2] = SORT32_GROUP_SIZE;
    params.elementLengths[MRG_QUE_3] = SORT32_GROUP_SIZE;
    params.validBit = groupNum == MRG_BLOCK_2 ? 0b0011 : (groupNum == MRG_BLOCK_3 ? 0b0111 : 0b1111);

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = pairOut[0];
    srcList.src2 = pairOut[SORT32_GROUP_PAIR_NUM];
    srcList.src3 = pairOut[SORT32_GROUP_PAIR_NUM * MRG_QUE_2];
    srcList.src4 = pairOut[SORT32_GROUP_PAIR_NUM * MRG_QUE_3];
    AscendC::MrgSort<float>(sort32Local, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(pairOut, sort32Local, sort32Repeats * SORT32_SEGMENT_PAIR_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void SortByKeyWithPayload512(const LocalTensor<float> &pairOut,
                                               const LocalTensor<float> &keyLocal,
                                               const LocalTensor<uint32_t> &payloadLocal,
                                               LocalTensor<float> &sortTmpLocal,
                                               int64_t sort32Repeats)
{
    AscendC::Sort32(sortTmpLocal, keyLocal, payloadLocal, sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();
    MergeSort32Segments(sortTmpLocal, pairOut, sort32Repeats);
}

__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum,
                                 const LocalTensor<float> &mrgSrc, int32_t mrgSrcNum,
                                 LocalTensor<float> &tmpTensor)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = mrgDstNum;
    params.elementLengths[1] = mrgSrcNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = mrgDst;
    srcList.src2 = mrgSrc;

    AscendC::MrgSort<float>(tmpTensor, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void MrgBasicBlock(const LocalTensor<float> &dst, const LocalTensor<float> &src, int64_t blockNum,
                                     int64_t basicBlockSize)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[MRG_QUE_0] = basicBlockSize;
    params.elementLengths[MRG_QUE_1] = basicBlockSize;
    params.elementLengths[MRG_QUE_2] = basicBlockSize;
    params.elementLengths[MRG_QUE_3] = basicBlockSize;
    params.ifExhaustedSuspension = false;
    if (blockNum == MRG_BLOCK_2) {
        params.validBit = 0b0011;
    } else if (blockNum == MRG_BLOCK_3) {
        params.validBit = 0b0111;
    } else if (blockNum == MRG_BLOCK_4) {
        params.validBit = 0b1111;
    } else {
        AscendC::DataCopy(dst, src, basicBlockSize * VALUE_AND_INDEX_NUM);
        return;
    }
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_1];
    srcList.src3 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_2];
    srcList.src4 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_3];
    AscendC::MrgSort<float>(dst, srcList, params);
}

template <bool needMrg = true>
__aicore__ inline void SparseTopK(const LocalTensor<float> &dst, const LocalTensor<float> &needsMerging,
                                  const LocalTensor<float> &tmp, int64_t topk, int64_t mergSize)
{
    if (!needMrg) {
        AscendC::DataCopy(dst, needsMerging, mergSize * VALUE_AND_INDEX_NUM);
        return;
    }
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = topk;
    params.elementLengths[1] = mergSize;
    params.ifExhaustedSuspension = (topk == mergSize);
    params.validBit = 0b0011;
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = dst;
    srcList.src2 = needsMerging;
    AscendC::MrgSort<float>(tmp, srcList, params);
    AscendC::DataCopy(dst, tmp, topk * VALUE_AND_INDEX_NUM);
}


__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal, const LocalTensor<uint32_t> &sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;
    uint8_t src1Pattern = 2;
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void DecodeIndexFromPayload(const LocalTensor<uint32_t> &indexOutLocal,
                                              const LocalTensor<uint32_t> &payloadLocal, int64_t count)
{
    AscendC::ShiftLeft(indexOutLocal, payloadLocal, INDEX_MASK_SHIFT, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(indexOutLocal, indexOutLocal, INDEX_MASK_SHIFT, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void FixInvalidIndex(const LocalTensor<int32_t> &indexOutLocal,
                                       const LocalTensor<int32_t> &payloadLocal, int64_t count)
{
    for (int64_t idx = 0; idx < count; ++idx) {
        if (payloadLocal.GetValue(idx) == INVALID_INDEX) {
            indexOutLocal.SetValue(idx, INVALID_INDEX);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void DecodeSlotsFromSlot14(const LocalTensor<int32_t> &slotLocal,
                                             const LocalTensor<int32_t> &tmpLocal, int64_t count)
{
    AscendC::Adds(tmpLocal, slotLocal, static_cast<int32_t>(1), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(tmpLocal.template ReinterpretCast<uint32_t>(), tmpLocal.template ReinterpretCast<uint32_t>(),
                        INVALID_FLAG_SHIFT, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(tmpLocal, tmpLocal, INVALID_SLOT_DELTA, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(slotLocal, slotLocal, tmpLocal, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void DecodeSlotFromPayload(const LocalTensor<uint32_t> &slotOutLocal,
                                             const LocalTensor<uint32_t> &payloadLocal,
                                             const LocalTensor<int32_t> &tmpLocal, int64_t count)
{
    AscendC::ShiftRight(slotOutLocal, payloadLocal, INDEX_BITS, count);
    AscendC::PipeBarrier<PIPE_V>();
    DecodeSlotsFromSlot14(slotOutLocal.template ReinterpretCast<int32_t>(), tmpLocal, count);
}

__aicore__ inline void BuildHitMissKey(const LocalTensor<float> &keyLocal,
                                       const LocalTensor<uint32_t> &payloadLocal, int64_t count)
{
    AscendC::ShiftRight(keyLocal.template ReinterpretCast<uint32_t>(), payloadLocal, INDEX_BITS, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(keyLocal.template ReinterpretCast<int32_t>(), keyLocal.template ReinterpretCast<int32_t>(),
                  HIT_MISS_KEY_BASE_BITS, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildEvictCandidateKeyFromPayload(const LocalTensor<float> &keyLocal,
                                                         const LocalTensor<float> &scoreLocal,
                                                         const LocalTensor<uint32_t> &payloadLocal,
                                                         const LocalTensor<float> &tmpLocal, int64_t count)
{
    AscendC::Muls(keyLocal, scoreLocal, -1.0f, count);
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> slot14Local = tmpLocal[count].template ReinterpretCast<int32_t>();
    AscendC::ShiftRight(slot14Local.template ReinterpretCast<uint32_t>(), payloadLocal, INDEX_BITS, count);
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> invalidMaskLocal = tmpLocal.template ReinterpretCast<uint8_t>();
    AscendC::CompareScalar(invalidMaskLocal, slot14Local, INVALID_SLOT14, AscendC::CMPMODE::EQ, count);
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<float> invalidKeyLocal = tmpLocal[count * 2];
    AscendC::Duplicate(invalidKeyLocal, INVALID_EVICT_KEY, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(keyLocal, invalidMaskLocal, invalidKeyLocal, keyLocal,
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

} // namespace LIServiceVec
#endif // lightning_indexer_decode_update_VECTOR_H
