/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_vector.h
 * \brief
 */
#ifndef quant_lightning_indexer_VECTOR_H
#define quant_lightning_indexer_VECTOR_H

#include "kernel_operator.h"
#include "quant_lightning_indexer_vector.h"

namespace QLIServiceVec {
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

/**
  src: 传入的初始化空间
  eleNum: 需要初始化的元素个数需为64整数倍，元素将被初始化为交错排布的-inf，-1
 */
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

/**
  src: logits和索引，前logitsNum为logits，后logitsNum为索引
  tmp: 计算使用到的临时空间，大小与src一致
  logitsNum: 排序的元素个数, 暂只支持[128,256,384,512,1024,2048]
 */
__aicore__ inline void SortAll(LocalTensor<float> &src, LocalTensor<float> &tmp, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort32(tmp, src, src[logitsNum].ReinterpretCast<uint32_t>(), sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();

    int64_t mrgGroups = sort32Repeats;
    int64_t mrgElements = BLOCK_BYTES;
    int64_t i = 0;
    AscendC::LocalTensor<float> srcTensor;
    AscendC::LocalTensor<float> dstTensor;
    while (true) {
        if (i % CONST_TWO == 0) {
            srcTensor = tmp;
            dstTensor = src;
        } else {
            srcTensor = src;
            dstTensor = tmp;
        }
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgElements;
        params.elementLengths[MRG_QUE_1] = mrgElements;
        params.elementLengths[MRG_QUE_2] = mrgElements;
        params.elementLengths[MRG_QUE_3] = mrgElements;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = srcTensor[0];
        srcList.src2 = srcTensor[MRG_QUE_1 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src3 = srcTensor[MRG_QUE_2 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src4 = srcTensor[MRG_QUE_3 * VALUE_AND_INDEX_NUM * mrgElements];
        if (mrgGroups <= MRG_BLOCK_4) {
            params.repeatTimes = 1;
            if (mrgGroups == 1) {
                break;
            } else if (mrgGroups == MRG_BLOCK_2) {
                params.validBit = 0b0011;
            } else if (mrgGroups == MRG_BLOCK_3) {
                params.validBit = 0b0111;
            } else if (mrgGroups == MRG_BLOCK_4) {
                params.validBit = 0b1111;
            }
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            break;
        } else {
            params.repeatTimes = mrgGroups / MRG_BLOCK_4;
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            mrgElements = mrgElements * MRG_BLOCK_4;
            mrgGroups = mrgGroups / MRG_BLOCK_4;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (i % CONST_TWO == 0) {
        AscendC::DataCopy(src, tmp, logitsNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
  mrgDst: 合并进的Tensor
  mrgSrc: 待合并的Tensor
  tmpTensor：空间为mrgDst+mrgSrc
 */
__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum, LocalTensor<float> &mrgSrc,
                                 int32_t mrgSrcNum, LocalTensor<float> &tmpTensor)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = mrgSrcNum;
    params.elementLengths[1] = mrgDstNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = mrgSrc;
    srcList.src2 = mrgDst;

    AscendC::MrgSort<float>(tmpTensor, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal, const LocalTensor<uint32_t> &sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;     // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2;  // 固定模式2,表示筛选出奇数索引的数
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

}  // namespace QLIServiceVec
#endif  // quant_lightning_indexer_VECTOR_H