/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pse.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_PSE_H
#define FLASH_ATTENTION_SCORE_PSE_H

#include "kernel_operator.h"
#include "util.h"

constexpr static int64_t pseS1S2 = 0;
constexpr static int64_t pse1S2 = 1;
constexpr static int64_t pseSlopeBn = 2;
constexpr static int64_t pseSlopeN = 3;

constexpr static uint8_t pseEncodeALibiS2Full = 0x11;

enum class PseTypeEnum {
    PSE_OUTER_MUL_ADD_TYPE = 0, // default
    PSE_OUTER_ADD_MUL_TYPE,
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};

struct PseInfo {
    int64_t blockCount;
    int64_t bSSOffset; // boidx * s1 * s2
    int64_t boIdx;
    int64_t gSize;
    int64_t goIdx;
    int64_t loopIdx;
    int64_t n2G;
    int64_t n2oIdx;
    int64_t pseBSize;
    int64_t pseS1Size;        // for alibi
    int64_t pseS2ComputeSize; // for alibi, do not need assignment
    int64_t pseS2Size;        // for alibi
    uint32_t pseShapeType;
    int64_t readS2Size; // for alibi, do not need assignment
    int64_t s1BaseSize;
    int64_t s1Size;
    int64_t s1oIdx;
    int64_t s2AlignedSize;
    int64_t s2BaseNratioSize;
    int64_t s2LoopCount;
    int64_t s2RealSize;
    int64_t s2Size;
    int64_t s2SizeAcc; // accumulated sum of s2 size
    int64_t s2StartIdx;
    int64_t vec1S1BaseSize;
    int64_t vec1S1RealSize;
    uint32_t pseEncodeType; // for distinguish alibi
    uint32_t pseType; // 0: outer, mul-add   1:outer, add-mul   2:inner, mul-add   3:inner, mul-add-sqrt
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t vecCoreOffset = 0;
    bool needCast;
    bool align8 = false;
    bool pseEndogenous = false;
};

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyInCommon(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
                                  int64_t s1Size, int64_t s2Size, int64_t actualS2Len, int32_t dtypeSize,
                                  int32_t alignedS2Size)
{
    if constexpr (hasPse == true) {
        uint32_t shapeArray[] = {static_cast<uint32_t>(s1Size), static_cast<uint32_t>(alignedS2Size)};
        dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        dstTensor.SetSize(s1Size * alignedS2Size);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s1Size;
        dataCopyParams.blockLen = CeilDiv(s2Size * dtypeSize, blockBytes); // 单位32B
        dataCopyParams.dstStride = alignedS2Size * dtypeSize / blockBytes - dataCopyParams.blockLen;                                      // gap
        if (actualS2Len * dtypeSize % blockBytes == 0) {
            dataCopyParams.srcStride =
                (actualS2Len * dtypeSize - dataCopyParams.blockLen * blockBytes) / blockBytes; // srcGap
            DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
        } else {
            dataCopyParams.blockLen = s2Size * dtypeSize; // 单位Byte
            dataCopyParams.srcStride = (actualS2Len * dtypeSize - dataCopyParams.blockLen);
            dataCopyParams.dstStride = (alignedS2Size - s2Size) * dtypeSize / blockBytes;
            DataCopyPadParams dataCopyPadParams;
            dataCopyPadParams.isPad = false;
            DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
        }
    }
}

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
                                  int64_t s1Size, int64_t s2Size, int64_t actualS2Len, int64_t alignedSize = 16)
{
    if constexpr (hasPse == true) {
        int32_t dtypeSize = sizeof(INPUT_T);
        int32_t alignedS2Size = CeilDiv(s2Size, alignedSize) * alignedSize;
        DataCopyInCommon<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, s2Size,
            actualS2Len, dtypeSize, alignedS2Size);
    }
}

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyInAlign8(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
                                  int64_t s1Size, int64_t s2Size, int64_t actualS2Len)
{
    if constexpr (hasPse == true) {
        int32_t dtypeSize = sizeof(INPUT_T);
        if (dtypeSize == 0){
            return;
        }
        int32_t alignedS2Size = CeilDiv(s2Size, 32 / dtypeSize) * (32 / dtypeSize);
        DataCopyInCommon<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, s2Size,
            actualS2Len, dtypeSize, alignedS2Size);
    }
}

/*
dst = BroadcastAdd(src0, src1)
src0 shape: (s1, s2)
src1 shape: (1, s2)
dst  shape: (s1, s2)
*/
template <typename T, bool hasPse>
__aicore__ inline void BroadcastAdd(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor,
                                    int64_t src0Offset, int32_t src1Size, int32_t repeatTimes)
{
    if constexpr (hasPse == true) {
        /* Total data number of single step should be smaller than 256bytes.
         * If larger, we need to do add multiple times. */
        int32_t innerLoop = src1Size / repeatMaxSize;   // s2轴整块计算次数
        int32_t innerRemain = src1Size % repeatMaxSize; // s2轴尾块计算量
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.src0BlkStride = 1;
        binaryRepeatParams.src0RepStride = src1Size / blockSize;
        binaryRepeatParams.src1BlkStride = 1;
        binaryRepeatParams.src1RepStride = 0;
        binaryRepeatParams.dstRepStride = binaryRepeatParams.src0RepStride;
        binaryRepeatParams.blockNumber = binaryRepeatParams.src0RepStride;

        for (int32_t j = 0; j < innerLoop; j++) {
            auto innerOffset = j * repeatMaxSize;
            auto ubOffset = src0Offset + innerOffset;
            Add(src0Tensor[ubOffset], src0Tensor[ubOffset], src1Tensor[innerOffset], repeatMaxSize, repeatTimes,
                binaryRepeatParams);
        }
        if (innerRemain > 0) {
            auto innerOffset = innerLoop * repeatMaxSize;
            auto ubOffset = src0Offset + innerOffset;
            Add(src0Tensor[ubOffset], src0Tensor[ubOffset], src1Tensor[innerOffset], innerRemain, repeatTimes,
                binaryRepeatParams);
        }
    }
}

template <typename T, bool hasPse>
__aicore__ inline void PseBroadcastAdd(int32_t s1Size, int32_t s2Size, int32_t computeSize, const LocalTensor<T> &pseUb,
                                       const LocalTensor<T> &dstTensor, uint32_t pseShapeType)
{
    if constexpr (hasPse == true) {
        if (pseShapeType == pseS1S2 || pseShapeType == pseSlopeBn || pseShapeType == pseSlopeN) {
            Add(dstTensor, dstTensor, pseUb, computeSize);
        } else {
            /* Total repeated times should be <= repeatMaxTimes. If larger,
             * we need to do multiple inner loops. */
            int32_t s1OuterLoop = s1Size / repeatMaxTimes;
            int32_t s1OuterRemain = s1Size % repeatMaxTimes;
            for (int32_t s1OuterIdx = 0; s1OuterIdx < s1OuterLoop; s1OuterIdx++) {
                int32_t s1OuterOffset = s1OuterIdx * repeatMaxTimes * s2Size;
                BroadcastAdd<T, hasPse>(dstTensor, pseUb, s1OuterOffset, s2Size, repeatMaxTimes);
            }
            if (s1OuterRemain > 0) {
                int32_t s1OuterOffset = s1OuterLoop * repeatMaxTimes * s2Size;
                BroadcastAdd<T, hasPse>(dstTensor, pseUb, s1OuterOffset, s2Size, s1OuterRemain);
            }
        }
    }
}
template <bool hasPse> __aicore__ inline int64_t PseComputeOffset(PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = 0;
        int64_t s1Offset = 0;
        int64_t s2Offset = pseInfo.s2StartIdx + pseInfo.s2LoopCount * pseInfo.s2BaseNratioSize;
        int64_t gOffset = 0;
        if (pseInfo.pseShapeType == pseS1S2) {
            // b, n2, g, s1, s2
            bOffset = pseInfo.bSSOffset * pseInfo.n2G;
            n2Offset = pseInfo.n2oIdx * pseInfo.gSize * pseInfo.s1Size * pseInfo.s2Size;
            gOffset = pseInfo.goIdx * pseInfo.s1Size * pseInfo.s2Size;
            s1Offset = (pseInfo.s1oIdx * pseInfo.s1BaseSize + pseInfo.vecCoreOffset +
                       pseInfo.loopIdx * pseInfo.vec1S1BaseSize) * pseInfo.s2Size;
        } else if (pseInfo.pseShapeType == pse1S2) {
            // b, n2, g, 1, s2
            bOffset = pseInfo.s2SizeAcc * pseInfo.n2G;
            n2Offset = pseInfo.n2oIdx * pseInfo.gSize * pseInfo.s2Size;
            gOffset = pseInfo.goIdx * pseInfo.s2Size;
        }
        if (pseInfo.pseBSize == 1) {
            bOffset = 0;
        }
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    } else {
        return 0;
    }
}

template <LayOutTypeEnum layOutType, bool hasPse> __aicore__ inline int64_t PseAlibiComputeOffset(PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        int64_t bOffset = (pseInfo.boIdx % pseInfo.pseBSize) * pseInfo.n2G * pseInfo.pseS2Size * pseInfo.pseS1Size;
        int64_t n2Offset = pseInfo.n2oIdx * pseInfo.gSize * pseInfo.pseS2Size * pseInfo.pseS1Size;
        int64_t gOffset = pseInfo.goIdx * pseInfo.pseS2Size * pseInfo.pseS1Size;
        int64_t row = pseInfo.s1oIdx * pseInfo.s1BaseSize + pseInfo.vecCoreOffset +
                      pseInfo.loopIdx * pseInfo.vec1S1BaseSize;
        int64_t column = pseInfo.s2StartIdx + pseInfo.s2LoopCount * pseInfo.s2BaseNratioSize;
        int64_t m = 0;
        int64_t k = 0;
        if constexpr (layOutType != LayOutTypeEnum::LAYOUT_TND) {
            int64_t threshold = pseInfo.s1Size - pseInfo.pseS1Size;
            if (row >= threshold) {
                m = row - threshold;
                k = column;
            } else {
                m = row % pseInfo.pseS1Size;
                k = pseInfo.pseS2Size - (row - column) - (pseInfo.pseS1Size - m);
            }
        } else {
            int64_t threshold = pseInfo.pseS2Size - pseInfo.pseS1Size;
            int64_t posVal = row - column - threshold;
            if (threshold >= 0) {
                if (posVal >= 0) {
                    m = posVal;
                    k = 0;
                } else {
                    m = 0;
                    k = -posVal;
                }
            } else {
                m = posVal;
                k = 0;
            }
        }
        int64_t s1Offset = m * pseInfo.pseS2Size;
        int64_t s2Offset = k;
        pseInfo.readS2Size = Min(pseInfo.s2AlignedSize, pseInfo.pseS2Size - k);
        pseInfo.pseS2ComputeSize = Align(pseInfo.readS2Size);

        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    } else {
        return 0;
    }
}

template <bool hasPse> __aicore__ inline bool NeedPseAlibiCompute(PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        // Alibi编码只计算下三角
        if (pseInfo.s1oIdx * pseInfo.s1BaseSize + pseInfo.vecCoreOffset +
            (pseInfo.loopIdx + 1) * pseInfo.vec1S1BaseSize <=
            pseInfo.s2StartIdx + pseInfo.s2LoopCount * pseInfo.s2BaseNratioSize) {
            return false;
        }
        return true;
    } else {
        return false;
    }
}

template <typename INPUT_T, typename T, LayOutTypeEnum layOutType, bool hasPse>
__aicore__ inline void PseAlibiCopyIn(LocalTensor<T> &dstTensor, LocalTensor<INPUT_T> &tmpTensor,
                                      GlobalTensor<INPUT_T> &srcTensor, PseInfo &pseInfo, int64_t alignedSize = 16)
{
    if constexpr (hasPse == true) {
        if (!NeedPseAlibiCompute<hasPse>(pseInfo)) {
            return;
        }
        int64_t offset = PseAlibiComputeOffset<layOutType, hasPse>(pseInfo);
        if constexpr (IsSameType<INPUT_T, T>::value) {
            if (!pseInfo.align8){
                DataCopyIn<INPUT_T, hasPse>(dstTensor, srcTensor, offset, pseInfo.vec1S1RealSize, pseInfo.readS2Size,
                                        pseInfo.pseS2Size, alignedSize);
            } else {
                DataCopyInAlign8<INPUT_T, hasPse>(dstTensor, srcTensor, offset, pseInfo.vec1S1RealSize,
                        pseInfo.readS2Size, pseInfo.pseS2Size);
            }
            return;
        }

        DataCopyIn<INPUT_T, hasPse>(tmpTensor, srcTensor, offset, pseInfo.vec1S1RealSize, pseInfo.readS2Size,
                                    pseInfo.pseS2Size, alignedSize);
        if (pseInfo.needCast) {
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(dstTensor, tmpTensor, RoundMode::CAST_NONE, pseInfo.vec1S1RealSize * pseInfo.pseS2ComputeSize);
        }
        return;
    }
}

template <typename T, bool hasPse>
__aicore__ inline void PseSlopeCopyIn(LocalTensor<T> &dstTensor, LocalTensor<half> &helpTensor,
                                      __gm__ uint8_t *pseSlope, GlobalTensor<half> &alibiGm, PseInfo &pseInfo,
                                      int64_t alignedSize = 16) {
    if constexpr (hasPse == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = pseInfo.n2oIdx * pseInfo.gSize;
        int64_t gOffset = pseInfo.goIdx;

        if (pseInfo.pseShapeType == pseSlopeBn) {
            bOffset = pseInfo.boIdx * pseInfo.n2G;
        }
        int64_t offset = bOffset + n2Offset + gOffset;

        DataCopyIn<half, hasPse>(helpTensor, alibiGm, 0, pseInfo.vec1S1RealSize,
                                 pseInfo.s2RealSize, pseInfo.pseAlibiBaseS2, alignedSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (pseInfo.needCast) {
            int64_t computeSize = pseInfo.vec1S1RealSize * pseInfo.s2AlignedSize;
            Cast(dstTensor, helpTensor, RoundMode::CAST_NONE, computeSize);
            pipe_barrier(PIPE_V);

            int64_t s1Offset = pseInfo.s1oIdx * pseInfo.s1BaseSize + pseInfo.vecCoreOffset +
                               pseInfo.loopIdx * pseInfo.vec1S1BaseSize;
            int64_t s2Offset = pseInfo.s2StartIdx + pseInfo.s2LoopCount * pseInfo.s2BaseNratioSize;

            float posShift = float(s2Offset + pseInfo.kvStartIdx - s1Offset - pseInfo.qStartIdx);

            Adds(dstTensor, dstTensor, posShift, computeSize);
            pipe_barrier(PIPE_V);
            Abs(dstTensor, dstTensor, computeSize);
            pipe_barrier(PIPE_V);
            float slopes = ((__gm__ T *)pseSlope)[offset] * -1;
            if (pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                Sqrt(dstTensor, dstTensor, computeSize);
                pipe_barrier(PIPE_V);
            }
            Muls(dstTensor, dstTensor, slopes, computeSize);
            pipe_barrier(PIPE_V);
        }
    }
}

template <typename T, bool hasPse>
__aicore__ inline void PseSlopeCast(LocalTensor<T> &dstTensor, LocalTensor<half> &helpTensor,
                                    __gm__ uint8_t *pseSlope, PseInfo &pseInfo) {
    if constexpr (hasPse == true) {
        int64_t bOffset = 0;
        int64_t n2Offset = pseInfo.n2oIdx * pseInfo.gSize;
        int64_t gOffset = pseInfo.goIdx;

        if (pseInfo.pseShapeType == pseSlopeBn) {
            bOffset = pseInfo.boIdx * pseInfo.n2G;
        }
        int64_t offset = bOffset + n2Offset + gOffset;
        int64_t computeSize = pseInfo.vec1S1RealSize * pseInfo.s2AlignedSize;
        Cast(dstTensor, helpTensor, RoundMode::CAST_NONE, computeSize);
        pipe_barrier(PIPE_V);

        int64_t s1Offset = pseInfo.s1oIdx * pseInfo.s1BaseSize + pseInfo.vecCoreOffset +
                           pseInfo.loopIdx * pseInfo.vec1S1BaseSize;
        int64_t s2Offset = pseInfo.s2StartIdx + pseInfo.s2LoopCount * pseInfo.s2BaseNratioSize;

        float posShift = float(s2Offset + pseInfo.kvStartIdx - s1Offset - pseInfo.qStartIdx);

        Adds(dstTensor, dstTensor, posShift, computeSize);
        pipe_barrier(PIPE_V);
        Abs(dstTensor, dstTensor, computeSize);
        pipe_barrier(PIPE_V);
        float slopes = ((__gm__ T *)pseSlope)[offset] * -1;
        if (pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            Sqrt(dstTensor, dstTensor, computeSize);
            pipe_barrier(PIPE_V);
        }
        Muls(dstTensor, dstTensor, slopes, computeSize);
        pipe_barrier(PIPE_V);
    }
}

template <typename INPUT_T, typename T, LayOutTypeEnum layOutType, bool hasPse>
__aicore__ inline void PseCopyIn(LocalTensor<T> &dstTensor, LocalTensor<INPUT_T> &tmpTensor,
                                 GlobalTensor<INPUT_T> &srcTensor, PseInfo &pseInfo, int64_t alignedSize = 16)
{
    if constexpr (hasPse == true) {
        if (pseInfo.pseEncodeType == pseEncodeALibiS2Full) {
            return PseAlibiCopyIn<INPUT_T, T, layOutType, hasPse>(dstTensor, tmpTensor, srcTensor, pseInfo, alignedSize);
        }
        int64_t offset = PseComputeOffset<hasPse>(pseInfo);
        int64_t s1Size = pseInfo.pseShapeType == pse1S2 ? (pseInfo.blockCount == 0 ? 1 : pseInfo.blockCount) :
                                                          pseInfo.vec1S1RealSize;

        if constexpr (IsSameType<INPUT_T, T>::value) {
            if (!pseInfo.align8){
                DataCopyIn<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, pseInfo.s2RealSize,
                                            pseInfo.s2Size, alignedSize);
            } else {
                DataCopyInAlign8<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, pseInfo.s2RealSize, pseInfo.s2Size);
            }
            return;
        }
        DataCopyIn<INPUT_T, hasPse>(tmpTensor, srcTensor, offset, s1Size, pseInfo.s2RealSize, pseInfo.s2Size,
                                    alignedSize);
        if (pseInfo.needCast) {
            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(dstTensor, tmpTensor, RoundMode::CAST_NONE, s1Size * pseInfo.s2AlignedSize);
        }
        return;
    }
}

template <typename T, bool hasPse>
__aicore__ inline void PseAlibiCompute(LocalTensor<T> &dstTensor, LocalTensor<T> &pseTensor, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        if (!NeedPseAlibiCompute<hasPse>(pseInfo)) {
            return;
        }
        Add(dstTensor, dstTensor, pseTensor, pseInfo.vec1S1RealSize * pseInfo.pseS2ComputeSize);
        return;
    }
}

template <typename T, bool hasPse>
__aicore__ inline void PseCompute(LocalTensor<T> &dstTensor, LocalTensor<T> &pseTensor, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        if (pseInfo.pseEncodeType == pseEncodeALibiS2Full) {
            return PseAlibiCompute<T, hasPse>(dstTensor, pseTensor, pseInfo);
        }
        int64_t computeSize = (pseInfo.pseShapeType == pseS1S2 || pseInfo.pseShapeType == pseSlopeBn ||
                               pseInfo.pseShapeType == pseSlopeN)
                              ? pseInfo.vec1S1RealSize * pseInfo.s2AlignedSize
                              : pseInfo.s2AlignedSize;
        PseBroadcastAdd<T, hasPse>(pseInfo.vec1S1RealSize, pseInfo.s2AlignedSize, computeSize, pseTensor,
                                   dstTensor, pseInfo.pseShapeType);
        return;
    }
}

template <bool hasPse>
__aicore__ inline void PseInnerAlibiCreate(GlobalTensor<half> &dstTensor, LocalTensor<half> &helpTensor, PseInfo &pseInfo) {
    if constexpr (hasPse == true) {
        if (pseInfo.pseType != (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE && pseInfo.pseType != (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            return;
        }
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        float tmpValue = -1.0;

        for (int64_t i = 0; i < pseInfo.pseAlibiBaseS1; i++) {
            CreateVecIndex(helpTensor, (half)(i * tmpValue), pseInfo.pseAlibiBaseS2);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(dstTensor[i * pseInfo.pseAlibiBaseS2], helpTensor, pseInfo.pseAlibiBaseS2);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        }
    }
}
#endif
