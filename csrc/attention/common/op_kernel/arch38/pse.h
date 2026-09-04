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
 * \file pse.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_PSE_H
#define FLASH_ATTENTION_SCORE_PSE_H

#include "kernel_operator.h"
#include "util_regbase.h"

namespace regbaseutil {
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
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

struct PseInfo {
    int64_t pseBSize;         // pse输入batch大小
    int64_t pseS1Size;        // for alibi
    int64_t pseS2ComputeSize; // for alibi, do not need assignment
    int64_t pseS2Size;        // for alibi
    int64_t readS2Size;       // for alibi, do not need assignment
    uint32_t pseLayoutType;   // pse输入shape的layout
    uint32_t pseEncodeType;   // for distinguish alibi
    uint32_t pseType;         // 0: outer, mul-add   1:outer, add-mul   2:inner, mul-add   3:inner, mul-add-sqrt
    uint32_t pseStride;
    int64_t qStartIdx;
    int64_t kvStartIdx;
};

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyInCommon(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                        int64_t offset, int64_t s1Size, int64_t s2Size, int64_t actualS2Len,
                                        int32_t s2BaseSize)
{
    if constexpr (hasPse == true) {
        if (s1Size == 0 || s2Size == 0) {
            return;
        }
        int32_t dtypeSize = sizeof(INPUT_T);
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s1Size;
        dataCopyParams.blockLen = CeilDiv(s2Size * dtypeSize, blockBytes); // 单位32B
        dataCopyParams.dstStride = CeilDiv(s2BaseSize * dtypeSize, blockBytes) - dataCopyParams.blockLen;
        if (actualS2Len * dtypeSize % blockBytes == 0) {
            dataCopyParams.srcStride = (actualS2Len * dtypeSize - dataCopyParams.blockLen * blockBytes) / blockBytes;
            DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
        } else {
            DataCopyExtParams dataCopyExtParams;
            dataCopyExtParams.blockCount = s1Size;
            dataCopyExtParams.blockLen = s2Size * dtypeSize;
            dataCopyExtParams.srcStride = actualS2Len * dtypeSize - dataCopyExtParams.blockLen;
            dataCopyExtParams.dstStride = CeilDiv(s2BaseSize * dtypeSize, blockBytes) - CeilDiv(s2Size * dtypeSize, blockBytes);
            DataCopyPadExtParams<INPUT_T> dataCopyPadParams;
            DataCopyPad(dstTensor, srcTensor[offset], dataCopyExtParams, dataCopyPadParams);
        }
    }
}

template <typename INPUT_T, bool hasPse>
__aicore__ inline void DataCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
                                  int64_t s1Size, int64_t s2Size, int64_t s2BaseSize, int64_t actualS2Len)
{
    if constexpr (hasPse == true) {
        DataCopyInCommon<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, s2Size, actualS2Len, s2BaseSize);
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
            actualS2Len, alignedS2Size);
    }
}

template <bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline int64_t PseComputeOffset(const RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        if constexpr (isInfer) {
            return runInfo.pseShiftOffset;
        } else {
            int64_t bOffset = 0;
            int64_t n2Offset = 0;
            int64_t s1Offset = 0;
            int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
            int64_t gOffset = 0;
            if (pseInfo.pseLayoutType == pseS1S2) {
                // b, n2, g, s1, s2
                bOffset = runInfo.b1SSOffset * constInfo.n2G;
                n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS1Size * runInfo.actualS2Size;
                gOffset = runInfo.goIdx * runInfo.actualS1Size * runInfo.actualS2Size;
                s1Offset = (runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset) * runInfo.actualS2Size;
            } else if (pseInfo.pseLayoutType == pse1S2) {
                // b, n2, g, 1, s2
                bOffset = runInfo.s2SizeAcc * constInfo.n2G;
                n2Offset = runInfo.n2oIdx * constInfo.gSize * runInfo.actualS2Size;
                gOffset = runInfo.goIdx * runInfo.actualS2Size;
                s1Offset = 0;
            }
            if (pseInfo.pseBSize == 1) {
                bOffset = 0;
            }
            return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
        }
    } else {
        return 0;
    }
}

template <bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline int64_t PseAlibiComputeOffset(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        int64_t bOffset = (runInfo.boIdx % pseInfo.pseBSize) * constInfo.n2G * pseInfo.pseS2Size * pseInfo.pseS1Size;
        int64_t n2Offset = runInfo.n2oIdx * constInfo.gSize * pseInfo.pseS2Size * pseInfo.pseS1Size;
        int64_t gOffset = runInfo.goIdx * pseInfo.pseS2Size * pseInfo.pseS1Size;
        int64_t row = runInfo.s1oIdx * constInfo.s1BaseSize;
        int64_t column = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
        int64_t m = 0;
        int64_t k = 0;
        if (constInfo.layoutType != (uint32_t)LayOutTypeEnum::LAYOUT_TND) {
            int64_t threshold = runInfo.actualS1Size - pseInfo.pseS1Size;
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
        int64_t s1Offset = m * pseInfo.pseS2Size + runInfo.vecCoreOffset * pseInfo.pseS2Size;
        int64_t s2Offset = k;
        pseInfo.readS2Size = Min(runInfo.s2AlignedSize, pseInfo.pseS2Size - k);
        pseInfo.pseS2ComputeSize = Align(pseInfo.readS2Size);

        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    } else {
        return 0;
    }
}

template <bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline bool NeedPseAlibiCompute(const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        // Alibi编码只计算下三角
        if (runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.halfS1RealSize <=
            runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize) {
            return false;
        }
        return true;
    } else {
        return false;
    }
}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void PseAlibiCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                      const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        if (!NeedPseAlibiCompute<hasPse>(runInfo, constInfo, pseInfo)) {
            return;
        }
        int64_t offset = PseAlibiComputeOffset<hasPse>(runInfo, constInfo, pseInfo);
        if constexpr (IsSameType<INPUT_T, T>::value) {
            DataCopyIn<INPUT_T, hasPse>(dstTensor, srcTensor, offset, runInfo.halfS1RealSize, pseInfo.readS2Size,
                                        constInfo.s2BaseSize, pseInfo.pseS2Size);
            return;
        }

        DataCopyIn<INPUT_T, hasPse>(dstTensor, srcTensor, offset, runInfo.halfS1RealSize, pseInfo.readS2Size,
                                    constInfo.s2BaseSize, pseInfo.pseS2Size);
        return;
    }
}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void PseCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                 const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        if (pseInfo.pseEncodeType == pseEncodeALibiS2Full) {
            return PseAlibiCopyIn<T, INPUT_T, hasPse>(dstTensor, srcTensor, runInfo, constInfo, pseInfo);
        }
        int64_t offset = PseComputeOffset<hasPse>(runInfo, constInfo, pseInfo);
        int64_t s1Size = pseInfo.pseLayoutType == pse1S2 ? 1 : runInfo.halfS1RealSize;
        int64_t pseS2Size;
        if constexpr (isInfer) {
            pseS2Size = pseInfo.pseS2Size;
        } else {
            pseS2Size = runInfo.actualS2Size;
        }
        if constexpr (IsSameType<INPUT_T, T>::value) {
            DataCopyIn<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, runInfo.s2RealSize,
                                        constInfo.s2BaseSize, pseS2Size);
            return;
        }
        DataCopyIn<INPUT_T, hasPse>(dstTensor, srcTensor, offset, s1Size, runInfo.s2RealSize, constInfo.s2BaseSize,
                                    pseS2Size);
        return;
    }
}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void PseCopyIn(TQue<QuePosition::VECIN, 1> &pseInQue, GlobalTensor<INPUT_T> &srcTensor,
                                 const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo, PseInfo &pseInfo)
{
    if constexpr (hasPse == true) {
        LocalTensor<INPUT_T> pseUb = pseInQue.template AllocTensor<INPUT_T>();
        if (pseInfo.pseEncodeType == pseEncodeALibiS2Full) {
            PseAlibiCopyIn<T, INPUT_T, hasPse>(pseUb, srcTensor, runInfo, constInfo, pseInfo);
            pseInQue.template EnQue(pseUb);
            return;
        }
        int64_t offset = PseComputeOffset<hasPse>(runInfo, constInfo, pseInfo);
        int64_t s1Size = pseInfo.pseLayoutType == pse1S2 ? 1 : runInfo.halfS1RealSize;
        int64_t pseS2Size;
        if constexpr (isInfer) {
            pseS2Size = pseInfo.pseS2Size;
        } else {
            pseS2Size = runInfo.actualS2Size;
        }
        if constexpr (IsSameType<INPUT_T, T>::value) {
            DataCopyIn<INPUT_T, hasPse>(pseUb, srcTensor, offset, s1Size, runInfo.s2RealSize, constInfo.s2BaseSize,
                                        pseS2Size);
            pseInQue.template EnQue(pseUb);
            return;
        }
        DataCopyIn<INPUT_T, hasPse>(pseUb, srcTensor, offset, s1Size, runInfo.s2RealSize, constInfo.s2BaseSize,
                                    pseS2Size);
        pseInQue.template EnQue(pseUb);
        return;
    }
}

template <typename T, typename INPUT_T, bool hasPse, bool isInfer = false, bool hasRope = false>
__aicore__ inline void ComputeInnerPseOffset(float &slopes, float &posShift, const RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo,
                                             PseInfo &pseInfo, __gm__ uint8_t *pseSlope)
{
    if constexpr (hasPse)
    {
        if (pseInfo.pseType != (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE &&
            pseInfo.pseType != (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            return;
        }
        int64_t bOffset = 0;
        int64_t n2Offset = runInfo.n2oIdx * constInfo.gSize;
        int64_t gOffset = runInfo.goIdx;
        if (pseInfo.pseLayoutType == pseSlopeBn) {
            bOffset = runInfo.boIdx * constInfo.n2G;
        }
        int64_t offset = bOffset + n2Offset + gOffset;
        slopes = ((__gm__ T *)pseSlope)[offset] * -1;
        int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset;
        int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
        if constexpr (isInfer) {
            s1Offset += (runInfo.nextTokensPerBatch < 0) ? -runInfo.nextTokensPerBatch : 0;
        }
        posShift = float(s2Offset + pseInfo.kvStartIdx - s1Offset - pseInfo.qStartIdx);
        return;
    }
}
}
#endif
