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
 * \file anti_quant.h
 * \brief
 */

#ifndef ANTIQUANT_H
#define ANTIQUANT_H

#include "kernel_operator.h"

using AscendC::Adds;
using AscendC::AIC;
using AscendC::BinaryRepeatParams;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::Muls;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_REPEAT_BYTE_SIZE;
using AscendC::PipeBarrier;
using AscendC::TBuf;

struct Mc2BroadCastPerGroupLoopParams {
    uint32_t mainLoopGroupCount;
    uint32_t tailGroupSize;
};

struct Mc2AntiQuantTensorShape {
    uint32_t srcK{1};
    uint32_t srcN{1};
    uint32_t srcOrigK{1};
    uint32_t srcOrigN{1};
    bool scaleKFull{true};
    uint32_t scaleK{1};
    uint32_t scaleN{1};
    uint32_t scaleOrigK{1};
    uint32_t scaleOrigN{1};
};

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithBroadcastHelper(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                    const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                    uint8_t repeatTimesDim0, const BinaryRepeatParams &params,
                                                    uint32_t loopDim1, uint32_t tailDim1)
{
    // (dim0, dim1)
    constexpr uint32_t oneRepeatSize = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if constexpr (HasOffset) {
        for (int i = 0; i < loopDim1; ++i) {
            Add(src[i * oneRepeatSize], src[i * oneRepeatSize], offset, oneRepeatSize, repeatTimesDim0, params);
        }
        if (tailDim1 > 0) {
            Add(src[loopDim1 * oneRepeatSize], src[loopDim1 * oneRepeatSize], offset, tailDim1, repeatTimesDim0,
                params);
        }
        PipeBarrier<PIPE_V>();
    }
    for (int i = 0; i < loopDim1; ++i) {
        Mul(dst[i * oneRepeatSize], src[i * oneRepeatSize], scale, oneRepeatSize, repeatTimesDim0, params);
    }
    if (tailDim1 > 0) {
        Mul(dst[loopDim1 * oneRepeatSize], src[loopDim1 * oneRepeatSize], scale, tailDim1, repeatTimesDim0, params);
    }
}

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithBroadcastPerChannel(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                        const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                        const Mc2AntiQuantTensorShape &tensorShape)
{
    // per-channel
    //   src shape (n, k_align) ori_shape (n, k)
    //   scale/offset (n_align8, 32B)
    constexpr uint32_t oneRepeatSize = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    auto n = tensorShape.srcK;
    auto kAlign = tensorShape.srcN;
    auto oriK = tensorShape.srcOrigN;
    int32_t mainLoop = oriK / oneRepeatSize;
    int32_t tailElements = oriK - mainLoop * oneRepeatSize;
    BinaryRepeatParams repeatParams;
    repeatParams.dstRepStride = kAlign / (ONE_BLK_SIZE / sizeof(T));
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(n < 256, { KERNEL_LOG(KERNEL_ERROR, "repeatTimes must < 256, actual is %d", n); });
    int32_t dstRepStride = kAlign / (ONE_BLK_SIZE / sizeof(T));
    ASCENDC_ASSERT(dstRepStride < 256, {
        KERNEL_LOG(KERNEL_ERROR, "dstRepStride/src1RepStride(%d) must < 256, actual is %d", dstRepStride);
    });
#endif
    Mc2AddMulWithBroadcastHelper<T, HasOffset>(dst, src, scale, offset, n, repeatParams, mainLoop, tailElements);
}

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithBroadcastPerGroupAtRepeat(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                              const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                              const Mc2BroadCastPerGroupLoopParams &loopParams,
                                                              const BinaryRepeatParams &repeatParamsK,
                                                              uint32_t groupSize, uint8_t repeatTimes)
{
    constexpr uint32_t elemsOneBlock = 32 / sizeof(T);
    constexpr uint32_t oneMaskSize = ONE_REPEAT_BYTE_SIZE / sizeof(T);

    uint32_t maskLoop = groupSize / oneMaskSize;
    uint32_t maskTail = groupSize - maskLoop * oneMaskSize;
    for (uint32_t i = 0; i < loopParams.mainLoopGroupCount; ++i) {
        Mc2AddMulWithBroadcastHelper<T, HasOffset>(dst[i * groupSize], src[i * groupSize], scale[i * elemsOneBlock],
                                                   offset[i * elemsOneBlock], repeatTimes, repeatParamsK, maskLoop,
                                                   maskTail);
    }

    if (loopParams.tailGroupSize > 0) {
        maskLoop = loopParams.tailGroupSize / oneMaskSize;
        maskTail = loopParams.tailGroupSize - maskLoop * oneMaskSize;
        Mc2AddMulWithBroadcastHelper<T, HasOffset>(
            dst[loopParams.mainLoopGroupCount * groupSize], src[loopParams.mainLoopGroupCount * groupSize],
            scale[loopParams.mainLoopGroupCount * elemsOneBlock], offset[loopParams.mainLoopGroupCount * elemsOneBlock],
            repeatTimes, repeatParamsK, maskLoop, maskTail);
    }
}

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithBroadcastPerGroup(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                      const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                      const Mc2AntiQuantTensorShape &tensorShape, uint32_t groupSize)
{
    // per-group
    //   src:
    //     shape (n, k_align)
    //     ori_shape: (n, k) k = x * groupSize + tailGroupSize
    int32_t mainLoopK;
    int32_t tailK;
    BinaryRepeatParams repeatParamsK;
    Mc2BroadCastPerGroupLoopParams loopParams;
    auto kAlign = tensorShape.srcN;
    auto k = tensorShape.srcOrigN;
    auto n = tensorShape.srcOrigK;
    loopParams.mainLoopGroupCount = k / groupSize;
    loopParams.tailGroupSize = k - loopParams.mainLoopGroupCount * groupSize;
    if (tensorShape.scaleKFull) {
        // full load group_count
        //   scale/offset:
        //     shape ((n*gc)_align, 32B)
        //     ori_shape (n, gc, 32B)
        auto realGroupCount = loopParams.mainLoopGroupCount + (loopParams.tailGroupSize > 0 ? 1 : 0);

        repeatParamsK.dstRepStride = kAlign / (ONE_BLK_SIZE / sizeof(T));
        repeatParamsK.src0BlkStride = 1;
        repeatParamsK.src0RepStride = repeatParamsK.dstRepStride;
        repeatParamsK.src1BlkStride = 0;
        repeatParamsK.src1RepStride = realGroupCount;
    } else {
        // not full load group_count
        //   scale/offset:
        //     shape (n, gc_align, 32B)
        //     ori_shape (n, gc, 32B)
        repeatParamsK.dstRepStride = kAlign / (ONE_BLK_SIZE / sizeof(T));
        repeatParamsK.src0BlkStride = 1;
        repeatParamsK.src0RepStride = repeatParamsK.dstRepStride;
        repeatParamsK.src1BlkStride = 0;
        repeatParamsK.src1RepStride = tensorShape.scaleN;
    }

    constexpr uint32_t repeatMax = 255;
    int32_t repeatLoop = n / repeatMax;
    int32_t repeatTail = n % repeatMax;
    for (int repeatIdx = 0; repeatIdx < repeatLoop; repeatIdx++) {
        uint32_t srcOffset = repeatIdx * repeatMax * k;
        uint32_t antiquantOffset = repeatIdx * repeatMax * (ONE_BLK_SIZE / sizeof(T));
        Mc2AddMulWithBroadcastPerGroupAtRepeat<T, HasOffset>(dst[srcOffset], src[srcOffset], scale[antiquantOffset],
                                                             offset[antiquantOffset], loopParams, repeatParamsK,
                                                             groupSize, repeatMax);
    }
    if (repeatTail > 0) {
        int32_t srcOffset = repeatLoop * repeatMax * k;
        int32_t antiquantOffset = repeatLoop * repeatMax * (ONE_BLK_SIZE / sizeof(T));
        Mc2AddMulWithBroadcastPerGroupAtRepeat<T, HasOffset>(dst[srcOffset], src[srcOffset], scale[antiquantOffset],
                                                             offset[antiquantOffset], loopParams, repeatParamsK,
                                                             groupSize, repeatTail);
    }
}

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithoutBroadcastHelper(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                       const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                       uint8_t repeatTimesDim0, const BinaryRepeatParams &params,
                                                       uint32_t loopDim1, uint32_t tailDim1)
{
    constexpr uint32_t oneRepeatSize = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if constexpr (HasOffset) {
        for (int i = 0; i < loopDim1; ++i) {
            Add(src[i * oneRepeatSize], src[i * oneRepeatSize], offset[i * oneRepeatSize], oneRepeatSize,
                repeatTimesDim0, params);
        }
        if (tailDim1 > 0) {
            Add(src[loopDim1 * oneRepeatSize], src[loopDim1 * oneRepeatSize], offset[loopDim1 * oneRepeatSize],
                tailDim1, repeatTimesDim0, params);
        }
        PipeBarrier<PIPE_V>();
    }
    for (int i = 0; i < loopDim1; ++i) {
        Mul(dst[i * oneRepeatSize], src[i * oneRepeatSize], scale[i * oneRepeatSize], oneRepeatSize, repeatTimesDim0,
            params);
    }
    if (tailDim1 > 0) {
        Mul(dst[loopDim1 * oneRepeatSize], src[loopDim1 * oneRepeatSize], scale[loopDim1 * oneRepeatSize], tailDim1,
            repeatTimesDim0, params);
    }
}

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithoutBroadcastPerChannel(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                           const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                           const Mc2AntiQuantTensorShape &tensorShape)
{
    // src
    //   shape (k, n)
    // scale/offset
    //   shape (1, n)
    auto k = tensorShape.srcK;
    auto n = tensorShape.srcN;
    constexpr uint32_t oneRepeatSize = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    int32_t mainLoop = n / oneRepeatSize;
    int32_t tailElements = n - mainLoop * oneRepeatSize;
    BinaryRepeatParams repeatParams;
    repeatParams.dstRepStride = n / (ONE_BLK_SIZE / sizeof(T));
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1BlkStride = 1;
    repeatParams.src1RepStride = 0;
#if defined(__CCE_KT_TEST__)
    int32_t dstRepStride = n / (ONE_BLK_SIZE / sizeof(T));
    ASCENDC_ASSERT(dstRepStride < 256,
                   { KERNEL_LOG(KERNEL_ERROR, "dstRepStride/src0RepStride(%d) must < 256", dstRepStride); });
#endif
    Mc2AddMulWithoutBroadcastHelper<T, HasOffset>(dst, src, scale, offset, k, repeatParams, mainLoop, tailElements);
}

template <typename SrcDataType, typename ScaleOffsetDataType, typename DstDataType, bool IsTranspose, bool HasOffset>
__aicore__ inline void Mc2AntiQuant(LocalTensor<DstDataType> &dst, const LocalTensor<SrcDataType> &src,
                                    const ScaleOffsetDataType &scale, const ScaleOffsetDataType &offset,
                                    TBuf<> &sharedTmpBuffer, const int64_t groupSize = 0)
{
    if ASCEND_IS_AIC {
        return;
    }

#if defined(__CCE_KT_TEST__)
    if (IsSameType<SrcDataType, float>::value) {
        ASCENDC_ASSERT((IsSameType<ScaleOffsetDataType, float>::value && IsSameType<DstDataType, float>::value),
                       { KERNEL_LOG(KERNEL_ERROR, "dtype of src, scale, offset, dst must be float"); });
    }
    if (IsSameType<SrcDataType, half>::value) {
        ASCENDC_ASSERT((IsSameType<ScaleOffsetDataType, half>::value && IsSameType<DstDataType, half>::value),
                       { KERNEL_LOG(KERNEL_ERROR, "dtype of src, scale, offset, dst must be half"); });
    }
#endif
    uint32_t calCount = src.GetSize();
    if constexpr (IsSameType<SrcDataType, float>::value || IsSameType<SrcDataType, half>::value) {
        // preprocess: f32->f32
        // preprocess: f16->f16
        if constexpr (HasOffset) {
            Adds(src, src, offset, calCount);
            PipeBarrier<PIPE_V>();
        }
        Muls(dst, src, scale, calCount);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__)
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support this scenario"); });
#endif
    }
}

template <typename SrcDataType, typename ScaleOffsetDataType, typename DstDataType, bool HasOffset>
__aicore__ inline void Mc2AscendAntiQuantPerGroupWithTranspose(LocalTensor<DstDataType> &dst,
                                                               const LocalTensor<SrcDataType> &src,
                                                               const LocalTensor<ScaleOffsetDataType> &scale,
                                                               const LocalTensor<ScaleOffsetDataType> &offset,
                                                               const Mc2AntiQuantTensorShape &tensorShape,
                                                               TBuf<> &sharedTmpBuffer, uint32_t groupSize)
{
    if constexpr (IsSameType<SrcDataType, float>::value || IsSameType<SrcDataType, half>::value) {
        // preprocess: f32->f32
        // preprocess: f16->f16
        Mc2AddMulWithBroadcastPerGroup<SrcDataType, HasOffset>(dst, src, scale, offset, tensorShape, groupSize);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__)
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support this scenario"); });
#endif
    }
}

template <typename SrcDataType, typename ScaleOffsetDataType, typename DstDataType, bool HasOffset>
__aicore__ inline void Mc2AscendAntiQuantPerChannelWithTranspose(
    LocalTensor<DstDataType> &dst, const LocalTensor<SrcDataType> &src, const LocalTensor<ScaleOffsetDataType> &scale,
    const LocalTensor<ScaleOffsetDataType> &offset, const Mc2AntiQuantTensorShape &tensorShape, TBuf<> &sharedTmpBuffer)
{
    if constexpr (IsSameType<SrcDataType, float>::value || IsSameType<SrcDataType, half>::value) {
        // preprocess: f32->f32
        // preprocess: f16->f16
        Mc2AddMulWithBroadcastPerChannel<SrcDataType, HasOffset>(dst, src, scale, offset, tensorShape);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__)
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support this scenario"); });
#endif
    }
}

template <typename T, bool HasOffset>
__aicore__ inline void Mc2AddMulWithoutBroadcastPerGroup(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                         const LocalTensor<T> &scale, const LocalTensor<T> &offset,
                                                         const Mc2AntiQuantTensorShape &tensorShape, uint32_t groupSize,
                                                         uint32_t preGroupSize)
{
    // src
    //   (k, n)
    //   k = preGroupSize + x * groupSize + tailGroupSize
    // scale
    //   (gc, n)
    // offset
    //   (gc, n)
    auto nAlign = tensorShape.srcN;
    auto scaleNAlign = tensorShape.scaleN;
    auto k = tensorShape.srcK;

    auto mainK = k - preGroupSize;
    auto mainLoopGroupCount = mainK / groupSize;
    auto tailGroupSize = mainK - mainLoopGroupCount * groupSize;
    int32_t oneRepeatSize = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    int32_t mainLoopN = nAlign / oneRepeatSize;
    int32_t tailN = nAlign - mainLoopN * oneRepeatSize;
    BinaryRepeatParams repeatParamsN;
    repeatParamsN.dstRepStride = nAlign / (ONE_BLK_SIZE / sizeof(T));
    repeatParamsN.src0BlkStride = 1;
    repeatParamsN.src0RepStride = repeatParamsN.dstRepStride;
    repeatParamsN.src1BlkStride = 1;
    repeatParamsN.src1RepStride = 0;

    uint32_t offsetSrc = 0;
    uint32_t offsetScaleOffset = 0;
    if (preGroupSize > 0) {
        Mc2AddMulWithoutBroadcastHelper<T, HasOffset>(dst, src, scale, offset, preGroupSize, repeatParamsN, mainLoopN,
                                                      tailN);
        offsetSrc = preGroupSize * nAlign;
        offsetScaleOffset = nAlign;
    }
    for (int i = 0; i < mainLoopGroupCount; ++i) {
        Mc2AddMulWithoutBroadcastHelper<T, HasOffset>(
            dst[offsetSrc + i * groupSize * nAlign], src[offsetSrc + i * groupSize * nAlign],
            scale[offsetScaleOffset + i * scaleNAlign], offset[offsetScaleOffset + i * scaleNAlign], groupSize,
            repeatParamsN, mainLoopN, tailN);
    }
    if (tailGroupSize > 0) {
        Mc2AddMulWithoutBroadcastHelper<T, HasOffset>(dst[offsetSrc + mainLoopGroupCount * groupSize * nAlign],
                                                      src[offsetSrc + mainLoopGroupCount * groupSize * nAlign],
                                                      scale[offsetScaleOffset + mainLoopGroupCount * scaleNAlign],
                                                      offset[offsetScaleOffset + mainLoopGroupCount * scaleNAlign],
                                                      tailGroupSize, repeatParamsN, mainLoopN, tailN);
    }
}

template <typename SrcDataType, typename ScaleOffsetDataType, typename DstDataType, bool HasOffset>
__aicore__ inline void Mc2AscendAntiQuantPerGroupWithoutTranspose(
    LocalTensor<DstDataType> &dst, const LocalTensor<SrcDataType> &src, const LocalTensor<ScaleOffsetDataType> &scale,
    const LocalTensor<ScaleOffsetDataType> &offset, const Mc2AntiQuantTensorShape &tensorShape, TBuf<> &sharedTmpBuffer,
    uint32_t groupSize, uint32_t preGroupSize)
{
    if constexpr (IsSameType<SrcDataType, float>::value || IsSameType<SrcDataType, half>::value) {
        // preprocess: f32->f32
        // preprocess: f16->f16
        Mc2AddMulWithoutBroadcastPerGroup<SrcDataType, HasOffset>(dst, src, scale, offset, tensorShape, groupSize,
                                                                  preGroupSize);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__)
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support this scenario"); });
#endif
    }
}

template <typename SrcDataType, typename ScaleOffsetDataType, typename DstDataType, bool HasOffset>
__aicore__ inline void Mc2AscendAntiQuantPerChannelWithoutTranspose(
    LocalTensor<DstDataType> &dst, const LocalTensor<SrcDataType> &src, const LocalTensor<ScaleOffsetDataType> &scale,
    const LocalTensor<ScaleOffsetDataType> &offset, const Mc2AntiQuantTensorShape &tensorShape, TBuf<> &sharedTmpBuffer)
{
    if constexpr (IsSameType<SrcDataType, float>::value || IsSameType<SrcDataType, half>::value) {
        // preprocess: f32->f32
        // preprocess: f16->f16
        Mc2AddMulWithoutBroadcastPerChannel<SrcDataType, HasOffset>(dst, src, scale, offset, tensorShape);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__)
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support this scenario"); });
#endif
    }
}

template <typename SrcDataType, typename ScaleOffsetDataType, typename DstDataType, bool IsTranspose, bool HasOffset>
__aicore__ inline void Mc2AntiQuant(LocalTensor<DstDataType> &dst, const LocalTensor<SrcDataType> &src,
                                    const LocalTensor<ScaleOffsetDataType> &scale,
                                    const LocalTensor<ScaleOffsetDataType> &offset,
                                    const Mc2AntiQuantTensorShape &tensorShape, TBuf<> &sharedTmpBuffer,
                                    const int64_t groupSize = 0, const uint32_t preGroupSize = 0)
{
    if ASCEND_IS_AIC {
        return;
    }

#if defined(__CCE_KT_TEST__)
    if (IsSameType<SrcDataType, float>::value) {
        ASCENDC_ASSERT((IsSameType<ScaleOffsetDataType, float>::value && IsSameType<DstDataType, float>::value),
                       { KERNEL_LOG(KERNEL_ERROR, "dtype of src, scale, offset, dst must be float"); });
    }
    if (IsSameType<SrcDataType, half>::value) {
        ASCENDC_ASSERT((IsSameType<ScaleOffsetDataType, half>::value && IsSameType<DstDataType, half>::value),
                       { KERNEL_LOG(KERNEL_ERROR, "dtype of src, scale, offset, dst must be half"); });
    }
#endif

    if (groupSize == 0) {
        if constexpr (IsTranspose) {
            // src (n, k)
            // scale/offset (n, 32B)
            Mc2AscendAntiQuantPerChannelWithTranspose<SrcDataType, ScaleOffsetDataType, DstDataType, HasOffset>(
                dst, src, scale, offset, tensorShape, sharedTmpBuffer);
        } else {
            // src (k, n)
            // scale/offset (1, n)
            Mc2AscendAntiQuantPerChannelWithoutTranspose<SrcDataType, ScaleOffsetDataType, DstDataType, HasOffset>(
                dst, src, scale, offset, tensorShape, sharedTmpBuffer);
        }
    } else {
        if constexpr (IsTranspose) {
            // src (n, k)
            // scale/offset
            //   full load group-count (n, gc, 32B)
            //   not full load group-count (n*gc, 32B)
#if defined(__CCE_KT_TEST__)
            ASCENDC_ASSERT(preGroupSize == 0,
                           { KERNEL_LOG(KERNEL_ERROR, "preGroupSize must = 0, actual is %d", preGroupSize); });
#endif
            Mc2AscendAntiQuantPerGroupWithTranspose<SrcDataType, ScaleOffsetDataType, DstDataType, HasOffset>(
                dst, src, scale, offset, tensorShape, sharedTmpBuffer, groupSize);
        } else {
            // src (k, n)
            // scale/offset (gc, n)
            Mc2AscendAntiQuantPerGroupWithoutTranspose<SrcDataType, ScaleOffsetDataType, DstDataType, HasOffset>(
                dst, src, scale, offset, tensorShape, sharedTmpBuffer, groupSize, preGroupSize);
        }
    }
}

#endif
