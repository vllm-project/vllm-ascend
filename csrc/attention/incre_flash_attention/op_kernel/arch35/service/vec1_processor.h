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
 * \file vec1_processor.h
 * \brief
 */

#ifndef VEC1_PROCESSOR_H
#define VEC1_PROCESSOR_H

#include "../incre_flash_attention_pub.h"
#include "../vector_api/vf_softmax_const.h"
#if __has_include("../../../../common/op_kernel/arch35/vf/vf_flash_decode.h")
#include "../../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#else
#include "../../../common/arch35/vf/vf_flash_decode.h"
#endif

struct Vec1TaskParam {
    uint32_t dealRowCount;
    uint32_t columnCount;
    uint32_t actualColumnCount;
    uint32_t qSeqSize;
    uint64_t bIdx;
    uint64_t n2Idx;
    uint64_t gIdx;
    uint32_t qHeadNum;
    uint32_t splitKVNum;
    uint32_t gSize;
    uint32_t flashDecodeS2Idx;
    uint64_t attenMaskOffset;
    uint64_t attenMaskOffsetPre;
    uint64_t pseShiftS;
    uint64_t pseShiftOffset;
    uint64_t qPaddingBeginOffset;
    uint32_t attenMaskKVSize;
    uint32_t sparseMode;
    float scaleValue;
    bool isFirstSInnerLoop;
    bool isLastSInnerLoop;
    bool softmaxLseFlag;
};

template <typename IFAT>
class Vec1Processor
{
public:
    using T = float;
    using Q_T = typename IFAT::queryType;
    using MM_OUT_T = T;
    using MM_IN_T = Q_T;
    using PSE_SHIFT_T = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;

    static constexpr IFAProfile PROFILE = IFAT::profile;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr bool ATTEN_MASK = IFAT::attenMask;
    static constexpr bool PSE_SHIFT = IFAT::pseShift;

    __aicore__ inline Vec1Processor(){};
    __aicore__ inline void Process(TQue<QuePosition::VECOUT, 1>& vec1ResOutUb, TSCM<TPosition::VECIN, 1>& mm2AScm,
                              LocalTensor<T>& softmaxMaxOutUb, LocalTensor<T>& softmaxSumOutUb,
                              LocalTensor<T>& softmaxExpUb, LocalTensor<MM_OUT_T>& mm1ResUb,
                              LocalTensor<T>& softmaxMaxInUb, LocalTensor<T>& softmaxSumInUb,
                              LocalTensor<uint8_t>& softmaxTmpUb, GlobalTensor<uint8_t>& attenMaskGm,
                              TQue<QuePosition::VECIN, 1>& maskInputQue, GlobalTensor<PSE_SHIFT_T>& pseShiftGm,
                              TQue<QuePosition::VECIN, 1>& pseInputQue, TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue,
                              GlobalTensor<T>& softmaxLseGm, const Vec1TaskParam& taskParam, GlobalTensor<T>& softmaxMaxGm, 
                              GlobalTensor<T>& softmaxSumGm);
private:
    __aicore__ inline void Softmax(TQue<QuePosition::VECOUT, 1>& vec1ResOutUb, TSCM<TPosition::VECIN, 1>& mm2AScm,
                              LocalTensor<T>& softmaxMaxOutUb, LocalTensor<T>& softmaxSumOutUb,
                              LocalTensor<T>& softmaxExpUb, LocalTensor<MM_OUT_T>& mm1ResUb,
                              LocalTensor<T>& softmaxMaxInUb, LocalTensor<T>& softmaxSumInUb,
                              LocalTensor<uint8_t>& softmaxTmpUb, GlobalTensor<uint8_t>& attenMaskGm,
                              TQue<QuePosition::VECIN, 1>& maskInputQue, GlobalTensor<PSE_SHIFT_T>& pseShiftGm,
                              TQue<QuePosition::VECIN, 1>& pseInputQue, const Vec1TaskParam& taskParam);
    __aicore__ inline void AttenMaskCopyIn(GlobalTensor<uint8_t>& attenMaskGm,
                                           TQue<QuePosition::VECIN, 1>& maskInputQue, const Vec1TaskParam& taskParam);
    __aicore__ inline void AttenMaskCopyInPFA(GlobalTensor<uint8_t>& attenMaskGm,
                                              TQue<QuePosition::VECIN, 1>& maskInputQue, const Vec1TaskParam& taskParam);
    __aicore__ inline void PseShiftCopyIn(GlobalTensor<PSE_SHIFT_T>& pseShiftGm,
                                          TQue<QuePosition::VECIN, 1>& pseInputQue, const Vec1TaskParam& taskParam);
    __aicore__ inline void Vec1CopyIn(GlobalTensor<uint8_t>& attenMaskGm, TQue<QuePosition::VECIN, 1>& maskInputQue,
                                      GlobalTensor<PSE_SHIFT_T>& pseShiftGm, TQue<QuePosition::VECIN, 1>& pseInputQue,
                                      const Vec1TaskParam& taskParam);
    __aicore__ inline void Vec1Compute(LocalTensor<Q_T>& bmm2InResUb, LocalTensor<T>& softmaxMaxOutUb,
                                      LocalTensor<T>& softmaxSumOutUb, LocalTensor<T>& softmaxExpUb,
                                      LocalTensor<MM_OUT_T>& mm1ResUb, LocalTensor<T>& softmaxMaxInUb,
                                      LocalTensor<T>& softmaxSumInUb,LocalTensor<uint8_t>& softmaxTmpUb, TQue<QuePosition::VECIN, 1>& maskInputQue,
                                      TQue<QuePosition::VECIN, 1>& pseInputQue, const Vec1TaskParam& taskParam);
    __aicore__ inline void Vec1CopyResToL1(TSCM<TPosition::VECIN, 1>& mm2AScm, LocalTensor<Q_T>& bmm2InResUb,
                                           const Vec1TaskParam& taskParam);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue,
                                                       LocalTensor<T>& softmaxMaxOutUb, LocalTensor<T>& softmaxSumOutUb,
                                                       const Vec1TaskParam& taskParam, GlobalTensor<T>& softmaxMaxGm, GlobalTensor<T>& softmaxSumGm);
    __aicore__ inline void CopyFixedUbToGm(TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue, const GlobalTensor<T> &dst,
                                           const LocalTensor<T>& src, size_t size);
    __aicore__ inline void SoftmaxLseCopyOut(GlobalTensor<T>& softmaxLseGm, TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue,
                                             LocalTensor<T>& softmaxMaxOutUb, LocalTensor<T>& softmaxSumOutUb, const Vec1TaskParam& taskParam);
};

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::SoftmaxLseCopyOut(
    GlobalTensor<T>& softmaxLseGm, TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue, LocalTensor<T>& softmaxMaxOutUb,
    LocalTensor<T>& softmaxSumOutUb, const Vec1TaskParam& taskParam)
{
    LocalTensor<T> lseUb = softmaxLseOutputQue.template AllocTensor<T>();
    FaVectorApi::ComputeLogSumExp_VF(lseUb, softmaxSumOutUb, softmaxMaxOutUb, taskParam.dealRowCount);
    softmaxLseOutputQue.template EnQue(lseUb);
    softmaxLseOutputQue.DeQue<T>();
    DataCopyExtParams intriParams1;
    intriParams1.blockLen = sizeof(float);
    intriParams1.blockCount = taskParam.dealRowCount;
    intriParams1.srcStride = 0;
    intriParams1.dstStride = 0;
    DataCopyPad(softmaxLseGm[(taskParam.bIdx * taskParam.qHeadNum + taskParam.n2Idx * taskParam.gSize) *
                taskParam.qSeqSize + taskParam.gIdx * PROFILE.G + taskParam.qPaddingBeginOffset], lseUb, intriParams1);
    softmaxLseOutputQue.FreeTensor(lseUb);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::ComputeLogSumExpAndCopyToGm(
    TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue, LocalTensor<T>& softmaxMaxOutUb, LocalTensor<T>& softmaxSumOutUb,
    const Vec1TaskParam& taskParam, GlobalTensor<T>& softmaxMaxGm, GlobalTensor<T>& softmaxSumGm)
{
    size_t size = taskParam.dealRowCount * FP32_ONE_BLOCK_SIZE;
    size_t offset =
        taskParam.bIdx * taskParam.qHeadNum * taskParam.splitKVNum * FP32_ONE_BLOCK_SIZE +
        taskParam.n2Idx * taskParam.splitKVNum * taskParam.gSize * FP32_ONE_BLOCK_SIZE +
        taskParam.gIdx * PROFILE.G * FP32_ONE_BLOCK_SIZE +
        taskParam.flashDecodeS2Idx * taskParam.gSize *
            FP32_ONE_BLOCK_SIZE; // logSumExp在gm上的shape(B, N1, splitKVNum, FP32_ONE_BLOCK_SIZE)

    CopyFixedUbToGm(softmaxLseOutputQue, softmaxMaxGm[offset], softmaxMaxOutUb, size);
    CopyFixedUbToGm(softmaxLseOutputQue, softmaxSumGm[offset], softmaxSumOutUb, size);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::CopyFixedUbToGm(TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue, 
                                                            const GlobalTensor<T> &dst,
                                                            const LocalTensor<T>& src, 
                                                            size_t size)
{
    LocalTensor<T> tmp = softmaxLseOutputQue.AllocTensor<T>();
    DataCopy(tmp, src, size);

    softmaxLseOutputQue.EnQue(tmp);
    softmaxLseOutputQue.DeQue();
    DataCopy(dst, tmp, size);
    softmaxLseOutputQue.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::AttenMaskCopyIn(GlobalTensor<uint8_t>& attenMaskGm,
                                                            TQue<QuePosition::VECIN, 1>& maskInputQue,
                                                            const Vec1TaskParam& taskParam)
{
    LocalTensor<uint8_t> maskUb = maskInputQue.AllocTensor<uint8_t>(); // david datacopy暂不支持bool，改为uint8_t
    maskUb.SetSize(taskParam.columnCount);
    DataCopy(maskUb, attenMaskGm[taskParam.attenMaskOffset], taskParam.columnCount);
    maskInputQue.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::AttenMaskCopyInPFA(GlobalTensor<uint8_t>& attenMaskGm,
                                                              TQue<QuePosition::VECIN, 1>& maskInputQue,
                                                              const Vec1TaskParam& taskParam)
{
    LocalTensor<uint8_t> maskUb = maskInputQue.AllocTensor<uint8_t>();
    if (taskParam.sparseMode == 4) { // 4:sparse band模式
        maskUb.SetSize(taskParam.columnCount * taskParam.dealRowCount * 2); // 2: band模式需要两块mask ub
    } else {
        maskUb.SetSize(taskParam.columnCount * taskParam.dealRowCount);
    }

    size_t lenOfType = sizeof(uint8_t); // 每个数据的长度
    DataCopyExtParams intriParams;
    intriParams.blockCount = taskParam.dealRowCount;
    intriParams.blockLen = taskParam.columnCount * lenOfType;
    intriParams.srcStride = (static_cast<int64_t>(taskParam.attenMaskKVSize) - taskParam.columnCount) * lenOfType;
    if (taskParam.actualColumnCount < taskParam.columnCount) {
        intriParams.blockLen = taskParam.actualColumnCount * lenOfType;
        intriParams.srcStride = (static_cast<int64_t>(taskParam.attenMaskKVSize) - taskParam.actualColumnCount) * lenOfType;
    }
    intriParams.dstStride = 0;

    DataCopyPadExtParams<uint8_t> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 1;
    padParams.rightPadding = 0; // 伪量化场景仅支持bool/int8/uint8, KV方向尾块按mask类型对齐后长度不变
    DataCopyPad(maskUb, attenMaskGm[taskParam.attenMaskOffset], intriParams, padParams);
    if (taskParam.sparseMode == 4) { // 4:sparse band模式
        DataCopyPad(maskUb[taskParam.columnCount * taskParam.dealRowCount],
                    attenMaskGm[taskParam.attenMaskOffsetPre], intriParams, padParams);
    }
    maskInputQue.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::PseShiftCopyIn(GlobalTensor<PSE_SHIFT_T>& pseShiftGm,
                                                           TQue<QuePosition::VECIN, 1>& pseInputQue,
                                                           const Vec1TaskParam& taskParam)
{
    LocalTensor<PSE_SHIFT_T> pseShiftUb = pseInputQue.AllocTensor<PSE_SHIFT_T>();
    uint32_t pseShiftSizeAlign = ALIGN(taskParam.actualColumnCount, 16U);
    pseShiftUb.SetSize(taskParam.dealRowCount * pseShiftSizeAlign);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<PSE_SHIFT_T> copyInPadParams;
    copyInParams.blockLen = taskParam.actualColumnCount * sizeof(PSE_SHIFT_T);
    copyInParams.blockCount = taskParam.dealRowCount;
    copyInParams.srcStride = (taskParam.pseShiftS - taskParam.actualColumnCount) * sizeof(PSE_SHIFT_T);
    copyInParams.dstStride = (PROFILE.S - taskParam.actualColumnCount) * sizeof(PSE_SHIFT_T) / BYTE_BLOCK;

    copyInPadParams.isPad = false;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = 0;
    copyInPadParams.paddingValue = 0;

    DataCopyPad(pseShiftUb, pseShiftGm[taskParam.pseShiftOffset + taskParam.gIdx * PROFILE.G * taskParam.pseShiftS],
             copyInParams, copyInPadParams);
    pseInputQue.EnQue(pseShiftUb);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::Vec1CopyIn(GlobalTensor<uint8_t>& attenMaskGm,
                                                       TQue<QuePosition::VECIN, 1>& maskInputQue,
                                                       GlobalTensor<PSE_SHIFT_T>& pseShiftGm,
                                                       TQue<QuePosition::VECIN, 1>& pseInputQue,
                                                       const Vec1TaskParam& taskParam)
{
    if constexpr (ATTEN_MASK) {
        if (taskParam.qSeqSize > 1) {
            AttenMaskCopyInPFA(attenMaskGm, maskInputQue, taskParam);
        } else {
            AttenMaskCopyIn(attenMaskGm, maskInputQue, taskParam);
        }
    }
    if constexpr (PSE_SHIFT) {
        PseShiftCopyIn(pseShiftGm, pseInputQue, taskParam);
    }
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::Vec1Compute(
    LocalTensor<Q_T>& bmm2InResUb, LocalTensor<T>& softmaxMaxOutUb, LocalTensor<T>& softmaxSumOutUb,
    LocalTensor<T>& softmaxExpUb, LocalTensor<MM_OUT_T>& mm1ResUb, LocalTensor<T>& softmaxMaxInUb,
    LocalTensor<T>& softmaxSumInUb, LocalTensor<uint8_t>& softmaxTmpUb, TQue<QuePosition::VECIN, 1>& maskInputQue,
    TQue<QuePosition::VECIN, 1>& pseInputQue, const Vec1TaskParam& taskParam)
{
    LocalTensor<uint8_t> attenMaskUb;
    LocalTensor<PSE_SHIFT_T> pseUb;
    if constexpr (ATTEN_MASK) {
        attenMaskUb = maskInputQue.DeQue<uint8_t>();
    }
    if constexpr (PSE_SHIFT) {
        pseUb = pseInputQue.DeQue<PSE_SHIFT_T>();
    }
    SoftMaxShapeInfo srcShape = {taskParam.dealRowCount, taskParam.columnCount, taskParam.dealRowCount,
                                 taskParam.actualColumnCount};
    if constexpr (FLASH_DECODE) {
        SoftmaxFlashV2_VF<T, MM_IN_T, PSE_SHIFT_T, PSE_SHIFT, ATTEN_MASK, PROFILE.S, false, false>(
            bmm2InResUb, softmaxSumOutUb, softmaxMaxOutUb, mm1ResUb, softmaxExpUb,
            softmaxSumInUb, softmaxMaxInUb, softmaxTmpUb, taskParam.scaleValue, attenMaskUb,
            pseUb, PROFILE.G, srcShape);
    } else {
        if (taskParam.qSeqSize == 1) {
            SoftmaxFlashV2_VF<T, MM_IN_T, PSE_SHIFT_T, PSE_SHIFT, ATTEN_MASK, PROFILE.S, false, false>(
                bmm2InResUb, softmaxSumOutUb, softmaxMaxOutUb, mm1ResUb, softmaxExpUb,
                softmaxSumInUb, softmaxMaxInUb, softmaxTmpUb, taskParam.scaleValue, attenMaskUb,
                pseUb, PROFILE.G, srcShape);
        } else if (taskParam.sparseMode == 4) { // 4:sparse band模式
            SoftmaxFlashV2_VF<T, MM_IN_T, PSE_SHIFT_T, PSE_SHIFT, ATTEN_MASK, PROFILE.S, true, true>(
                bmm2InResUb, softmaxSumOutUb, softmaxMaxOutUb, mm1ResUb, softmaxExpUb,
                softmaxSumInUb, softmaxMaxInUb, softmaxTmpUb, taskParam.scaleValue, attenMaskUb,
                pseUb, PROFILE.G, srcShape);
        } else {
            SoftmaxFlashV2_VF<T, MM_IN_T, PSE_SHIFT_T, PSE_SHIFT, ATTEN_MASK, PROFILE.S, false, true>(
                bmm2InResUb, softmaxSumOutUb, softmaxMaxOutUb, mm1ResUb, softmaxExpUb,
                softmaxSumInUb, softmaxMaxInUb, softmaxTmpUb, taskParam.scaleValue, attenMaskUb,
                pseUb, PROFILE.G, srcShape);
        }
    }
    if constexpr (ATTEN_MASK) {
        maskInputQue.FreeTensor(attenMaskUb);
    }
    if constexpr (PSE_SHIFT) {
        pseInputQue.FreeTensor(pseUb);
    }
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::Vec1CopyResToL1(TSCM<TPosition::VECIN, 1>& mm2AScm,
                                                            LocalTensor<Q_T>& bmm2InResUb,
                                                            const Vec1TaskParam& taskParam)
{
    LocalTensor<Q_T> bmm2LeftScmTensor = mm2AScm.template AllocTensor<Q_T>();
    uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(Q_T);
    uint16_t lenBurst = taskParam.dealRowCount;
    uint16_t nBurst = PROFILE.S / elementTypeSize;  // 注意要整块G*PROFILE.S都拷到L1
    uint16_t srcStride =
        ALIGN((uint16_t)taskParam.dealRowCount, (uint16_t)PROFILE.G) - taskParam.dealRowCount + 1;  // 解写写ub的bank冲突
    uint16_t dstStride = ALIGN((uint16_t)taskParam.dealRowCount, (uint16_t)16) - taskParam.dealRowCount;
    DataCopyParams intriParamsTemp;
    intriParamsTemp.blockCount = nBurst;
    intriParamsTemp.blockLen = lenBurst;
    intriParamsTemp.dstStride = dstStride; 
    intriParamsTemp.srcStride = srcStride;
    DataCopy(bmm2LeftScmTensor, bmm2InResUb, intriParamsTemp);

    mm2AScm.template EnQue(bmm2LeftScmTensor);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::Softmax(
    TQue<QuePosition::VECOUT, 1>& vec1ResOutUb, TSCM<TPosition::VECIN, 1>& mm2AScm, LocalTensor<T>& softmaxMaxOutUb,
    LocalTensor<T>& softmaxSumOutUb, LocalTensor<T>& softmaxExpUb, LocalTensor<MM_OUT_T>& mm1ResUb,
    LocalTensor<T>& softmaxMaxInUb, LocalTensor<T>& softmaxSumInUb, LocalTensor<uint8_t>& softmaxTmpUb,
    GlobalTensor<uint8_t>& attenMaskGm, TQue<QuePosition::VECIN, 1>& maskInputQue,
    GlobalTensor<PSE_SHIFT_T>& pseShiftGm, TQue<QuePosition::VECIN, 1>& pseInputQue, const Vec1TaskParam& taskParam)
{
    if (taskParam.isFirstSInnerLoop) {
        Duplicate(softmaxMaxInUb, SOFTMAX_MIN_NUM, taskParam.dealRowCount * 32 / sizeof(T));
        Duplicate(softmaxSumInUb, FLOAT_ZERO, taskParam.dealRowCount * 32 / sizeof(T));
    }
    Vec1CopyIn(attenMaskGm, maskInputQue, pseShiftGm, pseInputQue, taskParam);
    LocalTensor<Q_T> bmm2InResUb = vec1ResOutUb.AllocTensor<Q_T>();
    Vec1Compute(bmm2InResUb, softmaxMaxOutUb, softmaxSumOutUb, softmaxExpUb, mm1ResUb, softmaxMaxInUb, softmaxSumInUb,
                softmaxTmpUb, maskInputQue, pseInputQue, taskParam);
    vec1ResOutUb.EnQue(bmm2InResUb);
    vec1ResOutUb.DeQue<Q_T>();
    Vec1CopyResToL1(mm2AScm, bmm2InResUb, taskParam);
    vec1ResOutUb.FreeTensor(bmm2InResUb);
}

template <typename IFAT>
__aicore__ inline void Vec1Processor<IFAT>::Process(
    TQue<QuePosition::VECOUT, 1>& vec1ResOutUb, TSCM<TPosition::VECIN, 1>& mm2AScm, LocalTensor<T>& softmaxMaxOutUb,
    LocalTensor<T>& softmaxSumOutUb, LocalTensor<T>& softmaxExpUb, LocalTensor<MM_OUT_T>& mm1ResUb,
    LocalTensor<T>& softmaxMaxInUb, LocalTensor<T>& softmaxSumInUb, LocalTensor<uint8_t>& softmaxTmpUb,
    GlobalTensor<uint8_t>& attenMaskGm, TQue<QuePosition::VECIN, 1>& maskInputQue,
    GlobalTensor<PSE_SHIFT_T>& pseShiftGm, TQue<QuePosition::VECIN, 1>& pseInputQue,
    TQue<QuePosition::VECOUT, 1>& softmaxLseOutputQue, GlobalTensor<T>& softmaxLseGm, const Vec1TaskParam& taskParam,
    GlobalTensor<T>& softmaxMaxGm, GlobalTensor<T>& softmaxSumGm)
{
    Softmax(vec1ResOutUb,mm2AScm, softmaxMaxOutUb, softmaxSumOutUb, softmaxExpUb, mm1ResUb, softmaxMaxInUb, softmaxSumInUb,
            softmaxTmpUb, attenMaskGm, maskInputQue, pseShiftGm, pseInputQue, taskParam);
    if constexpr (FLASH_DECODE) {
        if (unlikely(taskParam.isLastSInnerLoop == true)) {
            ComputeLogSumExpAndCopyToGm(softmaxLseOutputQue, softmaxMaxOutUb, softmaxSumOutUb, taskParam,
                                        softmaxMaxGm, softmaxSumGm);
        }
        return;
    }
    if (taskParam.softmaxLseFlag) {
        SoftmaxLseCopyOut(softmaxLseGm, softmaxLseOutputQue, softmaxMaxOutUb, softmaxSumOutUb, taskParam);
    }
}
#endif