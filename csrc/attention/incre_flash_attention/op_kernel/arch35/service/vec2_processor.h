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
 * \file vec2_processor.h
 * \brief
 */

#ifndef VEC2_PROCESSOR_H
#define VEC2_PROCESSOR_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "../incre_flash_attention_pub.h"
#include "../matmul_modules/ifa_matmul_policy.h"
#include "../vector_api/vf_flashupdate.h"
#include "../vector_api/vf_row_invalid.h"

struct Vec2TaskParam {
    uint32_t startRow;
    uint32_t dealRowCount;
    uint32_t columnCount;
    uint32_t actualColumnCount;
    
    uint64_t attenOutOffset;
    uint32_t splitKVNum;
    uint32_t gSize;
    uint32_t flashDecodeS2Idx;
    uint32_t attenOutStride;
    uint32_t attenOutHeadNum;
    bool isPFABSH;
    bool isRowInvalid;
    bool isFirstSInnerLoop;
    bool isLastSInnerLoop;
};

template <typename IFAT>
class Vec2Processor
{
public:
    using T = float;
    using OUT_T = typename IFAT::outputType;
    using MM_OUT_T = T;
    static constexpr IFAProfile PROFILE = IFAT::profile; 
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr bool ATTEN_MASK = IFAT::attenMask;
    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);

    __aicore__ inline Vec2Processor(){};
    __aicore__ inline void Process(LocalTensor<MM_OUT_T>& bmm2ResUb, TQue<QuePosition::VECOUT, 1>& updateResOutputQue,
                              LocalTensor<T>& softmaxSumUb, LocalTensor<T>& softmaxExpUb, LocalTensor<T>& softmaxMaxUb,
                              GlobalTensor<OUT_T> attentionOutGm, GlobalTensor<T> accumOutGm,
                              const Vec2TaskParam& taskParam);
    __aicore__ inline void Bmm2Out(GlobalTensor<OUT_T> attentionOutGm, LocalTensor<MM_OUT_T>& bmm2PreResUb,
                                   TQue<QuePosition::VECOUT, 1>& updateResOutputQue, const Vec2TaskParam& taskParam);

private:
    __aicore__ inline void RowDivs(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   const Vec2TaskParam& taskParam);
    __aicore__ inline void RowInvalid(LocalTensor<MM_OUT_T>& bmm2PreResUb, LocalTensor<T>& softmaxMaxUb,
                                      const Vec2TaskParam& taskParam);
    __aicore__ inline void PostQuant();
    __aicore__ inline void Bmm2FDOut(GlobalTensor<T> accumOutGm, LocalTensor<MM_OUT_T>& bmm2PreResUb,
                                     TQue<QuePosition::VECOUT, 1>& updateResOutputQue, const Vec2TaskParam& taskParam);
    __aicore__ inline void Bmm2FDDataCopyOut(GlobalTensor<T> accumOutGm, LocalTensor<MM_OUT_T>& attenOutUb,
                                             const Vec2TaskParam& taskParam);

    __aicore__ inline void Bmm2DataCopyOut(GlobalTensor<OUT_T> attentionOutGm, LocalTensor<OUT_T>& attenOutUb,
                                           const Vec2TaskParam& taskParam);
    LocalTensor<MM_OUT_T> bmm2PreResUb;                             
};
template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::Process(LocalTensor<MM_OUT_T>& bmm2ResUb,
                                                    TQue<QuePosition::VECOUT, 1>& updateResOutputQue,
                                                    LocalTensor<T>& softmaxSumUb, LocalTensor<T>& softmaxExpUb,
                                                    LocalTensor<T>& softmaxMaxUb, GlobalTensor<OUT_T> attentionOutGm,
                                                    GlobalTensor<T> accumOutGm, const Vec2TaskParam& taskParam)
{
    if (taskParam.isFirstSInnerLoop) {
        bmm2PreResUb = updateResOutputQue.AllocTensor<T>();
        DataCopy(bmm2PreResUb, bmm2ResUb, taskParam.dealRowCount * PROFILE.D);
    } else {
        FlashUpdate<MM_OUT_T, MM_OUT_T>(bmm2PreResUb, bmm2ResUb, bmm2PreResUb, softmaxExpUb, taskParam.dealRowCount,
                                        taskParam.actualColumnCount, PROFILE.D);
    }

    if (taskParam.isLastSInnerLoop) {
        RowDivs(bmm2PreResUb, bmm2PreResUb, softmaxSumUb, taskParam);
        if constexpr (ATTEN_MASK) {
            RowInvalid(bmm2PreResUb, softmaxMaxUb, taskParam);
        }
        if constexpr (FLASH_DECODE) {
            Bmm2FDOut(accumOutGm, bmm2PreResUb, updateResOutputQue, taskParam);
        } else {
            Bmm2Out(attentionOutGm, bmm2PreResUb, updateResOutputQue, taskParam);
        }
    }
}

template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::RowDivs(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                                    const Vec2TaskParam& taskParam)
{
    uint32_t dtypeMask = REPEAT_ELEMENT_NUM;
    uint32_t dLoop = PROFILE.D / dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = PROFILE.D / BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = PROFILE.D / BLOCK_ELEMENT_NUM;

    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[i * dtypeMask], src0Ub[i * dtypeMask], src1Ub, dtypeMask, taskParam.dealRowCount, repeatParamsDiv);
    }
}

template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::RowInvalid(LocalTensor<MM_OUT_T>& bmm2PreResUb, 
    LocalTensor<T>& softmaxMaxUb, const Vec2TaskParam& taskParam)
{
    if (!taskParam.isRowInvalid) {
        return;
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isRowInvalidNeedUpdate = false;
    uint16_t reduceSize = 8; // 8: maxUb每8个位置储存相同的值
    for (uint32_t i = 0; i < taskParam.dealRowCount; i++) {
        float maxValue = softmaxMaxUb.GetValue(i * reduceSize);
        if (maxValue == SOFTMAX_MIN_NUM) {
            isRowInvalidNeedUpdate = true;
            break;
        }
    }
    if (isRowInvalidNeedUpdate) {
        RowInvalidUpdate<T>(bmm2PreResUb, softmaxMaxUb, taskParam.dealRowCount, taskParam.columnCount,
            taskParam.actualColumnCount, reduceSize);
    }
}

template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::Bmm2DataCopyOut(GlobalTensor<OUT_T> attentionOutGm,
                                                            LocalTensor<OUT_T>& attenOutUb,
                                                            const Vec2TaskParam& taskParam)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = taskParam.dealRowCount;
    dataCopyParams.blockLen = taskParam.actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (PROFILE.D - taskParam.actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    if (taskParam.isPFABSH) {
        dataCopyParams.dstStride = (taskParam.attenOutHeadNum - 1) * taskParam.actualColumnCount * sizeof(OUT_T);
    } else {
        dataCopyParams.dstStride = 0;
    }
    DataCopyPad(attentionOutGm[taskParam.attenOutOffset + taskParam.startRow * taskParam.attenOutStride], attenOutUb,
                dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::Bmm2FDDataCopyOut(GlobalTensor<T> accumOutGm,
                                                              LocalTensor<MM_OUT_T>& attenOutUb,
                                                              const Vec2TaskParam& taskParam)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = taskParam.dealRowCount;
    dataCopyParams.blockLen = taskParam.actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (PROFILE.D - taskParam.actualColumnCount) / (BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(accumOutGm[taskParam.attenOutOffset * taskParam.splitKVNum +
                           taskParam.flashDecodeS2Idx * taskParam.gSize * taskParam.actualColumnCount +
                           taskParam.startRow * taskParam.actualColumnCount],
                attenOutUb, dataCopyParams);
}
template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::Bmm2Out(GlobalTensor<OUT_T> attentionOutGm,
                                                    LocalTensor<MM_OUT_T>& bmm2PreResUb,
                                                    TQue<QuePosition::VECOUT, 1>& updateResOutputQue,
                                                    const Vec2TaskParam& taskParam)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor;
    if constexpr (!POST_QUANT) {
        tmpBmm2ResCastTensor = bmm2PreResUb.template ReinterpretCast<OUT_T>();
        Cast(tmpBmm2ResCastTensor, bmm2PreResUb, AscendC::RoundMode::CAST_ROUND,
             taskParam.dealRowCount * PROFILE.D);
    } else {
        PostQuant();
    }

    updateResOutputQue.EnQue(bmm2PreResUb);
    updateResOutputQue.DeQue<OUT_T>();
    Bmm2DataCopyOut(attentionOutGm, tmpBmm2ResCastTensor, taskParam);
    updateResOutputQue.FreeTensor(bmm2PreResUb);
}

template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::Bmm2FDOut(GlobalTensor<T> accumOutGm, LocalTensor<MM_OUT_T>& bmm2PreResUb,
                                                      TQue<QuePosition::VECOUT, 1>& updateResOutputQue,
                                                      const Vec2TaskParam& taskParam)
{
    updateResOutputQue.EnQue(bmm2PreResUb);
    updateResOutputQue.DeQue<T>();
    Bmm2FDDataCopyOut(accumOutGm, bmm2PreResUb, taskParam);
    updateResOutputQue.FreeTensor(bmm2PreResUb);
}

template <typename IFAT>
__aicore__ inline void Vec2Processor<IFAT>::PostQuant(){}
#endif