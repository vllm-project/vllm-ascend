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
 * \file service_dequant.h
 * \brief
 */

#ifndef SERVICE_DEQUANT_H
#define SERVICE_DEQUANT_H

#include "mla_prolog_comm.h"
#include "mla_prolog_vector_comm.h"

namespace MlaProlog {

/**
 * @brief DequantPerTokenQc 用于对Qc做反量化；按行做dequant流程，给oriRow * col的数据做反量化
 * @param outputGm 输出tensor
 * @param inputGm 输入tensor
 * @param deqScaleQcQrWGm deqScaleQcQr反量化系数，原shape[1,ND]
 * @param deQuantScaleQcQrLocal deQuantScaleQcQr反量化系数，原shape[BS,1]
 * @param shareTmpUb 临时buffer
 * @param dequantRowColStrideParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param oriRow 一共有多少行
*/
template <typename T, typename C, typename O>
__aicore__ inline void
DequantPerTokenQc(const GlobalTensor<O> &outputGm, const GlobalTensor<T> &inputGm,
                  const GlobalTensor<C> &deqScaleQcQrWGm, const LocalTensor<float> deQuantScaleQcQrLocal,
                  const LocalTensor<uint8_t> &shareTmpUb, Rectangle dequantRowColStrideParams, uint32_t oriRow)
{
    int64_t count = dequantRowColStrideParams.row * dequantRowColStrideParams.col;

    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();                         // count * sizeof(T)
    LocalTensor<C> scaleLocal = inputLocal[count + 16].template ReinterpretCast<C>();    // count * sizeof(C)
    LocalTensor<C> computeLocal = scaleLocal[count + 16].template ReinterpretCast<C>();  // count * sizeof(C)
    LocalTensor<O> outputLocal = computeLocal[count + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams copyParams{
        static_cast<uint16_t>(dequantRowColStrideParams.row),
        static_cast<uint16_t>(dequantRowColStrideParams.col * sizeof(T) / 32U),
        static_cast<uint16_t>((dequantRowColStrideParams.stride - dequantRowColStrideParams.col) * sizeof(T) / 32U), 0};

    Rectangle rectangleParams{
        static_cast<uint32_t>(1),     // row
        static_cast<uint32_t>(count), // col
        static_cast<uint32_t>(count)  // columnStride
    };

    for (int64_t rowOffset = 0; rowOffset < oriRow; rowOffset += dequantRowColStrideParams.row) {
        int64_t inputOffset = rowOffset * dequantRowColStrideParams.stride;
        int64_t outputOffset = rowOffset * dequantRowColStrideParams.col;
        // copy in
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], copyParams);
        DataCopy(scaleLocal, deqScaleQcQrWGm[inputOffset], copyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // compute
        Dequant(computeLocal, inputLocal, scaleLocal, deQuantScaleQcQrLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
        // cast
        Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, count);
    }
}

/**
 * @brief CastPerTokenQc 用于对Qc做类型转换；按行做cast流程，给oriRow * col的数据做类型转换
 * @param outputGm 输出tensor
 * @param inputGm 输入tensor
 * @param shareTmpUb 临时buffer
 * @param castRowColStrideParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param oriRow 一共有多少行
*/
template <typename T, typename O>
__aicore__ inline void CastPerTokenQc(const GlobalTensor<O> &outputGm, const GlobalTensor<T> &inputGm,
                                      const LocalTensor<uint8_t> &shareTmpUb, Rectangle castRowColStrideParams,
                                      uint32_t oriRow)
{
    int64_t count = castRowColStrideParams.row * castRowColStrideParams.col;

    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();                       // count * sizeof(T)
    LocalTensor<O> outputLocal = inputLocal[count + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams copyParams{
        static_cast<uint16_t>(castRowColStrideParams.row),
        static_cast<uint16_t>(castRowColStrideParams.col * sizeof(T) / 32U),
        static_cast<uint16_t>((castRowColStrideParams.stride - castRowColStrideParams.col) * sizeof(T) / 32U), 0};

    for (int64_t rowOffset = 0; rowOffset < oriRow; rowOffset += castRowColStrideParams.row) {
        int64_t inputOffset = rowOffset * castRowColStrideParams.stride;
        int64_t outputOffset = rowOffset * castRowColStrideParams.col;
        // copy in
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], copyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // cast
        Cast(outputLocal, inputLocal, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, count);
    }
}

/**
   DequantSplitNQc 用于enableGroupComputeOpt场景
   场景特征：半量化，BS = 1, headsize = 8
 * @brief 用于对Qc做反量化；按N方向切分的dequant,理解为做一个head的
 * @param outputGm 输出位置
 * @param inputGm 输入tensor
 * @param deqScaleQcQrWGm deqScaleQcQr反量化系数，原shape[1,ND]
 * @param deQuantScaleQcQrLocal deQuantScaleQcQr反量化系数，原shape[BS,1]
 * @param shareTmpUb 临时buffer
 * @param dequantRowColStrideParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param oriCol 一列真实的长度
 * @param dstStride 目的数据行间的偏移
 * @param subBlockIdx_ 这是奇数核还是偶数核
 */
// 用于enableGroupComputeOpt场景
template <typename T, typename C, typename O>
__aicore__ inline void
DequantSplitNQc(const GlobalTensor<O> &outputGm, const GlobalTensor<T> &inputGm, const GlobalTensor<C> &deqScaleQcQrWGm,
                const LocalTensor<C> &deQuantScaleQcQrLocal, const LocalTensor<uint8_t> &shareTmpUb,
                Rectangle dequantRowColStrideParams, uint32_t oriCol, uint32_t dstStride, uint32_t subBlockIdx_)
{
    int64_t count = dequantRowColStrideParams.col * 1;
    LocalTensor<T> inputLocal = shareTmpUb.ReinterpretCast<T>();                         // count * sizeof(T)
    LocalTensor<C> scaleLocal = inputLocal[count + 16].template ReinterpretCast<C>();    // count * sizeof(C)
    LocalTensor<C> computeLocal = scaleLocal[count + 16].template ReinterpretCast<C>();  // count * sizeof(C)
    LocalTensor<O> outputLocal = computeLocal[count + 16].template ReinterpretCast<O>(); // count * sizeof(O)

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(1), static_cast<uint16_t>(count * sizeof(T) / 32U),
        static_cast<uint16_t>((dequantRowColStrideParams.stride - dequantRowColStrideParams.col) * sizeof(T) / 32U), 0};

    DataCopyParams outputCopyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(count * sizeof(O) / 32U), 0,
                                    static_cast<uint16_t>(dstStride * sizeof(O) / 32U)};

    Rectangle rectangleParams{
        static_cast<uint32_t>(1),     // row
        static_cast<uint32_t>(count), // col
        static_cast<uint32_t>(count)  // columnStride
    };

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    int64_t scaleOffset = subBlockIdx_ * count;
    DataCopy(scaleLocal, deqScaleQcQrWGm[scaleOffset], dequantRowColStrideParams.col);
    for (int64_t rowOffset = 0; rowOffset < dequantRowColStrideParams.row; rowOffset++) {
        int64_t inputOffset = rowOffset * oriCol + subBlockIdx_ * count;
        int64_t outputOffset = rowOffset * oriCol + subBlockIdx_ * count;
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(inputLocal, inputGm[inputOffset], inputCopyParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // compute
        Dequant(computeLocal, inputLocal, scaleLocal, deQuantScaleQcQrLocal, rectangleParams);
        AscendC::PipeBarrier<PIPE_V>();
        // cast
        Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // copy out
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(outputGm[outputOffset], outputLocal, count);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

} // namespace MlaProlog
#endif
