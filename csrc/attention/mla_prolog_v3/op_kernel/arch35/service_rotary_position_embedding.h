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
 * \file service_rotary_position_embedding.h
 * \brief
 */

#ifndef SERVICE_ROTARY_POSITION_EMBEDDING_H
#define SERVICE_ROTARY_POSITION_EMBEDDING_H

#include "mla_prolog_comm.h"
#include "mla_prolog_vector_comm.h"
#include "service_rope.h"

namespace MlaProlog {

template <typename T, typename C>
__aicore__ inline void PreprocessRopeInput(const GlobalTensor<T> &inputGm, LocalTensor<uint8_t> &shareTmpUb,
                                           const Rectangle &ropeParams, LocalTensor<C> &kFp32Local,
                                           LocalTensor<T> &kLocal, int64_t cnt)
{
    int64_t baseOffset;
    kLocal = shareTmpUb.ReinterpretCast<T>();

    if constexpr (std::is_same<T, int32_t>::value) {
        baseOffset = cnt;
    } else {
        baseOffset = cnt >> 1;
    }
    kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];

    DataCopyExtParams copyParams{static_cast<uint16_t>(ropeParams.row),
                                 static_cast<uint32_t>(ropeParams.col * sizeof(T)),
                                 static_cast<uint32_t>((ropeParams.stride - ropeParams.col) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(kFp32Local, inputGm, copyParams, padParams);
    } else {
        DataCopyPad(kLocal, inputGm, copyParams, padParams);
    }
}

/**
 * @brief RotaryPosEmbPerTensor 对一个tensor进行RotartPosEmb，tensor的维度为[row * col]
          行与行之间sin/cos公用；
          C：ropeComputType, float
 * @param outputLocal 输出tensor
 * @param inputGm 输入tensor
 * @param cosLocal cos系数
 * @param sinLocal sin系数
 * @param shareTmpUb 临时buffer，需要大小为 cnt * 5 * sizeof(float)
 * @param ropeParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param channelDeqScaleGm 量化参数；该tensor的每个元素不同
 * @param scale 量化参数；该tensor共用
 */
template <typename T, typename C, typename O, bool enableDequant = false, bool enableRope = true>
__aicore__ inline void RotaryPosEmbPerTensor(LocalTensor<O> &outputLocal, const GlobalTensor<T> &inputGm,
                                             const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal,
                                             LocalTensor<uint8_t> &shareTmpUb, Rectangle ropeParams,
                                             LocalTensor<float> channelDeqScaleLocal = LocalTensor<float>(),
                                             LocalTensor<float> scale = LocalTensor<float>())
{
    int64_t cnt = ropeParams.row * ropeParams.col;
    LocalTensor<C> kFp32Local;
    LocalTensor<T> kLocal;
    PreprocessRopeInput<T, C>(inputGm, shareTmpUb, ropeParams, kFp32Local, kLocal, cnt);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<C> kFp32OutputLocal = kFp32Local[cnt];

    if constexpr (std::is_same<T, int32_t>::value || enableDequant) { // 反量化
        Rectangle rectangleParams{
            static_cast<uint32_t>(1),   // row
            static_cast<uint32_t>(cnt), // col
            static_cast<uint32_t>(cnt)  // columnStride
        };
        if constexpr (std::is_same<T, float>::value) {
            Dequant(kFp32Local, kFp32Local, channelDeqScaleLocal, scale, rectangleParams);
        } else {
            Dequant(kFp32Local, kLocal, channelDeqScaleLocal, scale, rectangleParams);
        }
        PipeBarrier<PIPE_V>();
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }
    if constexpr (enableRope) {
        if constexpr (std::is_same<O, C>::value) {
            RotaryPosEmb<C>(outputLocal, kFp32Local, cosLocal, sinLocal, ropeParams.row, ropeParams.col, 0);
            PipeBarrier<PIPE_V>();
        } else {
            RotaryPosEmb<C>(kFp32OutputLocal, kFp32Local, cosLocal, sinLocal, ropeParams.row, ropeParams.col, 0);
            PipeBarrier<PIPE_V>();
            Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
            PipeBarrier<PIPE_V>();
        }
    } else {
        DataSyncBarrier<MemDsbT::UB>();
        if constexpr (std::is_same<O, C>::value) {
            DataCopy(outputLocal, kFp32Local, cnt);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(outputLocal, kFp32Local, RoundMode::CAST_RINT, cnt);
            PipeBarrier<PIPE_V>();
        }
    }
}

/**
 * @brief RotaryPosEmbPerHead 进行row行col列的RotaryPosEmb
          每行的量化系数，sin/cos均不同
          C：ropeComputType，float
 * @param outputLocal 输出tensor
 * @param inputGm 输入tensor
 * @param cosLocal cos系数
 * @param sinLocal sin系数
 * @param shareTmpUb 临时buffer
 * @param ropeParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param strideScale 一段的真实长度，描述channelDeqScaledGm数据排布
 * @param channelDeqScaleGm 量化参数：最终使用shape[1,col]
 * @param deQuantScale 量化参数；最终使用shape[row,8]
 */
template <typename T, typename C, typename O, bool enableDequant = false, bool enableRope = true>
__aicore__ inline void RotaryPosEmbPerHead(LocalTensor<O> &outputLocal, const GlobalTensor<T> &inputGm,
                                           const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal,
                                           LocalTensor<uint8_t> &shareTmpUb, Rectangle ropeParams, int64_t strideScale,
                                           GlobalTensor<float> channelDeqScaleGm = GlobalTensor<float>(),
                                           LocalTensor<float> deQuantScale = LocalTensor<float>())
{
    // 在 BS = 1 场景可能存在有row为零的情况，提前返回减少运算
    if (ropeParams.row == 0) {
        return;
    }

    int64_t cnt = ropeParams.row * ropeParams.col;
    LocalTensor<C> kFp32Local;
    LocalTensor<T> kLocal;
    PreprocessRopeInput<T, C>(inputGm, shareTmpUb, ropeParams, kFp32Local, kLocal, cnt);
    // scale参数可以和rope使用的空间复用
    LocalTensor<C> scaleLocal = kFp32Local[cnt];
    LocalTensor<C> kFp32OutputLocal = scaleLocal[ropeParams.col];
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if constexpr (std::is_same<T, int32_t>::value || enableDequant) { // 反量化
        DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(ropeParams.col * sizeof(C)),
                                      static_cast<uint32_t>((strideScale - ropeParams.col) * sizeof(C)), 0, 0};
        DataCopyPadExtParams<C> padParams1{false, 0, 0, 0};
        DataCopyPad(scaleLocal, channelDeqScaleGm, copyParams1, padParams1); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        uint8_t blockNumPerRow = ropeParams.col / (ALIGN_BLOCK_SIZE / sizeof(C));
        // row  col stride
        Rectangle rectangleParams{static_cast<uint32_t>(ropeParams.row), static_cast<uint32_t>(ropeParams.col),
                                  static_cast<uint32_t>(ropeParams.col)};
        if constexpr (std::is_same<T, float>::value) {
            Dequant(kFp32Local, kFp32Local, scaleLocal, deQuantScale, rectangleParams);
        } else {
            Dequant(kFp32Local, kLocal, scaleLocal, deQuantScale, rectangleParams);
        }
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (enableRope) {
        RotaryPosEmb<C>(kFp32OutputLocal, kFp32Local, cosLocal, sinLocal, ropeParams.row, ropeParams.col, ropeParams.col);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<O, C>::value) {
            DataCopy(outputLocal, kFp32OutputLocal, cnt);
        } else {
            Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        }
    } else {
        DataSyncBarrier<MemDsbT::UB>();
        if constexpr (std::is_same<O, C>::value) {
            DataCopy(outputLocal, kFp32Local, cnt);
        } else {
            Cast(outputLocal, kFp32Local, RoundMode::CAST_RINT, cnt);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, typename O>
__aicore__ inline void RopePostQuantPerChannel(LocalTensor<O> &outputLocal, LocalTensor<T> &inputLocal,
                                               LocalTensor<float> &quantScaleLocal, LocalTensor<uint8_t> &shareTmpUb,
                                               int64_t cnt)
{
    LocalTensor<float> inFp32;
    if constexpr (std::is_same<T, float>::value) {
        inFp32 = inputLocal;
    } else {
        inFp32 = shareTmpUb.ReinterpretCast<float>()[cnt];
        Cast(inFp32, inputLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }
    Mul(inFp32, inFp32, quantScaleLocal, cnt);
    PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outputLocal, inFp32, shareTmpUb, cnt);
    PipeBarrier<PIPE_V>();
}
} // namespace MlaProlog
#endif