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
 * \file service_rope.h
 * \brief
 */

#ifndef SERVICE_ROPE_H
#define SERVICE_ROPE_H

#include "vf/vf_rope.h"

namespace MlaProlog {
/**
 * @brief RotaryPosEmb, 同时做row行的RotaryPosEmb，每一行的元素为col
 * @param outputLocal 输出tensor [row * col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row * col]
 * @param cosLocal cos系数tensor [(row - 1) * sinCosRepStride + col]
 * @param sinLocal sin系数tensor [(row - 1) * sinCosRepStride + col] - 1 应已在sin中
 * @param row 待处理的行数
 * @param col 待处理的列数  col <= 512 / sizeof(C)
 * @param sinCosRepStride 行与行之间sin/cos系数的偏移，单位为元素个数。
 */
template <typename C>
__aicore__ inline void RotaryPosEmb(const LocalTensor<C> &outputLocal, const LocalTensor<C> &inputLocal,
                                    const LocalTensor<C> &cosLocal, const LocalTensor<C> &sinLocal,
                                    uint64_t row, uint64_t col, uint8_t sinCosRepStride)
{
    DataSyncBarrier<MemDsbT::UB>();
    RotaryPosEmbVF<C>(outputLocal, inputLocal, cosLocal, sinLocal,
                      static_cast<uint32_t>(row), col, col, static_cast<uint64_t>(sinCosRepStride));
}
} // namespace MlaProlog
#endif