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
 * \file service_gather_sin_cos.h
 * \brief
 */

#ifndef SERVICE_GATHER_SIN_COS_H
#define SERVICE_GATHER_SIN_COS_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

/**
 * @brief GatherSinCos 读取rope所需的sin和cos系数，并进行预处理，rope中sin的-1系数会在此处融合进sin中；
 * @param cosLocal cos搬运的位置;
 * @param sinLocal sin搬运的位置;rope中sin的-1系数会在此处融合进sin中；
 * @param cosGm cos在GM的位置，可以认为一个token有col个系数；整体连续排布
 * @param sinGm sin在GM的位置，同上
 * @param tokenIndex 从哪个token开始读取
 * @param curVecToken 读取多少个token的系数
 * @param shareTmpUb 临时buffer 内部需要的空间为 [2 * curVecToken * col * sizeof(bfloat16_t)]
 * @param row 行数；预留参数
 * @param col 列数
 */
template <typename T, typename O>
__aicore__ inline void GatherSinCos(LocalTensor<O> &cosLocal, LocalTensor<O> &sinLocal, const GlobalTensor<T> &cosGm,
                                    const GlobalTensor<T> &sinGm, int64_t tokenIndex, int64_t curVecToken,
                                    LocalTensor<uint8_t> &shareTmpUb, int64_t row, int64_t col)
{
    int64_t offset = tokenIndex * col;
    int64_t curDataSize = curVecToken * col;
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if constexpr (IsSameType<O, float>::value) {
        auto tmpUbBf16 = shareTmpUb.ReinterpretCast<bfloat16_t>();
        DataCopy(tmpUbBf16, cosGm[offset], curDataSize);
        DataCopy(tmpUbBf16[curDataSize], sinGm[offset], curDataSize);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(cosLocal, tmpUbBf16, RoundMode::CAST_NONE, curDataSize);
        Cast(sinLocal, tmpUbBf16[curDataSize], RoundMode::CAST_NONE, curDataSize);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        DataCopy(cosLocal, cosGm[offset], curDataSize);
        DataCopy(sinLocal, sinGm[offset], curDataSize);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    uint8_t blockNumPerRow = col / (ALIGN_BLOCK_SIZE / sizeof(O));
    Muls<O>(sinLocal, sinLocal, -1.0f, col >> 1, curVecToken, {1, 1, blockNumPerRow, blockNumPerRow});
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace MlaProlog

#endif