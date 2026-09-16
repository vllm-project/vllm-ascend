/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_local_buffer.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla 共用的 Cube L1/L0 local buffer 初始化实现。
 */

#ifndef SMLA_CUBE_LOCAL_BUFFER_H
#define SMLA_CUBE_LOCAL_BUFFER_H

#include <stdint.h>
#include "static_buffer.h"
#include "smla_common_defs.h"

namespace AttentionCommon {

template <typename Q_T, typename T>
__aicore__ inline void InitCubeLocalBuffer(
    fa_base_matmul::StaticBuffer<Q_T> (&l1QBufs)[3], fa_base_matmul::StaticBuffer<Q_T> (&l1RightBufs)[3],
    fa_base_matmul::StaticBuffer<Q_T> (&l0ABufs)[2], fa_base_matmul::RingBuffer<Q_T> &l0A,
    fa_base_matmul::StaticBuffer<Q_T> (&l0BBufs)[2], fa_base_matmul::RingBuffer<Q_T> &l0B,
    fa_base_matmul::StaticBuffer<T> (&l0CBufs)[2], fa_base_matmul::RingBuffer<T> &l0C, uint32_t l1BaseAddr)
{
    if ASCEND_IS_AIC {
        uint32_t l1Addr = l1BaseAddr;

        l1QBufs[0] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 0};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
        l1QBufs[1] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 1};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
        l1QBufs[2] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 2};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);

        l1RightBufs[0] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 0};
        l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);
        l1RightBufs[1] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 1};
        l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);
        l1RightBufs[2] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 2};
        l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);

        uint32_t l0aAddr = 0;
        l0ABufs[0] = {LocalTensor<Q_T>(TPosition::A2, l0aAddr, L0A_ELEM_PER_BUF), 0};
        l0aAddr += L0A_ELEM_PER_BUF * sizeof(Q_T);
        l0ABufs[1] = {LocalTensor<Q_T>(TPosition::A2, l0aAddr, L0A_ELEM_PER_BUF), 1};

        uint32_t l0bAddr = 0;
        l0BBufs[0] = {LocalTensor<Q_T>(TPosition::B2, l0bAddr, L0B_ELEM_PER_BUF), 0};
        l0bAddr += L0B_ELEM_PER_BUF * sizeof(Q_T);
        l0BBufs[1] = {LocalTensor<Q_T>(TPosition::B2, l0bAddr, L0B_ELEM_PER_BUF), 1};

        uint32_t l0cAddr = 0;
        l0CBufs[0] = {LocalTensor<T>(TPosition::CO1, l0cAddr, L0C_ELEM_PER_BUF), 0};
        l0cAddr += L0C_ELEM_PER_BUF * sizeof(T);
        l0CBufs[1] = {LocalTensor<T>(TPosition::CO1, l0cAddr, L0C_ELEM_PER_BUF), 1};

        l0A = fa_base_matmul::RingBuffer<Q_T>(l0ABufs, 2);
        l0B = fa_base_matmul::RingBuffer<Q_T>(l0BBufs, 2);
        l0C = fa_base_matmul::RingBuffer<T>(l0CBufs, 2);

        SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(0));
        SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(1));
        SetFlag<HardEvent::M_MTE1>(INNERCORE_L0AB(0));
        SetFlag<HardEvent::M_MTE1>(INNERCORE_L0AB(1));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(0));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(1));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(2));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(0));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(1));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(2));
    }
}

} // namespace AttentionCommon

#endif // SMLA_CUBE_LOCAL_BUFFER_H
