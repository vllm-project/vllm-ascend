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
 * \file basic_block_vf_mx.h
 * \brief
 */
#ifndef GMMSQ_BASIC_BLOCK_VF_MX_H
#define GMMSQ_BASIC_BLOCK_VF_MX_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::VECTOR_REG_WIDTH;

namespace GMMSQWeightQuant {

template <typename xType, typename wType>
struct MxA8W4NzParams {
    uint64_t loopKNum;
    uint64_t innerLoopNum;
    uint64_t loopKDstStride;
    uint64_t innerDstStride;
    uint64_t nRealSizeAlign;
    __ubuf__ wType *weightLowBitPhyAddr;
    __ubuf__ xType *weightHighBitPhyAddr;
};

static constexpr uint32_t E2M1_SHIFT_RIGHT_SIZE = 0x2;
static constexpr uint32_t SHIFT_LEFT_SIZE = 0x4;
static constexpr uint32_t E2M1_AND_MASK = 0x9C;

} // namespace GMMSQWeightQuant
#endif // GMMSQ_BASIC_BLOCK_VF_MX_H
