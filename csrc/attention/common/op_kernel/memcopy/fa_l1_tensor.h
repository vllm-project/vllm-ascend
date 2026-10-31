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
 * \file fa_l1_tensor.h
 * \brief
 */
#ifndef FA_L1_TENSOR_H
#define FA_L1_TENSOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::LocalTensor;

enum class L1Format {
    NZ = 0
};

enum class ScaleTrans {
    NO_TRANS = 0,
    ND2NZ = 1,
    DN2NZ = 2
};

template <typename Q_T, L1Format FORMAT>
struct FaL1Tensor {
    LocalTensor<Q_T> tensor;
    uint32_t rowCount;
};

#endif