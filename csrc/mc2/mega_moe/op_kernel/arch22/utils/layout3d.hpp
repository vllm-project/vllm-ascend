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
 * \file layout3d.hpp
 * \brief
 */

#ifndef LAYOUT_3D_HPP
#define LAYOUT_3D_HPP
#include "kernel_operator.h"
#include "../template_linear_algebra_v2/catlass.hpp"
class Layout3D {
public:
    int64_t strides[2];

    CATLASS_DEVICE
    Layout3D()
    {
    }

    CATLASS_DEVICE
    Layout3D(int64_t stride0, int64_t stride1)
    {
        strides[0] = stride0;
        strides[1] = stride1;
    }

    CATLASS_DEVICE
    int64_t operator()(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * strides[0] + dim1 * strides[1] + dim2;
    }
};
#endif // LAYOUT_3D_HPP