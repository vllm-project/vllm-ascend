/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
#define __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__

#include "runtime/tiling_context.h"
#include "matmul_v3_common_advanced.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
class Mc2MatMulV3Tiling {
public:
    explicit Mc2MatMulV3Tiling(gert::TilingContext *context) : context_(context) {};
    virtual ~Mc2MatMulV3Tiling() = default;
    virtual ge::graphStatus DoTiling();

protected:
    virtual ge::graphStatus GetShapeAttrsInfo();

    virtual ge::graphStatus GetArgs();

    virtual ge::graphStatus CheckArgs();

protected:
    gert::TilingContext *context_ = nullptr;
    Mc2MatMulV3Args args_;
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
