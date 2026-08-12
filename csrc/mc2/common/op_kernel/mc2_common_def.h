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
 * \file mc2_common_def.h
 * \brief
 */
#ifndef KERNEL_MC2_COMMON_DEF_H
#define KERNEL_MC2_COMMON_DEF_H

namespace AscendC {
enum CommAlgType {
    COMM_ALG_DEFAULT = 0,
    COMM_ALG_FULL_MESH = 1,
    COMM_ALG_DOUBLE_RING = 2,
    COMM_ALG_SWITCH_WING = 3,
    COMM_ALG_RESERVED
};

struct HcclCombinOpParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
} // namespace AscendC
#endif // KERNEL_MC2_COMMON_DEF_H
