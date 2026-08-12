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
 * \file matmul_base.h
 * \brief
 */

#ifndef MC2_MATMUL_BASE_H
#define MC2_MATMUL_BASE_H

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "../../../../../3rd/mat_mul_v3/op_kernel/mat_mul_base_kernel.h"
#else
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "../../../../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#else
#include "../../../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#include "../../../../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#endif
#endif

namespace MC2KernelTemplate {
// 基本输入输出和偏移
struct MC2MMBaseGmAddrs {
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR cGM;
    GM_ADDR biasGM;
    uint64_t aOffset;
    uint64_t bOffset;
    uint64_t cOffset;
};

// matmul计算节点的数据上下文
template <typename AdditionalGmAddrDataType, typename TilingDataType>
struct MC2MMContext {
    // 基本输入输出和偏移
    MC2MMBaseGmAddrs baseData;
    // 不同场景下额外的输入输出和偏移
    AdditionalGmAddrDataType additionalData;
    // mamtul的tiling
    TilingDataType *tilingDataPtr;
};
}; // namespace MC2KernelTemplate

#endif
