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
 * \file quant_batch_matmul_v4_compile_info.h
 * \brief
 */
#ifndef MC2_OP_HOST_QUANT_BATCH_MATMUL_V3_COMPILE_INFO_H__
#define MC2_OP_HOST_QUANT_BATCH_MATMUL_V3_COMPILE_INFO_H__
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

struct Mc2QuantBatchMatmulV3CompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    bool supportL0c2Out;
    bool supportL12BtBf16;
    platform_ascendc::SocVersion socVersion;
    std::string socVersionStr = "";
};

} // namespace optiling
#endif // __OP_HOST_QUANT_BATCH_MATMUL_V3_COMPILE_INFO_H__
