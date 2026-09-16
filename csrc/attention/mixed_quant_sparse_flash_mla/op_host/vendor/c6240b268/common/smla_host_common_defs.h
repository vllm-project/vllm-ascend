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
 * \file smla_host_common_defs.h
 * \brief sparse_flash_mla / mixed_quant_sparse_flash_mla / quant_sparse_flash_mla 三个算子 op_host 侧共用的
 *        枚举、Tiling 入参信息 POD 与索引常量定义。
 */

#ifndef MQSMLA_C624_PRIVATE_SMLA_HOST_COMMON_DEFS_H
#define MQSMLA_C624_PRIVATE_SMLA_HOST_COMMON_DEFS_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include "register/tilingdata_base.h"

namespace optiling {

// ------------------公共定义--------------------------
struct MQSMLAC624TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct MQSMLAC624TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
    const gert::StorageShape *shape;
};

enum class MQSMLAC624Layout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

enum class MQSMLAC624Axis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3, // sparse_indices的K和key的D枚举值相同，表达相同位置, 最后一维
    T = 5,
    Bn = 6, // block number
    Bs = 7  // block size
};

enum class MQSMLAC624TemplateMode : uint32_t {
    SWA_TEMPLATE_MODE = 0,
    HCA_TEMPLATE_MODE = 1,
    CSA_TEMPLATE_MODE = 2,
    ORI_SPARSE_TEMPLATE_MODE = 3,
    ORI_CMP_SPARSE_TEMPLATE_MODE = 4
};

// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_IDX_FOUR = 4;

// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

// Outputs Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

} // namespace optiling

#endif // MQSMLA_C624_PRIVATE_SMLA_HOST_COMMON_DEFS_H
