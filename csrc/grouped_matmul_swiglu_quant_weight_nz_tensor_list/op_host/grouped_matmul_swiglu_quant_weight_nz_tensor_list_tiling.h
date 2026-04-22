/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_weight_nz_tensor_list_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H

#include <set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GMMSwigluBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, mLimit);
  TILING_DATA_FIELD_DEF(uint64_t, isPreFill);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluBaseParamsOp, GMMSwigluBaseParams)

BEGIN_TILING_DATA_DEF(GMMSwiglu)
  TILING_DATA_FIELD_DEF(uint32_t, maxProcessRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, groupListLen);
  TILING_DATA_FIELD_DEF(uint32_t, tokenLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluOp, GMMSwiglu)

BEGIN_TILING_DATA_DEF(GMMSwigluQuantTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluBaseParams, gmmSwigluBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(GMMSwiglu, gmmSwiglu);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantWeightNzTensorList, GMMSwigluQuantTilingData)
}

namespace GroupedMatmulSwigluQuantWeightNzTensorListTiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t GROUPLIST_INDEX = 4;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t USER_WORKSPACE_LIMIT = 256 * 1024 * 1024;
constexpr int64_t PREFILL_USER_WORKSPACE_LIMIT = 64 * 1024 * 1024;
constexpr int64_t DOUBLE_WORKSPACE_SPLIT = 2;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr uint32_t PREFILL_M_MIN_SIZE = 16 * 1024;

const std::set<std::array<int64_t, 2>> PREFILL_WHITE_LIST = {   // used for preFill case
    {{2048, 1536}},
    {{4096, 3072}}
};
} // namespace GroupedMatmulSwigluQuantWeightNzTensorListTiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H
