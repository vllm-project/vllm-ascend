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
 * \file moe_gating_top_k_hash_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKHashTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rowCount);
TILING_DATA_FIELD_DEF(int64_t, perCoreRowCount);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRowCount);
TILING_DATA_FIELD_DEF(int64_t, expertCount);
TILING_DATA_FIELD_DEF(int64_t, addBias);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kGroup);
TILING_DATA_FIELD_DEF(int64_t, groupCount);
TILING_DATA_FIELD_DEF(int64_t, perGroupExpertCount);
TILING_DATA_FIELD_DEF(int64_t, perGroupExpertCountAlign);
TILING_DATA_FIELD_DEF(int64_t, groupSelectMode);
TILING_DATA_FIELD_DEF(int64_t, renorm);
TILING_DATA_FIELD_DEF(int64_t, normType);
TILING_DATA_FIELD_DEF(int64_t, outFlag);
TILING_DATA_FIELD_DEF(int64_t, hashFlag);
TILING_DATA_FIELD_DEF(int64_t, vmsCount);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(int64_t, calTmpBufUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopKHash, MoeGatingTopKHashTilingData)

BEGIN_TILING_DATA_DEF(MoeGatingTopKHashRegbaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, rowCount);
TILING_DATA_FIELD_DEF(int64_t, perCoreRowCount);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRowCount);
TILING_DATA_FIELD_DEF(int64_t, expertCount);
TILING_DATA_FIELD_DEF(int64_t, addBias);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kGroup);
TILING_DATA_FIELD_DEF(int64_t, groupCount);
TILING_DATA_FIELD_DEF(int64_t, perGroupExpertCount);
TILING_DATA_FIELD_DEF(int64_t, perGroupExpertCountAlign);
TILING_DATA_FIELD_DEF(int64_t, groupSelectMode);
TILING_DATA_FIELD_DEF(int64_t, renorm);
TILING_DATA_FIELD_DEF(int64_t, normType);
TILING_DATA_FIELD_DEF(int64_t, outFlag);
TILING_DATA_FIELD_DEF(int64_t, hashFlag);
TILING_DATA_FIELD_DEF(int64_t, vmsCount);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopKHash_10000, MoeGatingTopKHashRegbaseTilingData)
struct MoeGatingTopKHashCompileInfo {};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H
