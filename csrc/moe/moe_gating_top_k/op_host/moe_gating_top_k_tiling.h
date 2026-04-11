/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>


#include "../tiling_base/tiling_base.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"

#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "math_util.h"
//#include "util/extern_math_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGatingTopKTilingData)
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
TILING_DATA_FIELD_DEF(int64_t, vmsCount);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(int64_t, calTmpBufUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopK, MoeGatingTopKTilingData)

BEGIN_TILING_DATA_DEF(MoeGatingTopKRegbaseTilingData)
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
TILING_DATA_FIELD_DEF(int64_t, vmsCount);
TILING_DATA_FIELD_DEF(float, routedScalingFactor);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGatingTopK_10000, MoeGatingTopKRegbaseTilingData)
struct MoeGatingTopKCompileInfo {};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_GATING_TOP_K_H
