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
 * \file new_mc2_tiling_utils.h
 * \brief
 */

#ifndef __NEW_MC2_TILING_UTILS_H__
#define __NEW_MC2_TILING_UTILS_H__

#include <cstdint>
#include <map>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "formulaic_tiling_datatype.h"
#include "graph/utils/type_utils.h"
#include "matmul_formulaic_tiling.h"
#include "platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_type.h"
#include "../../../3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h"

namespace mc2tiling {
void NewUpdateMatmulV3Args(optiling::mc2_matmul_v3_advanced::Mc2MatMulV3Args &mmV3Args,
                           const mc2tiling::TilingArgs &args, const char *opName);

ge::graphStatus NewGetMatmulV3PriorityPolicy(const NpuArch npuArch, std::vector<int32_t> &priorities,
                                             const char *opName);
} // namespace mc2tiling

#endif
