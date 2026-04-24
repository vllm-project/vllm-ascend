/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file hc_pre_inv_rms_tiling.h
 * \brief
 */
#ifndef HC_PRE_INV_RMS_TILING_H_
#define HC_PRE_INV_RMS_TILING_H_

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


BEGIN_TILING_DATA_DEF(HcPreInvRmsFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, A);  // A轴大小
TILING_DATA_FIELD_DEF(int64_t, R);  // R轴大小
TILING_DATA_FIELD_DEF(int64_t, blockNumA);  // 使用核数
TILING_DATA_FIELD_DEF(int64_t, blockFactorA);   // 每个核处理的A个数
TILING_DATA_FIELD_DEF(int64_t, blockTailFactorA);   // 尾核处理的A个数
TILING_DATA_FIELD_DEF(int64_t, ubFactorA);  // 每次UB循环处理的A个数
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HcPreInvRms, HcPreInvRmsFullLoadTilingData)

struct HcPreInvRmsCompileInfo {};

} // namespace optiling

#endif // HC_PRE_INV_RMS_TILING_H_
