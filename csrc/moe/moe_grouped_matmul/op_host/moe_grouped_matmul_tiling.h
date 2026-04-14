/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeGroupedMatmulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, group_num);
TILING_DATA_FIELD_DEF(uint32_t, core_num);
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, k);
TILING_DATA_FIELD_DEF(uint32_t, single_m);
TILING_DATA_FIELD_DEF(uint32_t, single_n);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm_tiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeGroupedMatmul, MoeGroupedMatmulTilingData)
} // namespace optiling
