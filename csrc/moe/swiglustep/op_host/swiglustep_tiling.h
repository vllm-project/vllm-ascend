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
 * \file swiglustep_tiling.h
 * \brief SwigluStep tiling data structure (shared by host and kernel)
 */
#ifndef SWIGLUSTEP_TILING_H
#define SWIGLUSTEP_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SwiglustepTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalLength);   // = M (rows of x[M,2N] / out[M,N])
    TILING_DATA_FIELD_DEF(int64_t, N);             // gate/up width (x last dim / 2)
    TILING_DATA_FIELD_DEF(int64_t, formerNum);      // number of former cores
    TILING_DATA_FIELD_DEF(int64_t, formerLength);   // former core rows
    TILING_DATA_FIELD_DEF(int64_t, tailNum);        // number of tail cores (=1)
    TILING_DATA_FIELD_DEF(int64_t, tailLength);     // tail core rows
    TILING_DATA_FIELD_DEF(int64_t, tileLength);     // UB tile rows (tileM)
    TILING_DATA_FIELD_DEF(float, limit);            // clamp limit (Step-3.7=7.0)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Swiglustep, SwiglustepTilingData)
}  // namespace optiling

#endif  // SWIGLUSTEP_TILING_H
