/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_v310_tiling.h
 * \brief
 */
#ifndef FUSED_GDN_GATING_V310_TILING_H
#define FUSED_GDN_GATING_V310_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedGdnGatingV310TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, alignedLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    TILING_DATA_FIELD_DEF(uint32_t, numHeads);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);
    TILING_DATA_FIELD_DEF(float, beta);
    TILING_DATA_FIELD_DEF(float, inv_beta);
    TILING_DATA_FIELD_DEF(float, threshold);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedGdnGatingV310, FusedGdnGatingV310TilingData)
} // namespace optiling

#endif // FUSED_GDN_GATING_V310_TILING_H