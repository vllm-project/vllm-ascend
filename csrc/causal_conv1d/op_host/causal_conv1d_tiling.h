/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_data.h
 * \brief
 */

#ifndef ASCEND_OPS_CAUSAL_CONV1D_TILING_DATA_H
#define ASCEND_OPS_CAUSAL_CONV1D_TILING_DATA_H

#include <cstdint>

// #include "register/tilingdata_base.h"
// #include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(CausalConv1dTilingData)
    TILING_DATA_FIELD_DEF(int64_t, dim);
    TILING_DATA_FIELD_DEF(int64_t, cuSeqlen);
    TILING_DATA_FIELD_DEF(int64_t, seqLen);
    TILING_DATA_FIELD_DEF(int64_t, inputMode);

    TILING_DATA_FIELD_DEF(int64_t, width);

    TILING_DATA_FIELD_DEF(int64_t, stateLen);
    TILING_DATA_FIELD_DEF(int64_t, numCacheLines);

    TILING_DATA_FIELD_DEF(int64_t, batch);

    TILING_DATA_FIELD_DEF(int64_t, activationMode);
    TILING_DATA_FIELD_DEF(int64_t, padSlotId);

    TILING_DATA_FIELD_DEF(int64_t, hasBias);

    TILING_DATA_FIELD_DEF(int64_t, dimTileSize);
    TILING_DATA_FIELD_DEF(int64_t, blocksPerSeq);
END_TILING_DATA_DEF;
struct CausalConv1dCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};
REGISTER_TILING_DATA_CLASS(CausalConv1d, CausalConv1dTilingData)

} // namespace optiling

#endif // ASCEND_OPS_CAUSAL_CONV1D_TILING_DATA_H