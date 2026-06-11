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
 * \file decode_update_tiling.h
 * \brief
 */
 
#ifndef ASCEND_OPS_DECODE_UPDATE_TILING_H
#define ASCEND_OPS_DECODE_UPDATE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct DecodeUpdateCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
    bool is_ascendc = false;
};

BEGIN_TILING_DATA_DEF(DecodeUpdateTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, formerLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint32_t, hDim);
    TILING_DATA_FIELD_DEF(uint32_t, updateType);
    TILING_DATA_FIELD_DEF(uint32_t, sp);
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AttentionUpdateV2, DecodeUpdateTilingData);
} // namespace optiling

#endif