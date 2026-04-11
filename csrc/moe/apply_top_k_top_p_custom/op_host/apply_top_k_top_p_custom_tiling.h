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
 * \file apply_top_k_top_p_custom_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (28) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */
#ifndef __APPLY_TOP_K_TOP_P_CUSTOM_TILINGDATA_H__
#define __APPLY_TOP_K_TOP_P_CUSTOM_TILINGDATA_H__

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyTopKTopPCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailBatch);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, dataNumInit);
    TILING_DATA_FIELD_DEF(uint32_t, dataNumInitAligned);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElementAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tailUbFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, tailUbFactorElementAligned);
    TILING_DATA_FIELD_DEF(uint32_t, calUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, iterateTimes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyTopKTopPCustom, ApplyTopKTopPCustomTilingData)

struct TilingForApplyTopKTopPCustomCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

}  // namespace optiling
#endif  // __APPLY_TOP_K_TOP_P_CUSTOM_TILINGDATA_H__