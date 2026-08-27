/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file data_copy_transpose_tiling_def.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(CopyTransposeTiling)
TILING_DATA_FIELD_DEF(uint32_t, dstShapeB);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeS);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, dstShapeH);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeB);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeN);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeS);
TILING_DATA_FIELD_DEF(uint32_t, srcShapeHN);
TILING_DATA_FIELD_DEF(uint32_t, originalShapeNLen);
TILING_DATA_FIELD_DEF(uint32_t, shapeSHValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsValue);
TILING_DATA_FIELD_DEF(uint32_t, shapeNsnValue);
TILING_DATA_FIELD_DEF(uint32_t, invalidParamCopyTransposeTiling);
TILING_DATA_FIELD_DEF(uint32_t, shapeBHValue);
TILING_DATA_FIELD_DEF(uint32_t, paramsAlign);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CopyTransposeTilingOp, CopyTransposeTiling)

} // namespace optiling
