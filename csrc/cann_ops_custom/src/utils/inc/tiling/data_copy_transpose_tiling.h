/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file data_copy_transpose_tiling.h
 * \brief
 */

#pragma once

#include <vector>
#include <graph/tensor.h>
#include "data_copy_transpose_tiling_def.h"

namespace optiling {

inline void GetDataCopyTransposeTiling(const ge::Shape &dstShape, const ge::Shape &srcShape, const uint32_t typeSize,
                                       optiling::CopyTransposeTiling &tiling)
{
    std::vector<int64_t> dstShapeInfo = dstShape.GetDims();
    std::vector<int64_t> srcShapeInfo = srcShape.GetDims();

    tiling.set_dstShapeB(dstShapeInfo[0]);
    tiling.set_dstShapeN(dstShapeInfo[1]);
    tiling.set_dstShapeS(dstShapeInfo[2]);
    tiling.set_dstShapeH(dstShapeInfo[3]);
    tiling.set_dstShapeHN(tiling.get_dstShapeH() / tiling.get_dstShapeN());

    tiling.set_srcShapeB(srcShapeInfo[0]);
    tiling.set_srcShapeN(srcShapeInfo[1]);
    tiling.set_srcShapeS(srcShapeInfo[2]);
    tiling.set_srcShapeHN(srcShapeInfo[3]);
    tiling.set_originalShapeNLen(tiling.get_srcShapeHN() * typeSize);
    tiling.set_shapeSHValue(tiling.get_dstShapeS() * tiling.get_dstShapeH());
    tiling.set_shapeNsValue(tiling.get_dstShapeN() * tiling.get_dstShapeS());
    tiling.set_shapeNsnValue(tiling.get_dstShapeN() * tiling.get_srcShapeS() * tiling.get_srcShapeN());
    tiling.set_shapeBHValue(tiling.get_dstShapeB() * tiling.get_dstShapeH());
}

} // namespace optiling
