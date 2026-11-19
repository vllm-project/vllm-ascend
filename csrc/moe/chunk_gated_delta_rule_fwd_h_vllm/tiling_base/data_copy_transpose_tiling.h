/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
    constexpr int64_t B_INDEX = 0;
    constexpr int64_t N_INDEX = 1;
    constexpr int64_t S_INDEX = 2;
    constexpr int64_t H_INDEX = 3;
    std::vector<int64_t> dstShapeInfo = dstShape.GetDims();
    std::vector<int64_t> srcShapeInfo = srcShape.GetDims();

    tiling.set_dstShapeB(dstShapeInfo[B_INDEX]);
    tiling.set_dstShapeN(dstShapeInfo[N_INDEX]);
    tiling.set_dstShapeS(dstShapeInfo[S_INDEX]);
    tiling.set_dstShapeH(dstShapeInfo[H_INDEX]);
    tiling.set_dstShapeHN(tiling.get_dstShapeH() / tiling.get_dstShapeN());

    tiling.set_srcShapeB(srcShapeInfo[B_INDEX]);
    tiling.set_srcShapeN(srcShapeInfo[N_INDEX]);
    tiling.set_srcShapeS(srcShapeInfo[S_INDEX]);
    tiling.set_srcShapeHN(srcShapeInfo[H_INDEX]);
    tiling.set_originalShapeNLen(tiling.get_srcShapeHN() * typeSize);
    tiling.set_shapeSHValue(tiling.get_dstShapeS() * tiling.get_dstShapeH());
    tiling.set_shapeNsValue(tiling.get_dstShapeN() * tiling.get_dstShapeS());
    tiling.set_shapeNsnValue(tiling.get_dstShapeN() * tiling.get_srcShapeS() * tiling.get_srcShapeN());
    tiling.set_shapeBHValue(tiling.get_dstShapeB() * tiling.get_dstShapeH());
}

} // namespace optiling
