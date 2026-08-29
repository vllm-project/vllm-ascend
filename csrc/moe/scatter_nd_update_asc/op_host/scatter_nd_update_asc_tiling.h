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
 * \file scatter_nd_update_asc_tiling.h
 * \brief
 */
#ifndef SCATTER_ND_UPDATE_ASC_TILING_H_
#define SCATTER_ND_UPDATE_ASC_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterNdUpdateAscTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, bAlign);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, blockFactorTail);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
// 非连续首轴(view)适配，与 ops-nn a2(arch22) 方案一致
TILING_DATA_FIELD_DEF(uint64_t, isViewStride0);
TILING_DATA_FIELD_DEF(uint64_t, varStride0Elements);
TILING_DATA_FIELD_DEF(uint64_t, firstDimStrideRows);
// indexDim：indices 第二维（scatter_nd 的 K），支持 1/2；varDim1：var dim1，K=2 计算 linearRow 用
TILING_DATA_FIELD_DEF(int64_t, indexDim);
TILING_DATA_FIELD_DEF(int64_t, varDim1);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ScatterNdUpdateAsc, ScatterNdUpdateAscTilingData)

struct ScatterNdUpdateAscCompileInfo {
};

class ScatterNdUpdateAscTiling {
public:
    explicit ScatterNdUpdateAscTiling(gert::TilingContext* context) : context_(context)
    {}
    ~ScatterNdUpdateAscTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus GetInputShapeInfo();
    ge::graphStatus GetInputDtypeInfo();
    ge::graphStatus HandleViewStride();
    ge::graphStatus PostTiling();

private:
    ScatterNdUpdateAscTilingData tilingData_;

    // platform info
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    // shape attr info
    int64_t a_ = 0;
    int64_t b_ = 0;
    int64_t bAlign_ = 0;
    int64_t c_ = 0;
    // var 维度信息：rank 及各维大小（供 HandleViewStride 校验 trailing 维 stride 连续性）
    int64_t varDimNum_ = 0;
    int64_t varDims_[4] = {0};
    int64_t varDtypeSize_ = 0;
    int64_t indicesDtypeSize_ = 0;
    // 非连续首轴(view)相关，与 ops-nn a2(arch22) 方案一致
    uint64_t isViewStride0_ = 0;
    uint64_t varStride0Elements_ = 0;
    uint64_t firstDimStrideRows_ = 1;
    // indices 第二维（scatter_nd 的 K，1/2）与 var dim1（K=2 计算 linearRow）
    int64_t indexDim_ = 1;
    int64_t varDim1_ = 1;

    // split info
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blockNum_ = 0;
    int64_t ubFactor_ = 0;

    // kernel mode
    bool indicesCache_ = false;
    bool splitVar = false;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling
#endif // SCATTER_ND_UPDATE_ASC_TILING_H_