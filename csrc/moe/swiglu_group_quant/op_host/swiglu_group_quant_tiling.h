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
 * \file swiglu_blocl_quant_tiling.h
 * \brief
 */

#ifndef SWIGLU_BLOCK_QUANT_TILING_H
#define SWIGLU_BLOCK_QUANT_TILING_H


#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
// ----------公共定义----------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// ----------算子TilingData定义----------
BEGIN_TILING_DATA_DEF(SwigluGroupQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, splitD);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(int64_t, ue8m0Scale);
TILING_DATA_FIELD_DEF(int64_t, outputOrigin);
TILING_DATA_FIELD_DEF(float, clampValue);
TILING_DATA_FIELD_DEF(int64_t, hasClampValue);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, gLoop);
TILING_DATA_FIELD_DEF(int64_t, gFactor);
TILING_DATA_FIELD_DEF(int64_t, tailGFactor);
TILING_DATA_FIELD_DEF(int64_t, groupListType);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGroupQuant, SwigluGroupQuantTilingData)

// ----------算子CompileInfo定义----------
struct SwigluGroupQuantCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class SwigluGroupQuantTiling {
public:
    explicit SwigluGroupQuantTiling(gert::TilingContext* tilingContext) : context_(tilingContext)
    {
    }
    ~SwigluGroupQuantTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcMxQuantOpTiling();
    ge::graphStatus CalcGroupQuantOpTiling();
    ge::graphStatus CalcFp8QuantOpTiling();
    ge::graphStatus CalcGroupIndexTiling();
    void SetTilingData();
private:
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    SwigluGroupQuantTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t splitD_ = 0;
    int64_t scaleCol_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_= 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
    int64_t quantMode_ = 0;
    int64_t splitFactor_ = 0;
    int64_t roundScale_ = 0;
    int64_t ue8m0Scale_ = 0;
    double clampValue_ = 0.0;
    int64_t hasClampValue_ = 0;
    int64_t outputOrigin_ = 0;
    bool hasTopkWeight_ = false;
    int64_t g_ = 0;
    int64_t gLoop_ = 0;
    int64_t gFactor_ = 0;
    int64_t tailGFactor_ = 0;
    int64_t groupListType_ = 0;
    bool hasGroupIndex_ = false;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

}  // namespace optiling
#endif  // SWIGLU_CLIP_QUANT_TILING_H
