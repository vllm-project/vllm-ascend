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
 * \file load_index_kv_cache_tiling.h
 * \brief
 */

#ifndef LOAD_INDEX_KV_CACHE_TILING_H
#define LOAD_INDEX_KV_CACHE_TILING_H


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
BEGIN_TILING_DATA_DEF(LoadIndexKvCacheTilingData)
TILING_DATA_FIELD_DEF(int64_t, bn); // block_num
TILING_DATA_FIELD_DEF(int64_t, bs); // block_size
TILING_DATA_FIELD_DEF(int64_t, d); // head_dim
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);  // 头核共需要处理多少行
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);  // 尾核共需要处理多少行
TILING_DATA_FIELD_DEF(int64_t, blockStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LoadIndexKvCache, LoadIndexKvCacheTilingData)

// ----------算子CompileInfo定义----------
struct LoadIndexKvCacheCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class LoadIndexKvCacheTiling {
public:
    explicit LoadIndexKvCacheTiling(gert::TilingContext* tilingContext) : context_(tilingContext)
    {
    }
    ~LoadIndexKvCacheTiling() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CheckShapesAndAttrs();
    ge::graphStatus CalcOpTiling();
private:
    gert::TilingContext *context_ = nullptr;
    LoadIndexKvCacheTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bn_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t n_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t blockStride_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    int64_t tilingKey_ = 0;
};

}  // namespace optiling
#endif  // LOAD_INDEX_KV_CACHE_TILING_H
