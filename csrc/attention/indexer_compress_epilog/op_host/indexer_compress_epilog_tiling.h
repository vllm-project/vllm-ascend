/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_compress_epilog_tiling.h
 * \brief
 */

#ifndef INDEXER_COMPRESS_EPILOG_TILING_H
#define INDEXER_COMPRESS_EPILOG_TILING_H


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
BEGIN_TILING_DATA_DEF(IndexerCompressEpilogTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);  // 一行多少个scale
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);  // 头核共需要处理多少行
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);  // 尾核共需要处理多少行
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);   // 头核需要几次ub搬入
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);  // 尾核需要几次ub搬入
TILING_DATA_FIELD_DEF(int64_t, rowFactor);  // ub一次标准处理行数
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);  // 头核最后一次ub处理行数
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);  // 尾核最后一次ub处理行数
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexerCompressEpilog, IndexerCompressEpilogTilingData)

// ----------算子CompileInfo定义----------
struct IndexerCompressEpilogCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class IndexerCompressEpilogTiling {
public:
    explicit IndexerCompressEpilogTiling(gert::TilingContext* tilingContext) : context_(tilingContext)
    {
    }
    ~IndexerCompressEpilogTiling() = default;
    
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
private:
    gert::TilingContext *context_ = nullptr;
    IndexerCompressEpilogTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t scaleCol_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_= 0;
    int64_t quantMode_ = 1;
    bool roundScale_ = true;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    int64_t tilingKey_ = 0;
};

}  // namespace optiling
#endif  // INDEXER_COMPRESS_EPILOG_TILING_H
