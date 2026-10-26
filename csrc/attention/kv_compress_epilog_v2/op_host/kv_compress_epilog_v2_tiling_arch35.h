/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_COMPRESS_EPILOG_V2_TILING_ARCH35_H
#define KV_COMPRESS_EPILOG_V2_TILING_ARCH35_H

#include <cstdint>
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr int64_t KCEV2_CACHE_INPUT_INDEX = 0;
constexpr int64_t KCEV2_X_INPUT_INDEX = 1;
constexpr int64_t KCEV2_SLOT_INPUT_INDEX = 2;
constexpr int64_t KCEV2_CACHE_OUTPUT_INDEX = 0;
constexpr int64_t KCEV2_GROUP_ATTR_INDEX = 0;
constexpr int64_t KCEV2_MODE_ATTR_INDEX = 1;
constexpr int64_t KCEV2_ROUND_ATTR_INDEX = 2;
constexpr int64_t KCEV2_X_SCALE_ATTR_INDEX = 3;
constexpr int64_t KCEV2_BLOCK_STRIDE_ATTR_INDEX = 4;
constexpr int64_t KCEV2_GROUP_SIZE_16 = 16;
constexpr int64_t KCEV2_GROUP_SIZE_32 = 32;
constexpr int64_t KCEV2_MODE_MXFP8 = 2;
constexpr int64_t KCEV2_CACHE_RANK_2D = 2;
constexpr int64_t KCEV2_CACHE_RANK_4D = 4;
constexpr int64_t KCEV2_LAYOUT_1 = 1;
constexpr int64_t KCEV2_LAYOUT_2 = 2;
constexpr int64_t KCEV2_CACHE_COL_DIM_INDEX = 3;
constexpr int64_t KCEV2_MODE_MXFP4 = 4;
constexpr int64_t KCEV2_WORKSPACE_SIZE = 32;

BEGIN_TILING_DATA_DEF(KvCompressEpilogV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, cacheRows);
TILING_DATA_FIELD_DEF(int64_t, cacheRowStride);
TILING_DATA_FIELD_DEF(int64_t, dataCol);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, concatCol);
TILING_DATA_FIELD_DEF(int64_t, kvCacheCol);
TILING_DATA_FIELD_DEF(int64_t, padCol);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(int64_t, perGroupSize);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
// Layout is derived from cache rank: 1 for [N, C], 2 for [B, S, 1, C].
TILING_DATA_FIELD_DEF(int64_t, layout);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, blockStride);
// Layout2 physical token bundle fields.  scalePerToken is retained for ABI
// compatibility and is no longer rounded up for layout2.
TILING_DATA_FIELD_DEF(int64_t, scalePerToken);
TILING_DATA_FIELD_DEF(int64_t, scaleBytes);
TILING_DATA_FIELD_DEF(int64_t, tokenStride);
TILING_DATA_FIELD_DEF(int64_t, cacheCol);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCompressEpilogV2, KvCompressEpilogV2TilingData)

struct KvCompressEpilogV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class KvCompressEpilogV2Tiling {
public:
    explicit KvCompressEpilogV2Tiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAndDtypeInfo();
    ge::graphStatus GetAttributes();
    ge::graphStatus ValidateAndCalculateLayout();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();
    void DumpTilingInfo();

    gert::TilingContext *context_ = nullptr;
    KvCompressEpilogV2TilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    int64_t bs_ = 0;
    int64_t d_ = 0;
    int64_t cacheRows_ = 0;
    int64_t cacheRowStride_ = 0;
    int64_t cacheRank_ = 0;
    int64_t blockSize_ = 1;
    int64_t blockStride_ = 0;
    int64_t layout_ = 1;
    int64_t dataCol_ = 0;
    int64_t scaleCol_ = 0;
    int64_t concatCol_ = 0;
    int64_t kvCacheCol_ = 0;
    int64_t padCol_ = 0;
    int64_t scalePerToken_ = 0;
    int64_t scaleBytes_ = 0;
    int64_t tokenStride_ = 0;
    int64_t quantGroupSize_ = KCEV2_GROUP_SIZE_32;
    int64_t quantMode_ = KCEV2_MODE_MXFP8;
    int64_t roundScale_ = 1;
    float xScale_ = 1.0f;
    ge::DataType cacheDtype_ = ge::DT_UNDEFINED;
    ge::DataType outputDtype_ = ge::DT_UNDEFINED;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    ge::DataType slotDtype_ = ge::DT_UNDEFINED;
};

}  // namespace optiling

#endif
