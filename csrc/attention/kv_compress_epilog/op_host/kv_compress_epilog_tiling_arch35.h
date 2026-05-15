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
 * \file kv_compress_epilog_tiling_arch35.h
 * \brief
 */

#ifndef KV_COMPRESS_EPILOG_TILING_ARCH35_H
#define KV_COMPRESS_EPILOG_TILING_ARCH35_H

#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {

// ---------- Constants ----------
constexpr int64_t X_INPUT_INDEX = 1;
constexpr int64_t SLOT_MAPPING_INDEX = 2;
constexpr int64_t KV_COMPRESS_CACHE_OUTPUT_INDEX = 0;

constexpr int64_t QUANT_GROUP_SIZE_ATTR_INDEX = 0;
constexpr int64_t QUANT_MODE_ATTR_INDEX = 1;
constexpr int64_t ROUND_SCALE_ATTR_INDEX = 2;

constexpr int64_t QUANT_MDOE_GROUP_FP8 = 1;
constexpr int64_t QUANT_MDOE_GROUP_MXFP8 = 2;

constexpr int64_t DEFAULT_QUANT_GROUP_SIZE = 128;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;

constexpr int64_t SLICE_SIZE = 64;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
// per_block量化,每128个f16需要量化出一个scale, 因此切分尾轴时，以128为factor进行切分
constexpr int64_t PER_BLOCK_FP16 = 128;

// ---------- TilingData Structure ----------
BEGIN_TILING_DATA_DEF(KvCompressEpilogTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, kvCacheCol);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, concatCol);
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
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCompressEpilog, KvCompressEpilogTilingData)

// ---------- CompileInfo Structure ----------
struct KvCompressEpilogCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ---------- Tiling Class ----------
class KvCompressEpilogTiling {
public:
    explicit KvCompressEpilogTiling(gert::TilingContext* context) : context_(context) {}
    ~KvCompressEpilogTiling() = default;

    ge::graphStatus RunTiling();

protected:
    // Main tiling workflow methods
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

    // Helper methods
    ge::graphStatus GetInputShapes();
    ge::graphStatus GetAttributes();
    ge::graphStatus GetDtypeInfo();
    void CountTilingKey();

    // Validation
    ge::graphStatus ValidateShapes();
    ge::graphStatus ValidateDtypes();

    // Utilities
    uint64_t GetTilingKey() const;
    void DumpTilingInfo();

private:
    // Context
    gert::TilingContext* context_ = nullptr;
    KvCompressEpilogTilingData tilingData_;
    uint64_t tilingKey_ = 0;

    // Platform info
    uint64_t coreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t ubSize_ = 0;

    // Shape info from inputs
    int64_t bs_ = 0;  // First dimension of x (to be partitioned)
    int64_t d_ = 0;  // Second dimension of x
    int64_t kvCacheCol_ = 0;  // kvCache cols
    int64_t scaleCol_ = 0;  // scale cols

    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_= 0;

    // Attributes
    int64_t quantGroupSize_ = DEFAULT_QUANT_GROUP_SIZE;
    int64_t quantMode_ = 1;
    int64_t roundScale_ = 1;

    // Data types
    ge::DataType xDtype_ = ge::DT_BF16;
    ge::DataType slotMappingDtype_ = ge::DT_INT32;
    ge::DataType kvCacheDtype_ = ge::DT_FLOAT8_E5M2;
};

}  // namespace optiling

#endif  // KV_COMPRESS_EPILOG_TILING_ARCH35_H
