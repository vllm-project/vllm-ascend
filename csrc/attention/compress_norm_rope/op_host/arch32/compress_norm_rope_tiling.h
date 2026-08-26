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
 * \file compress_norm_rope_tiling.h
 * \brief CompressNormRope A2/A3 tiling 声明（两阶段重构版，扁平实现，保留 ASCENDC_TPL key 分发）
 */

#ifndef COMPRESS_NORM_ROPE_TILING_H
#define COMPRESS_NORM_ROPE_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/arch32/compress_norm_rope_template_tiling_key.h"
#include "../../op_kernel/arch32/compress_norm_rope_tiling_data.h"
#include "platform/platform_info.h"

#ifdef ASCENDC_OP_TEST
#define CMP_EXTERN_C extern "C"
#else
#define CMP_EXTERN_C
#endif

namespace optiling {

// INPUT（顺序对齐 def）
constexpr uint32_t MM_KV_INPUT_INDEX = 0;
constexpr uint32_t MM_SCORE_INPUT_INDEX = 1;
constexpr uint32_t STATE_CACHE_INPUT_INDEX = 2;
constexpr uint32_t APE_INPUT_INDEX = 3;
constexpr uint32_t NORM_WEIGHT_INPUT_INDEX = 4;
constexpr uint32_t ROPE_SIN_INPUT_INDEX = 5;
constexpr uint32_t ROPE_COS_INPUT_INDEX = 6;
// INPUT(OPTION)
constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 7;
constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 8;
constexpr uint32_t SEQ_USED_INPUT_INDEX = 9;
constexpr uint32_t START_POS_INPUT_INDEX = 10;

// ATTR
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 0;
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
constexpr uint32_t COFF_ATTR_INDEX = 2;
constexpr uint32_t NORM_EPS_ATTR_INDEX = 3;
constexpr uint32_t ROTARY_MODE_ATTR_INDEX = 4;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 5;
constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 6;
constexpr uint32_t MM_KV_STRIDE_DIM0_ATTR_INDEX = 7;
constexpr uint32_t MM_SCORE_STRIDE_DIM0_ATTR_INDEX = 8;

// OUTPUT
constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;

constexpr uint32_t COMPRESS_NORM_ROPE_DIM_NUM_1 = 1;
constexpr uint32_t COMPRESS_NORM_ROPE_DIM_NUM_2 = 2;
constexpr uint32_t COMPRESS_NORM_ROPE_DIM_NUM_3 = 3;
constexpr uint32_t COMPRESS_NORM_ROPE_DIM_INDEX_0 = 0;
constexpr uint32_t COMPRESS_NORM_ROPE_DIM_INDEX_1 = 1;
constexpr uint32_t COMPRESS_NORM_ROPE_DIM_INDEX_2 = 2;

// 生产实际调用组合（vllm-ascend RATIO_CONFIG）：c4={coff:2}, c128={coff:1}
constexpr uint32_t CMP_RATIO_C4 = 4;
constexpr uint32_t CMP_RATIO_C128 = 128;
constexpr uint32_t COFF_OVERLAP = 2;
constexpr uint32_t COFF_DISABLE = 1;
constexpr uint32_t D_CHUNK_SIZE_C128 = 64;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;  // SyncAll（c128 两阶段）需要 batch 调度

struct CompressNormRopeCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    const gert::Tensor *tensor;
};

enum class TemplateId : uint8_t {
    NORMAL = 0,
    EMPTY_X = 1
};

struct CompressNormRopeContext {
    const char *opName = nullptr;
    const char *opType = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;

    RequiredParaInfo mmKv;
    RequiredParaInfo mmScore;
    RequiredParaInfo stateCache;
    RequiredParaInfo ape;
    RequiredParaInfo normWeight;
    RequiredParaInfo ropeSin;
    RequiredParaInfo ropeCos;
    OptionalParaInfo stateBlockTable;
    OptionalParaInfo cuSeqlens;
    OptionalParaInfo seqUsed;
    OptionalParaInfo startPos;
    RequiredParaInfo cmpKv;

    const int *ropeHeadDim = nullptr;
    const int *coff = nullptr;
    const int *cmpRatio = nullptr;
    const float *normEps = nullptr;
    const int *rotaryMode = nullptr;
    const int *cacheMode = nullptr;
    const int *stateCacheStrideDim0 = nullptr;
    const int *mmKvStrideDim0 = nullptr;
    const int *mmScoreStrideDim0 = nullptr;
    TemplateId templateId = TemplateId::NORMAL;

    ge::DataType dtype = ge::DT_BF16;
    size_t *workSpaces = nullptr;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
};

CMP_EXTERN_C ge::graphStatus TilingCompressNormRope(gert::TilingContext *context);

} // namespace optiling

#endif // COMPRESS_NORM_ROPE_TILING_H
