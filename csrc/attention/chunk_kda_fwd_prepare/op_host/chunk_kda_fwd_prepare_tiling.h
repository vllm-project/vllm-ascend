/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_TILING_H
#define CHUNK_KDA_FWD_PREPARE_TILING_H

#include <cstddef>
#include "chunk_kda_fwd_prepare_output_mask.h"
#include "register/tilingdata_base.h"

namespace optiling {

// 字段顺序和类型必须与设备侧读取保持一致。
BEGIN_TILING_DATA_DEF(ChunkKdaFwdPrepareTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, seqNum);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, qkHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, valueHeadNum);
TILING_DATA_FIELD_DEF(uint32_t, totalChunks);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, headsPerPartition);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, lowerBound);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, inputSequenceMajor);
TILING_DATA_FIELD_DEF(bool, hasDtBias);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdPrepare, ChunkKdaFwdPrepareTilingData)

enum ChunkKdaFwdPrepareInputIndex : size_t {
    PREPARE_INPUT_Q = 0,
    PREPARE_INPUT_K,
    PREPARE_INPUT_V,
    PREPARE_INPUT_G,
    PREPARE_INPUT_BETA,
    PREPARE_INPUT_A_LOG,
    PREPARE_INPUT_DT_BIAS,
    PREPARE_INPUT_CU_SEQLENS,
    PREPARE_INPUT_CHUNK_INDICES,
};

enum ChunkKdaFwdPrepareAttrIndex : size_t {
    PREPARE_ATTR_LAYOUT = 0,
    PREPARE_ATTR_SCALE,
    PREPARE_ATTR_CHUNK_SIZE,
    PREPARE_ATTR_EPSILON,
    PREPARE_ATTR_USE_QK_L2NORM,
    PREPARE_ATTR_USE_GATE,
    PREPARE_ATTR_USE_BETA_SIGMOID,
    PREPARE_ATTR_ALLOW_NEG_EIGVAL,
    PREPARE_ATTR_SAFE_GATE,
    PREPARE_ATTR_LOWER_BOUND,
    PREPARE_ATTR_USE_EXP2,
    PREPARE_ATTR_OUTPUT_MODE,
};

struct ChunkKdaFwdPrepareCompileInfo {};

} // namespace optiling

#endif // CHUNK_KDA_FWD_PREPARE_TILING_H
