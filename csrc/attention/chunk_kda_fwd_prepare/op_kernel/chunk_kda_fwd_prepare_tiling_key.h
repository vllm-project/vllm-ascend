/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
#define CHUNK_KDA_FWD_PREPARE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "chunk_kda_fwd_prepare_policy.h"

namespace KdaPrepare {

ASCENDC_TPL_ARGS_DECL(
    ChunkKdaFwdPrepare,
    ASCENDC_TPL_DTYPE_DECL(D_T_GATE, CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                           CHUNK_KDA_FWD_PREPARE_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(D_T_BETA, CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                           CHUNK_KDA_FWD_PREPARE_TPL_FP32),
    ASCENDC_TPL_UINT_DECL(NORM_MODE, 1, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
                          CHUNK_KDA_FWD_PREPARE_NORM_L2),
    ASCENDC_TPL_UINT_DECL(BETA_MODE, 2, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_BETA_RAW,
                          CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID,
                          CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID),
    ASCENDC_TPL_UINT_DECL(GATE_MODE, 2, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
                          CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS,
                          CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID),
    ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1),
    ASCENDC_TPL_BOOL_DECL(SAFE_GATE, 0, 1),
    ASCENDC_TPL_UINT_DECL(OUTPUT_MODE, 2, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE,
                          CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE,
                          CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE,
                          CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD));

// The A5 inference binding supplies raw BF16 gate, preprocessed FP32 beta,
// fused Q/K L2 norm and exp2. Legacy dtype/gate modes stay in ChunkKdaFwd.
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_GATE, CHUNK_KDA_FWD_PREPARE_TPL_BF16),
        ASCENDC_TPL_DTYPE_SEL(D_T_BETA, CHUNK_KDA_FWD_PREPARE_TPL_FP32),
        ASCENDC_TPL_UINT_SEL(NORM_MODE, ASCENDC_TPL_UI_LIST, CHUNK_KDA_FWD_PREPARE_NORM_L2),
        ASCENDC_TPL_UINT_SEL(BETA_MODE, ASCENDC_TPL_UI_LIST, CHUNK_KDA_FWD_PREPARE_BETA_RAW),
        ASCENDC_TPL_UINT_SEL(GATE_MODE, ASCENDC_TPL_UI_LIST, CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID),
        ASCENDC_TPL_BOOL_SEL(USE_EXP2, 1),
        ASCENDC_TPL_BOOL_SEL(SAFE_GATE, 1),
        ASCENDC_TPL_UINT_SEL(OUTPUT_MODE, ASCENDC_TPL_UI_LIST, CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD)));

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
