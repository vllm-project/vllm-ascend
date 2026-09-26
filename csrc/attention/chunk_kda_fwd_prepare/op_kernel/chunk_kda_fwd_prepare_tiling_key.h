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

// SAFE_GATE 只为 SafeSigmoid 置位，避免生成数学等价的重复实例。
#define CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                      \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, EXP_VALUE,    \
    SAFE_VALUE, OUTPUT_VALUE)                                                \
    ASCENDC_TPL_ARGS_SEL(                                                     \
        ASCENDC_TPL_DTYPE_SEL(D_T_GATE, GATE_TYPE),                          \
        ASCENDC_TPL_DTYPE_SEL(D_T_BETA, BETA_TYPE),                          \
        ASCENDC_TPL_UINT_SEL(NORM_MODE, ASCENDC_TPL_UI_LIST, NORM_VALUE),    \
        ASCENDC_TPL_UINT_SEL(BETA_MODE, ASCENDC_TPL_UI_LIST, BETA_VALUE),    \
        ASCENDC_TPL_UINT_SEL(GATE_MODE, ASCENDC_TPL_UI_LIST, GATE_VALUE),    \
        ASCENDC_TPL_BOOL_SEL(USE_EXP2, EXP_VALUE),                           \
        ASCENDC_TPL_BOOL_SEL(SAFE_GATE, SAFE_VALUE),                         \
        ASCENDC_TPL_UINT_SEL(OUTPUT_MODE, ASCENDC_TPL_UI_LIST, OUTPUT_VALUE))

#define CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT(                                   \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, EXP_VALUE,    \
    SAFE_VALUE)                                                              \
    CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE,           \
        EXP_VALUE, SAFE_VALUE, CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE),          \
    CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE,           \
        EXP_VALUE, SAFE_VALUE, CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE),     \
    CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE,           \
        EXP_VALUE, SAFE_VALUE, CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE),          \
    CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE,           \
        EXP_VALUE, SAFE_VALUE, CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD)

#define CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                      \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, SAFE_VALUE)   \
    CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT(                                        \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, 0,        \
        SAFE_VALUE),                                                         \
    CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT(                                        \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, 1,        \
        SAFE_VALUE)

#define CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                     \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE)                            \
    CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE,                        \
        CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP, 0),                     \
    CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE,                        \
        CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS, 0),                             \
    CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE,                        \
        CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID, 1)

#define CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                                \
    GATE_TYPE, BETA_TYPE, NORM_VALUE)                                        \
    CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                          \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, CHUNK_KDA_FWD_PREPARE_BETA_RAW),  \
    CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                          \
        GATE_TYPE, BETA_TYPE, NORM_VALUE,                                    \
        CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID),                                 \
    CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                          \
        GATE_TYPE, BETA_TYPE, NORM_VALUE,                                    \
        CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID)

#define CHUNK_KDA_FWD_PREPARE_SEL_NORM(GATE_TYPE, BETA_TYPE)                \
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                                     \
        GATE_TYPE, BETA_TYPE, CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY),         \
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                                     \
        GATE_TYPE, BETA_TYPE, CHUNK_KDA_FWD_PREPARE_NORM_L2)

#define CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(GATE_TYPE)                      \
    CHUNK_KDA_FWD_PREPARE_SEL_NORM(                                          \
        GATE_TYPE, CHUNK_KDA_FWD_PREPARE_TPL_BF16),                          \
    CHUNK_KDA_FWD_PREPARE_SEL_NORM(                                          \
        GATE_TYPE, CHUNK_KDA_FWD_PREPARE_TPL_FP32)

ASCENDC_TPL_SEL(
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(CHUNK_KDA_FWD_PREPARE_TPL_BF16),
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(CHUNK_KDA_FWD_PREPARE_TPL_FP32));

#undef CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE
#undef CHUNK_KDA_FWD_PREPARE_SEL_NORM
#undef CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE
#undef CHUNK_KDA_FWD_PREPARE_SEL_GATE
#undef CHUNK_KDA_FWD_PREPARE_SEL_EXP
#undef CHUNK_KDA_FWD_PREPARE_SEL_OUTPUT
#undef CHUNK_KDA_FWD_PREPARE_SEL_ONE

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
