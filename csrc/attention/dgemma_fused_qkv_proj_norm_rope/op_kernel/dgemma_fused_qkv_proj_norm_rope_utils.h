/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_utils.h
 *  \brief cross-core FFTS sync helpers (mirrors matmul_allreduce_add_rmsnorm_utils.h). */
#ifndef DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_UTILS_H
#define DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_UTILS_H

#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t FFTS_SYNC_INTERNEL_MODE = 0;
constexpr int32_t FFTS_SYNC_AICORE_GROUP_MODE = 2;
// Initial AIC<->AIV alignment flag (mirrors MC2 template). Cube waits on this at
// entry; vector fires it once before the split loop so the cube never runs ahead
// of a vector core that has not yet reached the consume loop.
constexpr int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 6;
constexpr int32_t QKV_PIPE_DEPTH = 2;

template <pipe_t pipe>
__aicore__ inline void FFTSCrossCoreSync(uint64_t mode, uint64_t flag_id)
{
    uint64_t config = 1 | (mode << 4) | (flag_id << 8);
    ffts_cross_core_sync(pipe, config);
}
#endif // DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_UTILS_H
