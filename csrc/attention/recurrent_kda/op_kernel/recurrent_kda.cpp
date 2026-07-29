/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/recurrent_kda.h"
#else
#include "recurrent_kda.h"
#endif
#include "recurrent_kda_tiling_data.h"


using namespace AscendC;
using namespace matmul;
using namespace RecurrentKda;


extern "C" __global__ __aicore__ void
recurrent_kda(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR gate, GM_ADDR beta, GM_ADDR initialState,
              GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR numAcceptedTokens,
              GM_ADDR out, GM_ADDR finalState, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(RecurrentKdaTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    RKDA<bfloat16_t, bfloat16_t, DTYPE_STATE> op(&tilingData);
    RKDAInitParams initParams{query, key, value, gate, beta, initialState, cuSeqlens, ssmStateIndices,
                              aLog, dtBias, numAcceptedTokens, out, finalState};
    op.Init(initParams, &pipe);
    op.Process();
}
