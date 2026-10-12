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
 * \file msa_index_score.cpp
 * \brief MsaIndexScore kernel 入口。
 *
 * 入参顺序与 op_def / aclnn 对齐：
 *   query, key, block_table, scale, atten_mask, actual_seq_qlen, actual_seq_klen, start_loc, score
 * atten_mask 仅做 host 校验；device 侧按 sparse_mode 解析因果，不消费该 GM。
 */

#include "kernel_operator.h" // force-rebuild-arch22-v65-int8-4slot-dual-aiv
#include "lib/matmul_intf.h"
#include "msa_index_score_common.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/msa_index_score_kernel.h"
#else
#include "arch22/msa_index_score_kernel.h"
#endif

using namespace MsaIndexScoreNs;

#define MSA_INVOKE_KERNEL(ElementQ, IsQuant) \
    do { \
        MsaIndexScoreKernel<ElementQ, IsQuant> op; \
        op.Init(query, key, blockTable, scale, actualSeqQlen, actualSeqKlen, startLoc, score, userWs, tilingData); \
        op.Process(); \
    } while (0)

extern "C" __global__ __aicore__ void msa_index_score(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale,
                                                      GM_ADDR attenMask, GM_ADDR actualSeqQlen, GM_ADDR actualSeqKlen,
                                                      GM_ADDR startLoc, GM_ADDR score, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)attenMask; // host 校验；device 按 sparse_mode 解析 rightDownCausal
    // 950 非量化 PA：S 在 UB、无 K scratch，user workspace 可能为 0。不能因此跳过整个 kernel 流程。
    GM_ADDR userWs = AscendC::GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        userWs = workspace;
    }
    GET_TILING_DATA_WITH_STRUCT(MsaIndexScoreTilingData, tilingDataIn, tiling);
    const MsaIndexScoreTilingData *__restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(MSA_TILING_KEY_BF16)) {
        MSA_INVOKE_KERNEL(bfloat16_t, false);
    } else if (TILING_KEY_IS(MSA_TILING_KEY_FP16)) {
        MSA_INVOKE_KERNEL(half, false);
    } else if (TILING_KEY_IS(MSA_TILING_KEY_FP16_INT8)) {
        MSA_INVOKE_KERNEL(half, true);
#if (__CCE_AICORE__ == 310)
    } else if (TILING_KEY_IS(MSA_TILING_KEY_FP8_E4M3FN)) {
        MSA_INVOKE_KERNEL(fp8_e4m3fn_t, false);
    } else if (TILING_KEY_IS(MSA_TILING_KEY_FP8_E5M2)) {
        MSA_INVOKE_KERNEL(fp8_e5m2_t, false);
    } else if (TILING_KEY_IS(MSA_TILING_KEY_HIFLOAT8)) {
        MSA_INVOKE_KERNEL(hifloat8_t, false);
#endif
    }
}
