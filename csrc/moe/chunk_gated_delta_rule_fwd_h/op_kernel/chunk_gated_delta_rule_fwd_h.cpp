/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief
 */

// #include "chunk_gated_delta_rule_fwd_h.h"
#include "gemm/kernel/gdn_fwd_h_kernel.hpp"
#include "lib/matmul_intf.h"

using namespace Catlass;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    using workspaceType = float;
    // dtype: 0 - fp16, 1 - bf16, 2 - fp32
    if (gdnFwdHTilingData->dataType == 1) {
        if (gdnFwdHTilingData->stateDataType == 2) {
            if (gdnFwdHTilingData->gDataType == 2) {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<bfloat16_t, float, float, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            } else {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<bfloat16_t, bfloat16_t, float, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            }
        } else {
            if (gdnFwdHTilingData->gDataType == 2) {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<bfloat16_t, float, bfloat16_t, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            } else {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<bfloat16_t, bfloat16_t, bfloat16_t, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            }
        }
    } else {
        if (gdnFwdHTilingData->stateDataType == 2) {
            if (gdnFwdHTilingData->gDataType == 2) {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<half, float, float, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            } else {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<half, half, float, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            }
        } else {
            if (gdnFwdHTilingData->gDataType == 2) {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<half, float, half, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            } else {
                using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<half, half, half, workspaceType>;
                GDNFwdHKernel gdnFwdH;
                gdnFwdH.Init(k, w, u, g, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                gdnFwdH.Process();
            }
        }
    }
}
