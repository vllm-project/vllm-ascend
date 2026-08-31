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

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "arch20/compat_310p.h"
#include "arch20/gemm/kernel/gdn_fwd_h_kernel.hpp"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
#include "arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "arch22/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif

#include "lib/matmul_intf.h"

using namespace Catlass;

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
// 310P keeps the arch20 kernel ABI: no TileShapes/kGated templates and no gk path.
// The host still passes the unified gk argument; arch20 ignores it.
extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens,
                                                         GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                         GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)gk;
    // 310P binary codegen cannot register non-default tiling keys here; host forces key 0.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData =
        reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

    using workspaceType = float;
    // dtype: 0 - fp16, 1 - bf16, 2 - fp32
#ifndef CATLASS_UNIFIED_CORE
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
    } else
#endif
    {
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
#else
namespace GDN {

template <typename InputT, typename GT, typename StateT, typename WorkspaceT, typename TileShapes, bool kGated>
__aicore__ inline void ChunkGatedDeltaRuleFwdHKernelImpl(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                                         GM_ADDR inital_state, GM_ADDR cu_seqlens,
                                                         GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                         GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<InputT, GT, StateT, WorkspaceT, TileShapes, kGated>;
    GDNFwdHKernel gdnFwdH;
    gdnFwdH.Init(k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    gdnFwdH.Process();
}

template <typename TileShapes>
__aicore__ inline void ChunkGatedDeltaRuleFwdHDispatch(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk,
                                                       GM_ADDR inital_state, GM_ADDR cu_seqlens,
                                                       GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
                                                       GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user)
{
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData =
        reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);
    using WorkspaceT = float;
    bool useGk = gdnFwdHTilingData->useGk;
    // dtype: 0 - fp16, 1 - bf16, 2 - fp32
    if (gdnFwdHTilingData->dataType == 1) {
        if (gdnFwdHTilingData->stateDataType == 2) {
            if (gdnFwdHTilingData->gDataType == 2) {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, float, float, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, float, float, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            } else {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, bfloat16_t, float, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, bfloat16_t, float, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            }
        } else {
            if (gdnFwdHTilingData->gDataType == 2) {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, float, bfloat16_t, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, float, bfloat16_t, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            } else {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, bfloat16_t, bfloat16_t, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<bfloat16_t, bfloat16_t, bfloat16_t, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            }
        }
    } else {
        if (gdnFwdHTilingData->stateDataType == 2) {
            if (gdnFwdHTilingData->gDataType == 2) {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, float, float, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, float, float, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            } else {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, half, float, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, half, float, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            }
        } else {
            if (gdnFwdHTilingData->gDataType == 2) {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, float, half, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, float, half, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            } else {
                if (useGk) {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, half, half, WorkspaceT, TileShapes, true>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                } else {
                    ChunkGatedDeltaRuleFwdHKernelImpl<half, half, half, WorkspaceT, TileShapes, false>(
                        k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
                }
            }
        }
    }
}

} // namespace GDN

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g,
                                                         GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                         GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleFwdHDispatch<Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::ChunkGatedDeltaRuleFwdHDispatch<Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
            k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, tiling, user);
    }
}
#endif
