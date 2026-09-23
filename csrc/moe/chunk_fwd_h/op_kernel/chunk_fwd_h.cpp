/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_fwd_h_tiling_key.h"
#include "chunk_fwd_h_policy.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_fwd_h_cube.h"
#include "arch35/chunk_fwd_h_vec.h"
#else
#include "arch22/chunk_fwd_h_cube.h"
#include "arch22/chunk_fwd_h_vec.h"
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace GDN {

template <int D_T_G>
struct FwdHGateDTypeTraits;

template <>
struct FwdHGateDTypeTraits<CHUNK_FWD_H_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct FwdHGateDTypeTraits<CHUNK_FWD_H_TPL_FP32> {
    using type = float;
};

template <int D_T_G, int V_DIM, bool USE_GK, bool USE_EXP2, bool STATE_FP32,
          bool STATE_V_FIRST>
struct FwdHKernelTraits {
    static_assert(V_DIM == static_cast<int>(FWD_H_V), "ChunkFwdH only supports V=128.");
    using GateT = typename FwdHGateDTypeTraits<D_T_G>::type;
    using CompilePolicy = FwdHCompilePolicy<
        USE_GK ? FwdHGateMode::KEY_GK : FwdHGateMode::SCALAR_G, USE_EXP2, STATE_FP32>;
    static constexpr bool STATE_V_FIRST_VALUE = STATE_V_FIRST;
};

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline void RunFwdHTyped(const FwdHKernelArgs &args)
{
    if ASCEND_IS_AIC {
        const bool hasCubeWork = args.tiling.useInitialState || args.tiling.storeFinalState ||
            args.tiling.seqlen > static_cast<uint32_t>(FWD_H_CHUNK);
        if (!hasCubeWork) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ChunkFwdHCubeArch35<CompilePolicy, STATE_V_FIRST> cube;
#else
        AscendC::TPipe pipe;
        ChunkFwdHCubeArch22<CompilePolicy, STATE_V_FIRST> cube;
#endif
        cube.Init(args);
        cube.Process();
    }
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ChunkFwdHVecArch35<GateT, CompilePolicy, STATE_V_FIRST> vec;
        vec.Init(args);
#else
        AscendC::TPipe pipe;
        ChunkFwdHVecArch22<GateT, CompilePolicy, STATE_V_FIRST> vec;
        vec.Init(args);
#endif
        vec.Process();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <int D_T_G, int V_DIM, bool USE_GK, bool USE_EXP2, bool STATE_FP32,
          bool STATE_V_FIRST>
__global__ __aicore__ void chunk_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
    GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GDN::FwdHKernelArgs args{};
    args.k = k;
    args.w = w;
    args.u = u;
    args.g = g;
    args.gk = gk;
    args.initialState = initial_state;
    args.cuSeqlens = cu_seqlens;
    args.chunkIndices = chunk_indices;
    args.h = h;
    args.vNew = v_new;
    args.finalState = final_state;
    args.workspace = AscendC::GetUserWorkspace(workspace);
    __gm__ const ChunkFwdHTilingData *tilingData =
        reinterpret_cast<__gm__ const ChunkFwdHTilingData *>(tiling);
    args.tiling.batch = static_cast<uint32_t>(tilingData->batch);
    args.tiling.seqlen = static_cast<uint32_t>(tilingData->seqlen);
    args.tiling.kNumHead = static_cast<uint32_t>(tilingData->kNumHead);
    args.tiling.vNumHead = static_cast<uint32_t>(tilingData->vNumHead);
    args.tiling.kHeadDim = static_cast<uint32_t>(tilingData->kHeadDim);
    args.tiling.vHeadDim = static_cast<uint32_t>(tilingData->vHeadDim);
    args.tiling.chunkSize = static_cast<uint32_t>(tilingData->chunkSize);
    args.tiling.useInitialState = tilingData->useInitialState;
    args.tiling.storeFinalState = tilingData->storeFinalState;
    args.tiling.isVariedLen = tilingData->isVariedLen;
    args.tiling.shapeBatch = static_cast<uint32_t>(tilingData->shapeBatch);
    args.tiling.tokenBatch = static_cast<uint32_t>(tilingData->tokenBatch);
    args.tiling.vWorkspaceOffset = static_cast<uint64_t>(tilingData->vWorkspaceOffset);
    args.tiling.vUpdateWorkspaceOffset = static_cast<uint64_t>(tilingData->vUpdateWorkspaceOffset);
    args.tiling.kDecayWorkspaceOffset = static_cast<uint64_t>(tilingData->kDecayWorkspaceOffset);
    args.tiling.hWorkspaceOffset = static_cast<uint64_t>(tilingData->hWorkspaceOffset);
    using Traits = GDN::FwdHKernelTraits<
        D_T_G, V_DIM, USE_GK, USE_EXP2, STATE_FP32, STATE_V_FIRST>;
    GDN::RunFwdHTyped<typename Traits::GateT, typename Traits::CompilePolicy,
                      Traits::STATE_V_FIRST_VALUE>(args);
}
#endif
