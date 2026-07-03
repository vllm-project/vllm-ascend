/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd.cpp
 * \brief
 */

#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif
#include "recompute_wu_fwd_struct.h"
#include "recompute_wu_fwd_common.h"
#include "recompute_wu_fwd_cube.h"
#include "recompute_wu_fwd_vector.h"

template <class... Dims>
using GemmCubeTileShape = tla::Shape<Dims...>;
using namespace tla;

namespace GDN {

template <typename kType, typename betaType>
struct RecomputeWUFwdTileShapes128 {
    using L1TileShape = GemmCubeTileShape<_128, _128, _256>;
    using L0TileShape = GemmCubeTileShape<_128, _128, _128>;
};

template <typename kType, typename betaType>
struct RecomputeWUFwdTileShapes256 {
    using L1TileShape = GemmCubeTileShape<_128, _256, _256>;
    using L0TileShape = GemmCubeTileShape<_128, _256, _64>;
};

template <typename kType, typename betaType, int VDim, typename TileShapes>
__aicore__ inline void RecomputeWUFwdKernelImplTyped(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR w, GM_ADDR u, GM_ADDR workspace, const RecomputeWUFwdTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        RecomputeWUFwdProcess<kType, betaType, typename TileShapes::L1TileShape, typename TileShapes::L0TileShape>
            recomputeWUFwdProcess(k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace);
        recomputeWUFwdProcess.Init(*tilingData);
        recomputeWUFwdProcess.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe tPipe;
        RecomputeWUFwdVectorProcess<kType, betaType> recomputeWUFwdVectorProcess(
            k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace);
        recomputeWUFwdVectorProcess.Init(*tilingData, &tPipe);
        recomputeWUFwdVectorProcess.Process();
    }
}

template <typename kType, typename betaType, int VDim>
__aicore__ inline void RecomputeWUFwdKernelImpl(
    GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR w, GM_ADDR u, GM_ADDR workspace, const RecomputeWUFwdTilingData *tilingData)
{
    if constexpr (VDim == 256) {
        RecomputeWUFwdKernelImplTyped<kType, betaType, VDim, RecomputeWUFwdTileShapes256<kType, betaType>>(
            k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace, tilingData);
    } else {
        RecomputeWUFwdKernelImplTyped<kType, betaType, VDim, RecomputeWUFwdTileShapes128<kType, betaType>>(
            k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, workspace, tilingData);
    }
}

} // namespace GDN

#ifndef TORCH_MODE
using namespace AscendC;

__global__ __aicore__ void recompute_wu_fwd(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR g, GM_ADDR gk,
                                            GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR w, GM_ADDR u,
                                            GM_ADDR workspace, GM_ADDR tiling)
{
    (void)gk;
    AscendC::AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(GDN::RecomputeWUFwdTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::RecomputeWUFwdTilingData, tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RecomputeWUFwdKernelImpl<DTYPE_K, DTYPE_BETA, 128>(
            k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, userWS, &tilingData);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GDN::RecomputeWUFwdKernelImpl<DTYPE_K, DTYPE_BETA, 256>(
            k, v, beta, A, g, cu_seqlens, chunk_indices, w, u, userWS, &tilingData);
    }
}
#endif
