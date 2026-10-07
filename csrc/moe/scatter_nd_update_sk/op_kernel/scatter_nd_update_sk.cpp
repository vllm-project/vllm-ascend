/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update.cpp
 * \brief scatter_nd_update arch22 kernel entry
 */
#include "./arch22/scatter_nd_update.h"
#include "./arch22/scatter_nd_update_linear_index.h"
#include "./arch22/scatter_nd_update_no_sort.h"
#include "./arch22/scatter_nd_update_large_index.h"

using namespace ScatterNdUpdate;

template <typename VarDtype, template <typename, bool> class ScatterKernelT>
__aicore__ inline void RunScatterAfterSync(GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace,
                                           const ScatterNdUpdateSkArch22TilingData& tilingData, AscendC::TPipe& tpipe,
                                           bool isView)
{
    AscendC::SyncAll();
    tpipe.Destroy();
    AscendC::TPipe pipe;
    if (isView) {
        ScatterKernelT<VarDtype, true> op2(updates, varRef, workspace, tilingData, pipe);
        op2.Process();
    } else {
        ScatterKernelT<VarDtype, false> op2(updates, varRef, workspace, tilingData, pipe);
        op2.Process();
    }
}

template <typename VarDtype, bool IsSort, typename IdxDtype, template <typename, bool> class ScatterKernelT>
__aicore__ inline void RunLinearIndexAndScatter(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace,
                                                const ScatterNdUpdateSkArch22TilingData& tilingData,
                                                AscendC::TPipe& tpipe, bool isView)
{
    ScatterNdUpdate::LinearIndexKernel<IsSort, IdxDtype> op1(indices, workspace, tilingData, tpipe);
    op1.Process();
    RunScatterAfterSync<VarDtype, ScatterKernelT>(updates, varRef, workspace, tilingData, tpipe, isView);
}

template <typename VarDtype>
__aicore__ inline void RunLargeIndex(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                                     const ScatterNdUpdateSkArch22TilingData& tilingData, AscendC::TPipe& tpipe,
                                     bool isView)
{
    if (isView) {
        ScatterNdUpdate::LargeIndexKernel<VarDtype, true> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    } else {
        ScatterNdUpdate::LargeIndexKernel<VarDtype, false> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    }
}

// NOTE: the build system passes --impl_mode=high_performance,optional (ascendc_bin_param_build.py),
// which may define HIGH_PERFORMANCE=1. The HP kernel is intentionally NOT compiled/used here:
// it skips SyncAll and is non-deterministic for duplicate indices (missed/polluted rows).
// Always take the deterministic path: LinearIndex + Sort + Scatter with SyncAll.
#ifdef HIGH_PERFORMANCE
#undef HIGH_PERFORMANCE
#endif

extern "C" __global__ __aicore__ void scatter_nd_update_sk(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
#if (defined(DTYPE_VAR))
    // tilingKey: indexType * 10 + sortFlag
    // indexType: 1=int32, 2=int64(cast), 3=int64(large); sortFlag: 0=unsorted, 1=sorted
    bool isView = tilingData.viewTiling.isViewStride0 != 0;
    if (TILING_KEY_IS(11)) {
        RunLinearIndexAndScatter<DTYPE_VAR, true, int, ScatterNdUpdate::ScatterNdUpdateKernel>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(10)) {
        RunLinearIndexAndScatter<DTYPE_VAR, false, int, ScatterNdUpdate::ScatterNdUpdateKernelNoSort>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(21)) {
        RunLinearIndexAndScatter<DTYPE_VAR, true, int64_t, ScatterNdUpdate::ScatterNdUpdateKernel>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(20)) {
        RunLinearIndexAndScatter<DTYPE_VAR, false, int64_t, ScatterNdUpdate::ScatterNdUpdateKernelNoSort>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(30)) {
        RunLargeIndex<DTYPE_VAR>(indices, updates, varRef, tilingData, tpipe, isView);
    }
#endif
}
