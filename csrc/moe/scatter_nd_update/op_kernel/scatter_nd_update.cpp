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
 * \brief scatter_nd_update arch32 kernel entry
 */
#include "./arch32/scatter_nd_update.h"
#include "./arch32/scatter_nd_update_linear_index.h"
#include "./arch32/scatter_nd_update_no_sort.h"
#include "./arch32/scatter_nd_update_large_index.h"
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
#include "./arch32/scatter_nd_update_hp.h"
#endif

using namespace ScatterNdUpdate;

template <typename ValueT, template <typename, bool> class ScatterKernelT>
__aicore__ inline void RunScatterAfterSync(GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace,
                                           const ScatterNdUpdateArch32TilingData& tilingData, AscendC::TPipe& tpipe,
                                           bool isView)
{
    AscendC::SyncAll();
    tpipe.Destroy();
    AscendC::TPipe pipe;
    if (isView) {
        ScatterKernelT<ValueT, true> op2(updates, varRef, workspace, tilingData, pipe);
        op2.Process();
    } else {
        ScatterKernelT<ValueT, false> op2(updates, varRef, workspace, tilingData, pipe);
        op2.Process();
    }
}

template <typename ValueT, bool IsSort, typename IndexType, template <typename, bool> class ScatterKernelT>
__aicore__ inline void RunLinearIndexAndScatter(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace,
                                                const ScatterNdUpdateArch32TilingData& tilingData,
                                                AscendC::TPipe& tpipe, bool isView)
{
    ScatterNdUpdate::LinearIndexKernel<IsSort, IndexType> op1(indices, workspace, tilingData, tpipe);
    op1.Process();
    RunScatterAfterSync<ValueT, ScatterKernelT>(updates, varRef, workspace, tilingData, tpipe, isView);
}

template <typename ValueT>
__aicore__ inline void RunLargeIndex(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                                     const ScatterNdUpdateArch32TilingData& tilingData, AscendC::TPipe& tpipe,
                                     bool isView)
{
    if (isView) {
        ScatterNdUpdate::LargeIndexKernel<ValueT, true> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    } else {
        ScatterNdUpdate::LargeIndexKernel<ValueT, false> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    }
}

#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
template <typename ValueT, typename IndicesT>
__aicore__ inline void RunHp(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                             const ScatterNdUpdateArch32TilingData& tilingData, AscendC::TPipe& tpipe, bool isView)
{
    if (isView) {
        ScatterNdUpdate::ScatterNdUpdateHpKernel<ValueT, IndicesT, true> op(indices, updates, varRef, tilingData,
                                                                             tpipe);
        op.Process();
    } else {
        ScatterNdUpdate::ScatterNdUpdateHpKernel<ValueT, IndicesT, false> op(indices, updates, varRef, tilingData,
                                                                              tpipe);
        op.Process();
    }
}
#endif

extern "C" __global__ __aicore__ void scatter_nd_update(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
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
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    // HP path: split by indices, fuse LinearIndex + Scatter on each core, no SyncAll.
    // Trade-off: writes to duplicate indices are non-deterministic across cores.
    // tilingKey 30 (int64 large index) is not representable as int32 linearIndex,
    // so it still falls back to the deterministic LargeIndex kernel.
    if (TILING_KEY_IS(11) || TILING_KEY_IS(10)) {
        RunHp<DTYPE_VAR, int>(indices, updates, varRef, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(21) || TILING_KEY_IS(20)) {
        RunHp<DTYPE_VAR, int64_t>(indices, updates, varRef, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(30)) {
        RunLargeIndex<DTYPE_VAR>(indices, updates, varRef, tilingData, tpipe, isView);
    }
#else
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
#endif
}
