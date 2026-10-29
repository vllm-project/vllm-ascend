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
#include "./arch22/scatter_nd_update_large_index.h"
#include "./arch22/scatter_nd_update_hp.h"

using namespace ScatterNdUpdate;

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

template <typename VarDtype, typename IndicesT>
__aicore__ inline void RunHp(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                             const ScatterNdUpdateSkArch22TilingData& tilingData, AscendC::TPipe& tpipe, bool isView)
{
    if (isView) {
        ScatterNdUpdate::ScatterNdUpdateHpKernel<VarDtype, IndicesT, true> op(indices, updates, varRef, tilingData,
                                                                              tpipe);
        op.Process();
    } else {
        ScatterNdUpdate::ScatterNdUpdateHpKernel<VarDtype, IndicesT, false> op(indices, updates, varRef, tilingData,
                                                                               tpipe);
        op.Process();
    }
}

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
    // HP-only dispatch: split by indices, fuse LinearIndex + Scatter on each core, no SyncAll.
    // NOTE: writes to duplicate indices are non-deterministic across cores.
    // tilingKey 30 (int64 large index) is not representable as int32 linearIndex,
    // so it still falls back to the deterministic LargeIndex kernel.
    bool isView = tilingData.viewTiling.isViewStride0 != 0;
    if (TILING_KEY_IS(11) || TILING_KEY_IS(10)) {
        RunHp<DTYPE_VAR, int>(indices, updates, varRef, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(21) || TILING_KEY_IS(20)) {
        RunHp<DTYPE_VAR, int64_t>(indices, updates, varRef, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(30)) {
        RunLargeIndex<DTYPE_VAR>(indices, updates, varRef, tilingData, tpipe, isView);
    }
#endif
}
