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
 * \file scatter_nd_update_v2.cpp
 * \brief ScatterNdUpdateV2 算子入口
 */
#include "scatter_nd_update_v2.h"
#include "scatter_nd_update_linear_index.h"
#include "scatter_nd_update_no_sort.h"
#include "scatter_nd_update_large_index.h"

extern "C" __global__ __aicore__ void scatter_nd_update_v2(GM_ADDR varRef, GM_ADDR indices,
    GM_ADDR updates, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
#if (defined(DTYPE_VAR))
    // tilingKey: indexType * 10 + sortFlag
    // indexType: 1=int32, 2=int64(cast), 3=int64(large); sortFlag: 0=非排序, 1=排序
    if (TILING_KEY_IS(11)) {
        ScatterNdUpdateV2::LinearIndexKernel<true, int> op1(indices, workSpace, tilingData, tpipe);
        op1.Process();
        AscendC::SyncAll();
        tpipe.Destroy();
        AscendC::TPipe pipe;
        ScatterNdUpdateV2::ScatterNdUpdateV2Kernel<DTYPE_VAR> op2(updates, output, workSpace, tilingData, pipe);
        op2.Process();
    } else if (TILING_KEY_IS(10)) {
        ScatterNdUpdateV2::LinearIndexKernel<false, int> op1(indices, workSpace, tilingData, tpipe);
        op1.Process();
        AscendC::SyncAll();
        tpipe.Destroy();
        AscendC::TPipe pipe;
        ScatterNdUpdateV2::ScatterNdUpdateV2KernelNoSort<DTYPE_VAR> op2(updates, output, workSpace, tilingData, pipe);
        op2.Process();
    } else if (TILING_KEY_IS(21)) {
        ScatterNdUpdateV2::LinearIndexKernel<true, int64_t> op1(indices, workSpace, tilingData, tpipe);
        op1.Process();
        AscendC::SyncAll();
        tpipe.Destroy();
        AscendC::TPipe pipe;
        ScatterNdUpdateV2::ScatterNdUpdateV2Kernel<DTYPE_VAR> op2(updates, output, workSpace, tilingData, pipe);
        op2.Process();
    } else if (TILING_KEY_IS(20)) {
        ScatterNdUpdateV2::LinearIndexKernel<false, int64_t> op1(indices, workSpace, tilingData, tpipe);
        op1.Process();
        AscendC::SyncAll();
        tpipe.Destroy();
        AscendC::TPipe pipe;
        ScatterNdUpdateV2::ScatterNdUpdateV2KernelNoSort<DTYPE_VAR> op2(updates, output, workSpace, tilingData, pipe);
        op2.Process();
    } else if (TILING_KEY_IS(30)) {
        ScatterNdUpdateV2::LargeIndexKernel<DTYPE_VAR> op(indices, updates, output, tilingData, tpipe);
        op.Process();
    }
#endif
}
