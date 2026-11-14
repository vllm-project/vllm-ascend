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
 * \file grouped_matmul_situ_quant.cpp
 * \brief GroupedMatmul(A8W4 MSD) + dequant + SiTU-GLU + per-token quant 的 kernel 入口。
 *
 * 只实现 Atlas A2 / A3 的 A8W4 MSD 路径（x=INT8, weight=INT4），tiling key = 2。
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch32/grouped_matmul_situ_quant_a8w4_msd_pipeline.h"
#include "arch32/grouped_matmul_situ_quant_utils_kernel.h"

using namespace AscendC;
using namespace matmul;
using namespace GroupedMatmulDequantSituQuant;

extern "C" __global__ __aicore__ void grouped_matmul_situ_quant(GM_ADDR x, GM_ADDR xScale, GM_ADDR weight,
                                                                GM_ADDR weightScale, GM_ADDR groupList,
                                                                GM_ADDR weightAssistanceMatrix, GM_ADDR bias,
                                                                GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

#if defined(GMM_SITU_QUANT_A8W4_MSD)
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        // The member macro materializes the enclosing tiling record. Decode
        // it once and keep every referenced member alive through Process().
        GET_TILING_DATA_WITH_STRUCT(GMMSituQuantTilingData, tilingData, tiling);
        const auto& gmmSituQuant_ = tilingData.gmmSituQuant;
        using xType = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t, false>;
        using weightType = MatmulType<TPosition::GM, wFormat, int4b_t, false>;
        using yType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
        using matmulType = MMImplTypeCustom<xType, weightType, yType>;
        matmulType::MT mm;
        // Cube 分块尺寸要按 live 行数在运行时二选一（每专家平均行数少的时候 baseM/baseN
        // 互换）。主分块在 mmTilingData，
        // 备选在 mmTilingDataAlt。mid 的调度器（singleM/singleN/blockDimM）读的是
        // baseParams 里的 baseM/baseN，必须和交给 mm.Init 的 TCubeTiling 同步切换，
        // 所以这里拷一份本地副本，只改这两个字段。
        GMMSituQuantBaseParams baseParams = tilingData.gmmSituQuantBaseParams;
        GMMSituQuantPipelineSchedule<matmulType> op(mm, &baseParams, &gmmSituQuant_, &tPipe);
        op.Init(x, weight, weightScale, xScale, weightAssistanceMatrix, groupList, y, yScale, userWorkspace);
        // live 行数要读 groupList 才知道，所以分块选择放在 op.Init 之后、op.Process 之前：
        // 此时 pre/mid/post 都还没绑 baseParams（它们是 Process 里才 Init 的）。
        const int64_t altLimit = baseParams.altRowsPerExpertLimit;
        const int64_t groupNum = gmmSituQuant_.groupListLen;
        const bool useAlt = (altLimit > 0) && (groupNum > 0) && (wFormat == CubeFormat::NZ) &&
                            (op.GetLiveRows() <= altLimit * groupNum);
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            const auto* selectedMmTiling = useAlt ? &tilingData.mmTilingDataAlt : &tilingData.mmTilingData;
            baseParams.baseM = selectedMmTiling->baseM;
            baseParams.baseN = selectedMmTiling->baseN;
            mm.Init(selectedMmTiling);
        }
        op.Process();
    }
#endif
}
