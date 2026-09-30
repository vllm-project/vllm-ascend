/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "gmm_dequant_situ_quant.h"
#include "gmm_dequant_situ_quant_tiling.h"

// The dynamic TensorList descriptor starts with the byte offset of its address
// array. The existing compute classes consume that array without modification.
__aicore__ inline GM_ADDR GetExpertPointerTable(GM_ADDR tensorList)
{
    auto *descriptor = reinterpret_cast<__gm__ uint64_t *>(tensorList);
    return tensorList + descriptor[0];
}

extern "C" __global__ __aicore__ void gmm_dequant_situ_quant(
    GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale, GM_ADDR groupList,
    GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GmmDequantSituQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GmmDequantSituQuantTilingData, data, tiling);
    GM_ADDR wPtrTbl = GetExpertPointerTable(weight);
    GM_ADDR scPtrTbl = GetExpertPointerTable(weightScale);
    GM_ADDR packedA = AscendC::GetUserWorkspace(workspace);
    GM_ADDR rawAcc = packedA + data.packedBytes;
    const int32_t E = data.experts;
    const int32_t K = data.k;
    const int32_t N = data.n;
    const int32_t C = data.capacity;
    const int32_t glType = data.groupListType;
    const int32_t hasLinear = data.hasLinear;
    const int32_t nzInput = data.weightNz;
    const float beta = data.beta;
    const float invBeta = data.invBeta;
    const float linBeta = data.linearBeta;
    const float invLinBeta = data.invLinearBeta;
    GlobalTensor<int64_t> glGM;
    glGM.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
    TPipe pipe;
    if ASCEND_IS_AIC {
        AscendC::AscendCUtils::SetOverflow(1);
        if (nzInput != 0) {
            GmsqExactMsdCube<MsdNzMt, true> kernel;
            kernel.Init(wPtrTbl, packedA, rawAcc, glGM, E, K, N, glType, &pipe);
            kernel.Process();
        } else {
            GmsqExactMsdCube<> kernel;
            kernel.Init(wPtrTbl, packedA, rawAcc, glGM, E, K, N, glType, &pipe);
            kernel.Process();
        }
    }
    if ASCEND_IS_AIV {
        GmsqExactMsdVector kernel;
        kernel.Init(x, wPtrTbl, scPtrTbl, packedA, rawAcc, xScale, y, yScale,
                    glGM, E, K, N, C, glType, beta, invBeta, hasLinear, linBeta,
                    invLinBeta, &pipe);
        kernel.Process();
    }
}
