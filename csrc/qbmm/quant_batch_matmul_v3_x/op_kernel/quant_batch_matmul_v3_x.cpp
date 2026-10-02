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
 * \file quant_batch_matmul_v3_x.cpp
 * \brief Kernel entry for QuantBatchMatmulV3X on Ascend 310P. Instantiates QBMInt8Compute with the SCALE_UINT64 template parameter.
 * \author Feodor Pisnitchenko
 */

#include "quant_batch_matmul_v3_x_utils.h"
#include "quant_batch_matmul_v3_x_tiling_key.h"

#include "kernel_operator.h"

// The opc compile injects the tiling data struct (class QBMTilingData)
// generated from the op's REGISTER_TILING_DATA_CLASS. Alias it to the
// name the compute classes expect.
typedef QBMTilingData QBMParams;

#include "quant_batch_matmul_v3_x_int8.h"

using namespace AscendC;
using namespace QBM;

template <int TRANS, int KERNEL_TEMPLATE_TYPE, int PERTOKEN, int OPTIONATTR>
__global__ __aicore__ void quant_batch_matmul_v3_x(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
    GM_ADDR workSpace, GM_ADDR tiling)
{
    TPipe tPipe;
    GM_ADDR workspace = GetUserWorkspace(workSpace);
    if (workspace == nullptr) {
        workspace = workSpace;
    }

    GET_TILING_DATA_WITH_STRUCT(QBMTilingData, tilingData, tiling);

    QBMInt8Compute<SCALE_UINT64> op;
    op.Init(x1, x2, scale, offset, bias, pertokenScale, y,
            workspace, &tilingData, &tPipe);
    op.Process();
}
