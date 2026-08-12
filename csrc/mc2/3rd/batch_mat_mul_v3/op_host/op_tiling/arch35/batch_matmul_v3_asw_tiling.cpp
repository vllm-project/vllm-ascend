/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_asw_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
using namespace strategy;
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2BatchMatMulV3, Mc2BatchMatMulV3AswTiling, DAV_3510, BASE);
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2BatchMatMulV3, Mc2BatchMatMulV3AswTiling, DAV_RESV, BASE); // supportMmadS8S4平台

ge::graphStatus Mc2BatchMatMulV3AswTiling::DoOpTiling()
{
    Mc2MatMulV3TilingHelper::ResetBase(context_, compileInfo_, args_, runInfo_);
    Mc2MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    if (Mc2MatMulV3TilingHelper::CheckIfDoubleAswt(context_, args_, batchInfo_->batchC)) {
        aswtModel_ = Mc2MatMulV3Model::DOUBLE_ASWT;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2BatchMatMulV3AswTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey().SetTrans(args_.isATrans, args_.isBTrans).SetModel(aswtModel_).GetTilingKey();
}

uint64_t Mc2BatchMatMulV3AswTiling::GetBlockDim() const
{
    return compileInfo_.aicNum;
}
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
