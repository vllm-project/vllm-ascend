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
 * \file batch_matmul_v3_asw_basic_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_basic_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
using namespace strategy;
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2BatchMatMulV3, Mc2BatchMatMulV3AswBasicTiling, DAV_3510, ASW_BASIC);

bool Mc2BatchMatMulV3AswBasicTiling::IsCapable()
{
    if (Mc2MatMulV3TilingHelper::CheckIfDoubleAswt(context_, args_, batchInfo_->batchC)) {
        return false;
    }

    bool isEqualBatch = batchInfo_->batchA0 == batchInfo_->batchB0 && batchInfo_->batchA1 == batchInfo_->batchB1 &&
                        batchInfo_->batchA2 == batchInfo_->batchB2 && batchInfo_->batchA3 == batchInfo_->batchB3;
    if (args_.hasBias || !isEqualBatch) {
        return false;
    }
    return true;
}

ge::graphStatus Mc2BatchMatMulV3AswBasicTiling::DoOpTiling()
{
    Mc2MatMulV3TilingHelper::ResetBase(context_, compileInfo_, args_, runInfo_);
    Mc2MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);

    // l1开2db后依然只使用了一半的空间，则开启4 db。该字段仅在基础api场景生效
    uint64_t abL1TensorSize = runInfo_.baseK * runInfo_.stepKa * (runInfo_.baseM + runInfo_.baseN) * args_.aDtypeSize;
    if (args_.hasBias) {
        abL1TensorSize += runInfo_.baseN * sizeof(args_.biasType);
    }
    if (abL1TensorSize * NUM_FOUR <= compileInfo_.l1Size) {
        runInfo_.l1BufferNum = NUM_FOUR;
    } else {
        runInfo_.l1BufferNum = NUM_TWO;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2BatchMatMulV3AswBasicTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(aswtModel_)
        .SetApiLevel(Mc2MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

uint64_t Mc2BatchMatMulV3AswBasicTiling::GetBlockDim() const
{
    return compileInfo_.aicNum;
}
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
