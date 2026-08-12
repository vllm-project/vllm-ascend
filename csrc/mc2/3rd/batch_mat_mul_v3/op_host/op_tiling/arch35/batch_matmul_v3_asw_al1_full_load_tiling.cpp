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
 * \file batch_matmul_v3_asw_al1_full_load_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_al1_full_load_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
using namespace strategy;
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2BatchMatMulV3, Mc2BatchMatMulV3AswAL1FullLoadTiling, DAV_3510, AL1_FULL_LOAD);

bool Mc2BatchMatMulV3AswAL1FullLoadTiling::IsCapable()
{
    if (batchInfo_->batchA > 1UL) { // matrix A should not have batch when AL1FullLoad
        return false;
    }
    if (batchInfo_->batchB < 1UL) { // matrix B should have batch when AL1FullLoad
        return false;
    }
    if (args_.mValue > BASIC_BLOCK_SIZE_256) {
        return false;
    }
    // get align m,k,n value
    uint64_t c0Size = BLOCK_BYTE_SIZE / args_.aDtypeSize;
    uint64_t alignMValue =
        args_.isATrans ? ops::CeilAlign(args_.mValue, c0Size) : ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKaValue =
        args_.isATrans ? ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16) : ops::CeilAlign(args_.kValue, c0Size);
    uint64_t alignKbValue =
        args_.isBTrans ? ops::CeilAlign(args_.kValue, c0Size) : ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignNValue =
        args_.isBTrans ? ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16) : ops::CeilAlign(args_.nValue, c0Size);
    uint64_t alignMatASize = batchInfo_->batchA * alignMValue * alignKaValue * args_.aDtypeSize;
    uint64_t alignMatBSize = batchInfo_->batchB * alignKbValue * alignNValue * args_.bDtypeSize;
    if (alignMatASize * NUM_TWO > compileInfo_.l1Size) {
        return false;
    }
    if (alignMatBSize < (compileInfo_.l1Size) * compileInfo_.aicNum &&
        batchInfo_->batchB < 4UL * compileInfo_.aicNum) { // each core needs to loop at least 4 batch
        return false;
    }
    return true;
}

ge::graphStatus Mc2BatchMatMulV3AswAL1FullLoadTiling::DoOpTiling()
{
    Mc2MatMulV3TilingHelper::ResetBase(context_, compileInfo_, args_, runInfo_);
    Mc2MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    Mc2MatMulV3AswFullLoadTiling::DoAL1FullLoad(isAl1MulCoreLoad_, args_.batchInfo->batchB, args_.batchInfo->batchBias);
    if (Mc2MatMulV3TilingHelper::CheckIfDoubleAswt(context_, args_, batchInfo_->batchC)) {
        aswtModel_ = Mc2MatMulV3Model::DOUBLE_ASWT;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2BatchMatMulV3AswAL1FullLoadTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(aswtModel_)
        .SetFullLoad(Mc2MatMulV3FullLoad::A_FULL_LOAD)
        .GetTilingKey();
}
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
