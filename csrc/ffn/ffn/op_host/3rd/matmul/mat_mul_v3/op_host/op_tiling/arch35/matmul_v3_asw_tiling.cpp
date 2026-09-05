/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_asw_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3AswTiling, DAV_3510, BASE);

void MatMulV3AswTiling::CalcTailBasicBlock()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfo_.aicNum ? 0UL : mnCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    if (tailCnt != 0UL) {
        while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= compileInfo_.aicNum &&
               (!args_.isATrans || MathUtil::CeilDivision(runInfo_.baseM, runInfo_.tailInfo.mCnt) * args_.aDtypeSize >
                                       BASIC_BLOCK_K_128_BYTE)) {
            runInfo_.tailInfo.mCnt += 1UL;
            // swiglu 单 matmul：N 尾块必须保持 32 对齐整块（半 tile 布局要求 curNL1%32==0），
            // 尾波禁止 N 方向拆分，只允许 M 方向拆分（M 拆分子 tile 的 N 保持 baseN 不变）。
            if (!args_.swigluSingleNAlign32 &&
                runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum &&
                (args_.isBTrans || MathUtil::CeilDivision(runInfo_.baseN, runInfo_.tailInfo.nCnt) * args_.bDtypeSize >
                                       BASIC_BLOCK_K_128_BYTE)) {
                runInfo_.tailInfo.nCnt += 1UL;
            }
        }
    }
}

ge::graphStatus MatMulV3AswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::GetRebalanceBlock(compileInfo_, args_, runInfo_, context_);
    CalcTailBasicBlock();
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3AswTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetL0C2Out(MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_))
        .GetTilingKey();
}

ge::graphStatus MatMulV3AswTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3TilingData>(tiling);
}
} // namespace matmul_v3_advanced
} // namespace optiling
