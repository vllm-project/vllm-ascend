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
 * \file matmul_v3_asw_basic_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_basic_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "common/op_host/math_util.h"

using Ops::Transformer::MathUtil;
namespace optiling {
namespace mc2_matmul_v3_advanced {

constexpr uint64_t FP32_SPLIT_K_THRESHOLD = 8192UL;
using namespace strategy;
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2MatMulV3, Mc2MatMulV3AswBasicApiTiling, DAV_3510, BASIC_ASWT);

bool Mc2MatMulV3AswBasicApiTiling::IsCapable()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_256);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256);
    if (mCore * nCore > compileInfo_.aicNum) {
        OP_LOGD(args_.opName, "mCnt_[%lu] and nCnt_[%lu] is not enter in Mc2MatMulV3 basic api", mCore, nCore);
        return false;
    }
    if (args_.bFormat != ge::FORMAT_ND || args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for basic api");
        return false;
    }
    if (args_.aDtypeSize == DATA_SIZE_FP32 && !args_.isHf32 && args_.bFormat == ge::FORMAT_ND &&
        args_.kValue > FP32_SPLIT_K_THRESHOLD) {
        OP_LOGD(args_.opName, "fp32 big k is not supported for basic api");
        return false;
    }

    OP_LOGI(args_.opName, "Mc2MatMulV3 tiling enable state basic api");
    return true;
}


uint64_t Mc2MatMulV3AswBasicApiTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetL0C2Out(Mc2MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(Mc2MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
