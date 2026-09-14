/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mask_checker_sparse_flash_mla.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus MaskChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(context.oriMaskMode != 0 && context.oriMaskMode != 3 &&
                    context.oriMaskMode != 4, // 3: RightDownCausal模式 4: Band模式
                OP_LOGE_FOR_INVALID_VALUE(Op(context), "ori_mask_mode", std::to_string(context.oriMaskMode).c_str(),
                                          "0, 3 or 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context.cmpMaskMode != 0 && context.cmpMaskMode != 3, // 3: RightDownCausal模式
        OP_LOGE_FOR_INVALID_VALUE(Op(context), "cmp_mask_mode", std::to_string(context.cmpMaskMode).c_str(), "0 or 3"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.oriWinLeft < -1 || context.oriWinRight < -1,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    Op(context), "ori_win_left and ori_win_right",
                    (std::to_string(context.oriWinLeft) + ", " + std::to_string(context.oriWinRight)).c_str(),
                    "Both values must be -1 or non-negative"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckFeature(const CheckContext &context) const
{
    if (context.oriMaskMode != 4) { // 4: Band模式
        OP_CHECK_IF(context.oriWinLeft != -1 || context.oriWinRight != -1,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        Op(context), "ori_win_left and ori_win_right",
                        (std::to_string(context.oriWinLeft) + ", " + std::to_string(context.oriWinRight)).c_str(),
                        "Ori_win_left and ori_win_right must be -1 when ori_mask_mode is not 4"),
                    return ge::GRAPH_FAILED);
    }

    if (!context.cmpKv.present) {
        OP_CHECK_IF(context.cmpMaskMode != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Op(context), "cmp_mask_mode",
                                                          std::to_string(context.cmpMaskMode).c_str(),
                                                          "Cmp_mask_mode must be 0 when cmp_kv is absent"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
