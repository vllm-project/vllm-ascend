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
 * \file mask_checker.cpp
 * \brief Checker for mask_mode, attn_mask, win_left, win_right parameters (文档约束: Mask参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "mask_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;
ge::graphStatus MaskChecker::CheckSingleParaMaskMode(const FaTilingInfo &faInfo)
{
    const std::vector<int64_t> maskModeList = {static_cast<int64_t>(MaskMode::NO_MASK),
                                               static_cast<int64_t>(MaskMode::CAUSAL),
                                               static_cast<int64_t>(MaskMode::BAND)};
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != CheckValueSupport(static_cast<int64_t>(faInfo.maskMode), maskModeList),
        OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "mask_mode", std::to_string(faInfo.maskMode).c_str(), "0, 3 or 4"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaAttnMask(const FaTilingInfo &faInfo)
{
    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;
    if (attnMaskTensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *attnMaskDesc = faInfo.opParamInfo.attnMask.desc;
    OP_CHECK_IF(attnMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "TensorDesc of attn_mask"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(attnMaskDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "attn_mask",
                                          Ops::Base::ToString(attnMaskDesc->GetDataType()).c_str(), "INT8"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(attnMaskDesc, ATTN_MASK_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = attnMaskTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "attn_mask", (std::to_string(dimNum) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);

    int64_t dim0 = attnMaskTensor->GetStorageShape().GetDim(0);
    int64_t dim1 = attnMaskTensor->GetStorageShape().GetDim(1);
    OP_CHECK_IF(
        dim0 != SPARSE_OPTIMIZE_ATTENTION_SIZE || dim1 != SPARSE_OPTIMIZE_ATTENTION_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "attn_mask",
                                              ("[" + std::to_string(dim0) + ", " + std::to_string(dim1) + "]").c_str(),
                                              "The shape of attn_mask must be [2048, 2048]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaWindowParams(const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(faInfo.winLeft < -1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "win_left", std::to_string(faInfo.winLeft).c_str(), ">= -1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.winRight < -1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "win_right", std::to_string(faInfo.winRight).c_str(), ">= -1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (CheckSingleParaMaskMode(faInfo) != ge::GRAPH_SUCCESS || CheckSingleParaAttnMask(faInfo) != ge::GRAPH_SUCCESS ||
        CheckSingleParaWindowParams(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;

    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        OP_CHECK_IF(attnMaskTensor != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "attn_mask", "not empty",
                                                          "When mask_mode=0 (no mask mode), attn_mask must be empty"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            faInfo.winLeft != -1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "win_left", std::to_string(faInfo.winLeft).c_str(),
                                                  "When mask_mode=0 (no mask mode), win_left must be -1"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            faInfo.winRight != -1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "win_right", std::to_string(faInfo.winRight).c_str(),
                                                  "When mask_mode=0 (no mask mode), win_right must be -1"),
            return ge::GRAPH_FAILED);
    }
    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        OP_CHECK_IF(attnMaskTensor == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "attn_mask", "empty",
                                                          "When mask_mode=3 (causal mode), attn_mask must be provided"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            faInfo.winLeft != -1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "win_left", std::to_string(faInfo.winLeft).c_str(),
                                                  "When mask_mode=3 (causal mode), win_left must be -1"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            faInfo.winRight != -1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "win_right", std::to_string(faInfo.winRight).c_str(),
                                                  "When mask_mode=3 (causal mode), win_right must be -1"),
            return ge::GRAPH_FAILED);
    }
    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::BAND)) {
        OP_CHECK_IF(attnMaskTensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        faInfo.opName, "attn_mask",
                        "The value of attn_mask cannot be nullptr when the mask_mode=4 (band mode)"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            faInfo.winLeft < -1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "win_left", std::to_string(faInfo.winLeft).c_str(),
                                                  "The value of win_left must be greater than or equal to -1"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            faInfo.winRight < -1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "win_right", std::to_string(faInfo.winRight).c_str(),
                                                  "The value of win_right must be greater than or equal to -1"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
