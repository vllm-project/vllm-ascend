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
 * \brief Checker for mask_mode, attn_mask parameters (文档约束: Mask参数组)
 *
 * mask_mode <-> FIA sparseMode mapping:
 *   The fia*mla compute path has only ever been exercised with sparseMode in
 *   {NO_MASK(0), RIGHT_DOWN(3)} — hard gate at
 *   fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_mla.cpp:79
 *   (IsCapableSparseLayoutCheckMla), values from fia_tiling_info.h:69,72.
 *   This repo's flash_attn MaskMode enum is {NO_MASK=0, CAUSAL=3, BAND=4}
 *   (fa_tiling_info.h:106-110) and the FIA kernel implements right-down
 *   causality via `sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL`
 *   (fia_kernel_noquant_mla.h:466). CAUSAL and SPARSE_MODE_RIGHT_DOWN are
 *   numerically identical (3), so the mapping is:
 *     mask_mode 0 (NO_MASK)  -> sparseMode 0 (NO_MASK)     [no attn_mask]
 *     mask_mode 3 (CAUSAL)   -> sparseMode 3 (RIGHT_DOWN) [attn_mask required]
 *   Every other value (incl. BAND=4) is REJECTED.
 *   FIA additionally requires attn_mask when sparseMode != 0
 *   (fia_tiling_info.cpp mask_checker CheckDimAndShape), so flash_attn's
 *   "attn_mask required for CAUSAL" existence rule is kept.
 *   The left/right window attrs are NOT part of the interface — no checks.
 */

#include <map>
#include <numeric>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "mask_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus MaskChecker::CheckSingleParaMaskMode(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    // only {0 (NO_MASK), 3 (CAUSAL == FIA SPARSE_MODE_RIGHT_DOWN)} are valid.
    // BAND(4)/LEFT_UP/etc. route in FIA MLA to nothing — reject.
    const std::vector<int64_t> maskModeList = {static_cast<int64_t>(MaskMode::NO_MASK),
                                               static_cast<int64_t>(MaskMode::CAUSAL)};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(static_cast<int64_t>(faInfo.maskMode), maskModeList),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "mask_mode",
                                                      std::to_string(faInfo.maskMode).c_str(),
                                                      "The value of mask_mode can only be 0 (NO_MASK) or 3 (CAUSAL, == FIA RIGHT_DOWN)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckSingleParaAttnMask(const FlashMlaWithKvcacheTilingInfo &faInfo)
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

ge::graphStatus MaskChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (CheckSingleParaMaskMode(faInfo) != ge::GRAPH_SUCCESS || CheckSingleParaAttnMask(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskChecker::CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    auto &attnMaskTensor = faInfo.opParamInfo.attnMask.tensor;

    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        OP_CHECK_IF(attnMaskTensor != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "attn_mask", "not empty",
                                                          "When mask_mode=0 (no mask mode), attn_mask must be empty"),
                    return ge::GRAPH_FAILED);
    }
    if (faInfo.maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        OP_CHECK_IF(attnMaskTensor == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "attn_mask", "empty",
                                                          "When mask_mode=3 (causal/right-down), attn_mask must be provided"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling
