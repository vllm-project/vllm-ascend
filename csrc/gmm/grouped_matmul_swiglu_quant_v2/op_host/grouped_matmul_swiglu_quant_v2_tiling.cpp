/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_tiling.cpp
 * \brief
 */

#include "grouped_matmul_swiglu_quant_v2_tiling.h"
#include <climits>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "tiling_base/tiling_base.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "grouped_matmul_swiglu_quant_v2_fusion_tiling.h"
#include "grouped_matmul_swiglu_quant_v2_base_tiling.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::GroupedMatmulSwigluQuantV2Tiling;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2FusionTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2BaseTiling, 1);

static ge::graphStatus GroupedMatmulSwigluQuantV2TilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
            OPS_REPORT_CUBE_INNER_ERR("GroupedMatmulSwigluQuantV2TilingFunc", "Tilingcontext is null"),
            return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompileInfo<GMMSwigluV2CompileInfo>();
    if (compileInfoPtr->supportL12BtBf16) {
        std::vector<int32_t> registerList = {2};
        OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc", "Using the tiling strategy in the mxfp8");
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }else {
        std::vector<int32_t> registerList = {0,1};
        OP_LOGD("GroupedMatmulSwigluQuantV2TilingFunc", "Using the tiling strategy in the int8");
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}

ASCENDC_EXTERN_C graphStatus TilingPrepareForGMMSwigluQuantV2(gert::TilingParseContext *context)
{
    // get info
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum_ = ascendcPlatform.GetCoreNumAiv();
    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
    OP_LOGD(context->GetNodeName(), "ubSize is %lu, aicNum is %u.", compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluQuantV2)
    .Tiling(GroupedMatmulSwigluQuantV2TilingFunc)
    .TilingParse<GMMSwigluV2CompileInfo>(TilingPrepareForGMMSwigluQuantV2);
} // namespace optiling
