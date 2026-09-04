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
 * \file prompt_flash_attention_tiling_register.cpp
 * \brief
 */
#include "err/ops_err.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "register/op_def_registry.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint32_t ACTUAL_SEQ_Q_INDEX_PFA = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX_PFA = 6;
static ge::graphStatus TilingPrepareForPromptFlashAttention(gert::TilingParseContext* context) {
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
                    return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<PromptFlashAttentionCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null!"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);

    compileInfoPtr->socShortName = ascendcPlatform.GetSocVersion();
    if (compileInfoPtr->socShortName == platform_ascendc::SocVersion::ASCEND310P) {
        // sys workspace size default value
        compileInfoPtr->defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    } else {
        compileInfoPtr->defaultSysWorkspaceSize = 0;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PromptFlashAttention)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX_PFA, ACTUAL_SEQ_KV_INDEX_PFA})
    .Tiling(TilingPromptFlashAttention)
    .TilingParse<PromptFlashAttentionCompileInfo>(TilingPrepareForPromptFlashAttention); // Register entrance functions to the framework
}  // namespace optiling
