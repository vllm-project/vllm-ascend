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
 * \file flash_mla_with_kvcache_tiling.cpp
 * \brief FlashMlaWithKvcache Tiling主入口
 */

#include <cmath>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "flash_mla_with_kvcache_tiling.h"
#include "flash_mla_with_kvcache_tiling_common.h"
#include "../op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h"
#include "flash_mla_with_kvcache_tiling_info_parser.h"
#include "checkers/flash_mla_with_kvcache_checker.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
using namespace flash_mla_with_kvcache;

static bool IsEmptyInput(gert::TilingContext *context)
{
    (void)context;
    return false;
}

ASCENDC_EXTERN_C ge::graphStatus TilingFlashMlaWithKvcache(gert::TilingContext *context)
{
    OP_LOGW(context, "FlashMlaWithKvcache TilingFlashMlaWithKvcache start.");

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        if (IsEmptyInput(context)) {
            return ge::GRAPH_SUCCESS;
        }
    }

    FlashMlaWithKvcacheTilingInfo faInfo;
    FlashMlaWithKvcacheInfoParser faInfoParser(context);
    if (faInfoParser.Parse(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FlashMlaWithKvcacheChecker faChecker;
    faChecker.Init(faInfo);
    if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &faInfo);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashMlaWithKvcache(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<FlashMlaWithKvcacheCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

// 注册tiling函数：
IMPL_OP_OPTILING(FlashMlaWithKvcache)
    .Tiling(TilingFlashMlaWithKvcache)

    .TilingParse<FlashMlaWithKvcacheCompileInfo>(TilingPrepareForFlashMlaWithKvcache);

} // namespace optiling