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
 * \file matmul_v3_compile_info_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMPILE_INFO_ADVANCED_H__
#define __OP_HOST_MATMUL_V3_COMPILE_INFO_ADVANCED_H__

#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "../matmul_v3_compile_info.h"

#include "../../../../../common/utils/mc2_log.h"
#include "mc2_log_compat.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {

inline ge::graphStatus InitCompileInfo(fe::PlatFormInfos *platformInfo, Mc2MatmulV3CompileInfo *compileInfoPtr)
{
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT("Mc2MatMul", "platformInfo"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(compileInfoPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT("Mc2MatMul", "compileInfoPtr"),
                    return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    std::string mmad;
    bool res = platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    bool supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;
    compileInfoPtr->socVersion =
        supportMmadS8S4 ? platform_ascendc::SocVersion::RESERVED_VERSION : ascendcPlatform.GetSocVersion();
    compileInfoPtr->supportL0c2out = false;   // Not used
    compileInfoPtr->supportL12BtBf16 = false; // Not used
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, compileInfoPtr->btSize);
    OP_LOGI(
        "Mc2MatMul",
        "parse compile info success soc:%d, aicNum:%lu, aivNum:%lu, ubSize:%lu, l1Size:%lu, l2Size:%lu, l0ASize:%lu, "
        "l0BSize:%lu, "
        "l0CSize:%lu, btSize:%lu",
        static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->aicNum, compileInfoPtr->aivNum,
        compileInfoPtr->ubSize, compileInfoPtr->l1Size, compileInfoPtr->l2Size, compileInfoPtr->l0ASize,
        compileInfoPtr->l0BSize, compileInfoPtr->l0CSize, compileInfoPtr->btSize);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InitCompileInfo(gert::TilingParseContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("Mc2MatMul", "context"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    auto compileInfoPtr = context->GetCompiledInfo<Mc2MatmulV3CompileInfo>();
    return InitCompileInfo(platformInfo, compileInfoPtr);
}
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_COMPILE_INFO_ADVANCED_H__
