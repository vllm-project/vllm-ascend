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
 * \file matmul_v3_platform_common.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__
#define __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__

#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_infos_def.h"
#include "mc2_log.h"
#include "platform/soc_spec.h"
namespace optiling {
const std::initializer_list<NpuArch> Mc2AdvancedSocVersion = {NpuArch::DAV_3510,
                                                              NpuArch::DAV_RESV}; // supportMmadS8S4平台

template <typename T>
inline typename std::enable_if<
    std::is_same<T, gert::TilingParseContext>::value || std::is_same<T, gert::TilingContext>::value, bool>::type
Mc2IsAdvancedSocVersion(T *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("Mc2MatMulV3", "context"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "platformInfo"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    std::string mmad;
    bool res = platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    bool supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;
    NpuArch npuArch = supportMmadS8S4 ? NpuArch::DAV_RESV : ascendcPlatform.GetCurNpuArch();
    return std::find(Mc2AdvancedSocVersion.begin(), Mc2AdvancedSocVersion.end(), npuArch) !=
           Mc2AdvancedSocVersion.end();
}
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_PLATFORM_COMMON_H__
