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
 * \file mc2_platform_info.cpp
 * \brief
 */

#include "platform/platform_info.h"
#include "mc2_common_log.h"
#include "runtime/rt_external_base.h"

namespace ops {
constexpr uint32_t VERSION_SIZE = 32;

bool IsTargetPlatformSocVersion(const char *nodeName, const std::set<std::string> &targetPlatform)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info!");
        return false;
    }
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return targetPlatform.count(short_soc_version) > 0;
}

bool IsTargetPlatformNpuArch(const char *nodeName, const std::set<std::string> &targetPlatform)
{
    char versionValNpuArch[VERSION_SIZE];
    if (rtGetSocSpec("version", "NpuArch", versionValNpuArch, VERSION_SIZE) != RT_ERROR_NONE) {
        OPS_LOG_E(nodeName, "Cannot get npuArch info!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get Platform NpuArch %s", versionValNpuArch);
    return (targetPlatform.count(versionValNpuArch) > 0);
}
} // namespace ops
