/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_SOC_VERSION_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_SOC_VERSION_UTILS_H

#include <map>
#include <string>

#include "opdev/platform.h"

namespace ops::ut {

inline op::SocVersion ParseOpSocVersion(const std::string &socVersion)
{
    static const std::map<std::string, op::SocVersion> socMap = {
        {"Ascend310P", op::SocVersion::ASCEND310P},
        {"Ascend910B", op::SocVersion::ASCEND910B},
        {"Ascend950", op::SocVersion::ASCEND950},
    };
    const auto it = socMap.find(socVersion);
    return it == socMap.end() ? op::SocVersion::ASCEND910B : it->second;
}

inline void SetPlatformSocVersion(const std::string &socVersion)
{
    op::SetPlatformSocVersion(ParseOpSocVersion(socVersion));
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_GMM_SOC_VERSION_UTILS_H
