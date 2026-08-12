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
 * \file mc2_platform_info.h
 * \brief
 */

#ifndef OPS_TRANSFORMER_MC2_PLATFORM_INFO_H
#define OPS_TRANSFORMER_MC2_PLATFORM_INFO_H

#include <set>
#include <string>
#include "platform/soc_spec.h"

namespace ops {
const std::set<std::string> PLATFORM_A2 = {"Ascend910B"};
const std::set<std::string> PLATFORM_A3 = {"Ascend910_93"};
const std::set<std::string> NPUARCH_A5 = {std::to_string(static_cast<uint32_t>(NpuArch::DAV_3510))};

// distinguish between 910B and 910_93
bool IsTargetPlatformSocVersion(const char *nodeName, const std::set<std::string> &targetPlatform);

// generally used
bool IsTargetPlatformNpuArch(const char *nodeName, const std::set<std::string> &targetPlatform);

} // namespace ops
#endif // OPS_TRANSFORMER_MC2_PLATFORM_INFO_H
