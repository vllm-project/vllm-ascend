/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file host_utils.h
 * \brief
 */

#ifndef UTILS_HOST_UTILS_H
#define UTILS_HOST_UTILS_H
#ifndef __NPU_ARCH__
#include "tiling/platform/platform_ascendc.h"
#endif
namespace Act {
namespace Gemm {

static int64_t GetCoreNum()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return 0;
    }
    else {
        return ascendcPlatform->GetCoreNumAic();
    }
}

static size_t GetSysWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return 0;
    }
    else {
        return static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    }
}
} // namespace Gemm
} // namespace Act
#endif