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
 * \file tiling_util.h
 * \brief
 */

#pragma once

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "platform/soc_spec.h"
#include "log/log.h"

namespace Ops {
namespace NN {
namespace OpTiling {
static const gert::Shape g_vec_1_shape = {1};

static bool IsRegbaseNpuArch(NpuArch npuArch)
{
    const static std::set<NpuArch> regbaseNpuArchs = {
        NpuArch::DAV_3510,
        NpuArch::DAV_5102};
    return regbaseNpuArchs.find(npuArch) != regbaseNpuArchs.end();
}

static inline bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    OP_LOGI(context, "Current NpuArch is %u", static_cast<uint32_t>(npuArch));
    return IsRegbaseNpuArch(npuArch);
}

static inline bool IsRegbaseSocVersion(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    OP_LOGI(context, "Current NpuArch is %u", static_cast<uint32_t>(npuArch));
    return IsRegbaseNpuArch(npuArch);
}

inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}
} // namespace OpTiling
} // namespace NN
} // namespace Ops