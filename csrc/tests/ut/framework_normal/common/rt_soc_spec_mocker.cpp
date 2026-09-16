/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include <string>
#include <runtime/rt_external_base.h>

namespace {
struct RtSocSpecMockState {
    std::string shortSocVersion = "Ascend910B";
    std::string npuArch = "";
    bool rtGetSocSpecFail = false;
};
RtSocSpecMockState &MockState()
{
    static RtSocSpecMockState state;
    return state;
}
} // namespace

extern "C" {
void SetRtSocSpecShortVersion(const char *version)
{
    MockState().shortSocVersion = version ? version : "";
}

void SetRtSocSpecNpuArch(const char *arch)
{
    MockState().npuArch = arch ? arch : "";
}

void SetRtSocSpecFail(bool fail)
{
    MockState().rtGetSocSpecFail = fail;
}

rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen)
{
    if (MockState().rtGetSocSpecFail) {
        return static_cast<rtError_t>(1);
    }
    if (label == nullptr || key == nullptr || val == nullptr) {
        return static_cast<rtError_t>(1);
    }
    if (std::string(label) == "version" && std::string(key) == "Short_SoC_version") {
        if (maxLen == 0) {
            return static_cast<rtError_t>(1);
        }
        const auto &v = MockState().shortSocVersion;
        auto copyLen = static_cast<uint32_t>(v.size());
        if (copyLen > maxLen - 1) {
            copyLen = maxLen - 1;
        }
        if (memcpy_s(val, maxLen, v.c_str(), copyLen) != EOK) {
            return static_cast<rtError_t>(1);
        }
        val[copyLen] = '\0';
        return RT_ERROR_NONE;
    }
    if (std::string(label) == "version" && std::string(key) == "NpuArch") {
        if (maxLen == 0) {
            return static_cast<rtError_t>(1);
        }
        const auto &v = MockState().npuArch;
        auto copyLen = static_cast<uint32_t>(v.size());
        if (copyLen > maxLen - 1) {
            copyLen = maxLen - 1;
        }
        if (memcpy_s(val, maxLen, v.c_str(), copyLen) != EOK) {
            return static_cast<rtError_t>(1);
        }
        val[copyLen] = '\0';
        return RT_ERROR_NONE;
    }
    return RT_ERROR_NONE;
}
}
