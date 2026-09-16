/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform.h"
#include <iostream>

namespace op {

SocVersion g_socVersion = SocVersion::ASCEND910B;
PlatformInfo *g_platformInfo = new PlatformInfo();

bool PlatformInfo::CheckSupport(SocSpec socSpec, SocSpecAbility ability) const
{
    return true;
}

PlatformInfo::~PlatformInfo()
{
    if (impl_ != nullptr) {
        delete impl_;
    }
}

bool PlatformInfo::Valid() const
{
    return valid_;
}

void PlatformInfo::SetPlatformImpl(PlatformInfoImpl *impl)
{
    impl_ = impl;
    valid_ = true;
}

SocVersion PlatformInfo::GetSocVersion() const
{
    return g_socVersion;
}

const std::string &PlatformInfo::GetSocLongVersion() const
{
    return "";
}

int32_t PlatformInfo::GetDeviceId() const
{
    return 0;
}

int64_t PlatformInfo::GetBlockSize() const
{
    return 0;
}

uint32_t PlatformInfo::GetCubeCoreNum() const
{
    return 0;
}

uint32_t PlatformInfo::GetVectorCoreNum() const
{
    return 0;
}

bool PlatformInfo::GetFftsPlusMode() const
{
    return true;
}

fe::PlatFormInfos *PlatformInfo::GetPlatformInfos() const
{
    return nullptr;
}

const PlatformInfo &GetCurrentPlatformInfo()
{
    return *g_platformInfo;
}

NpuArch PlatformInfo::GetCurNpuArch() const
{
    static const std::map<SocVersion, NpuArch> soc2ArchMap = {
        {SocVersion::ASCEND910, NpuArch::DAV_1001},    {SocVersion::ASCEND910B, NpuArch::DAV_2201},
        {SocVersion::ASCEND910_93, NpuArch::DAV_2201}, {SocVersion::ASCEND950, NpuArch::DAV_3510},
        {SocVersion::ASCEND310P, NpuArch::DAV_2002},   {SocVersion::ASCEND310B, NpuArch::DAV_3002},
        {SocVersion::ASCEND610LITE, NpuArch::DAV_3102}};
    const auto it = soc2ArchMap.find(g_socVersion);
    if (it != soc2ArchMap.end()) {
        return it->second;
    }
    std::cout << "Error, Unsupported SocVersion, plz modyfy this function" << std::endl;
    return NpuArch::DAV_RESV;
}

ge::AscendString ToString(SocVersion socVersion)
{
    static const std::map<SocVersion, std::string> kSocVersionMap = {
        {SocVersion::ASCEND910, "Ascend910"},
        {SocVersion::ASCEND910B, "Ascend910B"},
        {SocVersion::ASCEND910_93, "Ascend910_93"},
        {SocVersion::ASCEND950, "Ascend950"},
        {SocVersion::ASCEND910E, "Ascend910E"},
        {SocVersion::ASCEND310, "Ascend310"},
        {SocVersion::ASCEND310P, "Ascend310P"},
        {SocVersion::ASCEND310B, "Ascend310B"},
        {SocVersion::ASCEND310C, "Ascend310C"},
        {SocVersion::ASCEND610LITE, "Ascend610LITE"},
        {SocVersion::KIRINX90, "KirinX90"},
        {SocVersion::KIRIN9030, "Kirin9030"},
        {SocVersion::RESERVED_VERSION, "UnknowSocVersion"},
    };
    static const std::string reserved("UnknowSocVersion");
    const auto it = kSocVersionMap.find(socVersion);
    if (it != kSocVersionMap.end()) {
        return ge::AscendString((it->second).c_str());
    } else {
        return ge::AscendString(reserved.c_str());
    }
}

void SetPlatformSocVersion(SocVersion socVersion)
{
    g_socVersion = socVersion;
}

void SetPlatformNpuArch(NpuArch npuArch)
{
    static const std::map<NpuArch, SocVersion> NpuArchSocVersionMap = {
        {NpuArch::DAV_3510, SocVersion::ASCEND950},
        {NpuArch::DAV_RESV, SocVersion::RESERVED_VERSION},
    };
    g_socVersion = SocVersion::RESERVED_VERSION;
    const auto it = NpuArchSocVersionMap.find(npuArch);
    if (it != NpuArchSocVersionMap.end()) {
        g_socVersion = it->second;
    }
}

} // namespace op
