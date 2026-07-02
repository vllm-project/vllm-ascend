/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARCH_ARCH_HPP
#define ARCH_ARCH_HPP

#include "../../attn_infra/base_defs.hpp"

namespace NpuArch::Arch
{

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7U * 1024U;
    static constexpr uint32_t UB_SIZE = 192U * 1024U;
    static constexpr uint32_t L1_SIZE = 512U * 1024U;
    static constexpr uint32_t L0A_SIZE = 64U * 1024U;
    static constexpr uint32_t L0B_SIZE = 64U * 1024U;
    static constexpr uint32_t L0C_SIZE = 128U * 1024U;
};

struct PositionGM {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::GM;
};

struct PositionL1 {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::A1;
};

struct PositionL0A {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::A2;
};

struct PositionL0B {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::B2;
};

struct PositionL0C {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::CO1;
};

struct PositionUB {
    static constexpr AscendC::TPosition POSITION = AscendC::TPosition::VECCALC;
};

} // namespace NpuArch::Arch

#endif // ARCH_ARCH_HPP