/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RESOURCE_HPP
#define RESOURCE_HPP

#include "../../attn_infra/base_defs.hpp"
#include "../../attn_infra/arch/local_tensor_buffer.hpp"

namespace NpuArch::Arch
{

template<class ArchTag>
struct Resource
{};

template<>
struct Resource<Arch::AtlasA5> {
public:
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::VECCALC> ubBuf;
    LocalTensorBuffer<Arch::AtlasA5, AscendC::TPosition::C2PIPE2GM> fpBuf;

    __aicore__ inline
    Resource()
    {
        AscendC::InitSocState();
    }

    __aicore__ inline
    ~Resource()
    {
        AscendC::InitSocState();
    }
};

template<>
struct Resource<Arch::AtlasA2> {
public:
    AscendC::TPipe pipe;

    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::VECCALC> ubBuf;

    __aicore__ inline
    Resource()
    {
        pipe.Destroy();
    }
};

} // namespace NpuArch::Arch

#endif // INCLUDE_ARCH_RESOURCE_HPP